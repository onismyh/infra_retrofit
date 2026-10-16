"""一个规划年里煤电各 hub 的路径约束（`_add_plant_path_constraints`）：用水、空冷份额、捕集量与残余排放（含原址重建部分与
掺烧各档）。重建部分的拆分在 `constraints_rebuilt`，掺烧档位在 `constraints_blend`；`model_resources` 用这里的两个向量约束工具。"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..constants import WATER_FLOW_SCALE   # 流量缩放系数，用于数值稳定
from ._shared import GRB, PATHWAY_INDEX, gp
from .assumptions import OptimizationAssumptions
from .constraints_blend import add_blend_level_constraints
from .constraints_rebuilt import RUNNING, add_rebuilt_split, rebuilt_dot
from .scenario import OptimizationScenario, PATHWAYS
from .year_types import GrbExpr, GrbMVar, RebuiltShares, YearData


def _add_vector_equality(model, lhs, rhs, length: int, name: str) -> None:
    model.addConstrs((lhs[idx] == rhs[idx] for idx in range(length)), name=name)


def _add_vector_upper_bound(model, lhs, rhs, length: int, name: str, rows=None) -> None:
    """lhs[i] <= rhs[i]，i 取 rows（缺省为全部 0..length-1）；约束名的下标仍是 i。"""
    model.addConstrs((lhs[idx] <= rhs[idx] for idx in (range(length) if rows is None else rows)), name=name)


def _path_dot(coeffs, variables, p: int, skip_zero: bool = False):
    """Σ_k coeffs[p, k] · variables[p, k]（k 为路径列）；skip_zero 时略去系数为 0 的项（只用于两个捕集矩阵，沿用拆分前的写法）。"""
    return gp.quicksum(
        float(coeffs[p, k]) * variables[p, k]
        for k in range(len(PATHWAYS))
        if not skip_zero or float(coeffs[p, k]) != 0.0
    )


class PlantPathTerms(NamedTuple):
    """`_add_plant_path_constraints` 的变量与表达式，字段同 `YearPayload` 的同名字段（表达式的列表每厂一项）。"""

    captured_mt_by_plant: GrbMVar
    biomass_use_gj: GrbMVar
    ammonia_use_kg: GrbMVar
    water_use_m3: GrbMVar
    total_reduction_mt: GrbExpr
    blend_level_b: GrbMVar
    blend_level_a: GrbMVar
    blend_layers_b: list[list[GrbExpr]]
    blend_layers_a: list[list[GrbExpr]]
    bio_penalty_by_plant: list[GrbExpr]
    plant_reduction_exprs: list[GrbExpr]
    air_share: GrbMVar
    air_new: GrbMVar
    biomass_blend_x_share: list[GrbExpr]
    beccs_blend_x_share: list[GrbExpr]
    ammonia_blend_x_share: list[GrbExpr]
    rebuilt_share: RebuiltShares
    rebuilt_air_share: RebuiltShares
    rebuilt_blend_x_share: dict[tuple[int, int, int], GrbExpr]


def _add_plant_path_constraints(
    model,
    share,
    rebuild_class,
    year_data: YearData,
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_suffix: str = "",
) -> PlantPathTerms:
    pathway_count = len(PATHWAYS)
    sfx = f"_{year_suffix}" if year_suffix else ""
    captured_mt_by_plant = model.addMVar(plant_count, lb=0.0, name=f"captured_mt_by_plant{sfx}")
    water_use_m3 = model.addMVar(plant_count, lb=0.0, name=f"water_use_m3{sfx}")

    # 湿冷改空冷。`air_share[p, k]` 是 hub p 发电量中既走路径 k 又采用空冷的部分，
    # 因此 0 <= air_share <= share，hub 的改造比例为 sum_k air_share[p, k]。
    # 按路径分列可让每一项都保持线性：另一种写法是用单一空冷比例去乘
    # 各路径份额，那是双线性的，需要二元变量或 McCormick 包络。它还对应
    # 一个真实的自由度：一个 hub 聚合了约 10 台机组，运营方要选择其中
    # 哪几台同时配捕集岛和空冷凝汽器。
    allow_air = bool(year_data.allow_air_cooling_retrofit) and year_data.air_water_intensity is not None
    # 退役路径不发电，空冷份额的上界为零（2026-10-02 前由已装存量 >= 各路径空冷份额之和压低，存量有富余时可任取）；
    # 不许空冷改造时全为零（2026-10-16 前是等式约束 `air_share_off` / `air_new_off`）。
    air_ub = np.full((plant_count, pathway_count), 1.0 if allow_air else 0.0)
    air_ub[:, PATHWAY_INDEX["retire"]] = 0.0
    air_share = model.addMVar((plant_count, pathway_count), lb=0.0, ub=air_ub, name=f"air_share{sfx}")
    # 本年新建的空冷改造（占仍湿冷装机的份额）：在役（寿命内历年新建之和）>= 运行路径上的空冷份额之和，到寿命退出，
    # capex 计在新建上（`model_linking.add_capacity_vintages`，2026-10-02 起；此前是只增不减、永不到期的已装存量，
    # capex 计在它的增量上）。
    air_new = model.addMVar(plant_count, lb=0.0, ub=GRB.INFINITY if allow_air else 0.0, name=f"air_new{sfx}")

    # 有到期装机的 hub 的份额按重建热耗类拆出原址重建部分（与未重建部分毛热耗不同）。
    rebuilt_share, rebuilt_air_share = add_rebuilt_split(
        model, share, rebuild_class, air_share, year_data.expired_share, year_data.rebuilt_class_share,
        plant_count, allow_air, sfx,
    )
    blend = add_blend_level_constraints(model, share, plant_count, scenario, assumptions, year_data, sfx, rebuilt_share)

    eta = float(scenario.capture_rate)
    plant_reduction_exprs: list[GrbExpr] = []
    ccs_penalty_captured = year_data.ccs_penalty_captured_matrix
    air_penalty_captured = year_data.air_penalty_captured_matrix
    un, ccs, bio, beccs, amm = (PATHWAY_INDEX[k] for k in ("unabated", "ccs", "biomass", "beccs", "ammonia"))

    for plant_idx in range(plant_count):
        E_p = float(year_data.emissions_mt[plant_idx])               # 基准
        E_op = float(year_data.emissions_operating_mt[plant_idx])    # 经效率调整
        E_rt = float(year_data.emissions_retrofit_mt[plant_idx])     # 效率 × CF 提升
        G_bp = year_data.generation_by_pathway[plant_idx, :]
        s_un = share[plant_idx, un]
        s_ccs = share[plant_idx, ccs]
        s_bio = share[plant_idx, bio]
        s_beccs = share[plant_idx, beccs]
        s_amm = share[plant_idx, amm]

        # 用水量（按 WATER_FLOW_SCALE 缩放以保证数值稳定）；各路径发电量
        # （含改造提升，退役 = 0）乘以该路径的用水强度，其中各路径的
        # 空冷部分改按空冷强度计。
        air_intensity = year_data.air_water_intensity if allow_air else None
        water_expr = gp.quicksum(
            float(G_bp[path_idx])
            * (
                float(year_data.water_intensity[plant_idx, path_idx]) * share[plant_idx, path_idx]
                - (
                    float(
                        year_data.water_intensity[plant_idx, path_idx]
                        - air_intensity[plant_idx, path_idx]
                    )
                    * air_share[plant_idx, path_idx]
                    if air_intensity is not None
                    else 0.0
                )
            )
            / WATER_FLOW_SCALE
            for path_idx in range(pathway_count)
        )
        model.addConstr(water_use_m3[plant_idx] == water_expr, name=f"water_use_{plant_idx}{sfx}")
        if allow_air:
            # 退役列的上界为零，那一行是 0 <= 退役份额，不加。
            for path_idx in RUNNING:
                model.addConstr(
                    air_share[plant_idx, path_idx] <= share[plant_idx, path_idx],
                    name=f"air_share_le_share_{plant_idx}_{path_idx}{sfx}",
                )

        # 物理捕集的 CO2 不同于 BECCS 的净减排。除锅炉自身烟气的 eta x 之外，
        # 捕集装置还会捕获同一锅炉里燃烧的每一份惩罚燃料的 eta
        # （CCS 能耗惩罚、BECCS 掺烧惩罚、空冷背压）；这些吨数在残余排放里
        # 按 (1-eta) 计了排放，但从未计入这里，于是它们在账面上被减掉了，
        # 却没有经过输送和封存。
        captured_expr = E_rt * eta * (s_ccs + s_beccs) + blend.beccs_penalty_captured_exprs[plant_idx]
        if ccs_penalty_captured is not None:
            captured_expr = captured_expr + _path_dot(ccs_penalty_captured, share, plant_idx, skip_zero=True)
        if allow_air and air_penalty_captured is not None:
            captured_expr = captured_expr + _path_dot(air_penalty_captured, air_share, plant_idx, skip_zero=True)

        # 先算各路径下的实际（残余）排放，再算相对基准的减排量。
        # CF 提升与效率比都取 1 时，这恰好化简为经典的
        # 各路径减排比例乘以基准排放。
        # 效率惩罚燃料（CCS 固定部分 + 随掺烧档位变化部分）与其他煤一样燃烧并
        # 排放，因此其排放属于残余排放的一部分。
        residual_expr = (
            E_op * s_un
            + E_rt * (1.0 - eta) * s_ccs
            + (E_rt * s_bio - blend.bio_red_exprs[plant_idx])
            + (E_rt * (1.0 - eta) * s_beccs - blend.beccs_blend_red_exprs[plant_idx])
            + (E_rt * s_amm - blend.amm_red_exprs[plant_idx])
            + _path_dot(year_data.ccs_penalty_emissions_matrix, share, plant_idx)
            + blend.bio_penalty_emissions_exprs[plant_idx]
            # 空冷的背压惩罚：与其他煤一样燃烧并排放。
            + (_path_dot(year_data.air_penalty_emissions_matrix, air_share, plant_idx) if allow_air else 0.0)
        )
        for c, r in rebuilt_share.get(plant_idx, {}).items():
            # 重建部分（逐类）：乘路径份额的各项按同式再加一遍，份额换成该类的重建部分 r、系数换成该类与未重建部分之差
            # （乘 Σβz 的掺烧项已含重建部分；空冷背压按 hub 毛热耗计，没有差）。
            delta = year_data.rebuilt_deltas[c]
            d_op = float(delta.emissions_operating_mt[plant_idx])
            d_rt = float(delta.emissions_retrofit_mt[plant_idx])
            captured_expr = captured_expr + (
                d_rt * eta * (r[ccs] + r[beccs]) + rebuilt_dot(delta.ccs_penalty_captured_matrix[plant_idx], r)
            )
            residual_expr = residual_expr + (
                d_op * r[un]
                + d_rt * (1.0 - eta) * (r[ccs] + r[beccs])
                + d_rt * (r[bio] + r[amm])
                + rebuilt_dot(delta.ccs_penalty_emissions_matrix[plant_idx], r)
            )
        model.addConstr(captured_mt_by_plant[plant_idx] == captured_expr, name=f"captured_balance_{plant_idx}{sfx}")
        plant_red = E_p - residual_expr
        plant_reduction_exprs.append(plant_red)

    return PlantPathTerms(
        captured_mt_by_plant=captured_mt_by_plant, biomass_use_gj=blend.biomass_use_gj, ammonia_use_kg=blend.ammonia_use_kg,
        water_use_m3=water_use_m3, total_reduction_mt=gp.quicksum(plant_reduction_exprs),
        blend_level_b=blend.blend_level_b, blend_level_a=blend.blend_level_a,
        blend_layers_b=blend.blend_layers_b, blend_layers_a=blend.blend_layers_a,
        bio_penalty_by_plant=blend.bio_penalty_by_plant, plant_reduction_exprs=plant_reduction_exprs,
        air_share=air_share, air_new=air_new, biomass_blend_x_share=blend.biomass_blend_x_share,
        beccs_blend_x_share=blend.beccs_blend_x_share, ammonia_blend_x_share=blend.ammonia_blend_x_share,
        rebuilt_share=rebuilt_share, rebuilt_air_share=rebuilt_air_share, rebuilt_blend_x_share=blend.rebuilt_blend_x_share,
    )
