from __future__ import annotations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    gp = None
    GRB = None

from .scenario import OptimizationAssumptions, OptimizationScenario, PATHWAYS
from ._shared import PATHWAY_INDEX
from .year_types import YearData, YearPayload

# 导入流量缩放系数，用于数值稳定
from ..constants import AMMONIA_FLOW_SCALE, WATER_FLOW_SCALE, BIOMASS_FLOW_SCALE


def _add_vector_equality(model, lhs, rhs, length: int, name: str) -> None:
    model.addConstrs((lhs[idx] == rhs[idx] for idx in range(length)), name=name)


def _add_vector_upper_bound(model, lhs, rhs, length: int, name: str) -> None:
    model.addConstrs((lhs[idx] <= rhs[idx] for idx in range(length)), name=name)


def _add_mccormick_product(model, share_var, binary_var, name: str):
    """线性化 z = share_var * binary_var，其中 binary 取值 {0,1}，share 取值 [0,1]。"""
    z = model.addVar(lb=0.0, ub=1.0, name=name)
    model.addConstr(z <= share_var,                      name=f"{name}_u1")
    model.addConstr(z <= binary_var,                     name=f"{name}_u2")
    model.addConstr(z >= share_var - (1.0 - binary_var), name=f"{name}_lo")
    return z


def _build_air_retrofit_capex(
    model,
    year_data: YearData,
    payload: YearPayload,
    plant_count: int,
    yr_sfx: str,
    prev_payload: YearPayload | None = None,
):
    """湿冷凝汽器改为空冷的一次性 capex。

    按已装存量的增量计费，而不是按本期改造份额的增量：与 CCS 改造一样，
    空冷凝汽器一旦建成就一直在，所以空冷份额在某一期下降、下一期回升的 hub
    不能付两次钱。存量只设下限（>= 本期改造份额，>= 上期存量），
    这就够了，因为目标是最小化且系数为正。
    """
    if not bool(year_data.allow_air_cooling_retrofit):
        return 0.0
    capex_per_plant = year_data.air_retrofit_capex_per_plant
    if capex_per_plant is None:
        return 0.0
    installed = payload.air_installed
    terms = []
    for plant_idx in range(plant_count):
        coeff = float(capex_per_plant[plant_idx])
        if coeff <= 0:
            continue
        if prev_payload is None:
            terms.append(coeff * installed[plant_idx])
        else:
            previous = prev_payload.air_installed[plant_idx]
            model.addConstr(
                installed[plant_idx] >= previous, name=f"air_installed_mono_{plant_idx}_{yr_sfx}"
            )
            terms.append(coeff * (installed[plant_idx] - previous))
    return gp.quicksum(terms) if terms else 0.0


def _build_blend_upgrade_capex(
    model,
    capacity_mw,
    blend_level_b,
    blend_level_a,
    assumptions: "OptimizationAssumptions",
    plant_count: int,
    sfx: str = "",
    prev_blend_level_b=None,
    prev_blend_level_a=None,
):
    """掺烧升级 capex 表达式。prev_* 为 None 时，视为从档位 0 起步。"""
    cap = np.asarray(capacity_mw, dtype=np.float64)
    coeff_b = cap * assumptions.biomass_upgrade_capex_cny_per_mw_per_level
    coeff_a = cap * assumptions.ammonia_upgrade_capex_cny_per_mw_per_level
    if prev_blend_level_b is None:
        return coeff_b @ blend_level_b + coeff_a @ blend_level_a
    # 用辅助变量实现 max(0, delta)，防止退役电厂把掺烧档位
    # 重置为 0 时出现负 CAPEX
    delta_b_pos = model.addMVar(plant_count, lb=0.0, name=f"blend_delta_b_pos{sfx}")
    delta_a_pos = model.addMVar(plant_count, lb=0.0, name=f"blend_delta_a_pos{sfx}")
    model.addConstr(delta_b_pos >= blend_level_b - prev_blend_level_b, name=f"blend_delta_b_lb{sfx}")
    model.addConstr(delta_a_pos >= blend_level_a - prev_blend_level_a, name=f"blend_delta_a_lb{sfx}")
    return coeff_b @ delta_b_pos + coeff_a @ delta_a_pos


def _add_blend_level_constraints(
    model,
    share,
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_data: YearData,
    sfx: str,
) -> tuple:
    """为每个厂添加掺烧档位二元变量、线性化用的 zeta 以及用量约束。

    Returns
    -------
    select_b : MVar (plant_count, L_b+1)  — 档位 0 = 不掺生物质
    select_a : MVar (plant_count, L_a+1)  — 档位 0 = 不掺氨
    blend_level_b : MVar (plant_count,)   — 数值档位索引（Σ l·select_b）
    blend_level_a : MVar (plant_count,)
    biomass_use_gj : MVar (plant_count,)
    ammonia_use_kg : MVar (plant_count,)
    bio_red_exprs   : list[LinExpr]  — E_retrofit · Σ β_b[l] · zeta_bio[p,l]，每厂一项
    beccs_blend_red_exprs : list[LinExpr]  — E_retrofit · Σ β_b[l]·zeta_beccs[p,l]，每厂一项
        （只含掺烧替代部分；η·share 的捕集部分由调用方加上）
    amm_red_exprs   : list[LinExpr]  — E_retrofit · Σ β_a[l] · zeta_amm[p,l]，每厂一项
    bio_penalty_exprs : list[LinExpr] — 每个厂随掺烧档位变化的能耗惩罚成本
    bio_penalty_emissions_exprs : list[LinExpr] — 惩罚燃料额外排放的 CO2（Mt）
    """
    blend_b = np.asarray(scenario.biomass_blend_levels, dtype=np.float64)
    blend_a = np.asarray(scenario.ammonia_blend_levels, dtype=np.float64)
    L_b = len(blend_b)
    L_a = len(blend_a)
    lhv = float(assumptions.nh3_lhv_gj_per_kg)

    # select_b/select_a：第 0 列 = "不掺烧"，第 l+1 列 = "掺烧档位 l"
    # 独热二元变量（一个 hub 选一个档位），或 hub 容量中改造到各档位的
    # 连续份额；见 `OptimizationAssumptions.hub_decisions_continuous`。
    sel_vtype = GRB.CONTINUOUS if assumptions.hub_decisions_continuous else GRB.BINARY
    select_b = model.addMVar((plant_count, L_b + 1), lb=0.0, ub=1.0, vtype=sel_vtype, name=f"sel_b{sfx}")
    select_a = model.addMVar((plant_count, L_a + 1), lb=0.0, ub=1.0, vtype=sel_vtype, name=f"sel_a{sfx}")
    blend_level_b = model.addMVar(plant_count, lb=0.0, ub=float(L_b), name=f"blv_b{sfx}")
    blend_level_a = model.addMVar(plant_count, lb=0.0, ub=float(L_a), name=f"blv_a{sfx}")
    biomass_use_gj = model.addMVar(plant_count, lb=0.0, name=f"biomass_use_gj{sfx}")
    ammonia_use_kg = model.addMVar(plant_count, lb=0.0, name=f"ammonia_use_kg{sfx}")

    bio_red_exprs: list[object] = []
    beccs_blend_red_exprs: list[object] = []
    amm_red_exprs: list[object] = []
    bio_penalty_exprs: list[object] = []  # 每个厂随掺烧档位变化的能耗惩罚
    bio_penalty_emissions_exprs: list[object] = []  # 惩罚燃料额外产生的 CO2（Mt）

    bio_pen_coeff_data = year_data.biomass_penalty_coeff_per_level
    bio_pen_em_coeff_data = year_data.biomass_penalty_emissions_coeff_per_level
    # BECCS 下，掺烧惩罚燃料与被捕集的烟气在同一台锅炉里燃烧，所以只有
    # 未捕集的份额排放（Fan et al. 2023 SI eq. S42）——而被捕集的份额是
    # 实实在在的吨数，必须经管道输送并封存。
    beccs_pen_em_coeff_data = year_data.beccs_penalty_emissions_coeff_per_level
    beccs_pen_cap_coeff_data = year_data.beccs_penalty_captured_coeff_per_level
    beccs_penalty_captured_exprs: list[object] = []

    for p in range(plant_count):
        # 改造后的运行带有效率比（重建电厂）和改造后的
        # CF 提升；燃料用量还要再按电厂热耗率缩放。
        E_rt = float(year_data.emissions_retrofit_mt[p])
        hr_p = float(year_data.heat_rate_eff[p])
        G_bp = year_data.generation_by_pathway[p, :]
        G_bio = float(G_bp[PATHWAY_INDEX["biomass"]])
        G_beccs = float(G_bp[PATHWAY_INDEX["beccs"]])
        G_amm = float(G_bp[PATHWAY_INDEX["ammonia"]])
        bio_pen_coeff = float(bio_pen_coeff_data[p]) if hasattr(bio_pen_coeff_data, '__getitem__') and not isinstance(bio_pen_coeff_data, (int, float)) else float(bio_pen_coeff_data)
        bio_pen_em_coeff = float(bio_pen_em_coeff_data[p]) if hasattr(bio_pen_em_coeff_data, '__getitem__') and not isinstance(bio_pen_em_coeff_data, (int, float)) else float(bio_pen_em_coeff_data)
        beccs_pen_em_coeff = float(beccs_pen_em_coeff_data[p]) if hasattr(beccs_pen_em_coeff_data, '__getitem__') and not isinstance(beccs_pen_em_coeff_data, (int, float)) else float(beccs_pen_em_coeff_data)
        beccs_pen_cap_coeff = float(beccs_pen_cap_coeff_data[p]) if hasattr(beccs_pen_cap_coeff_data, '__getitem__') and not isinstance(beccs_pen_cap_coeff_data, (int, float)) else float(beccs_pen_cap_coeff_data)
        s_bio = share[p, PATHWAY_INDEX["biomass"]]
        s_beccs = share[p, PATHWAY_INDEX["beccs"]]
        s_amm = share[p, PATHWAY_INDEX["ammonia"]]

        # --- 每类路径恰好选一个掺烧档位 ---
        model.addConstr(select_b[p, :].sum() == 1.0, name=f"sel_b_{p}{sfx}")
        model.addConstr(select_a[p, :].sum() == 1.0, name=f"sel_a_{p}{sfx}")

        # --- 若选档位 0（不掺烧），相关份额必须为 0 ---
        model.addConstr(s_bio + s_beccs <= 1.0 - select_b[p, 0], name=f"nb_b_{p}{sfx}")
        model.addConstr(s_amm <= 1.0 - select_a[p, 0], name=f"nb_a_{p}{sfx}")

        # --- 数值掺烧档位（用于计算升级成本） ---
        model.addConstr(
            blend_level_b[p] == gp.quicksum(level * select_b[p, level] for level in range(L_b + 1)),
            name=f"blv_b_{p}{sfx}",
        )
        model.addConstr(
            blend_level_a[p] == gp.quicksum(level * select_a[p, level] for level in range(L_a + 1)),
            name=f"blv_a_{p}{sfx}",
        )

        # --- 线性化：zeta = select_b[p, l+1] × share（二元变量 × [0,1] 内的连续变量） ---
        bio_use_expr = gp.LinExpr()
        bio_red = gp.LinExpr()
        beccs_blend_red = gp.LinExpr()
        bio_penalty = gp.LinExpr()  # 随掺烧档位变化的能耗惩罚（成本）
        bio_penalty_emissions = gp.LinExpr()  # 同一份惩罚燃料，折为排放的 CO2（Mt）
        beccs_penalty_captured = gp.LinExpr()  # BECCS 惩罚燃料中被捕集的份额（Mt）

        z_bio_all: list[object] = []
        z_beccs_all: list[object] = []
        for level, beta_b in enumerate(blend_b):
            bin_b = select_b[p, level + 1]
            z_bio = _add_mccormick_product(model, s_bio, bin_b, f"zb_{p}_{level}{sfx}")
            z_beccs = _add_mccormick_product(model, s_beccs, bin_b, f"zbc_{p}_{level}{sfx}")
            # 两条路径共用改造到该档位的容量。独热形式下此式自动成立；
            # 档位取份额时它才真正起约束作用，也才有物理含义。
            model.addConstr(z_bio + z_beccs <= bin_b, name=f"zlvl_b_{p}_{level}{sfx}")
            z_bio_all.append(z_bio)
            z_beccs_all.append(z_beccs)
            bio_use_expr += hr_p * beta_b * (G_bio * z_bio + G_beccs * z_beccs) / BIOMASS_FLOW_SCALE
            bio_red  += E_rt * beta_b * z_bio
            beccs_blend_red += E_rt * beta_b * z_beccs
            # 能耗惩罚：β_b × coeff × (G_bio·z_bio + G_beccs·z_beccs)
            bio_penalty += beta_b * bio_pen_coeff * (G_bio * z_bio + G_beccs * z_beccs)
            bio_penalty_emissions += beta_b * (
                bio_pen_em_coeff * G_bio * z_bio + beccs_pen_em_coeff * G_beccs * z_beccs
            )
            beccs_penalty_captured += beta_b * beccs_pen_cap_coeff * G_beccs * z_beccs

        # 一条路径的份额在各档位间至多分摊一次（多个档位份额为正时，
        # 减排量不会被重复计算）。
        model.addConstr(gp.quicksum(z_bio_all) <= s_bio, name=f"zsum_bio_{p}{sfx}")
        model.addConstr(gp.quicksum(z_beccs_all) <= s_beccs, name=f"zsum_beccs_{p}{sfx}")
        model.addConstr(biomass_use_gj[p] == bio_use_expr, name=f"bu_{p}{sfx}")

        amm_use_expr = gp.LinExpr()
        amm_red = gp.LinExpr()
        z_amm_all: list[object] = []
        for level, beta_a in enumerate(blend_a):
            bin_a = select_a[p, level + 1]
            z_amm = _add_mccormick_product(model, s_amm, bin_a, f"za_{p}_{level}{sfx}")
            z_amm_all.append(z_amm)

            amm_use_expr += G_amm * hr_p / lhv * beta_a * z_amm / AMMONIA_FLOW_SCALE
            amm_red  += E_rt * beta_a * z_amm

        model.addConstr(gp.quicksum(z_amm_all) <= s_amm, name=f"zsum_amm_{p}{sfx}")
        model.addConstr(ammonia_use_kg[p] == amm_use_expr, name=f"au_{p}{sfx}")

        bio_red_exprs.append(bio_red)
        beccs_blend_red_exprs.append(beccs_blend_red)
        amm_red_exprs.append(amm_red)
        bio_penalty_exprs.append(bio_penalty)
        bio_penalty_emissions_exprs.append(bio_penalty_emissions)
        beccs_penalty_captured_exprs.append(beccs_penalty_captured)

    return (
        select_b, select_a,
        blend_level_b, blend_level_a,
        biomass_use_gj, ammonia_use_kg,
        bio_red_exprs, beccs_blend_red_exprs, amm_red_exprs,
        bio_penalty_exprs, bio_penalty_emissions_exprs,
        beccs_penalty_captured_exprs,
    )


def _add_plant_path_constraints(
    model,
    share,
    year_data: YearData,
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_suffix: str = "",
):
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
    air_share = model.addMVar((plant_count, pathway_count), lb=0.0, ub=1.0, name=f"air_share{sfx}")
    # 已装存量：capex 按历史最高值计费，已改造 hub 的空冷份额在某一期下降、
    # 下一期回升时不会重复计费。与 CCS 的 `retrofit_installed` 是同一手法；
    # 没有它，空冷凝汽器可能被付两次钱。
    air_installed = model.addMVar(plant_count, lb=0.0, ub=1.0, name=f"air_installed{sfx}")
    if not allow_air:
        model.addConstr(air_share == 0.0, name=f"air_share_off{sfx}")
        model.addConstr(air_installed == 0.0, name=f"air_installed_off{sfx}")
    else:
        model.addConstrs(
            (
                air_installed[plant_idx] >= gp.quicksum(
                    air_share[plant_idx, path_idx] for path_idx in range(pathway_count)
                )
                for plant_idx in range(plant_count)
            ),
            name=f"air_installed_ge_share{sfx}",
        )

    (
        select_b, select_a,
        blend_level_b, blend_level_a,
        biomass_use_gj, ammonia_use_kg,
        bio_red_exprs, beccs_blend_red_exprs, amm_red_exprs,
        bio_penalty_exprs, bio_penalty_emissions_exprs,
        beccs_penalty_captured_exprs,
    ) = _add_blend_level_constraints(model, share, plant_count, scenario, assumptions, year_data, sfx)

    eta = float(scenario.capture_rate)
    plant_reduction_exprs: list[object] = []
    ccs_penalty_captured = year_data.ccs_penalty_captured_matrix
    air_penalty_captured = year_data.air_penalty_captured_matrix

    for plant_idx in range(plant_count):
        E_p = float(year_data.emissions_mt[plant_idx])               # 基准
        E_op = float(year_data.emissions_operating_mt[plant_idx])    # 经效率调整
        E_rt = float(year_data.emissions_retrofit_mt[plant_idx])     # 效率 × CF 提升
        G_bp = year_data.generation_by_pathway[plant_idx, :]
        s_un = share[plant_idx, PATHWAY_INDEX["unabated"]]
        s_ccs = share[plant_idx, PATHWAY_INDEX["ccs"]]
        s_bio = share[plant_idx, PATHWAY_INDEX["biomass"]]
        s_beccs = share[plant_idx, PATHWAY_INDEX["beccs"]]
        s_amm = share[plant_idx, PATHWAY_INDEX["ammonia"]]

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
            for path_idx in range(pathway_count):
                model.addConstr(
                    air_share[plant_idx, path_idx] <= share[plant_idx, path_idx],
                    name=f"air_share_le_share_{plant_idx}_{path_idx}{sfx}",
                )

        # 物理捕集的 CO2 不同于 BECCS 的净减排。除锅炉自身烟气的 eta x 之外，
        # 捕集装置还会捕获同一锅炉里燃烧的每一份惩罚燃料的 eta
        # （CCS 能耗惩罚、BECCS 掺烧惩罚、空冷背压）；这些吨数在残余排放里
        # 按 (1-eta) 计了排放，但从未计入这里，于是它们在账面上被减掉了，
        # 却没有经过输送和封存。
        captured_expr = E_rt * eta * (s_ccs + s_beccs) + beccs_penalty_captured_exprs[plant_idx]
        if ccs_penalty_captured is not None:
            captured_expr = captured_expr + gp.quicksum(
                float(ccs_penalty_captured[plant_idx, path_idx]) * share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
                if float(ccs_penalty_captured[plant_idx, path_idx]) != 0.0
            )
        if allow_air and air_penalty_captured is not None:
            captured_expr = captured_expr + gp.quicksum(
                float(air_penalty_captured[plant_idx, path_idx]) * air_share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
                if float(air_penalty_captured[plant_idx, path_idx]) != 0.0
            )
        model.addConstr(captured_mt_by_plant[plant_idx] == captured_expr, name=f"captured_balance_{plant_idx}{sfx}")

        # 先算各路径下的实际（残余）排放，再算相对基准的减排量。
        # CF 提升与效率比都取 1 时，这恰好化简为经典的
        # 各路径减排比例乘以基准排放。
        # 效率惩罚燃料（CCS 固定部分 + 随掺烧档位变化部分）与其他煤一样燃烧并
        # 排放，因此其排放属于残余排放的一部分。
        residual_expr = (
            E_op * s_un
            + E_rt * (1.0 - eta) * s_ccs
            + (E_rt * s_bio - bio_red_exprs[plant_idx])
            + (E_rt * (1.0 - eta) * s_beccs - beccs_blend_red_exprs[plant_idx])
            + (E_rt * s_amm - amm_red_exprs[plant_idx])
            + gp.quicksum(
                float(year_data.ccs_penalty_emissions_matrix[plant_idx, path_idx]) * share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
            )
            + bio_penalty_emissions_exprs[plant_idx]
            # 空冷的背压惩罚：与其他煤一样燃烧并排放。
            + (
                gp.quicksum(
                    float(year_data.air_penalty_emissions_matrix[plant_idx, path_idx])
                    * air_share[plant_idx, path_idx]
                    for path_idx in range(pathway_count)
                )
                if allow_air
                else 0.0
            )
        )
        plant_red = E_p - residual_expr
        plant_reduction_exprs.append(plant_red)

    total_reduction_mt = gp.quicksum(plant_reduction_exprs)
    total_bio_penalty = gp.quicksum(bio_penalty_exprs)
    return (
        captured_mt_by_plant, biomass_use_gj, ammonia_use_kg, water_use_m3, total_reduction_mt,
        select_b, select_a, blend_level_b, blend_level_a,
        total_bio_penalty, plant_reduction_exprs, air_share, air_installed,
    )
