"""单个规划年的变量与年内约束：路径份额、CO2 管网平衡、资源节点上限、水约束、部门目标。"""
from __future__ import annotations

import numpy as np

from ..constants_industry import POWER_TARGET_GROUP
from ._shared import GRB, PATHWAY_INDEX, PreparedInputs, SolveState, gp
from .constraints import _add_forced_pathway_activation_constraints, _add_plant_path_constraints
from .model_index import ModelIndex
from .model_resources import add_basin_withdrawal_cap, add_resource_balances
from .scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from .year_matrices import _build_year_matrices
from .year_types import GrbExpr, GrbMVar, YearData, YearPayload


def add_year_block(
    model,
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    idx: ModelIndex,
    years: tuple[int, ...],
    year_index: int,
    state: SolveState,
) -> YearPayload:
    """建立 `years[year_index]` 这一年的全部变量与年内约束，返回 payload 供跨期约束与成本使用。

    各子系统按下面的先后加入模型；变量与约束的加入顺序决定模型指纹，调换会让新旧求解无法逐字节对照。
    """
    from .industry import add_industry_year

    year = years[year_index]
    plant_count, storage_count, edge_count = idx.plant_count, idx.storage_count, idx.edge_count
    interval_years = scenario.interval_years(years, year_index, assumptions)
    year_data = _build_year_matrices(prepared, scenario, assumptions, year, state)
    ammonia_node_count = len(year_data.ammonia_nodes)
    water_node_count = len(year_data.water_nodes)
    biomass_link_count = len(prepared.biomass_links)
    ammonia_link_count = len(year_data.ammonia_links)
    water_link_count = len(year_data.water_links)
    year_suffix = str(year)

    # --- 变量 ---
    share = model.addMVar((plant_count, len(PATHWAYS)), lb=0.0, name=f"share_{year_suffix}")
    # 到期机组原址重建（二元）或重建份额（连续），见 `hub_decisions_continuous`。
    rebuild = model.addMVar(
        plant_count, lb=0.0, ub=1.0,
        vtype=GRB.CONTINUOUS if assumptions.hub_decisions_continuous else GRB.BINARY,
        name=f"rebuild_{year_suffix}",
    )
    co2_flow_fwd = model.addMVar(edge_count, lb=0.0, name=f"co2_flow_fwd_{year_suffix}")
    co2_flow_bwd = model.addMVar(edge_count, lb=0.0, name=f"co2_flow_bwd_{year_suffix}")
    co2_node_outflow = model.addMVar(idx.n_nodes, lb=-GRB.INFINITY, name=f"co2_node_outflow_{year_suffix}")
    # build_edge 是"已建成"的锁存标志；add_cap 是"本期新增容量"。
    build_edge = model.addMVar(edge_count, vtype=GRB.BINARY, name=f"build_edge_{year_suffix}")
    add_cap = model.addMVar(edge_count, vtype=GRB.BINARY, name=f"add_cap_{year_suffix}")
    # 容量按管径档整根新增，`new_cap_mtpa` 是派生的连续总量。
    pipe_tiers = tuple(float(t) for t in year_data.pipe_tiers_mtpa)
    pipe_count = model.addMVar(
        (edge_count, len(pipe_tiers)), vtype=GRB.INTEGER, lb=0.0,
        ub=float(assumptions.max_parallel_pipes), name=f"pipe_count_{year_suffix}",
    )
    new_cap_mtpa = model.addMVar(edge_count, lb=0.0, name=f"new_cap_mtpa_{year_suffix}")
    model.addConstrs(
        (
            new_cap_mtpa[e] == gp.quicksum(pipe_tiers[k] * pipe_count[e, k] for k in range(len(pipe_tiers)))
            for e in range(edge_count)
        ),
        name=f"new_cap_from_tiers_{year_suffix}",
    )
    biomass_flow_gj = model.addMVar(biomass_link_count, lb=0.0, name=f"biomass_flow_gj_{year_suffix}")
    ammonia_flow_kg = model.addMVar(ammonia_link_count, lb=0.0, name=f"ammonia_flow_kg_{year_suffix}")
    water_flow_m3 = model.addMVar(water_link_count, lb=0.0, name=f"water_flow_m3_{year_suffix}")
    # 总缺口 = 各组缺口之和，老读者继续读标量。
    target_shortfall_mt = model.addVar(lb=0.0, name=f"target_shortfall_mt_{year_suffix}")
    biomass_slack_gj = model.addMVar(idx.biomass_node_count, lb=0.0, name=f"biomass_slack_gj_{year_suffix}")
    ammonia_slack_kg = model.addMVar(ammonia_node_count, lb=0.0, name=f"ammonia_slack_kg_{year_suffix}")
    water_slack_m3 = model.addMVar(water_node_count, lb=0.0, name=f"water_slack_m3_{year_suffix}")
    injectivity_slack_mtpa = model.addMVar(storage_count, lb=0.0, name=f"injectivity_slack_mtpa_{year_suffix}")
    storage_slack_mt = model.addMVar(storage_count, lb=0.0, name=f"storage_slack_mt_{year_suffix}")
    edge_slack_mtpa = model.addMVar(edge_count, lb=0.0, name=f"edge_slack_mtpa_{year_suffix}")
    edge_flow_mtpa = model.addMVar(edge_count, lb=0.0, name=f"edge_flow_mtpa_{year_suffix}")
    storage_use_mtpa = model.addMVar(storage_count, lb=0.0, name=f"storage_use_mtpa_{year_suffix}")

    (
        captured_mt_by_plant, biomass_use_gj, ammonia_use_kg, water_use_m3, total_reduction_mt,
        select_b, select_a, blend_level_b, blend_level_a,
        total_bio_penalty, plant_reduction_exprs, air_share, air_installed,
    ) = _add_plant_path_constraints(
        model, share, year_data, plant_count, scenario, assumptions, year_suffix=year_suffix,
    )
    model.addConstrs((share[plant_idx, :].sum() == 1.0 for plant_idx in range(plant_count)), name=f"share_sum_{year_suffix}")
    retrofit_installed = _add_retrofit_stock(model, share, plant_count, year_suffix)
    _add_expiry_rules(model, share, rebuild, idx.retirement_years, year, plant_count, year_suffix)

    # --- CO2 管网节点平衡；工业捕集在 hub 自己的节点进入同一张图，所以工业年块建在两段之间 ---
    _add_co2_power_storage_balance(
        model, prepared, idx, co2_node_outflow, co2_flow_fwd, co2_flow_bwd,
        captured_mt_by_plant, storage_use_mtpa, year_suffix,
    )
    industry_payload = add_industry_year(
        model, prepared.industry, year_data.industry, year_suffix,
        h2_link_cost=year_data.industry_h2_link_cost_cny_per_kg,
        h2_hub_membership=year_data.industry_h2_hub_membership,
    )
    _add_co2_industry_pipe_balance(
        model, prepared, idx, co2_node_outflow, co2_flow_fwd, co2_flow_bwd,
        industry_payload["captured_by_hub"], edge_flow_mtpa, year_suffix,
    )

    # --- 生物质 / 氨 / 水：链路平衡与节点上限；流域取水指标 ---
    add_resource_balances(
        model, year_data, assumptions, year, plant_count, idx.biomass_node_count,
        biomass_flow_gj=biomass_flow_gj, ammonia_flow_kg=ammonia_flow_kg, water_flow_m3=water_flow_m3,
        biomass_use_gj=biomass_use_gj, ammonia_use_kg=ammonia_use_kg, water_use_m3=water_use_m3,
        biomass_slack_gj=biomass_slack_gj, ammonia_slack_kg=ammonia_slack_kg, water_slack_m3=water_slack_m3,
        industry_h2_flow_kg=industry_payload["h2_flow_kg"], year_suffix=year_suffix,
    )
    water_basin_slack_m3, water_basin_use_m3 = add_basin_withdrawal_cap(
        model, year_data, share, air_share, industry_payload["withdrawal_by_hub_scaled"],
        plant_count, year_suffix,
    )

    # --- 注入速率、部门目标、路径开关 ---
    _add_injectivity_limit(model, year_data, storage_use_mtpa, injectivity_slack_mtpa, storage_count, year_suffix)
    target_shortfall_by_group = _add_sector_targets(
        model, year_data, scenario, idx, year, total_reduction_mt,
        industry_payload["residual_by_group"], target_shortfall_mt, year_suffix,
    )
    _add_pathway_switches(model, share, scenario, year_data, idx, year, plant_count, year_suffix)

    return YearPayload(
        year=year,
        interval_years=interval_years,
        year_data=year_data,
        share=share,
        co2_flow_fwd=co2_flow_fwd,
        co2_flow_bwd=co2_flow_bwd,
        build_edge=build_edge,
        add_cap=add_cap,
        pipe_count=pipe_count,
        rebuild=rebuild,
        retrofit_installed=retrofit_installed,
        new_cap_mtpa=new_cap_mtpa,
        biomass_flow_gj=biomass_flow_gj,
        ammonia_flow_kg=ammonia_flow_kg,
        water_flow_m3=water_flow_m3,
        target_shortfall_mt=target_shortfall_mt,
        target_shortfall_by_group=target_shortfall_by_group,
        biomass_slack_gj=biomass_slack_gj,
        ammonia_slack_kg=ammonia_slack_kg,
        water_slack_m3=water_slack_m3,
        water_basin_slack_m3=water_basin_slack_m3,
        water_basin_use_m3=water_basin_use_m3,
        injectivity_slack_mtpa=injectivity_slack_mtpa,
        storage_slack_mt=storage_slack_mt,
        edge_slack_mtpa=edge_slack_mtpa,
        edge_flow_mtpa=edge_flow_mtpa,
        storage_use_mtpa=storage_use_mtpa,
        captured_mt_by_plant=captured_mt_by_plant,
        biomass_use_gj=biomass_use_gj,
        ammonia_use_kg=ammonia_use_kg,
        water_use_m3=water_use_m3,
        air_share=air_share,
        air_installed=air_installed,
        select_b=select_b,
        select_a=select_a,
        blend_level_b=blend_level_b,
        blend_level_a=blend_level_a,
        plant_reduction_exprs=plant_reduction_exprs,
        total_reduction_mt=total_reduction_mt,
        total_bio_penalty=total_bio_penalty,
        industry=industry_payload,
    )


def _add_retrofit_stock(model, share: GrbMVar, plant_count: int, year_suffix: str) -> GrbMVar:
    """改造存量（一次性 capex 的计费基数），两列：

      j=0 捕集岛 >= share_ccs + share_beccs，按 CCS capex 计价
      j=1 BECCS 增量 >= share_beccs，按 (BECCS - CCS) capex 计价

    CCS↔BECCS 切换只为捕集岛付一次钱。存量只设下界（跨期单调在 `model_linking`），capex 计在存量增量上，
    份额暂时下降不会重复触发。两列的 capex 系数在 `YearData.retrofit_stock_capex`。
    """
    ccs_k, beccs_k = PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]
    retrofit_installed = model.addMVar(
        (plant_count, 2), lb=0.0, name=f"retrofit_installed_{year_suffix}"
    )
    model.addConstrs(
        (retrofit_installed[p, 0] >= share[p, ccs_k] + share[p, beccs_k] for p in range(plant_count)),
        name=f"retrofit_installed_lb_capture_{year_suffix}",
    )
    model.addConstrs(
        (retrofit_installed[p, 1] >= share[p, beccs_k] for p in range(plant_count)),
        name=f"retrofit_installed_lb_beccs_{year_suffix}",
    )
    return retrofit_installed


def _add_expiry_rules(
    model,
    share: GrbMVar,
    rebuild: GrbMVar,
    retirement_years: np.ndarray,
    year: int,
    plant_count: int,
    year_suffix: str,
) -> None:
    """到期机组：退役或原址重建；未到期机组不能重建。"""
    retire_idx = PATHWAY_INDEX["retire"]
    for plant_idx in range(plant_count):
        if year >= retirement_years[plant_idx]:
            model.addConstr(
                share[plant_idx, retire_idx] >= 1.0 - rebuild[plant_idx],
                name=f"expire_retire_or_rebuild_{plant_idx}_{year_suffix}",
            )
        else:
            model.addConstr(rebuild[plant_idx] == 0, name=f"no_rebuild_{plant_idx}_{year_suffix}")


def _add_co2_power_storage_balance(
    model,
    prepared: PreparedInputs,
    idx: ModelIndex,
    co2_node_outflow: GrbMVar,
    co2_flow_fwd: GrbMVar,
    co2_flow_bwd: GrbMVar,
    captured_mt_by_plant: GrbMVar,
    storage_use_mtpa: GrbMVar,
    year_suffix: str,
) -> None:
    """节点净流出的定义；煤电节点净流出 = 捕集量，封存节点净流出 = -注入量。"""
    B = idx.incidence
    model.addConstr(
        co2_node_outflow == B @ co2_flow_fwd - B @ co2_flow_bwd,
        name=f"co2_flow_define_{year_suffix}",
    )
    for p_idx, plant_id in enumerate(idx.plant_ids):
        if plant_id in prepared.network.plant_node_ids:
            mapped_node = prepared.network.plant_node_ids[plant_id]
            n_idx = idx.node_idx_dict.get(mapped_node)
            if n_idx is None:
                raise ValueError(f"plant_node_ids maps plant {plant_id!r} → {mapped_node!r} which is not in network nodes")
            model.addConstr(
                co2_node_outflow[n_idx] == captured_mt_by_plant[p_idx],
                name=f"co2_plant_inject_{p_idx}_{year_suffix}",
            )
    for s_idx, storage_id in enumerate(idx.storage_ids):
        if storage_id in prepared.network.storage_node_ids:
            mapped_node = prepared.network.storage_node_ids[storage_id]
            n_idx = idx.node_idx_dict.get(mapped_node)
            if n_idx is None:
                raise ValueError(f"storage_node_ids maps hub {storage_id!r} → {mapped_node!r} which is not in network nodes")
            model.addConstr(
                co2_node_outflow[n_idx] == -storage_use_mtpa[s_idx],
                name=f"co2_storage_absorb_{s_idx}_{year_suffix}",
            )


def _add_co2_industry_pipe_balance(
    model,
    prepared: PreparedInputs,
    idx: ModelIndex,
    co2_node_outflow: GrbMVar,
    co2_flow_fwd: GrbMVar,
    co2_flow_bwd: GrbMVar,
    industry_captured_by_hub: list[GrbExpr],
    edge_flow_mtpa: GrbMVar,
    year_suffix: str,
) -> None:
    """工业 hub 节点净流出 = 捕集量（形式与煤电相同）；管道中间节点守恒；边流量 = 正向 + 反向。"""
    for hub_idx, hub_id in enumerate(idx.industry_hub_ids):
        n_idx = idx.node_idx_dict[prepared.network.industry_node_ids[hub_id]]
        model.addConstr(
            co2_node_outflow[n_idx] == industry_captured_by_hub[hub_idx],
            name=f"co2_industry_inject_{hub_idx}_{year_suffix}",
        )
    model.addConstrs(
        (co2_node_outflow[i] == 0.0 for i in idx.pipeline_indices),
        name=f"co2_pipeline_balance_{year_suffix}",
    )
    model.addConstrs(
        (edge_flow_mtpa[e] == co2_flow_fwd[e] + co2_flow_bwd[e] for e in range(idx.edge_count)),
        name=f"edge_flow_define_{year_suffix}",
    )


def _add_injectivity_limit(
    model,
    year_data: YearData,
    storage_use_mtpa: GrbMVar,
    injectivity_slack_mtpa: GrbMVar,
    storage_count: int,
    year_suffix: str,
) -> None:
    """本年可用注入速率 = 可建速率 x 部署进度。"""
    injectivity_year = np.asarray(year_data.storage_injectivity_mtpa, dtype=np.float64)
    model.addConstrs(
        (
            storage_use_mtpa[storage_idx]
            <= float(injectivity_year[storage_idx]) + injectivity_slack_mtpa[storage_idx]
            for storage_idx in range(storage_count)
        ),
        name=f"injectivity_limit_{year_suffix}",
    )


def _add_sector_targets(
    model,
    year_data: YearData,
    scenario: OptimizationScenario,
    idx: ModelIndex,
    year: int,
    total_reduction_mt: GrbExpr,
    industry_residual_by_group: dict[str, GrbExpr],
    target_shortfall_mt: gp.Var,
    year_suffix: str,
) -> dict[str, gp.Var]:
    """部门残余排放上限：residual_g(y) <= cap_fraction_g(y) x baseline_g(2030) + shortfall_g(y)。

    煤电整体为 power 组；工业 hub 经 SECTOR_TARGET_GROUP 映射到 steel / cement / chemicals。
    返回各组缺口变量；总缺口 `target_shortfall_mt` 约束为其和。
    """
    caps = year_data.sector_cap_fraction
    target_shortfall_by_group: dict[str, gp.Var] = {}
    residual_exprs: dict[str, object] = {
        POWER_TARGET_GROUP: float(year_data.emissions_mt.sum()) - total_reduction_mt
    }
    residual_exprs.update(industry_residual_by_group)
    for group, residual in residual_exprs.items():
        if group not in caps:
            raise ValueError(
                f"sector_targets_{scenario.sector_target_source}.csv has no {year} cap for group {group!r}"
            )
        shortfall = model.addVar(lb=0.0, name=f"target_shortfall_{group}_{year_suffix}")
        target_shortfall_by_group[group] = shortfall
        model.addConstr(
            residual - shortfall <= float(caps[group]) * float(idx.sector_base_2030[group]),
            name=f"sector_target_{group}_{year_suffix}",
        )
    model.addConstr(
        target_shortfall_mt == gp.quicksum(target_shortfall_by_group.values()),
        name=f"target_shortfall_total_{year_suffix}",
    )
    return target_shortfall_by_group


def _add_pathway_switches(
    model,
    share: GrbMVar,
    scenario: OptimizationScenario,
    year_data: YearData,
    idx: ModelIndex,
    year: int,
    plant_count: int,
    year_suffix: str,
) -> None:
    """情景关掉的路径份额为零；强制激活的路径按 `_add_forced_pathway_activation_constraints`。"""
    for pathway, pathway_idx in PATHWAY_INDEX.items():
        if not scenario.path_enabled(pathway):
            model.addConstrs(
                (share[plant_idx, pathway_idx] == 0.0 for plant_idx in range(plant_count)),
                name=f"disable_{pathway}_{year_suffix}",
            )
    _add_forced_pathway_activation_constraints(
        model, share, scenario, year_data, plant_count,
        name_suffix=f"_{year_suffix}",
        retired_mask=np.array([year >= idx.retirement_years[p] for p in range(plant_count)]),
    )
