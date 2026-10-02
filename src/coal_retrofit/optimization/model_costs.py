"""一年的成本表达式：运行成本、资源采购、运输封存、一次性 capex、松弛惩罚，及残值台账。

成本口径（作者决定 2026-09-22）：改造 capex 一次性计 + 固定运维 + 能耗按模型自身煤价电价，期末对未折旧 capex
计残值（`salvage.py`）。不用平准化每吨成本。捕集岛与工业路线能力的 capex 计在本年新建量上，按建设年分代、到寿命
退出（`vintage`，2026-09-30 起）；掺烧升级、空冷与原址重建计在存量增量上，搁浅资产计在新增提前退役上（`retirement`）。
"""
from __future__ import annotations

import numpy as np

from ._shared import (
    _COST_SCALE,
    PreparedInputs,
    _discount_factor,
    _year_objective_weight,
    gp,
)
from .constraints import _build_air_retrofit_capex, _build_blend_upgrade_capex, _rebuilt_dot
from .retirement import add_retirement_rate_limit, retirement_flows
from .scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from .year_types import GrbExpr, YearPayload


def add_year_costs(
    model,
    payload: YearPayload,
    year_payloads: list[YearPayload],
    year_position: int,
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    plant_count: int,
    edge_count: int,
    storage_count: int,
) -> None:
    """写入 payload 的 `cost_exprs`、`salvage_ledger`、`objective_expr`。

    年度项（运行、资源、运输封存、松弛）乘折现 x 年金权重，一次性 capex 只乘折现。
    `cost_exprs` 的键序即目标函数的求和顺序；辅助变量与约束的加入顺序决定模型指纹，两者都不要调换。
    """
    from .industry import CCS as _IND_CCS, H2 as _IND_H2, industry_capex_expr

    interval_weight = _year_objective_weight(int(payload.interval_years), scenario.discount_rate)
    df = _discount_factor(int(payload.year), scenario.discount_base_year, scenario.discount_rate)
    prev_payload = None if year_position == 0 else year_payloads[year_position - 1]

    annual = {
        **_operating_costs(payload, plant_count),
        **_resource_costs(payload, prepared),
        **_transport_storage_costs(payload, prepared, edge_count, storage_count),
    }
    early_new, rebuilt_new = retirement_flows(model, payload, prev_payload, plant_count)
    one_off = _one_off_capex(model, payload, prev_payload, scenario, assumptions, early_new, plant_count, edge_count)
    add_retirement_rate_limit(model, payload, scenario, early_new, rebuilt_new, plant_count)

    payload.cost_exprs = {name: df * interval_weight * expr / _COST_SCALE for name, expr in annual.items()}
    payload.cost_exprs.update({name: df * expr / _COST_SCALE for name, expr in one_off.items()})
    payload.cost_exprs["slack_penalty"] = df * interval_weight * _slack_penalty(payload, assumptions) / _COST_SCALE
    # 本年未折现 capex 及其经济寿命，供期末残值。搁浅资产是损失不是资产，不入台账。
    payload.salvage_ledger = [
        ("ccs_retrofit_capex", one_off["ccs_retrofit_capex"], int(assumptions.ccs_retrofit_lifetime_years)),
        ("pipe_capex", one_off["pipe_capex"], int(assumptions.pipeline_lifetime_years)),
        ("blend_upgrade_capex", one_off["blend_upgrade_capex"], int(assumptions.blend_upgrade_lifetime_years)),
        ("air_retrofit_capex", one_off["air_retrofit_capex"], int(assumptions.air_retrofit_lifetime_years)),
        ("rebuild_capex", one_off["rebuild_capex"], int(assumptions.rebuild_lifetime_years)),
    ]
    # 工业：年度部分（捕集能耗与耗材、在役且在用的捕集能力按建设年单价的固定运维、氢路线固定运维与非氢运行差额、
    # 按链路买的氢）用同一折现/年金权重；改造 capex 一次性计在本年新建能力上，与煤电 `ccs_retrofit_capex` 同法。
    assert payload.industry_ccs is not None, "add_capacity_vintages must run before add_year_costs"
    payload.cost_exprs["industry_cost"] = (
        df * interval_weight * (payload.industry.annual_cost_expr + gp.quicksum(payload.industry_ccs.fixed_om))
        / _COST_SCALE
    )
    ind_lives = payload.industry.year_data.capex_lifetime_years
    ind_ccs_capex = industry_capex_expr(payload.industry, routes=(_IND_CCS,))
    ind_h2_capex = industry_capex_expr(payload.industry, routes=(_IND_H2,))
    payload.cost_exprs["industry_capex"] = (
        df * (ind_ccs_capex + ind_h2_capex) / _COST_SCALE
    )
    payload.salvage_ledger += [
        ("industry_ccs_capex", ind_ccs_capex, int(ind_lives[_IND_CCS])),
        ("industry_h2_capex", ind_h2_capex, int(ind_lives[_IND_H2])),
    ]
    payload.objective_expr = gp.quicksum(list(payload.cost_exprs.values()))


def _operating_costs(payload: YearPayload, plant_count: int) -> dict[str, GrbExpr]:
    """煤电运行项与碳成本（CNY/yr，未折现未缩放），键与 `cost_exprs` 同名同序。"""
    year_data = payload.year_data

    # 基线净运行成本（煤 + 运维 - 电）：未改造按基线发电量，改造路径含 CF 提升，退役为零。
    baseline_net = gp.quicksum(
        float(year_data.baseline_net_matrix[p, k]) * payload.share[p, k]
        for p in range(plant_count) for k in range(len(PATHWAYS))
    )

    # 碳成本：碳价 x 残余排放，煤电与工业都计。碳价为零时是 0.0。
    carbon_price_t = float(year_data.carbon_price)
    carbon_cost = 0.0
    if carbon_price_t > 0:
        carbon_cost = carbon_price_t * 1e6 * gp.quicksum(
            float(year_data.emissions_mt[p]) - payload.plant_reduction_exprs[p]
            for p in range(plant_count)
        )
        carbon_cost = carbon_cost + carbon_price_t * 1e6 * gp.quicksum(
            payload.industry.residual_by_group.values()
        )

    # 掺烧替代的煤（负成本）：baseline_net 对所有运行路径按全煤热耗计费，所以生物质与氨都要返还。
    coal_savings_vec = year_data.coal_savings_per_gj
    coal_savings = gp.quicksum(
        float(coal_savings_vec[p]) * payload.biomass_use_gj[p] for p in range(plant_count)
    )
    nh3_savings_vec = year_data.coal_savings_per_kg_nh3
    if nh3_savings_vec is not None:
        coal_savings = coal_savings + gp.quicksum(
            float(nh3_savings_vec[p]) * payload.ammonia_use_kg[p] for p in range(plant_count)
        )

    # 路径增量运维。
    incremental_om = gp.quicksum(
        float(year_data.fixed_cost_matrix[p, k]) * payload.share[p, k]
        for p in range(plant_count) for k in range(len(PATHWAYS))
    )

    # 能耗惩罚：CCS 固定项 + 空冷背压项（仅转换份额）；生物质档位相关项在 constraints 里线性化。
    energy_penalty_cost = gp.quicksum(
        float(year_data.energy_penalty_matrix[p, k]) * payload.share[p, k]
        for p in range(plant_count) for k in range(len(PATHWAYS))
    )
    if year_data.air_penalty_cost_matrix is not None and bool(
        year_data.allow_air_cooling_retrofit
    ):
        energy_penalty_cost = energy_penalty_cost + gp.quicksum(
            float(year_data.air_penalty_cost_matrix[p, k]) * payload.air_share[p, k]
            for p in range(plant_count) for k in range(len(PATHWAYS))
        )
    # 上面三项的系数按未重建部分的毛热耗算；部分到期 hub 的重建部分再加两部分之差 x 重建部分的份额
    # （`constraints._add_rebuilt_split`）。掺烧惩罚的差已在 `total_bio_penalty` 里。
    delta = year_data.rebuilt_delta
    baseline_net = baseline_net + gp.quicksum(
        _rebuilt_dot(delta.baseline_net_matrix[p], part) for p, part in payload.rebuilt_share.items()
    )
    energy_penalty_cost = energy_penalty_cost + gp.quicksum(
        _rebuilt_dot(delta.energy_penalty_matrix[p], part) for p, part in payload.rebuilt_share.items()
    ) + gp.quicksum(
        _rebuilt_dot(delta.air_penalty_cost_matrix[p], part) for p, part in payload.rebuilt_air_share.items()
    )
    # 捕集岛固定运维：在役且在用的捕集岛 x 建设年的单价（`vintage`，闲置不付）。2026-09-30 前按当年单价 x 捕集份额计。
    assert payload.ccs_island is not None, "add_capacity_vintages must run before add_year_costs"
    ccs_om_cost = gp.quicksum(payload.ccs_island.fixed_om)
    return {
        "baseline_net_cost": baseline_net,
        "carbon_cost": carbon_cost,
        "coal_savings_credit": -coal_savings,
        "energy_penalty_cost": energy_penalty_cost + payload.total_bio_penalty,
        "ccs_om_cost": ccs_om_cost,
        "incremental_om": incremental_om,
    }


def _resource_costs(payload: YearPayload, prepared: PreparedInputs) -> dict[str, GrbExpr]:
    """生物质、氨、水的链路采购成本（CNY/yr，未折现未缩放）。"""
    year_data = payload.year_data
    biomass_cost = gp.quicksum(
        float(year_data.biomass_link_cost_cny_per_gj[link_idx]) * payload.biomass_flow_gj[link_idx]
        for link_idx in range(len(prepared.biomass_links))
    )
    ammonia_cost = gp.quicksum(
        float(year_data.ammonia_link_cost_cny_per_kg[link_idx]) * payload.ammonia_flow_kg[link_idx]
        for link_idx in range(len(year_data.ammonia_links))
    )
    water_link_cost = year_data.water_link_cost_cny_per_m3
    water_cost = gp.quicksum(
        float(water_link_cost[link_idx]) * payload.water_flow_m3[link_idx]
        for link_idx in range(len(year_data.water_links))
    ) if len(water_link_cost) > 0 else 0.0
    return {"biomass_cost": biomass_cost, "ammonia_cost": ammonia_cost, "water_cost": water_cost}


def _transport_storage_costs(
    payload: YearPayload, prepared: PreparedInputs, edge_count: int, storage_count: int
) -> dict[str, GrbExpr]:
    """CO2 运输运维与封存（CNY/yr，未折现未缩放）。边系数（CNY/Mt）已含长度、单位运维率与海上倍率。"""
    edge_opex_coeff = payload.year_data.edge_route_opex_coeff
    transport_opex = gp.quicksum(
        float(edge_opex_coeff[e])
        * (payload.co2_flow_fwd[e] + payload.co2_flow_bwd[e])
        for e in range(edge_count)
    )
    storage_cost_lookup = prepared.storages["storage_cost_cny_per_t"].astype(float).to_numpy()
    storage_cost = gp.quicksum(
        float(storage_cost_lookup[s]) * 1_000_000.0 * payload.storage_use_mtpa[s]
        for s in range(storage_count)
    )
    return {"transport_opex": transport_opex, "storage_cost": storage_cost}


def _one_off_capex(
    model,
    payload: YearPayload,
    prev_payload: YearPayload | None,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    early_new: list[GrbExpr],
    plant_count: int,
    edge_count: int,
) -> dict[str, GrbExpr]:
    """一次性 capex（CNY，未折现未缩放），键与 `cost_exprs` 同名同序；逐厂搁浅资产另写入 `payload.stranded_by_plant`。

    非首年按增量计：原址重建与掺烧升级为增量加辅助变量与约束，空冷改造只加存量单调约束（后两者在
    `constraints` 里加）；搁浅资产计在新增提前退役 `early_new` 上（`retirement.retirement_flows`）。这些加入模型的
    先后决定模型指纹。
    """
    year_data = payload.year_data
    yr_sfx = str(payload.year)
    capacity_mw = year_data.capacity_mw
    expired_share = year_data.expired_share

    # 管道 capex：各管径档整根计。
    tier_capex = np.asarray(year_data.edge_tier_capex, dtype=np.float64)
    pipe_capex = gp.quicksum(
        float(tier_capex[edge_idx, k]) * payload.pipe_count[edge_idx, k]
        for edge_idx in range(edge_count) for k in range(tier_capex.shape[1])
    )

    # 捕集岛：本年单价 x 本年新建（`vintage`；2026-09-30 前计在单调存量的增量上）。
    stock_coeff = np.asarray(year_data.retrofit_stock_capex, dtype=np.float64)
    ccs_retrofit_capex = gp.quicksum(
        float(stock_coeff[p, j]) * payload.retrofit_new[p, j]
        for j in range(stock_coeff.shape[1]) for p in range(plant_count)
        if float(stock_coeff[p, j]) > 0.0
    )

    # 搁浅资产：每单位新增提前退役的剩余账面价值（`plant_matrices`）x 新增提前退役。到期退役不计；重建后退役也不计，
    # 它的 capex 已在目标函数里，期末不计残值（`retirement.add_retired_rebuild_offset`）。
    payload.stranded_by_plant = [
        float(year_data.stranded_per_plant[p]) * early_new[p] if float(year_data.stranded_per_plant[p]) > 0 else 0.0
        for p in range(plant_count)
    ]
    stranded_capex = gp.quicksum(payload.stranded_by_plant)

    # 掺烧升级、空冷改造。
    if prev_payload is None:
        blend_upgrade_capex = _build_blend_upgrade_capex(
            model, capacity_mw, payload.blend_level_b, payload.blend_level_a,
            assumptions, plant_count, sfx=f"_{yr_sfx}",
        )
        air_retrofit_capex = _build_air_retrofit_capex(
            model, year_data, payload, plant_count, yr_sfx, prev_payload=None
        )
    else:
        blend_upgrade_capex = _build_blend_upgrade_capex(
            model, capacity_mw, payload.blend_level_b, payload.blend_level_a,
            assumptions, plant_count,
            prev_blend_level_b=prev_payload.blend_level_b,
            prev_blend_level_a=prev_payload.blend_level_a,
            sfx=f"_{yr_sfx}",
        )
        air_retrofit_capex = _build_air_retrofit_capex(
            model, year_data, payload, plant_count, yr_sfx, prev_payload=prev_payload
        )

    # 原址重建 capex：容量 x 新建成本比例 x 重建份额的增量（rebuild 锁存）；只有本年有到期装机的 hub 能重建
    # （`model_year._add_expiry_rules`），2026-10-02 前按整个 hub 到期计。
    rebuild_cost_per_mw = assumptions.stranded_asset_base_cny_per_kw * scenario.rebuild_capex_fraction * 1000.0
    if prev_payload is None:
        rebuild_capex = gp.quicksum(
            float(capacity_mw[p]) * rebuild_cost_per_mw * payload.rebuild[p]
            for p in range(plant_count)
            if float(expired_share[p]) > 0.0
        )
    else:
        rebuild_capex_terms = []
        for p in range(plant_count):
            if float(expired_share[p]) <= 0.0:
                continue
            delta_rebuild = model.addVar(lb=0.0, name=f"rebuild_delta_{p}_{yr_sfx}")
            model.addConstr(
                delta_rebuild >= payload.rebuild[p] - prev_payload.rebuild[p],
                name=f"rebuild_delta_lb_{p}_{yr_sfx}",
            )
            rebuild_capex_terms.append(float(capacity_mw[p]) * rebuild_cost_per_mw * delta_rebuild)
        rebuild_capex = gp.quicksum(rebuild_capex_terms) if rebuild_capex_terms else 0.0

    return {
        "stranded_capex": stranded_capex,
        "ccs_retrofit_capex": ccs_retrofit_capex,
        "pipe_capex": pipe_capex,
        "blend_upgrade_capex": blend_upgrade_capex,
        "air_retrofit_capex": air_retrofit_capex,
        "rebuild_capex": rebuild_capex,
    }


def _slack_penalty(payload: YearPayload, assumptions: OptimizationAssumptions) -> GrbExpr:
    """松弛惩罚（CNY/yr，资源松弛按缩放单位乘回）。流域指标松弛与节点松弛同价，两个制度不因数值原因分高下。"""
    year_data = payload.year_data
    _amm_scale = float(year_data.ammonia_flow_scale)
    _wat_scale = float(year_data.water_flow_scale)
    _bio_scale = float(year_data.biomass_flow_scale)
    return (
        payload.target_shortfall_mt * assumptions.slack_penalty_cny_per_unit
        + payload.biomass_slack_gj.sum() * 2_000.0 * _bio_scale
        + payload.ammonia_slack_kg.sum() * 1_000.0 * _amm_scale
        + payload.water_slack_m3.sum() * 1_000.0 * _wat_scale
        + (payload.water_basin_slack_m3.sum() * 1_000.0 * _wat_scale
           if payload.water_basin_slack_m3 is not None else 0.0)
        + payload.injectivity_slack_mtpa.sum() * assumptions.slack_penalty_cny_per_unit
        + payload.storage_slack_mt.sum() * assumptions.slack_penalty_cny_per_unit
        + payload.edge_slack_mtpa.sum() * assumptions.slack_penalty_cny_per_unit
    )
