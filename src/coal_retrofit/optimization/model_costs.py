"""一年的成本表达式：运行成本、资源采购、运输封存、一次性 capex、松弛惩罚，及残值台账。

成本口径（作者决定 2026-09-22）：改造 capex 一次性计 + 固定运维 + 能耗按模型自身煤价电价，期末对未折旧 capex
计残值（`salvage.py`）。不用平准化每吨成本。捕集岛、空冷改造、掺烧升级与工业路线能力的 capex 计在本年新建量上，按建设年
分代、到寿命退出（`vintage`，捕集岛与工业 2026-09-30 起，空冷与掺烧 2026-10-02 起）；原址重建计在重建份额的增量上，
搁浅资产计在新增提前退役上（`retirement`）。
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
from .constraints import _rebuilt_dot
from .retirement import add_retirement_rate_limit, retirement_flows
from .scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from .vintage import alive_vintages
from .year_types import GrbExpr, GrbMVar, YearData, YearPayload


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
    """写入 payload 的 `cost_exprs`、`cost_weights`、`salvage_ledger`、`objective_expr`。

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
        **_transport_storage_costs(payload, year_payloads, year_position, prepared, assumptions, edge_count, storage_count),
    }
    early_new, rebuilt_new = retirement_flows(model, payload, prev_payload, plant_count)
    one_off = _one_off_capex(model, payload, prev_payload, scenario, assumptions, early_new, plant_count, edge_count)
    add_retirement_rate_limit(model, payload, scenario, early_new, rebuilt_new, plant_count)

    payload.cost_exprs = {name: df * interval_weight * expr / _COST_SCALE for name, expr in annual.items()}
    payload.cost_exprs.update({name: df * expr / _COST_SCALE for name, expr in one_off.items()})
    payload.cost_exprs["slack_penalty"] = df * interval_weight * _slack_penalty(payload, assumptions) / _COST_SCALE
    # 本年建成资产的残值基数（未折现 capex）及其经济寿命，供期末残值。分代资产只计本年建成、最后一个规划年仍在用的部分
    # （`vintage`，2026-10-02 起；此前按建成量计）；管道与原址重建按建成量计，后者另扣期末已关停的（`retirement`）。
    # 搁浅资产是损失不是资产，不入台账。
    year_data = payload.year_data
    assert payload.ccs_island is not None and payload.air_cooling is not None, "add_capacity_vintages must run first"
    assert payload.biomass_blend is not None and payload.ammonia_blend is not None
    unit_b, unit_a = _blend_unit_capex(year_data, assumptions)
    payload.salvage_ledger = [
        ("ccs_retrofit_capex", _priced(_island_unit_capex(year_data), payload.ccs_island.end_in_use),
         int(assumptions.ccs_retrofit_lifetime_years)),
        ("pipe_capex", one_off["pipe_capex"], int(assumptions.pipeline_lifetime_years)),
        ("blend_upgrade_capex",
         sum((_priced(unit_b, layer.end_in_use) for layer in payload.biomass_blend), 0.0)
         + sum((_priced(unit_a, layer.end_in_use) for layer in payload.ammonia_blend), 0.0),
         int(assumptions.blend_upgrade_lifetime_years)),
        ("air_retrofit_capex", _priced(_air_unit_capex(year_data), payload.air_cooling.end_in_use),
         int(assumptions.air_retrofit_lifetime_years)),
        ("rebuild_capex", one_off["rebuild_capex"], int(assumptions.rebuild_lifetime_years)),
    ]
    # 工业：年度部分（捕集能耗与耗材、在役且在用的捕集能力按建设年单价的固定运维、氢路线固定运维与非氢运行差额、
    # 按链路买的氢）用同一折现/年金权重；改造 capex 一次性计在本年新建能力上，与煤电 `ccs_retrofit_capex` 同法。
    assert payload.industry_ccs is not None and payload.industry_h2 is not None
    payload.cost_exprs["industry_cost"] = (
        df * interval_weight * (payload.industry.annual_cost_expr + gp.quicksum(payload.industry_ccs.fixed_om))
        / _COST_SCALE
    )
    ind_lives = payload.industry.year_data.capex_lifetime_years
    ind_unit = np.asarray(payload.industry.year_data.capex_cny_per_mt, dtype=np.float64)
    ind_ccs_capex = industry_capex_expr(payload.industry, routes=(_IND_CCS,))
    ind_h2_capex = industry_capex_expr(payload.industry, routes=(_IND_H2,))
    payload.cost_exprs["industry_capex"] = (
        df * (ind_ccs_capex + ind_h2_capex) / _COST_SCALE
    )
    payload.salvage_ledger += [
        ("industry_ccs_capex", _priced(ind_unit[:, _IND_CCS], payload.industry_ccs.end_in_use),
         int(ind_lives[_IND_CCS])),
        ("industry_h2_capex", _priced(ind_unit[:, _IND_H2], payload.industry_h2.end_in_use), int(ind_lives[_IND_H2])),
    ]
    # 上面各项乘的折现权重，结果表除以它得本年不折现的值（`results._build_cost_breakdown`）。
    payload.cost_weights = {
        **{name: ("annual", df * interval_weight) for name in (*annual, "slack_penalty", "industry_cost")},
        **{name: ("one_off", df) for name in (*one_off, "industry_capex")},
    }
    assert payload.cost_weights.keys() == payload.cost_exprs.keys()
    payload.objective_expr = gp.quicksum(list(payload.cost_exprs.values()))


def _operating_costs(payload: YearPayload, plant_count: int) -> dict[str, GrbExpr]:
    """煤电运行项与碳成本（CNY/yr，未折现未缩放），键与 `cost_exprs` 同名同序。"""
    year_data = payload.year_data

    # 基线净运行成本（煤 + 运维 - 电量电费 - 容量电费）相对参照"全部维持不改造运行"的差（增量口径，2026-10-10 起；
    # 参照不进目标，`plant_matrices` 的 `baseline_reference`）：未改造为零，改造路径为 CF 提升的差，退役为省下的参照。
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

    # 路径增量运维（每 MWh 附加项）+ 掺烧能力的固定运维：在用的掺烧能力 x 每单位的运维（生物质 2026-10-02 起，
    # 掺氨 2026-10-10 起）。
    incremental_om = (
        gp.quicksum(
            float(year_data.fixed_cost_matrix[p, k]) * payload.share[p, k]
            for p in range(plant_count) for k in range(len(PATHWAYS))
        )
        + _priced(year_data.biomass_blend_om_per_level, payload.blend_level_b)
        + _priced(year_data.ammonia_blend_om_per_level, payload.blend_level_a)
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
    # 上面三项的系数按未重建部分的毛热耗算；拆出重建部分的 hub 再逐类加该类与未重建部分之差 x 该类重建部分的份额
    # （`constraints._add_rebuilt_split`）。掺烧惩罚的差已在 `bio_penalty_by_plant` 里。
    deltas = year_data.rebuilt_deltas
    baseline_net = baseline_net + gp.quicksum(
        _rebuilt_dot(deltas[c].baseline_net_matrix[p], part)
        for p, classes in payload.rebuilt_share.items() for c, part in classes.items()
    )
    energy_penalty_cost = energy_penalty_cost + gp.quicksum(
        _rebuilt_dot(deltas[c].energy_penalty_matrix[p], part)
        for p, classes in payload.rebuilt_share.items() for c, part in classes.items()
    ) + gp.quicksum(
        _rebuilt_dot(deltas[c].air_penalty_cost_matrix[p], part)
        for p, classes in payload.rebuilt_air_share.items() for c, part in classes.items()
    )
    # 捕集岛固定运维：在役且在用的捕集岛 x 建设年的单价（`vintage`，闲置不付）。2026-09-30 前按当年单价 x 捕集份额计。
    assert payload.ccs_island is not None, "add_capacity_vintages must run before add_year_costs"
    ccs_om_cost = gp.quicksum(payload.ccs_island.fixed_om)
    return {
        "coal_operating_delta": baseline_net,
        "carbon_cost": carbon_cost,
        "coal_savings_credit": -coal_savings,
        "energy_penalty_cost": energy_penalty_cost + gp.quicksum(payload.bio_penalty_by_plant),
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
    payload: YearPayload,
    year_payloads: list[YearPayload],
    year_position: int,
    prepared: PreparedInputs,
    assumptions: OptimizationAssumptions,
    edge_count: int,
    storage_count: int,
) -> dict[str, GrbExpr]:
    """CO2 运输运维与封存（CNY/yr，未折现未缩放）。

    运输运维 = 按流量计的一项（边系数 CNY/Mt 已含长度、单位运维率与海上倍率；2026-10-02 起缺省单位运维率为 0）+ 管道固定
    运维：在役各代管道的 capex x `pipe_fixed_om_fraction`（2026-10-02 起）。在役与流量上限同一判据（t − v < 管道寿命，
    `model_linking.add_capacity_constraints`），含到寿命后原址重铺的那一代；闲置的管也付。比例为 0 时不加这一项。
    """
    edge_opex_coeff = payload.year_data.edge_route_opex_coeff
    transport_opex = gp.quicksum(
        float(edge_opex_coeff[e])
        * (payload.co2_flow_fwd[e] + payload.co2_flow_bwd[e])
        for e in range(edge_count)
    )
    pipe_om_fraction = float(assumptions.pipe_fixed_om_fraction)
    if not pipe_om_fraction >= 0.0:  # 写成 not >= 也拦下 NaN
        raise ValueError(f"pipe_fixed_om_fraction must be >= 0, got {pipe_om_fraction}")
    if pipe_om_fraction > 0.0:
        years = [int(p.year) for p in year_payloads]
        transport_opex = transport_opex + pipe_om_fraction * gp.quicksum(
            _pipe_capex(year_payloads[v], edge_count)
            for v in alive_vintages(years, year_position, int(assumptions.pipeline_lifetime_years))
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

    分代资产（捕集岛、掺烧升级、空冷改造）计在本年新建量上（`vintage`）；原址重建非首年按增量计，加辅助变量与约束；
    搁浅资产计在新增提前退役 `early_new` 上（`retirement.retirement_flows`）。这些加入模型的先后决定模型指纹。
    """
    year_data = payload.year_data
    yr_sfx = str(payload.year)
    capacity_mw = year_data.capacity_mw
    expired_share = year_data.expired_share

    pipe_capex = _pipe_capex(payload, edge_count)

    # 捕集岛：本年单价 x 本年新建（`vintage`；2026-09-30 前计在单调存量的增量上）。
    ccs_retrofit_capex = _priced(_island_unit_capex(year_data), payload.retrofit_new[:, 0])

    # 搁浅资产：每单位新增提前退役的剩余账面价值（`plant_matrices`）x 新增提前退役。到期退役不计；重建后退役也不计，
    # 它的 capex 已在目标函数里，期末不计残值（`retirement.add_retired_rebuild_offset`）。
    payload.stranded_by_plant = [
        float(year_data.stranded_per_plant[p]) * early_new[p] if float(year_data.stranded_per_plant[p]) > 0 else 0.0
        for p in range(plant_count)
    ]
    stranded_capex = gp.quicksum(payload.stranded_by_plant)

    # 掺烧升级、空冷改造：本年单价 x 本年新建，掺烧逐档位层（`vintage`，2026-10-02 起；此前掺烧计在档位 Σ l·select
    # 的正增量上、空冷计在只增不减的已装存量的增量上，都不到期）。
    unit_b, unit_a = _blend_unit_capex(year_data, assumptions)
    blend_upgrade_capex = sum(
        (_priced(unit_b, payload.blend_new_b[:, j]) for j in range(payload.blend_new_b.shape[1])), 0.0
    ) + sum((_priced(unit_a, payload.blend_new_a[:, j]) for j in range(payload.blend_new_a.shape[1])), 0.0)
    air_retrofit_capex = _priced(_air_unit_capex(year_data), payload.air_new)

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


def _pipe_capex(payload: YearPayload, edge_count: int) -> GrbExpr:
    """本年新铺管道的 capex（CNY）：各管径档整根计，单根单价含类别与海上倍率（`year_matrices._edge_matrices`）。"""
    tier_capex = np.asarray(payload.year_data.edge_tier_capex, dtype=np.float64)
    return gp.quicksum(
        float(tier_capex[edge_idx, k]) * payload.pipe_count[edge_idx, k]
        for edge_idx in range(edge_count) for k in range(tier_capex.shape[1])
    )


def _priced(unit: np.ndarray, amount: GrbMVar | None) -> GrbExpr:
    """Σ_i 单价_i x 数量_i，只取单价为正的单元；没有数量（`StockYear.end_in_use` 为 None）或没有这样的单元时为 0.0。"""
    if amount is None:
        return 0.0
    terms = [float(u) * amount[i] for i, u in enumerate(np.asarray(unit, dtype=np.float64)) if float(u) > 0.0]
    return gp.quicksum(terms) if terms else 0.0


def _island_unit_capex(year_data: YearData) -> np.ndarray:
    """本年建成的每单位捕集岛（占装机的份额）的 capex，CNY，每厂一项（`year_matrices`，含学习曲线）。"""
    return np.asarray(year_data.retrofit_stock_capex, dtype=np.float64)[:, 0]


def _air_unit_capex(year_data: YearData) -> np.ndarray:
    """每单位空冷改造（占仍湿冷装机的份额）的 capex，CNY，每厂一项（`plant_matrices._air_cooling_matrices`）。"""
    return np.asarray(year_data.air_retrofit_capex_per_plant, dtype=np.float64)


def _blend_unit_capex(year_data: YearData, assumptions: OptimizationAssumptions) -> tuple[np.ndarray, np.ndarray]:
    """每单位掺烧能力（一个档位层 x 占装机的份额）的 capex，CNY，每厂一项；生物质（BECCS 共用）、氨各一份。

    掺烧能力的固定运维按同一单价计（`plant_matrices` 的 `biomass_blend_om_per_level`、`ammonia_blend_om_per_level`），
    改这里须同改。
    """
    capacity_mw = np.asarray(year_data.capacity_mw, dtype=np.float64)
    return (
        capacity_mw * float(assumptions.biomass_upgrade_capex_cny_per_mw_per_level),
        capacity_mw * float(assumptions.ammonia_upgrade_capex_cny_per_mw_per_level),
    )


def _slack_penalty(payload: YearPayload, assumptions: OptimizationAssumptions) -> GrbExpr:
    """松弛惩罚（CNY/yr，资源松弛按缩放单位乘回）。流域指标松弛与节点松弛同价，两个制度不因数值原因分高下。"""
    year_data = payload.year_data
    _amm_scale = float(year_data.ammonia_flow_scale)
    _wat_scale = float(year_data.water_flow_scale)
    _bio_scale = float(year_data.biomass_flow_scale)
    return (
        payload.target_shortfall_mt * assumptions.slack_penalty_cny_per_unit
        + payload.biomass_slack_gj.sum() * 2_000.0 * _bio_scale
        + payload.h2_slack_kg.sum() * 1_000.0 * _amm_scale
        + payload.water_slack_m3.sum() * 1_000.0 * _wat_scale
        + (payload.water_basin_slack_m3.sum() * 1_000.0 * _wat_scale
           if payload.water_basin_slack_m3 is not None else 0.0)
        + payload.injectivity_slack_mtpa.sum() * assumptions.slack_penalty_cny_per_unit
        + payload.storage_slack_mt.sum() * assumptions.slack_penalty_cny_per_unit
        + payload.edge_slack_mtpa.sum() * assumptions.slack_penalty_cny_per_unit
    )
