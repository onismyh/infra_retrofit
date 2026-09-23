"""一年的成本表达式：运行成本、资源采购、运输封存、一次性 capex、松弛惩罚，及残值台账。

成本口径（作者决定 2026-09-22）：改造 capex 一次性计在存量增量上 + 固定运维 + 能耗按模型自身
煤价电价，期末对未折旧 capex 计残值（`salvage.py`）。不用平准化每吨成本。
"""
from __future__ import annotations

import numpy as np

from ._shared import (
    _COST_SCALE,
    PATHWAY_INDEX,
    PreparedInputs,
    _discount_factor,
    _year_objective_weight,
    gp,
)
from .constraints import _build_air_retrofit_capex, _build_blend_upgrade_capex
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
    retirement_years: np.ndarray,
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
    one_off = _one_off_capex(
        model, payload, prev_payload, scenario, assumptions, retirement_years, plant_count, edge_count,
    )
    _add_retirement_rate_limit(model, payload, prev_payload, scenario, retirement_years, plant_count)

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
    # 工业：年度部分（固定运维、捕集能耗与耗材、氢路线非氢运行差额、按链路买的氢）用同一折现/年金权重；
    # 改造 capex 一次性计在路线份额增量上，与煤电 `ccs_retrofit_capex` 同法。
    payload.cost_exprs["industry_cost"] = (
        df * interval_weight * payload.industry.annual_cost_expr / _COST_SCALE
    )
    prev_industry = None if prev_payload is None else prev_payload.industry
    ind_lives = payload.industry.year_data.capex_lifetime_years
    ind_ccs_capex = industry_capex_expr(payload.industry, prev_industry, routes=(_IND_CCS,))
    ind_h2_capex = industry_capex_expr(payload.industry, prev_industry, routes=(_IND_H2,))
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

    # 基线净运行成本（煤 + 运维 - 电）：未改造按基线发电量，改造路径含 CF 提升（重建机组用其热耗），退役为零。
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
    ccs_om_cost = gp.quicksum(
        float(year_data.ccs_om_matrix[p, k]) * payload.share[p, k]
        for p in range(plant_count) for k in range(len(PATHWAYS))
    )
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
    retirement_years: np.ndarray,
    plant_count: int,
    edge_count: int,
) -> dict[str, GrbExpr]:
    """一次性 capex（CNY，未折现未缩放），键与 `cost_exprs` 同名同序。

    非首年按增量计：搁浅资产、原址重建与掺烧升级为增量加辅助变量与约束，空冷改造只加存量单调约束
    （后两者在 `constraints` 里加）。这些加入模型的先后决定模型指纹。
    """
    year_data = payload.year_data
    yr_sfx = str(payload.year)
    current_year = int(payload.year)
    retire_idx = PATHWAY_INDEX["retire"]
    capacity_mw = year_data.capacity_mw

    # 管道 capex：各管径档整根计。
    tier_capex = np.asarray(year_data.edge_tier_capex, dtype=np.float64)
    pipe_capex = gp.quicksum(
        float(tier_capex[edge_idx, k]) * payload.pipe_count[edge_idx, k]
        for edge_idx in range(edge_count) for k in range(tier_capex.shape[1])
    )

    # 搁浅资产、CCS 改造（计在存量增量上）、掺烧升级、空冷改造。
    if prev_payload is None:
        stranded_capex = gp.quicksum(
            float(year_data.stranded_per_plant[p]) * payload.share[p, retire_idx]
            for p in range(plant_count)
            if float(year_data.stranded_per_plant[p]) > 0
        )
        stock_coeff = np.asarray(year_data.retrofit_stock_capex, dtype=np.float64)
        ccs_retrofit_capex = gp.quicksum(
            float(stock_coeff[p, j]) * payload.retrofit_installed[p, j]
            for j in range(stock_coeff.shape[1]) for p in range(plant_count)
            if float(stock_coeff[p, j]) > 0.0
        )
        blend_upgrade_capex = _build_blend_upgrade_capex(
            model, capacity_mw, payload.blend_level_b, payload.blend_level_a,
            assumptions, plant_count, sfx=f"_{yr_sfx}",
        )
        air_retrofit_capex = _build_air_retrofit_capex(
            model, year_data, payload, plant_count, yr_sfx, prev_payload=None
        )
    else:
        # 搁浅资产按新增退役份额计。
        stranded_terms = []
        for p in range(plant_count):
            coeff = float(year_data.stranded_per_plant[p])
            if coeff <= 0:
                continue
            delta_ret = model.addVar(lb=0.0, name=f"stranded_delta_{p}_{yr_sfx}")
            model.addConstr(
                delta_ret >= payload.share[p, retire_idx] - prev_payload.share[p, retire_idx],
                name=f"stranded_delta_lb_{p}_{yr_sfx}",
            )
            stranded_terms.append(coeff * delta_ret)
        stranded_capex = gp.quicksum(stranded_terms) if stranded_terms else 0.0
        # CCS 改造 capex 计在存量增量上（历史最高份额），份额回落再回升不重复计费。
        prev_installed = prev_payload.retrofit_installed
        stock_coeff = np.asarray(year_data.retrofit_stock_capex, dtype=np.float64)
        ccs_retrofit_capex = gp.quicksum(
            float(stock_coeff[p, j])
            * (payload.retrofit_installed[p, j] - prev_installed[p, j])
            for j in range(stock_coeff.shape[1]) for p in range(plant_count)
            if float(stock_coeff[p, j]) > 0.0
        )
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

    # 原址重建 capex：容量 x 新建成本比例，只在首次激活期计（rebuild 锁存，增量只在激活期为 1）。
    rebuild_cost_per_mw = assumptions.stranded_asset_base_cny_per_kw * scenario.rebuild_capex_fraction * 1000.0
    if prev_payload is None:
        rebuild_capex = gp.quicksum(
            float(capacity_mw[p]) * rebuild_cost_per_mw * payload.rebuild[p]
            for p in range(plant_count)
            if current_year >= retirement_years[p]
        )
    else:
        rebuild_capex_terms = []
        for p in range(plant_count):
            if current_year < retirement_years[p]:
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


def _add_retirement_rate_limit(
    model,
    payload: YearPayload,
    prev_payload: YearPayload | None,
    scenario: OptimizationScenario,
    retirement_years: np.ndarray,
    plant_count: int,
) -> None:
    """自愿退役速率上限（不含到期强制退役）。两边除以总发电量，系数为发电份额，避免 4e7 量级系数。"""
    retire_idx = PATHWAY_INDEX["retire"]
    yr_sfx = str(payload.year)
    if scenario.max_new_retirement_share_per_period > 0:
        generation = payload.year_data.generation
        total_gen = float(generation.sum())
        voluntary_plants = [p for p in range(plant_count) if int(payload.year) < retirement_years[p]]
        if voluntary_plants and total_gen > 0:
            if prev_payload is None:
                model.addConstr(
                    gp.quicksum(float(generation[p]) / total_gen * payload.share[p, retire_idx] for p in voluntary_plants)
                    <= scenario.max_new_retirement_share_per_period,
                    name=f"max_retire_rate_{yr_sfx}",
                )
            else:
                model.addConstr(
                    gp.quicksum(
                        float(generation[p]) / total_gen * (payload.share[p, retire_idx] - prev_payload.share[p, retire_idx])
                        for p in voluntary_plants
                    ) <= scenario.max_new_retirement_share_per_period,
                    name=f"max_retire_rate_{yr_sfx}",
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
        + payload.ammonia_slack_kg.sum() * 1_000.0 * _amm_scale
        + payload.water_slack_m3.sum() * 1_000.0 * _wat_scale
        + (payload.water_basin_slack_m3.sum() * 1_000.0 * _wat_scale
           if payload.water_basin_slack_m3 is not None else 0.0)
        + payload.injectivity_slack_mtpa.sum() * assumptions.slack_penalty_cny_per_unit
        + payload.storage_slack_mt.sum() * assumptions.slack_penalty_cny_per_unit
        + payload.edge_slack_mtpa.sum() * assumptions.slack_penalty_cny_per_unit
    )
