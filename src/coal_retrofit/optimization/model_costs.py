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
from .year_types import YearPayload


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
    """写入 payload 的 `cost_exprs`、`salvage_ledger`、`objective_expr`。"""
    from .industry import CCS as _IND_CCS, H2 as _IND_H2, industry_capex_expr

    year_data = payload.year_data
    interval_years = int(payload.interval_years)
    current_year = int(payload.year)
    interval_weight = _year_objective_weight(interval_years, scenario.discount_rate)
    df = _discount_factor(current_year, scenario.discount_base_year, scenario.discount_rate)
    yr_sfx = str(payload.year)
    retire_idx = PATHWAY_INDEX["retire"]
    plant_reduction_exprs = payload.plant_reduction_exprs
    total_bio_penalty = payload.total_bio_penalty

    # 基线净运行成本（煤 + 运维 - 电）：未改造按基线发电量，改造路径含 CF 提升（重建机组用其热耗），退役为零。
    baseline_net = gp.quicksum(
        float(year_data.baseline_net_matrix[p, k]) * payload.share[p, k]
        for p in range(plant_count) for k in range(len(PATHWAYS))
    )

    # 碳成本：碳价 x 残余排放，煤电与工业都计。
    carbon_price_t = float(year_data.carbon_price)
    carbon_cost = 0.0
    if carbon_price_t > 0:
        carbon_cost = carbon_price_t * 1e6 * gp.quicksum(
            float(year_data.emissions_mt[p]) - plant_reduction_exprs[p]
            for p in range(plant_count)
        )
        carbon_cost = carbon_cost + carbon_price_t * 1e6 * gp.quicksum(
            payload.industry["residual_by_group"].values()
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

    # 资源采购。
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

    # CO2 运输与封存。边系数（CNY/Mt）已含长度、单位运维率与海上倍率。
    edge_opex_coeff = year_data.edge_route_opex_coeff
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

    # 管道 capex：各管径档整根计。
    tier_capex = np.asarray(year_data.edge_tier_capex, dtype=np.float64)
    pipe_capex = gp.quicksum(
        float(tier_capex[edge_idx, k]) * payload.pipe_count[edge_idx, k]
        for edge_idx in range(edge_count) for k in range(tier_capex.shape[1])
    )

    # 松弛惩罚（资源松弛按缩放单位乘回）。流域指标松弛与节点松弛同价，两个制度不因数值原因分高下。
    _amm_scale = float(year_data.ammonia_flow_scale)
    _wat_scale = float(year_data.water_flow_scale)
    _bio_scale = float(year_data.biomass_flow_scale)
    slack_cost = (
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

    # 一次性 capex：搁浅资产、CCS 改造（计在存量增量上）、掺烧升级、空冷改造。
    capacity_mw = year_data.capacity_mw
    if year_position == 0:
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
        prev_payload = year_payloads[year_position - 1]
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
    if year_position == 0:
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

    # 自愿退役速率上限（不含到期强制退役）。两边除以总发电量，系数为发电份额，避免 4e7 量级系数。
    if scenario.max_new_retirement_share_per_period > 0:
        generation = year_data.generation
        total_gen = float(generation.sum())
        voluntary_plants = [p for p in range(plant_count) if int(payload.year) < retirement_years[p]]
        if voluntary_plants and total_gen > 0:
            if year_position == 0:
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

    payload.cost_exprs = {
        "baseline_net_cost":   df * interval_weight * baseline_net / _COST_SCALE,
        "carbon_cost":         df * interval_weight * carbon_cost / _COST_SCALE if carbon_price_t > 0 else 0.0,
        "coal_savings_credit": df * interval_weight * (-coal_savings) / _COST_SCALE,
        "energy_penalty_cost": df * interval_weight * (energy_penalty_cost + total_bio_penalty) / _COST_SCALE,
        "ccs_om_cost":         df * interval_weight * ccs_om_cost / _COST_SCALE,
        "incremental_om":      df * interval_weight * incremental_om / _COST_SCALE,
        "biomass_cost":        df * interval_weight * biomass_cost / _COST_SCALE,
        "ammonia_cost":        df * interval_weight * ammonia_cost / _COST_SCALE,
        "water_cost":          df * interval_weight * water_cost / _COST_SCALE,
        "transport_opex":      df * interval_weight * transport_opex / _COST_SCALE,
        "storage_cost":        df * interval_weight * storage_cost / _COST_SCALE,
        "stranded_capex":      df * stranded_capex / _COST_SCALE,
        "ccs_retrofit_capex":  df * ccs_retrofit_capex / _COST_SCALE,
        "pipe_capex":          df * pipe_capex / _COST_SCALE,
        "blend_upgrade_capex": df * blend_upgrade_capex / _COST_SCALE,
        "air_retrofit_capex": df * air_retrofit_capex / _COST_SCALE,
        "rebuild_capex":      df * rebuild_capex / _COST_SCALE,
        "slack_penalty":       df * interval_weight * slack_cost / _COST_SCALE,
    }
    # 本年未折现 capex 及其经济寿命，供期末残值。搁浅资产是损失不是资产，不入台账。
    payload.salvage_ledger = [
        ("ccs_retrofit_capex", ccs_retrofit_capex, int(assumptions.ccs_retrofit_lifetime_years)),
        ("pipe_capex", pipe_capex, int(assumptions.pipeline_lifetime_years)),
        ("blend_upgrade_capex", blend_upgrade_capex, int(assumptions.blend_upgrade_lifetime_years)),
        ("air_retrofit_capex", air_retrofit_capex, int(assumptions.air_retrofit_lifetime_years)),
        ("rebuild_capex", rebuild_capex, int(assumptions.rebuild_lifetime_years)),
    ]
    # 工业：年度部分（固定运维、捕集能耗与耗材、氢路线非氢运行差额、按链路买的氢）用同一折现/年金权重；
    # 改造 capex 一次性计在路线份额增量上，与煤电 `ccs_retrofit_capex` 同法。
    payload.cost_exprs["industry_cost"] = (
        df * interval_weight * payload.industry["annual_cost_cny"] / _COST_SCALE
    )
    prev_industry = None if year_position == 0 else year_payloads[year_position - 1].industry
    ind_lives = payload.industry["year_data"]["capex_lifetime_years"]
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
