"""煤电厂侧逐年系数矩阵：发电与排放、能耗惩罚、改造 capex 与运维、耗水强度、空冷改造。

所有矩阵形状 (plant_count, len(PATHWAYS))，列序同 PATHWAYS = unabated/retire/ccs/biomass/beccs/ammonia。
"""
from __future__ import annotations

import numpy as np

from ._shared import PATHWAY_INDEX, PreparedInputs
from .scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario


def _plant_operating_matrices(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
) -> dict[str, object]:
    """发电、排放、运行成本、能耗惩罚、CCS/BECCS capex 与运维、搁浅资产。"""
    # 本年利用小时：`annual_generation_mwh` 是当前省级统计，按情景小时轨迹逐年缩放，
    # 发电、基线排放与每 MWh 成本同步移动。
    fleet_hours_now = float(prepared.plants["fleet_hours_now"].iloc[0]) if "fleet_hours_now" in prepared.plants.columns else 0.0
    hours_scale = float(scenario.operating_hours_scale(int(year), fleet_hours_now)) if fleet_hours_now > 0 else 1.0
    generation = prepared.plants["annual_generation_mwh"].astype(float).to_numpy() * hours_scale
    emissions_mt = prepared.plants["baseline_emissions_mt"].astype(float).to_numpy() * hours_scale
    # 改造路径优先调度，发电量乘 CF 提升。
    generation_retrofit = generation * scenario.retrofit_cf_boost
    generation_by_pathway = np.column_stack([
        generation,                    # unabated
        np.zeros_like(generation),     # retire
        generation_retrofit,           # ccs
        generation_retrofit,           # biomass
        generation_retrofit,           # beccs
        generation_retrofit,           # ammonia
    ])
    design_retirement_year = prepared.plants["retirement_year"].astype(int).to_numpy()
    # 到期后原址重建的机组按 rebuild_efficiency（超超临界）运行，热耗与排放强度按效率比缩放。
    eff_ratio = np.where(
        year >= design_retirement_year,
        assumptions.coal_plant_base_efficiency / max(float(scenario.rebuild_efficiency), 1e-9),
        1.0,
    )
    heat_rate_eff = assumptions.heat_rate_gj_per_mwh * eff_ratio
    # 三个排放基数：基线（退役避免量 + 目标分母）、运行（效率修正）、改造（效率修正 x CF 提升）。
    emissions_operating_mt = emissions_mt * eff_ratio
    emissions_retrofit_mt = emissions_operating_mt * scenario.retrofit_cf_boost
    # 成本基数：退役列保留基线发电量（退役成本按原发电量计），物理量用 generation_by_pathway（退役列为零）。
    generation_cost_basis = generation_by_pathway.copy()
    generation_cost_basis[:, PATHWAY_INDEX["retire"]] = generation
    pathway_fixed_costs = np.array(
        [
            assumptions.fixed_cost_cny_per_mwh("unabated"),
            assumptions.fixed_cost_cny_per_mwh("retire"),
            assumptions.fixed_cost_cny_per_mwh("ccs") * scenario.ccs_cost_multiplier,
            assumptions.fixed_cost_cny_per_mwh("biomass"),
            assumptions.fixed_cost_cny_per_mwh("beccs") * scenario.ccs_cost_multiplier,
            assumptions.fixed_cost_cny_per_mwh("ammonia"),
        ],
        dtype=np.float64,
    )
    # 能耗惩罚（效率损失 → 多烧煤）：CCS 项只随份额变；生物质项随掺烧档位变，在 constraints 里 McCormick 线性化。
    eta_coal = assumptions.coal_plant_base_efficiency
    coal_price_per_plant = np.array([
        assumptions.province_coal_cost(str(prov))
        for prov in prepared.plants["province_name"]
    ], dtype=np.float64)
    # CCS 能耗惩罚直接以单位出力的额外燃料比表示，逐年下降：额外煤成本 = ratio(year) x hr_eff x 煤价 [CNY/MWh]
    eps_ratio_ccs = assumptions.ccs_energy_penalty_ratio(year)
    ccs_penalty_per_mwh_per_plant = eps_ratio_ccs * heat_rate_eff * coal_price_per_plant
    energy_penalty_per_pathway = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    energy_penalty_matrix = generation_cost_basis * (ccs_penalty_per_mwh_per_plant[:, None] * energy_penalty_per_pathway[None, :])
    # 生物质档位效率惩罚系数：penalty = G_p x β_b x (ε_per_ratio / η) x hr_eff x 煤价_p
    biomass_penalty_coeff_per_level = (
        assumptions.biomass_efficiency_penalty_per_ratio / eta_coal * heat_rate_eff * coal_price_per_plant
    )

    # 能耗惩罚的排放（Mt）：补效率损失多烧的煤在同一锅炉燃烧，经同一捕集装置，只有未捕集份额排放
    # （Fan et al. 2023 Nat Clim Change SI eq. S42）；捕集份额是真实流量，计入捕集量。
    uncaptured = 1.0 - float(scenario.capture_rate)
    emission_factor_t_per_gj = assumptions.coal_emission_factor_t_per_mwh / assumptions.heat_rate_gj_per_mwh
    ccs_penalty_emissions_per_mwh_per_plant = (
        eps_ratio_ccs * heat_rate_eff * emission_factor_t_per_gj * uncaptured / 1_000_000.0
    )
    ccs_penalty_emissions_matrix = (
        generation_cost_basis
        * (ccs_penalty_emissions_per_mwh_per_plant[:, None] * energy_penalty_per_pathway[None, :])
    )
    ccs_penalty_captured_matrix = (
        generation_cost_basis
        * (eps_ratio_ccs * heat_rate_eff * emission_factor_t_per_gj * float(scenario.capture_rate)
           / 1_000_000.0)[:, None]
        * energy_penalty_per_pathway[None, :]
    )
    # 生物质掺烧的惩罚燃料：纯掺烧路径全部排放，BECCS 按 capture_rate 捕集。
    biomass_penalty_emissions_coeff_per_level = (
        assumptions.biomass_efficiency_penalty_per_ratio / eta_coal * heat_rate_eff * emission_factor_t_per_gj / 1_000_000.0
    )
    beccs_penalty_emissions_coeff_per_level = biomass_penalty_emissions_coeff_per_level * uncaptured
    beccs_penalty_captured_coeff_per_level = (
        biomass_penalty_emissions_coeff_per_level * float(scenario.capture_rate)
    )

    # CCS/BECCS 改造 capex（含学习曲线）。
    lf = assumptions.ccs_learning_factor(year)
    ccs_capex_per_mw = np.array([
        0.0,
        0.0,
        assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * scenario.ccs_cost_multiplier * lf,   # ccs
        0.0,
        assumptions.beccs_retrofit_capex_cny_per_kw * 1000.0 * scenario.ccs_cost_multiplier * lf,  # beccs
        0.0,
    ], dtype=np.float64)
    capacity_mw = prepared.plants["total_capacity_mw"].astype(float).to_numpy()
    ccs_retrofit_capex_matrix = capacity_mw[:, None] * ccs_capex_per_mw[None, :]

    # CCS 固定运维 = ccs_om_fraction x 学习后 capex，按改造容量 MW 计（An et al. 2025 SI Table 7），不按 MWh。
    ccs_om_per_mw = np.array([
        0.0,
        0.0,
        assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * lf * assumptions.ccs_om_fraction,
        0.0,
        assumptions.beccs_retrofit_capex_cny_per_kw * 1000.0 * lf * assumptions.ccs_om_fraction,
        0.0,
    ], dtype=np.float64)
    ccs_om_matrix = capacity_mw[:, None] * ccs_om_per_mw[None, :]

    fixed_cost_matrix = generation_cost_basis * pathway_fixed_costs[None, :]

    # 基线净运行成本（煤 + 运维 - 电）：未改造列按基线发电量，改造列含 CF 提升，退役列为零。
    elec_price_year = scenario.electricity_price_for_year(year)
    net_operating_cost_per_mwh = (
        heat_rate_eff * coal_price_per_plant
        + assumptions.baseline_om_cost_cny_per_mwh
        - elec_price_year
    )
    baseline_net_matrix = generation_by_pathway * net_operating_cost_per_mwh[:, None]

    # 搁浅资产：按剩余设计寿命占会计寿命的比例计。
    remaining_life = np.maximum(0, design_retirement_year - year)
    fraction_remaining = np.minimum(1.0, remaining_life / max(1, assumptions.stranded_asset_accounting_life))
    stranded_per_plant = capacity_mw * assumptions.stranded_asset_base_cny_per_kw * 1000.0 * fraction_remaining

    return {
        "hours_scale": hours_scale,
        "generation": generation,
        "generation_by_pathway": generation_by_pathway,
        "generation_cost_basis": generation_cost_basis,
        "emissions_mt": emissions_mt,
        "emissions_operating_mt": emissions_operating_mt,
        "emissions_retrofit_mt": emissions_retrofit_mt,
        "heat_rate_eff": heat_rate_eff,
        "capacity_mw": capacity_mw,
        "coal_price_per_plant": coal_price_per_plant,
        "fixed_cost_matrix": fixed_cost_matrix,
        "energy_penalty_matrix": energy_penalty_matrix,
        "biomass_penalty_coeff_per_level": biomass_penalty_coeff_per_level,
        "ccs_penalty_emissions_matrix": ccs_penalty_emissions_matrix,
        "ccs_penalty_captured_matrix": ccs_penalty_captured_matrix,
        "biomass_penalty_emissions_coeff_per_level": biomass_penalty_emissions_coeff_per_level,
        "beccs_penalty_emissions_coeff_per_level": beccs_penalty_emissions_coeff_per_level,
        "beccs_penalty_captured_coeff_per_level": beccs_penalty_captured_coeff_per_level,
        "ccs_retrofit_capex_matrix": ccs_retrofit_capex_matrix,
        "ccs_om_matrix": ccs_om_matrix,
        "baseline_net_matrix": baseline_net_matrix,
        "stranded_per_plant": stranded_per_plant,
    }


def _water_intensity_matrices(
    prepared: PreparedInputs, scenario: OptimizationScenario, assumptions: OptimizationAssumptions
) -> tuple[np.ndarray, np.ndarray]:
    """逐路径耗水强度及其全空冷对应矩阵（m3/MWh）。

    捕集路径取表内带捕集值而非单一倍率：捕集增量随冷却方式不同（Wang 2023：冷却塔与空冷 x1.9–2.2，
    直流 x1.2–1.5）。空冷矩阵取 min(空冷, 湿冷)，转换不能反而多耗水（海水冷却厂淡水为零）。
    """
    water_base = prepared.plants["baseline_water_intensity_m3_per_mwh"].astype(float).to_numpy()
    capture_base = prepared.plants["capture_water_intensity_m3_per_mwh"].astype(float).to_numpy()
    water_intensity = np.zeros((len(prepared.plants), len(PATHWAYS)), dtype=np.float64)
    water_intensity[:, PATHWAY_INDEX["unabated"]] = water_base
    water_intensity[:, PATHWAY_INDEX["retire"]] = 0.0
    water_intensity[:, PATHWAY_INDEX["ccs"]] = capture_base * scenario.ccs_water_multiplier_adjustment
    water_intensity[:, PATHWAY_INDEX["biomass"]] = water_base * assumptions.biomass_water_multiplier
    water_intensity[:, PATHWAY_INDEX["beccs"]] = capture_base * scenario.beccs_water_multiplier_adjustment
    water_intensity[:, PATHWAY_INDEX["ammonia"]] = water_base * assumptions.ammonia_water_multiplier

    if "air_consumption_intensity_m3_per_mwh" in prepared.plants.columns:
        air_base = prepared.plants["air_consumption_intensity_m3_per_mwh"].astype(float).to_numpy()
        air_capture = prepared.plants["air_consumption_ccs_intensity_m3_per_mwh"].astype(float).to_numpy()
    else:
        air_base = water_base.copy()
        air_capture = capture_base.copy()
    air_water_intensity = np.zeros_like(water_intensity)
    air_water_intensity[:, PATHWAY_INDEX["unabated"]] = air_base
    air_water_intensity[:, PATHWAY_INDEX["retire"]] = 0.0
    air_water_intensity[:, PATHWAY_INDEX["ccs"]] = air_capture * scenario.ccs_water_multiplier_adjustment
    air_water_intensity[:, PATHWAY_INDEX["biomass"]] = air_base * assumptions.biomass_water_multiplier
    air_water_intensity[:, PATHWAY_INDEX["beccs"]] = air_capture * scenario.beccs_water_multiplier_adjustment
    air_water_intensity[:, PATHWAY_INDEX["ammonia"]] = air_base * assumptions.ammonia_water_multiplier
    air_water_intensity = np.minimum(air_water_intensity, water_intensity)
    return water_intensity, air_water_intensity


def _air_cooling_matrices(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    plant: dict[str, object],
) -> dict[str, object]:
    """湿冷→空冷改造：capex、背压能耗的排放/捕集/燃料成本矩阵，都只按仍湿冷的份额计。

    `air_share` 是按全空冷强度计价的发电份额，驱到 1 只转换仍湿冷的部分（基线强度已混入现有空冷），
    所以 capex 与背压惩罚都乘 still_wet；否则 87% 已空冷的宁夏 hub 会按全厂重建收费。
    """
    generation_by_pathway = plant["generation_by_pathway"]
    generation_cost_basis = plant["generation_cost_basis"]
    heat_rate_eff = plant["heat_rate_eff"]
    coal_price_per_plant = plant["coal_price_per_plant"]
    capacity_mw = plant["capacity_mw"]

    already_air_share = (
        prepared.plants["already_air_share"].astype(float).clip(0.0, 1.0).to_numpy()
        if "already_air_share" in prepared.plants.columns
        else np.zeros(len(prepared.plants), dtype=np.float64)
    )
    still_wet = 1.0 - already_air_share
    # 一次性 capex 计在新转换份额上，与 CCS 改造同一约定。
    air_retrofit_capex_per_plant = (
        capacity_mw * 1000.0 * float(assumptions.air_retrofit_capex_cny_per_kw) * still_wet
    )
    # 空冷背压升高：每 MWh 多烧煤、多排 CO2，按本厂基线排放比例计。
    penalty_ratio = float(assumptions.air_retrofit_efficiency_penalty_pp) / max(
        1e-6, float(assumptions.coal_plant_base_efficiency)
    )
    air_penalty_gross_matrix = (
        generation_by_pathway
        * assumptions.coal_emission_factor_t_per_mwh / 1_000_000.0
        * penalty_ratio
        * still_wet[:, None]
    )
    air_penalty_gross_matrix[:, PATHWAY_INDEX["retire"]] = 0.0
    # 同一锅炉同一捕集装置：捕集路径上背压惩罚燃料按 (1-η) 排放、按 η 捕集，与 CCS 能耗惩罚一致。
    capture_pathway_mask = np.zeros(len(PATHWAYS), dtype=np.float64)
    capture_pathway_mask[[PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]]] = 1.0
    air_penalty_emissions_matrix = air_penalty_gross_matrix * (
        1.0 - capture_pathway_mask[None, :] * float(scenario.capture_rate)
    )
    air_penalty_captured_matrix = (
        air_penalty_gross_matrix * capture_pathway_mask[None, :] * float(scenario.capture_rate)
    )
    # 多烧的煤也要买（只计碳价会低估约 17%）。
    air_penalty_cost_matrix = (
        generation_cost_basis
        * (penalty_ratio * heat_rate_eff * coal_price_per_plant)[:, None]
        * still_wet[:, None]
    )
    air_penalty_cost_matrix[:, PATHWAY_INDEX["retire"]] = 0.0
    return {
        "already_air_share": already_air_share,
        "air_retrofit_capex_per_plant": air_retrofit_capex_per_plant,
        "air_penalty_emissions_matrix": air_penalty_emissions_matrix,
        "air_penalty_captured_matrix": air_penalty_captured_matrix,
        "air_penalty_cost_matrix": air_penalty_cost_matrix,
        "allow_air_cooling_retrofit": bool(assumptions.allow_air_cooling_retrofit),
    }
