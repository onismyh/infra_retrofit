"""煤电厂侧逐年系数矩阵：发电与排放、能耗惩罚、改造 capex 与运维、耗水强度、空冷改造。

所有矩阵形状 (plant_count, len(PATHWAYS))，列序同 PATHWAYS = unabated/retire/ccs/biomass/beccs/ammonia。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import GJ_PER_MWH
from ._shared import PATHWAY_INDEX, PreparedInputs
from .scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario


def _plant_operating_matrices(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
) -> dict[str, Any]:
    """发电、排放、运行成本、能耗惩罚、CCS/BECCS capex、搁浅资产。

    随毛热耗变的系数（`by_heat_rate`）按未重建部分的毛热耗算；原址重建部分按重建热耗分类，每类与未重建部分算出的
    系数之差放在 `rebuilt_deltas`，乘该类重建部分的份额加进约束与成本（`constraints._add_rebuilt_split`）。
    """
    # 本年利用小时：`annual_generation_mwh` 是当前省级统计，按情景小时轨迹逐年缩放，
    # 发电、基线排放与每 MWh 成本同步移动。
    fleet_hours_now = float(prepared.plants["fleet_hours_now"].iloc[0]) if "fleet_hours_now" in prepared.plants.columns else 0.0
    hours_scale = float(scenario.operating_hours_scale(int(year), fleet_hours_now)) if fleet_hours_now > 0 else 1.0
    # 部分负荷修正（缺省关为 1，`scenario.part_load_factor`）：全国一个系数，乘在基线排放与下面各部分的毛热耗上。
    part_load = float(scenario.part_load_factor(int(year), fleet_hours_now))
    generation = prepared.plants["annual_generation_mwh"].astype(float).to_numpy() * hours_scale
    emissions_mt = prepared.plants["baseline_emissions_mt"].astype(float).to_numpy() * hours_scale * part_load
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
    # hub 毛热耗（基线排放按它算，`data_prep._prepare_plants`）与各部分的毛热耗（`data_prep._with_expiry`）：
    # 未重建部分（未到期机组；全部到期的 hub 取各类重建热耗的装机加权平均）与各类原址重建部分（到期机组逐台
    # min(机组毛热耗, 3.6 / rebuild_efficiency，空冷机组加 +15 g/kWh)，按这个值分类）。本年没有某类到期装机的 hub，
    # 该类取未重建部分的毛热耗，差恰为零。2026-10-02 前全国一个热耗 8.5714，整个 hub 自 `retirement_year` 起乘
    # 0.42 / rebuild_efficiency，不论到期装机重建还是退役。
    hub_heat_rate = prepared.plants["heat_rate_gj_per_mwh"].astype(float).to_numpy() * part_load
    heat_rate_unexpired = prepared.plants[f"heat_rate_unexpired_{year}"].astype(float).to_numpy() * part_load
    expired_share = prepared.plants[f"expired_share_{year}"].astype(float).to_numpy()
    classes = range(sum(str(column).startswith("heat_rate_rebuilt_c") for column in prepared.plants.columns))
    rebuilt_class_share = np.column_stack(
        [prepared.plants[f"expired_share_c{c}_{year}"].astype(float).to_numpy() for c in classes]
    )
    rebuilt_heat_rates = [
        np.where(
            rebuilt_class_share[:, c] > 0.0,
            prepared.plants[f"heat_rate_rebuilt_c{c}"].astype(float).to_numpy() * part_load,
            heat_rate_unexpired,
        )
        for c in classes
    ]
    # 成本基数：退役列保留基线发电量（退役成本按原发电量计），物理量用 generation_by_pathway（退役列为零）。
    generation_cost_basis = generation_by_pathway.copy()
    generation_cost_basis[:, PATHWAY_INDEX["retire"]] = generation
    # 每 MWh 附加项不乘 `ccs_cost_multiplier`：CCS、生物质、BECCS 为 0（掺烧运维按掺烧能力计，见下方
    # `biomass_blend_om_per_level`；2026-10-02 前生物质、BECCS 各 30 元/MWh，2026-09-23 前 CCS、BECCS 两项都乘）。
    pathway_fixed_costs = np.array(
        [
            assumptions.fixed_cost_cny_per_mwh("unabated"),
            assumptions.fixed_cost_cny_per_mwh("retire"),
            assumptions.fixed_cost_cny_per_mwh("ccs"),
            assumptions.fixed_cost_cny_per_mwh("biomass"),
            assumptions.fixed_cost_cny_per_mwh("beccs"),
            assumptions.fixed_cost_cny_per_mwh("ammonia"),
        ],
        dtype=np.float64,
    )
    coal_price_per_plant = np.array([
        assumptions.province_coal_cost(str(prov))
        for prov in prepared.plants["province_name"]
    ], dtype=np.float64)
    eps_ratio_ccs = assumptions.ccs_energy_penalty_ratio(year)
    energy_penalty_per_pathway = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    uncaptured = 1.0 - float(scenario.capture_rate)
    emission_factor_t_per_gj = assumptions.coal_emission_factor_t_per_gj
    elec_price_year = scenario.electricity_price_for_year(year)
    # 容量电价（缺省 0，`scenario.coal_capacity_price_cny_per_kw_yr`）：按装机收，退役列为零（各列份额相加为 1，即按 1 − 退役份额收）；
    # 电量电价减去 容量电价 ÷ 全机组现状利用小时，按现状小时算的总收入不变。
    capacity_price = float(scenario.coal_capacity_price_cny_per_kw_yr)
    capacity_revenue_matrix = np.zeros_like(generation_by_pathway)
    if capacity_price != 0.0:
        if not 0.0 < capacity_price < float("inf"):
            raise ValueError(f"coal_capacity_price_cny_per_kw_yr must be positive and finite, got {capacity_price}")
        if not fleet_hours_now > 0.0:
            raise ValueError("coal_capacity_price_cny_per_kw_yr needs fleet_hours_now (current fleet hours) > 0")
        elec_price_year -= capacity_price * 1000.0 / fleet_hours_now
        capacity_kw = prepared.plants["total_capacity_mw"].astype(float).to_numpy() * 1000.0
        capacity_revenue_matrix[:] = (capacity_kw * capacity_price)[:, None]
        capacity_revenue_matrix[:, PATHWAY_INDEX["retire"]] = 0.0

    def by_heat_rate(heat_rate: np.ndarray) -> dict[str, np.ndarray]:
        """按给定毛热耗算的系数，键同 `YearData` 的字段。效率损失折算燃料按这部分自己的效率 η = 3.6 / 毛热耗。"""
        eta = GJ_PER_MWH / heat_rate
        # 两个排放基数（基线排放之外）：运行（基线排放按毛热耗 / hub 毛热耗缩放）、改造（再乘 CF 提升）。
        emissions_operating_mt = emissions_mt * (heat_rate / hub_heat_rate)
        # 能耗惩罚（效率损失 -> 多烧煤）：CCS 项只随份额变；生物质项随掺烧档位变，在 constraints 里 McCormick 线性化。
        # CCS 能耗惩罚直接以单位出力的额外燃料比表示，逐年下降：额外燃料 = ratio(year) x 毛热耗 [GJ/MWh]。
        ccs_penalty_fuel = eps_ratio_ccs * heat_rate
        # 生物质档位效率惩罚：每单位掺烧比例多烧 (ε_per_ratio / η) x 毛热耗 [GJ/MWh]，乘 G_p x β_b 即惩罚燃料。
        biomass_penalty_fuel = assumptions.biomass_efficiency_penalty_per_ratio / eta * heat_rate
        # 能耗惩罚的排放（Mt）：补效率损失多烧的煤在同一锅炉燃烧，经同一捕集装置，只有未捕集份额排放
        # （Fan et al. 2023 Nat Clim Change SI eq. S42）；捕集份额是真实流量，计入捕集量。
        ccs_penalty_mt = generation_cost_basis * (
            (ccs_penalty_fuel * emission_factor_t_per_gj / 1_000_000.0)[:, None] * energy_penalty_per_pathway[None, :]
        )
        # 生物质掺烧的惩罚燃料：纯掺烧路径全部排放，BECCS 按 capture_rate 捕集。
        biomass_penalty_emissions = biomass_penalty_fuel * emission_factor_t_per_gj / 1_000_000.0
        return {
            "heat_rate_eff": heat_rate,
            "emissions_operating_mt": emissions_operating_mt,
            "emissions_retrofit_mt": emissions_operating_mt * scenario.retrofit_cf_boost,
            "energy_penalty_matrix": generation_cost_basis * (
                (ccs_penalty_fuel * coal_price_per_plant)[:, None] * energy_penalty_per_pathway[None, :]
            ),
            "biomass_penalty_coeff_per_level": biomass_penalty_fuel * coal_price_per_plant,
            "ccs_penalty_emissions_matrix": ccs_penalty_mt * uncaptured,
            "ccs_penalty_captured_matrix": ccs_penalty_mt * float(scenario.capture_rate),
            "biomass_penalty_emissions_coeff_per_level": biomass_penalty_emissions,
            "beccs_penalty_emissions_coeff_per_level": biomass_penalty_emissions * uncaptured,
            "beccs_penalty_captured_coeff_per_level": biomass_penalty_emissions * float(scenario.capture_rate),
            # 基线净运行成本（煤 + 运维 - 电量电费 - 容量电费）：未改造列按基线发电量，改造列含 CF 提升，退役列为零。
            "baseline_net_matrix": generation_by_pathway * (
                heat_rate * coal_price_per_plant + assumptions.baseline_om_cost_cny_per_mwh - elec_price_year
            )[:, None] - capacity_revenue_matrix,
        }

    unexpired_terms = by_heat_rate(heat_rate_unexpired)

    # CCS/BECCS 改造 capex（含学习曲线）。BECCS 的捕集岛就是 CCS 捕集岛，同价；生物质改造
    # 另由掺烧能力的 capex 计（`model_costs._blend_unit_capex`）。
    lf = assumptions.ccs_learning_factor(year)
    capture_island_capex_per_mw = assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * scenario.ccs_cost_multiplier * lf
    ccs_capex_per_mw = np.array([
        0.0,
        0.0,
        capture_island_capex_per_mw,   # ccs
        0.0,
        capture_island_capex_per_mw,   # beccs
        0.0,
    ], dtype=np.float64)
    capacity_mw = prepared.plants["total_capacity_mw"].astype(float).to_numpy()
    ccs_retrofit_capex_matrix = capacity_mw[:, None] * ccs_capex_per_mw[None, :]
    # 捕集岛的固定运维（ccs_om_fraction x 这笔 capex，每年）在 `year_matrices` 里随 capex 系数一起给。
    # 生物质掺烧能力的固定运维（CNY/yr，每单位掺烧能力 = 一个档位层 x 占装机的份额）：capex 单价
    # （`model_costs._blend_unit_capex`）x biomass_upgrade_om_fraction，乘在用的掺烧能力计入 `incremental_om`。
    biomass_blend_om_per_level = (
        capacity_mw * float(assumptions.biomass_upgrade_capex_cny_per_mw_per_level)
        * float(assumptions.biomass_upgrade_om_fraction)
    )

    fixed_cost_matrix = generation_cost_basis * pathway_fixed_costs[None, :]

    # 搁浅资产（每单位新增提前退役，`retirement.retirement_flows`）：新建成本 x 未到期装机的平均剩余账面份额
    # ℓ / (1 - f)。ℓ 是全 hub 的剩余账面份额（到期装机为 0，`data_prep._with_expiry`）；全部到期的 hub 为零，
    # 到期退役不罚。2026-10-02 前按 hub 的剩余寿命份额 min(1, max(0, retirement_year - 年) / 20) 计在退役份额的
    # 增量上（整数 hub 与现在的 ℓ 相同）。
    fraction_remaining = prepared.plants[f"remaining_life_fraction_{year}"].astype(float).to_numpy()
    unexpired = 1.0 - expired_share
    book_of_unexpired = np.divide(
        fraction_remaining, unexpired, out=np.zeros_like(unexpired), where=unexpired > 0.0
    )
    stranded_per_plant = capacity_mw * assumptions.stranded_asset_base_cny_per_kw * 1000.0 * book_of_unexpired

    return {
        "hours_scale": hours_scale,
        "part_load_factor": part_load,
        "generation": generation,
        "generation_by_pathway": generation_by_pathway,
        "generation_cost_basis": generation_cost_basis,
        "emissions_mt": emissions_mt,
        **unexpired_terms,
        "expired_share": expired_share,
        "capacity_mw": capacity_mw,
        "coal_price_per_plant": coal_price_per_plant,
        "fixed_cost_matrix": fixed_cost_matrix,
        "biomass_blend_om_per_level": biomass_blend_om_per_level,
        "cfb_share": prepared.plants["cfb_share"].astype(float).to_numpy(),
        "ccs_retrofit_capex_matrix": ccs_retrofit_capex_matrix,
        "stranded_per_plant": stranded_per_plant,
        "rebuilt_class_share": rebuilt_class_share,
        "rebuilt_heat_rates": rebuilt_heat_rates,
        "rebuilt_deltas": [
            {name: value - unexpired_terms[name] for name, value in by_heat_rate(heat_rate).items()}
            for heat_rate in rebuilt_heat_rates
        ],
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
    plant: dict[str, Any],
) -> dict[str, Any]:
    """湿冷→空冷改造：capex、背压能耗的排放/捕集/燃料成本矩阵，都只按仍湿冷的份额计。

    `air_share` 是按全空冷强度计价的发电份额，驱到 1 只转换仍湿冷的部分（基线强度已混入现有空冷），
    所以 capex 与背压惩罚都乘 still_wet；否则 87% 已空冷的宁夏 hub 会按全厂重建收费。
    燃料成本按未重建部分的毛热耗算，各类重建部分与它之差另给（`air_penalty_cost_rebuilt_deltas`，并入 `RebuiltDelta`）。
    """
    generation_by_pathway = plant["generation_by_pathway"]
    generation_cost_basis = plant["generation_cost_basis"]
    coal_price_per_plant = plant["coal_price_per_plant"]
    capacity_mw = plant["capacity_mw"]

    already_air_share = (
        prepared.plants["already_air_share"].astype(float).clip(0.0, 1.0).to_numpy()
        if "already_air_share" in prepared.plants.columns
        else np.zeros(len(prepared.plants), dtype=np.float64)
    )
    still_wet = 1.0 - already_air_share
    # 一次性 capex 计在新建的空冷能力上（在役 `air_retrofit_lifetime_years` 年，`model_linking.add_capacity_vintages`），
    # 与 CCS 改造同一约定。
    air_retrofit_capex_per_plant = (
        capacity_mw * 1000.0 * float(assumptions.air_retrofit_capex_cny_per_kw) * still_wet
    )
    # 空冷背压升高：效率降 pp 个百分点，每 MWh 多烧 (pp / η) x 毛热耗的煤（η = 3.6 / 毛热耗）。多排的 CO2 按 hub 毛热耗计，
    # 不随重建变；下面的燃料成本按各部分自己的毛热耗计（docs/参数调研_20261001.md §4 第 15 条，作者决定 2026-10-02 维持）。
    # 毛热耗都乘部分负荷修正（缺省关为 1；各部分的已在 `_plant_operating_matrices` 里乘过）。
    heat_rate = prepared.plants["heat_rate_gj_per_mwh"].astype(float).to_numpy() * plant["part_load_factor"]
    penalty_pp = float(assumptions.air_retrofit_efficiency_penalty_pp)
    penalty_ratio = penalty_pp / (GJ_PER_MWH / heat_rate)
    air_penalty_gross_matrix = (
        generation_by_pathway
        * (heat_rate * assumptions.coal_emission_factor_t_per_gj / 1_000_000.0 * penalty_ratio * still_wet)[:, None]
    )
    air_penalty_gross_matrix[:, PATHWAY_INDEX["retire"]] = 0.0
    # 同一锅炉同一捕集装置：捕集路径上背压惩罚燃料按 (1 − 捕集率) 排放、按捕集率捕集，与 CCS 能耗惩罚一致。
    capture_pathway_mask = np.zeros(len(PATHWAYS), dtype=np.float64)
    capture_pathway_mask[[PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]]] = 1.0
    air_penalty_emissions_matrix = air_penalty_gross_matrix * (
        1.0 - capture_pathway_mask[None, :] * float(scenario.capture_rate)
    )
    air_penalty_captured_matrix = (
        air_penalty_gross_matrix * capture_pathway_mask[None, :] * float(scenario.capture_rate)
    )
    # 多烧的煤也要买（只计碳价会低估约 17%）。
    def penalty_cost(part_heat_rate: np.ndarray) -> np.ndarray:
        cost = (
            generation_cost_basis
            * (penalty_pp / (GJ_PER_MWH / part_heat_rate) * part_heat_rate * coal_price_per_plant)[:, None]
            * still_wet[:, None]
        )
        cost[:, PATHWAY_INDEX["retire"]] = 0.0
        return cost

    air_penalty_cost_matrix = penalty_cost(plant["heat_rate_eff"])
    return {
        "air_retrofit_capex_per_plant": air_retrofit_capex_per_plant,
        "air_penalty_emissions_matrix": air_penalty_emissions_matrix,
        "air_penalty_captured_matrix": air_penalty_captured_matrix,
        "air_penalty_cost_matrix": air_penalty_cost_matrix,
        "air_penalty_cost_rebuilt_deltas": [
            penalty_cost(heat_rate) - air_penalty_cost_matrix for heat_rate in plant["rebuilt_heat_rates"]
        ],
        "allow_air_cooling_retrofit": bool(assumptions.allow_air_cooling_retrofit),
    }
