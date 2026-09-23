"""工业点源的逐年系数：每个 hub、每条路线的减排、捕集、取水、年度成本与一次性 capex。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants_industry import (
    INDUSTRY_CAPTURE_LIFETIME_YEARS,
    INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2,
    INDUSTRY_CCS_FIXED_OM_FRACTION,
    INDUSTRY_H2_ABATEMENT_FRACTION,
    INDUSTRY_H2_LIFETIME_YEARS,
    INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION,
    INDUSTRY_H2_USES_ADVANCED_QUOTA,
    INDUSTRY_ROUTES,
    SECTOR_HAS_H2_ROUTE,
    capture_capex_cny_per_t_yr,
    capture_steam_co2_t_per_t,
    capture_variable_cost_cny_per_t,
    h2_route_capex_cny_per_t_yr,
    h2_route_opex_delta_cny_per_t,
    water_quota,
)
from .industry_inputs import IndustryInputs

ROUTE_INDEX = {name: index for index, name in enumerate(INDUSTRY_ROUTES)}
UNABATED, CCS, H2 = ROUTE_INDEX["unabated"], ROUTE_INDEX["ccs"], ROUTE_INDEX["h2"]

# Feedstock key used when reading the advanced/general quota ratio per sector. The hub table
# carries a capacity-weighted blend of its members' feedstocks but not the mix itself, so the
# ratio is taken on the sector's default row and applied to the hub's own blended intensity.
# The spread across feedstocks is small (0.61-0.73 for the three H2-route sectors), so this
# cannot move a result; it is recorded rather than hidden.
_DEFAULT_FEEDSTOCK = "default"


@dataclass(frozen=True)
class IndustryYearData:
    """`industry_year_data` 的输出：一个规划年里每个工业 hub、每条路线的系数。

    路线矩阵形状 (hub_count, len(INDUSTRY_ROUTES))、列序同 INDUSTRY_ROUTES；其余形状在旁注明。
    成本分两部分：`opex_cny` 是年度部分，每个运行年按路线份额计；`capex_cny` 是一次性改造 capex，
    按路线份额增量计（`model_industry.industry_capex_expr`）。氢路线买氢不在 `opex_cny` 里，
    由求解器按链路采购。
    """

    hub_ids: list[str]
    sectors: np.ndarray                  # (hub_count,)
    target_groups: np.ndarray            # (hub_count,)，部门目标组
    output_scale: np.ndarray             # (hub_count,)，产量指数，2030 = 1
    baseline_emissions_mt: np.ndarray    # (hub_count,)，已乘产量指数
    route_available: np.ndarray          # bool
    reduction_mt: np.ndarray             # CCS 列已扣除放空的再生蒸汽 CO2
    captured_mt: np.ndarray
    opex_cny: np.ndarray                 # 年度：固定运维 + 能耗 + 耗材 + 氢路线非氢运行差额
    capex_cny: np.ndarray                # 一次性：整个 hub 在该路线上的改造 capex
    h2_demand_kg_per_share: np.ndarray   # (hub_count,)，氢路线份额为 1 时的年需氢量，kg
    water_m3: np.ndarray
    h2_price_cny_per_kg: float           # 全国供给加权均价，只作报告
    capture_learning_factor: float
    capex_lifetime_years: dict[int, int]  # 路线下标 -> 经济寿命（年），供期末残值


def _h2_price_for_year(prices: dict[int, float], year: int) -> float:
    """Hydrogen price in `year`, falling back to the nearest tabulated year.

    Matches how `scenario.carbon_price_for_year` handles off-grid years.
    """
    if year in prices:
        return float(prices[year])
    if not prices:
        raise ValueError("no hydrogen prices available")
    nearest = min(prices, key=lambda candidate: abs(candidate - year))
    return float(prices[nearest])


def _advanced_quota_ratio(sector: str) -> float:
    """Advanced-value / general-value water quota for a sector (both from GB/T 18916)."""
    general = water_quota(sector, _DEFAULT_FEEDSTOCK, advanced=False)
    advanced = water_quota(sector, _DEFAULT_FEEDSTOCK, advanced=True)
    if general <= 0:
        raise ValueError(f"sector {sector!r} has a non-positive general water quota")
    return float(advanced) / float(general)


def _output_scale(industry: IndustryInputs, sectors: np.ndarray, year: int) -> np.ndarray:
    if not industry.output_index:
        return np.ones(len(sectors), dtype=np.float64)
    available_years = sorted({y for _, y in industry.output_index})
    nearest = year if year in available_years else min(available_years, key=lambda y: abs(y - year))
    return np.array(
        [float(industry.output_index[(str(sector), nearest)]) for sector in sectors], dtype=np.float64
    )


def industry_year_data(
    industry: IndustryInputs, scenario, assumptions, year: int
) -> IndustryYearData:
    """Per-year cost, emission and water coefficients for every industrial hub and route.

    All arrays are shaped `(hub_count, len(INDUSTRY_ROUTES))` unless noted. Costs are split
    into an ANNUAL part (`opex_cny` = fixed O&M + energy + consumables + non-hydrogen operating
    delta, charged every operating year on the route share) and a ONE-TIME part (`capex_cny`
    = retrofit capex of the whole hub on that route, charged on the increment of the route
    share). The H2 route's hydrogen purchase is NOT in `opex_cny`: it is bought per link in
    the solver. `reduction_mt[:, CCS]` is net of the vented reboiler-steam CO2.

    Args:
        industry: Prepared industrial inputs.
        scenario: `OptimizationScenario`; capture rate, cost multipliers, discount rate.
        assumptions: `OptimizationAssumptions`; the CCS learning curve is read from it.
        year: Planning year.

    Returns:
        Coefficient arrays plus scalars the solver needs.
    """
    hubs = industry.hubs
    sectors = hubs["sector"].astype(str).to_numpy()
    n = len(hubs)
    output_scale = _output_scale(industry, sectors, int(year))
    co2_mt = hubs["co2_mt_per_year"].astype(float).to_numpy() * output_scale
    production_t = hubs["production_kt_per_year"].astype(float).to_numpy() * 1_000.0 * output_scale
    base_water_m3 = hubs["water_m3_per_year"].astype(float).to_numpy() * output_scale
    production_now = hubs["production_kt_per_year"].astype(float).to_numpy()
    h2_intensity_t_per_t = np.divide(
        hubs["h2_demand_kt_per_year"].astype(float).to_numpy(),
        np.where(production_now > 0, production_now, np.nan),
    )
    h2_intensity_t_per_t = np.nan_to_num(h2_intensity_t_per_t, nan=0.0)

    capture_rate = float(scenario.capture_rate)
    # Same exogenous learning curve the coal retrofits get, so the two sectors' capture costs
    # decline together. Using a different curve for industry would make the sectoral split a
    # function of an arbitrary modelling choice rather than of the technologies.
    learning = float(assumptions.ccs_learning_factor(year))
    cost_multiplier = float(scenario.industry_cost_multiplier)
    h2_multiplier = float(scenario.industry_h2_cost_multiplier)
    h2_price_mean = _h2_price_for_year(industry.h2_price_cny_per_kg, int(year))
    rate = float(scenario.discount_rate)
    # Energy for capture is priced at the model's own prices: the hub's provincial coal price
    # (same lookup the coal hubs use) for reboiler steam, the scenario electricity price for
    # compression and auxiliaries. So a coal-price or power-price sensitivity moves the two
    # sectors' capture costs together instead of leaving industry on a frozen literature price.
    provinces = hubs["province"].astype(str).to_numpy() if "province" in hubs.columns else np.array([""] * n)
    coal_price_gj = np.array([float(assumptions.province_coal_cost(p)) for p in provinces], dtype=np.float64)
    elec_price_mwh = float(scenario.electricity_price_for_year(int(year)))
    emission_factor_t_per_gj = float(assumptions.coal_emission_factor_t_per_mwh) / float(assumptions.heat_rate_gj_per_mwh)

    route_available = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=bool)
    route_available[:, UNABATED] = True
    route_available[:, CCS] = True
    reduction_mt = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    captured_mt = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    opex_cny = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    capex_cny = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    water_m3 = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    h2_demand_kg = np.zeros(n, dtype=np.float64)  # kg H2 per unit share on the H2 route

    # unabated: the hub as it runs today. No abatement, no extra cost, its existing intake.
    water_m3[:, UNABATED] = base_water_m3

    # ccs: capture `capture_rate` of everything the hub emits, combustion and process alike —
    # which is the whole point for cement, where 63% of emissions are calcination and no fuel
    # switch can touch them.
    captured_mt[:, CCS] = co2_mt * capture_rate
    captured_t = captured_mt[:, CCS] * 1e6
    # Retrofit capex of the capture island sized to the hub's captured tonnage (CNY per t/a of
    # capacity x t/a captured), learning-adjusted like the coal retrofits; fixed O&M as a share
    # of that capex; energy and consumables per tonne captured at this year's prices.
    capex_unit = np.array([capture_capex_cny_per_t_yr(s) for s in sectors], dtype=np.float64)
    capex_unit = capex_unit * learning * cost_multiplier
    variable_unit = np.array(
        [capture_variable_cost_cny_per_t(s, float(c), elec_price_mwh) for s, c in zip(sectors, coal_price_gj)],
        dtype=np.float64,
    ) * cost_multiplier
    capex_cny[:, CCS] = captured_t * capex_unit
    opex_cny[:, CCS] = captured_t * (capex_unit * INDUSTRY_CCS_FIXED_OM_FRACTION + variable_unit)
    # The reboiler steam is raised in a coal boiler whose CO2 is vented, so the route's net
    # reduction is captured minus that steam CO2 (zero for the compression-only chemical
    # streams). Same convention as the coal side's energy-penalty emissions.
    steam_co2_per_t = np.array(
        [capture_steam_co2_t_per_t(s, emission_factor_t_per_gj) for s in sectors], dtype=np.float64
    )
    reduction_mt[:, CCS] = captured_mt[:, CCS] * (1.0 - steam_co2_per_t)
    water_m3[:, CCS] = base_water_m3 + captured_t * INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2

    # h2: only where the sector has a route at all.
    quota_ratio = {s: _advanced_quota_ratio(s) for s in set(sectors) if SECTOR_HAS_H2_ROUTE.get(s, False)}
    for hub_idx in range(n):
        sector = sectors[hub_idx]
        if not SECTOR_HAS_H2_ROUTE.get(sector, False):
            continue
        if h2_intensity_t_per_t[hub_idx] <= 0.0 or production_t[hub_idx] <= 0.0:
            # A hub with an H2-capable sector but no hydrogen demand in the point-source table
            # cannot be priced. Leave the route closed rather than price it at zero.
            continue
        route_available[hub_idx, H2] = True
        reduction_mt[hub_idx, H2] = co2_mt[hub_idx] * float(INDUSTRY_H2_ABATEMENT_FRACTION[sector])
        k_kg_per_t = float(h2_intensity_t_per_t[hub_idx]) * 1000.0
        # Capex of the rebuilt route on the hub's whole output; fixed O&M on it; and the
        # non-hydrogen operating delta backed out of the literature anchor (module docstring).
        # The annual part BEFORE the hydrogen purchase can be negative (the anchor credits the
        # avoided fossil feedstock); the floor in the solver keeps annual + purchase >= 0.
        # `h2_multiplier` scales the route's own costs (capex and the anchor premium),
        # never the hydrogen: scaling the negative backed-out delta would invert the knob.
        route_capex_unit = h2_route_capex_cny_per_t_yr(sector) * h2_multiplier
        opex_delta_unit = h2_route_opex_delta_cny_per_t(
            sector, float(h2_intensity_t_per_t[hub_idx]), rate, h2_multiplier
        )
        capex_cny[hub_idx, H2] = production_t[hub_idx] * route_capex_unit
        opex_cny[hub_idx, H2] = production_t[hub_idx] * (
            route_capex_unit * INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION + opex_delta_unit
        )
        h2_demand_kg[hub_idx] = k_kg_per_t * production_t[hub_idx]
        ratio = quota_ratio[sector] if INDUSTRY_H2_USES_ADVANCED_QUOTA else 1.0
        water_m3[hub_idx, H2] = base_water_m3[hub_idx] * ratio

    # NB: `scenario.water_multiplier` scales AVAILABILITY, not demand, so it is deliberately
    # not applied here — it is applied once, to the basin residual, in `_basin_cap_data`.
    return IndustryYearData(
        hub_ids=hubs["hub_id"].astype(str).tolist(),
        sectors=sectors,
        target_groups=hubs["target_group"].astype(str).to_numpy(),
        output_scale=output_scale,
        baseline_emissions_mt=co2_mt,
        route_available=route_available,
        reduction_mt=reduction_mt,
        captured_mt=captured_mt,
        opex_cny=opex_cny,
        capex_cny=capex_cny,
        h2_demand_kg_per_share=h2_demand_kg,
        water_m3=water_m3,
        h2_price_cny_per_kg=h2_price_mean,
        capture_learning_factor=learning,
        # Economic lives per route, read by the solver's end-of-horizon salvage credit.
        capex_lifetime_years={CCS: INDUSTRY_CAPTURE_LIFETIME_YEARS, H2: INDUSTRY_H2_LIFETIME_YEARS},
    )


def basin_membership(industry: IndustryInputs, basin_codes: list[str]) -> np.ndarray:
    """(n_basins, n_hubs) indicator of which basin each industrial hub sits in.

    Kept separate from the coal membership rather than concatenated: the two index spaces stay
    distinct, so nothing downstream can silently read a hub index as a plant index.

    Args:
        industry: Prepared industrial inputs, carrying `basin_code`.
        basin_codes: Basin codes in the order the cap constraint uses.

    Returns:
        Indicator matrix.

    Raises:
        ValueError: A hub fell outside every basin in the cap table.
    """
    if "basin_code" not in industry.hubs.columns:
        raise ValueError("industry hubs carry no basin_code; prepare_industry ran without the official-quota budget")
    hub_basins = industry.hubs["basin_code"].astype(str).to_numpy()
    membership = np.zeros((len(basin_codes), len(hub_basins)), dtype=np.float64)
    for row, code in enumerate(basin_codes):
        membership[row, :] = (hub_basins == str(code)).astype(np.float64)
    unmatched = int(len(hub_basins) - membership.sum())
    if unmatched:
        raise ValueError(f"{unmatched} industrial hubs fell outside every basin in water_basin_caps.csv")
    return membership
