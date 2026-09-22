"""Industrial point sources as DECISION AGENTS, co-optimised with the coal fleet.

Before this module the 2 552 industrial sources existed only as inputs: `builders/industry.py`
clustered them into 390 sector hubs, and `builders/water_quota.py` subtracted their current
withdrawal from the basin reservation so the coal fleet could compete for it. Nothing in
`optimization/` ever read them. Their abatement was exogenous — zero — and the basin residual
they created was handed entirely to coal.

What is joint here:

1. **The CO2 network.** Industrial capture is injected at the hub's own node in the same
   pipeline graph the coal hubs use, competes for the same edge capacity, and is stored in the
   same sinks against the same injectivity limits. Every industrial hub also gets direct
   candidate arcs to its nearest sinks, the same rule the coal hubs have.
2. **The basin water cap.** Industrial site water and the water penalty of industrial capture
   are counted against the same `residual_m3_per_year` that coal power draws on.
3. **The hydrogen supply.** A hub on the H2 route draws hydrogen from the SAME green-ammonia
   nodes that supply coal-side co-firing, within the same radius, and pays each node's own
   plant-gate LCOH plus transport -- so the two hydrogen users compete for quantity, and the
   price industry pays is the marginal one, not the national supply-weighted mean.
4. **The emission targets.** Either the legacy single joint reduction floor, or (from
   2026-09-10) one residual cap per sector group -- steel, cement, chemicals, and power for
   the coal fleet -- read from `inputs/sector_targets_<source>.csv`. With sector caps the
   carbon price is normally zero; when it is not, it is charged on industrial residuals
   exactly as on coal residuals (before 2026-09-10 only coal paid it).

Three routes per hub — `unabated`, `ccs`, `h2` — with `h2` available only where
`SECTOR_HAS_H2_ROUTE` is true (so cement and EAF steel get capture or nothing, which is the
right answer for a process-CO2 source). Route shares are continuous in [0, 1] and MONOTONE
across planning years: a hub that installs capture cannot uninstall it, and it cannot swap
capture for hydrogen.

COST BASIS (author's decision 2026-09-22; the same one the coal side has always used). Every
industrial route is priced as

    capex   = unit retrofit capex x capacity built,  charged ONCE on the route-share increment
              (shares are monotone, so the increment is the newly built stock)
    annual  = fixed O&M (a share of that capex per year)
            + energy and consumables at the model's OWN coal and electricity prices
            + (H2 route) the non-hydrogen operating delta against the incumbent
            + (H2 route) hydrogen bought per supply link in the solver

and the undepreciated part of every capex is credited back at the end of the horizon
(`salvage_credit` in the solver). NO levelised per-tonne cost enters the objective any more.
Between 2026-09-10 and 2026-09-22 the ACCA21 levelised capture costs were split into a
capital share de-annualised at CRF and an annual remainder; that kept the levelised
figure's implicit capital-recovery assumption inside a model whose whole point is to decide
when to build. The ACCA21 ranges survive as a cross-check
(`constants_industry.levelised_capture_cost_cny_per_t`).

CCS route (`constants_industry.INDUSTRY_CCS_*`): capex per tonne of annual capture capacity
from Chinese project filings (cement 1 150, steel 1 000, high-concentration chemicals 450
CNY/(t/a)), 5%/a fixed O&M, reboiler steam raised in a site coal boiler at the hub's
provincial coal price, electricity at the scenario price, 15 or 5 CNY/t consumables. The
steam CO2 is VENTED and subtracted from the route's reduction (before 2026-09-22 the ACCA21
unit cost was taken as energy-inclusive and nothing was vented). The coal-side learning
curve scales the capex (and with it the fixed O&M), as before.

H2 route (`INDUSTRY_H2_ROUTE_*`): capex per tonne of annual product capacity (H2-DRI shaft +
EAF 3 500; ammonia / methanol hydrogen tie-in 500 CNY/(t/a)), 3.5%/a fixed O&M, and a
non-hydrogen operating delta backed out of the literature premium anchor:

    opex_delta = premium_ref - k * P_ref - capex * (CRF(r, life) + fom)

so the anchor is reproduced exactly at its own reference hydrogen price and moved to the
price actually paid through the hub's hydrogen intensity k. `opex_delta` is negative for
steel (avoided coke and BF opex exceed the EAF power bill); the solver's floor
`annual + hydrogen purchase >= 0` is the statement that switching cannot be cheaper than
running the sunk incumbent, and it still credits the avoided fossil feedstock in full, so H2
uptake remains an upper bound.

KNOWN BIASES that remain:
* Electrolysis water is not charged (10-22 L/kg H2): it belongs to the basin of the
  electrolyser, and the coal side's ammonia co-firing does not charge it either.
* Output follows an exogenous index (TIMES CN60 when `industry_output_index_source` is set);
  there is no plant-level retirement decision for industry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover
    gp = None

from ..constants import AMMONIA_FLOW_SCALE, NH3_H2_RATIO, WATER_FLOW_SCALE
from ..constants_industry import (
    INDUSTRY_CAPTURE_LIFETIME_YEARS,
    INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2,
    INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR,
    INDUSTRY_CCS_FIXED_OM_FRACTION,
    INDUSTRY_H2_ABATEMENT_FRACTION,
    INDUSTRY_H2_LIFETIME_YEARS,
    INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION,
    INDUSTRY_H2_USES_ADVANCED_QUOTA,
    INDUSTRY_ROUTES,
    INDUSTRY_SECTORS,
    SECTOR_HAS_H2_ROUTE,
    SECTOR_TARGET_GROUP,
    capture_capex_cny_per_t_yr,
    capture_steam_co2_t_per_t,
    capture_variable_cost_cny_per_t,
    h2_route_capex_cny_per_t_yr,
    h2_route_opex_delta_cny_per_t,
    water_quota,
)
from ..paths import ProjectPaths

logger = logging.getLogger(__name__)

ROUTE_INDEX = {name: index for index, name in enumerate(INDUSTRY_ROUTES)}
UNABATED, CCS, H2 = ROUTE_INDEX["unabated"], ROUTE_INDEX["ccs"], ROUTE_INDEX["h2"]

# Feedstock key used when reading the advanced/general quota ratio per sector. The hub table
# carries a capacity-weighted blend of its members' feedstocks but not the mix itself, so the
# ratio is taken on the sector's default row and applied to the hub's own blended intensity.
# The spread across feedstocks is small (0.61-0.73 for the three H2-route sectors), so this
# cannot move a result; it is recorded rather than hidden.
_DEFAULT_FEEDSTOCK = "default"


@dataclass(frozen=True)
class IndustryInputs:
    """Industrial hubs prepared for optimisation, plus the hydrogen price path."""

    hubs: pd.DataFrame
    # industry_hubs.csv filtered to INDUSTRY_SECTORS, with `basin_code` added.
    h2_price_cny_per_kg: dict[int, float]
    # National supply-weighted LCOH per planning year, REPORTING ONLY since 2026-09-10: the
    # model buys hydrogen per link at the node's own price.
    output_index: dict[tuple[str, int], float] = field(default_factory=dict)
    # {(sector, year): production index, 2030 = 1}; empty holds output flat.


def _national_h2_price(paths: ProjectPaths, usd_to_cny: float) -> dict[int, float]:
    """Supply-weighted mean LCOH per year, CNY/kg, from the repo's own hydrogen supply curve.

    Reported beside the per-link prices so a reader can see how far the marginal price the
    model pays sits from the mean of the whole potential.

    Raises:
        FileNotFoundError: The supply curve has not been built.
        ValueError: The curve lacks the columns this needs.
    """
    path = paths.inputs_dir / "ammonia_supply_curve.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; industry's H2 route is priced from it. Build it with "
            "scripts/build_ammonia_supply.py, or disable industry."
        )
    curve = pd.read_csv(path, usecols=["year", "h2_supply_kg_per_year", "weighted_lcoh_usd_per_kg_h2"])
    missing = {"year", "h2_supply_kg_per_year", "weighted_lcoh_usd_per_kg_h2"} - set(curve.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    prices: dict[int, float] = {}
    for year, block in curve.groupby(curve["year"].astype(int)):
        weight = block["h2_supply_kg_per_year"].astype(float)
        total = float(weight.sum())
        if total <= 0:
            raise ValueError(f"{path}: zero hydrogen supply in {year}, cannot weight LCOH")
        lcoh_usd = float((block["weighted_lcoh_usd_per_kg_h2"].astype(float) * weight).sum() / total)
        prices[int(year)] = lcoh_usd * float(usd_to_cny)
    return prices


def prepare_industry(
    paths: ProjectPaths, assumptions, output_index: dict[tuple[str, int], float] | None = None
) -> IndustryInputs:
    """Load and prepare the industrial hubs. Call only when industry is enabled.

    Args:
        paths: Project paths.
        assumptions: `OptimizationAssumptions`; `usd_to_cny` and `water_budget` are read.
        output_index: {(sector, year): index}; None or empty holds output flat.

    Returns:
        Prepared industrial inputs.

    Raises:
        FileNotFoundError: `industry_hubs.csv` has not been built.
        ValueError: A hub carries a sector this module has no parameters for.
    """
    path = paths.inputs_dir / "industry_hubs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with `builders.industry.write_industry_inputs`, or "
            "leave `include_industry` off."
        )
    hubs = pd.read_csv(path)
    # 西藏不参与减排：备选管网里已经没有西藏点源（`builders.network_branches`
    # 同一张排除表），优化侧必须用同一个点源集合，否则会给一个没有管网接入的点源
    # 派任务。部门碳目标是"各组自身 2030 基线的比例"，基线随点源集合一起缩放，
    # 因此剔除既不放松也不收紧目标。
    from ..builders.network_branches import EXCLUDED_PROVINCES as _EXCLUDED_PROVINCES

    dropped = hubs["province"].astype(str).str.strip().str.lower().isin(_EXCLUDED_PROVINCES)
    if bool(dropped.any()):
        logger.info("industry: dropped %d hub(s) in non-abating provinces", int(dropped.sum()))
        hubs = hubs.loc[~dropped].reset_index(drop=True)
    hubs = hubs[hubs["sector"].astype(str).isin(INDUSTRY_SECTORS)].reset_index(drop=True)
    if hubs.empty:
        raise ValueError(f"{path} has no rows in the in-scope sectors {sorted(INDUSTRY_SECTORS)}")
    unknown = sorted(set(hubs["sector"].astype(str)) - set(INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR))
    if unknown:
        raise ValueError(f"no capture capex sourced for sector(s) {unknown}; refusing to guess")
    hubs["target_group"] = hubs["sector"].astype(str).map(SECTOR_TARGET_GROUP)
    if hubs["target_group"].isna().any():
        raise ValueError("a hub's sector has no entry in SECTOR_TARGET_GROUP")

    # Basin of the hub's own location, matching how `_prepare_plants` attributes coal hubs:
    # the withdrawal permit follows the site, not the intake. Only needed for the basin cap.
    if str(getattr(assumptions, "water_budget", "runoff")) == "official_quota":
        from ..builders.water import _assign_basin_codes

        # industry_hubs.csv already names the columns `latitude`/`longitude`, which is what
        # `_assign_basin_codes` expects; no rename needed (plants.csv needs one, this does not).
        hubs["basin_code"] = _assign_basin_codes(paths, hubs)

    prices = _national_h2_price(paths, float(assumptions.usd_to_cny))
    index = dict(output_index or {})
    if index:
        sectors_missing = sorted(
            {str(s) for s in hubs["sector"]} - {sector for sector, _ in index}
        )
        if sectors_missing:
            raise ValueError(f"industry output index has no rows for sector(s) {sectors_missing}")
    logger.info(
        "industry: %d hubs, %.0f Mt CO2/yr baseline, %.1f 亿 m3/yr water; national mean H2 price %s CNY/kg; output index %s",
        len(hubs),
        float(hubs["co2_mt_per_year"].sum()),
        float(hubs["water_m3_per_year"].sum()) / 1e8,
        {y: round(p, 1) for y, p in sorted(prices.items()) if y in (2030, 2060)},
        "on" if index else "flat",
    )
    return IndustryInputs(hubs=hubs, h2_price_cny_per_kg=prices, output_index=index)


def prepare_industry_h2_links(
    hubs: pd.DataFrame, ammonia_supply: pd.DataFrame, radius_km: float
) -> pd.DataFrame:
    """Candidate (hub, ammonia node) hydrogen links per supply year, within `radius_km`.

    Only hubs whose sector has an H2 route get links. Nodes are the same 0.5-degree green
    ammonia nodes the coal side draws on; `lcoh_usd_per_kg` is the node's plant-gate hydrogen
    cost before Haber-Bosch, which is what an industrial hub taking hydrogen pays.

    Args:
        hubs: Prepared industrial hubs (needs `hub_id`, `sector`, `longitude`, `latitude`).
        ammonia_supply: `ammonia_supply_curve.csv` as loaded by `_prepare_ammonia_supply`.
        radius_km: Matching radius, the coal side's `resource_match_radius_km`.

    Returns:
        Columns `year, hub_id, ammonia_node_id, distance_km, lcoh_usd_per_kg`.
    """
    from .data_prep import _haversine_distances_km

    columns = ["year", "hub_id", "ammonia_node_id", "distance_km", "lcoh_usd_per_kg"]
    eligible = hubs[hubs["sector"].astype(str).map(lambda s: bool(SECTOR_HAS_H2_ROUTE.get(s, False)))]
    if eligible.empty or ammonia_supply.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for year, nodes in ammonia_supply.groupby("year", sort=True):
        nodes = nodes.reset_index(drop=True)
        lons = nodes["longitude"].astype(float).to_numpy()
        lats = nodes["latitude"].astype(float).to_numpy()
        node_ids = nodes["ammonia_node_id"].astype(str).to_numpy()
        lcoh = nodes["weighted_lcoh_usd_per_kg_h2"].astype(float).to_numpy()
        for hub in eligible.itertuples(index=False):
            distances = _haversine_distances_km(float(hub.longitude), float(hub.latitude), lons, lats)
            for node_idx in np.flatnonzero(distances <= radius_km):
                rows.append({
                    "year": int(year),
                    "hub_id": str(hub.hub_id),
                    "ammonia_node_id": node_ids[node_idx],
                    "distance_km": round(float(distances[node_idx]), 3),
                    "lcoh_usd_per_kg": float(lcoh[node_idx]),
                })
    links = pd.DataFrame(rows, columns=columns)
    unreachable = sorted(set(eligible["hub_id"].astype(str)) - set(links["hub_id"].astype(str)))
    if unreachable:
        logger.warning(
            "industry: %d H2-capable hubs have no hydrogen node within %.0f km; their H2 route "
            "is closed by the balance (first few: %s)", len(unreachable), radius_km, unreachable[:5],
        )
    logger.info("Industry H2 links: %d (across all years)", len(links))
    return links


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
) -> dict[str, object]:
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
        Dict of coefficient arrays plus scalars the solver needs.
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
    cost_multiplier = float(getattr(scenario, "industry_cost_multiplier", 1.0))
    h2_multiplier = float(getattr(scenario, "industry_h2_cost_multiplier", 1.0))
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
    return {
        "hub_ids": hubs["hub_id"].astype(str).tolist(),
        "sectors": sectors,
        "target_groups": hubs["target_group"].astype(str).to_numpy(),
        "output_scale": output_scale,
        "baseline_emissions_mt": co2_mt,
        "route_available": route_available,
        "reduction_mt": reduction_mt,
        "captured_mt": captured_mt,
        "opex_cny": opex_cny,
        "capex_cny": capex_cny,
        # Kept under the old key for readers that sum a single cost matrix; it is the annual
        # part only and no longer contains the hydrogen purchase.
        "annual_cost_cny": opex_cny,
        "h2_demand_kg_per_share": h2_demand_kg,
        "water_m3": water_m3,
        "h2_price_cny_per_kg": h2_price_mean,
        "capture_learning_factor": learning,
        # Economic lives per route, read by the solver's end-of-horizon salvage credit.
        "capex_lifetime_years": {CCS: INDUSTRY_CAPTURE_LIFETIME_YEARS, H2: INDUSTRY_H2_LIFETIME_YEARS},
    }


def add_industry_year(
    model,
    industry: IndustryInputs,
    ydata: dict,
    year_suffix: str,
    h2_link_cost: np.ndarray | None = None,
    h2_hub_membership=None,
) -> dict:
    """Add one planning year's industrial variables and within-year constraints.

    Args:
        model: The Gurobi model under construction.
        industry: Prepared industrial inputs.
        ydata: Output of `industry_year_data` for this year.
        year_suffix: Year string appended to every variable and constraint name.
        h2_link_cost: Delivered hydrogen cost per link, CNY per scaled kg (None: no links).
        h2_hub_membership: Sparse (hub, link) incidence for the hydrogen links.

    Returns:
        Payload with the route share variable and the derived expressions the solver wires
        into the CO2 network, the basin cap, the targets, the hydrogen nodes and the objective.
    """
    if gp is None:  # pragma: no cover
        raise RuntimeError("gurobipy is required to build the industrial block.")
    n = len(industry.hubs)
    n_routes = len(INDUSTRY_ROUTES)
    share = model.addMVar((n, n_routes), lb=0.0, ub=1.0, name=f"ind_share_{year_suffix}")
    model.addConstrs(
        (share[hub, :].sum() == 1.0 for hub in range(n)), name=f"ind_share_sum_{year_suffix}"
    )
    unavailable = np.argwhere(~ydata["route_available"])
    if len(unavailable):
        model.addConstrs(
            (share[int(hub), int(route)] == 0.0 for hub, route in unavailable),
            name=f"ind_route_unavailable_{year_suffix}",
        )
    captured = ydata["captured_mt"]
    captured_by_hub = [
        gp.quicksum(float(captured[hub, r]) * share[hub, r] for r in range(n_routes))
        for hub in range(n)
    ]
    reduction = ydata["reduction_mt"]
    baseline = ydata["baseline_emissions_mt"]
    reduction_by_hub = [
        gp.quicksum(float(reduction[hub, r]) * share[hub, r] for r in range(n_routes))
        for hub in range(n)
    ]
    total_reduction = gp.quicksum(reduction_by_hub)
    groups = ydata["target_groups"]
    residual_by_group = {
        str(group): gp.quicksum(
            float(baseline[hub]) - reduction_by_hub[hub] for hub in np.flatnonzero(groups == group)
        )
        for group in sorted(set(str(g) for g in groups))
    }
    water = ydata["water_m3"]
    # Scaled to the same Mm3 units the coal side's basin expression uses, so the two can be
    # added inside one constraint without a hidden unit change.
    withdrawal_by_hub = [
        gp.quicksum(float(water[hub, r]) / WATER_FLOW_SCALE * share[hub, r] for r in range(n_routes))
        for hub in range(n)
    ]
    opex = ydata["opex_cny"]
    annual_cost = gp.quicksum(
        float(opex[hub, r]) * share[hub, r] for hub in range(n) for r in (CCS,)
    )

    # Hydrogen: per-link purchase, hub balance, and the floored H2-route annual cost.
    h2_demand = ydata["h2_demand_kg_per_share"]
    link_count = 0 if h2_link_cost is None else int(len(h2_link_cost))
    h2_flow_kg = model.addMVar(link_count, lb=0.0, name=f"ind_h2_flow_kg_{year_suffix}")
    h2_route_cost = model.addMVar(n, lb=0.0, name=f"ind_h2_cost_{year_suffix}")
    if link_count:
        hub_draw = h2_hub_membership @ h2_flow_kg
        purchase_by_hub = [None] * n
        # Group link costs by hub once, so each hub's purchase is a short expression.
        coo = h2_hub_membership.tocoo()
        links_of_hub: dict[int, list[int]] = {}
        for hub_idx, link_idx in zip(coo.row.tolist(), coo.col.tolist()):
            links_of_hub.setdefault(int(hub_idx), []).append(int(link_idx))
        for hub in range(n):
            hub_links = links_of_hub.get(hub, [])
            purchase_by_hub[hub] = gp.quicksum(
                float(h2_link_cost[l]) * h2_flow_kg[l] for l in hub_links
            ) if hub_links else gp.LinExpr()
            model.addConstr(
                hub_draw[hub] == float(h2_demand[hub]) / AMMONIA_FLOW_SCALE * share[hub, H2],
                name=f"ind_h2_balance_{hub}_{year_suffix}",
            )
            if float(h2_demand[hub]) > 0.0:
                model.addConstr(
                    h2_route_cost[hub] >= float(opex[hub, H2]) * share[hub, H2] + purchase_by_hub[hub],
                    name=f"ind_h2_cost_lb_{hub}_{year_suffix}",
                )
    else:
        # No hydrogen links at all: the route can only be taken by hubs with no demand, which
        # `industry_year_data` has already closed. Force the balance explicitly anyway.
        for hub in range(n):
            if float(h2_demand[hub]) > 0.0:
                model.addConstr(share[hub, H2] == 0.0, name=f"ind_h2_no_supply_{hub}_{year_suffix}")
    annual_cost = annual_cost + h2_route_cost.sum()

    return {
        "share": share,
        "year_suffix": year_suffix,
        "captured_by_hub": captured_by_hub,
        "reduction_by_hub": reduction_by_hub,
        "total_reduction_mt": total_reduction,
        "residual_by_group": residual_by_group,
        "withdrawal_by_hub_scaled": withdrawal_by_hub,
        "annual_cost_cny": annual_cost,
        "h2_flow_kg": h2_flow_kg,
        "h2_route_cost": h2_route_cost,
        "year_data": ydata,
    }


def add_industry_monotonicity(model, payloads: list[dict], hub_count: int) -> None:
    """Abatement routes are irreversible: `ccs` and `h2` shares never fall between years.

    With the shares summing to one this also forbids swapping capture for hydrogen — you do
    not demolish a capture island to build a DRI shaft — and it is what lets the one-time
    capital charge be levied on the share INCREMENT: the increment is the newly built stock.

    Args:
        model: The Gurobi model.
        payloads: One payload per planning year, in chronological order.
        hub_count: Number of industrial hubs.
    """
    for index in range(1, len(payloads)):
        current = payloads[index]["share"]
        previous = payloads[index - 1]["share"]
        suffix = payloads[index]["year_suffix"]
        for route in (CCS, H2):
            model.addConstrs(
                (current[hub, route] >= previous[hub, route] for hub in range(hub_count)),
                name=f"ind_mono_{INDUSTRY_ROUTES[route]}_{suffix}",
            )


def industry_capex_expr(payload: dict, previous: dict | None, routes: tuple[int, ...] = (CCS, H2)):
    """One-time capital charge for the year: capex coefficient x route-share increment.

    Shares are monotone (`add_industry_monotonicity`), so `share_t - share_{t-1}` is the newly
    built stock and never negative; in the first year the whole share is new.

    Args:
        payload: The year's industry payload (`share` variables and `year_data`).
        previous: The previous year's payload, or None in the first year.
        routes: Route indices to include; the solver asks for CCS and H2 separately so each
            can carry its own economic life in the salvage credit.
    """
    share = payload["share"]
    capex = payload["year_data"]["capex_cny"]
    n = share.shape[0]
    terms = []
    for hub in range(n):
        for route in routes:
            coeff = float(capex[hub, route])
            if coeff <= 0.0:
                continue
            if previous is None:
                terms.append(coeff * share[hub, route])
            else:
                terms.append(coeff * (share[hub, route] - previous["share"][hub, route]))
    return gp.quicksum(terms) if terms else 0.0


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


H2_PER_NH3 = NH3_H2_RATIO
