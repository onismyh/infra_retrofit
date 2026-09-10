"""Industrial point sources as DECISION AGENTS, co-optimised with the coal fleet.

Before this module the 2 552 industrial sources existed only as inputs: `builders/industry.py`
clustered them into 390 sector hubs, and `builders/water_quota.py` subtracted their current
withdrawal from the basin reservation so the coal fleet could compete for it. Nothing in
`optimization/` ever read them. Their abatement was exogenous — zero — and the basin residual
they created was handed entirely to coal.

What is joint here, precisely three things:

1. **The CO2 network.** Industrial capture is injected at the hub's own node in the same
   pipeline graph the coal hubs use, competes for the same edge capacity, and is stored in the
   same 35 sinks against the same injectivity limits. This is what the author asked for by
   "作为基础设施减排".
2. **The basin water cap.** Industrial site water and the water penalty of industrial capture
   are counted against the same `residual_m3_per_year` that coal power draws on. Basin K's
   residual is 2.1e8 m3 of which 2.4e8 is added-back modelled industry, so this is the
   constraint the whole exercise turns on.
3. **One emission target.** `total_reduction + industry_reduction >= target * (coal_baseline +
   industry_baseline)`. The solver, not the modeller, decides which sector abates.

Three routes per hub — `unabated`, `ccs`, `h2` — with `h2` available only where
`SECTOR_HAS_H2_ROUTE` is true (so cement and EAF steel get capture or nothing, which is the
right answer for a process-CO2 source). Route shares are continuous in [0, 1] and MONOTONE
across planning years: a hub that installs capture cannot uninstall it, and it cannot swap
capture for hydrogen. Monotonicity is what makes the levelised cost basis legitimate — see
below — and it is why there is no separate one-time CAPEX term.

COST BASIS, and why it differs from the coal side. The coal fleet is costed per MW and per MWh
with capital charged once on the increment of an installed stock. Industry is costed per tonne,
LEVELISED: `INDUSTRY_CAPTURE_COST_CNY_PER_T` and `h2_premium_cny_per_t` already contain capital
recovery, exactly as Tang et al. (2023) define `Z_cap = CAPEX*CRF + OPEX`. Charging a levelised
cost in every operating year is correct as long as the decision cannot be reversed for free —
which the monotonicity constraints guarantee. Splitting it back into a one-time CAPEX would
have needed a CRF and a capital share that neither Chinese source publishes;
`INDUSTRY_CAPTURE_CAPEX_SHARE` and `INDUSTRY_H2_CAPEX_SHARE` therefore survive only to split the
REPORTED cost in `industry_detail.csv`, and are not in the model's arithmetic.

KNOWN BIASES, stated because they push the answer in a known direction:

* **The H2 premium is a greenfield-vs-greenfield comparison** (Transition Asia's LCOS
  contrasts a new H2-DRI-EAF against a new BF-BOF). Our hubs are existing plants whose blast
  furnaces are sunk capital, so the true retrofit premium is higher by roughly the incumbent's
  annualised capital. **H2 uptake here is therefore an upper bound.**
  `scenario.industry_h2_cost_multiplier` exists to test it.
* **Electrolysis water is not charged** (10-22 L/kg H2), because it belongs to the basin of the
  electrolyser, not of the industrial hub, and the coal side's ammonia co-firing does not charge
  it either. See the note at the foot of `constants_industry`.
* **Hydrogen supply is not capacity-constrained and not spatially matched.** Industry pays a
  national supply-weighted LCOH from the same `ammonia_supply_curve.csv` that prices coal-side
  ammonia, but does not draw from those nodes, so the two hydrogen users do not compete for
  supply. At 2030 that curve offers 1 555 Mt H2/yr against the 103 Mt/yr full industrial
  substitution would need, so the missing cap is not binding at national scale; the missing
  SPATIAL matching is the real gap and is the first thing to add.
* **Industrial output is exogenous.** No retirement, no demand response, no material
  efficiency. A hub can only change how it makes the same tonnes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover
    gp = None

from ..constants import WATER_FLOW_SCALE
from ..constants_industry import (
    INDUSTRY_CAPTURE_COST_CNY_PER_T,
    INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2,
    INDUSTRY_H2_ABATEMENT_FRACTION,
    INDUSTRY_H2_USES_ADVANCED_QUOTA,
    INDUSTRY_ROUTES,
    INDUSTRY_SECTORS,
    SECTOR_HAS_H2_ROUTE,
    capture_cost_cny_per_t,
    h2_premium_cny_per_t,
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
    # National supply-weighted LCOH per planning year, derived from ammonia_supply_curve.csv.


def _national_h2_price(paths: ProjectPaths, usd_to_cny: float) -> dict[int, float]:
    """Supply-weighted mean LCOH per year, CNY/kg, from the repo's own hydrogen supply curve.

    Deliberately the SAME dataset that prices coal-side ammonia co-firing, so the two hydrogen
    uses in this model cannot be priced off different curves.

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


def prepare_industry(paths: ProjectPaths, assumptions) -> IndustryInputs:
    """Load and prepare the industrial hubs. Call only when industry is enabled.

    Args:
        paths: Project paths.
        assumptions: `OptimizationAssumptions`; `usd_to_cny` and `water_budget` are read.

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
    hubs = hubs[hubs["sector"].astype(str).isin(INDUSTRY_SECTORS)].reset_index(drop=True)
    if hubs.empty:
        raise ValueError(f"{path} has no rows in the in-scope sectors {sorted(INDUSTRY_SECTORS)}")
    unknown = sorted(set(hubs["sector"].astype(str)) - set(INDUSTRY_CAPTURE_COST_CNY_PER_T))
    if unknown:
        raise ValueError(f"no capture cost sourced for sector(s) {unknown}; refusing to guess")

    # Basin of the hub's own location, matching how `_prepare_plants` attributes coal hubs:
    # the withdrawal permit follows the site, not the intake. Only needed for the basin cap.
    if str(getattr(assumptions, "water_budget", "runoff")) == "official_quota":
        from ..builders.water import _assign_basin_codes

        # industry_hubs.csv already names the columns `latitude`/`longitude`, which is what
        # `_assign_basin_codes` expects; no rename needed (plants.csv needs one, this does not).
        hubs["basin_code"] = _assign_basin_codes(paths, hubs)

    prices = _national_h2_price(paths, float(assumptions.usd_to_cny))
    logger.info(
        "industry: %d hubs, %.0f Mt CO2/yr baseline, %.1f 亿 m3/yr water; H2 price %s CNY/kg",
        len(hubs),
        float(hubs["co2_mt_per_year"].sum()),
        float(hubs["water_m3_per_year"].sum()) / 1e8,
        {y: round(p, 1) for y, p in sorted(prices.items()) if y in (2030, 2060)},
    )
    return IndustryInputs(hubs=hubs, h2_price_cny_per_kg=prices)


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


def industry_year_data(
    industry: IndustryInputs, scenario, assumptions, year: int
) -> dict[str, object]:
    """Per-year cost, emission and water coefficients for every industrial hub and route.

    All arrays are shaped `(hub_count, len(INDUSTRY_ROUTES))` unless noted.

    Args:
        industry: Prepared industrial inputs.
        scenario: `OptimizationScenario`; capture rate, cost multipliers, water multiplier.
        assumptions: `OptimizationAssumptions`; the CCS learning curve is read from it.
        year: Planning year.

    Returns:
        Dict of coefficient arrays plus scalars the solver needs.
    """
    hubs = industry.hubs
    sectors = hubs["sector"].astype(str).to_numpy()
    n = len(hubs)
    co2_mt = hubs["co2_mt_per_year"].astype(float).to_numpy()
    production_t = hubs["production_kt_per_year"].astype(float).to_numpy() * 1_000.0
    base_water_m3 = hubs["water_m3_per_year"].astype(float).to_numpy()
    h2_intensity_t_per_t = np.divide(
        hubs["h2_demand_kt_per_year"].astype(float).to_numpy(),
        np.where(hubs["production_kt_per_year"].astype(float).to_numpy() > 0,
                 hubs["production_kt_per_year"].astype(float).to_numpy(), np.nan),
    )
    h2_intensity_t_per_t = np.nan_to_num(h2_intensity_t_per_t, nan=0.0)

    capture_rate = float(scenario.capture_rate)
    # Same exogenous learning curve the coal retrofits get, so the two sectors' capture costs
    # decline together. Using a different curve for industry would make the sectoral split a
    # function of an arbitrary modelling choice rather than of the technologies.
    learning = float(assumptions.ccs_learning_factor(year))
    cost_multiplier = float(getattr(scenario, "industry_cost_multiplier", 1.0))
    h2_multiplier = float(getattr(scenario, "industry_h2_cost_multiplier", 1.0))
    h2_price = _h2_price_for_year(industry.h2_price_cny_per_kg, int(year))

    route_available = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=bool)
    route_available[:, UNABATED] = True
    route_available[:, CCS] = True
    reduction_mt = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    captured_mt = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    annual_cost_cny = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    water_m3 = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)

    # unabated: the hub as it runs today. No abatement, no extra cost, its existing intake.
    water_m3[:, UNABATED] = base_water_m3

    # ccs: capture `capture_rate` of everything the hub emits, combustion and process alike —
    # which is the whole point for cement, where 63% of emissions are calcination and no fuel
    # switch can touch them.
    captured_mt[:, CCS] = co2_mt * capture_rate
    reduction_mt[:, CCS] = captured_mt[:, CCS]
    capture_unit_cost = np.array([capture_cost_cny_per_t(s) for s in sectors], dtype=np.float64)
    annual_cost_cny[:, CCS] = (
        captured_mt[:, CCS] * 1e6 * capture_unit_cost * learning * cost_multiplier
    )
    water_m3[:, CCS] = base_water_m3 + captured_mt[:, CCS] * 1e6 * INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2

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
        premium = h2_premium_cny_per_t(sector, h2_price, float(h2_intensity_t_per_t[hub_idx]))
        annual_cost_cny[hub_idx, H2] = production_t[hub_idx] * premium * h2_multiplier
        ratio = quota_ratio[sector] if INDUSTRY_H2_USES_ADVANCED_QUOTA else 1.0
        water_m3[hub_idx, H2] = base_water_m3[hub_idx] * ratio

    # NB: `scenario.water_multiplier` scales AVAILABILITY, not demand, so it is deliberately
    # not applied here — it is applied once, to the basin residual, in `_basin_cap_data`.
    return {
        "hub_ids": hubs["hub_id"].astype(str).tolist(),
        "sectors": sectors,
        "baseline_emissions_mt": co2_mt,
        "route_available": route_available,
        "reduction_mt": reduction_mt,
        "captured_mt": captured_mt,
        "annual_cost_cny": annual_cost_cny,
        "water_m3": water_m3,
        "h2_price_cny_per_kg": h2_price,
        "capture_learning_factor": learning,
    }


def add_industry_year(model, industry: IndustryInputs, ydata: dict, year_suffix: str) -> dict:
    """Add one planning year's industrial variables and within-year constraints.

    Args:
        model: The Gurobi model under construction.
        industry: Prepared industrial inputs.
        ydata: Output of `industry_year_data` for this year.
        year_suffix: Year string appended to every variable and constraint name.

    Returns:
        Payload with the route share variable and the derived expressions the solver wires
        into the CO2 network, the basin cap, the joint target and the objective.
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
    total_reduction = gp.quicksum(
        float(reduction[hub, r]) * share[hub, r] for hub in range(n) for r in range(n_routes)
    )
    water = ydata["water_m3"]
    # Scaled to the same Mm3 units the coal side's basin expression uses, so the two can be
    # added inside one constraint without a hidden unit change.
    withdrawal_by_hub = [
        gp.quicksum(float(water[hub, r]) / WATER_FLOW_SCALE * share[hub, r] for r in range(n_routes))
        for hub in range(n)
    ]
    cost = ydata["annual_cost_cny"]
    annual_cost = gp.quicksum(
        float(cost[hub, r]) * share[hub, r] for hub in range(n) for r in range(n_routes)
    )
    return {
        "share": share,
        "year_suffix": year_suffix,
        "captured_by_hub": captured_by_hub,
        "total_reduction_mt": total_reduction,
        "withdrawal_by_hub_scaled": withdrawal_by_hub,
        "annual_cost_cny": annual_cost,
        "year_data": ydata,
    }


def add_industry_monotonicity(model, payloads: list[dict], hub_count: int) -> None:
    """Abatement routes are irreversible: `ccs` and `h2` shares never fall between years.

    With the shares summing to one this also forbids swapping capture for hydrogen — you do
    not demolish a capture island to build a DRI shaft — and it is what makes charging a
    LEVELISED cost per operating year legitimate (see the module docstring).

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
