"""End-of-horizon salvage credit on one-time capex (author's decision 2026-09-22).

Every capex the objective books once -- coal capture islands, blend upgrades, air-cooling
retrofits, pipelines, site rebuilds, industrial capture and H2 routes -- buys an asset with an
economic life. A retrofit built in 2060 on a horizon that ends in 2070 has served a third of
its 30-year life when the model stops looking; charging its whole capex makes the last period
under-invest, and levelising instead would bring back the capital-recovery assumption the
author rejected. The fix is the standard one: straight-line depreciation over the economic
life, and the undepreciated remainder credited back at the horizon end, discounted from there.

    credit = df(T_end) * sum_t sum_items  max(0, 1 - (T_end - t) / L_item) * capex_item(t)

with `T_end = last planning year + its interval` (2070 on the 2030/40/50/60 grid). The credit
can never exceed the capex it refers to and is discounted further than the charge, so it
cannot make building profitable on its own; it only stops the horizon from punishing late
investment. It is booked as one negative `salvage_credit` entry on the last payload (zero on
the others, so the cost-breakdown schema is identical across years). With
`end_of_horizon_salvage=False` the key is NOT written at all, so a run reproducing the
pre-2026-09-22 objective also reproduces its `cost_breakdown.csv` schema.

KNOWN SIMPLIFICATION: the economic life is the asset's, not the host unit's. A 2060
air-cooling retrofit on a unit that retires exogenously in 2065 still earns half its capex
back at 2070, and a 2030 capture island (20 a) keeps operating to 2070 with no replacement
capex (pipelines, by contrast, do expire at `pipeline_lifetime_years` in the solver). Both
effects are second order at the fleet level and symmetric between coal and industry.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover
    gp = None

from ._shared import _discount_factor

if TYPE_CHECKING:  # pragma: no cover
    from .scenario import OptimizationAssumptions, OptimizationScenario

logger = logging.getLogger(__name__)


def horizon_end_year(year_payloads: list[dict]) -> int:
    """Last planning year plus the interval it stands for."""
    last = year_payloads[-1]
    return int(last["year"]) + int(last["interval_years"])


def remaining_fraction(build_year: int, life_years: int, end_year: int) -> float:
    """Undepreciated share of an asset built in `build_year` when the horizon ends."""
    life = max(1, int(life_years))
    served = max(0, int(end_year) - int(build_year))
    return max(0.0, 1.0 - served / life)


def _add_salvage_credit(
    year_payloads: list[dict],
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    cost_scale: float,
) -> None:
    """Append `salvage_credit` to every payload's `cost_exprs` and refresh the objective.

    Args:
        year_payloads: Solver payloads in planning-year order; each carries `salvage_ledger`
            as a list of `(name, undiscounted capex expr, life_years)`.
        scenario: `OptimizationScenario`; discount rate and base year.
        assumptions: `OptimizationAssumptions`; `end_of_horizon_salvage` switch.
        cost_scale: The solver's objective scale (`_COST_SCALE`), applied to the credit as
            to every other cost expression.
    """
    if not bool(assumptions.end_of_horizon_salvage):
        return  # objective_expr already assembled without the key
    if gp is None:  # pragma: no cover
        raise ImportError("gurobipy is required to build the salvage credit")
    for payload in year_payloads:
        payload["cost_exprs"]["salvage_credit"] = 0.0
    end_year = horizon_end_year(year_payloads)
    df_end = _discount_factor(end_year, scenario.discount_base_year, scenario.discount_rate)
    terms = []
    for payload in year_payloads:
        build_year = int(payload["year"])
        for name, expr, life in payload.get("salvage_ledger", []):
            frac = remaining_fraction(build_year, life, end_year)
            if frac <= 0.0:
                continue
            if isinstance(expr, (int, float)):
                if float(expr) != 0.0:
                    raise ValueError(f"salvage ledger item {name!r} in {build_year} is a constant {expr}")
                continue  # builder returned 0.0: nothing of this kind can be built this year
            terms.append(frac * expr)
    if terms:
        year_payloads[-1]["cost_exprs"]["salvage_credit"] = -df_end * gp.quicksum(terms) / float(cost_scale)
    for payload in year_payloads:
        payload["objective_expr"] = gp.quicksum(list(payload["cost_exprs"].values()))
    logger.info("salvage credit at horizon end %d (df=%.4f) on %d capex items", end_year, df_end, len(terms))
