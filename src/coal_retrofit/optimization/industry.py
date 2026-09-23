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
4. **The emission targets.** One residual cap per sector group -- steel, cement, chemicals,
   and power for the coal fleet -- read from `inputs/sector_targets_<source>.csv`. The
   carbon price is normally zero under caps; when it is not, it is charged on industrial
   residuals exactly as on coal residuals.

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

文件分工（2026-09-23 从本文件拆出，本文件只留设计说明并转导出，调用方的 import 不用改）：
`industry_inputs.py` 输入准备（hub 表、氢链路）；`industry_matrices.py` 逐年系数
（`IndustryYearData`）；`model_industry.py` 变量、约束与一次性 capex 表达式（`IndustryPayload`）。
"""
from __future__ import annotations

from .industry_inputs import IndustryInputs, prepare_industry, prepare_industry_h2_links
from .industry_matrices import (
    CCS,
    H2,
    ROUTE_INDEX,
    UNABATED,
    IndustryYearData,
    basin_membership,
    industry_year_data,
)
from .model_industry import (
    IndustryPayload,
    add_industry_monotonicity,
    add_industry_year,
    industry_capex_expr,
)

__all__ = [
    "CCS",
    "H2",
    "ROUTE_INDEX",
    "UNABATED",
    "IndustryInputs",
    "IndustryPayload",
    "IndustryYearData",
    "add_industry_monotonicity",
    "add_industry_year",
    "basin_membership",
    "industry_capex_expr",
    "industry_year_data",
    "prepare_industry",
    "prepare_industry_h2_links",
]
