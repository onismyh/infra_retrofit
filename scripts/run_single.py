"""Run a single experiment scenario.

Usage:
    python scripts/run_single.py <name> [--json]

Example:
    python scripts/run_single.py BASE
    python scripts/run_single.py RQ2_no_learning
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace

from _bootstrap import ROOT

sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.optimization._shared import SolveState, PATHWAY_INDEX
from coal_retrofit.optimization.results import (
    _build_cost_breakdown,
    _build_pathway_table,
    _build_province_table,
    _build_edge_table,
    _build_storage_table,
    _build_supply_table,
    _build_sanity_checks,
    _build_plant_detail_table,
    _build_biomass_flow_table,
    _build_ammonia_flow_table,
    _build_water_flow_table,
    _build_slack_detail_table,
    _build_co2_flow_direction_table,
    _build_plant_cost_table,
)
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.constants import PLANNING_YEARS


EXPERIMENTS: dict[str, tuple[dict, dict]] = {
    # name: (scenario_overrides, assumption_overrides)
    # === Core scenarios (MIPGap=1%, 2060-only emission target) ===
    # mip_gap 0.01, not 0.02. BASE was the only run in the study solved at twice everyone
    # else's tolerance, which makes its certified interval twice as wide as its partner's in
    # the one contrast it appears in (Fig 3, "is basin water accounted at all"). That
    # contrast is currently unresolved -- [-0.73, +1.25]% -- and roughly half that width is
    # BASE's own tolerance rather than anything physical. Re-solved at 0.01 with the rest.
    "BASE": ({"mip_gap": 0.01}, {}),
    # The fourth cell of the 2x2 (water constraint on/off) x (cooling adaptation on/off).
    # Without it the headline comparison is contaminated: BASE already converts ~90 GW to dry
    # cooling on water PRICE alone, so `WA_*_noair - BASE` mixes the cost of the water
    # constraint with the cost of losing that price-driven conversion. Only
    # `WA_*_noair - BASE_noair` measures the constraint inside a frozen-cooling model, which
    # is what the published literature actually does.
    "BASE_noair": ({"mip_gap": 0.02}, {"allow_air_cooling_retrofit": False}),
    "BASE_zero": ({"emission_target_fraction": (0.0, 0.0, 0.0, 1.0), "mip_gap": 0.01}, {}),
    "BASE_neg": ({"emission_target_fraction": (0.0, 0.0, 0.0, 1.05), "mip_gap": 0.01}, {}),
    # === RQ3: Pathway restriction scenarios ===
    "RQ3_no_ammonia": ({"pathway_disable": ("ammonia",)}, {}),
    "RQ3_no_biomass": ({"pathway_disable": ("biomass", "beccs")}, {}),
    "RQ3_no_ccs": ({"pathway_disable": ("ccs", "beccs")}, {}),
    "RQ3_retire_only": ({"pathway_disable": ("unabated", "ccs", "biomass", "beccs", "ammonia")}, {}),
    "RQ3_ccs_only": ({"pathway_disable": ("unabated", "biomass", "ammonia")}, {}),
    # === Water constraint (grid-based, 200km, SSP1-2.6, x0.20) ===
    "WA_grid_200km": ({"water_mode": "grid_supply"}, {}),
    # === Climate-driven water availability (qtot local runoff, province-budgeted) ===
    # 2 hydrology models x gfdl-esm4 x {SSP1-2.6, SSP3-7.0} x {annual mean, dry season}.
    # The dry season is the lowest three consecutive months, which is when thermal plants
    # are actually curtailed; on annual means the constraint barely binds.
    "WA_cwatm_126_annual": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "annual"}, {}),
    "WA_cwatm_370_annual": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp370", "water_season": "annual"}, {}),
    "WA_cwatm_126_dry":    ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {}),
    "WA_cwatm_370_dry":    ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp370", "water_season": "dry"}, {}),
    "WA_wgap_126_annual":  ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp126", "water_season": "annual"}, {}),
    "WA_wgap_370_annual":  ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp370", "water_season": "annual"}, {}),
    "WA_wgap_126_dry":     ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp126", "water_season": "dry"}, {}),
    "WA_wgap_370_dry":     ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp370", "water_season": "dry"}, {}),
    # === T1: reserve the non-power share of the abstraction allowance ===
    # `existing_withdrawal_share` has always defaulted to 0.0, i.e. power was offered the
    # basin's entire 20% extractable allowance as though agriculture, households and other
    # industry did not exist. NOTE that "at 0.0 the constraint never binds" -- what this
    # comment used to say -- is FALSE, and was read off a basin aggregate. At s = 0 the
    # 2030 solve already puts 33 of 373 water nodes at >=99.9% of budget, carrying 20.1%
    # of fleet water demand (2040: 54 and 27.9%), and converts 76.7 GW more cooling than
    # BASE. The reservation is a step up in binding intensity, not a switch. At 0.0 even
    # the Hai basin -- 177 GW on 396x10^8 m3/yr of renewable runoff, the tightest in the fleet
    # -- uses only 22% of its budget, so `WA_*_dry` lands within the MIP gap of BASE.
    #
    # 0.85 reserves the non-power share of the allowance, power keeping the ~15% it holds in
    # national abstraction statistics (China Environment Statistical Yearbook basin supply).
    # Basis note for Methods: the allowance is an ABSTRACTION quantity while the constraint
    # acts on CONSUMPTION, so this is deliberately conservative -- it grants power ~7x more
    # consumptive headroom than the ~2% of national depletion it actually holds. Deducting
    # non-power CONSUMPTION from a consumption allowance instead is basis-consistent but
    # degenerate: the share exceeds 1.0 in Hai/Yellow/Huai/Northwest and zeroes 765 GW.
    #
    # CORRECTED 2026-08-18. The earlier note here ("at 0.85 the unabated fleet still fits
    # every basin, Hai peaks at 0.86 of budget") was wrong by ~4x: it was computed against a
    # double-blended intensity (see plot_fig2_constraint_response.py:227-235). Recomputed on
    # the ensemble median, 2030, the UNABATED fleet already exceeds this budget in four
    # basins -- Hai 3.80, Yellow 1.80, NW Interior 1.25, Huai 1.10 -- so 0.85 does not leave
    # today's fleet feasible; the shortfall is absorbed by the big-M water slack. Attaching
    # capture takes Hai to 8.05. 0.85 is therefore NOT presented as a calibrated central
    # case: it is one point on the reservation-share axis, and the paper reports the whole
    # axis (WA_cwatm_126_dry_wd056 / wd070 / wd085 / wd090). Two facts on that axis do not
    # depend on where the dial sits: Hai cannot host full capture at ANY reservation share,
    # including zero (s* < 0 on the median and in 75% of the 20 members), and the four-basin
    # / 764.7 GW verdict is flat over 0.65 <= share <= 0.89, a 24-point plateau.
    "WA_cwatm_126_dry_wd085": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    "WA_cwatm_370_dry_wd085": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp370", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    # === Reservation-share sweep: the parameter that CREATES the binding constraint ========
    # `existing_withdrawal_share` is the single number that decides whether water binds at all
    # (at 0.0 nothing binds anywhere; at 0.85 four basins cross), yet until now exactly one
    # value had ever been solved. Every headline in the paper -- the four-basin verdict, the
    # +2.06%, the 411 GW of conversion -- inherits it, and a referee will ask for the response
    # curve rather than the point. The value is also known to be mis-anchored: the measured
    # 0.830 non-power share is relative to China's own 370e8 m3 consumptive quota, but it is
    # applied on top of Richter's 20% extractable fraction, so the basis-consistent number
    # would exceed 1.0. Bracketing it is therefore the honest presentation: report the response
    # over 0.56-0.90 and let the reader see where each basin crosses.
    #   0.559  the national residual-allocation value (3 basins cross, ~449 GW)
    #   0.70   between the residual value and the used one
    #   0.90   tighter than used, to show the curve does not hinge on 0.85
    "WA_cwatm_126_dry_wd056": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.559}),
    "WA_cwatm_126_dry_wd070": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.70}),
    "WA_cwatm_126_dry_wd090": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.90}),
    # The same binding treatment on the OTHER hydrology model. Without these, EVERY run in
    # which water actually binds comes from CWatM, so "water binds" is confounded with "CWatM":
    # the treatment pairs and the degeneracy floor that judges them would share one hydrology,
    # and the Fig 5 dissociation would rest on a single model's runoff. WaterGAP2 is the
    # independent replicate that makes the contrast a treatment rather than a model artefact.
    "WA_wgap_126_dry_wd085": ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    "WA_wgap_370_dry_wd085": ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp370", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    # GCM DIVERSITY, AND A BRACKET AROUND THE HAI SIGN CHANGE.
    # The four members solved above are all gfdl-esm4, so the ensemble spread the paper
    # reports is a hydrology-model and SSP spread with a single GCM inside it. These two add
    # a second and third GCM, and they are chosen by where they sit relative to the result
    # the paper turns on: the critical reservation share s* for full capture in the Hai at
    # 2030, which is negative in 13 of 20 members. Ranked by that statistic the solved set
    # currently spans -0.627 (wgap|gfdl|ssp370) to +0.050 (cwatm|gfdl|ssp126); ipsl-cm6a-lr
    # under ssp126 sits at -0.029, immediately below the sign change, and mpi-esm1-2-hr under
    # ssp370 at +0.293 is the wettest Hai in the ensemble. Together they bracket s* = 0 from
    # both sides with GCMs the solved set does not contain.
    "WA_cwatm_ipsl126_dry":       ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|ipsl-cm6a-lr|ssp126", "water_season": "dry"}, {}),
    "WA_cwatm_ipsl126_dry_wd085": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|ipsl-cm6a-lr|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    "WA_cwatm_mpi370_dry":        ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|mpi-esm1-2-hr|ssp370", "water_season": "dry"}, {}),
    "WA_cwatm_mpi370_dry_wd085":  ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|mpi-esm1-2-hr|ssp370", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    # Cooling frozen under a binding constraint: with adaptation unavailable, every response
    # has to run through early retirement. Completes the (constraint on/off) x (cooling on/off)
    # 2x2 in a world where the constraint actually bites.
    "WA_cwatm_126_dry_wd085_noair": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85, "allow_air_cooling_retrofit": False}),
    # DIAGNOSTIC, not a policy scenario. The frozen-cooling run above sits EXACTLY on the
    # early-retirement cap in the two periods the figures quote -- 2030 share_retire = 0.150000
    # and 2040 = 0.300000, i.e. 1x and 2x `max_new_retirement_share_per_period` (0.15) to six
    # decimals -- and then buys the rest of its shortfall at the big-M water penalty (18.5% of
    # its objective). So "forbidding the retrofit costs 150 Mt of capture" may be measuring
    # where two exogenous constants happen to land rather than the value of the adaptation
    # channel. Raising the cap to 0.50 unbinds it; if the capture drop survives, the finding is
    # real and the cap was incidental. If it collapses, the magnitude was manufactured and only
    # the DIRECTION may be reported. Either way the answer must reach the caption.
    "WA_cwatm_126_dry_wd085_noair_capfree": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "max_new_retirement_share_per_period": 0.50}, {"existing_withdrawal_share": 0.85, "allow_air_cooling_retrofit": False}),

    # === Retirement-cap-free replicates of the HEADLINE pair =============================
    # Audit finding (verified from plant_detail.csv): every published run sits at EXACTLY
    # max_new_retirement_share_per_period = 0.15 in 2050 -- BASE 0.150000, WA_cwatm_126_dry
    # 0.150000, ..._wd085 0.150000, ..._370_dry_wd085 0.150000, ..._wgap_126_dry_wd085
    # 0.150000. The 2050 retirement DIFFERENCE between any two of them is therefore
    # identically zero by construction, and the 2040 difference (0.0017 -> 0.0466) is the
    # displacement of a trajectory that is truncated one period later. No figure may report
    # a 2050 retirement effect off the capped runs. These two lift the cap to 0.50 -- shown
    # non-binding by _noair_capfree, which peaks at 0.2467 -- on BOTH sides of the headline
    # contrast, so the retirement response can be measured where it is free.
    "WA_cwatm_126_dry_capfree":       ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "max_new_retirement_share_per_period": 0.50}, {}),
    "WA_cwatm_126_dry_wd085_capfree": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "max_new_retirement_share_per_period": 0.50}, {"existing_withdrawal_share": 0.85}),

    # === Biomass resource realism: the largest single exposure in the scenario set =========
    # inputs/biomass_supply_curve.csv totals 30.05 EJ/yr, which is 2 003 Mt/yr of dry matter
    # at the file's own 15 GJ/t conversion. China's ENTIRE annual crop-straw production is
    # ~900 Mt and forestry residue adds ~200 Mt, so the supply curve offers roughly twice all
    # the agricultural and forestry biomass the country grows -- before any of it is eaten,
    # fed, ploughed back for soil carbon, or made into paper. It is a gross potential surface
    # (data/中国生物质潜力/bioenergy_s3_AFE_abandon.tif, ~4.4e6 km2 of positive pixels =
    # 46% of China's land area), not an available-for-energy resource, and no availability
    # mask was ever applied. The headline run draws 25.5 EJ = 85% of it.
    #
    # DIRECTION OF THE ERROR: RETRACTED, IT IS NOT DETERMINED.
    # This block previously asserted that an inflated biomass supply runs AGAINST the water
    # finding -- biomass displaces capture, capture carries the 1.82x water multiplier, so
    # correcting the supply could only strengthen the water claim. That reasoning is wrong,
    # and the refutation is one line of scenario.py: `beccs_water_multiplier = 1.82`, the
    # SAME multiplier as CCS. BECCS is 13.5% of 2060 generation and biomass co-firing 28.5%,
    # so cutting the supply removes a water-free wedge (co-firing at 1.00x) and a
    # water-hungry one (BECCS at 1.82x) together, and the replacement splits between CCS
    # retrofit (1.82x, raises water) and early retirement (zero water, lowers it) under a
    # hard emission target. The net sign is genuinely indeterminate a priori.
    # No direction is claimed until bio015 and bio042 are solved. If it turns out the other
    # way it is the paper's conclusion that moves, not a caveat.
    #
    # Two multipliers, both anchored to the residue literature rather than tuned:
    #   0.15 -> 4.5 EJ: ~230 Mt straw actually available after feed/fertiliser/fibre uses,
    #                   plus ~1 EJ forestry residue. The realistic case.
    #   0.42 -> 12.6 EJ: ALL collectable straw (~765 Mt) diverted to energy plus forestry
    #                   residue. A generous upper bound, not a central estimate.
    # Both are solved on the binding side AND the control so the water contrast survives.
    #
    # mip_gap 0.02, NOT the 0.01 the rest of the study uses. Cutting biomass supply removes the
    # water-free wedge (co-firing) and the water-hungry one (BECCS) together, so the replacement
    # has to split between CCS retrofit and retirement and the integer part gets hard: the
    # bio015 CONTROL took 10.03 h to reach 1.009%. At 2% the certified interval on this pair is
    # roughly twice as wide, which these runs can afford -- they were registered to test the
    # DIRECTION of the biomass mechanism, and after the source-sink rebuild that mechanism is
    # withdrawn anyway (round 1 and round 2 disagree in sign on the biomass response while
    # agreeing on the objective to 0.002 pp; see 源汇聚类与划分_审查.md §9.4). The bio015 control
    # is already solved at 1.009% and is NOT re-run, so that pair is asymmetric by construction:
    # the interval formula uses each run's own gap, so it stays valid, just wider on one side.
    "WA_cwatm_126_dry_wd085_bio015": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "biomass_supply_multiplier": 0.15, "mip_gap": 0.02}, {"existing_withdrawal_share": 0.85}),
    "WA_cwatm_126_dry_bio015":       ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "biomass_supply_multiplier": 0.15}, {}),
    "WA_cwatm_126_dry_wd085_bio042": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "biomass_supply_multiplier": 0.42, "mip_gap": 0.02}, {"existing_withdrawal_share": 0.85}),
    # The bio042 CONTROL was missing, so that arm had a treatment with nothing to difference
    # against. Without it the 0.42 case can only be read against the full-supply control,
    # which moves the biomass ceiling and the water reservation at the same time.
    "WA_cwatm_126_dry_bio042":       ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "biomass_supply_multiplier": 0.42, "mip_gap": 0.02}, {}),
    # SEED REPLICATES. Identical scenario, identical assumptions, identical inputs -- only the
    # Gurobi seed differs, supplied via COAL_RETROFIT_GUROBI_SEED. This is the only construction
    # in the study that measures DEGENERACY: the model, its parameters and its feasible set are
    # bit-identical, so any disagreement between these runs is the solver settling on a
    # different point of a flat optimum. Everything previously used as a "degeneracy floor"
    # (the `*_nobias` pairs) perturbs the physics instead -- Hai's bias factor is 0.412 under
    # watergap2-2e|gfdl-esm4 and 0.791 under cwatm|gfdl-esm4, so switching the correction off
    # multiplies its available water by 2.43x and 1.26x respectively. Those are perturbations
    # of different size, which is a second reason the nobias pairs cannot serve as a floor.
    "WA_cwatm_126_dry_wd085_seed2": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    "WA_cwatm_126_dry_wd085_seed3": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    "WA_cwatm_126_dry_wd085_seed4": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    "WA_cwatm_126_dry_wd085_seed5": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    "WA_cwatm_126_dry_wd085_seed6": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85}),
    # Replicates of the CONTROL side. A floor measured on one run and applied to a difference
    # of two understates by about sqrt(2); with replicates on both sides the sampling
    # distribution of the DIFFERENCE is estimated directly (5 x 4 = 20 pairs) instead of
    # dividing an effect by a 2-sample range, whose 5-95% span is a factor of 31.
    "WA_cwatm_126_dry_seed2": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {}),
    "WA_cwatm_126_dry_seed3": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {}),
    "WA_cwatm_126_dry_seed4": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {}),

    # === Air-retrofit capex sensitivity: the parameter the headline actually rests on ======
    # `air_retrofit_capex_cny_per_kw=300` is a placeholder. Its docstring cites "roughly
    # 200-400 CNY/kW in the Chinese literature", but plan/*.md still lists the engineering cost
    # as 需检索 -- no source was ever attached, and searching found none: wet-to-air conversion
    # is rare enough in China (air cooling is normally chosen at the new-build stage) that no
    # public CNY/kW figure exists.
    #
    # The one quantitative retrofit study in hand is Zhai et al. 2022 (DOE/OSTI 1882420,
    # reference/water/_drycool_retrofit_text.txt): dry-cooling retrofit at 12 western US coal
    # EGUs cuts water consumption 93%, raises LCOE by 5.4 USD/MWh (12%), and costs 0.7% of net
    # capacity as a monthly average. Priced like-for-like the model charges only 2.54 USD/MWh
    # (0.80 capex at CRF(6%,20) + 1.73 fuel for the 1.5 pp penalty), i.e. Zhai is 2.13x dearer.
    # Holding the efficiency penalty fixed, that residual implies ~1370 CNY/kW over 20 years.
    #
    # 1370 is an UPPER anchor, not a correction: Zhai assumes new-build ACC cost with no
    # site-difficulty premium, on US steel and labour, converted at market FX rather than PPP,
    # and its 5.4 USD/MWh also absorbs O&M changes. China builds most of the world's air-cooled
    # coal capacity, so its true cost is genuinely below the US figure. The defensible claim is
    # therefore a RANGE, and the paper must show the water-carbon trade-off survives across it
    # rather than quoting the +0.15% point estimate.
    "WA_cwatm_126_dry_wd085_air1000": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85, "air_retrofit_capex_cny_per_kw": 1000.0}),
    "WA_cwatm_126_dry_wd085_air1370": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"existing_withdrawal_share": 0.85, "air_retrofit_capex_cny_per_kw": 1370.0}),
    # === Bias-correction on/off: how much of the constraint owes to the correction? ===
    # The basin factors (builders/water.basin_bias_factors) are baked into the availability
    # columns; setting apply_bias_correction=False serves raw modelled runoff instead (a
    # faithful unde-bias: the factor divides out of the availability exactly). These two are
    # the decisive members: SSP3-7.0 dry is the driest, and WaterGAP2 carries the largest bias
    # in the basins where the constraint binds. `_nobias` reproduces the uncorrected world;
    # the delta vs the corrected `..._dry` runs is the T4 robustness result.
    "WA_cwatm_370_dry_nobias": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp370", "water_season": "dry"}, {"apply_bias_correction": False}),
    "WA_wgap_126_dry_nobias":  ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp126", "water_season": "dry"}, {"apply_bias_correction": False}),
    # === Wet-to-dry cooling conversion: on/off and capex sensitivity ===
    # The counterfactual the paper needs. With cooling technology frozen, every response to
    # scarcity has to run through early retirement, which is exactly the criticism a reviewer
    # would make. `_noair` reproduces that frozen world so the difference can be quantified.
    "WA_cwatm_126_dry_noair": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"allow_air_cooling_retrofit": False}),
    "WA_cwatm_370_dry_noair": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp370", "water_season": "dry"}, {"allow_air_cooling_retrofit": False}),
    "SA_air_capex_low":  ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"air_retrofit_capex_cny_per_kw": 200.0}),
    "SA_air_capex_high": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"air_retrofit_capex_cny_per_kw": 400.0}),
    # Dry-cooling backpressure penalty, spanning the three sources cited in
    # OptimizationAssumptions.air_retrofit_efficiency_penalty_pp. LOW keeps the value the study
    # used before (1.5 pp) as an explicit lower bound rather than deleting it; HIGH is the upper
    # end of the Zhang et al. 2014 back-solve at 4500 operating hours, and doubles as the crude
    # stand-in for the temperature dependence Qin et al. 2023 report but this model does not
    # carry. The default sits at 2.0 pp, the point where the measured Chinese unit data
    # (1.93/2.11 pp) and the lower bound of the Zhang back-solve (1.91 pp) meet.
    "SA_air_penalty_low":  ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"air_retrofit_efficiency_penalty_pp": 0.015}),
    "SA_air_penalty_high": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"air_retrofit_efficiency_penalty_pp": 0.028}),
    # === Retirement-rate cap and retirement cost: the two unsourced parameters the water
    # story leans on. Both are run against the binding dry-season member, not BASE, because
    # that is where they actually bite.
    "SA_retire_cap_010": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "max_new_retirement_share_per_period": 0.10}, {}),
    "SA_retire_cap_025": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "max_new_retirement_share_per_period": 0.25}, {}),
    "SA_retire_cost_250": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"retire_cost_cny_per_mwh": 250.0}),
    "SA_retire_cost_650": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry"}, {"retire_cost_cny_per_mwh": 650.0}),
    # === No hard 2060 target: is "no water-carbon trade-off" physics or model structure? ===
    "DIAG_no_target": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "emission_target_fraction": (0.0, 0.0, 0.0, 0.0)}, {}),
    # === Water shadow-price sweep (Fig 3 frontier) ===
    # No availability constraint: the price alone drives the trade-off, so the locus of
    # (water use, abatement) points is a clean Pareto frontier and the adder at each point
    # is the shadow price of water there. Needed because a MIP has no usable constraint duals.
    "WP_0002": ({"water_price_adder_cny_per_m3": 2.0}, {}),
    "WP_0005": ({"water_price_adder_cny_per_m3": 5.0}, {}),
    "WP_0010": ({"water_price_adder_cny_per_m3": 10.0}, {}),
    "WP_0020": ({"water_price_adder_cny_per_m3": 20.0}, {}),
    "WP_0050": ({"water_price_adder_cny_per_m3": 50.0}, {}),
    "WP_0100": ({"water_price_adder_cny_per_m3": 100.0}, {}),
    "WP_0200": ({"water_price_adder_cny_per_m3": 200.0}, {}),
    "WP_0500": ({"water_price_adder_cny_per_m3": 500.0}, {}),
    "WP_1000": ({"water_price_adder_cny_per_m3": 1000.0}, {}),
    # === Sensitivity Analysis ===
    # Discount Rate
    "SA_discount_3pct": ({"discount_rate": 0.03}, {}),
    "SA_discount_8pct": ({"discount_rate": 0.08}, {}),
    # CCS CAPEX (base=4200; BECCS maintains 25% premium)
    "SA_ccs_capex_low": ({}, {"ccs_retrofit_capex_cny_per_kw": 3000.0, "beccs_retrofit_capex_cny_per_kw": 3750.0}),
    "SA_ccs_capex_high": ({}, {"ccs_retrofit_capex_cny_per_kw": 6000.0, "beccs_retrofit_capex_cny_per_kw": 7500.0}),
    # Biomass Cost
    "SA_biomass_cost_150": ({"biomass_cost_multiplier": 1.5}, {}),
    "SA_biomass_cost_200": ({"biomass_cost_multiplier": 2.0}, {}),
    # Ammonia Cost
    "SA_ammonia_cost_50": ({"ammonia_cost_multiplier": 0.5}, {}),
    "SA_ammonia_cost_70": ({"ammonia_cost_multiplier": 0.7}, {}),
    # Replacement Power Cost
    "SA_retire_cost_500": ({}, {"retire_cost_cny_per_mwh": 500.0}),
    "SA_retire_cost_1000": ({}, {"retire_cost_cny_per_mwh": 1000.0}),
    # Pipeline Corridor Reuse
    "SA_pipe_mid": ({}, {"corridor_capex_multiplier": 0.6}),
    "SA_pipe_full": ({}, {"corridor_capex_multiplier": 1.0}),
    # Storage Injectivity
    "SA_injectivity_half": ({"injectivity_multiplier": 0.5}, {}),
    # Carbon Price
    "SA_carbon_low": ({"carbon_price_cny_per_t_by_year": (60.0, 250.0, 440.0, 630.0)}, {}),
    "SA_carbon_high": ({"carbon_price_cny_per_t_by_year": (240.0, 1000.0, 1760.0, 2520.0)}, {}),

    # === v9.1: the official 用水总量控制指标 basis ("_oq") ================================
    # These REPLACE the `_wd085` family rather than extend it, and results from the two
    # families MUST NOT be differenced -- they are not the same model. Under `_wd085` the
    # whole water rule is one node constraint on CONSUMPTION whose right-hand side is
    # `qtot x 0.20 x 0.15`; under `_oq` it is two constraints on two bases, and the family is
    # a three-rung LADDER rather than an on/off pair:
    #
    #   BASE            no water rule at all
    #   *_oq_envonly    node  <= qtot x 0.20                      environmental flow, on
    #                                                             consumption (a depletion rule)
    #   *_oq            + basin <= 用水总量控制指标 - 非电既有取水  allocation, on withdrawal
    #                                                             (what the 公报 meters)
    #
    # That ladder is the point of the switch. Under `_wd085` the environmental-flow standard
    # and the allocation rule were ALIASED into one product and no experiment could say which
    # was binding; here BASE->envonly prices the first and envonly->oq prices the second,
    # separately. `existing_withdrawal_share` is absent throughout on purpose -- it was the
    # ASSUMED allocation rule, and the basin cap reads the real one off 国办发〔2013〕2号
    # instead. Setting both would count the allocation twice.
    #
    # `*_oq_envonly` is the CONTROL arm, replacing v9's `WA_*_dry` (which was s=0).
    #
    # Basin caps come from inputs/water_basin_caps.csv (scripts/build_water_basin_caps.py).
    # The apportionment-key sensitivity (demand- vs area-weighted province->basin split, which
    # flips the sign of the Northwest residual) is a REBUILD of that file, not a scenario knob:
    # rebuild with kind="area" and re-solve into a separate tree. See docs/官方指标口径水预算.md.
    #
    # mip_gap 0.01 throughout, seeds included (CLAUDE.md 二.2: a seed family measures solver
    # degeneracy, and a loosened gap there would be read as degeneracy).

    # --- the three hydrology arms, control and treatment ---------------------------------
    # Three forcings behind one treatment switch, so "the cap binds" can be told apart from
    # "CWatM says so". This matters LESS here than under `_wd085`: the binding side is now the
    # basin cap, which carries no hydrology at all. These runs are the test of that claim.
    "WA_cwatm_126_dry_oq_envonly": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "apply_basin_cap": False}),
    "WA_cwatm_126_dry_oq":         ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota"}),
    "WA_cwatm_370_dry_oq_envonly": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp370", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "apply_basin_cap": False}),
    "WA_cwatm_370_dry_oq":         ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp370", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota"}),
    "WA_wgap_126_dry_oq_envonly":  ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "apply_basin_cap": False}),
    "WA_wgap_126_dry_oq":          ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota"}),

    # --- seed replicates: the degeneracy floor, BOTH sides of the contrast ----------------
    # Identical model, identical parameters, only COAL_RETROFIT_GUROBI_SEED differs, so any
    # disagreement is the solver settling on a different point of a flat optimum. Both arms are
    # replicated because a floor measured on one run cannot judge a DIFFERENCE of two.
    # SEEDS = (2, 3, 4) in plot_ed_water_on_off, i.e. k = 4 with the base run, d2 = 2.059.
    "WA_cwatm_126_dry_oq_envonly_seed2": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "apply_basin_cap": False}),
    "WA_cwatm_126_dry_oq_envonly_seed3": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "apply_basin_cap": False}),
    "WA_cwatm_126_dry_oq_envonly_seed4": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "apply_basin_cap": False}),
    "WA_cwatm_126_dry_oq_seed2":         ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota"}),
    "WA_cwatm_126_dry_oq_seed3":         ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota"}),
    "WA_cwatm_126_dry_oq_seed4":         ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota"}),

    # --- cooling frozen, and the retirement cap lifted ------------------------------------
    # Dry conversion is a far bigger lever here than under `_wd085`: on the withdrawal basis
    # converting a once-through condenser removes ~90 m3/MWh instead of ~1, and the pipeline
    # check confirmed it is HOW the model complies (Northwest 16.9% -> 80.8% dry, utilisation
    # 1.000 with zero slack). Forbidding it is what prices that channel.
    "WA_cwatm_126_dry_oq_noair": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "allow_air_cooling_retrofit": False}),
    # Every capped run in the study sits at EXACTLY max_new_retirement_share_per_period, so a
    # retirement difference read off the capped pair measures the cap. Lifted to 0.50 here.
    # BOTH sides of the frozen-cooling contrast need the cap in the same place, or the single
    # reported effect moves TWO factors at once -- whether the dry-cooling retrofit is allowed
    # AND where the retirement cap sits. plot_fig5_pathway_succession pairs
    # `<BINDS>_capfree` against `<BINDS>_noair_capfree` for exactly that reason and falls back
    # to a confounded pair with a printed warning when the former is missing.
    "WA_cwatm_126_dry_oq_capfree": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "max_new_retirement_share_per_period": 0.50, "mip_gap": 0.01}, {"water_budget": "official_quota"}),
    "WA_cwatm_126_dry_oq_noair_capfree": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "max_new_retirement_share_per_period": 0.50, "mip_gap": 0.01}, {"water_budget": "official_quota", "allow_air_cooling_retrofit": False}),

    # --- air-cooling capex sweep ----------------------------------------------------------
    # The compliance lever's price. Two points bracketing the default so the response reads as
    # an elasticity rather than a single point.
    "WA_cwatm_126_dry_oq_air1000": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "air_retrofit_capex_cny_per_kw": 1000.0}),
    "WA_cwatm_126_dry_oq_air1370": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "air_retrofit_capex_cny_per_kw": 1370.0}),

    # --- bias-correction off, on the control arm -----------------------------------------
    # Serves the environmental-flow half only: the basin cap carries no hydrology, so turning
    # the runoff correction off cannot touch it. Attached to `_envonly` for that reason.
    # NOT a degeneracy floor -- the nobias pairs perturb the physics by different amounts per
    # basin (Hai 0.412 under watergap2, 0.791 under cwatm), which is why the seed family exists.
    "WA_cwatm_370_dry_oq_envonly_nobias": ({"water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp370", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "apply_basin_cap": False, "apply_bias_correction": False}),
    "WA_wgap_126_dry_oq_envonly_nobias":  ({"water_mode": "grid_supply", "water_scenario_id": "watergap2-2e|gfdl-esm4|ssp126", "water_season": "dry", "mip_gap": 0.01}, {"water_budget": "official_quota", "apply_basin_cap": False, "apply_bias_correction": False}),
}


def run(name: str, threads: int = 0, time_limit: int = 36000,
        mip_gap: float | None = None) -> dict:
    """Solve one registered scenario.

    *mip_gap* overrides the registry entry for this invocation only. It exists so a campaign can
    trade tolerance for wall-clock without editing -- and thereby permanently changing -- the
    scenario definitions. WHAT IT COSTS: the certified interval on any contrast involving the run
    widens roughly in proportion, and for the `*_seed*` replicates it destroys the measurement
    outright, because a seed family stops being a probe of solver DEGENERACY once the runs are
    allowed to stop that far apart on tolerance alone. Keep seeds at the tight gap.
    """
    scenario_kw, assumption_kw = EXPERIMENTS[name]
    if mip_gap is not None:
        scenario_kw = {**scenario_kw, "mip_gap": float(mip_gap)}
    paths = ProjectPaths(ROOT)
    assumptions = replace(OptimizationAssumptions(), **assumption_kw) if assumption_kw else OptimizationAssumptions()
    base_kw = {
        "experiment_id": name,
        "description": name,
        "planning_years": tuple(PLANNING_YEARS),
    }
    base_kw.update(scenario_kw)
    if threads > 0:
        base_kw["solver_threads"] = threads
    if time_limit != 36000:
        base_kw["solver_time_limit"] = time_limit
    scenario = OptimizationScenario(**base_kw)

    t0 = time.time()
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.effective_years(list(prepared.available_ammonia_years) or list(PLANNING_YEARS))
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, state)
    elapsed = time.time() - t0
    solver_quality = solution.get("solver_quality", {})

    gen = prepared.plants["annual_generation_mwh"].astype(float).to_numpy()
    total_gen = gen.sum()

    # Build high-resolution tables
    pathway_tables, province_tables, edge_tables = [], [], []
    storage_tables, supply_tables, cost_tables = [], [], []
    plant_detail_tables, biomass_flow_tables, ammonia_flow_tables = [], [], []
    water_flow_tables, slack_detail_tables, co2_direction_tables = [], [], []
    plant_cost_tables = []
    sanity_tables = []
    prev_share_values = None
    prev_retrofit_installed = None

    state_track = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )

    year_summaries = {}
    for year_index, year in enumerate(years):
        ys = solution["year_solutions"][year]
        share = ys["share"]
        year_data = ys["year_data"]
        interval_years = scenario.interval_years(years, year_index, assumptions)
        state_before = state_track.clone()

        # Aggregate summary
        pathway_shares = {}
        for pw, idx in PATHWAY_INDEX.items():
            pathway_shares[pw] = float((gen * share[:, idx]).sum() / total_gen) if total_gen > 0 else 0.0
        year_summaries[int(year)] = {
            "status": ys["status"],
            "objective_cny": float(ys["objective_cny"]),
            "pathway_shares": pathway_shares,
            "cost_breakdown": {k: float(v) for k, v in ys["cost_breakdown_cny"].items()},
            "target_shortfall_mt": float(ys["slacks"]["target_shortfall_mt"]),
            "solver_quality": solver_quality,
        }

        # High-resolution tables
        pw_table = _build_pathway_table(
            prepared, scenario, year, share,
            ys["captured_mt_by_plant"], ys["blend_level_b"], ys["blend_level_a"],
        )
        prov_table = _build_province_table(pw_table)
        pathway_tables.append(pw_table)
        province_tables.append(prov_table)
        edge_tables.append(_build_edge_table(prepared, year, ys["edge_flow_mtpa"], ys["build_edge"], ys["new_cap_mtpa"], state_before, assumptions))
        storage_tables.append(_build_storage_table(prepared, year, ys["storage_use_mtpa"], state_before, interval_years))
        supply_tables.append(_build_supply_table(prepared, year, year_data, ys["biomass_flow_gj"], ys["ammonia_flow_kg"], ys["water_flow_m3"], ys["slacks"].get("water_basin_use_m3")))
        cost_tables.append(_build_cost_breakdown(year, ys["cost_breakdown_cny"]))
        sanity_tables.append(_build_sanity_checks(year, ys["slacks"], pw_table, prov_table))
        plant_detail_tables.append(_build_plant_detail_table(
            prepared, scenario, year, share,
            ys["captured_mt_by_plant"], ys["biomass_use_gj"], ys["ammonia_use_kg"],
            ys["water_use_m3"], ys["blend_level_b"], ys["blend_level_a"],
            ys.get("air_share"),
        ))
        biomass_flow_tables.append(_build_biomass_flow_table(prepared, year, ys["biomass_flow_gj"]))
        ammonia_flow_tables.append(_build_ammonia_flow_table(year_data, year, ys["ammonia_flow_kg"], prepared.plants))
        water_flow_tables.append(_build_water_flow_table(year_data, year, ys["water_flow_m3"], prepared.plants))
        slack_detail_tables.append(_build_slack_detail_table(prepared, year, year_data, ys["slacks"]))
        co2_direction_tables.append(_build_co2_flow_direction_table(prepared, year, ys["co2_flow_fwd"], ys["co2_flow_bwd"]))
        plant_cost_tables.append(_build_plant_cost_table(
            prepared, scenario, assumptions, year, year_data, share,
            ys["captured_mt_by_plant"], ys["biomass_use_gj"], ys["water_use_m3"],
            ys["blend_level_b"], ys["blend_level_a"],
            prev_share_values=prev_share_values,
            retrofit_installed=ys["retrofit_installed"],
            prev_retrofit_installed=prev_retrofit_installed,
            capex_pathway_indices=solution.get("capex_pathway_indices", ()),
        ))

        if scenario.carry_state_between_years:
            state_track.edge_added_stock_mtpa = state_track.edge_added_stock_mtpa + ys["new_cap_mtpa"]
            state_track.remaining_storage_mt = np.maximum(0.0, state_track.remaining_storage_mt - ys["storage_use_mtpa"] * interval_years)
        prev_share_values = share
        prev_retrofit_installed = ys["retrofit_installed"]

    # Save CSV artifacts
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_tables = {
        "pathway_shares.csv": pd.concat(pathway_tables, ignore_index=True, sort=False),
        "province_pathways.csv": pd.concat(province_tables, ignore_index=True, sort=False),
        "plant_detail.csv": pd.concat(plant_detail_tables, ignore_index=True, sort=False),
        "network_edges.csv": pd.concat(edge_tables, ignore_index=True, sort=False),
        "storage_utilization.csv": pd.concat(storage_tables, ignore_index=True, sort=False),
        "resource_use.csv": pd.concat(supply_tables, ignore_index=True, sort=False),
        "biomass_flows.csv": pd.concat(biomass_flow_tables, ignore_index=True, sort=False),
        "ammonia_flows.csv": pd.concat(ammonia_flow_tables, ignore_index=True, sort=False),
        "water_flows.csv": pd.concat(water_flow_tables, ignore_index=True, sort=False),
        "co2_flow_direction.csv": pd.concat(co2_direction_tables, ignore_index=True, sort=False),
        "plant_cost.csv": pd.concat(plant_cost_tables, ignore_index=True, sort=False),
        "slack_detail.csv": pd.concat(slack_detail_tables, ignore_index=True, sort=False),
        "cost_breakdown.csv": pd.concat(cost_tables, ignore_index=True, sort=False),
        "sanity_checks.csv": pd.concat(sanity_tables, ignore_index=True, sort=False),
    }
    for fname, df in csv_tables.items():
        df.to_csv(out_dir / fname, index=False)

    result = {
        "name": name,
        "global_objective_cny": float(solution["objective_cny"]),
        "solver_quality": solver_quality,
        "years": year_summaries,
        "planning_years": [int(y) for y in years],
        "elapsed_seconds": round(elapsed, 1),
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run a single optimization scenario")
    parser.add_argument("name", nargs="?", default="BASE", help="Scenario name (or --list)")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument("--threads", type=int, default=0, help="Gurobi thread limit (0=auto)")
    parser.add_argument("--time-limit", type=int, default=36000, help="Solver time limit in seconds")
    parser.add_argument("--mip-gap", type=float, default=None,
                        help="Override the scenario's MIPGap for this run only "
                             "(do NOT use on *_seed* replicates; see run())")
    args = parser.parse_args()

    name = args.name
    if args.list:
        for k in EXPERIMENTS:
            print(k)
        return
    if name not in EXPERIMENTS:
        print(f"Unknown experiment: {name}. Use --list to see options.", file=sys.stderr)
        sys.exit(1)

    result = run(name, threads=args.threads, time_limit=args.time_limit,
                 mip_gap=args.mip_gap)
    out_file = ROOT / "results" / f"{name}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    obj = result["global_objective_cny"]
    status = "optimal" if not any(
        y["status"] != "optimal" for y in result["years"].values()
    ) else "WARNING"
    print(f"{name}: obj={obj:.2e}, status={status}, time={result['elapsed_seconds']}s")
    for yr, yd in sorted(result["years"].items()):
        shares = yd["pathway_shares"]
        parts = " ".join(f"{pw}={v:.1%}" for pw, v in sorted(shares.items()) if v > 0.005)
        print(f"  {yr}: {parts}")


if __name__ == "__main__":
    main()
