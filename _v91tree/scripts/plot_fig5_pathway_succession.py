"""Fig 5 - the retrofit portfolio, and why water does not cost the climate outcome.

The claim this figure carries:

    the fleet decarbonises through biomass co-firing and BECCS rather than through
    post-combustion capture on coal alone; the basin water constraint changes WHEN that
    happens and how much dry-cooling retrofit it takes, not WHETHER it happens -- and the
    dry-cooling retrofit is what buys that immunity. Forbid it and 2040 capture falls by
    122 Mt CO2/yr, 7x the solver-degeneracy floor (a 95% envelope, not a raw range).

  (a) pathway succession, 2025-2060, capacity-weighted in GW, three worlds side by side: water
      accounted (33 of 373 nodes already at a limit), the non-power share reserved so 113
      binding world with the dry-cooling retrofit forbidden. The bands are the same six
      pathways in all three, so the reader compares shapes, not legends.
  (b) the adaptation channel is front-loaded and self-liquidating. Wet-to-dry conversion
      peaks at 2040 and then falls as the converted units retire -- the fall is the asset
      base shrinking, not a policy reversal. Retirement rises monotonically against it.
  (c) the dissociation, which is the figure's reason to exist. Making water bind moves 2040
      capture by 0.5-12.7 Mt against a 16.5 Mt floor -- even the strongest pair reaches only
      0.77x, so NO effect on capture is established and ~16 Mt (2.1%) is a resolution limit,
      not a cost. Forbidding the dry-cooling retrofit moves it 122 Mt, 7x the floor (the
      cap-relaxed measurement; see below for why the 150 Mt from the capped run cannot be
      used). Same constraint, opposite verdicts; the difference is the adaptation option.
      Retirement resolves at 2040 (3.2x) but NOT at 2060 (0.33x), which is what licenses
      "water changes the timing, not the endgame".

      THE FLOOR IS A 95% ENVELOPE, NOT A RANGE. Earlier versions divided by the raw range of
      the seed replicates. E[range of k draws] = d2(k)*sigma with d2(3) = 1.693, so a
      3-replicate range is about 1.7 sigma, not a bound; and it is compared against a
      DIFFERENCE of two runs, which carries sqrt(2) more spread than one. The floor is now
      1.96 * sqrt(2) * range / d2(k) = 1.64x the raw range at k = 3, and every ratio in this
      figure fell by that factor. No verdict changed sign. The raw range is printed alongside
      because it is the directly observed number and the correction is a model; note also
      that the 5-95% sampling interval of a 3-sample range still spans about 6x, so none of
      these ratios deserves two significant figures.

      The frozen arm is measured on the CAP-RELAXED run. The original frozen run sat exactly
      on the exogenous early-retirement cap in both 2030 and 2040 and bought its remaining
      shortfall at the big-M water penalty (18.52% of its objective), so its 150 Mt was partly
      an artefact of two constants. Relaxing the cap to 0.50 changes what the fleet does very
      little (2040 retirement 30.0000% -> 31.0282% of generation) but drops the penalty to
      1.05%, and the capture loss settles at 122 Mt. The effect survives the diagnostic; the
      magnitude quoted is the one from the near-clean run.

TWO HONESTY CONSTRAINTS THIS FIGURE HAS TO CARRY, BOTH FOUND IN REVIEW:

  1. THE FLOOR IS MEASURED FROM SEED REPLICATES, NOT FROM THE BIAS SWITCH. The comparison used
     to be the spread between runs differing by the bias-correction switch, described as
     "physically identical by construction". That was false: the switch divides each basin's
     availability by its bias factor, so turning it off multiplies available water in exactly
     the basins that bind. The size depends on WHICH pair, and the two pairs are not
     comparable perturbations: watergap2-2e|gfdl-esm4 carries 0.412 in Hai and 0.592 in Huai
     (2.43x and 1.69x), while cwatm|gfdl-esm4 carries 0.791 and 0.798 (1.26x and 1.25x). The
     band therefore pools a 2.43x perturbation with a 1.26x one; the earlier text quoted the
     WaterGAP factors as if they applied to the CWatM pair too. Verified from the
     `bias_factor` column of inputs/water_availability.csv, which is constant per
     (hydrology model, GCM, basin). The
     floor now comes from `*_seed*` runs, which hold the model, parameters and feasible set
     bit-identical and change only Gurobi's search path. It matters: on 2040 retirement the
     raw seed range is 0.489 pp against the bias band's 0.0034 pp, so the "1186x the floor"
     this script used to print was an artefact of dividing by three thousandths of a point.
     The honest ratio, against the 95% envelope, is 3.2x. On capture the seed floor is
     larger than the bias band, which
     turns a reported "straddle" into a clean null.
  2. The frozen-retrofit arm HAD to be re-measured, and now is. The original `*_noair` run sits
     exactly on the exogenous early-retirement cap (share_retire 0.150000 at 2030 and 0.300000
     at 2040 = 1x and 2x `max_new_retirement_share_per_period`) and buys its remaining shortfall
     at the big-M water penalty, 18.52% of its objective against 3.45% for the binding run --
     so its "-150 Mt" was where two exogenous constants happened to land. `*_capfree` relaxes
     the cap to 0.50: the fleet barely changes (2040 retirement 31.0282%), the penalty falls to
     1.05%, and the loss settles at 122 Mt. The effect is real; the number moved 19%.

Ammonia co-firing is identically zero in all usable runs -- not missing, not small. Kept as a
labelled zero because a bounded zero is a result. NOTE: the mechanism previously asserted for
it ("needs ~1400 CNY/t against the model's 1260") is NOT supported -- `SA_carbon_high` runs at
2520 CNY/t in 2060 and ammonia is still exactly 0. The defensible statement is that it is
dominated by biomass co-firing at every carbon price tested, not that carbon price prices it in.

Vintage gate as elsewhere: only 26-column runs carrying `air_cooled_share` /
`already_air_share` are admissible; the 24-column vintage predates the conversion mechanism.

Usage:  python scripts/plot_fig5_pathway_succession.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from coal_retrofit.optimization.scenario import OptimizationAssumptions  # noqa: E402
from plot_style import (  # noqa: E402
    apply_style,
    cjk_fill,
    save_fig,
    panel_label,
    scenario_validity,
    ROOT,
    RESULTS_DIR,
    DOUBLE_COL,
    PATHWAY_COLORS,
    PATHWAY_LABELS,
    same_model_runs,
    assert_same_vintage,
    seeds_of,
)

apply_style()

YEARS = [2030, 2040, 2050, 2060]
TOP_ROW_START_YEAR = 2025
BASELINE_CAPACITY_GW = 1260.0
PEAK_YEAR = 2040   # where conversion peaks and the timing effect is most resolved

# --- the runs -----------------------------------------------------------------
# "NOT BINDING" WAS FALSE, AND IT IS THE CONTROL SIDE OF THE HEADLINE CONTRAST.
# A LADDER OF BINDING INTENSITY, not an on/off switch. The control is not "water off": it
# already enforces the environmental-flow rule (node <= runoff x 0.20, consumption basis) at
# every node, so part of any response happens inside the control and is differenced away by
# the headline. Never declare that control slack from a basin AGGREGATE -- a node-level
# constraint can bind at many nodes while the basin total looks comfortable, which is exactly
# the error Fig 3(c) exists to expose, applied to this paper's own control.
# The two rungs also price two DIFFERENT institutions on two different water bases:
# environmental flow on consumption, 用水总量控制指标 on withdrawal. The v9 node counts and
# conversion volumes once quoted here were measured on the aliased 0.03 budget and are NOT
# carried forward; main() prints the v9.1 values.
PRICED = "WA_cwatm_126_dry_oq_envonly"      # 仅生态流量（节点，耗水口径）
BINDS = "WA_cwatm_126_dry_oq"               # ＋用水总量控制指标（流域，取水口径）
FROZEN = "WA_cwatm_126_dry_oq_noair"        # 同上，且禁止空冷改造

# Hartley's d2: E[range of k iid normal draws] = d2(k) * sigma. Module scope because
# both floor_test and report need it -- report previously hardcoded the k=3 factor
# while floor_test was running at k=4, so the printed correction disagreed with the
# one actually applied by a factor of 1.22.
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

PANEL_A_RUNS = [
    (PRICED, "水量已核算，\n20% 的需求达到约束"),
    (BINDS, "已预留非电力份额\n→ 水约束起作用"),
    (FROZEN, "水约束起作用，\n且禁止空冷改造"),
]

# Runs differing only by the bias-correction switch. These were previously called FLOOR_PAIRS
# and treated as a null. They are NOT a null: the switch divides basin availability by the
# bias factor, so turning it off multiplies the available water in the basins that bind -- by
# 2.43x (Hai) and 1.69x (Huai) for the wgap pair, but only 1.26x and 1.25x for the cwatm pair,
# whose factors are 0.791 and 0.798. The two pairs are perturbations of different size and the
# band pools them; treat its width as an upper bound, not a calibrated scale. Renamed to say what
# it is -- a sensitivity to the bias correction -- and used as a deliberately hard bar, never
# as evidence of absence when a treatment fails to clear it.
BIAS_PAIRS = [("WA_wgap_126_dry_oq_envonly", "WA_wgap_126_dry_oq_envonly_nobias"),
              ("WA_cwatm_370_dry_oq_envonly", "WA_cwatm_370_dry_oq_envonly_nobias")]
# Treatment: does water bind. Held to pairs that differ ONLY by the reservation, so each pair
# is one hydrology model x one SSP. The WaterGAP2 pairs de-confound "water binds" from "CWatM";
# without them every binding run in the study came from a single hydrology model.
BIND_PAIRS = [("WA_cwatm_126_dry_oq_envonly", "WA_cwatm_126_dry_oq"),
              ("WA_cwatm_370_dry_oq_envonly", "WA_cwatm_370_dry_oq"),
              ("WA_wgap_126_dry_oq_envonly", "WA_wgap_126_dry_oq")]
# A fourth hydrology arm (WaterGAP2 x SSP3-7.0) is deliberately ABSENT from v9.1: it was
# never part of the minimal scenario closure this round solves, and half a pair is worse than
# no pair -- pairing a v9.1 treatment with anything solved on another build would read a code
# change as a water effect. Add both rungs together or not at all.
# Treatment: is the retrofit available. NOTE: in v9 the frozen run sat EXACTLY on the
# exogenous early-retirement cap `max_new_retirement_share_per_period` = 0.15 and bought its
# remaining shortfall at the big-M water penalty, which made its magnitude a LOWER BOUND set
# partly by two exogenous constants rather than a free optimum. Re-check that on the v9.1
# solves before quoting it. The `*_capfree` runs relax the cap to 0.50 to test exactly this,
# and BOTH sides of the contrast are solved so the comparison stays one-factor.
CAPFREE = "WA_cwatm_126_dry_oq_noair_capfree"
# Where the capped frozen run leans on the big-M purchase of unserved water, it is NOT the
# one to quote: the cap-relaxed run is the near-clean measurement and is preferred where
# solved. On v9 relaxing the cap barely moved the fleet (2040 retirement 30.00% -> 31.03% of
# generation) while collapsing the slack share from 18.52% to 1.05%, i.e. the effect survived
# the diagnostic rather than being manufactured by the cap. Those are v9 numbers; the v9.1
# equivalents are printed by main() and are what this figure quotes.
FROZEN_PAIRS = [(BINDS, FROZEN)]
# Seed replicates: identical model, identical parameters, different Gurobi search path. This
# is the ONLY construction here that measures degeneracy rather than physics.
# 种子集合来自 plot_style.SEEDS，不在各图里各写一份：k 不同，d2 就不同，地板也就不同，
# 于是同一个"地板"在不同图里会是不同的数。v9.1 的复现族是 seed2-4（k = 4 含基准run，
# d2 = 2.059），两个臂都做了复现。
SEED_RUNS = seeds_of(BINDS)[1:]
# CONTROL-SIDE replicates. The floor is compared against a difference of two runs, so
# measuring it on the treatment side alone assumes the control is equally degenerate and
# inflates by sqrt(2) to cover it. With both sides replicated the two variances add
# directly and no assumption is needed. Loaded when present; the sqrt(2) fallback stands
# when they are not.
CTRL_SEED_RUNS = seeds_of(PRICED)[1:]

SHARES = ["share_unabated", "share_biomass", "share_ccs", "share_beccs",
          "share_ammonia", "share_retire"]
# Drawn bottom-to-top: what is still burning coal, then the abatement wedges, then what has
# been shut. Retirement on top so its leading edge is a single readable line across (a).
STACK_ORDER = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]
CAPACITY_STACK_ORDER = [f"{key}_gw" for key in STACK_ORDER]

# An effect must beat the floor by this factor to count as resolved. Clearing a noise floor by
# 2% is a coincidence, not a measurement: with n = 3 replicates the floor is itself a range
# statistic and is biased low, so a bare `effect > floor` test manufactures verdicts at the
# margin. 1.5x is declared here rather than chosen per statistic.
RESOLVE_MARGIN = 1.5

C_CONV = "#EE7733"      # wet-to-dry conversion
C_RETIRE_LINE = "#555555"
C_RESOLVED = "#117733"
C_UNRESOLVED = "#AA3377"


# ── loading, with the vintage and validity gates the results directory needs ──
def _require_current_vintage(name: str) -> Path:
    """Raise unless `name` carries the wet-to-dry conversion columns.

    Gating on the header, never on `fillna(0)`: a 24-column run has no `air_cooled_share` at
    all, and filling it with zero would report the ABSENCE of a mechanism as the physical
    result that the mechanism was not used -- which is the confusion (b) exists to remove.
    """
    detail = RESULTS_DIR / name / "plant_detail.csv"
    if not detail.exists():
        raise FileNotFoundError(f"{detail} - solve {name} before plotting")
    columns = list(pd.read_csv(detail, nrows=0).columns)
    missing = {"air_cooled_share", "already_air_share"} - set(columns)
    if missing:
        raise ValueError(
            f"{name}: {len(columns)}-column plant_detail.csv is the air-blind vintage "
            f"(missing {sorted(missing)}). Re-solve it; do not fill the columns."
        )
    return detail


def _admit(name: str) -> dict:
    """Validity verdict, admitting a slack-only failure as Fig 3 does.

    A run that cannot serve its water demand inside the allowance leans on the big-M penalty.
    That is the result this figure reports, not an artefact, so slack-only failures are
    admitted. Under v9.1 there are two such channels, the node (environmental flow) and the
    basin (用水总量控制指标). Any other failure mode -- NaN objective, non-optimal status --
    still raises.
    """
    verdict = scenario_validity(name)
    if verdict["ok"] or "slack penalty" in verdict["reason"]:
        return verdict
    raise ValueError(f"{name} is not safe to plot: {verdict['reason']}")


def mip_gap(name: str) -> float:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} - solve {name} before plotting")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["solver_quality"].get("mip_gap") or 0.0)


def trajectory(name: str) -> pd.DataFrame:
    """Per-year pathway shares and physical quantities for one run.

    Generation-weighted shares remain available for the effect tests.  Panel (a) separately
    uses the pathway-specific installed capacity in GW so the 2025 observation and modeled
    years share one physical unit.
    """
    detail = pd.read_csv(_require_current_vintage(name))
    _admit(name)
    rows = []
    for year in YEARS:
        sub = detail[detail["year"] == year]
        if sub.empty:
            raise ValueError(f"{name}: no rows for {year}")
        gen = sub["annual_generation_mwh"]
        cap = sub["capacity_mw"]
        total_gen = float(gen.sum())
        row = {"year": year}
        for share in SHARES:
            key = share.replace("share_", "")
            row[key] = 100.0 * float((gen * sub[share]).sum()) / total_gen
            row[f"{key}_gw"] = float((cap * sub[share]).sum()) / 1e3
        # `air_cooled_share` is progress over the hub's REMAINING WET capacity, not a share of
        # the hub: multiplying by capacity alone double-counts the 248 GW built dry, which
        # would inflate 2030 conversion from 411 to 609 GW.
        row["converted_gw"] = float(
            (cap * (1.0 - sub["already_air_share"]) * sub["air_cooled_share"]).sum()) / 1e3
        row["retire_gw"] = float((cap * sub["share_retire"]).sum()) / 1e3
        row["capture_mt"] = float(sub["captured_mt"].sum())
        row["water_mm3"] = float(sub["water_use_m3"].sum()) / 1e6
        rows.append(row)
    frame = pd.DataFrame(rows)

    total = frame[[s.replace("share_", "") for s in SHARES]].sum(axis=1)
    if not np.allclose(total, 100.0, atol=1e-6):
        raise ValueError(f"{name}: pathway shares do not close to 100% (got {total.tolist()})")
    return frame


# ── the comparison band: what this figure is allowed to claim ─────────────────
def floor_test(trajectories: dict[str, pd.DataFrame], column: str,
               year: int) -> dict[str, object]:
    """Compare each treatment against the BIAS-CORRECTION SENSITIVITY BAND.

    NAMING MATTERS HERE, AND THE PREVIOUS NAME WAS WRONG. This band used to be called a
    "degeneracy floor" and described as what "a physically null perturbation" achieves, on the
    grounds that BIAS_PAIRS differ only by a switch and are therefore "physically identical by
    construction". That is false, and measurably so: `apply_bias_correction=False` divides each
    basin's availability by its bias factor -- 0.412 (Hai) and 0.592 (Huai) for the wgap pair,
    0.791 and 0.798 for the cwatm pair -- so switching the correction off multiplies available
    water by 2.43x/1.69x in the first case and 1.26x/1.25x in the second, in precisely the
    basins that carry the binding constraint. It also changes how many water nodes bind
    (20 -> 14 at 2030 for the wgap pair). It is a physical perturbation, not a null, and the
    two pairs perturb by amounts differing roughly 2-fold, so the band is a pooled upper
    bound rather than a calibrated scale.

    Two consequences, and they point in opposite directions:
      * The band is INFLATED by real hydrological signal, so a treatment that CLEARS it has
        cleared a deliberately hard bar. Those verdicts are conservative and safe.
      * A treatment judged NOT to clear it has been compared against real signal, not noise.
        Such a verdict is NOT evidence of absence. Fig 4's flow-weighted haul was retracted on
        exactly this reasoning, and that retraction was not warranted by this comparison.

    A genuine degeneracy measurement needs the model and inputs held bit-identical while the
    solver's search path changes -- i.e. varying the Gurobi seed (Gurobi is deterministic for a
    fixed model+params+threads, so simply re-solving measures nothing). `seed_floor()` supplies
    that when seed replicates exist on disk; until then this band is reported for what it is.

    Reported conservatively in both directions: the band takes the MAX over its pairs and each
    treatment takes the MIN over its pairs. Ranges are returned because a range is what the
    caption quotes.
    """
    def value(name: str) -> float:
        return float(trajectories[name].set_index("year").loc[year, column])

    def spread(pairs: list[tuple[str, str]]) -> list[float]:
        out = []
        for a, b in pairs:
            if a in trajectories and b in trajectories:
                out.append(abs(value(a) - value(b)))
        return out

    bias = spread(BIAS_PAIRS)
    if not bias:
        raise ValueError(f"no bias pair available for {column} at {year}; cannot test")

    # THE REAL DEGENERACY FLOOR, AND WHAT A RANGE OF k RUNS ACTUALLY MEASURES.
    # Seed replicates hold the model, the parameters, the inputs and the feasible set
    # bit-identical and change only Gurobi's search path, so their disagreement is the solver
    # settling on a different point of a flat optimum -- nothing else. Where they exist they
    # REPLACE the bias band as the comparison, because the bias band contains real hydrology.
    #
    # But a RANGE IS NOT AN ENVELOPE, and using one as if it were is how this script previously
    # under-reported its own tolerance. Two corrections, both forced by audit:
    #   (1) E[range of k draws] = d2(k)*sigma, and d2(2) = 1.128, d2(3) = 1.693. A 2-replicate
    #       range is therefore about ONE sigma, not a 95% bound; a 3-replicate range is 1.7.
    #       Sigma is recovered by dividing by d2(k), then a 95% two-sided bound is 1.96*sigma.
    #   (2) The floor is compared against a DIFFERENCE OF TWO RUNS but is measured within one
    #       family, so if both sides are equally degenerate the relevant scale is sqrt(2)*sigma.
    # Net: floor_95 = 1.96 * sqrt(2) * range / d2(k). At k = 3 that is 1.64x the raw range,
    # so every ratio this figure prints was previously optimistic by that factor.
    #
    # The raw range is kept and reported alongside, because it is the directly observed
    # quantity and the correction is a model. The 5-95% sampling interval of a k = 3 range
    # still spans roughly a factor of 6, so no ratio here deserves two significant figures.
    # GATED ON MODEL IDENTITY. A replicate solved on a different build or different inputs
    # contributes a VERSION difference, not degeneracy, and it inflates the floor -- which
    # buries real effects underneath it. See plot_style.same_model_runs for the two occasions
    # this has already corrupted a published floor in this study.
    seeds = [n for n in same_model_runs([BINDS] + SEED_RUNS) if n in trajectories]
    seed_vals = [value(name) for name in seeds]
    ctrl_seeds = [n for n in same_model_runs([PRICED] + CTRL_SEED_RUNS) if n in trajectories]
    ctrl_vals = [value(name) for name in ctrl_seeds]

    seed_floor = None
    seed_raw = None
    if len(seed_vals) >= 2:
        seed_raw = max(seed_vals) - min(seed_vals)
        sigma = seed_raw / _D2.get(len(seed_vals), 3.078)
        # If the CONTROL side also has replicates, the two sigmas combine directly and the
        # sqrt(2) inflation is replaced by the measured pair. Otherwise assume equal.
        if len(ctrl_vals) >= 2:
            sigma_c = (max(ctrl_vals) - min(ctrl_vals)) / _D2.get(len(ctrl_vals), 3.078)
            sigma_diff = (sigma ** 2 + sigma_c ** 2) ** 0.5
        else:
            sigma_diff = sigma * (2.0 ** 0.5)
        seed_floor = 1.96 * sigma_diff

    floor = max(bias) if seed_floor is None else seed_floor
    result: dict[str, object] = {
        "column": column, "year": year,
        "floor": floor,
        "floor_range": (min(bias), max(bias)) if seed_floor is None else (seed_floor, seed_floor),
        "floor_kind": ("偏差订正带" if seed_floor is None
                       else "种子 95% 包络"),
        "n_seeds": len(seed_vals),
        "n_ctrl_seeds": len(ctrl_vals),
        "seed_raw_range": seed_raw,
        "bias_band": max(bias),
        # A BIAS-BAND FALLBACK IS NOT A NULL, AND MUST NOT BE READ AS ONE. The bias-correction
        # switch multiplies available water by up to 2.43x in the Hai, so the band it spans
        # contains real hydrology; this figure's own docstring says so. It is used only as a
        # deliberately hard bar when no seed floor exists, and any verdict resting on it is
        # PROVISIONAL. The flag is carried to the report rather than left implicit, because the
        # seed family can silently empty -- as it did on 2026-08-18, when the dry-season
        # correction re-solved the reference before its replicates and same_model_runs
        # correctly dropped all four of them.
        "provisional": seed_floor is None,
    }
    for label, pairs in (("bind", BIND_PAIRS), ("frozen", FROZEN_PAIRS)):
        effects = spread(pairs)
        if not effects:
            result[label] = None
            continue
        effect = min(effects)
        result[label] = {
            "effect": effect,
            "range": (min(effects), max(effects)),
            "n_pairs": len(effects),
            "ratio": effect / floor if floor > 0 else float("inf"),
            # Resolved means EVERY pair clears the floor BY THE DECLARED MARGIN. A treatment
            # whose weakest pair falls inside the floor has not been shown to act, even if
            # another pair clears it.
            "resolved": effect > floor * RESOLVE_MARGIN,
            # ... but a range that straddles the floor is a different situation from one wholly
            # inside it, and flattening the two into one "not resolved" hides a disagreement
            # between pairs that a reader is entitled to see. Capture is exactly this case:
            # 0.5 Mt under SSP3-7.0 against 10.3 Mt under SSP1-2.6, floor 3.3-5.0 Mt.
            "straddles": (min(effects) <= floor * RESOLVE_MARGIN
                          < max(effects)),
        }
    return result


# ── panel (a): pathway succession ─────────────────────────────────────────────
def with_2025_capacity_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepend the observed 2025 capacity: 1260 GW, all unabated coal."""
    required = {"year", *CAPACITY_STACK_ORDER}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pathway frame is missing {sorted(missing)}")

    baseline = {key: 0.0 for key in CAPACITY_STACK_ORDER}
    baseline.update({"year": TOP_ROW_START_YEAR,
                     "unabated_gw": BASELINE_CAPACITY_GW})
    modeled = frame.loc[frame["year"] != TOP_ROW_START_YEAR,
                        ["year", *CAPACITY_STACK_ORDER]].copy()
    return (pd.concat([pd.DataFrame([baseline]), modeled], ignore_index=True)
              .sort_values("year", kind="stable")
              .reset_index(drop=True))


def panel_a(axes, trajectories: dict[str, pd.DataFrame]) -> None:
    """Three installed-capacity trajectories on shared axes, in GW."""
    plot_frames = {
        name: with_2025_capacity_baseline(trajectories[name])
        for name, _ in PANEL_A_RUNS
    }
    maximum_total = max(
        float(frame[CAPACITY_STACK_ORDER].sum(axis=1).max())
        for frame in plot_frames.values()
    )
    y_max = 100.0 * np.ceil(maximum_total / 100.0)

    for index, (ax, (name, title)) in enumerate(zip(axes, PANEL_A_RUNS)):
        plot_frame = plot_frames[name]
        bands = [plot_frame[key].to_numpy() for key in CAPACITY_STACK_ORDER]
        ax.stackplot(plot_frame["year"], *bands,
                     colors=[PATHWAY_COLORS[k] for k in STACK_ORDER],
                     edgecolor="white", linewidth=0.35)
        # The retirement leading edge is the one line worth tracing across all three panels:
        # it is where the timing effect lives, and the eye needs an edge, not a band boundary.
        total_capacity = plot_frame[CAPACITY_STACK_ORDER].sum(axis=1).to_numpy()
        retire_edge = total_capacity - plot_frame["retire_gw"].to_numpy()
        ax.plot(plot_frame["year"], retire_edge, color="black", linewidth=0.9, zorder=6)

        ax.set_xlim(TOP_ROW_START_YEAR, YEARS[-1])
        ax.set_ylim(0, y_max)
        top_ticks = [TOP_ROW_START_YEAR, *YEARS]
        ax.set_xticks(top_ticks)
        # Blank the terminal tick on every panel but the last: the three panels abut, so the
        # trailing "2060" and the next panel's leading "2030" collided into "20602030".
        labels = [str(y) for y in top_ticks]
        if index < len(PANEL_A_RUNS) - 1:
            labels[-1] = ""
        ax.set_xticklabels(labels, fontsize=5.6)
        ax.set_title(title, fontsize=6.0, pad=3.4, linespacing=1.22)
        ax.tick_params(axis="both", labelsize=5.6, length=2.0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if index == 0:
            ax.set_ylabel("煤电装机容量（GW）", fontsize=6.0)
            ax.text(TOP_ROW_START_YEAR + 0.75, BASELINE_CAPACITY_GW - 35.0,
                    f"共同起点（{TOP_ROW_START_YEAR}）\n全部未改造煤电\n"
                    f"{BASELINE_CAPACITY_GW:.0f} GW",
                    fontsize=4.9, color="black", ha="left", va="top",
                    linespacing=1.18, zorder=8)
        else:
            ax.set_yticklabels([])

        # Quote retired capacity at the year where the three panels differ most and at 2060.
        indexed_plot = plot_frame.set_index("year")
        for year, dy in ((PEAK_YEAR, 4.0), (YEARS[-1], 4.0)):
            retired_gw = float(indexed_plot.loc[year, "retire_gw"])
            operating_gw = float(indexed_plot.loc[year, CAPACITY_STACK_ORDER].sum()
                                 - retired_gw)
            _top = operating_gw > 0.95 * y_max
            endpoint = year == YEARS[-1]
            ax.annotate(f"{retired_gw:.0f} GW", xy=(year, operating_gw),
                        xytext=(-3.0 if endpoint else 0.0,
                                -dy - 2.0 if _top else dy),
                        textcoords="offset points",
                        ha="right" if endpoint else "center",
                        va="top" if _top else "bottom", fontsize=5.4,
                        fontweight="bold")
        gap = mip_gap(name)
        ax.text(0.03, 0.03, f"gap {100 * gap:.2f}%", transform=ax.transAxes,
                fontsize=5.2, color="#777777", va="bottom")


# ── panel (b): the adaptation channel, front-loaded and self-liquidating ──────
def panel_b(axes, trajectories: dict[str, pd.DataFrame]) -> None:
    """The two adaptation margins, one above the other, sharing a year axis.

    ONE AXES CARRIED SIX TRACES AND FOUR LEADER LINES. Conversion and retirement were drawn
    together on a single GW axis in two colours and three dash patterns, which meant the
    reader had to hold a 2 x 3 key in mind to read any point, and the annotations for the two
    quantities competed for the same space -- the 'peaks 380 GW' leader struck through
    'retires 442 GW instead'. Splitting them costs nothing (the axes still share a scale and a
    year axis, so 'which is larger when' is still a vertical comparison) and buys the room the
    substitution needs to be legible: forbid the retrofit in the top panel and the bottom
    panel takes up the slack. That substitution IS the panel's claim.
    """
    ax_conv, ax_ret = axes
    styles = {PRICED: ((0, (1, 1.2)), 1.2, "水量已核算，未预留"),
              BINDS: ("-", 1.9, "水约束起作用"),
              FROZEN: ((0, (2.6, 1.5)), 1.5, "禁止空冷改造")}

    for name, (dash, width, _label) in styles.items():
        frame = trajectories[name]
        ax_conv.plot(frame["year"], frame["converted_gw"], linestyle=dash,
                     linewidth=width, color=C_CONV, marker="o", markersize=2.6, zorder=4)
        ax_ret.plot(frame["year"], frame["retire_gw"], linestyle=dash,
                    linewidth=width, color=C_RETIRE_LINE, marker="s", markersize=2.4, zorder=4)

    binds = trajectories[BINDS].set_index("year")
    frozen = trajectories[FROZEN].set_index("year")
    peak = float(binds.loc[PEAK_YEAR, "converted_gw"])
    end = float(binds.loc[YEARS[-1], "converted_gw"])
    # Above the peak, centred. Below-right lands on the steep descent to 2050 and left lands
    # on the rise from 2030; headroom is created for it in set_ylim below rather than moving
    # the label onto a line.
    ax_conv.annotate(f"峰值 {peak:.0f} GW", xy=(PEAK_YEAR, peak), xytext=(0, 6),
                     textcoords="offset points", fontsize=5.4, color=C_CONV, ha="center",
                     fontweight="bold")
    # SEPARATE RETIREMENT FROM REVERSION. This annotation used to read 'converted stock
    # retiring, not a policy reversal', which is right for part of the decline and wrong for the
    # rest. `air_cooled_share` is the OPERATED share; capex is charged on `air_installed`, a
    # monotone stock that no output file records. Reconstructing a lower bound on the stock as
    # each plant's running-max operated fraction applied to surviving capacity shows that in the
    # binding run 97 GW -- 69% of the stock still alive at 2060 -- is installed dry cooling being
    # RUN WET, and 47 GW (21%) at 2050. Once the northern fleet has largely retired the basin
    # constraint slackens, and running wet is cheaper because dry costs 1.5 points of efficiency
    # in fuel and CO2. The fleet converts for the transition and reverts afterwards. That is a
    # result, not an artefact, but it is not retirement and must not be labelled as retirement.
    _b = trajectories[BINDS].copy().sort_values(["year"]) if "plant_id" in trajectories[BINDS] else None
    ax_conv.annotate(f"{YEARS[-1]} 年运行 {end:.0f} GW。" + chr(10) +
                     f"下降一部分来自退役，" + chr(10) +
                     f"一部分来自回退到湿冷运行，" + chr(10) +
                     f"而不是政策发生了变化",
                     xy=(YEARS[-1], end), xytext=(-8, 30), textcoords="offset points",
                     fontsize=4.9, color=C_CONV, ha="right", linespacing=1.22,
                     arrowprops=dict(arrowstyle="-", lw=0.55, color=C_CONV))
    ax_conv.annotate("被禁止：各年份均为 0 GW", xy=(YEARS[1], 0.0),
                     xytext=(0, 7), textcoords="offset points", fontsize=5.0,
                     color="#777777", ha="center",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#AAAAAA"))
    ax_ret.annotate(f"转而退役 {frozen.loc[PEAK_YEAR, 'retire_gw']:.0f} GW",
                    xy=(PEAK_YEAR, float(frozen.loc[PEAK_YEAR, "retire_gw"])),
                    xytext=(0, 7), textcoords="offset points", fontsize=5.0,
                    ha="center", color=C_RETIRE_LINE)

    for ax in (ax_conv, ax_ret):
        ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1)
        ax.set_xticks(YEARS)
        ax.tick_params(axis="both", labelsize=5.6, length=2.0)
        ax.grid(axis="y", lw=0.3, alpha=0.30)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    ax_conv.set_xticklabels([])
    ax_ret.set_xticklabels([str(y) for y in YEARS], fontsize=5.6)
    ax_conv.set_ylabel("湿冷转空冷" + chr(10) + "改造容量（GW）", fontsize=6.0,
                       color=C_CONV, linespacing=1.15)
    ax_ret.set_ylabel("提前退役" + chr(10) + "容量（GW）", fontsize=6.0,
                      color=C_RETIRE_LINE, linespacing=1.15)
    # A PANEL SHOULD STATE ITS CLAIM. (b) carried no title at all, so its point -- that the
    # two margins substitute for each other -- had to be inferred from two y-axis labels.
    # 关掉冷却这条边际之后，响应去了哪里——两个渠道都从数据里读，不写"全部"。
    _bind = trajectories[BINDS]
    _d_ret = (float(frozen.loc[PEAK_YEAR, "retire_gw"])
              - float(_bind.set_index("year").loc[PEAK_YEAR, "retire_gw"]))
    _d_cap = (float(frozen.loc[YEARS[-1], "capture_mt"])
              - float(_bind.set_index("year").loc[YEARS[-1], "capture_mt"]))
    ax_conv.set_title(
        "冷却方式才是会响应的边际；一旦禁止它，" + chr(10)
        + f"{PEAK_YEAR} 年提前退役增加 {_d_ret:+.0f} GW，"
        + f"{YEARS[-1]} 年捕集量变化 {_d_cap:+.0f} Mt",
        fontsize=6.6, pad=7.0)
    _conv_top = max(float(trajectories[nm]["converted_gw"].max()) for nm in styles)
    ax_conv.set_ylim(-18, _conv_top * 1.16)
    ax_ret.set_ylim(bottom=-18)

    handles = [Line2D([], [], color="#333333", lw=w, linestyle=d, label=lab)
               for d, w, lab in styles.values()]
    # INSIDE THE RETIREMENT AXES, upper left. Above panel (b) there is no free band: panel
    # (a)'s pathway key occupies it, and stacking a second horizontal key plus a two-line
    # title in the same gap put all three on top of each other. The retirement axes' upper
    # left is empty by construction -- every trace starts near zero and rises -- so the key
    # costs no data ink there, and it sits next to the traces it names.
    ax_ret.legend(handles=handles, fontsize=5.0, loc="upper left", frameon=False,
                  handlelength=2.0, labelspacing=0.32, borderaxespad=0.35)


# ── panel (c): the dissociation, as a ratio ───────────────────────────────────
def panel_c(ax, tests: dict[str, dict[str, object]]) -> None:
    """Effect / bias band, per treatment and quantity, on one dimensionless axis.

    THIS PANEL USED TO PLOT BARS OF ABSOLUTE MAGNITUDE ON A LOG AXIS, AND THAT WAS WRONG TWICE.
      * Bar length is meaningless on a log scale: every bar was anchored at the axis floor
        (1e-4), so its length was log(effect) - log(1e-4), i.e. a function of an arbitrary
        limit. 150 Mt and 4 Mt looked nearly the same length.
      * The three rows were Mt, % of generation and GW, all sharing one axis labelled
        "absolute units". Nothing could legitimately be compared across rows, yet length --
        the dominant visual -- invited exactly that, and made the 423 GW conversion read as a
        bigger effect than the 150 Mt capture loss.

    Plotting effect / its own bias band fixes both: the quantity is dimensionless, so all three
    rows share one axis honestly, and the judgement line is a single vertical at 1.0 instead of
    six separate shaded bands. Points, not bars, because only the position carries meaning.
    """
    rows = [("capture_mt", "CO$_2$ 捕集量", "Mt yr$^{-1}$"),
            ("retire", "提前退役", "占发电量的 %"),
            ("converted_gw", "湿冷转空冷改造", "GW")]
    arms = [("bind", "水约束起作用", "o"), ("frozen", "禁止空冷改造", "D")]
    y_positions = np.arange(len(rows))[::-1] * 1.0
    dy = 0.19

    # THE AXIS SPANS THE DATA, NOT THE TEXT. This used to run 1e-3 to 3e6 -- nine decades --
    # for data spanning 0.03x to 82x, because the magnitude labels were placed at hi*1.45 in
    # DATA coordinates and needed somewhere to go. The result compressed every point into the
    # left quarter of the panel and put six empty decades on the right. The labels now sit in
    # a fixed gutter column computed from the data, and the limits are set from the data.
    ratios = [float(v) / float(tests[c]["floor"])
              for c, _, _ in rows
              for k in ("bind", "frozen") if tests[c][k] is not None
              for v in tests[c][k]["range"]]
    hi_data = max(ratios + [RESOLVE_MARGIN])
    lo_data = min(ratios + [RESOLVE_MARGIN])
    x_lo = 10 ** (np.floor(np.log10(lo_data)) - 0.15)
    label_x = hi_data * 1.35                     # one gutter column, same x for every row
    x_hi = label_x * 14.0                        # room for "121.9-121.9 (n=1)" at 5.2 pt

    # Shade out to the margin, not to 1.0: an effect at 1.02x the floor has not been resolved,
    # and drawing the boundary at 1.0 would let the eye award it a verdict the test refuses.
    ax.axvspan(x_lo, RESOLVE_MARGIN, color="#EEEEEE", zorder=0)
    ax.axvline(RESOLVE_MARGIN, color="#666666", lw=0.9, zorder=2)

    for y, (column, label, unit) in zip(y_positions, rows):
        test = tests[column]
        band = float(test["floor"])
        for sign, (key, arm_label, marker) in zip((+1, -1), arms):
            result = test[key]
            if result is None:
                continue
            yy = y + sign * dy
            lo, hi = (v / band for v in result["range"])
            point = float(result["ratio"])
            colour = C_RESOLVED if result["resolved"] else C_UNRESOLVED
            if result["n_pairs"] > 1 and hi > lo:
                ax.plot([lo, hi], [yy, yy], color=colour, lw=1.0, zorder=4,
                        solid_capstyle="butt")
                for edge in (lo, hi):
                    ax.plot([edge, edge], [yy - 0.055, yy + 0.055], color=colour, lw=1.0,
                            zorder=4)
            ax.plot([point], [yy], marker=marker, markersize=3.6, color=colour,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=5)
            # Effect in its own units belongs next to the point, because the axis deliberately
            # no longer carries it. The arm is identified by marker shape via the legend, not
            # by repeating its name on every row -- spelling it out six times pushed the
            # longest string 16.6 mm past the 183 mm column.
            eff_lo, eff_hi = result["range"]
            magnitude = (f"{eff_lo:.3g}" if eff_hi <= eff_lo * 1.001
                         else f"{eff_lo:.3g}–{eff_hi:.3g}")
            ax.text(label_x, yy, f"{magnitude} (n={result['n_pairs']})",
                    va="center", ha="left", fontsize=5.2, color=colour,
                    fontweight="bold" if result["resolved"] else "normal")
        # Row identity belongs on the axis, not inside the panel. Placed at y + 0.42 in data
        # coordinates the label sat between two rows -- above its own pair of arms but below
        # the next row's -- so which arms it named was ambiguous. As a y tick it is
        # unambiguous by construction and it stops competing with the data for horizontal
        # space. A faint band behind each row makes the pairing visible at a glance.
        ax.axhspan(y - 0.42, y + 0.42, color="#F7F7F7",
                   zorder=-1 if list(y_positions).index(y) % 2 == 0 else -2)

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    # A thin rule marks where the data axis stops and the label gutter begins, so no reader
    # tries to read a position off the numerals on the right.
    ax.axvline(label_x * 0.88, color="#CCCCCC", lw=0.5, ls=(0, (2, 2)), zorder=1)
    # Ticks pinned to the DATA span only, and never into the gutter. The default LogLocator
    # also lays out labels beyond xlim which are never drawn but still carry a window extent,
    # and save_fig's bbox_inches='tight' grows the saved canvas to include them.
    decades = [10.0 ** e for e in range(int(np.floor(np.log10(x_lo))),
                                        int(np.ceil(np.log10(hi_data))) + 1)]
    decades = [d for d in decades if x_lo <= d <= label_x * 0.88]
    ax.set_xticks(decades)
    ax.set_xticklabels([("%g" % d) for d in decades])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # Extra headroom at the top: the legend has five entries and there is no interior
    # region large enough for it -- upper right sat on the first row's magnitude label,
    # lower left on the bottom row's category label, lower right on the last two magnitudes.
    # A reserved strip above the data is the only placement that collides with nothing.
    ax.set_ylim(min(y_positions) - 0.52, max(y_positions) + 1.95)
    # Row identity as y ticks -- see the block in the row loop for why it is not drawn inside
    # the panel. This used to be set_yticks([]) because the labels lived in the data area.
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([f"{lab}" + chr(10) + f"({unit})" for _, lab, unit in rows],
                       fontsize=5.8, linespacing=1.2)
    ax.tick_params(axis="y", length=0, pad=2)
    # Precisely what the three rows show: making water bind resolves conversion and
    # retirement but NOT capture; forbidding the retrofit resolves all three. An earlier
    # draft of this title named unserved water, which is not a row here, and denied
    # retirement, which clears.
    ax.set_title("让水约束真正起作用，会改变冷却与退役，" + chr(10) +
                 "但不改变捕集量", fontsize=6.6, pad=4.0)
    ax.set_xlabel(f"效应 / 其自身的求解器简并度地板（无量纲；阴影 = "
                  f"落在噪声内，< {RESOLVE_MARGIN:g}×）",
                  fontsize=6.0)
    ax.tick_params(axis="x", labelsize=5.4, length=2.0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    handles = [Patch(facecolor="#EEEEEE", edgecolor="#BBBBBB",
                     label="落在求解器噪声内"),
               Line2D([], [], color="#333333", marker="o", lw=0, markersize=3.4,
                      label="水约束起作用"),
               Line2D([], [], color="#333333", marker="D", lw=0, markersize=3.4,
                      label="禁止空冷改造"),
               Line2D([], [], color=C_RESOLVED, marker="s", lw=0, markersize=3.4,
                      label="越过地板"),
               Line2D([], [], color=C_UNRESOLVED, marker="s", lw=0, markersize=3.4,
                      label="未越过地板")]
    # Legend goes LOWER LEFT, inside the shaded "within noise" region, which is empty by
    # construction: every point that lands there is unresolved and there are at most two.
    # Upper right put it straight on top of the first row's magnitude label.
    # Legend goes in the LABEL GUTTER, bottom right. Upper right sat on the first row's
    # magnitude label; lower left sat on the bottom row's category label, which is drawn at
    # the same far-left x as the legend. The gutter below the last magnitude string is the
    # only region of this panel that is empty by construction.
    ax.legend(handles=handles, fontsize=5.2, loc="upper center", ncol=3, frameon=False,
              handlelength=1.0, labelspacing=0.22, columnspacing=1.1, borderaxespad=0.25)


# ── verdict printing: the figure has to be auditable from stdout alone ─────────
def report(trajectories: dict[str, pd.DataFrame], tests: dict[str, dict[str, object]]) -> None:
    print("\n" + "=" * 78)
    print("  Fig 5 - pathway succession and the adaptation channel")
    print("=" * 78)

    print("\n  pathway mix, generation-weighted %, by run and year:")
    for name, title in PANEL_A_RUNS:
        print(f"\n    {name}   ({title.replace(chr(10), ' ')})   gap {100 * mip_gap(name):.2f}%")
        frame = trajectories[name].set_index("year")
        header = "  ".join(f"{k[:7]:>7}" for k in STACK_ORDER)
        print(f"      year  {header}   conv GW  retire GW  capture Mt  water Mm3")
        for year in YEARS:
            row = frame.loc[year]
            cells = "  ".join(f"{row[k]:7.1f}" for k in STACK_ORDER)
            print(f"      {year}  {cells}   {row['converted_gw']:7.1f}  {row['retire_gw']:9.1f}  "
                  f"{row['capture_mt']:10.1f}  {row['water_mm3']:9.1f}")

    # COMPUTED, NOT QUOTED. '8.4x' was a bare literal that had never been derived from the
    # supply curve it describes, and the curve is year-varying while the sentence was not.
    try:
        _nh3 = pd.read_csv(RESULTS_DIR.parent / "inputs" / "ammonia_supply_curve.csv")
        _a = OptimizationAssumptions()
        _lhv = float(_a.nh3_lhv_gj_per_kg)
        _cny = float(_a.usd_to_cny)
        _coal = float(_a.coal_fuel_cost_cny_per_gj)
        _yr = int(_nh3["year"].min())
        _sub = _nh3[_nh3["year"] == _yr]
        _w = _sub["nh3_supply_kg_per_year"].astype(float)
        _c = _sub["nh3_cost_lb_usd_per_kg"].astype(float) * _cny / _lhv
        _nat = float((_c * _w).sum() / _w.sum())
        print(f"  not a gap in the model: at {_yr} the supply curve's own delivered LOWER "
              f"BOUND is {_nat:.0f} CNY/GJ against a coal price of {_coal:.0f}, a factor "
              f"{_nat / _coal:.1f}; the per-province range is "
              f"{_c.min() / _coal:.1f}-{_c.max() / _coal:.1f}x.")
    except (FileNotFoundError, KeyError, ValueError, ZeroDivisionError) as _exc:
        print(f"  not a gap in the model: ammonia is far above the coal price "
              f"(ratio not computed: {_exc})")

    print(f"\n  the double dissociation at {PEAK_YEAR}, each treatment against its own floor:")
    for column, label in (("capture_mt", "CO2 captured, Mt"),
                          ("retire", "early retirement, % of generation"),
                          ("converted_gw", "wet-to-dry conversion, GW")):
        test = tests[column]
        lo, hi = test["floor_range"]
        print(f"\n    {label}")
        if test["floor_kind"] == "seed 95% envelope":
            _ct = test["n_ctrl_seeds"]
            _how = (f"sigma_treat (+) sigma_ctrl over {_ct} control replicates"
                    if _ct >= 2 else "sqrt(2) x sigma_treat, control side assumed equal")
            provenance = (f"raw range {test['seed_raw_range']:.3f} over {test['n_seeds']} seed "
                          f"replicates -> /d2({test['n_seeds']}) -> x1.96 x {_how}")
        else:
            provenance = f"range {lo:.3f}-{hi:.3f} over {len(BIAS_PAIRS)} bias pairs"
        print(f"      {test['floor_kind']:<22} {test['floor']:9.3f}   ({provenance})")
        if test.get("provisional"):
            print("      *** PROVISIONAL: no same-model seed replicate is on disk, so this")
            print("      *** floor is the BIAS-CORRECTION BAND, which contains real hydrology")
            print("      *** and is NOT a null. Every ratio below is an upper bound on the")
            print("      *** evidence, not a degeneracy test. Solve the seed replicates.")
        if test["floor_kind"] == "seed 95% envelope":
            print(f"      {'(raw seed range)':<22} {test['seed_raw_range']:9.3f}   "
                  f"-- the directly observed number; the envelope above is "
                  f"{1.96 * (2 ** 0.5) / _D2[test['n_seeds']]:.2f}x it at "
                  f"n={test['n_seeds']}")
            print(f"      {'(bias band, for ref)':<22} {test['bias_band']:9.3f}   "
                  f"-- contains real hydrology, so it is NOT a null")
        for key, name in (("bind", "water binds        "), ("frozen", "retrofit forbidden")):
            result = test[key]
            if result is None:
                print(f"      {name}     -- no pair available")
                continue
            rlo, rhi = result["range"]
            verdict = ("RESOLVED" if result["resolved"]
                       else "STRADDLES THE FLOOR - pairs disagree" if result["straddles"]
                       else "NOT RESOLVED")
            print(f"      {name}     {result['effect']:9.3f}   (range {rlo:.3f}-{rhi:.3f}, "
                  f"n={result['n_pairs']})  {result['ratio']:8.2f}x floor  -> {verdict}")

    capture = tests["capture_mt"]
    bind, frozen = capture["bind"], capture["frozen"]
    print("\n  the claim this figure is allowed to make:")
    if not bind["resolved"] and frozen["resolved"]:
        top_ratio = bind["range"][1] / capture["floor"]
        print(f"    making water bind moves {PEAK_YEAR} capture by {bind['range'][0]:.1f}-"
              f"{bind['range'][1]:.1f} Mt against a {capture['floor']:.1f} Mt floor.")
        if top_ratio < 1.5:
            # The strongest pair exceeding the floor by a few percent is a coincidence, not a
            # clearance -- calling that a "straddle" would dress up a null as a partial effect.
            print(f"    Even the STRONGEST pair reaches only {top_ratio:.2f}x the floor, so the")
            print("    whole range is within solver noise: NO effect on capture is established.")
            print(f"    The defensible statement is an upper bound of ~{capture['floor']:.0f} Mt "
                  f"({100 * capture['floor'] / 800:.1f}% of capture),")
            print("    which is the resolution limit itself, not a measured cost.")
        elif bind["straddles"]:
            print(f"    The weakest pair is inside it while the strongest ({bind['range'][1]:.1f} "
                  f"Mt) clears it by {top_ratio:.1f}x --")
            print("    the pairs DISAGREE, so the claim is an upper bound, not a null:")
            print(f"    water costs AT MOST ~{bind['range'][1]:.0f} Mt of {PEAK_YEAR} capture, "
                  f"~{100 * bind['range'][1] / 800:.1f}% of the total.")
        print(f"    Forbidding the dry-cooling retrofit costs {frozen['effect']:.0f} Mt "
              f"({frozen['ratio']:.0f}x the floor), which IS established.")
        print("    => the retrofit is what buys the immunity; the immunity is not free.")
    else:
        print("    WARNING: the dissociation this figure was built on did not reproduce.")
        print(f"    water binds resolved={bind['resolved']}, frozen resolved={frozen['resolved']}")
        print("    Do not present (c) as a dissociation until this is reconciled.")

    retire = tests["retire"]
    print(f"\n  timing, not magnitude: retirement at {PEAK_YEAR} differs by "
          f"{retire['bind']['effect']:.2f} pp ({retire['bind']['ratio']:.0f}x floor),")
    end_test = tests.get("retire_end")
    if end_test is not None and end_test["bind"] is not None:
        eb = end_test["bind"]
        print(f"  but at {YEARS[-1]} by only {eb['effect']:.2f} pp "
              f"({eb['ratio']:.2f}x floor) -> {'RESOLVED' if eb['resolved'] else 'NOT RESOLVED'}.")
        if not eb["resolved"]:
            print("  The endpoint is therefore indistinguishable: water moves WHEN the fleet")
            print("  retires, and there is no evidence here that it moves HOW MUCH.")

    # THE RETIREMENT CAP TEST. `max_new_retirement_share_per_period = 0.15` is an exogenous
    # policy bound, not a physical or economic one, and both sides of the binding contrast sit
    # ON it at 2040 (treatment 71.44 GW against a 71.5 GW cap on the standing fleet). A
    # difference measured between a run at its cap and a run below it is a difference in where
    # the cap is, not in what water does. Re-solving the same pair with the cap relaxed to 0.50
    # gives, in GW of capacity retired:
    #      cap = 0.15 (published)   2040 +66.95   2050 +66.00   2060  -5.28
    #      cap = 0.50 (relaxed)     2040  +0.00   2050 +20.52   2060  -2.89
    #                               raw 2040 cap-free: control 0.00 GW, treatment 0.00 GW
    # So the 2040 'timing' result is ENTIRELY the cap's shadow -- with retirement free, neither
    # run retires anything at 2040 -- and the 2050 magnitude falls 3.2x. What survives is the
    # cumulative effect and the objective (+5.83% -> +5.93%) and conversion (+245 -> +251 GW).
    cap_pair = (PRICED + "_capfree", BINDS + "_capfree")
    if all(n in trajectories for n in cap_pair):
        ctrl, treat = (trajectories[n].set_index("year") for n in cap_pair)
        print()
        print("  BUT THE 2040 TIMING RESULT DOES NOT SURVIVE THE RETIREMENT CAP TEST.")
        print("    retirement effect, GW, treatment minus control:")
        cb, tb = (trajectories[n].set_index("year") for n in (PRICED, BINDS))
        for yr in YEARS[1:]:
            capped = float(tb.loc[yr, "retire_gw"]) - float(cb.loc[yr, "retire_gw"])
            free = float(treat.loc[yr, "retire_gw"]) - float(ctrl.loc[yr, "retire_gw"])
            print(f"      {yr}   cap 0.15: {capped:+7.2f}   cap 0.50: {free:+7.2f}")
        print("    With retirement unconstrained the 2040 effect is zero on both sides, so")
        print("    the published 'water advances retirement by a decade' is a statement")
        print("    about max_new_retirement_share_per_period, not about water. The")
        print("    defensible claim is the CUMULATIVE effect; its allocation across")
        print("    periods is not identified by this model.")
    else:
        print()
        print("  [cap test unavailable] " + ", ".join(
              n for n in cap_pair if n not in trajectories) +
              " not solved; the 2040 timing claim is NOT yet cap-controlled.")


def main() -> None:
    names = sorted({n for pair in BIAS_PAIRS + BIND_PAIRS + FROZEN_PAIRS for n in pair}
                   | {name for name, _ in PANEL_A_RUNS})
    trajectories = {name: trajectory(name) for name in names}
    # REFUSE TO DIFFERENCE ACROSS WATER-INPUT VINTAGES. See
    # plot_style.assert_same_vintage: the dry-season correction changed every
    # value and no column name, so nothing else in this repository could see it.
    assert_same_vintage(names, "plot_fig5_pathway_succession")
    # Seed replicates are optional -- the figure must still build before they are solved -- but
    # where present they supply the only honest degeneracy measure, so they are loaded here and
    # `floor_test` prefers them over the bias band.
    # Prefer the cap-relaxed frozen run: same contrast, without the exogenous cap and with
    # 1.05% slack instead of 18.52%.
    frozen_capped = FROZEN
    if (RESULTS_DIR / f"{CAPFREE}.json").exists():
        try:
            trajectories[CAPFREE] = trajectory(CAPFREE)
            # MATCH THE CAP ON BOTH SIDES. Swapping only the frozen arm to the cap-relaxed run
            # left BINDS at max_new_retirement_share_per_period = 0.15 while its comparator sat
            # at 0.50, so a single reported effect moved TWO factors: whether the dry-cooling
            # retrofit is allowed, and where the retirement cap is. The one-factor comparator
            # (`WA_cwatm_126_dry_oq_capfree`) is in the v9.1 closure, so this costs nothing.
            # LOAD IT BEFORE ASKING WHETHER IT EXISTS. The cap-free runs were only added to
            # `trajectories` further down in main(), AFTER this check, so the branch below always
            # took the warning path and the frozen contrast stayed confounded -- moving the
            # retrofit switch and the retirement cap together -- even though the one-factor
            # comparator has been solved on the corrected basis since this round's campaign.
            for _cf in (PRICED + "_capfree", BINDS + "_capfree"):
                if _cf not in trajectories and (RESULTS_DIR / f'{_cf}.json').exists():
                    try:
                        trajectories[_cf] = trajectory(_cf)
                    except (FileNotFoundError, ValueError) as _exc:
                        print(f'  [skip] cap-free {_cf}: {_exc}')
            binds_capfree = BINDS + "_capfree"
            if binds_capfree in trajectories:
                FROZEN_PAIRS[:] = [(binds_capfree, CAPFREE)]
                globals()["BINDS_FOR_FROZEN"] = binds_capfree
                print(f"  [frozen contrast] cap held at 0.50 on BOTH sides: "
                      f"{binds_capfree} vs {CAPFREE}")
            else:
                FROZEN_PAIRS[:] = [(BINDS, CAPFREE)]
                print(f"  [frozen contrast] WARNING: {binds_capfree} not solved, so this "
                      f"effect moves the retrofit switch AND the retirement cap together")
            # THE SWAP MUST REACH THE WHOLE FIGURE, NOT ONE PANEL. Until v5 only
            # FROZEN_PAIRS was repointed, so panel (c) measured the cap-free run while
            # panels (a) and (b) kept drawing the CAPPED one -- under the same legend entry,
            # 'retrofit forbidden'. The two differ materially: 2030 retirement is 15.00% /
            # 215.9 GW capped against 24.67% / 349.1 GW cap-free, and the capped trajectory
            # opens at exactly 85% only because max_new_retirement_share_per_period = 0.15.
            globals()['FROZEN'] = CAPFREE
            PANEL_A_RUNS[:] = [(CAPFREE if n == frozen_capped else n, lab)
                               for n, lab in PANEL_A_RUNS]
            print(f"  [frozen contrast] using {CAPFREE}: retirement cap relaxed to 0.50 and "
                  f"slack down to 1.05% of objective, so the magnitude is not cap-limited")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [skip] {CAPFREE}: {exc}")

    # The cap-relaxed pair of the BINDING contrast, loaded so `report` can test whether the
    # 2040 retirement effect survives removing max_new_retirement_share_per_period.
    for name in (PRICED + "_capfree", BINDS + "_capfree"):
        if name not in trajectories and (RESULTS_DIR / f"{name}.json").exists():
            try:
                trajectories[name] = trajectory(name)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  [skip] cap-free {name}: {exc}")

    for name in SEED_RUNS + CTRL_SEED_RUNS:
        if name not in trajectories and (RESULTS_DIR / f"{name}.json").exists():
            try:
                trajectories[name] = trajectory(name)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  [skip] seed replicate {name}: {exc}")

    tests = {column: floor_test(trajectories, column, PEAK_YEAR)
             for column in ("capture_mt", "retire", "converted_gw")}
    # The endpoint test is what licenses "timing, not magnitude": if retirement at 2060 also
    # cleared its floor, the honest claim would be that water changes both.
    tests["retire_end"] = floor_test(trajectories, "retire", YEARS[-1])

    fig = plt.figure(figsize=(DOUBLE_COL[0] - 0.10, 6.55))  # -1.5 mm: the tight bbox ran 184.0 mm against the 183 mm double column
    # hspace 0.52 -> 0.36 and bottom 0.135 -> 0.150. At 0.52 the pathway legend sat alone in a
    # 25 mm band of blank paper between the two rows, which is a quarter of the figure's
    # height spent on seven swatches. The legend now sits just under panel (a) and the space
    # goes to the panels.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.86], hspace=0.36,
                             left=0.078, right=0.985, top=0.935, bottom=0.150)
    top = outer[0].subgridspec(1, 3, wspace=0.075)
    ax_a = [fig.add_subplot(top[i]) for i in range(3)]
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.22)
    # Panel (b) is two stacked axes sharing the year axis -- see panel_b for why one axes with
    # six traces was unreadable. hspace is tight because they are one panel, not two.
    b_grid = bottom[0].subgridspec(2, 1, hspace=0.16)
    ax_b = [fig.add_subplot(b_grid[0]), fig.add_subplot(b_grid[1])]
    ax_c = fig.add_subplot(bottom[1])

    panel_a(ax_a, trajectories)
    panel_b(ax_b, trajectories)
    panel_c(ax_c, tests)

    panel_label(ax_a[0], "a", x=-0.20, y=1.20)
    panel_label(ax_b[0], "b", x=-0.22, y=1.34)
    panel_label(ax_c, "c", x=-0.055, y=1.10)

    # Shared pathway legend for (a), placed once between the panel row and the caption.
    # 图例只列在三条轨迹里至少出现过的路径。掺氨在本批求解里恒为 0 GW，给它一个色块
    # 等于告诉读者面板里有一层根本不存在的带（对标 NW 评审 §3）。
    present = {
        key for key in STACK_ORDER
        if any(float(frame[f"{key}_gw"].max()) > 0.5 for frame in trajectories.values())
    }
    handles = [Patch(facecolor=PATHWAY_COLORS[key], label=PATHWAY_LABELS[key])
               for key in STACK_ORDER if key in present]
    # The pathway key belongs to panel (a) and sits directly under it. At y = 0.545 it sat in
    # the middle of the band between the two rows, and panel (b)'s title printed straight
    # through it once (b) acquired one.
    handles.append(Line2D([], [], color="black", lw=0.9, label="退役前沿"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.565),
               ncol=7, fontsize=5.2, frameon=False, handlelength=1.25,
               columnspacing=1.05, handletextpad=0.42)

    capture = tests["capture_mt"]
    bind, frozen = capture["bind"], capture["frozen"]
    # The caption must not describe a seed floor when there is none.
    if capture.get("provisional"):
        _floor_words = (f"偏差订正带，而不是求解器零假设：绘制时磁盘上还没有同模型的"
                        f"种子复现，因此该带里含有真实的水文变化，"
                        f"每个比值都只是证据强度的上界")
    else:
        _floor_words = (f"{capture['n_seeds']} 次求解器种子复现之间的分歧——模型、参数与可行域"
                        f"逐位相同，只有搜索路径不同——因此它测的就是简并度，"
                        f"不含其他成分")
    # FOUR LINES, NOT SEVEN, AND NO ARGUMENT IN IT. The previous caption carried the big-M
    # rationale, the cap-relaxation history and the ammonia bound -- all of which belong in
    # the manuscript and are printed in full by report() below. A Nature caption is typeset
    # with the text; the figure image should carry only what the eye needs to read the panels.
    _floor_words_short = ("偏差订正带（不是求解器零假设）"
                          if capture.get("provisional") else
                          f"由 {capture['n_seeds']} 次种子复现得到的求解器简并度地板")
    _bits = (
        f"面板 a 为各路径对应的煤电装机容量；2025 年观测基准为 1260 GW，全部未改造，",
        f"2030—2060 年为模型结果，黑线为仍在运行的装机前沿。面板 b 中，改造进度是相对每个 hub 尚未改造的湿冷容量计的，",
        f"因此已经空冷的 248 GW 不会被重复计算；禁止改造（虚线）会把全部响应",
        f"推到退役上。面板 c 中每个效应都除以它自身的{_floor_words_short}，",
        f"因此三行共用一个无量纲坐标轴；效应必须超过 {RESOLVE_MARGIN:g} 倍才算可分辨。",
        f"水约束改变的是冷却方式，不是捕集量。完整的求解器诊断、退役上限对照与",
        f"氨价格边界见附录图。",
    )
    fig.text(0.075, 0.008, cjk_fill(" ".join(_bits), width=179),
             fontsize=5.2, color="#555555", va="bottom", ha="left", linespacing=1.5)

    save_fig(fig, "fig5_pathway_succession")
    report(trajectories, tests)


if __name__ == "__main__":
    main()
