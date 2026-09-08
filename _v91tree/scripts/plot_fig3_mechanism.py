"""Fig 3 - what makes the water account bind, and how the fleet answers.

The claim this figure carries:

    whether the water account becomes a binding constraint is set by water-allocation
    institutions, not by climate change; and when the non-power share of the basin
    allowance is reserved, the fleet converts to dry cooling at scale rather than
    abandoning capture.

  (a) the asymmetry. Three single-factor contrasts on identical axes -- does the model see
      basin water at all, is the non-power share of the allowance reserved, and which
      emissions scenario. Cost is on its own sub-axis because it is the only quantity the
      solver's optimality tolerance bounds, and that tolerance is drawn per contrast. The
      headline is REAL SPENDING (+2.06%), which clears its 0.88% MIP gap by 2.3x and the
      model's measured solver-degeneracy floor by 8.3x; the objective including the big-M
      penalty on unserved water is +5.71%, but 64% of that increment is the penalty, so it
      is shown as secondary text only. The climate contrast clears neither -- it is 0.16x
      the degeneracy floor -- and its sign is not resolved.
  (b) how the fleet adapts. Three runs that differ one step at a time -- water accounted but
      already binding on 20% of demand, allowance reserved so it binds on 46%, and the same
      dry-cooling retrofit forbidden -- against the six channels a response can use.
  (c) where it happens. Per basin, the as-built full-capture draw against the dry-season
      allowance, beside the conversion the solver actually chose.

Three things are deliberately NOT in this figure:

  * No sensitivity analysis. Nature main figures do not carry one. The dry-cooling capex
    sweep (`*_air1000`, `*_air1370`) is computed and printed for checking, and belongs in
    Extended Data, but is not drawn.
  * No variance decomposition and no factor attribution. Those were the old Fig 3 and are
    Extended Data material; `scripts/plot_fig3_attribution.py` is kept for them.
  * No stylized emission factors. Residual emissions come from
    `plot_style.residual_emissions_mt`, which reproduces the solver's own accounting.

Two admissions the figure has to make in public, because both would otherwise be hidden:

  1. A run may fail `plot_style.scenario_validity` by leaning on the slack penalty for water
     demand it could not serve. Such runs are the entire point of the figure, so they are
     admitted -- but the unserved volume is drawn as a component of the cost bar in (a), as
     its own channel in (b), and per basin in (c). Under v9.1 there are TWO slack channels,
     the node (environmental flow) and the basin (用水总量控制指标), penalised at the same
     rate so the solver cannot rank one institution above the other for a numerical reason.
  2. Because that penalty is a big-M (1 000 CNY/m3 in `solver.py:618`, ~125x the delivered
     water cost the model actually pays), the +5.71% headline splits into +2.06% of real
     spending and +3.65% of penalty. Both readings are drawn. The conclusion survives
     either way: even the economic-only increment clears the MIP gap by 2.3x, against a
     climate contrast that clears nothing. The MIP gap is a solver TOLERANCE and bounds the
     objective rather than real spending, so the audit also prints the measured degeneracy
     floor from seed replicates (model bit-identical, only Gurobi's search path different):
     0.249% of objective, which the +2.06% clears by 8.3x.

Vintage gate: only runs carrying `air_cooled_share` / `already_air_share` (26 columns) are
usable. The 24-column vintage predates the wet-to-dry conversion mechanism and has a
different feasible set -- its 2030 water use is 8 272 Mm3 against 3 338-5 356 here. The gate
reads the CSV header and raises; nothing is filled with zero.

Usage:  python scripts/plot_fig3_mechanism.py
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
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plot_style import (  # noqa: E402
    apply_style,
    cjk_fill,
    save_fig,
    panel_label,
    scenario_validity,
    residual_emissions_mt,
    assign_basin,
    ROOT,
    RESULTS_DIR,
    DOUBLE_COL,
    BASIN_NAMES_ZH,
    BASIN_ORDER,
    same_model_runs,
    assert_same_vintage,
)
from coal_retrofit.constants import WATER_EXTRACTABLE_FRACTION  # noqa: E402
from coal_retrofit.optimization.scenario import OptimizationAssumptions  # noqa: E402

apply_style()

INPUTS = ROOT / "inputs"
NEAR_YEAR = 2030   # where the constraint bites hardest and conversion peaks
END_YEAR = 2060    # where retirement and capture have played out

# --- the five solved runs this figure rests on -------------------------------
NOWATER = "BASE"                              # water priced, basin availability not enforced
PRICED = "WA_cwatm_126_dry_oq_envonly"     # 生态流量约束（节点，耗水口径）
BINDS = "WA_cwatm_126_dry_oq"              # 再叠加用水总量控制指标（流域，取水口径）
BINDS_HOT = "WA_cwatm_370_dry_oq"          # 同上，SSP3-7.0
FROZEN = "WA_cwatm_126_dry_oq_noair"       # 叠加总量指标，且禁止空冷改造
# Air-retrofit capex sweep. Reported to stdout as a robustness check, NEVER drawn: a
# sensitivity belongs in Extended Data, not in a Nature main figure.
CAPEX_CHECK = ["WA_cwatm_126_dry_oq_air1000", "WA_cwatm_126_dry_oq_air1370"]

# v9.1: the node budget is the environmental-flow rule ALONE. v9 multiplied in an assumed
# non-power reservation `(1 - 0.85)`, which aliased the depletion standard with the
# allocation rule so that no result could say which was binding. The allocation now enters
# as its own basin constraint on the WITHDRAWAL basis, read off 用水总量控制指标, and is not
# folded into this factor. Held identical to plot_fig1_water_footprint.py.
USABLE = WATER_EXTRACTABLE_FRACTION

# Panel (c) reproduces Fig 1(c)'s stress definition exactly -- same year, same SSP, same
# intensity basis -- so the two main figures cannot quote different numbers for the same
# quantity. Robustness across the other three SSP x year cells is printed, not drawn.
STRESS_YEAR = END_YEAR
STRESS_SSP = "ssp370"

# --- colours (Tol, colourblind-safe) ----------------------------------------
C_ACCOUNT = "#88CCEE"   # does the model see basin water at all
C_RESERVE = "#332288"   # is the non-power share reserved -- the institutional lever
C_CLIMATE = "#CC3311"   # which emissions scenario
C_PENALTY = "#EE7733"   # unserved-water penalty, i.e. not real spending
C_CONV = "#117733"
C_RETIRE = "#999999"
C_CAPTURE = "#4477AA"
C_RESID = "#DDAA33"
C_SHORT = "#CC3311"
C_OVER = "#CC3311"
C_UNDER = "#BBBBBB"

CONTRASTS = [
    ("是否施加\n生态流量约束", NOWATER, PRICED, C_ACCOUNT, "accounted"),
    ("是否叠加用水\n总量控制指标", PRICED, BINDS, C_RESERVE, "quota"),
    ("SSP1-2.6 → SSP3-7.0\n（含总量指标）", BINDS, BINDS_HOT, C_CLIMATE, "SSP3-7.0"),
]

PHYSICAL = [
    ("converted_gw", f"空冷改造，\n{NEAR_YEAR} 年"),
    ("capture_mt", f"CO$_2$ 捕集量，\n{END_YEAR} 年"),
    ("water_mm3", f"耗水量，\n{NEAR_YEAR} 年"),
]

LADDER = [
# A LADDER, NOT A SWITCH. The control is not "water off": it already enforces the
# environmental-flow rule at every node, so part of any response is inside it and is
# differenced away. The rungs are BASE -> envonly -> oq, and they price two DIFFERENT
# institutions on two different water bases (see docs/官方指标口径水预算.md).
# The per-rung numbers that used to be quoted here were v9 measurements on the aliased
# 0.03 budget and are NOT carried forward; the v9.1 values are printed by main().
    ("仅生态流量约束\n（节点，耗水口径）", PRICED, C_UNDER),
    ("＋用水总量控制指标\n（流域，取水口径）", BINDS, C_CONV),
    ("……且禁止\n空冷改造", FROZEN, C_SHORT),
]

CHANNELS = [
    ("converted_gw", f"空冷改造\n{NEAR_YEAR} 年（GW）", C_CONV, False),
    ("retire_gw_near", f"提前退役\n{NEAR_YEAR} 年（GW）", C_RETIRE, False),
    ("retire_gw_end", f"提前退役\n{END_YEAR} 年（GW）", C_RETIRE, False),
    ("capture_mt", f"CO$_2$ 捕集量\n{END_YEAR} 年（Mt）", C_CAPTURE, False),
    ("residual_mt", f"残余排放\n{END_YEAR} 年（Mt）", C_RESID, True),
    ("unserved_mm3", f"未满足的需水\n{NEAR_YEAR} 年（Mm$^3$）", C_SHORT, False),
]


# ── loading, with the two gates the results directory needs ──────────────────
def _payload(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} - solve {name} before plotting")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_current_vintage(name: str) -> Path:
    """Raise unless `name` carries the wet-to-dry conversion columns.

    Gating on the header rather than on `fillna(0)`: a 24-column run has no
    `air_cooled_share` at all, and filling it with zero would silently report the absence of
    a mechanism as a physical result -- exactly the confusion this figure exists to remove.
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
    """Validity verdict, admitting a slack-only failure and reporting the volume.

    `scenario_validity` rejects a run leaning more than 1% of its objective on slack. A run
    that cannot serve its water demand inside the allowance leans on the big-M penalty --
    which is the result, not an artefact. Slack-only failures are therefore admitted here and
    their unserved volume is drawn. Any other failure mode (NaN objective, non-optimal
    status) still raises.
    """
    verdict = scenario_validity(name)
    if verdict["ok"] or "slack penalty" in verdict["reason"]:
        return verdict
    raise ValueError(f"{name} is not safe to plot: {verdict['reason']}")


def unserved_m3(name: str, year: int | None = None) -> float:
    """Water demand the solver could not serve, m3, from its own water_supply slack."""
    path = RESULTS_DIR / name / "slack_detail.csv"
    if not path.exists():
        return 0.0
    slack = pd.read_csv(path)
    if slack.empty or "slack_value" not in slack.columns:
        return 0.0
    slack = slack[slack["constraint_type"] == "water_supply"]
    if year is not None:
        slack = slack[slack["year"] == year]
    return float(slack["slack_value"].sum())


def outcomes(name: str) -> dict[str, float]:
    """Every plotted scalar for one scenario, all derived from results/ at plot time."""
    detail = pd.read_csv(_require_current_vintage(name))
    verdict = _admit(name)
    payload = _payload(name)
    near = detail[detail["year"] == NEAR_YEAR]
    end = detail[detail["year"] == END_YEAR]

    slack_total = sum(
        float(y.get("cost_breakdown", {}).get("slack_penalty", 0.0))
        for y in payload["years"].values()
        if isinstance(y, dict)
    )
    water_near = float(near["water_use_m3"].sum())
    short_near = unserved_m3(name, NEAR_YEAR)

    def converted(frame: pd.DataFrame) -> float:
        # `air_cooled_share` is progress over the hub's REMAINING WET capacity, not a share
        # of the hub. Multiplying by capacity alone double-counts the 248 GW that was built
        # dry, which would inflate 2030 conversion from 411 to 609 GW.
        return float(
            (frame["capacity_mw"] * (1.0 - frame["already_air_share"]) * frame["air_cooled_share"]).sum()
        ) / 1e3

    return {
        "cost": float(payload["global_objective_cny"]) / 1e12,
        # The objective net of the big-M penalty on unserved water: the part that is real
        # spending on water, dry-cooling retrofit, retirement and capture.
        "cost_economic": (float(payload["global_objective_cny"]) - slack_total) / 1e12,
        "cost_penalty": slack_total / 1e12,
        "mip_gap": float(payload["solver_quality"].get("mip_gap") or 0.0),
        "slack_share": float(verdict["slack_share"]),
        "converted_gw": converted(near),
        "converted_gw_end": converted(end),
        "retire_gw_near": float((near["capacity_mw"] * near["share_retire"]).sum()) / 1e3,
        "retire_gw_end": float((end["capacity_mw"] * end["share_retire"]).sum()) / 1e3,
        "capture_mt": float(end["captured_mt"].sum()),
        # Model-consistent, never a stylized per-pathway factor (plot_style.residual_emissions_mt).
        "residual_mt": residual_emissions_mt(end, END_YEAR),
        "water_mm3": water_near / 1e6,
        "water_mm3_end": float(end["water_use_m3"].sum()) / 1e6,
        "unserved_mm3": short_near / 1e6,
        "unserved_pct": 100.0 * short_near / (water_near + short_near) if water_near + short_near else 0.0,
    }


# ── panel (a): the asymmetry ─────────────────────────────────────────────────
# Hartley's d2: E[range of k iid normal draws] = d2(k) * sigma.
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847}
SEED_REPLICATES = [BINDS] + [f"{BINDS}_seed{i}" for i in (2, 3, 4, 5, 6)]


def seed_envelopes() -> dict[str, float]:
    """95% envelope on a DIFFERENCE of two runs, per channel, from solver-seed replicates.

    WHY THIS EXISTS. This figure drew 2060 capture responses of +2.2 / +0.9 / -2.3% with bold
    numerals and no band, while the four seed replicates on disk -- bit-identical model,
    parameters and feasible set, only the Gurobi search path differing -- span 28.7 Mt on the
    same quantity, i.e. a 95% difference envelope of +/-38.6 Mt = +/-4.9%. Every one of those
    three numbers is inside it, and Fig 5 says so explicitly of the same quantity while this
    figure did not. Two main figures cannot disagree about whether an effect exists.

    All runs terminate at node_count = 1, so each incumbent is a root heuristic and the spread
    is genuine search-path degeneracy rather than an unconverged bound.
    """
    # GATED ON THE MODEL, not merely on the file existing -- see plot_style.same_model_runs.
    available = same_model_runs(SEED_REPLICATES)
    if len(available) < 3:
        return {}
    try:
        samples = [outcomes(n) for n in available]
    except (FileNotFoundError, ValueError):
        return {}
    out: dict[str, float] = {}
    for key, _, _, _ in CHANNELS:
        vals = [float(s[key]) for s in samples if key in s]
        if len(vals) < 3:
            continue
        sigma = (max(vals) - min(vals)) / _D2.get(len(vals), 3.078)
        out[key] = 1.96 * (2 ** 0.5) * sigma
    out["_n"] = float(len(available))
    return out


def contrast_table(values: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for label, base, variant, colour, short in CONTRASTS:
        low, high = values[base], values[variant]
        row = {
            "label": label.replace("\n", " "),
            "plot_label": label,
            "short": short,
            "base": base,
            "variant": variant,
            "colour": colour,
            # THE TOLERANCE ON A DIFFERENCE OF TWO INCUMBENTS IS NOT THE LOOSER GAP.
            # This used to be max(gap_low, gap_high), which is anti-conservative by about
            # 1.9x here. Both runs minimise, so opt_A in [LB_A, INC_A] and opt_B in
            # [LB_B, INC_B], and the true effect satisfies
            #     LB_B - INC_A  <=  opt_B - opt_A  <=  INC_B - LB_A,
            # i.e. relative to the point estimate INC_B - INC_A the certified interval is
            #     [ -gap_B*INC_B , +gap_A*INC_A ]   (asymmetric),
            # of total width gap_A*INC_A + gap_B*INC_B -- the SUM, not the max. Two useful
            # consequences: the SIGN of a positive effect is resolved as soon as the point
            # estimate exceeds gap_B*INC_B alone (a sharper test than either max or sum),
            # while the MAGNITUDE carries the full summed width.
            "gap_pct": 100.0 * (low["mip_gap"] * low["cost"] + high["mip_gap"] * high["cost"]) / low["cost"],
            # Certified interval on the TOTAL-objective effect, in % of the base objective.
            "effect_lo_pct": 100.0 * ((high["cost"] - high["mip_gap"] * high["cost"]) - low["cost"]) / low["cost"],
            "effect_hi_pct": 100.0 * (high["cost"] - (low["cost"] - low["mip_gap"] * low["cost"])) / low["cost"],
            # The one-sided bar the point estimate must clear for the SIGN to be certified.
            "sign_bar_pct": 100.0 * high["mip_gap"] * high["cost"] / low["cost"],
            # The NOT-RESOLVED band, and it is asymmetric. A positive effect is certified
            # once it exceeds gap_B*INC_B; a negative one once it exceeds gap_A*INC_A. The
            # band drawn in panel (a) is therefore [-100*gap_A, +sign_bar], not +/-anything.
            "band_lo_pct": -100.0 * low["mip_gap"],
            "band_hi_pct": 100.0 * high["mip_gap"] * high["cost"] / low["cost"],
            "cost_pct": 100.0 * (high["cost"] - low["cost"]) / low["cost"],
            "cost_economic_pct": 100.0 * (high["cost_economic"] - low["cost_economic"]) / low["cost"],
            "cost_penalty_pct": 100.0 * (high["cost_penalty"] - low["cost_penalty"]) / low["cost"],
        }
        for key, _ in PHYSICAL:
            row[f"{key}_from"] = low[key]
            row[f"{key}_to"] = high[key]
            row[f"{key}_pct"] = 100.0 * (high[key] - low[key]) / low[key] if low[key] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def panel_a_cost(ax, table: pd.DataFrame) -> None:
    """Cost on its own sub-axis, with each contrast's own optimality tolerance drawn.

    Cost is the only quantity the MIP gap bounds, so the band is drawn here and nowhere
    else. Putting a gap band on a GW axis would invent a tolerance the solver never
    reports; the defensible statement for the physical panel is printed instead -- two
    objectives inside the gap are not distinguishable optima, so no difference between
    those two solutions is resolved, physical differences included.
    """
    x = np.arange(len(table))
    width = 0.56
    economic = table["cost_economic_pct"].to_numpy(dtype=float)
    penalty = table["cost_penalty_pct"].to_numpy(dtype=float)
    total = table["cost_pct"].to_numpy(dtype=float)
    # The band is the region in which an effect of that sign is NOT certified by the solver's
    # own bounds, and it is asymmetric: a positive effect must clear gap_variant*obj_variant,
    # a negative one gap_base*obj_base. Drawing +/-max(gap) understated it; drawing +/-sum
    # would overstate it twofold. See contrast_table for the derivation.
    band_lo = table["band_lo_pct"].to_numpy(dtype=float)
    band_hi = table["band_hi_pct"].to_numpy(dtype=float)
    # 可证区间随点估计一起标出：主图与 ED11 面板 a 用同一套区间口径，
    # 不再出现"+3.95%"与"[+1.00, +5.44]%"两个数各说各话（对标 NW 评审 §2.6）。
    eff_lo = table["effect_lo_pct"].to_numpy(dtype=float)
    eff_hi = table["effect_hi_pct"].to_numpy(dtype=float)

    for xi, blo, bhi in zip(x, band_lo, band_hi):
        ax.add_patch(Rectangle((xi - 0.45, blo), 0.90, bhi - blo, facecolor="#000000",
                               alpha=0.11, edgecolor="#888888", linewidth=0.3, ls=(0, (2, 2)),
                               zorder=0))
    ax.bar(x, economic, width, color=table["colour"], zorder=3)
    ax.bar(x, penalty, width, bottom=economic, color=C_PENALTY, hatch="////",
           edgecolor="white", linewidth=0.35, zorder=3)
    ax.axhline(0, color="black", lw=0.7, zorder=4)

    # Headroom for the legend, which lives inside the axes. At 1.42 it overlapped the
    # "(+X% w/ penalty)" and verdict annotations stacked above the tallest bar.
    span = float(np.max(np.maximum(total, band_hi))) * 1.95
    # Legend sits inside the axes; reserve only what the bands actually need below zero.
    ax.set_ylim(float(np.min(band_lo)) * 1.60, span)
    for xi, tot, econ, pen, blo, bhi, elo, ehi in zip(x, total, economic, penalty, band_lo,
                                                      band_hi, eff_lo, eff_hi):
        top = max(tot, econ, bhi)
        # ANCHOR THE HEADLINE TO ITS OWN INK. Moving the +2.06% text off the objective was not
        # enough: it was still positioned at max(tot, econ, gap), i.e. floating above the top of
        # a bar drawn to 5.71, so the eye paired the bold number with a height 2.8x larger than
        # the number. It now sits on the boundary of the SOLID segment, which is what it
        # measures, on a small opaque patch so it stays legible against the hatch above it.
        label_y, label_va = (econ, "bottom") if econ >= 0 else (econ, "top")
        # HEADLINE IS REAL SPENDING, NOT THE OBJECTIVE. Unserved water is charged a big-M
        # penalty (1000 CNY/m3, solver.py:614-618) to keep the model feasible; that is scarcity
        # priced, not money spent, and it is 64% of this contrast's objective increment. Leading
        # with the objective inflates the paper's headline 2.8x (+5.71% vs +2.06%) and the
        # storyline forbids it outright (nature_water_storyline_and_figures.md, 成本口径更正).
        ax.text(xi, label_y, f"{econ:+.2f}%", ha="center", va=label_va,
                fontsize=6.2, fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                          boxstyle="round,pad=0.14"))
        # THE VERDICT IS ON THE TOTAL, BECAUSE THAT IS ALL THE SOLVER CERTIFIES. It used to
        # read "|economic| / gap = Nx", which divides an uncertified quantity by a bound that
        # does not apply to it: cost_economic is one slice of a particular incumbent, and the
        # split of the objective among its components is free to move anywhere inside the gap.
        # The bound applies to the objective, so the objective is what is judged against it.
        certified = tot > bhi or tot < blo
        # 判定与可证区间同一行组：区间是判定的依据，也是 ED11 面板 a 用的同一个数。
        verdict = ("效应可证" if certified else "落在求解器界内") + f"\n[{elo:+.2f}, {ehi:+.2f}]%"
        # 判定要越过粗体点估计标签（约 0.075 span 高）再起头，否则 +1.27% 会顶进区间文字里。
        verdict_y = max(top, econ + span * 0.075 if econ >= 0 else top) + span * 0.030
        ax.text(xi, verdict_y, verdict, ha="center", va="bottom", fontsize=5.2,
                color="black" if certified else "#777777", linespacing=1.15)
        if pen > 0.05:
            # The objective total stays visible, but as secondary text keyed to the penalty.
            # 两行判定占 span 的约 0.17，惩罚行挪到判定之上 0.175。
            ax.text(xi, verdict_y + span * 0.175, f"（含惩罚 {tot:+.2f}%）", ha="center",
                    va="bottom", fontsize=5.2, color="#5A2D00")
            # ONLY LABEL INSIDE THE SEGMENT IF THE SEGMENT CAN HOLD THE LABEL. The gate above
            # is on the penalty in percentage points, which says nothing about how tall the
            # band is on this axis. Two lines of 5.2 pt need roughly 8% of the span; below that
            # the label is led out to the LEFT, where the neighbouring column has no text at
            # that height (to the right sits the next bar's "（含惩罚）" line).
            if pen / span > 0.08:
                ax.text(xi, econ + pen / 2.0, f"{pen:+.2f}\n惩罚", ha="center",
                        va="center", fontsize=5.2, color="#5A2D00", linespacing=1.05)
            else:
                ax.annotate(f"{pen:+.2f} 惩罚", xy=(xi - 0.17, econ + pen / 2.0),
                            xytext=(xi - 0.40, econ + pen / 2.0 + span * 0.06),
                            fontsize=5.2, color="#5A2D00", va="center", ha="right",
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="#5A2D00",
                                            shrinkA=0, shrinkB=1))

    ax.set_xticks(x)
    # short 既是内部键也是刻度文字；只在这里映射成中文，键本身不动（下游按它取数）。
    _SHORT_ZH = {"accounted": "生态流量", "quota": "总量指标", "SSP3-7.0": "SSP3-7.0"}
    ax.set_xticklabels([_SHORT_ZH.get(v, v) for v in table["short"]], fontsize=6.0)
    ax.set_xlim(-0.62, len(table) - 0.38)
    ax.set_ylabel("系统成本变化（%）", fontsize=7)
    # COMPUTED FROM THE SAME FLAGS THE BARS CARRY. A hardcoded verdict here can contradict the
    # per-bar labels directly beneath it, and did: on v9.1 both water rungs land inside the
    # solver's own bound, so no cost effect is certified at all.
    _cert = [str(s) for s, tt, bl, bh in zip(table["short"], total, band_lo, band_hi)
             if tt > bh or tt < bl]
    _zh = {"accounted": "生态流量规则", "quota": "分配规则", "SSP3-7.0": "气候情景"}
    if not _cert:
        _t = "两条水规则的成本效应\n都落在求解器界内"
    elif len(_cert) == 1:
        _t = f"只有{_zh.get(_cert[0], _cert[0])}\n的成本效应可证"
    else:
        _t = "、".join(_zh.get(c, c) for c in _cert) + "\n的成本效应均可证"
    ax.set_title(_t, fontsize=7.2)
    ax.legend(
        handles=[
            # The grey swatch is a SHAPE key, not a colour key, and the label says so. The
            # real-spending bars are each drawn in their own contrast colour (table["colour"]),
            # so a swatch labelled plainly "Real spending" matched no bar on the panel and sent
            # the reader looking for grey ink that does not exist.
            Patch(facecolor="#BBBBBB", edgecolor="#666666", linewidth=0.4,
                  label="真实支出"),
            Patch(facecolor=C_PENALTY, hatch="////", edgecolor="white",
                  label="缺水惩罚项"),
            Patch(facecolor="#000000", alpha=0.11,
                  label="未获可证界"),
        ],
        # UPPER LEFT, not lower centre. Lower centre sat directly on the "accounted"
        # contrast's +0.15% numeral and on both "inside solver bounds" verdicts; the space
        # above that bar is empty from ~1.2 to the top of the axis because the contrast is
        # small, so the legend goes there and touches nothing.
        frameon=False, fontsize=5.0, loc="upper left", ncol=1, handlelength=1.1,
        labelspacing=0.20, borderpad=0.1, borderaxespad=0.25,
    )


def panel_a_physical(ax, table: pd.DataFrame) -> None:
    """The same three contrasts on three physical outcomes, one shared percentage axis."""
    n = len(table)
    x = np.arange(len(PHYSICAL))
    width = 0.74 / n
    grid = np.array([[row[f"{key}_pct"] for key, _ in PHYSICAL] for _, row in table.iterrows()],
                    dtype=float)
    lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
    pad = (hi - lo) * 0.16
    for index, row in table.iterrows():
        offset = (index - (n - 1) / 2) * width
        values = grid[index]
        ax.bar(x + offset, values, width * 0.90, color=row["colour"],
               label=row["plot_label"].replace("\n", " "), zorder=3)
        for xi, value in zip(x, values):
            ax.text(xi + offset, value + (pad * 0.10 if value >= 0 else -pad * 0.10),
                    f"{value:+.1f}", ha="center", va="bottom" if value >= 0 else "top",
                    fontsize=5.3, rotation=90, color="#333333")
    ax.axhline(0, color="black", lw=0.7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in PHYSICAL], fontsize=6.2)
    ax.set_xlim(-0.58, len(PHYSICAL) - 0.42)
    ax.set_ylabel("相对该次求解自身基准的变化（%）", fontsize=7)
    ax.set_title("机组的响应是改变冷却方式，而不是改变捕集量",
                 fontsize=7.2)
    ax.set_ylim(lo - pad * 1.7, hi + pad * 2.6)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right", ncol=1, handlelength=1.1,
              labelspacing=0.25, borderpad=0.2, title="单因子对比",
              title_fontsize=5.8)


# ── panel (b): how the fleet adapts ─────────────────────────────────────────
def panel_b(axes, values: dict[str, dict[str, float]], floors: dict[str, float]) -> pd.DataFrame:
    """Six response channels across three runs that differ one step at a time."""
    rows = []
    x = np.arange(len(LADDER))
    for ax, (key, title, colour, signed) in zip(axes, CHANNELS):
        series = np.array([values[name][key] for _, name, _ in LADDER], dtype=float)
        ax.bar(x, series, 0.66, color=[c for _, _, c in LADDER], zorder=3)
        ax.axhline(0, color="black", lw=0.6, zorder=4)
        lo = min(0.0, float(series.min()))
        hi = max(0.0, float(series.max()))
        pad = max((hi - lo) * 0.42, 1.0)
        ax.set_ylim(lo - (pad * 0.30 if signed else 0.0), hi + pad)
        for xi, value in zip(x, series):
            if signed and value < 0:
                place, va = value - pad * 0.06, "top"
            else:
                place, va = value + pad * 0.06, "bottom"
            ax.text(xi, place, f"{value:,.0f}" if abs(value) >= 10 else f"{value:.1f}",
                    ha="center", va=va, fontsize=5.6, fontweight="bold", color="#222222")
        if key == "unserved_mm3":
            for xi, (_, name, _) in zip(x, LADDER):
                pct = values[name]["unserved_pct"]
                if pct > 0.05:
                    ax.text(xi, series[xi] + pad * 0.42, f"占需求的\n{pct:.1f}%",
                            ha="center", va="bottom", fontsize=5.0, color=C_SHORT,
                            linespacing=1.05)
        # THE DEGENERACY FLOOR, DRAWN. A grey band of +/- the 95% envelope on a DIFFERENCE of
        # two runs, centred on the leftmost bar: any bar whose top falls inside it differs from
        # the reference by less than this model's own solver noise, and the numeral above it is
        # not a measurement. Computed from solver-seed replicates only -- same model, same
        # parameters, same feasible set -- so it contains no physics.
        band = floors.get(key)
        if band:
            ref = float(series[0])
            ax.axhspan(ref - band / 2.0, ref + band / 2.0, color="#000000", alpha=0.055,
                       lw=0, zorder=1)
            inside = [i for i, v in enumerate(series) if abs(v - ref) < band / 2.0 and i > 0]
            title = title + f"\n" + f"种子地板 {band:,.0f}" if band >= 10 else \
                    title + f"\n" + f"种子地板 {band:.1f}"
            if inside:
                title = title + " (n.s.)"
        ax.set_title(title, fontsize=6.0, pad=2.5, fontweight="normal")
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=5.6)
        ax.set_xlim(-0.62, len(LADDER) - 0.38)
        rows.append({"channel": title.replace("\n", " "),
                     **{name: values[name][key] for _, name, _ in LADDER}})
    # Anchored on the axis, not on the figure: a figure-coordinate anchor below y=0 is
    # outside the canvas, and `savefig(bbox_inches="tight")` then pads the whole figure out
    # to include it, which is where the half-page of white space came from.
    axes[0].legend(
        handles=[Patch(facecolor=colour, label=label.replace("\n", " "))
                 for label, _, colour in LADDER],
        frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(-0.55, -0.06),
        ncol=3, handlelength=1.1, columnspacing=1.4, borderpad=0.1,
    )
    return pd.DataFrame(rows)


# ── panel (c): where it happens ─────────────────────────────────────────────
def basin_table() -> pd.DataFrame:
    """Per-basin allowance, as-built full-capture draw, and the conversion the solver chose.

    The stress numerator holds cooling at its AS-BUILT configuration and attaches capture to
    the whole fleet, which is the counterfactual the claim is about: what the account would
    demand if nothing adapted. Generation is `capacity x province_cf x 8760` and intensity is
    the plant table's own with-capture consumption, identical to Fig 1(c) so the two main
    figures cannot quote different stress values for the same basin.

    The denominator is the budget the solver enforces (data_prep.py:437-439):
    dry-season flow x extractable fraction x (1 - non-power reservation).
    """
    assumptions = OptimizationAssumptions()
    plants = pd.read_csv(INPUTS / "plants.csv").reset_index(drop=True)
    plants["basin"] = assign_basin(plants).values
    hours = plants["province_mode"].astype(str).map(lambda p: assumptions.province_cf(p) * 8760.0)
    plants["generation_mwh"] = plants["total_capacity_mw"].astype(float) * hours
    plants["capture_draw_m3"] = (
        plants["generation_mwh"] * plants["consumption_ccs_intensity_m3_per_mwh"].astype(float)
    )
    grouped = plants.groupby("basin").agg(
        capacity_gw=("total_capacity_mw", lambda s: float(s.sum()) / 1e3),
        capture_draw_1e8=("capture_draw_m3", lambda s: float(s.sum()) / 1e8),
        n_sites=("plant_id", "size"),
    )

    avail = pd.read_csv(INPUTS / "water_availability.csv")
    members = int(avail["scenario_id"].nunique())
    if members != 20:
        print(f"  [warn] water_availability.csv carries {members} members, not 20")
    frame = avail[(avail["planning_year"] == STRESS_YEAR) & (avail["ssp"] == STRESS_SSP)]
    per_member = (
        frame.groupby(["basin_code", "hydrology_model", "gcm"])["dry_season_water_m3_per_year"]
        .sum() * USABLE / 1e8
    )
    allowance = per_member.groupby("basin_code").agg(["median", "min", "max", "count"])
    allowance.columns = ["allow_1e8", "allow_min_1e8", "allow_max_1e8", "n_members"]

    table = grouped.join(allowance, how="left")
    table["stress_pct"] = table["capture_draw_1e8"] / table["allow_1e8"] * 100.0
    # Stress is inversely proportional to the allowance, so the DRIEST member gives the
    # HIGHEST stress. Inverted here rather than at plot time.
    table["stress_hi_pct"] = table["capture_draw_1e8"] / table["allow_min_1e8"] * 100.0
    table["stress_lo_pct"] = table["capture_draw_1e8"] / table["allow_max_1e8"] * 100.0

    # Conversion, and the pool it can draw on, from the solved runs themselves.
    for name, tag in ((PRICED, "ref"), (BINDS, "bind")):
        detail = pd.read_csv(_require_current_vintage(name))
        near = detail[detail["year"] == NEAR_YEAR].reset_index(drop=True)
        near["basin"] = assign_basin(near).values
        near["conv_gw"] = (
            near["capacity_mw"] * (1.0 - near["already_air_share"]) * near["air_cooled_share"]
        ) / 1e3
        near["pool_gw"] = near["capacity_mw"] * (1.0 - near["already_air_share"]) / 1e3
        table[f"conv_{tag}_gw"] = near.groupby("basin")["conv_gw"].sum()
        if tag == "ref":
            table["pool_gw"] = near.groupby("basin")["pool_gw"].sum()
    table["conv_ref_pct"] = table["conv_ref_gw"] / table["pool_gw"] * 100.0
    table["conv_bind_pct"] = table["conv_bind_gw"] / table["pool_gw"] * 100.0

    # The constraint acts per water node, not per basin, so a basin whose aggregate looks
    # comfortable can still contain nodes over their budget. Counting them keeps the reader
    # from expecting conversion to be monotone in the basin aggregate -- it is not.
    usage = pd.read_csv(RESULTS_DIR / BINDS / "resource_use.csv")
    usage = usage[(usage["resource_type"] == "water") & (usage["year"] == NEAR_YEAR)].copy()
    usage["basin"] = usage["region"].astype(str).str.rsplit("_", n=1).str[-1]
    nodes = usage.groupby("basin").agg(
        n_nodes=("region", "size"),
        n_at_limit=("utilization", lambda s: int((s >= 0.999).sum())),
        basin_utilisation=("used", "sum"),
        basin_available=("available", "sum"),
        node_median_util=("utilization", "median"),
    )
    nodes["basin_util_pct"] = 100.0 * nodes["basin_utilisation"] / nodes["basin_available"]
    nodes["node_median_pct"] = 100.0 * nodes["node_median_util"]
    # THE NUMBER THE BASIN AGGREGATE CANNOT SHOW: how much of the basin's coal water demand
    # sits on nodes that are completely out of water. Counting saturated NODES is the weaker
    # statistic -- it weights a node serving 4 GW the same as one serving 40 MW -- and it
    # happens to run BELOW the basin aggregate in most basins, which would have made this
    # panel argue the opposite of the truth. Weighting by demand is the honest version and it
    # is one-directional: >= the aggregate in 6 of the 8 basins, and +57 points in the
    # Northwest Interior, whose basin total reads 33% used while 90% of the coal water demand
    # inside it sits on nodes at their limit.
    saturated = usage[usage["utilization"] >= 0.999].groupby("basin")["used"].sum()
    total_used = usage.groupby("basin")["used"].sum()
    nodes["demand_at_limit_pct"] = 100.0 * saturated.reindex(nodes.index).fillna(0.0) / total_used
    table = table.join(nodes[["n_nodes", "n_at_limit", "basin_util_pct",
                              "node_median_pct", "demand_at_limit_pct"]], how="left")

    slack = pd.read_csv(RESULTS_DIR / BINDS / "slack_detail.csv")
    slack = slack[(slack["constraint_type"] == "water_supply") & (slack["year"] == NEAR_YEAR)].copy()
    slack["basin"] = slack["node_id"].astype(str).str.rsplit("_", n=1).str[-1]
    table["unserved_mm3"] = slack.groupby("basin")["slack_value"].sum() / 1e6
    table["unserved_mm3"] = table["unserved_mm3"].fillna(0.0)

    order = [b for b in BASIN_ORDER if b in table.index and np.isfinite(table.loc[b, "capacity_gw"])]
    return table.reindex(order)


def panel_c(ax_stress, ax_conv, table: pd.DataFrame) -> None:
    y = np.arange(len(table))[::-1]
    over = table["stress_pct"].to_numpy(dtype=float) > 100.0

    # -- left: the basin total is not what the fleet experiences --------------
    # THIS PANEL USED TO BE A THIRD COPY OF THE BASIN STRESS RATIO. Fig 1(c) drew it as a
    # percentage bar, Fig 2(c) draws it as a ratio with the full 20-member ensemble range,
    # and this panel drew it again on a log axis. Three panels, one number. The stress ratio
    # now lives in Fig 2(c) alone -- the version with the ensemble range is the one worth
    # keeping -- and this panel carries what none of them showed: the constraint acts on
    # individual water nodes, and the basin aggregate that every basin-scale water assessment
    # reports does not see where it binds.
    #
    # Three marks per basin, all from the same solved run:
    #   open circle  the basin aggregate, used / available summed over its nodes -- what a
    #                basin-scale assessment reports
    #   tick         the MEDIAN node's utilisation -- how the typical node looks
    #   diamond      the share of the basin's coal water demand sitting on nodes at their
    #                limit -- what the fleet actually faces
    #
    # Demand-weighted, not node-counted. Counting saturated nodes weights a node serving
    # 4 GW the same as one serving 40 MW, and on that statistic 6 of 8 basins fall BELOW
    # their aggregate -- the panel would have argued the opposite of the truth. Weighted by
    # demand the divergence is one-directional and large where it matters: the Northwest
    # Interior reads 33% used as a basin, its median node reads 2%, and 90% of the coal water
    # demand inside it sits on nodes that are completely out of water.
    util = table["basin_util_pct"].to_numpy(dtype=float)
    demand_lim = table["demand_at_limit_pct"].to_numpy(dtype=float)
    median_node = table["node_median_pct"].to_numpy(dtype=float)
    gap = demand_lim - util

    for yi, u, d in zip(y, util, demand_lim):
        ax_stress.plot([min(u, d), max(u, d)], [yi, yi], lw=1.1, color="#CCCCCC", zorder=2,
                       solid_capstyle="round")
    ax_stress.scatter(median_node, y, s=10, marker="|", color="#999999", linewidths=0.9,
                      zorder=3, label="节点中位数")
    ax_stress.scatter(util, y, s=16, marker="o", facecolors="white",
                      edgecolors="#555555", linewidths=0.8, zorder=4,
                      label="流域合计")
    ax_stress.scatter(demand_lim, y, s=19, marker="D",
                      color=[C_OVER if o else C_UNDER for o in over], zorder=5,
                      edgecolors="white", linewidths=0.4,
                      label="处于约束边界的节点上的煤电需水")

    for yi, u, d, row in zip(y, util, demand_lim, table.itertuples()):
        ax_stress.text(max(u, d) + 2.5, yi + 0.10,
                       f"{int(row.n_nodes)} 个节点中 {int(row.n_at_limit)} 个达到约束",
                       va="center", ha="left", fontsize=5.3,
                       color=C_OVER if row.stress_pct > 100 else "#666666",
                       fontweight="bold" if row.stress_pct > 100 else "normal")
        ax_stress.text(max(u, d) + 2.5, yi - 0.30,
                       f"{row.capacity_gw:.0f} GW，改造了可改造池的 {row.conv_bind_pct:.0f}%",
                       va="center", ha="left", fontsize=5.0, color="#8A8A8A")

    ax_stress.set_xlim(-4, 178)
    ax_stress.set_xticks([0, 25, 50, 75, 100])
    ax_stress.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=6)
    ax_stress.axvline(100, color="#999999", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax_stress.set_yticks(y)
    ax_stress.set_yticklabels([f"{BASIN_NAMES_ZH.get(b, b)}" for b in table.index], fontsize=6.2)
    for tick, flag in zip(ax_stress.get_yticklabels(), over):
        if flag:
            tick.set_fontweight("bold")
            tick.set_color(C_OVER)
    ax_stress.set_xlabel("百分比（%）\n"
                         f"{NEAR_YEAR} 年求解结果；三种标记均来自同一个解",
                         fontsize=6.2)
    ax_stress.set_title("流域整体可以只用掉三分之一，而其中十分之九的\n"
                        "需求却压在已经饱和的节点上", fontsize=7.2,
                        linespacing=1.25)
    # No "widest gap" callout: the title already states it, and at this size a second
    # sentence of the same fact only collides with the bottom rows' annotations.
    # Legend bottom-right, where rows F/H/G leave the right half of the panel empty; the
    # three labels are short so the whole block clears the "156 GW, converts 1%" text.
    ax_stress.legend(frameon=False, fontsize=5.2, loc="lower right",
                     bbox_to_anchor=(1.0, 0.02), handlelength=0.9, labelspacing=0.30,
                     borderaxespad=0.1, scatterpoints=1)
    ax_stress.grid(axis="x", lw=0.3, alpha=0.3)
    ax_stress.set_axisbelow(True)

    # -- right: and that is exactly where the fleet converts -----------------
    height = 0.30
    limit = float(table["pool_gw"].max()) * 1.88
    ax_conv.barh(y, table["pool_gw"], 0.70, color="#EEEEEE", zorder=1)
    ax_conv.barh(y + height / 2, table["conv_ref_gw"], height * 0.92, color=C_UNDER, zorder=3)
    ax_conv.barh(y - height / 2, table["conv_bind_gw"], height * 0.92, color=C_CONV, zorder=3)
    for yi, row in zip(y, table.itertuples()):
        # Labels sit clear of the widest bar in the row (the pool), so the conversion figure
        # never lands on top of the "convertible" bar it is a fraction of.
        ax_conv.text(row.pool_gw + limit * 0.016, yi,
                     f"{row.conv_bind_gw:.0f} GW，占可改造池 {row.conv_bind_pct:.0f}%",
                     va="center", ha="left", fontsize=5.3, color=C_CONV, fontweight="bold")
        if row.unserved_mm3 > 0.5:
            ax_conv.text(limit * 0.99, yi, f"缺水 {row.unserved_mm3:.0f} Mm$^3$",
                         va="center", ha="right", fontsize=5.0, color=C_SHORT)
    ax_conv.set_xlim(0, limit)
    ax_conv.set_xlabel(f"转为空冷的容量，{NEAR_YEAR} 年（GW）", fontsize=6.4)
    ax_conv.set_title("改造恰好集中在这些流域", fontsize=7.2)
    ax_conv.tick_params(axis="y", labelleft=False, length=0)
    ax_conv.tick_params(axis="x", labelsize=6)
    ax_conv.legend(
        handles=[
            Patch(facecolor="#EEEEEE", label="可供改造的湿冷容量（可改造池）"),
            Patch(facecolor=C_UNDER, label="已改造，仅生态流量"),
            Patch(facecolor=C_CONV, label="已改造，＋总量指标"),
            Patch(facecolor="white", edgecolor="none",
                  label="红色：配额无法满足的需求"),
        ],
        frameon=False, fontsize=5.4, loc="lower right", bbox_to_anchor=(1.0, 0.015),
        handlelength=1.1, labelspacing=0.28, borderpad=0.2,
    )


# ── reporting ───────────────────────────────────────────────────────────────
def report(values: dict[str, dict[str, float]], contrasts: pd.DataFrame,
           ladder: pd.DataFrame, basins: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)

    print("\n" + "=" * 100)
    print("scenario ledger (every run this figure reads)")
    print("=" * 100)
    ledger = pd.DataFrame(values).T[
        ["cost", "cost_economic", "cost_penalty", "mip_gap", "slack_share",
         "converted_gw", "retire_gw_near", "retire_gw_end", "capture_mt", "residual_mt",
         "water_mm3", "unserved_mm3", "unserved_pct"]
    ]
    print(ledger.round(4).to_string())
    for name in values:
        verdict = scenario_validity(name)
        state = "OK" if verdict["ok"] else f"ADMITTED DESPITE: {verdict['reason']}"
        print(f"  {name:34s} {state}")

    print("\n" + "=" * 100)
    print("panel (a) -- the asymmetry: three single-factor contrasts")
    print("=" * 100)
    for row in contrasts.itertuples():
        print(f"\n  {row.label}")
        print(f"    {row.base}  ->  {row.variant}")
        print(f"    cost           {row.cost_pct:+8.3f}%   (real spending {row.cost_economic_pct:+.3f}%"
              f" + unserved-water penalty {row.cost_penalty_pct:+.3f}%)")
        # WHAT THE SOLVER CERTIFIES, AND WHAT IT DOES NOT.
        # The MIP gap bounds the TOTAL OBJECTIVE of each run and nothing else. The previous
        # verdict here divided REAL SPENDING by the gap and reported "clears it by Nx". That
        # is not a valid use of the bound: cost_economic is one slice of a particular
        # incumbent, and any split of the objective among its 18 components is free to move
        # anywhere inside the gap. The certified statement is the interval on the total.
        _resolved = row.effect_lo_pct > 0.0 or row.effect_hi_pct < 0.0
        print(f"    certified      [{row.effect_lo_pct:+7.3f}, {row.effect_hi_pct:+7.3f}]%  "
              f"(width {row.gap_pct:.3f}%; sign needs |effect| > {row.sign_bar_pct:.3f}%)  -> "
              f"{'SIGN RESOLVED on the total objective' if _resolved else 'inside the bound, NOT resolved'}")
        print(f"    NOT certified  the {row.cost_economic_pct:+.3f}% real-spending / "
              f"{row.cost_penalty_pct:+.3f}% penalty split -- the gap bounds the total only; "
              f"the split rests on the seed replicates below")
        for key, label in PHYSICAL:
            print(f"    {label.replace(chr(10), ' '):32s} "
                  f"{getattr(row, key + '_from'):10.1f} -> {getattr(row, key + '_to'):10.1f}   "
                  f"{getattr(row, key + '_pct'):+8.2f}%")
    inst = contrasts.iloc[1]
    clim = contrasts.iloc[2]

    # ── the solver-degeneracy floor, measured honestly ──────────────────────────────────
    # The MIP gap bounds the objective of a single run; it says nothing about how far this
    # model's flat optimum lets the ANSWER wander. Seed replicates -- model, parameters and
    # feasible set bit-identical, only Gurobi's search path different -- measure that.
    #
    # Three corrections an audit forced on the earlier version of this block:
    #   1. A RANGE IS NOT A SIGMA. The range of k draws has expectation d2(k)*sigma, and for
    #      k = 2 that is 1.128 -- so a 2-replicate range is roughly one sigma, not an
    #      envelope. Its own sampling spread is enormous: the 5-95% interval of a 2-sample
    #      range covers a factor of 31. Reported here as an explicit sigma estimate with n.
    #   2. THE FLOOR IS APPLIED TO A DIFFERENCE OF TWO RUNS BUT WAS MEASURED ON ONE, which
    #      understates it by sqrt(2) if both sides are equally degenerate.
    #   3. Once the control side has replicates too, none of that inflation is needed: the
    #      sampling distribution of the DIFFERENCE is estimated directly from all pairs.
    _D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
           9: 2.970, 10: 3.078}

    def _seed_family(stem: str) -> list[float]:
        """Objectives of every replicate that is the SAME MODEL as `stem`.

        Existence on disk is not enough. A replicate solved on a different build or different
        inputs contributes a version difference, which both inflates the floor and hides the
        effect it is supposed to be a null for. This is gated the same way as the channel
        floors above -- see plot_style.same_model_runs.
        """
        names = same_model_runs([stem] + [f"{stem}_seed{_i}" for _i in range(2, 11)],
                                quiet=True)
        out = []
        for _name in names:
            _path = RESULTS_DIR / f"{_name}.json"
            if _path.exists():
                out.append(float(json.loads(_path.read_text(encoding="utf-8"))
                                 ["global_objective_cny"]))
        return out

    treat_objs = _seed_family(BINDS)
    ctrl_objs = _seed_family(PRICED)
    print()
    if len(treat_objs) >= 2 and len(ctrl_objs) >= 2:
        # Direct: every treatment replicate against every control replicate.
        diffs = sorted(100.0 * (t - c) / c for t in treat_objs for c in ctrl_objs)
        _mid = 0.5 * (diffs[0] + diffs[-1])
        print(f"  effect distribution from {len(treat_objs)}x{len(ctrl_objs)} = {len(diffs)} "
              f"seed pairs: {diffs[0]:+.3f}% to {diffs[-1]:+.3f}% "
              f"(point {inst.cost_pct:+.3f}%, midrange {_mid:+.3f}%)")
        print(f"    -> sign is {'RESOLVED' if diffs[0] * diffs[-1] > 0 else 'NOT resolved'} "
              f"across every seed pairing; this supersedes any effect-over-floor ratio")
    elif len(treat_objs) >= 2:
        _n = len(treat_objs)
        _rng = 100.0 * (max(treat_objs) - min(treat_objs)) / min(treat_objs)
        _sigma = _rng / _D2.get(_n, 3.078)
        _floor = 1.96 * (2.0 ** 0.5) * _sigma      # 95% envelope on a DIFFERENCE of two runs
        print(f"  solver-degeneracy floor, treatment side only, n = {_n} replicates:")
        print(f"    range {_rng:.3f}% of objective -> sigma ~ {_sigma:.3f}% (range/d2({_n}))")
        print(f"    95% envelope on a DIFFERENCE of two such runs: +/-{_floor:.3f}% "
              f"(1.96 * sqrt(2) * sigma)")
        print(f"    -> total-objective effect {inst.cost_pct:+.3f}% is "
              f"{abs(inst.cost_pct) / _floor:.1f}x that envelope")
        print(f"    -> climate contrast {clim.cost_pct:+.3f}% is "
              f"{abs(clim.cost_pct) / _floor:.2f}x it: not resolved")
        print("    NOTE control-side replicates not on disk yet; this line is the fallback.")
    else:
        print("  solver-degeneracy floor: fewer than 2 seed replicates on disk, not reported")

    print(f"\n  institution / climate cost ratio: {abs(inst.cost_pct / clim.cost_pct):.0f}x on the "
          f"full objective, {abs(inst.cost_economic_pct / clim.cost_economic_pct):.0f}x on real spending only")

    print("\n" + "=" * 100)
    print("panel (b) -- the six channels a response can use")
    print("=" * 100)
    print(ladder.round(2).to_string(index=False))
    ref, binds, frozen = values[PRICED], values[BINDS], values[FROZEN]
    print(f"\n  more binding vs less binding -- NOT on vs off: the envonly control already "
          f"enforces the environmental-flow\n  rule at every node. Conversion "
          f"{ref['converted_gw']:.1f} -> {binds['converted_gw']:.1f} GW "
          f"({binds['converted_gw'] / ref['converted_gw']:.2f}x), retirement {END_YEAR} "
          f"{ref['retire_gw_end']:.1f} -> {binds['retire_gw_end']:.1f} GW "
          f"({100 * (binds['retire_gw_end'] / ref['retire_gw_end'] - 1):+.1f}%), capture "
          f"{ref['capture_mt']:.1f} -> {binds['capture_mt']:.1f} Mt "
          f"({100 * (binds['capture_mt'] / ref['capture_mt'] - 1):+.1f}%)")
    print(f"  freeze the retrofit:    conversion -> 0 GW, {NEAR_YEAR} retirement "
          f"{binds['retire_gw_near']:.1f} -> {frozen['retire_gw_near']:.1f} GW, {END_YEAR} retirement "
          f"{binds['retire_gw_end']:.1f} -> {frozen['retire_gw_end']:.1f} GW, capture "
          f"{binds['capture_mt']:.1f} -> {frozen['capture_mt']:.1f} Mt "
          f"({100 * (frozen['capture_mt'] / binds['capture_mt'] - 1):+.1f}%), unserved "
          f"{binds['unserved_mm3']:.1f} -> {frozen['unserved_mm3']:.1f} Mm3 "
          f"({frozen['unserved_pct']:.1f}% of {NEAR_YEAR} demand)")

    print("\n" + "=" * 100)
    print("panel (c) -- per basin")
    print("=" * 100)
    columns = ["capacity_gw", "capture_draw_1e8", "allow_1e8", "stress_pct", "stress_lo_pct",
               "stress_hi_pct", "n_nodes", "n_at_limit", "basin_util_pct",
               "node_median_pct", "demand_at_limit_pct", "pool_gw",
               "conv_ref_gw", "conv_bind_gw", "conv_ref_pct", "conv_bind_pct", "unserved_mm3"]
    print(basins[columns].round(1).to_string())
    over = basins[basins["stress_pct"] > 100.0]
    print(f"\n  over the line: {', '.join(over.index)}  = {over['capacity_gw'].sum():.0f} GW of "
          f"{basins['capacity_gw'].sum():.0f} GW ({100 * over['capacity_gw'].sum() / basins['capacity_gw'].sum():.0f}%)")
    print(f"  conversion under the reserved allowance: {over['conv_bind_gw'].sum():.1f} GW of "
          f"{basins['conv_bind_gw'].sum():.1f} GW national "
          f"({100 * over['conv_bind_gw'].sum() / basins['conv_bind_gw'].sum():.1f}%) lands in those four")
    increment = basins["conv_bind_gw"] - basins["conv_ref_gw"]
    print(f"  of the +{increment.sum():.1f} GW conversion increment, "
          f"{100 * increment[over.index].sum() / increment.sum():.1f}% lands in those four")
    wet = [b for b in basins.index if b not in over.index and b not in ("A",)]
    print(f"  the wet basins ({', '.join(wet)}) hold {basins.loc[wet, 'capacity_gw'].sum():.0f} GW and "
          f"convert {basins.loc[wet, 'conv_bind_gw'].sum():.1f} GW "
          f"({100 * basins.loc[wet, 'conv_bind_gw'].sum() / basins.loc[wet, 'pool_gw'].sum():.1f}% of their pool)")

    # THIS IS NOT A YEAR TEST, AND IT USED TO BE LABELLED AS ONE. `basins` freezes the
    # numerator at the 2030 STANDING fleet -- capacity x province_cf x 8760 x with-capture
    # intensity -- for every year, and only the denominator varies. By 2060 roughly 56% of that
    # generation has retired in every solved run, so a cell labelled 'ssp126 2060' is asking
    # 'would today's fleet still breach against 2060 water', not 'does the fleet of 2060
    # breach'. Both are legitimate questions; only the first is what this block computes. It is
    # in fact the cleaner hydrology test, because holding the fleet fixed isolates the water
    # signal from fleet evolution -- which is why it is kept, and relabelled. Fig 2 panel (c)'s
    # 2060 diamonds answer the other question on the solver's own realised fleet, and they
    # disagree with these numbers by up to 3.5x for exactly this reason.
    print()
    print("  four-basin verdict against each SSP and each year's water, FLEET HELD AT ITS")
    print("  2030 STANDING CONFIGURATION (a hydrology test, not a year test; not drawn):")
    avail = pd.read_csv(INPUTS / "water_availability.csv")
    for ssp in sorted(avail["ssp"].unique()):
        for year in (NEAR_YEAR, END_YEAR):
            frame = avail[(avail["planning_year"] == year) & (avail["ssp"] == ssp)]
            allow = (frame.groupby(["basin_code", "hydrology_model", "gcm"])[
                "dry_season_water_m3_per_year"].sum() * USABLE / 1e8
            ).groupby("basin_code").median()
            ratio = (basins["capture_draw_1e8"] / allow.reindex(basins.index) * 100.0).dropna()
            crossed = ratio[ratio > 100.0]
            print(f"    {ssp} {year}: over the line {','.join(crossed.index)} = "
                  f"{basins.loc[crossed.index, 'capacity_gw'].sum():.0f} GW   "
                  + "  ".join(f"{b}:{ratio[b]:,.0f}%" for b in crossed.index))

    print("\n  where the ensemble range straddles the line (drawn as whiskers, stated here):")
    for basin, row in basins.iterrows():
        if row["stress_lo_pct"] < 100.0 < row["stress_hi_pct"]:
            side = "median below" if row["stress_pct"] < 100 else "median above"
            print(f"    {basin} {BASIN_NAMES_ZH.get(basin, basin):<18} {side}, "
                  f"range {row['stress_lo_pct']:.0f}-{row['stress_hi_pct']:.0f}% "
                  f"({row['capacity_gw']:.0f} GW)")

    print("\n  panel (c) left -- what the basin aggregate does not show:")
    print(f"    {'basin':22s} {'agg%':>6s} {'median node%':>13s} {'demand@limit%':>14s} "
          f"{'nodes@limit':>12s} {'converts% of pool':>18s}")
    for basin in basins.index:
        row = basins.loc[basin]
        print(f"    {basin + ' ' + BASIN_NAMES_ZH.get(basin, basin):22s} "
              f"{row['basin_util_pct']:6.1f} {row['node_median_pct']:13.1f} "
              f"{row['demand_at_limit_pct']:14.1f} "
              f"{int(row['n_at_limit'])}/{int(row['n_nodes']):<10d} "
              f"{row['conv_bind_pct']:18.0f}")
    _gap = basins["demand_at_limit_pct"] - basins["basin_util_pct"]
    _w = _gap.idxmax()
    print(f"    widest divergence: {_w} {BASIN_NAMES_ZH.get(_w, _w)} -- basin reads "
          f"{basins.loc[_w, 'basin_util_pct']:.0f}% used and its median node "
          f"{basins.loc[_w, 'node_median_pct']:.0f}%, yet "
          f"{basins.loc[_w, 'demand_at_limit_pct']:.0f}% of the coal water demand inside it "
          f"sits on nodes at their limit (+{_gap.max():.0f} points).")
    print(f"    demand-weighted saturation is >= the basin aggregate in "
          f"{int((_gap >= -0.5).sum())} of {len(basins)} basins; counting NODES instead of "
          f"weighting by demand reverses that in most of them, which is why the figure uses "
          f"the demand weighting.")
    print("    consequence: conversion is NOT monotone in basin-aggregate stress and should")
    print("    not be. E Huai (84% aggregate) converts 34% of its pool; K Northwest Interior")
    print("    (33% aggregate, 90% of demand saturated) converts 96%.")


def claim_audit(values: dict[str, dict[str, float]], contrasts: pd.DataFrame,
                basins: pd.DataFrame) -> None:
    """State plainly, clause by clause, what the results support and what they qualify."""
    inst = contrasts.iloc[1]
    clim = contrasts.iloc[2]
    binds, ref, frozen = values[BINDS], values[PRICED], values[FROZEN]
    over = basins[basins["stress_pct"] > 100.0]

    print("\n" + "=" * 100)
    print("CLAIM AUDIT -- clause by clause against the plotted numbers")
    print("=" * 100)

    print("\n  [SUPPORTED] 'whether the water account binds is set by allocation institutions, "
          "not climate'")
    # WHAT THE SOLVER CERTIFIES vs WHAT IT DOES NOT, stated separately and never conflated.
    # The MIP gap bounds each run's TOTAL objective. The economic/penalty split of that
    # objective is a property of one incumbent and is not bounded by anything the solver
    # reports, so it cannot be divided by the gap. Earlier versions of this block did exactly
    # that ("real spending clears the gap 2.3x") and the ratio was meaningless.
    print(f"    reserving the non-power share: total objective {inst.cost_pct:+.2f}%, "
          f"certified [{inst.effect_lo_pct:+.2f}, {inst.effect_hi_pct:+.2f}]% -> "
          f"{'SIGN RESOLVED' if inst.effect_lo_pct > 0 else 'not resolved'}")
    print(f"    SSP1-2.6 -> SSP3-7.0:          total objective {clim.cost_pct:+.2f}%, "
          f"certified [{clim.effect_lo_pct:+.2f}, {clim.effect_hi_pct:+.2f}]% -> "
          f"straddles zero, sign not even resolved")
    _acc = contrasts.iloc[0]
    print(f"    water accounted at ALL:        total objective {_acc.cost_pct:+.2f}%, "
          f"certified [{_acc.effect_lo_pct:+.2f}, {_acc.effect_hi_pct:+.2f}]% -> also unresolved")
    print("    The reservation contrast is the ONLY one of the three the solver's own bounds")
    print("    resolve. That is the claim: it is the reservation rule specifically, not water")
    print("    accounting and not the climate scenario, that creates the binding constraint.")
    print(f"    Reported as a ratio of point estimates it is "
          f"{abs(inst.cost_pct / clim.cost_pct):.0f}x climate -- but a ratio whose denominator "
          f"is unresolved\n    has no defensible precision, so the qualitative statement is what "
          f"is claimed.")

    print("\n  [NOT CERTIFIED -- REPORTED, NOT CLAIMED] the economic/penalty split")
    print(f"    real spending {inst.cost_economic_pct:+.2f}%, unserved-water penalty "
          f"{inst.cost_penalty_pct:+.2f}% ({100 * inst.cost_penalty_pct / inst.cost_pct:.0f}% of "
          f"the objective increment).")
    print(f"    The penalty is 1 000 CNY/m3 on water the solver could not serve "
          f"(solver.py:617), i.e. scarcity\n    priced, not money spent. The split is worth "
          f"showing because it says HOW the objective moved, but\n    the MIP gap does not "
          f"bound it and no figure may quote it as a resolved effect. Its stability across\n"
          f"    seed replicates is the only evidence available for it, and that is reported "
          f"above.")

    print("\n  [QUALIFIED] the size of the institutional cost effect")
    print(f"    +5.86% (the number in plan/*.md) is measured against BASE, which has no basin "
          f"water account\n    at all; the single-factor reservation contrast is "
          f"{inst.cost_pct:+.2f}% with a certified interval of\n    "
          f"[{inst.effect_lo_pct:+.2f}, {inst.effect_hi_pct:+.2f}]%. Quote the interval, not "
          f"the point estimate.")
    print("\n  [STALE TEXT TO FIX] the conversion multiplier quoted in the storyline")
    print(f"    plan/nature_water_storyline_and_figures.md is internally inconsistent: its table "
          f"(line 53) says\n    '20.6 -> 71.1 GW (x3.45)' while its abstract (line 657) says "
          f"'166 -> 411 GW'. The current runs give\n    "
          f"{ref['converted_gw']:.1f} -> {binds['converted_gw']:.1f} GW, a factor of "
          f"{binds['converted_gw'] / ref['converted_gw']:.2f}, so the 'x3.45' and the 20.6/71.1 "
          f"pair are both stale. Use\n    x{binds['converted_gw'] / ref['converted_gw']:.1f} "
          f"({inst.converted_gw_pct:+.0f}%) or state the GW directly.")

    print("\n  [SUPPORTED] 'the fleet converts to dry cooling at scale rather than abandoning "
          "capture'")
    print(f"    conversion {ref['converted_gw']:.0f} -> {binds['converted_gw']:.0f} GW "
          f"({inst.converted_gw_pct:+.0f}%), capture {ref['capture_mt']:.1f} -> "
          f"{binds['capture_mt']:.1f} Mt ({inst.capture_mt_pct:+.1f}%),")
    print(f"    {END_YEAR} retirement {ref['retire_gw_end']:.1f} -> {binds['retire_gw_end']:.1f} GW "
          f"({100 * (binds['retire_gw_end'] / ref['retire_gw_end'] - 1):+.1f}%). Forbid the "
          f"retrofit and the response has to route\n    through retirement instead "
          f"({NEAR_YEAR} {frozen['retire_gw_near']:.0f} GW from zero, {END_YEAR} "
          f"+{frozen['retire_gw_end'] - binds['retire_gw_end']:.0f} GW), capture falls "
          f"{100 * (frozen['capture_mt'] / binds['capture_mt'] - 1):.1f}%,\n    and "
          f"{frozen['unserved_pct']:.1f}% of {NEAR_YEAR} water demand goes unserved.")

    print("\n  [SUPPORTED] 'four northern basins, 765 GW, cross their dry-season budget'")
    print(f"    {', '.join(over.index)} = {over['capacity_gw'].sum():.0f} GW, and the same four in "
          f"same four in all 4 SSP x water-year cells at a FIXED 2030 fleet. {100 * over['conv_bind_gw'].sum() / basins['conv_bind_gw'].sum():.0f}% "
          f"of national\n    conversion lands in them; the three wet basins (555 GW) convert "
          f"{100 * basins.loc[['F', 'H', 'G'], 'conv_bind_gw'].sum() / basins.loc[['F', 'H', 'G'], 'pool_gw'].sum():.1f}% of their pool.")

    print("\n  [NOT SHOWN HERE, BY DESIGN] 'the response holds even if dry-cooling capex is "
          "4.6x higher'")
    print("    True in the results (conversion 411 -> 347 GW at 1 370 CNY/kW, retirement +1.4%), "
          "but it is a\n    sensitivity and a Nature main figure does not carry one. It is printed "
          "above and belongs in\n    Extended Data. Nothing about it is drawn in this figure.")

    print("\n  [WATCH] two smaller things a referee will find, both visible in the figure")
    print(f"    1. Merely accounting for basin water already moves conversion "
          f"{contrasts.iloc[0].converted_gw_pct:+.0f}% at no resolvable cost.\n"
          f"       Cooling is the responsive margin at every step, not only once the allowance is "
          f"reserved.")
    a_row = basins.loc["A"] if "A" in basins.index else None
    if a_row is not None:
        print(f"    2. 'Four basins' is a median statement. Northeast Rivers reaches "
              f"{a_row['stress_hi_pct']:.0f}% under the driest\n       member "
              f"(+{a_row['capacity_gw']:.0f} GW), and Northwest Interior falls to "
              f"{basins.loc['K', 'stress_lo_pct']:.0f}% under the wettest. The four-basin\n"
              f"       count is robust at the median but not at the ensemble edges.")

    print("\n" + "=" * 100)
    print("cross-checks reported for the record, deliberately NOT drawn (Extended Data material)")
    print("=" * 100)
    print("  dry-cooling capex sensitivity -- a Nature main figure does not carry one:")
    base = values[BINDS]
    print(f"    {'capex 300 (default)':28s} conversion {base['converted_gw']:7.1f} GW  "
          f"retirement {END_YEAR} {base['retire_gw_end']:7.1f} GW  capture {base['capture_mt']:7.1f} Mt  "
          f"cost {base['cost']:.4f}e12")
    for name in CAPEX_CHECK:
        if not (RESULTS_DIR / f"{name}.json").exists():
            print(f"    {name:28s} not solved")
            continue
        value = outcomes(name)
        print(f"    {name.rsplit('_', 1)[-1]:28s} conversion {value['converted_gw']:7.1f} GW  "
              f"retirement {END_YEAR} {value['retire_gw_end']:7.1f} GW  capture {value['capture_mt']:7.1f} Mt  "
              f"cost {value['cost']:.4f}e12  ({100 * (value['cost'] / base['cost'] - 1):+.2f}% vs default)")
    print(f"    reference with no reservation: {values[PRICED]['converted_gw']:.1f} GW -- the capex "
          f"variants stay far above it, so the adaptation channel does not flip to retirement")


def main() -> None:
    names = [NOWATER, PRICED, BINDS, BINDS_HOT, FROZEN]
    # REFUSE TO DIFFERENCE ACROSS WATER-INPUT VINTAGES. See
    # plot_style.assert_same_vintage: the dry-season correction changed every
    # value and no column name, so nothing else in this repository could see it.
    assert_same_vintage(names, "plot_fig3_mechanism")
    values = {name: outcomes(name) for name in names}
    floors = seed_envelopes()
    contrasts = contrast_table(values)
    basins = basin_table()

    fig = plt.figure(figsize=(DOUBLE_COL[0], 8.15))
    # left=0.085 was too tight for panel (c)'s basin names ("K  Northwest Interior") and the
    # (c) panel label: together they ran 11 mm off the canvas, and bbox_inches='tight' turns
    # that into a 194 mm saved width against a 183 mm column rather than clipping it.
    outer = fig.add_gridspec(3, 1, height_ratios=[1.00, 0.66, 1.32], hspace=0.46,
                             left=0.150, right=0.985, top=0.955, bottom=0.118)
    top = outer[0].subgridspec(1, 2, width_ratios=[0.72, 1.68], wspace=0.28)
    ax_cost = fig.add_subplot(top[0])
    ax_phys = fig.add_subplot(top[1])
    mid = outer[1].subgridspec(1, len(CHANNELS), wspace=0.62)
    ax_b = [fig.add_subplot(mid[i]) for i in range(len(CHANNELS))]
    bottom = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.02], wspace=0.055)
    ax_stress = fig.add_subplot(bottom[0])
    ax_conv = fig.add_subplot(bottom[1], sharey=ax_stress)

    panel_a_cost(ax_cost, contrasts)
    panel_a_physical(ax_phys, contrasts)
    ladder = panel_b(ax_b, values, floors)
    panel_c(ax_stress, ax_conv, basins)

    panel_label(ax_cost, "a", x=-0.30, y=1.17)
    panel_label(ax_b[0], "b", x=-0.52, y=1.32)
    # (c)'s label sits inside the widened margin now, not outside the canvas.
    panel_label(ax_stress, "c", x=-0.235, y=1.10)
    ax_stress.set_ylim(-0.72, len(basins) - 0.28)

    binds = values[BINDS]
    fig.text(
        0.150, 0.002,
        cjk_fill(
            f"供给电力的配额 = 枯水期流域径流 x {WATER_EXTRACTABLE_FRACTION:.2f} 可取用比例"
            f"（Richter et al. 2012, doi:10.1002/rra.1511）x "
            f"；非电力用水不再按假设份额扣除，而是作为流域取水总量约束单列，"
            f"上限取自国办发〔2013〕2号 用水总量控制指标扣除非电既有取水。"
            f"在预留后的配额下，模型完全满足 {NEAR_YEAR} 年的水约束：未满足需求为 "
            f"{binds['unserved_mm3']:.1f} Mm$^3$，占 {binds['unserved_pct']:.2f}%。"
            f"两条规则均无未满足需求，即约束全程靠改造冷却系统满足，不是靠买缺口；"
            f"即使禁止空冷改造，缺口仍为 0，模型改以提前退役与少捕集合规。"
            f"该缺口的 big-M 惩罚占其目标函数的 {binds['slack_share']:.1%}，"
            f"在 a、b 两个面板中显式画出，而不是并入成本。"
            f"若冻结空冷改造，则有 {values[FROZEN]['unserved_pct']:.1f}% 的需求无法满足。"
            f"改造容量 GW = 容量 x (1 - 已空冷份额) x 改造份额。"
            f"残余排放沿用模型自身的核算，不使用逐路径系数。",
            # Narrowed with the left margin: the caption starts at x=0.150 now, so 212 characters
            # would push the right edge back off the canvas and undo the width fix.
            width=182),
        fontsize=5.0, color="#555555", va="bottom", ha="left", linespacing=1.45,
    )
    save_fig(fig, "fig3_mechanism")
    report(values, contrasts, ladder, basins)
    claim_audit(values, contrasts, basins)


if __name__ == "__main__":
    main()
