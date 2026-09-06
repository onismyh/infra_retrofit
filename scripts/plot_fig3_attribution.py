"""Fig 3 - where the uncertainty in a water-constrained coal transition actually comes from.

Panel (a) used to attribute outcomes to factors using relative system cost. That was invalid
and has been removed. Across the ensemble contrasts (annual vs dry season, CWatM vs WaterGAP2,
SSP1-2.6 vs SSP3-7.0, water-constrained or not) the objectives differ by less than the solver's
own optimality tolerance: the cost spread is ~0.4% against MIP gaps of 0.6-1.1%. A cost ranking
built on that is a ranking of solver noise. Cost still resolves ONE thing -- switching cooling
adaptation off costs +8.7%, which clears the gap -- and that is all it is used for.

The physical quantities are 1-2 orders clearer, so panel (a) attributes on quantities instead.
It also shows which quantities do NOT separate the factors: 2030 air-cooling conversion spans
1.3% to 100% of the main case across the five factors, while 2060 capture and retirement stay
inside +/-3% for every one of them. The fleet reaches the same end state by adjusting cooling,
not by retiring or capturing differently. That is a real result, not a gap in the figure.

  (a) effect of each factor on four physical outcomes, with the dropped cost metric explained
  (b) what happens when the water constraint actually binds (existing_withdrawal_share 0.85)
  (c) what basin bias correction did to the apparent hydrology-model spread, 20-member ensemble

Panel (c) carries a separate methodological point. Before bias correction the two hydrology
models disagree by a factor of 1.87 in the Hai basin and the cross-basin spread of their ratio
is 0.95; after correction that spread collapses to 0.08. An uncorrected multi-model ensemble
reports its own bias as a confidence interval.

Vintage note: only runs from 2026-08-10 or later carry `air_cooled_share` / `already_air_share`
(26 columns). Earlier runs predate the wet-to-dry conversion mechanism entirely and have a
different feasible set, so mixing them in would attribute a code change to hydrology or SSP.
Every scenario used here is vintage-checked by `_has_air_columns` and validity-checked by
`plot_style.scenario_validity`.

Usage:  python scripts/plot_fig3_attribution.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (  # noqa: E402
    apply_style,
    save_fig,
    panel_label,
    scenario_validity,
    require_valid_scenarios,
    ROOT,
    RESULTS_DIR,
    DOUBLE_COL,
    BASIN_NAMES_ZH,
    BASIN_ORDER,
)

apply_style()

INPUTS = ROOT / "inputs"
NEAR_YEAR = "2030"   # where the water constraint bites and cooling is being converted
END_YEAR = "2060"    # where retirement and capture have played out
ANCHOR = "WA_cwatm_126_dry"

# Each factor differs from the anchor in exactly one thing.
FACTORS: list[tuple[str, str, str]] = [
    ("Water constrained\nat all", "BASE", "constraint"),
    ("Cooling frozen\n(no adaptation)", "WA_cwatm_126_dry_noair", "adaptation"),
    ("Annual mean\ninstead of dry season", "WA_cwatm_126_annual", "accounting"),
    ("WaterGAP2\ninstead of CWatM", "WA_wgap_126_dry", "hydrology"),
    ("SSP3-7.0\ninstead of SSP1-2.6", "WA_cwatm_370_dry", "climate"),
]

# Panel (b): raising existing_withdrawal_share to 0.85 is what finally makes water bind.
# air1000/air1370 are air-retrofit capex sensitivities and are included only if solved.
BINDING: list[tuple[str, str]] = [
    ("Main case\nshare 0.60", ANCHOR),
    ("Water binding\nshare 0.85", "WA_cwatm_126_dry_wd085"),
    ("Binding, air\ncapex 1000", "WA_cwatm_126_dry_wd085_air1000"),
    ("Binding, air\ncapex 1370", "WA_cwatm_126_dry_wd085_air1370"),
    ("Binding,\ncooling frozen", "WA_cwatm_126_dry_wd085_noair"),
]

METRICS: list[tuple[str, str, str]] = [
    ("converted_gw", f"Air-cooling conversion, {NEAR_YEAR} (GW)", "#117733"),
    ("water_1e8", f"Water consumption, {NEAR_YEAR} (10$^8$ m$^3$)", "#4477AA"),
    ("capture_mt", f"CO$_2$ captured, {END_YEAR} (Mt)", "#CC3311"),
    ("retire_gw", f"Early retirement, {END_YEAR} (GW)", "#999999"),
]

GROUP_COLOUR = {
    "constraint": "#332288",
    "adaptation": "#117733",
    "accounting": "#DDAA33",
    "hydrology": "#4477AA",
    "climate": "#CC3311",
}
# Anything moving a quantity by less than this is not worth a physical interpretation.
FLAT_PCT = 5.0

def load(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} - run {name} first")
    return json.loads(path.read_text(encoding="utf-8"))


def _has_air_columns(name: str) -> bool:
    """Vintage gate: pre-2026-08-10 runs have 24 columns and no cooling-conversion mechanism.

    Their 2030 water use is 8272 Mm3 against 4806-5075 in the current model, i.e. a different
    feasible set. Silently filling the missing columns with zero would report a code change as
    a hydrology or SSP effect, which is precisely what this figure exists to rule out.
    """
    detail = RESULTS_DIR / name / "plant_detail.csv"
    if not detail.exists():
        return False
    columns = pd.read_csv(detail, nrows=0).columns
    return {"air_cooled_share", "already_air_share"}.issubset(columns)


def usable(names: list[str], allow_slack: bool = False) -> list[str]:
    """Scenarios that exist, are the current vintage, and are physically meaningful.

    Strict mode (allow_slack=False) delegates the validity gate to
    `plot_style.require_valid_scenarios`, which is the single place the
    infeasible/NaN-objective and >1%-slack failures are defined.
    """
    present = []
    for name in names:
        if not (RESULTS_DIR / f"{name}.json").exists():
            print(f"  [skip] {name}: not solved yet")
            continue
        if not _has_air_columns(name):
            print(f"  [skip] {name}: stale vintage, no air-cooling columns")
            continue
        present.append(name)

    if not allow_slack:
        return require_valid_scenarios(present)

    # The wd085 family leans ~3.5% on unserved water by construction -- that is the result
    # panel (b) is reporting, not a solver artefact -- so a slack-only failure is kept and
    # its unserved volume is drawn alongside. Every other failure mode still drops the run.
    keep = []
    for name in present:
        verdict = scenario_validity(name)
        if verdict["ok"] or "slack penalty" in verdict["reason"]:
            keep.append(name)
        else:
            print(f"  [skip] {name}: {verdict['reason']}")
    return keep
    return keep


def outcomes(name: str) -> dict[str, float]:
    data = load(name)
    detail = pd.read_csv(RESULTS_DIR / name / "plant_detail.csv")
    near = detail[detail["year"] == int(NEAR_YEAR)]
    end = detail[detail["year"] == int(END_YEAR)]
    slack_path = RESULTS_DIR / name / "slack_detail.csv"
    slack = pd.read_csv(slack_path) if slack_path.exists() else pd.DataFrame()

    def unserved(frame: pd.DataFrame, year: str) -> float:
        if frame.empty or "slack_value" not in frame.columns:
            return 0.0
        return float(frame[frame["year"] == int(year)]["slack_value"].sum())

    water_near = float(near["water_use_m3"].sum())
    short_near = unserved(slack, NEAR_YEAR)
    return {
        "cost_Ttn": data["global_objective_cny"] / 1e12,
        # air_cooled_share is progress over the hub's remaining wet units, not a capacity
        # share, so it has to be scaled by the fraction that was still wet.
        "converted_gw": float(
            (near["capacity_mw"] * (1.0 - near["already_air_share"]) * near["air_cooled_share"]).sum()
        ) / 1000.0,
        "water_1e8": water_near / 1e8,
        "unserved_1e8": short_near / 1e8,
        # Demand the solver could not serve, as a share of what it wanted -- the honest
        # denominator for a run that buys its way out with slack.
        "unserved_pct": 100.0 * short_near / (water_near + short_near) if water_near + short_near > 0 else 0.0,
        "capture_mt": float(end["captured_mt"].sum()),
        "retire_gw": float((end["capacity_mw"] * end["share_retire"]).sum()) / 1000.0,
        "water_end_1e8": float(end["water_use_m3"].sum()) / 1e8,
        "mip_gap": float(data["solver_quality"].get("mip_gap") or 0.0),
        "slack_share": scenario_validity(name)["slack_share"],
    }


# -- panel (a): physical attribution ----------------------------------------
def build_table() -> tuple[pd.DataFrame, dict[str, float]]:
    anchor = outcomes(ANCHOR)
    rows = []
    for label, name, group in FACTORS:
        if name not in usable([name]):
            continue
        value = outcomes(name)
        row = {"label": label, "scenario": name, "group": group,
               "d_cost_pct": (value["cost_Ttn"] - anchor["cost_Ttn"]) / anchor["cost_Ttn"] * 100,
               "mip_gap": max(value["mip_gap"], anchor["mip_gap"])}
        for key, _, _ in METRICS:
            row[key] = value[key]
            base = anchor[key]
            row[f"rel_{key}"] = (value[key] - base) / base * 100 if base else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    # Order by the metric that actually moves, so the reader sees a ranking not a jumble.
    frame = frame.reindex(frame["rel_converted_gw"].abs().sort_values().index).reset_index(drop=True)
    return frame, anchor


def panel_a(ax, frame: pd.DataFrame, anchor: dict[str, float]) -> float:
    """Grouped relative change on four physical quantities, anchored on the main case."""
    y = np.arange(len(frame))
    n = len(METRICS)
    height = 0.78 / n
    tolerance = float(frame["mip_gap"].max()) * 100

    for index, (key, label, colour) in enumerate(METRICS):
        offset = (n - 1 - index - (n - 1) / 2) * height
        values = frame[f"rel_{key}"].to_numpy(dtype=float)
        ax.barh(y + offset, values, height * 0.92, color=colour, label=label,
                edgecolor="white", linewidth=0.25)
        for yi, value in zip(y, values):
            if not np.isfinite(value):
                continue
            ax.text(value + (1.6 if value >= 0 else -1.6), yi + offset,
                    f"{value:+.0f}%" if abs(value) >= 1 else "~0",
                    va="center", ha="left" if value >= 0 else "right", fontsize=5.1,
                    color="#666666" if abs(value) < FLAT_PCT else "black")

    # The band the cost metric could never escape, drawn on the same axis so the reader can
    # see that the physical bars clear it by an order of magnitude.
    ax.axvspan(-tolerance, tolerance, color="#000000", alpha=0.10, zorder=0)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"], fontsize=6.0)
    ax.set_xlabel(f"Change vs main case ({ANCHOR.replace('WA_', '').replace('_', ' ')}), %")
    ax.set_title("Cooling separates the factors;\nretirement and capture do not")
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", bbox_to_anchor=(0.0, 0.30),
              handlelength=1.0, labelspacing=0.25, borderpad=0.2)

    finite = frame[[f"rel_{k}" for k, _, _ in METRICS]].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = float(np.min(finite)), float(np.max(finite))
    pad = max((hi - lo) * 0.22, 12.0)
    ax.set_xlim(lo - pad, hi + pad * 0.9)
    ax.set_ylim(-0.62, len(frame) - 0.38)
    cost_note = textwrap.fill(
        f"Cost was dropped as the attribution metric: across these factors it spans only "
        f"{frame['d_cost_pct'].abs().min():.2f}-{frame['d_cost_pct'].abs().max():.2f}%, against a "
        f"{tolerance:.2f}% MIP gap (grey band). Only 'cooling frozen' "
        f"({frame.loc[frame['group'] == 'adaptation', 'd_cost_pct'].max():+.1f}%) clears the solver "
        f"tolerance; every other cost difference is solver noise.",
        width=86,
    )
    ax.text(0.0, -0.20, cost_note, transform=ax.transAxes, fontsize=5.2,
            va="top", ha="left", color="#444444", linespacing=1.5)
    return tolerance


# -- panel (b): what a binding water constraint does ------------------------
def binding_table() -> pd.DataFrame:
    rows = []
    for label, name in BINDING:
        if name not in usable([name], allow_slack=True):
            continue
        value = outcomes(name)
        rows.append({"label": label, "scenario": name,
                     "converted_gw": value["converted_gw"],
                     "water_1e8": value["water_1e8"],
                     "water_end_1e8": value["water_end_1e8"],
                     "retire_gw": value["retire_gw"],
                     "unserved_1e8": value["unserved_1e8"],
                     "unserved_pct": value["unserved_pct"],
                     "cost_Ttn": value["cost_Ttn"],
                     "slack_share": value["slack_share"]})
    return pd.DataFrame(rows)


def panel_b(ax, frame: pd.DataFrame) -> None:
    """Conversion GW against water consumption, with unserved demand drawn on top.

    Conversion (hundreds of GW) and water (tens of 10^8 m3) cannot share one scale without
    one of them collapsing into the axis, so water gets a twin axis. Both start at zero and
    each axis is coloured to its series.
    """
    x = np.arange(len(frame))
    width = 0.36
    green, blue, red = "#117733", "#4477AA", "#CC3311"
    conv = frame["converted_gw"].to_numpy(dtype=float)
    served = frame["water_1e8"].to_numpy(dtype=float)
    short = frame["unserved_1e8"].to_numpy(dtype=float)

    bar_conv = ax.bar(x - width / 2, conv, width, color=green)
    ax.set_ylim(0, max(conv.max(), 1.0) * 1.22)
    ax.set_ylabel(f"Air-cooling conversion, {NEAR_YEAR} (GW)", color=green, fontsize=6.6)
    ax.tick_params(axis="y", colors=green, labelsize=6.2)

    twin = ax.twinx()
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(blue)
    bar_water = twin.bar(x + width / 2, served, width, color=blue)
    # Unserved demand stacked on the water bar: without it, the frozen-cooling run looks
    # like it simply uses less water, when in fact it fails to supply what it needs.
    bar_short = twin.bar(x + width / 2, short, width, bottom=served, color=red,
                         hatch="////", edgecolor="white", linewidth=0.3)
    twin.set_ylim(0, max((served + short).max(), 1.0) * 1.22)
    twin.set_ylabel(f"Water, {NEAR_YEAR} (10$^8$ m$^3$)", color=blue, fontsize=6.6,
                    labelpad=1.5)
    twin.tick_params(axis="y", colors=blue, labelsize=6.2, pad=1.0)

    head_c = ax.get_ylim()[1] * 0.02
    head_w = twin.get_ylim()[1] * 0.02
    for xi in range(len(frame)):
        ax.text(xi - width / 2, conv[xi] + head_c, f"{conv[xi]:.0f}",
                ha="center", va="bottom", fontsize=5.8, color=green)
        twin.text(xi + width / 2, served[xi] + short[xi] + head_w, f"{served[xi]:.0f}",
                  ha="center", va="bottom", fontsize=5.8, color=blue)
        pct = frame["unserved_pct"].iloc[xi]
        if pct > 0.5:
            twin.text(xi + width / 2, served[xi] + short[xi] + head_w * 3.6,
                      f"{pct:.0f}%\nshort", ha="center", va="bottom", fontsize=4.9,
                      color=red, fontweight="bold", linespacing=1.05)

    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"], fontsize=5.4)
    ax.set_xlim(-0.62, len(frame) - 0.38)
    ax.set_title("Raising the withdrawal share\nis what makes water bind")
    ax.legend([bar_conv, bar_water, bar_short],
              ["Conversion (GW)", "Water consumed", "Water unserved"],
              frameon=False, fontsize=5.4, loc="upper left", ncol=1,
              handlelength=1.0, labelspacing=0.22, borderpad=0.2)

    frozen = frame[frame["scenario"].str.endswith("noair")]
    reference = frame[frame["scenario"] == "WA_cwatm_126_dry_wd085"]
    if not frozen.empty and not reference.empty:
        row = frozen.iloc[0]
        # Compared against wd085, not the main case: wd085_noair differs from wd085 in
        # cooling alone, whereas the main case also differs in withdrawal share, and
        # attributing the sum of both changes to cooling would overstate it.
        delta = (row["cost_Ttn"] / reference.iloc[0]["cost_Ttn"] - 1) * 100
        pending = [name for _, name in BINDING if not (RESULTS_DIR / f"{name}.json").exists()]
        tail = f" Not yet solved: {', '.join(pending)}." if pending else ""
        note = textwrap.fill(
            f"Freezing cooling at the same 0.85 withdrawal share costs +{delta:.0f}%, but that "
            f"is mostly slack penalty ({row['slack_share']:.0%} of the objective), not "
            f"economics: it leaves {row['unserved_pct']:.0f}% of {NEAR_YEAR} water demand "
            f"unserved.{tail}",
            width=62,
        )
        ax.text(0.0, -0.20, note, transform=ax.transAxes, fontsize=5.2,
                va="top", ha="left", color=red, linespacing=1.5)


# -- panel (c): bias correction over the 20-member ensemble ------------------
def bias_table() -> pd.DataFrame:
    """Per-basin runoff ratio between the two hydrology models, per GCM, before and after.

    The table now carries five GCMs per model per SSP (20 members), so the ratio gets a
    range across GCMs rather than a single line -- which is what separates genuine climate
    spread from the systematic wet bias the correction removes.
    """
    frame = pd.read_csv(INPUTS / "water_availability.csv")
    frame = frame[(frame["planning_year"] == int(NEAR_YEAR)) & (frame["ssp"] == "ssp126")]
    frame = frame.assign(corrected=frame["local_runoff_m3_per_year"] * frame["bias_factor"])
    grouped = frame.groupby(["hydrology_model", "gcm", "basin_code"]).agg(
        raw=("local_runoff_m3_per_year", "sum"),
        corrected=("corrected", "sum"),
    ).reset_index()

    models = sorted(grouped["hydrology_model"].unique())
    if len(models) < 2:
        raise ValueError(f"need two hydrology models to show a spread, found {models}")
    cw = next(m for m in models if "cwatm" in m.lower())
    wg = next(m for m in models if "watergap" in m.lower())

    index = ["gcm", "basin_code"]
    left = grouped[grouped["hydrology_model"] == wg].set_index(index)
    right = grouped[grouped["hydrology_model"] == cw].set_index(index)
    ratio = pd.DataFrame({
        "before": left["raw"] / right["raw"],
        "after": left["corrected"] / right["corrected"],
    }).reset_index()
    return ratio


def panel_c(ax) -> pd.DataFrame:
    ratio = bias_table()
    stats = ratio.groupby("basin_code")[["before", "after"]].agg(["min", "mean", "max"])
    order = [b for b in BASIN_ORDER if b in stats.index]
    stats = stats.reindex(order)
    n_gcm = int(ratio["gcm"].nunique())

    y = np.arange(len(order))[::-1]
    for yi, before, after in zip(y, stats[("before", "mean")], stats[("after", "mean")]):
        ax.plot([before, after], [yi, yi], color="#BBBBBB", lw=0.9, zorder=1)
    for key, colour, label in (("before", "#CC3311", "Before bias correction"),
                              ("after", "#117733", "After")):
        mean = stats[(key, "mean")].to_numpy(dtype=float)
        lo = mean - stats[(key, "min")].to_numpy(dtype=float)
        hi = stats[(key, "max")].to_numpy(dtype=float) - mean
        ax.errorbar(mean, y, xerr=[lo, hi], fmt="o", ms=4.2, color=colour,
                    ecolor=colour, elinewidth=0.8, capsize=1.6, zorder=3, label=label)
    ax.axvline(1.0, color="black", lw=0.8, ls="--")

    worst = int(np.argmax(np.abs(np.log(stats[("before", "mean")].to_numpy(dtype=float)))))
    ax.text(stats[("before", "mean")].iloc[worst], y[worst] + 0.34,
            f"{stats[('before', 'mean')].iloc[worst]:.2f}x",
            fontsize=6.3, color="#CC3311", fontweight="bold", ha="center", va="bottom")

    low = float(min(stats[("before", "min")].min(), stats[("after", "min")].min()))
    high = float(max(stats[("before", "max")].max(), stats[("after", "max")].max()))
    span = high - low
    ax.set_xlim(low - 0.06 * span, high + 0.08 * span)
    ax.set_yticks(y)
    ax.set_yticklabels([BASIN_NAMES_ZH.get(b, b) for b in order], fontsize=6.3)
    ax.set_xlabel("WaterGAP2 runoff / CWatM runoff")
    ax.set_title("Model 'uncertainty' was mostly bias")
    ax.legend(frameon=False, fontsize=6.0, loc="lower right", bbox_to_anchor=(1.0, 0.06))
    ax.text(0.98, 0.005, f"bars: range over {n_gcm} GCMs", transform=ax.transAxes,
            fontsize=5.4, ha="right", va="bottom", color="#666666")

    return pd.DataFrame({
        "basin": order,
        "before": stats[("before", "mean")].to_numpy(dtype=float),
        "after": stats[("after", "mean")].to_numpy(dtype=float),
        "before_min": stats[("before", "min")].to_numpy(dtype=float),
        "before_max": stats[("before", "max")].to_numpy(dtype=float),
    })


def report_factorial() -> None:
    """State plainly whether the 2x2x2 hydrology x SSP x accounting design is complete."""
    cells = {
        ("cwatm", "126", "dry"): "WA_cwatm_126_dry",
        ("cwatm", "370", "dry"): "WA_cwatm_370_dry",
        ("wgap", "126", "dry"): "WA_wgap_126_dry",
        ("wgap", "370", "dry"): "WA_wgap_370_dry",
        ("cwatm", "126", "annual"): "WA_cwatm_126_annual",
        ("cwatm", "370", "annual"): "WA_cwatm_370_annual",
        ("wgap", "126", "annual"): "WA_wgap_126_annual",
        ("wgap", "370", "annual"): "WA_wgap_370_annual",
    }
    print("\nfactorial completeness (hydrology x SSP x accounting)")
    present = 0
    for (model, ssp, season), name in cells.items():
        exists = (RESULTS_DIR / f"{name}.json").exists()
        current = exists and _has_air_columns(name)
        present += current
        state = "usable" if current else ("STALE vintage (air-blind)" if exists else "not solved")
        print(f"  {model:6s} ssp{ssp:4s} {season:7s} {name:24s} {state}")
    print(f"  -> {present}/8 cells usable; the factorial is "
          f"{'complete' if present == 8 else 'INCOMPLETE and is not presented as one'}")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL[0] * 1.42, 3.9),
                             gridspec_kw={"width_ratios": [1.20, 1.05, 1.0]})
    print("panel (a) scenario screening")
    frame, anchor = build_table()
    tolerance = panel_a(axes[0], frame, anchor)
    print("panel (b) scenario screening")
    binding = binding_table()
    panel_b(axes[1], binding)
    table_c = panel_c(axes[2])
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    # Wider gutter: panel (b) carries a right-hand twin axis whose label would otherwise
    # run into panel (c)'s basin names.
    fig.subplots_adjust(wspace=0.42, bottom=0.24, top=0.86, left=0.105, right=0.985)
    save_fig(fig, "fig3_attribution")

    print(f"\npanel a -- physical outcomes vs {ANCHOR}")
    print(f"  anchor: conversion {anchor['converted_gw']:.1f} GW, water {anchor['water_1e8']:.1f}e8 m3 "
          f"({NEAR_YEAR}); capture {anchor['capture_mt']:.1f} Mt, retirement {anchor['retire_gw']:.1f} GW "
          f"({END_YEAR})")
    columns = ["label", "group", "d_cost_pct"] + [f"rel_{k}" for k, _, _ in METRICS]
    print(frame[columns].round(2).to_string(index=False))
    print(f"\n  MIP gap across these runs: +/-{tolerance:.2f}% -- the cost spread is "
          f"{frame['d_cost_pct'].abs().min():.2f}-{frame['d_cost_pct'].abs().max():.2f}%")
    for key, label, _ in METRICS:
        rel = frame[f"rel_{key}"].abs()
        verdict = "separates the factors" if rel.max() > FLAT_PCT else "does NOT separate the factors"
        print(f"    {label:44s} spans {rel.min():5.1f}-{rel.max():6.1f}%  -> {verdict}")

    print("\npanel b -- binding water constraint")
    print(binding[["label", "converted_gw", "water_1e8", "unserved_1e8", "unserved_pct",
                   "retire_gw", "cost_Ttn", "slack_share"]].round(2).to_string(index=False))

    print("\npanel c -- WaterGAP2 / CWatM runoff ratio by basin (GCM mean, raw range)")
    print(table_c.round(3).to_string(index=False))
    spread = lambda col: table_c[col].max() - table_c[col].min()  # noqa: E731
    print(f"\n  cross-basin spread   before {spread('before'):.2f}   after {spread('after'):.2f}")

    report_factorial()


if __name__ == "__main__":
    main()
