"""Water figures for the Nature Water framing.

Replaces `main_fig6_water_network`, which compared BASE against `WA_grid_200km` — two runs
that differ only in whether water is *priced*, both drawing on SSP1-2.6, so it showed no
climate signal at all.

These panels are built on the rebuilt water inputs: ISIMIP `qtot` (locally generated runoff)
budgeted per province, rather than `dis` (routed discharge) summed node by node, which
counted the same water tens of times and left the constraint unable to bind.

  water_fig1_supply_demand — basin/province supply vs the water a full-CCS fleet would need,
                             annual mean against dry season, SSP1-2.6 against SSP3-7.0
  water_fig2_constraint    — what the constraint does to pathways, cost and CCS deployment

Usage:
    python scripts/plot_water_nature.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plot_style import (
    apply_style, save_fig, panel_label, RESULTS_DIR, ROOT,
    CN_TO_EN, PATHWAY_COLORS, PATHWAY_LABELS, DOUBLE_COL,
    require_valid_scenarios,
)
from coal_retrofit.optimization.scenario import OptimizationAssumptions

apply_style()

INPUTS = ROOT / "inputs"
EXTRACTABLE = 0.20          # environmental-flow rule: 20% of renewable runoff is usable
YEARS = [2030, 2040, 2050, 2060]

MEMBERS = {
    "cwatm|gfdl-esm4|ssp126": ("CWatM", "SSP1-2.6"),
    "cwatm|gfdl-esm4|ssp370": ("CWatM", "SSP3-7.0"),
    "watergap2-2e|gfdl-esm4|ssp126": ("WaterGAP2", "SSP1-2.6"),
    "watergap2-2e|gfdl-esm4|ssp370": ("WaterGAP2", "SSP3-7.0"),
}
SSP_COLORS = {"SSP1-2.6": "#4477AA", "SSP3-7.0": "#CC3311"}


# ── Data ──────────────────────────────────────────────────────────────────────

def _fleet_water_demand_by_province() -> pd.Series:
    """Water a fully CCS-retrofitted fleet would consume, m3/yr by province (English)."""
    assumptions = OptimizationAssumptions()
    plants = pd.read_csv(INPUTS / "plants.csv")
    cf = plants["province_mode"].map(lambda p: assumptions.province_cf(str(p)))
    generation = plants["total_capacity_mw"].astype(float) * cf * 8760.0
    demand = (
        generation
        * plants["weighted_water_intensity_m3_per_mwh"].astype(float)
        * assumptions.ccs_water_multiplier
        * 1.15  # retrofit CF boost
    )
    return demand.groupby(plants["province_mode"]).sum()


def _supply_by_province(year: int) -> pd.DataFrame:
    """Usable water by province and climate member, annual mean and dry season (m3/yr)."""
    availability = pd.read_csv(INPUTS / "water_availability.csv")
    nodes = pd.read_csv(INPUTS / "water_nodes.csv")
    node_province = dict(zip(nodes["water_node_id"], nodes["province_name"]))
    frame = availability[availability["planning_year"].astype(int) == year].copy()
    frame["province"] = frame["water_node_id"].map(node_province).map(CN_TO_EN)
    frame = frame.dropna(subset=["province"])
    grouped = frame.groupby(["scenario_id", "province"], as_index=False).agg(
        annual=("available_water_m3_per_year", "sum"),
        dry=("dry_season_water_m3_per_year", "sum"),
    )
    grouped[["annual", "dry"]] *= EXTRACTABLE
    return grouped


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def water_fig1_supply_demand() -> None:
    demand = _fleet_water_demand_by_province()
    supply = _supply_by_province(2050)

    stress = supply.copy()
    stress["demand"] = stress["province"].map(demand).fillna(0.0)
    stress = stress[stress["demand"] > 0]
    stress["ratio_annual"] = stress["demand"] / stress["annual"].replace(0, np.nan) * 100
    stress["ratio_dry"] = stress["demand"] / stress["dry"].replace(0, np.nan) * 100

    median_dry = stress.groupby("province")["ratio_dry"].median().dropna()
    order = median_dry.sort_values(ascending=False).index.tolist()[:16]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL[0], 4.4),
                                     gridspec_kw={"width_ratios": [1.15, 1.0]})

    # (a) demand-to-supply ratio, annual vs dry season, one marker per climate member
    y = np.arange(len(order))
    ax_a.axvline(100, color="#CC3311", lw=1.0, ls="--", zorder=1)
    for offset, (col, label, marker) in enumerate(
        [("ratio_annual", "Annual mean", "o"), ("ratio_dry", "Dry season (3-month low)", "D")]
    ):
        for member, (hydro, ssp) in MEMBERS.items():
            sub = stress[stress["scenario_id"] == member].set_index("province")
            values = [sub[col].get(p, np.nan) for p in order]
            ax_a.scatter(values, y + (offset - 0.5) * 0.32,
                         s=16 if offset else 12, marker=marker,
                         facecolor=SSP_COLORS[ssp] if offset else "none",
                         edgecolor=SSP_COLORS[ssp], linewidth=0.8,
                         alpha=0.9, zorder=3,
                         label=f"{label} · {ssp}" if hydro == "CWatM" else None)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(order, fontsize=6.5)
    ax_a.invert_yaxis()
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Full-CCS fleet water demand as % of usable supply (2050)")
    ax_a.set_xlim(0.5, 2000)
    ax_a.text(105, len(order) - 0.4, "supply exhausted", fontsize=6, color="#CC3311", ha="left")
    ax_a.legend(loc="lower right", fontsize=6, frameon=False)
    ax_a.spines[["top", "right"]].set_visible(False)
    panel_label(ax_a, "a")

    # (b) how many provinces cross 100%, by year and member
    counts = []
    for year in YEARS:
        year_supply = _supply_by_province(year)
        year_supply["demand"] = year_supply["province"].map(demand).fillna(0.0)
        year_supply = year_supply[year_supply["demand"] > 0]
        for member, (hydro, ssp) in MEMBERS.items():
            sub = year_supply[year_supply["scenario_id"] == member]
            for col, season in [("annual", "Annual mean"), ("dry", "Dry season")]:
                over = int((sub["demand"] > sub[col]).sum())
                capacity_share = float(
                    sub.loc[sub["demand"] > sub[col], "demand"].sum() / sub["demand"].sum() * 100
                ) if sub["demand"].sum() > 0 else 0.0
                counts.append({"year": year, "member": member, "hydro": hydro, "ssp": ssp,
                               "season": season, "n_over": over, "share": capacity_share})
    counts = pd.DataFrame(counts)

    for season, ls in [("Annual mean", ":"), ("Dry season", "-")]:
        for ssp, colour in SSP_COLORS.items():
            sub = counts[(counts["season"] == season) & (counts["ssp"] == ssp)]
            grouped = sub.groupby("year")["share"]
            ax_b.plot(grouped.mean().index, grouped.mean().values, ls=ls, color=colour, lw=1.6,
                      marker="o" if season == "Dry season" else "s", markersize=4,
                      label=f"{season} · {ssp}")
            ax_b.fill_between(grouped.min().index, grouped.min().values, grouped.max().values,
                              color=colour, alpha=0.12, lw=0)
    ax_b.set_xticks(YEARS)
    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("Coal capacity in water-short provinces (%)")
    ax_b.set_title("Share of the fleet whose CCS water demand\nexceeds local usable supply", fontsize=7.5)
    ax_b.legend(fontsize=6, frameon=False, loc="center left")
    ax_b.spines[["top", "right"]].set_visible(False)
    panel_label(ax_b, "b")

    fig.tight_layout()
    save_fig(fig, "water_fig1_supply_demand")


# ── Figure 2 ──────────────────────────────────────────────────────────────────

WATER_RUNS = [
    ("BASE", "No water\nconstraint", "#333333", "none"),
    ("WA_cwatm_126_annual", "CWatM\nSSP1-2.6", "#88CCEE", "annual"),
    ("WA_cwatm_370_annual", "CWatM\nSSP3-7.0", "#EE99AA", "annual"),
    ("WA_wgap_126_annual", "WaterGAP2\nSSP1-2.6", "#88CCEE", "annual"),
    ("WA_wgap_370_annual", "WaterGAP2\nSSP3-7.0", "#EE99AA", "annual"),
    ("WA_cwatm_126_dry", "CWatM\nSSP1-2.6", "#4477AA", "dry"),
    ("WA_cwatm_370_dry", "CWatM\nSSP3-7.0", "#CC3311", "dry"),
    ("WA_wgap_126_dry", "WaterGAP2\nSSP1-2.6", "#117733", "dry"),
    ("WA_wgap_370_dry", "WaterGAP2\nSSP3-7.0", "#DDAA33", "dry"),
]


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def water_fig2_constraint_effect() -> None:
    valid = set(require_valid_scenarios([r[0] for r in WATER_RUNS]))
    runs_all = [r for r in WATER_RUNS if r[0] in valid and _load(r[0])]
    # Annual-mean members sit on top of the unconstrained run, so panels a and b show the
    # dry-season members; panel c keeps every member to make that flatness visible.
    runs = [r for r in runs_all if r[3] in ("none", "dry")]
    if len(runs) < 2:
        print("  [skip] water_fig2 needs at least two solved water runs")
        return

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL[0], 3.4))
    ax_a, ax_b, ax_c = axes

    # (a) capture fleet (CCS + BECCS) share over time
    for key, label, colour, season in runs:
        payload = _load(key)
        shares = [
            float(payload["years"][str(y)]["pathway_shares"].get("ccs", 0))
            + float(payload["years"][str(y)]["pathway_shares"].get("beccs", 0))
            for y in YEARS
        ]
        ax_a.plot(YEARS, [s * 100 for s in shares], marker="o", markersize=4,
                  color=colour, lw=2.2 if key == "BASE" else 1.5,
                  ls="-" if key == "BASE" else "--", label=label.replace("\n", " "))
    ax_a.set_xticks(YEARS)
    ax_a.set_ylabel("Capture fleet, CCS + BECCS (% of generation)")
    ax_a.set_xlabel("Year")
    ax_a.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax_a.spines[["top", "right"]].set_visible(False)
    panel_label(ax_a, "a")

    # (b) 2060 pathway mix
    paths = ["unabated", "retire", "ccs", "biomass", "beccs", "ammonia"]
    x = np.arange(len(runs))
    bottom = np.zeros(len(runs))
    for pathway in paths:
        values = np.array([
            float(_load(k)["years"]["2060"]["pathway_shares"].get(pathway, 0)) * 100
            for k, _, _, _ in runs
        ])
        if values.max() < 0.05:
            continue
        ax_b.bar(x, values, bottom=bottom, color=PATHWAY_COLORS.get(pathway, "#999999"),
                 label=PATHWAY_LABELS.get(pathway, pathway), width=0.68, linewidth=0)
        bottom += values
    x = np.arange(len(runs))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([lbl for _, lbl, _, _ in runs], fontsize=5.5, rotation=0)
    ax_b.set_ylabel("2060 generation share (%)")
    ax_b.legend(fontsize=5.5, frameon=False, ncol=2, loc="lower center")
    ax_b.spines[["top", "right"]].set_visible(False)
    panel_label(ax_b, "b")

    # (c) system cost premium vs the unconstrained run
    base_objective = float(_load("BASE")["global_objective_cny"])
    # Every member, so the flatness of the annual-mean runs is visible next to the
    # dry-season ones: on annual means the constraint is effectively absent.
    cost_runs = [r for r in runs_all if r[0] != "BASE"]
    xc = np.arange(len(cost_runs))
    premiums = [(float(_load(k)["global_objective_cny"]) / base_objective - 1) * 100
                for k, _, _, _ in cost_runs]
    ax_c.bar(xc, premiums, color=[c for _, _, c, _ in cost_runs], width=0.72, linewidth=0)
    for xi, value in zip(xc, premiums):
        ax_c.text(xi, value + 0.4, f"{value:+.1f}", ha="center", fontsize=5.2)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([f"{lbl.replace(chr(10), " ")}\n{season}" for _, lbl, _, season in cost_runs],
                         fontsize=4.4, rotation=90)
    ax_c.set_ylabel("System cost vs unconstrained (%)")
    ax_c.axhline(0, color="#444444", lw=0.8)
    ax_c.spines[["top", "right"]].set_visible(False)
    panel_label(ax_c, "c")

    fig.tight_layout()
    save_fig(fig, "water_fig2_constraint_effect")


def main() -> None:
    print("Generating Nature Water figures ...")
    water_fig1_supply_demand()
    water_fig2_constraint_effect()
    print(f"Saved to {RESULTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
