"""Fig 3 for the Nature Water framing: the water-carbon frontier by shadow pricing.

A MIP has no usable constraint duals, so the shadow price of water is recovered
parametrically: the model is re-solved with a charge added to every m3 delivered to a
plant, swept from 0 to 1000 CNY/m3 with the availability constraint switched off. Each
solve is a point on the Pareto frontier between water use and abatement, and the adder at
that point IS the shadow price of water there.

The climate-constrained runs are then placed on the frontier by their realised water use,
which reads off the shadow price their availability constraint implies.

  (a) frontier: system water use against abatement, annotated with the shadow price
  (b) technology substitution along the frontier
  (c) the shadow price implied by each climate scenario

Usage:
    python scripts/plot_water_fig3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plot_style import (
    apply_style, save_fig, panel_label, RESULTS_DIR, DOUBLE_COL,
    PATHWAY_COLORS, PATHWAY_LABELS, residual_emissions_mt, baseline_emissions_mt,
)

apply_style()

YEARS = [2030, 2040, 2050, 2060]
FRONTIER_YEAR = 2060

# (scenario, water price adder CNY/m3)
SWEEP = [("BASE", 0.0)] + [
    (f"WP_{v:04d}", float(v)) for v in [2, 5, 10, 20, 50, 100, 200, 500, 1000]
]

CLIMATE_RUNS = [
    ("WA_cwatm_126_dry", "CWatM SSP1-2.6 dry", "#4477AA"),
    ("WA_cwatm_370_dry", "CWatM SSP3-7.0 dry", "#CC3311"),
    ("WA_wgap_126_dry", "WaterGAP2 SSP1-2.6 dry", "#117733"),
    ("WA_wgap_370_dry", "WaterGAP2 SSP3-7.0 dry", "#DDAA33"),
    ("WA_cwatm_370_annual", "CWatM SSP3-7.0 annual", "#EE99AA"),
]


def _exists(name: str) -> bool:
    return (RESULTS_DIR / f"{name}.json").exists() and (RESULTS_DIR / name / "plant_detail.csv").exists()


def _scenario_metrics(name: str) -> dict | None:
    """Water use, abatement, pathway mix and cost for one solved scenario."""
    if not _exists(name):
        return None
    with open(RESULTS_DIR / f"{name}.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    detail = pd.read_csv(RESULTS_DIR / name / "plant_detail.csv")
    year_rows = detail[detail["year"] == FRONTIER_YEAR]
    if year_rows.empty:
        return None
    baseline = baseline_emissions_mt(year_rows)
    residual = residual_emissions_mt(year_rows, FRONTIER_YEAR)
    shares = payload["years"][str(FRONTIER_YEAR)]["pathway_shares"]
    return {
        "scenario": name,
        "water_bn_m3": float(year_rows["water_use_m3"].sum()) / 1e8,
        "abated_gt": (baseline - residual) / 1000.0,
        "residual_mt": residual,
        "cost_trillion": float(payload["global_objective_cny"]) / 1e12,
        **{f"share_{k}": float(v) for k, v in shares.items()},
    }


def main_fig3() -> None:
    frontier = []
    for name, price in SWEEP:
        metrics = _scenario_metrics(name)
        if metrics is None:
            print(f"  [skip] {name}: not solved yet")
            continue
        metrics["price"] = price
        frontier.append(metrics)
    if len(frontier) < 3:
        print("  [abort] need at least three sweep points")
        return
    frontier = pd.DataFrame(frontier).sort_values("price").reset_index(drop=True)

    climate = []
    for name, label, colour in CLIMATE_RUNS:
        metrics = _scenario_metrics(name)
        if metrics is None:
            continue
        metrics.update({"label": label, "colour": colour})
        climate.append(metrics)
    climate = pd.DataFrame(climate)

    # Implied shadow price: the sweep price at which the unconstrained model would choose
    # the same amount of water as the constrained run actually used.
    if not climate.empty:
        order = frontier.sort_values("water_bn_m3")
        climate["implied_price"] = np.interp(
            climate["water_bn_m3"], order["water_bn_m3"], order["price"]
        )

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(DOUBLE_COL[0], 3.6))

    # (a) frontier — the informative axis is COST, not abatement: the 2060 target is a hard
    # constraint, so the system always delivers ~5.9 GtCO2 and simply substitutes retirement
    # for capture as water gets dearer. Abatement is plotted on the right axis to show that.
    ax_a.plot(frontier["water_bn_m3"], frontier["cost_trillion"], "-o", color="#333333",
              lw=1.6, markersize=4, zorder=3, label="System cost")
    for _, row in frontier.iterrows():
        if row["price"] in (0.0, 20.0, 100.0, 200.0, 500.0, 1000.0):
            ax_a.annotate(f"{row['price']:.0f}", (row["water_bn_m3"], row["cost_trillion"]),
                          textcoords="offset points", xytext=(5, 3), fontsize=5.5, color="#666666")
    ax_r = ax_a.twinx()
    ax_r.plot(frontier["water_bn_m3"], frontier["abated_gt"], "--s", color="#4477AA",
              lw=1.2, markersize=3, zorder=2, label="CO$_2$ abated")
    ax_r.set_ylim(0, 8)
    ax_r.set_ylabel(r"CO$_2$ abated in 2060 (Gt)", color="#4477AA", fontsize=7)
    ax_r.tick_params(axis="y", labelcolor="#4477AA", labelsize=6)
    ax_r.spines["top"].set_visible(False)
    if not climate.empty:
        for _, row in climate.iterrows():
            ax_a.scatter(row["water_bn_m3"], row["cost_trillion"], s=34, marker="D",
                         color=row["colour"], edgecolor="white", linewidth=0.6, zorder=4,
                         label=row["label"])
    ax_a.set_xlabel(r"System water use in 2060 (10$^8$ m$^3$ yr$^{-1}$)")
    ax_a.set_ylabel("System cost (trillion CNY)")
    ax_a.set_title("Water-carbon frontier\n(labels: shadow price, CNY per m3)", fontsize=7.5)
    ax_a.legend(fontsize=4.6, frameon=False, loc="upper right")
    ax_a.spines["top"].set_visible(False)
    panel_label(ax_a, "a")

    # (b) technology substitution along the frontier
    paths = ["unabated", "retire", "ccs", "biomass", "beccs", "ammonia"]
    bottom = np.zeros(len(frontier))
    xpos = np.arange(len(frontier))
    for pathway in paths:
        col = f"share_{pathway}"
        if col not in frontier.columns:
            continue
        values = frontier[col].fillna(0).to_numpy() * 100
        if values.max() < 0.05:
            continue
        ax_b.bar(xpos, values, bottom=bottom, width=0.85, linewidth=0,
                 color=PATHWAY_COLORS.get(pathway, "#999999"),
                 label=PATHWAY_LABELS.get(pathway, pathway))
        bottom += values
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([f"{p:.0f}" for p in frontier["price"]], fontsize=5.5, rotation=90)
    ax_b.set_xlabel("Water shadow price (CNY per m3)")
    ax_b.set_ylabel("2060 generation share (%)")
    ax_b.set_title("How the system adapts", fontsize=7.5, pad=14)
    ax_b.legend(fontsize=5, frameon=False, ncol=2, loc="lower left")
    ax_b.spines[["top", "right"]].set_visible(False)
    panel_label(ax_b, "b")

    # (c) shadow price implied by each climate scenario
    if not climate.empty:
        order = climate.sort_values("implied_price")
        y = np.arange(len(order))
        ax_c.barh(y, order["implied_price"], color=order["colour"], height=0.6, linewidth=0)
        for yi, value in zip(y, order["implied_price"]):
            ax_c.text(value * 1.05 + 0.5, yi, f"{value:.0f}", va="center", fontsize=6)
        ax_c.set_yticks(y)
        ax_c.set_yticklabels(order["label"], fontsize=5.5)
        ax_c.set_xlabel("Implied shadow price (CNY per m3)", fontsize=7)
        ax_c.set_title("What each climate scenario\nis worth in water price", fontsize=7.5)
        ax_c.spines[["top", "right"]].set_visible(False)
        # Reference: what industry actually pays
        ax_c.axvline(4.0, color="#666666", ls="--", lw=0.9)
        ax_c.text(4.4, len(order) - 0.6, "industrial tariff 4", fontsize=5.5, color="#666666")
    panel_label(ax_c, "c")

    fig.tight_layout()
    save_fig(fig, "water_fig3_shadow_price_frontier")

    print("\nFrontier:")
    print(frontier[["scenario", "price", "water_bn_m3", "abated_gt", "cost_trillion",
                    "share_ccs", "share_beccs", "share_retire"]].round(3).to_string(index=False))
    if not climate.empty:
        print("\nImplied shadow prices:")
        print(climate[["label", "water_bn_m3", "abated_gt", "implied_price"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main_fig3()
