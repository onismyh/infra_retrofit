"""The water-cost trade-off, on real resource cost.

Two corrections are built into this figure.

1. The swept water charge is an accounting device, not a resource cost. `global_objective_cny`
   includes it, so plotting the objective against water use mostly plots the tax: the
   objective rises 5.1x across the sweep while the real system cost rises 1.44x. Here the
   charge is discounted back out year by year and only the residual - the genuine cost of
   coal plants retiring earlier, retrofitting differently and moving water further - is
   plotted.

2. The frontier's fine structure is not resolvable at the solver settings used. Between 0
   and 20 CNY/m3 the real-cost differences (0.007-0.053 trillion) are smaller than each
   solve's own optimality tolerance (0.086-0.139 trillion), which is why a naive slope
   between neighbouring points can even come out negative. Tolerance is drawn on the figure
   and the unresolvable segment is shaded, rather than presenting noise as a marginal cost.

Usage:
    python scripts/plot_water_cost_tradeoff.py
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
from plot_style import apply_style, save_fig, panel_label, RESULTS_DIR, DOUBLE_COL
from coal_retrofit.optimization.scenario import OptimizationScenario

apply_style()

YEARS = [2030, 2040, 2050, 2060]
FRONTIER_YEAR = 2060
PRICES = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

CLIMATE_RUNS = [
    ("WA_cwatm_126_dry", "CWatM SSP1-2.6 dry", "#4477AA"),
    ("WA_cwatm_370_dry", "CWatM SSP3-7.0 dry", "#CC3311"),
    ("WA_wgap_126_dry", "WaterGAP2 SSP1-2.6 dry", "#117733"),
    ("WA_wgap_370_dry", "WaterGAP2 SSP3-7.0 dry", "#DDAA33"),
]


def _annuity(n: int, rate: float) -> float:
    return (1.0 - (1.0 + rate) ** -n) / rate


def _metrics(name: str, price: float) -> dict | None:
    payload_path = RESULTS_DIR / f"{name}.json"
    detail_path = RESULTS_DIR / name / "plant_detail.csv"
    if not payload_path.exists() or not detail_path.exists():
        return None
    with open(payload_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    detail = pd.read_csv(detail_path)
    scenario = OptimizationScenario(experiment_id="x", description="x")
    rate, base_year = scenario.discount_rate, scenario.discount_base_year

    charge = 0.0
    water_by_year = {}
    for year in YEARS:
        water = float(detail[detail["year"] == year]["water_use_m3"].sum())
        water_by_year[year] = water
        discount = 1.0 / (1.0 + rate) ** max(0, year - base_year)
        charge += discount * _annuity(10, rate) * price * water

    objective = float(payload["global_objective_cny"])
    gap = float(payload.get("solver_quality", {}).get("mip_gap", 0.0) or 0.0)
    return {
        "scenario": name,
        "price": price,
        "water_1e8": water_by_year[FRONTIER_YEAR] / 1e8,
        "real_cost_T": (objective - charge) / 1e12,
        "tolerance_T": objective * gap / 1e12,
        "retire": float(payload["years"][str(FRONTIER_YEAR)]["pathway_shares"].get("retire", 0)) * 100,
    }


def main() -> None:
    rows = [_metrics("BASE" if p == 0 else f"WP_{p:04d}", float(p)) for p in PRICES]
    frontier = pd.DataFrame([r for r in rows if r]).sort_values("water_1e8").reset_index(drop=True)
    if len(frontier) < 4:
        print("  [abort] not enough sweep points")
        return

    climate = []
    for name, label, colour in CLIMATE_RUNS:
        metrics = _metrics(name, 0.0)
        if metrics:
            metrics.update({"label": label, "colour": colour})
            climate.append(metrics)
    climate = pd.DataFrame(climate)

    # Marginal cost of saving water, only where the difference clears both solves' tolerance
    marginal = []
    ordered = frontier.sort_values("water_1e8", ascending=False).reset_index(drop=True)
    for i in range(len(ordered) - 1):
        wet, dry = ordered.iloc[i], ordered.iloc[i + 1]
        saved = (wet["water_1e8"] - dry["water_1e8"]) * 1e8
        extra = (dry["real_cost_T"] - wet["real_cost_T"]) * 1e12
        tolerance = (wet["tolerance_T"] + dry["tolerance_T"]) * 1e12
        marginal.append({
            "cum_saved_1e8": (frontier["water_1e8"].max() - dry["water_1e8"]),
            "saved_1e8": saved / 1e8,
            "cny_per_m3": extra / saved if saved > 0 else np.nan,
            "resolved": abs(extra) > tolerance,
            "price": dry["price"],
        })
    marginal = pd.DataFrame(marginal)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(DOUBLE_COL[0], 3.5))

    # (a) frontier on real resource cost
    ax_a.errorbar(frontier["water_1e8"], frontier["real_cost_T"], yerr=frontier["tolerance_T"],
                  fmt="-o", color="#333333", lw=1.5, markersize=4, capsize=2, elinewidth=0.8,
                  ecolor="#AAAAAA", zorder=3, label="Real system cost (± solver tolerance)")
    for _, row in frontier.iterrows():
        if row["price"] in (0, 50, 100, 200, 500, 1000):
            ax_a.annotate(f"{row['price']:.0f}", (row["water_1e8"], row["real_cost_T"]),
                          textcoords="offset points", xytext=(4, 5), fontsize=5.5, color="#666666")
    if not climate.empty:
        for _, row in climate.iterrows():
            ax_a.scatter(row["water_1e8"], row["real_cost_T"], s=34, marker="D",
                         color=row["colour"], edgecolor="white", linewidth=0.6, zorder=4,
                         label=row["label"])
    ax_a.set_xlabel(r"Water use in 2060 (10$^8$ m$^3$ yr$^{-1}$)")
    ax_a.set_ylabel("Real system cost (trillion CNY)")
    ax_a.set_title("Water-cost frontier\n(labels: water price, CNY per m3)", fontsize=7.5, pad=12)
    ax_a.legend(fontsize=4.6, frameon=False, loc="upper right")
    ax_a.spines[["top", "right"]].set_visible(False)
    panel_label(ax_a, "a")

    # (b) cost of saving water — a MACC for water
    resolved = marginal[marginal["resolved"]]
    unresolved = marginal[~marginal["resolved"]]
    if not unresolved.empty:
        ax_b.axvspan(0, unresolved["cum_saved_1e8"].max(), color="#DDDDDD", alpha=0.55, lw=0)
        ax_b.text(unresolved["cum_saved_1e8"].max() / 2, ax_b.get_ylim()[1] * 0.5,
                  "below solver\ntolerance", fontsize=5.5, ha="center", color="#777777")
    if not resolved.empty:
        ax_b.step(resolved["cum_saved_1e8"], resolved["cny_per_m3"], where="post",
                  color="#CC3311", lw=1.6)
        ax_b.scatter(resolved["cum_saved_1e8"], resolved["cny_per_m3"], s=16, color="#CC3311", zorder=3)
    ax_b.axhline(4.0, color="#666666", ls="--", lw=0.9)
    ax_b.text(1, 5.5, "industrial tariff 4", fontsize=5.5, color="#666666")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"Water saved vs unconstrained (10$^8$ m$^3$ yr$^{-1}$)")
    ax_b.set_ylabel("Marginal cost of saving (CNY per m3)")
    ax_b.set_title("What each cubic metre saved costs", fontsize=7.5, pad=12)
    ax_b.spines[["top", "right"]].set_visible(False)
    panel_label(ax_b, "b")

    # (c) the mechanism: retirement replaces capture
    ax_c.plot(frontier["water_1e8"], frontier["retire"], "-o", color="#777777",
              lw=1.5, markersize=4)
    ax_c.set_xlabel(r"Water use in 2060 (10$^8$ m$^3$ yr$^{-1}$)")
    ax_c.set_ylabel("Retirement share of generation (%)")
    ax_c.set_title("The adjustment channel", fontsize=7.5, pad=12)
    ax_c.invert_xaxis()
    ax_c.spines[["top", "right"]].set_visible(False)
    panel_label(ax_c, "c")

    fig.tight_layout()
    save_fig(fig, "water_fig3b_cost_tradeoff")

    print("\nFrontier on real resource cost:")
    print(frontier[["scenario", "price", "water_1e8", "real_cost_T", "tolerance_T", "retire"]]
          .round(3).to_string(index=False))
    print("\nMarginal cost of saving water:")
    print(marginal.round(2).to_string(index=False))
    span = frontier["real_cost_T"].max() - frontier["real_cost_T"].min()
    print(f"\nReal cost spans {frontier['real_cost_T'].min():.2f} -> {frontier['real_cost_T'].max():.2f} "
          f"trillion CNY (+{span:.2f}, x{frontier['real_cost_T'].max()/frontier['real_cost_T'].min():.2f}) "
          f"while water falls {frontier['water_1e8'].max():.1f} -> {frontier['water_1e8'].min():.1f} x10^8 m3")


if __name__ == "__main__":
    main()
