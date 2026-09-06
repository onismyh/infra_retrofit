"""Sensitivity analysis tornado chart + ammonia appearance conditions.

Reads all SA_* and BASE_tight results from experiment_results_clean.json.
Generates:
  - ed_fig3_sensitivity: 2×2 panel (tornado, pathway robustness, ammonia conditions, retirement sensitivity)

Usage:
    python scripts/plot_sensitivity_tornado.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "SimHei", "axes.unicode_minus": False, "font.size": 8, "mathtext.default": "regular",
    "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.facecolor": "white",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "legend.fontsize": 7, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


def _load():
    with open(RESULTS_DIR / "experiment_results_clean.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _get_cost(data, key):
    if key in data:
        return data[key]["global_objective_cny"] / 1e9
    return None


def _get_shares(data, key, year):
    if key not in data:
        return {}
    return data[key].get("years", {}).get(str(year), {}).get("pathway_shares", {})


def fig_sensitivity_4panel(data):
    """4-panel sensitivity figure for Extended Data."""
    base_cost = _get_cost(data, "BASE")

    # Define sensitivity pairs: (label, low_scenario, high_scenario)
    sensitivity_pairs = [
        ("Discount rate", "SA_discount_3pct", "SA_discount_8pct", "3%", "8%"),
        ("CCS CAPEX", "SA_ccs_capex_low", "SA_ccs_capex_high", "900", "1800"),
        ("Replacement cost", "SA_retire_cost_500", "SA_retire_cost_1000", "500", "1000"),
        ("Biomass cost", None, "SA_biomass_cost_200", "BASE", "×2.0"),
        ("Ammonia cost", "SA_ammonia_cost_50", "SA_ammonia_cost_70", "×0.5", "×0.7"),
        ("Pipeline corridor", "SA_pipe_mid", "SA_pipe_full", "×0.6", "×1.0"),
        ("Storage injectivity", None, "SA_injectivity_half", "BASE", "×0.5"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) Tornado chart: total cost deviation from BASE
    labels, lows, highs = [], [], []
    for label, low_key, high_key, low_label, high_label in sensitivity_pairs:
        low_cost = _get_cost(data, low_key) if low_key else base_cost
        high_cost = _get_cost(data, high_key) if high_key else base_cost
        if low_cost is None or high_cost is None:
            continue
        low_delta = (low_cost - base_cost) / base_cost * 100
        high_delta = (high_cost - base_cost) / base_cost * 100
        labels.append(label)
        lows.append(min(low_delta, high_delta))
        highs.append(max(low_delta, high_delta))

    # Sort by range
    ranges = [h - l for l, h in zip(lows, highs)]
    order = np.argsort(ranges)
    labels = [labels[i] for i in order]
    lows = [lows[i] for i in order]
    highs = [highs[i] for i in order]

    y = np.arange(len(labels))
    ax_a.barh(y, [h - max(l, 0) for l, h in zip(lows, highs)],
              left=[max(l, 0) for l in lows],
              color="#CC3311", alpha=0.7, height=0.5, label="Cost increase")
    ax_a.barh(y, [min(0, l) for l in lows],
              color="#0077BB", alpha=0.7, height=0.5, label="Cost decrease")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels)
    ax_a.set_xlabel("Δ Cost vs. BASE (%)")
    ax_a.axvline(0, color="black", linewidth=0.5)
    ax_a.legend(fontsize=6)
    ax_a.text(0.02, 0.98, "(a) Cost sensitivity", transform=ax_a.transAxes,
              fontweight="bold", fontsize=8, va="top")

    # (b) Biomass 2030 share across all scenarios — robustness
    all_sa = [k for k in data.keys() if k.startswith("SA_")]
    bio_shares = []
    sa_labels = []
    for sa in sorted(all_sa):
        shares = _get_shares(data, sa, 2030)
        bio = shares.get("biomass", 0) * 100
        bio_shares.append(bio)
        sa_labels.append(sa.replace("SA_", "").replace("_", " "))

    y_b = np.arange(len(sa_labels))
    colors_b = ["#228833" if b > 80 else "#CCBB44" if b > 70 else "#CC3311" for b in bio_shares]
    ax_b.barh(y_b, bio_shares, color=colors_b, height=0.6, edgecolor="white", linewidth=0.3)
    ax_b.axvline(85, color="black", linewidth=0.5, linestyle="--")
    ax_b.text(85.5, len(sa_labels) - 0.5, "BASE", fontsize=6, va="top")
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(sa_labels, fontsize=5.5)
    ax_b.set_xlabel("Biomass share 2030 (%)")
    ax_b.set_xlim(70, 100)
    ax_b.text(0.02, 0.98, "(b) Biomass robustness (2030)", transform=ax_b.transAxes,
              fontweight="bold", fontsize=8, va="top")

    # (c) Ammonia appearance conditions — when does ammonia > 1%?
    nh3_data = []
    for sa in sorted(all_sa):
        shares = _get_shares(data, sa, 2030)
        nh3 = shares.get("ammonia", 0) * 100
        if nh3 > 0.5:  # threshold for "appears"
            nh3_data.append((sa.replace("SA_", "").replace("_", " "), nh3))

    # Add BASE scenarios
    for sc in ["BASE", "BASE_neg", "BASE_zero"]:
        shares = _get_shares(data, sc, 2030)
        nh3 = shares.get("ammonia", 0) * 100
        if nh3 > 0.5:
            nh3_data.append((sc, nh3))

    if nh3_data:
        nh3_data.sort(key=lambda x: x[1])
        nh3_labels, nh3_vals = zip(*nh3_data)
        y_c = np.arange(len(nh3_labels))
        ax_c.barh(y_c, nh3_vals, color="#EE6677", height=0.5, edgecolor="white")
        ax_c.set_yticks(y_c)
        ax_c.set_yticklabels(nh3_labels, fontsize=6)
        ax_c.set_xlabel("Ammonia share 2030 (%)")
        ax_c.axvline(1, color="black", linewidth=0.5, linestyle=":")
        ax_c.text(1.1, len(nh3_labels) - 0.5, "1%\nthreshold", fontsize=5, va="top")
    else:
        ax_c.text(0.5, 0.5, "Ammonia never exceeds 0.5%\nin any scenario",
                  transform=ax_c.transAxes, ha="center", va="center", fontsize=8)
    ax_c.text(0.02, 0.98, "(c) Ammonia appearance", transform=ax_c.transAxes,
              fontweight="bold", fontsize=8, va="top")

    # (d) Retirement share 2050 across scenarios
    ret_data = []
    for sa in sorted(all_sa):
        shares = _get_shares(data, sa, 2050)
        ret = shares.get("retire", 0) * 100
        ret_data.append((sa.replace("SA_", "").replace("_", " "), ret))
    ret_data.sort(key=lambda x: x[1])
    ret_labels, ret_vals = zip(*ret_data)
    y_d = np.arange(len(ret_labels))
    colors_d = ["#CC3311" if r < 50 else "#999999" for r in ret_vals]
    ax_d.barh(y_d, ret_vals, color=colors_d, height=0.6, edgecolor="white", linewidth=0.3)
    ax_d.axvline(74, color="black", linewidth=0.5, linestyle="--")
    ax_d.text(74.5, len(ret_labels) - 0.5, "BASE", fontsize=6, va="top")
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(ret_labels, fontsize=5.5)
    ax_d.set_xlabel("Retirement share 2050 (%)")
    ax_d.text(0.02, 0.98, "(d) Retirement sensitivity (2050)", transform=ax_d.transAxes,
              fontweight="bold", fontsize=8, va="top")

    fig.tight_layout()
    _save(fig, "ed_fig3_sensitivity")


def main():
    data = _load()
    print(f"Loaded {len(data)} scenarios. Generating sensitivity figure...")
    fig_sensitivity_4panel(data)
    print(f"\nSensitivity figure saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
