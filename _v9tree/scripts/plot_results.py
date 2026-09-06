"""Academic-quality figures for coal retrofit optimization results.

Style: Arial, Nature/Science sizing, colorblind-safe, vector output.
Reads results/experiment_results.json → outputs to results/figures/

Usage:
    python scripts/plot_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Global style (academic-plotting skill) ───────────────────────────────────
plt.rcParams.update({
    "font.family": "SimHei", "axes.unicode_minus": False,
    "font.size": 8,
    "mathtext.default": "regular",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.facecolor": "white",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ── Palettes ─────────────────────────────────────────────────────────────────
PATHWAY_COLORS = {
    "unabated": "#BBBBBB",
    "retire": "#999999", "ccs": "#4477AA", "biomass": "#228833",
    "beccs": "#CCBB44", "ammonia": "#EE6677",
}
PATHWAY_LABELS = {
    "unabated": "Unabated operation",
    "retire": "Retirement", "ccs": "CCS", "biomass": "Biomass co-firing",
    "beccs": "BECCS", "ammonia": "Ammonia co-firing",
}
PATHWAY_ORDER = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]
SCENARIO_COLORS = {"low": "#4477AA", "base": "#999999", "high": "#CC3311"}
DIVERGING = {"neg": "#0077BB", "pos": "#CC3311"}

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _load() -> dict:
    clean_path = RESULTS_DIR / "experiment_results_clean.json"
    raw_path = RESULTS_DIR / "experiment_results.json"
    path = clean_path if clean_path.exists() else raw_path
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Remove excluded scenarios and metadata
    raw.pop("RQ2_no_learning", None)
    raw.pop("RQ2_sparse", None)
    raw.pop("experiment_results", None)
    raw.pop("_metadata", None)
    return raw


def _is_feasible(scenario_data: dict) -> bool:
    """Check if a scenario is feasible (no large slack retained)."""
    return not scenario_data.get("infeasible", False)


def _clean_shares(shares: dict) -> dict:
    """Round pathway shares below 0.1% to zero for clean visualization."""
    cleaned = {k: (v if v >= 0.001 else 0.0) for k, v in shares.items()}
    total = sum(cleaned.values())
    if total > 0:
        cleaned = {k: v / total for k, v in cleaned.items()}
    return cleaned


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


def _get_shares(result, year_str):
    return result.get("years", {}).get(year_str, {}).get("pathway_shares", {})


# ── Fig 1: BASE pathway shares (RQ1) ────────────────────────────────────────
def fig1(data):
    base = data["BASE"]
    years = sorted(int(y) for y in base["years"])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(years))
    bottom = np.zeros(len(years))
    for pw in PATHWAY_ORDER:
        vals = [_clean_shares(_get_shares(base, str(y))).get(pw, 0) * 100 for y in years]
        ax.bar(x, vals, 0.55, bottom=bottom, label=PATHWAY_LABELS[pw],
               color=PATHWAY_COLORS[pw], edgecolor="white", linewidth=0.3)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Generation share (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, bbox_to_anchor=(0, 1.02), loc="lower left", fontsize=6.5)
    ax.text(0.0, 1.08, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    _save(fig, "fig1_pathway_shares")


# ── Fig 2: Cost breakdown waterfall (RQ1) ────────────────────────────────────
def fig2(data):
    base = data["BASE"]
    years = sorted(int(y) for y in base["years"])
    cost_labels = {
        "Baseline net": "baseline_net_cost",
        "Carbon cost": "carbon_cost",
        "Coal savings": "coal_savings_credit",
        "Energy penalty": "energy_penalty_cost",
        "CCS O&M": "ccs_om_cost",
        "Incr. O&M": "incremental_om",
        "Biomass": "biomass_cost",
        "Ammonia": "ammonia_cost",
        "CO$_2$ transport": "transport_opex",
        "Storage": "storage_cost",
        "Stranded asset": "stranded_capex",
        "CCS CAPEX": "ccs_retrofit_capex",
        "Pipeline": "pipe_capex",
        "Blend upgrade": "blend_upgrade_capex",
    }
    fig, axes = plt.subplots(1, len(years), figsize=(7.0, 3.5), sharey=False)
    for col, (ax, year) in enumerate(zip(axes, years)):
        bd = base["years"][str(year)]["cost_breakdown"]
        labels, values = [], []
        for label, key in cost_labels.items():
            v = bd.get(key, 0) / 1e9
            if abs(v) > 0.5:
                labels.append(label)
                values.append(v)
        colors = [DIVERGING["pos"] if v > 0 else DIVERGING["neg"] for v in values]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, height=0.6, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("B CNY", fontsize=7)
        ax.text(0.0, 1.05, f"({chr(97+col)}) {year}", transform=ax.transAxes,
                fontweight="bold", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.4)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig2_cost_breakdown")


# ── Fig 3+4 combined: 4-panel sensitivity figure ────────────────────────────
def fig3_4_combined(data):
    """2x2 sensitivity figure: carbon price (top) and learning rate (bottom)."""
    carbon_scenarios = [
        ("Low (60\u2192300)", "RQ2_low_carbon", SCENARIO_COLORS["low"], "v"),
        ("Base (120\u2192600)", "BASE", SCENARIO_COLORS["base"], "o"),
        ("High (200\u2192800)", "RQ2_high_carbon", SCENARIO_COLORS["high"], "^"),
    ]
    learning_scenarios = [
        ("Base (LR=7.5%)", "BASE", "#999999", "o"),
        ("Fast learning (LR=15%)", "RQ2_fast_learning", "#CC3311", "D"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) Carbon price → CCS/BECCS share
    for label, key, color, marker in carbon_scenarios:
        if key not in data:
            continue
        r = data[key]
        yrs = sorted(int(y) for y in r["years"])
        ccs = [(_clean_shares(r["years"][str(y)]["pathway_shares"]).get("ccs", 0) +
                _clean_shares(r["years"][str(y)]["pathway_shares"]).get("beccs", 0)) * 100
               for y in yrs]
        ax_a.plot(yrs, ccs, marker=marker, color=color, label=label,
                  markersize=4, markeredgecolor="white", markeredgewidth=0.3)
    ax_a.set_ylabel("CCS + BECCS share (%)")
    ax_a.set_xlabel("Year")
    ax_a.legend(fontsize=6.5)
    ax_a.text(0.0, 1.05, "(a)", transform=ax_a.transAxes, fontweight="bold", fontsize=9)

    # (b) Carbon price → total cost (skip infeasible)
    labels_b, costs_b, colors_b, hatches_b = [], [], [], []
    for label, key, color, _ in carbon_scenarios:
        if key not in data:
            continue
        inf = not _is_feasible(data[key])
        labels_b.append(label + ("*" if inf else ""))
        costs_b.append(data[key]["global_objective_cny"] / 1e9)
        colors_b.append(color)
        hatches_b.append("//" if inf else "")
    x_b = np.arange(len(labels_b))
    bars_b = ax_b.bar(x_b, costs_b, width=0.5, color=colors_b, edgecolor="white")
    for bar, h in zip(bars_b, hatches_b):
        bar.set_hatch(h)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(labels_b, rotation=15, ha="right", fontsize=6.5)
    ax_b.set_ylabel("Total discounted cost (B CNY)")
    ax_b.text(0.0, 1.05, "(b)", transform=ax_b.transAxes, fontweight="bold", fontsize=9)

    # (c) Learning rate → CCS deployment
    for label, key, color, marker in learning_scenarios:
        if key not in data:
            continue
        r = data[key]
        yrs = sorted(int(y) for y in r["years"])
        ccs = [(_clean_shares(r["years"][str(y)]["pathway_shares"]).get("ccs", 0) +
                _clean_shares(r["years"][str(y)]["pathway_shares"]).get("beccs", 0)) * 100
               for y in yrs]
        ax_c.plot(yrs, ccs, marker=marker, color=color, label=label,
                  markersize=4, markeredgecolor="white", markeredgewidth=0.3)
    ax_c.set_ylabel("CCS + BECCS share (%)")
    ax_c.set_xlabel("Year")
    ax_c.legend(fontsize=6.5)
    ax_c.text(0.0, 1.05, "(c)", transform=ax_c.transAxes, fontweight="bold", fontsize=9)

    # (d) Learning rate → total cost
    labels_d, costs_d, colors_d = [], [], []
    for label, key, color, _ in learning_scenarios:
        if key not in data:
            continue
        labels_d.append(label)
        costs_d.append(data[key]["global_objective_cny"] / 1e9)
        colors_d.append(color)
    x_d = np.arange(len(labels_d))
    ax_d.bar(x_d, costs_d, width=0.5, color=colors_d, edgecolor="white")
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(labels_d, rotation=15, ha="right", fontsize=6.5)
    ax_d.set_ylabel("Total discounted cost (B CNY)")
    ax_d.text(0.0, 1.05, "(d)", transform=ax_d.transAxes, fontweight="bold", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig3_sensitivity_combined")


# ── Fig 5: Pathway marginal value (RQ3) ─────────────────────────────────────
def fig5(data):
    if "BASE" not in data:
        return
    base_cost = data["BASE"]["global_objective_cny"] / 1e9
    variants = [
        ("No ammonia", "RQ3_no_ammonia"),
        ("No CCS/BECCS", "RQ3_no_ccs"),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    labels, deltas, infeasible_flags = [], [], []
    for label, key in variants:
        if key not in data:
            continue
        is_inf = data[key].get("infeasible", False)
        val = data[key]["global_objective_cny"] / 1e9 - base_cost
        if val > 0:
            labels.append(label)
            deltas.append(val)
            infeasible_flags.append(is_inf)

    y = np.arange(len(labels))
    bars = ax.barh(y, deltas, height=0.5, color=DIVERGING["pos"], edgecolor="white", linewidth=0.3)

    for i, (bar, label, d, inf) in enumerate(zip(bars, labels, deltas, infeasible_flags)):
        bar_right = bar.get_width()
        pct = d / base_cost * 100
        if label == "No ammonia":
            annotation = f"+{d:,.0f} B CNY  (+{pct:.1f}%, ammonia contributes <1% cost saving)"
        elif label == "No CCS/BECCS":
            annotation = f"+{d:,.0f} B CNY  (+{pct:.1f}%)"
        else:
            annotation = f"+{d:,.0f} B CNY"
        if inf:
            annotation += "  *infeasible"
        ax.text(bar_right * 1.05, bar.get_y() + bar.get_height() / 2,
                annotation, ha="left", va="center", fontsize=6.5)

    ax.set_xscale("log")
    ax.set_xlim(0.5, max(deltas) * 8 if deltas else 20000)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("\u0394 Cost vs. BASE (B CNY, log scale)")
    ax.text(0.0, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig5_pathway_value")


# ── Fig 6: Pathway comparison across restrictions (RQ3) ──────────────────────
def fig6(data):
    variants = [
        ("BASE", "BASE"),
        ("No\nammonia", "RQ3_no_ammonia"),
        ("No CCS/\nBECCS", "RQ3_no_ccs"),
    ]
    years_show = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, len(years_show), figsize=(7.0, 3.2), sharey=True)

    for col, (ax, year) in enumerate(zip(axes, years_show)):
        avail = [(vl, key) for vl, key in variants
                 if key in data and str(year) in data[key]["years"]]
        x = np.arange(len(avail))
        bottom = np.zeros(len(avail))
        for pw in PATHWAY_ORDER:
            vals = [_clean_shares(data[key]["years"][str(year)]["pathway_shares"]).get(pw, 0) * 100
                    for _, key in avail]
            ax.bar(x, vals, 0.6, bottom=bottom,
                   label=PATHWAY_LABELS[pw] if col == 0 else "",
                   color=PATHWAY_COLORS[pw], edgecolor="white", linewidth=0.3)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([vl for vl, _ in avail], fontsize=6, rotation=45, ha="right")
        ax.text(0.0, 1.05, f"({chr(97+col)}) {year}", transform=ax.transAxes,
                fontweight="bold", fontsize=9)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Generation share (%)")
    axes[0].legend(loc="upper left", ncol=1, fontsize=6)
    fig.tight_layout()
    _save(fig, "fig6_pathway_comparison")


# ── Fig 7: CCS-only vs BASE cost structure (RQ3) ────────────────────────────
def fig7(data):
    if "RQ3_ccs_only" not in data:
        return
    cost_components = {
        "Carbon cost": "carbon_cost",
        "CCS CAPEX": "ccs_retrofit_capex",
        "CCS O&M": "ccs_om_cost",
        "Energy penalty": "energy_penalty_cost",
        "Pipeline": "pipe_capex",
        "Storage": "storage_cost",
        "Biomass": "biomass_cost",
        "Ammonia": "ammonia_cost",
        "Stranded asset": "stranded_capex",
    }
    scenarios = [("BASE", "BASE", "#4477AA"), ("CCS/BECCS only", "RQ3_ccs_only", "#CC3311")]
    n_comp = len(cost_components)
    x = np.arange(n_comp)
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    totals = {}
    for i, (label, key, color) in enumerate(scenarios):
        if key not in data or "2050" not in data[key]["years"]:
            continue
        bd = data[key]["years"]["2050"]["cost_breakdown"]
        vals = [bd.get(comp_key, 0) / 1e9 for comp_key in cost_components.values()]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color,
               edgecolor="white", linewidth=0.3, alpha=0.85)
        total = sum(v for v in bd.values() if isinstance(v, (int, float))) / 1e9
        totals[label] = total

    ax.set_xticks(x)
    ax.set_xticklabels(list(cost_components.keys()), rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("Period cost at 2050 (B CNY)")
    ax.legend(fontsize=7)
    ax.text(0.0, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # Annotate totals
    for i, (label, _, color) in enumerate(scenarios):
        if label in totals:
            ax.annotate(f"Total: {totals[label]:,.0f} B",
                        xy=(0.02 + i * 0.5, 0.97), xycoords="axes fraction",
                        ha="left", va="top", fontsize=7, color=color, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig7_ccs_cost_penalty")



# ── Fig 8: Emissions trajectory over time ────────────────────────────────────
def fig8(data):
    """Emissions trajectory: model-consistent residual CO2 index by year.

    Computed from results/<scenario>/plant_detail.csv via
    plot_style.residual_emissions_mt (retrofit CF boost, rebuilt-plant
    efficiency, chosen blend levels, penalty emissions).
    """
    import pandas as pd
    from plot_style import residual_emissions_mt, baseline_emissions_mt

    scenarios_to_plot = [
        ("BASE", "BASE", "#999999", "o"),
        ("No CCS", "RQ3_no_ccs", "#4477AA", "^"),
        ("High carbon", "SA_carbon_high", "#CC3311", "D"),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for label, key, color, marker in scenarios_to_plot:
        detail_path = RESULTS_DIR / key / "plant_detail.csv"
        if not detail_path.exists():
            continue
        detail = pd.read_csv(detail_path)
        years = sorted(int(y) for y in detail["year"].unique())
        base = baseline_emissions_mt(detail[detail["year"] == years[0]])
        if base <= 0:
            continue
        emissions_idx = [
            residual_emissions_mt(detail[detail["year"] == y], y) / base * 100.0
            for y in years
        ]
        inf = key in data and not _is_feasible(data[key])
        ls = "--" if inf else "-"
        lbl = label + (" *" if inf else "")
        ax.plot(years, emissions_idx, marker=marker, color=color, label=lbl,
                markersize=4, markeredgecolor="white", markeredgewidth=0.3,
                linestyle=ls)

    ax.set_ylabel("Residual emissions (% of unabated)")
    ax.set_xlabel("Year")

    # 2060 carbon neutrality reference line
    ax.axhline(0, color="#333333", linewidth=0.6, linestyle="--")
    ax.text(2028, 1.5,
            "2060 carbon neutrality target", fontsize=6, color="#333333", va="bottom")

    ax.legend(fontsize=6, loc="upper right")
    ax.text(0.0, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # Secondary annotation: absolute residual MtCO2 for BASE at last year
    base_detail_path = RESULTS_DIR / "BASE" / "plant_detail.csv"
    if base_detail_path.exists():
        _d = pd.read_csv(base_detail_path)
        last_yr = int(_d["year"].max())
        rem_mt = residual_emissions_mt(_d[_d["year"] == last_yr], last_yr)
        _b = baseline_emissions_mt(_d[_d["year"] == last_yr])
        rem_pct = rem_mt / _b * 100.0
        ax.annotate(f"BASE {last_yr}: {rem_mt:+.0f} MtCO$_2$/yr",
                    xy=(last_yr, rem_pct),
                    xytext=(last_yr - 8, rem_pct + 12),
                    fontsize=6, color="#999999",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

    fig.tight_layout()
    _save(fig, "fig8_emissions_trajectory")


def main():
    data = _load()
    print(f"Loaded {len(data)} experiments. Generating figures...")
    fig1(data)
    fig2(data)
    fig3_4_combined(data)
    fig5(data)
    fig6(data)
    fig7(data)
    fig8(data)
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
