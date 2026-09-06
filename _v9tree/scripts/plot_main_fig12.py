"""Main figures 1 and 2: emissions trajectory and cost waterfall.

Fig 1 — Multi-scenario residual emissions index (2030-2060).
Fig 2 — BASE scenario cost breakdown waterfall, one panel per year.
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
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (
    apply_style, save_fig, panel_label, panel_label_inside,
    ROOT, RESULTS_DIR, FIGURES_DIR, BASE_DIR,
    PATHWAY_COLORS, PATHWAY_LABELS, PATHWAY_ORDER,
    DIVERGING, DOUBLE_COL, FULL_PAGE,
    residual_emissions_mt, baseline_emissions_mt,
    require_valid_scenarios,
)

apply_style()

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_PATH = RESULTS_DIR / "experiment_results_clean.json"
YEARS = [2030, 2040, 2050, 2060]
YEAR_STRS = [str(y) for y in YEARS]


def _load_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


_PLANT_DETAIL_CACHE: dict[str, "pd.DataFrame | None"] = {}


def _scenario_plant_detail(sc_key: str):
    if sc_key not in _PLANT_DETAIL_CACHE:
        path = RESULTS_DIR / sc_key / "plant_detail.csv"
        _PLANT_DETAIL_CACHE[sc_key] = pd.read_csv(path) if path.exists() else None
    return _PLANT_DETAIL_CACHE[sc_key]


def _emission_index(sc_key: str, year: int) -> float | None:
    """Model-consistent residual emission index (% of unabated baseline),
    computed from plant_detail.csv via plot_style.residual_emissions_mt."""
    detail = _scenario_plant_detail(sc_key)
    if detail is None:
        return None
    yr = detail[detail["year"] == year]
    if yr.empty:
        return None
    base = baseline_emissions_mt(yr)
    if base <= 0:
        return None
    return residual_emissions_mt(yr, year) / base * 100.0


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def main_fig1_emissions_trajectory() -> None:
    """Multi-scenario emissions trajectory (all scenarios, one panel)."""

    data = _load_data()

    # Scenario definitions: (key, color, linestyle, linewidth, marker, label)
    scenarios = [
        ("BASE",           "#333333", "solid",    2.0, "o", "BASE (all pathways)"),
        ("BASE_zero",      "#999999", "dashed",   1.2, "s", "Net zero"),
        ("BASE_neg",       "#CCBB44", "solid",    1.5, "v", "Net negative"),
        ("RQ3_no_ammonia", "#EE6677", "dashed",   1.5, "s", "No ammonia"),
        ("RQ3_no_ccs",     "#4477AA", "dashdot",  1.5, "^", "No CCS"),
        ("RQ3_no_biomass", "#228833", "dotted",   1.5, "D", "No biomass*"),
        ("RQ3_retire_only", "#555555", (0, (5, 1)), 1.1, "*", "Retire only"),
        ("RQ3_ccs_only",   "#117733", (0, (3, 1)), 1.1, "<", "CCS/BECCS only"),
        ("WA_grid_200km",  "#0077BB", "dashed",   1.5, "P", "Water constraint"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    # Drop scenarios that are not physical solutions. Two have shipped into this figure
    # before: RQ3_retire_only solves infeasible_or_unbounded (NaN objective, all shares
    # zero) yet drew a flat 0% line reading as "retirement alone reaches zero emissions";
    # RQ3_ccs_only reports optimal but 89% of its objective is the ghost-resource slack
    # penalty. Both are reported in Extended Data instead, with the reason stated.
    valid = set(require_valid_scenarios([s[0] for s in scenarios]))

    for sc_key, color, ls, lw, marker, label in scenarios:
        sc = data.get(sc_key)
        if sc is None or sc_key not in valid:
            continue

        infeasible = sc.get("infeasible", False)
        # Degenerate solves (e.g. zero-filled infeasible models) carry no slack to
        # flag, so also treat non-optimal year statuses as infeasible markers.
        if not infeasible:
            statuses = {
                str(yr_data.get("status", ""))
                for yr_data in sc.get("years", {}).values()
                if isinstance(yr_data, dict)
            }
            infeasible = bool(statuses) and statuses != {"optimal"}

        # Build emission index per year (use available years)
        ys: list[float] = []
        xs: list[int] = []
        for yr_str in YEAR_STRS:
            yr_data = sc["years"].get(yr_str)
            if yr_data is None:
                continue
            idx = _emission_index(sc_key, int(yr_str))
            if idx is None:
                continue
            ys.append(idx)
            xs.append(int(yr_str))

        # Infeasible scenarios: override to dotted
        actual_ls = "dotted" if infeasible else ls

        ax.plot(
            xs, ys,
            color=color,
            linestyle=actual_ls,
            linewidth=lw,
            marker=marker,
            markersize=6,
            label=label,
            zorder=3,
        )

        # No inline endpoint labels — the legend provides identification.
        # (Removed: all 6 scenario lines converge near 0% at 2060, causing
        #  severe label overlap when annotated at the endpoint.)

    # Net-negative shaded band
    ax.axhspan(-5, 0, color="#BBBBBB", alpha=0.10, zorder=0, label="_nolegend_")
    ax.text(
        2031, -2.5, "Net negative zone",
        fontsize=6, color="#888888", va="center", ha="left", style="italic",
    )

    # Carbon neutral reference line
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", zorder=1,
               label="Carbon neutral")

    ax.set_xlim(2028, 2063)
    ax.set_xticks(YEARS)
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual emissions (% of unabated)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", fontsize=6.5, frameon=False, ncol=2,
              handlelength=2.0, handletextpad=0.6)

    fig.tight_layout()
    save_fig(fig, "main_fig1_emissions_trajectory")


# ── Figure 2 ──────────────────────────────────────────────────────────────────

# Display categories: (label, [keys], is_negative)
# Negative bars (credits) are drawn left of zero.
COST_GROUPS: list[tuple[str, list[str]]] = [
    ("Carbon cost",    ["carbon_cost"]),
    ("Biomass + blend",["biomass_cost", "blend_upgrade_capex"]),
    ("CCS invest",     ["ccs_retrofit_capex", "pipe_capex"]),
    ("CCS operate",    ["ccs_om_cost", "energy_penalty_cost",
                        "storage_cost", "transport_opex"]),
    ("Stranded value", ["stranded_capex"]),
    ("Rebuild capex",  ["rebuild_capex"]),
    ("Water cost",     ["water_cost"]),
    ("Other O&M",      ["incremental_om", "baseline_net_cost", "ammonia_cost"]),
    ("Coal savings",   ["coal_savings_credit"]),
    ("Slack penalty",  ["slack_penalty"]),
]

# Colors: positive groups first (9), negative group last
GROUP_COLORS: list[str] = [
    "#CC3311",   # Carbon cost
    "#228833",   # Biomass + blend
    "#4477AA",   # CCS invest
    "#0077BB",   # CCS operate
    "#CCBB44",   # Stranded value
    "#AA4477",   # Rebuild capex
    "#44AA99",   # Water cost
    "#999999",   # Other O&M
    "#66CCEE",   # Coal savings (credit, shown left of zero)
    "#661111",   # Slack penalty (infeasibility indicator)
]

_B = 1e9  # billion CNY divisor


def _group_values(cost: dict[str, float]) -> list[float]:
    """Return summed value in billion CNY for each group in COST_GROUPS."""
    vals: list[float] = []
    for _label, keys in COST_GROUPS:
        vals.append(sum(cost.get(k, 0.0) for k in keys) / _B)
    return vals


def main_fig2_cost_breakdown() -> None:
    """BASE scenario 4-period cost waterfall (horizontal bars, shared y-axis)."""

    data = _load_data()
    base = data["BASE"]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.18, right=0.94, top=0.84, bottom=0.10)

    group_labels = [label for label, _ in COST_GROUPS]
    n_groups = len(group_labels)
    y_pos = np.arange(n_groups)
    bar_height = 0.55

    panels = list(zip("abcd", YEARS, axes))

    for panel_letter, year, ax in panels:
        yr_str = str(year)
        yr_data = base["years"].get(yr_str, {})
        cost = yr_data.get("cost_breakdown", {})
        vals = _group_values(cost)

        total_b = sum(vals)

        # Estimate x-axis span for bar-width legibility check.
        # Use the positive and negative extents of all bars in this panel.
        pos_max = max((v for v in vals if v > 0), default=1.0)
        neg_min = min((v for v in vals if v < 0), default=-1.0)
        x_span = pos_max - neg_min  # total axis width in data units

        # Draw bars
        for i, (val, color) in enumerate(zip(vals, GROUP_COLORS)):
            if abs(val) < 1.0:  # skip items < 1B CNY
                continue
            ax.barh(
                y_pos[i], val,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )
            # Annotate bar value.
            # Compute perceived luminance to pick contrasting text color.
            r_ch = int(color[1:3], 16) / 255.0
            g_ch = int(color[3:5], 16) / 255.0
            b_ch = int(color[5:7], 16) / 255.0
            luminance = 0.2126 * r_ch + 0.7152 * g_ch + 0.0722 * b_ch
            is_dark_bar = luminance < 0.45
            # Skip annotation if bar is narrower than ~8% of the panel's x-range:
            # text at 5pt needs ~5 characters ≈ 8% of a 300dpi panel to be legible.
            bar_fraction = abs(val) / x_span
            min_fraction = 0.10 if is_dark_bar else 0.07
            if bar_fraction >= min_fraction:
                inside_offset = 3.0
                if val >= 0:
                    x_text = val - inside_offset
                    ha = "right"
                else:
                    x_text = val + inside_offset
                    ha = "left"
                txt_color = "white" if is_dark_bar else "#222222"
                ax.text(
                    x_text, y_pos[i],
                    f"{val:+.0f}B",
                    fontsize=5,
                    va="center",
                    ha=ha,
                    color=txt_color,
                    clip_on=False,
                    zorder=4,
                )

        # Zero reference line
        ax.axvline(0, color="black", linewidth=0.6, zorder=3)

        # Panel title
        ax.set_title(str(year), fontweight="bold", fontsize=9)

        # Total cost annotation
        ax.text(
            0.97, 0.02,
            f"Total: {total_b:+.0f}B CNY",
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="bottom",
            color="#333333",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("Cost (B CNY)", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)

        # y-axis labels only on leftmost panel
        if panel_letter == "a":
            ax.set_yticks(y_pos)
            ax.set_yticklabels(group_labels, fontsize=7)
            ax.spines["left"].set_visible(True)
        else:
            ax.tick_params(axis="y", left=False)

        panel_label(ax, panel_letter)

    # Shared legend (color patches)
    legend_handles = [
        mpatches.Patch(facecolor=GROUP_COLORS[i], edgecolor="none",
                       label=group_labels[i])
        for i in range(n_groups)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.57, 0.97),
    )

    save_fig(fig, "main_fig2_cost_breakdown")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    main_fig1_emissions_trajectory()
    main_fig2_cost_breakdown()


if __name__ == "__main__":
    main()
