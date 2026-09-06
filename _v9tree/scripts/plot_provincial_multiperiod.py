"""Multi-period provincial pathway bar chart grouped by 6 regions.

Recommended by provincial policy analyst as "the single most impactful
figure for policymakers." Shows 2030/2040/2050 side by side.

Usage:
    python scripts/plot_provincial_multiperiod.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    "font.family": "SimHei", "axes.unicode_minus": False, "font.size": 8, "mathtext.default": "regular",
    "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.facecolor": "white",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

PATHWAY_COLORS = {
    "unabated": "#BBBBBB",
    "retire": "#999999", "ccs": "#4477AA", "biomass": "#228833",
    "beccs": "#CCBB44", "ammonia": "#EE6677",
}
PATHWAY_ORDER = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]
PATHWAY_LABELS = {
    "unabated": "Unabated",
    "retire": "Retirement", "ccs": "CCS", "biomass": "Biomass",
    "beccs": "BECCS", "ammonia": "Ammonia",
}

# 6-region grouping
REGIONS = {
    "North": ["Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia"],
    "Northeast": ["Liaoning", "Jilin", "Heilongjiang"],
    "Northwest": ["Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"],
    "East": ["Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong"],
    "Southwest": ["Chongqing", "Sichuan", "Guizhou", "Yunnan"],
    "South-Central": ["Henan", "Hubei", "Hunan", "Guangdong", "Guangxi", "Hainan"],
}

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "BASE"
FIG = ROOT / "results" / "figures"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


def fig_provincial_multiperiod():
    """3-period provincial stacked bar chart grouped by 6 regions."""
    pp = pd.read_csv(RES / "province_pathways.csv")
    years = [2030, 2040, 2050]

    # Build province-year-pathway share table
    pivot = pp.pivot_table(index=["province_name", "year"], columns="pathway",
                            values="generation_share", fill_value=0).reset_index()

    # Build ordered province list by region, sorted by 2050 CCS share within each region
    ccs_2050 = pivot[pivot["year"] == 2050].set_index("province_name")
    ccs_2050["ccs_total"] = ccs_2050.get("ccs", 0) + ccs_2050.get("beccs", 0)

    ordered_provinces = []
    region_boundaries = []
    region_labels = []
    for region_name, provs in REGIONS.items():
        avail = [p for p in provs if p in ccs_2050.index]
        avail_sorted = sorted(avail, key=lambda p: ccs_2050.loc[p, "ccs_total"], reverse=True)
        region_boundaries.append(len(ordered_provinces))
        region_labels.append(region_name)
        ordered_provinces.extend(avail_sorted)

    n_prov = len(ordered_provinces)
    n_years = len(years)
    bar_width = 0.25
    group_width = n_years * bar_width + 0.15

    # 7.5 in = 190.5 mm nominal and 185.4 mm saved -- over the 183 mm limit.
    fig, ax = plt.subplots(figsize=(7.1, 5.5))

    for yi, yr in enumerate(years):
        yr_data = pivot[pivot["year"] == yr].set_index("province_name")
        x_positions = np.arange(n_prov) * group_width + yi * bar_width
        bottom = np.zeros(n_prov)

        for pw in PATHWAY_ORDER:
            if pw not in yr_data.columns:
                continue
            vals = np.array([yr_data.loc[p, pw] * 100 if p in yr_data.index else 0
                            for p in ordered_provinces])
            label = f"{PATHWAY_LABELS[pw]} ({yr})" if yi == 0 else ""
            ax.bar(x_positions, vals, bar_width, bottom=bottom,
                   color=PATHWAY_COLORS[pw], edgecolor="white", linewidth=0.1,
                   label=label if pw == "retire" and yi == 0 else
                         (PATHWAY_LABELS[pw] if yi == 0 else ""))
            bottom += vals

    # X-axis labels
    x_centers = np.arange(n_prov) * group_width + (n_years - 1) * bar_width / 2
    ax.set_xticks(x_centers)
    ax.set_xticklabels(ordered_provinces, rotation=90, fontsize=5.5)
    ax.set_ylabel("Generation share (%)")
    ax.set_ylim(0, 105)

    # Region dividers and labels
    for i, (boundary, label) in enumerate(zip(region_boundaries, region_labels)):
        if i > 0:
            x_div = boundary * group_width - group_width * 0.3
            ax.axvline(x_div, color="#cccccc", linewidth=0.5, linestyle="-")
        # Region label at top
        if i < len(region_boundaries) - 1:
            x_mid = (boundary + region_boundaries[i + 1]) / 2 * group_width
        else:
            x_mid = (boundary + n_prov) / 2 * group_width
        ax.text(x_mid, 103, label, ha="center", va="bottom", fontsize=6.5,
                fontstyle="italic", color="#666666")

    # Year labels in legend area
    from matplotlib.patches import Patch
    legend_handles = [Patch(fc=PATHWAY_COLORS[pw], label=PATHWAY_LABELS[pw])
                      for pw in PATHWAY_ORDER if pw != "ammonia"]
    # Add year indicators
    for yi, yr in enumerate(years):
        legend_handles.append(Patch(fc="none", ec="black", linewidth=0.5,
                                     label=f"Bar {yi+1} = {yr}"))

    ax.legend(handles=legend_handles, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), fontsize=6)

    fig.tight_layout()
    _save(fig, "ed_provincial_multiperiod")


def fig_provincial_typology_summary():
    """Compact summary: province typology + key stats."""
    pp = pd.read_csv(RES / "province_pathways.csv")
    detail = pd.read_csv(RES / "plant_detail.csv")

    # 2050 shares
    s50 = pp[pp["year"] == 2050].pivot_table(
        index="province_name", columns="pathway", values="generation_share", fill_value=0
    ).reset_index()
    s50["ccs_total"] = s50.get("ccs", 0) + s50.get("beccs", 0)

    # Classify
    def classify(row):
        if row["retire"] >= 0.99:
            return "Full Retiree"
        elif row["retire"] >= 0.80:
            return "Retire-Dominant"
        elif row["ccs_total"] >= 0.40:
            return "CCS-Heavy"
        elif row.get("beccs", 0) >= 0.05:
            return "BECCS-Notable"
        else:
            return "CCS-Moderate"

    s50["typology"] = s50.apply(classify, axis=1)

    # Get provincial capacity
    cap = detail[detail["year"] == 2030].groupby("province_name")["capacity_mw"].sum() / 1000
    s50["capacity_gw"] = s50["province_name"].map(cap)

    # Average distance to storage
    dist = detail[detail["year"] == 2050].groupby("province_name")["min_distance_to_storage_km"].mean()
    s50["avg_dist_km"] = s50["province_name"].map(dist)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # (a) CCS share vs distance, colored by typology
    type_colors = {
        "Full Retiree": "#999999", "Retire-Dominant": "#bbbbbb",
        "CCS-Moderate": "#4477AA", "CCS-Heavy": "#003366", "BECCS-Notable": "#CCBB44",
    }
    for typ, color in type_colors.items():
        mask = s50["typology"] == typ
        if mask.any():
            sub = s50[mask]
            ax1.scatter(sub["avg_dist_km"], sub["ccs_total"] * 100,
                       s=sub["capacity_gw"] * 2, c=color, alpha=0.7,
                       edgecolors="black", linewidth=0.3, label=typ)
            # Label a few key provinces
            for _, r in sub.iterrows():
                if r["ccs_total"] > 0.40 or r["capacity_gw"] > 80 or r["typology"] == "BECCS-Notable":
                    ax1.annotate(r["province_name"][:6], (r["avg_dist_km"], r["ccs_total"] * 100),
                                fontsize=5.0, ha="left", va="bottom")

    ax1.set_xlabel("Avg. distance to storage (km)")
    ax1.set_ylabel("CCS + BECCS share 2050 (%)")
    ax1.legend(fontsize=5.5, markerscale=0.8, loc="upper right")
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=9, va="top")

    # (b) Typology count + capacity
    type_summary = s50.groupby("typology").agg(
        n=("province_name", "count"),
        total_gw=("capacity_gw", "sum"),
    ).reindex(["Full Retiree", "Retire-Dominant", "CCS-Moderate", "CCS-Heavy", "BECCS-Notable"])
    type_summary = type_summary.dropna()

    y = np.arange(len(type_summary))
    bars = ax2.barh(y, type_summary["total_gw"].values,
                    color=[type_colors[t] for t in type_summary.index],
                    height=0.6, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{t}\n({n} prov.)" for t, n in
                         zip(type_summary.index, type_summary["n"].values)], fontsize=6)
    ax2.set_xlabel("Total fleet capacity (GW)")
    for bar, gw in zip(bars, type_summary["total_gw"].values):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{gw:.0f}", va="center", fontsize=6)
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9, va="top")

    fig.tight_layout()
    _save(fig, "ed_provincial_typology")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("Generating provincial analysis figures...")
    fig_provincial_multiperiod()
    fig_provincial_typology_summary()
    print(f"\nFigures saved to {FIG}")


if __name__ == "__main__":
    main()
