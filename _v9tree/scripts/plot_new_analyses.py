"""New analysis figures from expert review round 3.

1. Provincial decarbonization trajectory heatmap
2. CCS water penalty analysis
3. Biomass supply feasibility

Usage:
    python scripts/plot_new_analyses.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path

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

PATHWAY_COLORS = {
    "unabated": "#BBBBBB",
    "retire": "#999999", "ccs": "#4477AA", "biomass": "#228833",
    "beccs": "#CCBB44", "ammonia": "#EE6677",
}

# Chinese → English province name mapping for resource_use.csv
CN_TO_EN = {
    "安徽省": "Anhui", "北京市": "Beijing", "重庆市": "Chongqing",
    "福建省": "Fujian", "甘肃省": "Gansu", "广东省": "Guangdong",
    "广西壮族自治区": "Guangxi", "贵州省": "Guizhou", "海南省": "Hainan",
    "河北省": "Hebei", "黑龙江省": "Heilongjiang", "河南省": "Henan",
    "湖北省": "Hubei", "湖南省": "Hunan", "内蒙古自治区": "Inner Mongolia",
    "江苏省": "Jiangsu", "江西省": "Jiangxi", "吉林省": "Jilin",
    "辽宁省": "Liaoning", "宁夏回族自治区": "Ningxia", "青海省": "Qinghai",
    "陕西省": "Shaanxi", "山东省": "Shandong", "上海市": "Shanghai",
    "山西省": "Shanxi", "四川省": "Sichuan", "天津市": "Tianjin",
    "西藏自治区": "Tibet", "新疆维吾尔自治区": "Xinjiang",
    "新疆维吾尔族自治区": "Xinjiang", "云南省": "Yunnan", "浙江省": "Zhejiang",
}

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


# ── Fig 1: Provincial Decarbonization Trajectory Heatmap ─────────────────────
def fig_provincial_heatmap():
    pp = pd.read_csv(RESULTS_DIR / "BASE" / "province_pathways.csv")
    years = sorted(pp["year"].unique())

    # For each province-year, find dominant pathway and its share
    rows = []
    for (yr, prov), grp in pp.groupby(["year", "province_name"]):
        total_gen = grp["province_generation_mwh"].iloc[0]
        if total_gen == 0:
            continue
        dominant = grp.loc[grp["generation_share"].idxmax()]
        rows.append({
            "year": yr, "province": prov,
            "dominant": dominant["pathway"],
            "share": dominant["generation_share"],
            "total_gw": total_gen / (0.55 * 8760) / 1000,  # approx GW
        })
    df = pd.DataFrame(rows)

    # Sort provinces by total capacity (2030)
    cap_order = df[df["year"] == years[0]].sort_values("total_gw", ascending=True)["province"].tolist()
    provinces = [p for p in cap_order if p in df["province"].unique()]

    # Build heatmap arrays
    pw_to_num = {"unabated": 0, "biomass": 1, "ccs": 2, "retire": 3, "beccs": 4, "ammonia": 5}
    pw_colors = ["#BBBBBB", "#228833", "#4477AA", "#999999", "#CCBB44", "#EE6677"]
    cmap = mcolors.ListedColormap(pw_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    heat = np.full((len(provinces), len(years)), np.nan)
    share_text = np.full((len(provinces), len(years)), "", dtype=object)

    for _, row in df.iterrows():
        if row["province"] in provinces:
            yi = years.index(row["year"])
            pi = provinces.index(row["province"])
            heat[pi, yi] = pw_to_num.get(row["dominant"], pw_to_num["retire"])
            share_text[pi, yi] = f"{row['share'] * 100:.0f}%"

    fig, ax = plt.subplots(figsize=(5.0, 7.0))
    im = ax.imshow(heat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # Annotate cells
    for i in range(len(provinces)):
        for j in range(len(years)):
            txt = share_text[i, j]
            if txt:
                # White text on dark backgrounds, black on light
                color = "white" if heat[i, j] in (0, 1) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=color)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years])
    ax.set_yticks(range(len(provinces)))
    ax.set_yticklabels(provinces, fontsize=6)
    ax.set_xlabel("Year")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l.capitalize())
                       for l, c in PATHWAY_COLORS.items() if l != "ammonia"]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, 1.06), ncol=4, fontsize=6.5)

    fig.tight_layout()
    _save(fig, "new_provincial_heatmap")


# ── Fig 2: CCS Water Penalty ────────────────────────────────────────────────
def fig_water_penalty():
    detail = pd.read_csv(RESULTS_DIR / "BASE" / "plant_detail.csv")

    # Panel (a): Plant-level water change 2030 vs 2050
    d30 = detail[detail["year"] == 2030][["plant_id", "water_use_m3", "province_name"]].rename(
        columns={"water_use_m3": "water_2030"})
    d50 = detail[detail["year"] == 2050][["plant_id", "water_use_m3", "dominant_pathway"]].rename(
        columns={"water_use_m3": "water_2050"})
    merged = d30.merge(d50, on="plant_id").dropna()
    # Only keep plants that had water use in both periods
    merged = merged[(merged["water_2030"] > 0) & (merged["water_2050"] > 0)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # (a) Scatter: 2030 vs 2050 water use
    for pw, color in PATHWAY_COLORS.items():
        mask = merged["dominant_pathway"] == pw
        if mask.any():
            sub = merged[mask]
            ax1.scatter(sub["water_2030"] / 1e6, sub["water_2050"] / 1e6,
                       s=15, c=color, alpha=0.6, label=pw.capitalize(), edgecolors="none")

    # y=x line
    lim = max(merged["water_2030"].max(), merged["water_2050"].max()) / 1e6 * 1.1
    ax1.plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Water use 2030 (M m$^3$/yr)")
    ax1.set_ylabel("Water use 2050 (M m$^3$/yr)")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.legend(fontsize=6, markerscale=2)
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=9, va="top")

    # Annotate water increase region
    ax1.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.05, color="red")
    ax1.text(lim * 0.15, lim * 0.85, "Water\nincrease", fontsize=6, color="#CC3311", alpha=0.7)

    # (b) Provincial total water by year
    prov_water = detail.groupby(["year", "province_name"])["water_use_m3"].sum().reset_index()
    prov_30 = prov_water[prov_water["year"] == 2030].set_index("province_name")["water_use_m3"]
    prov_50 = prov_water[prov_water["year"] == 2050].set_index("province_name")["water_use_m3"]

    # Top 10 provinces by 2050 water use
    top10 = prov_50.nlargest(10)
    provs = top10.index.tolist()
    x = np.arange(len(provs))
    w = 0.35

    ax2.bar(x - w / 2, [prov_30.get(p, 0) / 1e6 for p in provs], w,
            label="2030", color="#228833", alpha=0.7)
    ax2.bar(x + w / 2, [prov_50.get(p, 0) / 1e6 for p in provs], w,
            label="2050", color="#4477AA", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(provs, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Water consumption (M m$^3$/yr)")
    ax2.legend(fontsize=6)
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9, va="top")

    fig.tight_layout()
    _save(fig, "new_water_penalty")


# ── Fig 3: Biomass Supply Feasibility ────────────────────────────────────────
def fig_biomass_feasibility():
    resource = pd.read_csv(RESULTS_DIR / "BASE" / "resource_use.csv")
    flows = pd.read_csv(RESULTS_DIR / "BASE" / "biomass_flows.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # (a) Provincial biomass supply vs demand (2030)
    bio = resource[(resource["resource_type"] == "biomass") & (resource["year"] == 2030)]
    prov_bio = bio.groupby("province_name")[["used", "available"]].sum().reset_index()
    prov_bio["utilization"] = np.where(prov_bio["available"] > 0,
                                        prov_bio["used"] / prov_bio["available"] * 100, 0)
    # Map Chinese province names to English
    prov_bio["province_en"] = prov_bio["province_name"].map(CN_TO_EN).fillna(prov_bio["province_name"])
    prov_bio = prov_bio[prov_bio["available"] > 0].sort_values("utilization", ascending=False).head(15)

    x = np.arange(len(prov_bio))
    ax1.barh(x, prov_bio["available"].values / 1e9, color="#cccccc", label="Available")
    ax1.barh(x, prov_bio["used"].values / 1e9, color="#228833", alpha=0.8, label="Used")
    ax1.set_yticks(x)
    ax1.set_yticklabels(prov_bio["province_en"].values, fontsize=6)
    ax1.set_xlabel("Biomass supply (B GJ)")
    ax1.legend(fontsize=6)
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=9, va="top")

    # Highlight provinces > 50% utilization
    for i, (_, row) in enumerate(prov_bio.iterrows()):
        if row["utilization"] > 50:
            ax1.text(row["used"] / 1e9 + 0.02, i, f'{row["utilization"]:.0f}%',
                    va="center", fontsize=5.5, color="#CC3311", fontweight="bold")

    # (b) Biomass transport distance distribution (2030)
    fl30 = flows[flows["year"] == 2030]
    if len(fl30) > 0:
        distances = fl30["distance_km"].values
        weights = fl30["flow_gj"].values
        ax2.hist(distances, bins=30, weights=weights / weights.sum() * 100,
                color="#228833", alpha=0.7, edgecolor="white")

        # Weighted statistics
        wmean = np.average(distances, weights=weights)
        wp90 = np.percentile(np.repeat(distances, (weights / weights.min()).astype(int)), 90)
        ax2.axvline(wmean, color="black", linewidth=1, linestyle="--")
        ax2.text(wmean + 2, ax2.get_ylim()[1] * 0.9, f"Mean: {wmean:.0f} km",
                fontsize=6, va="top")
        ax2.axvline(wp90, color="#CC3311", linewidth=0.8, linestyle=":")
        ax2.text(wp90 + 2, ax2.get_ylim()[1] * 0.75, f"P90: {wp90:.0f} km",
                fontsize=6, color="#CC3311", va="top")

    ax2.set_xlabel("Transport distance (km)")
    ax2.set_ylabel("Share of biomass flow (%)")
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9, va="top")

    fig.tight_layout()
    _save(fig, "new_biomass_feasibility")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating new analysis figures...")
    fig_provincial_heatmap()
    fig_water_penalty()
    fig_biomass_feasibility()
    print(f"\nNew figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
