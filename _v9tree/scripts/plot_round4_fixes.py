"""Comprehensive new analyses from expert review round 4.

Batch B: Storage utilization, Distance-vs-CCS scatter
Batch C: Enhanced water, biomass, heatmap
Batch D: Stranded asset analysis

Usage:
    python scripts/plot_round4_fixes.py
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
RES = ROOT / "results" / "BASE"
FIG = ROOT / "results" / "figures"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH B: New figures using fixed spatial data
# ═══════════════════════════════════════════════════════════════════════════════

def fig_storage_utilization():
    """Storage hub injectivity utilization (2050) — shows injectivity is binding constraint."""
    stor = pd.read_csv(RES / "storage_utilization.csv")
    s50 = stor[stor["year"] == 2050].copy()
    s50 = s50[s50["injectivity_mtpa"] > 0]  # exclude zero-injectivity hubs
    s50["util_pct"] = s50["injectivity_utilization"] * 100
    s50 = s50.sort_values("util_pct", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.0))

    # (a) Injectivity utilization bar chart
    y = np.arange(len(s50))
    colors = ["#CC3311" if u >= 99 else "#4477AA" if u > 0 else "#cccccc"
              for u in s50["util_pct"].values]
    ax1.barh(y, s50["util_pct"].values, color=colors, height=0.7, edgecolor="white", linewidth=0.2)
    ax1.set_yticks(y)
    labels = []
    for _, r in s50.iterrows():
        prov = r.get("province", "")
        prov_str = f" ({prov})" if pd.notna(prov) and prov else ""
        labels.append(f'{r["storage_hub_id"]}{prov_str}')
    ax1.set_yticklabels(labels, fontsize=4.5)
    ax1.set_xlabel("Injectivity utilization (%)")
    ax1.axvline(100, color="black", linewidth=0.5, linestyle="--")
    saturated = (s50["util_pct"] >= 99).sum()
    ax1.text(0.98, 0.02, f"{saturated} hubs saturated", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=7, color="#CC3311", fontweight="bold")
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=9, va="top")

    # (b) Capacity vs injectivity scatter
    s50_active = s50[s50["storage_use_mtpa"] > 0]
    ax2.scatter(s50_active["available_capacity_mt"] / 1000,
                s50_active["storage_use_mtpa"],
                s=40, c=["#CC3311" if u >= 99 else "#4477AA" for u in s50_active["util_pct"]],
                edgecolors="black", linewidth=0.3, alpha=0.7)
    ax2.set_xlabel("Available capacity (Gt)")
    ax2.set_ylabel("Injection rate 2050 (Mtpa)")
    ax2.set_xscale("log")
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9, va="top")

    # Annotate: large capacity + low utilization
    for _, r in s50_active.iterrows():
        if r["available_capacity_mt"] > 100000 and r["util_pct"] < 5:
            prov = r.get("province", r["storage_hub_id"])
            ax2.annotate(str(prov)[:8], (r["available_capacity_mt"] / 1000, r["storage_use_mtpa"]),
                        fontsize=5, ha="left", va="bottom")

    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(fc="#CC3311", label="Saturated (100%)"),
                         Patch(fc="#4477AA", label="Active (<100%)")],
               fontsize=6, loc="upper left")
    fig.tight_layout()
    _save(fig, "ed_storage_utilization")


def fig_distance_vs_ccs():
    """Distance to nearest storage vs CCS adoption (2050)."""
    detail = pd.read_csv(RES / "plant_detail.csv")
    d50 = detail[detail["year"] == 2050].copy()
    d50["dist"] = pd.to_numeric(d50["min_distance_to_storage_km"], errors="coerce")
    d50 = d50.dropna(subset=["dist"])
    d50["ccs_total"] = d50["share_ccs"] + d50["share_beccs"]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Color by dominant pathway
    for pw, color in [("retire", "#999999"), ("ccs", "#4477AA"), ("beccs", "#CCBB44"), ("biomass", "#228833")]:
        mask = d50["dominant_pathway"] == pw
        if mask.any():
            sub = d50[mask]
            ax.scatter(sub["dist"], sub["ccs_total"] * 100,
                      s=sub["capacity_mw"] / 200, c=color, alpha=0.6,
                      edgecolors="black", linewidth=0.2, label=pw.upper())

    ax.set_xlabel("Distance to nearest storage (km)")
    ax.set_ylabel("CCS + BECCS share (%)")
    ax.set_xlim(0, d50["dist"].max() * 1.05)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=6, markerscale=2, loc="center right")
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")

    # Add threshold annotation
    ccs_plants = d50[d50["ccs_total"] > 0.5]
    if len(ccs_plants) > 0:
        max_ccs_dist = ccs_plants["dist"].max()
        ax.axvline(max_ccs_dist, color="#CC3311", linewidth=0.6, linestyle=":")
        ax.text(max_ccs_dist + 5, 50, f"Max CCS\ndist: {max_ccs_dist:.0f} km",
                fontsize=6, color="#CC3311")

    fig.tight_layout()
    _save(fig, "ed_distance_vs_ccs")


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH C: Enhanced existing figures
# ═══════════════════════════════════════════════════════════════════════════════

def fig_provincial_heatmap_enhanced():
    """Provincial heatmap with mixed pathway markers and capacity bar."""
    pp = pd.read_csv(RES / "province_pathways.csv")
    years = sorted(pp["year"].unique())

    rows = []
    for (yr, prov), grp in pp.groupby(["year", "province_name"]):
        total_gen = grp["province_generation_mwh"].iloc[0]
        if total_gen == 0:
            continue
        dominant = grp.loc[grp["generation_share"].idxmax()]
        # Check for mixed pathways (secondary > 10%)
        secondary_max = grp[grp["pathway"] != dominant["pathway"]]["generation_share"].max()
        has_mixed = secondary_max > 0.10 if pd.notna(secondary_max) else False
        rows.append({
            "year": yr, "province": prov,
            "dominant": dominant["pathway"],
            "share": dominant["generation_share"],
            "total_gw": total_gen / (0.55 * 8760) / 1000,
            "mixed": has_mixed,
            "secondary_share": secondary_max if pd.notna(secondary_max) else 0,
        })
    df = pd.DataFrame(rows)

    # Sort by 2030 capacity
    cap_order = df[df["year"] == years[0]].sort_values("total_gw", ascending=True)["province"].tolist()
    provinces = [p for p in cap_order if p in df["province"].unique()]

    pw_to_num = {"unabated": 0, "biomass": 1, "ccs": 2, "retire": 3, "beccs": 4, "ammonia": 5}
    pw_colors = ["#BBBBBB", "#228833", "#4477AA", "#999999", "#CCBB44", "#EE6677"]
    cmap = mcolors.ListedColormap(pw_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    heat = np.full((len(provinces), len(years)), np.nan)
    share_text = np.full((len(provinces), len(years)), "", dtype=object)
    mixed_markers = []

    for _, row in df.iterrows():
        if row["province"] in provinces:
            yi = years.index(row["year"])
            pi = provinces.index(row["province"])
            heat[pi, yi] = pw_to_num.get(row["dominant"], pw_to_num["retire"])
            share_text[pi, yi] = f"{row['share'] * 100:.0f}%"
            if row["mixed"]:
                mixed_markers.append((yi, pi))

    fig, (ax_main, ax_cap) = plt.subplots(1, 2, figsize=(6.0, 7.0),
                                            gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05})

    im = ax_main.imshow(heat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    for i in range(len(provinces)):
        for j in range(len(years)):
            txt = share_text[i, j]
            if txt:
                color = "white" if heat[i, j] in (0, 1) else "black"
                ax_main.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=color)

    # Mixed pathway markers (small triangle)
    for (yj, pi) in mixed_markers:
        ax_main.plot(yj + 0.35, pi - 0.35, marker="v", color="white",
                    markersize=4, markeredgecolor="black", markeredgewidth=0.3)

    ax_main.set_xticks(range(len(years)))
    ax_main.set_xticklabels([str(y) for y in years])
    ax_main.set_yticks(range(len(provinces)))
    ax_main.set_yticklabels(provinces, fontsize=6)
    ax_main.set_xlabel("Year")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#228833", label="Biomass"),
        Patch(facecolor="#4477AA", label="CCS"),
        Patch(facecolor="#999999", label="Retirement"),
        Patch(facecolor="#CCBB44", label="BECCS"),
    ]
    ax_main.legend(handles=legend_elements, loc="upper center",
                  bbox_to_anchor=(0.5, 1.06), ncol=4, fontsize=6.5)

    # Capacity bar (right side)
    cap_2030 = df[df["year"] == years[0]].set_index("province")["total_gw"]
    cap_vals = [cap_2030.get(p, 0) for p in provinces]
    ax_cap.barh(range(len(provinces)), cap_vals, color="#cccccc", edgecolor="white", height=0.7)
    ax_cap.set_yticks([])
    ax_cap.set_xlabel("GW")
    ax_cap.set_xlim(0, max(cap_vals) * 1.1 if cap_vals else 1)
    ax_cap.tick_params(axis="x", labelsize=6)
    ax_cap.spines["left"].set_visible(False)

    fig.tight_layout()
    _save(fig, "new_provincial_heatmap")


def fig_biomass_feasibility_enhanced():
    """Biomass feasibility: 2030+2040 comparison + node saturation."""
    resource = pd.read_csv(RES / "resource_use.csv")
    flows = pd.read_csv(RES / "biomass_flows.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.5),
                              gridspec_kw={"width_ratios": [2, 2, 1.5]})
    ax1, ax2, ax3 = axes

    # (a) Provincial supply vs demand — 2030
    bio30 = resource[(resource["resource_type"] == "biomass") & (resource["year"] == 2030)]
    prov30 = bio30.groupby("province_name")[["used", "available"]].sum().reset_index()
    prov30["utilization"] = np.where(prov30["available"] > 0,
                                      prov30["used"] / prov30["available"] * 100, 0)
    prov30["province_en"] = prov30["province_name"].map(CN_TO_EN).fillna(prov30["province_name"])
    prov30 = prov30[prov30["available"] > 0].sort_values("utilization", ascending=False).head(12)

    x = np.arange(len(prov30))
    ax1.barh(x, prov30["available"].values / 1e9, color="#cccccc", label="Available")
    ax1.barh(x, prov30["used"].values / 1e9, color="#228833", alpha=0.8, label="Used")
    ax1.set_yticks(x)
    ax1.set_yticklabels(prov30["province_en"].values, fontsize=5.5)
    ax1.set_xlabel("Biomass (B GJ)")
    ax1.legend(fontsize=5.5)
    for i, (_, r) in enumerate(prov30.iterrows()):
        if r["utilization"] > 50:
            ax1.text(r["used"] / 1e9 + 0.01, i, f'{r["utilization"]:.0f}%',
                    va="center", fontsize=5, color="#CC3311", fontweight="bold")
    ax1.text(0.02, 0.98, "(a) 2030", transform=ax1.transAxes, fontweight="bold", fontsize=8, va="top")

    # (b) Transport distance — 2030 vs 2040
    for yr, color, ls in [(2030, "#228833", "-"), (2040, "#4477AA", "--")]:
        fl = flows[flows["year"] == yr]
        if len(fl) > 0:
            d = fl["distance_km"].values
            w = fl["flow_gj"].values
            bins = np.linspace(0, 200, 25)
            hist, edges = np.histogram(d, bins=bins, weights=w)
            hist = hist / hist.sum() * 100
            centers = (edges[:-1] + edges[1:]) / 2
            ax2.plot(centers, hist, color=color, linestyle=ls, linewidth=1.2, label=str(yr))
            wmean = np.average(d, weights=w)
            ax2.axvline(wmean, color=color, linewidth=0.6, linestyle=":")
            ax2.text(wmean + 2, ax2.get_ylim()[1] * 0.9 if yr == 2030 else ax2.get_ylim()[1] * 0.75,
                    f"Mean {yr}: {wmean:.0f}km", fontsize=5, color=color)

    ax2.set_xlabel("Transport distance (km)")
    ax2.set_ylabel("Share of flow (%)")
    ax2.legend(fontsize=6)
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=8, va="top")

    # (c) Node saturation count by province
    bio_nodes = resource[(resource["resource_type"] == "biomass") & (resource["year"] == 2030)]
    bio_nodes["saturated"] = bio_nodes["utilization"] >= 0.99
    prov_sat = bio_nodes.groupby("province_name").agg(
        total=("utilization", "count"),
        saturated=("saturated", "sum")
    ).reset_index()
    prov_sat["province_en"] = prov_sat["province_name"].map(CN_TO_EN).fillna(prov_sat["province_name"])
    prov_sat["sat_pct"] = prov_sat["saturated"] / prov_sat["total"] * 100
    prov_sat = prov_sat.sort_values("sat_pct", ascending=True).tail(12)

    y3 = np.arange(len(prov_sat))
    ax3.barh(y3, prov_sat["sat_pct"].values, color="#CC3311", alpha=0.7, height=0.6)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(prov_sat["province_en"].values, fontsize=5.5)
    ax3.set_xlabel("Nodes at 100% (%)")
    ax3.set_xlim(0, 100)
    ax3.text(0.02, 0.98, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=8, va="top")

    fig.tight_layout()
    _save(fig, "new_biomass_feasibility")


def fig_water_penalty_enhanced():
    """Water penalty with water stress context."""
    detail = pd.read_csv(RES / "plant_detail.csv")

    d30 = detail[detail["year"] == 2030][["plant_id", "water_use_m3", "province_name", "capacity_mw"]].rename(
        columns={"water_use_m3": "water_2030"})
    d50 = detail[detail["year"] == 2050][["plant_id", "water_use_m3", "dominant_pathway"]].rename(
        columns={"water_use_m3": "water_2050"})
    merged = d30.merge(d50, on="plant_id").dropna()
    merged = merged[(merged["water_2030"] > 0) & (merged["water_2050"] > 0)]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0),
                              gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    ax1, ax2, ax3 = axes

    # (a) Scatter: 2030 vs 2050 water
    for pw, color in PATHWAY_COLORS.items():
        mask = merged["dominant_pathway"] == pw
        if mask.any():
            sub = merged[mask]
            ax1.scatter(sub["water_2030"] / 1e6, sub["water_2050"] / 1e6,
                       s=15, c=color, alpha=0.6, label=pw.capitalize(), edgecolors="none")
    lim = max(merged["water_2030"].max(), merged["water_2050"].max()) / 1e6 * 1.1
    ax1.plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Water 2030 (M m$^3$)")
    ax1.set_ylabel("Water 2050 (M m$^3$)")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.05, color="red")
    ax1.legend(fontsize=5, markerscale=2)
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=8, va="top")

    # (b) CCS water increase distribution
    ccs_plants = merged[merged["dominant_pathway"].isin(["ccs", "beccs"])]
    if len(ccs_plants) > 0:
        pct_increase = (ccs_plants["water_2050"] - ccs_plants["water_2030"]) / ccs_plants["water_2030"] * 100
        pct_increase = pct_increase[pct_increase.between(-50, 200)]
        ax2.hist(pct_increase, bins=20, color="#4477AA", alpha=0.7, edgecolor="white")
        med = pct_increase.median()
        ax2.axvline(med, color="black", linewidth=1, linestyle="--")
        ax2.text(med + 2, ax2.get_ylim()[1] * 0.9, f"Median: +{med:.0f}%",
                fontsize=6, va="top")
    ax2.set_xlabel("Water change (%)")
    ax2.set_ylabel("Number of CCS plants")
    ax2.axvline(0, color="grey", linewidth=0.5)
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=8, va="top")

    # (c) Provincial water intensity (2050) — top provinces
    prov_water = detail[detail["year"] == 2050].groupby("province_name").agg(
        water_m3=("water_use_m3", "sum"),
        gen_mwh=("annual_generation_mwh", "sum"),
    ).reset_index()
    prov_water["intensity"] = prov_water["water_m3"] / prov_water["gen_mwh"]
    prov_water = prov_water[prov_water["gen_mwh"] > 0].sort_values("intensity", ascending=True).tail(10)

    y3 = np.arange(len(prov_water))
    ax3.barh(y3, prov_water["intensity"].values, color="#4477AA", alpha=0.7, height=0.6)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(prov_water["province_name"].values, fontsize=5.5)
    ax3.set_xlabel("Water intensity\n(m$^3$/MWh)")
    ax3.text(0.02, 0.98, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=8, va="top")

    fig.tight_layout()
    _save(fig, "new_water_penalty")


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH D: Stranded asset analysis
# ═══════════════════════════════════════════════════════════════════════════════

def fig_stranded_assets():
    """Stranded asset quantification: early retirement capacity and value."""
    detail = pd.read_csv(RES / "plant_detail.csv")
    cost = pd.read_csv(RES / "plant_cost.csv")

    # Find plants that retire before their design retirement year
    years = sorted(detail["year"].unique())
    plants_2030 = detail[detail["year"] == 2030][
        ["plant_id", "province_name", "capacity_mw", "retirement_year"]
    ].copy()

    # Track when each plant actually retires (share_retire > 0.9)
    actual_retire = {}
    for yr in years:
        yr_data = detail[detail["year"] == yr]
        for _, r in yr_data.iterrows():
            pid = r["plant_id"]
            if pid not in actual_retire and r["share_retire"] > 0.9:
                actual_retire[pid] = yr

    plants_2030["actual_retire_year"] = plants_2030["plant_id"].map(actual_retire)
    plants_2030["early_retire"] = plants_2030["actual_retire_year"] < plants_2030["retirement_year"]
    plants_2030["years_early"] = plants_2030["retirement_year"] - plants_2030["actual_retire_year"]
    plants_2030["years_early"] = plants_2030["years_early"].clip(lower=0)

    # Stranded value: years_early × capacity × stranded_asset_base (3500 CNY/kW) / accounting_life (20 yr)
    plants_2030["stranded_value_b"] = (plants_2030["years_early"] * plants_2030["capacity_mw"]
                                        * 3500 / 20) / 1e6  # B CNY

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 3.0))

    # (a) Retirement wave — cumulative retired capacity by year
    cum_gw = {}
    for yr in years:
        yr_data = detail[detail["year"] == yr]
        retired_gw = yr_data[yr_data["share_retire"] > 0.9]["capacity_mw"].sum() / 1000
        cum_gw[yr] = retired_gw

    ax1.bar(list(cum_gw.keys()), list(cum_gw.values()), width=6,
            color="#999999", edgecolor="white")
    ax1.set_ylabel("Retired capacity (GW)")
    ax1.set_xlabel("Year")
    total_gw = plants_2030["capacity_mw"].sum() / 1000
    for yr, gw in cum_gw.items():
        ax1.text(yr, gw + total_gw * 0.02, f"{gw:.0f}", ha="center", fontsize=6)
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=8, va="top")

    # (b) Years of early retirement distribution
    early = plants_2030[plants_2030["years_early"] > 0]
    if len(early) > 0:
        bins = np.arange(0, early["years_early"].max() + 5, 5)
        ax2.hist(early["years_early"], bins=bins, weights=early["capacity_mw"] / 1000,
                color="#CC3311", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Years of early retirement")
    ax2.set_ylabel("Capacity (GW)")
    total_early_gw = early["capacity_mw"].sum() / 1000
    total_stranded_b = plants_2030["stranded_value_b"].sum()
    ax2.text(0.98, 0.98, f"Total: {total_early_gw:.0f} GW\n{total_stranded_b:.0f} B CNY",
            transform=ax2.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=8, va="top")

    # (c) Top 10 provinces by stranded value
    prov_strand = plants_2030.groupby("province_name").agg(
        stranded_gw=("capacity_mw", lambda x: x[plants_2030.loc[x.index, "years_early"] > 0].sum() / 1000),
        stranded_value=("stranded_value_b", "sum"),
    ).reset_index().sort_values("stranded_value", ascending=True).tail(10)

    y3 = np.arange(len(prov_strand))
    ax3.barh(y3, prov_strand["stranded_value"].values, color="#CC3311", alpha=0.7, height=0.6)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(prov_strand["province_name"].values, fontsize=5.5)
    ax3.set_xlabel("Stranded value (B CNY)")
    ax3.text(0.02, 0.98, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=8, va="top")

    fig.tight_layout()
    _save(fig, "ed_stranded_assets")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("Generating round 4 figures...")
    fig_storage_utilization()
    fig_distance_vs_ccs()
    fig_provincial_heatmap_enhanced()
    fig_biomass_feasibility_enhanced()
    fig_water_penalty_enhanced()
    fig_stranded_assets()
    print(f"\nAll round 4 figures saved to {FIG}")


if __name__ == "__main__":
    main()
