"""Main text figures (Fig 1-6) for coal retrofit paper.

Generates 6 publication figures with consistent styling from plot_style:
  main_fig1_pathway_allocation  — Stacked bar, pathway shares by period
  main_fig2_cost_waterfall      — 4-period cost component bars
  main_fig3_ammonia_dominance   — 3-panel: marginal value + breakeven + phase diagram
  main_fig4_provincial_heatmap  — 29 provinces × 4 years matrix + GW sidebar
  main_fig5_co2_network         — EPSG:2380 pipeline map
  main_fig6_water_penalty       — 3-panel: scatter + CCS Δ% + provincial intensity

Usage:
    python scripts/plot_main.py
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
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (
    apply_style, save_fig, panel_label, panel_label_inside,
    PATHWAY_COLORS, PATHWAY_LABELS, PATHWAY_ORDER, pathway_legend,
    clean_shares, format_pct, annotate_value,
    SINGLE_COL, DOUBLE_COL, HALF_PAGE, FULL_PAGE,
    ROOT, RESULTS_DIR, FIGURES_DIR, BASE_DIR,
    REGIONS, CN_TO_EN, EN_TO_CN,
    SCENARIO_COLORS, DIVERGING,
    MAP_SINGLE, MAP_DOUBLE,
)

apply_style()

YEARS = [2030, 2040, 2050, 2060]


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_json() -> dict:
    with open(RESULTS_DIR / "experiment_results_clean.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _base(data: dict) -> dict:
    return data.get("BASE")


def _shares(result: dict, year: int) -> dict:
    return result.get("years", {}).get(str(year), {}).get("pathway_shares", {})


def _cost_bd(result: dict, year: int) -> dict:
    return result.get("years", {}).get(str(year), {}).get("cost_breakdown", {})


def fig6_water_penalty() -> None:
    """3-panel comparison of BASE and grid-water-constrained results."""
    base = pd.read_csv(BASE_DIR / "plant_detail.csv")
    water_path = RESULTS_DIR / "WA_grid_200km" / "plant_detail.csv"
    if not water_path.exists():
        print("  [skip] main_fig6_water_penalty: WA_grid_200km results not found")
        return
    water = pd.read_csv(water_path)
    required_share_cols = {f"share_{pw}" for pw in PATHWAY_ORDER}
    if not required_share_cols.issubset(set(water.columns)):
        print("  [skip] main_fig6_water_penalty: WA_grid_200km was not rerun with the current pathway schema")
        return

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(FULL_PAGE[0], 3.5))

    def _province_summary(frame: pd.DataFrame) -> pd.DataFrame:
        year_frame = frame[frame["year"] == 2050].copy()
        rows = []
        for province, group in year_frame.groupby("province_name"):
            gen = group["annual_generation_mwh"].astype(float)
            pathway_gen = {
                pw: float((gen * group[f"share_{pw}"].astype(float)).sum())
                for pw in PATHWAY_ORDER
            }
            dominant = max(pathway_gen, key=pathway_gen.get)
            rows.append({
                "province_name": province,
                "water_use_m3": float(group["water_use_m3"].sum()),
                "annual_generation_mwh": float(gen.sum()),
                "dominant_pathway": dominant,
            })
        return pd.DataFrame(rows)

    base50 = _province_summary(base)[["province_name", "water_use_m3"]].rename(
        columns={"water_use_m3": "water_base"}
    )
    water50 = _province_summary(water).rename(columns={"water_use_m3": "water_grid"})
    merged = base50.merge(
        water50[["province_name", "water_grid", "dominant_pathway"]],
        on="province_name",
        how="inner",
    )

    for pw in PATHWAY_ORDER:
        sub = merged[merged["dominant_pathway"] == pw]
        if len(sub) > 0:
            ax_a.scatter(
                sub["water_base"] / 1e6,
                sub["water_grid"] / 1e6,
                s=22,
                color=PATHWAY_COLORS[pw],
                alpha=0.6,
                label=PATHWAY_LABELS[pw],
                edgecolors="none",
            )

    lim = max(merged["water_base"].max(), merged["water_grid"].max()) / 1e6
    ax_a.plot([0, lim * 1.05], [0, lim * 1.05], "k--", linewidth=0.5, alpha=0.5)
    ax_a.set_xlabel("BASE provincial water use 2050 (M m3)")
    ax_a.set_ylabel("Water-constrained provincial use 2050 (M m3)")
    ax_a.legend(fontsize=5.5, markerscale=2)
    panel_label(ax_a, "a")

    x_b = np.arange(2)
    bottoms = np.zeros(2)
    for pw in PATHWAY_ORDER:
        vals = []
        for frame in (base, water):
            yr = frame[frame["year"] == 2050]
            total_gen = float(yr["annual_generation_mwh"].sum())
            share_col = f"share_{pw}"
            value = (
                float((yr["annual_generation_mwh"] * yr[share_col]).sum()) / total_gen
                if total_gen > 0 and share_col in yr
                else 0.0
            )
            vals.append(value * 100)
        ax_b.bar(
            x_b,
            vals,
            0.55,
            bottom=bottoms,
            color=PATHWAY_COLORS[pw],
            edgecolor="white",
            linewidth=0.3,
            label=PATHWAY_LABELS[pw],
        )
        bottoms += np.array(vals)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(["BASE", "Water"])
    ax_b.set_ylabel("Generation share in 2050 (%)")
    ax_b.set_ylim(0, 100)
    panel_label(ax_b, "b")

    def _province_intensity(frame: pd.DataFrame) -> pd.DataFrame:
        year_frame = frame[frame["year"] == 2050].copy()
        grouped = year_frame.groupby("province_name").agg(
            total_water=("water_use_m3", "sum"),
            total_gen=("annual_generation_mwh", "sum"),
        ).reset_index()
        grouped["intensity"] = np.where(
            grouped["total_gen"] > 0,
            grouped["total_water"] / grouped["total_gen"],
            np.nan,
        )
        return grouped[["province_name", "intensity"]]

    prov = _province_intensity(base).merge(
        _province_intensity(water),
        on="province_name",
        suffixes=("_base", "_water"),
    )
    prov["change_pct"] = np.where(
        prov["intensity_base"] > 0,
        (prov["intensity_water"] - prov["intensity_base"]) / prov["intensity_base"] * 100,
        np.nan,
    )
    prov = prov.dropna(subset=["change_pct"])
    prov = prov.reindex(prov["change_pct"].abs().sort_values().tail(15).index)

    y_c = np.arange(len(prov))
    colors = [
        DIVERGING["pos"] if value > 0 else DIVERGING["neg"]
        for value in prov["change_pct"]
    ]
    ax_c.barh(
        y_c,
        prov["change_pct"],
        color=colors,
        height=0.6,
        edgecolor="white",
        linewidth=0.3,
    )
    ax_c.axvline(0, color="black", linewidth=0.5)
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(prov["province_name"], fontsize=6)
    ax_c.set_xlabel("Water intensity change vs BASE (%)")
    panel_label(ax_c, "c")

    handles = [
        Patch(facecolor=PATHWAY_COLORS[pw], label=PATHWAY_LABELS[pw])
        for pw in PATHWAY_ORDER
    ]
    ax_b.legend(
        handles=handles,
        fontsize=5.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )

    fig.tight_layout()
    save_fig(fig, "main_fig6_water_penalty", subdir="main")


# ── Fig 1: Pathway Allocation ───────────────────────────────────────────────

def fig1_pathway_allocation(data: dict) -> None:
    """Stacked bar: operating capacity (excluding retired) in GW."""
    base = _base(data)

    # Read plant_detail for capacity data
    plant_detail = pd.read_csv(BASE_DIR / "plant_detail.csv")

    # Calculate operating capacity by pathway (excluding retire) by year
    cap_by_pathway = {yr: {pw: 0.0 for pw in PATHWAY_ORDER if pw != "retire"} for yr in YEARS}
    cap_operating = {yr: 0.0 for yr in YEARS}

    for yr in YEARS:
        yr_data = plant_detail[plant_detail["year"] == yr]

        for pw in PATHWAY_ORDER:
            if pw == "retire":
                continue
            share_col = f"share_{pw}"
            if share_col in yr_data.columns:
                cap = (yr_data["capacity_mw"] * yr_data[share_col]).sum() / 1000
                cap_by_pathway[yr][pw] = cap
                cap_operating[yr] += cap

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = np.arange(len(YEARS))
    width = 0.55
    bottoms = np.zeros(len(YEARS))

    for pw in PATHWAY_ORDER:
        if pw == "retire":
            continue
        vals = [cap_by_pathway[yr][pw] for yr in YEARS]
        ax.bar(x, vals, width, bottom=bottoms,
               color=PATHWAY_COLORS[pw], edgecolor="white",
               linewidth=0.3, label=PATHWAY_LABELS[pw])
        for i, v in enumerate(vals):
            if v > cap_operating[YEARS[i]] * 0.05:  # Only label if > 5% of operating
                c = "white" if v > cap_operating[YEARS[i]] * 0.15 else "black"
                ax.text(x[i], bottoms[i] + v / 2, f"{v:.0f}",
                        ha="center", va="center", fontsize=6.5,
                        fontweight="bold", color=c)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([str(yr) for yr in YEARS])
    ax.set_ylabel("Operating capacity (GW)")
    ax.set_xlim(-0.5, len(YEARS) - 0.5)
    pathway_legend(ax, ncol=4, loc="upper center", bbox=(0.5, 1.12))
    save_fig(fig, "main_fig1_pathway_allocation", subdir="main")


# ── Fig 2: Cost Waterfall ────────────────────────────────────────────────────

COST_ITEMS = [
    ("Baseline net", "baseline_net_cost"),
    ("Carbon cost", "carbon_cost"),
    ("Coal savings", "coal_savings_credit"),
    ("Energy penalty", "energy_penalty_cost"),
    ("CCS O&M", "ccs_om_cost"),
    ("Incremental O&M", "incremental_om"),
    ("Biomass fuel", "biomass_cost"),
    ("Ammonia fuel", "ammonia_cost"),
    ("Transport OPEX", "transport_opex"),
    ("Storage cost", "storage_cost"),
    ("Stranded CAPEX", "stranded_capex"),
    ("CCS CAPEX", "ccs_retrofit_capex"),
    ("Pipeline CAPEX", "pipe_capex"),
    ("Blend upgrade", "blend_upgrade_capex"),
]


def fig2_cost_waterfall(data: dict) -> None:
    """4-period cost component horizontal bars."""
    base = _base(data)
    fig, axes = plt.subplots(1, 4, figsize=(FULL_PAGE[0], 5.0))

    for idx, (yr, ax) in enumerate(zip(YEARS, axes)):
        cbd = _cost_bd(base, yr)
        labels, vals = [], []
        for label, key in COST_ITEMS:
            v = cbd.get(key, 0) / 1e9
            if abs(v) > 0.5:
                labels.append(label)
                vals.append(v)
        # Total
        total = sum(vals)
        labels.append("TOTAL")
        vals.append(total)

        colors = []
        for i, v in enumerate(vals):
            if labels[i] == "TOTAL":
                colors.append("#333333")
            elif v > 0:
                colors.append(DIVERGING["pos"])
            else:
                colors.append(DIVERGING["neg"])

        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, height=0.6, edgecolor="white",
                linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if idx == 0 else [], fontsize=6)
        ax.set_title(f"{yr}", fontsize=9, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.4)
        ax.set_xlabel("B CNY", fontsize=7)

        for i, v in enumerate(vals):
            if abs(v) > 15:
                ha = "left" if v > 0 else "right"
                offset = max(abs(v) * 0.03, 2)
                ax.text(v + (offset if v > 0 else -offset), i,
                        f"{v:,.0f}", fontsize=5, ha=ha, va="center")

        panel_label(ax, chr(ord("a") + idx),
                    x=-0.15 if idx == 0 else -0.05, y=1.05)

    fig.tight_layout()
    save_fig(fig, "main_fig2_cost_waterfall", subdir="main")


# ── Fig 3: Ammonia Dominance (3 panels) ─────────────────────────────────────

# Analytical LCOE parameters (from optimization model / plot_ammonia_analysis)
_HR = 9.0           # GJ/MWh heat rate
_NH3_LHV = 0.0186   # GJ/kg ammonia LHV
_COAL_COST = 35.0    # CNY/GJ coal fuel cost
_COAL_EF = 0.82      # tCO2/MWh emission factor
_CCS_CAP_RATE = 0.90
_CCS_CAPEX = 1200.0  # CNY/kW
_CCS_OM = 50.0       # CNY/MWh
_BIO_OM = 30.0        # CNY/MWh
_BIO_COST_GJ = 50.0   # CNY/GJ
_NH3_OM = 80.0         # CNY/MWh incremental O&M for ammonia
_CF = 0.55
_HOURS = 8760
_CRF = 0.06 * (1 + 0.06) ** 20 / ((1 + 0.06) ** 20 - 1)  # ~0.0872


def _ammonia_lcoe(cp, nh3_price, blend=0.20):
    nh3_kg = blend * _HR / _NH3_LHV
    fuel = nh3_kg * nh3_price
    coal_save = blend * _HR * _COAL_COST
    abatement = blend * _COAL_EF * cp
    return fuel - coal_save + _NH3_OM - abatement


def _biomass_lcoe(cp, blend=0.15):
    fuel = blend * _HR * _BIO_COST_GJ
    coal_save = blend * _HR * _COAL_COST
    abatement = blend * _COAL_EF * cp
    return fuel - coal_save + _BIO_OM - abatement


def _ccs_lcoe(cp):
    annual_capex = _CCS_CAPEX * 1000 * _CRF / (_CF * _HOURS)
    eff_pen = 0.08 * _HR * _COAL_COST
    storage_transport = 60.0 * _CCS_CAP_RATE * _COAL_EF
    abatement = _CCS_CAP_RATE * _COAL_EF * cp
    return annual_capex + _CCS_OM + eff_pen + storage_transport - abatement


def fig3_ammonia_dominance(data: dict) -> None:
    """3-panel: (a) marginal value, (b) breakeven, (c) phase diagram."""
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(FULL_PAGE[0], 3.5))

    # ── (a) Marginal value ──
    base_cost = _base(data)["global_objective_cny"] / 1e9
    # All available scenarios
    all_scenarios = [
        ("BASE", "BASE", "BASE"),
        ("BASE zero", "BASE_zero", None),
        ("BASE neg", "BASE_neg", None),
        ("No ammonia", "RQ3_no_ammonia", "RQ3_no_ammonia"),
        ("No CCS", "RQ3_no_ccs", None),
        ("No biomass", "RQ3_no_biomass", None),
    ]
    labels_a, deltas, colors_a, hatches_a = [], [], [], []
    for label, key1, key2 in all_scenarios:
        sc = data.get(key1) or (data.get(key2) if key2 else None)
        if sc is None:
            continue
        cost = sc["global_objective_cny"] / 1e9
        infeasible = sc.get("infeasible", False)
        delta = cost - base_cost
        pct = delta / base_cost * 100
        labels_a.append(f"{label}\n({pct:+.1f}%)")
        deltas.append(delta)
        colors_a.append(DIVERGING["neg"] if delta < 0 else DIVERGING["pos"])
        hatches_a.append("///" if infeasible else "")

    y_a = np.arange(len(labels_a))
    bars = ax_a.barh(y_a, deltas, color=colors_a, height=0.55,
                     edgecolor="white", linewidth=0.3)
    for bar, h in zip(bars, hatches_a):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor("black")
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels(labels_a, fontsize=6.5)
    ax_a.set_xlabel("$\\Delta$Cost vs BASE (B CNY)")
    ax_a.axvline(0, color="black", linewidth=0.4)
    for i, d in enumerate(deltas):
        ha = "left" if d > 0 else "right"
        offset = max(abs(d) * 0.03, 8)
        ax_a.text(d + (offset if d > 0 else -offset), i,
                  f"{d:+,.0f}B", fontsize=6, ha=ha, va="center")
    # Infeasible legend
    if any(hatches_a):
        ax_a.plot([], [], "s", color="#CC3311", markerfacecolor="white",
                  markeredgecolor="black", label="Infeasible")
        inf_patch = Patch(facecolor="#CC3311", edgecolor="black",
                          hatch="///", label="Infeasible", alpha=0.5)
        ax_a.legend(handles=[inf_patch], fontsize=5.5, loc="lower right")
    panel_label(ax_a, "a")

    # ── (b) Breakeven ammonia price ──
    cp_range = np.linspace(0, 1500, 300)
    blend_nh3 = 0.20
    nh3_kg = blend_nh3 * _HR / _NH3_LHV
    coal_sav = blend_nh3 * _HR * _COAL_COST
    co2_red = blend_nh3 * _COAL_EF

    competitors = [
        ("vs Biomass 15%", _biomass_lcoe(cp_range, 0.15), "#228833", "-"),
        ("vs Biomass 30%", _biomass_lcoe(cp_range, 0.30), "#228833", "--"),
        ("vs CCS retrofit", _ccs_lcoe(cp_range), "#4477AA", "-"),
    ]
    for label, comp_cost, color, ls in competitors:
        be = (comp_cost + coal_sav - _NH3_OM + co2_red * cp_range) / nh3_kg
        ax_b.plot(cp_range, be, color=color, linestyle=ls, linewidth=1.0,
                  label=label)

    # Shade competitive region
    bio15_be = (_biomass_lcoe(cp_range, 0.15) + coal_sav - _NH3_OM +
                co2_red * cp_range) / nh3_kg
    ccs_be = (_ccs_lcoe(cp_range) + coal_sav - _NH3_OM +
              co2_red * cp_range) / nh3_kg
    floor = np.minimum(bio15_be, ccs_be)
    ax_b.fill_between(cp_range, 0, np.clip(floor, 0, None),
                      alpha=0.08, color="#EE6677", label="NH$_3$ competitive")

    ax_b.axhline(3.0, color="#EE6677", linewidth=0.7, linestyle=":")
    ax_b.text(1480, 3.1, "Current NH$_3$\n(3 CNY/kg)", fontsize=5.5,
              ha="right", color="#EE6677")
    ax_b.axvline(80, color="#888888", linewidth=0.7, linestyle=":")
    ax_b.text(90, 4.8, "China\nETS", fontsize=5.5, color="#888888", va="top")

    ax_b.set_xlabel("Carbon price (CNY/tCO$_2$)")
    ax_b.set_ylabel("Break-even NH$_3$ price (CNY/kg)")
    ax_b.set_xlim(0, 1500)
    ax_b.set_ylim(0, 5)
    ax_b.legend(fontsize=5.5, loc="upper left")
    panel_label(ax_b, "b")

    # ── (c) Phase diagram ──
    cp_grid = np.linspace(0, 1500, 200)
    nh3_grid = np.linspace(0, 5, 200)
    CP, NH3P = np.meshgrid(cp_grid, nh3_grid)

    cost_bio = _biomass_lcoe(CP, 0.15)
    cost_ccs = _ccs_lcoe(CP)
    cost_nh3 = _ammonia_lcoe(CP, NH3P, 0.20)

    costs = np.stack([cost_bio, cost_ccs, cost_nh3], axis=0)
    winner = np.argmin(costs, axis=0)

    cmap_phase = ListedColormap(["#228833", "#4477AA", "#EE6677"])
    ax_c.pcolormesh(CP, NH3P, winner, cmap=cmap_phase, shading="auto",
                    alpha=0.5)
    ax_c.contour(CP, NH3P, winner, levels=[0.5, 1.5], colors="black",
                 linewidths=0.6)

    ax_c.text(200, 1.0, "Biomass", fontsize=8, fontweight="bold",
              color="#228833")
    ax_c.text(900, 4.0, "CCS", fontsize=8, fontweight="bold",
              color="#4477AA")
    nh3_wins = winner == 2
    if nh3_wins.any():
        nh3_ys, nh3_xs = np.where(nh3_wins)
        cx = cp_grid[int(np.mean(nh3_xs))]
        cy = nh3_grid[int(np.mean(nh3_ys))]
        ax_c.text(cx, cy, "NH$_3$", fontsize=7, fontweight="bold",
                  color="#EE6677", ha="center", va="center")
    else:
        ax_c.text(1400, 0.3, "NH$_3$\n(not viable)", fontsize=5.5,
                  color="#EE6677", ha="right", fontstyle="italic")

    ax_c.plot(80, 3.0, "k*", markersize=8, zorder=10)
    ax_c.annotate("Current\nprices", xy=(80, 3.0), xytext=(200, 3.8),
                  fontsize=6, arrowprops=dict(arrowstyle="-", color="black",
                                             lw=0.5))
    ax_c.set_xlabel("Carbon price (CNY/tCO$_2$)")
    ax_c.set_ylabel("NH$_3$ price (CNY/kg)")
    ax_c.set_xlim(0, 1500)
    ax_c.set_ylim(0, 5)
    handles_c = [Patch(facecolor=c, alpha=0.5, label=l)
                 for c, l in [("#228833", "Biomass"), ("#4477AA", "CCS"),
                               ("#EE6677", "Ammonia")]]
    ax_c.legend(handles=handles_c, fontsize=5.5, loc="upper right")
    panel_label(ax_c, "c")

    fig.tight_layout()
    save_fig(fig, "main_fig3_ammonia_dominance", subdir="main")


# ── Fig 4: Provincial Heatmap ───────────────────────────────────────────────

def fig4_provincial_heatmap() -> None:
    """29 provinces × 4 years heatmap + GW capacity sidebar."""
    df = pd.read_csv(BASE_DIR / "province_pathways.csv")

    records = []
    for (yr, prov), grp in df.groupby(["year", "province_name"]):
        if yr not in YEARS:
            continue
        total_gen = grp["province_generation_mwh"].iloc[0]
        if total_gen == 0:
            continue
        pw_shares = dict(zip(grp["pathway"], grp["generation_share"]))
        dominant = max(pw_shares, key=pw_shares.get)
        dominant_share = pw_shares[dominant]
        secondary_vals = sorted(pw_shares.values(), reverse=True)
        mixed = len(secondary_vals) > 1 and secondary_vals[1] > 0.10
        records.append({
            "province": prov, "year": yr, "dominant": dominant,
            "share": dominant_share, "mixed": mixed, "gen_mwh": total_gen,
        })

    rdf = pd.DataFrame(records)

    # Sort by 2030 capacity (ascending)
    gen_2030 = rdf[rdf["year"] == 2030].set_index("province")["gen_mwh"]
    cap_gw = gen_2030 / (0.55 * 8760) / 1e6
    cap_gw = cap_gw.sort_values(ascending=True)
    provinces = cap_gw.index.tolist()

    pw_code = {pw: idx for idx, pw in enumerate(PATHWAY_ORDER)}
    pw_cmap = ListedColormap([PATHWAY_COLORS[p] for p in PATHWAY_ORDER])

    n_prov, n_year = len(provinces), len(YEARS)
    grid = np.full((n_prov, n_year), np.nan)
    share_grid: dict = {}
    mixed_grid: dict = {}

    for _, row in rdf.iterrows():
        if row["province"] not in provinces:
            continue
        i = provinces.index(row["province"])
        j = YEARS.index(row["year"])
        grid[i, j] = pw_code.get(row["dominant"], pw_code["retire"])
        share_grid[(i, j)] = row["share"]
        mixed_grid[(i, j)] = row["mixed"]

    fig = plt.figure(figsize=(7.5, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax_heat = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax_heat.imshow(grid, cmap=pw_cmap, aspect="auto", vmin=-0.5, vmax=len(PATHWAY_ORDER) - 0.5,
                   interpolation="nearest")

    for i in range(n_prov):
        for j in range(n_year):
            s = share_grid.get((i, j), 0)
            if s > 0:
                dark_bg = grid[i, j] in [pw_code["biomass"], pw_code["ccs"], pw_code["ammonia"]]
                color = "white" if (dark_bg and s > 0.5) else "black"
                ax_heat.text(j, i, f"{s * 100:.0f}%", ha="center",
                             va="center", fontsize=5.5, fontweight="bold",
                             color=color)
            if mixed_grid.get((i, j), False):
                ax_heat.plot(j + 0.3, i + 0.3, "v", color="black",
                             markersize=3, zorder=5)

    ax_heat.set_xticks(range(n_year))
    ax_heat.set_xticklabels([str(yr) for yr in YEARS])
    ax_heat.set_yticks(range(n_prov))
    ax_heat.set_yticklabels(provinces, fontsize=6)
    ax_heat.tick_params(length=0)

    handles = [Patch(facecolor=PATHWAY_COLORS[pw], edgecolor="white",
                     linewidth=0.3, label=PATHWAY_LABELS[pw])
               for pw in PATHWAY_ORDER]
    handles.append(plt.Line2D([0], [0], marker="v", color="black",
                              linestyle="none", markersize=4,
                              label="Mixed (2nd >10%)"))
    ax_heat.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=6,
                   frameon=False)

    # Sidebar: capacity bars
    bar_vals = [cap_gw.get(p, 0) for p in provinces]
    ax_bar.barh(range(n_prov), bar_vals, color="#666666", height=0.7,
                edgecolor="white", linewidth=0.3)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("GW", fontsize=7)
    ax_bar.set_xlim(0, max(bar_vals) * 1.3 if bar_vals else 1)
    ax_bar.spines["left"].set_visible(False)
    for i, v in enumerate(bar_vals):
        if v > 3:
            ax_bar.text(v + 0.5, i, f"{v:.0f}", fontsize=5, va="center")

    save_fig(fig, "main_fig4_provincial_heatmap", subdir="main")


# ── Fig 5: CO2 Network ──────────────────────────────────────────────────────

def fig5_co2_network() -> None:
    """CO2 pipeline network map (EPSG:2380)."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("  [skip] geopandas not available — skipping fig5")
        return

    TARGET_CRS = "EPSG:2380"

    # Load shapefiles
    provinces = gpd.read_file(
        ROOT / "data" / "ChinaMap" / "provinces.shp"
    ).to_crs(TARGET_CRS)
    country = provinces.dissolve()

    # Province name mapping
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    provinces["province_en"] = provinces["NAME"].map(cn_to_en)

    # Projected bounds
    corners = gpd.GeoDataFrame(
        geometry=[Point(80, 15), Point(150, 50)], crs="EPSG:4326"
    ).to_crs(TARGET_CRS)
    scs_corners = gpd.GeoDataFrame(
        geometry=[Point(106.5, 2.8), Point(123, 24.5)], crs="EPSG:4326"
    ).to_crs(TARGET_CRS)

    def _reproj(df, lon_col, lat_col):
        valid = df.dropna(subset=[lon_col, lat_col])
        gdf = gpd.GeoDataFrame(
            valid,
            geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]),
            crs="EPSG:4326",
        ).to_crs(TARGET_CRS)
        gdf["px"] = gdf.geometry.x
        gdf["py"] = gdf.geometry.y
        return gdf

    # Load data
    co2 = pd.read_csv(BASE_DIR / "co2_flow_direction.csv")
    nodes = pd.read_csv(ROOT / "inputs" / "pipeline_nodes.csv")
    candidates = pd.read_csv(ROOT / "inputs" / "pipeline_candidate_edges.csv")
    plants = pd.read_csv(BASE_DIR / "plant_detail.csv")

    yr50_co2 = co2[co2["year"] == 2050]
    yr50_plants = plants[plants["year"] == 2050].copy()

    # Geometry lookup for curved corridor edges
    from shapely import wkt as shapely_wkt
    geom_lookup = candidates.set_index("edge_id")["geometry_wkt"].to_dict()

    nodes_proj = _reproj(nodes, "lon", "lat")
    node_xy = nodes_proj.set_index("node_id")[["px", "py"]].to_dict("index")
    plants_proj = _reproj(yr50_plants, "centroid_longitude",
                          "centroid_latitude")

    fig, ax = plt.subplots(figsize=(6, 6))
    provinces.plot(ax=ax, facecolor="none", edgecolor="black",
                   linewidth=0.2, zorder=1)
    country.plot(ax=ax, facecolor="none", edgecolor="black",
                 linewidth=0.75, zorder=2)
    ax.set_xlim(corners.geometry.iloc[0].x, corners.geometry.iloc[1].x)
    ax.set_ylim(corners.geometry.iloc[0].y, corners.geometry.iloc[1].y)
    ax.set_axis_off()

    # Pipeline edges — use curved geometry for corridor edges
    if len(yr50_co2) > 0:
        max_flow = max(yr50_co2["net_flow_mtpa"].max(), 0.01)
        for _, edge in yr50_co2.iterrows():
            if edge["net_flow_mtpa"] <= 0.01:
                continue
            lw = max(0.15, edge["net_flow_mtpa"] / max_flow * 1.8)
            drawn = False
            wkt_str = geom_lookup.get(edge.get("edge_id", ""), "")
            if isinstance(wkt_str, str) and wkt_str.startswith("LINESTRING"):
                try:
                    geom = shapely_wkt.loads(wkt_str)
                    gdf_e = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(TARGET_CRS)
                    xs, ys = zip(*gdf_e.geometry.iloc[0].coords)
                    ax.plot(list(xs), list(ys), color="#4477AA", linewidth=lw,
                            alpha=0.45, solid_capstyle="round", zorder=3)
                    drawn = True
                except Exception:
                    pass
            if not drawn:
                src = node_xy.get(edge["source_node"])
                snk = node_xy.get(edge["sink_node"])
                if src and snk:
                    ax.plot([src["px"], snk["px"]], [src["py"], snk["py"]],
                            color="#4477AA", linewidth=lw, alpha=0.45,
                            solid_capstyle="round", zorder=3)

    # Storage sinks
    stor_nodes = nodes_proj[
        nodes_proj["node_type"].str.contains("storage", case=False, na=False)
    ]
    storage_csv = pd.read_csv(BASE_DIR / "storage_utilization.csv")
    yr50_stor = storage_csv[storage_csv["year"] == 2050]
    active_hubs = yr50_stor[
        yr50_stor["storage_use_mtpa"] > 0
    ]["storage_hub_id"].tolist()
    active_stor = stor_nodes[stor_nodes["storage_hub_id"].isin(active_hubs)]
    if len(active_stor) > 0:
        ax.scatter(active_stor["px"], active_stor["py"], s=25,
                   c="#EE6677", marker="^", edgecolors="black",
                   linewidth=0.3, label="Storage sink", zorder=7)

    # Plants by pathway — skip retirement, distinguish CCS and BECCS
    ccs_only = plants_proj[(plants_proj["share_ccs"] > 0.1) & (plants_proj["share_beccs"] <= 0.01)]
    beccs = plants_proj[plants_proj["share_beccs"] > 0.01]
    bio = plants_proj[(plants_proj["share_biomass"] > 0.1)
                      & (plants_proj["share_ccs"] <= 0.1)
                      & (plants_proj["share_beccs"] <= 0.01)]

    if len(bio) > 0:
        ax.scatter(bio["px"], bio["py"], s=bio["capacity_mw"] / 500,
                   c="#228833", alpha=0.6, edgecolors="black", linewidth=0.15,
                   label="Biomass co-firing", zorder=4)
    if len(ccs_only) > 0:
        ax.scatter(ccs_only["px"], ccs_only["py"], s=ccs_only["capacity_mw"] / 500,
                   c="#4477AA", alpha=0.7, marker="s", edgecolors="black",
                   linewidth=0.2, label="CCS", zorder=5)
    if len(beccs) > 0:
        ax.scatter(beccs["px"], beccs["py"], s=beccs["capacity_mw"] / 400,
                   c="#CCBB44", alpha=0.7, marker="D", edgecolors="black",
                   linewidth=0.2, label="BECCS", zorder=6)

    # Basin labels
    basin_info = [
        (123, 44, "Songliao\nBasin"), (118, 38, "Bohai Bay\nBasin"),
        (110, 39, "Ordos\nBasin"), (105, 30, "Sichuan\nBasin"),
        (120, 34, "Subei\nBasin"),
    ]
    basin_pts = gpd.GeoDataFrame(
        basin_info, columns=["lon", "lat", "label"],
        geometry=[Point(lo, la) for lo, la, _ in basin_info],
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    for _, row in basin_pts.iterrows():
        ax.annotate(
            row["label"], xy=(row.geometry.x, row.geometry.y),
            fontsize=7, fontstyle="italic", color="#666666",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                      alpha=0.7),
            zorder=8,
        )

    # Source/sink legend
    ax.legend(loc="lower left", fontsize=7, markerscale=1.5, framealpha=0.9)

    # Pipeline flow legend
    if len(yr50_co2) > 0:
        from matplotlib.legend import Legend
        from matplotlib.lines import Line2D as L2D
        max_flow = max(yr50_co2["net_flow_mtpa"].max(), 0.01)
        ph = []
        for fl in [5, 10, 20]:
            lw_l = max(0.15, fl / max_flow * 1.8)
            ph.append(L2D([], [], color="#4477AA", linewidth=lw_l, alpha=0.5,
                          solid_capstyle="round", label=f"{fl} Mtpa"))
        pipe_leg = Legend(ax, ph, [h.get_label() for h in ph],
                         loc="upper right", fontsize=6,
                         title="CO$_2$ flow", title_fontsize=7,
                         framealpha=0.9, bbox_to_anchor=(0.99, 0.93))
        ax.add_artist(pipe_leg)

    # Scale bar (500 km)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([x0, x0 + 500_000], [y0, y0], "k-", linewidth=1.5, zorder=20)
    ax.text(x0 + 250_000, y0 - (ylim[1] - ylim[0]) * 0.02, "500 km",
            ha="center", va="top", fontsize=6, zorder=20)

    # North arrow
    ax.annotate("N", xy=(0.95, 0.95), xycoords="axes fraction",
                fontsize=9, fontweight="bold", ha="center")
    ax.annotate("", xy=(0.95, 0.93), xycoords="axes fraction",
                xytext=(0.95, 0.87), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # SCS inset
    pos = ax.get_position()
    inset = fig.add_axes([pos.x1 - 0.12, pos.y0 + 0.01, 0.11, 0.16])
    provinces.plot(ax=inset, facecolor="#f5f5f5", edgecolor="black",
                   linewidth=0.2)
    country.plot(ax=inset, facecolor="none", edgecolor="black",
                 linewidth=0.75)
    inset.set_xlim(scs_corners.geometry.iloc[0].x,
                   scs_corners.geometry.iloc[1].x)
    inset.set_ylim(scs_corners.geometry.iloc[0].y,
                   scs_corners.geometry.iloc[1].y)
    inset.set_xticks([])
    inset.set_yticks([])

    fig.tight_layout()
    save_fig(fig, "main_fig5_co2_network", subdir="main")


# ── Fig 6: Water Penalty ────────────────────────────────────────────────────

def _fig6_water_penalty_base_only() -> None:
    """3-panel: (a) scatter, (b) CCS water increase %, (c) provincial bars."""
    plant = pd.read_csv(BASE_DIR / "plant_detail.csv")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=FULL_PAGE)

    # (a) 2030 vs 2050 water use scatter
    p30 = plant[plant["year"] == 2030][
        ["plant_id", "water_use_m3", "dominant_pathway"]
    ].rename(columns={"water_use_m3": "water_2030"})
    p50 = plant[plant["year"] == 2050][
        ["plant_id", "water_use_m3", "dominant_pathway"]
    ].rename(columns={"water_use_m3": "water_2050",
                       "dominant_pathway": "pathway_2050"})
    merged = p30.merge(p50, on="plant_id", how="inner")

    for pw in PATHWAY_ORDER:
        sub = merged[merged["pathway_2050"] == pw]
        if len(sub) > 0:
            ax_a.scatter(sub["water_2030"] / 1e6, sub["water_2050"] / 1e6,
                         s=8, color=PATHWAY_COLORS[pw], alpha=0.6,
                         label=PATHWAY_LABELS[pw], edgecolors="none")

    lim = max(merged["water_2030"].max(), merged["water_2050"].max()) / 1e6
    ax_a.plot([0, lim * 1.05], [0, lim * 1.05], "k--", linewidth=0.5,
              alpha=0.5)
    ax_a.set_xlabel("Water use 2030 (M m³)")
    ax_a.set_ylabel("Water use 2050 (M m³)")
    ax_a.legend(fontsize=5.5, markerscale=2)
    panel_label(ax_a, "a")

    # (b) CCS water increase % distribution
    ccs_plants = merged[merged["pathway_2050"] == "ccs"].copy()
    ccs_plants["pct_change"] = np.where(
        ccs_plants["water_2030"] > 0,
        (ccs_plants["water_2050"] - ccs_plants["water_2030"])
        / ccs_plants["water_2030"] * 100,
        np.nan,
    )
    ccs_plants = ccs_plants.dropna(subset=["pct_change"])

    if len(ccs_plants) > 0:
        # Handle edge case where all values are identical
        unique_vals = ccs_plants["pct_change"].nunique()
        bins = min(20, max(5, unique_vals))
        ax_b.hist(ccs_plants["pct_change"], bins=bins,
                  color=PATHWAY_COLORS["ccs"], alpha=0.7,
                  edgecolor="white", linewidth=0.3)
        med = ccs_plants["pct_change"].median()
        ax_b.axvline(med, color="black", linewidth=1, linestyle="--")
        ax_b.text(med + 2, ax_b.get_ylim()[1] * 0.85,
                  f"Median: {med:+.0f}%", fontsize=6, fontweight="bold")

    ax_b.set_xlabel("Water use change (%)")
    ax_b.set_ylabel("Number of CCS plants")
    panel_label(ax_b, "b")

    # (c) Provincial water intensity (m³/MWh)
    p50_full = plant[plant["year"] == 2050].copy()
    prov_water = p50_full.groupby("province_name").agg(
        total_water=("water_use_m3", "sum"),
        total_gen=("annual_generation_mwh", "sum"),
    ).reset_index()
    prov_water["intensity"] = np.where(
        prov_water["total_gen"] > 0,
        prov_water["total_water"] / prov_water["total_gen"],
        np.nan,
    )
    prov_water = prov_water.dropna(subset=["intensity"])
    prov_water = prov_water.sort_values("intensity", ascending=True).tail(15)

    y_c = np.arange(len(prov_water))
    ax_c.barh(y_c, prov_water["intensity"], color="#4477AA", height=0.6,
              edgecolor="white", linewidth=0.3)
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(prov_water["province_name"], fontsize=6)
    ax_c.set_xlabel("Water intensity (m³/MWh)")
    panel_label(ax_c, "c")

    fig.tight_layout()
    save_fig(fig, "main_fig6_water_penalty", subdir="main")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading experiment results...")
    data = _load_json()

    print("Generating main text figures...")
    fig1_pathway_allocation(data)
    fig2_cost_waterfall(data)
    fig3_ammonia_dominance(data)
    fig4_provincial_heatmap()
    fig5_co2_network()
    fig6_water_penalty()

    print(f"\nAll main figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
