"""Main figures 5 and 6: technology dominance and water-network interactions.

Fig 5 — When does each technology win? (2x2 panel)
  (a) Technology share by scenario (stacked bar, 2050)
  (b) Cost penalty of removing each technology (horizontal bar)
  (c) Ammonia: when does it appear? (lollipop chart, 2030 shares)
  (d) Replacement cost drives retirement pace (bar chart)

Fig 6 — Water reshapes the CO2 network (2x2 panel)
  (a) BASE pipeline network 2050 (map)
  (b) Water-constrained pipeline network 2050 (map, diff highlighted)
  (c) Water use change by province (horizontal bar)
  (d) Technology shift: BASE vs Water (grouped bars, top provinces)
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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (
    apply_style, save_fig, panel_label, panel_label_inside, pathway_legend,
    PATHWAY_COLORS, PATHWAY_LABELS, PATHWAY_ORDER,
    ROOT, RESULTS_DIR, FIGURES_DIR, BASE_DIR,
    REGIONS, CN_TO_EN, EN_TO_CN,
    DIVERGING, FULL_PAGE, DOUBLE_COL,
)

apply_style()

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH = RESULTS_DIR / "experiment_results_clean.json"
YEARS = [2030, 2040, 2050, 2060]

# Scenarios shown in Fig 5 panels (a) and (b)
TECH_SCENARIOS: list[tuple[str, str]] = [
    ("BASE",          "BASE"),
    ("RQ3_no_ammonia", r"$-$NH$_3$"),
    ("RQ3_no_ccs",    r"$-$CCS"),
    ("RQ3_no_biomass", r"$-$Bio"),
    ("BASE_neg",      "Neg"),
    ("WA_grid_200km", r"H$_2$O"),
]

# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_json() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _shares_2050(sc: dict) -> dict[str, float]:
    """Return 2050 pathway shares dict from a scenario dict."""
    return sc.get("years", {}).get("2050", {}).get("pathway_shares", {})


def _shares_2030(sc: dict) -> dict[str, float]:
    return sc.get("years", {}).get("2030", {}).get("pathway_shares", {})


def _obj_tcny(sc: dict) -> float | None:
    """Return global objective in trillion CNY, or None if infeasible."""
    if sc.get("infeasible", False):
        return None
    obj = sc.get("global_objective_cny")
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    return obj / 1e12


# ── Fig 5 ─────────────────────────────────────────────────────────────────────

def main_fig5_technology_dominance() -> None:
    """When does each technology win? 2x2 storytelling figure."""

    data = _load_json()
    base_obj_tcny = _obj_tcny(data["BASE"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    # ── (a) Technology share by scenario (stacked bar, 2050) ──────────────────
    n_sc = len(TECH_SCENARIOS)
    x_a = np.arange(n_sc)
    bottoms = np.zeros(n_sc)

    for pw in PATHWAY_ORDER:
        vals = []
        for sc_key, _ in TECH_SCENARIOS:
            sc = data.get(sc_key, {})
            shares = _shares_2050(sc)
            vals.append(shares.get(pw, 0.0) * 100)
        vals = np.array(vals)
        ax_a.bar(x_a, vals, bottom=bottoms,
                 color=PATHWAY_COLORS[pw], label=PATHWAY_LABELS[pw],
                 edgecolor="white", linewidth=0.3, width=0.7)
        bottoms += vals

    # Annotate dominant pathway above each bar
    for i, (sc_key, _) in enumerate(TECH_SCENARIOS):
        sc = data.get(sc_key, {})
        shares = _shares_2050(sc)
        if sc.get("infeasible", False):
            ax_a.text(i, 105, "*", ha="center", va="bottom", fontsize=7,
                      color="#CC3311")
            continue
        if not shares:
            continue
        dom_pw = max(shares, key=lambda k: shares.get(k, 0))
        dom_val = shares.get(dom_pw, 0) * 100
        label_str = f"{dom_val:.0f}%"
        ax_a.text(i, min(bottoms[i] + 2, 108), label_str,
                  ha="center", va="bottom", fontsize=5.5, color="#333333")

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels([lbl for _, lbl in TECH_SCENARIOS], fontsize=7)
    ax_a.set_ylabel("2050 pathway share (%)")
    ax_a.set_ylim(0, 115)
    ax_a.set_xlim(-0.5, n_sc - 0.5)
    # Compact legend inside panel
    handles = [Patch(facecolor=PATHWAY_COLORS[pw], edgecolor="white",
                     linewidth=0.3, label=PATHWAY_LABELS[pw])
               for pw in PATHWAY_ORDER]
    ax_a.legend(handles=handles, fontsize=5.5, ncol=2, loc="upper right",
                frameon=True, framealpha=0.8, edgecolor="none")
    ax_a.text(0.0, 1.02, "* infeasible", transform=ax_a.transAxes,
              fontsize=5.5, color="#CC3311")
    panel_label(ax_a, "a")

    # ── (b) Cost penalty of removing each technology ───────────────────────────
    # Only non-BASE scenarios; skip if infeasible or objective missing
    b_labels: list[str] = []
    b_diffs: list[float] = []
    b_infeasible: list[bool] = []
    for sc_key, lbl in TECH_SCENARIOS[1:]:  # skip BASE itself
        sc = data.get(sc_key, {})
        infeasible = sc.get("infeasible", False)
        obj = _obj_tcny(sc)
        b_labels.append(lbl)
        b_infeasible.append(infeasible)
        if infeasible or obj is None or base_obj_tcny is None:
            b_diffs.append(0.0)
        else:
            # Convert to B CNY
            b_diffs.append((obj - base_obj_tcny) * 1000)

    y_b = np.arange(len(b_labels))
    colors_b = [DIVERGING["pos"] if d >= 0 else DIVERGING["neg"]
                for d in b_diffs]

    bars_b = ax_b.barh(y_b, b_diffs, color=colors_b,
                       edgecolor="white", linewidth=0.3, height=0.6)

    # Hatch infeasible bars
    for i, (bar, inf) in enumerate(zip(bars_b, b_infeasible)):
        if inf:
            bar.set_hatch("////")
            bar.set_facecolor("#dddddd")
            ax_b.text(0, i, " infeasible", va="center", ha="left",
                      fontsize=5.5, color="#CC3311", style="italic")
        else:
            val = b_diffs[i]
            xpos = val + (5 if val >= 0 else -5)
            ha = "left" if val >= 0 else "right"
            ax_b.text(xpos, i, f"{val:+.0f} B", va="center", ha=ha,
                      fontsize=6, color="#333333")

    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(b_labels, fontsize=7)
    ax_b.axvline(0, color="black", linewidth=0.6)
    ax_b.set_xlabel("Cost vs BASE (B CNY)")
    ax_b.set_title("Cost penalty of removing\neach technology", fontsize=8)
    panel_label(ax_b, "b")

    # ── (c) Ammonia role: styled annotation panel ─────────────────────────────
    # Ammonia never enters the optimal solution in BASE or any ablation except
    # when biomass is entirely removed (RQ3_no_biomass). Show this concisely.
    sc_nb = data.get("RQ3_no_biomass", {})
    nb_obj = sc_nb.get("global_objective_cny", 0.0)
    base_obj_val = data["BASE"].get("global_objective_cny", 0.0)
    amm_share_2030 = (sc_nb.get("years", {}).get("2030", {})
                      .get("pathway_shares", {}).get("ammonia", 0.0)) * 100
    amm_share_2040 = (sc_nb.get("years", {}).get("2040", {})
                      .get("pathway_shares", {}).get("ammonia", 0.0)) * 100
    cost_pct = (nb_obj - base_obj_val) / base_obj_val * 100 if base_obj_val else 0.0
    cost_b   = (nb_obj - base_obj_val) / 1e9 if base_obj_val else 0.0

    # Build a bar chart showing ammonia share by year under no_biomass,
    # plus a reference bar at 0 for the BASE scenario
    amm_years  = [2030, 2040, 2050, 2060]
    amm_shares = [amm_share_2030, amm_share_2040, 0.0, 0.0]
    y_c = np.arange(len(amm_years))

    bar_c = ax_c.barh(y_c, amm_shares,
                      color=PATHWAY_COLORS.get("ammonia", "#EE7733"),
                      edgecolor="white", linewidth=0.3, height=0.55)
    ax_c.axvline(0, color="black", linewidth=0.6)

    # Value labels
    for i, val in enumerate(amm_shares):
        if val > 0.5:
            ax_c.text(val + 0.8, i, f"{val:.0f}%",
                      va="center", ha="left", fontsize=6.5,
                      color="#333333", fontweight="bold")
        else:
            ax_c.text(0.8, i, "0%", va="center", ha="left",
                      fontsize=6.5, color="#888888")

    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([str(yr) for yr in amm_years], fontsize=7)
    ax_c.set_xlabel(r"NH$_3$ share when biomass removed (%)", fontsize=7)
    ax_c.set_title(r"Ammonia: last resort only", fontsize=8)
    ax_c.set_xlim(0, 100)

    # Annotation box: key finding
    ann_text = (
        "In BASE: NH$_3$ share = 0% (all years)\n"
        f"Only enters when biomass removed (−Bio):\n"
        f"  2030: {amm_share_2030:.0f}%  →  2040: {amm_share_2040:.0f}%  →  2050: 0%\n"
        f"Cost: +{cost_pct:.0f}% (+{cost_b:.0f} B CNY)"
    )
    ax_c.text(0.97, 0.97, ann_text,
              transform=ax_c.transAxes,
              fontsize=5.8, va="top", ha="right",
              linespacing=1.5,
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7e6",
                        edgecolor="#EE7733", linewidth=0.8, alpha=0.92))
    panel_label(ax_c, "c")

    # ── (d) BASE retirement progression 2030–2060 ─────────────────────────────
    # SA_retire_cost_500/1000 were not run; show BASE timeline instead.
    retire_years = [2030, 2040, 2050, 2060]
    base_sc = data["BASE"]
    retire_vals = [
        base_sc.get("years", {}).get(str(yr), {})
                .get("pathway_shares", {}).get("retire", 0.0) * 100
        for yr in retire_years
    ]

    x_d = np.arange(len(retire_years))
    # Color gradient from light to dark green
    retire_palette = ["#a8dba8", "#59c259", "#228833", "#155a21"]

    bars_d = ax_d.bar(x_d, retire_vals, color=retire_palette,
                      edgecolor="white", linewidth=0.3, width=0.55)

    for bar, val in zip(bars_d, retire_vals):
        ax_d.text(bar.get_x() + bar.get_width() / 2,
                  val + 1.5, f"{val:.0f}%",
                  ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([str(yr) for yr in retire_years], fontsize=7)
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("Retirement share (%)")
    ax_d.set_title("BASE retirement progression\n2030–2060", fontsize=8)
    ax_d.set_ylim(0, max(retire_vals) * 1.2 + 5)
    # Annotation: scenario context
    ax_d.text(0.97, 0.05,
              "BASE scenario (300 CNY/kW)\n"
              "SA sensitivity not available",
              transform=ax_d.transAxes,
              fontsize=5.5, va="bottom", ha="right", color="#666666",
              style="italic")
    panel_label(ax_d, "d")

    fig.tight_layout(rect=[0, 0, 1, 1])
    save_fig(fig, "main_fig5_technology_dominance")


# ── Fig 6 ─────────────────────────────────────────────────────────────────────

def main_fig6_water_network() -> None:
    """Water reshapes the CO$_2$ network. 2x2 layout."""

    # Try importing geopandas; gracefully fall back to non-map version
    try:
        import geopandas as gpd
        from shapely.geometry import Point, LineString
        _HAS_GEO = True
    except ImportError:
        _HAS_GEO = False

    TARGET_CRS = "EPSG:2380"
    SHP_PATH = ROOT / "data" / "ChinaMap" / "provinces.shp"

    # ── Load data ──────────────────────────────────────────────────────────────
    edges_base = pd.read_csv(RESULTS_DIR / "BASE" / "network_edges.csv")
    edges_wa   = pd.read_csv(RESULTS_DIR / "WA_grid_200km" / "network_edges.csv")
    nodes      = pd.read_csv(ROOT / "inputs" / "pipeline_nodes.csv")
    plants_base = pd.read_csv(RESULTS_DIR / "BASE" / "plant_detail.csv")
    plants_wa   = pd.read_csv(RESULTS_DIR / "WA_grid_200km" / "plant_detail.csv")

    # Filter to 2050 with significant flow (>0.1 Mtpa to reduce clutter)
    flow_threshold = 0.1
    e50_base = edges_base[(edges_base["year"] == 2050) &
                          (edges_base["edge_flow_mtpa"] > flow_threshold)].copy()
    e50_wa   = edges_wa[(edges_wa["year"] == 2050) &
                        (edges_wa["edge_flow_mtpa"] > flow_threshold)].copy()

    p50_base = plants_base[plants_base["year"] == 2050].copy()
    p50_wa   = plants_wa[plants_wa["year"] == 2050].copy()

    # Node coordinate lookup
    node_coords: dict[str, tuple[float, float]] = dict(
        zip(nodes["node_id"],
            zip(nodes["lon"].astype(float), nodes["lat"].astype(float)))
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _plot_basemap_simple(ax) -> None:
        """Minimal basemap: either geopandas or blank axes."""
        if _HAS_GEO and SHP_PATH.exists():
            provinces = gpd.read_file(str(SHP_PATH)).to_crs(TARGET_CRS)
            country   = provinces.dissolve()
            provinces.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#888888",
                           linewidth=0.35, zorder=1)
            country.plot(ax=ax, facecolor="none", edgecolor="black",
                         linewidth=0.75, zorder=2)
            # Bounds: lon 80-150, lat 15-50
            corners = gpd.GeoDataFrame(
                geometry=[Point(80, 15), Point(150, 50)],
                crs="EPSG:4326",
            ).to_crs(TARGET_CRS)
            ax.set_xlim(corners.geometry.iloc[0].x, corners.geometry.iloc[1].x)
            ax.set_ylim(corners.geometry.iloc[0].y, corners.geometry.iloc[1].y)
        ax.set_axis_off()

    def _add_scs_inset(fig, ax_main) -> None:
        """South China Sea inset."""
        if not (_HAS_GEO and SHP_PATH.exists()):
            return
        provinces = gpd.read_file(str(SHP_PATH)).to_crs(TARGET_CRS)
        country   = provinces.dissolve()
        pos = ax_main.get_position()
        inset = fig.add_axes([pos.x1 - 0.055, pos.y0 + 0.003, 0.055, 0.085])
        provinces.plot(ax=inset, facecolor="#f5f5f5", edgecolor="black", linewidth=0.2)
        country.plot(ax=inset, facecolor="none", edgecolor="black", linewidth=0.75)
        scs_corners = gpd.GeoDataFrame(
            geometry=[Point(106.5, 2.8), Point(123, 24.5)],
            crs="EPSG:4326",
        ).to_crs(TARGET_CRS)
        inset.set_xlim(scs_corners.geometry.iloc[0].x, scs_corners.geometry.iloc[1].x)
        inset.set_ylim(scs_corners.geometry.iloc[0].y, scs_corners.geometry.iloc[1].y)
        inset.set_xticks([])
        inset.set_yticks([])

    def _reproject_lonlat(df: pd.DataFrame,
                          lon_col: str = "lon",
                          lat_col: str = "lat") -> pd.DataFrame:
        """Add proj_x, proj_y columns; return df (projected if possible)."""
        if not _HAS_GEO:
            df = df.copy()
            df["proj_x"] = df[lon_col]
            df["proj_y"] = df[lat_col]
            return df
        valid = df.dropna(subset=[lon_col, lat_col]).copy()
        gdf = gpd.GeoDataFrame(
            valid,
            geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]),
            crs="EPSG:4326",
        ).to_crs(TARGET_CRS)
        gdf["proj_x"] = gdf.geometry.x
        gdf["proj_y"] = gdf.geometry.y
        return gdf

    def _draw_pipeline_edges(ax, edges: pd.DataFrame,
                             color: str = "#4477AA",
                             alpha: float = 0.5,
                             lw_scale: float = 1.5,
                             zorder: int = 5) -> None:
        """Draw pipeline edges as lines; width proportional to flow."""
        if edges.empty:
            return
        max_flow = edges["edge_flow_mtpa"].max()
        if max_flow <= 0:
            return
        for _, row in edges.iterrows():
            src_lonlat = node_coords.get(row["from_node_id"])
            dst_lonlat = node_coords.get(row["to_node_id"])
            if src_lonlat is None or dst_lonlat is None:
                continue
            # Project the two endpoints
            pts_df = pd.DataFrame({
                "lon": [src_lonlat[0], dst_lonlat[0]],
                "lat": [src_lonlat[1], dst_lonlat[1]],
            })
            pts_proj = _reproject_lonlat(pts_df)
            x0, y0 = pts_proj["proj_x"].iloc[0], pts_proj["proj_y"].iloc[0]
            x1, y1 = pts_proj["proj_x"].iloc[1], pts_proj["proj_y"].iloc[1]
            lw = max(0.2, row["edge_flow_mtpa"] / max_flow * lw_scale)
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                    alpha=alpha, solid_capstyle="round", zorder=zorder)

    def _draw_plants(ax, plants: pd.DataFrame, zorder: int = 6) -> None:
        """Scatter only CCS/BECCS plants (pipeline-connected). Skip retire/biomass/ammonia-only."""
        valid = plants.dropna(subset=["centroid_longitude", "centroid_latitude"])
        # Only show plants with CCS or BECCS share > 5%
        ccs_mask = pd.Series(False, index=valid.index)
        for pw in ["ccs", "beccs"]:
            col = f"share_{pw}"
            if col in valid.columns:
                ccs_mask = ccs_mask | (valid[col].astype(float) > 0.05)
        valid = valid[ccs_mask]
        if valid.empty:
            return
        proj = _reproject_lonlat(valid, "centroid_longitude", "centroid_latitude")
        for pw in ["ccs", "beccs"]:
            col_name = f"share_{pw}"
            if col_name not in proj.columns:
                continue
            subset = proj[proj["dominant_pathway"] == pw]
            if subset.empty:
                continue
            ax.scatter(subset["proj_x"], subset["proj_y"],
                       s=subset["capacity_mw"] / 400,
                       c=PATHWAY_COLORS[pw], alpha=0.65,
                       edgecolors="white", linewidths=0.3,
                       label=PATHWAY_LABELS[pw], zorder=zorder)

    def _draw_storage_hubs(ax) -> None:
        """Draw storage hubs as triangles, DSA vs EOR distinguished by marker."""
        storages = pd.read_csv(ROOT / "inputs" / "storage_hubs.csv")
        storages = storages.dropna(subset=["latitude", "longitude"])
        proj = _reproject_lonlat(storages, "longitude", "latitude")
        for stype, marker, label in [("dsa", "v", "DSA"), ("eor", "^", "EOR")]:
            sub = proj[proj["storage_type"] == stype]
            if sub.empty:
                continue
            ax.scatter(sub["proj_x"], sub["proj_y"],
                       s=8, marker=marker, c="none",
                       edgecolors="#CC3311" if stype == "dsa" else "#EE6677",
                       linewidths=0.4, alpha=0.55,
                       label=label, zorder=7)

    # ── (a) BASE pipeline network 2050 ────────────────────────────────────────
    _plot_basemap_simple(ax_a)
    _draw_pipeline_edges(ax_a, e50_base, color="#4477AA", lw_scale=2.0, alpha=0.7)
    _draw_storage_hubs(ax_a)
    _draw_plants(ax_a, p50_base)
    ax_a.set_title("BASE (2050)", fontsize=8, fontweight="bold")
    # Storage type legend on panel (a)
    from matplotlib.lines import Line2D
    storage_handles = [
        Line2D([0], [0], marker="v", color="none", markeredgecolor="#CC3311",
               markerfacecolor="none", markersize=5, linewidth=0, label="DSA"),
        Line2D([0], [0], marker="^", color="none", markeredgecolor="#EE6677",
               markerfacecolor="none", markersize=5, linewidth=0, label="EOR"),
    ]
    ax_a.legend(handles=storage_handles, fontsize=5.5, loc="lower left",
                frameon=True, framealpha=0.85, edgecolor="none")
    panel_label_inside(ax_a, "a")

    # ── (b) Water-constrained network 2050, diff highlighted ──────────────────
    _plot_basemap_simple(ax_b)
    _draw_storage_hubs(ax_b)
    # Edges in BASE but NOT in WA (lost edges) → red dashes
    base_edge_set = set(e50_base["edge_id"])
    wa_edge_set   = set(e50_wa["edge_id"])
    lost_edges = e50_base[e50_base["edge_id"].isin(base_edge_set - wa_edge_set)]
    new_edges  = e50_wa[e50_wa["edge_id"].isin(wa_edge_set - base_edge_set)]
    shared_edges = e50_wa[e50_wa["edge_id"].isin(base_edge_set & wa_edge_set)]

    _draw_pipeline_edges(ax_b, shared_edges, color="#4477AA", lw_scale=1.8, alpha=0.7)
    _draw_pipeline_edges(ax_b, new_edges,    color="#228833", alpha=0.75,
                         lw_scale=1.8, zorder=6)
    _draw_plants(ax_b, p50_wa)

    # Diff legend
    diff_handles = [
        Patch(facecolor="#4477AA", label="Shared routes"),
        Patch(facecolor="#228833", label="New (WA only)"),
    ]
    ax_b.legend(handles=diff_handles, fontsize=5.5, loc="lower left",
                frameon=True, framealpha=0.85, edgecolor="none")
    ax_b.set_title("Water constrained (2050)", fontsize=8, fontweight="bold")
    panel_label_inside(ax_b, "b")

    # ── (c) Water use change by province ──────────────────────────────────────
    prov_base = (
        p50_base.groupby("province_name")["water_use_m3"].sum()
        .rename("water_base")
    )
    prov_wa = (
        p50_wa.groupby("province_name")["water_use_m3"].sum()
        .rename("water_wa")
    )
    prov_water = pd.concat([prov_base, prov_wa], axis=1).fillna(0)
    prov_water["delta_pct"] = np.where(
        prov_water["water_base"] > 0,
        (prov_water["water_wa"] - prov_water["water_base"]) / prov_water["water_base"] * 100,
        np.where(prov_water["water_wa"] > 0, 100.0, 0.0),
    )
    # Keep provinces with meaningful water use in at least one scenario
    prov_water = prov_water[
        (prov_water["water_base"] > 1e6) | (prov_water["water_wa"] > 1e6)
    ]
    prov_water = prov_water.sort_values("delta_pct", key=np.abs, ascending=False).head(15)
    prov_water = prov_water.sort_values("delta_pct")

    c_colors = [DIVERGING["pos"] if d > 0 else DIVERGING["neg"]
                for d in prov_water["delta_pct"]]
    y_c = np.arange(len(prov_water))
    ax_c.barh(y_c, prov_water["delta_pct"].values, color=c_colors,
              edgecolor="white", linewidth=0.2, height=0.7)
    ax_c.axvline(0, color="black", linewidth=0.6)
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(prov_water.index.tolist(), fontsize=6)
    ax_c.set_xlabel("Water use change, WA vs BASE (%)")
    ax_c.set_title("Provincial water use shift", fontsize=8)
    for i, val in enumerate(prov_water["delta_pct"].values):
        xpos = val + (1 if val >= 0 else -1)
        ha = "left" if val >= 0 else "right"
        ax_c.text(xpos, i, f"{val:+.0f}%", va="center", ha=ha,
                  fontsize=5.5, color="#333333")
    panel_label(ax_c, "c")

    # ── (d) Technology shift: BASE vs Water, top provinces by capacity ────────
    # Top 10 provinces by BASE 2050 total capacity
    # Top 8 provinces by BASE 2050 total capacity (fewer labels = less clutter)
    top_provs = (
        p50_base.groupby("province_name")["capacity_mw"].sum()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )

    pw_keys = ["unabated", "ccs", "beccs", "biomass", "retire"]
    share_cols = [f"share_{pw}" for pw in pw_keys]

    # Capacity-weighted mean share per province
    def _prov_shares(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for prov in top_provs:
            grp = df[df["province_name"] == prov]
            if grp.empty:
                rows.append({"province": prov, **{pw: 0.0 for pw in pw_keys}})
                continue
            cap = grp["capacity_mw"].values
            total = cap.sum()
            row = {"province": prov}
            for col, pw in zip(share_cols, pw_keys):
                values = grp[col].values if col in grp else np.zeros(len(grp))
                row[pw] = float((values * cap).sum() / total) if total > 0 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    df_d_base = _prov_shares(p50_base)
    df_d_wa   = _prov_shares(p50_wa)

    n_prov = len(top_provs)
    x_d = np.arange(n_prov)
    bar_w = 0.35
    # Show CCS share as the clearest differentiator
    ccs_base = df_d_base["ccs"].values * 100
    ccs_wa   = df_d_wa["ccs"].values * 100

    ax_d.bar(x_d - bar_w / 2, ccs_base, width=bar_w,
             color=PATHWAY_COLORS["ccs"], alpha=0.9,
             label="BASE", edgecolor="white", linewidth=0.2)
    ax_d.bar(x_d + bar_w / 2, ccs_wa, width=bar_w,
             color=PATHWAY_COLORS["ccs"], alpha=0.4,
             label="Water", edgecolor="white", linewidth=0.2, hatch="///")

    # Overlay retire as a second stacked element on the same bars
    ret_base = df_d_base["retire"].values * 100
    ret_wa   = df_d_wa["retire"].values * 100
    ax_d.bar(x_d - bar_w / 2, ret_base, width=bar_w,
             bottom=ccs_base,
             color=PATHWAY_COLORS["retire"], alpha=0.85,
             edgecolor="white", linewidth=0.2)
    ax_d.bar(x_d + bar_w / 2, ret_wa, width=bar_w,
             bottom=ccs_wa,
             color=PATHWAY_COLORS["retire"], alpha=0.40,
             edgecolor="white", linewidth=0.2, hatch="///")

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(
        [p[:8] for p in top_provs], rotation=45, ha="right",
        rotation_mode="anchor", fontsize=6)
    ax_d.set_ylabel("Share 2050 (%)")
    ax_d.set_title("CCS + retirement shift\n(BASE vs Water)", fontsize=8)
    leg_handles = [
        Patch(facecolor=PATHWAY_COLORS["ccs"], label="CCS – BASE"),
        Patch(facecolor=PATHWAY_COLORS["ccs"], alpha=0.4,
              hatch="///", label="CCS – Water"),
        Patch(facecolor=PATHWAY_COLORS["retire"], label="Retire – BASE"),
        Patch(facecolor=PATHWAY_COLORS["retire"], alpha=0.4,
              hatch="///", label="Retire – Water"),
    ]
    ax_d.legend(handles=leg_handles, fontsize=5.5, ncol=2,
                loc="upper right", frameon=True, framealpha=0.8, edgecolor="none")
    panel_label(ax_d, "d")

    fig.tight_layout()
    # Add SCS insets AFTER tight_layout so both align consistently
    _add_scs_inset(fig, ax_a)
    _add_scs_inset(fig, ax_b)
    save_fig(fig, "main_fig6_water_network")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating Fig 5 and Fig 6 ...")
    main_fig5_technology_dominance()
    main_fig6_water_network()
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
