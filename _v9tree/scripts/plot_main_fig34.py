"""Figures 3 and 4 for the coal-retrofit paper.

Fig 3: 2x2 layout — provincial dominant-pathway choropleth maps (BASE,
       Net negative, Water constrained) and biomass transport hexbin.
Fig 4: Provincial technology-transition stacked bars (all provinces,
       4 time-steps, mini-alluvial connecting lines, region separators).

Usage:
    python scripts/plot_main_fig34.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (
    apply_style, save_fig, panel_label, panel_label_inside, pathway_legend,
    PATHWAY_COLORS, PATHWAY_LABELS, PATHWAY_ORDER,
    ROOT, RESULTS_DIR, FIGURES_DIR, BASE_DIR,
    REGIONS, CN_TO_EN, EN_TO_CN,
    FULL_PAGE, MAP_SINGLE,
)
apply_style()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _dominant_pathway_province(plant_detail: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute capacity-weighted dominant pathway per province for a given year.

    Returns a DataFrame with columns: province_name, dominant_pathway.
    """
    yr = plant_detail[plant_detail["year"] == year].copy()
    pathway_names = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]
    share_cols = [f"share_{pw}" for pw in pathway_names]
    rows = []
    for prov, grp in yr.groupby("province_name"):
        cap = grp["capacity_mw"].values
        total_cap = cap.sum()
        if total_cap == 0:
            continue
        weighted = {
            pw: float(((grp[col].values if col in grp else np.zeros(len(grp))) * cap).sum() / total_cap)
            for col, pw in zip(share_cols, pathway_names)
        }
        dom = max(weighted, key=weighted.get)
        rows.append({"province_name": prov, "dominant_pathway": dom})
    return pd.DataFrame(rows)


def _load_provinces_local():
    """Load and reproject shapefile, mapping Chinese NAME → English.

    Uses plot_style.CN_TO_EN which covers both Xinjiang name variants.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required for map panels") from exc

    shp_path = ROOT / "data" / "ChinaMap" / "provinces.shp"
    provinces = gpd.read_file(shp_path)
    provinces = provinces.to_crs("EPSG:2380")
    # CN_TO_EN keys match the raw (possibly garbled) bytes from the shapefile,
    # exactly as plot_spatial._load_provinces does.
    provinces["province_en"] = provinces["NAME"].map(CN_TO_EN)
    return provinces


def _plot_map_basemap(ax, provinces_gdf, color_map: dict[str, str]):
    """Draw choropleth by dominant pathway color, then country outline.

    color_map: {province_en → hex color string}
    """
    # Fill each province with its pathway color; grey for unmatched
    default_color = "#e0e0e0"
    provinces_gdf = provinces_gdf.copy()
    provinces_gdf["_color"] = provinces_gdf["province_en"].map(color_map).fillna(default_color)
    provinces_gdf.plot(
        ax=ax,
        color=provinces_gdf["_color"],
        edgecolor="black",
        linewidth=0.5,
        zorder=1,
    )
    # Country outline
    country = provinces_gdf.dissolve()
    country.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.75, zorder=2)

    # Clip to main-map extent (lon 80-150, lat 15-55)
    from shapely.geometry import Point
    import geopandas as gpd
    corners = gpd.GeoDataFrame(
        geometry=[Point(80, 15), Point(150, 55)],
        crs="EPSG:4326",
    ).to_crs("EPSG:2380")
    ax.set_xlim(corners.geometry.iloc[0].x, corners.geometry.iloc[1].x)
    ax.set_ylim(corners.geometry.iloc[0].y, corners.geometry.iloc[1].y)
    ax.set_axis_off()


def _add_scale_bar(ax):
    """500 km scale bar, lower-left."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    bar_len = 500_000  # metres
    ax.plot([x0, x0 + bar_len], [y0, y0], "k-", linewidth=1.5, zorder=20)
    ax.text(
        x0 + bar_len / 2,
        y0 - (ylim[1] - ylim[0]) * 0.02,
        "500 km",
        ha="center", va="top", fontsize=6, zorder=20,
    )


def _add_north_arrow(ax):
    """North arrow, upper-right."""
    ax.annotate(
        "N", xy=(0.95, 0.95), xycoords="axes fraction",
        fontsize=9, fontweight="bold", ha="center", va="center",
    )
    ax.annotate(
        "", xy=(0.95, 0.93), xycoords="axes fraction",
        xytext=(0.95, 0.87), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )


def _add_scs_inset(fig, ax_main, provinces_gdf, color_map: dict[str, str]):
    """South China Sea inset in bottom-right of the map panel."""
    from shapely.geometry import Point
    import geopandas as gpd

    default_color = "#e0e0e0"
    provinces_colored = provinces_gdf.copy()
    provinces_colored["_color"] = provinces_colored["province_en"].map(color_map).fillna(default_color)

    pos = ax_main.get_position()
    inset_ax = fig.add_axes([pos.x1 - 0.12, pos.y0 + 0.01, 0.11, 0.16])

    provinces_colored.plot(
        ax=inset_ax,
        color=provinces_colored["_color"],
        edgecolor="black",
        linewidth=0.5,
    )
    country = provinces_gdf.dissolve()
    country.plot(ax=inset_ax, facecolor="none", edgecolor="black", linewidth=0.75)

    # SCS bounds: lon 106.5-123, lat 2.8-24.5
    corners_scs = gpd.GeoDataFrame(
        geometry=[Point(106.5, 2.8), Point(123, 24.5)],
        crs="EPSG:4326",
    ).to_crs("EPSG:2380")
    inset_ax.set_xlim(corners_scs.geometry.iloc[0].x, corners_scs.geometry.iloc[1].x)
    inset_ax.set_ylim(corners_scs.geometry.iloc[0].y, corners_scs.geometry.iloc[1].y)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.set_title("")


def _draw_choropleth_panel(
    fig, ax, provinces_gdf, plant_detail_path: Path, year: int, label: str, title: str
):
    """Draw a single dominant-pathway choropleth panel."""
    detail = pd.read_csv(plant_detail_path)
    dom = _dominant_pathway_province(detail, year)
    color_map = {
        row["province_name"]: PATHWAY_COLORS[row["dominant_pathway"]]
        for _, row in dom.iterrows()
    }
    _plot_map_basemap(ax, provinces_gdf, color_map)
    _add_scale_bar(ax)
    _add_north_arrow(ax)
    panel_label_inside(ax, label)
    ax.set_title(title, fontsize=8, pad=4)
    _add_scs_inset(fig, ax, provinces_gdf, color_map)


# ── Fig 3 ─────────────────────────────────────────────────────────────────────

def main_fig3_scenario_maps():
    """2x2 layout: 3 choropleth maps + 1 biomass hexbin."""
    try:
        import geopandas  # noqa: F401
    except ImportError:
        print("  [skip] main_fig3_scenario_maps: geopandas not available")
        return

    provinces_gdf = _load_provinces_local()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.06,
                        wspace=0.05, hspace=0.12)

    # ── (a) BASE 2050 ──────────────────────────────────────────────────────
    ax_a = axes[0, 0]
    _draw_choropleth_panel(
        fig, ax_a,
        provinces_gdf,
        RESULTS_DIR / "BASE" / "plant_detail.csv",
        year=2050, label="a", title="BASE (2050)",
    )

    # ── (b) Net negative 2050 ─────────────────────────────────────────────
    ax_b = axes[0, 1]
    _draw_choropleth_panel(
        fig, ax_b,
        provinces_gdf,
        RESULTS_DIR / "BASE_neg" / "plant_detail.csv",
        year=2050, label="b", title="Net negative (2050)",
    )

    # ── (c) Water constrained 2050 ────────────────────────────────────────
    ax_c = axes[1, 0]
    _draw_choropleth_panel(
        fig, ax_c,
        provinces_gdf,
        RESULTS_DIR / "WA_grid_200km" / "plant_detail.csv",
        year=2050, label="c", title="Water constraint (2050)",
    )

    # ── (d) Biomass transport hexbin ──────────────────────────────────────
    ax_d = axes[1, 1]
    bio = pd.read_csv(RESULTS_DIR / "BASE" / "biomass_flows.csv")
    bio_30 = bio[bio["year"] == 2030].copy()
    bio_30 = bio_30[bio_30["flow_gj"] > 0]
    panel_year = 2030
    if bio_30.empty:
        # 2030 may be all-unabated; fall back to the earliest year with positive flow
        positive = bio[bio["flow_gj"] > 0]
        if not positive.empty:
            panel_year = int(positive["year"].min())
            bio_30 = positive[positive["year"] == panel_year].copy()

    if bio_30.empty:
        ax_d.text(0.5, 0.5, "no biomass transport", transform=ax_d.transAxes,
                  ha="center", va="center", fontsize=8, color="#666666")
        ax_d.set_axis_off()
        panel_label(ax_d, "d")
        # Fall through to the shared legend / save without the hexbin panel.
    else:
        dist = bio_30["distance_km"].values
        flow_pj = bio_30["flow_gj"].values / 1e6

        hb = ax_d.hexbin(
            dist, flow_pj,
            gridsize=25,
            cmap="YlGnBu",
            mincnt=1,
        )
        cb = fig.colorbar(hb, ax=ax_d, pad=0.02, shrink=0.85)
        cb.set_label("Count", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Weighted mean and 90th percentile
        weights = bio_30["flow_gj"].values
        w_mean = float(np.average(dist, weights=weights))
        # Weighted 90th percentile via sorted CDF
        sort_idx = np.argsort(dist)
        cumw = np.cumsum(weights[sort_idx])
        cumw_norm = cumw / cumw[-1]
        pct90 = float(dist[sort_idx][np.searchsorted(cumw_norm, 0.90)])

        ax_d.axvline(w_mean, color="#CC3311", linewidth=1.2, linestyle="--", zorder=5,
                     label=f"Mean {w_mean:.0f} km")
        ax_d.axvline(pct90, color="#4477AA", linewidth=1.2, linestyle=":", zorder=5,
                     label=f"90th pct {pct90:.0f} km")
        ax_d.annotate(
            f"Mean\n{w_mean:.0f} km",
            xy=(w_mean, ax_d.get_ylim()[1] * 0.85),
            xytext=(w_mean + 8, ax_d.get_ylim()[1] * 0.85),
            fontsize=6, color="#CC3311",
            arrowprops=dict(arrowstyle="->", color="#CC3311", lw=0.8),
        )
        ax_d.set_xlabel("Transport distance (km)", fontsize=8)
        ax_d.set_ylabel(f"Biomass flow (PJ), {panel_year}", fontsize=8)
        ax_d.tick_params(labelsize=7)
        ax_d.grid(True, which="major", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
        ax_d.set_axisbelow(True)
        ax_d.legend(fontsize=6, loc="upper right", frameon=False)
        panel_label(ax_d, "d")

    # ── Shared pathway legend at top ──────────────────────────────────────
    handles = [
        Patch(facecolor=PATHWAY_COLORS[pw], edgecolor="white",
              linewidth=0.3, label=PATHWAY_LABELS[pw])
        for pw in PATHWAY_ORDER
    ]
    fig.legend(
        handles=handles, ncol=5,
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        fontsize=7, frameon=False,
    )

    save_fig(fig, "main_fig3_scenario_maps")


# ── Fig 4 ─────────────────────────────────────────────────────────────────────

def _bezier_connect(ax, x0, x1, y_bottom, y_top, color, alpha=0.3):
    """Draw a filled bezier patch connecting two horizontal bar segments.

    (x0, y_bottom) is the right edge (start_x, y center of bottom bar)
    (x1, y_top) is the left edge (end_x, y center of top bar)

    We draw a filled quadrilateral with bezier-curved sides to represent
    how a share 'flows' from one year bar to the next.
    """
    # y_bottom and y_top are (y_center, half_height) tuples
    yb_ctr, yb_h = y_bottom
    yt_ctr, yt_h = y_top

    # Four corners: bottom-bar right edge (top/bottom), top-bar left edge (top/bottom)
    bl_b = (x0, yb_ctr - yb_h)  # bottom-left bottom
    bl_t = (x0, yb_ctr + yb_h)  # bottom-left top
    tr_b = (x1, yt_ctr - yt_h)  # top-right bottom
    tr_t = (x1, yt_ctr + yt_h)  # top-right top

    # Bezier control points at x midpoint
    xm = (x0 + x1) / 2.0

    # Build the path: go from bl_t → tr_t (cubic bezier) → tr_b → bl_b (reverse bezier)
    verts = [
        bl_t,
        (xm, bl_t[1]),
        (xm, tr_t[1]),
        tr_t,
        tr_b,
        (xm, tr_b[1]),
        (xm, bl_b[1]),
        bl_b,
        bl_t,  # close
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    path = MPath(verts, codes)
    patch = mpatches.PathPatch(
        path, facecolor=color, edgecolor="none", alpha=alpha, zorder=1,
    )
    ax.add_patch(patch)


def main_fig4_provincial_transitions():
    """Horizontal stacked bars per province across 4 time-steps with alluvial links."""
    years = [2030, 2040, 2050, 2060]
    df = pd.read_csv(RESULTS_DIR / "BASE" / "province_pathways.csv")

    # Build per-province-year share pivot
    # generation_share = fraction of provincial generation from each pathway
    pivot = df.pivot_table(
        index=["province_name", "year"],
        columns="pathway",
        values="generation_share",
        aggfunc="sum",
    ).fillna(0.0).reset_index()

    # Ensure all pathways present
    for pw in PATHWAY_ORDER:
        if pw not in pivot.columns:
            pivot[pw] = 0.0

    # Compute province capacity (GW) from province_generation_mwh (same for all pathways)
    prov_gen = (
        df[df["year"] == 2030]
        .drop_duplicates("province_name")[["province_name", "province_generation_mwh"]]
        .copy()
    )
    # MWh / (CF * annual_hours) = MW; divide by 1e3 for GW
    prov_gen["capacity_gw"] = prov_gen["province_generation_mwh"] / (0.55 * 8760) / 1e3

    # Determine province order: grouped by region, sorted by 2030 capacity descending
    # Build region → list of (province, capacity_gw)
    all_data_provinces = set(pivot["province_name"].unique())
    ordered_provinces = []  # list of (region_name, province_name)
    region_boundaries = []  # y-indices where a new region starts

    running_idx = 0
    for region_name, region_provs in REGIONS.items():
        present = [p for p in region_provs if p in all_data_provinces]
        if not present:
            continue
        cap_map = prov_gen.set_index("province_name")["capacity_gw"].to_dict()
        present_sorted = sorted(present, key=lambda p: cap_map.get(p, 0.0), reverse=True)
        region_boundaries.append((running_idx, region_name))
        for p in present_sorted:
            ordered_provinces.append((region_name, p))
        running_idx += len(present_sorted)

    n_provinces = len(ordered_provinces)
    n_years = len(years)

    # Layout constants
    bar_height = 0.18
    bar_gap = 0.05       # gap between consecutive year-bars within a province
    province_spacing = 1.0  # total vertical space per province
    # Each province occupies province_spacing in y.
    # 4 bars at bar_height with bar_gap between them.
    # Bars are centered within the province slot.
    # Bar i within slot: offset from center = (i - 1.5) * (bar_height + bar_gap)
    bars_total_height = n_years * bar_height + (n_years - 1) * bar_gap
    bar_y_offsets = [
        (i - (n_years - 1) / 2.0) * (bar_height + bar_gap)
        for i in range(n_years)
    ]

    # Figure height: n_provinces * province_spacing + margins
    fig_height = max(9.0, n_provinces * province_spacing * 0.32 + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    # y positions: province 0 at bottom, n_provinces-1 at top
    # We index provinces bottom-to-top (reversed display order so top province is first)
    # Display order: first region at top → province 0 in ordered_provinces at top
    # y_center for province i (0=top of list displayed) = (n_provinces - 1 - i) * province_spacing
    def prov_y(i):
        return (n_provinces - 1 - i) * province_spacing

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-province_spacing * 0.6, n_provinces * province_spacing)
    ax.set_axis_off()

    # Draw bars and collect edge positions for bezier connectors
    # edge_map[province_idx][year_idx] = list of (x_right, y_ctr, y_half, color)
    # one entry per pathway segment (left-to-right stacked)
    edge_map: dict[int, dict[int, list[tuple]]] = {
        pi: {yi: [] for yi in range(n_years)}
        for pi in range(n_provinces)
    }

    for pi, (region_name, prov) in enumerate(ordered_provinces):
        y_ctr = prov_y(pi)

        for yi, yr in enumerate(years):
            row = pivot[(pivot["province_name"] == prov) & (pivot["year"] == yr)]
            if row.empty:
                continue
            bar_y = y_ctr + bar_y_offsets[yi]
            x_left = 0.0
            for pw in PATHWAY_ORDER:
                share = float(row[pw].values[0]) if pw in row.columns else 0.0
                if share < 1e-6:
                    continue
                rect = mpatches.Rectangle(
                    (x_left, bar_y - bar_height / 2),
                    share, bar_height,
                    linewidth=0,
                    facecolor=PATHWAY_COLORS[pw],
                    zorder=3,
                )
                ax.add_patch(rect)
                x_right = x_left + share
                # Store right-edge info for connector drawing
                edge_map[pi][yi].append((x_left, x_right, bar_y, bar_height / 2, pw))
                x_left = x_right

        # Province label on left
        cap_gw = prov_gen.set_index("province_name")["capacity_gw"].get(prov, 0.0)
        ax.text(
            -0.03, y_ctr,
            prov,
            ha="right", va="center", fontsize=6.5,
            color="black",
        )
        # Capacity annotation on right
        ax.text(
            1.02, y_ctr,
            f"{cap_gw:.1f} GW",
            ha="left", va="center", fontsize=5,
            color="#555555",
        )

    # Draw bezier connectors between consecutive year bars
    for pi in range(n_provinces):
        y_ctr = prov_y(pi)
        for yi in range(n_years - 1):
            # Build segment lookup: pathway → (x_left, x_right) for year yi and yi+1
            segs_a = {pw: (xl, xr, bary, bh) for xl, xr, bary, bh, pw in edge_map[pi][yi]}
            segs_b = {pw: (xl, xr, bary, bh) for xl, xr, bary, bh, pw in edge_map[pi][yi + 1]}
            for pw in PATHWAY_ORDER:
                if pw not in segs_a or pw not in segs_b:
                    continue
                xl_a, xr_a, bary_a, bh_a = segs_a[pw]
                xl_b, xr_b, bary_b, bh_b = segs_b[pw]
                share_a = xr_a - xl_a
                share_b = xr_b - xl_b
                if share_a < 1e-4 and share_b < 1e-4:
                    continue
                # x positions: right edge of year-bar A → left edge of year-bar B
                # But bars are stacked at same x range (0-1); the "time" axis
                # is encoded in y position via bar_y_offsets.
                # The connector goes from (xr_a, bary_a) to (xl_b, bary_b)
                # using the share width as half-height of the bezier band.
                # We represent the band width as min(share_a, share_b)/2 in y.
                band_half = min(share_a, share_b) * bar_height / 2.0
                _bezier_connect(
                    ax,
                    x0=xr_a,   # right edge of segment in bar yi
                    x1=xl_b,   # left edge of segment in bar yi+1
                    y_bottom=(bary_a, band_half),
                    y_top=(bary_b, band_half),
                    color=PATHWAY_COLORS[pw],
                    alpha=0.25,
                )

    # Region separators and labels
    # First, draw alternating light-gray bands per region group for visual separation
    band_colors = ["#f7f7f7", "#ffffff"]  # alternate between very light gray and white
    for bi, (reg_start_idx, region_name) in enumerate(region_boundaries):
        # Determine the end index of this region
        if bi + 1 < len(region_boundaries):
            reg_end_idx = region_boundaries[bi + 1][0] - 1
        else:
            reg_end_idx = n_provinces - 1
        # y extents: from bottom of last province to top of first province in this region
        y_top = prov_y(reg_start_idx) + province_spacing * 0.5
        y_bot = prov_y(reg_end_idx) - province_spacing * 0.5
        band_color = band_colors[bi % 2]
        if band_color != "#ffffff":  # skip white (already background)
            ax.axhspan(y_bot, y_top, xmin=0.0, xmax=1.0,
                       facecolor=band_color, edgecolor="none", zorder=0, alpha=0.6)

    # Thin horizontal separator lines between each province for readability
    for pi in range(n_provinces - 1):
        sep_line_y = prov_y(pi) - province_spacing * 0.5
        ax.axhline(sep_line_y, color="#dddddd", linewidth=0.3, zorder=0,
                   xmin=0.0, xmax=1.0)

    for reg_start_idx, region_name in region_boundaries:
        # Find the last province index in this region
        # region_boundaries gives start; next boundary gives end
        # Draw a line just above the top province of the region
        y_top_prov = prov_y(reg_start_idx)
        sep_y = y_top_prov + province_spacing * 0.55
        ax.axhline(sep_y, color="#cccccc", linewidth=0.6, zorder=0, xmin=-0.2, xmax=1.1,
                   clip_on=False)
        ax.text(
            -0.03, sep_y + 0.05,
            region_name,
            ha="right", va="bottom", fontsize=8,
            fontweight="bold", color="#222222",
        )

    # Year labels at the top (one per bar_y_offset, at the top province y_ctr)
    top_y_ctr = prov_y(0)
    for yi, yr in enumerate(years):
        bar_y_top = top_y_ctr + bar_y_offsets[yi]
        ax.text(
            (yi + 0.5) / n_years if False else 0.125 + yi * 0.25,
            1.02,
            str(yr),
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            transform=ax.transAxes,
        )

    # Pathway legend at top
    handles = [
        Patch(facecolor=PATHWAY_COLORS[pw], edgecolor="white",
              linewidth=0.3, label=PATHWAY_LABELS[pw])
        for pw in PATHWAY_ORDER
    ]
    ax.legend(
        handles=handles, ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        fontsize=7, frameon=False,
    )

    # X-axis ticks at top (share %)
    for tick_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ax.text(
            tick_val, -province_spacing * 0.45,
            f"{int(tick_val * 100)}%",
            ha="center", va="top", fontsize=6, color="#666666",
        )
        ax.plot(
            [tick_val, tick_val],
            [-province_spacing * 0.35, n_provinces * province_spacing - 0.3],
            color="#eeeeee", linewidth=0.4, zorder=0,
        )

    # Secondary y-axis label for capacity
    ax.text(
        1.02, n_provinces * province_spacing * 0.5,
        "2030 capacity",
        ha="left", va="center", fontsize=6, color="#555555", rotation=90,
    )

    panel_label(ax, "a", x=-0.01, y=1.04)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    save_fig(fig, "main_fig4_provincial_transitions")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating main figures 3 and 4...")
    main_fig3_scenario_maps()
    main_fig4_provincial_transitions()
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
