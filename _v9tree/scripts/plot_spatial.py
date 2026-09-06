"""Provincial choropleth maps + CO2 network + resource utilization.

All maps use Albers Equal-Area Conic projection (standard parallels 25N/47N, CM 105E)
following the reference plot.ipynb style for academic publication quality.

Two-layer basemap: provinces (thin 0.2) + country outline (thick 0.75).
SCS inset with both layers. Arial font throughout.

Usage:
    python scripts/plot_spatial.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# ── Style (academic, Arial) ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "SimHei", "axes.unicode_minus": False, "font.size": 8, "mathtext.default": "regular",
    "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.facecolor": "white",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 7, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

TARGET_CRS = "EPSG:2380"

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

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "results" / "figures"

# English → Chinese province name mapping
EN_TO_CN = {
    "Anhui": "安徽省", "Beijing": "北京市", "Chongqing": "重庆市",
    "Fujian": "福建省", "Gansu": "甘肃省", "Guangdong": "广东省",
    "Guangxi": "广西壮族自治区", "Guizhou": "贵州省", "Hainan": "海南省",
    "Hebei": "河北省", "Heilongjiang": "黑龙江省", "Henan": "河南省",
    "Hubei": "湖北省", "Hunan": "湖南省", "Inner Mongolia": "内蒙古自治区",
    "Jiangsu": "江苏省", "Jiangxi": "江西省", "Jilin": "吉林省",
    "Liaoning": "辽宁省", "Ningxia": "宁夏回族自治区", "Qinghai": "青海省",
    "Shaanxi": "陕西省", "Shandong": "山东省", "Shanghai": "上海市",
    "Shanxi": "山西省", "Sichuan": "四川省", "Tianjin": "天津市",
    "Tibet": "西藏自治区", "Xinjiang": "新疆维吾尔自治区", "Yunnan": "云南省",
    "Zhejiang": "浙江省",
}


# ── Projected bounds (matching reference plot.ipynb) ─────────────────────────
# Main map: lon 80-150, lat 15-50 → projected
# SCS inset: lon 106.5-123, lat 2.8-24.5 → projected
def _compute_bounds():
    corners = gpd.GeoDataFrame(
        {"x": [80, 150, 106.5, 123], "y": [15, 50, 2.8, 24.5]},
        geometry=[Point(80, 15), Point(150, 50), Point(106.5, 2.8), Point(123, 24.5)],
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    return corners

_BOUNDS = None

def _get_bounds():
    global _BOUNDS
    if _BOUNDS is None:
        _BOUNDS = _compute_bounds()
    return _BOUNDS

def _main_xlim():
    b = _get_bounds()
    return (b.geometry.iloc[0].x, b.geometry.iloc[1].x)

def _main_ylim():
    b = _get_bounds()
    return (b.geometry.iloc[0].y, b.geometry.iloc[1].y)

def _scs_xlim():
    b = _get_bounds()
    return (b.geometry.iloc[2].x, b.geometry.iloc[3].x)

def _scs_ylim():
    b = _get_bounds()
    return (b.geometry.iloc[2].y, b.geometry.iloc[3].y)


# ── Data loading with caching ────────────────────────────────────────────────
_CACHE: dict = {}


def _load_provinces() -> gpd.GeoDataFrame:
    if "provinces" in _CACHE:
        return _CACHE["provinces"]
    provinces = gpd.read_file(ROOT / "data" / "ChinaMap" / "provinces.shp")
    provinces = provinces.to_crs(TARGET_CRS)
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    provinces["province_en"] = provinces["NAME"].map(cn_to_en)
    _CACHE["provinces"] = provinces
    return provinces


def _load_country() -> gpd.GeoDataFrame:
    if "country" in _CACHE:
        return _CACHE["country"]
    country = _load_provinces().dissolve()
    _CACHE["country"] = country
    return country


def _reproject_points(df: pd.DataFrame, lon_col: str = "lon",
                      lat_col: str = "lat") -> gpd.GeoDataFrame:
    """Convert a DataFrame with lon/lat columns to projected GeoDataFrame."""
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    gdf["proj_x"] = gdf.geometry.x
    gdf["proj_y"] = gdf.geometry.y
    return gdf


# ── Basemap helpers ──────────────────────────────────────────────────────────

def _plot_basemap(ax, facecolor="#f5f5f5"):
    """Plot China basemap: provinces (thin) + country outline (thick).

    Reference style (plot.ipynb):
      china.plot(edgecolor="black", linewidth=0.2)
      china_country.plot(edgecolor="black", linewidth=0.75)
    """
    provinces = _load_provinces()
    country = _load_country()
    provinces.plot(ax=ax, facecolor=facecolor, edgecolor="black",
                   linewidth=0.2, zorder=1)
    country.plot(ax=ax, facecolor="none", edgecolor="black",
                 linewidth=0.75, zorder=2)
    ax.set_xlim(_main_xlim())
    ax.set_ylim(_main_ylim())
    ax.set_axis_off()


def _plot_choropleth(ax, merged, column, cmap, vmin=0, vmax=100,
                     legend=True, legend_kwds=None):
    """Plot choropleth with province borders (thin) + country border (thick)."""
    country = _load_country()
    merged.plot(column=column, ax=ax, cmap=cmap,
                edgecolor="black", linewidth=0.2,
                legend=legend, vmin=vmin, vmax=vmax,
                legend_kwds=legend_kwds or {})
    country.plot(ax=ax, facecolor="none", edgecolor="black",
                 linewidth=0.75, zorder=10)
    ax.set_xlim(_main_xlim())
    ax.set_ylim(_main_ylim())
    ax.set_axis_off()


def _add_scale_bar(ax):
    """Add a 500 km scale bar in the lower-left."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Position in lower-left
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    bar_len = 500_000  # 500 km in meters
    ax.plot([x0, x0 + bar_len], [y0, y0], 'k-', linewidth=1.5, zorder=20)
    ax.text(x0 + bar_len / 2, y0 - (ylim[1] - ylim[0]) * 0.02,
            '500 km', ha='center', va='top', fontsize=6, zorder=20)


def _add_north_arrow(ax):
    """Add north arrow in upper-right corner."""
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=9, fontweight='bold', ha='center', va='center')
    ax.annotate('', xy=(0.95, 0.93), xycoords='axes fraction',
                xytext=(0.95, 0.87), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))


def _add_scs_inset(fig, ax_main):
    """Add South China Sea inset (reference plot.ipynb style).

    Both province and country layers are drawn in the inset.
    """
    provinces = _load_provinces()
    country = _load_country()
    pos = ax_main.get_position()
    inset_ax = fig.add_axes([pos.x1 - 0.12, pos.y0 + 0.01, 0.11, 0.16])
    provinces.plot(ax=inset_ax, facecolor="#f5f5f5", edgecolor="black", linewidth=0.2)
    country.plot(ax=inset_ax, facecolor="none", edgecolor="black", linewidth=0.75)
    inset_ax.set_xlim(_scs_xlim())
    inset_ax.set_ylim(_scs_ylim())
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.set_title("")


def _add_map_elements(ax):
    """Add scale bar + north arrow to a map axis."""
    _add_scale_bar(ax)
    _add_north_arrow(ax)


# ── Utility ──────────────────────────────────────────────────────────────────

def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


def _dominant_pathway_by_province(plant_detail: pd.DataFrame, year: int) -> pd.DataFrame:
    yr = plant_detail[plant_detail["year"] == year].copy()
    pathway_names = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]
    share_cols = [f"share_{pw}" for pw in pathway_names]
    rows = []
    for prov, grp in yr.groupby("province_name"):
        cap = grp["capacity_mw"].values
        total_cap = cap.sum()
        if total_cap == 0:
            continue
        shares = {}
        for col, pw in zip(share_cols, pathway_names):
            values = grp[col].values if col in grp else np.zeros(len(grp))
            shares[pw] = float((values * cap).sum() / total_cap)
        rows.append({"province_en": prov, **shares})
    return pd.DataFrame(rows)


# ── Fig: Provincial CCS+BECCS / Retirement maps (2050) ──────────────────────
def fig_provincial_maps():
    provinces = _load_provinces()
    plant_detail = pd.read_csv(ROOT / "results" / "BASE" / "plant_detail.csv")
    dom = _dominant_pathway_by_province(plant_detail, 2050)
    legend_kwds = {"shrink": 0.6, "orientation": "horizontal", "pad": 0.03}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5))

    # (a) CCS+BECCS share
    ax = axes[0]
    dom["ccs_beccs"] = dom["ccs"].fillna(0) + dom["beccs"].fillna(0)
    merged = provinces.merge(dom[["province_en", "ccs_beccs"]], on="province_en", how="left")
    merged["ccs_beccs"] = merged["ccs_beccs"].fillna(0) * 100
    _plot_choropleth(ax, merged, "ccs_beccs", "Blues",
                     legend_kwds={**legend_kwds, "label": "CCS+BECCS share (%)"})
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9,
            va="top")
    _add_map_elements(ax)

    # (b) Retirement share
    ax = axes[1]
    merged2 = provinces.merge(dom[["province_en", "retire"]], on="province_en", how="left")
    merged2["retire"] = merged2["retire"].fillna(1.0) * 100
    _plot_choropleth(ax, merged2, "retire", "Greys",
                     legend_kwds={**legend_kwds, "label": "Retirement share (%)"})
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9,
            va="top")
    _add_map_elements(ax)

    fig.tight_layout()
    _add_scs_inset(fig, axes[0])
    _add_scs_inset(fig, axes[1])
    _save(fig, "fig_provincial_maps")


# ── Fig: Fleet Age Map ──────────────────────────────────────────────────────
def fig_fleet_age_map():
    provinces = _load_provinces()
    plant_detail = pd.read_csv(ROOT / "results" / "BASE" / "plant_detail.csv")
    yr30 = plant_detail[plant_detail["year"] == 2030].copy()
    legend_kwds = {"shrink": 0.6, "orientation": "horizontal", "pad": 0.03}

    prov_age = []
    for prov, grp in yr30.groupby("province_name"):
        cap = grp["capacity_mw"].values
        total = cap.sum()
        if total == 0:
            continue
        if "retirement_year" in grp.columns:
            avg_ret_yr = float((grp["retirement_year"].values * cap).sum() / total)
            prov_age.append({"province_en": prov, "avg_retirement_year": avg_ret_yr})
        else:
            ret_share = float((grp["share_retire"].values * cap).sum() / total)
            prov_age.append({"province_en": prov, "retire_share_2030": ret_share})

    df_age = pd.DataFrame(prov_age)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    if "avg_retirement_year" in df_age.columns:
        merged = provinces.merge(df_age[["province_en", "avg_retirement_year"]],
                                on="province_en", how="left")
        merged["avg_retirement_year"] = merged["avg_retirement_year"].fillna(2040)
        _plot_choropleth(ax, merged, "avg_retirement_year", "RdYlBu",
                         vmin=2035, vmax=2060,
                         legend_kwds={**legend_kwds, "label": "Avg. design retirement year"})
    else:
        yr50 = plant_detail[plant_detail["year"] == 2050].copy()
        prov_ret = []
        for prov, grp in yr50.groupby("province_name"):
            cap = grp["capacity_mw"].values
            total = cap.sum()
            if total == 0:
                continue
            ret = float((grp["share_retire"].values * cap).sum() / total)
            prov_ret.append({"province_en": prov, "retire_2050": ret * 100})
        df_ret = pd.DataFrame(prov_ret)
        merged = provinces.merge(df_ret, on="province_en", how="left")
        merged["retire_2050"] = merged["retire_2050"].fillna(100)
        _plot_choropleth(ax, merged, "retire_2050", "RdYlBu_r",
                         legend_kwds={**legend_kwds, "label": "Retirement share 2050 (%)"})

    _add_map_elements(ax)
    fig.tight_layout()
    _add_scs_inset(fig, ax)
    _save(fig, "fig_fleet_age_map")


# ── Fig: Provincial Stacked Bar (2050) ───────────────────────────────────────
def fig_provincial_bar():
    plant_detail = pd.read_csv(ROOT / "results" / "BASE" / "plant_detail.csv")
    yr50 = plant_detail[plant_detail["year"] == 2050].copy()

    pathway_names = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]
    share_cols = [f"share_{pw}" for pw in pathway_names]

    prov_data = []
    for prov, grp in yr50.groupby("province_name"):
        cap = grp["capacity_mw"].values
        total = cap.sum()
        if total == 0:
            continue
        row = {"province": prov, "total_gw": total / 1000}
        for col, pw in zip(share_cols, pathway_names):
            values = grp[col].values if col in grp else np.zeros(len(grp))
            row[pw] = float((values * cap).sum() / total)
        prov_data.append(row)

    df = pd.DataFrame(prov_data).sort_values("ccs", ascending=False)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    for pw in PATHWAY_ORDER:
        vals = df[pw].values * 100
        ax.bar(x, vals, bottom=bottom, color=PATHWAY_COLORS[pw],
               label=PATHWAY_LABELS[pw], edgecolor="white", linewidth=0.2, width=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df["province"].values, rotation=90, fontsize=5.5)
    ax.set_ylabel("Generation share (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=5, bbox_to_anchor=(0.5, 0.98), loc="upper center", fontsize=6)
    ax.text(0.0, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_provincial_bar_2050")


# ── Fig: CO2 Network (2050) ─────────────────────────────────────────────────
def fig_co2_network():
    co2 = pd.read_csv(ROOT / "results" / "BASE" / "co2_flow_direction.csv")
    nodes = pd.read_csv(ROOT / "inputs" / "pipeline_nodes.csv")
    candidates = pd.read_csv(ROOT / "inputs" / "pipeline_candidate_edges.csv")
    plants = pd.read_csv(ROOT / "results" / "BASE" / "plant_detail.csv")
    storage_util = pd.read_csv(ROOT / "results" / "BASE" / "storage_utilization.csv")

    yr50_co2 = co2[co2["year"] == 2050].copy()
    yr50_plants = plants[plants["year"] == 2050].copy()

    # Build geometry lookup: edge_id → WKT (for curved corridor edges)
    geom_lookup = candidates.set_index("edge_id")["geometry_wkt"].to_dict()

    # Reproject all pipeline nodes (includes plant, storage_hub, corridor nodes)
    if len(nodes) > 0:
        nodes_proj = _reproject_points(nodes, "lon", "lat")
        node_coords = nodes_proj.set_index("node_id")[["proj_x", "proj_y"]].to_dict("index")
    else:
        node_coords = {}

    # Reproject plants for pathway markers
    plant_proj = _reproject_points(
        yr50_plants.dropna(subset=["centroid_longitude"]),
        "centroid_longitude", "centroid_latitude",
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_basemap(ax)

    # ── Pipelines: trunk edges (corridor / triangulation) get flow-scaled widths
    # and darker colour; feeder/branch edges are drawn thin and light ──
    from matplotlib.lines import Line2D
    from shapely import wkt as shapely_wkt
    pipe_handles = []
    PIPE_COLOR = "#4477AA"
    BRANCH_COLOR = "#9DB8D0"
    BRANCH_CLASSES = {
        "runtime_plant_branch", "runtime_storage_branch",
        "hub_to_corridor_branch", "corridor_to_storage_branch",
    }
    if len(yr50_co2) > 0 and node_coords:
        max_flow = max(yr50_co2["net_flow_mtpa"].max(), 0.01)
        for _, edge in yr50_co2.iterrows():
            flow = float(edge["net_flow_mtpa"])
            if flow <= 0.01:
                continue
            is_branch = str(edge.get("edge_class", "")) in BRANCH_CLASSES
            if is_branch:
                lw, color, alpha, zorder = 0.4, BRANCH_COLOR, 0.5, 2
            else:
                lw = 0.4 + 2.6 * (flow / max_flow)
                color, alpha, zorder = PIPE_COLOR, 0.55 + 0.35 * (flow / max_flow), 3
            drawn = False
            # Try curved geometry for corridor edges
            wkt_str = geom_lookup.get(edge.get("edge_id", ""), "")
            if isinstance(wkt_str, str) and wkt_str.startswith("LINESTRING"):
                try:
                    geom = shapely_wkt.loads(wkt_str)
                    gdf_e = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(TARGET_CRS)
                    xs, ys = zip(*gdf_e.geometry.iloc[0].coords)
                    ax.plot(list(xs), list(ys),
                            color=color, linewidth=lw, alpha=alpha,
                            solid_capstyle="round", zorder=zorder)
                    drawn = True
                except Exception:
                    pass
            # Fallback: straight line between nodes
            if not drawn:
                src = node_coords.get(edge["source_node"])
                snk = node_coords.get(edge["sink_node"])
                if src and snk:
                    ax.plot([src["proj_x"], snk["proj_x"]],
                            [src["proj_y"], snk["proj_y"]],
                            color=color, linewidth=lw, alpha=alpha,
                            solid_capstyle="round", zorder=zorder)

        # Flow magnitude legend (trunk classes + feeder swatch)
        for fl in [f for f in [5, 10, 20, 40] if f <= max_flow + 0.01]:
            lw_leg = 0.4 + 2.6 * (fl / max_flow)
            pipe_handles.append(Line2D([], [], color=PIPE_COLOR, linewidth=lw_leg,
                                       alpha=0.7, solid_capstyle="round",
                                       label=f"{fl} Mtpa"))
        pipe_handles.append(Line2D([], [], color=BRANCH_COLOR, linewidth=0.4,
                                   alpha=0.6, solid_capstyle="round",
                                   label="feeder"))

    # ── Storage sinks (from pipeline_nodes with coordinates) ──
    storage_nodes = nodes_proj[nodes_proj["node_type"] == "storage_hub"].copy()
    yr50_stor = storage_util[(storage_util["year"] == 2050) & (storage_util["storage_use_mtpa"] > 0.001)]
    active_hubs = set(yr50_stor["storage_hub_id"])
    active_stor = storage_nodes[storage_nodes["storage_hub_id"].isin(active_hubs)]
    if len(active_stor) > 0:
        ax.scatter(active_stor["proj_x"], active_stor["proj_y"],
                   s=50, c="#EE6677", marker="^", edgecolors="black",
                   linewidth=0.4, label="Storage sink", zorder=7)

    # ── Source plants by pathway (skip retirement) ──
    ccs_only = plant_proj[(plant_proj["share_ccs"] > 0.1) & (plant_proj["share_beccs"] <= 0.01)]
    beccs = plant_proj[plant_proj["share_beccs"] > 0.01]
    bio = plant_proj[(plant_proj["share_biomass"] > 0.1)
                     & (plant_proj["share_ccs"] <= 0.1)
                     & (plant_proj["share_beccs"] <= 0.01)]

    if len(bio) > 0:
        ax.scatter(bio["proj_x"], bio["proj_y"],
                   s=bio["capacity_mw"] / 250, c="#228833", alpha=0.6,
                   edgecolors="black", linewidth=0.2, label="Biomass co-firing", zorder=4)
    if len(ccs_only) > 0:
        ax.scatter(ccs_only["proj_x"], ccs_only["proj_y"],
                   s=ccs_only["capacity_mw"] / 250, c="#4477AA", alpha=0.7,
                   marker="s", edgecolors="black", linewidth=0.3, label="CCS", zorder=5)
    if len(beccs) > 0:
        ax.scatter(beccs["proj_x"], beccs["proj_y"],
                   s=beccs["capacity_mw"] / 180, c="#CCBB44", alpha=0.7,
                   marker="D", edgecolors="black", linewidth=0.3, label="BECCS", zorder=6)

    # Basin labels
    basin_info = [
        (123, 44, "Songliao\nBasin"),
        (118, 38, "Bohai Bay\nBasin"),
        (110, 39, "Ordos\nBasin"),
        (105, 30, "Sichuan\nBasin"),
        (120, 34, "Subei\nBasin"),
    ]
    basin_pts = gpd.GeoDataFrame(
        basin_info, columns=["lon", "lat", "label"],
        geometry=[Point(lon, lat) for lon, lat, _ in basin_info],
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    for _, row in basin_pts.iterrows():
        ax.annotate(row["label"], xy=(row.geometry.x, row.geometry.y),
                   fontsize=7, fontstyle="italic", color="#666666",
                   ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                   zorder=8)

    # Source/sink legend (lower-left)
    ax.legend(loc="lower left", fontsize=7, markerscale=1.5, framealpha=0.9)

    # Pipeline flow legend (upper-right)
    if pipe_handles:
        from matplotlib.legend import Legend
        pipe_leg = Legend(ax, pipe_handles, [h.get_label() for h in pipe_handles],
                         loc="upper right", fontsize=6,
                         title="CO$_2$ pipeline flow",
                         title_fontsize=7, framealpha=0.9,
                         bbox_to_anchor=(0.99, 0.93))
        ax.add_artist(pipe_leg)

    _add_map_elements(ax)
    fig.tight_layout()
    _add_scs_inset(fig, ax)
    _save(fig, "fig_co2_network_2050")


# ── Fig: Plant Cost Map (2050) ───────────────────────────────────────────────
def fig_plant_cost_map():
    cost = pd.read_csv(ROOT / "results" / "BASE" / "plant_cost.csv")
    detail = pd.read_csv(ROOT / "results" / "BASE" / "plant_detail.csv")
    yr50_cost = cost[cost["year"] == 2050].copy()
    yr50_detail = detail[detail["year"] == 2050][
        ["plant_id", "centroid_longitude", "centroid_latitude", "capacity_mw"]
    ].copy()
    yr50 = yr50_cost.merge(yr50_detail, on="plant_id", how="left",
                           suffixes=("", "_detail")).dropna(
        subset=["centroid_longitude"]
    )
    # Ensure capacity_mw column exists (may come from either side of merge)
    if "capacity_mw" not in yr50.columns and "capacity_mw_detail" in yr50.columns:
        yr50["capacity_mw"] = yr50["capacity_mw_detail"]
    yr50["total_billion"] = yr50["total_plant_cost_cny"] / 1e9

    # Reproject
    yr50_proj = _reproject_points(yr50, "centroid_longitude", "centroid_latitude")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _plot_basemap(ax)

    sc = ax.scatter(
        yr50_proj["proj_x"], yr50_proj["proj_y"],
        s=yr50_proj["capacity_mw"] / 80,
        c=yr50_proj["total_billion"],
        cmap="RdYlBu_r", alpha=0.7,
        edgecolors="black", linewidth=0.3,
        vmin=-5, vmax=20, zorder=5,
    )
    plt.colorbar(sc, ax=ax, label="Net cost (B CNY)", shrink=0.6, pad=0.02,
                 orientation="horizontal")
    _add_map_elements(ax)
    fig.tight_layout()
    _add_scs_inset(fig, ax)
    _save(fig, "fig_plant_cost_2050")


# ── Fig: Biomass Utilization ─────────────────────────────────────────────────
def fig_biomass_utilization():
    provinces = _load_provinces()
    resource = pd.read_csv(ROOT / "results" / "BASE" / "resource_use.csv")
    legend_kwds = {"shrink": 0.6, "orientation": "horizontal", "pad": 0.03}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5))

    for col, (ax, year) in enumerate(zip(axes, [2030, 2050])):
        bio = resource[(resource["resource_type"] == "biomass") &
                       (resource["year"] == year)].copy()
        if len(bio) == 0:
            continue
        prov_bio = bio.groupby("province_name")[["used", "available"]].sum().reset_index()
        prov_bio["utilization"] = np.where(
            prov_bio["available"] > 0,
            prov_bio["used"] / prov_bio["available"] * 100, 0,
        )
        prov_bio["province_en"] = prov_bio["province_name"]
        merged = provinces.merge(prov_bio[["province_en", "utilization"]],
                                on="province_en", how="left")
        merged["utilization"] = merged["utilization"].fillna(0)
        _plot_choropleth(ax, merged, "utilization", "YlOrRd",
                         legend_kwds={**legend_kwds, "label": "Utilization (%)"})
        ax.text(0.02, 0.98, f"({chr(97 + col)})", transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="top")
        _add_map_elements(ax)

    fig.tight_layout()
    _save(fig, "fig_biomass_utilization")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating spatial figures (Albers EAC)...")
    fig_provincial_maps()
    fig_fleet_age_map()
    fig_provincial_bar()
    fig_co2_network()
    fig_plant_cost_map()
    print(f"\nAll spatial figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
