"""Visualise the candidate CO2 pipeline network.

Basemap style follows reference/plot.ipynb:
  - EPSG:2380 projection
  - Province boundaries (thin 0.2) + country outline (thick 0.75)
  - South China Sea inset
  - No axis, no grid

Output: inputs/figures/candidate_network.png + .pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt as shapely_wkt

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family": "SimHei", "axes.unicode_minus": False, "font.size": 8,
    "axes.grid": False, "axes.facecolor": "white",
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})

TARGET_CRS = "EPSG:2380"
PROVINCE_SHP = ROOT / "data" / "ChinaMap" / "provinces.shp"
BOUNDARY_SHP = ROOT / "data" / "ChinaMap" / "boundary.shp"

# Bounding box (same as reference notebook)
_BOUND_PTS = [(80, 15), (150, 50), (106.5, 2.8), (123, 24.5)]


def _get_bounds():
    """Reproject bounding points to TARGET_CRS."""
    gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in _BOUND_PTS], crs="EPSG:4326"
    ).to_crs(TARGET_CRS)
    return [g for g in gdf.geometry]


def _reproject_points(df, lon_col, lat_col):
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(x, y) for x, y in zip(df[lon_col], df[lat_col])],
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    out = df.copy()
    out["proj_x"] = gdf.geometry.x.values
    out["proj_y"] = gdf.geometry.y.values
    return out


def _reproject_linestring(wkt_str):
    geom = shapely_wkt.loads(str(wkt_str))
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(TARGET_CRS)
    return zip(*gdf.geometry.iloc[0].coords)


def _draw_basemap(ax, provinces, country):
    """Draw province + country basemap layers (reference style)."""
    provinces.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.2, zorder=0)
    country.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.75, zorder=1)


def main():
    nodes = pd.read_csv(ROOT / "inputs" / "pipeline_nodes.csv")
    edges = pd.read_csv(ROOT / "inputs" / "pipeline_candidate_edges.csv")

    # Load basemap
    provinces = gpd.read_file(PROVINCE_SHP).to_crs(TARGET_CRS)
    country = gpd.read_file(BOUNDARY_SHP).to_crs(TARGET_CRS)
    bounds = _get_bounds()  # [SW, NE, SCS_SW, SCS_NE]

    # Reproject nodes
    nodes_proj = _reproject_points(nodes, "lon", "lat")
    coord = nodes_proj.set_index("node_id")[["proj_x", "proj_y"]].to_dict("index")

    fig = plt.figure(figsize=(8, 8))
    plt.rc("font", size=10)
    ax = fig.add_subplot(1, 1, 1)

    # ── Basemap ──
    _draw_basemap(ax, provinces, country)

    # ── Helper: draw edge ──
    def _draw_edge(e, color, lw, alpha, zorder, ls="-"):
        wkt_str = e.get("geometry_wkt", "")
        if pd.notna(wkt_str) and str(wkt_str).startswith("LINESTRING"):
            xs, ys = _reproject_linestring(wkt_str)
            ax.plot(list(xs), list(ys), color=color, linewidth=lw, alpha=alpha,
                    linestyle=ls, solid_capstyle="round", zorder=zorder)
        else:
            f, t = coord.get(e["from_node_id"]), coord.get(e["to_node_id"])
            if f and t:
                ax.plot([f["proj_x"], t["proj_x"]], [f["proj_y"], t["proj_y"]],
                        color=color, linewidth=lw, alpha=alpha,
                        linestyle=ls, solid_capstyle="round", zorder=zorder)

    # ── 1. Existing corridors (curved) ──
    corridor_edges = edges[edges["edge_class"] == "existing_main_corridor"]
    gas_edges = corridor_edges[corridor_edges["corridor_type"] == "gas"]
    oil_edges = corridor_edges[corridor_edges["corridor_type"] == "oil"]

    for _, e in gas_edges.iterrows():
        _draw_edge(e, "#CC3311", 0.8, 0.35, 3)
    for _, e in oil_edges.iterrows():
        _draw_edge(e, "#EE7733", 0.6, 0.35, 3)

    # ── 2. Branch edges ──
    branch_edges = edges[edges["edge_class"].isin(["hub_to_corridor_branch", "corridor_to_storage_branch"])]
    for _, e in branch_edges.iterrows():
        _draw_edge(e, "#777777", 0.6, 0.65, 2, ls="--")

    # ── 3. Triangulation candidates ──
    tri_edges = edges[edges["edge_class"] == "triangulation_candidate"]
    storage_ids = set(nodes[nodes["node_type"] == "storage_hub"]["node_id"])
    plant_ids = set(nodes[nodes["node_type"] == "plant"]["node_id"])

    for _, e in tri_edges.iterrows():
        f, t = coord.get(e["from_node_id"]), coord.get(e["to_node_id"])
        if not (f and t):
            continue
        fid, tid = e["from_node_id"], e["to_node_id"]
        is_src_sink = ((fid in plant_ids and tid in storage_ids) or
                       (fid in storage_ids and tid in plant_ids))
        color = "#4477AA" if is_src_sink else "#888888"
        alpha = 0.65 if is_src_sink else 0.5
        ax.plot([f["proj_x"], t["proj_x"]], [f["proj_y"], t["proj_y"]],
                color=color, linewidth=0.6, alpha=alpha, zorder=2)

    # ── 4. Nodes ──
    plants = nodes_proj[nodes_proj["node_type"] == "plant"]
    ax.scatter(plants["proj_x"], plants["proj_y"],
               s=3, c="#4477AA", alpha=0.5, edgecolors="none", zorder=5)

    storage = nodes_proj[nodes_proj["node_type"] == "storage_hub"]
    ax.scatter(storage["proj_x"], storage["proj_y"],
               s=45, c="#882255", marker="D", edgecolors="black",
               linewidth=0.4, alpha=0.8, zorder=6)

    for _, row in storage.iterrows():
        ax.annotate(str(row["storage_hub_id"]),
                    xy=(row["proj_x"], row["proj_y"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=5, color="#882255", fontweight="bold", zorder=7)

    # ── Axes: off, set bounds ──
    ax.set_axis_off()
    ax.set_xlim(bounds[0].x, bounds[1].x)
    ax.set_ylim(bounds[0].y, bounds[1].y)

    # ── Legend ──
    n_plants = len(plants)
    n_storage = len(storage)
    legend_handles = [
        Line2D([], [], color="#CC3311", linewidth=1.0, alpha=0.7,
               label=f"Gas corridor ({len(gas_edges)})"),
        Line2D([], [], color="#EE7733", linewidth=0.8, alpha=0.7,
               label=f"Oil corridor ({len(oil_edges)})"),
        Line2D([], [], color="#BBBBBB", linewidth=0.5, alpha=0.4, linestyle="--",
               label=f"Branch ({len(branch_edges)})"),
        Line2D([], [], color="#4477AA", linewidth=0.5, alpha=0.4,
               label=f"Delaunay ({len(tri_edges)})"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#4477AA",
               markersize=3, alpha=0.6, label=f"Plant ({n_plants})"),
        Line2D([], [], marker="D", color="none", markerfacecolor="#882255",
               markeredgecolor="black", markeredgewidth=0.4,
               markersize=5, label=f"Storage ({n_storage})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              frameon=False, bbox_to_anchor=(1.15, 0.85))

    # ── Title ──
    ax.set_title(
        f"CO$_2$ Candidate Pipeline Network\n"
        f"{n_plants} plants + {n_storage} storage hubs | {len(edges)} edges",
        fontsize=10,
    )

    # ── South China Sea inset ──
    ax_scs = fig.add_axes([0.75, 0.18, 0.2, 0.2])
    _draw_basemap(ax_scs, provinces, country)
    # Redraw corridors on inset
    for _, e in gas_edges.iterrows():
        wkt_str = e.get("geometry_wkt", "")
        if pd.notna(wkt_str) and str(wkt_str).startswith("LINESTRING"):
            geom = shapely_wkt.loads(str(wkt_str))
            gdf_e = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(TARGET_CRS)
            xs, ys = zip(*gdf_e.geometry.iloc[0].coords)
            ax_scs.plot(list(xs), list(ys), color="#CC3311", linewidth=0.5, alpha=0.5)
    ax_scs.set_xlim(bounds[2].x, bounds[3].x)
    ax_scs.set_ylim(bounds[2].y, bounds[3].y)
    ax_scs.set_xticks([])
    ax_scs.set_yticks([])
    ax_scs.set_title("")
    ax_scs.set_xlabel("")
    ax_scs.set_ylabel("")

    fig.tight_layout()
    out_dir = ROOT / "inputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"candidate_network.{ext}")
    plt.close(fig)
    print(f"Saved to {out_dir / 'candidate_network.png'}")


if __name__ == "__main__":
    main()
