"""CO2 candidate network: oil/gas corridor backbone + pruned Delaunay triangulation.

Strategy
--------
1. Load gas + oil shapefiles -> sample waypoints every ~50 km -> build corridor graph G.
2. Load 300 plant-hub centroids + extract ~103 injection-center centroids.
3. Connect each terminal to its nearest corridor waypoint (branch edge added to G).
4. Delaunay triangulation on terminal nodes; keep only edges that are BOTH:
   - <= MAX_EDGE_KM (500 km) in direct distance, AND
   - NOT already "covered" by a corridor path (existing_path > direct_km * MAX_STRETCH=1.20)
5. Plot: oil/gas corridor geometry (real curves) + pruned triangulation edges + terminals.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
TRIANG_CRS = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
MAX_EDGE_KM = 500.0
MAX_STRETCH = 1.20
WAYPOINT_SPACING_KM = 50.0
R = 6371.0088


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lo1, la1, lo2, la2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def load_plant_sites() -> pd.DataFrame:
    hubs = pd.read_csv(ROOT / "inputs" / "plants_hub_300.csv")
    return (
        hubs[["centroid_longitude", "centroid_latitude"]]
        .rename(columns={"centroid_longitude": "lon", "centroid_latitude": "lat"})
        .dropna()
        .reset_index(drop=True)
    )


def extract_injection_centers() -> pd.DataFrame:
    base = ROOT / "data" / "封存汇图层-Fan"
    configs = [
        ("dsa", base / "DSA-storage potential.tif", 1.0),
        ("eor", base / "EOR-storage potential.tif", 10.0),
    ]
    rows: list[dict] = []
    for stype, spath, min_storage_mt in configs:
        with rasterio.open(spath) as src:
            storage = src.read(1).astype(np.float64)
            nodata = src.nodata
            transform = src.transform
            crs = src.crs
        if nodata is not None:
            storage[storage == nodata] = 0.0
        valid = storage > 0
        dilated = ndimage.binary_dilation(valid, iterations=2)
        labeled, _ = ndimage.label(dilated)
        rows_idx, cols_idx = np.where(valid)
        x_proj = transform.c + (cols_idx + 0.5) * transform.a
        y_proj = transform.f + (rows_idx + 0.5) * transform.e
        sv = storage[valid]
        comp_ids = labeled[valid]
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        for cid in np.unique(comp_ids):
            mask = comp_ids == cid
            total_st = float(sv[mask].sum())
            if total_st < min_storage_mt:
                continue
            w = sv[mask] / sv[mask].sum()
            cx = float((x_proj[mask] * w).sum())
            cy = float((y_proj[mask] * w).sum())
            lon, lat = tr.transform(cx, cy)
            rows.append({"lon": lon, "lat": lat, "storage_type": stype, "storage_mt": total_st})
    return pd.DataFrame(rows)


def _cumulative_geodesic(coords: list) -> list[float]:
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
    return cum


def _sample_waypoints(geom, spacing_km: float) -> list[tuple[float, float]]:
    coords = list(geom.coords)
    if len(coords) < 2:
        return [(coords[0][0], coords[0][1])] if coords else []
    cum = _cumulative_geodesic(coords)
    total_km = cum[-1]
    if total_km <= 0:
        return [(coords[0][0], coords[0][1])]
    n = max(1, int(np.ceil(total_km / spacing_km)))
    waypoints = []
    seg = 0
    for i in range(n + 1):
        d = i * total_km / n
        while seg < len(cum) - 2 and cum[seg + 1] <= d:
            seg += 1
        seg_len = cum[seg + 1] - cum[seg]
        t = (d - cum[seg]) / seg_len if seg_len > 0 else 0.0
        lon = coords[seg][0] + t * (coords[seg + 1][0] - coords[seg][0])
        lat = coords[seg][1] + t * (coords[seg + 1][1] - coords[seg][1])
        waypoints.append((round(lon, 5), round(lat, 5)))
    return waypoints


def build_corridor_graph() -> tuple[nx.Graph, list, list]:
    """Load shapefiles, sample waypoints, build weighted corridor graph.

    Returns
    -------
    graph : nx.Graph  (nodes = (lon, lat) tuples, edge weight = km)
    oil_segs : list of (xs, ys) for plotting original pipeline curves
    gas_segs : list of (xs, ys) for plotting original pipeline curves
    """
    graph = nx.Graph()
    oil_segs: list = []
    gas_segs: list = []

    def process_layer(gdf: gpd.GeoDataFrame, seg_list: list) -> None:
        for _, row in gdf.iterrows():
            geom = row.geometry
            parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
            for part in parts:
                xs, ys = zip(*part.coords)
                seg_list.append((xs, ys))
                wps = _sample_waypoints(part, WAYPOINT_SPACING_KM)
                if len(wps) < 2:
                    continue
                for wp in wps:
                    if wp not in graph:
                        graph.add_node(wp)
                for i in range(len(wps) - 1):
                    a, b = wps[i], wps[i + 1]
                    d = haversine_km(a[0], a[1], b[0], b[1])
                    if not graph.has_edge(a, b):
                        graph.add_edge(a, b, weight=d)

    data_dir = ROOT / "data" / "油气管道源数据矢量化"
    for filename, seg_list, label in [
        ("gas_pipelines.shp", gas_segs, "gas"),
        ("oil_pipeline.shp", oil_segs, "oil"),
    ]:
        gdf = gpd.read_file(data_dir / filename)
        if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        process_layer(gdf, seg_list)

    return graph, oil_segs, gas_segs


def connect_terminals_to_corridor(
    graph: nx.Graph,
    terminals: pd.DataFrame,
) -> dict[int, tuple[float, float]]:
    """Add each terminal as a graph node connected to nearest corridor waypoint.

    Returns mapping: terminal integer index -> node key (lon, lat) in graph.
    """
    corridor_nodes = list(graph.nodes())
    tr = Transformer.from_crs("EPSG:4326", TRIANG_CRS, always_xy=True)
    c_lons = np.array([n[0] for n in corridor_nodes])
    c_lats = np.array([n[1] for n in corridor_nodes])
    c_x, c_y = tr.transform(c_lons, c_lats)
    tree = cKDTree(np.column_stack([c_x, c_y]))

    terminal_keys: dict[int, tuple[float, float]] = {}
    for pos, (idx, row) in enumerate(terminals.iterrows()):
        lon, lat = float(row["lon"]), float(row["lat"])
        tx, ty = tr.transform(lon, lat)
        _, nn_idx = tree.query([tx, ty])
        nearest = corridor_nodes[int(nn_idx)]
        branch_d = haversine_km(lon, lat, nearest[0], nearest[1])
        node_key = (round(lon, 5), round(lat, 5))
        if node_key not in graph:
            graph.add_node(node_key)
            graph.add_edge(node_key, nearest, weight=branch_d)
        terminal_keys[pos] = node_key
    return terminal_keys


def build_pruned_triangulation(
    graph: nx.Graph,
    terminals: pd.DataFrame,
    terminal_keys: dict[int, tuple[float, float]],
) -> list[tuple[float, float, float, float]]:
    """Delaunay triangulation with corridor-coverage pruning."""
    tr = Transformer.from_crs("EPSG:4326", TRIANG_CRS, always_xy=True)
    x_p, y_p = tr.transform(terminals["lon"].to_numpy(), terminals["lat"].to_numpy())
    pts_xy = np.column_stack([x_p, y_p])
    simplices = Delaunay(pts_xy, qhull_options="QJ").simplices

    cand_pairs: set[tuple[int, int]] = set()
    for simplex in simplices:
        for a, b in combinations(sorted(int(v) for v in simplex), 2):
            cand_pairs.add((a, b))

    kept: list[tuple[float, float, float, float]] = []
    pruned = 0
    for a, b in cand_pairs:
        row_a = terminals.iloc[a]
        row_b = terminals.iloc[b]
        loa, la = float(row_a["lon"]), float(row_a["lat"])
        lob, lb = float(row_b["lon"]), float(row_b["lat"])
        direct_km = haversine_km(loa, la, lob, lb)
        if direct_km > MAX_EDGE_KM:
            continue
        key_a = terminal_keys[a]
        key_b = terminal_keys[b]
        try:
            existing_km = float(nx.shortest_path_length(graph, key_a, key_b, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            existing_km = float("inf")
        if np.isfinite(existing_km) and existing_km <= direct_km * MAX_STRETCH:
            pruned += 1
            continue
        kept.append((loa, la, lob, lb))
    print(f"  Delaunay pairs evaluated: {len(cand_pairs)}, pruned by corridor: {pruned}, kept: {len(kept)}")
    return kept


def main() -> None:
    out_dir = ROOT / "inputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading plant sites...")
    plants = load_plant_sites()
    print(f"  {len(plants)} plant hubs")

    print("Extracting injection centers...")
    injections = extract_injection_centers()
    dsa = injections[injections["storage_type"] == "dsa"]
    eor = injections[injections["storage_type"] == "eor"]
    print(f"  DSA: {len(dsa)}, EOR: {len(eor)}, total: {len(injections)}")

    print("Building corridor graph from shapefiles...")
    graph, oil_segs, gas_segs = build_corridor_graph()
    print(f"  Corridor nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

    print("Connecting terminals to corridor...")
    terminals = pd.concat([
        plants.assign(ttype="plant"),
        injections[["lon", "lat"]].assign(ttype="injection"),
    ], ignore_index=True).dropna().reset_index(drop=True)
    print(f"  Terminal nodes: {len(terminals)}")
    terminal_keys = connect_terminals_to_corridor(graph, terminals)

    print("Building pruned Delaunay triangulation...")
    tri_edges = build_pruned_triangulation(graph, terminals, terminal_keys)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 15), dpi=150)
    ax.set_facecolor("#f0f0e8")
    fig.patch.set_facecolor("#f8f8f4")

    # triangulation candidate edges (background)
    for loa, la, lob, lb in tri_edges:
        ax.plot([loa, lob], [la, lb], color="#566573", lw=1.0, alpha=0.60, zorder=1)

    # oil/gas corridor real curves (foreground skeleton)
    for xs, ys in oil_segs:
        ax.plot(xs, ys, color="#c0392b", lw=1.5, alpha=0.85, zorder=3)
    for xs, ys in gas_segs:
        ax.plot(xs, ys, color="#d68910", lw=1.2, alpha=0.75, zorder=3)

    # injection centers
    ax.scatter(dsa["lon"], dsa["lat"], marker="D", s=70, color="#8e44ad",
               zorder=7, alpha=0.95, edgecolors="white", linewidths=0.6)
    ax.scatter(eor["lon"], eor["lat"], marker="D", s=22, color="#e74c3c",
               zorder=6, alpha=0.70, edgecolors="none")

    # plant hubs
    ax.scatter(plants["lon"], plants["lat"], marker="^", s=10, color="#2980b9",
               zorder=5, alpha=0.55, edgecolors="none")

    legend_items = [
        mpatches.Patch(color="#c0392b", label="石油管道走廊"),
        mpatches.Patch(color="#d68910", label="天然气管道走廊"),
        mpatches.Patch(color="#566573", label=f"Delaunay 补边 ({len(tri_edges)} 条, <=500km, 走廊已覆盖剪枝)"),
        plt.scatter([], [], marker="^", color="#2980b9", s=40,
                    label=f"煤电厂集群 ({len(plants)} 个 hub)"),
        plt.scatter([], [], marker="D", color="#8e44ad", s=55, edgecolors="white",
                    label=f"DSA 注入中心 ({len(dsa)} 个)"),
        plt.scatter([], [], marker="D", color="#e74c3c", s=25,
                    label=f"EOR 注入中心 ({len(eor)} 个, >10 Mt)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=10, framealpha=0.93,
              edgecolor="#cccccc", facecolor="white", title="图例", title_fontsize=10)

    ax.set_xlim(72, 137)
    ax.set_ylim(17, 55)
    ax.set_xlabel("经度 (E)", fontsize=12)
    ax.set_ylabel("纬度 (N)", fontsize=12)
    ax.set_title(
        f"CO2 候选管网 | 油气走廊骨架 + Delaunay 补边\n"
        f"{len(plants)} 个煤电厂 hub + {len(dsa)} 个 DSA + {len(eor)} 个 EOR"
        f" | 走廊覆盖剪枝 stretch <= {MAX_STRETCH}",
        fontsize=13, pad=12,
    )
    ax.grid(True, color="white", linewidth=0.5, alpha=0.8)

    plt.tight_layout()
    out_path = out_dir / "co2_network_corridor_triangulation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
