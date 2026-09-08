"""Add the source-to-sink candidate arcs the model was designed to have and never got.

WHAT IS MISSING. `OptimizationAssumptions` declares `top_k_storage_pairs = 5` and
`direct_fallback_capex_multiplier = 2.8`, and `data_prep._edge_capex_multiplier` prices an edge
class called `runtime_direct_fallback`. Nothing in the repository ever emits that class, and
`top_k_storage_pairs` is never read anywhere. The candidate network is therefore only what
`builders.network` produces: existing oil and gas corridors, their branches, and a Delaunay
triangulation over terminals that is then filtered to be PLANAR.

WHAT THAT COSTS. Planarity is not a property CO2 pipelines have -- they cross. Enforcing it
greedily drops whichever crossing edge comes later in the file, and that disconnects terminals:

    7 plants (20.4 GW) have no path to any storage hub at all, one of them 24.5 km from one
    401 GW is routed at more than 1.5x its straight-line distance to the nearest sink
    the worst case, P0077, sits 12.0 km from a hub and is routed 617.7 km (51x)

At the model's own levelised transport cost of 0.179 CNY/(t km) that is a capacity-weighted
+17.7 CNY/t of pure routing artefact, against a CCS retrofit capex of about 118 CNY/t. For the
140 GW carrying more than 100 CNY/t it roughly doubles the delivered cost of capture.

WHAT THIS SCRIPT DOES. For every plant it adds direct candidate arcs to its k nearest storage
hubs, length = geodesic x TORTUOSITY, class `runtime_direct_fallback` so the 2.8x premium the
assumptions already carry is applied. It adds candidates; it removes nothing. The optimiser
still chooses whether to build them, and at 2.8x capex it will only do so where the corridor
network really is the long way round.

Usage:
    python scripts/build_direct_source_sink_edges.py            # report only
    python scripts/build_direct_source_sink_edges.py --write    # rewrite pipeline_candidate_edges.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from _bootstrap import ROOT

# Terrain factor on a straight line. IEAGHG/ZEP and the SimCCS-family studies use 1.2-1.4 when
# a routed cost surface is unavailable; 1.3 is the middle of that range.
TORTUOSITY = 1.3
TOP_K = 5
EDGE_CLASS = "runtime_direct_fallback"
INPUTS = ROOT / "inputs"


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def nearest_sink_distance(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Network and geodesic distance from every plant node to its nearest storage hub."""
    sinks = nodes[nodes["node_type"] == "storage_hub"]
    sink_ids = set(sinks["node_id"].astype(str))
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.from_node_id), str(row.to_node_id), w=float(row.length_km))

    out = []
    for row in nodes[nodes["node_type"] == "plant"].itertuples(index=False):
        geo = float(haversine_km(row.lon, row.lat,
                                 sinks["lon"].to_numpy(), sinks["lat"].to_numpy()).min())
        net = np.nan
        node_id = str(row.node_id)
        if node_id in graph:
            lengths = nx.single_source_dijkstra_path_length(graph, node_id, weight="w")
            reach = [lengths[s] for s in sink_ids if s in lengths]
            if reach:
                net = min(reach)
        out.append({"node_id": node_id, "plant_id": row.plant_id, "geo_km": geo, "net_km": net})
    return pd.DataFrame(out)


def build_direct_edges(nodes: pd.DataFrame, edges: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    plants = nodes[nodes["node_type"] == "plant"].reset_index(drop=True)
    sinks = nodes[nodes["node_type"] == "storage_hub"].reset_index(drop=True)
    existing = {
        tuple(sorted((str(r.from_node_id), str(r.to_node_id))))
        for r in edges.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    next_index = len(edges) + 1
    for plant in plants.itertuples(index=False):
        d = haversine_km(plant.lon, plant.lat, sinks["lon"].to_numpy(), sinks["lat"].to_numpy())
        for rank in np.argsort(d)[:top_k]:
            sink = sinks.iloc[int(rank)]
            key = tuple(sorted((str(plant.node_id), str(sink["node_id"]))))
            if key in existing:
                continue
            existing.add(key)
            length = float(d[int(rank)]) * TORTUOSITY
            geom = LineString([(float(plant.lon), float(plant.lat)),
                               (float(sink["lon"]), float(sink["lat"]))])
            rows.append({
                "edge_id": f"edge_{next_index:05d}",
                "feature_index": next_index,
                "part_index": 1,
                "from_node_id": str(plant.node_id),
                "to_node_id": str(sink["node_id"]),
                "length_km": round(length, 3),
                "direct_length_km": round(float(d[int(rank)]), 3),
                "tortuosity": TORTUOSITY,
                "geometry_wkt": geom.wkt,
                "corridor_type": "direct_source_sink",
                "existing_corridor_flag": 0,
                "edge_class": EDGE_CLASS,
                "source": "source_sink_top_k_rule",
                "year_basis": str(edges["year_basis"].iloc[0]) if len(edges) else "2025",
            })
            next_index += 1
    return pd.DataFrame(rows, columns=list(edges.columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite inputs/pipeline_candidate_edges.csv")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    nodes = pd.read_csv(INPUTS / "pipeline_nodes.csv", encoding="utf-8-sig")
    edges = pd.read_csv(INPUTS / "pipeline_candidate_edges.csv", encoding="utf-8-sig")
    plants = pd.read_csv(INPUTS / "plants.csv", encoding="utf-8-sig")
    gw = plants.set_index("plant_id")["total_capacity_mw"].div(1e3)

    before = nearest_sink_distance(nodes, edges)
    new_edges = build_direct_edges(nodes, edges, top_k=args.top_k)
    after = nearest_sink_distance(nodes, pd.concat([edges, new_edges], ignore_index=True))

    m = before.merge(after, on=["node_id", "plant_id"], suffixes=("_b", "_a"))
    m["gw"] = m["plant_id"].map(gw)
    m["ratio_b"] = m["net_km_b"] / m["geo_km_b"]
    m["ratio_a"] = m["net_km_a"] / m["geo_km_a"]
    total = m["gw"].sum()

    print(f"added {len(new_edges)} direct source-sink arcs (top {args.top_k}, tortuosity {TORTUOSITY})")
    print(f"  edges {len(edges)} -> {len(edges) + len(new_edges)}")
    print()
    print("                                   before      after")
    print(f"  plants with no path to a sink   {int(m['net_km_b'].isna().sum()):7d}    {int(m['net_km_a'].isna().sum()):7d}")
    for thr in (1.5, 2.0, 3.0):
        b = m[m["ratio_b"] > thr]["gw"].sum()
        a = m[m["ratio_a"] > thr]["gw"].sum()
        print(f"  GW routed > {thr}x straight line  {b:7.1f}    {a:7.1f}")
    print(f"  median detour ratio             {m['ratio_b'].median():7.2f}    {m['ratio_a'].median():7.2f}")
    print(f"  p90 detour ratio                {m['ratio_b'].quantile(0.9):7.2f}    {m['ratio_a'].quantile(0.9):7.2f}")
    print(f"  max detour ratio                {m['ratio_b'].max():7.2f}    {m['ratio_a'].max():7.2f}")
    saved = (m["net_km_b"] - m["net_km_a"]).fillna(0.0)
    per_gw = float((saved * m["gw"]).sum() / total)
    print(f"  capacity-weighted route saving  {per_gw:7.0f} km  = {per_gw * 0.179:.1f} CNY/t CO2")

    if args.write:
        out = pd.concat([edges, new_edges], ignore_index=True)
        backup = INPUTS / "pipeline_candidate_edges.csv.bak_pre_direct"
        if not backup.exists():
            edges.to_csv(backup, index=False, encoding="utf-8-sig")
            print(f"\n  original saved to {backup.name}")
        out.to_csv(INPUTS / "pipeline_candidate_edges.csv", index=False, encoding="utf-8-sig")
        print(f"  wrote {len(out)} edges to inputs/pipeline_candidate_edges.csv")


if __name__ == "__main__":
    main()
