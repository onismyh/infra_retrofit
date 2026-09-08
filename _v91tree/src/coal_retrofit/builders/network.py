from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Delaunay
from shapely.geometry import LineString
from shapely import wkt as shapely_wkt
from shapely.strtree import STRtree

from ..artifacts import write_csv
from ..constants import (
    NETWORK_COORD_DECIMALS,
    NETWORK_DETOUR_FACTOR,
    NETWORK_DIRECT_SINK_TOP_K,
    NETWORK_EDGE_CLASS_DIRECT,
    NETWORK_EDGE_CLASS_MAIN,
    NETWORK_EDGE_CLASS_TRIANGULATION,
    NETWORK_TRIANGULATION_CRS,
    NETWORK_TRIANGULATION_MAX_EDGE_KM,
    NETWORK_TRIANGULATION_MAX_STRETCH,
    STATIC_LAYER_YEAR_BASIS,
    TARGET_GEO_CRS,
)
from ..paths import ProjectPaths
from ..spatial import geodesic_length_km, geometry_parts
from .network_branches import build_attachment_tables

logger = logging.getLogger(__name__)


def load_corridor_layer(paths: ProjectPaths, filename: str, corridor_type: str) -> tuple[gpd.GeoDataFrame, Path, list[str]]:
    path = paths.find_data_file(filename)
    gdf = gpd.read_file(path)
    fields = [column for column in gdf.columns if column != "geometry"]
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS defined")
    if gdf.crs.to_string() != TARGET_GEO_CRS:
        gdf = gdf.to_crs(TARGET_GEO_CRS)
    gdf = gdf.reset_index(drop=True).copy()
    gdf["corridor_type"] = corridor_type
    gdf["source"] = paths.rel(path)
    gdf["year_basis"] = STATIC_LAYER_YEAR_BASIS
    return gdf, path, fields


def build_corridor_tables(paths: ProjectPaths, layers: list[tuple[gpd.GeoDataFrame, Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_stats: dict[tuple[float, float], dict[str, object]] = {}
    edge_rows: list[dict[str, object]] = []
    edge_counter = 0

    for gdf, path in layers:
        source = paths.rel(path)
        corridor_type = str(gdf["corridor_type"].iloc[0])
        for feature_index, row in enumerate(gdf.itertuples(index=False), start=1):
            for part_index, part in enumerate(geometry_parts(row.geometry), start=1):
                coords = list(part.coords)
                if len(coords) < 2:
                    continue

                wp_from = (round(coords[0][0], NETWORK_COORD_DECIMALS), round(coords[0][1], NETWORK_COORD_DECIMALS))
                wp_to   = (round(coords[-1][0], NETWORK_COORD_DECIMALS), round(coords[-1][1], NETWORK_COORD_DECIMALS))

                for wp in [wp_from, wp_to]:
                    stats = node_stats.setdefault(
                        wp,
                        {"lon": wp[0], "lat": wp[1], "degree": 0, "sources": set()},
                    )
                    stats["sources"].add(source)

                node_stats[wp_from]["degree"] = int(node_stats[wp_from]["degree"]) + 1
                node_stats[wp_to]["degree"]   = int(node_stats[wp_to]["degree"])   + 1

                length_km = geodesic_length_km(coords)
                edge_counter += 1
                edge_rows.append(
                    {
                        "edge_id": f"edge_{edge_counter:05d}",
                        "feature_index": feature_index,
                        "part_index": part_index,
                        "from_node_key": wp_from,
                        "to_node_key": wp_to,
                        "length_km": round(length_km, 3),
                        "direct_length_km": round(length_km, 3),
                        "tortuosity": 1.0,
                        "geometry_wkt": part.wkt,
                        "corridor_type": corridor_type,
                        "existing_corridor_flag": 1,
                        "edge_class": NETWORK_EDGE_CLASS_MAIN,
                        "source": source,
                        "year_basis": STATIC_LAYER_YEAR_BASIS,
                    }
                )

    node_keys = sorted(node_stats.keys(), key=lambda item: (item[0], item[1]))
    node_id_map = {key: f"node_{index:05d}" for index, key in enumerate(node_keys, start=1)}

    nodes = pd.DataFrame(
        {
            "node_id": node_id_map[key],
            "lon": node_stats[key]["lon"],
            "lat": node_stats[key]["lat"],
            "node_type": "corridor_junction" if int(node_stats[key]["degree"]) > 1 else "corridor_endpoint",
            "degree": int(node_stats[key]["degree"]),
            "source": ";".join(sorted(node_stats[key]["sources"])),
            "year_basis": STATIC_LAYER_YEAR_BASIS,
        }
        for key in node_keys
    ).sort_values(["lon", "lat", "node_id"]).reset_index(drop=True)

    edges = pd.DataFrame(
        {
            "edge_id": row["edge_id"],
            "feature_index": row["feature_index"],
            "part_index": row["part_index"],
            "from_node_id": node_id_map[row["from_node_key"]],
            "to_node_id": node_id_map[row["to_node_key"]],
            "length_km": row["length_km"],
            "direct_length_km": row.get("direct_length_km", row["length_km"]),
            "tortuosity": row.get("tortuosity", 1.0),
            "geometry_wkt": row.get("geometry_wkt", ""),
            "corridor_type": row["corridor_type"],
            "existing_corridor_flag": row["existing_corridor_flag"],
            "edge_class": row["edge_class"],
            "source": row["source"],
            "year_basis": row["year_basis"],
        }
        for row in edge_rows
    ).sort_values(["corridor_type", "feature_index", "part_index", "edge_id"]).reset_index(drop=True)

    return nodes, edges


def _unordered_edge_key(from_node_id: str, to_node_id: str) -> tuple[str, str]:
    return tuple(sorted((str(from_node_id), str(to_node_id))))


def _next_edge_index(edges: pd.DataFrame) -> int:
    """One past the highest numeric suffix in `edge_id`.

    Not `len(edges) + 1`: once the crossing filter has removed rows, the highest id in the table
    exceeds its length, and numbering from the length silently reissues ids that are already in
    use. `prepare_inputs` keys on `edge_id`, so duplicates are dropped rather than rejected -- an
    earlier build lost 410 candidate edges that way without a single warning.
    """
    if edges.empty or "edge_id" not in edges.columns:
        return 1
    suffixes = (
        edges["edge_id"].astype(str).str.extract(r"(\d+)$", expand=False).dropna().astype(int)
    )
    return int(suffixes.max()) + 1 if len(suffixes) else len(edges) + 1


def _project_points(nodes: pd.DataFrame) -> np.ndarray:
    transformer = Transformer.from_crs(TARGET_GEO_CRS, NETWORK_TRIANGULATION_CRS, always_xy=True)
    x, y = transformer.transform(
        nodes["lon"].astype(float).to_numpy(),
        nodes["lat"].astype(float).to_numpy(),
    )
    return np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])


def _knn_candidate_pairs(points_xy: np.ndarray, k: int = 3) -> set[tuple[int, int]]:
    candidate_pairs: set[tuple[int, int]] = set()
    point_count = len(points_xy)
    if point_count < 2:
        return candidate_pairs
    for idx in range(point_count):
        deltas = points_xy - points_xy[idx]
        distance_sq = np.einsum("ij,ij->i", deltas, deltas)
        ranked = np.argsort(distance_sq)
        for neighbor_idx in ranked[1 : min(point_count, k + 1)]:
            candidate_pairs.add(tuple(sorted((idx, int(neighbor_idx)))))
    return candidate_pairs


def _triangulation_candidate_pairs(points_xy: np.ndarray) -> set[tuple[int, int]]:
    point_count = len(points_xy)
    if point_count < 2:
        return set()
    if point_count == 2:
        return {(0, 1)}
    try:
        simplices = Delaunay(points_xy, qhull_options="QJ").simplices
    except Exception:
        return _knn_candidate_pairs(points_xy, k=3)

    candidate_pairs: set[tuple[int, int]] = set()
    for simplex in simplices:
        for idx_a, idx_b in combinations(sorted(int(value) for value in simplex), 2):
            candidate_pairs.add((idx_a, idx_b))
    return candidate_pairs


def _build_length_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(str(row.node_id))
    for row in edges.itertuples(index=False):
        graph.add_edge(
            str(row.from_node_id),
            str(row.to_node_id),
            length_km=float(row.length_km),
        )
    return graph


def build_triangulation_candidate_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    terminals = nodes.loc[
        nodes["node_type"].isin(["plant", "storage_hub"]),
        ["node_id", "lon", "lat", "node_type", "plant_id", "storage_hub_id", "source", "year_basis"],
    ].dropna(subset=["lon", "lat"]).reset_index(drop=True)
    if len(terminals) < 2:
        return pd.DataFrame(columns=list(edges.columns))

    points_xy = _project_points(terminals)
    candidate_pairs = _triangulation_candidate_pairs(points_xy)
    if not candidate_pairs:
        return pd.DataFrame(columns=list(edges.columns))

    graph = _build_length_graph(nodes, edges)
    existing_edge_keys = {
        _unordered_edge_key(str(row.from_node_id), str(row.to_node_id))
        for row in edges.itertuples(index=False)
    }

    edge_rows: list[dict[str, object]] = []
    next_edge_index = _next_edge_index(edges)
    feature_index = 1

    for idx_a, idx_b in sorted(candidate_pairs):
        node_a = terminals.iloc[idx_a]
        node_b = terminals.iloc[idx_b]
        from_node_id = str(node_a["node_id"])
        to_node_id = str(node_b["node_id"])
        edge_key = _unordered_edge_key(from_node_id, to_node_id)
        if edge_key in existing_edge_keys:
            continue

        direct_length_km = geodesic_length_km(
            [
                (float(node_a["lon"]), float(node_a["lat"])),
                (float(node_b["lon"]), float(node_b["lat"])),
            ]
        )
        if direct_length_km > NETWORK_TRIANGULATION_MAX_EDGE_KM:
            continue

        # A new pipeline does not run along the great circle. Existing-corridor edges in this
        # table already carry their routed polyline length, so leaving triangulation edges at
        # the raw geodesic made new build look cheaper than corridor reuse by construction.
        routed_length_km = direct_length_km * NETWORK_DETOUR_FACTOR

        try:
            existing_path_km = float(nx.shortest_path_length(graph, from_node_id, to_node_id, weight="length_km"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            existing_path_km = float("inf")

        if np.isfinite(existing_path_km) and existing_path_km <= routed_length_km * NETWORK_TRIANGULATION_MAX_STRETCH:
            continue

        line_geom = LineString([
            (float(node_a["lon"]), float(node_a["lat"])),
            (float(node_b["lon"]), float(node_b["lat"]))
        ])

        edge_rows.append(
            {
                "edge_id": f"edge_{next_edge_index:05d}",
                "feature_index": feature_index,
                "part_index": 1,
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "length_km": round(float(routed_length_km), 3),
                "direct_length_km": round(float(direct_length_km), 3),
                "tortuosity": NETWORK_DETOUR_FACTOR,
                "geometry_wkt": line_geom.wkt,
                "corridor_type": "triangulation",
                "existing_corridor_flag": 0,
                "edge_class": NETWORK_EDGE_CLASS_TRIANGULATION,
                "source": "source_sink_delaunay_rule",
                "year_basis": STATIC_LAYER_YEAR_BASIS,
            }
        )
        existing_edge_keys.add(edge_key)
        next_edge_index += 1
        feature_index += 1

    if not edge_rows:
        return pd.DataFrame(columns=list(edges.columns))
    return pd.DataFrame(edge_rows)


def _merge_network_tables(
    main_nodes: pd.DataFrame,
    main_edges: pd.DataFrame,
    branch_nodes: pd.DataFrame,
    branch_edges: pd.DataFrame,
    corridor_degree_additions: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = main_nodes.copy()
    if corridor_degree_additions:
        added_degree = nodes["node_id"].map(corridor_degree_additions).fillna(0).astype(int)
        nodes["degree"] = nodes["degree"].astype(int) + added_degree
        nodes["node_type"] = nodes["degree"].map(
            lambda degree: "corridor_junction" if int(degree) > 1 else "corridor_endpoint"
        )

    if not branch_nodes.empty:
        nodes = pd.concat([nodes, branch_nodes], ignore_index=True, sort=False)

    edges = main_edges.copy()
    if not branch_edges.empty:
        edges = pd.concat([edges, branch_edges], ignore_index=True, sort=False)

    node_columns = [
        "node_id",
        "lon",
        "lat",
        "node_type",
        "degree",
        "source",
        "year_basis",
        "plant_id",
        "plant_index",
        "storage_hub_id",
        "storage_hub_index",
        "province",
    ]
    edge_columns = [
        "edge_id",
        "feature_index",
        "part_index",
        "from_node_id",
        "to_node_id",
        "length_km",
        "direct_length_km",
        "tortuosity",
        "geometry_wkt",
        "corridor_type",
        "existing_corridor_flag",
        "edge_class",
        "source",
        "year_basis",
    ]

    for column in node_columns:
        if column not in nodes.columns:
            nodes[column] = pd.NA
    for column in edge_columns:
        if column not in edges.columns:
            edges[column] = pd.NA

    return (
        nodes[node_columns].sort_values(["node_id"]).reset_index(drop=True),
        edges[edge_columns].sort_values(["edge_id"]).reset_index(drop=True),
    )


_EDGE_CLASS_PRIORITY = {
    "existing_main_corridor": 0,
    "hub_to_corridor_branch": 1,
    "corridor_to_storage_branch": 1,
    "triangulation_candidate": 2,
}


def _build_edge_geometry(row: pd.Series, node_coords: dict[str, tuple[float, float]]) -> LineString | None:
    """Build a Shapely LineString for an edge, using WKT if available."""
    wkt_str = row.get("geometry_wkt", "")
    if isinstance(wkt_str, str) and wkt_str.startswith("LINESTRING"):
        return shapely_wkt.loads(wkt_str)
    from_id = str(row["from_node_id"])
    to_id = str(row["to_node_id"])
    c1 = node_coords.get(from_id)
    c2 = node_coords.get(to_id)
    if c1 and c2:
        return LineString([c1, c2])
    return None


def _terminal_ids(nodes: pd.DataFrame) -> tuple[set[str], set[str]]:
    plants = set(nodes.loc[nodes["node_type"] == "plant", "node_id"].astype(str))
    sinks = set(nodes.loc[nodes["node_type"] == "storage_hub", "node_id"].astype(str))
    return plants, sinks


def _unreached_terminals(nodes: pd.DataFrame, edges: pd.DataFrame) -> set[str]:
    """Plant and sink nodes that cannot reach a sink over *edges*."""
    plants, sinks = _terminal_ids(nodes)
    graph = nx.Graph()
    graph.add_nodes_from(plants | sinks)
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.from_node_id), str(row.to_node_id))
    component = {node: idx for idx, part in enumerate(nx.connected_components(graph)) for node in part}
    sink_components = {component[s] for s in sinks if s in component}
    return {p for p in plants if component.get(p) not in sink_components}


def _repair_connectivity(
    nodes: pd.DataFrame, kept: pd.DataFrame, dropped: pd.DataFrame
) -> pd.DataFrame:
    """Put back the shortest dropped edges needed to reconnect every terminal.

    The crossing filter is a tidiness rule, not a physical one, and applied blindly it severs
    terminals: on the network this replaces, 7 plants (20.4 GW) could reach no sink at all, one
    of them 24.5 km from one. SimCCS treats source and sink connectivity as an invariant that
    survives however heavily the cost surface discourages crossings, and that is the behaviour
    reproduced here: filter first, then restore the cheapest edges that buy back connectivity.
    """
    if dropped.empty:
        return kept
    order = dropped.sort_values("length_km")
    restored: list[int] = []
    for position in range(len(order)):
        if not _unreached_terminals(nodes, kept):
            break
        candidate = order.iloc[[position]]
        trial = pd.concat([kept, candidate], ignore_index=True)
        if len(_unreached_terminals(nodes, trial)) < len(_unreached_terminals(nodes, kept)):
            kept = trial
            restored.append(position)
    if restored:
        logger.info("restored %d crossing edges to keep every terminal connected", len(restored))
    return kept


def _remove_crossing_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Remove lower-priority edges that cross higher-priority ones, then repair connectivity.

    Priority: existing corridors > branches > triangulation candidates. The result is planar
    wherever planarity costs nothing, and connected everywhere -- see `_repair_connectivity`.
    """
    node_coords = {
        str(row.node_id): (float(row.lon), float(row.lat))
        for row in nodes.itertuples(index=False)
    }

    # Build geometries and sort by priority
    records: list[tuple[int, int, LineString]] = []  # (priority, df_index, geom)
    for idx, row in edges.iterrows():
        geom = _build_edge_geometry(row, node_coords)
        if geom is not None and not geom.is_empty:
            priority = _EDGE_CLASS_PRIORITY.get(row.get("edge_class", ""), 2)
            records.append((priority, idx, geom))

    # Sort: lowest priority number first (keep these)
    records.sort(key=lambda x: x[0])

    kept_indices: list[int] = []
    kept_geoms: list[LineString] = []

    for priority, idx, geom in records:
        crosses_any = False
        for kept_geom in kept_geoms:
            if geom.crosses(kept_geom):
                crosses_any = True
                break
        if not crosses_any:
            kept_indices.append(idx)
            kept_geoms.append(geom)

    removed = len(edges) - len(kept_indices)
    if removed > 0:
        logger.info("removed %d crossing candidate edges (%d kept)", removed, len(kept_indices))

    kept = edges.loc[kept_indices].reset_index(drop=True)
    dropped = edges.drop(index=kept_indices).reset_index(drop=True)
    return _repair_connectivity(nodes, kept, dropped).reset_index(drop=True)


def build_direct_sink_edges(
    nodes: pd.DataFrame, edges: pd.DataFrame, top_k: int = NETWORK_DIRECT_SINK_TOP_K
) -> pd.DataFrame:
    """Direct candidate arcs from every plant to its *top_k* nearest sinks.

    This is the rule `OptimizationAssumptions.top_k_storage_pairs` has always declared and that
    nothing ever produced. The arcs are priced at `direct_fallback_capex_multiplier` (2.8x), the
    premium the assumptions already carry for a dedicated line built outside any corridor, so
    the optimiser takes one only where the corridor network really is the long way round.

    They are ADDED, never substituted: the candidate set can only grow, so no solution that was
    reachable before becomes unreachable.
    """
    plants = nodes.loc[nodes["node_type"] == "plant", ["node_id", "lon", "lat"]].dropna().reset_index(drop=True)
    sinks = nodes.loc[nodes["node_type"] == "storage_hub", ["node_id", "lon", "lat"]].dropna().reset_index(drop=True)
    if plants.empty or sinks.empty:
        return pd.DataFrame(columns=list(edges.columns))

    existing = {
        _unordered_edge_key(str(row.from_node_id), str(row.to_node_id))
        for row in edges.itertuples(index=False)
    }
    sink_lon = sinks["lon"].astype(float).to_numpy()
    sink_lat = sinks["lat"].astype(float).to_numpy()

    rows: list[dict[str, object]] = []
    next_index = _next_edge_index(edges)
    for plant in plants.itertuples(index=False):
        lon, lat = float(plant.lon), float(plant.lat)
        separation = np.array([
            geodesic_length_km([(lon, lat), (float(sl), float(sa))])
            for sl, sa in zip(sink_lon, sink_lat)
        ])
        for rank in np.argsort(separation)[: max(0, int(top_k))]:
            sink = sinks.iloc[int(rank)]
            key = _unordered_edge_key(str(plant.node_id), str(sink["node_id"]))
            if key in existing:
                continue
            existing.add(key)
            direct_km = float(separation[int(rank)])
            geom = LineString([(lon, lat), (float(sink["lon"]), float(sink["lat"]))])
            rows.append({
                "edge_id": f"edge_{next_index:05d}",
                "feature_index": next_index,
                "part_index": 1,
                "from_node_id": str(plant.node_id),
                "to_node_id": str(sink["node_id"]),
                "length_km": round(direct_km * NETWORK_DETOUR_FACTOR, 3),
                "direct_length_km": round(direct_km, 3),
                "tortuosity": NETWORK_DETOUR_FACTOR,
                "geometry_wkt": geom.wkt,
                "corridor_type": "direct_source_sink",
                "existing_corridor_flag": 0,
                "edge_class": NETWORK_EDGE_CLASS_DIRECT,
                "source": "source_sink_top_k_rule",
                "year_basis": STATIC_LAYER_YEAR_BASIS,
            })
            next_index += 1

    if not rows:
        return pd.DataFrame(columns=list(edges.columns))
    return pd.DataFrame(rows, columns=list(edges.columns))


def build_network_tables(paths: ProjectPaths, use_corridors: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_corridors:
        gas_layer, gas_path, _ = load_corridor_layer(paths, "gas_pipelines.shp", "gas")
        oil_layer, oil_path, _ = load_corridor_layer(paths, "oil_pipeline.shp", "oil")
        main_nodes, main_edges = build_corridor_tables(paths, [(gas_layer, gas_path), (oil_layer, oil_path)])

        corridor_nodes = main_nodes.loc[
            main_nodes["node_type"].isin(["corridor_junction", "corridor_endpoint"]),
            ["node_id", "lon", "lat"],
        ].copy()
        branch_nodes, branch_edges, corridor_degree_additions = build_attachment_tables(
            paths=paths,
            corridor_nodes=corridor_nodes,
            main_edges=main_edges,
            next_node_index=len(main_nodes) + 1,
            next_edge_index=len(main_edges) + 1,
        )
        nodes, edges = _merge_network_tables(main_nodes, main_edges, branch_nodes, branch_edges, corridor_degree_additions)
    else:
        # Terminal-only network: plants + storage hubs, no corridors
        nodes, edges = _build_terminal_only_nodes(paths)

    triangulation_edges = build_triangulation_candidate_edges(nodes, edges)
    if not triangulation_edges.empty:
        edges = pd.concat([edges, triangulation_edges], ignore_index=True, sort=False)
        edges = edges.sort_values(["edge_id"]).reset_index(drop=True)
    # Prefer planarity, but never at the cost of disconnecting a terminal
    edges = _remove_crossing_edges(nodes, edges)
    # Direct plant-to-sink arcs last, so they are never themselves filtered out
    direct_edges = build_direct_sink_edges(nodes, edges)
    if not direct_edges.empty:
        edges = pd.concat([edges, direct_edges], ignore_index=True, sort=False)
        edges = edges.sort_values(["edge_id"]).reset_index(drop=True)
    duplicates = int(edges["edge_id"].duplicated().sum())
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate edge_id values; downstream code keys on edge_id and would "
            f"drop them silently"
        )
    stranded = _unreached_terminals(nodes, edges)
    if stranded:
        raise ValueError(
            f"{len(stranded)} plant nodes still cannot reach a storage hub: {sorted(stranded)[:8]}"
        )
    logger.info(
        "candidate network: %d nodes, %d edges (%s)",
        len(nodes), len(edges),
        ", ".join(f"{k} {v}" for k, v in edges["edge_class"].value_counts().items()),
    )
    return nodes, edges


def _build_terminal_only_nodes(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build node table from plants and storage hubs only (no corridor nodes)."""
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    storage = pd.read_csv(paths.inputs_dir / "storage_hubs.csv")

    node_rows: list[dict] = []
    node_index = 1
    for _, row in plants.iterrows():
        node_rows.append({
            "node_id": f"node_{node_index:05d}",
            "lon": row["centroid_longitude"],
            "lat": row["centroid_latitude"],
            "node_type": "plant",
            "degree": 0,
            "source": row.get("source", ""),
            "year_basis": row.get("year_basis", ""),
            "plant_id": row["plant_id"],
            "plant_index": row["plant_index"],
            "storage_hub_id": pd.NA,
            "storage_hub_index": pd.NA,
            "province": row.get("province_mode", ""),
        })
        node_index += 1
    for _, row in storage.iterrows():
        node_rows.append({
            "node_id": f"node_{node_index:05d}",
            "lon": row["longitude"],
            "lat": row["latitude"],
            "node_type": "storage_hub",
            "degree": 0,
            "source": row.get("source", ""),
            "year_basis": row.get("year_basis", ""),
            "plant_id": pd.NA,
            "plant_index": pd.NA,
            "storage_hub_id": row["storage_hub_id"],
            "storage_hub_index": row["storage_hub_index"],
            "province": row.get("province", ""),
        })
        node_index += 1

    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(columns=[
        "edge_id", "feature_index", "part_index", "from_node_id", "to_node_id",
        "length_km", "direct_length_km", "tortuosity", "geometry_wkt",
        "corridor_type", "existing_corridor_flag", "edge_class", "source", "year_basis",
    ])
    return nodes, edges


def write_network_inputs(paths: ProjectPaths, use_corridors: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = build_network_tables(paths, use_corridors=use_corridors)
    write_csv(nodes, paths.inputs_dir / "pipeline_nodes.csv")
    write_csv(edges, paths.inputs_dir / "pipeline_candidate_edges.csv")
    return nodes, edges
