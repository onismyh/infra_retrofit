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
        "industry_hub_id",
        "industry_index",
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
    # 工业点源和煤电一样是"必须能到汇"的源端：它们现在进了备选网络，
    # 去交叉之后的连通性修复必须把它们一起照顾到，否则又回到求解时拉直连线的老路。
    sources = set(nodes.loc[nodes["node_type"].isin(["plant", "industry_hub"]), "node_id"].astype(str))
    sinks = set(nodes.loc[nodes["node_type"] == "storage_hub", "node_id"].astype(str))
    return sources, sinks


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


MIN_STITCH_LENGTH_KM = 1.0


def _repair_connectivity(
    nodes: pd.DataFrame, kept: pd.DataFrame, dropped: pd.DataFrame
) -> pd.DataFrame:
    """把每一个源（煤电 + 工业点源）重新接回封存汇，且不重新引入交叉。

    去交叉是整洁性规则，不是物理规则；照字面执行会切断源端，工业点源进网后有 124 个
    片区因此拿不到汇。原实现从"被删的交叉边"里挑最短的恢复——可是一条边当初被删正是
    因为它与保留边相交，恢复它就等于把交叉放回图上（实测 136 条，62 处新建线交叉）。
    这里改为按连通片缝合：每次取一个缺汇片区，在它与含汇片区之间找"最短且不与任何
    保留边相交"的节点对，新增一条新建候选边（source 记为 connectivity_stitch_rule）。
    片区数严格递减，因此必然终止；只有在所有候选对都相交时才退回最短那条。
    `dropped` 仅用于日志，不再回填。
    """
    coords = {str(r.node_id): (float(r.lon), float(r.lat)) for r in nodes.itertuples(index=False)}
    sources, sinks = _terminal_ids(nodes)
    geometries = [g for g in (_build_edge_geometry(row, coords) for _, row in kept.iterrows())
                  if g is not None and not g.is_empty]

    def _components(frame: pd.DataFrame) -> dict[str, int]:
        graph = nx.Graph()
        graph.add_nodes_from(coords)
        for row in frame.itertuples(index=False):
            graph.add_edge(str(row.from_node_id), str(row.to_node_id))
        return {n: i for i, part in enumerate(nx.connected_components(graph)) for n in part}

    def _haversine_km(lon0: float, lat0: float, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        lo, la = np.radians(lon0), np.radians(lat0)
        hav = np.sin((lats - la) / 2.0) ** 2 + np.cos(la) * np.cos(lats) * np.sin((lons - lo) / 2.0) ** 2
        return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))

    stitched, forced = 0, 0
    while True:
        component = _components(kept)
        with_sink = {component[s] for s in sinks if s in component}
        needy = sorted({component[s] for s in sources if s in component and component[s] not in with_sink})
        if not needy:
            break
        group_ids = [n for n, c in component.items() if c == needy[0]]
        target_ids = [n for n, c in component.items() if c in with_sink]
        if not group_ids or not target_ids:
            break
        target_lon = np.radians(np.array([coords[n][0] for n in target_ids], dtype=np.float64))
        target_lat = np.radians(np.array([coords[n][1] for n in target_ids], dtype=np.float64))
        pairs: list[tuple[float, str, str]] = []
        for node_a in group_ids:
            distances = _haversine_km(coords[node_a][0], coords[node_a][1], target_lon, target_lat)
            for position in np.argsort(distances)[:12]:
                pairs.append((float(distances[position]), node_a, target_ids[int(position)]))
        pairs.sort()
        tree = STRtree(geometries)
        chosen = None
        for direct_km, node_a, node_b in pairs[:400]:
            line = LineString([coords[node_a], coords[node_b]])
            if not any(line.crosses(geometries[int(k)]) for k in tree.query(line)):
                chosen = (direct_km, node_a, node_b, line)
                break
        if chosen is None:
            direct_km, node_a, node_b = pairs[0]
            chosen = (direct_km, node_a, node_b, LineString([coords[node_a], coords[node_b]]))
            forced += 1
        direct_km, node_a, node_b, line = chosen
        direct_km = max(float(direct_km), MIN_STITCH_LENGTH_KM)
        stitched += 1
        record = {column: pd.NA for column in kept.columns}
        record.update({
            "edge_id": f"edge_stitch_{stitched:05d}",
            "feature_index": -1,
            "part_index": 1,
            "from_node_id": node_a,
            "to_node_id": node_b,
            "length_km": round(direct_km * NETWORK_DETOUR_FACTOR, 3),
            "direct_length_km": round(direct_km, 3),
            "tortuosity": NETWORK_DETOUR_FACTOR,
            "geometry_wkt": line.wkt,
            "corridor_type": "triangulation",
            "existing_corridor_flag": 0,
            "edge_class": NETWORK_EDGE_CLASS_TRIANGULATION,
            "source": "connectivity_stitch_rule",
            "year_basis": STATIC_LAYER_YEAR_BASIS,
        })
        kept = pd.concat([kept, pd.DataFrame([record])], ignore_index=True)
        geometries.append(line)
    if stitched:
        logger.info("stitched %d candidate edges (%d unavoidably crossing); %d dropped edges left out",
                    stitched, forced, len(dropped))
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

    # Sort: lowest priority number first (keep these); within a priority, SHORTEST first.
    # 同优先级下按建表顺序取舍是任意的，长边先占位会把一片短边全挤掉：实测按长度排序，
    # 保留边 1224 -> 1269，连通片 173 -> 157，每个源平均可选汇 32 -> 56。
    records.sort(key=lambda x: (x[0], x[2].length))

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


EXCLUDED_REGION_NAMES = ("西藏", "西藏自治区")


def _drop_edges_over_excluded_region(paths, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """丢掉穿过不参与减排省份（西藏）的三角剖分候选边。

    这些边不是任何源或汇的接入线——西藏没有封存汇，工业点源已在装载时剔除——
    它们只是 Delaunay 在空旷西部连出来的长边，留着既不会被选中，又让图上看着像有管网。
    既有管廊与接入支线不动：那是真实存在的管道。
    """
    import geopandas as gpd
    from shapely.geometry import LineString as _LS

    path = paths.data_dir / "ChinaMap" / "provinces.shp"
    if not path.exists():
        return edges
    prov = gpd.read_file(path)
    hit = prov[prov["NAME"].astype(str).isin(EXCLUDED_REGION_NAMES)]
    if hit.empty:
        return edges
    region = hit.to_crs("EPSG:4326").geometry.union_all()
    coords = {str(r.node_id): (float(r.lon), float(r.lat)) for r in nodes.itertuples(index=False)}
    drop = []
    for row in edges.itertuples(index=False):
        if str(row.edge_class) not in (NETWORK_EDGE_CLASS_TRIANGULATION, NETWORK_EDGE_CLASS_DIRECT):
            continue
        a, b = coords.get(str(row.from_node_id)), coords.get(str(row.to_node_id))
        if a is None or b is None:
            continue
        if _LS([a, b]).intersects(region):
            drop.append(str(row.edge_id))
    if drop:
        logger.info("dropped %d triangulation edges crossing %s", len(drop), EXCLUDED_REGION_NAMES[0])
        edges = edges[~edges["edge_id"].astype(str).isin(set(drop))].reset_index(drop=True)
    return edges


def _excluded_region(paths: ProjectPaths):
    """不参与减排省份的几何；没有图层时返回 None。"""
    path = paths.data_dir / "ChinaMap" / "provinces.shp"
    if not path.exists():
        return None
    hit = gpd.read_file(path)
    hit = hit[hit["NAME"].astype(str).isin(EXCLUDED_REGION_NAMES)]
    return None if hit.empty else hit.to_crs("EPSG:4326").geometry.union_all()


def _merge_components(paths: ProjectPaths, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """把去交叉后残留的连通片并成一张网，每次补一条最短、不交叉、不过西藏的连线。

    `_repair_connectivity` 只保证"每个源能到某个汇"，这不够：去交叉把图切成 45 片以后，
    南海那个 200 Mtpa 的离岸汇 S002 落在 35 节点的小片里，全国只有 28 个源够得着它，于是
    IND_*_t95 的 2060 年有 26 个小汇被超注入 196 Mt/yr，而 S002 闲置 162 Mt/yr。
    备选网络不该替优化器决定"谁不许去哪个汇"；要不要真建这些管段仍由优化器决定。
    """
    coords = {str(r.node_id): (float(r.lon), float(r.lat)) for r in nodes.itertuples(index=False)}
    region = _excluded_region(paths)
    geometries = [g for g in (_build_edge_geometry(row, coords) for _, row in edges.iterrows())
                  if g is not None and not g.is_empty]
    ids = list(coords)
    lon = np.radians(np.array([coords[n][0] for n in ids], dtype=np.float64))
    lat = np.radians(np.array([coords[n][1] for n in ids], dtype=np.float64))
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.from_node_id), str(row.to_node_id))

    added: list[dict[str, object]] = []
    forced = 0
    while True:
        component = {n: i for i, part in enumerate(nx.connected_components(graph)) for n in part}
        if len(set(component.values())) <= 1:
            break
        label = np.array([component[n] for n in ids])
        candidates: list[tuple[float, int, int]] = []
        for group in sorted(set(label.tolist())):
            inside = np.flatnonzero(label == group)
            outside = np.flatnonzero(label != group)
            for index in inside:
                hav = (np.sin((lat[outside] - lat[index]) / 2.0) ** 2
                       + np.cos(lat[index]) * np.cos(lat[outside])
                       * np.sin((lon[outside] - lon[index]) / 2.0) ** 2)
                dist = 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
                pick = int(np.argmin(dist))
                candidates.append((float(dist[pick]), int(index), int(outside[pick])))
        candidates.sort()
        chosen = None
        for direct_km, a_index, b_index in candidates[:600]:
            node_a, node_b = ids[a_index], ids[b_index]
            line = LineString([coords[node_a], coords[node_b]])
            if region is not None and line.intersects(region):
                continue
            if any(line.crosses(other) for other in geometries):
                continue
            chosen = (direct_km, node_a, node_b, line)
            break
        if chosen is None:
            direct_km, a_index, b_index = candidates[0]
            node_a, node_b = ids[a_index], ids[b_index]
            chosen = (direct_km, node_a, node_b, LineString([coords[node_a], coords[node_b]]))
            forced += 1
        direct_km, node_a, node_b, line = chosen
        direct_km = max(float(geodesic_length_km([coords[node_a], coords[node_b]])),
                        MIN_STITCH_LENGTH_KM)
        graph.add_edge(node_a, node_b)
        geometries.append(line)
        record = {column: pd.NA for column in edges.columns}
        record.update({
            "edge_id": f"edge_merge_{len(added) + 1:05d}",
            "feature_index": -3,
            "part_index": 1,
            "from_node_id": node_a,
            "to_node_id": node_b,
            "length_km": round(direct_km * NETWORK_DETOUR_FACTOR, 3),
            "direct_length_km": round(direct_km, 3),
            "tortuosity": NETWORK_DETOUR_FACTOR,
            "geometry_wkt": line.wkt,
            "corridor_type": "triangulation",
            "existing_corridor_flag": 0,
            "edge_class": NETWORK_EDGE_CLASS_TRIANGULATION,
            "source": "component_merge_rule",
            "year_basis": STATIC_LAYER_YEAR_BASIS,
        })
        added.append(record)
    if added:
        logger.info("merged the network into one component with %d links (%d unavoidable)",
                    len(added), forced)
        edges = pd.concat([edges, pd.DataFrame(added)], ignore_index=True, sort=False)
    return edges


def build_direct_sink_edges(
    nodes: pd.DataFrame, edges: pd.DataFrame, top_k: int = NETWORK_DIRECT_SINK_TOP_K
) -> pd.DataFrame:
    """每个源到其最近 top_k 个汇的专线候选弧，但只保留不与已有边相交的那些。

    专线是真实存在的接入方式（`direct_fallback_capex_multiplier` 2.8 倍就是为它准备的
    溢价），去掉它会把"必须全程走共享干线"当成硬约束，2050 年水泥捕集因此少了约 70 Mt。
    但原实现把这些弧放在去交叉之后追加，于是它们以直线穿过整张图——正是图上那些"很奇怪
    的线"。这里改为：逐条测试，与既有候选边相交的直接丢弃，被接受的立刻计入测试集合，
    所以专线之间也不会互相交叉。源包括煤电与工业点源。
    """
    sources = nodes.loc[
        nodes["node_type"].isin(["plant", "industry_hub"]), ["node_id", "lon", "lat"]
    ].dropna().reset_index(drop=True)
    sinks = nodes.loc[nodes["node_type"] == "storage_hub", ["node_id", "lon", "lat"]].dropna().reset_index(drop=True)
    if sources.empty or sinks.empty:
        return pd.DataFrame(columns=list(edges.columns))

    coords = {str(r.node_id): (float(r.lon), float(r.lat)) for r in nodes.itertuples(index=False)}
    kept_geoms = [g for g in (_build_edge_geometry(row, coords) for _, row in edges.iterrows())
                  if g is not None and not g.is_empty]
    existing = {
        frozenset((str(r.from_node_id), str(r.to_node_id)))
        for r in edges.itertuples(index=False)
    }
    sink_lon = np.radians(sinks["lon"].to_numpy(dtype=np.float64))
    sink_lat = np.radians(sinks["lat"].to_numpy(dtype=np.float64))
    sink_ids = sinks["node_id"].astype(str).to_numpy()

    next_index = 1
    rows: list[dict[str, object]] = []
    order = []
    for row in sources.itertuples(index=False):
        lon0, lat0 = np.radians(float(row.lon)), np.radians(float(row.lat))
        hav = (np.sin((sink_lat - lat0) / 2.0) ** 2
               + np.cos(lat0) * np.cos(sink_lat) * np.sin((sink_lon - lon0) / 2.0) ** 2)
        dist = 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
        for position in np.argsort(dist)[:top_k]:
            order.append((float(dist[position]), str(row.node_id), str(sink_ids[position])))
    order.sort()                                    # 短的先试，长的更可能被挡掉

    tree = STRtree(kept_geoms)
    pending: list[LineString] = []
    for direct_km, from_id, to_id in order:
        if frozenset((from_id, to_id)) in existing:
            continue
        line = LineString([coords[from_id], coords[to_id]])
        if any(line.crosses(kept_geoms[int(k)]) for k in tree.query(line)):
            continue
        if any(line.crosses(other) for other in pending):
            continue
        pending.append(line)
        existing.add(frozenset((from_id, to_id)))
        rows.append({
            "edge_id": f"edge_direct_{next_index:05d}",
            "feature_index": -2,
            "part_index": 1,
            "from_node_id": from_id,
            "to_node_id": to_id,
            "length_km": round(direct_km * NETWORK_DETOUR_FACTOR, 3),
            "direct_length_km": round(direct_km, 3),
            "tortuosity": NETWORK_DETOUR_FACTOR,
            "geometry_wkt": line.wkt,
            "corridor_type": "direct",
            "existing_corridor_flag": 0,
            "edge_class": NETWORK_EDGE_CLASS_DIRECT,
            "source": "direct_sink_rule",
            "year_basis": STATIC_LAYER_YEAR_BASIS,
        })
        next_index += 1
    if not rows:
        return pd.DataFrame(columns=list(edges.columns))
    logger.info("kept %d of %d direct source-sink arcs (crossing-free)", len(rows), len(order))
    return pd.DataFrame(rows)

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
    # 西藏不参与减排：先把穿过西藏的三角剖分候选边去掉，再去交叉。
    edges = _drop_edges_over_excluded_region(paths, nodes, edges)
    # Prefer planarity, but never at the cost of disconnecting a terminal
    edges = _remove_crossing_edges(nodes, edges)
    # 专线（源 -> 最近的 k 个汇）在去交叉之后补回，但逐条做相交检验：穿过已有管网的
    # 一律丢弃。既保住"可以为一个源单建一条专线"这个真实选项（2.8 倍造价溢价），
    # 又不会再出现那些横穿全图的直线。
    # 先把去交叉切出来的孤片并回一张网，再补专线（专线要对着合并后的图做相交检验）
    edges = _merge_components(paths, nodes, edges)
    direct_edges = build_direct_sink_edges(nodes, edges)
    if not direct_edges.empty:
        edges = pd.concat([edges, direct_edges], ignore_index=True, sort=False)
        edges = _drop_edges_over_excluded_region(paths, nodes, edges)
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
