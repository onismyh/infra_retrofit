"""源—汇之间的新建候选边。`build_triangulation_candidate_edges`：煤电与封存端点投到等积投影上做 Delaunay 三角剖分（退化时
回退 k=3 近邻），按最大边长与相对既有路径的伸长率筛选；`build_direct_sink_edges`：缝合并片之后再补每个源（煤电 + 工业点源）
到最近 k 个汇的专线，现有路线不比直线绕远 2.8 倍以上的不加。两类边都是直线，长度为大地线 × 绕行系数
（`network_edges.make_edge_record`），可以与已有边相交。"""
from __future__ import annotations

import logging
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import LineString

from ..constants import (
    NETWORK_DETOUR_FACTOR,
    NETWORK_DIRECT_SINK_MAX_STRETCH,
    NETWORK_DIRECT_SINK_TOP_K,
    NETWORK_EDGE_CLASS_DIRECT,
    NETWORK_EDGE_CLASS_TRIANGULATION,
    NETWORK_TRIANGULATION_CRS,
    NETWORK_TRIANGULATION_MAX_EDGE_KM,
    NETWORK_TRIANGULATION_MAX_STRETCH,
    TARGET_GEO_CRS,
)
from ..frames import iter_rows
from ..spatial import geodesic_length_km, haversine_km
from .network_edges import make_edge_record, next_edge_index, unordered_edge_key
from .network_repair import node_coordinates

logger = logging.getLogger(__name__)


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
            other = int(neighbor_idx)
            candidate_pairs.add((min(idx, other), max(idx, other)))
    return candidate_pairs


def _triangulation_candidate_pairs(points_xy: np.ndarray) -> set[tuple[int, int]]:
    point_count = len(points_xy)
    if point_count < 2:
        return set()
    if point_count == 2:
        return {(0, 1)}
    try:
        simplices = Delaunay(points_xy, qhull_options="QJ").simplices
    except (QhullError, ValueError) as exc:
        logger.warning("Delaunay 三角化失败（%s），候选边改用 k=3 近邻", exc)
        return _knn_candidate_pairs(points_xy, k=3)

    candidate_pairs: set[tuple[int, int]] = set()
    for simplex in simplices:
        for idx_a, idx_b in combinations(sorted(int(value) for value in simplex), 2):
            candidate_pairs.add((idx_a, idx_b))
    return candidate_pairs


def _build_length_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in iter_rows(nodes):
        graph.add_node(str(row.node_id))
    for row in iter_rows(edges):
        u, v, length = str(row.from_node_id), str(row.to_node_id), float(row.length_km)
        if graph.has_edge(u, v):    # 同一对节点的平行边取最短的那条，否则后读的覆盖前面的
            length = min(length, graph[u][v]["length_km"])
        graph.add_edge(u, v, length_km=length)
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
    existing_edge_keys = {unordered_edge_key(row.from_node_id, row.to_node_id) for row in iter_rows(edges)}

    edge_rows: list[dict[str, object]] = []
    edge_index = next_edge_index(edges)
    feature_index = 1

    for idx_a, idx_b in sorted(candidate_pairs):
        node_a = terminals.iloc[idx_a]
        node_b = terminals.iloc[idx_b]
        from_node_id = str(node_a["node_id"])
        to_node_id = str(node_b["node_id"])
        edge_key = unordered_edge_key(from_node_id, to_node_id)
        if edge_key in existing_edge_keys:
            continue

        coords = [(float(node_a["lon"]), float(node_a["lat"])), (float(node_b["lon"]), float(node_b["lat"]))]
        direct_length_km = geodesic_length_km(coords)
        if direct_length_km > NETWORK_TRIANGULATION_MAX_EDGE_KM:
            continue

        # 新建管道不会沿大圆走。本表中既有走廊的边已带有实际路由的折线长度，所以把三角
        # 剖分边留在原始大地线长度上，曾使新建管道从构造上就显得比复用走廊便宜。
        routed_length_km = direct_length_km * NETWORK_DETOUR_FACTOR

        try:
            existing_path_km = float(nx.shortest_path_length(graph, from_node_id, to_node_id, weight="length_km"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            existing_path_km = float("inf")

        if np.isfinite(existing_path_km) and existing_path_km <= routed_length_km * NETWORK_TRIANGULATION_MAX_STRETCH:
            continue

        edge_rows.append(
            make_edge_record(
                f"edge_{edge_index:05d}", feature_index, from_node_id, to_node_id, direct_length_km,
                geometry_wkt=LineString(coords).wkt, corridor_type="triangulation",
                edge_class=NETWORK_EDGE_CLASS_TRIANGULATION, source="source_sink_delaunay_rule",
            )
        )
        existing_edge_keys.add(edge_key)
        edge_index += 1
        feature_index += 1

    if not edge_rows:
        return pd.DataFrame(columns=list(edges.columns))
    return pd.DataFrame(edge_rows)


def build_direct_sink_edges(
    nodes: pd.DataFrame, edges: pd.DataFrame, top_k: int = NETWORK_DIRECT_SINK_TOP_K
) -> pd.DataFrame:
    """每个源到其最近 top_k 个汇的专线候选弧，只在现有路线绕得远时才加。

    专线是真实存在的接入方式（`direct_fallback_capex_multiplier` 2.8 倍就是为它准备的溢价），去掉它会把"必须全程走共享
    干线"当成硬约束，2050 年水泥捕集曾因此少了约 70 Mt。源到该汇在 *edges* 上的最短路不长于直线管长的
    `NETWORK_DIRECT_SINK_MAX_STRETCH` 倍时，专线按 2.8 倍单价算一般不比沿网络走便宜，不加；源到不了该汇时照加。专线可以与已有边
    相交（2026-10-16 起；此前与已有边相交的一律丢弃，3 680 条候选只留 135 条，源到近处的汇被一条走廊挡住就只能绕远）。
    源包括煤电与工业点源；最短路只走 *edges*，不走本函数新加的专线。
    """
    sources = nodes.loc[
        nodes["node_type"].isin(["plant", "industry_hub"]), ["node_id", "lon", "lat"]
    ].dropna().reset_index(drop=True)
    sinks = nodes.loc[nodes["node_type"] == "storage_hub", ["node_id", "lon", "lat"]].dropna().reset_index(drop=True)
    if sources.empty or sinks.empty:
        return pd.DataFrame(columns=list(edges.columns))

    coords = node_coordinates(nodes)
    graph = _build_length_graph(nodes, edges)
    existing = {unordered_edge_key(r.from_node_id, r.to_node_id) for r in iter_rows(edges)}
    sink_lon = sinks["lon"].to_numpy(dtype=np.float64)
    sink_lat = sinks["lat"].to_numpy(dtype=np.float64)
    sink_ids = sinks["node_id"].astype(str).to_numpy()
    stretch = NETWORK_DETOUR_FACTOR * NETWORK_DIRECT_SINK_MAX_STRETCH

    order = []
    farthest: dict[str, float] = {}
    for row in iter_rows(sources):
        dist = haversine_km(float(row.lon), float(row.lat), sink_lon, sink_lat)
        nearest = np.argsort(dist)[:top_k]
        farthest[str(row.node_id)] = float(dist[nearest].max())
        for position in nearest:
            order.append((float(dist[position]), str(row.node_id), str(sink_ids[position])))
    order.sort()

    reach: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    for direct_km, from_id, to_id in order:
        edge_key = unordered_edge_key(from_id, to_id)
        if edge_key in existing:
            continue
        if from_id not in reach:    # 只需知道最短路是否短于门槛，按该源最远的那个汇设截断
            reach[from_id] = nx.single_source_dijkstra_path_length(
                graph, from_id, cutoff=farthest[from_id] * stretch, weight="length_km")
        if reach[from_id].get(to_id, float("inf")) <= direct_km * stretch:
            continue
        existing.add(edge_key)
        rows.append(
            make_edge_record(
                f"edge_direct_{len(rows) + 1:05d}", -2, from_id, to_id, direct_km,
                geometry_wkt=LineString([coords[from_id], coords[to_id]]).wkt, corridor_type="direct",
                edge_class=NETWORK_EDGE_CLASS_DIRECT, source="direct_sink_rule",
            )
        )
    if not rows:
        return pd.DataFrame(columns=list(edges.columns))
    logger.info("kept %d of %d direct source-sink arcs (existing route longer than %.1f x straight)",
                len(rows), len(order), NETWORK_DIRECT_SINK_MAX_STRETCH)
    return pd.DataFrame(rows)
