"""源—汇之间的新建候选边。`build_triangulation_candidate_edges`：煤电与封存端点投到等积投影上做 Delaunay 三角剖分（退化时
回退 k=3 近邻），按最大边长与相对既有路径的伸长率筛选；`build_direct_sink_edges`：去交叉之后再补每个源（煤电 + 工业点源）
到最近 k 个汇、与已有边不相交的专线。两类边都是直线，长度为大地线 × 绕行系数（`network_edges.make_edge_record`）。"""
from __future__ import annotations

import logging
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import LineString
from shapely.strtree import STRtree

from ..constants import (
    NETWORK_DETOUR_FACTOR,
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
from .network_repair import edge_geometries, node_coordinates

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

    coords = node_coordinates(nodes)
    kept_geoms = edge_geometries(edges, coords)
    existing = {unordered_edge_key(r.from_node_id, r.to_node_id) for r in iter_rows(edges)}
    sink_lon = sinks["lon"].to_numpy(dtype=np.float64)
    sink_lat = sinks["lat"].to_numpy(dtype=np.float64)
    sink_ids = sinks["node_id"].astype(str).to_numpy()

    order = []
    for row in iter_rows(sources):
        dist = haversine_km(float(row.lon), float(row.lat), sink_lon, sink_lat)
        for position in np.argsort(dist)[:top_k]:
            order.append((float(dist[position]), str(row.node_id), str(sink_ids[position])))
    order.sort()                                    # 短的先试，长的更可能被挡掉

    tree = STRtree(kept_geoms)
    pending: list[LineString] = []
    rows: list[dict[str, object]] = []
    for direct_km, from_id, to_id in order:
        edge_key = unordered_edge_key(from_id, to_id)
        if edge_key in existing:
            continue
        line = LineString([coords[from_id], coords[to_id]])
        if any(line.crosses(kept_geoms[int(k)]) for k in tree.query(line)):
            continue
        if any(line.crosses(other) for other in pending):
            continue
        pending.append(line)
        existing.add(edge_key)
        rows.append(
            make_edge_record(
                f"edge_direct_{len(rows) + 1:05d}", -2, from_id, to_id, direct_km,
                geometry_wkt=line.wkt, corridor_type="direct",
                edge_class=NETWORK_EDGE_CLASS_DIRECT, source="direct_sink_rule",
            )
        )
    if not rows:
        return pd.DataFrame(columns=list(edges.columns))
    logger.info("kept %d of %d direct source-sink arcs (crossing-free)", len(rows), len(order))
    return pd.DataFrame(rows)
