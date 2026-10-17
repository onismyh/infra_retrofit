"""候选网络的清理与连通性修复：剔除穿过西藏的候选边、把缺汇的源缝回汇、把残留的连通片并成一张网，以及建网末尾的
"每个源都能到汇"检查。由 `network.build_network_tables` 调用。相交的边都保留，缝合与并片的连线也可以与已有边相交
（2026-10-16 起；此前按"既有走廊 > 接入支线 > 三角剖分"的优先级删相交的边，连线也尽量不交叉）。
"""
from __future__ import annotations

import logging

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from ..constants import NETWORK_EDGE_CLASS_DIRECT, NETWORK_EDGE_CLASS_TRIANGULATION
from ..frames import iter_rows
from ..paths import ProjectPaths
from ..spatial import geodesic_length_km, haversine_km
from .network_edges import make_edge_record

logger = logging.getLogger(__name__)


def build_edge_geometry(row: pd.Series, node_coords: dict[str, tuple[float, float]]) -> LineString | None:
    """为一条边构建 Shapely LineString：有 WKT 时直接用 WKT，否则连两端节点的直线；端点不全在 *node_coords* 里返回 None。"""
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


def node_coordinates(nodes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """节点 id 到 (经度, 纬度)。"""
    return {str(r.node_id): (float(r.lon), float(r.lat)) for r in iter_rows(nodes)}


def _terminal_ids(nodes: pd.DataFrame) -> tuple[set[str], set[str]]:
    # 工业点源和煤电一样是"必须能到汇"的源端：它们现在进了备选网络，
    # 连通性修复必须把它们一起照顾到，否则又回到求解时拉直连线的老路。
    sources = set(nodes.loc[nodes["node_type"].isin(["plant", "industry_hub"]), "node_id"].astype(str))
    sinks = set(nodes.loc[nodes["node_type"] == "storage_hub", "node_id"].astype(str))
    return sources, sinks


def _unreached_terminals(nodes: pd.DataFrame, edges: pd.DataFrame) -> set[str]:
    """经 *edges* 无法到达任何汇的电厂节点与汇节点。"""
    plants, sinks = _terminal_ids(nodes)
    graph = nx.Graph()
    graph.add_nodes_from(plants | sinks)
    for row in iter_rows(edges):
        graph.add_edge(str(row.from_node_id), str(row.to_node_id))
    component = {node: idx for idx, part in enumerate(nx.connected_components(graph)) for node in part}
    sink_components = {component[s] for s in sinks if s in component}
    return {p for p in plants if component.get(p) not in sink_components}


MIN_STITCH_LENGTH_KM = 1.0


def _repair_connectivity(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """把到不了任何汇的源（煤电 + 工业点源）缝回含汇的连通片。

    每次取一个缺汇片区，在它与含汇片区之间连一条最短的新建候选边（source 记为 connectivity_stitch_rule）；
    片区数严格递减，因此必然终止。连线可以与已有边相交（2026-10-16 起；此前先按交叉删边、再挑不与保留边相交的
    最短节点对来缝，工业点源进网后有 124 个片区要这样缝回去）。
    """
    coords = node_coordinates(nodes)
    sources, sinks = _terminal_ids(nodes)

    def _components(frame: pd.DataFrame) -> dict[str, int]:
        graph = nx.Graph()
        graph.add_nodes_from(coords)
        for row in iter_rows(frame):
            graph.add_edge(str(row.from_node_id), str(row.to_node_id))
        return {n: i for i, part in enumerate(nx.connected_components(graph)) for n in part}

    stitched = 0
    while True:
        component = _components(edges)
        with_sink = {component[s] for s in sinks if s in component}
        needy = sorted({component[s] for s in sources if s in component and component[s] not in with_sink})
        if not needy:
            break
        # 排序：连通片按集合迭代，顺序随进程的字串 hash 变；等距时取 id 小的，建表才可复现
        group_ids = sorted(n for n, c in component.items() if c == needy[0])
        target_ids = sorted(n for n, c in component.items() if c in with_sink)
        if not group_ids or not target_ids:
            break
        target_lon = np.array([coords[n][0] for n in target_ids], dtype=np.float64)
        target_lat = np.array([coords[n][1] for n in target_ids], dtype=np.float64)
        pairs: list[tuple[float, str, str]] = []
        for node_a in group_ids:
            distances = haversine_km(coords[node_a][0], coords[node_a][1], target_lon, target_lat)
            position = int(np.argmin(distances))
            pairs.append((float(distances[position]), node_a, target_ids[position]))
        direct_km, node_a, node_b = min(pairs)
        direct_km = max(direct_km, MIN_STITCH_LENGTH_KM)
        stitched += 1
        record: dict[str, object] = {column: pd.NA for column in edges.columns}
        record.update(
            make_edge_record(
                f"edge_stitch_{stitched:05d}", -1, node_a, node_b, direct_km,
                geometry_wkt=LineString([coords[node_a], coords[node_b]]).wkt, corridor_type="triangulation",
                edge_class=NETWORK_EDGE_CLASS_TRIANGULATION, source="connectivity_stitch_rule",
            )
        )
        edges = pd.concat([edges, pd.DataFrame([record])], ignore_index=True)
    if stitched:
        logger.info("stitched %d candidate edges", stitched)
    return edges.reset_index(drop=True)


EXCLUDED_REGION_NAMES = ("西藏", "西藏自治区")


def _excluded_region(paths: ProjectPaths) -> BaseGeometry | None:
    """不参与减排省份的几何；没有图层或图层里没有这个省时返回 None。"""
    path = paths.data_dir / "ChinaMap" / "provinces.shp"
    if not path.exists():
        return None
    hit = gpd.read_file(path)
    hit = hit[hit["NAME"].astype(str).isin(EXCLUDED_REGION_NAMES)]
    return None if hit.empty else hit.to_crs("EPSG:4326").geometry.union_all()


def _drop_edges_over_excluded_region(paths: ProjectPaths, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """丢掉穿过不参与减排省份（西藏）的三角剖分候选边与专线。

    这些边不是任何源或汇的接入线——西藏没有封存汇，工业点源已在装载时剔除——
    它们只是 Delaunay 在空旷西部连出来的长边，留着既不会被选中，又让图上看着像有管网。
    既有管廊与接入支线不动：那是真实存在的管道。
    """
    region = _excluded_region(paths)
    if region is None:
        return edges
    coords = node_coordinates(nodes)
    drop = []
    for row in iter_rows(edges):
        if str(row.edge_class) not in (NETWORK_EDGE_CLASS_TRIANGULATION, NETWORK_EDGE_CLASS_DIRECT):
            continue
        a, b = coords.get(str(row.from_node_id)), coords.get(str(row.to_node_id))
        if a is None or b is None:
            continue
        if LineString([a, b]).intersects(region):
            drop.append(str(row.edge_id))
    if drop:
        logger.info("dropped %d triangulation edges crossing %s", len(drop), EXCLUDED_REGION_NAMES[0])
        edges = edges[~edges["edge_id"].astype(str).isin(set(drop))].reset_index(drop=True)
    return edges


def _merge_components(paths: ProjectPaths, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """把残留的连通片并成一张网，每次补一条最短、不过西藏的连线（可以与已有边相交，2026-10-16 起；此前还要求不交叉，
    只在找不到时才退回最短那条）。

    `_repair_connectivity` 只保证"每个源能到某个汇"，这不够：2026-09-12 建网时按交叉删边把图切成 45 片，
    南海那个 200 Mtpa 的离岸汇 S002 落在 35 节点的小片里，全国只有 28 个源够得着它，于是
    IND_*_t95 的 2060 年有 26 个小汇被超注入 196 Mt/yr，而 S002 闲置 162 Mt/yr。
    备选网络不该替优化器决定"谁不许去哪个汇"；要不要真建这些管段仍由优化器决定。
    所有候选连线都过西藏时退回最短那条，它在建网末尾随穿过西藏的边一起删掉（西藏一侧的管廊碎片因此仍是孤片）。
    """
    coords = node_coordinates(nodes)
    region = _excluded_region(paths)
    ids = list(coords)
    lon = np.array([coords[n][0] for n in ids], dtype=np.float64)
    lat = np.array([coords[n][1] for n in ids], dtype=np.float64)
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    for row in iter_rows(edges):
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
                dist = haversine_km(lon[index], lat[index], lon[outside], lat[outside])
                pick = int(np.argmin(dist))
                candidates.append((float(dist[pick]), int(index), int(outside[pick])))
        candidates.sort()
        chosen = None
        for _, a_index, b_index in candidates[:600]:
            line = LineString([coords[ids[a_index]], coords[ids[b_index]]])
            if region is None or not line.intersects(region):
                chosen = (ids[a_index], ids[b_index], line)
                break
        if chosen is None:
            _, a_index, b_index = candidates[0]
            chosen = (ids[a_index], ids[b_index], LineString([coords[ids[a_index]], coords[ids[b_index]]]))
            forced += 1
        node_a, node_b, line = chosen
        direct_km = max(float(geodesic_length_km([coords[node_a], coords[node_b]])),
                        MIN_STITCH_LENGTH_KM)
        graph.add_edge(node_a, node_b)
        record: dict[str, object] = {column: pd.NA for column in edges.columns}
        record.update(
            make_edge_record(
                f"edge_merge_{len(added) + 1:05d}", -3, node_a, node_b, direct_km,
                geometry_wkt=line.wkt, corridor_type="triangulation",
                edge_class=NETWORK_EDGE_CLASS_TRIANGULATION, source="component_merge_rule",
            )
        )
        added.append(record)
    if added:
        logger.info("merged the network into one component with %d links (%d over %s)",
                    len(added), forced, EXCLUDED_REGION_NAMES[0])
        edges = pd.concat([edges, pd.DataFrame(added)], ignore_index=True, sort=False)
    return edges
