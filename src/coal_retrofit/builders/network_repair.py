"""候选网络的清理与连通性修复：剔除穿过西藏的候选边、按优先级去交叉、把缺汇的源缝回汇、
把残留的连通片并成一张网，以及建网末尾的"每个源都能到汇"检查。由 `network.build_network_tables` 调用。
"""
from __future__ import annotations

import logging

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString
from shapely.strtree import STRtree

from ..constants import (
    NETWORK_DETOUR_FACTOR,
    NETWORK_EDGE_CLASS_DIRECT,
    NETWORK_EDGE_CLASS_TRIANGULATION,
    STATIC_LAYER_YEAR_BASIS,
)
from ..paths import ProjectPaths
from ..spatial import geodesic_length_km

logger = logging.getLogger(__name__)


_EDGE_CLASS_PRIORITY = {
    "existing_main_corridor": 0,
    "hub_to_corridor_branch": 1,
    "corridor_to_storage_branch": 1,
    "triangulation_candidate": 2,
}


def _build_edge_geometry(row: pd.Series, node_coords: dict[str, tuple[float, float]]) -> LineString | None:
    """为一条边构建 Shapely LineString；有 WKT 时直接用 WKT。"""
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
    """经 *edges* 无法到达任何汇的电厂节点与汇节点。"""
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
    """删去与高优先级边相交的低优先级边，然后修复连通性。

    优先级：既有走廊 > 支线 > 三角剖分候选边。结果在平面性不需付出代价的地方都是平面的，
    并且处处连通——见 `_repair_connectivity`。
    """
    node_coords = {
        str(row.node_id): (float(row.lon), float(row.lat))
        for row in nodes.itertuples(index=False)
    }

    # 构建几何并按优先级排序
    records: list[tuple[int, int, LineString]] = []  # (priority, df_index, geom)
    for idx, row in edges.iterrows():
        geom = _build_edge_geometry(row, node_coords)
        if geom is not None and not geom.is_empty:
            priority = _EDGE_CLASS_PRIORITY.get(row.get("edge_class", ""), 2)
            records.append((priority, idx, geom))

    # 排序：优先级数字最小的在前（这些要保留）；同一优先级内，最短的在前。
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
