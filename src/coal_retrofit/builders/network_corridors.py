"""既有油气干线走廊：读图层（`load_corridor_layer`），按要素的每个部件建一条走廊边、端点按坐标去重成走廊节点
（`build_corridor_tables`）。走廊边按路由折线的大地线长计价、`tortuosity` 为 1、`existing_corridor_flag` 为 1。"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import pandas as pd

from ..constants import NETWORK_COORD_DECIMALS, NETWORK_EDGE_CLASS_MAIN, STATIC_LAYER_YEAR_BASIS, TARGET_GEO_CRS
from ..paths import ProjectPaths
from ..spatial import geodesic_length_km, geometry_parts
from .network_edges import make_edge_record

Waypoint = tuple[float, float]


class _CorridorPart(NamedTuple):
    """一条走廊边的原料：要素号、部件号、两端（坐标取 `NETWORK_COORD_DECIMALS` 位）、折线长、WKT、走廊类型与来源。"""

    feature_index: int
    part_index: int
    wp_from: Waypoint
    wp_to: Waypoint
    length_km: float
    wkt: str
    corridor_type: str
    source: str


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


def _round_waypoint(coord: tuple[float, ...]) -> Waypoint:
    return (round(coord[0], NETWORK_COORD_DECIMALS), round(coord[1], NETWORK_COORD_DECIMALS))


def build_corridor_tables(paths: ProjectPaths, layers: list[tuple[gpd.GeoDataFrame, Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """走廊节点表与走廊边表。节点按坐标排序编号 `node_{i:05d}`，度数大于 1 的是 `corridor_junction`，否则 `corridor_endpoint`；
    边按收录顺序编号 `edge_{i:05d}`，再按 (走廊类型, 要素号, 部件号, 边号) 排序。"""
    degree: Counter[Waypoint] = Counter()
    sources: defaultdict[Waypoint, set[str]] = defaultdict(set)
    parts: list[_CorridorPart] = []

    for gdf, path in layers:
        source = paths.rel(path)
        corridor_type = str(gdf["corridor_type"].iloc[0])
        for feature_index, row in enumerate(gdf.itertuples(index=False), start=1):
            for part_index, part in enumerate(geometry_parts(row.geometry), start=1):
                coords = list(part.coords)
                if len(coords) < 2:
                    continue
                wp_from, wp_to = _round_waypoint(coords[0]), _round_waypoint(coords[-1])
                for wp in (wp_from, wp_to):
                    degree[wp] += 1
                    sources[wp].add(source)
                parts.append(
                    _CorridorPart(feature_index, part_index, wp_from, wp_to, geodesic_length_km(coords), part.wkt,
                                  corridor_type, source)
                )

    node_keys = sorted(degree)
    node_id_map = {key: f"node_{index:05d}" for index, key in enumerate(node_keys, start=1)}
    nodes = pd.DataFrame(
        {
            "node_id": node_id_map[key],
            "lon": key[0],
            "lat": key[1],
            "node_type": "corridor_junction" if degree[key] > 1 else "corridor_endpoint",
            "degree": degree[key],
            "source": ";".join(sorted(sources[key])),
            "year_basis": STATIC_LAYER_YEAR_BASIS,
        }
        for key in node_keys
    ).sort_values(["lon", "lat", "node_id"]).reset_index(drop=True)

    edges = pd.DataFrame(
        make_edge_record(
            f"edge_{index:05d}", part.feature_index, node_id_map[part.wp_from], node_id_map[part.wp_to], part.length_km,
            part_index=part.part_index, tortuosity=1.0, geometry_wkt=part.wkt, corridor_type=part.corridor_type,
            existing_corridor_flag=1, edge_class=NETWORK_EDGE_CLASS_MAIN, source=part.source,
        )
        for index, part in enumerate(parts, start=1)
    ).sort_values(["corridor_type", "feature_index", "part_index", "edge_id"]).reset_index(drop=True)

    return nodes, edges
