"""候选管网构建：Delaunay 三角化失败时回退到 k=3 近邻并记一条 warning；其他异常照常抛出。
`network_edges`：边记录工厂的键序与长度、无向边键、下一个边号；`network_corridors`：走廊节点去重与度数。"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import QhullError
from shapely.geometry import LineString

from coal_retrofit.builders import network_triangulation as tri
from coal_retrofit.builders.network_corridors import build_corridor_tables
from coal_retrofit.builders.network_edges import EDGE_COLUMNS, make_edge_record, next_edge_index, unordered_edge_key
from coal_retrofit.constants import NETWORK_DETOUR_FACTOR, NETWORK_EDGE_CLASS_MAIN, STATIC_LAYER_YEAR_BASIS
from coal_retrofit.paths import ProjectPaths


def test_triangulation_falls_back_to_knn_on_qhull_error(monkeypatch, caplog) -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])  # 共线点，Qhull 会退化

    def _raise_qhull(*args, **kwargs):
        raise QhullError("QH6154 Qhull precision error")

    monkeypatch.setattr(tri, "Delaunay", _raise_qhull)
    with caplog.at_level(logging.WARNING, logger=tri.logger.name):
        pairs = tri._triangulation_candidate_pairs(points)
    assert pairs == tri._knn_candidate_pairs(points, k=3)
    assert any("Delaunay 三角化失败" in record.message for record in caplog.records)


def test_triangulation_other_errors_propagate(monkeypatch) -> None:
    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("not a qhull failure")

    monkeypatch.setattr(tri, "Delaunay", _raise_runtime)
    with pytest.raises(RuntimeError):
        tri._triangulation_candidate_pairs(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))


def test_make_edge_record_straight_edge() -> None:
    record = make_edge_record(
        "edge_00007", 3, "node_a", "node_b", 1.0004999,
        geometry_wkt="LINESTRING (0 0, 1 1)", corridor_type="triangulation", edge_class="triangulation_candidate",
        source="source_sink_delaunay_rule",
    )
    assert list(record) == list(EDGE_COLUMNS)
    assert record["direct_length_km"] == 1.0
    assert record["length_km"] == 1.137    # 先乘绕行系数再舍入；先舍成 1.000 再乘是 1.136
    assert record["tortuosity"] == NETWORK_DETOUR_FACTOR
    assert (record["part_index"], record["existing_corridor_flag"], record["year_basis"]) == (1, 0, STATIC_LAYER_YEAR_BASIS)
    assert record["geometry_wkt"] == "LINESTRING (0 0, 1 1)"


def test_make_edge_record_corridor_and_branch() -> None:
    corridor = make_edge_record(
        "edge_00001", 1, "a", "b", 250.5, part_index=2, tortuosity=1.0, existing_corridor_flag=1,
        geometry_wkt="LINESTRING (0 0, 2 2)", corridor_type="gas", edge_class="existing_main_corridor", source="gas.shp",
    )
    assert corridor["length_km"] == corridor["direct_length_km"] == 250.5
    assert (corridor["part_index"], corridor["existing_corridor_flag"]) == (2, 1)
    branch = make_edge_record(
        "edge_00002", 0, "a", "c", 3.0, geometry_wkt=None, corridor_type="main_corridor",
        edge_class="hub_to_corridor_branch", source="plants.csv", year_basis="2025",
    )
    assert branch["geometry_wkt"] is pd.NA
    assert branch["year_basis"] == "2025"


def test_unordered_edge_key_is_symmetric() -> None:
    assert unordered_edge_key("node_00002", "node_00001") == ("node_00001", "node_00002")
    assert unordered_edge_key("node_00001", "node_00002") == ("node_00001", "node_00002")
    assert unordered_edge_key(2, 10) == ("10", "2")    # 按字串排序


def test_next_edge_index_uses_max_suffix() -> None:
    edges = pd.DataFrame({"edge_id": ["edge_00001", "edge_00410", "edge_direct_00003"]})
    assert next_edge_index(edges) == 411
    assert next_edge_index(pd.DataFrame(columns=["edge_id"])) == 1
    assert next_edge_index(pd.DataFrame({"from_node_id": ["a"]})) == 1
    assert next_edge_index(pd.DataFrame({"edge_id": ["edge_a", "edge_b"]})) == 3    # 没有数字后缀时取行数 + 1


def test_build_corridor_tables_dedupes_endpoints(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    gas = gpd.GeoDataFrame(
        {"corridor_type": ["gas", "gas"]},
        geometry=[LineString([(100, 30), (102, 30)]), LineString([(100, 31), (101, 31), (100, 31)])],
        crs="EPSG:4326",
    )
    oil = gpd.GeoDataFrame({"corridor_type": ["oil"]}, geometry=[LineString([(102, 30), (103, 30)])], crs="EPSG:4326")
    nodes, edges = build_corridor_tables(
        paths, [(gas, tmp_path / "data" / "gas.shp"), (oil, tmp_path / "data" / "oil.shp")]
    )
    assert nodes["node_id"].tolist() == ["node_00001", "node_00002", "node_00003", "node_00004"]
    assert list(zip(nodes["lon"], nodes["lat"])) == [(100, 30), (100, 31), (102, 30), (103, 30)]
    assert nodes["degree"].tolist() == [1, 2, 2, 1]
    assert nodes["node_type"].tolist() == ["corridor_endpoint", "corridor_junction", "corridor_junction", "corridor_endpoint"]
    assert nodes.loc[2, "source"] == "data/gas.shp;data/oil.shp"
    assert list(edges.columns) == list(EDGE_COLUMNS)
    assert edges["edge_id"].tolist() == ["edge_00001", "edge_00002", "edge_00003"]
    assert edges["corridor_type"].tolist() == ["gas", "gas", "oil"]
    assert edges.loc[1, "from_node_id"] == edges.loc[1, "to_node_id"] == "node_00002"    # 闭合环两端同一节点
    assert (edges["tortuosity"] == 1.0).all() and (edges["existing_corridor_flag"] == 1).all()
    assert (edges["edge_class"] == NETWORK_EDGE_CLASS_MAIN).all()
    assert (edges["length_km"] == edges["direct_length_km"]).all()
    assert edges.loc[0, "length_km"] > 190.0    # 30°N 上 2° 经度约 193 km
