"""候选管网的相交边（2026-10-16 起）：缝合、并片与专线都可以与已有边相交，不再为避开交叉绕远；专线只在现有路线长于
直线管长 2.8 倍（`NETWORK_DIRECT_SINK_MAX_STRETCH`）时才加。测试里的走廊边 C1–C2 横在源与汇的直线上。"""
from __future__ import annotations

import pandas as pd
import pytest

from coal_retrofit.builders.network_edges import make_edge_record
from coal_retrofit.builders.network_repair import _merge_components, _repair_connectivity
from coal_retrofit.builders.network_triangulation import build_direct_sink_edges
from coal_retrofit.constants import (
    NETWORK_DETOUR_FACTOR,
    NETWORK_DIRECT_SINK_MAX_STRETCH,
    NETWORK_EDGE_CLASS_DIRECT,
    NETWORK_EDGE_CLASS_MAIN,
    NETWORK_EDGE_CLASS_TRIANGULATION,
)
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.spatial import haversine_km


def _nodes(*rows: tuple[str, float, float, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["node_id", "lon", "lat", "node_type"])


def _edge(edge_id: str, a: str, b: str, km: float, edge_class: str = NETWORK_EDGE_CLASS_TRIANGULATION,
          wkt: str | None = None) -> dict[str, object]:
    record = make_edge_record(edge_id, 1, a, b, km, geometry_wkt=wkt, corridor_type="t", edge_class=edge_class,
                              source="test")
    record["length_km"] = km    # 测试直接给图上的长度
    return record


CORRIDOR = _edge("edge_c", "C1", "C2", 111.0, NETWORK_EDGE_CLASS_MAIN, "LINESTRING (100.5 29.5, 100.5 30.5)")


def test_stitch_takes_the_shortest_link_even_across_a_corridor() -> None:
    """缺汇的源 P 缝到最近的汇 S，哪怕连线横穿走廊 C1–C2；此前会改连远处不交叉的 S2。已有边一条不删。"""
    nodes = _nodes(("P", 100.0, 30.0, "plant"), ("S", 101.0, 30.0, "storage_hub"), ("S2", 100.0, 32.0, "storage_hub"),
                   ("C1", 100.5, 29.5, "corridor_junction"), ("C2", 100.5, 30.5, "corridor_junction"))
    edges = pd.DataFrame([CORRIDOR])
    out = _repair_connectivity(nodes, edges)
    assert len(out) == 2 and out.loc[0, "edge_id"] == "edge_c"
    stitch = out.iloc[1]
    assert {stitch["from_node_id"], stitch["to_node_id"]} == {"P", "S"}
    assert stitch["source"] == "connectivity_stitch_rule"
    assert stitch["direct_length_km"] == pytest.approx(float(haversine_km(100.0, 30.0, 101.0, 30.0)), abs=1e-3)


def test_merge_links_the_nearest_pair_even_across_a_corridor(tmp_path) -> None:
    """两片 {P, S, C1, C2} 与 {Q, T}：并片取最近的一对 S–Q，连线横穿走廊 C1–C2 也照连（没有省界图层时不判西藏）。"""
    nodes = _nodes(("P", 100.0, 30.0, "plant"), ("S", 100.2, 30.0, "storage_hub"),
                   ("Q", 101.0, 30.0, "plant"), ("T", 101.2, 30.0, "storage_hub"),
                   ("C1", 100.6, 29.0, "corridor_junction"), ("C2", 100.6, 31.0, "corridor_junction"))
    corridor = _edge("edge_c", "C1", "C2", 222.0, NETWORK_EDGE_CLASS_MAIN, "LINESTRING (100.6 29.0, 100.6 31.0)")
    edges = pd.DataFrame([_edge("e1", "P", "S", 19.0), _edge("e2", "Q", "T", 19.0), _edge("e3", "S", "C1", 112.0),
                          corridor])
    out = _merge_components(ProjectPaths(tmp_path), nodes, edges)
    merged = out[out["source"] == "component_merge_rule"]
    assert len(out) == 5 and len(merged) == 1
    assert {merged.iloc[0]["from_node_id"], merged.iloc[0]["to_node_id"]} == {"S", "Q"}


def test_direct_arcs_cross_edges_and_skip_short_detours() -> None:
    """P 的三个汇：到 S1 没有路（直线横穿走廊）→ 加专线；到 S3 有 100 km 的路，短于直线管长的 2.8 倍 → 不加；
    到 S4 要绕 250 km，长于它自己的 2.8 倍（153 km）、但短于按最远汇 S1 设的最短路截断（306 km）→ 加，
    这样测的是逐汇门槛，而不是截断。"""
    nodes = _nodes(("P", 100.0, 30.0, "plant"), ("S1", 101.0, 30.0, "storage_hub"),
                   ("S3", 100.0, 30.5, "storage_hub"), ("S4", 99.5, 30.0, "storage_hub"),
                   ("M", 100.1, 30.25, "corridor_junction"), ("N", 99.8, 29.0, "corridor_junction"),
                   ("C1", 100.5, 29.5, "corridor_junction"), ("C2", 100.5, 30.5, "corridor_junction"))
    edges = pd.DataFrame([CORRIDOR, _edge("e1", "P", "M", 50.0), _edge("e2", "M", "S3", 50.0),
                          _edge("e3", "P", "N", 125.0), _edge("e4", "N", "S4", 125.0)])
    stretch = NETWORK_DETOUR_FACTOR * NETWORK_DIRECT_SINK_MAX_STRETCH
    xy = nodes.set_index("node_id")
    cutoff = float(haversine_km(100.0, 30.0, *xy.loc["S1", ["lon", "lat"]])) * stretch    # 最远的汇是 S1
    for sink, detour in (("S3", 100.0), ("S4", 250.0)):    # 两条路分别落在门槛两侧，都在截断以内
        limit = float(haversine_km(100.0, 30.0, *xy.loc[sink, ["lon", "lat"]])) * stretch
        assert (detour <= limit) == (sink == "S3") and detour < cutoff
    out = build_direct_sink_edges(nodes, edges)
    assert set(out["edge_class"]) == {NETWORK_EDGE_CLASS_DIRECT}
    assert sorted(out["to_node_id"]) == ["S1", "S4"] and set(out["from_node_id"]) == {"P"}
