"""逐厂明细的 `min_distance_to_storage_km`：沿候选管网到最近汇的最短管长（`results_plant._network_km_to_nearest_sink`）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from coal_retrofit.optimization.network import RuntimeNetwork
from coal_retrofit.optimization.results_plant import _network_km_to_nearest_sink


def _network() -> RuntimeNetwork:
    """P1 经交汇点 J 到汇 S1（两条平行边 100 / 80 km，再 150 km），另有一条到 S2 的直连弧 300 km；P2 不连通。"""
    nodes = pd.DataFrame({"node_id": ["P1", "P2", "J", "S1", "S2"]})
    edges = pd.DataFrame({
        "edge_id": ["e1", "e2", "e3", "e4"],
        "from_node_id": ["P1", "J", "P1", "J"],
        "to_node_id": ["J", "S1", "S2", "P1"],
        "length_km": [100.0, 150.0, 300.0, 80.0],
    })
    return RuntimeNetwork(
        nodes=nodes, edges=edges, incidence=np.zeros((len(nodes), len(edges))),
        plant_node_ids={"p1": "P1", "p2": "P2"}, storage_node_ids={"s1": "S1", "s2": "S2"}, industry_node_ids={},
    )


def test_distance_follows_the_network_not_only_direct_arcs() -> None:
    """取经 J 的 80 + 150 = 230 km（平行边取短的），不是直连弧的 300 km；到不了汇的 hub 不在结果里（明细记 NaN）。
    2026-10-17 前只看直连弧，p1 得 300。"""
    km = _network_km_to_nearest_sink(_network())
    assert km == {"p1": 230.0}
