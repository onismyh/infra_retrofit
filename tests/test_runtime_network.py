"""运行期网络的可达性断言：管网输入与代码不配套时直接报错，而不是静默求解。

v7 输入（`a303f05` 的仓库根 `inputs/`，35 汇 / 923 边）不含工业节点；运行期直连弧去掉之后，按"接最近节点"
挂上去的工业点源有一批落在无汇的连通片里（实测 386 个中 33 个）。这里用三节点玩具网
复现这一情形：N1(煤电)—N2(汇) 连通，N3 是孤立的走廊节点。
"""
from __future__ import annotations

import pandas as pd
import pytest

from coal_retrofit.optimization.network import build_runtime_network
from coal_retrofit.optimization.scenario import OptimizationScenario
from coal_retrofit.paths import ProjectPaths


def _write_network(root) -> ProjectPaths:
    inputs = root / "inputs"
    inputs.mkdir()
    pd.DataFrame(
        {
            "node_id": ["N1", "N2", "N3"],
            "lon": [112.0, 112.5, 120.0],
            "lat": [37.0, 37.0, 45.0],
            "node_type": ["plant", "storage_hub", "corridor_junction"],
            "plant_id": ["P1", pd.NA, pd.NA],
            "storage_hub_id": [pd.NA, "S1", pd.NA],
        }
    ).to_csv(inputs / "pipeline_nodes.csv", index=False)
    pd.DataFrame(
        {
            "edge_id": ["E1"],
            "from_node_id": ["N1"],
            "to_node_id": ["N2"],
            "length_km": [50.0],
            "offshore_length_km": [0.0],
            "existing_corridor_flag": [0],
            "edge_class": ["triangulation_candidate"],
            "corridor_type": ["triangulation"],
            "source": ["toy"],
            "year_basis": ["toy"],
        }
    ).to_csv(inputs / "pipeline_candidate_edges.csv", index=False)
    return ProjectPaths(root)


def _build(paths: ProjectPaths, hub_lon: float, hub_lat: float):
    plants = pd.DataFrame({"plant_id": ["P1"], "centroid_longitude": [112.0], "centroid_latitude": [37.0]})
    storages = pd.DataFrame({"storage_hub_id": ["S1"], "longitude": [112.5], "latitude": [37.0]})
    hubs = pd.DataFrame({"hub_id": ["H1"], "longitude": [hub_lon], "latitude": [hub_lat]})
    scenario = OptimizationScenario(experiment_id="T-NET", description="toy")
    return build_runtime_network(paths, plants, storages, scenario, industry_hubs=hubs)


def test_runtime_hub_next_to_a_connected_node_is_accepted(tmp_path) -> None:
    network = _build(_write_network(tmp_path), hub_lon=112.1, hub_lat=37.0)
    assert network.industry_node_ids == {"H1": "industry::H1"}


def test_runtime_hub_in_a_component_without_sink_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="1 个参与求解的源到不了任何封存汇.*industry H1"):
        _build(_write_network(tmp_path), hub_lon=120.1, hub_lat=45.0)


def test_edge_table_without_offshore_lengths_is_rejected(tmp_path) -> None:
    """2026-10-10 起海上倍率按每条边的海上段计：边表没有 `offshore_length_km` 就报错，提示补算，而不是按陆上静默求解。"""
    paths = _write_network(tmp_path)
    edges_path = paths.inputs_dir / "pipeline_candidate_edges.csv"
    pd.read_csv(edges_path).drop(columns="offshore_length_km").to_csv(edges_path, index=False)
    with pytest.raises(ValueError, match="offshore_length_km.*--offshore-only"):
        _build(paths, hub_lon=112.1, hub_lat=37.0)


def test_runtime_branch_to_an_offshore_sink_is_offshore_end_to_end(tmp_path) -> None:
    """运行期支线没有几何：接海上汇的整条记海上段，接电厂、工业的记 0。"""
    paths = _write_network(tmp_path)
    plants = pd.DataFrame({"plant_id": ["P1"], "centroid_longitude": [112.0], "centroid_latitude": [37.0]})
    storages = pd.DataFrame({
        "storage_hub_id": ["S1", "S2"], "longitude": [112.5, 112.6], "latitude": [37.0, 37.0], "offshore": [False, True],
    })
    hubs = pd.DataFrame({"hub_id": ["H1"], "longitude": [112.1], "latitude": [37.0]})
    scenario = OptimizationScenario(experiment_id="T-NET", description="toy")
    edges = build_runtime_network(paths, plants, storages, scenario, industry_hubs=hubs).edges.set_index("edge_id")
    sea = edges.loc["edge_runtime_storage_S2"]
    assert sea["offshore_length_km"] == pytest.approx(sea["length_km"]) and sea["length_km"] > 0
    assert edges.loc["edge_runtime_industry_H1", "offshore_length_km"] == 0.0
