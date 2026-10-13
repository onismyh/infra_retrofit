"""运行期网络的断言：管网输入与代码不配套时直接报错，而不是静默求解。

v7 输入（`a303f05` 的仓库根 `inputs/`，35 汇 / 923 边）不含工业节点。2026-10-13 前运行期按"接最近节点"给缺节点的源与汇
补支线，挂上去的工业点源有一批落在无汇的连通片里（实测 386 个中 33 个）；现在缺节点直接报错。这里用玩具网复现：
N1(煤电)—N2(汇) 连通，N3(工业)、N4 是孤立的节点。
"""
from __future__ import annotations

import pandas as pd
import pytest

from coal_retrofit.optimization.network import build_runtime_network
from coal_retrofit.paths import ProjectPaths


def _write_network(root) -> ProjectPaths:
    inputs = root / "inputs"
    inputs.mkdir()
    pd.DataFrame(
        {
            "node_id": ["N1", "N2", "N3", "N4"],
            "lon": [112.0, 112.5, 120.0, 112.1],
            "lat": [37.0, 37.0, 45.0, 37.0],
            "node_type": ["plant", "storage_hub", "industry_hub", "industry_hub"],
            "plant_id": ["P1", pd.NA, pd.NA, pd.NA],
            "storage_hub_id": [pd.NA, "S1", pd.NA, pd.NA],
            "industry_hub_id": [pd.NA, pd.NA, "H_FAR", "H_NEAR"],
        }
    ).to_csv(inputs / "pipeline_nodes.csv", index=False)
    pd.DataFrame(
        {
            "edge_id": ["E1", "E2"],
            "from_node_id": ["N1", "N4"],
            "to_node_id": ["N2", "N1"],
            "length_km": [50.0, 10.0],
            "offshore_length_km": [0.0, 0.0],
            "existing_corridor_flag": [0, 0],
            "edge_class": ["triangulation_candidate", "triangulation_candidate"],
            "corridor_type": ["triangulation", "triangulation"],
            "source": ["toy", "toy"],
            "year_basis": ["toy", "toy"],
        }
    ).to_csv(inputs / "pipeline_candidate_edges.csv", index=False)
    return ProjectPaths(root)


def _build(paths: ProjectPaths, hub_id: str, storage_id: str = "S1"):
    plants = pd.DataFrame({"plant_id": ["P1"], "centroid_longitude": [112.0], "centroid_latitude": [37.0]})
    storages = pd.DataFrame({"storage_hub_id": [storage_id], "longitude": [112.5], "latitude": [37.0]})
    hubs = pd.DataFrame({"hub_id": [hub_id], "longitude": [112.1], "latitude": [37.0]})
    return build_runtime_network(paths, plants, storages, industry_hubs=hubs)


def test_hub_connected_to_a_sink_is_accepted(tmp_path) -> None:
    network = _build(_write_network(tmp_path), "H_NEAR")
    assert network.industry_node_ids == {"H_FAR": "N3", "H_NEAR": "N4"}
    assert network.plant_node_ids == {"P1": "N1"} and network.storage_node_ids == {"S1": "N2"}


def test_hub_in_a_component_without_sink_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="1 个参与求解的源到不了任何封存汇.*industry H_FAR"):
        _build(_write_network(tmp_path), "H_FAR")


@pytest.mark.parametrize(("hub_id", "storage_id", "label"), [("H_NEW", "S1", "industry H_NEW"), ("H_NEAR", "S9", "storage S9")])
def test_source_or_sink_missing_from_the_network_is_rejected(tmp_path, hub_id, storage_id, label) -> None:
    """2026-10-13 前运行期给缺节点的源与汇补接最近节点的支线；现在要求它们都已在管网输入里。"""
    with pytest.raises(ValueError, match=f"1 个参与求解的源或汇不在 .*pipeline_nodes.csv 里（{label}）"):
        _build(_write_network(tmp_path), hub_id, storage_id)


def test_edge_table_without_offshore_lengths_is_rejected(tmp_path) -> None:
    """2026-10-10 起海上倍率按每条边的海上段计：边表没有 `offshore_length_km` 就报错，提示补算，而不是按陆上静默求解。"""
    paths = _write_network(tmp_path)
    edges_path = paths.inputs_dir / "pipeline_candidate_edges.csv"
    pd.read_csv(edges_path).drop(columns="offshore_length_km").to_csv(edges_path, index=False)
    with pytest.raises(ValueError, match="offshore_length_km.*--offshore-only"):
        _build(paths, "H_NEAR")
