"""汇变体（`coal_retrofit.sink_inputs`，2026-10-16 起）：文件名、建表只写变体的文件、求解按情景的 `sink_variant` 读三张表，
汇表与管网不配套（缺文件、汇坐标对不上）时报错。不求解，不需要 Gurobi。"""
from __future__ import annotations

import pandas as pd
import pytest

from coal_retrofit.builders import network as network_builder
from coal_retrofit.builders import storage
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.scenario import OptimizationScenario
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.sink_inputs import (
    SINK_INPUT_STEMS,
    check_sink_variant,
    cluster_variant,
    require_sink_inputs,
    sink_input_path,
)
from toy_inputs import YEARS, _toy_assumptions, _write_toy_inputs

VARIANT = "300km_average"


def test_variant_file_names(tmp_path) -> None:
    assert cluster_variant(300.0, "complete") == "300km_complete" and cluster_variant(300, "average") == VARIANT
    assert sink_input_path(tmp_path, "storage_hubs") == tmp_path / "storage_hubs.csv"
    assert sink_input_path(tmp_path, "pipeline_candidate_edges", VARIANT) == (
        tmp_path / "pipeline_candidate_edges_300km_average.csv")
    assert check_sink_variant("") == ""
    with pytest.raises(ValueError, match="汇变体只能是 .*得到 '100km_complete'"):
        sink_input_path(tmp_path, "storage_hubs", "100km_complete")
    with pytest.raises(ValueError, match="不是随汇变体换文件的输入表"):
        sink_input_path(tmp_path, "industry_hubs", VARIANT)


def test_missing_variant_files_name_the_build_commands(tmp_path) -> None:
    for stem in SINK_INPUT_STEMS:
        (tmp_path / f"{stem}.csv").write_text("x\n", encoding="utf-8")
    require_sink_inputs(tmp_path)
    (tmp_path / "storage_hubs_300km_average.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=(
            "没有汇变体 300km_average 的 pipeline_nodes_300km_average.csv、pipeline_candidate_edges_300km_average.csv："
            "先运行 scripts/build_storage_inputs.py --cluster-km 300 --linkage average，"
            "再运行 scripts/build_network_inputs.py --sinks 300km_average")):
        require_sink_inputs(tmp_path, VARIANT)


def _scenario(sink_variant: str = "") -> OptimizationScenario:
    return OptimizationScenario(experiment_id="TEST-SINKS", description="toy", planning_years=YEARS,
                                sector_target_source="toy", sink_variant=sink_variant)


def _write_variant(paths: ProjectPaths, *, rebuild_network: bool = True) -> None:
    """toy 的汇 S1 挪到 (113.0, 37.0) 作为变体的汇表；*rebuild_network* 时管网节点随之挪（在新汇表上重建），否则照抄缺省管网。"""
    inputs = paths.inputs_dir
    pd.read_csv(inputs / "storage_hubs.csv").assign(longitude=113.0, storage_all_mt=500.0).to_csv(
        sink_input_path(inputs, "storage_hubs", VARIANT), index=False)
    nodes = pd.read_csv(inputs / "pipeline_nodes.csv")
    if rebuild_network:
        nodes.loc[nodes["storage_hub_id"] == "S1", "lon"] = 113.0
    nodes.to_csv(sink_input_path(inputs, "pipeline_nodes", VARIANT), index=False)
    pd.read_csv(inputs / "pipeline_candidate_edges.csv").to_csv(
        sink_input_path(inputs, "pipeline_candidate_edges", VARIANT), index=False)


def test_prepare_inputs_reads_the_variant_tables(tmp_path) -> None:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_variant(paths)
    default = prepare_inputs(paths, _scenario(), _toy_assumptions())
    variant = prepare_inputs(paths, _scenario(VARIANT), _toy_assumptions())
    assert default.storages["available_capacity_mt"].tolist() == [1000.0]
    assert variant.storages["available_capacity_mt"].tolist() == [500.0]
    sink_node = variant.network.storage_node_ids["S1"]
    assert variant.network.nodes.set_index("node_id").loc[sink_node, "lon"] == 113.0
    assert default.network.nodes.set_index("node_id").loc[sink_node, "lon"] == 112.5


def test_variant_without_its_network_is_rejected(tmp_path) -> None:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_variant(paths)
    sink_input_path(paths.inputs_dir, "pipeline_candidate_edges", VARIANT).unlink()
    with pytest.raises(FileNotFoundError, match="pipeline_candidate_edges_300km_average.csv"):
        prepare_inputs(paths, _scenario(VARIANT), _toy_assumptions())


def test_network_built_on_another_sink_table_is_rejected(tmp_path) -> None:
    """汇 id 按序号编、各变体共用：管网照抄了缺省汇表上建的，S1 找得到但坐标不对，也要报错。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_variant(paths, rebuild_network=False)
    with pytest.raises(ValueError, match=(
            r"1 个汇在 pipeline_nodes_300km_average.csv 里的坐标与汇表不同（S1）：管网不是在这张汇表上建的，"
            r"重新运行 scripts/build_network_inputs.py --sinks 300km_average")):
        prepare_inputs(paths, _scenario(VARIANT), _toy_assumptions())


def test_storage_builder_writes_the_variant_table_only(monkeypatch, tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.inputs_dir.mkdir(parents=True)
    calls: list[tuple[float | None, str]] = []

    def fake_build(paths, offshore_basin_dilation_px=6, cluster_km=None, cluster_linkage="complete") -> pd.DataFrame:
        calls.append((cluster_km, cluster_linkage))
        return pd.DataFrame({"storage_hub_id": ["S001"]})

    monkeypatch.setattr(storage, "build_storage_hub_dataframe", fake_build)
    _, path = storage.write_storage_hubs(paths, cluster_km=300.0, cluster_linkage="average")
    assert path == paths.inputs_dir / "storage_hubs_300km_average.csv" and path.exists()
    assert not (paths.inputs_dir / "storage_hubs.csv").exists()
    _, path = storage.write_storage_hubs(paths)
    assert path == paths.inputs_dir / "storage_hubs.csv"
    # 没登记的变体在读栅格之前就报错
    with pytest.raises(ValueError, match="得到 '100km_complete'"):
        storage.write_storage_hubs(paths, cluster_km=100.0)
    assert calls == [(300.0, "average"), (None, "complete")]


def test_network_builder_writes_the_variant_tables(monkeypatch, tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.inputs_dir.mkdir(parents=True)
    seen: list[str] = []

    def fake_tables(paths, use_corridors=True, sink_variant=""):
        seen.append(sink_variant)
        return pd.DataFrame({"node_id": ["N1"]}), pd.DataFrame({"edge_id": ["E1"]})

    monkeypatch.setattr(network_builder, "build_network_tables", fake_tables)
    network_builder.write_network_inputs(paths, use_corridors=False, sink_variant=VARIANT)
    assert seen == [VARIANT]
    assert sorted(p.name for p in paths.inputs_dir.iterdir()) == [
        "pipeline_candidate_edges_300km_average.csv", "pipeline_nodes_300km_average.csv"]
