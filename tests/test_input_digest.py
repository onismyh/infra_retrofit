"""溯源记录：输入摘要只摘本情景实际读取的文件、按实际读取的目录计算；mip_focus 读回模型上的实际值。"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from coal_retrofit.optimization.data_prep import _input_files, prepare_inputs
from coal_retrofit.optimization.scenario import OptimizationScenario
from coal_retrofit.optimization.solver_provenance import _input_digest, _run_provenance
from toy_inputs import YEARS, _toy_assumptions, _write_toy_inputs

WATER = {"water_mode": "grid_supply", "water_scenario_id": "toy|gcm|ssp126", "water_season": "dry"}


def _scenario(**overrides) -> OptimizationScenario:
    return OptimizationScenario(
        experiment_id="TEST-DIGEST", description="toy", planning_years=YEARS,
        sector_target_source="toy", **overrides,
    )


def _reads_during_prepare(paths, scenario: OptimizationScenario, monkeypatch) -> set[Path]:
    """跑一遍 `prepare_inputs`，返回 `pd.read_csv` 与 `gpd.read_file` 实际读到的文件。

    求解侧只经这两个函数读文件；以后若加了别的读法，这里要跟着记。
    """
    seen: set[Path] = set()
    real_read_csv = pd.read_csv

    def recording_read_csv(path, *args, **kwargs):
        seen.add(Path(path).resolve())
        return real_read_csv(path, *args, **kwargs)

    def fake_read_file(path, *args, **kwargs):
        # toy 没有真的流域多边形：记下路径，返回一个罩住 toy 全部点位的一级区。
        seen.add(Path(path).resolve())
        return gpd.GeoDataFrame(
            {"code": ["B1"], "name": ["toy"]}, geometry=[box(80.0, 20.0, 130.0, 55.0)], crs="EPSG:4326"
        )

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)
    monkeypatch.setattr(gpd, "read_file", fake_read_file)
    prepare_inputs(paths, scenario, _toy_assumptions())
    return seen


def test_input_files_are_what_prepare_inputs_reads_without_water(tmp_path, monkeypatch) -> None:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    scenario = _scenario()
    files = _input_files(paths, scenario)
    assert _reads_during_prepare(paths, scenario, monkeypatch) == {p.resolve() for p in files.values()}
    assert "water_basin_caps" not in files and "basin_polygons" not in files


def test_input_files_are_what_prepare_inputs_reads_with_water(tmp_path, monkeypatch) -> None:
    """有水约束时多读流域指标，以及 data/ 下给电厂与工业 hub 分流域用的一级区多边形。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    pd.DataFrame(
        [{"basin_code": "B1", "planning_year": year, "residual_m3_per_year": 6.0e6} for year in YEARS]
    ).to_csv(paths.inputs_dir / "water_basin_caps.csv", index=False)
    polygons = paths.data_dir / "ChinaBasins" / "basin_l1.gpkg"
    polygons.parent.mkdir(parents=True)
    polygons.write_bytes(b"toy")  # `load_basins` 先查文件在不在；内容由 fake_read_file 给
    scenario = _scenario(**WATER)
    files = _input_files(paths, scenario)
    assert _reads_during_prepare(paths, scenario, monkeypatch) == {p.resolve() for p in files.values()}
    assert files["basin_polygons"] == polygons


def test_digest_covers_the_files_that_were_read(tmp_path) -> None:
    scenario = _scenario()
    paths_a = _write_toy_inputs(tmp_path / "tree_a", retirement_year=9999)
    paths_b = _write_toy_inputs(tmp_path / "tree_b", retirement_year=9999)
    nodes = pd.read_csv(paths_b.inputs_dir / "pipeline_nodes.csv")
    nodes.loc[nodes["node_id"] == "N2", "lon"] = 112.6
    nodes.to_csv(paths_b.inputs_dir / "pipeline_nodes.csv", index=False)

    a = _input_digest(paths_a.inputs_dir, _input_files(paths_a, scenario))
    b = _input_digest(paths_b.inputs_dir, _input_files(paths_b, scenario))
    assert a["digest_plants"] == b["digest_plants"]
    # 两棵树只差管网：摘要必须能区分，这正是仓库根 v7 与 _indtree 的差别所在。
    assert a["digest_pipeline_nodes"] != b["digest_pipeline_nodes"]
    assert str(a["input_dir"]).endswith("tree_a/inputs")
    # 只摘读了的文件：无水约束时没有流域指标，求解不读的 water_supply_links 也不在内。
    assert set(a) == {"input_dir"} | {f"digest_{key}" for key in _input_files(paths_a, scenario)}
    assert "digest_water_basin_caps" not in a and "digest_water_links" not in a
    # `plot_style.input_vintage` 靠它判断两次求解是否用的同一份水。
    assert a["digest_water_availability"] is not None
    # 缺的文件记 None，不报错。
    (paths_a.inputs_dir / "storage_hubs.csv").unlink()
    assert _input_digest(paths_a.inputs_dir, _input_files(paths_a, scenario))["digest_storage_hubs"] is None


def test_mip_focus_is_read_back_from_the_model(tmp_path, monkeypatch) -> None:
    """溯源记模型上实际生效的 MIPFocus：未设环境变量时是 `_new_gurobi_model` 的缺省 1，不是 0。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    from coal_retrofit.optimization._shared import _new_gurobi_model

    monkeypatch.delenv("COAL_RETROFIT_MIPFOCUS", raising=False)
    model = _new_gurobi_model("provenance")
    try:
        assert _run_provenance(model, tmp_path, {})["mip_focus"] == 1
        model.Params.MIPFocus = 2
        assert _run_provenance(model, tmp_path, {})["mip_focus"] == 2
    finally:
        model.dispose()
