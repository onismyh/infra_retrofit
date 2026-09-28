"""溯源记录：输入摘要只摘本情景实际读取的文件、按实际读取的目录、在读完文件时计算；
mip_focus 读回模型上的实际值。"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from coal_retrofit.optimization._shared import SolveState
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
    # 两棵树只差管网：摘要必须能区分，这正是 v7（a303f05 的仓库根 inputs/）与 _indtree 的差别所在。
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


def _rewrite_plants(paths) -> None:
    """改一个数，模拟求解期间有人重建了输入。"""
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    plants["total_capacity_mw"] = plants["total_capacity_mw"].astype(float) * 2.0
    plants.to_csv(paths.inputs_dir / "plants.csv", index=False)


def test_digest_is_taken_when_prepare_inputs_reads(tmp_path) -> None:
    """摘要在 `prepare_inputs` 读完文件时算好，之后改写输入不影响已记下的值。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    scenario = _scenario()
    prepared = prepare_inputs(paths, scenario, _toy_assumptions())
    at_read = _input_digest(paths.inputs_dir, _input_files(paths, scenario))
    assert prepared.input_digest == at_read
    _rewrite_plants(paths)
    assert _input_digest(paths.inputs_dir, _input_files(paths, scenario))["digest_plants"] != at_read["digest_plants"]
    assert prepared.input_digest == at_read


def test_solver_records_the_digest_taken_at_read_time(tmp_path) -> None:
    """求解结果的溯源照抄 `PreparedInputs.input_digest`，不在求解结束后重算：读完输入后改写的文件不算数。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    from coal_retrofit.optimization.solver import _solve_joint_multi_period

    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    scenario = _scenario(solver_time_limit=60)
    assumptions = _toy_assumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    _rewrite_plants(paths)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    quality = _solve_joint_multi_period(prepared, scenario, assumptions, YEARS, state)["solver_quality"]
    assert {k: quality[k] for k in prepared.input_digest} == prepared.input_digest
    assert quality["digest_plants"] != _input_digest(paths.inputs_dir, _input_files(paths, scenario))["digest_plants"]


def test_seed_and_mip_focus_are_read_back_from_the_model() -> None:
    """溯源记模型上实际生效的 Seed 与 MIPFocus（由情景字段 `solver_seed`、`mip_focus` 经 `_new_gurobi_model` 设上）：
    缺省是 Gurobi 的 Seed 0 与 `_new_gurobi_model` 的 MIPFocus 1，不是 0。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    from coal_retrofit.optimization._shared import _new_gurobi_model

    model = _new_gurobi_model("provenance")
    seeded = _new_gurobi_model("provenance", seed=3, mip_focus=2)
    try:
        assert {k: _run_provenance(model, {})[k] for k in ("seed", "mip_focus")} == {"seed": 0, "mip_focus": 1}
        assert {k: _run_provenance(seeded, {})[k] for k in ("seed", "mip_focus")} == {"seed": 3, "mip_focus": 2}
        model.Params.MIPFocus = 2
        assert _run_provenance(model, {})["mip_focus"] == 2
    finally:
        model.dispose()
        seeded.dispose()
