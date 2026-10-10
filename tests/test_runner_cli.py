"""`python -m coal_retrofit run` 在 toy 上走通：15 张结果表、result.json 与 `resolved` 段；
`--set` 配 `--as` 另起结果名、不动登记情景的结果；`--tree` 换求解树；已有结果时拒绝，`--force` 才覆盖；
情景字段 `solver_seed`、`mip_focus` 设到模型上；情景 `warm_start = "lp_relax"` 的一次 run 与手工的两次 run
逐字节相同。需要 Gurobi。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from coal_retrofit import cli
from coal_retrofit.optimization import solver
from coal_retrofit.scenarios import diff_resolved
from toy_inputs import _write_targets, _write_toy_inputs

pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

TABLES = {
    "pathway_shares.csv", "province_pathways.csv", "plant_detail.csv", "industry_detail.csv",
    "network_edges.csv", "storage_utilization.csv", "resource_use.csv", "biomass_flows.csv",
    "ammonia_flows.csv", "water_flows.csv", "co2_flow_direction.csv", "plant_cost.csv",
    "slack_detail.csv", "cost_breakdown.csv", "sanity_checks.csv",
}

REGISTRY = """
[TOY]
tree = "toy"

[TOY.scenario]
planning_years = [2050, 2060]
sector_target_source = "toy"
electricity_price_cny_per_mwh_by_year = [490.0, 550.0]
carbon_price_cny_per_t_by_year = [150.0, 300.0]

[TOY.assumptions]
storage_national_injection_mtpa_by_year = [500.0, 800.0, 1000.0, 1500.0]

[TOY_WARM]
extends = "TOY"

[TOY_WARM.scenario]
warm_start = "lp_relax"
warm_start_time_limit = 60
"""


def _toy_tree(root: Path) -> None:
    paths = _write_toy_inputs(root, retirement_year=9999)
    _write_targets(paths, {2050: 0.5, 2060: 0.5})
    # 结果表要读封存汇的三列说明字段，求解测试的 toy 没有。
    hubs = pd.read_csv(paths.inputs_dir / "storage_hubs.csv")
    hubs["province"], hubs["source"], hubs["year_basis"] = "Shanxi", "toy", "toy"
    hubs.to_csv(paths.inputs_dir / "storage_hubs.csv", index=False)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry(tmp_path: Path) -> Path:
    _toy_tree(tmp_path / "toy")
    registry = tmp_path / "scenarios"
    registry.mkdir()
    (registry / "toy.toml").write_text(REGISTRY, encoding="utf-8")
    return registry


def test_run_writes_tables_result_and_resolved(tmp_path, capsys) -> None:
    registry = _registry(tmp_path)
    argv = ["--registry", str(registry), "run", "TOY", "--threads", "1", "--time-limit", "60"]

    assert cli.main(argv) == 0
    assert any(line.startswith("TOY: obj=") for line in capsys.readouterr().out.splitlines())  # Gurobi 日志在前
    results = tmp_path / "toy" / "results"
    assert {p.name for p in (results / "TOY").glob("*.csv")} == TABLES
    base = _load(results / "TOY.json")
    resolved = base["resolved"]
    assert base["name"] == "TOY" and resolved["registered_as"] == "TOY" and resolved["set"] == {}
    assert resolved["tree"] == (tmp_path / "toy").resolve().as_posix()
    assert resolved["options"] == {"threads": 1, "time_limit": 60, "mip_gap": None, "sol_dir": None, "force": False}
    assert resolved["scenario"]["solver_threads"] == 1 and resolved["scenario"]["solver_time_limit"] == 60
    assert resolved["scenario"]["planning_years"] == [2050, 2060]
    assert resolved["assumptions"]["storage_national_injection_mtpa_by_year"] == [500.0, 800.0, 1000.0, 1500.0]
    assert resolved["warm_start"] is None and set(resolved["code"]) == {"commit", "dirty"}
    # 摘要按逻辑名记，`input_files` 同名：check_run_provenance --pair 靠它分辨两边读的是不是同一个文件。
    assert resolved["input_files"]["sector_targets"] == "inputs/sector_targets_toy.csv"
    assert {f"digest_{key}" for key in resolved["input_files"]} == {
        key for key in base["solver_quality"] if key.startswith("digest_")}

    # 已有结果：求解之前就拒绝，结果不动。
    capsys.readouterr()
    assert cli.main(argv) == 1
    assert "TOY.json 已存在：要覆盖加 --force" in capsys.readouterr().err
    assert _load(results / "TOY.json") == base

    # --set 配 --as：结果另起名，登记情景名下的结果不动；两段 resolved 只差覆盖的那一项。
    assert cli.main([*argv, "--set", "assumptions.air_retrofit_capex_cny_per_kw=400", "--as", "TOY_air400"]) == 0
    other = _load(results / "TOY_air400.json")
    assert other["name"] == "TOY_air400" and other["resolved"]["registered_as"] == "TOY"
    assert other["resolved"]["set"] == {"assumptions.air_retrofit_capex_cny_per_kw": 400.0}
    assert diff_resolved(resolved, other["resolved"]) == [("assumptions.air_retrofit_capex_cny_per_kw", 300.0, 400.0)]
    assert {p.name for p in (results / "TOY_air400").glob("*.csv")} == TABLES
    assert _load(results / "TOY.json") == base

    # --tree 换求解树：读那棵树的 inputs/，结果也写在那里。
    shutil.copytree(tmp_path / "toy" / "inputs", tmp_path / "elsewhere" / "inputs")
    assert cli.main([*argv, "--tree", str(tmp_path / "elsewhere")]) == 0
    moved = _load(tmp_path / "elsewhere" / "results" / "TOY.json")
    assert diff_resolved(resolved, moved["resolved"]) == [
        ("tree", resolved["tree"], (tmp_path / "elsewhere").resolve().as_posix())]
    assert moved["global_objective_cny"] == base["global_objective_cny"]


def test_seed_and_mip_focus_reach_the_model(tmp_path) -> None:
    """情景字段 `solver_seed`、`mip_focus` 经运行器设到 Gurobi 模型上：溯源记的是从模型上读回的值。"""
    registry = _registry(tmp_path)
    assert cli.main(["--registry", str(registry), "run", "TOY", "--threads", "1", "--time-limit", "60",
                     "--set", "scenario.solver_seed=3", "--set", "scenario.mip_focus=2", "--as", "TOY_seed3"]) == 0
    quality = _load(tmp_path / "toy" / "results" / "TOY_seed3.json")["solver_quality"]
    assert (quality["seed"], quality["mip_focus"]) == (3, 2)


@pytest.mark.usefixtures("ascii_tmp_path")
def test_warm_start_matches_the_manual_two_steps(tmp_path, monkeypatch) -> None:
    """情景 `warm_start = "lp_relax"` 的一次 run 与此前手工的两次 run（第 1 步设 LP_RELAX 与 WRITE_SOL，第 2 步设
    START_SOL）是同一对求解：第 1 步的 .sol、第 2 步的目标函数与 15 张表逐字节相同。两边走的是同一段代码，所以另核
    第 1 步确实解的是 LP 松弛、第 2 步确实按这份 .sol 设了 MIP start。"""
    starts: list[Path] = []
    apply_start = solver._apply_rounded_start

    def recording_start(model, path, assumptions):
        starts.append(Path(path))
        return apply_start(model, path, assumptions)

    monkeypatch.setattr(solver, "_apply_rounded_start", recording_start)
    registry = _registry(tmp_path)
    argv = ["--registry", str(registry), "run", "TOY_WARM", "--threads", "1", "--time-limit", "60"]
    assert cli.main(argv) == 0
    results = tmp_path / "toy" / "results"
    assert starts == [results / "TOY_WARM.lp.sol"]
    sol = tmp_path / "manual.lp.sol"
    manual = [*argv, "--set", "scenario.warm_start=none", "--as"]
    monkeypatch.setenv("COAL_RETROFIT_LP_RELAX", "1")
    monkeypatch.setenv("COAL_RETROFIT_WRITE_SOL", str(sol))
    assert cli.main([*manual, "MANUAL_LP"]) == 0
    monkeypatch.delenv("COAL_RETROFIT_LP_RELAX")
    monkeypatch.delenv("COAL_RETROFIT_WRITE_SOL")
    monkeypatch.setenv("COAL_RETROFIT_START_SOL", str(sol))
    assert cli.main([*manual, "MANUAL"]) == 0
    assert starts[1:] == [sol]

    assert (results / "TOY_WARM.lp.sol").read_bytes() == sol.read_bytes()
    auto = _load(results / "TOY_WARM.json")
    assert auto["global_objective_cny"] == _load(results / "MANUAL.json")["global_objective_cny"]
    for table in sorted(TABLES):
        assert (results / "TOY_WARM" / table).read_bytes() == (results / "MANUAL" / table).read_bytes(), table
    step_one = auto["resolved"]["warm_start"]
    assert step_one["method"] == "lp_relax" and step_one["time_limit"] == 60 and step_one["sol_sha256"]
    assert step_one["objective_cny"] == _load(results / "MANUAL_LP.json")["global_objective_cny"]
    assert step_one["objective_cny"] < auto["global_objective_cny"]  # LP 松弛是 MIP 的下界，toy 上严格低
