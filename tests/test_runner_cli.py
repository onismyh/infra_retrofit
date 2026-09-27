"""`python -m coal_retrofit run` 在 toy 上走通：15 张结果表、result.json 与 `resolved` 段；
`--set` 配 `--as` 另起结果名、不动登记情景的结果；`--tree` 换求解树。需要 Gurobi。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from coal_retrofit import cli
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
storage_deployment_fraction_by_year = [1.0, 1.0, 1.0, 1.0]
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


def test_run_writes_tables_result_and_resolved(tmp_path, capsys) -> None:
    _toy_tree(tmp_path / "toy")
    registry = tmp_path / "scenarios"
    registry.mkdir()
    (registry / "toy.toml").write_text(REGISTRY, encoding="utf-8")
    argv = ["--registry", str(registry), "run", "TOY", "--threads", "1", "--time-limit", "60"]

    assert cli.main(argv) == 0
    assert any(line.startswith("TOY: obj=") for line in capsys.readouterr().out.splitlines())  # Gurobi 日志在前
    results = tmp_path / "toy" / "results"
    assert {p.name for p in (results / "TOY").glob("*.csv")} == TABLES
    base = _load(results / "TOY.json")
    resolved = base["resolved"]
    assert base["name"] == "TOY" and resolved["registered_as"] == "TOY" and resolved["set"] == {}
    assert resolved["tree"] == (tmp_path / "toy").resolve().as_posix()
    assert resolved["options"] == {"threads": 1, "time_limit": 60, "mip_gap": None}
    assert resolved["scenario"]["solver_threads"] == 1 and resolved["scenario"]["solver_time_limit"] == 60
    assert resolved["scenario"]["planning_years"] == [2050, 2060]
    assert resolved["assumptions"]["storage_deployment_fraction_by_year"] == [1.0, 1.0, 1.0, 1.0]

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
