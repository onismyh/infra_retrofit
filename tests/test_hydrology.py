"""水文边界条件的入口（`coal_retrofit.hydrology` 与命令行的 `--water` / `--season` / `--targets`、`water-members`）：
成员解析与数据校验、自动结果名、覆盖项记进 `resolved.set`，数据不全的成员在求解之前就拒绝。不求解，不需要 Gurobi。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from coal_retrofit import cli, runner
from coal_retrofit.constants import PLANNING_YEARS
from coal_retrofit.hydrology import check_member, check_members, load_members, member_id, water_overrides
from coal_retrofit.runner import build_parameters
from coal_retrofit.scenarios import ScenarioRegistryError, diff_resolved, load_registry

REPO = Path(__file__).resolve().parents[1]
MEMBERS = ("cwatm|gfdl-esm4|ssp126", "cwatm|gfdl-esm4|ssp370")
NAN = float("nan")
REGISTRY = """
[BASE]
tree = "tree"

[BASE.scenario]
mip_gap = 0.03

[WET]
extends = "BASE"

[WET.scenario]
water_mode = "high_water_stress"
water_scenario_id = "cwatm|gfdl-esm4|ssp126"
water_season = "dry"
"""


def _water_tree(root: Path, members: tuple[str, ...] = MEMBERS) -> Path:
    """每个成员两个水节点（同一流域 A）、四个规划年都齐全的最小输入。"""
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    pd.DataFrame({"scenario_id": members, "scenario_family": "baseline"}).to_csv(
        inputs / "water_scenarios.csv", index=False)
    pd.DataFrame({"water_node_id": ["N1", "N2"]}).to_csv(inputs / "water_nodes.csv", index=False)
    pd.DataFrame([
        {"water_node_id": node, "planning_year": year, "scenario_id": member, "basin_code": "A",
         "available_water_m3_per_year": 10.0, "dry_season_water_m3_per_year": 2.0, "bias_factor": 1.1}
        for member in members for year in PLANNING_YEARS for node in ("N1", "N2")
    ]).to_csv(inputs / "water_availability.csv", index=False)
    pd.DataFrame([
        {"scenario_id": member, "planning_year": year, "basin_code": "A", "domestic_m3_per_year": 1.0,
         "irrigation_m3_per_year": 1.0, "dry_season_domestic_m3_per_year": 1.0, "dry_season_irrigation_m3_per_year": 0.5}
        for member in members for year in PLANNING_YEARS
    ]).to_csv(inputs / "water_basin_use.csv", index=False)
    return root


def _registry(tmp_path: Path) -> str:
    _water_tree(tmp_path / "tree")
    directory = tmp_path / "scenarios"
    directory.mkdir()
    (directory / "t.toml").write_text(REGISTRY, encoding="utf-8")
    return str(directory)


def _fake_run(calls: list) -> Callable:
    def fake(spec, **kwargs):
        calls.append((spec, kwargs))
        return {"name": kwargs["name"] or spec.name, "global_objective_cny": 1.0, "years": {},
                "elapsed_seconds": 0.0, "solver_quality": {}}
    return fake


class Reached(Exception):
    """`runner.run` 过了求解之前的各项检查，走到了建模求解。"""


def _stop_at_solve(monkeypatch) -> None:
    def solve(*args, **kwargs):
        raise Reached
    monkeypatch.setattr(runner, "solve", solve)


def _at(frame: pd.DataFrame, year: int, node: str | None = None) -> pd.Series:
    """第一个成员 *year* 年（*node* 节点）的行。"""
    mask = (frame["scenario_id"] == MEMBERS[0]) & (frame["planning_year"] == year)
    return mask & (frame["water_node_id"] == node) if node is not None else mask


def _with(frame: pd.DataFrame, mask: pd.Series, column: str, value: object) -> pd.DataFrame:
    frame = frame.copy()
    frame.loc[mask, column] = value
    return frame


def _rewrite(root: Path, mutate: Callable) -> None:
    inputs = root / "inputs"
    availability, use = mutate(pd.read_csv(inputs / "water_availability.csv"),
                               pd.read_csv(inputs / "water_basin_use.csv"))
    availability.to_csv(inputs / "water_availability.csv", index=False)
    if use is None:
        (inputs / "water_basin_use.csv").unlink()
    else:
        use.to_csv(inputs / "water_basin_use.csv", index=False)


def test_repo_members_are_complete(capsys) -> None:
    """`_indtree` 的 20 个成员（2 个水文模型 × 5 个 GCM × 2 个 SSP）四个规划年的数据都齐全。"""
    problems = check_members(REPO / "_indtree", PLANNING_YEARS)
    assert len(problems) == 20 and not any(problems.values())
    assert cli.main(["water-members"]) == 0
    out = capsys.readouterr().out
    assert "20 个成员" in out and "watergap2-2e/ukesm1-0-ll/ssp370    high_pressure  齐全" in out


def test_water_option_reproduces_the_registered_water_scenario(monkeypatch) -> None:
    """`run ST_BASE --water cwatm/gfdl-esm4/ssp126 --season dry` 与登记的 ST_WA_cwatm_126_dry_oq 参数相同，只是结果名不同。"""
    calls: list = []
    monkeypatch.setattr(cli, "run", _fake_run(calls))
    assert cli.main(["run", "ST_BASE", "--water", "cwatm/gfdl-esm4/ssp126", "--season", "dry"]) == 0
    spec, kwargs = calls[0]
    assert kwargs["name"] == "ST_BASE_cwatm_gfdl-esm4_ssp126_dry"
    assert kwargs["applied_sets"] == {"scenario.water_mode": "grid_supply",
                                      "scenario.water_scenario_id": "cwatm|gfdl-esm4|ssp126",
                                      "scenario.water_season": "dry"}
    registry = load_registry()

    def params(name: str, s=None) -> dict:
        scenario, assumptions = build_parameters(s or registry.get(name), name)
        return {"tree": "t", "scenario": asdict(scenario), "assumptions": asdict(assumptions)}

    assert diff_resolved(params("ST_WA_cwatm_126_dry_oq"), params("x", spec)) == []


def test_water_overrides() -> None:
    members = pd.DataFrame({"scenario_id": list(MEMBERS)})
    member = member_id("cwatm/gfdl-esm4/ssp370", members)
    assert member == "cwatm|gfdl-esm4|ssp370" and member_id(member, members) == member
    # 无水约束的基底另设 water_mode；季节不给时取基底的（缺省 annual）
    assert water_overrides({}, member, None) == (
        {"water_mode": "grid_supply", "water_scenario_id": member}, "_cwatm_gfdl-esm4_ssp370_annual")
    # 已有水约束的基底保留它的 water_mode 与季节
    wet = {"water_mode": "high_water_stress", "water_season": "dry"}
    assert water_overrides(wet, member, None) == ({"water_scenario_id": member}, "_cwatm_gfdl-esm4_ssp370_dry")
    assert water_overrides(wet, member, "annual") == (
        {"water_scenario_id": member, "water_season": "annual"}, "_cwatm_gfdl-esm4_ssp370_annual")
    assert water_overrides(wet, None, None) == ({}, "")
    with pytest.raises(ScenarioRegistryError, match="只配 --water"):
        water_overrides({}, None, "dry")
    with pytest.raises(ScenarioRegistryError, match="是不是 cwatm/gfdl-esm4/ssp370"):
        member_id("cwatm/gfdl/ssp370", members)


def test_cli_boundary_options(tmp_path, monkeypatch, capsys) -> None:
    directory = _registry(tmp_path)
    calls: list = []
    monkeypatch.setattr(cli, "run", _fake_run(calls))
    base = ["--registry", directory, "run"]

    assert cli.main([*base, "WET", "--water", "cwatm/gfdl-esm4/ssp370", "--targets", "total"]) == 0
    spec, kwargs = calls[-1]
    assert kwargs["name"] == "WET_cwatm_gfdl-esm4_ssp370_dry_total" and spec.name == "WET"
    assert spec.scenario["water_mode"] == "high_water_stress"  # 基底已有水约束：不改 water_mode
    assert kwargs["applied_sets"] == {"scenario.water_scenario_id": "cwatm|gfdl-esm4|ssp370",
                                      "scenario.sector_target_mode": "total"}
    assert "结果名：WET_cwatm_gfdl-esm4_ssp370_dry_total" in capsys.readouterr().out

    # --as 说了算；--set 与边界选项一起记进 resolved.set
    assert cli.main([*base, "BASE", "--water", "cwatm/gfdl-esm4/ssp126", "--set", "scenario.mip_gap=0.05",
                     "--as", "MINE"]) == 0
    spec, kwargs = calls[-1]
    assert kwargs["name"] == "MINE" and spec.scenario["mip_gap"] == 0.05
    assert kwargs["applied_sets"] == {"scenario.mip_gap": 0.05, "scenario.water_mode": "grid_supply",
                                      "scenario.water_scenario_id": "cwatm|gfdl-esm4|ssp126"}
    # --set 先给了水模式：按并过 --set 的值判断，不再另设 water_mode
    assert cli.main([*base, "BASE", "--set", "scenario.water_mode=base_water", "--water", "cwatm/gfdl-esm4/ssp126",
                     "--as", "MINE2"]) == 0
    spec, kwargs = calls[-1]
    assert spec.scenario["water_mode"] == "base_water"
    assert kwargs["applied_sets"] == {"scenario.water_mode": "base_water",
                                      "scenario.water_scenario_id": "cwatm|gfdl-esm4|ssp126"}
    # 只用 --targets：结果名是基底名加 _total
    assert cli.main([*base, "BASE", "--targets", "total"]) == 0
    assert calls[-1][1]["name"] == "BASE_total"
    # 不用边界选项：结果名仍是登记情景名
    assert cli.main([*base, "BASE"]) == 0
    assert calls[-1][1]["name"] is None and calls[-1][1]["applied_sets"] == {}
    capsys.readouterr()

    # 与 --set 改同一字段、成员不存在、--season 不配 --water：都在求解之前报错，退出码 1
    count = len(calls)
    for argv, message in (
        (["--season", "dry", "--water", "cwatm/gfdl-esm4/ssp126", "--set", "scenario.water_season=annual",
          "--as", "X"], "只留一处"),
        (["--water", "cwatm/ipsl-cm6a-lr/ssp126"], "没有成员 'cwatm/ipsl-cm6a-lr/ssp126'"),
        (["--season", "dry"], "只配 --water"),
    ):
        assert cli.main([*base, "BASE", *argv]) == 1
        assert message in capsys.readouterr().err
    # --set 仍须配 --as：自动结果名不含 --set 的覆盖
    with pytest.raises(SystemExit) as excinfo:
        cli.main([*base, "BASE", "--water", "cwatm/gfdl-esm4/ssp126", "--set", "scenario.mip_gap=0.05"])
    assert excinfo.value.code == 2 and "必须用 --as" in capsys.readouterr().err
    assert len(calls) == count

    # show 显示覆盖项与 run 会用的结果名；带 --set 时 run 要另给 --as，不显示自动结果名
    show = ["--registry", directory, "show", "BASE", "--water", "cwatm/gfdl-esm4/ssp370", "--season", "dry"]
    assert cli.main(show) == 0
    shown = capsys.readouterr().out
    assert "--water/--season/--targets → scenario.water_scenario_id = 'cwatm|gfdl-esm4|ssp370'" in shown
    assert "run 的结果名（不给 --as 时）：BASE_cwatm_gfdl-esm4_ssp370_dry" in shown
    assert "* water_season = 'dry'    （缺省 'annual'）" in shown
    assert cli.main([*show, "--set", "scenario.mip_gap=0.05"]) == 0
    assert "run 的结果名" not in capsys.readouterr().out


@pytest.mark.parametrize(("registered", "message"), [
    ("BASE_total", "自动拼出的结果名 BASE_total：与登记表里的情景同名，"),
    ("base_TOTAL", "自动拼出的结果名 BASE_total：与登记表里的情景同名（与 base_TOTAL 只差大小写"),
])
def test_auto_name_must_not_clash_with_a_registered_scenario(tmp_path, capsys, registered: str, message: str) -> None:
    directory = _registry(tmp_path)
    (Path(directory) / "u.toml").write_text(f'[{registered}]\nextends = "BASE"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--registry", directory, "run", "BASE", "--targets", "total"])
    assert excinfo.value.code == 2 and message in capsys.readouterr().err


def test_as_name_errors_quote_only_the_name(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--registry", _registry(tmp_path), "run", "BASE", "--as", "a b"])
    assert "--as 'a b'：结果名只许" in capsys.readouterr().err


def test_members_come_from_the_tree_the_run_reads(tmp_path, monkeypatch, capsys) -> None:
    """`run --tree` 时成员表读那棵树的。"""
    directory = _registry(tmp_path)
    _water_tree(tmp_path / "other", members=("watergap2-2e|gfdl-esm4|ssp126",))
    calls: list = []
    monkeypatch.setattr(cli, "run", _fake_run(calls))
    argv = ["--registry", directory, "run", "BASE", "--water", "watergap2-2e/gfdl-esm4/ssp126"]
    assert cli.main(argv) == 1
    assert "没有成员 'watergap2-2e/gfdl-esm4/ssp126'" in capsys.readouterr().err
    assert cli.main([*argv, "--tree", str(tmp_path / "other")]) == 0
    assert calls[-1][1]["tree"] == (tmp_path / "other").absolute()
    assert calls[-1][1]["name"] == "BASE_watergap2-2e_gfdl-esm4_ssp126_annual"


def test_sector_target_mode_is_checked_early(tmp_path, capsys) -> None:
    assert cli.main(["--registry", _registry(tmp_path), "show", "BASE", "--set", "scenario.sector_target_mode=totl"]) == 1
    assert "只能是 'sector'、'total'，写的是 'totl'；是不是 total？" in capsys.readouterr().err


# (改坏数据, 报的问题, 不读那一列时不报的 check_member 参数)
GAPS = {
    "缺节点行": (lambda a, u: (a[~_at(a, 2040, "N2")], u), "2040：1 个水节点没有可用水量（如 N2）", None),
    "多余节点": (lambda a, u: (pd.concat([a, a[_at(a, 2040, "N2")].assign(water_node_id="N9")]), u),
             "2040：1 个水节点不在 water_nodes.csv（如 N9）", None),
    "重复节点": (lambda a, u: (pd.concat([a, a[_at(a, 2040, "N2")]]), u), "2040：1 个水节点重复（如 N2）", None),
    "枯水期径流为空": (lambda a, u: (_with(a, _at(a, 2050), "dry_season_water_m3_per_year", NAN), u),
                "2050：2 个水节点的 dry_season_water_m3_per_year 为空、非数或为负（如 N1, N2）", {"season": "annual"}),
    "全年径流为负": (lambda a, u: (_with(a, _at(a, 2030, "N1"), "available_water_m3_per_year", -1.0), u),
               "2030：1 个水节点的 available_water_m3_per_year 为空、非数或为负（如 N1）", {"season": "dry"}),
    "偏差因子为零": (lambda a, u: (_with(a, _at(a, 2060, "N1"), "bias_factor", 0.0), u),
               "2060：1 个水节点的 bias_factor 为空、非数或不为正（如 N1）", {"check_bias": False}),
    "没有流域码": (lambda a, u: (_with(a, _at(a, 2030, "N2"), "basin_code", NAN), u), "2030：1 个水节点没有流域码（如 N2）", None),
    "流域缺耗水": (lambda a, u: (_with(a, _at(a, 2060), "basin_code", "B"), u), "2060：1 个流域没有生活与灌溉耗水（如 B）", None),
    "耗水重复": (lambda a, u: (a, pd.concat([u, u[_at(u, 2040)]])), "2040：1 个流域的耗水重复（如 A）", None),
    "枯水期耗水为空": (lambda a, u: (a, _with(u, _at(u, 2050), "dry_season_irrigation_m3_per_year", NAN)),
                "2050：1 个流域的耗水为空、非数或为负（如 A）", {"season": "annual"}),
    "没有耗水表": (lambda a, u: (a, None), "没有 water_basin_use.csv（scripts/build_water_use.py 生成）", None),
    "缺列": (lambda a, u: (a.drop(columns="bias_factor"), u), "water_availability.csv 缺列 ['bias_factor']",
           {"check_bias": False}),
}


@pytest.mark.parametrize("gap", list(GAPS))
def test_check_member_reports_each_gap(tmp_path, gap: str) -> None:
    mutate, message, quiet = GAPS[gap]
    _rewrite(_water_tree(tmp_path), mutate)
    assert check_member(tmp_path, MEMBERS[0], PLANNING_YEARS) == [message]
    if quiet is not None:  # 只核模型这次读的列：不读的那一列坏了不拦
        assert check_member(tmp_path, MEMBERS[0], PLANNING_YEARS, **quiet) == []


def test_incomplete_member_is_reported_and_refused_before_solving(tmp_path, monkeypatch, capsys) -> None:
    directory = _registry(tmp_path)
    _rewrite(tmp_path / "tree", lambda a, u: (_with(a[~_at(a, 2040, "N2")], _at(a, 2060), "basin_code", "B"), u))
    assert check_member(tmp_path / "tree", MEMBERS[0], PLANNING_YEARS) == [
        "2040：1 个水节点没有可用水量（如 N2）", "2060：1 个流域没有生活与灌溉耗水（如 B）"]
    assert check_member(tmp_path / "tree", MEMBERS[1], PLANNING_YEARS) == []

    assert cli.main(["--registry", directory, "water-members"]) == 1
    out = capsys.readouterr().out
    assert "cwatm/gfdl-esm4/ssp126  baseline       2 处不全" in out and "    2040：1 个水节点没有可用水量（如 N2）" in out
    assert "cwatm/gfdl-esm4/ssp370  baseline       齐全" in out
    # 登记的水约束情景：求解之前就拒绝；换成数据齐全的成员就过了检查、走到求解
    _stop_at_solve(monkeypatch)
    assert cli.main(["--registry", directory, "run", "WET"]) == 1
    err = capsys.readouterr().err
    assert "水文成员 cwatm|gfdl-esm4|ssp126 的数据不全" in err and "2060：1 个流域没有生活与灌溉耗水（如 B）" in err
    with pytest.raises(Reached):
        cli.main(["--registry", directory, "run", "WET", "--water", "cwatm/gfdl-esm4/ssp370"])


def test_run_checks_only_what_it_reads(tmp_path, monkeypatch, capsys) -> None:
    """预检只核这次求解读的季节；无水约束、或有水约束而成员留空时不预检。"""
    directory = _registry(tmp_path)
    _rewrite(tmp_path / "tree", lambda a, u: (_with(a, _at(a, 2050), "dry_season_water_m3_per_year", NAN), u))
    _stop_at_solve(monkeypatch)
    run = ["--registry", directory, "run"]
    assert cli.main([*run, "WET"]) == 1
    assert "dry_season_water_m3_per_year 为空" in capsys.readouterr().err
    for argv in (["WET", "--set", "scenario.water_season=annual", "--as", "ANNUAL"],
                 ["BASE", "--set", "scenario.water_scenario_id=cwatm|gfdl-esm4|ssp126", "--set",
                  "scenario.water_season=dry", "--as", "DRYID"],
                 ["WET", "--set", 'scenario.water_scenario_id=""', "--as", "FAMILY"]):
        with pytest.raises(Reached):
            cli.main([*run, *argv])


def test_water_members_tree_choice(tmp_path, capsys) -> None:
    directory = _registry(tmp_path)
    assert load_members(tmp_path / "tree")["scenario_id"].tolist() == list(MEMBERS)
    (Path(directory) / "u.toml").write_text('[ELSEWHERE]\ntree = "other"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--registry", directory, "water-members"])
    assert excinfo.value.code == 2 and "2 棵求解树" in capsys.readouterr().err
    assert cli.main(["--registry", directory, "water-members", "--tree", str(tmp_path / "tree")]) == 0
    assert cli.main(["--registry", directory, "water-members", "--tree", str(tmp_path / "other")]) == 1
    assert "water_scenarios.csv 不存在" in capsys.readouterr().err
