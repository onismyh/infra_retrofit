"""情景登记表（`coal_retrofit.scenarios`）与命令行（`coal_retrofit.cli`）：不求解，不需要 Gurobi。"""
from __future__ import annotations

import filecmp
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from coal_retrofit import cli
from coal_retrofit.runner import build_parameters
from coal_retrofit.scenarios import (
    ScenarioRegistryError,
    apply_sets,
    diff_resolved,
    experiments_dict,
    load_registry,
    parse_set,
    shown_path,
)

REPO = Path(__file__).resolve().parents[1]

FAMILY = """
[BASE]
abstract = true
tree = "tree"

[BASE.scenario]
mip_gap = 0.03
carbon_price_cny_per_t_by_year = [0, 0, 0, 0]

[CHILD]
extends = "BASE"
note = "child"

[CHILD.assumptions]
apply_basin_cap = false

[GRANDCHILD]
extends = "CHILD"

[GRANDCHILD.scenario]
mip_gap = 0.02
"""


def _registry_dir(tmp_path: Path, text: str, filename: str = "t.toml") -> Path:
    directory = tmp_path / "scenarios"
    directory.mkdir(exist_ok=True)
    (directory / filename).write_text(text, encoding="utf-8")
    return directory


def test_repo_registry_builds_every_runnable_scenario() -> None:
    registry = load_registry()
    names = registry.runnable()
    assert names and "ST_COMMON" not in names
    assert list(experiments_dict(registry)) == names
    for name in names:
        spec = registry.get(name)
        assert spec.tree is not None
        scenario, assumptions = build_parameters(spec, name)
        assert scenario.experiment_id == name and scenario.solver_threads == 8
        if name.startswith("ST_"):
            # CLAUDE.md 二.2：ST_ 系 MIPGap 统一 3%；二.6：只在 v9.2 管网（_indtree/inputs/）上求解。
            assert scenario.mip_gap == 0.03 and shown_path(spec.tree) == "_indtree"


def test_st_water_scenario_differs_from_base_only_in_water() -> None:
    """ST_WA 与 ST_BASE 是要相减的一对：两者只许差水的三项，别的差异都会混进水的效应。"""
    registry = load_registry()

    def params(name: str) -> dict:
        scenario, assumptions = build_parameters(registry.get(name), name)
        return {"tree": "t", "scenario": asdict(scenario), "assumptions": asdict(assumptions)}

    diffs = diff_resolved(params("ST_BASE"), params("ST_WA_cwatm_126_dry_oq"))
    assert [key for key, _, _ in diffs] == [
        "scenario.water_mode", "scenario.water_scenario_id", "scenario.water_season",
    ]


def test_inheritance_merges_sections_and_keeps_note_local(tmp_path) -> None:
    registry = load_registry(_registry_dir(tmp_path, FAMILY))
    assert registry.runnable() == ["CHILD", "GRANDCHILD"]
    spec = registry.get("GRANDCHILD")
    assert registry.lineage("GRANDCHILD") == ["GRANDCHILD", "CHILD", "BASE"]
    assert spec.tree == tmp_path / "tree"
    assert spec.note == "" and registry.get("CHILD").note == "child"
    assert spec.scenario == {"mip_gap": 0.02, "carbon_price_cny_per_t_by_year": (0.0, 0.0, 0.0, 0.0)}
    assert spec.assumptions == {"apply_basin_cap": False}
    # 整数写在浮点字段里读成浮点：与原字典里的 0.0 逐位相同，repr 也相同。
    assert repr(spec.scenario["carbon_price_cny_per_t_by_year"]) == "(0.0, 0.0, 0.0, 0.0)"
    with pytest.raises(ScenarioRegistryError, match="抽象基底"):
        registry.get("BASE")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[X]\nextend = "Y"\ntree = "t"\n', "是不是 extends"),
        ('[X]\ntree = "t"\n[X.scenario]\nmip_gapp = 0.1\n', "是不是 mip_gap"),
        ('[X]\ntree = "t"\n[X.scenario]\napply_basin_cap = false\n', "是 assumptions 节的字段"),
        ('[X]\ntree = "t"\n[X.scenario]\nsolver_threads = 4\n', "--threads"),
        ('[X]\ntree = "t"\n[X.scenario]\nexperiment_id = "X"\n', "取结果名"),
        ('[X]\ntree = "t"\n[X.scenario]\nmip_gap = "0.03"\n', "应为 float"),
        ('[X]\ntree = "t"\n[X.assumptions]\nmax_parallel_pipes = true\n', "应为 int"),
        ('[X]\ntree = "t"\n[X.scenario]\nplanning_years = [2030.5]\n', "应为 int"),
        ('[X]\ntree = "t"\n[X.assumptions]\nprovince_operating_hours = [1.0]\n', "应为 dict"),
        ('[A]\nextends = "B"\ntree = "t"\n[B]\nextends = "A"\n', "成环"),
        ('[A]\nextends = "NOPE"\ntree = "t"\n', "不在登记表"),
        ("[A]\n[A.scenario]\nmip_gap = 0.1\n", "没有 tree"),
        ('["bad name"]\ntree = "t"\n', "情景名只许"),
        ('["A."]\ntree = "t"\n', "情景名只许"),  # Windows 去掉目录名末尾的点，会写进 A/
        ('["A.Json"]\ntree = "t"\n', "情景名只许"),  # 结果目录 A.Json/ 与 A 的 A.json 同路径（Windows 不分大小写）
        ('[A]\ntree = "t"\n[a]\ntree = "t"\n', "只差大小写"),  # Windows 上是同一个结果文件
        ('[A]\nabstract = "yes"\n', "abstract 应为 bool"),
        ("A = 1\n", "顶层只能是情景表"),
        ('[A]\ntree = "t"\nscenario = 1\n', "应是一张表"),
        ("[A\n", "TOML 语法错误"),
    ],
)
def test_registry_rejects_mistakes(tmp_path, text: str, message: str) -> None:
    with pytest.raises(ScenarioRegistryError, match=message):
        load_registry(_registry_dir(tmp_path, text))


def test_duplicate_names_across_files_are_rejected(tmp_path) -> None:
    directory = _registry_dir(tmp_path, '[A]\ntree = "t"\n', "a.toml")
    _registry_dir(tmp_path, '[A]\ntree = "u"\n', "b.toml")
    with pytest.raises(ScenarioRegistryError, match="重复"):
        load_registry(directory)


def test_empty_registry_directory_is_an_error(tmp_path) -> None:
    with pytest.raises(ScenarioRegistryError, match="没有 \\*.toml"):
        load_registry(tmp_path)


def test_parse_set_reads_toml_values_and_bare_strings() -> None:
    assert parse_set("assumptions.air_retrofit_capex_cny_per_kw=400") == (
        "assumptions", "air_retrofit_capex_cny_per_kw", 400.0)
    assert parse_set("scenario.water_scenario_id=cwatm|gfdl-esm4|ssp126")[2] == "cwatm|gfdl-esm4|ssp126"
    assert parse_set('scenario.water_mode="grid_supply"')[2] == "grid_supply"
    assert parse_set("scenario.carbon_price_cny_per_t_by_year=[1, 2, 3, 4]")[2] == (1.0, 2.0, 3.0, 4.0)
    assert parse_set("assumptions.apply_basin_cap=false")[2] is False
    for bad in ("mip_gap=0.1", "scenario.mip_gap", "setting.mip_gap=0.1"):
        with pytest.raises(ScenarioRegistryError, match="应写成"):
            parse_set(bad)
    with pytest.raises(ScenarioRegistryError, match="应为 float"):
        parse_set("scenario.mip_gap=abc")
    # 漏了引号或括号：不当字符串收下（否则 water_season 会静默变成 '"dry'，按全年水量算）。
    for broken in ('scenario.water_season="dry', "scenario.water_season='dry", "scenario.planning_years=[2030, 2040",
                   "assumptions.province_coal_cost_cny_per_gj={Beijing = 38.6"):
        with pytest.raises(ScenarioRegistryError, match="读不成"):
            parse_set(broken)


def test_dict_fields_are_replaced_whole(tmp_path) -> None:
    """字典型字段整张替换，不按键合并：子情景、`--set` 只写一个省，别的省不从父情景或缺省继承。"""
    text = """
[P]
abstract = true
tree = "t"

[P.assumptions]
province_coal_cost_cny_per_gj = {Anhui = 40.0, Beijing = 38.6}

[C]
extends = "P"

[C.assumptions]
province_coal_cost_cny_per_gj = {Anhui = 45.0}
"""
    spec = load_registry(_registry_dir(tmp_path, text)).get("C")
    assert spec.assumptions == {"province_coal_cost_cny_per_gj": {"Anhui": 45.0}}
    new, _ = apply_sets(spec, ["assumptions.province_coal_cost_cny_per_gj={Beijing = 50}"])
    assert build_parameters(new, "C")[1].province_coal_cost_cny_per_gj == {"Beijing": 50.0}


def test_apply_sets_later_wins_and_records_what_was_applied(tmp_path) -> None:
    spec = load_registry(_registry_dir(tmp_path, FAMILY)).get("CHILD")
    new, applied = apply_sets(spec, ["scenario.mip_gap=0.05", "assumptions.apply_basin_cap=true",
                                     "scenario.mip_gap=0.04"])
    assert applied == {"scenario.mip_gap": 0.04, "assumptions.apply_basin_cap": True}
    assert new.scenario["mip_gap"] == 0.04 and new.assumptions["apply_basin_cap"] is True
    assert spec.scenario["mip_gap"] == 0.03 and spec.assumptions["apply_basin_cap"] is False  # 原情景不变


def test_diff_resolved_matches_tuples_to_lists_and_skips_names() -> None:
    a = {"tree": "t", "scenario": {"experiment_id": "A", "description": "A", "years": (1, 2)},
         "assumptions": {"x": 1.0}}
    b = {"tree": "t", "scenario": {"experiment_id": "B", "description": "B", "years": [1, 2]},
         "assumptions": {"x": 2.0}}
    assert diff_resolved(a, b) == [("assumptions.x", 1.0, 2.0)]
    assert diff_resolved(a, {**b, "tree": "u", "assumptions": {"x": 1.0}}) == [("tree", "t", "u")]


def test_cli_list_show_diff(tmp_path, capsys) -> None:
    directory = str(_registry_dir(tmp_path, FAMILY))
    assert cli.main(["--registry", directory, "list"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in listed] == ["CHILD", "GRANDCHILD"]
    assert cli.main(["--registry", directory, "show", "GRANDCHILD", "--set", "scenario.capture_rate=0.8"]) == 0
    shown = capsys.readouterr().out
    assert "* mip_gap = 0.02    （缺省 0.01）" in shown
    assert "* capture_rate = 0.8    （缺省 0.9）" in shown
    assert "  capacity_factor = 0.55" in shown
    assert cli.main(["--registry", directory, "diff", "CHILD", "GRANDCHILD"]) == 0
    assert capsys.readouterr().out.splitlines() == ["CHILD → GRANDCHILD：1 项不同", "  scenario.mip_gap: 0.03 → 0.02"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "CHILD", "--set", "scenario.mip_gap=0.1"], "必须用 --as"),
        (["run", "CHILD", "--set", "scenario.mip_gap=0.1", "--as", "GRANDCHILD"], "同名"),
        (["run", "CHILD", "--as", "BASE"], "同名"),
        (["run", "CHILD", "--as", "a b"], "结果名只许"),
        (["run", "CHILD", "--as", "X."], "结果名只许"),
        (["run", "CHILD", "--set", "scenario.mip_gap=0.1", "--as", "CHILD.json"], "结果名只许"),
        (["run", "CHILD", "--set", "scenario.mip_gap=0.1", "--as", "child"], "只差大小写"),
    ],
)
def test_cli_run_refuses_ambiguous_result_names(tmp_path, capsys, argv: list[str], message: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--registry", str(_registry_dir(tmp_path, FAMILY)), *argv])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_run_stops_before_solving_when_tree_has_no_inputs(tmp_path, capsys) -> None:
    directory = str(_registry_dir(tmp_path, FAMILY))
    assert cli.main(["--registry", directory, "run", "CHILD"]) == 1
    assert "没有 inputs/" in capsys.readouterr().err
    assert cli.main(["--registry", directory, "run", "CHILDE"]) == 1
    assert "是不是 CHILD" in capsys.readouterr().err
    assert cli.main(["--registry", directory, "run", "child"]) == 1
    assert "是不是 CHILD？（情景名区分大小写）" in capsys.readouterr().err


@pytest.mark.parametrize("directory", [REPO / "scripts", REPO / "_indtree" / "scripts"], ids=["scripts", "_indtree"])
def test_run_single_wrapper(monkeypatch, capsys, directory: Path) -> None:
    """两份薄壳：出图脚本 `from run_single import EXPERIMENTS` 的形状，原命令行的改写，`--list` 的原格式。"""
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.delitem(sys.modules, "_bootstrap", raising=False)  # 两份 _bootstrap 不同，各载各的
    alias = f"run_single_under_test_{directory.parent.name}"
    spec = importlib.util.spec_from_file_location(alias, directory / "run_single.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    assert module.EXPERIMENTS == experiments_dict(load_registry())
    scenario_kw, assumption_kw = module.EXPERIMENTS["ST_BASE"]
    assert isinstance(scenario_kw, dict) and isinstance(assumption_kw, dict)

    calls: list[list[str]] = []

    def fake_cli(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(module, "cli_main", fake_cli)
    for argv in ([], ["--threads", "8"], ["ST_CP_BASE", "--threads", "1"]):
        assert module.main(argv) == 0
    assert calls == [["run", "ST_BASE"], ["run", "ST_BASE", "--threads", "8"],
                     ["run", "ST_CP_BASE", "--threads", "1"]]  # 不写情景名时跑原来的缺省 ST_BASE
    capsys.readouterr()
    assert module.main(["--list"]) == 0 and len(calls) == 3
    assert capsys.readouterr().out.splitlines() == list(module.EXPERIMENTS)  # 原格式：每行一个名字


def test_two_run_single_copies_are_identical() -> None:
    assert filecmp.cmp(REPO / "scripts" / "run_single.py", REPO / "_indtree" / "scripts" / "run_single.py", shallow=False)
