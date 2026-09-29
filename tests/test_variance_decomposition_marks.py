"""`scripts/plot_ed_variance_decomposition.py` 的受约束流域标记（*）：只取登记表里带水约束的求解；没求解与旧格式
各自提示后跳过，都不当成"没有流域缺水"；图上没有 * 时，图注写明是没读到求解还是没有流域缺水。不求解，不需要 Gurobi。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from coal_retrofit.scenarios import load_registry

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    matplotlib = pytest.importorskip("matplotlib", reason="plot_style imports matplotlib")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    alias = "plot_ed_variance_decomposition_under_test"
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / "plot_ed_variance_decomposition.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    with matplotlib.rc_context():  # 脚本导入时调 apply_style()，别改掉同一进程里其他测试的 rcParams
        spec.loader.exec_module(module)
    return module


def test_binding_runs_are_registered_water_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """标记只取登记表里能求解、带水约束（`water_mode` 不是缺省的 `no_water`）、枯水期口径的情景，结果从它的 `tree`
    下读。原先写的两个 v9.1 情景名不在登记表里，结果读不到，图上一个 * 也没有。"""
    module = _load(monkeypatch)
    registry = load_registry()
    assert module.BINDING_RUNS
    for name in module.BINDING_RUNS:
        spec = registry.get(name)  # 没登记或是抽象基底时报错
        assert spec.scenario.get("water_mode", "no_water") != "no_water", name
        assert spec.scenario.get("water_season") == "dry", name
        assert spec.tree is not None and module.RESULTS_DIR == spec.tree / "results", name


def test_binding_basins_tells_unsolved_from_old_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """三份结果：没求解、旧格式（缺空冷两列）、现格式。前两种各自提示后跳过，只有现格式的进标记与读到的情景；
    标记只取 `water_supply` 松弛大于 1 m3 的流域（节点编号末段是流域代码；正好 1 m3 的不算），流域上限与生物质供给的松弛不算。"""
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "BINDING_RUNS", {"UNSOLVED": "未求解", "OLD": "旧格式", "NEW": "现格式"})
    (tmp_path / "OLD").mkdir()
    pd.DataFrame({"plant_id": ["P1"], "year": [2030]}).to_csv(tmp_path / "OLD" / "plant_detail.csv", index=False)
    (tmp_path / "NEW").mkdir()
    pd.DataFrame({"plant_id": ["P1"], "year": [2030], "air_cooled_share": [0.0], "already_air_share": [0.0]}).to_csv(
        tmp_path / "NEW" / "plant_detail.csv", index=False)
    pd.DataFrame({
        "year": [2030, 2030, 2060, 2060, 2060],
        "constraint_type": ["water_supply", "water_supply", "water_supply", "water_basin_quota", "biomass_supply"],
        "node_id": ["WC_0101_0202_H", "WC_0303_0404_K", "WC_0505_0606_A", "J", "BM_0707_C"],
        "slack_value": [5.0, 1.0, 1.01, 100.0, 50.0],
    }).to_csv(tmp_path / "NEW" / "slack_detail.csv", index=False)

    assert module.binding_basins() == (["A", "H"], ["NEW"])
    skips = [line.strip() for line in capsys.readouterr().out.splitlines() if "[skip]" in line]
    assert len(skips) == 2
    assert skips[0].startswith("[skip] UNSOLVED: no solved result")
    assert skips[1].startswith("[skip] OLD: old format")


def test_binding_note_says_why_there_is_no_star(monkeypatch: pytest.MonkeyPatch) -> None:
    """图注讲 * 的那一句分三种：标了；读到了求解但没有流域缺水；一个求解都没读到。后两种图上都没有 *，要写明是哪一种；
    读到了的只列读到的情景。"""
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "BINDING_RUNS", {"R1": "甲", "R2": "乙"})
    assert module.binding_note(used=["R2"], binding=["A", "H"]) == (
        "* = 按 乙的求解，任一规划年出现未满足需求的流域（只超出流域用水总量控制指标的不算）。")
    assert module.binding_note(used=["R1", "R2"], binding=[]) == (
        "按 甲、乙的求解，没有流域在任一规划年出现未满足需求（只超出流域用水总量控制指标的不算），所以图上没有 *。")
    assert module.binding_note(used=["R1"], binding=[]) == (
        "按 甲的求解，没有流域在任一规划年出现未满足需求（只超出流域用水总量控制指标的不算），所以图上没有 *。")
    assert module.binding_note(used=[], binding=[]) == "未标 *：甲、乙没有可用的求解结果。"
