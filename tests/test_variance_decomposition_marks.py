"""`scripts/plot_ed_variance_decomposition.py` 的受约束流域标记（*）：只取登记表里带水约束的求解；没求解与旧格式
各自提示后跳过，都不当成"没有流域缺水"。不求解，不需要 Gurobi。"""
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
    """标记只取登记表里能求解、带水约束、枯水期口径的情景。原先写的两个 v9.1 情景名不在登记表里，结果读不到，
    图上一个 * 也没有。登记情景的结果写在它的 `tree` 下，那就是脚本读的 `_indtree`
    （`test_scenarios.test_bootstrap_root_is_the_registered_tree`）。"""
    module = _load(monkeypatch)
    registry = load_registry()
    assert module.BINDING_RUNS
    for name in module.BINDING_RUNS:
        spec = registry.get(name)  # 没登记或是抽象基底时报错
        assert spec.scenario.get("water_mode") == "grid_supply", name
        assert spec.scenario.get("water_season") == "dry", name


def test_binding_basins_tells_unsolved_from_old_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """三份结果：没求解、旧格式（缺空冷两列）、现格式。前两种各自提示后跳过，只有现格式的进标记与读到的情景；
    标记只取 `water_supply` 松弛大于 1 m3 的流域（节点编号末段是流域代码），流域上限的松弛不算。"""
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "BINDING_RUNS", {"UNSOLVED": "未求解", "OLD": "旧格式", "NEW": "现格式"})
    (tmp_path / "OLD").mkdir()
    pd.DataFrame({"plant_id": ["P1"], "year": [2030]}).to_csv(tmp_path / "OLD" / "plant_detail.csv", index=False)
    (tmp_path / "NEW").mkdir()
    pd.DataFrame({"plant_id": ["P1"], "year": [2030], "air_cooled_share": [0.0], "already_air_share": [0.0]}).to_csv(
        tmp_path / "NEW" / "plant_detail.csv", index=False)
    pd.DataFrame({
        "year": [2030, 2030, 2060, 2060],
        "constraint_type": ["water_supply", "water_supply", "water_supply", "water_basin_quota"],
        "node_id": ["WC_0101_0202_H", "WC_0303_0404_K", "WC_0505_0606_A", "J"],
        "slack_value": [5.0, 0.5, 2.0, 100.0],
    }).to_csv(tmp_path / "NEW" / "slack_detail.csv", index=False)

    assert module.binding_basins() == (["A", "H"], ["NEW"])
    skips = [line.strip() for line in capsys.readouterr().out.splitlines() if "[skip]" in line]
    assert len(skips) == 2
    assert skips[0].startswith("[skip] UNSOLVED: no solved result")
    assert skips[1].startswith("[skip] OLD: old format")
