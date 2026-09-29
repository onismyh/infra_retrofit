"""`scripts/render_version.py` 的 `tree_figures`：只收 `figures/` 根目录的图（出图脚本都存在这里），给了 since 只收这之后
写出的，since 为 None 时全收；子目录 `main/`、`extended/` 里是已删脚本早先画的图，一律不收。上一轮画的、这次没重画的图
也还在 `_indtree/results/figures/` 里，`render_version` 正常运行靠 since 过滤不把它们拷进归档；`main()` 怎么传 since
（正常运行传画图前的时刻，`--skip-plots` 时传 None）不在本测试里。不求解，不需要 Gurobi。"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    alias = "render_version_under_test"
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / "render_version.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    return module


def test_tree_figures_keeps_only_root_figures_written_since(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    render_version = _load(monkeypatch)
    figures = tmp_path / "results" / "figures"
    (figures / "extended").mkdir(parents=True)
    (figures / "main").mkdir()
    old_ext = figures / "extended" / "ed_fig1_fleet_atlas.png"   # 已删脚本早先画的，子目录里的都不收
    old_main = figures / "main" / "ind_fig1_joint_allocation.pdf"
    stale = figures / "fig2_candidate_network.pdf"               # 上一轮画的，这次没重画
    new = figures / "fig3_power_pathways.png"                    # 本次画的
    new_en = figures / "fig3_power_pathways_en.pdf"              # 本次画的英文版
    for path, mtime in ((old_ext, 1.7e9), (old_main, 1.7e9), (stale, 1.2e9), (new, 1.7e9), (new_en, 1.7e9)):
        path.write_bytes(b"")
        os.utime(path, (mtime, mtime))
    assert sorted(render_version.tree_figures(tmp_path, since=1.5e9)) == [new, new_en]
    assert sorted(render_version.tree_figures(tmp_path, since=None)) == [stale, new, new_en]
    assert render_version.tree_figures(tmp_path / "missing", since=None) == []
