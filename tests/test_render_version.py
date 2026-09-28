"""`scripts/render_version.py` 的 `tree_figures`：给了 since 只收这之后写出的图，since 为 None 时全收。
`plot_ind_*` 也往 `_indtree/results/figures/` 存图，上一轮画的、这次没重画的图也还在那里，`render_version` 正常运行
靠这个过滤不把它们拷进归档；`main()` 怎么传 since（正常运行传画图前的时刻，`--skip-plots` 时传 None）不在本测试里。
不求解，不需要 Gurobi。"""
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


def test_tree_figures_keeps_only_figures_written_since(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    render_version = _load(monkeypatch)
    figures = tmp_path / "results" / "figures"
    (figures / "extended").mkdir(parents=True)
    old = figures / "extended" / "ind_ed1_target_level.pdf"   # plot_ind_ed1 早先留下的，与本次画的 ED 图同在 extended/
    stale = figures / "fig2_constraint_response.pdf"          # 上一轮画的，这次没重画
    new = figures / "extended" / "ed_fig1_fleet_atlas.png"     # 本次画的
    for path, mtime in ((old, 1.0e9), (stale, 1.2e9), (new, 1.7e9)):
        path.write_bytes(b"")
        os.utime(path, (mtime, mtime))
    assert render_version.tree_figures(tmp_path, since=1.5e9) == [(new, "extended")]
    assert sorted(render_version.tree_figures(tmp_path, since=None)) == [
        (new, "extended"), (old, "extended"), (stale, "figures")]
