"""`scripts/render_version.py` 的 `tree_figures`：给了 since 只收这之后写出的图，since 为 None 时全收；`main/` 与根目录的图
归到 `figures/`，`extended/` 的归到 `extended/`。上一轮画的、这次没重画的图（包括已删脚本如 `plot_ind_*` 画的）也还在
`_indtree/results/figures/` 里，`render_version` 正常运行靠这个过滤不把它们拷进归档；`main()` 怎么传 since（正常运行传
画图前的时刻，`--skip-plots` 时传 None）不在本测试里。不求解，不需要 Gurobi。"""
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
    (figures / "main").mkdir()
    old = figures / "extended" / "ind_ed1_target_level.pdf"   # plot_ind_ed1 早先留下的，与本次画的 ED 图同在 extended/
    old_main = figures / "main" / "ind_fig1_joint_allocation.pdf"  # plot_ind_fig1 早先留下的，main/ 归到 figures/
    stale = figures / "fig2_constraint_response.pdf"          # 上一轮画的，这次没重画
    new = figures / "extended" / "ed_fig1_fleet_atlas.png"     # 本次画的
    new_root = figures / "fig1_water_footprint.pdf"            # 本次画的，根目录归到 figures/
    new_main = figures / "main" / "fig_new.png"                # 本次画的（留下的脚本眼下不往 main/ 存）
    for path, mtime in ((old, 1.0e9), (old_main, 1.1e9), (stale, 1.2e9),
                        (new, 1.7e9), (new_root, 1.7e9), (new_main, 1.7e9)):
        path.write_bytes(b"")
        os.utime(path, (mtime, mtime))
    assert sorted(render_version.tree_figures(tmp_path, since=1.5e9)) == [
        (new, "extended"), (new_root, "figures"), (new_main, "figures")]
    assert sorted(render_version.tree_figures(tmp_path, since=None)) == [
        (new, "extended"), (old, "extended"), (new_root, "figures"), (stale, "figures"),
        (new_main, "figures"), (old_main, "figures")]
