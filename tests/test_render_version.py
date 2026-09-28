"""`scripts/render_version.py` 只归档本次画出的图：`plot_ind_*` 等不由它跑的脚本留在
`_indtree/results/figures/` 的旧图不拷进新版本（`--skip-plots` 时全拷）。不求解，不需要 Gurobi。"""
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
    (figures / "main").mkdir(parents=True)
    (figures / "extended").mkdir()
    old = figures / "main" / "ind_fig1_joint_allocation.pdf"   # plot_ind_* 早先留下的
    new = figures / "extended" / "ed_fig1_fleet_atlas.png"     # 本次画的
    for path, mtime in ((old, 1.0e9), (new, 1.7e9)):
        path.write_bytes(b"")
        os.utime(path, (mtime, mtime))
    assert render_version.tree_figures(tmp_path, since=1.5e9) == [(new, "extended")]
    assert sorted(render_version.tree_figures(tmp_path, since=None)) == [(new, "extended"), (old, "figures")]
