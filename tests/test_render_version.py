"""`scripts/render_version.py` 的 `tree_figures`：只收 `figures/` 根目录的图（出图脚本都存在这里），给了 since 只收这之后
写出的，since 为 None 时全收；子目录 `main/`、`extended/` 里是已删脚本早先画的图，一律不收，`save_fig` 的临时文件
（"." 开头）也不收。上一轮画的、这次没重画的图也还在 `_indtree/results/figures/` 里，`render_version` 正常运行靠 since
过滤不把它们拷进归档；`main()` 怎么传 since（正常运行传画图前的时刻，`--skip-plots` 时传 None）不在本测试里。
`copy_results` 归档出图读的 `results/solved/`（表与工作簿都拷；`SOLVED_DIR` 取自出图读的 `plot_io.SOLVED_DIR`），不拷求解树
`results/` 下的结果。
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
    tmp = figures / ".fig4_industry_routes.tmp.pdf"              # 出图中断留下的临时文件
    for path, mtime in ((old_ext, 1.7e9), (old_main, 1.7e9), (stale, 1.2e9), (new, 1.7e9), (new_en, 1.7e9),
                        (tmp, 1.7e9)):
        path.write_bytes(b"")
        os.utime(path, (mtime, mtime))
    assert sorted(render_version.tree_figures(tmp_path, since=1.5e9)) == [new, new_en]
    assert sorted(render_version.tree_figures(tmp_path, since=None)) == [stale, new, new_en]
    assert render_version.tree_figures(tmp_path / "missing", since=None) == []


def test_copy_results_archives_solved_results_not_tree_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    render_version = _load(monkeypatch)
    tree, solved, dest = tmp_path / "tree", tmp_path / "solved", tmp_path / "v1"
    (tree / "inputs").mkdir(parents=True)
    (tree / "inputs" / "storage_hubs.csv").write_text("storage_hub_id\n", encoding="utf-8")
    (tree / "inputs" / "pipeline_nodes_300km_average.csv").write_text("node_id\n", encoding="utf-8")  # 汇变体的表也拷
    (tree / "results" / "ST_LOCAL").mkdir(parents=True)                 # 本机求解树的结果，出图不读
    (tree / "results" / "ST_LOCAL.json").write_text("{}", encoding="utf-8")
    (solved / "ST_BASE").mkdir(parents=True)
    (solved / "ST_BASE.json").write_text("{}", encoding="utf-8")
    (solved / "ST_BASE" / "sources.csv").write_text("year\n", encoding="utf-8")
    (solved / "ST_BASE" / "ccs_results.xlsx").write_bytes(b"")
    (solved / "ST_ORPHAN.json").write_text("{}", encoding="utf-8")       # 没有同名目录，不收
    assert render_version.copy_results(tree, solved, dest) == ["ST_BASE"]
    data = dest / "data"
    assert sorted(p.name for p in (data / "ST_BASE").iterdir()) == ["ccs_results.xlsx", "sources.csv"]
    assert (data / "ST_BASE.json").is_file() and (data / "storage_hubs.csv").is_file()
    assert (data / "pipeline_nodes_300km_average.csv").is_file() and not (data / "pipeline_nodes.csv").exists()
    assert not (data / "ST_LOCAL.json").exists() and not (data / "ST_ORPHAN.json").exists()


def test_prepare_dest_refuses_a_populated_version_directory_unless_forced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    render_version = _load(monkeypatch)
    dest = tmp_path / "v1"
    render_version.prepare_dest(dest, force=False)          # 新目录：建好就行
    assert dest.is_dir()
    (dest / "figures").mkdir()
    render_version.prepare_dest(dest, force=False)          # 空的 figures/ 不算上一轮的归档
    (dest / "figures" / "fig1.png").write_bytes(b"")
    (dest / "render.log").write_text("old", encoding="utf-8")
    (dest / "README.md").write_text("人写的", encoding="utf-8")
    with pytest.raises(SystemExit, match="figures, render.log"):
        render_version.prepare_dest(dest, force=False)
    render_version.prepare_dest(dest, force=True)
    assert not (dest / "figures").exists() and not (dest / "render.log").exists()
    assert (dest / "README.md").read_text(encoding="utf-8") == "人写的"
