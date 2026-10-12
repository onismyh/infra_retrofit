"""出图脚本的命令行、读表与对账自检：`--lang` / `--scenario`，读输入表与求解结果（核模型分段号），图上要画的量与模型记的量
对不上就报错。不依赖 matplotlib：`render_version.py` 归档时从这里取 `SOLVED_DIR`。
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT, ROOT  # 数据树 _indtree/
from coal_retrofit.segment import check_segment

INPUTS_DIR = ROOT / "inputs"
# 结果图读入库的 `results/solved/<结果名>(.json)`（作者本机求解后拷进来，见该目录的 README.md）；`--scenario` 给路径时
# 读那里（例如本机求解树的 `_indtree/results/<结果名>`）。
SOLVED_DIR = REPO_ROOT / "results" / "solved"
# 结果图缺省画的主情景（scenarios/st.toml）。只画单个情景：ST_ 系的差值只能按可证区间报告（CLAUDE.md 二.2）。
MAIN_SCENARIO = "ST_WA_cwatm_126_dry_oq"
LANGS = ("zh", "en")         # 中文版、英文版（字体见 `plot_style.FONTS`）


def figure_cli(doc: str | None, *, scenario: bool = True) -> argparse.ArgumentParser:
    """出图脚本的统一参数：--lang zh|en|both（缺省两版都出），结果图另有 --scenario。"""
    lines = (doc or "").strip().splitlines()
    parser = argparse.ArgumentParser(description=lines[0] if lines else None,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=("zh", "en", "both"), default="both",
                        help="出中文版、英文版或两版（缺省 both；英文版文件名加 _en）")
    if scenario:
        parser.add_argument("--scenario", default=MAIN_SCENARIO,
                            help=f"结果名，读 results/solved/<结果名>/ 的结果表（缺省 {MAIN_SCENARIO}）；"
                                 "给路径（含 /）时读该目录，如 _indtree/results/ST_BASE")
    return parser


def langs(choice: str) -> tuple[str, ...]:
    return LANGS if choice == "both" else (choice,)


def result_dir(scenario: str) -> Path:
    """结果表目录：结果名在 `SOLVED_DIR` 下找，含路径分隔符的按路径（相对仓库根或绝对路径）。"""
    if "/" in scenario or "\\" in scenario:
        path = Path(scenario)
        return path if path.is_absolute() else REPO_ROOT / path
    return SOLVED_DIR / scenario


def read_result(scenario: str, table: str) -> pd.DataFrame:
    """读 `<结果目录>/<表>.csv`（`result_dir`）；同名 result.json 的模型分段号须是当前代码的（`read_result_json`）。"""
    read_result_json(scenario)
    path = result_dir(scenario) / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在：结果目录里缺这张表")
    return pd.read_csv(path, encoding="utf-8-sig")


@lru_cache(maxsize=None)
def read_result_json(scenario: str) -> dict:
    """读结果目录旁的 `<结果名>.json`，核模型分段号（不是当前代码的就抛 `StaleResultError`），打印求解时的提交号。"""
    folder = result_dir(scenario)
    path = folder.parent / f"{folder.name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在：先在本机求解 python -m coal_retrofit run {folder.name}，"
                                "再把结果拷进 results/solved/（见该目录的 README.md）")
    with path.open(encoding="utf-8") as f:
        result = json.load(f)
    check_segment(result, path)
    code = (result.get("resolved") or {}).get("code") or {}
    print(f"  {folder.name}：求解提交 {code.get('commit') or '未记录'}"
          f"{'（有未提交的改动）' if code.get('dirty') else ''}")
    return result


def read_input(name: str) -> pd.DataFrame:
    """读 `_indtree/inputs/<name>.csv`（列名去掉 BOM）。"""
    frame = pd.read_csv(INPUTS_DIR / f"{name}.csv", encoding="utf-8-sig")
    return frame


def check_close(what: str, got, expected, *, atol: float = 1e-3, rtol: float = 1e-6) -> None:
    """出图前的自检：图上要画的量与模型记的量对不上就报错，不出图。"""
    got_arr = np.asarray(got, dtype=float)
    exp_arr = np.asarray(expected, dtype=float)
    if got_arr.shape != exp_arr.shape or not np.allclose(got_arr, exp_arr, rtol=rtol, atol=atol):
        diff = np.nanmax(np.abs(got_arr - exp_arr)) if got_arr.shape == exp_arr.shape else float("nan")
        raise ValueError(f"自检不通过：{what}（最大偏差 {diff:.6g}），不出图")
