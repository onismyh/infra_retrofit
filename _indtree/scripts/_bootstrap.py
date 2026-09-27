"""求解树的路径引导：`ROOT` 是本树根，`SRC` 是仓库根的 `src/`。

2026-09-22 起代码只有仓库根一份（此前树内有第二份 `src/`，两份代码曾出现分叉），
但输入、数据与结果仍按树分开：输入构建与诊断脚本用 `ROOT` 读写 `ROOT/inputs`。
所以 `ROOT` 必须是树根——指向仓库根会让本树的脚本悄悄改读改写仓库根的 v7 输入
（35 汇 / 923 边、无工业节点），即 CLAUDE.md 二.6 的跨版本混用。
2026-09-27 起求解不经 `ROOT`：`run_single.py` 是 `python -m coal_retrofit` 的薄壳，求解树取自
情景登记的 `tree`（仓库根 `scenarios/st.toml`），从哪棵树的 `scripts/` 启动都一样。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
REPO_ROOT = Path(__file__).absolute().parents[2]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
