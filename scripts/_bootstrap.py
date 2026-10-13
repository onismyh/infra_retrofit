"""脚本的路径引导：`ROOT` 是数据树 `_indtree/`，`REPO_ROOT` 是仓库根，`SRC` 是仓库根的 `src/`。

2026-09-28 起脚本只有仓库根 `scripts/` 一份（此前 `_indtree/scripts/` 是第二份，两份曾出现分叉）。
输入构建与诊断脚本用 `ROOT` 读写 `_indtree/inputs`、`_indtree/results`（v9.2 管网，`ST_` 系在这里求解），出图脚本读
`_indtree/inputs`、把图写到 `_indtree/results/figures/`；读入库的 `results/solved/` 结果、读写仓库根 `results/figures/` 图幅归档、
读仓库根 `data/ChinaMapTHT/` 底图的用 `REPO_ROOT`。
求解不经 `ROOT`：`python -m coal_retrofit` 的求解树取自情景登记的 `tree`
（`scenarios/st.toml`）。两者不一致时，脚本读写的就不是求解用的那套输入与结果（CLAUDE.md 二.6），
所以 `tests/test_scenarios.py` 核对 `ROOT` 就是登记情景的 `tree`。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).absolute().parents[1]
ROOT = REPO_ROOT / "_indtree"
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
