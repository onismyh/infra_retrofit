"""把仓库根的 `src/` 放到 sys.path 最前面。

求解树（`_indtree/`）只存放输入与结果，代码只有仓库根一份（2026-09-22 起；此前树内
有第二份 `src/`，两份代码曾出现分叉）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
