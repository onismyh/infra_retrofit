"""旧求解树（v9）的路径引导：`ROOT` 是本树根，`SRC` 是仓库根的 `src/`。

树内 `src/` 已在 892c877（2026-09-22）删除，本树从此是冻结快照。出图、诊断、校验脚本
照常可用：它们用到的 builders.storage、constants、optimization.emissions、paths、spatial
与删除前逐字节相同，煤电侧排放参数的默认值也没变。求解与重建输入则不行——仓库根已是
另一个模型（工业强制开启、部门碳目标、新成本口径与期末残值）：`run_*` 复现不了 v9 的解，
`build_*` 会用新的管网构建器覆盖本树冻结的 `inputs/`。这两类入口在这里直接拒绝；
需要复现时，在删除前的提交上另开一份工作副本（那里树内 `src/` 还在）：

    git worktree add ../infra_v9 892c877^
    cd ../infra_v9/_v9tree && python scripts/run_single.py <name> --threads 8
"""
from __future__ import annotations

import sys
from pathlib import Path

import __main__

ROOT = Path(__file__).absolute().parents[1]
REPO_ROOT = Path(__file__).absolute().parents[2]
SRC = REPO_ROOT / "src"

_ENTRY = Path(getattr(__main__, "__file__", None) or "").name
if _ENTRY.startswith(("run_", "build_")):
    raise SystemExit(
        f"{ROOT.name} 是冻结快照（树内 src/ 已于 892c877 删除），{_ENTRY} 不能跑在仓库根的新模型上。"
        "复现请用删除前的提交：git worktree add ../infra_v9 892c877^"
    )

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
