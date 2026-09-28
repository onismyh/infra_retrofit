"""求解一个登记情景：`python -m coal_retrofit run` 的兼容入口。

    python scripts/run_single.py [<name>] [--threads 8] [--time-limit S] [--mip-gap G]
    python scripts/run_single.py --list

情景名要写在最前面，不写时跑 `ST_BASE`（原来的缺省）。原来 argparse 也接受情景名写在选项后面，
现在那样写会报 unrecognized arguments 退出，不会跑错情景。`--list` 照旧每行一个情景名（带求解树与说明的
列表用 `python -m coal_retrofit list`）。

情景登记在仓库根 `scenarios/*.toml`（2026-09-27 之前是本文件里的字典）。求解树由情景的 `tree` 决定
（`ST_` 系是 `_indtree/`）。其余选项（`--set`、`--as`、
`--tree`、`--force`、`--sol-dir`）照传给 `coal_retrofit.cli`，见那里的说明。2026-09-28 起已有同名结果时求解之前就拒绝，
要覆盖加 `--force`；`ST_` 系的热启动两步由运行器自动做。

出图脚本 `from run_single import EXPERIMENTS` 读登记表，形状与原来的字典相同：
{名: (scenario 覆盖, assumptions 覆盖)}，只含能求解的情景。
"""
from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  （把仓库根的 src/ 放进 sys.path）

from coal_retrofit.cli import main as cli_main
from coal_retrofit.scenarios import experiments_dict, load_registry

EXPERIMENTS: dict[str, tuple[dict, dict]] = experiments_dict(load_registry())


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if "--list" in args:
        print("\n".join(EXPERIMENTS))
        return 0
    if not args or args[0].startswith("-"):
        args = ["ST_BASE", *args]  # 原来的缺省情景
    return cli_main(["run", *args])


if __name__ == "__main__":
    sys.exit(main())
