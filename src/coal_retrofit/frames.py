"""DataFrame 逐行遍历：`iter_rows(frame)` 就是 `frame.itertuples(index=False)`，只是把行标成 Any。

pandas-stubs 把 itertuples 每个字段都标成 Scalar 联合类型（含日期、字串、复数），数值列上的 `float(row.x)` 过不了
mypy；列的类型由各输入表读入时的检查保证，静态类型帮不上忙。运行时与 itertuples 完全相同。
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd


def iter_rows(frame: pd.DataFrame) -> Iterator[Any]:
    """逐行的具名元组（`frame.itertuples(index=False)`），字段按列名取。"""
    return frame.itertuples(index=False)
