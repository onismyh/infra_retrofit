"""读 `inputs/` 的一张表：文件不在、缺列都在读入时报错，各 `_prepare_*` 共用。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_input_csv(path: Path, required: Iterable[str], *, not_found: str) -> pd.DataFrame:
    """读 `path`，核对 `required` 各列都在。

    文件不在时报 `FileNotFoundError`，文案为 `{path} not found` 接 `not_found`（怎么重建、哪个情景开关要它）；
    缺列时报 `ValueError`，列名按字典序列出。
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found{not_found}")
    table = pd.read_csv(path)
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    return table
