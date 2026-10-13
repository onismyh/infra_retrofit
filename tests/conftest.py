"""各测试文件共用的夹具。"""
from __future__ import annotations

from pathlib import Path

import pytest

@pytest.fixture
def ascii_tmp_path(tmp_path: Path) -> Path:
    """*tmp_path* 含非 ASCII 字符（如 Windows 用户名是中文）时跳过：Gurobi 在这样的路径下写不了文件，热启动的 .sol
    改写到系统临时目录（`run_controls.sol_path`），按 *tmp_path* 断言 .sol 位置的测试不成立；系统临时目录也含非 ASCII
    字符时 `sol_path` 直接拒绝（退出码 1），断言退出码 3 的测试也不成立。"""
    if not str(tmp_path).isascii():
        pytest.skip(f"tmp_path 含非 ASCII 字符（{tmp_path}），.sol 不写在它下面")
    return tmp_path
