"""各测试文件共用的夹具。"""
from __future__ import annotations

from pathlib import Path

import pytest

# 求解相关的兼容环境变量（前四个由 `run_controls.env_controls` 读，后两个是 `runner.SOLVER_ENV_FIELDS`）。本机 shell
# 里留着的值（如手工热启动两步时设的 COAL_RETROFIT_LP_RELAX）会改掉测试里的求解与参数，所以每个测试先清掉，要用的在
# 测试里再设。点源库目录、TIMES 表格这类指向数据的环境变量不动。
SOLVER_ENV = ("COAL_RETROFIT_LP_RELAX", "COAL_RETROFIT_START_SOL", "COAL_RETROFIT_WRITE_SOL",
              "COAL_RETROFIT_LOG_INCUMBENTS", "COAL_RETROFIT_GUROBI_SEED", "COAL_RETROFIT_MIPFOCUS")


@pytest.fixture(autouse=True)
def _clean_solver_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in SOLVER_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def ascii_tmp_path(tmp_path: Path) -> Path:
    """*tmp_path* 含非 ASCII 字符（如 Windows 用户名是中文）时跳过：Gurobi 在这样的路径下写不了文件，热启动的 .sol
    改写到系统临时目录（`run_controls.sol_path`），按 *tmp_path* 断言 .sol 位置的测试不成立。"""
    if not str(tmp_path).isascii():
        pytest.skip(f"tmp_path 含非 ASCII 字符（{tmp_path}），.sol 不写在它下面")
    return tmp_path
