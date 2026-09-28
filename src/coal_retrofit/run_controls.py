"""运行控制：热启动第 1 步与 .sol 的位置、兼容的 `COAL_RETROFIT_*` 环境变量、求解时的代码版本。由 `runner.run` 调用。

热启动（实现说明 §9.7）2026-09-28 之前要手工跑两次：第 1 步设 COAL_RETROFIT_LP_RELAX 与 COAL_RETROFIT_WRITE_SOL、
`--time-limit 1800`，第 2 步设 COAL_RETROFIT_START_SOL。情景 `warm_start = "lp_relax"` 时运行器在同一进程里照做：
两步各读一次输入、各建一次模型，与手工跑两次是同一对求解。第 1 步只写 .sol，不写结果表，它的状态、目标函数、
用时与 .sol 的摘要记进最终 result.json 的 `resolved.warm_start`。
"""
from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .optimization._shared import PreparedInputs, SolveState
from .optimization.data_prep import prepare_inputs
from .optimization.scenario import OptimizationAssumptions, OptimizationScenario
from .optimization.solver import SolveControls, _solve_joint_multi_period
from .paths import ProjectPaths
from .scenarios import REPO_ROOT, shown_path

logger = logging.getLogger(__name__)

# 运行器自己做热启动两步时，这两个环境变量再设就是两套办法叠在一起：直接拒绝。
WARM_START_ENV = ("COAL_RETROFIT_LP_RELAX", "COAL_RETROFIT_START_SOL")


class RunError(RuntimeError):
    """求解之前就拒绝：结果已存在、热启动与环境变量冲突、.sol 没有可用的 ASCII 路径。命令行退出码 1。"""


class WarmStartFailed(RuntimeError):
    """热启动第 1 步没有解（状态不可接受或没写出 .sol）：不跑第 2 步，也不写结果。命令行退出码 3。"""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__(
            f"热启动第 1 步（LP 松弛）没有可用的解：状态 {record.get('status')}，时限 {record.get('time_limit')} s"
        )
        self.record = record


def initial_state(prepared: PreparedInputs) -> SolveState:
    """求解开始时的管网与封存状态：没有新增管道，封存汇是满的。"""
    return SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )


def env_controls() -> SolveControls:
    """兼容的环境变量 → 求解开关。非空即设，与此前求解器自己读时相同。"""
    return SolveControls(
        relax=bool(os.environ.get("COAL_RETROFIT_LP_RELAX")),
        start_sol=_env_path("COAL_RETROFIT_START_SOL"),
        write_sol=_env_path("COAL_RETROFIT_WRITE_SOL"),
        log_incumbents=bool(os.environ.get("COAL_RETROFIT_LOG_INCUMBENTS")),
    )


def _env_path(key: str) -> Path | None:
    value = os.environ.get(key)
    return Path(value) if value else None


def check_warm_start_env() -> None:
    """情景 `warm_start = "lp_relax"` 时不许再设 COAL_RETROFIT_LP_RELAX / START_SOL。"""
    given = [key for key in WARM_START_ENV if os.environ.get(key)]
    if given:
        raise RunError(
            f"情景 warm_start = \"lp_relax\" 由运行器自己做热启动两步，不能再设 {'、'.join(given)}；去掉环境变量，"
            f"或用 --set scenario.warm_start=none --as <结果名> 照旧手工跑"
        )


def sol_path(tree: Path, name: str, sol_dir: Path | None) -> Path:
    """热启动第 1 步的 .sol 写在哪。Gurobi 在中文路径下写文件会失败，所以路径必须是 ASCII。

    给了 *sol_dir* 就写在那里，不是 ASCII 就报错。没给时写 `<树>/results/<结果名>.lp.sol`；这个路径含非 ASCII 字符时，
    改写到系统临时目录下新建的子目录里；临时目录也不是 ASCII，就报错并提示用 `--sol-dir`。
    """
    file_name = f"{name}.lp.sol"
    if sol_dir is not None:
        path = Path(sol_dir).absolute() / file_name
        if not str(path).isascii():
            raise RunError(f"--sol-dir {sol_dir}：路径含非 ASCII 字符，Gurobi 写不了 .sol")
        return path
    path = (Path(tree) / "results" / file_name).absolute()
    if str(path).isascii():
        return path
    temp_root = Path(tempfile.gettempdir()).absolute()
    if not str(temp_root).isascii():
        raise RunError(
            f"{path} 与系统临时目录 {temp_root} 都含非 ASCII 字符，Gurobi 写不了 .sol；用 --sol-dir 指定一个 ASCII 目录"
        )
    path = Path(tempfile.mkdtemp(prefix="coal_retrofit_", dir=temp_root)) / file_name
    logger.warning("结果目录的路径含非 ASCII 字符，热启动的 .sol 改写到 %s", path)
    return path


def lp_relaxation_start(
    paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions, sol: Path,
) -> dict[str, Any]:
    """热启动第 1 步：读输入、建模，把全部整数变量松弛后求解，解写到 *sol*；不写结果表。

    时限用 `warm_start_time_limit`，其余参数与第 2 步相同。返回记进 `resolved.warm_start` 的内容；
    没有解时抛 `WarmStartFailed`。
    """
    t0 = time.time()
    sol.parent.mkdir(parents=True, exist_ok=True)
    sol.unlink(missing_ok=True)  # 上次留下的 .sol 不能被第 2 步当成这次的
    step = replace(scenario, solver_time_limit=scenario.warm_start_time_limit)
    prepared = prepare_inputs(paths, step, assumptions)
    solution = _solve_joint_multi_period(
        prepared, step, assumptions, step.planning_years, initial_state(prepared),
        SolveControls(relax=True, write_sol=sol),
    )
    quality = solution["solver_quality"]
    record: dict[str, Any] = {
        "method": "lp_relax",
        "time_limit": step.solver_time_limit,
        "status": quality.get("status"),
        "objective_cny": float(solution["objective_cny"]),
        "runtime_seconds": quality.get("runtime_seconds"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "sol": shown_path(sol),
        "sol_sha256": hashlib.sha256(sol.read_bytes()).hexdigest()[:12] if sol.exists() else None,
    }
    del prepared, solution, quality
    gc.collect()  # 第 1 步的模型（ST_ 常驻数 GiB）在第 2 步建模之前放掉
    if not math.isfinite(record["objective_cny"]) or record["sol_sha256"] is None:
        raise WarmStartFailed(record)
    logger.info("热启动第 1 步：%s，LP 目标函数 %.6e 元，.sol 写在 %s", record["status"], record["objective_cny"], sol)
    return record


def code_state() -> dict[str, Any]:
    """求解时的代码版本：`git rev-parse HEAD`，以及已跟踪的文件有没有未提交的改动（未跟踪的文件不算）。

    包所在目录不是 git 仓库的顶层（没有 git，或不是以可编辑方式从本仓库安装）时两项都记 None 并告警：
    这次结果对应哪个提交核不了。
    """
    try:
        prefix, head = _git("rev-parse", "--show-prefix", "HEAD").splitlines()
        if prefix.strip():
            raise ValueError(f"{REPO_ROOT} 在仓库里的 {prefix.strip()}，不是仓库顶层")
        dirty = bool(_git("--no-optional-locks", "status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("记不了求解时的提交号（%s）：resolved.code 记空，这次结果对应哪个提交核不了", exc)
        return {"commit": None, "dirty": None}
    return {"commit": head.strip(), "dirty": dirty}


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, check=True, encoding="utf-8", timeout=60,
    )
    return completed.stdout
