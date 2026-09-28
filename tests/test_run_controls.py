"""运行控制（`coal_retrofit.run_controls`）与运行器的两步编排：不求解，不需要 Gurobi。

求解用桩函数代替：`runner.lp_relaxation_start`、`runner.solve` 记下收到的参数，返回最小的结果。
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from coal_retrofit import cli, run_controls, runner
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import SolveControls
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.run_controls import RunError, WarmStartFailed
from coal_retrofit.scenarios import ScenarioSpec

ENV = ("COAL_RETROFIT_LP_RELAX", "COAL_RETROFIT_START_SOL", "COAL_RETROFIT_WRITE_SOL", "COAL_RETROFIT_LOG_INCUMBENTS",
       "COAL_RETROFIT_GUROBI_SEED", "COAL_RETROFIT_MIPFOCUS")

REGISTRY = """
[WARM]
tree = "tree"

[WARM.scenario]
warm_start = "lp_relax"
warm_start_time_limit = 77

[PLAIN]
tree = "tree"
"""


# --- .sol 的位置 -------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("ascii_tmp_path")
def test_sol_path_defaults_to_the_results_directory(tmp_path) -> None:
    assert run_controls.sol_path(tmp_path / "tree", "ST_BASE", None) == tmp_path / "tree" / "results" / "ST_BASE.lp.sol"
    assert run_controls.sol_path(tmp_path / "tree", "ST_BASE", tmp_path / "sols") == tmp_path / "sols" / "ST_BASE.lp.sol"


@pytest.mark.usefixtures("ascii_tmp_path")
def test_sol_path_moves_to_the_temp_directory_when_the_tree_is_not_ascii(tmp_path, monkeypatch) -> None:
    """Gurobi 在中文路径下写不了文件：结果目录不是 ASCII 时改写到系统临时目录下新建的子目录。"""
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_root))
    path = run_controls.sol_path(tmp_path / "求解树", "ST_BASE", None)
    assert path.name == "ST_BASE.lp.sol" and path.parent.parent == temp_root and path.parent.is_dir()
    assert path.parent.name.startswith("coal_retrofit_")


def test_sol_path_refuses_non_ascii_paths(tmp_path, monkeypatch) -> None:
    with pytest.raises(RunError, match="路径含非 ASCII 字符"):
        run_controls.sol_path(tmp_path / "tree", "ST_BASE", tmp_path / "临时")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "临时"))
    with pytest.raises(RunError, match="用 --sol-dir 指定一个 ASCII 目录"):
        run_controls.sol_path(tmp_path / "求解树", "ST_BASE", None)


# --- 求解时的提交号 --------------------------------------------------------------------------------------------


def _fake_git(prefix: str, status: str):
    def fake(*args: str) -> str:
        if args[0] == "rev-parse":
            return f"{prefix}\n{'c' * 40}\n"
        return status
    return fake


def _failing_git(exc: Exception):
    def fake(*args: str) -> str:
        raise exc
    return fake


@pytest.mark.parametrize(("status", "dirty"), [("", False), (" M src/coal_retrofit/runner.py\n", True)])
def test_code_state_records_head_and_tracked_changes(monkeypatch, status: str, dirty: bool) -> None:
    monkeypatch.setattr(run_controls, "_git", _fake_git("", status))
    assert run_controls.code_state() == {"commit": "c" * 40, "dirty": dirty}


@pytest.mark.parametrize(
    "fake",
    [
        _fake_git("src/", ""),  # 包所在目录不是仓库顶层：记下的提交号对应不到这份代码
        _failing_git(subprocess.CalledProcessError(128, ["git"])),  # 不是 git 仓库
        _failing_git(FileNotFoundError("git")),  # 没装 git
    ],
)
def test_code_state_is_empty_when_git_cannot_tell(monkeypatch, caplog, fake) -> None:
    monkeypatch.setattr(run_controls, "_git", fake)
    with caplog.at_level(logging.WARNING):
        assert run_controls.code_state() == {"commit": None, "dirty": None}
    assert "记不了求解时的提交号" in caplog.text


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")
def test_code_state_on_a_real_repository(tmp_path, monkeypatch) -> None:
    """未跟踪的文件不算改动；包目录在仓库的子目录里时不记提交号。"""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com",
             "-c", "commit.gpgsign=false", *args], capture_output=True, check=True, encoding="utf-8",
        ).stdout

    git("init", "-q")
    (tmp_path / "a.txt").write_text("1", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-q", "-m", "init")
    head = git("rev-parse", "HEAD").strip()
    monkeypatch.setattr(run_controls, "REPO_ROOT", tmp_path)
    assert run_controls.code_state() == {"commit": head, "dirty": False}
    (tmp_path / "untracked.txt").write_text("x", encoding="utf-8")
    assert run_controls.code_state() == {"commit": head, "dirty": False}
    (tmp_path / "a.txt").write_text("2", encoding="utf-8")
    assert run_controls.code_state() == {"commit": head, "dirty": True}
    (tmp_path / "sub").mkdir()
    monkeypatch.setattr(run_controls, "REPO_ROOT", tmp_path / "sub")
    assert run_controls.code_state() == {"commit": None, "dirty": None}


# --- 兼容的环境变量 --------------------------------------------------------------------------------------------


def test_env_controls_reads_the_compatible_switches(monkeypatch) -> None:
    assert run_controls.env_controls() == SolveControls()
    monkeypatch.setenv("COAL_RETROFIT_LP_RELAX", "1")
    monkeypatch.setenv("COAL_RETROFIT_START_SOL", "a.sol")
    monkeypatch.setenv("COAL_RETROFIT_WRITE_SOL", "b.sol")
    monkeypatch.setenv("COAL_RETROFIT_LOG_INCUMBENTS", "1")
    assert run_controls.env_controls() == SolveControls(True, Path("a.sol"), Path("b.sol"), True)
    for key in ENV[:4]:
        monkeypatch.setenv(key, "")  # 空值与未设相同
    assert run_controls.env_controls() == SolveControls()


@pytest.mark.parametrize("key", ["COAL_RETROFIT_LP_RELAX", "COAL_RETROFIT_START_SOL"])
def test_warm_start_scenarios_refuse_the_manual_switches(monkeypatch, key: str) -> None:
    monkeypatch.setenv("COAL_RETROFIT_WRITE_SOL", "b.sol")  # 只决定第 2 步写不写 .sol，可以设
    run_controls.check_warm_start_env()
    monkeypatch.setenv(key, "x")
    with pytest.raises(RunError, match=f"不能再设 {key}"):
        run_controls.check_warm_start_env()


# --- 热启动第 1 步 ---------------------------------------------------------------------------------------------


def _step_one_stubs(monkeypatch, *, objective: float, write: bool) -> list:
    calls: list = []
    prepared = SimpleNamespace(network=SimpleNamespace(edges=[0, 1]),
                               storages=pd.DataFrame({"available_capacity_mt": [5.0]}))

    def prepare(paths, scenario, assumptions):
        calls.append(("prepare", scenario))
        return prepared

    def solve(prepared_, scenario, assumptions, years, state, controls):
        calls.append(("solve", scenario, controls))
        if write:
            controls.write_sol.write_text("x", encoding="utf-8")
        return {"objective_cny": objective, "solver_quality": {"status": "optimal", "runtime_seconds": 1.5}}

    monkeypatch.setattr(run_controls, "prepare_inputs", prepare)
    monkeypatch.setattr(run_controls, "_solve_joint_multi_period", solve)
    return calls


SCENARIO = OptimizationScenario(experiment_id="T", description="t", warm_start="lp_relax", warm_start_time_limit=123)


def test_step_one_relaxes_writes_the_sol_and_uses_its_own_time_limit(tmp_path, monkeypatch) -> None:
    calls = _step_one_stubs(monkeypatch, objective=1.0e9, write=True)
    sol = tmp_path / "results" / "T.lp.sol"
    record = run_controls.lp_relaxation_start(ProjectPaths(tmp_path), SCENARIO, OptimizationAssumptions(), sol)
    assert [call[0] for call in calls] == ["prepare", "solve"]
    assert all(call[1].solver_time_limit == 123 for call in calls)
    assert calls[1][2] == SolveControls(relax=True, write_sol=sol)
    assert record["method"] == "lp_relax" and record["time_limit"] == 123 and record["status"] == "optimal"
    assert record["objective_cny"] == 1.0e9 and record["sol_sha256"] == hashlib.sha256(b"x").hexdigest()[:12]


@pytest.mark.parametrize(("objective", "write"), [(1.0e9, False), (float("nan"), True)])
def test_step_one_without_a_usable_solution_fails(tmp_path, monkeypatch, objective: float, write: bool) -> None:
    """没写出 .sol，或目标函数不是有限值：抛 WarmStartFailed。上次留下的 .sol 先删掉，不会被第 2 步当成这次的。"""
    _step_one_stubs(monkeypatch, objective=objective, write=write)
    sol = tmp_path / "results" / "T.lp.sol"
    sol.parent.mkdir()
    sol.write_text("stale", encoding="utf-8")
    with pytest.raises(WarmStartFailed, match="热启动第 1 步（LP 松弛）没有可用的解：状态 optimal，时限 123 s"):
        run_controls.lp_relaxation_start(ProjectPaths(tmp_path), SCENARIO, OptimizationAssumptions(), sol)
    if write:
        assert sol.read_text(encoding="utf-8") == "x"
    else:
        assert not sol.exists()


# --- 运行器：两步编排与求解之前的拒绝 --------------------------------------------------------------------------------


@pytest.fixture
def cli_run(tmp_path, monkeypatch, capsys):
    """登记表两个情景（WARM 热启动、PLAIN 不热启动），求解树只有空的 inputs/；两步都换成桩。
    返回 (跑一次命令行的函数, 桩收到的调用列表, 结果目录)。"""
    registry = tmp_path / "scenarios"
    registry.mkdir()
    (registry / "t.toml").write_text(REGISTRY, encoding="utf-8")
    (tmp_path / "tree" / "inputs").mkdir(parents=True)
    calls: list = []
    record = {"method": "lp_relax", "status": "optimal"}
    state = {"objective": 1.0e9, "fail_step_one": False}

    def step_one(paths, scenario, assumptions, sol):
        calls.append(("step1", sol))
        if state["fail_step_one"]:
            raise WarmStartFailed({"status": "INFEASIBLE", "time_limit": scenario.warm_start_time_limit})
        return record

    def solve(paths, name, scenario, assumptions, controls=None):
        calls.append(("solve", controls))
        return {"name": name, "global_objective_cny": state["objective"], "solver_quality": {"status": "optimal"},
                "years": {}, "planning_years": list(scenario.planning_years), "elapsed_seconds": 0.0}

    monkeypatch.setattr(runner, "lp_relaxation_start", step_one)
    monkeypatch.setattr(runner, "solve", solve)

    def run(*argv: str) -> tuple[int, str]:
        code = cli.main(["--registry", str(registry), "run", *argv, "--threads", "1", "--time-limit", "60"])
        return code, capsys.readouterr().err

    return run, calls, state, tmp_path / "tree" / "results"


@pytest.mark.usefixtures("ascii_tmp_path")
def test_warm_start_runs_both_steps_and_records_step_one(cli_run, monkeypatch, tmp_path) -> None:
    run, calls, _, results = cli_run
    monkeypatch.setenv("COAL_RETROFIT_WRITE_SOL", str(tmp_path / "w.sol"))  # 兼容开关照旧作用于第 2 步
    monkeypatch.setenv("COAL_RETROFIT_LOG_INCUMBENTS", "1")
    monkeypatch.setattr(runner, "code_state", lambda: {"commit": "f" * 40, "dirty": True})
    assert run("WARM") == (0, "")
    sol = results / "WARM.lp.sol"
    assert calls == [("step1", sol), ("solve", SolveControls(start_sol=sol, write_sol=tmp_path / "w.sol",
                                                                log_incumbents=True))]
    resolved = json.loads((results / "WARM.json").read_text(encoding="utf-8"))["resolved"]
    assert resolved["warm_start"] == {"method": "lp_relax", "status": "optimal"}
    assert resolved["scenario"]["warm_start"] == "lp_relax" and resolved["scenario"]["warm_start_time_limit"] == 77
    assert resolved["options"] == {"threads": 1, "time_limit": 60, "mip_gap": None, "sol_dir": None, "force": False}
    assert resolved["code"] == {"commit": "f" * 40, "dirty": True}  # 原样记 code_state() 的返回值
    assert resolved["input_files"]["plants"] == "inputs/plants.csv"


@pytest.mark.usefixtures("ascii_tmp_path")
def test_sol_dir_moves_the_step_one_sol(cli_run, tmp_path) -> None:
    run, calls, _, results = cli_run
    assert run("WARM", "--sol-dir", str(tmp_path / "sols"))[0] == 0
    assert calls[0] == ("step1", tmp_path / "sols" / "WARM.lp.sol")
    assert json.loads((results / "WARM.json").read_text(encoding="utf-8"))["resolved"]["options"]["sol_dir"] == (
        (tmp_path / "sols").as_posix())


@pytest.mark.usefixtures("ascii_tmp_path")
def test_step_one_sol_is_named_after_the_result(cli_run) -> None:
    """.sol 按结果名起名：`--as` 另起名的 seed 族各写各的 .sol，并发时不互相覆盖。"""
    run, calls, _, results = cli_run
    assert run("WARM", "--set", "scenario.solver_seed=3", "--as", "WARM_s3")[0] == 0
    assert calls[0] == ("step1", results / "WARM_s3.lp.sol")


def test_existing_result_is_refused_before_solving_unless_forced(cli_run) -> None:
    run, calls, _, results = cli_run
    assert run("PLAIN")[0] == 0
    calls.clear()
    code, err = run("PLAIN")
    assert code == 1 and "PLAIN.json 已存在：要覆盖加 --force" in err and calls == []
    assert run("PLAIN", "--force")[0] == 0 and [call[0] for call in calls] == ["solve"]
    assert json.loads((results / "PLAIN.json").read_text(encoding="utf-8"))["resolved"]["options"]["force"] is True


@pytest.mark.parametrize(
    ("argv", "env", "message"),
    [
        (("WARM",), {"COAL_RETROFIT_START_SOL": "a.sol"}, "不能再设 COAL_RETROFIT_START_SOL"),
        (("WARM",), {"COAL_RETROFIT_LP_RELAX": "1"}, "不能再设 COAL_RETROFIT_LP_RELAX"),
        (("PLAIN", "--sol-dir", "sols"), {}, "--sol-dir 只配 warm_start = \"lp_relax\" 的情景"),
    ],
)
def test_conflicting_warm_start_options_are_refused(cli_run, monkeypatch, argv, env, message) -> None:
    run, calls, _, results = cli_run
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    code, err = run(*argv)
    assert code == 1 and message in err and calls == [] and not results.exists()


def test_runner_refuses_an_unknown_warm_start(cli_run, tmp_path) -> None:
    """登记表与 `--set` 按 `CHOICES` 拦住拼错的 warm_start；不经登记表直接构造情景调 `runner.run` 的，运行器再拦一次，
    不静默地不热启动。"""
    _, calls, _, results = cli_run
    spec = ScenarioSpec(name="X", source=tmp_path / "x.toml", tree=tmp_path / "tree",
                        scenario={"warm_start": "lp"}, assumptions={})
    with pytest.raises(RunError, match="warm_start = 'lp'：只能是"):
        runner.run(spec, threads=1, time_limit=60)
    assert calls == [] and not results.exists()


def test_manual_warm_start_still_works_without_the_field(cli_run, monkeypatch) -> None:
    """情景不热启动时手工设 START_SOL 照旧：只跑第 2 步，`resolved.warm_start` 记 None。"""
    run, calls, _, results = cli_run
    monkeypatch.setenv("COAL_RETROFIT_START_SOL", "a.sol")
    assert run("PLAIN")[0] == 0
    assert calls == [("solve", SolveControls(start_sol=Path("a.sol")))]
    assert json.loads((results / "PLAIN.json").read_text(encoding="utf-8"))["resolved"]["warm_start"] is None


@pytest.mark.usefixtures("ascii_tmp_path")
def test_step_one_failure_exits_3_without_step_two(cli_run) -> None:
    run, calls, state, results = cli_run
    state["fail_step_one"] = True
    code, err = run("WARM")
    assert code == 3 and "状态 INFEASIBLE，时限 77 s；没有跑第 2 步，也没有写结果" in err
    assert [call[0] for call in calls] == ["step1"] and not (results / "WARM.json").exists()


def test_no_usable_solution_exits_3(cli_run) -> None:
    run, _, state, results = cli_run
    state["objective"] = float("nan")
    code, err = run("PLAIN")
    assert code == 3 and "PLAIN 没有可用的解（求解状态 optimal）：结果表是补零的" in err
    assert (results / "PLAIN.json").exists()  # 照旧落盘，供查看求解状态
