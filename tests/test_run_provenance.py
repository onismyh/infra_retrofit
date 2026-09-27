"""`check_run_provenance.py --pair`（`scripts/` 与 `_indtree/scripts/` 两份逐字节相同）的判定：不求解，不需要 Gurobi。

结果是手写的 result.json：`solver_quality` 只填脚本读的键，`resolved` 只有参数与环境变量两部分。
"""
from __future__ import annotations

import filecmp
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
COPIES = [REPO / "scripts", REPO / "_indtree" / "scripts"]


def _result(env: dict[str, str] | None, objective: float = 4.0e12, mip_gap: float | None = 0.03,
            **scenario: object) -> dict:
    """一份 result.json；*env* 为 None 表示没有 `resolved` 段（2026-09-27 之前落盘）。"""
    payload: dict = {
        "global_objective_cny": objective,
        "solver_quality": {"fingerprint": "0x1", "num_vars": 10, "num_constrs": 5, "num_nonzeros": 30,
                           "threads_param": 8, "threads_pinned": True, "seed": 0, "mip_focus": 1,
                           "mip_gap": mip_gap},
    }
    if env is not None:
        payload["resolved"] = {"tree": "_indtree", "env": env,
                               "scenario": {"mip_gap": 0.03, **scenario}, "assumptions": {}}
    return payload


RESULTS = {
    "MIP": _result({}),
    "MIP_WRITE": _result({"COAL_RETROFIT_WRITE_SOL": "/s/x.sol"}),
    "WARM": _result({"COAL_RETROFIT_START_SOL": "/s/a.sol"}),
    "WARM_DRY": _result({"COAL_RETROFIT_START_SOL": "/s/b.sol"}, objective=4.5e12, water_season="dry"),
    "LP": _result({"COAL_RETROFIT_LP_RELAX": "1"}, objective=3.7e12, mip_gap=None),
    "OLD": _result(None),
}


def _load(monkeypatch: pytest.MonkeyPatch, directory: Path) -> ModuleType:
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.delitem(sys.modules, "_bootstrap", raising=False)  # 两份 _bootstrap 不同，各载各的
    alias = f"check_run_provenance_under_test_{directory.parent.name}"
    spec = importlib.util.spec_from_file_location(alias, directory / "check_run_provenance.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=COPIES, ids=["scripts", "_indtree"])
def check(request, monkeypatch, tmp_path, capsys) -> Callable[..., tuple[int, str]]:
    """跑一份脚本的 `main()`，结果目录由 `--results` 指到 *tmp_path*；返回 (退出码, 标准输出)。"""
    module = _load(monkeypatch, request.param)
    assert module.RESULTS == request.param.parent / "results"  # 缺省读本树的 results/
    for name, payload in RESULTS.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    def run(*argv: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["check_run_provenance.py", "--results", str(tmp_path), *argv])
        code = module.main()
        return code, capsys.readouterr().out

    return run


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ("MIP", "MIP_WRITE", "求解相关的环境变量相同"),  # WRITE_SOL 只决定写不写 .sol，不比
        ("WARM", "WARM_DRY", "参数差 1 项"),  # 都热启动：.sol 路径按情景不同，只看设没设
    ],
)
def test_pair_passes(check, a: str, b: str, message: str) -> None:
    code, out = check("--pair", a, b)
    assert code == 0 and message in out and "no hard failures" in out


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ("MIP", "LP", "COAL_RETROFIT_LP_RELAX 两边不同"),
        ("MIP", "WARM", "COAL_RETROFIT_START_SOL 两边不同"),
        ("OLD", "LP", "OLD 没有 resolved 段"),  # 旧结果核不了参数与环境变量：不放行
        ("OLD", "OLD", "没有 resolved 段"),
        ("MIP", "NOPE", "下没有 NOPE 的结果"),
    ],
)
def test_pair_fails(check, a: str, b: str, message: str) -> None:
    code, out = check("--pair", a, b)
    assert code == 1 and message in out
    assert "SIGN RESOLVED" not in out and "inside solver bounds" not in out  # 不能相减的一对不给判断


def test_results_only_with_pair(check) -> None:
    with pytest.raises(SystemExit) as excinfo:
        check()
    assert excinfo.value.code == 2


def test_two_copies_are_identical() -> None:
    assert filecmp.cmp(COPIES[0] / "check_run_provenance.py", COPIES[1] / "check_run_provenance.py", shallow=False)
