"""`scripts/check_run_provenance.py --pair` 的判定：不求解，不需要 Gurobi。

结果是手写的 result.json：`solver_quality` 只填脚本读的键，`resolved` 只有参数、环境变量与求解时的提交号几部分。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
CODE = {"commit": "a" * 40, "dirty": False}
FILES = {"plants": "inputs/plants.csv", "sector_targets": "inputs/sector_targets_times_cn60.csv"}


def _result(env: dict[str, str] | None, objective: float = 4.0e12, mip_gap: float | None = 0.03, *,
            code: dict | None = CODE, bound: float | None = None, digests: dict[str, str] | None = None,
            input_files: dict[str, str] | None = None, **scenario: object) -> dict:
    """一份 result.json。*env* 为 None 表示没有 `resolved` 段（2026-09-27 之前落盘）；*code* 为 None 表示 `resolved`
    里没有 `code`（加上这一项之前落盘）；*bound* 是 ObjBound，缺省不记（更早的结果没有这个键）。"""
    quality: dict = {"fingerprint": "0x1", "num_vars": 10, "num_constrs": 5, "num_nonzeros": 30,
                     "threads_param": 8, "threads_pinned": True, "seed": 0, "mip_focus": 1, "mip_gap": mip_gap,
                     **(digests or {})}
    if bound is not None:
        quality["objective_bound_cny"] = bound
    payload: dict = {"global_objective_cny": objective, "solver_quality": quality}
    if env is not None:
        resolved: dict = {"tree": "_indtree", "env": env, "scenario": {"mip_gap": 0.03, **scenario}, "assumptions": {}}
        if code is not None:
            resolved["code"] = code
        if input_files is not None:
            resolved["input_files"] = input_files
        payload["resolved"] = resolved
    return payload


RESULTS = {
    "MIP": _result({}),
    "MIP_WRITE": _result({"COAL_RETROFIT_WRITE_SOL": "/s/x.sol"}),
    "WARM": _result({"COAL_RETROFIT_START_SOL": "/s/a.sol"}),
    "WARM_DRY": _result({"COAL_RETROFIT_START_SOL": "/s/b.sol"}, objective=4.5e12, water_season="dry"),
    "AUTO": _result({}, warm_start="lp_relax"),  # 运行器自动两步：与手工设 START_SOL 是同一套流程
    "LP": _result({"COAL_RETROFIT_LP_RELAX": "1"}, objective=3.7e12, mip_gap=None),
    "LP_DRY": _result({"COAL_RETROFIT_LP_RELAX": "1"}, objective=3.9e12, mip_gap=None, water_season="dry"),
    "NOSOL": _result({}, objective=float("nan"), mip_gap=None),  # 求解状态不可接受：目标函数写成 NaN，结果表补零
    "OLD": _result(None),
    "NOCODE": _result({}, code=None),
    "OTHER_COMMIT": _result({}, code={"commit": "b" * 40, "dirty": False}),
    "DIRTY": _result({}, code={"commit": "a" * 40, "dirty": True}),
    "NOGIT": _result({}, code={"commit": None, "dirty": None}),
    "CP0": _result({}, carbon_price_cny_per_t_by_year=[0.0, 0.0, 0.0, 0.0]),
    "CP1": _result({}, carbon_price_cny_per_t_by_year=[50.0, 100.0, 150.0, 200.0]),
    "DIG": _result({}, digests={"digest_plants": "p1", "digest_sector_targets": "s1"}, input_files=FILES),
    "DIG_PLANTS": _result({}, digests={"digest_plants": "p2", "digest_sector_targets": "s1"}, input_files=FILES),
    "DIG_OTHER_FILE": _result({}, digests={"digest_plants": "p1", "digest_sector_targets": "s2"},
                              input_files={**FILES, "sector_targets": "inputs/sector_targets_none.csv"}),
    "DIG_WATER": _result({}, digests={"digest_plants": "p1", "digest_sector_targets": "s1", "digest_water_basin_caps": "w"},
                         input_files={**FILES, "water_basin_caps": "inputs/water_basin_caps.csv"}),
    # 下界记的是 ObjBound：区间按它算，不按 gap 反推（gap 反推会得出 [+1.850, +8.247]%）。
    "BOUND_C": _result({}, objective=1.00e12, bound=0.98e12),
    "BOUND_T": _result({}, objective=1.05e12, bound=1.03e12),
    # toy 的目标函数是负的：只给绝对区间。
    "NEG_C": _result({}, objective=-5.0e8, bound=-5.1e8),
    "NEG_T": _result({}, objective=-4.0e8, bound=-4.2e8),
    # 目标函数是正的，下界不是：hi 的分母 LB_c 不是正数，同样只给绝对区间。
    "LBNEG_C": _result({}, objective=1.0e9, bound=-1.0e8),
    "LBNEG_T": _result({}, objective=1.1e9, bound=0.9e9),
}


def _write_results(directory: Path) -> None:
    for name, payload in RESULTS.items():
        (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    alias = "check_run_provenance_under_test"
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / "check_run_provenance.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def check(monkeypatch, tmp_path, capsys) -> Callable[..., tuple[int, str]]:
    """跑脚本的 `main()`，结果目录由 `--results` 指到 *tmp_path*；返回 (退出码, 标准输出)。"""
    module = _load(monkeypatch)
    assert module.RESULTS == REPO / "_indtree" / "results"  # 缺省读数据树 _indtree/ 的 results/
    _write_results(tmp_path)

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
        ("AUTO", "WARM", "参数差 1 项"),  # 自动两步与手工 START_SOL 是同一套流程，算都热启动了
        ("DIG", "DIG_OTHER_FILE", "两边读的文件不同"),  # 部门目标换了来源：文件不同，摘要不比
        ("DIG", "DIG_WATER", "只有 DIG_WATER 读了"),  # 无水约束的一边不读水的文件
        ("DIG_WATER", "DIG", "只有 DIG_WATER 读了"),  # 两边换位置，点名的仍是读了水文件的那一边
    ],
)
def test_pair_passes(check, a: str, b: str, message: str) -> None:
    code, out = check("--pair", a, b)
    assert code == 0 and message in out and "no hard failures" in out


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ("MIP", "LP", "COAL_RETROFIT_LP_RELAX 两边不同"),
        ("MIP", "WARM", "只有 WARM 热启动了"),
        ("AUTO", "MIP", "只有 AUTO 热启动了"),
        ("OLD", "LP", "OLD 没有 resolved 段"),  # 旧结果核不了参数与环境变量：不放行
        ("MIP", "OLD", "OLD 没有 resolved 段"),  # 两个位置都要查
        ("OLD", "OLD", "没有 resolved 段"),
        ("LP", "LP_DRY", "两边都是 LP 松弛"),  # 热启动第 1 步都没接上第 2 步：环境变量相同，也不能相减
        ("MIP", "NOSOL", "NOSOL 没有可用的解"),
        ("NOSOL", "MIP", "NOSOL 没有可用的解"),
        ("MIP", "NOPE", "下没有 NOPE 的结果"),
        ("MIP", "NOCODE", "NOCODE 的 resolved 段没有 code"),  # 分不出是不是在模型改动之后求解的
        ("NOCODE", "MIP", "NOCODE 的 resolved 段没有 code"),
        ("CP0", "CP1", "carbon_price_cny_per_t_by_year 两边不同"),  # 目标函数的口径不同
        ("DIG", "DIG_PLANTS", "输入 plants 两边不同"),  # CLAUDE.md 二.6：不同输入版本不得相减
    ],
)
def test_pair_fails(check, a: str, b: str, message: str) -> None:
    code, out = check("--pair", a, b)
    assert code == 1 and message in out
    assert "SIGN RESOLVED" not in out and "inside solver bounds" not in out  # 不能相减的一对不给判断


def test_pair_mip_lp_is_exactly_one_failure(check) -> None:
    """只一边是 LP 松弛：只记环境变量那一条，不因两边同为 LP、没有可用的解之类的判定重复记。"""
    code, out = check("--pair", "MIP", "LP")
    assert code == 1 and "FAILURES (1)" in out


@pytest.mark.parametrize(
    ("b", "message"),
    [
        ("OTHER_COMMIT", "两边的提交号不同（aaaaaaaaa vs bbbbbbbbb）"),
        ("DIRTY", "DIRTY 求解时有未提交的改动"),
        ("NOGIT", "至少一边求解时记不了提交号"),
    ],
)
def test_pair_code_differences_only_warn(check, b: str, message: str) -> None:
    """提交号不同、有未提交的改动、记不了提交号：只告警（模型改动与否要人对着提交核），仍给区间判断。"""
    code, out = check("--pair", "MIP", b)
    assert code == 0 and "WARNINGS (1)" in out and message in out and "no hard failures" in out
    assert "certified [-3.000, +3.093]%  -> inside solver bounds" in out  # 两边同为 INC 4e12、gap 3%


def test_interval_uses_the_recorded_bound(check) -> None:
    """CLAUDE.md 二.3：lo = (LB_t − INC_c)/INC_c = +3.000%，hi = (INC_t − LB_c)/LB_c = +7.143%，下界取记下的 ObjBound。"""
    code, out = check("--pair", "BOUND_C", "BOUND_T")
    assert code == 0 and "effect +5.000%  certified [+3.000, +7.143]%  -> SIGN RESOLVED" in out


def test_interval_falls_back_to_the_gap(check) -> None:
    """没有 ObjBound 的结果按 gap 反推下界：LB = INC − gap·|INC|（两边 gap 3%）。"""
    code, out = check("--pair", "WARM", "WARM_DRY")
    assert code == 0 and "effect +12.500%  certified [+9.125, +15.979]%  -> SIGN RESOLVED" in out


def test_interval_is_absolute_when_the_objective_is_negative(check) -> None:
    """对照的目标函数是负数时相对区间的正负会反：只给绝对区间 [LB_t − INC_c, INC_t − LB_c]，单位元。"""
    code, out = check("--pair", "NEG_C", "NEG_T")
    assert code == 0
    assert "effect +1.0000e+08 元  certified [+8.0000e+07, +1.1000e+08] 元" in out and "SIGN RESOLVED" in out


def test_interval_is_absolute_when_the_bound_is_not_positive(check) -> None:
    """对照的目标函数是正的、下界不是：hi = (INC_t − LB_c)/LB_c 的分母不是正数，也只给绝对区间。"""
    code, out = check("--pair", "LBNEG_C", "LBNEG_T")
    assert code == 0 and "inside solver bounds" in out
    assert "effect +1.0000e+08 元  certified [-1.0000e+08, +1.2000e+09] 元" in out


def test_hardcoded_contrasts_only_warn(monkeypatch, tmp_path, capsys) -> None:
    """不带 `--pair` 的写死对照表照旧（那批 v9 结果都早于 resolved 段）：旧结果只注明比不了，缺一边只记 warning；
    两边都是 LP 松弛、有一边没有可用的解、没有 code，也只在 `--pair` 下判。"""
    module = _load(monkeypatch)
    _write_results(tmp_path)
    failures: list[str] = []
    warnings: list[str] = []
    contrasts = [("old", "OLD", "MIP"), ("missing", "MIP", "NOPE"), ("lp", "LP", "LP_DRY"), ("nosol", "MIP", "NOSOL"),
                 ("nocode", "MIP", "NOCODE")]
    module.check_contrasts(failures, warnings, False, contrasts=contrasts, results=tmp_path)
    assert failures == [] and warnings == ["missing: one side not solved yet"]
    assert "比不了" in capsys.readouterr().out


def test_results_only_with_pair(check) -> None:
    with pytest.raises(SystemExit) as excinfo:
        check()
    assert excinfo.value.code == 2
