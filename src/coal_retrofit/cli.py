"""命令行：`python -m coal_retrofit list | show | diff | run | water-members`。

    python -m coal_retrofit list
    python -m coal_retrofit show ST_WA_cwatm_126_dry_oq [--set 节.字段=值 ...]
    python -m coal_retrofit diff ST_BASE ST_WA_cwatm_126_dry_oq
    python -m coal_retrofit run ST_BASE [--threads 8] [--time-limit 36000] [--mip-gap G] [--force] [--sol-dir D]
    python -m coal_retrofit run ST_BASE --set assumptions.air_retrofit_capex_cny_per_kw=400 --as ST_BASE_air400
    python -m coal_retrofit run ST_BASE --set scenario.solver_seed=3 --as ST_BASE_seed3
    python -m coal_retrofit run ST_BASE --water cwatm/gfdl-esm4/ssp370 --season dry [--targets total]
    python -m coal_retrofit water-members [--tree _indtree]

情景登记在仓库根 `scenarios/*.toml`（格式见 `coal_retrofit.scenarios`），`--registry` 可换目录。

- `run` 写 `<tree>/results/<结果名>.json` 与同名目录下的九张表；结果名缺省是情景名。已有 `<结果名>.json` 时
  求解之前就拒绝，`--force` 才覆盖。
- 情景 `warm_start = "lp_relax"`（`ST_` 系）时自动做热启动两步（实现说明 §9.7）：第 1 步的 .sol 缺省写在
  `<tree>/results/<结果名>.lp.sol`，`--sol-dir` 换目录（须是 ASCII 路径），见 `run_controls.sol_path`。
- 用了 `--set` 就必须用 `--as` 另起结果名，且不能与登记表里的情景同名（不分大小写）：登记情景名下的结果
  只来自登记的参数。seed 族就是一个种子一个结果名。
- 水文边界条件（`coal_retrofit.hydrology`）：`--water 水文模型/GCM/SSP`（成员见 `water-members`）、`--season annual|dry`
  与碳目标入口 `--targets sector|total`（`scenario.sector_target_mode`）并到登记情景上（基底无水约束时另设
  `water_mode = "grid_supply"`），与 `--set` 一样记进 `resolved.set`。不给 `--as` 时结果名自动拼成
  `<情景>_<水文模型>_<GCM>_<SSP>_<季节>[_<目标>]`（`ST_BASE --water cwatm/gfdl-esm4/ssp370 --season dry` 即
  `ST_BASE_cwatm_gfdl-esm4_ssp370_dry`）。`--season` 只配 `--water`；与 `--set` 改同一字段时报错。`show` 也认这三个
  选项，并显示 run 会用的结果名。
- `water-members` 列出 `<tree>/inputs/water_scenarios.csv` 的成员，逐个核各规划年的可用水量与耗水数据（两个季节）；
  `--tree` 缺省取登记情景的求解树（只许有一棵）。有水约束且指定了成员的 run 在求解之前也核这个成员（只核它读的季节）。
- `--mip-gap` 只放宽这一次求解的 gap，结果名不变（原 `run_single.py` 的用法）；seed 复现不要用。
- `--tree` 换求解树（相对当前目录），结果也写在那棵树下。
- `show`、`diff` 按 `run` 的缺省选项（8 线程、36000 s）与当前的兼容环境变量构造参数，不读输入、不求解。

退出码：0 解出来了（`water-members`：成员数据都齐全）；1 登记表、`--set`、`--water` 等、环境变量写错，或求解之前就
拒绝（结果已存在、热启动与环境变量冲突、水文成员的数据不全），`water-members` 查出数据不全，读输入、Gurobi 出错之类
没有接住的异常也是 1（Python 的缺省，打印 traceback）；2 命令行写错；3 没有可用的解（求解状态不可接受，结果表补零；
或热启动第 1 步没有解，第 2 步没跑、没写结果）。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from .constants import PLANNING_YEARS
from .hydrology import MEMBERS_FILE, check_members, load_members, member_id, water_arg, water_overrides
from .optimization.scenario import OptimizationAssumptions, OptimizationScenario
from .run_controls import RunError, WarmStartFailed
from .runner import DEFAULT_THREADS, DEFAULT_TIME_LIMIT, SOLVER_ENV_FIELDS, build_parameters, run
from .scenarios import (
    CHOICES,
    DEFAULT_REGISTRY_DIR,
    NAME_RE,
    NAME_RULE,
    Registry,
    ScenarioRegistryError,
    ScenarioSpec,
    apply_sets,
    diff_resolved,
    load_registry,
    shown_path,
)

_NAME_FIELDS = ("experiment_id", "description")


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行并执行；返回退出码（见模块说明）。登记表写错或求解之前就拒绝时打印原因、返回 1。"""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        return int(args.handler(registry, args, parser))
    except (ScenarioRegistryError, RunError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m coal_retrofit", description="按情景登记表求解，查看、比较情景参数。")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_DIR,
                        help="情景登记目录（缺省：仓库根的 scenarios/）")
    sub = parser.add_subparsers(dest="command", required=True, metavar="命令")

    p = sub.add_parser("list", help="列出能求解的情景")
    p.set_defaults(handler=_cmd_list)

    p = sub.add_parser("show", help="解析后的全部参数，标出与缺省不同的项")
    p.add_argument("name")
    p.add_argument("--set", action="append", default=[], metavar="节.字段=值", help="先覆盖再显示，可重复")
    _add_boundary(p)
    p.set_defaults(handler=_cmd_show)

    p = sub.add_parser("diff", help="两个情景的参数差（相减之前核对只差该差的）")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(handler=_cmd_diff)

    p = sub.add_parser("run", help="求解一个情景并落盘")
    p.add_argument("name")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help="Gurobi 线程数（缺省 %(default)s；CLAUDE.md 二.1：要相减的两次求解必须同线程数；"
                        "0 = 交给 Gurobi 自动定，会告警）")
    p.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT, help="求解时限，秒（缺省 %(default)s）")
    p.add_argument("--mip-gap", type=float, default=None,
                   help="只对这次求解覆盖 MIPGap，结果名不变（seed 复现不要用，见 runner.build_parameters）")
    p.add_argument("--set", action="append", default=[], metavar="节.字段=值",
                   help="覆盖一个参数，可重复；必须配 --as")
    p.add_argument("--as", dest="result_name", default=None, metavar="结果名", help="另起结果名")
    p.add_argument("--tree", type=Path, default=None, help="换求解树（相对当前目录），结果也写在那里")
    p.add_argument("--force", action="store_true", help="已有同名结果时覆盖（缺省拒绝，求解之前就查）")
    p.add_argument("--sol-dir", type=Path, default=None, metavar="目录",
                   help="热启动第 1 步的 .sol 写在这个目录（须是 ASCII 路径；缺省 <tree>/results/）")
    _add_boundary(p)
    p.set_defaults(handler=_cmd_run)

    p = sub.add_parser("water-members", help="列出水文成员（水文模型 × GCM × SSP），逐个核数据")
    p.add_argument("--tree", type=Path, default=None, help="求解树（缺省：登记情景的求解树，只许有一棵）")
    p.set_defaults(handler=_cmd_water_members)
    return parser


def _add_boundary(p: argparse.ArgumentParser) -> None:
    """`show`、`run` 共用的水文边界条件与碳目标入口（`_apply_boundary`）。"""
    p.add_argument("--water", default=None, metavar="水文模型/GCM/SSP",
                   help="水文成员，如 cwatm/gfdl-esm4/ssp370（全部成员：water-members）")
    p.add_argument("--season", choices=CHOICES["water_season"], default=None,
                   help="可用水量取全年或枯水期（只配 --water）")
    p.add_argument("--targets", choices=CHOICES["sector_target_mode"], default=None,
                   help="碳目标入口：分部门上限或合计总量上限（scenario.sector_target_mode）")


def _cmd_list(registry: Registry, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    names = registry.runnable()
    width = max((len(name) for name in names), default=0)
    for name in names:
        spec = registry.specs[name]
        print(f"{name:{width}s}  [{_tree(spec)}]  {spec.note}".rstrip())
    return 0


def _cmd_show(registry: Registry, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    spec, applied = apply_sets(registry.get(args.name), args.set)
    spec, boundary, suffix = _apply_boundary(spec, args, applied, spec.tree)
    scenario, assumptions = build_parameters(spec, spec.name)
    print(f"{spec.name}（{shown_path(spec.source)}；{' ← '.join(registry.lineage(spec.name))}）")
    if spec.note:
        print(f"说明：{spec.note}")
    print(f"求解树：{_tree(spec)}")
    for key, value in applied.items():
        print(f"--set {key} = {value!r}")
    for key, value in boundary.items():
        print(f"--water/--season/--targets → {key} = {value!r}")
    if suffix and not args.set:
        print(f"run 的结果名（不给 --as 时）：{spec.name}{suffix}")
    for key, field_name in SOLVER_ENV_FIELDS.items():
        if os.environ.get(key):
            print(f"环境变量 {key}={os.environ[key]} → scenario.{field_name}")
    print(f"标 * 的与 dataclass 缺省不同；线程数、时限按 run 的缺省（{DEFAULT_THREADS}、{DEFAULT_TIME_LIMIT} s）。")
    pairs: dict[str, tuple[Any, Any]] = {
        "scenario": (scenario, OptimizationScenario(experiment_id="", description="")),
        "assumptions": (assumptions, OptimizationAssumptions()),
    }
    for section, (obj, default) in pairs.items():
        print(f"[{section}]")
        for field in fields(obj):
            value, base = getattr(obj, field.name), getattr(default, field.name)
            if field.name in _NAME_FIELDS or value == base:
                print(f"  {field.name} = {value!r}")
            else:
                print(f"* {field.name} = {value!r}    （缺省 {base!r}）")
    return 0


def _cmd_diff(registry: Registry, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    a, b = registry.get(args.a), registry.get(args.b)
    diffs = diff_resolved(_params(a), _params(b))
    if not diffs:
        print(f"{a.name} 与 {b.name} 参数完全相同")
        return 0
    print(f"{a.name} → {b.name}：{len(diffs)} 项不同")
    for key, value_a, value_b in diffs:
        print(f"  {key}: {value_a!r} → {value_b!r}")
    return 0


def _cmd_run(registry: Registry, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    spec = registry.get(args.name)
    if args.set and args.result_name is None:
        parser.error("用了 --set 就必须用 --as 另起结果名：登记情景名下的结果只来自登记的参数")
    if args.result_name is not None:
        _check_result_name(registry, args.result_name, "--as", parser)
    spec, applied = apply_sets(spec, args.set)
    tree = args.tree.absolute() if args.tree is not None else None
    spec, boundary, suffix = _apply_boundary(spec, args, applied, tree if tree is not None else spec.tree)
    name = args.result_name
    if name is None and suffix:
        name = f"{spec.name}{suffix}"
        _check_result_name(registry, name, "自动拼出的结果名", parser)
        print(f"结果名：{name}")
    sol_dir = args.sol_dir.absolute() if args.sol_dir is not None else None
    try:
        result = run(spec, name=name, tree=tree, threads=args.threads, time_limit=args.time_limit,
                     mip_gap=args.mip_gap, applied_sets={**applied, **boundary}, sol_dir=sol_dir, force=args.force)
    except WarmStartFailed as exc:
        print(f"错误：{exc}；没有跑第 2 步，也没有写结果", file=sys.stderr)
        return 3
    _print_summary(result)
    if not math.isfinite(result["global_objective_cny"]):
        status = result["solver_quality"].get("status")
        print(f"错误：{result['name']} 没有可用的解（求解状态 {status}）：结果表是补零的，不能拿来出图或相减",
              file=sys.stderr)
        return 3
    return 0


def _check_result_name(registry: Registry, name: str, label: str, parser: argparse.ArgumentParser) -> None:
    """另起的结果名（`--as` 或自动拼出的，*label* 说是哪种）要合 `NAME_RE`，且不与登记情景同名（不分大小写）。"""
    if not NAME_RE.fullmatch(name):
        parser.error(f"{label} {name!r}：结果名{NAME_RULE}")
    clash = registry.name_clash(name)
    if clash is not None:
        same = "" if clash == name else f"（与 {clash} 只差大小写，Windows 上是同一个文件）"
        parser.error(f"{label} {name}：与登记表里的情景同名{same}，会冒充登记情景的结果")


def _apply_boundary(
    spec: ScenarioSpec, args: argparse.Namespace, applied: dict[str, Any], tree: Path | None,
) -> tuple[ScenarioSpec, dict[str, Any], str]:
    """`--water`、`--season`、`--targets` 并到（已并过 `--set` 的）情景上：返回新情景、这三个选项的覆盖项
    {`scenario.字段`: 值} 与结果名后缀（水文部分 `_水文模型_GCM_SSP_季节` 见 `hydrology.water_overrides`，再加
    `_sector` / `_total`）。成员按 *tree* 的成员表核。"""
    member = None
    if args.water is not None:
        if tree is None:  # 能求解的情景都有求解树（`scenarios._resolve`）；这里只为类型收窄
            raise ScenarioRegistryError(f"{spec.name}：没有求解树，读不到成员表 {MEMBERS_FILE}")
        member = member_id(args.water, load_members(tree))
    overrides, suffix = water_overrides(spec.scenario, member, args.season)
    if args.targets is not None:
        overrides["sector_target_mode"] = args.targets
        suffix += f"_{args.targets}"
    boundary = {f"scenario.{key}": value for key, value in overrides.items()}
    both = sorted(set(boundary) & set(applied))
    if both:
        raise ScenarioRegistryError(f"{'、'.join(both)} 既由 --set 给、又由 --water / --season / --targets 给，只留一处")
    return replace(spec, scenario={**spec.scenario, **overrides}), boundary, suffix


def _cmd_water_members(registry: Registry, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.tree is not None:
        tree = args.tree.absolute()
    else:
        trees = sorted({spec.tree for spec in registry.specs.values() if not spec.abstract and spec.tree is not None})
        if len(trees) != 1:
            parser.error(f"登记情景有 {len(trees)} 棵求解树（{'、'.join(map(shown_path, trees))}），用 --tree 指定")
        tree = trees[0]
    members = load_members(tree)
    family = dict(zip(members["scenario_id"], members["scenario_family"])) if "scenario_family" in members else {}
    problems = check_members(tree, PLANNING_YEARS)
    years = "、".join(str(year) for year in PLANNING_YEARS)
    print(f"{shown_path(tree / 'inputs' / MEMBERS_FILE)}：{len(members)} 个成员；核 {years} 年每个水节点的可用水量、"
          "每个流域的生活与灌溉耗水")
    width = max((len(water_arg(member)) for member in problems), default=0)
    for member, items in problems.items():
        status = f"{len(items)} 处不全" if items else "齐全"
        print(f"{water_arg(member):{width}s}  {family.get(member, ''):13s}  {status}")
        for item in items:
            print(f"    {item}")
    print("用法：python -m coal_retrofit run <情景> --water <第一列> [--season annual|dry] [--targets sector|total]")
    return 1 if any(problems.values()) else 0


def _params(spec: ScenarioSpec) -> dict[str, Any]:
    """`diff` 比的内容，与 result.json 的 `resolved` 同形。"""
    scenario, assumptions = build_parameters(spec, spec.name)
    return {"tree": _tree(spec), "scenario": asdict(scenario), "assumptions": asdict(assumptions)}


def _tree(spec: ScenarioSpec) -> str:
    return shown_path(spec.tree) if spec.tree is not None else "（无）"


def _print_summary(result: dict[str, Any]) -> None:
    obj = result["global_objective_cny"]
    status = "optimal" if not any(
        y["status"] != "optimal" for y in result["years"].values()
    ) else "WARNING"
    print(f"{result['name']}: obj={obj:.2e}, status={status}, time={result['elapsed_seconds']}s")
    for yr, yd in sorted(result["years"].items()):
        shares = yd["pathway_shares"]
        parts = " ".join(f"{pw}={v:.1%}" for pw, v in sorted(shares.items()) if v > 0.005)
        print(f"  {yr}: {parts}")
