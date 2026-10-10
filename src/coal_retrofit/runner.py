"""求解一个登记情景并落盘：建模、求解、建九张结果表（2026-10-10 前是 15 张，`results_tables`），写 CSV、结果工作簿
与 result.json。

2026-09-27 从 `scripts/run_single.py` 的 `run()`、`main()` 搬来，求解、建表与写文件的逻辑不变。变的有两处：
参数取自登记表（`coal_retrofit.scenarios`），求解树取自情景的 `tree`（或命令行 `--tree`），不再取脚本所在目录；
result.json 末尾多一段 `resolved`，记全部参数、求解树、`--set` 覆盖项、运行选项与 `COAL_RETROFIT_*` 环境变量，
其余各键不变。

2026-09-28 另加（求解流程进情景定义，不改模型）：情景 `warm_start = "lp_relax"` 时在同一进程里做热启动两步
（`run_controls`）；Gurobi 的 seed 与 MIPFocus 是情景字段，环境变量仍兼容（`build_parameters`）；已有同名结果时
求解之前就拒绝，`force` 才覆盖；`resolved` 另记热启动第 1 步（`warm_start`）、读了哪些输入文件（`input_files`）
与求解时的提交号（`code`）。

2026-10-02 另加：同目录多写一个参照 ChinaCCS.xlsm 版式的结果工作簿 `ccs_results.xlsx`（`results_workbook`），它出错只记
日志并删掉工作簿（删不掉也只记日志），CSV 与 result.json 照写；管网节点的省（`results_regions.node_provinces`）在建模
之前定好，缺省界图层等错误在 MIP 求解之前就报（`warm_start = "lp_relax"` 的情景在热启动第 1 步之后）。

2026-10-10 另加：`resolved.model_segment` 记模型分段号（`segment.MODEL_SEGMENT`），读结果时与当前代码比对；结果表换成
按源、汇、管网、资源、系统分组的九张（`results_tables`），工作簿改由它们生成。
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .constants import PLANNING_YEARS
from .constants_industry import INDUSTRY_ROUTES
from .optimization._shared import PATHWAY_INDEX
from .optimization.data_prep import _input_files, prepare_inputs
from .optimization.results_regions import node_provinces
from .optimization.results_tables import build_result_tables, write_result_tables
from .optimization.results_workbook import write_ccs_workbook
from .optimization.scenario import OptimizationAssumptions, OptimizationScenario
from .optimization.solver import SolveControls, _solve_joint_multi_period
from .paths import ProjectPaths
from .run_controls import (
    RunError,
    check_warm_start_env,
    code_state,
    env_controls,
    initial_state,
    lp_relaxation_start,
    sol_path,
)
from .scenarios import ScenarioRegistryError, ScenarioSpec, convert_field, shown_path
from .segment import MODEL_SEGMENT

logger = logging.getLogger(__name__)

# CLAUDE.md 二.1：Gurobi 只在（模型, 参数, 线程数）都不变时可复现，要相减的两次求解必须同线程数，缺省固定为 8。
# 0 交给 Gurobi 按机器自动定，换一台机器线程数就变，只告警不拦。
DEFAULT_THREADS = 8
DEFAULT_TIME_LIMIT = 36000

# 2026-09-28 之前设 seed 与 MIPFocus 的唯一办法，保留兼容：非空时读进这两个情景字段（`resolved` 里记的是实际
# 生效的值）；登记表或 `--set` 已给该字段而值不同，就报错。求解器只读字段。
SOLVER_ENV_FIELDS: dict[str, str] = {
    "COAL_RETROFIT_GUROBI_SEED": "solver_seed",
    "COAL_RETROFIT_MIPFOCUS": "mip_focus",
}


def build_parameters(
    spec: ScenarioSpec, name: str, threads: int = DEFAULT_THREADS, time_limit: int = DEFAULT_TIME_LIMIT,
    mip_gap: float | None = None,
) -> tuple[OptimizationScenario, OptimizationAssumptions]:
    """按登记的覆盖项、运行选项与兼容的环境变量（`SOLVER_ENV_FIELDS`）构造两个参数 dataclass；
    *name* 是结果名，写进 `experiment_id`。

    *mip_gap* 只对这一次求解覆盖登记的 MIPGap。它让一批求解能拿容差换时间，而不必去改（从而永久改变）
    情景定义。代价：凡涉及这次求解的差值，可证区间大致按比例变宽；对 seed 复现（同一情景只换
    `solver_seed` 重解）则直接毁掉测量：各次求解一旦允许单凭容差停在相距那么远的地方，
    seed 族测的就不再是求解器的简并度。seed 复现保持登记的 gap。
    """
    scenario_kw, assumption_kw = dict(spec.scenario), dict(spec.assumptions)
    if mip_gap is not None:
        scenario_kw = {**scenario_kw, "mip_gap": float(mip_gap)}
    for key, field_name in SOLVER_ENV_FIELDS.items():
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            number = int(raw)
        except ValueError:
            raise ScenarioRegistryError(f"环境变量 {key}={raw!r} 应为整数") from None
        value = convert_field("scenario", field_name, number, f"环境变量 {key}")
        if field_name in scenario_kw and scenario_kw[field_name] != value:
            raise ScenarioRegistryError(
                f"环境变量 {key}={raw} 与登记表或 --set 给的 {field_name} = {scenario_kw[field_name]!r} 不一致；"
                f"只留一处（{field_name} 是情景字段，环境变量只为兼容）"
            )
        scenario_kw = {**scenario_kw, field_name: value}
    assumptions = replace(OptimizationAssumptions(), **assumption_kw) if assumption_kw else OptimizationAssumptions()
    base_kw: dict[str, Any] = {
        "experiment_id": name,
        "description": name,
        "planning_years": tuple(PLANNING_YEARS),
    }
    base_kw.update(scenario_kw)
    if threads > 0:
        base_kw["solver_threads"] = threads
    else:
        logger.warning(
            "%s: threads=%d，线程数交给 Gurobi 自动定，这次求解不能与其他求解相减（CLAUDE.md 二.1）", name, threads
        )
    # 总是写：命令行给的时限说了算，不随 dataclass 缺省变（此前只在不等于 36000 时才写）。
    base_kw["solver_time_limit"] = time_limit
    return OptimizationScenario(**base_kw), assumptions


def solve(
    paths: ProjectPaths, name: str, scenario: OptimizationScenario, assumptions: OptimizationAssumptions,
    controls: SolveControls | None = None,
) -> dict[str, Any]:
    """建模、求解，把九张结果表与结果工作簿 `ccs_results.xlsx` 写到 `<树>/results/<name>/`；返回 result.json 的内容
    （还没有 `resolved`）。

    *controls* 是只改搜索路径或只做诊断的开关（`SolveControls`：热启动第 2 步的 MIP start、LP 松弛诊断等），缺省全关。
    """
    t0 = time.time()
    prepared = prepare_inputs(paths, scenario, assumptions)
    node_province = node_provinces(prepared, paths.data_dir / "ChinaMap" / "provinces.shp")
    years = scenario.planning_years
    state = initial_state(prepared)
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, state, controls)
    elapsed = time.time() - t0
    solver_quality = solution["solver_quality"]

    # 逐年汇总进 result.json；九张结果表另建（`results_tables`）
    year_summaries = {}
    for year in years:
        ys = solution["year_solutions"][year]
        share = ys["share"]
        year_data = ys["year_data"]

        # 汇总按本年发电量加权（已乘利用小时轨迹）
        gen_year = np.asarray(year_data.generation, dtype=np.float64)
        total_gen_year = float(gen_year.sum())
        pathway_shares = {}
        for pw, idx in PATHWAY_INDEX.items():
            pathway_shares[pw] = float((gen_year * share[:, idx]).sum() / total_gen_year) if total_gen_year > 0 else 0.0
        iy = year_data.industry
        ish = ys["industry_share"]
        groups = np.asarray(iy.target_groups).astype(str)
        residual_hub = iy.baseline_emissions_mt - (iy.reduction_mt * ish).sum(axis=1)
        industry_summary = {
            "baseline_mt": float(iy.baseline_emissions_mt.sum()),
            "reduction_mt": float((iy.reduction_mt * ish).sum()),
            "residual_by_group_mt": {
                g: float(residual_hub[groups == g].sum()) for g in sorted(set(groups))
            },
            "captured_mt": float((iy.captured_mt * ish).sum()),
            "h2_kg": float(np.sum(ys["industry_h2_flow_kg"])),
            "water_m3": float((iy.water_m3 * ish).sum()),
            "cost_annual_cny": float(ys["cost_breakdown_cny"]["industry_cost"]),
            "cost_capex_cny": float(ys["cost_breakdown_cny"]["industry_capex"]),
            "h2_price_national_mean_cny_per_kg": float(iy.h2_price_cny_per_kg),
            "share_by_route": {
                route: float((iy.baseline_emissions_mt * ish[:, idx]).sum()
                             / max(float(iy.baseline_emissions_mt.sum()), 1e-9))
                for idx, route in enumerate(INDUSTRY_ROUTES)
            },
        }
        coal_baseline_year = float(np.asarray(year_data.emissions_mt, dtype=np.float64).sum())
        coal_reduction = float(ys["total_reduction_mt"])
        year_summaries[int(year)] = {
            "status": ys["status"],
            "objective_cny": float(ys["objective_cny"]),
            "hours_scale": float(year_data.hours_scale),
            "part_load_factor": float(year_data.part_load_factor),
            "coal_generation_twh": total_gen_year / 1e6,
            "coal_baseline_mt": coal_baseline_year,
            "pathway_shares": pathway_shares,
            "coal_reduction_mt": coal_reduction,
            "coal_residual_mt": coal_baseline_year - coal_reduction,
            "sector_cap_fraction": dict(year_data.sector_cap_fraction or {}),
            "storage_national_injection_mtpa": (float(year_data.storage_national_injection_mtpa) if np.isfinite(year_data.storage_national_injection_mtpa) else None),  # 不设上限时记 null
            "industry": industry_summary,
            "cost_breakdown": {k: float(v) for k, v in ys["cost_breakdown_cny"].items()},
            # 煤电参照（全部维持不改造运行）的基线净运行成本，CNY/yr、不折现、不进目标（增量口径，2026-10-10 起）；上面的 cost_breakdown 是折现并乘区间权重后的值，二者相加前先换到同一口径（system.csv 的 cost_cny 是不折现值）。
            "coal_operating_reference_cny": float(np.sum(year_data.baseline_reference_cny)),
            "target_shortfall_mt": float(ys["slacks"]["target_shortfall_mt"]),
            "target_shortfall_by_group_mt": {
                k: float(v) for k, v in ys["slacks"]["target_shortfall_by_group"].items()
            },
            "solver_quality": solver_quality,
        }

    # 写 CSV
    out_dir = paths.root / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_result_tables(prepared, solution, scenario, assumptions, node_province, initial_state(prepared))
    write_result_tables(out_dir, tables)
    workbook = out_dir / "ccs_results.xlsx"
    try:
        write_ccs_workbook(workbook, tables, solution, prepared, scenario, assumptions, node_province)
    except Exception:  # 工作簿只是同一组结果的另一种版式，出错不能连累已求得的解
        logger.exception("%s: 结果工作簿 ccs_results.xlsx 没有写成，CSV 与 result.json 照写", name)
        try:  # 写了一半的、或 --force 之前那次求解留下的工作簿都删掉，不与这次的 CSV 混放
            workbook.unlink(missing_ok=True)
        except OSError as err:  # 如 Windows 上在 Excel 里开着（保存多半也是因此失败的）
            logger.error("%s: %s 删不掉（%s），它不是这次求解的结果，请手动删除", name, shown_path(workbook), err)

    result = {
        "name": name,
        "global_objective_cny": float(solution["objective_cny"]),
        "solver_quality": solver_quality,
        "years": year_summaries,
        "planning_years": [int(y) for y in years],
        "elapsed_seconds": round(elapsed, 1),
    }
    return result


def run(
    spec: ScenarioSpec, *, name: str | None = None, tree: Path | None = None, threads: int = DEFAULT_THREADS,
    time_limit: int = DEFAULT_TIME_LIMIT, mip_gap: float | None = None,
    applied_sets: Mapping[str, Any] | None = None, sol_dir: Path | None = None, force: bool = False,
) -> dict[str, Any]:
    """求解 *spec*，写 `<树>/results/<结果名>.json` 与同名目录下的九张表和结果工作簿，返回 result.json 的内容。

    *name* 是结果名（缺省为情景名；`--as` 另起），*tree* 换掉情景登记的求解树（`--tree`）。
    *applied_sets* 是已经并进 *spec* 的 `--set` 覆盖项，只用来记进 `resolved`。
    已有 `<结果名>.json` 时求解之前就拒绝（`RunError`），*force* 才覆盖。情景 `warm_start = "lp_relax"` 时先做热启动
    第 1 步（`run_controls.lp_relaxation_start`；.sol 的位置见 `run_controls.sol_path`，*sol_dir* 换目录），第 1 步
    没有解就抛 `WarmStartFailed`，不跑第 2 步、不写结果。
    """
    name = name or spec.name
    tree = Path(tree) if tree is not None else spec.tree
    if tree is None or not (tree / "inputs").is_dir():
        raise ScenarioRegistryError(f"{spec.name}：求解树 {tree} 下没有 inputs/")
    scenario, assumptions = build_parameters(spec, name, threads, time_limit, mip_gap)
    out_file = tree / "results" / f"{name}.json"
    if out_file.exists() and not force:
        raise RunError(f"{shown_path(out_file)} 已存在：要覆盖加 --force，或先把旧结果移走")
    controls, sol = env_controls(), None
    if scenario.warm_start == "lp_relax":
        check_warm_start_env()
        sol = sol_path(tree, name, sol_dir)
    elif scenario.warm_start != "none":
        raise RunError(f"warm_start = {scenario.warm_start!r}：只能是 \"none\" 或 \"lp_relax\"")
    elif sol_dir is not None:
        raise RunError("--sol-dir 只配 warm_start = \"lp_relax\" 的情景：那是热启动第 1 步写 .sol 的目录")
    code = code_state()
    paths = ProjectPaths(tree)
    warm_start = None
    if sol is not None:
        warm_start = lp_relaxation_start(paths, scenario, assumptions, sol)
        controls = replace(controls, start_sol=sol)
    result = solve(paths, name, scenario, assumptions, controls)
    result["resolved"] = {
        "registered_as": spec.name,
        "registry_file": shown_path(spec.source),
        "tree": shown_path(tree),
        "set": dict(applied_sets or {}),
        "options": {"threads": threads, "time_limit": time_limit, "mip_gap": mip_gap,
                    "sol_dir": shown_path(sol_dir) if sol_dir is not None else None, "force": force},
        "env": {key: value for key, value in sorted(os.environ.items()) if key.startswith("COAL_RETROFIT_")},
        "scenario": asdict(scenario),
        "assumptions": asdict(assumptions),
        "warm_start": warm_start,
        # 摘要（solver_quality 的 digest_*）按逻辑名记；部门目标、产量指数的文件随情景的来源换，--pair 靠这里区分。
        "input_files": {key: _tree_relative(path, tree) for key, path in _input_files(paths, scenario).items()},
        "code": code,
        # 读结果时与当前代码比对（`segment.check_segment`），分段号不同的旧结果不能用。
        "model_segment": MODEL_SEGMENT,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return result


def _tree_relative(path: Path, tree: Path) -> str:
    """相对求解树的 posix 路径（如 `inputs/plants.csv`）；不在树下的按 `shown_path` 写。"""
    try:
        return Path(path).relative_to(tree).as_posix()
    except ValueError:
        return shown_path(path)
