"""求解一个登记情景并落盘：建模、求解、建 15 张结果表，写 CSV 与 result.json。

2026-09-27 从 `scripts/run_single.py` 的 `run()`、`main()` 搬来，求解、建表与写文件的逻辑不变。变的有两处：
参数取自登记表（`coal_retrofit.scenarios`），求解树取自情景的 `tree`（或命令行 `--tree`），不再取脚本所在目录；
result.json 末尾多一段 `resolved`，记全部参数、求解树、`--set` 覆盖项、运行选项与 `COAL_RETROFIT_*` 环境变量，
其余各键不变。

2026-09-28 另加（求解流程进情景定义，不改模型）：情景 `warm_start = "lp_relax"` 时在同一进程里做热启动两步
（`run_controls`）；Gurobi 的 seed 与 MIPFocus 是情景字段，环境变量仍兼容（`build_parameters`）；已有同名结果时
求解之前就拒绝，`force` 才覆盖；`resolved` 另记热启动第 1 步（`warm_start`）、读了哪些输入文件（`input_files`）
与求解时的提交号（`code`）。
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
import pandas as pd

from .constants import PLANNING_YEARS
from .constants_industry import INDUSTRY_ROUTES
from .optimization._shared import PATHWAY_INDEX
from .optimization.data_prep import _input_files, prepare_inputs
from .optimization.results import (
    _alive_edge_added_stock,
    _build_ammonia_flow_table,
    _build_biomass_flow_table,
    _build_co2_flow_direction_table,
    _build_cost_breakdown,
    _build_edge_table,
    _build_industry_detail_table,
    _build_pathway_table,
    _build_plant_cost_table,
    _build_plant_detail_table,
    _build_province_table,
    _build_sanity_checks,
    _build_slack_detail_table,
    _build_storage_table,
    _build_supply_table,
    _build_water_flow_table,
)
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
    """建模、求解，把 15 张结果表写到 `<树>/results/<name>/`；返回 result.json 的内容（还没有 `resolved`）。

    *controls* 是只改搜索路径或只做诊断的开关（`SolveControls`：热启动第 2 步的 MIP start、LP 松弛诊断等），缺省全关。
    """
    t0 = time.time()
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.planning_years
    state = initial_state(prepared)
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, state, controls)
    elapsed = time.time() - t0
    solver_quality = solution["solver_quality"]

    # 逐年建结果表
    pathway_tables, province_tables, edge_tables = [], [], []
    storage_tables, supply_tables, cost_tables = [], [], []
    plant_detail_tables, biomass_flow_tables, ammonia_flow_tables = [], [], []
    water_flow_tables, slack_detail_tables, co2_direction_tables = [], [], []
    plant_cost_tables = []
    industry_detail_tables = []
    sanity_tables = []
    prev_share_values = None
    prev_retrofit_installed = None

    state_track = initial_state(prepared)

    year_summaries = {}
    prev_industry_capacity = None
    new_cap_by_year: dict[int, np.ndarray] = {}  # 逐年新增管道容量，在役存量只数寿命内的
    for year_index, year in enumerate(years):
        ys = solution["year_solutions"][year]
        share = ys["share"]
        year_data = ys["year_data"]
        interval_years = scenario.interval_years(years, year_index, assumptions)
        state_track.edge_added_stock_mtpa = _alive_edge_added_stock(
            new_cap_by_year, year, assumptions.pipeline_lifetime_years, len(prepared.network.edges)
        )
        state_before = state_track.clone()

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
            "coal_generation_twh": total_gen_year / 1e6,
            "coal_baseline_mt": coal_baseline_year,
            "pathway_shares": pathway_shares,
            "coal_reduction_mt": coal_reduction,
            "coal_residual_mt": coal_baseline_year - coal_reduction,
            "sector_cap_fraction": dict(year_data.sector_cap_fraction or {}),
            "storage_deployment_fraction": float(year_data.storage_deployment_fraction),
            "industry": industry_summary,
            "cost_breakdown": {k: float(v) for k, v in ys["cost_breakdown_cny"].items()},
            "target_shortfall_mt": float(ys["slacks"]["target_shortfall_mt"]),
            "target_shortfall_by_group_mt": {
                k: float(v) for k, v in ys["slacks"]["target_shortfall_by_group"].items()
            },
            "solver_quality": solver_quality,
        }

        # 逐厂、逐边的结果表
        pw_table = _build_pathway_table(
            prepared, scenario, year, share, ys["captured_mt_by_plant"],
            ys["biomass_blend_x_share"], ys["beccs_blend_x_share"], ys["ammonia_blend_x_share"],
            year_data=year_data, plant_reduction_mt=ys["plant_reduction_mt"],
        )
        prov_table = _build_province_table(pw_table)
        pathway_tables.append(pw_table)
        province_tables.append(prov_table)
        edge_tables.append(_build_edge_table(
            prepared, year, ys["edge_flow_mtpa"], ys["build_edge"], ys["new_cap_mtpa"], state_before, assumptions,
            pipe_count=ys["pipe_count"], pipe_tiers=tuple(year_data.pipe_tiers_mtpa),
        ))
        storage_tables.append(_build_storage_table(
            prepared, year, ys["storage_use_mtpa"], state_before, interval_years,
            injectivity_mtpa=year_data.storage_injectivity_mtpa,
        ))
        supply_tables.append(_build_supply_table(prepared, year, year_data, ys["biomass_flow_gj"], ys["ammonia_flow_kg"], ys["water_flow_m3"], ys["slacks"]["water_basin_use_m3"]))
        cost_tables.append(_build_cost_breakdown(year, ys["cost_breakdown_cny"]))
        sanity_tables.append(_build_sanity_checks(year, ys["slacks"], pw_table, prov_table))
        plant_detail_tables.append(_build_plant_detail_table(
            prepared, scenario, year, share,
            ys["captured_mt_by_plant"], ys["biomass_use_gj"], ys["ammonia_use_kg"],
            ys["water_use_m3"], ys["blend_level_b"], ys["blend_level_a"],
            ys["air_share"], year_data=year_data, plant_reduction_mt=ys["plant_reduction_mt"],
            biomass_blend_x_share=ys["biomass_blend_x_share"], beccs_blend_x_share=ys["beccs_blend_x_share"],
            ammonia_blend_x_share=ys["ammonia_blend_x_share"],
        ))
        industry_detail_tables.append(_build_industry_detail_table(
            prepared, year, year_data.industry, ys["industry_share"],
            h2_flow_kg=ys["industry_h2_flow_kg"], year_data=year_data,
            capacity_mt=ys["industry_capacity_mt"], prev_capacity_mt=prev_industry_capacity,
        ))
        prev_industry_capacity = ys["industry_capacity_mt"]
        biomass_flow_tables.append(_build_biomass_flow_table(prepared, year, ys["biomass_flow_gj"]))
        ammonia_flow_tables.append(_build_ammonia_flow_table(year_data, year, ys["ammonia_flow_kg"], prepared.plants))
        water_flow_tables.append(_build_water_flow_table(year_data, year, ys["water_flow_m3"], prepared.plants))
        slack_detail_tables.append(_build_slack_detail_table(prepared, year, year_data, ys["slacks"]))
        co2_direction_tables.append(_build_co2_flow_direction_table(prepared, year, ys["co2_flow_fwd"], ys["co2_flow_bwd"]))
        plant_cost_tables.append(_build_plant_cost_table(
            prepared, year, year_data, share, ys["biomass_use_gj"],
            prev_share_values=prev_share_values,
            plant_reduction_mt=ys["plant_reduction_mt"],
            retrofit_installed=ys["retrofit_installed"],
            prev_retrofit_installed=prev_retrofit_installed,
            capex_pathway_indices=solution["capex_pathway_indices"],
        ))

        new_cap_by_year[year] = ys["new_cap_mtpa"]
        state_track.remaining_storage_mt = np.maximum(0.0, state_track.remaining_storage_mt - ys["storage_use_mtpa"] * interval_years)
        prev_share_values = share
        prev_retrofit_installed = ys["retrofit_installed"]

    # 写 CSV
    out_dir = paths.root / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_tables = {
        "pathway_shares.csv": pd.concat(pathway_tables, ignore_index=True, sort=False),
        "province_pathways.csv": pd.concat(province_tables, ignore_index=True, sort=False),
        "plant_detail.csv": pd.concat(plant_detail_tables, ignore_index=True, sort=False),
        "industry_detail.csv": pd.concat(industry_detail_tables, ignore_index=True, sort=False),
        "network_edges.csv": pd.concat(edge_tables, ignore_index=True, sort=False),
        "storage_utilization.csv": pd.concat(storage_tables, ignore_index=True, sort=False),
        "resource_use.csv": pd.concat(supply_tables, ignore_index=True, sort=False),
        "biomass_flows.csv": pd.concat(biomass_flow_tables, ignore_index=True, sort=False),
        "ammonia_flows.csv": pd.concat(ammonia_flow_tables, ignore_index=True, sort=False),
        "water_flows.csv": pd.concat(water_flow_tables, ignore_index=True, sort=False),
        "co2_flow_direction.csv": pd.concat(co2_direction_tables, ignore_index=True, sort=False),
        "plant_cost.csv": pd.concat(plant_cost_tables, ignore_index=True, sort=False),
        "slack_detail.csv": pd.concat(slack_detail_tables, ignore_index=True, sort=False),
        "cost_breakdown.csv": pd.concat(cost_tables, ignore_index=True, sort=False),
        "sanity_checks.csv": pd.concat(sanity_tables, ignore_index=True, sort=False),
    }
    for fname, df in csv_tables.items():
        df.to_csv(out_dir / fname, index=False)

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
    """求解 *spec*，写 `<树>/results/<结果名>.json` 与同名目录下的 15 张表，返回 result.json 的内容。

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
