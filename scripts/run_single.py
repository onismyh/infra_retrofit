"""求解一个已注册情景。

用法：
    python scripts/run_single.py <name> [--threads 8]
    python scripts/run_single.py --list
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import replace

from _bootstrap import ROOT

import numpy as np
import pandas as pd

from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.optimization._shared import SolveState, PATHWAY_INDEX
from coal_retrofit.constants_industry import INDUSTRY_ROUTES
from coal_retrofit.optimization.results import (
    _build_cost_breakdown,
    _build_pathway_table,
    _build_province_table,
    _build_edge_table,
    _build_storage_table,
    _build_supply_table,
    _build_sanity_checks,
    _build_plant_detail_table,
    _build_biomass_flow_table,
    _build_ammonia_flow_table,
    _build_water_flow_table,
    _build_slack_detail_table,
    _build_co2_flow_direction_table,
    _build_plant_cost_table,
    _build_industry_detail_table,
    _alive_edge_added_stock,
)
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.constants import PLANNING_YEARS

logger = logging.getLogger(__name__)

# CLAUDE.md 二.1：Gurobi 只在（模型, 参数, 线程数）都不变时可复现，要相减的两次求解必须同线程数，缺省固定为 8。
# 0 交给 Gurobi 按机器自动定，换一台机器线程数就变，只告警不拦。
DEFAULT_THREADS = 8

# 情景注册：name -> (scenario 覆盖, assumptions 覆盖)。
#
# ST_ 系（作者决定 2026-09-10/11）：部门碳目标（China TIMES CN60 四组轨迹，各组自身 2030 基线的比例）
# + 煤电利用小时轨迹 3600/3100/2000/1500 h + 工业产量指数；MIPGap 统一 3%（CLAUDE.md 二.2），
# 任何差值只能按可证区间报告。成本口径自 2026-09-22 起为改造 capex + 固定运维 + 能耗 + 期末残值，09-23 的
# PR #2 又改了目标函数与约束：PR #2 合入之前落盘的 _indtree/results/ 都不得与之后的新解相减。
_ST_COMMON: dict = {
    "mip_gap": 0.03,
    "sector_target_source": "times_cn60",
    "coal_operating_hours_by_year": (3600.0, 3100.0, 2000.0, 1500.0),
}
_ZERO_CARBON_PRICE: dict = {"carbon_price_cny_per_t_by_year": (0.0, 0.0, 0.0, 0.0)}
_CWATM_126_DRY: dict = {
    "water_mode": "grid_supply", "water_scenario_id": "cwatm|gfdl-esm4|ssp126", "water_season": "dry",
}

EXPERIMENTS: dict[str, tuple[dict, dict]] = {
    # 无水约束基准。
    "ST_BASE": ({**_ST_COMMON, **_ZERO_CARBON_PRICE}, {}),
    # 官方用水总量指标（流域上限）+ CWatM/GFDL SSP1-2.6 枯水期供水。
    "ST_WA_cwatm_126_dry_oq": ({**_ST_COMMON, **_ZERO_CARBON_PRICE, **_CWATM_126_DRY}, {}),
    # 碳价对照：默认碳价路径，煤电与工业同价；部门上限不收紧（sector_targets_none.csv 全为 1.0，
    # 即各组不高于自身 2030 水平），其余与 ST_BASE 相同。
    "ST_CP_BASE": (
        {**_ST_COMMON, "sector_target_source": "none", "industry_output_index_source": "times_cn60"},
        {},
    ),
}

def run(name: str, threads: int = DEFAULT_THREADS, time_limit: int = 36000,
        mip_gap: float | None = None) -> dict:
    """Solve one registered scenario.

    *mip_gap* overrides the registry entry for this invocation only. It exists so a campaign can
    trade tolerance for wall-clock without editing -- and thereby permanently changing -- the
    scenario definitions. WHAT IT COSTS: the certified interval on any contrast involving the run
    widens roughly in proportion, and for seed replicates (the same scenario re-solved under
    different `COAL_RETROFIT_GUROBI_SEED` values) it destroys the measurement outright, because a
    seed family stops being a probe of solver DEGENERACY once the runs are allowed to stop that far
    apart on tolerance alone. Keep seed replicates at the registry gap.
    """
    scenario_kw, assumption_kw = EXPERIMENTS[name]
    if mip_gap is not None:
        scenario_kw = {**scenario_kw, "mip_gap": float(mip_gap)}
    paths = ProjectPaths(ROOT)
    assumptions = replace(OptimizationAssumptions(), **assumption_kw) if assumption_kw else OptimizationAssumptions()
    base_kw = {
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
    if time_limit != 36000:
        base_kw["solver_time_limit"] = time_limit
    scenario = OptimizationScenario(**base_kw)

    t0 = time.time()
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.planning_years
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, state)
    elapsed = time.time() - t0
    solver_quality = solution["solver_quality"]

    # Build high-resolution tables
    pathway_tables, province_tables, edge_tables = [], [], []
    storage_tables, supply_tables, cost_tables = [], [], []
    plant_detail_tables, biomass_flow_tables, ammonia_flow_tables = [], [], []
    water_flow_tables, slack_detail_tables, co2_direction_tables = [], [], []
    plant_cost_tables = []
    industry_detail_tables = []
    sanity_tables = []
    prev_share_values = None
    prev_retrofit_installed = None

    state_track = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )

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

        # Aggregate summary, weighted by THIS year's generation (utilisation trajectory applied)
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

        # High-resolution tables
        pw_table = _build_pathway_table(
            prepared, scenario, year, share,
            ys["captured_mt_by_plant"], ys["blend_level_b"], ys["blend_level_a"],
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
            prepared, scenario, assumptions, year, year_data, share,
            ys["captured_mt_by_plant"], ys["biomass_use_gj"], ys["water_use_m3"],
            ys["blend_level_b"], ys["blend_level_a"],
            prev_share_values=prev_share_values,
            retrofit_installed=ys["retrofit_installed"],
            prev_retrofit_installed=prev_retrofit_installed,
            capex_pathway_indices=solution["capex_pathway_indices"],
        ))

        new_cap_by_year[year] = ys["new_cap_mtpa"]
        state_track.remaining_storage_mt = np.maximum(0.0, state_track.remaining_storage_mt - ys["storage_use_mtpa"] * interval_years)
        prev_share_values = share
        prev_retrofit_installed = ys["retrofit_installed"]

    # Save CSV artifacts
    out_dir = ROOT / "results" / name
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run a single optimization scenario")
    parser.add_argument("name", nargs="?", default="ST_BASE", help="Scenario name (or --list)")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="Gurobi thread count (default %(default)s; CLAUDE.md 二.1: runs that will be subtracted "
                             "must share it; 0 = Gurobi auto, logs a warning)")
    parser.add_argument("--time-limit", type=int, default=36000, help="Solver time limit in seconds")
    parser.add_argument("--mip-gap", type=float, default=None,
                        help="Override the scenario's MIPGap for this run only "
                             "(do NOT use on COAL_RETROFIT_GUROBI_SEED replicates; see run())")
    args = parser.parse_args()

    name = args.name
    if args.list:
        for k in EXPERIMENTS:
            print(k)
        return
    if name not in EXPERIMENTS:
        print(f"Unknown experiment: {name}. Use --list to see options.", file=sys.stderr)
        sys.exit(1)

    result = run(name, threads=args.threads, time_limit=args.time_limit,
                 mip_gap=args.mip_gap)
    out_file = ROOT / "results" / f"{name}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    obj = result["global_objective_cny"]
    status = "optimal" if not any(
        y["status"] != "optimal" for y in result["years"].values()
    ) else "WARNING"
    print(f"{name}: obj={obj:.2e}, status={status}, time={result['elapsed_seconds']}s")
    for yr, yd in sorted(result["years"].items()):
        shares = yd["pathway_shares"]
        parts = " ".join(f"{pw}={v:.1%}" for pw, v in sorted(shares.items()) if v > 0.005)
        print(f"  {yr}: {parts}")


if __name__ == "__main__":
    main()
