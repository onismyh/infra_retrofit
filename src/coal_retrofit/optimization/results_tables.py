"""一次求解的九张结果表（2026-10-10 起，取代此前的 15 张 CSV），按源、汇、管网、资源、系统分组。

| 表 | 一行是 | 构建 |
|---|---|---|
| `sources.csv` | 年 x 源（煤电 hub、工业 hub） | `results_sources._build_sources_table` |
| `source_routes.csv` | 年 x 源 x 路线 | `results_sources._build_source_route_table` |
| `sinks.csv` | 年 x 封存汇 | `results_network._build_sinks_table` |
| `network.csv` | 年 x 管段 | `results_network._build_network_table` |
| `resources.csv` | 年 x 资源节点（含流域取水指标） | `results_resources._build_resource_table` |
| `resource_flows.csv` | 年 x 资源链路（流量大于 1e-3 的） | `results_resources._build_resource_flow_table` |
| `costs.csv` | 年 x 实体 x 成本类别 x 细项 | `results_costs.build_costs_table` |
| `system.csv` | 年 x 成本类别 | `results_costs.build_system_table` |
| `checks.csv` | 年 x 检查项 | `results._build_sanity_checks`，另加成本对账一行 |

约定：数量的单位写在列名后缀（`_mtpa` Mt/yr，排放、捕集、注入、管道流量与能力等年量；`_mt` Mt，封存容量、剩余与累计
注入等存量；`_gj` GJ/yr、`_kg` kg/yr、`_m3` m3/yr、`_mw` MW、`_km` km、`_cny` 元）；一列里混有几种单位的（活动量、
资源量、资源流量）另有单位列。成本列 `cost_cny` 是不折现的当年值，`cost_discounted_cny` 是进目标函数的折现值
（`costs.csv` 的定义）；年度项是一年的费用，一次性项是当年的支出，`kind` 列区分，跨年加总前年度项要乘区间年数。
`sources`、`network`、`sinks` 的这两列是该实体在 `costs.csv` 里各行之和（两类混在一起），没有成本的实体为 0。
CSV 一律 utf-8-sig 编码。
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._shared import PreparedInputs, SolveState
from .results import (
    _alive_edge_added_stock,
    _build_industry_detail_table,
    _build_network_table,
    _build_pathway_table,
    _build_plant_detail_table,
    _build_province_table,
    _build_resource_flow_table,
    _build_resource_table,
    _build_sanity_checks,
    _build_sinks_table,
    _build_source_route_table,
    _build_sources_table,
)
from .results_costs import build_costs_table, build_system_table, cost_closure
from .scenario import OptimizationAssumptions, OptimizationScenario

TABLES = ("sources", "source_routes", "sinks", "network", "resources", "resource_flows", "costs", "system", "checks")
# 成本对账的容差：拆到各实体的合计与求解器合计的最大相对差（`results_costs.cost_closure`）。
CLOSURE_TOL = 1e-6
# 各表并入成本合计时，实体号所在的列与 `costs.csv` 的实体类型；源表的实体类型逐行取 `source_type`（记 None）。
_COST_KEYS = {"sources": ("source_id", None), "network": ("edge_id", "edge"), "sinks": ("sink_id", "sink")}


def build_result_tables(
    prepared: PreparedInputs,
    solution: Mapping[str, Any],
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    node_province: Mapping[str, str],
    state: SolveState,
) -> dict[str, pd.DataFrame]:
    """全部规划年的九张表，键见 `TABLES`。*state* 是求解开始时的管网与封存状态（`run_controls.initial_state`），
    这里逐年推进：在役管道只数寿命内的（`_alive_edge_added_stock`），剩余封存容量逐区间扣注入量。"""
    years = [int(y) for y in solution["year_solutions"]]
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES if name not in ("costs", "system")}
    life = int(assumptions.pipeline_lifetime_years)
    new_cap_by_year: dict[int, np.ndarray] = {}  # 逐年新增管道容量
    pipes_by_year: dict[int, np.ndarray] = {}  # 逐年逐边逐管径档新铺的整根管数
    state = state.clone()
    for index, year in enumerate(years):
        ys = solution["year_solutions"][year]
        yd = ys["year_data"]
        interval = scenario.interval_years(tuple(years), index, assumptions)
        state.edge_added_stock_mtpa = _alive_edge_added_stock(new_cap_by_year, year, life, len(prepared.network.edges))
        before = state.clone()
        pipes_by_year[year] = np.rint(ys["pipe_count"])
        in_service = pipes_by_year[year] + _alive_edge_added_stock(pipes_by_year, year, life, pipes_by_year[year].shape)

        pathways = _build_pathway_table(
            prepared, scenario, year, ys["share"],
            ys["biomass_blend_x_share"], ys["beccs_blend_x_share"], ys["ammonia_blend_x_share"],
            year_data=yd, air_share=ys["air_share"],
            rebuilt_share=ys["rebuilt_share"], rebuilt_blend_x_share=ys["rebuilt_blend_x_share"],
        )
        plant = _build_plant_detail_table(
            prepared, scenario, year, ys["share"],
            ys["captured_mt_by_plant"], ys["biomass_use_gj"], ys["ammonia_use_kg"],
            ys["water_use_m3"], ys["blend_level_b"], ys["blend_level_a"],
            ys["air_share"], year_data=yd, plant_reduction_mt=ys["plant_reduction_mt"],
            biomass_blend_x_share=ys["biomass_blend_x_share"], beccs_blend_x_share=ys["beccs_blend_x_share"],
            ammonia_blend_x_share=ys["ammonia_blend_x_share"], air_installed=ys["air_installed"],
        )
        industry = _build_industry_detail_table(
            prepared, year, yd.industry, ys["industry_share"], h2_flow_kg=ys["industry_h2_flow_kg"], year_data=yd,
            capacity_mt=ys["industry_capacity_mt"], new_capacity_mt=ys["industry_new_capacity_mt"],
        )
        parts["sources"].append(_build_sources_table(prepared, ys, plant, industry))
        parts["source_routes"].append(_build_source_route_table(prepared, ys, year, pathways))
        parts["sinks"].append(_build_sinks_table(prepared, year, ys, before, interval, node_province))
        parts["network"].append(_build_network_table(prepared, year, ys, before, assumptions, in_service, node_province))
        parts["resources"].append(_build_resource_table(prepared, year, ys))
        parts["resource_flows"].append(_build_resource_flow_table(prepared, year, ys))
        parts["checks"].append(_build_sanity_checks(
            year, ys["slacks"], pathways, _build_province_table(pathways),
            plant_reduction_mt=ys["plant_reduction_mt"], captured_mt=ys["captured_mt_by_plant"],
        ))

        new_cap_by_year[year] = ys["new_cap_mtpa"]
        state.remaining_storage_mt = np.maximum(0.0, state.remaining_storage_mt - ys["storage_use_mtpa"] * interval)

    tables = {name: pd.concat(frames, ignore_index=True, sort=False) for name, frames in parts.items()}
    costs = build_costs_table(prepared, solution, scenario, assumptions)
    system = build_system_table(solution, costs)
    for name, (key, entity_type) in _COST_KEYS.items():
        tables[name] = _with_cost_totals(tables[name], costs, key, entity_type)
    tables["checks"] = pd.concat([tables["checks"], _closure_checks(system)], ignore_index=True, sort=False)
    tables["costs"], tables["system"] = costs, system
    return {name: tables[name] for name in TABLES}


def write_result_tables(out_dir: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    """九张表写成 `<out_dir>/<表>.csv`，utf-8-sig 编码（Excel 直接打开不乱码）。"""
    for name, frame in tables.items():
        frame.to_csv(Path(out_dir) / f"{name}.csv", index=False, encoding="utf-8-sig")


def _with_cost_totals(frame: pd.DataFrame, costs: pd.DataFrame, key: str, entity_type: str | None) -> pd.DataFrame:
    """并入每个实体每年的成本合计（`cost_cny`、`cost_discounted_cny`，`costs.csv` 各行之和），没有成本行的记 0。
    *entity_type* 是这张表全体的实体类型；为 None 时逐行取 `source_type` 列（源表里煤电与工业混排）。"""
    types = frame["source_type"].astype(str) if entity_type is None else pd.Series(entity_type, index=frame.index)
    totals = costs.groupby(["year", "entity_type", "entity_id"])[["cost_cny", "cost_discounted_cny"]].sum()
    index = pd.MultiIndex.from_arrays([frame["year"].astype(int), types, frame[key].astype(str)])
    out = frame.copy()
    for column in ("cost_cny", "cost_discounted_cny"):
        out[column] = totals[column].reindex(index).fillna(0.0).to_numpy()
    return out


def _closure_checks(system: pd.DataFrame) -> pd.DataFrame:
    """每年一行 `cost_attribution_closure`：各类别拆到实体的合计与求解器合计的最大相对差（`results_costs.cost_closure`）。"""
    gaps = cost_closure(system)
    return pd.DataFrame({
        "year": gaps.index.astype(int), "check_name": "cost_attribution_closure",
        "status": np.where(gaps.to_numpy() > CLOSURE_TOL, "fail", "pass"), "metric": "relative",
        "value": gaps.to_numpy(), "threshold": CLOSURE_TOL,
        "detail": "Per-entity costs (costs.csv) should add up to the solver's cost of each category.",
    })
