"""逐年结果表。合理性检查（`_build_sanity_checks`，写进 `checks.csv`）在本模块；源、管网与封存、资源、工业各表在
`results_*` 模块，这里统一转出，调用方照旧从 `results` 导入。九张输出表由 `results_tables` 组装。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .results_industry import _build_industry_detail_table
from .results_network import _alive_edge_added_stock, _build_network_table, _build_sinks_table
from .results_plant import _build_pathway_table, _build_plant_detail_table, _build_province_table
from .results_resources import _build_resource_flow_table, _build_resource_table
from .results_sources import _build_source_route_table, _build_sources_table
from .year_types import SolveSlacks

__all__ = [
    "_alive_edge_added_stock",
    "_build_industry_detail_table",
    "_build_network_table",
    "_build_pathway_table",
    "_build_plant_detail_table",
    "_build_province_table",
    "_build_resource_flow_table",
    "_build_resource_table",
    "_build_sanity_checks",
    "_build_sinks_table",
    "_build_source_route_table",
    "_build_sources_table",
]


def _build_sanity_checks(
    year: int,
    slacks: SolveSlacks,
    pathways: pd.DataFrame,
    province_table: pd.DataFrame,
    *,
    plant_reduction_mt: np.ndarray,
    captured_mt: np.ndarray,
) -> pd.DataFrame:
    total_generation = float(pathways["annual_generation_mwh"].sum())
    path_shares = pathways.groupby("pathway", as_index=False)["annual_generation_mwh"].sum()
    max_path_share = float(path_shares["annual_generation_mwh"].max() / total_generation) if total_generation > 0 else 0.0
    province_generation = province_table.groupby("province_name", as_index=False)["annual_generation_mwh"].sum()
    province_peak = (
        float(province_generation["annual_generation_mwh"].max() / province_generation["annual_generation_mwh"].sum())
        if not province_generation.empty and float(province_generation["annual_generation_mwh"].sum()) > 0
        else 0.0
    )
    # 逐路径的减排量与捕集量（`results_plant._pathway_split` 照搬约束）逐厂相加应等于求解器的值，对不上说明约束改了、
    # 拆分没跟着改。差额除以该厂基线排放（不足 1 Mt 按 1 Mt）；容差取可行性容差 1e-6（份额合计为 1）的十倍。
    by_plant = pathways.groupby("plant_id", sort=False)[["abatement_mt", "captured_mt", "baseline_emissions_mt"]].sum()
    scale = np.maximum(by_plant["baseline_emissions_mt"].to_numpy(), 1.0)
    split_gap = max(
        float(np.max(np.abs(by_plant["abatement_mt"].to_numpy() - plant_reduction_mt) / scale, initial=0.0)),
        float(np.max(np.abs(by_plant["captured_mt"].to_numpy() - captured_mt) / scale, initial=0.0)),
    )
    rows = [
        {"year": year, "check_name": "target_shortfall", "status": "fail" if slacks["target_shortfall_mt"] > 1e-6 else "pass", "metric": "mt", "value": slacks["target_shortfall_mt"], "threshold": 0.0, "detail": "Emission target slack should remain zero."},
    ]
    # 部门碳目标下每个目标组一行，报告才能指明究竟是哪一个上限没达到。
    for group, value in sorted((slacks.get("target_shortfall_by_group") or {}).items()):
        rows.append({
            "year": year, "check_name": f"target_shortfall_{group}",
            "status": "fail" if float(value) > 1e-6 else "pass", "metric": "mt",
            "value": float(value), "threshold": 0.0,
            "detail": f"Residual cap of sector group '{group}' should be met without slack.",
        })
    rows += [
        {"year": year, "check_name": "biomass_overuse", "status": "warn" if float(np.sum(slacks["biomass_slack_gj"])) > 1e-3 else "pass", "metric": "GJ", "value": float(np.sum(slacks["biomass_slack_gj"])), "threshold": 0.0, "detail": "Biomass use should fit shared biomass-node availability within hub buffers."},
        {"year": year, "check_name": "green_h2_overuse", "status": "warn" if float(np.sum(slacks["h2_slack_kg"])) > 1e-3 else "pass", "metric": "kg H2", "value": float(np.sum(slacks["h2_slack_kg"])), "threshold": 0.0, "detail": "Green H2 drawn by coal ammonia (as H2) and industry H2 should fit the shared node H2 supply."},
        {"year": year, "check_name": "water_overuse", "status": "warn" if float(np.sum(slacks["water_slack_m3"])) > 1e-3 else "pass", "metric": "m3", "value": float(np.sum(slacks["water_slack_m3"])), "threshold": 0.0, "detail": "Consumptive water use should fit the shared grid-water-node availability proxy under local competition."},
        # 官方指标流域上限。总是输出（水预算关闭时值为零），这样读者能区分"上限守住了"
        # 与"上限从未施加"，缺了这一行就做不到。任何正值都表示某个流域的用水总量控制指标
        # 被突破，模型付了 big-M 而没有遵守。
        {"year": year, "check_name": "water_basin_quota_breach", "status": "warn" if float(np.sum(slacks.get("water_basin_slack_m3", np.zeros(0)))) > 1e-3 else "pass", "metric": "m3", "value": float(np.sum(slacks.get("water_basin_slack_m3", np.zeros(0)))), "threshold": 0.0, "detail": "Basin withdrawal should fit the official 用水总量控制指标 net of non-power use."},
        {"year": year, "check_name": "storage_or_network_stress", "status": "warn" if float(np.sum(slacks["injectivity_slack_mtpa"]) + np.sum(slacks["storage_slack_mt"]) + np.sum(slacks["edge_slack_mtpa"])) > 1e-6 else "pass", "metric": "aggregate_slack", "value": float(np.sum(slacks["injectivity_slack_mtpa"]) + np.sum(slacks["storage_slack_mt"]) + np.sum(slacks["edge_slack_mtpa"])), "threshold": 0.0, "detail": "Transport and storage slacks indicate infeasible corridor or sink assumptions."},
        {"year": year, "check_name": "pathway_split_closure", "status": "warn" if split_gap > 1e-5 else "pass", "metric": "relative", "value": split_gap, "threshold": 1e-5, "detail": "Per-pathway abatement and capture should add up to the solver's plant totals."},
        {"year": year, "check_name": "single_route_lock_in", "status": "warn" if max_path_share > 0.80 else "pass", "metric": "share", "value": max_path_share, "threshold": 0.80, "detail": "A single route dominating the annual mix may indicate lock-in."},
        {"year": year, "check_name": "province_concentration", "status": "warn" if province_peak > 0.35 else "pass", "metric": "share", "value": province_peak, "threshold": 0.35, "detail": "A single province carrying too much of the result should be reviewed."},
    ]
    return pd.DataFrame(rows)
