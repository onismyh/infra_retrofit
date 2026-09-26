"""逐年结果表。成本分项与诊断（合理性检查 `_build_sanity_checks`、逐节点松弛）在本模块；
煤电厂侧、管网封存、资源、工业四类表在 `results_*` 模块，这里统一转出，调用方照旧从 `results` 导入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import PreparedInputs
from .results_industry import _build_industry_detail_table
from .results_network import (
    _alive_edge_added_stock,
    _build_co2_flow_direction_table,
    _build_edge_table,
    _build_storage_table,
)
from .results_plant import (
    _build_pathway_table,
    _build_plant_cost_table,
    _build_plant_detail_table,
    _build_province_table,
)
from .results_resources import (
    _build_ammonia_flow_table,
    _build_biomass_flow_table,
    _build_supply_table,
    _build_water_flow_table,
)
from .year_types import YearData

__all__ = [
    "_alive_edge_added_stock",
    "_build_ammonia_flow_table",
    "_build_biomass_flow_table",
    "_build_co2_flow_direction_table",
    "_build_cost_breakdown",
    "_build_edge_table",
    "_build_industry_detail_table",
    "_build_pathway_table",
    "_build_plant_cost_table",
    "_build_plant_detail_table",
    "_build_province_table",
    "_build_sanity_checks",
    "_build_slack_detail_table",
    "_build_storage_table",
    "_build_supply_table",
    "_build_water_flow_table",
]


def _build_cost_breakdown(year: int, breakdown: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"year": year, "category": category, "cost_cny": cost} for category, cost in breakdown.items()])


def _build_sanity_checks(
    year: int,
    slacks: dict[str, object],
    pathways: pd.DataFrame,
    province_table: pd.DataFrame,
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
        {"year": year, "check_name": "ammonia_overuse", "status": "warn" if float(np.sum(slacks["ammonia_slack_kg"])) > 1e-3 else "pass", "metric": "kg", "value": float(np.sum(slacks["ammonia_slack_kg"])), "threshold": 0.0, "detail": "Ammonia use should fit shared ammonia-node availability under hub competition."},
        {"year": year, "check_name": "water_overuse", "status": "warn" if float(np.sum(slacks["water_slack_m3"])) > 1e-3 else "pass", "metric": "m3", "value": float(np.sum(slacks["water_slack_m3"])), "threshold": 0.0, "detail": "Consumptive water use should fit the shared grid-water-node availability proxy under local competition."},
        # 官方指标流域上限。总是输出（水预算关闭时值为零），这样读者能区分"上限守住了"
        # 与"上限从未施加"，缺了这一行就做不到。任何正值都表示某个流域的用水总量控制指标
        # 被突破，模型付了 big-M 而没有遵守。
        {"year": year, "check_name": "water_basin_quota_breach", "status": "warn" if float(np.sum(slacks.get("water_basin_slack_m3", np.zeros(0)))) > 1e-3 else "pass", "metric": "m3", "value": float(np.sum(slacks.get("water_basin_slack_m3", np.zeros(0)))), "threshold": 0.0, "detail": "Basin withdrawal should fit the official 用水总量控制指标 net of non-power use."},
        {"year": year, "check_name": "storage_or_network_stress", "status": "warn" if float(np.sum(slacks["injectivity_slack_mtpa"]) + np.sum(slacks["storage_slack_mt"]) + np.sum(slacks["edge_slack_mtpa"])) > 1e-6 else "pass", "metric": "aggregate_slack", "value": float(np.sum(slacks["injectivity_slack_mtpa"]) + np.sum(slacks["storage_slack_mt"]) + np.sum(slacks["edge_slack_mtpa"])), "threshold": 0.0, "detail": "Transport and storage slacks indicate infeasible corridor or sink assumptions."},
        {"year": year, "check_name": "single_route_lock_in", "status": "warn" if max_path_share > 0.80 else "pass", "metric": "share", "value": max_path_share, "threshold": 0.80, "detail": "A single route dominating the annual mix may indicate lock-in."},
        {"year": year, "check_name": "province_concentration", "status": "warn" if province_peak > 0.35 else "pass", "metric": "share", "value": province_peak, "threshold": 0.35, "detail": "A single province carrying too much of the result should be reviewed."},
    ]
    return pd.DataFrame(rows)


def _build_slack_detail_table(
    prepared: PreparedInputs,
    year: int,
    year_data: YearData,
    slacks: dict[str, object],
) -> pd.DataFrame:
    """所有资源 / 基础设施约束的逐节点松弛值。"""
    rows: list[dict[str, object]] = []

    bio_slack = np.asarray(slacks["biomass_slack_gj"], dtype=np.float64)
    for i, node in enumerate(prepared.biomass.itertuples(index=False)):
        if bio_slack[i] > 1e-6:
            rows.append({"year": year, "constraint_type": "biomass_supply", "node_id": str(node.biomass_node_id), "province": str(node.province_name), "slack_value": float(bio_slack[i]), "unit": "GJ"})

    amm_slack = np.asarray(slacks["ammonia_slack_kg"], dtype=np.float64)
    amm_nodes = year_data.ammonia_nodes
    for i in range(len(amm_nodes)):
        if amm_slack[i] > 1e-6:
            rows.append({"year": year, "constraint_type": "ammonia_supply", "node_id": str(amm_nodes.iloc[i]["ammonia_node_id"]), "province": str(amm_nodes.iloc[i].get("province_name", "")), "slack_value": float(amm_slack[i]), "unit": "kg"})

    water_slack = np.asarray(slacks["water_slack_m3"], dtype=np.float64)
    water_nodes = year_data.water_nodes
    for i in range(len(water_nodes)):
        if water_slack[i] > 1e-6:
            rows.append({"year": year, "constraint_type": "water_supply", "node_id": str(water_nodes.iloc[i]["water_node_id"]), "province": str(water_nodes.iloc[i].get("province_name", "")), "slack_value": float(water_slack[i]), "unit": "m3"})

    # 官方指标流域上限。无水约束或关掉流域上限时不存在（长度为零）。
    # 与 `water_supply` 分开报告，因为它是另一口径上的另一条规则：它是取水口径上的
    # 指标分配，而非耗水口径上的环境流量限值。
    basin_slack = np.asarray(slacks.get("water_basin_slack_m3", np.zeros(0)), dtype=np.float64)
    basin_codes = list(slacks.get("water_basin_codes") or year_data.water_basin_codes or [])
    for i, code in enumerate(basin_codes[:len(basin_slack)]):
        if basin_slack[i] > 1e-6:
            rows.append({"year": year, "constraint_type": "water_basin_quota", "node_id": str(code),
                         "province": "", "slack_value": float(basin_slack[i]), "unit": "m3"})

    inj_slack = np.asarray(slacks["injectivity_slack_mtpa"], dtype=np.float64)
    for i, hub in enumerate(prepared.storages.itertuples(index=False)):
        if inj_slack[i] > 1e-9:
            rows.append({"year": year, "constraint_type": "storage_injectivity", "node_id": str(hub.storage_hub_id), "province": str(hub.province), "slack_value": float(inj_slack[i]), "unit": "Mtpa"})

    cap_slack = np.asarray(slacks["storage_slack_mt"], dtype=np.float64)
    for i, hub in enumerate(prepared.storages.itertuples(index=False)):
        if cap_slack[i] > 1e-9:
            rows.append({"year": year, "constraint_type": "storage_capacity", "node_id": str(hub.storage_hub_id), "province": str(hub.province), "slack_value": float(cap_slack[i]), "unit": "Mt"})

    edge_slack = np.asarray(slacks["edge_slack_mtpa"], dtype=np.float64)
    for i, edge in enumerate(prepared.network.edges.itertuples(index=False)):
        if edge_slack[i] > 1e-9:
            rows.append({"year": year, "constraint_type": "edge_capacity", "node_id": str(edge.edge_id), "province": "", "slack_value": float(edge_slack[i]), "unit": "Mtpa"})

    if slacks["target_shortfall_mt"] > 1e-9:
        rows.append({"year": year, "constraint_type": "emission_target", "node_id": "global", "province": "", "slack_value": float(slacks["target_shortfall_mt"]), "unit": "Mt"})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["year", "constraint_type", "node_id", "province", "slack_value", "unit"])
