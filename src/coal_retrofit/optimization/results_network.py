"""CO2 管网与封存结果表：`network.csv`（逐边容量、流量与流向）与 `sinks.csv`（逐汇注入、剩余容量与成本）。"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ._shared import PreparedInputs, SolveState
from .scenario import OptimizationAssumptions
from .year_types import YearSolution


def _alive_edge_added_stock(
    new_cap_by_year: dict[int, np.ndarray], year: int, lifetime_years: int, shape: int | tuple[int, ...]
) -> np.ndarray:
    """`year` 年仍在寿命内的往期新增管道容量（Mtpa），不含 `year` 本年的新增；*shape* 是每年数组的形状。

    与求解器 `edge_capacity_limit` 同口径：`year - 建成年 < lifetime_years` 的管才在役。到寿命的管
    不再计入存量，它在原址重建的容量记在重建那一年的新增里。逐边逐管径档的根数同法累计（`results_tables.build_result_tables`）。
    """
    stock = np.zeros(shape, dtype=np.float64)
    for built_year, new_cap in new_cap_by_year.items():
        if int(built_year) < int(year) and int(year) - int(built_year) < int(lifetime_years):
            stock += np.asarray(new_cap, dtype=np.float64)
    return stock


NETWORK_COLUMNS = [
    "year", "edge_id", "from_node_id", "to_node_id", "from_province", "to_province", "length_km", "offshore_length_km",
    "corridor_type", "existing_corridor_flag", "edge_class", "available_stock_before_mtpa", "build_selected",
    "edge_active", "flow_fwd_mtpa", "flow_bwd_mtpa", "flow_mtpa", "net_flow_mtpa", "flow_from_node_id",
    "flow_to_node_id", "new_capacity_mtpa", "total_capacity_mtpa", "pipes_new_by_tier", "pipes_in_service_by_tier",
    "edge_slack_mtpa", "source", "year_basis",
]
SINK_COLUMNS = [
    "year", "sink_id", "storage_type", "offshore", "province", "longitude", "latitude", "injectivity_mtpa",
    "capacity_mt", "remaining_before_mt", "injected_mtpa", "remaining_after_mt", "cumulative_injected_mt",
    "injectivity_utilization", "storage_cost_before_credit_cny_per_t", "storage_cost_cny_per_t", "eor_credit_cny",
    "injectivity_slack_mtpa", "storage_slack_mt", "source", "year_basis",
]


def _build_network_table(
    prepared: PreparedInputs,
    year: int,
    ys: YearSolution,
    state_before: SolveState,
    assumptions: OptimizationAssumptions,
    pipes_in_service: np.ndarray,
    node_province: Mapping[str, str],
) -> pd.DataFrame:
    """逐边一行（含没有流量的边）。

    - 流量：`flow_fwd_mtpa`、`flow_bwd_mtpa` 是沿边定义方向（from → to）与反方向的流量，`flow_mtpa` 是两者之和（受管道
      容量约束的量，`model_year`），`net_flow_mtpa` = |正向 − 反向|，`flow_from_node_id` → `flow_to_node_id` 是净流向，
      没有流量的边为空。2026-10-10 前流向在另一张表 `co2_flow_direction.csv`。
    - 管道：`pipes_new_by_tier` 是本年按管径档铺设的整根管数，`pipes_in_service_by_tier`（2026-10-02 起）是本年在役的
      根数（寿命内的往年新铺加本年新铺，`_alive_edge_added_stock`），写法同为 "根数x档容量"，如 "1x2|1x20"。
      在役根数乘档容量的合计 = `total_capacity_mtpa` 减既有走廊容量。
    - `from_province`、`to_province`（2026-10-02 起）是起止节点的省（`results_regions.node_provinces`，海上记 Offshore）；
      `offshore_length_km` 是边落在陆地省界之外的长度（按陆上 1.5 倍计价的那段，`builders.network_offshore`）。
    - `edge_slack_mtpa` 是管道容量约束的松弛（应为零）。
    """
    edges = prepared.network.edges.copy()
    fwd = np.asarray(ys["co2_flow_fwd"], dtype=np.float64)
    bwd = np.asarray(ys["co2_flow_bwd"], dtype=np.float64)
    moving = fwd + bwd >= 1e-9
    forward = fwd >= bwd
    from_ids, to_ids = edges["from_node_id"].astype(str), edges["to_node_id"].astype(str)
    edges["year"] = year
    edges["from_province"] = from_ids.map(node_province)
    edges["to_province"] = to_ids.map(node_province)
    edges["available_stock_before_mtpa"] = (
        edges["existing_corridor_flag"].fillna(0).astype(float) * assumptions.existing_corridor_capacity_mtpa
        + state_before.edge_added_stock_mtpa
    )
    edges["build_selected"] = np.rint(np.asarray(ys["build_edge"], dtype=np.float64)).astype(int)
    edges["flow_fwd_mtpa"], edges["flow_bwd_mtpa"] = fwd, bwd
    edges["flow_mtpa"] = np.asarray(ys["edge_flow_mtpa"], dtype=np.float64)
    edges["net_flow_mtpa"] = np.abs(fwd - bwd)
    edges["flow_from_node_id"] = np.where(moving, np.where(forward, from_ids, to_ids), "")
    edges["flow_to_node_id"] = np.where(moving, np.where(forward, to_ids, from_ids), "")
    edges["new_capacity_mtpa"] = np.asarray(ys["new_cap_mtpa"], dtype=np.float64)
    edges["total_capacity_mtpa"] = edges["available_stock_before_mtpa"] + edges["new_capacity_mtpa"]
    edges["edge_active"] = ((edges["flow_mtpa"] > 1e-6) | (edges["new_capacity_mtpa"] > 1e-6)).astype(int)
    pipe_tiers = tuple(ys["year_data"].pipe_tiers_mtpa)
    edges["pipes_new_by_tier"] = _tier_strings(ys["pipe_count"], pipe_tiers)
    edges["pipes_in_service_by_tier"] = _tier_strings(pipes_in_service, pipe_tiers)
    edges["edge_slack_mtpa"] = np.asarray(ys["slacks"]["edge_slack_mtpa"], dtype=np.float64)
    return edges.reindex(columns=NETWORK_COLUMNS)


def _tier_strings(counts: np.ndarray, pipe_tiers: tuple[float, ...]) -> list[str]:
    """逐边的 "根数x档容量" 串，档之间用 | 隔开，没有管的边为空串；根数按最近整数取（求解器的整数变量带容差）。"""
    whole = np.rint(np.asarray(counts, dtype=np.float64)).astype(int)
    return [
        "|".join(f"{int(row[k])}x{pipe_tiers[k]:g}" for k in range(len(pipe_tiers)) if row[k] > 0)
        for row in whole
    ]


def _build_sinks_table(
    prepared: PreparedInputs,
    year: int,
    ys: YearSolution,
    state_before: SolveState,
    interval_years: int,
    node_province: Mapping[str, str],
) -> pd.DataFrame:
    """逐汇一行。

    - `storage_type` 是 dsa / eor，`offshore` 是海上汇的标记（`storage_hubs.csv`，没有这一列的输入记 False），`province`
      是汇所在节点的省（海上记 Offshore，`results_regions.node_provinces`）。
    - `injectivity_mtpa` 是单汇年注入能力上限（Fan 2025 逐格之和 x `injectivity_multiplier`，不随年份变，即约束所用的值；
      全国合计另受逐年部署上限），`capacity_mt` 是累计
      封存容量；`remaining_before_mt` / `remaining_after_mt` 是本区间注入前后的剩余容量（注入量 x 区间年数），
      `cumulative_injected_mt` = 容量 − 区间末剩余。
    - 每吨封存成本：`storage_cost_before_credit_cny_per_t` 是扣 EOR 抵扣前（按汇型与陆海定价），`storage_cost_cny_per_t`
      是扣后、即目标函数所用（`data_prep._prepare_storages`）；`eor_credit_cny` = 两者之差 x 注入量，CNY/yr。
    - `injectivity_slack_mtpa`、`storage_slack_mt` 是注入能力与累计容量约束的松弛（应为零）。
    """
    storages = prepared.storages
    ids = storages["storage_hub_id"].astype(str)
    use = np.asarray(ys["storage_use_mtpa"], dtype=np.float64)
    injectivity = np.asarray(ys["year_data"].storage_injectivity_mtpa, dtype=np.float64)
    capacity = storages["available_capacity_mt"].astype(float).to_numpy()
    remaining_after = np.maximum(0.0, state_before.remaining_storage_mt - use * interval_years)
    before_credit = storages["storage_cost_before_credit_cny_per_t"].astype(float).to_numpy()
    after_credit = storages["storage_cost_cny_per_t"].astype(float).to_numpy()
    table = pd.DataFrame({
        "year": year, "sink_id": ids, "storage_type": storages["storage_type"].astype(str),
        "offshore": storages["offshore"].astype(bool) if "offshore" in storages else False,
        "province": ids.map(prepared.network.storage_node_ids).map(node_province),
        "longitude": storages["longitude"], "latitude": storages["latitude"],
        "injectivity_mtpa": injectivity, "capacity_mt": capacity,
        "remaining_before_mt": state_before.remaining_storage_mt, "injected_mtpa": use,
        "remaining_after_mt": remaining_after, "cumulative_injected_mt": capacity - remaining_after,
        "injectivity_utilization": np.divide(use, injectivity, out=np.zeros(len(use)), where=injectivity > 0),
        "storage_cost_before_credit_cny_per_t": before_credit, "storage_cost_cny_per_t": after_credit,
        "eor_credit_cny": (before_credit - after_credit) * 1e6 * use,
        "injectivity_slack_mtpa": np.asarray(ys["slacks"]["injectivity_slack_mtpa"], dtype=np.float64),
        "storage_slack_mt": np.asarray(ys["slacks"]["storage_slack_mt"], dtype=np.float64),
        "source": storages.get("source", ""), "year_basis": storages.get("year_basis", ""),
    })
    return table.reindex(columns=SINK_COLUMNS)
