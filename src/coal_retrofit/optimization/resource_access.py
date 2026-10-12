"""资源侧（生物质、氨、工业氢）的链路关联矩阵与到厂成本。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._shared import PreparedInputs, _nearest_year
from .assumptions import OptimizationAssumptions


def _sparse_membership(rows: list[int], link_count: int, row_count: int):
    """CSR (row_count, link_count) 指示矩阵，每条链路一列一个非零。"""
    from scipy import sparse

    cols = list(range(len(rows)))
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(row_count, link_count)
    )


@dataclass(frozen=True, eq=False)   # eq=False：字段里有稀疏矩阵，自动生成的 __eq__ 会逐元素比较，按对象同一性即可
class BiomassAccess:
    """生物质链路关联矩阵（稀疏 CSR）与到厂成本。与年份无关：`solver` 每次求解建一次，各规划年共用。

    每条链路只碰一个厂和一个节点，CSR 约 1.7 MB（稠密曾是 7.58 GiB）。成本与可用量都已按 `flow_scale` 缩放。
    """

    hub_membership: Any            # CSR (plant_count, link_count)
    node_membership: Any           # CSR (biomass_node_count, link_count)
    available_scaled: np.ndarray   # 节点可用量 / flow_scale
    link_costs: np.ndarray         # 到厂成本（采购 + 预处理 + 运输）x flow_scale
    flow_scale: float


def _biomass_access_matrices(prepared: PreparedInputs, assumptions: OptimizationAssumptions) -> BiomassAccess:
    from scipy import sparse

    from ..constants import BIOMASS_FLOW_SCALE

    plant_index = {str(plant_id): idx for idx, plant_id in enumerate(prepared.plants["plant_id"].astype(str))}
    node_index = {str(node_id): idx for idx, node_id in enumerate(prepared.biomass["biomass_node_id"].astype(str))}
    link_count = len(prepared.biomass_links)
    hub_rows: list[int] = []
    hub_cols: list[int] = []
    node_rows: list[int] = []
    node_cols: list[int] = []
    link_costs = np.zeros(link_count, dtype=np.float64)
    for link_idx, link in enumerate(prepared.biomass_links.itertuples(index=False)):
        if str(link.plant_id) not in plant_index or str(link.biomass_node_id) not in node_index:
            continue
        hub_rows.append(plant_index[str(link.plant_id)])
        hub_cols.append(link_idx)
        node_rows.append(node_index[str(link.biomass_node_id)])
        node_cols.append(link_idx)
        # 到厂成本 = 采购 + 预处理 + 运输（固定 + 距离 x 变动）
        purchase_cost = float(link.cost_cny_per_gj)
        dist_km = float(link.distance_km) if hasattr(link, "distance_km") else 0.0
        transport_cost = (
            assumptions.biomass_pretreatment_cost_cny_per_gj
            + assumptions.biomass_transport_fixed_cost_cny_per_gj
            + assumptions.biomass_transport_variable_cost_cny_per_gj_km * dist_km
        )
        link_costs[link_idx] = (purchase_cost + transport_cost) * BIOMASS_FLOW_SCALE
    hub_membership = sparse.csr_matrix(
        (np.ones(len(hub_rows), dtype=np.float64), (hub_rows, hub_cols)),
        shape=(len(prepared.plants), link_count),
    )
    node_membership = sparse.csr_matrix(
        (np.ones(len(node_rows), dtype=np.float64), (node_rows, node_cols)),
        shape=(len(prepared.biomass), link_count),
    )
    available_scaled = prepared.biomass["available_gj"].astype(float).to_numpy() / BIOMASS_FLOW_SCALE
    return BiomassAccess(hub_membership, node_membership, available_scaled, link_costs, BIOMASS_FLOW_SCALE)


def _ammonia_access_data(
    prepared: PreparedInputs, year: int, assumptions: OptimizationAssumptions
) -> dict[str, Any]:
    """煤电侧 `year` 的氨链路：稀疏关联矩阵、到厂成本、按部署进度缩放的节点绿氢供给。

    节点是绿氢的产地（风光电解），上限只按绿氢计（`h2_supply_kg_per_year`，2026-10-10 起；此前按折成氨的
    `nh3_supply_kg_per_year` 计，另有一条全国绿氨上限）：煤电用的氨在节点上由绿氢合成，按 `NH3_H2_RATIO` 折成氢
    从同一上限里扣，工业氢路线直接扣（`model_resources.add_resource_balances`）。到厂的氨价 = 绿氢 x 折算系数
    + 合成岛电耗与 capex 年金 + 储存（`nh3_cost_lb_usd_per_kg`，`builders.supply_ammonia`）+ 运输。
    """
    from ..constants import AMMONIA_FLOW_SCALE
    source_year = _nearest_year(year, prepared.available_ammonia_years)
    nodes = prepared.ammonia_supply[prepared.ammonia_supply["year"].astype(int) == int(source_year)].copy()
    nodes = nodes.reset_index(drop=True)
    links = prepared.ammonia_links[prepared.ammonia_links["year"].astype(int) == int(source_year)].copy()
    links = links.reset_index(drop=True)
    node_index = {str(node_id): idx for idx, node_id in enumerate(nodes["ammonia_node_id"].astype(str))}
    plant_index = {str(plant_id): idx for idx, plant_id in enumerate(prepared.plants["plant_id"].astype(str))}
    link_count = len(links)
    hub_rows: list[int] = []
    node_rows: list[int] = []
    link_costs = np.zeros(link_count, dtype=np.float64)
    keep = np.zeros(link_count, dtype=bool)
    for link_idx, link in enumerate(links.itertuples(index=False)):
        if str(link.ammonia_node_id) not in node_index or str(link.plant_id) not in plant_index:
            # 丢弃的链路不能留零列，否则求解器会把它读成不受约束的自由流量。
            continue
        keep[link_idx] = True
        hub_rows.append(plant_index[str(link.plant_id)])
        node_rows.append(node_index[str(link.ammonia_node_id)])
        # 到厂成本 = 生产成本 + 距离 x 公路运输
        base_cost = float(link.cost_cny_per_kg)
        dist_km = float(link.distance_km) if hasattr(link, "distance_km") else 0.0
        transport_cost = assumptions.ammonia_transport_cost_cny_per_kg_km * dist_km
        link_costs[link_idx] = (base_cost + transport_cost) * AMMONIA_FLOW_SCALE
    if not keep.all():
        links = links.loc[keep].reset_index(drop=True)
        link_costs = link_costs[keep]
        link_count = len(links)
    hub_membership = _sparse_membership(hub_rows, link_count, len(prepared.plants))
    node_membership = _sparse_membership(node_rows, link_count, len(nodes))
    ramp = float(assumptions.ammonia_supply_deployment_fraction(int(year)))
    available_scaled = (
        nodes["h2_supply_kg_per_year"].astype(float).to_numpy() / AMMONIA_FLOW_SCALE * ramp
        if not nodes.empty else np.zeros(0, dtype=np.float64)
    )
    return {
        "nodes": nodes,
        "links": links,
        "hub_membership": hub_membership,
        "node_membership": node_membership,
        "h2_available_kg": available_scaled,
        "link_cost_cny_per_kg": link_costs,
        "ammonia_flow_scale": AMMONIA_FLOW_SCALE,
    }


def _industry_h2_access_data(
    prepared: PreparedInputs,
    year: int,
    assumptions: OptimizationAssumptions,
    ammonia_nodes: pd.DataFrame,
) -> dict[str, Any]:
    """工业氢链路：与煤电氨共用节点。氢流量按缩放 kg 计；链路成本 = 节点 LCOH + 管束车运输。"""
    from ..constants import AMMONIA_FLOW_SCALE

    empty = {
        "links": pd.DataFrame(columns=["year", "hub_id", "ammonia_node_id", "distance_km", "lcoh_usd_per_kg"]),
        "hub_membership": None,
        "node_membership": None,
        "link_cost_cny_per_kg": np.zeros(0, dtype=np.float64),
    }
    if prepared.industry_h2_links.empty:
        return empty
    source_year = _nearest_year(year, prepared.available_ammonia_years)
    links = prepared.industry_h2_links[
        prepared.industry_h2_links["year"].astype(int) == int(source_year)
    ].reset_index(drop=True)
    if links.empty:
        return empty
    node_index = {str(node_id): idx for idx, node_id in enumerate(ammonia_nodes["ammonia_node_id"].astype(str))}
    hub_index = {str(hub_id): idx for idx, hub_id in enumerate(prepared.industry.hubs["hub_id"].astype(str))}
    keep = np.zeros(len(links), dtype=bool)
    hub_rows: list[int] = []
    node_rows: list[int] = []
    costs = np.zeros(len(links), dtype=np.float64)
    for link_idx, link in enumerate(links.itertuples(index=False)):
        if str(link.ammonia_node_id) not in node_index or str(link.hub_id) not in hub_index:
            continue
        keep[link_idx] = True
        hub_rows.append(hub_index[str(link.hub_id)])
        node_rows.append(node_index[str(link.ammonia_node_id)])
        delivered = (
            float(link.lcoh_usd_per_kg) * float(assumptions.usd_to_cny)
            + float(assumptions.h2_transport_cost_cny_per_kg_km) * float(link.distance_km)
        )
        costs[link_idx] = delivered * AMMONIA_FLOW_SCALE
    links = links.loc[keep].reset_index(drop=True)
    costs = costs[keep]
    return {
        "links": links,
        "hub_membership": _sparse_membership(hub_rows, len(links), len(prepared.industry.hubs)),
        "node_membership": _sparse_membership(node_rows, len(links), len(ammonia_nodes)),
        "link_cost_cny_per_kg": costs,
    }
