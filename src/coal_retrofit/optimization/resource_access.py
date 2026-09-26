"""资源侧（生物质、氨、工业氢）的链路关联矩阵与到厂成本；空间距离工具。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._shared import PreparedInputs, _nearest_year
from .scenario import OptimizationAssumptions


def _haversine_distances_km(origin_lon: float, origin_lat: float, target_lons: np.ndarray, target_lats: np.ndarray) -> np.ndarray:
    origin_lon_rad = np.radians(origin_lon)
    origin_lat_rad = np.radians(origin_lat)
    target_lons_rad = np.radians(target_lons.astype(np.float64))
    target_lats_rad = np.radians(target_lats.astype(np.float64))
    delta_lon = target_lons_rad - origin_lon_rad
    delta_lat = target_lats_rad - origin_lat_rad
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(origin_lat_rad) * np.cos(target_lats_rad) * np.sin(delta_lon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _sparse_membership(rows: list[int], link_count: int, row_count: int):
    """CSR (row_count, link_count) 指示矩阵，每条链路一列一个非零。"""
    from scipy import sparse

    cols = list(range(len(rows)))
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(row_count, link_count)
    )


# `_biomass_access_matrices` 与年份无关，但每个规划年都会调一次且结果留在 year_payloads 里；
# 只缓存最近一次结果。键含三个表的 id 与长度及本函数读的三个假设字段；owner 保住表的引用，防 id 复用。
_BIOMASS_ACCESS_CACHE: dict[str, object] = {}


def _biomass_access_matrices(prepared: PreparedInputs, assumptions: OptimizationAssumptions):
    """生物质链路关联矩阵（稀疏 CSR）与到厂成本。

    返回 `(hub_membership, node_membership, available_scaled, link_costs, scale)`。
    每条链路只碰一个厂和一个节点，CSR 约 1.7 MB，稠密曾是 7.58 GiB。
    """
    from scipy import sparse

    from ..constants import BIOMASS_FLOW_SCALE

    key = (
        id(prepared.biomass_links), len(prepared.biomass_links),
        id(prepared.plants), len(prepared.plants),
        id(prepared.biomass), len(prepared.biomass),
        float(assumptions.biomass_pretreatment_cost_cny_per_gj),
        float(assumptions.biomass_transport_fixed_cost_cny_per_gj),
        float(assumptions.biomass_transport_variable_cost_cny_per_gj_km),
    )
    if _BIOMASS_ACCESS_CACHE.get("key") == key:
        return _BIOMASS_ACCESS_CACHE["value"]

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
    out = (
        hub_membership,
        node_membership,
        available_scaled,
        link_costs,
        BIOMASS_FLOW_SCALE,
    )
    _BIOMASS_ACCESS_CACHE["key"] = key
    _BIOMASS_ACCESS_CACHE["value"] = out
    _BIOMASS_ACCESS_CACHE["owner"] = (prepared.biomass_links, prepared.plants, prepared.biomass)
    return out


def _ammonia_access_data(
    prepared: PreparedInputs, year: int, assumptions: OptimizationAssumptions
) -> dict[str, Any]:
    """煤电侧 `year` 的氨链路：稀疏关联矩阵、到厂成本、按部署进度缩放的节点供给。"""
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
        nodes["nh3_supply_kg_per_year"].astype(float).to_numpy() / AMMONIA_FLOW_SCALE * ramp
        if not nodes.empty else np.zeros(0, dtype=np.float64)
    )
    return {
        "nodes": nodes,
        "links": links,
        "hub_membership": hub_membership,
        "node_membership": node_membership,
        "available_kg": available_scaled,
        "link_cost_cny_per_kg": link_costs,
        "ammonia_flow_scale": AMMONIA_FLOW_SCALE,
        "deployment_fraction": ramp,
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
