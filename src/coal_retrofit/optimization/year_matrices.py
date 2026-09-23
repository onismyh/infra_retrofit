"""组装单个规划年的全部系数：厂侧矩阵、资源与水链路、管网边系数、封存注入速率、部门上限。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._shared import PATHWAY_INDEX, PreparedInputs, SolveState
from .plant_matrices import _air_cooling_matrices, _plant_operating_matrices, _water_intensity_matrices
from .resource_access import _ammonia_access_data, _biomass_access_matrices, _industry_h2_access_data
from .scenario import OptimizationAssumptions, OptimizationScenario
from .water_access import _basin_cap_data, _water_access_data, _withdrawal_matrices
from .year_types import YearData


def _offshore_edge_mask(prepared: PreparedInputs) -> np.ndarray:
    """触及海上封存 hub 的边（海底管道 / 平台）。"""
    edges = prepared.network.edges
    storages = prepared.storages
    if "offshore" not in storages.columns:
        return np.zeros(len(edges), dtype=bool)
    offshore_hubs = {
        str(hub_id)
        for hub_id, flag in zip(storages["storage_hub_id"], storages["offshore"].astype(bool))
        if flag
    }
    offshore_nodes = {
        node_id
        for hub_id, node_id in prepared.network.storage_node_ids.items()
        if str(hub_id) in offshore_hubs
    }
    if not offshore_nodes:
        return np.zeros(len(edges), dtype=bool)
    return (
        edges["from_node_id"].astype(str).isin(offshore_nodes)
        | edges["to_node_id"].astype(str).isin(offshore_nodes)
    ).to_numpy()


def _edge_capex_multiplier(edge_class: str, existing_flag: int, assumptions: OptimizationAssumptions) -> float:
    label = str(edge_class)
    if label == "runtime_direct_fallback":
        return assumptions.direct_fallback_capex_multiplier
    if existing_flag and label == "existing_main_corridor":
        return assumptions.corridor_capex_multiplier
    if label == "triangulation_candidate":
        return 1.0
    return assumptions.branch_capex_multiplier


def _edge_matrices(
    prepared: PreparedInputs, assumptions: OptimizationAssumptions, state: SolveState
) -> dict[str, Any]:
    """管网边：存量与可新增容量、各管径档单根 capex、运输运维系数（均含海上倍率）。"""
    edge_base_stock = (
        prepared.network.edges["existing_corridor_flag"].fillna(0).astype(float).to_numpy() * assumptions.existing_corridor_capacity_mtpa
        + state.edge_added_stock_mtpa
    )
    edge_max_total = np.full(len(prepared.network.edges), assumptions.standard_pipe_capacity_mtpa * assumptions.max_parallel_pipes)
    edge_max_new = np.maximum(0.0, edge_max_total - edge_base_stock)
    edge_length_km = prepared.network.edges["length_km"].astype(float).to_numpy()
    offshore_edges = _offshore_edge_mask(prepared)
    offshore_factor = np.where(offshore_edges, assumptions.offshore_transport_multiplier, 1.0)
    edge_class_multiplier = prepared.network.edges.apply(
        lambda row: _edge_capex_multiplier(str(row["edge_class"]), int(row["existing_corridor_flag"]), assumptions),
        axis=1,
    ).astype(float).to_numpy()
    # 每 Mtpa 系数只用于报告；求解器按下面的管径档整根建。
    edge_capex_coeff = (
        edge_length_km * assumptions.pipe_capex_cny_per_mtpa_km * edge_class_multiplier * offshore_factor
    )
    tiers = tuple(float(t) for t in assumptions.pipe_capacity_tiers_mtpa)
    tier_capex_per_km = tuple(float(c) for c in assumptions.pipe_capex_cny_per_km_by_tier)
    if len(tiers) != len(tier_capex_per_km) or not tiers:
        raise ValueError("pipe_capacity_tiers_mtpa and pipe_capex_cny_per_km_by_tier must be non-empty and equal length")
    # (n_edges, n_tiers)：每条边每档一根管的 CNY。
    edge_tier_capex = (
        (edge_length_km * edge_class_multiplier * offshore_factor)[:, None]
        * np.asarray(tier_capex_per_km, dtype=np.float64)[None, :]
    )
    # 每条边的运输运维系数（CNY / Mt 流量）。
    edge_route_opex_coeff = (
        edge_length_km * assumptions.route_opex_cny_per_t_km * 1_000_000.0 * offshore_factor
    )
    return {
        "edge_base_stock_mtpa": edge_base_stock,
        "edge_max_new_mtpa": edge_max_new,
        "edge_min_build_mtpa": np.minimum(edge_max_new, float(min(tiers))),
        "edge_capex_coeff": edge_capex_coeff,
        "pipe_tiers_mtpa": tiers,
        "edge_tier_capex": edge_tier_capex,
        "edge_route_opex_coeff": edge_route_opex_coeff,
        "edge_offshore_mask": offshore_edges,
    }


def _build_year_matrices(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
    state: SolveState,
) -> YearData:
    biomass_hub_membership, biomass_node_membership, biomass_available, biomass_link_costs, biomass_flow_scale = _biomass_access_matrices(prepared, assumptions)
    from .industry import basin_membership as _industry_basin_membership
    from .industry import industry_year_data

    industry_basin_membership = None
    industry_data = industry_year_data(prepared.industry, scenario, assumptions, year)
    ammonia_data = _ammonia_access_data(prepared, year, assumptions)
    industry_h2_data = _industry_h2_access_data(prepared, year, assumptions, ammonia_data["nodes"])
    water_data = _water_access_data(prepared, scenario, assumptions, year)

    plant = _plant_operating_matrices(prepared, scenario, assumptions, year)
    water_intensity, air_water_intensity = _water_intensity_matrices(prepared, scenario, assumptions)
    # 取水孪生矩阵与流域指标：只在 official_quota 下建（流域指标是唯一按取水计的约束）。
    withdrawal_intensity, air_withdrawal_intensity, once_through_factor = _withdrawal_matrices(
        prepared, scenario, assumptions, water_intensity, air_water_intensity, year
    )
    basin_membership, basin_residual, basin_codes = _basin_cap_data(
        prepared, assumptions, scenario, year
    )
    if basin_codes:
        industry_basin_membership = _industry_basin_membership(prepared.industry, basin_codes)
    air = _air_cooling_matrices(prepared, scenario, assumptions, plant)
    edges = _edge_matrices(prepared, assumptions, state)

    coal_price_per_plant = plant["coal_price_per_plant"]
    coal_savings_per_gj = coal_price_per_plant * biomass_flow_scale  # 缩放后 CNY/TJ
    # 掺氨替代的煤，CNY / 缩放 kg NH3：LHV x 煤价。氨列的 baseline_net 按全煤热耗计费，没有此项会为没烧的煤付钱。
    coal_savings_per_kg_nh3 = (
        coal_price_per_plant * float(assumptions.nh3_lhv_gj_per_kg) * float(ammonia_data.get("ammonia_flow_scale", 1.0))
    )
    # 封存：2060 规模的可建注入速率 x 本年部署进度。
    storage_injectivity_mtpa = (
        prepared.storages["injectivity_mtpa"].astype(float).to_numpy()
        * float(assumptions.storage_deployment_fraction(int(year)))
    )
    # 本年各组上限，各组自身 2030 基线的比例。
    rows = prepared.sector_targets[prepared.sector_targets["planning_year"].astype(int) == int(year)]
    if rows.empty:
        raise ValueError(f"sector_targets_{scenario.sector_target_source}.csv has no rows for {year}")
    sector_cap_fraction: dict[str, float] = {
        str(row.sector_group): float(row.cap_fraction_of_2030) for row in rows.itertuples(index=False)
    }
    # 改造存量的 capex 系数，(plant_count, 1)：只有捕集岛一列，按 CCS capex 计（见 `model_year`）。
    capex_matrix = np.asarray(plant["ccs_retrofit_capex_matrix"], dtype=np.float64)
    retrofit_stock_capex = capex_matrix[:, [PATHWAY_INDEX["ccs"]]]

    return YearData(
        **{k: v for k, v in plant.items() if k not in ("generation_cost_basis", "coal_price_per_plant")},
        retrofit_stock_capex=retrofit_stock_capex,
        carbon_price=scenario.carbon_price_for_year(year),
        coal_savings_per_gj=coal_savings_per_gj,
        coal_savings_per_kg_nh3=coal_savings_per_kg_nh3,
        storage_injectivity_mtpa=storage_injectivity_mtpa,
        storage_deployment_fraction=float(assumptions.storage_deployment_fraction(int(year))),
        sector_cap_fraction=sector_cap_fraction,
        industry_h2_links=industry_h2_data["links"],
        industry_h2_hub_membership=industry_h2_data["hub_membership"],
        industry_h2_node_membership=industry_h2_data["node_membership"],
        industry_h2_link_cost_cny_per_kg=industry_h2_data["link_cost_cny_per_kg"],
        biomass_link_hub_membership=biomass_hub_membership,
        biomass_link_node_membership=biomass_node_membership,
        biomass_available=biomass_available,
        biomass_link_cost_cny_per_gj=biomass_link_costs,
        biomass_flow_scale=biomass_flow_scale,
        biomass_nodes=prepared.biomass[["biomass_node_id", "province_name"]].copy(),
        ammonia_year=int(ammonia_data["year"]),
        ammonia_nodes=ammonia_data["nodes"][["ammonia_node_id", "province_name"]].copy() if not ammonia_data["nodes"].empty else pd.DataFrame(columns=["ammonia_node_id", "province_name"]),
        ammonia_links=ammonia_data["links"],
        ammonia_link_hub_membership=ammonia_data["hub_membership"],
        ammonia_link_node_membership=ammonia_data["node_membership"],
        ammonia_available_kg=ammonia_data["available_kg"],
        ammonia_link_cost_cny_per_kg=ammonia_data["link_cost_cny_per_kg"],
        water_nodes=water_data["nodes"],
        water_links=water_data["links"],
        water_link_hub_membership=water_data["hub_membership"],
        water_link_node_membership=water_data["node_membership"],
        water_available_m3=water_data["available_m3"],
        water_link_cost_cny_per_m3=water_data["link_cost_cny_per_m3"],
        water_intensity=water_intensity,
        air_water_intensity=air_water_intensity,
        withdrawal_intensity=withdrawal_intensity,
        air_withdrawal_intensity=air_withdrawal_intensity,
        once_through_calibration=once_through_factor,
        water_basin_membership=basin_membership,
        water_basin_available_m3=basin_residual,
        water_basin_codes=basin_codes,
        industry=industry_data,
        # (n_basins, n_industry_hubs)，与 `water_basin_membership` 分开：求解器用 flatnonzero 把后者转成厂索引，
        # 共用索引空间会让 hub 索引被当成厂索引而不报错。
        industry_basin_membership=industry_basin_membership,
        **air,
        ammonia_flow_scale=ammonia_data.get("ammonia_flow_scale", 1.0),
        water_flow_scale=water_data.get("water_flow_scale", 1.0),
        **edges,
    )
