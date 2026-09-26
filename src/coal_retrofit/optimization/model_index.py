"""跨年共享的索引：节点编号、源/汇/管网节点集合、关联矩阵、各部门 2030 基线。"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..constants_industry import POWER_TARGET_GROUP
from ._shared import PATHWAY_INDEX, PreparedInputs
from .scenario import OptimizationAssumptions, OptimizationScenario

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelIndex:
    plant_count: int
    storage_count: int
    edge_count: int
    n_nodes: int
    biomass_node_count: int
    retirement_years: np.ndarray
    node_idx_dict: dict[str, int]
    plant_ids: list[str]
    storage_ids: list[str]
    industry_hub_ids: list[str]
    pipeline_indices: list[int]
    incidence: np.ndarray
    sector_base_2030: dict[str, float]
    # `retrofit_installed` 每一列按哪条路径的 capex 计价，即结果里的列顺序。只有捕集岛一列
    # （ccs + beccs 份额），按 CCS 计价；2026-09-23 前另有 BECCS 增量一列。
    capex_pathway_indices: tuple[int, ...]


def build_model_index(
    prepared: PreparedInputs, scenario: OptimizationScenario, assumptions: OptimizationAssumptions
) -> ModelIndex:
    from .industry import industry_year_data

    n_nodes = len(prepared.network.nodes)
    retirement_years = prepared.plants["retirement_year"].astype(int).to_numpy()
    node_idx_dict = {
        str(row.node_id): i
        for i, row in enumerate(prepared.network.nodes.itertuples(index=False))
    }
    plant_ids = prepared.plants["plant_id"].astype(str).tolist()
    storage_ids = prepared.storages["storage_hub_id"].astype(str).tolist()
    plant_n_indices: set[int] = set()
    for pid in plant_ids:
        if pid in prepared.network.plant_node_ids:
            plant_n_indices.add(node_idx_dict[prepared.network.plant_node_ids[pid]])
    storage_n_indices: set[int] = set()
    for sid in storage_ids:
        if sid in prepared.network.storage_node_ids:
            storage_n_indices.add(node_idx_dict[prepared.network.storage_node_ids[sid]])
    # 工业 hub 是源，必须先进 source_sink 集合再推导管网节点集合；否则它会落进管网集合、
    # 被加上 `co2_node_outflow == 0`，工业捕集量就凭空消失。
    industry_n_indices: set[int] = set()
    industry_hub_ids = prepared.industry.hubs["hub_id"].astype(str).tolist()
    missing_hubs = [
        hub_id for hub_id in industry_hub_ids
        if hub_id not in prepared.network.industry_node_ids
    ]
    if missing_hubs:
        # 未注册的源 hub 没有节点平衡把捕集量接进管网，其 CO2 就成了零成本处置。
        raise ValueError(
            f"{len(missing_hubs)} industrial hub(s) are absent from network.industry_node_ids "
            f"(first few: {missing_hubs[:5]}); their captured CO2 would vanish at zero cost"
        )
    for hub_id in industry_hub_ids:
        mapped_node = prepared.network.industry_node_ids[hub_id]
        n_idx = node_idx_dict.get(mapped_node)
        if n_idx is None:
            raise ValueError(
                f"industry_node_ids maps hub {hub_id!r} -> {mapped_node!r} which is not in network nodes"
            )
        industry_n_indices.add(n_idx)
    source_sink_indices = plant_n_indices | storage_n_indices | industry_n_indices
    pipeline_indices = [i for i in range(n_nodes) if i not in source_sink_indices]

    # 各组自身的 2030 冻结技术排放，上限按其比例给。与逐年矩阵走同一数据路径，
    # 无论 2030 是否在规划年里，基线都与 2030 约束所见一致。
    sector_base_2030: dict[str, float] = {}
    fleet_hours_now = float(prepared.plants["fleet_hours_now"].iloc[0]) if "fleet_hours_now" in prepared.plants.columns else 0.0
    scale_2030 = float(scenario.operating_hours_scale(2030, fleet_hours_now)) if fleet_hours_now > 0 else 1.0
    sector_base_2030[POWER_TARGET_GROUP] = float(
        prepared.plants["baseline_emissions_mt"].astype(float).sum() * scale_2030
    )
    base_2030 = industry_year_data(prepared.industry, scenario, assumptions, 2030)
    groups = base_2030.target_groups
    for group in sorted(set(str(g) for g in groups)):
        sector_base_2030[str(group)] = float(
            base_2030.baseline_emissions_mt[groups == group].sum()
        )
    logger.info("sector targets from %s; 2030 baselines (Mt): %s",
                scenario.sector_target_source,
                {g: round(v, 1) for g, v in sector_base_2030.items()})

    return ModelIndex(
        plant_count=len(prepared.plants),
        storage_count=len(prepared.storages),
        edge_count=len(prepared.network.edges),
        n_nodes=n_nodes,
        biomass_node_count=len(prepared.biomass),
        retirement_years=retirement_years,
        node_idx_dict=node_idx_dict,
        plant_ids=plant_ids,
        storage_ids=storage_ids,
        industry_hub_ids=industry_hub_ids,
        pipeline_indices=pipeline_indices,
        incidence=prepared.network.incidence,
        sector_base_2030=sector_base_2030,
        capex_pathway_indices=(PATHWAY_INDEX["ccs"],),
    )
