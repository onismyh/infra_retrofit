"""逐年容器：`YearData` 是一个规划年的全部系数，`YearPayload` 是这一年的变量与表达式。

两者此前都是 `dict[str, object]`：键名拼错要到运行到那一行才报错，读者也看不出有哪些键。
字段名与原字典键一一对应。工业侧（两者的 `industry` 字段）同样是 dataclass：
`industry_matrices.IndustryYearData` 与 `model_industry.IndustryPayload`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import gurobipy as gp
    from scipy import sparse

    from .industry_matrices import IndustryYearData
    from .model_industry import IndustryPayload

# gurobipy 13 的存根把 MVar 的标量下标 `x[i, j]` 与 `x.sum()` 标成 MVar / MLinExpr，而 `quicksum`、
# `LinExpr.__iadd__` 的存根只收 `float | Var | LinExpr`；运行时它们是 0 维对象，照常参与求和。
# 字段若直接标 `gp.MVar`，mypy 会在几十处正确的建模代码上报假阳性，所以按 Any 放行，别名只说明字段是什么。
GrbMVar = Any
GrbExpr = Any  # LinExpr、0 维 MLinExpr 或 float


@dataclass(frozen=True)
class YearData:
    """`_build_year_matrices` 的输出。煤电厂侧的路径矩阵形状 (plant_count, len(PATHWAYS))、列序同 PATHWAYS；
    其余形状不同的字段在旁边注明。"""

    # --- 煤电厂侧（`plant_matrices._plant_operating_matrices`）---
    hours_scale: float
    generation: np.ndarray
    generation_by_pathway: np.ndarray
    emissions_mt: np.ndarray
    emissions_operating_mt: np.ndarray
    emissions_retrofit_mt: np.ndarray
    heat_rate_eff: np.ndarray
    capacity_mw: np.ndarray
    fixed_cost_matrix: np.ndarray
    energy_penalty_matrix: np.ndarray
    biomass_penalty_coeff_per_level: np.ndarray
    ccs_penalty_emissions_matrix: np.ndarray
    ccs_penalty_captured_matrix: np.ndarray
    biomass_penalty_emissions_coeff_per_level: np.ndarray
    beccs_penalty_emissions_coeff_per_level: np.ndarray
    beccs_penalty_captured_coeff_per_level: np.ndarray
    ccs_retrofit_capex_matrix: np.ndarray
    ccs_om_matrix: np.ndarray
    baseline_net_matrix: np.ndarray
    stranded_per_plant: np.ndarray
    # 改造存量的 capex 系数 (plant_count, 1)：只有捕集岛一列，按 CCS capex 计（见 `model_year`）。
    retrofit_stock_capex: np.ndarray

    # --- 价格、封存部署、部门上限 ---
    carbon_price: float
    coal_savings_per_gj: np.ndarray
    coal_savings_per_kg_nh3: np.ndarray
    storage_injectivity_mtpa: np.ndarray
    storage_deployment_fraction: float
    sector_cap_fraction: dict[str, float]

    # --- 工业氢链路（与煤电氨共用节点；本年输入没有链路时两个关联矩阵为 None，
    #     有链路但节点或厂址都对不上时为 0 列稀疏矩阵）---
    industry_h2_links: pd.DataFrame
    industry_h2_hub_membership: sparse.csr_matrix | None
    industry_h2_node_membership: sparse.csr_matrix | None
    industry_h2_link_cost_cny_per_kg: np.ndarray

    # --- 生物质 ---
    biomass_link_hub_membership: sparse.csr_matrix
    biomass_link_node_membership: sparse.csr_matrix
    biomass_available: np.ndarray
    biomass_link_cost_cny_per_gj: np.ndarray
    biomass_flow_scale: float

    # --- 氨 ---
    ammonia_nodes: pd.DataFrame
    ammonia_links: pd.DataFrame
    ammonia_link_hub_membership: sparse.csr_matrix
    ammonia_link_node_membership: sparse.csr_matrix
    ammonia_available_kg: np.ndarray
    ammonia_link_cost_cny_per_kg: np.ndarray

    # --- 水：节点与链路、耗水与取水强度、流域指标（未激活时为 None / 空）---
    water_nodes: pd.DataFrame
    water_links: pd.DataFrame
    water_link_hub_membership: np.ndarray
    water_link_node_membership: np.ndarray
    water_available_m3: np.ndarray | None
    water_link_cost_cny_per_m3: np.ndarray
    water_intensity: np.ndarray
    air_water_intensity: np.ndarray
    withdrawal_intensity: np.ndarray | None
    air_withdrawal_intensity: np.ndarray | None
    water_basin_membership: np.ndarray | None
    water_basin_available_m3: np.ndarray | None
    water_basin_codes: list[str]

    # --- 工业（`industry_matrices.industry_year_data` 的输出）与其流域成员矩阵 (n_basins, n_industry_hubs) ---
    industry: IndustryYearData
    industry_basin_membership: np.ndarray | None

    # --- 湿冷→空冷改造（`plant_matrices._air_cooling_matrices`）---
    air_retrofit_capex_per_plant: np.ndarray
    air_penalty_emissions_matrix: np.ndarray
    air_penalty_captured_matrix: np.ndarray
    air_penalty_cost_matrix: np.ndarray
    allow_air_cooling_retrofit: bool

    # --- 求解器缩放单位 ---
    ammonia_flow_scale: float
    water_flow_scale: float

    # --- 管网边（`year_matrices._edge_matrices`）---
    edge_base_stock_mtpa: np.ndarray
    edge_max_new_mtpa: np.ndarray
    pipe_tiers_mtpa: tuple[float, ...]
    edge_tier_capex: np.ndarray
    edge_route_opex_coeff: np.ndarray


@dataclass
class YearPayload:
    """`add_year_block` 的输出：一年的变量、表达式与系数。

    `cost_exprs`、`salvage_ledger`、`objective_expr` 由 `add_year_costs` 写入，
    `_add_salvage_credit` 再补残值项并重算目标，所以这个类不冻结。
    """

    year: int
    interval_years: int
    year_data: YearData
    share: GrbMVar
    co2_flow_fwd: GrbMVar
    co2_flow_bwd: GrbMVar
    build_edge: GrbMVar
    add_cap: GrbMVar
    pipe_count: GrbMVar
    rebuild: GrbMVar
    retrofit_installed: GrbMVar
    new_cap_mtpa: GrbMVar
    biomass_flow_gj: GrbMVar
    ammonia_flow_kg: GrbMVar
    water_flow_m3: GrbMVar
    target_shortfall_mt: gp.Var
    target_shortfall_by_group: dict[str, gp.Var]
    biomass_slack_gj: GrbMVar
    ammonia_slack_kg: GrbMVar
    water_slack_m3: GrbMVar
    # 流域指标未激活时为 None。
    water_basin_slack_m3: GrbMVar | None
    water_basin_use_m3: GrbMVar | None
    injectivity_slack_mtpa: GrbMVar
    storage_slack_mt: GrbMVar
    edge_slack_mtpa: GrbMVar
    edge_flow_mtpa: GrbMVar
    storage_use_mtpa: GrbMVar
    captured_mt_by_plant: GrbMVar
    biomass_use_gj: GrbMVar
    ammonia_use_kg: GrbMVar
    water_use_m3: GrbMVar
    air_share: GrbMVar
    air_installed: GrbMVar
    select_b: GrbMVar
    select_a: GrbMVar
    blend_level_b: GrbMVar
    blend_level_a: GrbMVar
    plant_reduction_exprs: list[GrbExpr]
    total_reduction_mt: GrbExpr
    total_bio_penalty: GrbExpr
    # `model_industry.add_industry_year` 的输出。
    industry: IndustryPayload
    # 成本类别 -> 折现并缩放后的表达式（碳价为零时碳成本是 0.0）。
    cost_exprs: dict[str, GrbExpr] = field(default_factory=dict)
    # (名称, 未折现 capex 表达式, 经济寿命年)，供期末残值。
    salvage_ledger: list[tuple[str, GrbExpr, int]] = field(default_factory=list)
    # `add_year_costs` 之前为 None。
    objective_expr: GrbExpr = None
