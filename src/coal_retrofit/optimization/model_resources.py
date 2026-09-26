"""一年的资源约束：生物质 / 氨 / 水的链路平衡与节点上限、全国上限，及流域取水指标。"""
from __future__ import annotations

import numpy as np

from ..constants import NH3_H2_RATIO
from ._shared import gp
from .constraints import _add_vector_equality, _add_vector_upper_bound
from .scenario import PATHWAYS, OptimizationAssumptions
from .year_types import GrbExpr, GrbMVar, YearData


def add_resource_balances(
    model,
    year_data: YearData,
    assumptions: OptimizationAssumptions,
    year: int,
    plant_count: int,
    biomass_node_count: int,
    *,
    biomass_flow_gj: GrbMVar,
    ammonia_flow_kg: GrbMVar,
    water_flow_m3: GrbMVar,
    biomass_use_gj: GrbMVar,
    ammonia_use_kg: GrbMVar,
    water_use_m3: GrbMVar,
    biomass_slack_gj: GrbMVar,
    ammonia_slack_kg: GrbMVar,
    water_slack_m3: GrbMVar,
    industry_h2_flow_kg: GrbMVar,
    year_suffix: str,
) -> None:
    """链路流量与厂、节点的平衡与上限：三种资源的厂侧平衡在前，各自的节点上限与全国上限在后。"""
    ammonia_node_count = len(year_data.ammonia_nodes)
    water_node_count = len(year_data.water_nodes)
    biomass_plant_expr = year_data.biomass_link_hub_membership @ biomass_flow_gj
    biomass_node_expr = year_data.biomass_link_node_membership @ biomass_flow_gj
    ammonia_plant_expr = year_data.ammonia_link_hub_membership @ ammonia_flow_kg
    ammonia_node_expr = year_data.ammonia_link_node_membership @ ammonia_flow_kg
    water_plant_expr = year_data.water_link_hub_membership @ water_flow_m3
    water_node_expr = year_data.water_link_node_membership @ water_flow_m3

    _add_vector_equality(model, biomass_plant_expr, biomass_use_gj, plant_count, f"biomass_plant_balance_{year_suffix}")
    _add_vector_equality(model, ammonia_plant_expr, ammonia_use_kg, plant_count, f"ammonia_plant_balance_{year_suffix}")
    _add_vector_equality(model, water_plant_expr, water_use_m3, plant_count, f"water_plant_balance_{year_suffix}")
    _add_vector_upper_bound(
        model, biomass_node_expr, year_data.biomass_available + biomass_slack_gj,
        biomass_node_count, f"biomass_node_limit_{year_suffix}",
    )
    # 全国生物质上限（默认 16 EJ/yr），按求解器缩放单位。
    if assumptions.biomass_national_cap_gj_per_year > 0:
        model.addConstr(
            biomass_use_gj.sum()
            <= assumptions.biomass_national_cap_gj_per_year
            / float(year_data.biomass_flow_scale),
            name=f"biomass_national_cap_{year_suffix}",
        )
    # 工业氢与煤电氨共用节点，氢按 NH3 当量计入节点上限。
    if (
        year_data.industry_h2_node_membership is not None
        and int(industry_h2_flow_kg.shape[0]) > 0
    ):
        h2_node_draw_nh3_eq = (year_data.industry_h2_node_membership @ industry_h2_flow_kg) * (1.0 / NH3_H2_RATIO)
        ammonia_node_expr = ammonia_node_expr + h2_node_draw_nh3_eq
    _add_vector_upper_bound(
        model, ammonia_node_expr, year_data.ammonia_available_kg + ammonia_slack_kg,
        ammonia_node_count, f"ammonia_node_limit_{year_suffix}",
    )
    # 全国上限：(1) 煤电可用绿氨 Mt NH3/yr；(2) 所有用户从共享电解节点取的绿氢 Mt H2/yr。
    _nh3_scale = float(year_data.ammonia_flow_scale)
    _fleet_cap_mt = assumptions.ammonia_fleet_cap_mt(year)
    if _fleet_cap_mt > 0:
        model.addConstr(
            ammonia_use_kg.sum() <= _fleet_cap_mt * 1e9 / _nh3_scale,
            name=f"ammonia_fleet_cap_{year_suffix}",
        )
    _h2_cap_mt = assumptions.green_h2_national_cap_mt(year)
    if _h2_cap_mt > 0:
        model.addConstr(
            ammonia_node_expr.sum() * NH3_H2_RATIO <= _h2_cap_mt * 1e9 / _nh3_scale,
            name=f"green_h2_national_cap_{year_suffix}",
        )
    # 水节点上限（物理半边：环境流量规则，作用于耗水）。no_water 模式下不加。
    if year_data.water_available_m3 is not None:
        _add_vector_upper_bound(
            model, water_node_expr, year_data.water_available_m3 + water_slack_m3,
            water_node_count, f"water_node_limit_{year_suffix}",
        )


def add_basin_withdrawal_cap(
    model,
    year_data: YearData,
    share: GrbMVar,
    air_share: GrbMVar,
    industry_withdrawal_by_hub: list[GrbExpr],
    plant_count: int,
    year_suffix: str,
) -> tuple[GrbMVar | None, GrbMVar | None]:
    """流域取水指标（制度半边：用水总量控制指标，作用于取水，按流域）。

    有水约束（`water_mode` 不为 no_water）且 `apply_basin_cap` 为真时激活，条件见
    `water_access._basin_cap_data`。返回 (流域松弛, 流域取水量)，未激活时 (None, None)。
    """
    basin_membership = year_data.water_basin_membership
    water_basin_slack_m3 = None
    water_basin_use_m3 = None
    if basin_membership is not None:
        withdrawal = year_data.withdrawal_intensity
        air_withdrawal = year_data.air_withdrawal_intensity
        # 取水矩阵、流域余量与流域成员矩阵在同一组开关下生成（`_withdrawal_matrices` / `_basin_cap_data`）。
        assert withdrawal is not None and air_withdrawal is not None
        assert year_data.water_basin_available_m3 is not None
        allow_air = bool(year_data.allow_air_cooling_retrofit)
        generation = year_data.generation_by_pathway
        flow_scale = float(year_data.water_flow_scale)
        plant_withdrawal = [
            gp.quicksum(
                float(generation[plant_idx, path_idx])
                * (
                    float(withdrawal[plant_idx, path_idx]) * share[plant_idx, path_idx]
                    - (
                        float(withdrawal[plant_idx, path_idx]
                              - air_withdrawal[plant_idx, path_idx])
                        * air_share[plant_idx, path_idx]
                        if allow_air
                        else 0.0
                    )
                )
                / flow_scale
                for path_idx in range(len(PATHWAYS))
            )
            for plant_idx in range(plant_count)
        ]
        basin_count = basin_membership.shape[0]
        water_basin_slack_m3 = model.addMVar(
            basin_count, lb=0.0, name=f"water_basin_slack_m3_{year_suffix}"
        )
        # 流域取水量用命名变量而非匿名表达式，便于直接提取报告。
        water_basin_use_m3 = model.addMVar(
            basin_count, lb=0.0, name=f"water_basin_use_m3_{year_suffix}"
        )
        basin_available = year_data.water_basin_available_m3 / flow_scale
        # 工业现状取水与工业捕集的水耗与煤电共用同一余量（`write_basin_caps` 已把工业加回余量）。
        industry_basin = year_data.industry_basin_membership
        for basin_idx in range(basin_count):
            members = np.flatnonzero(basin_membership[basin_idx])
            code = year_data.water_basin_codes[basin_idx]
            basin_use = gp.quicksum(plant_withdrawal[int(p)] for p in members)
            if industry_basin is not None:
                ind_members = np.flatnonzero(industry_basin[basin_idx])
                basin_use = basin_use + gp.quicksum(
                    industry_withdrawal_by_hub[int(h)] for h in ind_members
                )
            model.addConstr(
                water_basin_use_m3[basin_idx] == basin_use,
                name=f"water_basin_use_{code}_{year_suffix}",
            )
            model.addConstr(
                water_basin_use_m3[basin_idx]
                <= float(basin_available[basin_idx]) + water_basin_slack_m3[basin_idx],
                name=f"water_basin_limit_{code}_{year_suffix}",
            )
    return water_basin_slack_m3, water_basin_use_m3
