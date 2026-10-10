"""求解后提取逐年解；求解失败时给同结构的零填充结果。"""
from __future__ import annotations

import numpy as np

from ..constants_industry import INDUSTRY_ROUTES
from ._shared import (
    _COST_SCALE,
    PreparedInputs,
    _expr_value,
    _var_scalar_value,
    _var_value,
)
from .industry_matrices import CCS, H2
from .model_index import ModelIndex
from .scenario import PATHWAYS
from .year_types import GrbExpr, RebuiltShares, YearPayload, YearSolution


def empty_year_solutions(
    prepared: PreparedInputs, idx: ModelIndex, year_payloads: list[YearPayload], status: str
) -> dict[int, YearSolution]:
    """与 `extract_year_solutions` 同键的零填充结果，供调用方识别失败。"""
    plant_count, storage_count, edge_count = idx.plant_count, idx.storage_count, idx.edge_count
    hub_count = len(prepared.industry.hubs)
    return {
        int(p.year): {
            "status": status,
            "objective_cny": 0.0,
            "share": np.zeros((plant_count, len(PATHWAYS))),
            "build_edge": np.zeros(edge_count),
            "rebuild": np.zeros(plant_count),
            "rebuild_class": np.zeros(p.year_data.rebuilt_class_share.shape),
            "new_cap_mtpa": np.zeros(edge_count),
            "edge_flow_mtpa": np.zeros(edge_count),
            "storage_use_mtpa": np.zeros(storage_count),
            "water_use_m3": np.zeros(plant_count),
            "water_flow_m3": np.zeros(len(p.year_data.water_links)),
            "biomass_use_gj": np.zeros(plant_count),
            "biomass_flow_gj": np.zeros(len(prepared.biomass_links)),
            "ammonia_use_kg": np.zeros(plant_count),
            "ammonia_flow_kg": np.zeros(len(p.year_data.ammonia_links)),
            "captured_mt_by_plant": np.zeros(plant_count),
            "air_share": np.zeros((plant_count, len(PATHWAYS))),
            "air_installed": np.zeros(plant_count),
            "blend_level_b": np.zeros(plant_count),
            "blend_level_a": np.zeros(plant_count),
            "retrofit_new": np.zeros((plant_count, len(idx.capex_pathway_indices))),
            "retrofit_alive": np.zeros(plant_count),
            "ccs_om_by_plant": np.zeros(plant_count),
            "stranded_by_plant": np.zeros(plant_count),
            "pipe_count": np.zeros((edge_count, len(p.year_data.pipe_tiers_mtpa))),
            "industry_share": np.zeros((hub_count, len(INDUSTRY_ROUTES))),
            "industry_new_capacity_mt": np.zeros((hub_count, len(INDUSTRY_ROUTES))),
            "industry_capacity_mt": np.zeros((hub_count, len(INDUSTRY_ROUTES))),
            "industry_ccs_om_by_hub": np.zeros(hub_count),
            "industry_h2_flow_kg": np.zeros(0),
            "plant_reduction_mt": np.zeros(plant_count),
            "biomass_blend_x_share": np.zeros(plant_count),
            "beccs_blend_x_share": np.zeros(plant_count),
            "ammonia_blend_x_share": np.zeros(plant_count),
            "rebuilt_share": np.zeros(_rebuilt_shape(p, plant_count)),
            "rebuilt_air_share": np.zeros(_rebuilt_shape(p, plant_count)),
            "rebuilt_blend_x_share": np.zeros(_rebuilt_shape(p, plant_count)),
            "bio_penalty_by_plant": np.zeros(plant_count),
            "total_reduction_mt": 0.0,
            "co2_flow_fwd": np.zeros(edge_count),
            "co2_flow_bwd": np.zeros(edge_count),
            "cost_breakdown_cny": {k: 0.0 for k in list(year_payloads[0].cost_exprs.keys())},
            "cost_weights": dict(p.cost_weights),
            "salvage_ledger": [],
            "slacks": {
                "target_shortfall_mt": 0.0,
                "target_shortfall_by_group": {},
                "biomass_slack_gj": np.zeros(len(prepared.biomass)),
                "h2_slack_kg": np.zeros(len(p.year_data.ammonia_nodes)),
                "water_slack_m3": np.zeros(len(p.year_data.water_nodes)),
                "water_basin_slack_m3": np.zeros(
                    len(p.year_data.water_basin_codes or [])),
                "water_basin_use_m3": np.zeros(
                    len(p.year_data.water_basin_codes or [])),
                "water_basin_codes": list(p.year_data.water_basin_codes or []),
                "injectivity_slack_mtpa": np.zeros(storage_count),
                "storage_slack_mt": np.zeros(storage_count),
                "edge_slack_mtpa": np.zeros(edge_count),
            },
            "year_data": p.year_data,
        }
        for p in year_payloads
    }


def extract_year_solutions(
    prepared: PreparedInputs, idx: ModelIndex, year_payloads: list[YearPayload], status: str
) -> dict[int, YearSolution]:
    """读出每年的变量值。资源流量按缩放因子乘回原单位（GJ、kg、m3）。"""
    plant_count, storage_count, edge_count = idx.plant_count, idx.storage_count, idx.edge_count
    hub_count = len(prepared.industry.hubs)
    year_solutions: dict[int, YearSolution] = {}
    for payload in year_payloads:
        year = int(payload.year)
        year_data = payload.year_data
        biomass_node_count = len(prepared.biomass)
        ammonia_node_count = len(year_data.ammonia_nodes)
        water_node_count = len(year_data.water_nodes)
        _amm_s = float(year_data.ammonia_flow_scale)
        _wat_s = float(year_data.water_flow_scale)
        _bio_s = float(year_data.biomass_flow_scale)
        industry = payload.industry
        island, air, industry_ccs, industry_h2 = (
            payload.ccs_island, payload.air_cooling, payload.industry_ccs, payload.industry_h2
        )
        assert island is not None and air is not None and industry_ccs is not None and industry_h2 is not None
        industry_alive = np.zeros((hub_count, len(INDUSTRY_ROUTES)))
        industry_alive[:, CCS] = _values(industry_ccs.alive)
        industry_alive[:, H2] = _values(industry_h2.alive)
        year_solutions[year] = {
            "status": status,
            "objective_cny": _expr_value(payload.objective_expr) * _COST_SCALE,
            "share": _var_value(payload.share, (plant_count, len(PATHWAYS))),
            "build_edge": _var_value(payload.build_edge, edge_count),
            "rebuild": _var_value(payload.rebuild, plant_count),
            "rebuild_class": _var_value(payload.rebuild_class, tuple(payload.rebuild_class.shape)),
            "new_cap_mtpa": _var_value(payload.new_cap_mtpa, edge_count),
            "edge_flow_mtpa": _var_value(payload.edge_flow_mtpa, edge_count),
            "co2_flow_fwd": _var_value(payload.co2_flow_fwd, edge_count),
            "co2_flow_bwd": _var_value(payload.co2_flow_bwd, edge_count),
            "storage_use_mtpa": _var_value(payload.storage_use_mtpa, storage_count),
            "water_use_m3": _var_value(payload.water_use_m3, plant_count) * _wat_s,
            "water_flow_m3": _var_value(payload.water_flow_m3, len(year_data.water_links)) * _wat_s,
            "biomass_use_gj": _var_value(payload.biomass_use_gj, plant_count) * _bio_s,
            "biomass_flow_gj": _var_value(payload.biomass_flow_gj, len(prepared.biomass_links)) * _bio_s,
            "ammonia_use_kg": _var_value(payload.ammonia_use_kg, plant_count) * _amm_s,
            "ammonia_flow_kg": _var_value(payload.ammonia_flow_kg, len(year_data.ammonia_links)) * _amm_s,
            "captured_mt_by_plant": _var_value(payload.captured_mt_by_plant, plant_count),
            "air_share": _var_value(payload.air_share, (plant_count, len(PATHWAYS))),
            "air_installed": _values(air.alive),
            "blend_level_b": _var_value(payload.blend_level_b, plant_count),
            "blend_level_a": _var_value(payload.blend_level_a, plant_count),
            "retrofit_new": _var_value(payload.retrofit_new, (plant_count, len(idx.capex_pathway_indices))),
            "retrofit_alive": _values(island.alive),
            "ccs_om_by_plant": _values(island.fixed_om),
            "stranded_by_plant": _values(payload.stranded_by_plant),
            "pipe_count": _var_value(payload.pipe_count, (edge_count, len(year_data.pipe_tiers_mtpa))),
            "industry_share": _var_value(industry.share, (hub_count, len(INDUSTRY_ROUTES))),
            "industry_new_capacity_mt": _var_value(industry.new_capacity_mt, (hub_count, len(INDUSTRY_ROUTES))),
            "industry_capacity_mt": industry_alive,
            "industry_ccs_om_by_hub": _values(industry_ccs.fixed_om),
            "industry_h2_flow_kg": _var_value(industry.h2_flow_kg, int(industry.h2_flow_kg.shape[0])) * _amm_s,
            # 逐厂减排量直接取约束自身的表达式，报告不必从 share 反推。
            "plant_reduction_mt": np.array(
                [_expr_value(expr) for expr in payload.plant_reduction_exprs], dtype=np.float64
            ),
            # 逐厂 Σβ·z，结果表除以路径份额得有效掺烧比例（连续 hub 下 blend_level 换算不出比例）。
            "biomass_blend_x_share": np.array([_expr_value(e) for e in payload.biomass_blend_x_share], dtype=np.float64),
            "beccs_blend_x_share": np.array([_expr_value(e) for e in payload.beccs_blend_x_share], dtype=np.float64),
            "ammonia_blend_x_share": np.array([_expr_value(e) for e in payload.ammonia_blend_x_share], dtype=np.float64),
            "rebuilt_share": _rebuilt_values(payload.rebuilt_share, _rebuilt_shape(payload, plant_count)),
            "rebuilt_air_share": _rebuilt_values(payload.rebuilt_air_share, _rebuilt_shape(payload, plant_count)),
            "rebuilt_blend_x_share": _rebuilt_blend_values(payload, plant_count),
            "bio_penalty_by_plant": _values(payload.bio_penalty_by_plant),
            "total_reduction_mt": _expr_value(payload.total_reduction_mt),
            "cost_breakdown_cny": {category: _expr_value(expr) * _COST_SCALE for category, expr in payload.cost_exprs.items()},
            "cost_weights": dict(payload.cost_weights),
            "salvage_ledger": [(name, _expr_value(expr), life) for name, expr, life in payload.salvage_ledger],
            "slacks": {
                "target_shortfall_mt": _var_scalar_value(payload.target_shortfall_mt),
                "target_shortfall_by_group": {
                    group: _var_scalar_value(var)
                    for group, var in payload.target_shortfall_by_group.items()
                },
                "biomass_slack_gj": _var_value(payload.biomass_slack_gj, biomass_node_count) * _bio_s,
                "h2_slack_kg": _var_value(payload.h2_slack_kg, ammonia_node_count) * _amm_s,
                "water_slack_m3": _var_value(payload.water_slack_m3, water_node_count) * _wat_s,
                # 流域指标未激活时为零长度数组，下游读者统一按数组处理。
                "water_basin_slack_m3": (
                    _var_value(payload.water_basin_slack_m3,
                               len(year_data.water_basin_codes)) * _wat_s
                    if payload.water_basin_slack_m3 is not None else np.zeros(0)),
                "water_basin_codes": list(year_data.water_basin_codes or []),
                "water_basin_use_m3": (
                    _var_value(payload.water_basin_use_m3,
                               len(year_data.water_basin_codes)) * _wat_s
                    if payload.water_basin_use_m3 is not None else np.zeros(0)),
                "injectivity_slack_mtpa": _var_value(payload.injectivity_slack_mtpa, storage_count),
                "storage_slack_mt": _var_value(payload.storage_slack_mt, storage_count),
                "edge_slack_mtpa": _var_value(payload.edge_slack_mtpa, edge_count),
            },
            "year_data": year_data,
        }
    return year_solutions


def _values(exprs: list[GrbExpr]) -> np.ndarray:
    """逐单元表达式（分代能力 `vintage.StockYear`、逐厂搁浅资产与生物质惩罚）的解值；常数 0.0 读作 0。"""
    return np.array([_expr_value(expr) for expr in exprs], dtype=np.float64)


def _rebuilt_shape(payload: YearPayload, plant_count: int) -> tuple[int, int, int]:
    """重建部分各数组的形状：(重建热耗类数, plant_count, len(PATHWAYS))。"""
    return len(payload.year_data.rebuilt_deltas), plant_count, len(PATHWAYS)


def _rebuilt_values(parts: RebuiltShares, shape: tuple[int, int, int]) -> np.ndarray:
    """各路径份额或空冷份额里各类的重建部分（`constraints._add_rebuilt_split`），(类数, plant_count, len(PATHWAYS))。"""
    out = np.zeros(shape)
    for p, classes in parts.items():
        for c, part in classes.items():
            for k, var in part.items():
                out[c, p, k] = _var_scalar_value(var)
    return out


def _rebuilt_blend_values(payload: YearPayload, plant_count: int) -> np.ndarray:
    """掺烧三列上各类重建部分的 Σβ_l·rz_l，(类数, plant_count, len(PATHWAYS))。"""
    out = np.zeros(_rebuilt_shape(payload, plant_count))
    for (p, c, k), expr in payload.rebuilt_blend_x_share.items():
        out[c, p, k] = _expr_value(expr)
    return out
