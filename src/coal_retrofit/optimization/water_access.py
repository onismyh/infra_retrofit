"""水侧逐年数据：节点可用水量、水链路矩阵与成本、取水强度矩阵、流域取水指标。"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._shared import PATHWAY_INDEX, PreparedInputs
from .scenario import OptimizationAssumptions, OptimizationScenario

logger = logging.getLogger(__name__)


def _water_scenario_family(mode: str) -> str:
    if str(mode) == "high_water_stress":
        return "high_pressure"
    return "baseline"  # base_water 与 grid_supply 共用


def _water_available_by_node(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
    nodes: pd.DataFrame,
) -> np.ndarray:
    """`year` 各节点煤电可用水量（m3/yr）= 可再生径流 x 可提取比例 [x (1 - 存量取水占比)]。

    输入表每行一个 (节点, 年, 气候成员)，成员 = 水文模型 x GCM x SSP。`water_scenario_id`
    选一个成员；留空则取该 family 按 id 排序的第一个成员，保证可复现。

    可提取比例与存量取水占比在求解器里只以乘积出现，二者无法分别识别（0.85 x 0.20 与
    0 x 0.03 是同一个模型）；官方指标口径下分配规则移到流域指标约束，这里只剩环境流量规则。
    """
    from ..constants import WATER_EXTRACTABLE_FRACTION

    frame = prepared.water_availability
    frame = frame[frame["planning_year"].astype(int) == int(year)]
    wanted = str(getattr(scenario, "water_scenario_id", "") or "")
    if wanted:
        frame = frame[frame["scenario_id"].astype(str) == wanted]
        if frame.empty:
            raise ValueError(f"water_scenario_id {wanted!r} not present in water_availability.csv")
    else:
        family = _water_scenario_family(scenario.water_mode)
        frame = frame[frame["scenario_family"].astype(str) == family]
        members = sorted(frame["scenario_id"].astype(str).unique())
        if not members:
            raise ValueError(f"No water availability rows for family {family!r} in {year}")
        if len(members) > 1:
            logger.info("water: family %s has %d members; using %s", family, len(members), members[0])
        frame = frame[frame["scenario_id"].astype(str) == members[0]]

    season = str(getattr(scenario, "water_season", "annual")).lower()
    column = "dry_season_water_m3_per_year" if season == "dry" else "available_water_m3_per_year"
    if column not in frame.columns:
        logger.warning("water: column %s missing, falling back to annual mean", column)
        column = "available_water_m3_per_year"

    lookup = frame.set_index("water_node_id")[column].to_dict()
    # 偏差校正已烘进可用量列；`bias_factor` 是乘性因子，关掉时精确除回（只改水平不改季节性）。
    if not float(getattr(assumptions, "apply_bias_correction", True)):
        bias = frame.set_index("water_node_id")["bias_factor"].to_dict()
        lookup = {
            node_id: val / bias.get(str(node_id), 1.0)
            for node_id, val in lookup.items()
            if float(bias.get(str(node_id), 1.0)) > 0
        }
    if str(getattr(assumptions, "water_budget", "runoff")) == "official_quota":
        usable = WATER_EXTRACTABLE_FRACTION
    else:
        usable = WATER_EXTRACTABLE_FRACTION * (1.0 - float(assumptions.existing_withdrawal_share))
    return np.array(
        [float(lookup.get(str(node_id), 0.0)) * usable * scenario.water_multiplier
         for node_id in nodes["water_node_id"].astype(str)],
        dtype=np.float64,
    )


def _withdrawal_matrices(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    water_intensity: np.ndarray,
    air_water_intensity: np.ndarray,
    year: int,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """逐路径取水强度（m3/MWh）及其空冷对应矩阵，供流域指标用；未激活时 (None, None, 1.0)。

    与耗水矩阵逐路径对应：捕集路径取表内带捕集取水值；掺烧路径保留原冷却系统，
    继承基线取水并乘与耗水相同的掺烧倍率；退役为零。
    """
    from ..builders.water_quota import calibrated_withdrawal_intensities

    if str(getattr(assumptions, "water_budget", "runoff")) != "official_quota":
        return None, None, 1.0
    if scenario.water_mode == "no_water":
        return None, None, 1.0
    if not bool(getattr(assumptions, "apply_basin_cap", True)):
        return None, None, 1.0

    plants = prepared.plants
    hours = plants["province_mode"].astype(str).map(assumptions.province_operating_hours)
    hours = hours.fillna(assumptions.capacity_factor * 8760.0)
    generation = plants["total_capacity_mw"].astype(float) * hours
    base, capture, air_base, air_capture, factor = calibrated_withdrawal_intensities(
        plants, generation
    )
    base_np = base.to_numpy()
    capture_np = capture.to_numpy()

    withdrawal = np.zeros_like(water_intensity)
    withdrawal[:, PATHWAY_INDEX["unabated"]] = base_np
    withdrawal[:, PATHWAY_INDEX["retire"]] = 0.0
    withdrawal[:, PATHWAY_INDEX["ccs"]] = capture_np * scenario.ccs_water_multiplier_adjustment
    withdrawal[:, PATHWAY_INDEX["biomass"]] = base_np * assumptions.biomass_water_multiplier
    withdrawal[:, PATHWAY_INDEX["beccs"]] = capture_np * scenario.beccs_water_multiplier_adjustment
    withdrawal[:, PATHWAY_INDEX["ammonia"]] = base_np * assumptions.ammonia_water_multiplier

    air_base_np = air_base.to_numpy()
    air_capture_np = air_capture.to_numpy()
    air_withdrawal = np.zeros_like(withdrawal)
    air_withdrawal[:, PATHWAY_INDEX["unabated"]] = air_base_np
    air_withdrawal[:, PATHWAY_INDEX["retire"]] = 0.0
    air_withdrawal[:, PATHWAY_INDEX["ccs"]] = air_capture_np * scenario.ccs_water_multiplier_adjustment
    air_withdrawal[:, PATHWAY_INDEX["biomass"]] = air_base_np * assumptions.biomass_water_multiplier
    air_withdrawal[:, PATHWAY_INDEX["beccs"]] = air_capture_np * scenario.beccs_water_multiplier_adjustment
    air_withdrawal[:, PATHWAY_INDEX["ammonia"]] = air_base_np * assumptions.ammonia_water_multiplier
    air_withdrawal = np.minimum(air_withdrawal, withdrawal)

    logger.info(
        "water: official-quota budget active for %d, once-through calibration k=%.3f, "
        "fleet withdrawal/consumption at unabated = %.1fx",
        year, factor,
        float(withdrawal[:, PATHWAY_INDEX["unabated"]].sum()
              / max(water_intensity[:, PATHWAY_INDEX["unabated"]].sum(), 1e-9)),
    )
    return withdrawal, air_withdrawal, factor


def _basin_cap_data(
    prepared: PreparedInputs,
    assumptions: OptimizationAssumptions,
    scenario: OptimizationScenario,
    year: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """流域取水指标的 (成员矩阵, 余量 m3, 流域码)；未激活时 (None, None, [])。

    机组按所在地 `plants.basin_code` 归流域（取水许可按此发放），不按取水节点所在流域。
    """
    if str(getattr(assumptions, "water_budget", "runoff")) != "official_quota":
        return None, None, []
    if scenario.water_mode == "no_water":
        return None, None, []
    if not bool(getattr(assumptions, "apply_basin_cap", True)):
        logger.info("water: official-quota budget with the basin cap OFF (environmental flow only)")
        return None, None, []

    caps = prepared.water_basin_caps
    caps = caps[caps["planning_year"].astype(int) == int(year)]
    if caps.empty:
        raise ValueError(f"water_basin_caps.csv has no rows for planning year {year}")

    plant_basins = prepared.plants["basin_code"].astype(str).to_numpy()
    codes = [str(code) for code in caps["basin_code"]]
    membership = np.zeros((len(codes), len(plant_basins)), dtype=np.float64)
    for row, code in enumerate(codes):
        membership[row, :] = (plant_basins == code).astype(np.float64)
    unmatched = int(len(plant_basins) - membership.sum())
    if unmatched:
        raise ValueError(f"{unmatched} hubs fell outside every basin in water_basin_caps.csv")
    residual = caps["residual_m3_per_year"].astype(float).to_numpy()
    # 余量已含 `write_basin_caps` 加回的工业现状取水：工业是决策主体，这份水由它自己占用。
    residual = residual * scenario.water_multiplier
    return membership, residual, codes


def _water_access_data(prepared: PreparedInputs, scenario: OptimizationScenario, assumptions: OptimizationAssumptions, year: int) -> dict[str, object]:
    """水链路关联矩阵、按计量水量定价的链路成本、节点可用量（缩放单位）。"""
    from ..constants import WATER_FLOW_SCALE
    nodes = prepared.water_nodes.copy().reset_index(drop=True)
    links = prepared.water_links.copy().reset_index(drop=True)
    plant_index = {str(plant_id): idx for idx, plant_id in enumerate(prepared.plants["plant_id"].astype(str))}
    node_index = {str(node_id): idx for idx, node_id in enumerate(nodes["water_node_id"].astype(str))}
    link_count = len(links)
    hub_membership = np.zeros((len(prepared.plants), link_count), dtype=np.float64)
    node_membership = np.zeros((len(nodes), link_count), dtype=np.float64)
    # 流量变量是耗水，水费按计量指标水量收，链路成本乘该厂的 指标/耗水 比；
    # 参数扫描的加价是影子价格，作用在物理水量上。
    charge_ratio = (
        prepared.plants["water_charge_ratio"].astype(float).to_numpy()
        if "water_charge_ratio" in prepared.plants.columns
        else np.ones(len(prepared.plants), dtype=np.float64)
    )
    link_cost_cny_per_m3 = np.zeros(link_count, dtype=np.float64)
    for link_idx, link in enumerate(links.itertuples(index=False)):
        if str(link.plant_id) not in plant_index or str(link.water_node_id) not in node_index:
            continue
        plant_idx = plant_index[str(link.plant_id)]
        hub_membership[plant_idx, link_idx] = 1.0
        node_membership[node_index[str(link.water_node_id)], link_idx] = 1.0
        if hasattr(link, "delivered_cost_cny_per_m3"):
            link_cost_cny_per_m3[link_idx] = float(link.delivered_cost_cny_per_m3) * charge_ratio[plant_idx]
        link_cost_cny_per_m3[link_idx] += float(scenario.water_price_adder_cny_per_m3)

    if scenario.water_mode == "no_water":
        available = None
    else:
        available = _water_available_by_node(prepared, scenario, assumptions, year, nodes)
    return {
        "nodes": nodes[["water_node_id", "province_name"]].copy(),
        "links": links,
        "hub_membership": hub_membership,
        "node_membership": node_membership,
        "available_m3": available / WATER_FLOW_SCALE if available is not None else None,
        "link_cost_cny_per_m3": link_cost_cny_per_m3 * WATER_FLOW_SCALE,
        "water_flow_scale": WATER_FLOW_SCALE,
    }
