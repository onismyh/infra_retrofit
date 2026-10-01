"""水侧逐年数据：节点可用水量、水链路矩阵与成本、取水强度矩阵、流域取水指标。"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ._shared import PATHWAY_INDEX, PreparedInputs
from .scenario import OptimizationAssumptions, OptimizationScenario

logger = logging.getLogger(__name__)


def _water_scenario_family(mode: str) -> str:
    if str(mode) == "high_water_stress":
        return "high_pressure"
    return "baseline"  # base_water 与 grid_supply 共用


def _water_member_rows(prepared: PreparedInputs, scenario: OptimizationScenario, year: int) -> pd.DataFrame:
    """`year` 所选气候成员在 `water_availability.csv` 里的行，一个节点一行。

    输入表每行一个 (节点, 年, 气候成员)，成员 = 水文模型 x GCM x SSP。`water_scenario_id`
    选一个成员；留空则取该 family 按 id 排序的第一个成员，保证可复现。
    """
    frame = prepared.water_availability
    frame = frame[frame["planning_year"].astype(int) == int(year)]
    wanted = str(scenario.water_scenario_id or "")
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
    return frame


def _baseline_use_by_node(prepared: PreparedInputs, baseline_use_m3: np.ndarray, node_ids: pd.Series) -> np.ndarray:
    """各节点的存量：归到该节点的煤电不改造同年耗水（m3/yr），每个 hub 归它最近的节点（`distance_rank` 为 1）。"""
    links = prepared.water_links
    nearest = links[links["distance_rank"].astype(int) == 1]
    plant_index = {plant_id: idx for idx, plant_id in enumerate(prepared.plants["plant_id"].astype(str))}
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    existing = np.zeros(len(node_ids), dtype=np.float64)
    for plant_id, node_id in zip(nearest["plant_id"].astype(str), nearest["water_node_id"].astype(str)):
        if plant_id in plant_index and node_id in node_index:
            existing[node_index[node_id]] += float(baseline_use_m3[plant_index[plant_id]])
    return existing


def _water_available_by_node(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
    nodes: pd.DataFrame,
    baseline_use_m3: np.ndarray,
) -> np.ndarray:
    """`year` 各节点煤电可用水量（m3/yr）= max(生态余量, 存量) x `water_multiplier`。

    生态余量（环境流量规则，作用于耗水）：流域径流的 20%（`WATER_EXTRACTABLE_FRACTION`）先扣掉该流域的生活与灌溉
    耗水（`water_basin_use.csv`），再按节点的径流份额分到节点，即 径流_n x 0.20 − 耗水_b x 径流_n / 径流_b。径流本身就是
    按流域预算、按本地径流比例分到节点的（`builders/water.build_water_availability_dataframe`），扣减用同一口径。
    工业不扣：它是决策主体，取水进流域取水指标（`_basin_cap_data`）。

    存量（作者决定 2026-10-01，存量不增）：归到该节点的煤电不改造同年耗水（`_baseline_use_by_node`）。余量为负的
    流域里，既有用户已用尽环境流量的 20%，这些节点只给煤电留不改造时的耗水。hub 可从 200 km 内的各个节点取水，改造
    多出的耗水（如加装 CCS）要由这些节点的空余满足（空冷或退役腾出的水、余量高于存量的节点），不够的记在节点松弛上。

    `water_season` 同时决定径流列与耗水列（枯水期两者是同一个三个月窗口）。偏差校正已烘进径流列，关掉时只把径流
    除回（耗水本就未校正）。v9 的 runoff 口径曾在这里乘 (1 - 存量取水占比)，与可提取比例无法分别识别，已删除。
    """
    from ..constants import WATER_EXTRACTABLE_FRACTION

    frame = _water_member_rows(prepared, scenario, year)
    member = str(frame["scenario_id"].iloc[0])
    prefix = "dry_season_" if str(scenario.water_season).lower() == "dry" else ""
    runoff_column = "dry_season_water_m3_per_year" if prefix else "available_water_m3_per_year"
    missing = {runoff_column, "basin_code"} - set(frame.columns)
    if missing:
        raise ValueError(f"water_availability.csv lacks {sorted(missing)}")

    rows = frame.set_index(frame["water_node_id"].astype(str))
    runoff = rows[runoff_column].astype(float)
    # `bias_factor` 是乘性因子，关掉校正时精确除回（只改水平不改季节性）。
    if not float(assumptions.apply_bias_correction):
        bias = rows["bias_factor"].astype(float)
        runoff = (runoff / bias.where(bias > 0)).fillna(0.0)
    basins = rows["basin_code"].astype(str).to_numpy()
    basin_runoff = runoff.groupby(basins).sum()

    use = prepared.water_basin_use
    use = use[(use["scenario_id"].astype(str) == member) & (use["planning_year"].astype(int) == int(year))]
    use_columns = [f"{prefix}domestic_m3_per_year", f"{prefix}irrigation_m3_per_year"]
    basin_use = use.set_index(use["basin_code"].astype(str))[use_columns].astype(float).sum(axis=1)
    absent = sorted(set(basin_runoff.index) - set(basin_use.index))
    if absent:
        raise ValueError(
            f"water_basin_use.csv has no {member} {year} rows for basins {absent}; run scripts/build_water_use.py"
        )
    total = basin_runoff.reindex(basins).to_numpy()
    share = np.divide(runoff.to_numpy(), total, out=np.zeros(len(runoff)), where=total > 0)
    residual = pd.Series(
        runoff.to_numpy() * WATER_EXTRACTABLE_FRACTION - basin_use.reindex(basins).to_numpy() * share,
        index=runoff.index,
    )
    node_ids = nodes["water_node_id"].astype(str)
    residual_by_node = residual.reindex(node_ids).fillna(0.0).to_numpy()
    existing = _baseline_use_by_node(prepared, baseline_use_m3, node_ids)
    return np.maximum(residual_by_node, existing) * scenario.water_multiplier


def _withdrawal_matrices(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    water_intensity: np.ndarray,
    air_water_intensity: np.ndarray,
    year: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """逐路径取水强度（m3/MWh）及其空冷对应矩阵，供流域指标用；无水约束或关掉流域上限时 (None, None)。

    与耗水矩阵逐路径对应：捕集路径取表内带捕集取水值；掺烧路径保留原冷却系统，
    继承基线取水并乘与耗水相同的掺烧倍率；退役为零。
    """
    from ..builders.water_quota import calibrated_withdrawal_intensities

    if scenario.water_mode == "no_water":
        return None, None
    if not bool(assumptions.apply_basin_cap):
        return None, None

    plants = prepared.plants
    # 与 `province_cf` 同样按换过写法的省名查（`_prepare_plants`）。
    hours = plants["province_name"].map(assumptions.province_operating_hours)
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
    return withdrawal, air_withdrawal


def _basin_cap_data(
    prepared: PreparedInputs,
    assumptions: OptimizationAssumptions,
    scenario: OptimizationScenario,
    year: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """流域取水指标的 (成员矩阵, 余量 m3, 流域码)；无水约束或关掉流域上限时 (None, None, [])。

    机组按所在地 `plants.basin_code` 归流域（取水许可按此发放），不按取水节点所在流域。
    """
    if scenario.water_mode == "no_water":
        return None, None, []
    if not bool(assumptions.apply_basin_cap):
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


def _water_access_data(
    prepared: PreparedInputs, scenario: OptimizationScenario, assumptions: OptimizationAssumptions, year: int,
    baseline_use_m3: np.ndarray,
) -> dict[str, Any]:
    """水链路关联矩阵、按计量水量定价的链路成本、节点可用量（缩放单位）。

    `baseline_use_m3` 是各 hub 不改造同年的耗水（m3/yr），节点可用量的存量项要用。
    """
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
        available = _water_available_by_node(prepared, scenario, assumptions, year, nodes, baseline_use_m3)
    return {
        "nodes": nodes[["water_node_id", "province_name"]].copy(),
        "links": links,
        "hub_membership": hub_membership,
        "node_membership": node_membership,
        "available_m3": available / WATER_FLOW_SCALE if available is not None else None,
        "link_cost_cny_per_m3": link_cost_cny_per_m3 * WATER_FLOW_SCALE,
        "water_flow_scale": WATER_FLOW_SCALE,
    }
