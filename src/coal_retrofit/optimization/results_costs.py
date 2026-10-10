"""逐实体成本表 `costs.csv` 与系统成本表 `system.csv`（2026-10-10 起）。

目标函数的每个成本类别（`model_costs.add_year_costs` 的 `cost_exprs`）按求解得到的变量值拆到承担它的实体上：煤电 hub
（`coal`）、工业 hub（`industry`）、管段（`edge`）、封存汇（`sink`），松弛惩罚与期末残值记在 `system`。每行是
（年, 实体, 类别, 细项）的不折现值 `cost_cny` 与进目标函数的折现值 `cost_discounted_cny`（不折现值 x 该类别的折现权重
`YearPayload.cost_weights`）。各式与 `model_costs` 同式，系数取同一组 `YearData`；按年按类别相加应等于求解器的
`cost_breakdown_cny`，`cost_closure` 给出差额，`system.csv` 两列并排。

细项（`item`）只在一个类别有几个来源时细分：能耗惩罚分捕集额外燃料、空冷背压、生物质效率损失；增量运维分路径运维与两类
掺烧能力的固定运维；掺烧升级分生物质与氨；工业运行费分捕集运行、捕集固定运维、氢路线；运输运维分按流量的一项与管道固定
运维；松弛惩罚按松弛种类；残值按资产类别（台账名）。其余细项与类别同名。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._shared import PreparedInputs
from .industry import CCS, H2
from .model_costs import _air_unit_capex, _blend_unit_capex, _island_unit_capex
from .salvage import remaining_fraction
from .scenario import OptimizationAssumptions, OptimizationScenario
from .vintage import alive_vintages
from .year_types import YearData, YearSolution

COST_COLUMNS = ["year", "entity_type", "entity_id", "category", "item", "kind", "cost_cny", "cost_discounted_cny"]
SYSTEM_COLUMNS = ["year", "category", "kind", "discount_weight", "cost_cny", "cost_discounted_cny",
                  "attributed_discounted_cny"]


def build_costs_table(
    prepared: PreparedInputs,
    solution: Mapping[str, Any],
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
) -> pd.DataFrame:
    """全部规划年的逐实体成本，只写非零行；列见 `COST_COLUMNS`。"""
    years = [int(y) for y in solution["year_solutions"]]
    solutions: list[YearSolution] = [solution["year_solutions"][y] for y in years]
    rows: list[dict[str, Any]] = []
    for position, (year, ys) in enumerate(zip(years, solutions, strict=True)):
        weights = ys["cost_weights"]
        entities = {
            "coal": (prepared.plants["plant_id"].astype(str).tolist(), _coal_costs(prepared, ys, assumptions)),
            "industry": (prepared.industry.hubs["hub_id"].astype(str).tolist(), _industry_costs(ys)),
            "edge": (prepared.network.edges["edge_id"].astype(str).tolist(),
                     _edge_costs(solutions, years, position, assumptions)),
            "sink": (prepared.storages["storage_hub_id"].astype(str).tolist(), _sink_costs(prepared, ys)),
        }
        for entity_type, (ids, parts) in entities.items():
            for (category, item), values in parts.items():
                kind, weight = weights[category]
                for entity_id, value in zip(ids, np.asarray(values, dtype=np.float64), strict=True):
                    if value != 0.0:
                        rows.append(_row(year, entity_type, entity_id, category, item, kind, value, weight))
        system = _slack_costs(ys, assumptions)
        if position == len(years) - 1 and "salvage_credit" in weights:
            system.update(_salvage_items(solutions, years, scenario, assumptions))
        for (category, item), value in system.items():
            kind, weight = weights[category]
            if value != 0.0:
                rows.append(_row(year, "system", category, category, item, kind, value, weight))
    return pd.DataFrame(rows, columns=COST_COLUMNS)


def build_system_table(solution: Mapping[str, Any], costs: pd.DataFrame) -> pd.DataFrame:
    """每年每个成本类别一行：求解器的合计（`cost_breakdown_cny`）与 `costs.csv` 拆到各实体后的合计并排。

    `cost_discounted_cny` 是进目标函数的值，各年各类相加即目标函数值；`cost_cny` 是除以折现权重后本年不折现的值：
    `kind` 为 annual 的是一年的费用，one_off 是本年的一次性支出，horizon_end 是期末的残值抵扣。
    """
    attributed = costs.groupby(["year", "category"])["cost_discounted_cny"].sum()
    rows = []
    for year, ys in solution["year_solutions"].items():
        for category, discounted in ys["cost_breakdown_cny"].items():
            kind, weight = ys["cost_weights"][category]
            rows.append({
                "year": int(year), "category": category, "kind": kind, "discount_weight": float(weight),
                "cost_cny": float(discounted) / float(weight) if weight else 0.0,
                "cost_discounted_cny": float(discounted),
                "attributed_discounted_cny": float(attributed.get((int(year), category), 0.0)),
            })
    return pd.DataFrame(rows, columns=SYSTEM_COLUMNS)


def cost_closure(system: pd.DataFrame) -> pd.Series:
    """逐年的最大相对差：|拆到各实体的合计 − 求解器的合计| / max(|求解器的合计|, 1 元)，按年取各类别的最大值。"""
    gap = (system["attributed_discounted_cny"] - system["cost_discounted_cny"]).abs()
    scale = system["cost_discounted_cny"].abs().clip(lower=1.0)
    return (gap / scale).groupby(system["year"]).max()


def _row(year: int, entity_type: str, entity_id: str, category: str, item: str, kind: str,
         value: float, weight: float) -> dict[str, Any]:
    return {"year": year, "entity_type": entity_type, "entity_id": entity_id, "category": category, "item": item,
            "kind": kind, "cost_cny": float(value), "cost_discounted_cny": float(value) * float(weight)}


def _positive_priced(unit: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """同 `model_costs._priced`，逐单元：单价为正的单元 单价 x 数量，其余为 0。"""
    unit = np.asarray(unit, dtype=np.float64)
    return np.where(unit > 0.0, unit * np.asarray(amount, dtype=np.float64), 0.0)


def _by_plant(prepared: PreparedInputs, links: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """逐链路的值按链路的 `plant_id` 归到各厂。"""
    index = {pid: i for i, pid in enumerate(prepared.plants["plant_id"].astype(str))}
    out = np.zeros(len(prepared.plants))
    if len(links):
        np.add.at(out, links["plant_id"].astype(str).map(index).to_numpy(dtype=int), np.asarray(values, dtype=np.float64))
    return out


def _coal_costs(
    prepared: PreparedInputs, ys: YearSolution, assumptions: OptimizationAssumptions
) -> dict[tuple[str, str], np.ndarray]:
    """煤电各类成本，逐厂，CNY（未折现）；与 `model_costs._operating_costs`、`_resource_costs`、`_one_off_capex` 同式。"""
    yd: YearData = ys["year_data"]
    share, air_share = ys["share"], ys["air_share"]
    rebuilt, rebuilt_air = ys["rebuilt_share"], ys["rebuilt_air_share"]
    deltas = yd.rebuilt_deltas
    bio_s, amm_s, wat_s = float(yd.biomass_flow_scale), float(yd.ammonia_flow_scale), float(yd.water_flow_scale)

    operating = (yd.baseline_net_matrix * share).sum(axis=1)
    capture_fuel = (yd.energy_penalty_matrix * share).sum(axis=1)
    air_fuel = np.zeros(len(share))
    if yd.air_penalty_cost_matrix is not None and bool(yd.allow_air_cooling_retrofit):
        air_fuel = (yd.air_penalty_cost_matrix * air_share).sum(axis=1)
    for c, delta in enumerate(deltas):
        operating = operating + (delta.baseline_net_matrix * rebuilt[c]).sum(axis=1)
        capture_fuel = capture_fuel + (delta.energy_penalty_matrix * rebuilt[c]).sum(axis=1)
        air_fuel = air_fuel + (delta.air_penalty_cost_matrix * rebuilt_air[c]).sum(axis=1)

    price = float(yd.carbon_price)
    carbon = price * 1e6 * (np.asarray(yd.emissions_mt) - ys["plant_reduction_mt"]) if price > 0 else np.zeros(len(share))
    savings = np.broadcast_to(np.asarray(yd.coal_savings_per_gj, dtype=np.float64), len(share)) / bio_s * ys["biomass_use_gj"]
    if yd.coal_savings_per_kg_nh3 is not None:
        savings = savings + np.asarray(yd.coal_savings_per_kg_nh3, dtype=np.float64) / amm_s * ys["ammonia_use_kg"]

    unit_b, unit_a = _blend_unit_capex(yd, assumptions)
    return {
        ("coal_operating_delta", "coal_operating_delta"): operating,
        ("carbon_cost", "carbon_cost"): carbon,
        ("coal_savings_credit", "coal_savings_credit"): -savings,
        ("energy_penalty_cost", "capture_fuel"): capture_fuel,
        ("energy_penalty_cost", "air_cooling_backpressure"): air_fuel,
        ("energy_penalty_cost", "biomass_efficiency"): ys["bio_penalty_by_plant"],
        ("ccs_om_cost", "ccs_om_cost"): ys["ccs_om_by_plant"],
        ("incremental_om", "pathway_om"): (yd.fixed_cost_matrix * share).sum(axis=1),
        ("incremental_om", "biomass_blend_om"): _positive_priced(yd.biomass_blend_om_per_level, ys["blend_level_b"]),
        ("incremental_om", "ammonia_blend_om"): _positive_priced(yd.ammonia_blend_om_per_level, ys["blend_level_a"]),
        ("biomass_cost", "biomass_cost"): _by_plant(
            prepared, prepared.biomass_links, yd.biomass_link_cost_cny_per_gj * ys["biomass_flow_gj"] / bio_s),
        ("ammonia_cost", "ammonia_cost"): _by_plant(
            prepared, yd.ammonia_links, yd.ammonia_link_cost_cny_per_kg * ys["ammonia_flow_kg"] / amm_s),
        ("water_cost", "water_cost"): _by_plant(
            prepared, yd.water_links,
            (yd.water_link_cost_cny_per_m3 * ys["water_flow_m3"] / wat_s) if len(yd.water_link_cost_cny_per_m3) else []),
        ("stranded_capex", "stranded_capex"): ys["stranded_by_plant"],
        ("ccs_retrofit_capex", "ccs_retrofit_capex"): _positive_priced(_island_unit_capex(yd), ys["retrofit_new"][:, 0]),
        ("blend_upgrade_capex", "biomass_blend"): _positive_priced(unit_b, ys["blend_new_b"].sum(axis=1)),
        ("blend_upgrade_capex", "ammonia_blend"): _positive_priced(unit_a, ys["blend_new_a"].sum(axis=1)),
        ("air_retrofit_capex", "air_retrofit_capex"): _positive_priced(_air_unit_capex(yd), ys["air_new"]),
        ("rebuild_capex", "rebuild_capex"): ys["rebuild_capex_by_plant"],
    }


def _industry_costs(ys: YearSolution) -> dict[tuple[str, str], np.ndarray]:
    """工业各类成本，逐 hub，CNY（未折现）；与 `model_industry.add_industry_year`、`model_costs.add_year_costs` 同式。"""
    iy = ys["year_data"].industry
    share = ys["industry_share"]
    price = float(ys["year_data"].carbon_price)
    residual = np.asarray(iy.baseline_emissions_mt) - (iy.reduction_mt * share).sum(axis=1)
    unit, new = iy.capex_cny_per_mt, ys["industry_new_capacity_mt"]
    return {
        ("carbon_cost", "carbon_cost"): price * 1e6 * residual if price > 0 else np.zeros(len(share)),
        ("industry_cost", "ccs_operating"): iy.opex_cny[:, CCS] * share[:, CCS],
        ("industry_cost", "ccs_fixed_om"): ys["industry_ccs_om_by_hub"],
        ("industry_cost", "h2_route"): ys["industry_h2_route_cost"],
        ("industry_capex", "ccs"): _positive_priced(unit[:, CCS], new[:, CCS]),
        ("industry_capex", "h2"): _positive_priced(unit[:, H2], new[:, H2]),
    }


def _pipe_capex(ys: YearSolution) -> np.ndarray:
    """本年新铺管道的 capex，逐边，CNY：各管径档的单根单价 x 根数（`model_costs._pipe_capex`）。"""
    return (np.asarray(ys["year_data"].edge_tier_capex, dtype=np.float64) * ys["pipe_count"]).sum(axis=1)


def _edge_costs(
    solutions: list[YearSolution], years: list[int], position: int, assumptions: OptimizationAssumptions
) -> dict[tuple[str, str], np.ndarray]:
    """管段各类成本，逐边，CNY（未折现）：按流量计的运输运维、在役各代管道的固定运维、本年新铺管道的 capex
    （`model_costs._transport_storage_costs`、`_one_off_capex`）。"""
    ys = solutions[position]
    coeff = np.asarray(ys["year_data"].edge_route_opex_coeff, dtype=np.float64)
    fixed_om = np.zeros(len(coeff))
    fraction = float(assumptions.pipe_fixed_om_fraction)
    if fraction > 0.0:
        for v in alive_vintages(years, position, int(assumptions.pipeline_lifetime_years)):
            fixed_om = fixed_om + fraction * _pipe_capex(solutions[v])
    return {
        ("transport_opex", "flow_opex"): coeff * (ys["co2_flow_fwd"] + ys["co2_flow_bwd"]),
        ("transport_opex", "pipe_fixed_om"): fixed_om,
        ("pipe_capex", "pipe_capex"): _pipe_capex(ys),
    }


def _sink_costs(prepared: PreparedInputs, ys: YearSolution) -> dict[tuple[str, str], np.ndarray]:
    """封存成本，逐汇，CNY/yr：每吨封存成本（已按汇型与陆海定价）x 注入量。"""
    unit = prepared.storages["storage_cost_cny_per_t"].astype(float).to_numpy()
    return {("storage_cost", "storage_cost"): unit * 1e6 * ys["storage_use_mtpa"]}


def _slack_costs(ys: YearSolution, assumptions: OptimizationAssumptions) -> dict[tuple[str, str], float]:
    """松弛惩罚，按松弛种类，CNY/yr（`model_costs._slack_penalty`；松弛量已乘回物理单位）。"""
    slacks = ys["slacks"]
    penalty = float(assumptions.slack_penalty_cny_per_unit)
    parts = {
        "target_shortfall": float(slacks["target_shortfall_mt"]) * penalty,
        "biomass_supply": float(np.sum(slacks["biomass_slack_gj"])) * 2_000.0,
        "green_h2_supply": float(np.sum(slacks["h2_slack_kg"])) * 1_000.0,
        "water_node": float(np.sum(slacks["water_slack_m3"])) * 1_000.0,
        "water_basin_quota": float(np.sum(slacks["water_basin_slack_m3"])) * 1_000.0,
        "storage_injectivity": float(np.sum(slacks["injectivity_slack_mtpa"])) * penalty,
        "storage_capacity": float(np.sum(slacks["storage_slack_mt"])) * penalty,
        "edge_capacity": float(np.sum(slacks["edge_slack_mtpa"])) * penalty,
    }
    return {("slack_penalty", item): value for item, value in parts.items()}


def _salvage_items(
    solutions: list[YearSolution], years: list[int], scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
) -> dict[tuple[str, str], float]:
    """期末残值抵扣，按资产类别（台账名），CNY（期末、未折现，负值）：Σ_建设年 未折旧比例 x 台账值（`salvage`）。"""
    end_year = years[-1] + scenario.interval_years(tuple(years), len(years) - 1, assumptions)
    items: dict[tuple[str, str], float] = {}
    for year, ys in zip(years, solutions, strict=True):
        for name, value, life in ys["salvage_ledger"]:
            frac = remaining_fraction(year, life, end_year)
            if frac > 0.0:
                key = ("salvage_credit", str(name))
                items[key] = items.get(key, 0.0) - frac * float(value)
    return items
