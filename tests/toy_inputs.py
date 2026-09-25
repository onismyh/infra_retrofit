"""toy 输入：1 座电厂 - 1 条边 - 1 个封存汇，外加一个水泥 hub；多个测试模块共用。

从 `test_multiperiod_investment_logic.py` 挪来。那个文件在模块级 `importorskip("gurobipy")`，
从它导入这些辅助函数的测试文件，在没有 Gurobi 的环境里会连同不求解的测试整体跳过（见
`test_discount_rate.py` 的说明）；本模块不导入 gurobipy。文件名不以 `test_` 开头，pytest 不收集。
"""
from __future__ import annotations

import pandas as pd

from coal_retrofit.optimization.scenario import OptimizationAssumptions
from coal_retrofit.paths import ProjectPaths

YEARS = (2050, 2060)
TOY_TARGET_YEARS = (2030, 2040, 2050, 2060)


def _write_toy_inputs(root, retirement_year: int) -> ProjectPaths:
    """最小但完整的 inputs/：1 座电厂 - 1 条边 - 1 个封存汇，资源放在远处。"""
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "plant_id": ["P1"],
            "province_mode": ["Shanxi"],
            "total_capacity_mw": [1000.0],
            "retirement_year": [retirement_year],
            "dominant_cooling_technology": ["recirculating"],
            "centroid_longitude": [112.0],
            "centroid_latitude": [37.0],
        }
    ).to_csv(inputs / "plants.csv", index=False)

    pd.DataFrame(
        {
            "storage_hub_id": ["S1"],
            "storage_type": ["dsa"],
            "storage_all_mt": [1000.0],
            "storage_dsa_mt": [1000.0],
            "storage_eor_mt": [0.0],
            "injectivity_dsa_avg_mtpa": [10.0],
            "injectivity_eor_avg_mtpa": [0.0],
            "latitude": [37.0],
            "longitude": [112.5],
        }
    ).to_csv(inputs / "storage_hubs.csv", index=False)

    pd.DataFrame(
        {
            "node_id": ["N1", "N2"],
            "lon": [112.0, 112.5],
            "lat": [37.0, 37.0],
            "node_type": ["plant", "storage_hub"],
            "plant_id": ["P1", pd.NA],
            "storage_hub_id": [pd.NA, "S1"],
        }
    ).to_csv(inputs / "pipeline_nodes.csv", index=False)

    pd.DataFrame(
        {
            "edge_id": ["E1"],
            "from_node_id": ["N1"],
            "to_node_id": ["N2"],
            "length_km": [100.0],
            "existing_corridor_flag": [0],
            "edge_class": ["triangulation_candidate"],
            "corridor_type": ["candidate"],
            "source": ["toy"],
            "year_basis": ["toy"],
        }
    ).to_csv(inputs / "pipeline_candidate_edges.csv", index=False)

    # 生物质 / 氨放在 >2000 km 之外，因此不会生成燃料链路。
    # 水节点紧挨电厂：厂级水平衡（供水流量 == 用水量）即使在 no_water 模式下
    # 也会施加，所以任何运行中的电厂都至少需要一条水链路。
    pd.DataFrame(
        {
            "biomass_node_id": ["B1"],
            "longitude": [90.0],
            "latitude": [50.0],
            "available_gj": [1.0e6],
            "base_cost_cny_per_gj": [22.0],
            "province_name": ["Xinjiang"],
        }
    ).to_csv(inputs / "biomass_supply_curve.csv", index=False)

    pd.DataFrame(
        {
            "ammonia_node_id": ["A1"],
            "year": [2050],
            "nh3_supply_kg_per_year": [1.0e9],
            "h2_supply_kg_per_year": [1.8e8],
            "weighted_lcoh_usd_per_kg_h2": [3.0],
            "nh3_cost_lb_usd_per_kg": [1.5],
            "longitude": [90.0],
            "latitude": [50.0],
            "province_name": ["Xinjiang"],
        }
    ).to_csv(inputs / "ammonia_supply_curve.csv", index=False)

    pd.DataFrame(
        {
            "water_node_id": ["W1"],
            "longitude": [112.1],
            "latitude": [37.0],
            "province_name": ["Shanxi"],
        }
    ).to_csv(inputs / "water_nodes.csv", index=False)

    pd.DataFrame(
        columns=["water_node_id", "planning_year", "scenario_family", "available_water_m3_per_year"]
    ).to_csv(inputs / "water_availability.csv", index=False)

    # 一个水泥点源（无氢路线），挂在电厂旁；上限 1.0 且碳价为零时它什么都不做，
    # 所以煤电侧的闭式解不受影响。
    pd.DataFrame(
        {
            "hub_id": ["C1"],
            "sector": ["cement"],
            "sector_zh": ["水泥"],
            "n_plants": [1],
            "province": ["Shanxi"],
            "longitude": [112.2],
            "latitude": [37.0],
            "capacity_kt_per_year": [1200.0],
            "production_kt_per_year": [1000.0],
            "co2_mt_per_year": [0.8],
            "process_co2_mt_per_year": [0.5],
            "h2_demand_kt_per_year": [0.0],
            "has_h2_route": [False],
            "water_intensity_m3_per_t": [1.0],
            "water_m3_per_year": [1.0e6],
            "mean_commission_year": [2010.0],
            "share_year_observed": [1.0],
        }
    ).to_csv(inputs / "industry_hubs.csv", index=False)
    pd.DataFrame(
        [{"sector": "cement", "planning_year": y, "output_index": 1.0} for y in TOY_TARGET_YEARS]
    ).to_csv(inputs / "industry_output_index_toy.csv", index=False)
    paths = ProjectPaths(root=root)
    _write_targets(paths, {y: 1.0 for y in TOY_TARGET_YEARS})
    return paths


def _write_targets(
    paths: ProjectPaths, power_caps: dict[int, float], cement_caps: dict[int, float] | None = None
) -> None:
    """写 sector_targets_toy.csv：power 组按给定上限，水泥组按 `cement_caps`（缺省恒为 1.0，不约束）。"""
    cement = cement_caps or {}
    pd.DataFrame(
        [
            {"sector_group": group, "planning_year": year, "cap_fraction_of_2030": cap}
            for year, power_cap in power_caps.items()
            for group, cap in (("power", power_cap), ("cement", cement.get(year, 1.0)))
        ]
    ).to_csv(paths.inputs_dir / "sector_targets_toy.csv", index=False)


def _toy_assumptions() -> OptimizationAssumptions:
    """默认假设，但去掉封存部署爬坡：toy 汇每年都必须提供满额 10 Mtpa，
    否则 2030 年目标会靠缺口松弛而不是捕集来满足
    （有爬坡时 2030 年只剩 1.7 Mtpa，低于目标所需的 ~2.6 Mt/yr）。"""
    return OptimizationAssumptions(storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0))
