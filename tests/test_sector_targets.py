"""单厂 toy 模型上的部门碳目标上限与利用小时轨迹。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from toy_inputs import _write_targets, _write_toy_inputs

YEARS = (2030, 2040)


def _solve(paths, scenario: OptimizationScenario) -> dict[str, object]:
    assumptions = OptimizationAssumptions(storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0))
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.effective_years(list(prepared.available_ammonia_years))
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    return _solve_joint_multi_period(prepared, scenario, assumptions, years, state)


def test_power_cap_binds_on_the_2030_baseline_and_hours_scale_generation(tmp_path) -> None:
    """2030 年上限 1.0（无事可做），2040 年上限为 2030 年基线的 0.4，利用小时
    从 4 000 降到 3 000。2040 年残余排放必须恰好等于 0.4 x E_2030，且 2040 年基线
    必须是 2030 年的 3/4，因为只有利用小时变了。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 1.0, 2040: 0.4})
    scenario = OptimizationScenario(
        experiment_id="TEST-SECTOR",
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
        coal_operating_hours_by_year=(4000.0, 3000.0),
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0),
        pathway_disable=("retire",),
        solver_time_limit=300,
    )
    solution = _solve(paths, scenario)
    assert solution["status"] == "optimal"
    y1 = solution["year_solutions"][2030]
    y2 = solution["year_solutions"][2040]

    e_2030 = float(y1["year_data"].emissions_mt[0])
    e_2040 = float(y2["year_data"].emissions_mt[0])
    assert e_2040 == pytest.approx(0.75 * e_2030, rel=1e-9)
    # 利用小时：toy 机组群只有一座 4 629.5 h 的山西电厂，2030 年缩放到 4 000。
    assert y1["year_data"].hours_scale == pytest.approx(4000.0 / 4629.5, rel=1e-6)

    # 2030 年：不要求减排，也没有买任何减排。
    assert y1["total_reduction_mt"] == pytest.approx(0.0, abs=1e-6)
    assert y1["slacks"]["target_shortfall_mt"] == pytest.approx(0.0, abs=1e-9)
    # 2040 年：残余排放恰在上限，靠捕集满足（退役已禁用），无松弛。
    residual_2040 = e_2040 - float(y2["total_reduction_mt"])
    assert residual_2040 == pytest.approx(0.4 * e_2030, rel=1e-4)
    assert y2["slacks"]["target_shortfall_by_group"]["power"] == pytest.approx(0.0, abs=1e-9)
    assert y2["share"][0, 2] > 0.0  # ccs
    assert y2["cost_breakdown_cny"]["carbon_cost"] == 0.0


def test_unmeetable_cap_is_reported_as_group_shortfall(tmp_path) -> None:
    """上限低于各路径所能达到的水平时，差额落入对应目标组的缺口，且标量缺口
    等于各组之和。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 1.0, 2040: -0.5})
    scenario = OptimizationScenario(
        experiment_id="TEST-SECTOR-SHORT",
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0),
        # 只有 CCS 能起作用：无 BECCS（toy 中生物质够不到），无退役。
        pathway_disable=("retire", "biomass", "beccs", "ammonia"),
        solver_time_limit=300,
    )
    solution = _solve(paths, scenario)
    assert solution["status"] == "optimal"
    y2 = solution["year_solutions"][2040]
    by_group = y2["slacks"]["target_shortfall_by_group"]
    assert set(by_group) == {"power", "cement"}
    assert by_group["power"] > 0.1
    assert by_group["cement"] == pytest.approx(0.0, abs=1e-9)
    assert y2["slacks"]["target_shortfall_mt"] == pytest.approx(by_group["power"], rel=1e-9)


def test_national_biomass_cap_limits_fleet_biomass_and_lands_in_shortfall(tmp_path) -> None:
    """生物质是唯一剩下的路径、2040 年要求减排 15% 时，不设上限的机组群靠掺烧满足上限；
    远低于该需求的全国上限恰好绑定，未满足的部分表现为 power 组的缺口。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    # 把生物质节点挪到电厂旁边，使燃料链路存在（fixture 故意把它放在 2 000 km 之外）。
    pd.DataFrame(
        {
            "biomass_node_id": ["B1"],
            "longitude": [112.2],
            "latitude": [37.0],
            "available_gj": [1.0e9],
            "base_cost_cny_per_gj": [22.0],
            "province_name": ["Shanxi"],
        }
    ).to_csv(paths.inputs_dir / "biomass_supply_curve.csv", index=False)
    _write_targets(paths, {2030: 1.0, 2040: 0.85})
    scenario = OptimizationScenario(
        experiment_id="TEST-BIOCAP",
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0),
        pathway_disable=("retire", "ccs", "beccs", "ammonia"),
        solver_time_limit=300,
    )

    def _run(cap_gj: float):
        assumptions = OptimizationAssumptions(
            storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0),
            biomass_national_cap_gj_per_year=cap_gj,
        )
        prepared = prepare_inputs(paths, scenario, assumptions)
        years = scenario.effective_years(list(prepared.available_ammonia_years))
        state = SolveState(
            edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
            remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
        )
        return _solve_joint_multi_period(prepared, scenario, assumptions, years, state)

    free = _run(0.0)
    assert free["status"] == "optimal"
    y_free = free["year_solutions"][2040]
    assert y_free["slacks"]["target_shortfall_by_group"]["power"] == pytest.approx(0.0, abs=1e-9)
    demand_gj = float(np.sum(y_free["biomass_use_gj"]))
    assert demand_gj > 0.0

    cap_gj = 0.5 * demand_gj
    capped = _run(cap_gj)
    assert capped["status"] == "optimal"
    y_cap = capped["year_solutions"][2040]
    assert float(np.sum(y_cap["biomass_use_gj"])) == pytest.approx(cap_gj, rel=1e-6)
    assert y_cap["slacks"]["target_shortfall_by_group"]["power"] > 0.0


def test_fleet_ammonia_cap_binds_and_lands_in_shortfall(tmp_path) -> None:
    """与生物质测试同构，唯一路径换成掺氨：不设上限时，2040 年的减排要求得以满足；
    机组群上限设为该需求的一半时恰好绑定，其余部分成为 power 组缺口。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    pd.DataFrame(
        {
            "ammonia_node_id": ["A1"],
            "year": [2050],
            "nh3_supply_kg_per_year": [1.0e12],
            "h2_supply_kg_per_year": [1.8e11],
            "weighted_lcoh_usd_per_kg_h2": [3.0],
            "nh3_cost_lb_usd_per_kg": [0.3],
            "longitude": [112.2],
            "latitude": [37.0],
            "province_name": ["Shanxi"],
        }
    ).to_csv(paths.inputs_dir / "ammonia_supply_curve.csv", index=False)
    _write_targets(paths, {2030: 1.0, 2040: 0.85})
    scenario = OptimizationScenario(
        experiment_id="TEST-NH3CAP",
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0),
        pathway_disable=("retire", "ccs", "biomass", "beccs"),
        solver_time_limit=300,
    )

    def _run(cap_mt: tuple[float, ...]):
        assumptions = OptimizationAssumptions(
            storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0),
            ammonia_fleet_cap_mt_by_year=cap_mt,
            green_h2_national_cap_mt_by_year=(),
        )
        prepared = prepare_inputs(paths, scenario, assumptions)
        years = scenario.effective_years(list(prepared.available_ammonia_years))
        state = SolveState(
            edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
            remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
        )
        return _solve_joint_multi_period(prepared, scenario, assumptions, years, state)

    free = _run(())
    assert free["status"] == "optimal"
    y_free = free["year_solutions"][2040]
    assert y_free["slacks"]["target_shortfall_by_group"]["power"] == pytest.approx(0.0, abs=1e-9)
    demand_kg = float(np.sum(y_free["ammonia_use_kg"]))
    assert demand_kg > 0.0

    cap_mt = 0.5 * demand_kg / 1e9
    capped = _run((cap_mt, cap_mt))
    assert capped["status"] == "optimal"
    y_cap = capped["year_solutions"][2040]
    assert float(np.sum(y_cap["ammonia_use_kg"])) == pytest.approx(cap_mt * 1e9, rel=1e-6)
    assert y_cap["slacks"]["target_shortfall_by_group"]["power"] > 0.0
