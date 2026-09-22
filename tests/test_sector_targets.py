"""Sector-target caps and the utilisation trajectory on the single-plant toy model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from test_multiperiod_investment_logic import _write_targets, _write_toy_inputs

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
    """Cap 1.0 in 2030 (nothing to do) and 0.4 of the 2030 baseline in 2040, with hours
    falling from 4 000 to 3 000. The 2040 residual must equal 0.4 x E_2030 exactly, and the
    2040 baseline must be 3/4 of the 2030 one because only the hours moved."""
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

    e_2030 = float(y1["year_data"]["emissions_mt"][0])
    e_2040 = float(y2["year_data"]["emissions_mt"][0])
    assert e_2040 == pytest.approx(0.75 * e_2030, rel=1e-9)
    # Hours: the toy fleet is one Shanxi plant at 4 629.5 h, scaled to 4 000 in 2030.
    assert y1["year_data"]["hours_scale"] == pytest.approx(4000.0 / 4629.5, rel=1e-6)

    # 2030: no abatement required, none bought.
    assert y1["total_reduction_mt"] == pytest.approx(0.0, abs=1e-6)
    assert y1["slacks"]["target_shortfall_mt"] == pytest.approx(0.0, abs=1e-9)
    # 2040: residual exactly at the cap, met by capture (retirement disabled), no slack.
    residual_2040 = e_2040 - float(y2["total_reduction_mt"])
    assert residual_2040 == pytest.approx(0.4 * e_2030, rel=1e-4)
    assert y2["slacks"]["target_shortfall_by_group"]["power"] == pytest.approx(0.0, abs=1e-9)
    assert y2["share"][0, 2] > 0.0  # ccs
    assert y2["cost_breakdown_cny"]["carbon_cost"] == 0.0


def test_unmeetable_cap_is_reported_as_group_shortfall(tmp_path) -> None:
    """A cap below what the pathways can reach lands in the named group's shortfall, and the
    scalar shortfall equals the sum over groups."""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 1.0, 2040: -0.5})
    scenario = OptimizationScenario(
        experiment_id="TEST-SECTOR-SHORT",
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0),
        # Only CCS can act: no BECCS (biomass is out of reach in the toy), no retirement.
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
    """With biomass the only pathway left and a 15% cut required in 2040, an uncapped fleet meets
    the cap by co-firing; a national ceiling far below that demand binds exactly and the
    unmet part shows up as the power group's shortfall."""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    # Move the biomass node next to the plant so a fuel link exists (the fixture parks it
    # 2 000 km away on purpose).
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
    """Same shape as the biomass test with ammonia co-firing as the only pathway: uncapped, the
    2040 cut is met; a fleet ceiling at half that demand binds exactly and the rest is
    power-group shortfall."""
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
