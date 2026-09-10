from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.results import _build_plant_cost_table
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.paths import ProjectPaths

YEARS = (2050, 2060)


def _write_toy_inputs(root, retirement_year: int) -> ProjectPaths:
    """Minimal but complete inputs/: 1 plant - 1 edge - 1 storage, resources far away."""
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

    # Biomass/ammonia placed >2000 km away so no fuel links are generated.
    # The water node sits next to the plant: the plant-level water balance
    # (supply flow == use) is enforced even in no_water mode, so any operating
    # plant needs at least one water link.
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

    return ProjectPaths(root=root)


def _toy_assumptions() -> OptimizationAssumptions:
    """Defaults, minus the storage deployment ramp: the toy sink must offer its full 10 Mtpa
    in every year, or the 2030 target is met through the shortfall slack instead of capture
    (the ramp would leave 1.7 Mtpa in 2030, below the ~2.6 Mt/yr the target needs)."""
    return OptimizationAssumptions(storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0))


def _solve_toy(paths: ProjectPaths, scenario: OptimizationScenario) -> dict[str, object]:
    assumptions = _toy_assumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.effective_years(list(prepared.available_ammonia_years))
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    return _solve_joint_multi_period(prepared, scenario, assumptions, years, state)


def _expected_ccs(
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    target_fraction: float,
    year: int = 2030,
) -> tuple[float, float]:
    """Closed-form CCS share and captured Mt for the single-plant toy model.

    Per unit CCS share, reduction vs baseline = (1 - boost*(1-eta)) * E minus the
    energy-penalty fuel emissions (extra coal burned for the capture efficiency loss,
    counted in the residual since the penalty-emissions fix). The toy plant is never
    expired, so heat_rate_eff equals the baseline heat rate.
    """
    gen = 1000.0 * assumptions.province_cf("Shanxi") * 8760.0
    e_mt = gen * assumptions.coal_emission_factor_t_per_mwh / 1e6
    boost = scenario.retrofit_cf_boost
    eta = scenario.capture_rate
    ef_t_per_gj = assumptions.coal_emission_factor_t_per_mwh / assumptions.heat_rate_gj_per_mwh
    # Penalty fuel burns in the same boiler, so only the uncaptured share is vented.
    penalty_emissions_mt = (
        gen
        * boost
        * assumptions.ccs_energy_penalty_ratio(year)
        * assumptions.heat_rate_gj_per_mwh
        * ef_t_per_gj
        * (1.0 - eta)
        / 1e6
    )
    red_per_share = (1.0 - boost * (1.0 - eta)) * e_mt - penalty_emissions_mt
    share = target_fraction * e_mt / red_per_share
    # The captured share of the penalty fuel is a real tonne on the pipeline (2026-09-10).
    penalty_captured_mt = penalty_emissions_mt / (1.0 - eta) * eta
    captured_mt = (boost * eta * e_mt + penalty_captured_mt) * share
    return share, captured_mt


def test_pipeline_tiers_size_the_pipe_to_the_flow_and_build_once(tmp_path) -> None:
    """Capacity comes in whole pipes of the diameter tiers (2 / 5 / 20 Mtpa): a ~2.3 Mt/yr
    flow gets the cheapest tier combination that covers it (one 5-Mtpa pipe: 3.5e6 CNY/km
    beats two 2-Mtpa pipes at 4.0e6), not a 20-Mtpa trunk. And an edge built in period 1
    must NOT be forced to add capacity again in period 2 (the old min-build latch bug)."""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    scenario = OptimizationScenario(
        experiment_id="TEST-MINBUILD",
        description="toy",
        planning_years=YEARS,
        emission_target_fraction=(0.5, 0.5),
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=("retire",),
        solver_time_limit=300,
    )
    solution = _solve_toy(paths, scenario)
    assert solution["status"] == "optimal"

    y1 = solution["year_solutions"][2050]
    y2 = solution["year_solutions"][2060]

    # With the retrofit CF boost wired in, a retrofitted share generates (and emits)
    # boost x baseline, capturing eta of that; the energy-penalty fuel emissions
    # further reduce the net reduction per share. See _expected_ccs for the algebra.
    assumptions = _toy_assumptions()
    s_ccs_expected, captured_expected = _expected_ccs(scenario, assumptions, 0.5, 2050)
    ccs_idx = 2  # PATHWAYS = (unabated, retire, ccs, biomass, beccs, ammonia)
    assert y1["share"][0, ccs_idx] == pytest.approx(s_ccs_expected, rel=1e-3)
    assert y1["edge_flow_mtpa"][0] == pytest.approx(captured_expected, rel=1e-3)

    # Captured CO2 (~2.3 Mt/yr) needs more than the 2-Mtpa tier; one 5-Mtpa pipe is the
    # cheapest cover, so exactly 5 Mtpa of new capacity in period 1 -- not a 20-Mtpa trunk.
    assert captured_expected > 2.0
    assert y1["new_cap_mtpa"][0] == pytest.approx(5.0, rel=1e-3)
    assert y1["pipe_count"][0].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert y1["build_edge"][0] == pytest.approx(1.0)

    # Period 2 reuses the capacity stock built in period 1: no new capacity, but the
    # latched build flag must stay 1 (irreversibility) and flow keeps using the edge.
    assert y2["new_cap_mtpa"][0] == pytest.approx(0.0, abs=1e-6)
    assert y2["build_edge"][0] == pytest.approx(1.0)
    # The energy-penalty ratio is lower in 2060, so a smaller share would meet the same 50%
    # target -- but the capture-share lock (2026-09-10) keeps the 2050 share running: a capture
    # island is not switched off. Captured tonnes in 2060 are therefore the LOCKED share times
    # the 2060 per-share capture (whose penalty component is the smaller 2060 one).
    s_2060_unlocked, captured_2060_unlocked = _expected_ccs(scenario, assumptions, 0.5, 2060)
    assert s_2060_unlocked < s_ccs_expected
    assert y2["share"][0, ccs_idx] == pytest.approx(s_ccs_expected, rel=1e-3)
    captured_expected_2060 = captured_2060_unlocked / s_2060_unlocked * s_ccs_expected
    assert y2["edge_flow_mtpa"][0] == pytest.approx(captured_expected_2060, rel=1e-3)
    assert y2["cost_breakdown_cny"]["pipe_capex"] == pytest.approx(0.0, abs=1.0)


def test_rebuild_capex_charged_once_at_activation(tmp_path) -> None:
    """A plant past its design life that chooses site rebuild pays the one-time rebuild
    CAPEX only in the activation period. Regression test for the repeated-charging bug:
    previously the full CAPEX was re-charged in every period with rebuild == 1."""
    paths = _write_toy_inputs(tmp_path, retirement_year=2040)
    scenario = OptimizationScenario(
        experiment_id="TEST-REBUILD",
        description="toy",
        planning_years=YEARS,
        emission_target_fraction=(0.0, 0.0),
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        solver_time_limit=300,
    )
    solution = _solve_toy(paths, scenario)
    assert solution["status"] == "optimal"

    y1 = solution["year_solutions"][2050]
    y2 = solution["year_solutions"][2060]

    # Operating is profitable (electricity price > coal + O&M), so the expired plant
    # rebuilds in period 1 and stays rebuilt (latched) in period 2.
    assert y1["rebuild"][0] == pytest.approx(1.0)
    assert y2["rebuild"][0] == pytest.approx(1.0)
    retire_idx = 1  # PATHWAYS = (unabated, retire, ccs, biomass, beccs, ammonia)
    assert y1["share"][0, retire_idx] == pytest.approx(0.0, abs=1e-6)

    # One-time CAPEX in period 1: 1000 MW x 3500 CNY/kW x 0.70 x 1000, discounted.
    expected_capex_t1 = 1000.0 * 3500.0 * 0.70 * 1000.0 / (1.0 + 0.06) ** (2050 - 2025)
    assert y1["cost_breakdown_cny"]["rebuild_capex"] == pytest.approx(expected_capex_t1, rel=1e-3)
    # No rebuild CAPEX in period 2 even though the plant is still rebuilt and operating.
    assert y2["cost_breakdown_cny"]["rebuild_capex"] == pytest.approx(0.0, abs=1.0)

    # The rebuilt plant operates at rebuild_efficiency (USC): its heat rate improves to
    # hr x 0.42/0.45, which must show up in the baseline net operating cost.
    assumptions = _toy_assumptions()
    hr_eff = assumptions.heat_rate_gj_per_mwh * assumptions.coal_plant_base_efficiency / scenario.rebuild_efficiency
    net_pm = (
        hr_eff * assumptions.province_coal_cost("Shanxi")
        + assumptions.baseline_om_cost_cny_per_mwh
        - scenario.electricity_price_for_year(2050)
    )
    gen = 1000.0 * assumptions.province_cf("Shanxi") * 8760.0
    rate = scenario.discount_rate
    annuity = (1.0 - (1.0 + rate) ** -10.0) / rate
    df = 1.0 / (1.0 + rate) ** (2050 - scenario.discount_base_year)
    expected_baseline_net = net_pm * gen * annuity * df
    assert y1["cost_breakdown_cny"]["baseline_net_cost"] == pytest.approx(expected_baseline_net, rel=1e-3)


def test_ccs_retrofit_capex_charged_on_installed_stock_not_share_delta(tmp_path) -> None:
    """A capture island once built is paid for once and keeps running.

    Targets 0.5 -> 0.3 -> 0.5 with retirement disabled. Before the capture-share lock
    (2026-09-10) the CCS share dipped in period 2 and rebounded in period 3, and this test
    guarded the stock-increment charging (no second capex on the rebound). With the lock the
    share cannot dip at all -- the 2030 share is held through 2040 and 2050, the lower 2040
    target is over-met, and CAPEX is still due only once, in period 1."""
    years3 = (2030, 2040, 2050)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    scenario = OptimizationScenario(
        experiment_id="TEST-CCSSTOCK",
        description="toy",
        planning_years=years3,
        emission_target_fraction=(0.5, 0.3, 0.5),
        carbon_price_cny_per_t_by_year=(0.0, 0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0, 490.0),
        pathway_disable=("retire",),
        solver_time_limit=300,
    )
    assumptions = _toy_assumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.effective_years(list(prepared.available_ammonia_years))
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, state)
    assert solution["status"] == "optimal"

    y1 = solution["year_solutions"][2030]
    y2 = solution["year_solutions"][2040]
    y3 = solution["year_solutions"][2050]

    # CCS share follows the target exactly (all pathway costs increase with share).
    ccs_idx = 2  # PATHWAYS = (unabated, retire, ccs, biomass, beccs, ammonia)
    # The energy-penalty ratio declines over time, so the share meeting a given target
    # differs by year even when the target is unchanged.
    s1_y2030, _ = _expected_ccs(scenario, assumptions, 0.5, 2030)
    s1_y2050, _ = _expected_ccs(scenario, assumptions, 0.5, 2050)
    s2_y2040, _ = _expected_ccs(scenario, assumptions, 0.3, 2040)
    assert s2_y2040 < s1_y2050 < s1_y2030
    assert y1["share"][0, ccs_idx] == pytest.approx(s1_y2030, rel=1e-3)
    # Locked: no dip in 2040, and 2050 needs no more than what is already running.
    assert y2["share"][0, ccs_idx] == pytest.approx(s1_y2030, rel=1e-3)
    assert y3["share"][0, ccs_idx] == pytest.approx(s1_y2030, rel=1e-3)

    # CAPEX only in period 1, on the full installed stock; periods 2 and 3 add nothing
    # (period 3's rebound stays within the already-installed stock).
    rate = scenario.discount_rate
    df1 = 1.0 / (1.0 + rate) ** (2030 - scenario.discount_base_year)
    capex_rate = assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * assumptions.ccs_learning_factor(2030)
    expected_capex_y1 = 1000.0 * capex_rate * s1_y2030 * df1
    assert y1["cost_breakdown_cny"]["ccs_retrofit_capex"] == pytest.approx(expected_capex_y1, rel=1e-3)
    assert y2["cost_breakdown_cny"]["ccs_retrofit_capex"] == pytest.approx(0.0, abs=1.0)
    assert y3["cost_breakdown_cny"]["ccs_retrofit_capex"] == pytest.approx(0.0, abs=1.0)

    # The per-plant cost report must mirror the model: installed-stock CAPEX in
    # period 1 only, nothing in periods 2 and 3.
    capex_indices = solution["capex_pathway_indices"]
    plant_cost_y1 = _build_plant_cost_table(
        prepared, scenario, assumptions, 2030, y1["year_data"], y1["share"],
        y1["captured_mt_by_plant"], y1["biomass_use_gj"], y1["water_use_m3"],
        y1["blend_level_b"], y1["blend_level_a"],
        retrofit_installed=y1["retrofit_installed"], capex_pathway_indices=capex_indices,
    )
    plant_cost_y3 = _build_plant_cost_table(
        prepared, scenario, assumptions, 2050, y3["year_data"], y3["share"],
        y3["captured_mt_by_plant"], y3["biomass_use_gj"], y3["water_use_m3"],
        y3["blend_level_b"], y3["blend_level_a"],
        prev_share_values=y2["share"],
        retrofit_installed=y3["retrofit_installed"],
        prev_retrofit_installed=y2["retrofit_installed"],
        capex_pathway_indices=capex_indices,
    )
    assert float(plant_cost_y1["ccs_retrofit_capex_cny"].iloc[0]) == pytest.approx(
        1000.0 * capex_rate * s1_y2030, rel=1e-3
    )
    assert float(plant_cost_y3["ccs_retrofit_capex_cny"].iloc[0]) == pytest.approx(0.0, abs=1.0)


def test_unit_cf_boost_recovers_unboosted_accounting(tmp_path) -> None:
    """Guard: with retrofit_cf_boost = 1.0 the wired-in per-pathway accounting must
    reduce exactly to the classic formulation (reduction = eta x share x baseline E,
    minus the energy-penalty fuel emissions that are counted in the residual)."""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    scenario = OptimizationScenario(
        experiment_id="TEST-BOOST1",
        description="toy",
        planning_years=YEARS,
        emission_target_fraction=(0.5, 0.5),
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=("retire",),
        retrofit_cf_boost=1.0,
        solver_time_limit=300,
    )
    solution = _solve_toy(paths, scenario)
    assert solution["status"] == "optimal"

    y1 = solution["year_solutions"][2050]
    ccs_idx = 2  # PATHWAYS = (unabated, retire, ccs, biomass, beccs, ammonia)
    assumptions = _toy_assumptions()
    s_expected, captured_expected = _expected_ccs(scenario, assumptions, 0.5, 2050)
    assert y1["share"][0, ccs_idx] == pytest.approx(s_expected, rel=1e-3)
    assert y1["edge_flow_mtpa"][0] == pytest.approx(captured_expected, rel=1e-3)
