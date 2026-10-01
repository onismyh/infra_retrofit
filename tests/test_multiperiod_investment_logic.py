from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from coal_retrofit.optimization.year_types import SolveResult

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.results import _build_plant_cost_table
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.paths import ProjectPaths
from toy_inputs import YEARS, _toy_assumptions, _write_targets, _write_toy_inputs  # noqa: E402


def _solve_toy(paths: ProjectPaths, scenario: OptimizationScenario) -> SolveResult:
    assumptions = _toy_assumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.planning_years
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
    """单厂 toy 模型的闭式解：CCS 份额与捕集量（Mt）。

    每单位 CCS 份额相对基线的减排量 = (1 - boost*(1-eta)) * E，再减去能耗惩罚燃料排放
    （为弥补捕集造成的效率损失而多烧的煤，自能耗惩罚排放修正起计入残余排放）。
    toy 电厂从不到期，所以 heat_rate_eff 等于基线热耗。
    """
    gen = 1000.0 * assumptions.province_cf("Shanxi") * 8760.0
    e_mt = gen * assumptions.coal_emission_factor_t_per_mwh / 1e6
    boost = scenario.retrofit_cf_boost
    eta = scenario.capture_rate
    ef_t_per_gj = assumptions.coal_emission_factor_t_per_mwh / assumptions.heat_rate_gj_per_mwh
    # 惩罚燃料在同一台锅炉里燃烧，所以只有未被捕集的部分排入大气。
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
    # 惩罚燃料中被捕集的部分是管道上真实输送的吨数（2026-09-10）。
    penalty_captured_mt = penalty_emissions_mt / (1.0 - eta) * eta
    captured_mt = (boost * eta * e_mt + penalty_captured_mt) * share
    return share, captured_mt


def test_pipeline_tiers_size_the_pipe_to_the_flow_and_build_once(tmp_path) -> None:
    """容量按管径档位（2 / 5 / 20 Mtpa）整根计：~2.3 Mt/yr 的流量取能覆盖它的最便宜档位组合
    （一根 5-Mtpa 管 3.5e6 CNY/km，比两根 2-Mtpa 管的 4.0e6 便宜），而不是一根 20-Mtpa 干线。
    并且第 1 期建成的边在第 2 期不得被迫再次新增容量（旧的最小建设量锁存 bug）。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2050: 0.5, 2060: 0.5})
    scenario = OptimizationScenario(
        experiment_id="TEST-MINBUILD",
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=("retire",),
        solver_time_limit=300,
    )
    solution = _solve_toy(paths, scenario)
    assert solution["status"] == "optimal"

    y1 = solution["year_solutions"][2050]
    y2 = solution["year_solutions"][2060]

    # 接入改造 CF 提升后，改造份额的发电量（及排放）为 boost x 基线，
    # 其中 eta 被捕集；能耗惩罚燃料排放进一步压低每单位份额的净减排。
    # 代数推导见 _expected_ccs。
    assumptions = _toy_assumptions()
    s_ccs_expected, captured_expected = _expected_ccs(scenario, assumptions, 0.5, 2050)
    ccs_idx = 2  # PATHWAYS = (unabated, retire, ccs, biomass, beccs, ammonia)
    assert y1["share"][0, ccs_idx] == pytest.approx(s_ccs_expected, rel=1e-3)
    assert y1["edge_flow_mtpa"][0] == pytest.approx(captured_expected, rel=1e-3)

    # 捕集的 CO2（~2.3 Mt/yr）超出 2-Mtpa 档；一根 5-Mtpa 管是最便宜的覆盖方案，
    # 所以第 1 期新增容量恰为 5 Mtpa——而不是一根 20-Mtpa 干线。
    assert captured_expected > 2.0
    assert y1["new_cap_mtpa"][0] == pytest.approx(5.0, rel=1e-3)
    assert y1["pipe_count"][0].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert y1["build_edge"][0] == pytest.approx(1.0)

    # 第 2 期沿用第 1 期建成的容量存量：不新增容量，但锁存的建设标志必须保持为 1
    # （不可逆），流量继续使用这条边。
    assert y2["new_cap_mtpa"][0] == pytest.approx(0.0, abs=1e-6)
    assert y2["build_edge"][0] == pytest.approx(1.0)
    # 2060 年能耗惩罚比例更低，更小的份额就能满足同样 50% 的目标——但捕集份额锁定
    # （2026-09-10）让 2050 年的份额继续运行：捕集岛不会被关停。因此 2060 年的捕集吨数
    # 是被锁定的份额乘以 2060 年每单位份额的捕集量（其中的惩罚部分取 2060 年较小的那个值）。
    s_2060_unlocked, captured_2060_unlocked = _expected_ccs(scenario, assumptions, 0.5, 2060)
    assert s_2060_unlocked < s_ccs_expected
    assert y2["share"][0, ccs_idx] == pytest.approx(s_ccs_expected, rel=1e-3)
    captured_expected_2060 = captured_2060_unlocked / s_2060_unlocked * s_ccs_expected
    assert y2["edge_flow_mtpa"][0] == pytest.approx(captured_expected_2060, rel=1e-3)
    assert y2["cost_breakdown_cny"]["pipe_capex"] == pytest.approx(0.0, abs=1.0)


def test_rebuild_capex_charged_once_at_activation(tmp_path) -> None:
    """超过设计寿命、选择原址重建的电厂，只在激活期支付一次性重建 CAPEX。
    针对重复计费 bug 的回归测试：以前在 rebuild == 1 的每一期都会重新计入全额 CAPEX。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=2040)
    scenario = OptimizationScenario(
        experiment_id="TEST-REBUILD",
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        solver_time_limit=300,
    )
    solution = _solve_toy(paths, scenario)
    assert solution["status"] == "optimal"

    y1 = solution["year_solutions"][2050]
    y2 = solution["year_solutions"][2060]

    # 运行有利可图（电价 > 煤 + 运维），所以到期电厂在第 1 期原址重建，
    # 并在第 2 期保持重建状态（锁存）。
    assert y1["rebuild"][0] == pytest.approx(1.0)
    assert y2["rebuild"][0] == pytest.approx(1.0)
    retire_idx = 1  # PATHWAYS = (unabated, retire, ccs, biomass, beccs, ammonia)
    assert y1["share"][0, retire_idx] == pytest.approx(0.0, abs=1e-6)

    # 第 1 期的一次性 CAPEX：1000 MW x 3500 CNY/kW x 0.70 x 1000，再折现。
    expected_capex_t1 = 1000.0 * 3500.0 * 0.70 * 1000.0 / (1.0 + 0.06) ** (2050 - 2025)
    assert y1["cost_breakdown_cny"]["rebuild_capex"] == pytest.approx(expected_capex_t1, rel=1e-3)
    # 第 2 期没有重建 CAPEX，尽管电厂仍处于重建状态并在运行。
    assert y2["cost_breakdown_cny"]["rebuild_capex"] == pytest.approx(0.0, abs=1.0)

    # 重建后的电厂按 rebuild_efficiency（超超临界，USC）运行：热耗改善为
    # hr x 0.42/0.45，这必须体现在基线净运行成本里。
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


def test_ccs_capture_island_charged_per_build_and_rebuilt_at_end_of_life(tmp_path) -> None:
    """捕集岛建成时付一次钱，在役到寿命（20 a）为止；份额仍在就得按当年单价重建。固定运维按建设年的单价。

    目标 0.5 -> 0.3 -> 0.5，禁用退役。捕集份额锁定（2026-09-10）让 2030 年的份额一直保持到 2040 与 2050 年，
    较低的 2040 年目标被超额完成。2030 年建的捕集岛 2040 年仍在役，不再付 capex，固定运维仍按 2030 年的单价；
    2050 年满 20 年退出，按 2050 年单价重建。2026-09-30 前存量永不退出，2050 年不付钱，固定运维按当年单价。"""
    years3 = (2030, 2040, 2050)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 0.5, 2040: 0.7, 2050: 0.5})
    scenario = OptimizationScenario(
        experiment_id="TEST-CCSSTOCK",
        description="toy",
        planning_years=years3,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0, 490.0),
        pathway_disable=("retire",),
        solver_time_limit=300,
    )
    assumptions = _toy_assumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.planning_years
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, state)
    assert solution["status"] == "optimal"

    y1 = solution["year_solutions"][2030]
    y2 = solution["year_solutions"][2040]
    y3 = solution["year_solutions"][2050]

    # CCS 份额严格跟随目标（所有路径的成本都随份额增加）。
    ccs_idx = 2  # PATHWAYS = (unabated, retire, ccs, biomass, beccs, ammonia)
    # 能耗惩罚比例随时间下降，所以即使目标不变，满足给定目标的份额也逐年不同。
    s1_y2030, _ = _expected_ccs(scenario, assumptions, 0.5, 2030)
    s1_y2050, _ = _expected_ccs(scenario, assumptions, 0.5, 2050)
    s2_y2040, _ = _expected_ccs(scenario, assumptions, 0.3, 2040)
    assert s2_y2040 < s1_y2050 < s1_y2030
    assert y1["share"][0, ccs_idx] == pytest.approx(s1_y2030, rel=1e-3)
    # 已锁定：2040 年不下降，2050 年所需也不超过已在运行的份额。
    assert y2["share"][0, ccs_idx] == pytest.approx(s1_y2030, rel=1e-3)
    assert y3["share"][0, ccs_idx] == pytest.approx(s1_y2030, rel=1e-3)

    # CAPEX：第 1 期建、第 2 期不付、第 3 期重建，都按建设年的单价（学习曲线逐年下降）。
    rate = scenario.discount_rate
    annuity = (1.0 - (1.0 + rate) ** -10.0) / rate
    capex_rate = {
        year: assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * assumptions.ccs_learning_factor(year)
        for year in years
    }
    assert capex_rate[2050] < capex_rate[2040] < capex_rate[2030]
    capex_paid = {2030: capex_rate[2030], 2040: 0.0, 2050: capex_rate[2050]}
    # 固定运维（每 MW 每年 = 单价 x ccs_om_fraction）按在用捕集岛的建设年：2040 年仍是 2030 年的岛。
    om_rate = {
        2030: capex_rate[2030] * assumptions.ccs_om_fraction,
        2040: capex_rate[2030] * assumptions.ccs_om_fraction,
        2050: capex_rate[2050] * assumptions.ccs_om_fraction,
    }
    capex_indices = solution["capex_pathway_indices"]
    for year, ys, prev in ((2030, y1, None), (2040, y2, y1), (2050, y3, y2)):
        df = 1.0 / (1.0 + rate) ** (year - scenario.discount_base_year)
        costs = ys["cost_breakdown_cny"]
        assert costs["ccs_retrofit_capex"] == pytest.approx(1000.0 * capex_paid[year] * s1_y2030 * df, rel=1e-3, abs=1.0), year
        assert costs["ccs_om_cost"] == pytest.approx(1000.0 * om_rate[year] * s1_y2030 * df * annuity, rel=1e-3), year
        # 逐厂成本表与模型一致（未折现）。
        plant_cost = _build_plant_cost_table(
            prepared, year, ys["year_data"], ys["share"], ys["biomass_use_gj"],
            prev_share_values=None if prev is None else prev["share"],
            plant_reduction_mt=ys["plant_reduction_mt"],
            retrofit_new=ys["retrofit_new"], ccs_om_by_plant=ys["ccs_om_by_plant"],
            capex_pathway_indices=capex_indices,
        )
        assert float(plant_cost["ccs_retrofit_capex_cny"].iloc[0]) == pytest.approx(
            1000.0 * capex_paid[year] * s1_y2030, rel=1e-3, abs=1.0
        ), year
        assert float(plant_cost["ccs_om_cny"].iloc[0]) == pytest.approx(1000.0 * om_rate[year] * s1_y2030, rel=1e-3), year
    assert float(y2["retrofit_alive"][0]) == pytest.approx(s1_y2030, rel=1e-3)


def test_retired_plant_pays_no_capture_island_om(tmp_path) -> None:
    """2030 年建捕集岛，2040 年目标归零、全厂退役：捕集岛仍在寿命内，但不再运行，不付固定运维，也不再建。
    只开放 CCS 与退役，退役速率不设上限；目标缺口的罚价调高 10 倍，否则留一部分捕集、付缺口比全退便宜。
    2026-09-30 前固定运维按当年份额计，结论相同；这里锁住的是分代以后仍然成立：在用的捕集岛只需覆盖当年的
    捕集份额，闲置的在役部分不付固定运维。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 0.5, 2040: 0.0})
    scenario = OptimizationScenario(
        experiment_id="TEST-CCSRETIRE", description="toy", planning_years=(2030, 2040),
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0),
        pathway_disable=("biomass", "beccs", "ammonia"),
        max_new_retirement_share_per_period=1.0,
        solver_time_limit=300,
    )
    assumptions = replace(_toy_assumptions(), slack_penalty_cny_per_unit=5.0e10)
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, scenario.planning_years, state)
    assert solution["status"] == "optimal"
    y1, y2 = solution["year_solutions"][2030], solution["year_solutions"][2040]
    assert float(y2["slacks"]["target_shortfall_mt"]) == pytest.approx(0.0, abs=1e-6)
    ccs_idx, retire_idx = 2, 1  # PATHWAYS = (unabated, retire, ccs, biomass, beccs, ammonia)
    assert float(y1["share"][0, ccs_idx]) > 0.1
    assert float(y2["share"][0, retire_idx]) == pytest.approx(1.0, abs=1e-6)
    assert float(y2["retrofit_alive"][0]) == pytest.approx(float(y1["retrofit_new"][0, 0]), rel=1e-6)
    assert float(y1["ccs_om_by_plant"][0]) > 0.0
    assert float(y2["ccs_om_by_plant"][0]) == pytest.approx(0.0, abs=1.0)
    assert y2["cost_breakdown_cny"]["ccs_om_cost"] == pytest.approx(0.0, abs=1.0)
    assert y2["cost_breakdown_cny"]["ccs_retrofit_capex"] == pytest.approx(0.0, abs=1.0)


def test_unit_cf_boost_recovers_unboosted_accounting(tmp_path) -> None:
    """守护测试：retrofit_cf_boost = 1.0 时，已接入的逐路径核算必须精确退化为
    经典形式（减排 = eta x 份额 x 基线 E，再减去计入残余排放的能耗惩罚燃料排放）。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2050: 0.5, 2060: 0.5})
    scenario = OptimizationScenario(
        experiment_id="TEST-BOOST1",
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
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
