from __future__ import annotations

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.results import _build_plant_cost_table
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.paths import ProjectPaths
from toy_inputs import YEARS, _toy_assumptions, _write_targets, _write_toy_inputs  # noqa: E402


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


def test_ccs_retrofit_capex_charged_on_installed_stock_not_share_delta(tmp_path) -> None:
    """捕集岛一旦建成，只付一次钱，并持续运行。

    目标 0.5 -> 0.3 -> 0.5，禁用退役。在捕集份额锁定（2026-09-10）之前，
    CCS 份额会在第 2 期下降、第 3 期回升，本测试守护的是按存量增量计价（回升时
    不再计第二笔 capex）。有了锁定，份额根本不能下降——2030 年的份额一直保持到
    2040 与 2050 年，较低的 2040 年目标被超额完成，而 CAPEX 仍只在第 1 期计一次。"""
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

    # CAPEX 只在第 1 期按全部已装存量计；第 2、3 期不再增加
    # （第 3 期的回升不超出已装存量）。
    rate = scenario.discount_rate
    df1 = 1.0 / (1.0 + rate) ** (2030 - scenario.discount_base_year)
    capex_rate = assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * assumptions.ccs_learning_factor(2030)
    expected_capex_y1 = 1000.0 * capex_rate * s1_y2030 * df1
    assert y1["cost_breakdown_cny"]["ccs_retrofit_capex"] == pytest.approx(expected_capex_y1, rel=1e-3)
    assert y2["cost_breakdown_cny"]["ccs_retrofit_capex"] == pytest.approx(0.0, abs=1.0)
    assert y3["cost_breakdown_cny"]["ccs_retrofit_capex"] == pytest.approx(0.0, abs=1.0)

    # 逐厂成本报表必须与模型一致：已装存量 CAPEX 只在第 1 期计，第 2、3 期为零。
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
