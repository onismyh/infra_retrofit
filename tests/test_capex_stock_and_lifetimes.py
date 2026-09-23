"""一次性 capex 的计费基数与资产寿命（2026-09-23 的模型改动），每项一组 toy 回归测试。

(a) 工业 capex 计在能力存量的增量上，不计在路线份额的增量上。
(b) BECCS 的捕集岛按 CCS 计价，生物质改造只走掺烧档位 capex，各收一次。
(c) 管道到寿命后可在原址重铺：累计新增上限、热启动与结果表都只数在役的管。
(d) 成本乘子两侧都只乘 capex 与随 capex 的固定运维，不乘能耗、耗材与 BECCS 的掺烧运维。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.constants_industry import INDUSTRY_CCS_FIXED_OM_FRACTION  # noqa: E402
from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState  # noqa: E402
from coal_retrofit.optimization.data_prep import prepare_inputs  # noqa: E402
from coal_retrofit.optimization.industry import (  # noqa: E402
    CCS,
    IndustryInputs,
    add_industry_monotonicity,
    add_industry_year,
    industry_capex_expr,
    industry_year_data,
)
from coal_retrofit.optimization.results_network import _alive_edge_added_stock  # noqa: E402
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario  # noqa: E402
from coal_retrofit.optimization.solver import _solve_joint_multi_period  # noqa: E402
from coal_retrofit.optimization.solver_start import _apply_rounded_start  # noqa: E402
from coal_retrofit.optimization.year_matrices import _build_year_matrices  # noqa: E402
from coal_retrofit.optimization.year_types import YearData  # noqa: E402
from test_multiperiod_investment_logic import _write_targets, _write_toy_inputs  # noqa: E402


# ---------------------------------------------------------- (a) 工业 capex 按能力存量计 ---
def _cement_hub(output_index: dict[tuple[str, int], float]) -> IndustryInputs:
    """一个水泥 hub（只有 CCS 路线），产量按给定指数逐年缩放。"""
    hubs = pd.DataFrame({
        "hub_id": ["C1"], "sector": ["cement"], "province": ["Shanxi"],
        "longitude": [112.0], "latitude": [37.0],
        "production_kt_per_year": [1000.0], "co2_mt_per_year": [0.8],
        "process_co2_mt_per_year": [0.5], "h2_demand_kt_per_year": [0.0],
        "water_m3_per_year": [1.0e6], "target_group": ["cement"],
    })
    return IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0, 2040: 18.0}, output_index=output_index)


def _capex_by_year(industry: IndustryInputs, ccs_share: dict[int, float]) -> tuple[dict[int, float], dict]:
    """固定各年 CCS 份额，最小化一次性 capex 之和，返回每年的 capex 与逐年系数。"""
    scenario = OptimizationScenario(experiment_id="T", description="toy")
    assumptions = OptimizationAssumptions()
    model = gp.Model()
    model.Params.OutputFlag = 0
    years = sorted(ccs_share)
    payloads, data = [], {}
    for year in years:
        data[year] = industry_year_data(industry, scenario, assumptions, year)
        payload = add_industry_year(model, industry, data[year], str(year))
        model.addConstr(payload.share[0, CCS] == ccs_share[year], name=f"fix_ccs_{year}")
        payloads.append(payload)
    add_industry_monotonicity(model, payloads, 1)
    exprs = [
        industry_capex_expr(payload, payloads[i - 1] if i else None, routes=(CCS,))
        for i, payload in enumerate(payloads)
    ]
    model.setObjective(gp.quicksum(exprs))
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    values = {year: float(expr.getValue()) for year, expr in zip(years, exprs)}
    return values, data


def test_industry_capex_charges_output_growth_at_a_constant_share() -> None:
    """份额两年都是 1，产量 1.0 -> 1.5：多出的 50% 捕集能力要付 capex。
    按份额增量计时 2040 年的增量为 0，这部分能力从来不付钱。"""
    industry = _cement_hub({("cement", 2030): 1.0, ("cement", 2040): 1.5})
    capex, data = _capex_by_year(industry, {2030: 1.0, 2040: 1.0})
    v30 = data[2030].capacity_mt_per_share[0, CCS]
    v40 = data[2040].capacity_mt_per_share[0, CCS]
    assert v40 == pytest.approx(1.5 * v30, rel=1e-12)
    assert capex[2030] == pytest.approx(data[2030].capex_cny_per_mt[0, CCS] * v30, rel=1e-9)
    assert capex[2040] == pytest.approx(data[2040].capex_cny_per_mt[0, CCS] * (v40 - v30), rel=1e-9)
    assert capex[2040] > 0.0


def test_industry_capex_not_recharged_on_idle_capacity_of_a_declining_sector() -> None:
    """产量 1.0 -> 0.5，份额 0.5 -> 0.8：2040 年只需 0.8 x 0.5 = 0.40 份 2030 年能力，
    2030 年已建 0.50 份，不必新建。按份额增量计会再收 0.3 x 0.5 份的钱。"""
    industry = _cement_hub({("cement", 2030): 1.0, ("cement", 2040): 0.5})
    capex, data = _capex_by_year(industry, {2030: 0.5, 2040: 0.8})
    v30 = data[2030].capacity_mt_per_share[0, CCS]
    assert capex[2030] == pytest.approx(data[2030].capex_cny_per_mt[0, CCS] * 0.5 * v30, rel=1e-9)
    assert capex[2040] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------- (b) BECCS 只收一次生物质改造费 ---
def _solve_toy(paths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> dict:
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, scenario.planning_years, state)
    assert solution["status"] == "optimal"
    return solution


def test_beccs_pays_the_capture_island_once_and_the_biomass_conversion_once(tmp_path) -> None:
    """只开放 BECCS，2050 年电力目标为零排放：必须掺生物质。捕集岛按 CCS capex 计一次，
    生物质改造按掺烧档位 capex 计一次；2026-09-23 前捕集岛之上还要再付 +1 000 CNY/kW。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    bio = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv")
    bio["longitude"], bio["latitude"], bio["province_name"], bio["available_gj"] = 112.05, 37.0, "Shanxi", 1.0e9
    bio.to_csv(paths.inputs_dir / "biomass_supply_curve.csv", index=False)
    _write_targets(paths, {2030: 1.0, 2040: 1.0, 2050: 0.0, 2060: 0.0})
    scenario = OptimizationScenario(
        experiment_id="TEST-BECCS", description="toy", planning_years=(2050, 2060),
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=("retire", "ccs", "biomass", "ammonia"),
        solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0))
    y50 = _solve_toy(paths, scenario, assumptions)["year_solutions"][2050]
    capacity_mw = 1000.0
    beccs_share = float(y50["share"][0, PATHWAY_INDEX["beccs"]])
    blend_level = float(y50["blend_level_b"][0])
    assert beccs_share > 0.5 and blend_level >= 1.0 - 1e-6
    assert y50["retrofit_installed"].shape == (1, 1)
    assert float(y50["retrofit_installed"][0, 0]) == pytest.approx(beccs_share, rel=1e-6)
    df = 1.0 / (1.0 + scenario.discount_rate) ** (2050 - scenario.discount_base_year)
    island = assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * capacity_mw * assumptions.ccs_learning_factor(2050)
    costs = y50["cost_breakdown_cny"]
    assert costs["ccs_retrofit_capex"] == pytest.approx(island * beccs_share * df, rel=1e-6)
    blend = assumptions.biomass_upgrade_capex_cny_per_mw_per_level * capacity_mw * blend_level
    assert costs["blend_upgrade_capex"] == pytest.approx(blend * df, rel=1e-6)
    assert not hasattr(assumptions, "beccs_retrofit_capex_cny_per_kw")


# ------------------------------------------------- (c) 管道到寿命后可原位重建 ---
def _one_pipe_assumptions() -> OptimizationAssumptions:
    """每条边只容一根 5 Mtpa 管（标准管 5 x 1 根）；封存不爬坡。"""
    return OptimizationAssumptions(
        storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0),
        standard_pipe_capacity_mtpa=5.0,
        max_parallel_pipes=1,
    )


def test_alive_edge_stock_drops_pipes_at_the_end_of_their_lifetime() -> None:
    """在役 = 建成年早于当年且未满寿命；满 30 年当年即退出，当年新建的不算往期存量。"""
    added = {2030: np.array([5.0, 0.0]), 2050: np.array([0.0, 2.0])}
    np.testing.assert_allclose(_alive_edge_added_stock(added, 2050, 30, 2), [5.0, 0.0])
    np.testing.assert_allclose(_alive_edge_added_stock(added, 2059, 30, 2), [5.0, 2.0])
    np.testing.assert_allclose(_alive_edge_added_stock(added, 2060, 30, 2), [0.0, 2.0])


def test_expired_pipeline_can_be_rebuilt_in_place(tmp_path) -> None:
    """1 座电厂 - 1 条边 - 1 个汇，只开放 CCS，两年都要捕集约一半排放（> 2 Mtpa，只有 5 Mtpa 档够用）。
    2030 年铺的管 2060 年满 30 年寿命，2060 年必须在原址重铺。2026-09-23 前累计新增上限
    把已到寿命的管也算进去（5 + 5 > 5），2060 年铺不了，只能走管道松弛或目标缺口。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 0.5, 2040: 0.5, 2050: 0.5, 2060: 0.5})
    scenario = OptimizationScenario(
        experiment_id="TEST-REBUILD", description="toy", planning_years=(2030, 2060),
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 550.0),
        pathway_disable=("retire", "biomass", "beccs", "ammonia"),
        solver_time_limit=300,
    )
    solution = _solve_toy(paths, scenario, _one_pipe_assumptions())
    for year in (2030, 2060):
        ys = solution["year_solutions"][year]
        assert float(ys["new_cap_mtpa"][0]) == pytest.approx(5.0, abs=1e-6), year
        assert float(np.sum(ys["slacks"]["edge_slack_mtpa"])) == pytest.approx(0.0, abs=1e-6), year
        assert float(ys["slacks"]["target_shortfall_mt"]) == pytest.approx(0.0, abs=1e-6), year


def test_warm_start_reseeds_an_expired_pipe(tmp_path) -> None:
    """LP 热启动取整同样只数在役的管：2030 年铺满的边 2050 年不能再铺，2060 年到寿命后重铺。"""
    assumptions = _one_pipe_assumptions()
    years, n_tiers = (2030, 2050, 2060), len(assumptions.pipe_capacity_tiers_mtpa)
    model = gp.Model()
    model.Params.OutputFlag = 0
    for year in years:
        model.addMVar((1, n_tiers), vtype=gp.GRB.INTEGER, ub=1.0, name=f"pipe_count_{year}")
        model.addMVar(1, vtype=gp.GRB.BINARY, name=f"add_cap_{year}")
        model.addMVar(1, vtype=gp.GRB.BINARY, name=f"build_edge_{year}")
    sol = tmp_path / "relaxed.sol"
    sol.write_text("".join(f"new_cap_mtpa_{year}[0] 5.0\n" for year in years), encoding="utf-8")
    _apply_rounded_start(model, sol, assumptions)
    model.update()
    five = assumptions.pipe_capacity_tiers_mtpa.index(5.0)
    start = {
        year: [model.getVarByName(f"pipe_count_{year}[0,{k}]").Start for k in range(n_tiers)]
        for year in years
    }
    assert start[2030][five] == 1.0 and sum(start[2030]) == 1.0
    assert sum(start[2050]) == 0.0
    assert start[2060][five] == 1.0 and sum(start[2060]) == 1.0
    assert model.getVarByName("add_cap_2060[0]").Start == 1.0


# ------------------------------------- (d) 成本乘子只乘 capex 与随 capex 的固定运维 ---
def _toy_year_matrices(root, year: int, **scenario_overrides) -> YearData:
    """toy 输入上 `year` 年的煤电系数矩阵。"""
    paths = _write_toy_inputs(root, retirement_year=9999)
    _write_targets(paths, {2030: 1.0, 2040: 1.0, 2050: 1.0, 2060: 1.0})
    scenario = OptimizationScenario(
        experiment_id="TEST-MULT", description="toy", planning_years=(2050, 2060),
        sector_target_source="toy", **scenario_overrides,
    )
    assumptions = OptimizationAssumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    return _build_year_matrices(prepared, scenario, assumptions, year, state)


def test_ccs_cost_multiplier_scales_capex_and_its_fixed_om_only(tmp_path) -> None:
    """煤电：乘子 2 使捕集岛 capex 与其固定运维翻倍；能耗惩罚与 BECCS 每 MWh 的掺烧运维
    （与纯掺烧同为 30 元/MWh）不变。2026-09-23 前固定运维不翻倍，掺烧运维却翻倍。"""
    base = _toy_year_matrices(tmp_path / "m1", 2050)
    doubled = _toy_year_matrices(tmp_path / "m2", 2050, ccs_cost_multiplier=2.0)
    for k in (PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]):
        assert base.ccs_retrofit_capex_matrix[0, k] > 0.0 and base.ccs_om_matrix[0, k] > 0.0
        assert base.energy_penalty_matrix[0, k] > 0.0
        assert doubled.ccs_retrofit_capex_matrix[0, k] == pytest.approx(2.0 * base.ccs_retrofit_capex_matrix[0, k], rel=1e-12)
        assert doubled.ccs_om_matrix[0, k] == pytest.approx(2.0 * base.ccs_om_matrix[0, k], rel=1e-12)
        assert doubled.energy_penalty_matrix[0, k] == pytest.approx(base.energy_penalty_matrix[0, k], rel=1e-12)
    assert base.fixed_cost_matrix[0, PATHWAY_INDEX["beccs"]] > 0.0
    np.testing.assert_array_equal(doubled.fixed_cost_matrix, base.fixed_cost_matrix)


def test_industry_cost_multiplier_leaves_energy_and_consumables_alone() -> None:
    """工业 CCS：乘子 2 使 capex 与固定运维翻倍；蒸汽、电与耗材按模型价格计，不乘。
    2026-09-23 前能耗与耗材也翻倍。"""
    industry = _cement_hub({("cement", 2030): 1.0})
    assumptions = OptimizationAssumptions()
    base = industry_year_data(industry, OptimizationScenario(experiment_id="T", description="toy"), assumptions, 2030)
    doubled = industry_year_data(
        industry, OptimizationScenario(experiment_id="T", description="toy", industry_cost_multiplier=2.0),
        assumptions, 2030,
    )
    assert doubled.capex_cny_per_mt[0, CCS] == pytest.approx(2.0 * base.capex_cny_per_mt[0, CCS], rel=1e-12)
    fixed_om = base.capacity_mt_per_share[0, CCS] * base.capex_cny_per_mt[0, CCS] * INDUSTRY_CCS_FIXED_OM_FRACTION
    energy_and_consumables = float(base.opex_cny[0, CCS]) - fixed_om
    assert energy_and_consumables > 0.0
    # 年度成本 = 固定运维 + 能耗与耗材；乘子只让固定运维那一份翻倍。
    assert doubled.opex_cny[0, CCS] == pytest.approx(2.0 * fixed_om + energy_and_consumables, rel=1e-9)
