"""分代能力（2026-09-30）：工业捕集与氢路线能力到寿命退出、要用就重建；捕集的固定运维按建设年的单价、
只付在役且在用的那部分。

前四条直接建工业年块，不经 toy 输入，与 `test_capex_stock_and_lifetimes._capex_by_year` 同法；最后一条走全模型，
核对目标函数与结果。煤电捕集岛的同类测试（到寿命重建、按建设年付运维、退役后不付运维）在
`test_multiperiod_investment_logic.py`；不求解的（在役判定、铭牌定规模）在 `test_capex_stock_no_solver.py`。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import SolveState, _discount_factor, _year_objective_weight  # noqa: E402
from coal_retrofit.optimization.data_prep import prepare_inputs  # noqa: E402
from coal_retrofit.optimization.industry import (  # noqa: E402
    CCS,
    H2,
    IndustryInputs,
    IndustryYearData,
    add_industry_capacity,
    add_industry_monotonicity,
    add_industry_year,
    industry_capex_expr,
    industry_year_data,
)
from coal_retrofit.optimization.results_industry import _build_industry_detail_table  # noqa: E402
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario  # noqa: E402
from coal_retrofit.optimization.solver import _solve_joint_multi_period  # noqa: E402
from test_capex_stock_no_solver import _cement_hub  # noqa: E402
from test_h2_route_multiplier import _steel_hub  # noqa: E402
from toy_inputs import _toy_assumptions, _write_targets, _write_toy_inputs  # noqa: E402


def _solve(
    industry: IndustryInputs, share: dict[int, float], route: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, IndustryYearData]]:
    """固定各年 `route` 的份额，按求解器的权重（capex 乘折现，固定运维再乘 10 年年金）最小化现值，
    返回逐年未折现的 capex、捕集固定运维与系数块。"""
    scenario = OptimizationScenario(experiment_id="T", description="toy")
    assumptions = OptimizationAssumptions()
    model = gp.Model()
    model.Params.OutputFlag = 0
    h2_link_cost = np.array([1.0]) if route == H2 else None
    h2_membership = sparse.csr_matrix(np.ones((1, 1))) if route == H2 else None
    years = sorted(share)
    payloads, data = [], {}
    for year in years:
        data[year] = industry_year_data(industry, scenario, assumptions, year)
        payload = add_industry_year(
            model, industry, data[year], str(year), h2_link_cost=h2_link_cost, h2_hub_membership=h2_membership,
        )
        model.addConstr(payload.share[0, route] == share[year], name=f"fix_share_{year}")
        payloads.append(payload)
    add_industry_monotonicity(model, payloads, 1)
    ccs, _ = add_industry_capacity(model, years, payloads)
    capex = [industry_capex_expr(payload, routes=(route,)) for payload in payloads]
    om = [gp.quicksum(stock.fixed_om) for stock in ccs]
    annuity = _year_objective_weight(10, scenario.discount_rate)
    model.setObjective(gp.quicksum(
        _discount_factor(year, scenario.discount_base_year, scenario.discount_rate) * (c + annuity * o)
        for year, c, o in zip(years, capex, om)
    ))
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    return (
        {year: float(expr.getValue()) for year, expr in zip(years, capex)},
        {year: float(expr.getValue()) for year, expr in zip(years, om)},
        data,
    )


def test_capture_capacity_retires_after_20_years_and_is_rebuilt_at_that_years_price() -> None:
    """份额与产量不变：2030 年建的捕集能力在 2040 年仍在役，2050 年满 20 年退出，按 2050 年的单价重建，
    2060 年又在役。2026-09-30 前存量永不退出，2050 年不付钱。"""
    years = (2030, 2040, 2050, 2060)
    capex, _, data = _solve(_cement_hub({("cement", y): 1.0 for y in years}), {y: 1.0 for y in years}, CCS)
    need = data[2030].capacity_mt_per_share[0, CCS]
    assert data[2050].capex_cny_per_mt[0, CCS] < data[2030].capex_cny_per_mt[0, CCS]
    assert capex[2030] == pytest.approx(data[2030].capex_cny_per_mt[0, CCS] * need, rel=1e-9)
    assert capex[2040] == pytest.approx(0.0, abs=1e-3)
    assert capex[2050] == pytest.approx(data[2050].capex_cny_per_mt[0, CCS] * need, rel=1e-9)
    assert capex[2060] == pytest.approx(0.0, abs=1e-3)


def test_capture_fixed_om_is_priced_at_the_build_year() -> None:
    """2030 年建的捕集能力 2040 年仍按 2030 年的单价付固定运维；学习曲线让 2040 年的单价更低，
    但为省这点运维提前重建不合算。2026-09-30 前按当年单价 x 当年捕集量计。"""
    years = (2030, 2040)
    _, om, data = _solve(_cement_hub({("cement", y): 1.0 for y in years}), {y: 1.0 for y in years}, CCS)
    need = data[2030].capacity_mt_per_share[0, CCS]
    om30, om40 = data[2030].fixed_om_cny_per_mt[0, CCS], data[2040].fixed_om_cny_per_mt[0, CCS]
    assert om40 < 0.9 * om30
    assert om[2030] == pytest.approx(om30 * need, rel=1e-9)
    assert om[2040] == pytest.approx(om30 * need, rel=1e-9)


def test_idle_capture_capacity_pays_no_fixed_om() -> None:
    """产量 1.0 -> 0.5，份额 0.5 -> 0.8：2030 年建了 0.50 份能力，2040 年只用 0.40 份，
    固定运维只付在用的 0.40 份。"""
    capex, om, data = _solve(_cement_hub({("cement", 2030): 1.0, ("cement", 2040): 0.5}), {2030: 0.5, 2040: 0.8}, CCS)
    v30 = data[2030].capacity_mt_per_share[0, CCS]
    om30 = data[2030].fixed_om_cny_per_mt[0, CCS]
    assert capex[2040] == pytest.approx(0.0, abs=1e-3)
    assert om[2030] == pytest.approx(om30 * 0.5 * v30, rel=1e-9)
    assert om[2040] == pytest.approx(om30 * 0.4 * v30, rel=1e-9)


def test_h2_route_capacity_retires_after_25_years() -> None:
    """氢路线寿命 25 a：2030 年建的产能在 2050 年仍在役（20 < 25），2060 年退出，按原价重建（路线 capex
    不随年份变）。与管道同一在役判定：寿命不是 10 年的整数倍时，在役到下一个整 10 年为止。"""
    capex, _, data = _solve(_steel_hub(), {y: 1.0 for y in (2030, 2040, 2050, 2060)}, H2)
    need = data[2030].capacity_mt_per_share[0, H2]
    unit = data[2030].capex_cny_per_mt[0, H2]
    assert capex[2030] == pytest.approx(unit * need, rel=1e-9)
    assert capex[2040] == pytest.approx(0.0, abs=1e-3) and capex[2050] == pytest.approx(0.0, abs=1e-3)
    assert capex[2060] == pytest.approx(unit * need, rel=1e-9)


def test_solver_books_capture_fixed_om_and_reports_alive_capacity(tmp_path) -> None:
    """全模型求解：水泥组 2030、2040、2050 年要比 2030 年基线减 10%、30%、50%（toy 水泥只有 CCS），三年都要新建
    捕集能力。目标函数里的 `industry_cost` 逐年等于折现、乘年金后的明细表 `cost_annual_cny` 之和，其中含在役且在用
    的捕集能力按建设年单价的固定运维；结果里的在役能力是寿命内历年新建之和，2050 年 2030 年建的已满 20 年退出。
    目标函数漏掉捕集固定运维、或结果把本年新建当作在役时，这里对不上。"""
    years = (2030, 2040, 2050)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {year: 1.0 for year in years}, {2030: 0.9, 2040: 0.7, 2050: 0.5})
    scenario = OptimizationScenario(
        experiment_id="TEST-INDOM", description="toy", planning_years=years, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0, 490.0),
        solver_time_limit=300, mip_gap=1e-9,
    )
    assumptions = _toy_assumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, state)
    assert solution["status"] == "optimal"
    ys = solution["year_solutions"]
    weight = _year_objective_weight(10, scenario.discount_rate)
    for year in years:
        table = _build_industry_detail_table(
            prepared, year, ys[year]["year_data"].industry, ys[year]["industry_share"],
            capacity_mt=ys[year]["industry_capacity_mt"], new_capacity_mt=ys[year]["industry_new_capacity_mt"],
            ccs_fixed_om_cny=ys[year]["industry_ccs_om_by_hub"],
        )
        df = _discount_factor(year, scenario.discount_base_year, scenario.discount_rate)
        booked = float(ys[year]["cost_breakdown_cny"]["industry_cost"])
        assert booked == pytest.approx(df * weight * float(table["cost_annual_cny"].sum()), rel=1e-6), year
        assert float(ys[year]["slacks"]["target_shortfall_mt"]) == pytest.approx(0.0, abs=1e-6), year
        assert float(ys[year]["industry_ccs_om_by_hub"].sum()) > 0.0, year
    cement = list(prepared.industry.hubs["hub_id"]).index("C1")
    new = [float(ys[year]["industry_new_capacity_mt"][cement, CCS]) for year in years]
    alive = [float(ys[year]["industry_capacity_mt"][cement, CCS]) for year in years]
    assert min(new) > 0.0
    assert alive[0] == pytest.approx(new[0], rel=1e-9, abs=1e-9)
    assert alive[1] == pytest.approx(new[0] + new[1], rel=1e-9, abs=1e-9)
    assert alive[2] == pytest.approx(new[1] + new[2], rel=1e-9, abs=1e-9)
