"""一次性 capex 的计费基数与资产寿命（2026-09-23 的模型改动），每项一组 toy 回归测试。

(a) 工业 capex 计在能力存量的增量上，不计在路线份额的增量上。
(b) BECCS 的捕集岛按 CCS 计价，生物质改造只走掺烧档位 capex，各收一次。
(c) 管道到寿命后可在原址重铺：累计新增上限、热启动与结果表都只数在役的管。
(d) 成本乘子两侧都只乘 capex 与随 capex 的固定运维，不乘能耗、耗材与 BECCS 的掺烧运维。

不求解的几条（(b) 的 BECCS 固定运维、(c) 的在役存量、(d) 的两侧成本乘子、工业明细表的 capital 列）
在 `test_capex_stock_no_solver.py`。
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

if TYPE_CHECKING:
    from coal_retrofit.optimization.year_types import SolveResult

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState, _discount_factor  # noqa: E402
from coal_retrofit.optimization.data_prep import prepare_inputs  # noqa: E402
from coal_retrofit.optimization.industry import (  # noqa: E402
    CCS,
    H2,
    IndustryInputs,
    add_industry_monotonicity,
    add_industry_year,
    industry_capex_expr,
    industry_year_data,
)
from coal_retrofit.optimization.results_industry import _build_industry_detail_table  # noqa: E402
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario  # noqa: E402
from coal_retrofit.optimization.solver import _solve_joint_multi_period  # noqa: E402
from coal_retrofit.optimization.solver_start import _apply_rounded_start  # noqa: E402
from test_capex_stock_no_solver import _cement_hub  # noqa: E402
from test_h2_route_multiplier import _steel_hub  # noqa: E402
from toy_inputs import _toy_assumptions, _write_targets, _write_toy_inputs  # noqa: E402


# ---------------------------------------------------------- (a) 工业 capex 按能力存量计 ---
def _capex_by_year(
    industry: IndustryInputs, share: dict[int, float], route: int = CCS,
) -> tuple[dict[int, float], dict]:
    """固定各年 `route` 的份额，最小化一次性 capex 的现值之和，返回每年未折现的 capex 与逐年系数。
    与求解器一样按 `_discount_factor` 折现：氢路线的单价不随年份下降，不折现时，产量增长的用例里
    2030 年的存量在一段区间里都是最优解，结果取决于求解器参数。
    氢路线没有氢链路时份额被钉在 0，所以测氢路线时给唯一的 hub 接一条链路。"""
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
    exprs = [
        industry_capex_expr(payload, payloads[i - 1] if i else None, routes=(route,))
        for i, payload in enumerate(payloads)
    ]
    model.setObjective(gp.quicksum(
        _discount_factor(year, scenario.discount_base_year, scenario.discount_rate) * expr
        for year, expr in zip(years, exprs)
    ))
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


def _steel_hub_indexed(output_index: dict[tuple[str, int], float]) -> IndustryInputs:
    """`_steel_hub` 加上逐年产量指数。"""
    return replace(_steel_hub(), output_index=output_index)


def test_h2_route_capex_charges_output_growth_at_a_constant_share() -> None:
    """氢路线与 CCS 同法：份额两年都是 1，产量 1.0 -> 1.5，多出的 50% 产能要付 capex。
    锁住三件事：氢路线的产能存量有下界，所需产能随产量指数变化，capex 计在存量的增量上。"""
    industry = _steel_hub_indexed({("steel_bf_bof", 2030): 1.0, ("steel_bf_bof", 2040): 1.5})
    capex, data = _capex_by_year(industry, {2030: 1.0, 2040: 1.0}, route=H2)
    k30 = data[2030].capacity_mt_per_share[0, H2]
    k40 = data[2040].capacity_mt_per_share[0, H2]
    assert k40 == pytest.approx(1.5 * k30, rel=1e-12)
    assert capex[2030] == pytest.approx(data[2030].capex_cny_per_mt[0, H2] * k30, rel=1e-9)
    assert capex[2040] == pytest.approx(data[2040].capex_cny_per_mt[0, H2] * (k40 - k30), rel=1e-9)
    assert capex[2040] > 0.0


def test_h2_route_capex_not_recharged_on_idle_capacity_of_a_declining_sector() -> None:
    """产量 1.0 -> 0.5，份额 0.5 -> 0.8：2040 年只需 0.40 份 2030 年产能，已建 0.50 份，不必新建。
    产能存量跨年不降；去掉这条单调约束，存量会缩回 0.40，2040 年的 capex 成为负数。"""
    industry = _steel_hub_indexed({("steel_bf_bof", 2030): 1.0, ("steel_bf_bof", 2040): 0.5})
    capex, data = _capex_by_year(industry, {2030: 0.5, 2040: 0.8}, route=H2)
    k30 = data[2030].capacity_mt_per_share[0, H2]
    assert capex[2030] == pytest.approx(data[2030].capex_cny_per_mt[0, H2] * 0.5 * k30, rel=1e-9)
    assert capex[2040] == pytest.approx(0.0, abs=1e-6)


def test_industry_capex_expr_charges_each_route_on_its_own_capacity_increment() -> None:
    """氢路线的 capex 与 CCS 同法：本年单位 capex x 本路线能力存量的增量，第一年按整个存量计。
    求解器分别按 CCS、H2 请求，好让残值台账里两项各带各的寿命，所以两列不得串。
    不求解：能力存量直接给数值，表达式只剩常数项。"""
    industry = _steel_hub()
    data = industry_year_data(industry, OptimizationScenario(experiment_id="T", description="toy"), OptimizationAssumptions(), 2030)
    unit = data.capex_cny_per_mt
    assert unit[0, CCS] > 0.0 and unit[0, H2] > 0.0
    previous = SimpleNamespace(capacity_mt=np.array([[0.0, 0.4, 0.1]]), year_data=data)
    current = SimpleNamespace(capacity_mt=np.array([[0.0, 0.6, 0.3]]), year_data=data)
    h2 = industry_capex_expr(current, previous, routes=(H2,))
    ccs = industry_capex_expr(current, previous, routes=(CCS,))
    assert h2.getConstant() == pytest.approx(unit[0, H2] * 0.2, rel=1e-12)
    assert ccs.getConstant() == pytest.approx(unit[0, CCS] * 0.2, rel=1e-12)
    assert industry_capex_expr(current, None, routes=(H2,)).getConstant() == pytest.approx(unit[0, H2] * 0.3, rel=1e-12)


def _add_steel_hub_near_h2_node(paths, steel_caps: dict[int, float]) -> None:
    """在 toy 输入上加一个长流程钢 hub（100 万 t/a，需氢 0.081 t/t，有氢路线）和紧挨它的氢节点，
    钢铁组按 `steel_caps` 设上限；煤电组与水泥组仍是 `_write_toy_inputs` 写的 1.0。"""
    inputs = paths.inputs_dir
    hubs = pd.read_csv(inputs / "industry_hubs.csv")
    steel = hubs.iloc[[0]].assign(
        hub_id="ST1", sector="steel_bf_bof", sector_zh="长流程钢", longitude=112.3,
        co2_mt_per_year=2.0, process_co2_mt_per_year=0.2, h2_demand_kt_per_year=81.0,
        has_h2_route=True, water_m3_per_year=3.0e6,
    )
    pd.concat([hubs, steel], ignore_index=True).to_csv(inputs / "industry_hubs.csv", index=False)
    index = pd.read_csv(inputs / "industry_output_index_toy.csv")
    pd.concat([index, index.assign(sector="steel_bf_bof")], ignore_index=True).to_csv(
        inputs / "industry_output_index_toy.csv", index=False
    )
    nodes = pd.read_csv(inputs / "ammonia_supply_curve.csv")
    near = nodes.iloc[[0]].assign(ammonia_node_id="A2", longitude=112.4, latitude=37.0, province_name="Shanxi")
    pd.concat([nodes, near], ignore_index=True).to_csv(inputs / "ammonia_supply_curve.csv", index=False)
    targets = pd.read_csv(inputs / "sector_targets_toy.csv")
    steel_rows = pd.DataFrame([
        {"sector_group": "steel", "planning_year": year, "cap_fraction_of_2030": cap}
        for year, cap in steel_caps.items()
    ])
    pd.concat([targets, steel_rows], ignore_index=True).to_csv(inputs / "sector_targets_toy.csv", index=False)


def test_solver_books_h2_route_capex_in_the_objective(tmp_path) -> None:
    """全模型求解：钢铁组 2040、2050 年要比 2030 年减 70%。长流程钢的 CCS 最多减约 63%（捕集 90% x
    (1 - 再生蒸汽放空约 0.30)），氢路线减 95%，所以必须新建氢路线产能。目标函数里的 `industry_capex`
    逐年等于折现后的明细表 `cost_capital_cny` 之和。目标函数漏掉氢路线 capex、残值台账里仍有时，能力存量
    没有上界，期末残值抵扣使目标无下界，求解返回 unbounded，在 status 断言处失败；台账里也一起漏掉时，
    2040 年两者差出氢路线那一份。"""
    years = (2030, 2040, 2050)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _add_steel_hub_near_h2_node(paths, {2030: 1.0, 2040: 0.3, 2050: 0.3})
    # 煤电不约束、碳价为零；关掉退役与掺氨，煤电侧不动，也不与钢铁 hub 争同一个氢节点。
    scenario = OptimizationScenario(
        experiment_id="TEST-H2CAPEX", description="toy", planning_years=years,
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0, 490.0),
        pathway_disable=("retire", "ammonia"),
        solver_time_limit=300,
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
    previous = None
    for year in years:
        capacity = ys[year]["industry_capacity_mt"]
        table = _build_industry_detail_table(
            prepared, year, ys[year]["year_data"].industry, ys[year]["industry_share"],
            capacity_mt=capacity, prev_capacity_mt=previous,
        )
        df = _discount_factor(year, scenario.discount_base_year, scenario.discount_rate)
        booked = float(ys[year]["cost_breakdown_cny"]["industry_capex"])
        assert booked == pytest.approx(df * float(table["cost_capital_cny"].sum()), rel=1e-6, abs=1.0), year
        assert float(ys[year]["slacks"]["target_shortfall_mt"]) == pytest.approx(0.0, abs=1e-6), year
        previous = capacity
    steel = list(prepared.industry.hubs["hub_id"]).index("ST1")
    new_h2_mt = float(ys[2040]["industry_capacity_mt"][steel, H2] - ys[2030]["industry_capacity_mt"][steel, H2])
    assert new_h2_mt > 0.1


# ------------------------------------------------- (b) BECCS 只收一次生物质改造费 ---
def _solve_toy(paths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> SolveResult:
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
