"""分代能力（2026-09-30）：工业捕集与氢路线能力到寿命退出、要用就重建；捕集的固定运维按建设年的单价、
只付在役且在用的那部分。煤电空冷与掺烧能力 2026-10-02 起同法分代；期末残值只计最后一年仍在用的部分。

前四条直接建工业年块，不经 toy 输入，与 `test_capex_stock_and_lifetimes._capex_by_year` 同法；第五至七条走全模型，
核对目标函数与结果、残值台账（氢路线、掺烧各层）；其余只建分代约束（空冷与掺烧的接线、掺烧能力按档位分层、期末在用量）。
煤电捕集岛的同类测试（到寿命重建、按建设年付运维、退役后不付运维）在 `test_multiperiod_investment_logic.py`；不求解的
（在役判定、铭牌定规模）在 `test_capex_stock_no_solver.py`。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import (  # noqa: E402
    PATHWAY_INDEX,
    SolveState,
    _discount_factor,
    _expr_value,
    _year_objective_weight,
)
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
from coal_retrofit.optimization.salvage import remaining_fraction  # noqa: E402
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario  # noqa: E402
from coal_retrofit.optimization.solver import _solve_joint_multi_period  # noqa: E402
from coal_retrofit.optimization.vintage import add_vintage_stock  # noqa: E402
from coal_retrofit.optimization.year_types import YearData  # noqa: E402
from test_blend_ratios import SCENARIO  # noqa: E402
from test_capex_stock_and_lifetimes import _add_steel_hub_near_h2_node, _solve_toy  # noqa: E402
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


def test_solver_salvages_only_the_h2_capacity_still_in_use_at_the_end(tmp_path) -> None:
    """全模型求解，残值台账的接线（`model_costs`）：钢铁组 2050、2060 年要比 2030 年减 70%，逼出 2050 年的氢路线产能
    （寿命 25 年，到期末 2070 年未折旧 0.2）；2060 年长流程钢产量指数降到 0.5，在用的氢路线产能随之减少，不再新建。
    台账里 2050 年氢路线一项是 2050 年单价 x 2060 年在用量，小于按建成量计（2026-10-02 前的口径）；残值抵扣是各项
    台账值的未折旧部分从期末折现之和。"""
    years = (2050, 2060)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _add_steel_hub_near_h2_node(paths, {2050: 0.3, 2060: 0.3})
    index_path = paths.inputs_dir / "industry_output_index_toy.csv"
    index = pd.read_csv(index_path)
    index.loc[(index["sector"] == "steel_bf_bof") & (index["planning_year"] == 2060), "output_index"] = 0.5
    index.to_csv(index_path, index=False)
    # 煤电不约束、碳价为零；关掉退役与掺氨，与 `test_solver_books_h2_route_capex_in_the_objective` 同。
    scenario = OptimizationScenario(
        experiment_id="TEST-H2SALVAGE", description="toy", planning_years=years, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=(490.0, 540.0),
        pathway_disable=("retire", "ammonia"), solver_time_limit=300, mip_gap=1e-9,
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
    steel = list(prepared.industry.hubs["hub_id"]).index("ST1")
    built = float(ys[2050]["industry_new_capacity_mt"][steel, H2])
    assert built > 0.1
    assert float(ys[2060]["industry_new_capacity_mt"][steel, H2]) == pytest.approx(0.0, abs=1e-9)
    in_use = float(ys[2060]["year_data"].industry.capacity_mt_per_share[steel, H2] * ys[2060]["industry_share"][steel, H2])
    assert in_use < 0.6 * built
    unit = float(ys[2050]["year_data"].industry.capex_cny_per_mt[steel, H2])
    booked = {name: value for name, value, _ in ys[2050]["salvage_ledger"]}
    assert booked["industry_h2_capex"] == pytest.approx(unit * in_use, rel=1e-6)
    end_year = 2070
    df_end = _discount_factor(end_year, scenario.discount_base_year, scenario.discount_rate)
    expected = -df_end * sum(
        remaining_fraction(year, life, end_year) * value
        for year in years for _, value, life in ys[year]["salvage_ledger"]
    )
    assert float(ys[2060]["cost_breakdown_cny"]["salvage_credit"]) == pytest.approx(expected, rel=1e-6)


def test_solver_books_blend_capex_om_and_salvage_on_every_level_layer(tmp_path) -> None:
    """全模型求解，掺烧 capex、固定运维与残值台账逐层相加（`model_costs`，2026-10-02 起）。只开 BECCS、独热档位，2060 年
    电力目标为零排放：2060 年才建掺烧能力，所选档位高于第 1 档（前置断言：在用的掺烧能力 Σ 档位下标 x 份额大于份额）。
    2060 年的 `blend_upgrade_capex` 是每层单价（装机 x 每档单价）x 各层新建之和，即 x 在用的掺烧能力；残值台账的同一项是
    同一单价 x 各层期末在用量之和，也等于它；`incremental_om` 只剩掺烧能力的固定运维（BECCS 与未改造的每 MWh 附加项为 0），
    是单价 x `biomass_upgrade_om_fraction` x 在用的掺烧能力。只计第 1 层时三者都只剩单价 x 份额。连续 hub 下同一情景
    高于第 1 档的份额太少。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    bio = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv")
    bio["longitude"], bio["latitude"], bio["province_name"], bio["available_gj"] = 112.05, 37.0, "Shanxi", 1.0e9
    bio.to_csv(paths.inputs_dir / "biomass_supply_curve.csv", index=False)
    _write_targets(paths, {2030: 1.0, 2040: 1.0, 2050: 1.0, 2060: 0.0})
    scenario = OptimizationScenario(
        experiment_id="TEST-BLENDLAYERS", description="toy", planning_years=(2050, 2060), sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=("retire", "ccs", "biomass", "ammonia"), solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(
        hub_decisions_continuous=False,
    )
    y60 = _solve_toy(paths, scenario, assumptions)["year_solutions"][2060]
    share = float(y60["share"][0, PATHWAY_INDEX["beccs"]])
    level = float(y60["blend_level_b"][0])
    assert share > 0.5 and level > share + 0.01
    unit = assumptions.biomass_upgrade_capex_cny_per_mw_per_level * 1000.0  # toy 电厂 1 000 MW
    df = _discount_factor(2060, scenario.discount_base_year, scenario.discount_rate)
    assert y60["cost_breakdown_cny"]["blend_upgrade_capex"] == pytest.approx(unit * level * df, rel=1e-6)
    booked = {name: value for name, value, _ in y60["salvage_ledger"]}
    assert booked["blend_upgrade_capex"] == pytest.approx(unit * level, rel=1e-6)
    kind, weight = y60["cost_weights"]["incremental_om"]
    assert kind == "annual"
    assert y60["cost_breakdown_cny"]["incremental_om"] == pytest.approx(
        unit * assumptions.biomass_upgrade_om_fraction * level * weight, rel=1e-6
    )


def _vintage_payloads(model, years, air_need, layers_b, layers_a) -> list[SimpleNamespace]:
    """`model_linking.add_capacity_vintages` 要的年块，一个 hub：运行路径上的空冷份额与各档位层的在用掺烧能力逐年固定
    （层数取 `layers_b`、`layers_a` 每年的长度），新建量是变量，捕集岛与工业不用。"""
    payloads = []
    for year, value, layer_b, layer_a in zip(years, air_need, layers_b, layers_a, strict=True):
        air = np.zeros((1, len(PATHWAYS)))
        air[0, PATHWAY_INDEX["unabated"]] = value
        payloads.append(SimpleNamespace(
            year=year, interval_years=10, share=model.addMVar((1, len(PATHWAYS)), lb=0.0, ub=0.0),
            retrofit_new=model.addMVar((1, 1), lb=0.0), year_data=SimpleNamespace(retrofit_stock_om=np.zeros((1, 1))),
            air_share=model.addMVar(air.shape, lb=air, ub=air), air_new=model.addMVar(1, lb=0.0),
            blend_new_b=model.addMVar((1, len(layer_b)), lb=0.0), blend_new_a=model.addMVar((1, len(layer_a)), lb=0.0),
            blend_layers_b=[list(layer_b)], blend_layers_a=[list(layer_a)], industry=None,
        ))
    return payloads


def _without_industry(monkeypatch) -> None:
    from coal_retrofit.optimization import model_linking

    monkeypatch.setattr(
        model_linking, "add_industry_capacity",
        lambda model, years, payloads, salvage_end_year=None: ([None] * len(years), [None] * len(years)),
    )


def test_air_and_blend_capacity_retire_after_20_years(monkeypatch) -> None:
    """煤电空冷与掺烧能力按建设年分代（`model_linking.add_capacity_vintages`）。一个 hub，2030、2035、2040、2050 年运行路径
    上的空冷份额与两类在用掺烧能力（各取一个档位层）都固定为 1.0、0.4、1.0、1.0，寿命 20 年，越晚建越便宜（同折现）：
    2030 年建 1.0；2035 年降、2040 年升回都在寿命内，不再建；2050 年 2030 年建的满 20 年退出，重建 1.0。2026-10-02 前空冷
    存量只增不减、掺烧 capex 计在改造档位（受锁定只增不减）的增量上，都不到期，2050 年不付钱。期末 2060 年，只有 2050 年
    建的残值比例为正（0.5），只为它建期末在用量。"""
    from coal_retrofit.optimization import model_linking

    _without_industry(monkeypatch)
    model = gp.Model()
    model.Params.OutputFlag = 0
    years, need = (2030, 2035, 2040, 2050), (1.0, 0.4, 1.0, 1.0)
    payloads = _vintage_payloads(model, years, need, [[v] for v in need], [[v] for v in need])
    model_linking.add_capacity_vintages(model, payloads, OptimizationAssumptions())
    model.setObjective(gp.quicksum(
        (1.0 - 0.01 * (year - 2030)) * (pl.air_new.sum() + pl.blend_new_b.sum() + pl.blend_new_a.sum())
        for year, pl in zip(years, payloads)
    ))
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    built = np.array([[pl.air_new.X[0], pl.blend_new_b.X[0, 0], pl.blend_new_a.X[0, 0]] for pl in payloads])
    np.testing.assert_allclose(built, np.outer([1.0, 0.0, 0.0, 1.0], np.ones(3)), atol=1e-9)
    for stocks in (
        [pl.air_cooling for pl in payloads], [pl.biomass_blend[0] for pl in payloads],
        [pl.ammonia_blend[0] for pl in payloads],
    ):
        np.testing.assert_allclose([_expr_value(s.alive[0]) for s in stocks], [1.0, 1.0, 1.0, 1.0], atol=1e-9)
        assert [s.end_in_use is not None for s in stocks] == [False, False, False, True]


@pytest.mark.parametrize("continuous", [True, False])
def test_blend_layers_count_the_share_at_or_above_each_level(continuous: bool) -> None:
    """掺烧能力按档位分层（`constraints._add_blend_level_constraints`，2026-10-02 起）：第 l 层是落在第 l 档及以上的份额，
    各层相加即在用的掺烧能力 `blend_level`。生物质份额 0.2，连续 hub 与独热档位相同：全落在最高档时各层（L 层）都是 0.2、
    能力 0.2L；全落在最低档时只有第一层是 0.2、能力 0.2。hub 全是 CFB，炉型上限不起作用。"""
    from coal_retrofit.optimization.constraints import _add_blend_level_constraints

    year_data = cast(YearData, SimpleNamespace(
        emissions_retrofit_mt=np.ones(1), heat_rate_eff=np.full(1, 9.0),
        generation_by_pathway=np.ones((1, len(PATHWAYS))),
        biomass_penalty_coeff_per_level=0.0, biomass_penalty_emissions_coeff_per_level=0.0,
        beccs_penalty_emissions_coeff_per_level=0.0, beccs_penalty_captured_coeff_per_level=0.0,
        cfb_share=np.ones(1),
    ))
    model = gp.Model()
    model.Params.OutputFlag = 0
    fixed = np.zeros((1, len(PATHWAYS)))
    fixed[0, PATHWAY_INDEX["biomass"]] = 0.2
    share = model.addMVar(fixed.shape, lb=fixed, ub=fixed)
    blocks = _add_blend_level_constraints(
        model, share, 1, SCENARIO, OptimizationAssumptions(hub_decisions_continuous=continuous), year_data, "",
    )
    level_b, layers_b, x_share = blocks[2], blocks[4], blocks[-3]
    n_levels = len(SCENARIO.biomass_blend_levels)
    top, bottom = [0.2] * n_levels, [0.2] + [0.0] * (n_levels - 1)
    for sense, layers, level in ((gp.GRB.MAXIMIZE, top, 0.2 * n_levels), (gp.GRB.MINIMIZE, bottom, 0.2)):
        model.setObjective(x_share[0], sense)
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL
        np.testing.assert_allclose([_expr_value(e) for e in layers_b[0]], layers, atol=1e-9)
        assert float(level_b.X[0]) == pytest.approx(level, abs=1e-9)


def test_idle_low_level_blend_capacity_cannot_stand_in_for_higher_levels(monkeypatch) -> None:
    """掺烧能力按档位层分代（2026-10-02 起）。一个 hub，生物质 2030 年份额 1.0 全落在最低档（各层 1、0、0、0、0），2040 年
    份额 0.2 全落在最高档（五层都是 0.2）：两年在用的掺烧能力都是 1.0，但 2030 年建的最低档能力顶替不了高档，2040 年
    第 2–5 层各新建 0.2、第 1 层不建（越晚建越便宜，2030 年不预建）。各档合成一类分代时，2040 年在役的 1.0 就够，
    不必新建。"""
    from coal_retrofit.optimization import model_linking

    _without_industry(monkeypatch)
    model = gp.Model()
    model.Params.OutputFlag = 0
    years, none = (2030, 2040), [0.0] * 5
    payloads = _vintage_payloads(model, years, (0.0, 0.0), [[1.0, 0.0, 0.0, 0.0, 0.0], [0.2] * 5], [none, none])
    model_linking.add_capacity_vintages(model, payloads, OptimizationAssumptions())
    model.setObjective(gp.quicksum(
        (1.0 - 0.01 * (year - 2030)) * (pl.air_new.sum() + pl.blend_new_b.sum() + pl.blend_new_a.sum())
        for year, pl in zip(years, payloads)
    ))
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    np.testing.assert_allclose(payloads[0].blend_new_b.X[0], [1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(payloads[1].blend_new_b.X[0], [0.0, 0.2, 0.2, 0.2, 0.2], atol=1e-9)


@pytest.mark.parametrize("om", [False, True])
def test_salvage_counts_only_capacity_still_in_use_at_the_end(om: bool) -> None:
    """期末残值只计最后一个规划年仍在用的能力（`vintage._end_in_use`，2026-10-02 起；此前按建成量计）。2050、2060 年各新建
    1.0（固定），2060 年只用 0.4，寿命 20 年、期末 2070 年：2050 年建的残值比例为零，不建期末在用量；2060 年建的期末在用量
    最多 0.4。单列固定运维（煤电捕集岛、工业捕集）与不单列（空冷、掺烧、氢路线）两种写法相同。"""
    model = gp.Model()
    model.Params.OutputFlag = 0
    years = (2050, 2060)
    new = [model.addMVar(1, lb=1.0, ub=1.0) for _ in years]
    stocks = add_vintage_stock(
        model, "t", years, new, required=[[0.8], [0.4]], life=20,
        om_unit=[np.ones(1), np.ones(1)] if om else None, salvage_end_year=2070,
    )
    assert stocks[0].end_in_use is None
    end_in_use = stocks[1].end_in_use
    model.setObjective(end_in_use.sum(), gp.GRB.MAXIMIZE)
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    assert float(end_in_use.X[0]) == pytest.approx(0.4, abs=1e-9)
