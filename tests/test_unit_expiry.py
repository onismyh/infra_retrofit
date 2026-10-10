"""机组级到期（2026-10-02）：到期装机份额 f 与剩余账面份额按机组装机加权，到期约束按 f 加。

toy hub 缺省是两台机组：600 MW 2005 年投产（2045 年到期）、400 MW 2015 年投产（2055 年到期）；hub 的平均投产年 2010，
`retirement_year` 2050，hub 毛热耗取两台按装机加权。连续 hub（缺省）按机组计，2050 年 f = 0.6；整数 hub 整个 hub 在
2050 年到期，f = 1。有的用例换机组的投产年与装机（hub 装机仍是 1 000 MW）。退役核算（`retirement`）的用例也在这里。
用到 Gurobi 的用例在函数里 `importorskip`。
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.builders.plants import unit_heat_rate_gj_per_mwh
from coal_retrofit.constants import COAL_STATION_SERVICE_RATE, STANDARD_COAL_GJ_PER_KG
from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState, _discount_factor, _year_objective_weight
from coal_retrofit.optimization.data_prep import _with_expiry, prepare_inputs
from coal_retrofit.optimization.results_costs import build_costs_table, build_system_table, cost_closure
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.optimization.year_matrices import _build_year_matrices
from toy_inputs import TOY_UNIT_TYPE, YEARS, _toy_assumptions, _write_toy_inputs, _write_toy_units

GUROBI = "gurobipy is required for solver integration tests"
RETIRE = PATHWAY_INDEX["retire"]
UNABATED = PATHWAY_INDEX["unabated"]


def _unit_toy(root, capacity=(600.0, 400.0), commission=(2005, 2015), combustion=None, cooling=None):
    """toy 输入，电厂换成给定的几台机组（缺省为上面那两台，机型与冷却方式同 toy 机组），hub 毛热耗随之取机组的。"""
    paths = _write_toy_inputs(root, retirement_year=2050)
    n = len(capacity)
    _write_toy_units(paths, pd.DataFrame(
        {
            "plant_id": ["P1"] * n, "capacity_mw": list(capacity), "commission_year": list(commission),
            "combustion": list(combustion or TOY_UNIT_TYPE["combustion"] * n),
            "cooling_technology": list(cooling or TOY_UNIT_TYPE["cooling_technology"] * n),
        }
    ))
    return paths


def _two_type_toy(root, first_unit: str = "subcritical"):
    """缺省两台机组换成不同机型：600 MW `first_unit`（2045 年到期）、400 MW 超临界（300MW 级，2055 年到期）。
    返回 (paths, 两台的毛热耗, hub 毛热耗)。"""
    paths = _unit_toy(root, combustion=(first_unit, "supercritical"))
    units = pd.read_csv(paths.inputs_dir / "plants_unit_hub.csv")
    hub_heat_rate = float(pd.read_csv(paths.inputs_dir / "plants.csv")["heat_rate_gj_per_mwh"].iloc[0])
    return paths, unit_heat_rate_gj_per_mwh(units).to_numpy(), hub_heat_rate


@pytest.mark.parametrize(
    ("continuous", "commission", "expired", "book"),
    [
        # 2040 年账面份额：600 MW 剩 5 年、400 MW 剩 15 年，会计寿命 20 年：(600 x 0.25 + 400 x 0.75) / 1000。
        (True, (2005, 2015), [0.0, 0.6, 1.0], [0.45, 0.1, 0.0]),
        (False, (2005, 2015), [0.0, 1.0, 1.0], [0.5, 0.0, 0.0]),
        # 400 MW 2010 年投产，投产年 + 40 正好是规划年 2050：当年算到期（`<=`）。
        (True, (2005, 2010), [0.0, 1.0, 1.0], [0.35, 0.0, 0.0]),
    ],
)
def test_expired_share_and_book_value_by_unit(tmp_path, continuous: bool, commission, expired, book) -> None:
    paths = _unit_toy(tmp_path, commission=commission)
    scenario = OptimizationScenario(experiment_id="T", description="toy", planning_years=(2040, 2050, 2060))
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    out = _with_expiry(paths, scenario, OptimizationAssumptions(hub_decisions_continuous=continuous), plants)
    for year, f, value in zip(scenario.planning_years, expired, book, strict=True):
        assert out.loc[0, f"expired_share_{year}"] == pytest.approx(f, abs=1e-12)
        assert out.loc[0, f"remaining_life_fraction_{year}"] == pytest.approx(value, abs=1e-12)
    assert out.loc[0, "expired_share_2060"] == 1.0  # 全部到期时恰为 1，与整数 hub 加同一条约束


def test_unit_map_has_to_match_plants(tmp_path) -> None:
    """映射表与 plants.csv 不是同一次聚类的（hub 装机对不上、多出 hub）、有空值或非正装机、缺列都报错；没有映射表也报错。"""
    paths = _unit_toy(tmp_path)
    scenario = OptimizationScenario(experiment_id="T", description="toy", planning_years=YEARS)
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    path = paths.inputs_dir / "plants_unit_hub.csv"
    units = pd.read_csv(path)
    for bad in (
        units.assign(capacity_mw=[600.0, 300.0]),
        pd.concat([units, units.iloc[[0]].assign(plant_id="P2", capacity_mw=100.0)]),
        units.assign(capacity_mw=[1000.0, 0.0]),
        units.assign(commission_year=[2005.0, np.nan]),
    ):
        bad.to_csv(path, index=False)
        with pytest.raises(ValueError, match="does not match plants.csv"):
            _with_expiry(paths, scenario, OptimizationAssumptions(), plants)
    units.drop(columns="commission_year").to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks required columns"):
        _with_expiry(paths, scenario, OptimizationAssumptions(), plants)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="build_plant_inputs.py --hubs"):
        _with_expiry(paths, scenario, OptimizationAssumptions(), plants)


def test_partly_expired_hub_needs_the_heat_rate_of_its_units(tmp_path) -> None:
    """部分到期 hub 的未重建部分由 hub 毛热耗反推，hub 值与机组表按装机加权的值差 1e-4 以上时报错：toy 的 600 / 400 MW
    亚临界为 8.742 / 8.993，2050 年 f = 0.6，hub 写成 8.5714 时会反推出 8.315。plants.csv 取四位小数不算不一致；
    整数 hub 不会部分到期，不查。"""
    paths = _unit_toy(tmp_path)
    scenario = OptimizationScenario(experiment_id="T", description="toy", planning_years=YEARS)
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    _with_expiry(paths, scenario, OptimizationAssumptions(), plants.round({"heat_rate_gj_per_mwh": 4}))
    stale = plants.assign(heat_rate_gj_per_mwh=8.5714)
    with pytest.raises(ValueError, match="differs from the capacity-weighted unit heat rates"):
        _with_expiry(paths, scenario, OptimizationAssumptions(), stale)
    _with_expiry(paths, scenario, OptimizationAssumptions(hub_decisions_continuous=False), stale)


@pytest.mark.parametrize(("cooling", "adder_g_per_kwh"), [("air", 15.0), ("recirculating", 0.0)])
def test_rebuilt_unit_keeps_its_cooling(tmp_path, cooling: str, adder_g_per_kwh: float) -> None:
    """原址重建不换冷却方式：2050 年已到期的 600 MW 亚临界（湿冷 8.742，空冷 9.160）重建为 3.6 / 0.45 = 8.0，空冷的另加
    +15 g/kWh（`builders.plants.unit_rebuild_heat_rate_cap_gj_per_mwh`）。400 MW 那台（湿冷，2055 年到期）重建为 8.0：
    600 MW 那台空冷时两台按重建热耗分两类（升序，它在第 1 类），湿冷时同为一类；2060 年两台都到期，各类份额相加为 f。"""
    paths = _unit_toy(tmp_path, cooling=(cooling, "recirculating"))
    scenario = OptimizationScenario(experiment_id="T", description="toy", planning_years=YEARS)
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    out = _with_expiry(paths, scenario, OptimizationAssumptions(), plants)
    adder = adder_g_per_kwh * (1.0 - COAL_STATION_SERVICE_RATE) * STANDARD_COAL_GJ_PER_KG
    first = 1 if cooling == "air" else 0
    assert [c for c in out.columns if c.startswith("heat_rate_rebuilt_c")] == [f"heat_rate_rebuilt_c{j}" for j in range(first + 1)]
    assert out.loc[0, "expired_share_2050"] == pytest.approx(0.6, abs=1e-12)
    assert out.loc[0, f"expired_share_c{first}_2050"] == pytest.approx(0.6, abs=1e-12)
    assert out.loc[0, f"heat_rate_rebuilt_c{first}"] == pytest.approx(3.6 / scenario.rebuild_efficiency + adder, rel=1e-12)
    assert out.loc[0, "heat_rate_rebuilt_c0"] == pytest.approx(3.6 / scenario.rebuild_efficiency, rel=1e-12)
    assert sum(out.loc[0, f"expired_share_c{j}_2060"] for j in range(first + 1)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("first_unit", ["subcritical", "igcc"])
def test_partly_expired_hub_has_two_heat_rates_and_keeps_the_book_value(tmp_path, first_unit) -> None:
    """2050 年 f = 0.6：未重建部分的毛热耗就是未到期那台 400 MW 的（由 hub 值反推，两部分按 f 加权还原 hub 值），
    重建部分取 min(到期那台, 3.6 / 0.45 = 8.0)：亚临界 8.74 重建为 8.0，IGCC 7.52 本来就优于 8.0、重建不变差。
    系数之差按两部分各自的毛热耗算：排放与毛热耗成正比，生物质与空冷惩罚的燃料按各自的效率折算、与毛热耗的平方成正比，
    空冷背压的排放仍按 hub 毛热耗、没有差；本年没有到期装机的重建热耗类差为零。2060 年全部到期，未重建部分取逐台重建热耗的
    装机加权平均：亚临界与超临界都重建为 8.0，只有一类、差为零；IGCC 7.52 与超临界 8.0 分两类（升序），各类的差是该类与
    平均之差，按类份额加权为零。
    每单位提前退役的搁浅资产按未到期装机的剩余账面份额计：2050 年 ℓ / (1 - f) = 0.1 / 0.4 = 0.25，即那台 400 MW 机组
    自己的（剩 5 年 / 20 年）；2060 年全部到期，为零。2026-10-02 前全国一个热耗 8.5714，整个 hub 自 2050 年起按 8.0 计。"""
    paths, unit_heat_rate, hub_heat_rate = _two_type_toy(tmp_path, first_unit)
    scenario = OptimizationScenario(
        experiment_id="T", description="toy", planning_years=YEARS, sector_target_source="toy"
    )
    assumptions = OptimizationAssumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    data = {year: _build_year_matrices(prepared, scenario, assumptions, year, state) for year in YEARS}
    rebuilt = np.minimum(unit_heat_rate, 3.6 / scenario.rebuild_efficiency)
    unexpired_hr, rebuilt_hr = unit_heat_rate[1], rebuilt[0]
    y50 = data[2050]
    (expired_class,) = np.flatnonzero(y50.rebuilt_class_share[0])
    assert y50.rebuilt_class_share[0, expired_class] == pytest.approx(0.6, abs=1e-12)
    delta = y50.rebuilt_deltas[expired_class]
    for c, other in enumerate(y50.rebuilt_deltas):
        for field in fields(other) if c != expired_class else ():
            assert not np.any(getattr(other, field.name)), (c, field.name)
    assert y50.expired_share[0] == pytest.approx(0.6, abs=1e-12)
    assert y50.heat_rate_eff[0] == pytest.approx(unexpired_hr, rel=1e-12)
    assert delta.heat_rate_eff[0] == pytest.approx(rebuilt_hr - unexpired_hr, rel=1e-9)
    assert delta.emissions_operating_mt[0] == pytest.approx(
        y50.emissions_mt[0] * (rebuilt_hr - unexpired_hr) / hub_heat_rate, rel=1e-9
    )
    price = assumptions.province_coal_cost("Shanxi")
    assert delta.biomass_penalty_coeff_per_level[0] == pytest.approx(
        assumptions.biomass_efficiency_penalty_per_ratio * (rebuilt_hr**2 - unexpired_hr**2) / 3.6 * price, rel=1e-9
    )
    assert delta.air_penalty_cost_matrix[0, UNABATED] == pytest.approx(
        y50.generation[0] * assumptions.air_retrofit_efficiency_penalty_pp
        * (rebuilt_hr**2 - unexpired_hr**2) / 3.6 * price,
        rel=1e-9,
    )
    assert y50.air_penalty_emissions_matrix[0, UNABATED] == pytest.approx(
        y50.generation[0] * assumptions.air_retrofit_efficiency_penalty_pp * hub_heat_rate**2 / 3.6
        * assumptions.coal_emission_factor_t_per_gj / 1e6, rel=1e-9,
    )
    y60 = data[2060]
    average = (600.0 * rebuilt[0] + 400.0 * rebuilt[1]) / 1000.0
    assert y60.expired_share[0] == 1.0
    assert y60.heat_rate_eff[0] == pytest.approx(average, rel=1e-12)
    if first_unit == "subcritical":
        np.testing.assert_allclose(y60.rebuilt_class_share[0], [1.0], atol=1e-12)
        for field in fields(y60.rebuilt_deltas[0]):
            assert not np.any(getattr(y60.rebuilt_deltas[0], field.name)), field.name
    else:
        np.testing.assert_allclose(y60.rebuilt_class_share[0], [0.6, 0.4], atol=1e-12)
        diffs = np.array([float(d.heat_rate_eff[0]) for d in y60.rebuilt_deltas])
        np.testing.assert_allclose(diffs, [rebuilt[0] - average, rebuilt[1] - average], rtol=1e-9)
        assert float(y60.rebuilt_class_share[0] @ diffs) == pytest.approx(0.0, abs=1e-12)
    assert data[2050].stranded_per_plant[0] == pytest.approx(1000.0 * 3500.0 * 1000.0 * 0.25, rel=1e-12)
    assert data[2060].stranded_per_plant[0] == 0.0


def test_expiry_rules_follow_the_expired_share() -> None:
    """f = 0：不能重建；f = 1：退役 >= 1 - 重建（与 2026-10-02 前同一条）；0 < f < 1：退役 >= f - 重建、重建 <= f。"""
    gp = pytest.importorskip("gurobipy", reason=GUROBI)
    from coal_retrofit.optimization.model_year import _add_expiry_rules

    expired = np.array([0.0, 1.0, 0.6])
    for objective, fixed_rebuild, expected in (("max_rebuild", None, expired), ("min_retire", 0.0, expired)):
        model = gp.Model()
        model.Params.OutputFlag = 0
        share = model.addMVar((3, len(PATHWAYS)), lb=0.0, ub=1.0)
        rebuild = model.addMVar(3, lb=0.0, ub=1.0 if fixed_rebuild is None else fixed_rebuild)
        _add_expiry_rules(model, share, rebuild, expired, 3, "t")
        if objective == "max_rebuild":
            model.setObjective(rebuild.sum(), gp.GRB.MAXIMIZE)
        else:
            model.setObjective(share[:, RETIRE].sum(), gp.GRB.MINIMIZE)
        model.optimize()
        value = rebuild.X if objective == "max_rebuild" else share.X[:, RETIRE]
        np.testing.assert_allclose(value, expected, atol=1e-9, err_msg=objective)
        names = [c.ConstrName for c in model.getConstrs()]
        assert [n for n in names if n.startswith("rebuild_le_expired")] == ["rebuild_le_expired_2_t"]


def test_rebuilt_split_bounds() -> None:
    """拆分只加在部分到期的 hub 与全部到期、重建热耗有两类以上的 hub 上（f = 0 与只有一类的 f = 1 不加）。
    hub 2：f = 0.6（一类），份额固定为未改造 0.5、CCS 0.3、退役 0.2，重建 0.5，CCS 的空冷份额 0.2：重建部分 Σr 在
    [f - 退役, 重建] = [0.4, 0.5] 之间；CCS 全由重建机组承担（r = 0.3）时空冷全在重建部分（ra = 0.2），全由未重建机组
    承担（r = 0）时全不在。hub 3：f = 1，两类的到期份额 0.7、0.3，份额同 hub 2，重建 0.8：各路径全由重建部分承担
    （Σ_c r_c = 份额），第 0 类承担的在 [0.8 - 0.3, 0.7] = [0.5, 0.7] 之间。不开空冷时不拆空冷份额。"""
    gp = pytest.importorskip("gurobipy", reason=GUROBI)
    from coal_retrofit.optimization.constraints import _add_rebuilt_split

    ccs = PATHWAY_INDEX["ccs"]
    fixed = np.zeros((4, len(PATHWAYS)))
    fixed[2:, UNABATED], fixed[2:, ccs], fixed[2:, RETIRE] = 0.5, 0.3, 0.2
    air = np.zeros_like(fixed)
    air[2, ccs] = 0.2
    expired = np.array([0.0, 1.0, 0.6, 1.0])
    class_share = np.array([[0.0, 0.0], [1.0, 0.0], [0.6, 0.0], [0.7, 0.3]])

    def split(allow_air: bool):
        model = gp.Model()
        model.Params.OutputFlag = 0
        share = model.addMVar(fixed.shape, lb=fixed, ub=fixed)
        air_share = model.addMVar(air.shape, lb=air, ub=air)
        rebuild_class = model.addMVar(class_share.shape, lb=0.0, ub=class_share)
        model.addConstr(rebuild_class[2, :].sum() == 0.5)
        model.addConstr(rebuild_class[3, :].sum() == 0.8)
        r, ra = _add_rebuilt_split(model, share, rebuild_class, air_share, expired, class_share, 4, allow_air, "_t")
        return model, r, ra

    for allow_air in (True, False):
        model, r, ra = split(allow_air)
        assert {p: set(parts) for p, parts in r.items()} == {2: {0}, 3: {0, 1}}
        assert set(ra) == ({2, 3} if allow_air else set())
        for part, bounds in ((r[2][0], (0.4, 0.5)), (r[3][0], (0.5, 0.7))):
            total = gp.quicksum(part.values())
            for sense, expected in zip((gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE), bounds, strict=True):
                model.setObjective(total, sense)
                model.optimize()
                assert total.getValue() == pytest.approx(expected, abs=1e-9)
                for k in (UNABATED, ccs):
                    assert r[3][0][k].X + r[3][1][k].X == pytest.approx(fixed[3, k], abs=1e-9)
    for rebuilt_ccs, rebuilt_air in ((0.3, 0.2), (0.0, 0.0)):
        model, r, ra = split(True)
        r[2][0][ccs].LB = r[2][0][ccs].UB = rebuilt_ccs
        for sense in (gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE):
            model.setObjective(ra[2][0][ccs], sense)
            model.optimize()
            assert ra[2][0][ccs].X == pytest.approx(rebuilt_air, abs=1e-9)


@pytest.mark.parametrize(
    ("electricity", "retire_cost", "retired", "rebuilt"),
    [((490.0, 550.0), None, 0.0, 0.6), ((0.0, 0.0), 0.0, 0.75, 0.0)],
)
def test_running_capacity_burns_at_its_own_heat_rate(tmp_path, electricity, retire_cost, retired, rebuilt) -> None:
    """2050 年 f = 0.6：600 MW 亚临界到期，400 MW 超临界未到期，只开放未改造与退役。运行有利可图时到期装机全部原址重建、
    不退：未改造份额 1 里重建部分 0.6 按重建热耗 8.0 燃烧，其余 0.4 按未到期那台的毛热耗；运行就亏（电价为零、退役不计
    替代电量的成本）时退到 f + 0.15 = 0.75、不重建，在运行的 0.25 全是未到期那台。逐厂减排与逐厂成本（`costs.csv`）的
    基线净运行成本都按这两部分算（排放按毛热耗 / hub 毛热耗缩放基线排放）。2026-10-02 前全国一个热耗 8.5714，整个 hub
    自 2050 年起按 8.0 燃烧，不论到期装机重建还是退役。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    from coal_retrofit.optimization.results_costs import build_costs_table

    paths, unit_heat_rate, hub_heat_rate = _two_type_toy(tmp_path)
    scenario = OptimizationScenario(
        experiment_id="TEST-TWO-PARTS", description="toy", planning_years=YEARS, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=electricity,
        pathway_disable=("ccs", "biomass", "beccs", "ammonia"), solver_time_limit=300,
    )
    assumptions = _toy_assumptions()
    if retire_cost is not None:
        assumptions = replace(assumptions, retire_cost_cny_per_mwh=retire_cost)
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, YEARS, state)
    assert solution["status"] == "optimal"
    ys = solution["year_solutions"][2050]
    year_data = ys["year_data"]
    assert year_data.expired_share[0] == pytest.approx(0.6, abs=1e-12)
    assert ys["rebuild"][0] == pytest.approx(rebuilt, abs=1e-6)
    assert ys["share"][0, RETIRE] == pytest.approx(retired, abs=1e-6)
    assert ys["rebuilt_share"][:, 0, UNABATED].sum() == pytest.approx(rebuilt, abs=1e-6)
    running_heat = rebuilt * 3.6 / scenario.rebuild_efficiency + (1.0 - retired - rebuilt) * unit_heat_rate[1]
    baseline = float(year_data.emissions_mt[0])
    assert ys["plant_reduction_mt"][0] == pytest.approx(baseline * (1.0 - running_heat / hub_heat_rate), rel=1e-6)
    costs = build_costs_table(prepared, solution, scenario, assumptions)
    delta_rows = costs[(costs["year"] == 2050) & (costs["entity_type"] == "coal")
                       & (costs["category"] == "coal_operating_delta")]  # 表只写非零行，缺行按 0
    margin = assumptions.baseline_om_cost_cny_per_mwh - scenario.electricity_price_for_year(2050)
    expected = float(year_data.generation[0]) * (
        running_heat * assumptions.province_coal_cost("Shanxi") + (1.0 - retired) * margin
    )
    # 表与目标里都是相对参照（全部维持不改造运行）的差，加回参照即基线净运行成本（增量口径，2026-10-10 起）。
    reference = float(year_data.baseline_reference_cny[0])
    assert float(delta_rows["cost_cny"].sum()) + reference == pytest.approx(expected, rel=1e-6)
    # 目标函数里的同一项（除去折现与年金系数）。
    weight = _discount_factor(2050, scenario.discount_base_year, scenario.discount_rate) * _year_objective_weight(
        scenario.interval_years(YEARS, 0, assumptions), scenario.discount_rate
    )
    assert ys["cost_breakdown_cny"]["coal_operating_delta"] / weight + reference == pytest.approx(expected, rel=1e-6)


def test_rebuild_picks_the_efficient_heat_rate_class(tmp_path) -> None:
    """全部到期、重建热耗分两类的 hub：600 MW IGCC（重建不变差，7.52）与 400 MW 亚临界（重建为 8.0）都在 2045 年到期。
    只开放未改造与退役，电价取两类燃料成本的中间，重建几乎不要钱，水与退役替代电量都不计价：只重建 IGCC 那一类并运行，
    亚临界那一类到期退役（全部到期，不计搁浅资产、不占自愿退役上限），排放按 IGCC 的重建热耗计。重建部分只有一类、
    取两类的装机加权平均 7.71 时（2026-10-02 按机组细化起到按类分开之前），这个电价下两类同进退、全部重建运行。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    paths = _unit_toy(tmp_path, commission=(2005, 2005), combustion=("igcc", "subcritical"))
    unit_heat_rate = unit_heat_rate_gj_per_mwh(pd.read_csv(paths.inputs_dir / "plants_unit_hub.csv")).to_numpy()
    assumptions = replace(
        _toy_assumptions(), retire_cost_cny_per_mwh=0.0, water_extraction_cost_cny_per_m3=0.0,
        water_transport_cost_cny_per_m3_km=0.0,
    )
    scenario = OptimizationScenario(
        experiment_id="TEST-REBUILD-CLASS", description="toy", planning_years=YEARS, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), pathway_disable=("ccs", "biomass", "beccs", "ammonia"),
        rebuild_capex_fraction=1e-3, solver_time_limit=300, mip_gap=1e-9,
    )
    rebuilt = np.minimum(unit_heat_rate, 3.6 / scenario.rebuild_efficiency)
    assert rebuilt[0] < rebuilt[1] - 0.4, "前提：IGCC 那一类的重建热耗低"
    electricity = 0.5 * (rebuilt[0] + rebuilt[1]) * assumptions.province_coal_cost("Shanxi") + (
        assumptions.baseline_om_cost_cny_per_mwh
    )
    scenario = replace(scenario, electricity_price_cny_per_mwh_by_year=(electricity, electricity))
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, YEARS, state)
    assert solution["status"] == "optimal"
    hub_heat_rate = float(prepared.plants["heat_rate_gj_per_mwh"].iloc[0])
    for year in YEARS:
        ys = solution["year_solutions"][year]
        np.testing.assert_allclose(ys["year_data"].rebuilt_class_share[0], [0.6, 0.4], atol=1e-12)
        np.testing.assert_allclose(ys["rebuild_class"][0], [0.6, 0.0], atol=1e-6)
        assert ys["share"][0, RETIRE] == pytest.approx(0.4, abs=1e-6)
        np.testing.assert_allclose(ys["rebuilt_share"][:, 0, UNABATED], [0.6, 0.0], atol=1e-6)
        baseline = float(ys["year_data"].emissions_mt[0])
        assert ys["plant_reduction_mt"][0] == pytest.approx(baseline * (1.0 - 0.6 * rebuilt[0] / hub_heat_rate), rel=1e-6)
        assert ys["stranded_by_plant"][0] == pytest.approx(0.0, abs=1e-3)


def test_partly_expired_hub_rebuilds_only_its_expired_share(tmp_path) -> None:
    """三台机组 500 / 300 / 200 MW，2005 / 2015 / 2022 年投产：f 在 2050 年为 0.5、2060 年为 0.8（非首年仍部分到期）。
    运行有利可图时到期装机都重建、未到期的不退，重建 capex 分两期按份额增量计。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    paths = _unit_toy(tmp_path, capacity=(500.0, 300.0, 200.0), commission=(2005, 2015, 2022))
    scenario = OptimizationScenario(
        experiment_id="TEST-UNIT-EXPIRY", description="toy", planning_years=YEARS, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        solver_time_limit=300,
    )
    assumptions = _toy_assumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, YEARS, state)
    assert solution["status"] == "optimal"
    y1, y2 = solution["year_solutions"][2050], solution["year_solutions"][2060]
    assert y1["rebuild"][0] == pytest.approx(0.5, abs=1e-6)
    assert y2["rebuild"][0] == pytest.approx(0.8, abs=1e-6)
    assert y1["share"][0, RETIRE] == pytest.approx(0.0, abs=1e-6)
    assert y2["share"][0, RETIRE] == pytest.approx(0.0, abs=1e-6)
    full = 1000.0 * 3500.0 * scenario.rebuild_capex_fraction * 1000.0
    assert y1["cost_breakdown_cny"]["rebuild_capex"] == pytest.approx(0.5 * full / 1.06 ** (2050 - 2025), rel=1e-3)
    assert y2["cost_breakdown_cny"]["rebuild_capex"] == pytest.approx(0.3 * full / 1.06 ** (2060 - 2025), rel=1e-3)


def _retirement_payloads(model, hubs):
    """两年（2050、2060）的 payload，份额与重建都固定。`hubs` 每项是 (f, 退役份额, 重建份额, 重建部分在运行的份额)，
    每个都是 (首年, 次年)；重建部分在运行的份额放在未改造路径上，只给部分到期的 hub。各 hub 发电量相同。"""
    from types import SimpleNamespace

    payloads = []
    for t, year in enumerate((2050, 2060)):
        expired = np.array([hub[0][t] for hub in hubs])
        fixed = np.zeros((len(hubs), len(PATHWAYS)))
        fixed[:, RETIRE] = [hub[1][t] for hub in hubs]
        rebuild = np.array([hub[2][t] for hub in hubs])
        rebuilt_share = {
            p: {0: {UNABATED: model.addVar(lb=hub[3][t], ub=hub[3][t])}}
            for p, hub in enumerate(hubs) if 0.0 < expired[p] < 1.0
        }
        payloads.append(SimpleNamespace(
            year=year, share=model.addMVar(fixed.shape, lb=fixed, ub=fixed),
            rebuild=model.addMVar(len(hubs), lb=rebuild, ub=rebuild), rebuilt_share=rebuilt_share,
            year_data=SimpleNamespace(expired_share=expired, generation=np.ones(len(hubs))),
        ))
    return payloads


def _values(exprs) -> list[float]:
    return [expr if isinstance(expr, float) else float((1.0 * expr).getValue()) for expr in exprs]


def test_new_retirement_counts_early_closures_and_retired_rebuilds() -> None:
    """新增提前退役 n^o 与重建后退役 n^r（`retirement.retirement_flows`）。两年、八个 hub：
    A f = 0.6、0.6，全退、不重建：n^o 首年 0.4（到期的 0.6 不算），次年 0；
    B f = 0.6、0.6，首年重建 0.6 全运行、次年全退：次年 n^o = 0.4，退掉的重建装机另记 n^r = 0.6（不计搁浅资产）；
    C f = 0、0.5，首年退 0.3、次年退到 0.65：一半装机到期，首年提前关停的按比例留下 0.15 未到期，次年 n^o = 0；
    D 同 C、次年全退：n^o = 0.5 - 0.15 = 0.35；
    E 整数 hub，f = 0、1，首年退 0.2、次年重建（只能整个 hub）后退到 0.5：到期当期退掉的是正常寿终，n^o = n^r = 0；
    F 整数 hub，f = 1、1，重建后从全运行退到 0.4：n^r = 0.4；
    G f = 0.5、1，退役都是 0.3，重建 0.2 全运行、次年重建到 0.7：在运行的重建装机增加，n^r = 0（不是 -0.5）；
    H f = 0.5、1，首年重建 0.5、只运行 0.2，次年退到 0.9：重建后关停的从 0.3 增到 0.4，n^r = 0.1。
    C 次年只退到 0.6 时不可行：首年提前关停、次年仍未到期的 0.15 要重启；H 次年只退到 0.5 时也不可行：首年重建后关停的
    0.3 要重启。"""
    gp = pytest.importorskip("gurobipy", reason=GUROBI)
    from coal_retrofit.optimization.retirement import retirement_flows

    hubs = [
        ((0.6, 0.6), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        ((0.6, 0.6), (0.0, 1.0), (0.6, 0.6), (0.6, 0.0)),
        ((0.0, 0.5), (0.3, 0.65), (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.5), (0.3, 1.0), (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 1.0), (0.2, 0.5), (0.0, 1.0), (0.0, 0.0)),
        ((1.0, 1.0), (0.0, 0.4), (1.0, 1.0), (0.0, 0.0)),
        ((0.5, 1.0), (0.3, 0.3), (0.2, 0.7), (0.2, 0.0)),
        ((0.5, 1.0), (0.3, 0.9), (0.5, 0.5), (0.2, 0.0)),
    ]
    model = gp.Model()
    model.Params.OutputFlag = 0
    first, second = _retirement_payloads(model, hubs)
    early_1, rebuilt_1 = retirement_flows(model, first, None, len(hubs))
    early_2, rebuilt_2 = retirement_flows(model, second, first, len(hubs))
    model.update()
    # 辅助变量只加在上期部分到期的 hub（A、B、G、H）上，目标压到下界 max(0, 在运行的重建装机的净减少)。
    names = [v.VarName for v in model.getVars() if v.VarName.startswith("rebuilt_retire_new")]
    assert names == [f"rebuilt_retire_new_{p}_2060" for p in (0, 1, 6, 7)]
    model.setObjective(gp.quicksum(rebuilt_2), gp.GRB.MINIMIZE)
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    np.testing.assert_allclose(
        [_values(early_1), _values(rebuilt_1), _values(early_2), _values(rebuilt_2)],
        [
            [0.4, 0.0, 0.3, 0.3, 0.2, 0.0, 0.0, 0.0],
            [0.0] * 8,
            [0.0, 0.4, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.0, 0.0, 0.4, 0.0, 0.1],
        ],
        atol=1e-9,
    )
    for hub in (((0.0, 0.5), (0.3, 0.6), (0.0, 0.0), (0.0, 0.0)), ((0.5, 1.0), (0.3, 0.5), (0.5, 0.5), (0.2, 0.0))):
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.DualReductions = 0  # 区分不可行与无界
        first, second = _retirement_payloads(model, [hub])
        retirement_flows(model, second, first, 1)
        model.optimize()
        assert model.Status == gp.GRB.INFEASIBLE, hub


@pytest.mark.parametrize(("retired", "feasible"), [(0.15, True), (0.2, False)])
def test_retired_rebuilds_count_toward_the_rate_limit(retired: float, feasible: bool) -> None:
    """重建后又关停的装机计入 15% 的自愿退役速率上限（`retirement.add_retirement_rate_limit`）：hub 两期都全部到期、
    全部重建，次年从全运行退到 1 - `retired`，n^r = `retired`。"""
    gp = pytest.importorskip("gurobipy", reason=GUROBI)
    from coal_retrofit.optimization.retirement import add_retirement_rate_limit, retirement_flows

    scenario = OptimizationScenario(experiment_id="T", description="toy", planning_years=YEARS)
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.DualReductions = 0
    first, second = _retirement_payloads(model, [((1.0, 1.0), (0.0, retired), (1.0, 1.0), (0.0, 0.0))])
    add_retirement_rate_limit(model, second, scenario, *retirement_flows(model, second, first, 1), 1)
    model.optimize()
    assert model.Status == (gp.GRB.OPTIMAL if feasible else gp.GRB.INFEASIBLE)


@pytest.mark.parametrize("last_expired", [0.8, 1.0])
def test_retired_rebuilds_get_no_salvage(last_expired: float) -> None:
    """期末重建后已关停的装机不计残值（`retirement.add_retired_rebuild_offset`），按先关最早建成的扣。规划年 2030-2060、
    期末 2070，重建寿命 30 年：2030、2040 年建的残值为零，2050 年的 1/3，2060 年的 2/3。一个 hub，f 依次 0.2、0.4、
    0.6、0.8（或末年全部到期），重建 0、0.2、0.5、0.8，没有提前退役，重建装机在运行的依次 0、0.2、0.4、0.2：期末已关停的
    0.6 里 0.2 算 2040 年建的（本就为零），0.3 算 2050 年的（当年新增全部），0.1 算 2060 年的，残值抵扣少
    df(2070) x 重建单价 x (0.3 / 3 + 0.1 x 2 / 3)。没有残值时不加变量。"""
    gp = pytest.importorskip("gurobipy", reason=GUROBI)
    from types import SimpleNamespace

    from coal_retrofit.optimization._shared import _COST_SCALE
    from coal_retrofit.optimization.retirement import add_retired_rebuild_offset
    from coal_retrofit.optimization.salvage import _add_salvage_credit

    years = (2030, 2040, 2050, 2060)
    scenario = OptimizationScenario(experiment_id="T", description="toy", planning_years=years)
    assumptions = OptimizationAssumptions()
    model = gp.Model()
    model.Params.OutputFlag = 0
    payloads = []
    expired = (0.2, 0.4, 0.6, last_expired)
    for year, f, rebuild, running in zip(years, expired, (0.0, 0.2, 0.5, 0.8), (0.0, 0.2, 0.4, 0.2), strict=True):
        fixed = np.zeros((1, len(PATHWAYS)))
        fixed[0, RETIRE] = f - running  # 提前退役为零：退役 = 到期未重建 + 重建后关停
        payloads.append(SimpleNamespace(
            year=year, interval_years=10, share=model.addMVar(fixed.shape, lb=fixed, ub=fixed),
            rebuild=model.addMVar(1, lb=rebuild, ub=rebuild),
            rebuilt_share={0: {0: {UNABATED: model.addVar(lb=running, ub=running)}}},
            year_data=SimpleNamespace(expired_share=np.array([f]), capacity_mw=np.array([1000.0])),
            salvage_ledger=[], cost_exprs={}, cost_weights={},
        ))
    model.update()
    count = model.NumVars
    add_retired_rebuild_offset(model, payloads, scenario, replace(assumptions, end_of_horizon_salvage=False), 1)
    model.update()
    assert model.NumVars == count and not any(pl.salvage_ledger for pl in payloads)
    add_retired_rebuild_offset(model, payloads, scenario, assumptions, 1)
    _add_salvage_credit(payloads, scenario, assumptions, _COST_SCALE)
    model.setObjective(gp.quicksum(pl.objective_expr for pl in payloads), gp.GRB.MINIMIZE)
    model.optimize()
    retired = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith("rebuild_retired_")}
    assert retired == pytest.approx({"rebuild_retired_0_2050": 0.3, "rebuild_retired_0_2060": 0.1}, abs=1e-9)
    unit_cost = 1000.0 * assumptions.stranded_asset_base_cny_per_kw * scenario.rebuild_capex_fraction * 1000.0
    df_end = _discount_factor(2070, scenario.discount_base_year, scenario.discount_rate)
    credit = payloads[-1].cost_exprs["salvage_credit"]
    lost = df_end * unit_cost * (0.3 / 3.0 + 0.1 * 2.0 / 3.0)
    assert float(credit.getValue()) * _COST_SCALE == pytest.approx(lost, rel=1e-9)


@pytest.mark.parametrize("salvage", [True, False])
def test_solver_wires_the_retirement_accounting(tmp_path, monkeypatch, salvage: bool) -> None:
    """求解器接上退役核算（toy hub 2050 年部分到期、2060 年全部到期）：2060 年的速率上限含重建后关停的辅助变量
    （`model_costs` 把 n^r 传给 `add_retirement_rate_limit`）；两年都能重建、重建残值都不为零，残值台账上各记一条
    `rebuild_retired`（`solver` 调 `add_retired_rebuild_offset`），关掉残值时不记。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    from coal_retrofit.optimization import model_costs
    from coal_retrofit.optimization import solver as solver_module

    models = []
    ledgers: dict[int, list[str]] = {}
    add_rate_limit, add_credit = model_costs.add_retirement_rate_limit, solver_module._add_salvage_credit

    def _rate_limit_spy(model, *args, **kwargs):
        models.append(model)
        return add_rate_limit(model, *args, **kwargs)

    def _credit_spy(year_payloads, *args, **kwargs):
        ledgers.update({int(pl.year): [item[0] for item in pl.salvage_ledger] for pl in year_payloads})
        return add_credit(year_payloads, *args, **kwargs)

    monkeypatch.setattr(model_costs, "add_retirement_rate_limit", _rate_limit_spy)
    monkeypatch.setattr(solver_module, "_add_salvage_credit", _credit_spy)
    paths = _unit_toy(tmp_path)
    scenario = OptimizationScenario(
        experiment_id="TEST-REBUILD-SALVAGE", description="toy", planning_years=YEARS, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        solver_time_limit=300,
    )
    assumptions = replace(_toy_assumptions(), end_of_horizon_salvage=salvage)
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    assert _solve_joint_multi_period(prepared, scenario, assumptions, YEARS, state)["status"] == "optimal"
    row = models[-1].getRow(models[-1].getConstrByName("max_retire_rate_2060"))
    assert {row.getVar(i).VarName: row.getCoeff(i) for i in range(row.size())} == {"rebuilt_retire_new_0_2060": 1.0}
    assert {year: names.count("rebuild_retired") for year, names in ledgers.items()} == {
        2050: int(salvage), 2060: int(salvage)
    }


def test_expired_capacity_retires_outside_the_rate_limit_and_without_stranded_cost(tmp_path) -> None:
    """三台机组 500 / 300 / 200 MW，2005 / 2015 / 2022 年投产，f 在 2050 年为 0.5、2060 年为 0.8；电价为零、退役不计
    替代电量的成本（缺省 450 元/MWh，比亏着运行还贵），运行就亏，只开放退役。到期装机退役不占 15% 的自愿退役上限、
    不计搁浅资产：2050 年退到 0.5 + 0.15，搁浅资产 = 新建成本 x 0.15 x 未到期装机的剩余账面份额
    (300 x 5 / 20 + 200 x 12 / 20) / 500 = 0.39。2060 年 300 MW 那台也到期：2050 年提前关停的 0.15 按未到期装机等比例摊，
    留到 2060 年仍未到期的是 0.15 x 0.2 / 0.5 = 0.06，剩下的 0.2 全退，新增提前退役 0.14，份额 200 x 2 / 20 / 200 = 0.1。
    2026-10-02 前到期按整个 hub 的 `retirement_year` 计，它之前退掉的装机（含已到寿命的机组）全算自愿、受 15% 所限、
    计搁浅资产。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    paths = _unit_toy(tmp_path, capacity=(500.0, 300.0, 200.0), commission=(2005, 2015, 2022))
    scenario = OptimizationScenario(
        experiment_id="TEST-VOLUNTARY-RETIRE", description="toy", planning_years=YEARS, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=(0.0, 0.0),
        pathway_disable=("ccs", "biomass", "beccs", "ammonia"), solver_time_limit=300,
    )
    assert scenario.max_new_retirement_share_per_period == 0.15
    assumptions = replace(_toy_assumptions(), retire_cost_cny_per_mwh=0.0)
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, YEARS, state)
    assert solution["status"] == "optimal"
    y1, y2 = solution["year_solutions"][2050], solution["year_solutions"][2060]
    assert y1["share"][0, RETIRE] == pytest.approx(0.65, abs=1e-6)
    assert y2["share"][0, RETIRE] == pytest.approx(1.0, abs=1e-6)
    assert y1["rebuild"][0] == pytest.approx(0.0, abs=1e-6)
    assert y2["rebuild"][0] == pytest.approx(0.0, abs=1e-6)
    new_build = 1000.0 * 3500.0 * 1000.0
    for ys, year, stranded in ((y1, 2050, new_build * 0.15 * 0.39), (y2, 2060, new_build * 0.14 * 0.1)):
        assert ys["stranded_by_plant"][0] == pytest.approx(stranded, rel=1e-6)
        assert ys["cost_breakdown_cny"]["stranded_capex"] == pytest.approx(stranded / 1.06 ** (year - 2025), rel=1e-6)
    # 搁浅资产拆到 hub 后与其余各类成本一起逐年对上求解器的合计（`results_costs.cost_closure`）。
    costs = build_costs_table(prepared, solution, scenario, assumptions)
    assert (costs["category"] == "stranded_capex").any()
    assert cost_closure(build_system_table(solution, costs)).max() < 1e-6


@pytest.mark.parametrize(("retirement_year", "retired", "book"), [(2055, 0.15, 0.25), (2050, 1.0, 0.0)])
def test_integer_hub_expires_as_a_whole(tmp_path, retirement_year, retired, book) -> None:
    """整数 hub 整个 hub 在 `retirement_year` 一起到期（f 只取 0 或 1）。运行就亏、只开放退役（同上一条）：2055 年到期的
    hub 2050 年只能自愿退 0.15，每单位搁浅资产按剩余账面份额 5 / 20 = 0.25 计，2060 年到期后全退、不计；2050 年到期的
    hub 2050 年就全退，不受 15% 所限、不计搁浅资产。不重建时同 2026-10-02 前；到期重建后又退的装机此后计入 15% 上限
    （`test_retired_rebuilds_count_toward_the_rate_limit`），此前到期后不计。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    paths = _write_toy_inputs(tmp_path, retirement_year=retirement_year)
    scenario = OptimizationScenario(
        experiment_id="TEST-INTEGER-RETIRE", description="toy", planning_years=YEARS, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=(0.0, 0.0),
        pathway_disable=("ccs", "biomass", "beccs", "ammonia"), solver_time_limit=300,
    )
    assumptions = replace(_toy_assumptions(), retire_cost_cny_per_mwh=0.0, hub_decisions_continuous=False)
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, YEARS, state)
    assert solution["status"] == "optimal"
    y1, y2 = solution["year_solutions"][2050], solution["year_solutions"][2060]
    assert y1["year_data"].expired_share[0] == float(retirement_year <= 2050)
    assert y1["share"][0, RETIRE] == pytest.approx(retired, abs=1e-6)
    assert y2["share"][0, RETIRE] == pytest.approx(1.0, abs=1e-6)
    stranded = 1000.0 * 3500.0 * 1000.0 * retired * book
    assert y1["stranded_by_plant"][0] == pytest.approx(stranded, rel=1e-6, abs=1e-3)
    assert y1["cost_breakdown_cny"]["stranded_capex"] == pytest.approx(stranded / 1.06 ** (2050 - 2025), rel=1e-6, abs=1e-3)
    assert y2["stranded_by_plant"][0] == pytest.approx(0.0, abs=1e-3)
