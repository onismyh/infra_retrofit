"""2026-09-22 的成本口径：显式 capex + 固定运维 + 能耗，并在期末计残值。

分两组。第一组是闭式检查：工业成本辅助函数必须复现它们据以分解的文献锚点，隐含的捕集
成本落在 ACCA21 交叉核对区间附近（长流程钢低于下限，允许至多低 10%）。第二组求解煤电 toy 模型，
检查求解器记入的残值抵扣等于它所计每笔 capex 按直线法的剩余部分，并从期末折现；其中一条
收紧 toy 里水泥 hub 的目标，核对工业两项 capex 也进了残值台账。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit import constants_industry as ci
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.industry import CCS, H2, UNABATED, IndustryInputs, industry_year_data
from coal_retrofit.optimization.salvage import horizon_end_year, remaining_fraction
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization._shared import SolveState
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from test_multiperiod_investment_logic import _toy_assumptions, _write_targets, _write_toy_inputs


# ---------------------------------------------------------------------------- 闭式检查 ---
@pytest.mark.parametrize("sector", sorted(ci.INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT))
def test_h2_premium_reproduces_its_anchor_and_floors_at_capital(sector: str) -> None:
    premium_ref, price_ref = ci.INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT[sector]
    k = {"steel_bf_bof": 0.081, "ammonia": 0.18, "methanol": 0.19}[sector]
    # 在锚点自身的氢价下，分解是精确的。
    assert ci.h2_premium_cny_per_t(sector, price_ref, k, 0.06) == pytest.approx(premium_ref, rel=1e-9)
    # 氢免费也不能把溢价压到 capex 年金以下（求解器的下限把固定运维放在年度项之内，
    # 把年金放在其外）。
    floor = ci.h2_route_capex_cny_per_t_yr(sector) * ci.capital_recovery_factor(0.06, ci.INDUSTRY_H2_LIFETIME_YEARS)
    assert ci.h2_premium_cny_per_t(sector, 0.0, k, 0.06) == pytest.approx(floor, rel=1e-9)
    assert floor > 0.0
    # 对氢价单调。
    assert ci.h2_premium_cny_per_t(sector, 30.0, k, 0.06) >= ci.h2_premium_cny_per_t(sector, 10.0, k, 0.06)
    # 成本乘数只乘路线 capex（年金与固定运维随之）：在锚点价格下溢价多出 0.3 x capex x (CRF + 固定运维比例)，
    # 且在任何价格下乘数越高都不会更便宜（这个旋钮不得反向）。
    capital = ci.h2_route_annual_capital_cny_per_t(sector, 0.06)
    assert ci.h2_premium_cny_per_t(sector, price_ref, k, 0.06, 1.3) == pytest.approx(premium_ref + 0.3 * capital, rel=1e-9)
    for price in (0.0, 8.0, 12.4, 20.0, 35.0):
        assert ci.h2_premium_cny_per_t(sector, price, k, 0.06, 1.2) >= ci.h2_premium_cny_per_t(sector, price, k, 0.06)


@pytest.mark.parametrize("sector", sorted(ci.INDUSTRY_SECTORS))
def test_levelised_capture_cost_sits_near_the_acca21_range(sector: str) -> None:
    lo, hi = ci.INDUSTRY_CAPTURE_COST_REFERENCE_CNY_PER_T[sector]
    implied = ci.levelised_capture_cost_cny_per_t(sector, 0.06, 38.2, 400.0)
    # 隐含成本多在 ACCA21 区间低端，长流程钢低于下限 348，所以允许低于区间下限 10%。
    assert 0.9 * lo <= implied <= hi, (sector, implied, lo, hi)


def test_capital_recovery_factor_is_reciprocal_of_annuity() -> None:
    crf = ci.capital_recovery_factor(0.06, 20)
    annuity = sum(1.0 / 1.06**t for t in range(1, 21))
    assert crf * annuity == pytest.approx(1.0, rel=1e-12)
    assert ci.capital_recovery_factor(0.0, 20) == pytest.approx(0.05)


def _toy_industry() -> IndustryInputs:
    hubs = pd.DataFrame(
        {
            "hub_id": ["S1", "C1", "A1"],
            "sector": ["steel_bf_bof", "cement", "ammonia"],
            "province": ["Shanxi", "Shandong", "Unknown"],
            "production_kt_per_year": [1000.0, 2000.0, 500.0],
            "co2_mt_per_year": [2.0, 1.2, 1.0],
            "h2_demand_kt_per_year": [81.0, 0.0, 90.0],
            "water_m3_per_year": [3.0e6, 1.0e6, 5.0e6],
            "basin_code": ["B1", "B1", "B1"],
        }
    )
    hubs["target_group"] = hubs["sector"].map(ci.SECTOR_TARGET_GROUP)
    return IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0, 2060: 8.0})


def test_industry_year_data_prices_capex_om_energy_explicitly() -> None:
    scenario = OptimizationScenario(experiment_id="T", description="toy")
    assumptions = OptimizationAssumptions()
    data = industry_year_data(_toy_industry(), scenario, assumptions, 2030)
    learning = assumptions.ccs_learning_factor(2030)
    ef_gj = assumptions.coal_emission_factor_t_per_mwh / assumptions.heat_rate_gj_per_mwh
    elec = scenario.electricity_price_for_year(2030)

    # 钢铁 hub：份额为 1 时 capex = 捕集能力 t/a x 单位 capex x 学习系数；opex = 固定运维 + 可变成本。
    captured_t = 2.0e6 * scenario.capture_rate
    capex_unit = ci.capture_capex_cny_per_t_yr("steel_bf_bof") * learning
    var_unit = ci.capture_variable_cost_cny_per_t("steel_bf_bof", assumptions.province_coal_cost("Shanxi"), elec)
    assert data.capacity_mt_per_share[0, CCS] == pytest.approx(captured_t / 1e6, rel=1e-12)
    assert data.capex_cny_per_mt[0, CCS] * data.capacity_mt_per_share[0, CCS] == pytest.approx(
        captured_t * capex_unit, rel=1e-9
    )
    assert data.opex_cny[0, CCS] == pytest.approx(
        captured_t * (capex_unit * ci.INDUSTRY_CCS_FIXED_OM_FRACTION + var_unit), rel=1e-9
    )
    # 再沸器（reboiler）蒸汽的 CO2 直接排放：胺法捕集的减排量 < 捕集量，合成氨两者相等。
    steam = ci.capture_steam_co2_t_per_t("steel_bf_bof", ef_gj)
    assert 0.2 < steam < 0.4
    assert data.reduction_mt[0, CCS] == pytest.approx(data.captured_mt[0, CCS] * (1.0 - steam), rel=1e-9)
    assert data.reduction_mt[2, CCS] == pytest.approx(data.captured_mt[2, CCS], rel=1e-9)
    # 未知省份不崩溃。合成氨捕集不用蒸汽，煤价不进这一项，所以这里测不出回退到哪个煤价。
    var_nat = ci.capture_variable_cost_cny_per_t("ammonia", assumptions.coal_fuel_cost_cny_per_gj, elec)
    capex_nat = ci.capture_capex_cny_per_t_yr("ammonia") * learning
    assert data.opex_cny[2, CCS] == pytest.approx(
        1.0e6 * scenario.capture_rate * (capex_nat * ci.INDUSTRY_CCS_FIXED_OM_FRACTION + var_nat), rel=1e-9
    )

    # H2 路线：capex 按全部产量计；年度项 = 固定运维 + opex 差额；氢逐链路购买。
    assert data.route_available[0, H2] and data.route_available[2, H2]
    assert not data.route_available[1, H2]
    production = 1000.0e3
    route_capex = ci.h2_route_capex_cny_per_t_yr("steel_bf_bof")
    delta = ci.h2_route_opex_delta_cny_per_t("steel_bf_bof", 0.081, scenario.discount_rate)
    assert data.capacity_mt_per_share[0, H2] == pytest.approx(production / 1e6, rel=1e-12)
    assert data.capex_cny_per_mt[0, H2] * data.capacity_mt_per_share[0, H2] == pytest.approx(
        production * route_capex, rel=1e-9
    )
    assert data.opex_cny[0, H2] == pytest.approx(
        production * (route_capex * ci.INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION + delta), rel=1e-9
    )
    assert data.h2_demand_kg_per_share[0] == pytest.approx(81.0e6, rel=1e-9)
    assert data.capex_lifetime_years == {
        CCS: ci.INDUSTRY_CAPTURE_LIFETIME_YEARS,
        H2: ci.INDUSTRY_H2_LIFETIME_YEARS,
    }
    assert data.capex_cny_per_mt[:, UNABATED].sum() == 0.0
    assert data.capacity_mt_per_share[:, UNABATED].sum() == 0.0


# ---------------------------------------------------------------------- 求解器：残值 ---
def test_remaining_fraction_is_straight_line() -> None:
    assert remaining_fraction(2050, 20, 2060) == pytest.approx(0.5)
    assert remaining_fraction(2040, 20, 2060) == pytest.approx(0.0)
    assert remaining_fraction(2030, 20, 2060) == pytest.approx(0.0)
    assert remaining_fraction(2050, 30, 2060) == pytest.approx(2.0 / 3.0)
    assert remaining_fraction(2060, 30, 2060) == pytest.approx(1.0)


def test_horizon_end_year_uses_last_interval() -> None:
    assert horizon_end_year([SimpleNamespace(year=2050, interval_years=10)]) == 2060
    assert horizon_end_year([
        SimpleNamespace(year=2030, interval_years=30), SimpleNamespace(year=2060, interval_years=30),
    ]) == 2090


def _solve(paths, salvage: bool, power_caps=(1.0, 1.0, 0.5), cement_caps=(1.0, 1.0, 1.0)):
    years = (2030, 2040, 2050)
    _write_targets(
        paths, dict(zip(years, power_caps, strict=True)), dict(zip(years, cement_caps, strict=True))
    )
    scenario = OptimizationScenario(
        experiment_id="TEST-SALVAGE",
        description="toy",
        planning_years=(2030, 2040, 2050),
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0, 490.0),
        pathway_disable=("retire",),
        solver_time_limit=300,
    )
    base = _toy_assumptions()
    assumptions = OptimizationAssumptions(
        storage_deployment_fraction_by_year=base.storage_deployment_fraction_by_year,
        end_of_horizon_salvage=salvage,
    )
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.effective_years(list(prepared.available_ammonia_years))
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    return scenario, assumptions, _solve_joint_multi_period(prepared, scenario, assumptions, years, state)


def test_salvage_credit_equals_straight_line_remainder_of_booked_capex(tmp_path) -> None:
    """捕集只在 2050 年建设，规划期止于 2060 年：捕集岛的一半与管道的三分之二
    尚未折旧，残值抵扣恰为这部分，并从 2060 年折现。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    scenario, assumptions, solution = _solve(paths, salvage=True)
    assert solution["status"] == "optimal"
    ys = solution["year_solutions"]
    lives = {
        "ccs_retrofit_capex": assumptions.ccs_retrofit_lifetime_years,
        "pipe_capex": assumptions.pipeline_lifetime_years,
        "blend_upgrade_capex": assumptions.blend_upgrade_lifetime_years,
        "air_retrofit_capex": assumptions.air_retrofit_lifetime_years,
        "rebuild_capex": assumptions.rebuild_lifetime_years,
    }
    rate, base = scenario.discount_rate, scenario.discount_base_year
    end_year = 2060
    df_end = 1.0 / (1.0 + rate) ** (end_year - base)
    expected = 0.0
    for year in (2030, 2040, 2050):
        df_t = 1.0 / (1.0 + rate) ** (year - base)
        for key, life in lives.items():
            booked = float(ys[year]["cost_breakdown_cny"][key])  # 已按 df_t 折现
            expected -= remaining_fraction(year, life, end_year) * booked / df_t * df_end
    assert ys[2030]["cost_breakdown_cny"]["salvage_credit"] == 0.0
    assert ys[2040]["cost_breakdown_cny"]["salvage_credit"] == 0.0
    credit = float(ys[2050]["cost_breakdown_cny"]["salvage_credit"])
    assert credit < 0.0
    assert credit == pytest.approx(expected, rel=1e-6)
    # 2050 年的捕集岛是真实建设的（目标迫使其建设），其中一半以残值收回。
    capex_2050 = float(ys[2050]["cost_breakdown_cny"]["ccs_retrofit_capex"])
    df_2050 = 1.0 / (1.0 + rate) ** (2050 - base)
    assert capex_2050 > 0.0
    assert abs(credit) >= 0.5 * capex_2050 * df_end / df_2050 * (1 - 1e-6)


def test_salvage_ledger_carries_both_industry_capex_items(tmp_path, monkeypatch) -> None:
    """工业两项 capex 各带自己的寿命进残值台账（捕集岛 20 a、氢路线 25 a）。水泥目标只在 2050 年收紧，
    逼出 2050 年的工业捕集岛：残值抵扣里要有它未折旧的一半，否则末期的工业 capex 被全额计入。"""
    from coal_retrofit.optimization import solver as solver_module

    ledgers: list[list[tuple[str, object, int]]] = []
    original = solver_module._add_salvage_credit

    def _spy(year_payloads, *args, **kwargs):
        ledgers.extend(list(payload.salvage_ledger) for payload in year_payloads)
        return original(year_payloads, *args, **kwargs)

    monkeypatch.setattr(solver_module, "_add_salvage_credit", _spy)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    scenario, assumptions, solution = _solve(paths, salvage=True, cement_caps=(1.0, 1.0, 0.5))
    assert solution["status"] == "optimal"
    lives = {name: life for ledger in ledgers for name, _, life in ledger}
    assert lives["industry_ccs_capex"] == ci.INDUSTRY_CAPTURE_LIFETIME_YEARS
    assert lives["industry_h2_capex"] == ci.INDUSTRY_H2_LIFETIME_YEARS

    ys = solution["year_solutions"]
    rate, base, end_year = scenario.discount_rate, scenario.discount_base_year, 2060
    df_end = 1.0 / (1.0 + rate) ** (end_year - base)
    booked_lives = {
        "ccs_retrofit_capex": assumptions.ccs_retrofit_lifetime_years,
        "pipe_capex": assumptions.pipeline_lifetime_years,
        "blend_upgrade_capex": assumptions.blend_upgrade_lifetime_years,
        "air_retrofit_capex": assumptions.air_retrofit_lifetime_years,
        "rebuild_capex": assumptions.rebuild_lifetime_years,
        # 水泥没有氢路线，工业 capex 全是捕集岛。
        "industry_capex": ci.INDUSTRY_CAPTURE_LIFETIME_YEARS,
    }
    expected = 0.0
    for year in (2030, 2040, 2050):
        df_t = 1.0 / (1.0 + rate) ** (year - base)
        for key, life in booked_lives.items():
            booked = float(ys[year]["cost_breakdown_cny"][key])  # 已按 df_t 折现
            expected -= remaining_fraction(year, life, end_year) * booked / df_t * df_end
    assert float(ys[2050]["cost_breakdown_cny"]["industry_capex"]) > 0.0
    assert float(ys[2050]["cost_breakdown_cny"]["salvage_credit"]) == pytest.approx(expected, rel=1e-6)


def test_salvage_switch_off_reproduces_pre_20260922_objective(tmp_path) -> None:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _, _, off = _solve(paths, salvage=False)
    _, _, on = _solve(paths, salvage=True)
    assert off["status"] == "optimal" and on["status"] == "optimal"
    # 关闭时：该键不存在，cost_breakdown 回到没有残值行的表结构。
    for year in (2030, 2040, 2050):
        assert "salvage_credit" not in off["year_solutions"][year]["cost_breakdown_cny"]

    def total(sol, key):
        return sum(float(sol["year_solutions"][y]["cost_breakdown_cny"][key]) for y in (2030, 2040, 2050))

    def objective(sol):
        return sum(
            float(v)
            for y in (2030, 2040, 2050)
            for v in sol["year_solutions"][y]["cost_breakdown_cny"].values()
        )

    # 决策相同（目标把 2050 年的建设钉死），所以两个目标值只差残值抵扣一项，
    # 且计残值的目标值更低。
    assert total(on, "ccs_retrofit_capex") == pytest.approx(total(off, "ccs_retrofit_capex"), rel=1e-4)
    assert objective(on) < objective(off)
    assert objective(off) - objective(on) == pytest.approx(-total(on, "salvage_credit"), rel=1e-4)
