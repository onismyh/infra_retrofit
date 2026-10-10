"""三个敏感性开关（2026-10-02 起，缺省都不改模型）：部分负荷修正、煤价乘子、煤电容量电价。不需要 Gurobi。

部分负荷修正 κ 乘在煤电各部分的毛热耗与基线排放上（`plant_matrices`），2030 年电力基线同乘（`model_index`）；煤价乘子乘在
分省煤价上，煤电与工业捕集蒸汽的用煤同时变；容量电价按未退役装机收，电量电价减去 容量电价 ÷ 全机组现状利用小时。
toy 电厂只有一座 1 000 MW、4 629.5 h 的山西电厂，旁边一个水泥 hub（`toy_inputs`）。
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.constants import part_load_heat_rate_factor
from coal_retrofit.constants_industry import capture_variable_cost_cny_per_t
from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.industry_matrices import CCS
from coal_retrofit.optimization.model_index import build_model_index
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.year_matrices import _build_year_matrices
from coal_retrofit.optimization.year_types import YearData
from toy_inputs import TOY_UNIT_TYPE, _write_toy_inputs, _write_toy_units

ST_HOURS = (3600.0, 3100.0, 2000.0, 1500.0)
SCENARIO = OptimizationScenario(experiment_id="TEST-SWITCH", description="toy", sector_target_source="toy")
# 部分到期的 hub：两台 500 MW，2005、2025 年投产，2050 年到期一半，重建部分与未重建部分的系数之差不为零。
HALF_EXPIRED_UNITS = pd.DataFrame({
    "plant_id": ["P1", "P1"], "capacity_mw": [500.0, 500.0], "commission_year": [2005, 2025],
    **{column: values * 2 for column, values in TOY_UNIT_TYPE.items()},
})


def _year_data(root, scenario: OptimizationScenario, assumptions: OptimizationAssumptions, year: int,
               retirement_year: int = 9999, units: pd.DataFrame | None = None) -> YearData:
    paths = _write_toy_inputs(root, retirement_year=retirement_year)
    if units is not None:
        _write_toy_units(paths, units)
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    return _build_year_matrices(prepared, scenario, assumptions, year, state)


@pytest.mark.parametrize(
    ("load", "expected"),
    [(0.45, 1.126), (0.40, 1.160), (0.30, 1.281), (0.20, 1.609), (0.10, 2.747), (0.4999, 1.100), (0.50, 1.0), (1.0, 1.0)],
)
def test_part_load_curve_is_the_mee_table_2(load: float, expected: float) -> None:
    """征求意见稿表 2 的调峰修正系数：F < 50% 按指数式，F >= 50% 为 1（式在 50% 处为 1.100，原文字面跳变）。"""
    assert part_load_heat_rate_factor(load) == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize(
    ("online", "expected"),
    [(7500.0, (1.110, 1.150, 1.353, 1.609)), (6500.0, (1.000, 1.111, 1.268, 1.466)), (0.0, (1.0, 1.0, 1.0, 1.0))],
)
def test_part_load_factor_on_the_st_hours(online: float, expected: tuple[float, ...]) -> None:
    """ST 的小时轨迹：在线 7 500 h 为主档、6 500 h 为低档；在线小时为 0 即关，四年都是 1。"""
    scenario = dataclasses.replace(SCENARIO, coal_operating_hours_by_year=ST_HOURS, coal_part_load_online_hours=online)
    factors = [scenario.part_load_factor(year, 4643.0) for year in scenario.planning_years]
    np.testing.assert_allclose(factors, expected, atol=5e-4)


def test_part_load_without_an_hours_path_uses_the_current_fleet_hours() -> None:
    """没设小时轨迹时利用小时取全机组当前平均：4 643 h、在线 7 500 h 时 F = 62%，不修正；在线 8 760 h 时 F = 53%，同样为 1。"""
    for online in (7500.0, 8760.0):
        scenario = dataclasses.replace(SCENARIO, coal_part_load_online_hours=online)
        assert scenario.part_load_factor(2030, 4643.0) == 1.0
    scenario = dataclasses.replace(SCENARIO, coal_part_load_online_hours=8760.0)
    assert scenario.part_load_factor(2030, 2000.0) == pytest.approx(part_load_heat_rate_factor(2000.0 / 8760.0))


@pytest.mark.parametrize("online", [-1.0, 8761.0, math.nan, 3000.0])
def test_part_load_rejects_impossible_online_hours(online: float) -> None:
    """在线小时须在 (0, 8760] 内，且不少于利用小时（ST 2030 年 3 600 h）。"""
    scenario = dataclasses.replace(SCENARIO, coal_operating_hours_by_year=ST_HOURS, coal_part_load_online_hours=online)
    with pytest.raises(ValueError, match="coal_part_load_online_hours"):
        scenario.part_load_factor(2030, 4643.0)


def _net(data: YearData) -> np.ndarray:
    """基线净运行成本的绝对值：进目标的是相对未改造列的差，参照另存（2026-10-10 起，`plant_matrices`）。"""
    return data.baseline_net_matrix + data.baseline_reference_cny[:, None]


@pytest.mark.parametrize("retirement_year", [9999, 2045])
def test_part_load_scales_heat_rates_emissions_and_penalties(tmp_path, retirement_year: int) -> None:
    """2050 年 2 000 h、在线 7 500 h（κ = 1.353）：排放、毛热耗、CCS 能耗惩罚与燃料成本乘 κ；生物质与空冷背压的惩罚燃料按
    (pp / η) x 毛热耗折算，η 也随之降，乘 κ²；发电量与耗水强度不变。2045 年到期的那组 2050 年运行的是重建机组，同样乘 κ。"""
    hours = dataclasses.replace(SCENARIO, coal_operating_hours_by_year=ST_HOURS)
    on = dataclasses.replace(hours, coal_part_load_online_hours=7500.0)
    assumptions = OptimizationAssumptions()
    off_data = _year_data(tmp_path / "off", hours, assumptions, 2050, retirement_year)
    on_data = _year_data(tmp_path / "on", on, assumptions, 2050, retirement_year)
    kappa = part_load_heat_rate_factor(2000.0 / 7500.0)
    assert (off_data.part_load_factor, on_data.part_load_factor) == (1.0, pytest.approx(kappa))
    assert kappa == pytest.approx(1.353, abs=5e-4)
    linear = ("emissions_mt", "heat_rate_eff", "emissions_operating_mt", "emissions_retrofit_mt", "energy_penalty_matrix",
              "ccs_penalty_emissions_matrix", "ccs_penalty_captured_matrix")
    quadratic = ("biomass_penalty_coeff_per_level", "biomass_penalty_emissions_coeff_per_level",
                 "air_penalty_emissions_matrix", "air_penalty_captured_matrix", "air_penalty_cost_matrix")
    for name in linear:
        np.testing.assert_allclose(getattr(on_data, name), kappa * getattr(off_data, name), rtol=1e-12, err_msg=name)
    for name in quadratic:
        np.testing.assert_allclose(getattr(on_data, name), kappa**2 * getattr(off_data, name), rtol=1e-12, err_msg=name)
    for name in ("generation", "generation_by_pathway", "water_intensity", "air_water_intensity"):
        np.testing.assert_array_equal(getattr(on_data, name), getattr(off_data, name), err_msg=name)
    # 基线净运行成本只有燃料一项随毛热耗变：差 = 发电量 x 毛热耗 x (κ − 1) x 煤价。
    fuel = off_data.generation_by_pathway * (off_data.heat_rate_eff * assumptions.province_coal_cost("Shanxi"))[:, None]
    np.testing.assert_allclose(
        _net(on_data) - _net(off_data), (kappa - 1.0) * fuel, rtol=1e-9, atol=1e-3
    )


def test_part_load_and_capacity_price_on_the_rebuilt_part(tmp_path) -> None:
    """部分到期的 hub（2050 年到期一半）：重建部分与未重建部分的系数之差同样乘 κ，平方项乘 κ²；容量电价与毛热耗无关，在差里相消。"""
    hours = dataclasses.replace(SCENARIO, coal_operating_hours_by_year=ST_HOURS)
    variants = {
        "off": hours,
        "part_load": dataclasses.replace(hours, coal_part_load_online_hours=7500.0),
        "capacity": dataclasses.replace(hours, coal_capacity_price_cny_per_kw_yr=100.0),
    }
    data = {name: _year_data(tmp_path / name, scenario, OptimizationAssumptions(), 2050, units=HALF_EXPIRED_UNITS)
            for name, scenario in variants.items()}
    assert data["off"].expired_share[0] == pytest.approx(0.5)
    off, on = data["off"].rebuilt_deltas[0], data["part_load"].rebuilt_deltas[0]
    assert np.all(off.heat_rate_eff != 0.0)
    kappa = part_load_heat_rate_factor(2000.0 / 7500.0)
    linear = ("heat_rate_eff", "emissions_operating_mt", "emissions_retrofit_mt", "energy_penalty_matrix",
              "ccs_penalty_emissions_matrix", "ccs_penalty_captured_matrix", "baseline_net_matrix")
    quadratic = ("biomass_penalty_coeff_per_level", "biomass_penalty_emissions_coeff_per_level",
                 "beccs_penalty_emissions_coeff_per_level", "beccs_penalty_captured_coeff_per_level", "air_penalty_cost_matrix")
    for name in linear:
        np.testing.assert_allclose(getattr(on, name), kappa * getattr(off, name), rtol=1e-9, err_msg=name)
    for name in quadratic:
        np.testing.assert_allclose(getattr(on, name), kappa**2 * getattr(off, name), rtol=1e-9, err_msg=name)
    np.testing.assert_allclose(data["capacity"].rebuilt_deltas[0].baseline_net_matrix, off.baseline_net_matrix,
                               rtol=1e-9, atol=1e-3)


def test_part_load_scales_the_2030_power_baseline(tmp_path) -> None:
    """2030 年电力基线乘 κ_2030（3 600 h / 7 500 h，κ = 1.110），与 2030 年的排放同口径，不然 2030 年要凭空多减约 10%。"""
    hours = dataclasses.replace(SCENARIO, coal_operating_hours_by_year=ST_HOURS)
    on = dataclasses.replace(hours, coal_part_load_online_hours=7500.0)
    assumptions = OptimizationAssumptions()
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    bases = [build_model_index(prepare_inputs(paths, s, assumptions), s, assumptions).sector_base_2030 for s in (hours, on)]
    kappa = part_load_heat_rate_factor(3600.0 / 7500.0)
    assert bases[1]["power"] == pytest.approx(kappa * bases[0]["power"], rel=1e-12)
    assert bases[1]["cement"] == bases[0]["cement"]  # 只修正煤电
    on_2030 = _year_data(tmp_path / "y", on, assumptions, 2030)
    assert bases[1]["power"] == pytest.approx(float(on_2030.emissions_mt.sum()), rel=1e-12)


def test_coal_price_multiplier_scales_plant_and_industry_coal(tmp_path) -> None:
    """煤价乘 0.89：煤电的燃料与各项惩罚燃料成本、掺烧替代的煤、工业捕集的蒸汽用煤都乘 0.89；排放与电价不变。"""
    base = OptimizationAssumptions()
    cheap = dataclasses.replace(base, coal_price_multiplier=0.89)
    assert cheap.province_coal_cost("Shanxi") == pytest.approx(0.89 * base.province_coal_cost("Shanxi"))
    assert cheap.province_coal_cost("Nowhere") == pytest.approx(0.89 * base.coal_fuel_cost_cny_per_gj)
    off_data = _year_data(tmp_path / "off", SCENARIO, base, 2040)
    on_data = _year_data(tmp_path / "on", SCENARIO, cheap, 2040)
    for name in ("energy_penalty_matrix", "biomass_penalty_coeff_per_level", "air_penalty_cost_matrix", "coal_savings_per_gj",
                 "coal_savings_per_kg_nh3"):
        np.testing.assert_allclose(getattr(on_data, name), 0.89 * getattr(off_data, name), rtol=1e-12, err_msg=name)
    np.testing.assert_array_equal(on_data.emissions_mt, off_data.emissions_mt)
    fuel = off_data.generation_by_pathway * (off_data.heat_rate_eff * base.province_coal_cost("Shanxi"))[:, None]
    np.testing.assert_allclose(
        _net(on_data) - _net(off_data), -0.11 * fuel, rtol=1e-9, atol=1e-3
    )
    # 水泥 CCS 的年度成本 = 捕集量 x (蒸汽用煤 + 电 + 耗材)：只有蒸汽用煤一项乘 0.89。
    elec = SCENARIO.electricity_price_for_year(2040)
    captured_t = off_data.industry.captured_mt[0, CCS] * 1e6
    waste_heat = base.industry_capture_waste_heat_share["cement"]
    steam_coal = (capture_variable_cost_cny_per_t("cement", base.province_coal_cost("Shanxi"), elec, waste_heat)
                  - capture_variable_cost_cny_per_t("cement", 0.0, elec, waste_heat))
    assert on_data.industry.opex_cny[0, CCS] - off_data.industry.opex_cny[0, CCS] == pytest.approx(
        -0.11 * steam_coal * captured_t, rel=1e-9
    )
    np.testing.assert_array_equal(on_data.industry.reduction_mt, off_data.industry.reduction_mt)


@pytest.mark.parametrize("multiplier", [0.0, -0.5, math.nan, math.inf])
def test_coal_price_multiplier_must_be_positive(multiplier: float) -> None:
    with pytest.raises(ValueError, match="coal_price_multiplier"):
        OptimizationAssumptions(coal_price_multiplier=multiplier).province_coal_cost("Shanxi")


@pytest.mark.parametrize(("hours_path", "year"), [((), 2030), (ST_HOURS, 2050)])
def test_capacity_price_moves_revenue_from_energy_to_capacity(tmp_path, hours_path, year: int) -> None:
    """容量电价 100 元/(kW·a)：运行各列多收 100 x 1 000 MW x 1 000，电量电价减 100 x 1 000 / 现状小时（4 629.5 h，21.6 元/MWh），
    退役列不变（为零）。按现状小时（没设小时轨迹）运行时未改造列的净成本不变；2 000 h 时容量电费多于少收的电量电费。"""
    base = dataclasses.replace(SCENARIO, coal_operating_hours_by_year=hours_path)
    paid = dataclasses.replace(base, coal_capacity_price_cny_per_kw_yr=100.0)
    assumptions = OptimizationAssumptions()
    off_data = _year_data(tmp_path / "off", base, assumptions, year)
    on_data = _year_data(tmp_path / "on", paid, assumptions, year)
    hours_now, capacity_kw = 4629.5, 1000.0 * 1000.0
    expected = on_data.generation_by_pathway * (100.0 * 1000.0 / hours_now) - 100.0 * capacity_kw
    expected[:, PATHWAY_INDEX["retire"]] = 0.0
    np.testing.assert_allclose(_net(on_data) - _net(off_data), expected, rtol=1e-9, atol=1e-3)
    assert _net(on_data)[0, PATHWAY_INDEX["retire"]] == 0.0
    unabated = PATHWAY_INDEX["unabated"]
    if not hours_path:  # 按现状小时：总收入不变
        assert _net(on_data)[0, unabated] == pytest.approx(_net(off_data)[0, unabated], rel=1e-9)
    else:  # 2 000 h：容量电费 1e8 元，少收的电量电费 2 000 h x 1 000 MW x 21.6 元/MWh = 4.32e7 元
        assert _net(on_data)[0, unabated] - _net(off_data)[0, unabated] == pytest.approx(
            100.0 * capacity_kw * (2000.0 / hours_now - 1.0), rel=1e-6
        )
    # 工业买电仍按全价：水泥 CCS 的年度成本不变。
    np.testing.assert_array_equal(on_data.industry.opex_cny, off_data.industry.opex_cny)


@pytest.mark.parametrize("price", [-100.0, math.nan, math.inf])
def test_capacity_price_must_be_positive(tmp_path, price: float) -> None:
    scenario = dataclasses.replace(SCENARIO, coal_capacity_price_cny_per_kw_yr=price)
    with pytest.raises(ValueError, match="coal_capacity_price_cny_per_kw_yr"):
        _year_data(tmp_path, scenario, OptimizationAssumptions(), 2030)
