"""工业 CCS 的可捕集份额、再生蒸汽的余热份额与氢路线口径（2026-10-02）。

可捕集份额按（部门，原料）给，按点源 CO2 加权到 hub（`prepare_industry`）；捕集量 = hub CO2 x 份额 x 捕集率，份额为 0 的
hub 不开放 CCS（电炉钢、焦炉煤气与天然气制甲醇）。合成氨、甲醇的氢路线减排比例也取这个份额，为 0 时不开放；混合原料的
hub 里，两条路线都只作用于份额为正的点源（能力的铭牌系数、氢路线的产量、需氢与取水）。长流程钢的需氢量取 63 kg/t。
余热份额按部门给，蒸汽用煤与蒸汽 CO2 都乘 (1 − 份额)，缺省 0。这些测试不求解，不依赖 Gurobi。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit import constants_industry as ci
from coal_retrofit.optimization.industry import CCS, H2, IndustryInputs, industry_year_data, prepare_industry
from coal_retrofit.optimization.results_industry import _build_industry_detail_table
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.scenarios import parse_set
from test_h2_route_multiplier import _steel_hub
from test_province_names import _cement_hubs

_SCENARIO = OptimizationScenario(experiment_id="T", description="toy")
_RATE = _SCENARIO.capture_rate


def _chemical_paths(root):
    """三个化工 hub（各产量与铭牌 100 万 t/a、取水 100 万 m3/a）：M1 煤制甲醇 3 Mt + 焦炉煤气制甲醇 1 Mt CO2（产量 60 / 40 万 t、
    铭牌 75 / 25 万 t、取水 70 / 30 万 m3），M2 只有焦炉煤气制甲醇，A1 气头合成氨。"""
    paths = _cement_hubs(root, ["Shanxi"] * 3)  # 借它写 `prepare_industry` 要读的氢供给曲线
    hubs = pd.read_csv(paths.inputs_dir / "industry_hubs.csv")
    hubs["hub_id"] = ["M1", "M2", "A1"]
    hubs["sector"] = ["methanol", "methanol", "ammonia"]
    hubs["co2_mt_per_year"] = [4.0, 1.0, 1.6]
    hubs["h2_demand_kt_per_year"] = [190.0, 190.0, 180.0]
    hubs.to_csv(paths.inputs_dir / "industry_hubs.csv", index=False)
    pd.DataFrame({
        "source_id": ["s1", "s2", "s3", "s4"],
        "hub_id": ["M1", "M1", "M2", "A1"],
        "sector": ["methanol", "methanol", "methanol", "ammonia"],
        "feedstock": ["Coal", "Coke oven gas", "Coke oven gas", "Gas"],
        "co2_mt_per_year": [3.0, 1.0, 1.0, 1.6],
        "production_kt_per_year": [600.0, 400.0, 1000.0, 1000.0],
        "capacity_kt_per_year": [750.0, 250.0, 1000.0, 1000.0],
        "water_m3_per_year": [0.7e6, 0.3e6, 1.0e6, 1.0e6],
    }).to_csv(paths.inputs_dir / "industry_sources.csv", index=False)
    return paths


def test_capturable_share_table() -> None:
    """化工按原料取值，表里没有的原料按该部门的 default 行（煤头）；长流程钢、水泥 1，电炉钢 0；没有 default 行的部门报错。"""
    assert ci.capturable_share("ammonia", "Coal") == 0.75
    assert ci.capturable_share("ammonia", "Gas") == 0.67
    assert ci.capturable_share("ammonia", "Oil") == ci.capturable_share("ammonia", "Anthracite") == 0.75
    assert ci.capturable_share("methanol", "Coal") == ci.capturable_share("methanol", "电石炉尾气") == 0.57
    assert ci.capturable_share("methanol", "Coke oven gas") == ci.capturable_share("methanol", "Gas") == 0.0
    assert ci.capturable_share("steel_bf_bof") == ci.capturable_share("cement") == 1.0
    assert ci.capturable_share("steel_eaf") == 0.0
    with pytest.raises(KeyError):
        ci.capturable_share("refinery")


def test_hub_share_is_weighted_by_source_co2_and_gates_ccs_and_h2(tmp_path) -> None:
    """M1 = 0.57 x 3/4；M2 = 0，CCS 与氢路线都不开放；A1 = 0.67。CCS 捕集量 = CO2 x 份额 x 捕集率，
    氢路线减排 = CO2 x 份额（2026-10-02 前分别为 CO2 x 捕集率与 CO2 x 0.90 / 0.95）。"""
    industry = prepare_industry(_chemical_paths(tmp_path), OptimizationAssumptions())
    share = industry.hubs.set_index("hub_id")["capturable_share"]
    assert share["M1"] == pytest.approx(0.57 * 3.0 / 4.0, rel=1e-12)
    assert share["M2"] == 0.0
    assert share["A1"] == pytest.approx(0.67, rel=1e-12)
    data = industry_year_data(industry, _SCENARIO, OptimizationAssumptions(), 2030)
    assert data.route_available[:, CCS].tolist() == [True, False, True]
    assert data.route_available[:, H2].tolist() == [True, False, True]
    expected = np.array([4.0 * share["M1"], 0.0, 1.6 * 0.67])
    np.testing.assert_allclose(data.captured_mt[:, CCS], expected * _RATE, rtol=1e-12)
    np.testing.assert_allclose(data.reduction_mt[:, CCS], expected * _RATE, rtol=1e-12)  # 纯流股不用蒸汽
    np.testing.assert_allclose(data.reduction_mt[:, H2], expected, rtol=1e-12)


def test_mixed_hub_routes_act_on_its_positive_share_sources_only(tmp_path) -> None:
    """M1 的焦炉煤气点源没有纯流股：氢路线的产量、能力、需氢、运行差额与取水只计煤制点源（产量 60%、铭牌 75%、取水 70%），
    两条路线的能力都按煤制点源的铭牌系数 0.75 / 0.6 = 1.25 定（整个 hub 为 1）。A1 只有一个点源，三个比例为 1，各系数与
    按整个 hub 算的逐位相同。"""
    industry = prepare_industry(_chemical_paths(tmp_path), OptimizationAssumptions())
    columns = ["abatable_production_share", "abatable_capacity_share", "abatable_water_share"]
    assert industry.hubs.set_index("hub_id").loc["M1", columns].tolist() == pytest.approx([0.6, 0.75, 0.7], rel=1e-12)
    data = industry_year_data(industry, _SCENARIO, OptimizationAssumptions(), 2030)
    production_t = 0.6e6
    assert data.capacity_mt_per_share[0, CCS] == pytest.approx(data.captured_mt[0, CCS] * 1.25, rel=1e-12)
    assert data.capacity_mt_per_share[0, H2] == pytest.approx(production_t * 1.25 / 1e6, rel=1e-12)
    assert data.h2_demand_kg_per_share[0] == pytest.approx(190.0 * production_t, rel=1e-12)
    assert data.h2_demand_kg_per_share[2] == 180.0 * 1.0e6
    assert data.capacity_mt_per_share[2, CCS] == data.captured_mt[2, CCS]
    capex = ci.h2_route_capex_cny_per_t_yr("methanol")
    delta = ci.h2_route_opex_delta_cny_per_t("methanol", 0.19, _SCENARIO.discount_rate)
    assert data.opex_cny[0, H2] == pytest.approx(
        production_t * 1.25 * capex * ci.INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION + production_t * delta, rel=1e-12
    )
    ratio = {s: ci.water_quota(s, "default", advanced=True) / ci.water_quota(s, "default") for s in ("methanol", "ammonia")}
    assert data.water_m3[0, H2] == pytest.approx(0.3e6 + 0.7e6 * ratio["methanol"], rel=1e-12)
    assert data.water_m3[2, H2] == 1.0e6 * ratio["ammonia"]


@pytest.mark.parametrize(("with_sources", "expected"), [(True, [0.6, 0.0, 1.0]), (False, [1.0, 1.0, 1.0])])
def test_detail_table_records_the_production_share_of_rebuilt_sources(
    tmp_path, with_sources: bool, expected: list[float]
) -> None:
    """工业明细表记份额为正的点源占 hub 产量的比例（图 4 按它把不改造的产量计入未改造）；没有点源表时为 1。"""
    paths = _chemical_paths(tmp_path)
    if not with_sources:
        (paths.inputs_dir / "industry_sources.csv").unlink()
    industry = prepare_industry(paths, OptimizationAssumptions())
    data = industry_year_data(industry, _SCENARIO, OptimizationAssumptions(), 2030)
    zeros = np.zeros_like(data.capacity_mt_per_share)
    table = _build_industry_detail_table(
        SimpleNamespace(industry=industry), 2030, data, np.tile([1.0, 0.0, 0.0], (3, 1)),  # type: ignore[arg-type]
        capacity_mt=zeros, new_capacity_mt=zeros, ccs_fixed_om_cny=np.zeros(3),
    )
    assert table["abatable_production_share"].tolist() == pytest.approx(expected, rel=1e-12)


def test_hub_without_point_sources_is_refused(tmp_path) -> None:
    """点源表里没有某个 hub 的点源时报错，不静默按部门缺省。"""
    paths = _chemical_paths(tmp_path)
    sources = pd.read_csv(paths.inputs_dir / "industry_sources.csv")
    sources[sources["hub_id"] != "A1"].to_csv(paths.inputs_dir / "industry_sources.csv", index=False)
    with pytest.raises(ValueError, match="A1"):
        prepare_industry(paths, OptimizationAssumptions())


def test_without_a_source_table_each_sector_takes_its_default_share(tmp_path) -> None:
    """只有 hub 表时（toy 输入）不加 `capturable_share` 列，各 hub 按部门 default 原料（煤头）：甲醇 0.57、合成氨 0.75。"""
    paths = _chemical_paths(tmp_path)
    (paths.inputs_dir / "industry_sources.csv").unlink()
    industry = prepare_industry(paths, OptimizationAssumptions())
    assert "capturable_share" not in industry.hubs.columns
    data = industry_year_data(industry, _SCENARIO, OptimizationAssumptions(), 2030)
    np.testing.assert_allclose(data.captured_mt[:, CCS], np.array([4.0 * 0.57, 1.0 * 0.57, 1.6 * 0.75]) * _RATE, rtol=1e-12)


def test_capturable_share_outside_the_unit_interval_is_refused() -> None:
    industry = _steel_hub()
    hubs = industry.hubs.assign(capturable_share=1.2)
    with pytest.raises(ValueError, match="capturable_share"):
        industry_year_data(IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0}), _SCENARIO, OptimizationAssumptions(), 2030)


def test_eaf_hub_can_only_stay_unabated() -> None:
    """电炉钢不开放 CCS（份额 0，2026-10-02 起），本来也没有氢路线。"""
    hubs = _steel_hub().hubs.assign(sector="steel_eaf", h2_demand_kt_per_year=0.0)
    data = industry_year_data(IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0}), _SCENARIO, OptimizationAssumptions(), 2030)
    assert data.route_available[0].tolist() == [True, False, False]
    assert data.captured_mt[0, CCS] == 0.0


@pytest.mark.parametrize(
    ("columns", "available"),
    [
        ({"capturable_share": 0.0}, [True, False, False]),
        ({"capturable_share": 0.57, "abatable_production_share": 0.0}, [True, True, False]),
    ],
)
def test_chemical_h2_route_stays_closed_without_a_share_or_production_to_rebuild(
    columns: dict[str, float], available: list[bool]
) -> None:
    """甲醇 hub 的氢路线两道门：hub 表直接给份额 0（没有比例列，改造产量按整个 hub）时不开放；份额为正、但份额为正的
    点源没有产量时也不开放，否则减排不花钱（CCS 按捕集量定规模，照常开放）。"""
    hubs = _steel_hub().hubs.assign(sector="methanol", h2_demand_kt_per_year=190.0, **columns)
    data = industry_year_data(IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0}), _SCENARIO, OptimizationAssumptions(), 2030)
    assert data.route_available[0].tolist() == available
    assert data.reduction_mt[0, H2] == 0.0


@pytest.mark.parametrize("table_h2_kt", [81.0, 0.0])
def test_steel_h2_route_buys_63_kg_per_tonne(table_h2_kt: float) -> None:
    """长流程钢 100 万 t/a 全部转氢需氢 6.3 万 t（PyPSA technology-data 的 MPP 修正值），不取点源表的 81 kg/t；
    常量覆盖点源表，点源表没有需氢量时氢路线照样开放。"""
    hubs = _steel_hub().hubs.assign(h2_demand_kt_per_year=table_h2_kt)
    data = industry_year_data(IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0}), _SCENARIO, OptimizationAssumptions(), 2030)
    assert data.route_available[0, H2]
    assert data.h2_demand_kg_per_share[0] == pytest.approx(63.0e6, rel=1e-12)
    assert data.reduction_mt[0, H2] == pytest.approx(0.95 * 2.0, rel=1e-12)


@pytest.mark.parametrize("share", [0.0, 0.6])
def test_waste_heat_share_cuts_steam_coal_and_steam_co2_of_its_sector_only(share: float) -> None:
    """长流程钢 + 水泥两个 hub，只给水泥设余热份额：水泥的蒸汽用煤与蒸汽 CO2 都乘 (1 − 份额)，钢铁不变；
    份额 0 与缺省（空表）相同。"""
    steel = _steel_hub().hubs
    hubs = pd.concat([steel, steel.assign(hub_id="C1", sector="cement", h2_demand_kt_per_year=0.0)], ignore_index=True)
    industry = IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0})
    assumptions = OptimizationAssumptions(industry_capture_waste_heat_share={"cement": share})
    base = industry_year_data(industry, _SCENARIO, OptimizationAssumptions(), 2030)
    data = industry_year_data(industry, _SCENARIO, assumptions, 2030)
    coal_per_t = ci.INDUSTRY_CCS_STEAM_GJ_PER_T_CO2["cement"] / ci.INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY
    coal_cost_per_t = coal_per_t * assumptions.province_coal_cost("Shanxi")
    captured_t = data.captured_mt[1, CCS] * 1e6
    assert data.opex_cny[1, CCS] == pytest.approx(base.opex_cny[1, CCS] - share * captured_t * coal_cost_per_t, rel=1e-12)
    steam_co2 = coal_per_t * assumptions.coal_emission_factor_t_per_gj
    assert data.reduction_mt[1, CCS] == pytest.approx(data.captured_mt[1, CCS] * (1.0 - (1.0 - share) * steam_co2), rel=1e-12)
    np.testing.assert_array_equal(data.opex_cny[0], base.opex_cny[0])
    np.testing.assert_array_equal(data.reduction_mt[0], base.reduction_mt[0])


@pytest.mark.parametrize("value", [{"cemnt": 0.5}, {"cement": 1.5}, {"cement": -0.1}, {"cement": float("nan")}])
def test_invalid_waste_heat_share_is_refused(value: dict[str, float]) -> None:
    """键写错（不是模型内的部门）或份额不在 [0, 1]（含 NaN）时报错。"""
    assumptions = OptimizationAssumptions(industry_capture_waste_heat_share=value)
    with pytest.raises(ValueError, match="industry_capture_waste_heat_share"):
        industry_year_data(_steel_hub(), _SCENARIO, assumptions, 2030)


def test_waste_heat_share_can_be_set_from_the_command_line() -> None:
    """敏感性按字段注释写的 `--set` 跑：值按 TOML 内联表读成 {部门: 份额}。"""
    assert parse_set("assumptions.industry_capture_waste_heat_share={cement=0.6}") == (
        "assumptions", "industry_capture_waste_heat_share", {"cement": 0.6},
    )
