"""工业氢路线的成本乘子（2026-09-23，与 (d) 对齐）：只乘路线 capex 与随之的固定运维，
由锚点反推的非氢运行差额固定在乘子为 1 时的值。

这些测试不求解，单独成文件以免被 Gurobi 门控（见 `test_discount_rate.py` 的说明）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from coal_retrofit import constants_industry as ci
from coal_retrofit.optimization.industry import H2, IndustryInputs, industry_year_data
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario


def _steel_hub() -> IndustryInputs:
    """一个长流程钢 hub：100 万 t/a，氢强度 0.081 t/t（有氢路线）。"""
    hubs = pd.DataFrame({
        "hub_id": ["S1"], "sector": ["steel_bf_bof"], "province": ["Shanxi"],
        "longitude": [112.0], "latitude": [37.0],
        "production_kt_per_year": [1000.0], "co2_mt_per_year": [2.0],
        "process_co2_mt_per_year": [0.2], "h2_demand_kt_per_year": [81.0],
        "water_m3_per_year": [3.0e6], "target_group": ["steel"],
    })
    return IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0})


def test_industry_h2_multiplier_scales_route_capex_and_its_fixed_om_only() -> None:
    """乘子 2 使路线 capex 与其固定运维翻倍；由锚点反推的非氢运行差额停在乘子为 1 时的值。
    2026-09-23 前差额里路线自身的成本也翻倍（锚点价下平准化溢价恰为 2 倍锚点溢价）。"""
    industry = _steel_hub()
    assumptions = OptimizationAssumptions()
    base_scenario = OptimizationScenario(experiment_id="T", description="toy")
    base = industry_year_data(industry, base_scenario, assumptions, 2030)
    doubled = industry_year_data(
        industry, OptimizationScenario(experiment_id="T", description="toy", industry_h2_cost_multiplier=2.0),
        assumptions, 2030,
    )
    production_t = 1.0e6
    capex = ci.h2_route_capex_cny_per_t_yr("steel_bf_bof")
    delta = ci.h2_route_opex_delta_cny_per_t("steel_bf_bof", 0.081, base_scenario.discount_rate)
    assert delta != 0.0
    assert base.route_available[0, H2] and doubled.route_available[0, H2]
    assert doubled.capex_cny_per_mt[0, H2] == pytest.approx(2.0 * base.capex_cny_per_mt[0, H2], rel=1e-12)
    assert base.opex_cny[0, H2] == pytest.approx(
        production_t * (capex * ci.INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION + delta), rel=1e-9
    )
    assert doubled.opex_cny[0, H2] == pytest.approx(
        production_t * (2.0 * capex * ci.INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION + delta), rel=1e-9
    )


@pytest.mark.parametrize(("sector", "increase"), [("steel_bf_bof", 0.2516), ("ammonia", 0.1415), ("methanol", 0.0189)])
def test_h2_multiplier_reach_at_the_anchor_price(sector: str, increase: float) -> None:
    """锚点氢价下乘子取 2 只让平准化溢价增加 capex x (CRF + 固定运维比例)：钢铁 25%、合成氨 14%、
    甲醇 2%，即 scenario.py 字段注释与实现说明 §四.1 写的数。2026-09-23 前为 +100%。"""
    premium_ref, price_ref = ci.INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT[sector]
    k = {"steel_bf_bof": 0.081, "ammonia": 0.18, "methanol": 0.19}[sector]
    rate = OptimizationScenario(experiment_id="T", description="toy").discount_rate
    doubled = ci.h2_premium_cny_per_t(sector, price_ref, k, rate, 2.0)
    assert doubled == pytest.approx(premium_ref + ci.h2_route_annual_capital_cny_per_t(sector, rate), rel=1e-9)
    assert doubled / premium_ref - 1.0 == pytest.approx(increase, abs=1e-4)
