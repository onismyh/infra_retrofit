"""2026-09-23 模型改动 (a)(c)(d) 里不求解的测试，外加工业明细表的 capital 列。

前两条从 `test_capex_stock_and_lifetimes.py` 挪来：那个文件在模块级 `importorskip("gurobipy")`，
放在那里的测试在没有 Gurobi 的环境里会整体跳过（见 `test_discount_rate.py` 的说明）。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.constants_industry import INDUSTRY_CCS_FIXED_OM_FRACTION
from coal_retrofit.optimization.industry import CCS, H2, IndustryInputs, industry_year_data
from coal_retrofit.optimization.results_industry import _build_industry_detail_table
from coal_retrofit.optimization.results_network import _alive_edge_added_stock
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from test_h2_route_multiplier import _steel_hub


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


# ------------------------------------------------- (c) 管道到寿命后可原位重建 ---
def test_alive_edge_stock_drops_pipes_at_the_end_of_their_lifetime() -> None:
    """在役 = 建成年早于当年且未满寿命；满 30 年当年即退出，当年新建的不算往期存量。"""
    added = {2030: np.array([5.0, 0.0]), 2050: np.array([0.0, 2.0])}
    np.testing.assert_allclose(_alive_edge_added_stock(added, 2050, 30, 2), [5.0, 0.0])
    np.testing.assert_allclose(_alive_edge_added_stock(added, 2059, 30, 2), [5.0, 2.0])
    np.testing.assert_allclose(_alive_edge_added_stock(added, 2060, 30, 2), [0.0, 2.0])


# ------------------------------------- (d) 成本乘子只乘 capex 与随 capex 的固定运维 ---
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


# ------------------------------------------------ (a) 明细表的 capital 按能力存量增量计 ---
def test_industry_detail_table_charges_capital_on_the_capacity_increment() -> None:
    """明细表的 capital 列与目标函数同法：单位 capex x 能力存量增量，CCS 与氢路线各算各的；
    不传上一年存量时（第一年）按整个存量计。2026-09-23 前 context model 不传存量，
    每年都按整个存量计，跨年重复计入。"""
    industry = _steel_hub()
    data = industry_year_data(industry, OptimizationScenario(experiment_id="T", description="toy"), OptimizationAssumptions(), 2030)
    unit = data.capex_cny_per_mt
    assert unit[0, CCS] > 0.0 and unit[0, H2] > 0.0
    prepared = SimpleNamespace(industry=industry)
    share = np.array([[0.2, 0.5, 0.3]])
    previous = np.array([[0.0, 0.4, 0.1]])
    current = np.array([[0.0, 0.6, 0.3]])

    later = _build_industry_detail_table(prepared, 2040, data, share, capacity_mt=current, prev_capacity_mt=previous)
    assert later.loc[0, "cost_capital_cny"] == pytest.approx(unit[0, CCS] * 0.2 + unit[0, H2] * 0.2, rel=1e-12)

    first = _build_industry_detail_table(prepared, 2030, data, share, capacity_mt=current)
    assert first.loc[0, "cost_capital_cny"] == pytest.approx(unit[0, CCS] * 0.6 + unit[0, H2] * 0.3, rel=1e-12)
