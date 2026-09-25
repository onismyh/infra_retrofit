"""2026-09-23 模型改动 (a)–(d) 里不求解的测试，外加工业明细表的 capital 列。

除明细表那条外，都从 `test_capex_stock_and_lifetimes.py` 挪来：那个文件在模块级 `importorskip("gurobipy")`，
放在那里的测试在没有 Gurobi 的环境里会整体跳过（见 `test_discount_rate.py` 的说明）。煤电的两条
借 toy 输入建逐年系数矩阵，建矩阵不用 Gurobi。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.constants_industry import INDUSTRY_CCS_FIXED_OM_FRACTION
from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.industry import CCS, H2, IndustryInputs, industry_year_data
from coal_retrofit.optimization.results_industry import _build_industry_detail_table
from coal_retrofit.optimization.results_network import _alive_edge_added_stock
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.year_matrices import _build_year_matrices
from coal_retrofit.optimization.year_types import YearData
from test_h2_route_multiplier import _steel_hub
from toy_inputs import _write_targets, _write_toy_inputs


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


def test_beccs_fixed_om_equals_the_capture_island_om_of_ccs(tmp_path) -> None:
    """(b) 的另一半，放在本节是为了借用 `_toy_year_matrices`：BECCS 的捕集岛就是 CCS 捕集岛，固定运维一列
    与 CCS 相同，都等于学习后 capex x `ccs_om_fraction`。2026-09-23 前按 BECCS 的 4 500 元/kW 计，比 CCS 高 29%。
    不求解。"""
    matrices = _toy_year_matrices(tmp_path / "m", 2050)
    ccs, beccs = PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]
    assert matrices.ccs_om_matrix[0, ccs] > 0.0
    assert matrices.ccs_om_matrix[0, beccs] == pytest.approx(matrices.ccs_om_matrix[0, ccs], rel=1e-12)
    fraction = OptimizationAssumptions().ccs_om_fraction
    assert matrices.ccs_om_matrix[0, ccs] == pytest.approx(matrices.ccs_retrofit_capex_matrix[0, ccs] * fraction, rel=1e-12)


# ------------------------------------------------ (a) 明细表的 capital 按能力存量增量计 ---
def test_industry_detail_table_charges_capital_on_the_capacity_increment() -> None:
    """明细表的 capital 列与目标函数同法：单位 capex x 能力存量增量，CCS 与氢路线各算各的；
    不传上一年存量时（第一年）按整个存量计。2026-09-23 前 `run_context_model` 不传存量，
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
