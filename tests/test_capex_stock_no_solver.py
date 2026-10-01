"""2026-09-23 模型改动 (a)–(d) 里不求解的测试，外加工业明细表的 capital 列，以及 2026-09-30 的分代在役判定与
铭牌定规模（求解的分代测试在 `test_capacity_vintages.py`）。

(a)–(d) 那几条从 `test_capex_stock_and_lifetimes.py` 挪来：那个文件在模块级 `importorskip("gurobipy")`，
放在那里的测试在没有 Gurobi 的环境里会整体跳过（见 `test_discount_rate.py` 的说明）。煤电的两条
借 toy 输入建逐年系数矩阵，建矩阵不用 Gurobi。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.constants_industry import (
    INDUSTRY_CCS_FIXED_OM_FRACTION,
    INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION,
    h2_route_capex_cny_per_t_yr,
)
from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.industry import CCS, H2, IndustryInputs, industry_year_data
from coal_retrofit.optimization.industry_inputs import prepare_industry
from coal_retrofit.optimization.results_industry import _build_industry_detail_table
from coal_retrofit.optimization.results_network import _alive_edge_added_stock
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.vintage import alive_vintages
from coal_retrofit.optimization.year_matrices import _build_year_matrices
from coal_retrofit.optimization.year_types import YearData
from test_h2_route_multiplier import _steel_hub
from test_province_names import _cement_hubs
from toy_inputs import _write_targets, _write_toy_inputs


def _cement_hub(output_index: dict[tuple[str, int], float]) -> IndustryInputs:
    """一个水泥 hub（只有 CCS 路线），产量按给定指数逐年缩放；铭牌产能与产量相同。"""
    hubs = pd.DataFrame({
        "hub_id": ["C1"], "sector": ["cement"], "province": ["Shanxi"],
        "longitude": [112.0], "latitude": [37.0],
        "capacity_kt_per_year": [1000.0], "production_kt_per_year": [1000.0], "co2_mt_per_year": [0.8],
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


def test_capacity_vintages_retire_at_the_end_of_their_life() -> None:
    """捕集岛、工业捕集与氢路线能力（2026-09-30）用与管道相同的在役判定：t - 建设年 < 寿命。
    捕集 20 a：2030 年建的到 2050 年退出；氢路线 25 a：2050 年仍在役，到 2060 年退出。"""
    years = (2030, 2040, 2050, 2060)
    assert alive_vintages(years, 0, 20) == [0]
    assert alive_vintages(years, 2, 20) == [1, 2]
    assert alive_vintages(years, 3, 20) == [2, 3]
    assert alive_vintages(years, 2, 25) == [0, 1, 2]
    assert alive_vintages(years, 3, 25) == [1, 2, 3]


# ------------------------------------- (d) 成本乘子只乘 capex 与随 capex 的固定运维 ---
def test_industry_cost_multiplier_leaves_energy_and_consumables_alone() -> None:
    """工业 CCS：乘子 2 使 capex 与固定运维翻倍；蒸汽、电与耗材（`opex_cny` 的 CCS 列）按模型价格计，不乘。
    2026-09-23 前能耗与耗材也翻倍。固定运维 2026-09-30 起单列（每 Mt/yr 能力，按建设年单价计在在役且在用的能力上）。"""
    industry = _cement_hub({("cement", 2030): 1.0})
    assumptions = OptimizationAssumptions()
    base = industry_year_data(industry, OptimizationScenario(experiment_id="T", description="toy"), assumptions, 2030)
    doubled = industry_year_data(
        industry, OptimizationScenario(experiment_id="T", description="toy", industry_cost_multiplier=2.0),
        assumptions, 2030,
    )
    assert doubled.capex_cny_per_mt[0, CCS] == pytest.approx(2.0 * base.capex_cny_per_mt[0, CCS], rel=1e-12)
    assert base.fixed_om_cny_per_mt[0, CCS] == pytest.approx(
        base.capex_cny_per_mt[0, CCS] * INDUSTRY_CCS_FIXED_OM_FRACTION, rel=1e-12
    )
    assert doubled.fixed_om_cny_per_mt[0, CCS] == pytest.approx(2.0 * base.fixed_om_cny_per_mt[0, CCS], rel=1e-12)
    assert base.opex_cny[0, CCS] > 0.0
    assert doubled.opex_cny[0, CCS] == pytest.approx(base.opex_cny[0, CCS], rel=1e-12)


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
    assert base.retrofit_stock_om[0, 0] > 0.0
    assert doubled.retrofit_stock_om[0, 0] == pytest.approx(2.0 * base.retrofit_stock_om[0, 0], rel=1e-12)
    for k in (PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]):
        assert base.ccs_retrofit_capex_matrix[0, k] > 0.0
        assert base.energy_penalty_matrix[0, k] > 0.0
        assert doubled.ccs_retrofit_capex_matrix[0, k] == pytest.approx(2.0 * base.ccs_retrofit_capex_matrix[0, k], rel=1e-12)
        assert doubled.energy_penalty_matrix[0, k] == pytest.approx(base.energy_penalty_matrix[0, k], rel=1e-12)
    assert base.fixed_cost_matrix[0, PATHWAY_INDEX["beccs"]] > 0.0
    np.testing.assert_array_equal(doubled.fixed_cost_matrix, base.fixed_cost_matrix)


def test_beccs_fixed_om_equals_the_capture_island_om_of_ccs(tmp_path) -> None:
    """(b) 的另一半，放在本节是为了借用 `_toy_year_matrices`：BECCS 的捕集岛就是 CCS 捕集岛，capex 与 CCS 相同；
    捕集岛只有一列（在役捕集岛 >= CCS + BECCS 份额，`vintage`），固定运维也就只有一个单价，等于学习后 capex x
    `ccs_om_fraction`。2026-09-23 前按 BECCS 的 4 500 元/kW 计，比 CCS 高 29%。不求解。"""
    matrices = _toy_year_matrices(tmp_path / "m", 2050)
    ccs, beccs = PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]
    capex = matrices.ccs_retrofit_capex_matrix
    assert capex[0, ccs] > 0.0
    assert capex[0, beccs] == pytest.approx(capex[0, ccs], rel=1e-12)
    assert matrices.retrofit_stock_capex.shape == matrices.retrofit_stock_om.shape == (1, 1)
    assert matrices.retrofit_stock_capex[0, 0] == pytest.approx(capex[0, ccs], rel=1e-12)
    fraction = OptimizationAssumptions().ccs_om_fraction
    assert matrices.retrofit_stock_om[0, 0] == pytest.approx(capex[0, ccs] * fraction, rel=1e-12)


def test_industry_capacity_is_sized_by_nameplate_but_never_below_output() -> None:
    """能力按铭牌产能定规模（2026-09-30）：铭牌大于产量按铭牌（这里 x 1.25），小于产量按产量（输入里 49 个 hub
    如此，40 个是长流程钢），相等不变。捕集量、减排、需氢、捕集能耗与耗材都按产量，不受影响；氢路线的固定运维
    按铭牌产能计，非氢运行差额按产量。2026-09-30 前能力与氢路线固定运维都按产量。"""
    hubs = pd.concat([_steel_hub(capacity_kt=c).hubs for c in (1250.0, 800.0, 1000.0)], ignore_index=True)
    hubs["hub_id"] = ["S1", "S2", "S3"]
    industry = IndustryInputs(hubs=hubs, h2_price_cny_per_kg={2030: 20.0})
    data = industry_year_data(industry, OptimizationScenario(experiment_id="T", description="toy"), OptimizationAssumptions(), 2030)
    factor = np.array([1.25, 1.0, 1.0])
    np.testing.assert_allclose(data.capacity_mt_per_share[:, CCS], data.captured_mt[:, CCS] * factor, rtol=1e-12)
    np.testing.assert_allclose(data.capacity_mt_per_share[:, H2], 1.0 * factor, rtol=1e-12)  # 产量 1 Mt/yr
    for same in (data.captured_mt[:, CCS], data.reduction_mt[:, CCS], data.reduction_mt[:, H2],
                 data.h2_demand_kg_per_share, data.opex_cny[:, CCS]):
        np.testing.assert_allclose(same, same[2], rtol=1e-12)
    route_om_per_mt = h2_route_capex_cny_per_t_yr("steel_bf_bof") * 1e6 * INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION
    assert data.opex_cny[0, H2] - data.opex_cny[2, H2] == pytest.approx(0.25 * route_om_per_mt, rel=1e-9)
    assert data.opex_cny[1, H2] == pytest.approx(data.opex_cny[2, H2], rel=1e-12)


@pytest.mark.parametrize(
    ("column", "value"),
    [("capacity_kt_per_year", np.nan), ("capacity_kt_per_year", 0.0), ("production_kt_per_year", 0.0)],
)
def test_prepare_industry_refuses_hubs_without_nameplate_or_output(tmp_path, column: str, value: float) -> None:
    """铭牌系数 = 铭牌 ÷ 现状产量：铭牌或产量缺失、不为正的 hub 读入时就报错，不让 NaN、inf 进模型，也不静默按产量建。"""
    paths = _cement_hubs(tmp_path, ["Shanxi", "Shanxi"])
    hubs = pd.read_csv(paths.inputs_dir / "industry_hubs.csv")
    hubs.loc[1, column] = value
    hubs.to_csv(paths.inputs_dir / "industry_hubs.csv", index=False)
    with pytest.raises(ValueError, match="C1"):
        prepare_industry(paths, OptimizationAssumptions())


# ------------------------------------------------ (a) 明细表的 capital 按新建能力计 ---
def test_industry_detail_table_charges_capital_on_new_capacity() -> None:
    """明细表的 capital 列与目标函数同法：单位 capex x 本年新建能力，CCS 与氢路线各算各的；年度列含求解器给的
    捕集固定运维（在役且在用的能力 x 建设年单价）。2026-09-23 前 `run_context_model` 不传存量，每年都按整个存量计，跨年
    重复计入；2026-09-30 前按单调存量的增量计。"""
    industry = _steel_hub()
    data = industry_year_data(industry, OptimizationScenario(experiment_id="T", description="toy"), OptimizationAssumptions(), 2030)
    unit, opex = data.capex_cny_per_mt, data.opex_cny
    assert unit[0, CCS] > 0.0 and unit[0, H2] > 0.0
    prepared = SimpleNamespace(industry=industry)
    share = np.array([[0.2, 0.5, 0.3]])
    alive = np.array([[0.0, 0.6, 0.3]])
    new = np.array([[0.0, 0.2, 0.3]])
    ccs_om = np.array([7.0e6])

    table = _build_industry_detail_table(
        prepared, 2040, data, share, capacity_mt=alive, new_capacity_mt=new, ccs_fixed_om_cny=ccs_om,
    )
    assert table.loc[0, "cost_capital_cny"] == pytest.approx(unit[0, CCS] * 0.2 + unit[0, H2] * 0.3, rel=1e-12)
    assert table.loc[0, "capacity_ccs_mt"] == pytest.approx(0.6) and table.loc[0, "capacity_h2_mt"] == pytest.approx(0.3)
    annual_h2 = max(0.0, float(opex[0, H2]) * 0.3)  # 没有氢链路，买氢为 0
    assert table.loc[0, "cost_annual_cny"] == pytest.approx(float(opex[0, CCS]) * 0.5 + 7.0e6 + annual_h2, rel=1e-12)
