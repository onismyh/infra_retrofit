"""2026-09-23 的 (e)：氨价里的合成岛 capex 年金在求解时按情景贴现率重算，不再单用 8%。

这些测试不求解，单独成文件以免被 Gurobi 门控：`test_capex_stock_and_lifetimes.py` 在模块级
`importorskip("gurobipy")`，放在那里的测试在没有 Gurobi 的环境里会整体跳过。
"""
from __future__ import annotations

import pandas as pd
import pytest

from coal_retrofit.builders.supply import reprice_hb_capex
from coal_retrofit.optimization.data_prep import _prepare_ammonia_supply
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.paths import ProjectPaths

# 2026-09-23 前构建输入时按 8%、20 年写进 ammonia_supply_curve.csv 的合成岛年金（v7 / v9 / v9.1 输入都是这个数）。
_HB_ANNUITY_AT_8PCT = 0.08912068272025675


def _prepared_ammonia(root, discount_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一个氨节点、一座电厂：按给定情景贴现率读入氨供给曲线，返回节点表与链路表。"""
    paths = ProjectPaths(root=root)
    paths.ensure_inputs_dir()
    pd.DataFrame({
        "ammonia_node_id": ["A1"], "year": [2050], "nh3_supply_kg_per_year": [1.0e9],
        "nh3_hb_capex_usd_per_kg": [_HB_ANNUITY_AT_8PCT], "nh3_cost_lb_usd_per_kg": [1.5],
        "longitude": [112.2], "latitude": [37.0], "province_name": ["Shanxi"],
    }).to_csv(paths.inputs_dir / "ammonia_supply_curve.csv", index=False)
    plants = pd.DataFrame({"plant_id": ["P1"], "centroid_longitude": [112.0], "centroid_latitude": [37.0]})
    scenario = OptimizationScenario(experiment_id="T", description="toy", discount_rate=discount_rate)
    return _prepare_ammonia_supply(paths, scenario, OptimizationAssumptions(), plants)


def test_ammonia_hb_annuity_follows_the_scenario_discount_rate(tmp_path) -> None:
    """CSV 里 8% 的合成岛年金在求解时换成按情景贴现率算的：6% 时每 kg 便宜 0.0128 USD，8% 时不变。
    2026-09-23 前 CSV 里的年金原样进模型，与情景贴现率无关。"""
    usd_to_cny = OptimizationAssumptions().usd_to_cny
    annuity_6pct = 875.0 * 0.06 / (1.0 - 1.06 ** -20) / 1000.0
    assert annuity_6pct == pytest.approx(0.076286, abs=1e-6)

    nodes, links = _prepared_ammonia(tmp_path / "r6", 0.06)
    expected = (1.5 - _HB_ANNUITY_AT_8PCT + annuity_6pct) * usd_to_cny
    assert nodes.loc[0, "nh3_hb_capex_usd_per_kg"] == pytest.approx(annuity_6pct, rel=1e-12)
    assert nodes.loc[0, "cost_cny_per_kg"] == pytest.approx(expected, rel=1e-12)
    assert links.loc[0, "cost_cny_per_kg"] == pytest.approx(expected, rel=1e-12)

    nodes8, _ = _prepared_ammonia(tmp_path / "r8", 0.08)
    assert nodes8.loc[0, "cost_cny_per_kg"] == pytest.approx(1.5 * usd_to_cny, rel=1e-12)


def test_reprice_hb_capex_leaves_its_input_alone_and_passes_through_tables_without_the_column() -> None:
    """不改动传入的表；没有合成岛列的表（toy 测试的输入）不拆分成本，原样返回。"""
    curve = pd.DataFrame({
        "ammonia_node_id": ["A1"], "nh3_hb_capex_usd_per_kg": [_HB_ANNUITY_AT_8PCT], "nh3_cost_lb_usd_per_kg": [1.5],
    })
    repriced = reprice_hb_capex(curve, 0.06)
    assert repriced.loc[0, "nh3_cost_lb_usd_per_kg"] < 1.5
    assert curve.loc[0, "nh3_hb_capex_usd_per_kg"] == _HB_ANNUITY_AT_8PCT
    assert curve.loc[0, "nh3_cost_lb_usd_per_kg"] == 1.5

    bare = curve.drop(columns="nh3_hb_capex_usd_per_kg")
    passed = reprice_hb_capex(bare, 0.06)
    assert passed is not bare
    pd.testing.assert_frame_equal(passed, bare)
