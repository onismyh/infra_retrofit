"""松弛罚额：模型侧 `model_costs._slack_penalty` 与结果侧 `results_costs._slack_costs` 读同一组常量（2026-10-11 起）。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from coal_retrofit.constants import (
    SLACK_PENALTY_CNY_PER_GJ_BIOMASS,
    SLACK_PENALTY_CNY_PER_KG_H2,
    SLACK_PENALTY_CNY_PER_M3_WATER,
)
from coal_retrofit.optimization.results_costs import _slack_costs
from coal_retrofit.optimization.scenario import OptimizationAssumptions

UNIT_SLACKS = {
    "target_shortfall_mt": 1.0,
    "biomass_slack_gj": np.ones(2),
    "h2_slack_kg": np.ones(3),
    "water_slack_m3": np.ones(1),
    "water_basin_slack_m3": np.ones(2),
    "injectivity_slack_mtpa": np.ones(1),
    "storage_slack_mt": np.ones(1),
    "edge_slack_mtpa": np.ones(1),
}


def test_h2_penalty_is_the_ammonia_penalty_converted_to_hydrogen() -> None:
    # 1000 元/kg NH3 ÷ 0.18 kg H2/kg NH3：钉死数值，常量定义里的除式改了就会被抓到
    assert SLACK_PENALTY_CNY_PER_KG_H2 == pytest.approx(5_555.56, abs=0.01)


def test_results_side_prices_each_slack_with_its_constant() -> None:
    assumptions = OptimizationAssumptions()
    generic = float(assumptions.slack_penalty_cny_per_unit)
    costs = _slack_costs({"slacks": UNIT_SLACKS}, assumptions)  # type: ignore[arg-type]
    assert costs[("slack_penalty", "biomass_supply")] == pytest.approx(2 * SLACK_PENALTY_CNY_PER_GJ_BIOMASS)
    assert costs[("slack_penalty", "green_h2_supply")] == pytest.approx(3 * SLACK_PENALTY_CNY_PER_KG_H2)
    assert costs[("slack_penalty", "water_node")] == pytest.approx(SLACK_PENALTY_CNY_PER_M3_WATER)
    assert costs[("slack_penalty", "water_basin_quota")] == pytest.approx(2 * SLACK_PENALTY_CNY_PER_M3_WATER)
    for item in ("target_shortfall", "storage_injectivity", "storage_capacity", "edge_capacity"):
        assert costs[("slack_penalty", item)] == pytest.approx(generic)


def _model_side(unit_slacks: dict, basin_slack, scales: tuple[float, float, float]) -> float:
    """模型侧罚额：用 numpy 数组冒充 MVar（只用到 `.sum()`），松弛变量取单位值，缩放系数由 *scales* 给。"""
    from coal_retrofit.optimization.model_costs import _slack_penalty

    biomass_scale, ammonia_scale, water_scale = scales
    year_data = SimpleNamespace(
        biomass_flow_scale=biomass_scale, ammonia_flow_scale=ammonia_scale, water_flow_scale=water_scale
    )
    payload = SimpleNamespace(year_data=year_data, **{**unit_slacks, "water_basin_slack_m3": basin_slack})
    return float(_slack_penalty(payload, OptimizationAssumptions()))  # type: ignore[arg-type]


def _results_side(unit_slacks: dict, basin_slack: np.ndarray, scales: tuple[float, float, float]) -> float:
    """结果侧罚额：松弛量已乘回物理单位，所以单位松弛 x 对应缩放。"""
    biomass_scale, ammonia_scale, water_scale = scales
    slacks = {
        **unit_slacks,
        "biomass_slack_gj": unit_slacks["biomass_slack_gj"] * biomass_scale,
        "h2_slack_kg": unit_slacks["h2_slack_kg"] * ammonia_scale,
        "water_slack_m3": unit_slacks["water_slack_m3"] * water_scale,
        "water_basin_slack_m3": basin_slack * water_scale,
    }
    return sum(_slack_costs({"slacks": slacks}, OptimizationAssumptions()).values())  # type: ignore[arg-type]


def test_model_side_total_equals_results_side_total() -> None:
    # `_slack_penalty` 只做算术，没装 gurobipy 也能跑（`_shared.gp` 为 None 时照常导入），不跳过
    # 三种资源的缩放系数各不相同，混用缩放（如氢用了水的系数）会让两侧对不上
    scales = (2.0, 3.0, 5.0)
    basin = np.ones(2)
    assert _model_side(UNIT_SLACKS, basin, scales) == pytest.approx(_results_side(UNIT_SLACKS, basin, scales), rel=1e-12)
    # 流域指标未激活：模型侧为 None，结果侧为零长度数组
    assert _model_side(UNIT_SLACKS, None, scales) == pytest.approx(
        _results_side(UNIT_SLACKS, np.zeros(0), scales), rel=1e-12
    )
