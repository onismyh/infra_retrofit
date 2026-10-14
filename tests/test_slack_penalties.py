"""松弛罚额：模型侧 `model_costs._slack_penalty` 与结果侧 `results_costs._slack_costs` 读同一组常量（2026-10-11 起）；
松弛开关 `OptimizationAssumptions.allow_slack`（2026-10-14 起，缺省关）管全部松弛与缺口变量的上界。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from coal_retrofit.constants import (
    SLACK_PENALTY_CNY_PER_GJ_BIOMASS,
    SLACK_PENALTY_CNY_PER_KG_H2,
    SLACK_PENALTY_CNY_PER_M3_WATER,
)
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.results_costs import _slack_costs
from coal_retrofit.optimization.assumptions import OptimizationAssumptions
from coal_retrofit.optimization.scenario import OptimizationScenario
from coal_retrofit.run_controls import initial_state
from toy_inputs import _write_targets, _write_toy_inputs

GUROBI = "gurobipy is required for solver integration tests"
# 年块里的松弛与缺口变量（名字去掉年份与下标），按部门目标的入口分；流域取水指标的松弛只在有水约束且开了流域上限时建，
# toy 没有，在 `test_water_constraints` 里查。
_RESOURCE_SLACKS = {"biomass_slack_gj", "h2_slack_kg", "water_slack_m3", "injectivity_slack_mtpa", "storage_slack_mt",
                    "edge_slack_mtpa"}
SLACK_STEMS = {
    "sector": {"target_shortfall_mt", "target_shortfall_power", "target_shortfall_cement", *_RESOURCE_SLACKS},
    "total": {"target_shortfall_mt", "target_shortfall_all", *_RESOURCE_SLACKS},
}

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


def test_slack_switch_defaults_off() -> None:
    assert OptimizationAssumptions().allow_slack is False


@pytest.mark.parametrize("mode", ["sector", "total"])
@pytest.mark.parametrize("allow_slack", [False, True])
def test_slack_switch_bounds_every_slack_variable(tmp_path, monkeypatch, allow_slack: bool, mode: str) -> None:
    """只开 CCS、电力与水泥组上限都为 0：捕集降不到零排放（合计入口下水泥的额度也抵不了电力）。开关关时每个松弛与缺口
    变量上界为 0，模型不可行；开时不设上界，缺口按罚额进目标、解最优。部门目标的两个入口（分部门、合计）各跑一遍。"""
    gp = pytest.importorskip("gurobipy", reason=GUROBI)
    from coal_retrofit.optimization import solver as solver_module

    models = []
    add_vintages = solver_module.add_capacity_vintages

    def _spy(model, *args, **kwargs):
        models.append(model)
        return add_vintages(model, *args, **kwargs)

    monkeypatch.setattr(solver_module, "add_capacity_vintages", _spy)
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 0.0, 2040: 0.0}, cement_caps={2030: 0.0, 2040: 0.0})
    years = (2030, 2040)
    scenario = OptimizationScenario(
        experiment_id="TEST-SLACK-SWITCH", description="toy", planning_years=years, sector_target_source="toy",
        sector_target_mode=mode, carbon_price_cny_per_t_by_year=(0.0, 0.0), electricity_price_cny_per_mwh_by_year=(400.0, 440.0),
        pathway_disable=("retire", "biomass", "beccs", "ammonia"), solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(allow_slack=allow_slack)
    prepared = prepare_inputs(paths, scenario, assumptions)
    solution = solver_module._solve_joint_multi_period(prepared, scenario, assumptions, years, initial_state(prepared))

    (model,) = models
    slack_vars = [v for v in model.getVars() if "slack" in v.VarName or "shortfall" in v.VarName]
    stems = {v.VarName.split("[")[0].rsplit("_", 1)[0] for v in slack_vars}
    assert stems == SLACK_STEMS[mode]
    upper = {float(v.UB) for v in slack_vars}
    if not allow_slack:
        assert upper == {0.0}
        assert solution["status"] in ("infeasible", "infeasible_or_unbounded")
        return
    assert min(upper) >= gp.GRB.INFINITY   # 读回来是 inf
    assert solution["status"] == "optimal"
    for year in years:
        assert float(solution["year_solutions"][year]["slacks"]["target_shortfall_mt"]) > 0.0


@pytest.mark.parametrize(
    ("status", "allow_slack", "expected"),
    [
        ("infeasible", False, "--set assumptions.allow_slack=true"),
        ("infeasible_or_unbounded", False, "--set assumptions.allow_slack=true"),
        ("infeasible", True, "已开仍不可行"),
        ("optimal", False, ""),
        ("time_limit", True, ""),
        (None, False, ""),
    ],
)
def test_infeasible_hint_depends_on_status_and_switch(status, allow_slack: bool, expected: str) -> None:
    from coal_retrofit.cli import _infeasible_hint

    hint = _infeasible_hint(status, allow_slack)
    assert (expected in hint) if expected else hint == ""
    if allow_slack:
        assert "--set assumptions.allow_slack=true" not in hint   # 已开着就不再让人去开
