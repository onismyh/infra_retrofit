"""水约束在单厂 toy 上的两条规则：节点上限（环境流量规则，作用于耗水）与流域取水指标（分配规则，作用于取水）。

toy 没有流域面图层，也没有 plants.csv 的取水定额表；两处都在调用函数内部导入，
所以直接替换模块属性即可（`monkeypatch` 在测试结束时还原）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

import coal_retrofit.builders.water as builders_water
import coal_retrofit.builders.water_quota as builders_water_quota
from coal_retrofit.optimization._shared import SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.paths import ProjectPaths
from toy_inputs import YEARS, _write_targets, _write_toy_inputs

WATER_SCENARIO_ID = "toy|gcm|ssp126"
# 枯水期径流 2e7 x 可提取比例后节点只剩 4e6 m3/yr，低于 toy 电厂的最小耗水；
# 流域余量 6e6 m3/yr 低于它的取水。两条上限都会绑定。
DRY_SEASON_M3 = 2.0e7
BASIN_RESIDUAL_M3 = 6.0e6


def _toy_basin_codes(paths: ProjectPaths, nodes: pd.DataFrame) -> np.ndarray:
    return np.array(["B1"] * len(nodes), dtype=object)


def _toy_withdrawal(plants: pd.DataFrame, generation_mwh: pd.Series):
    """(基线, 带捕集, 空冷基线, 空冷带捕集) 取水强度 m3/MWh 与直流冷却标定系数。"""
    ones = pd.Series(1.0, index=plants.index)
    return ones * 2.0, ones * 3.2, ones * 0.2, ones * 1.1, 1.0


def _write_water_inputs(paths: ProjectPaths, basin_caps: bool) -> None:
    pd.DataFrame(
        [
            {"water_node_id": "W1", "planning_year": year, "scenario_family": "baseline",
             "scenario_id": WATER_SCENARIO_ID, "available_water_m3_per_year": 2.0 * DRY_SEASON_M3,
             "dry_season_water_m3_per_year": DRY_SEASON_M3, "bias_factor": 1.0}
            for year in YEARS
        ]
    ).to_csv(paths.inputs_dir / "water_availability.csv", index=False)
    if basin_caps:
        pd.DataFrame(
            [{"basin_code": "B1", "planning_year": year, "residual_m3_per_year": BASIN_RESIDUAL_M3}
             for year in YEARS]
        ).to_csv(paths.inputs_dir / "water_basin_caps.csv", index=False)


def _solve(paths: ProjectPaths, experiment_id: str, **assumption_overrides) -> dict[str, object]:
    scenario = OptimizationScenario(
        experiment_id=experiment_id,
        description="toy",
        planning_years=YEARS,
        sector_target_source="toy",
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        water_mode="grid_supply",
        water_scenario_id=WATER_SCENARIO_ID,
        water_season="dry",
        solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(
        storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0), **assumption_overrides
    )
    prepared = prepare_inputs(paths, scenario, assumptions)
    years = scenario.planning_years
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    return _solve_joint_multi_period(prepared, scenario, assumptions, years, state)


def _toy_paths(tmp_path, basin_caps: bool) -> ProjectPaths:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {year: 0.5 for year in YEARS})
    _write_water_inputs(paths, basin_caps)
    return paths


def test_node_limit_binds_and_overdraw_lands_in_node_slack(tmp_path, monkeypatch) -> None:
    """只开生态流量（关掉流域上限）：节点可用量低于电厂最小耗水，超出部分只能记在节点松弛上，
    所以流经节点的水 = 可用量 + 松弛（缩放单位换回 m3 后仍成立）；厂用水另按耗水强度与解出的份额重算核对。
    流域指标不激活。"""
    monkeypatch.setattr(builders_water, "_assign_basin_codes", _toy_basin_codes)
    solution = _solve(_toy_paths(tmp_path, basin_caps=True), "TEST-WATER-NODE", apply_basin_cap=False)
    assert solution["status"] == "optimal"
    for year in YEARS:
        ys = solution["year_solutions"][year]
        year_data = ys["year_data"]
        assert year_data.water_basin_membership is None
        assert year_data.withdrawal_intensity is None
        assert ys["slacks"]["water_basin_codes"] == []

        available = float(year_data.water_available_m3[0]) * float(year_data.water_flow_scale)
        through_node = float(np.sum(ys["water_flow_m3"]))
        slack = float(ys["slacks"]["water_slack_m3"][0])
        # 厂侧水平衡：链路供水 = 厂用水。
        assert float(np.sum(ys["water_use_m3"])) == pytest.approx(through_node, rel=1e-9)
        assert slack > 0.0
        assert through_node == pytest.approx(available + slack, rel=1e-6)
        # 上限绑定后，上面几条对任何耗水系数都成立；按解出的份额从耗水强度重算，才核对得到耗水行本身。
        per_path = year_data.water_intensity * ys["share"]
        if year_data.allow_air_cooling_retrofit:
            per_path = per_path - (year_data.water_intensity - year_data.air_water_intensity) * ys["air_share"]
        recomputed = float((year_data.generation_by_pathway * per_path).sum())
        assert float(np.sum(ys["water_use_m3"])) == pytest.approx(recomputed, rel=1e-6)


def test_basin_quota_binds_at_the_residual_and_costs_more(tmp_path, monkeypatch) -> None:
    """流域上限打开：流域取水等于余量、流域松弛为零，即上限靠改变路径满足而非松弛；
    流域取水另按取水强度、工业取水与解出的份额重算核对。
    同一输入关掉流域上限（只剩环境流量规则）时流域字段全为空，目标值更低。"""
    monkeypatch.setattr(builders_water, "_assign_basin_codes", _toy_basin_codes)
    monkeypatch.setattr(builders_water_quota, "calibrated_withdrawal_intensities", _toy_withdrawal)
    paths = _toy_paths(tmp_path, basin_caps=True)

    capped = _solve(paths, "TEST-WATER-QUOTA")
    assert capped["status"] == "optimal"
    for year in YEARS:
        ys = capped["year_solutions"][year]
        year_data = ys["year_data"]
        assert year_data.water_basin_codes == ["B1"]
        assert year_data.withdrawal_intensity is not None
        assert year_data.industry_basin_membership is not None
        assert ys["slacks"]["water_basin_codes"] == ["B1"]
        assert float(ys["slacks"]["water_basin_use_m3"][0]) == pytest.approx(BASIN_RESIDUAL_M3, rel=1e-6)
        assert float(ys["slacks"]["water_basin_slack_m3"][0]) == pytest.approx(0.0, abs=1e-3)
        # 只看绑定的话，漏掉工业项或错用耗水强度的模型同样能过。按解出的份额重算流域取水：
        # 煤电走取水强度（空冷部分按空冷取水强度），加上流域内工业各路线的取水。
        w = year_data.withdrawal_intensity
        per_path = w * ys["share"]
        if year_data.allow_air_cooling_retrofit:
            per_path = per_path - (w - year_data.air_withdrawal_intensity) * ys["air_share"]
        coal = float(year_data.water_basin_membership[0] @ (year_data.generation_by_pathway * per_path).sum(axis=1))
        industry_by_hub = (np.asarray(year_data.industry.water_m3) * ys["industry_share"]).sum(axis=1)
        industry = float(year_data.industry_basin_membership[0] @ industry_by_hub)
        assert industry > 0.0
        assert float(ys["slacks"]["water_basin_use_m3"][0]) == pytest.approx(coal + industry, rel=1e-6)

    env_only = _solve(paths, "TEST-WATER-ENVONLY", apply_basin_cap=False)
    assert env_only["status"] == "optimal"
    for year in YEARS:
        year_data = env_only["year_solutions"][year]["year_data"]
        assert year_data.water_basin_membership is None
        assert year_data.water_basin_codes == []
    assert float(capped["objective_cny"]) > float(env_only["objective_cny"])
