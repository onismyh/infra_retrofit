"""水约束在单厂 toy 上的两条规则：节点上限（环境流量规则，作用于耗水）与流域取水指标（分配规则，作用于取水）。
节点可用量 = max(径流 x 0.20 − 生活与灌溉耗水, 不改造同年耗水)（存量不增）。流域上限逼出空冷改造，结果表的空冷列与
含空冷背压的逐路径拆分也在这里核对。

toy 没有流域面图层，也没有 plants.csv 的取水定额表；两处都在调用函数内部导入，
所以直接替换模块属性即可（`monkeypatch` 在测试结束时还原）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from coal_retrofit.optimization._shared import PreparedInputs
    from coal_retrofit.optimization.year_types import SolveResult

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

import coal_retrofit.builders.water as builders_water
import coal_retrofit.builders.water_quota as builders_water_quota
from coal_retrofit.optimization._shared import SolveState
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.paths import ProjectPaths
from toy_inputs import YEARS, _write_targets, _write_toy_inputs

WATER_SCENARIO_ID = "toy|gcm|ssp126"
# 枯水期径流 2e7 x 可提取比例后节点余量只有 4e6 m3/yr，低于 toy 电厂不改造的耗水（约 8.6e6），节点可用量
# 就是后者；流域余量 6e6 m3/yr 低于它的取水。两条上限都会绑定。
DRY_SEASON_M3 = 2.0e7
BASIN_RESIDUAL_M3 = 6.0e6
UNABATED = PATHWAYS.index("unabated")


def _toy_basin_codes(paths: ProjectPaths, nodes: pd.DataFrame) -> np.ndarray:
    return np.array(["B1"] * len(nodes), dtype=object)


def _toy_withdrawal(plants: pd.DataFrame, generation_mwh: pd.Series):
    """(基线, 带捕集, 空冷基线, 空冷带捕集) 取水强度 m3/MWh 与直流冷却标定系数。"""
    ones = pd.Series(1.0, index=plants.index)
    return ones * 2.0, ones * 3.2, ones * 0.2, ones * 1.1, 1.0


def _write_water_inputs(
    paths: ProjectPaths, basin_caps: bool, dry_season_m3: float = DRY_SEASON_M3, dry_use_m3: float = 0.0
) -> None:
    pd.DataFrame(
        [
            {"water_node_id": "W1", "planning_year": year, "scenario_family": "baseline",
             "scenario_id": WATER_SCENARIO_ID, "basin_code": "B1", "available_water_m3_per_year": 2.0 * dry_season_m3,
             "dry_season_water_m3_per_year": dry_season_m3, "bias_factor": 1.0}
            for year in YEARS
        ]
    ).to_csv(paths.inputs_dir / "water_availability.csv", index=False)
    # 节点余量要扣的生活与灌溉耗水（枯水期列给 `dry_use_m3`，全部记在生活上）。
    pd.DataFrame(
        [
            {"scenario_id": WATER_SCENARIO_ID, "planning_year": year, "basin_code": "B1",
             "domestic_m3_per_year": dry_use_m3, "irrigation_m3_per_year": 0.0,
             "dry_season_domestic_m3_per_year": dry_use_m3, "dry_season_irrigation_m3_per_year": 0.0}
            for year in YEARS
        ]
    ).to_csv(paths.inputs_dir / "water_basin_use.csv", index=False)
    if basin_caps:
        pd.DataFrame(
            [{"basin_code": "B1", "planning_year": year, "residual_m3_per_year": BASIN_RESIDUAL_M3}
             for year in YEARS]
        ).to_csv(paths.inputs_dir / "water_basin_caps.csv", index=False)


def _solve(
    paths: ProjectPaths, experiment_id: str, **assumption_overrides
) -> tuple[OptimizationScenario, PreparedInputs, SolveResult]:
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
    return scenario, prepared, _solve_joint_multi_period(prepared, scenario, assumptions, years, state)


def _toy_paths(tmp_path, basin_caps: bool, **water) -> ProjectPaths:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {year: 0.5 for year in YEARS})
    _write_water_inputs(paths, basin_caps, **water)
    return paths


def _existing_use_m3(year_data) -> float:
    """toy 电厂不改造同年的耗水，节点可用量的存量项。"""
    return float(year_data.generation_by_pathway[0, UNABATED] * year_data.water_intensity[0, UNABATED])


def test_node_limit_keeps_existing_use_and_overdraw_lands_in_node_slack(tmp_path, monkeypatch) -> None:
    """只开生态流量（关掉流域上限）。节点余量（4e6，再扣 1e8 的耗水就为负）低于电厂不改造的耗水，可用量就是
    后者（存量不增）。2050 年要靠 CCS 达标，CCS 多耗的水在 toy 里没法用空冷抵（toy 没有空冷耗水列），超出部分
    只能记在节点松弛上：流经节点的水 = 可用量 + 松弛（缩放单位换回 m3 后仍成立）。厂用水另按耗水强度与解出的
    份额重算核对。流域指标不激活。"""
    monkeypatch.setattr(builders_water, "_assign_basin_codes", _toy_basin_codes)
    paths = _toy_paths(tmp_path, basin_caps=True, dry_use_m3=1.0e8)
    _, _, solution = _solve(paths, "TEST-WATER-NODE", apply_basin_cap=False)
    assert solution["status"] == "optimal"
    slacks = []
    for year in YEARS:
        ys = solution["year_solutions"][year]
        year_data = ys["year_data"]
        assert year_data.water_basin_membership is None
        assert year_data.withdrawal_intensity is None
        assert ys["slacks"]["water_basin_codes"] == []

        available = float(year_data.water_available_m3[0]) * float(year_data.water_flow_scale)
        assert available == pytest.approx(_existing_use_m3(year_data), rel=1e-12)
        through_node = float(np.sum(ys["water_flow_m3"]))
        slack = float(ys["slacks"]["water_slack_m3"][0])
        # 厂侧水平衡：链路供水 = 厂用水。
        assert float(np.sum(ys["water_use_m3"])) == pytest.approx(through_node, rel=1e-9)
        assert through_node <= available + slack + 1e-3
        if slack > 0.0:
            assert through_node == pytest.approx(available + slack, rel=1e-6)
        slacks.append(slack)
        # 上限绑定后，上面几条对任何耗水系数都成立；按解出的份额从耗水强度重算，才核对得到耗水行本身。
        per_path = year_data.water_intensity * ys["share"]
        if year_data.allow_air_cooling_retrofit:
            per_path = per_path - (year_data.water_intensity - year_data.air_water_intensity) * ys["air_share"]
        recomputed = float((year_data.generation_by_pathway * per_path).sum())
        assert float(np.sum(ys["water_use_m3"])) == pytest.approx(recomputed, rel=1e-6)
    assert max(slacks) > 0.0, "前提：CCS 多耗的水超出存量"


def test_node_limit_is_the_residual_when_it_exceeds_existing_use(tmp_path, monkeypatch) -> None:
    """径流大到余量高于不改造耗水时，节点可用量 = 0.20 x 枯水期径流 − 枯水期生活与灌溉耗水。"""
    monkeypatch.setattr(builders_water, "_assign_basin_codes", _toy_basin_codes)
    paths = _toy_paths(tmp_path, basin_caps=True, dry_season_m3=2.0e8, dry_use_m3=1.0e7)
    _, _, solution = _solve(paths, "TEST-WATER-RESIDUAL", apply_basin_cap=False)
    assert solution["status"] == "optimal"
    for year in YEARS:
        year_data = solution["year_solutions"][year]["year_data"]
        available = float(year_data.water_available_m3[0]) * float(year_data.water_flow_scale)
        assert available == pytest.approx(0.20 * 2.0e8 - 1.0e7, rel=1e-12)
        assert available > _existing_use_m3(year_data)


def test_existing_use_cap_offsets_ccs_water_with_air_cooling(tmp_path, monkeypatch) -> None:
    """存量不增：toy 只有一个节点，余量为负时 CCS 多耗的水只能由本厂腾出（空冷，或每期至多 0.15 的自愿退役），否则记松弛。
    本测试核对空冷：给 toy 补上空冷耗水强度（湿冷 1.85、空冷 0.17，带捕集 3.37 与 0.31 m3/MWh）后，CCS 路径转空冷，
    节点流量不超过不改造耗水、没有松弛；同一电厂在余量充足时不转空冷、耗水高于不改造水平，目标值更低。只开生态流量。"""
    monkeypatch.setattr(builders_water, "_assign_basin_codes", _toy_basin_codes)

    def solve(experiment_id: str, **water):
        paths = _toy_paths(tmp_path / experiment_id, basin_caps=True, **water)
        plants = pd.read_csv(paths.inputs_dir / "plants.csv")
        plants["air_consumption_intensity_m3_per_mwh"] = 0.17
        plants["air_consumption_ccs_intensity_m3_per_mwh"] = 0.31
        plants.to_csv(paths.inputs_dir / "plants.csv", index=False)
        _, _, solution = _solve(paths, experiment_id, apply_basin_cap=False)
        assert solution["status"] == "optimal"
        return solution

    ccs = PATHWAYS.index("ccs")
    capped = solve("TEST-WATER-EXISTING", dry_use_m3=1.0e8)
    free = solve("TEST-WATER-FREE", dry_season_m3=2.0e8)
    for year in YEARS:
        ys, ys_free = capped["year_solutions"][year], free["year_solutions"][year]
        existing = _existing_use_m3(ys["year_data"])
        assert float(ys["share"][0, ccs]) > 0.1, "前提：靠 CCS 达标"
        assert float(np.sum(ys["water_flow_m3"])) <= existing * (1 + 1e-9)
        assert float(ys["slacks"]["water_slack_m3"][0]) == pytest.approx(0.0, abs=1e-3)
        assert float(ys["air_share"][0, ccs]) > 1e-3
        assert float(np.abs(ys_free["air_share"]).max()) < 1e-6
        assert float(np.sum(ys_free["water_flow_m3"])) > existing
    assert float(capped["objective_cny"]) > float(free["objective_cny"])


def test_basin_quota_binds_at_the_residual_and_costs_more(tmp_path, monkeypatch) -> None:
    """流域上限打开：流域取水等于余量、流域松弛为零，即上限靠改变路径满足而非松弛；
    流域取水另按取水强度、工业取水与解出的份额重算核对。
    同一输入关掉流域上限（只剩环境流量规则）时流域字段全为空，目标值更低。"""
    monkeypatch.setattr(builders_water, "_assign_basin_codes", _toy_basin_codes)
    monkeypatch.setattr(builders_water_quota, "calibrated_withdrawal_intensities", _toy_withdrawal)
    paths = _toy_paths(tmp_path, basin_caps=True)

    _, _, capped = _solve(paths, "TEST-WATER-QUOTA")
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

    _, _, env_only = _solve(paths, "TEST-WATER-ENVONLY", apply_basin_cap=False)
    assert env_only["status"] == "optimal"
    for year in YEARS:
        year_data = env_only["year_solutions"][year]["year_data"]
        assert year_data.water_basin_membership is None
        assert year_data.water_basin_codes == []
    assert float(capped["objective_cny"]) > float(env_only["objective_cny"])


def test_air_columns_and_pathway_split_with_air_cooling(tmp_path, monkeypatch) -> None:
    """流域上限绑定时 toy 电厂在运行路径上转空冷。明细表的空冷运行份额是运行路径上空冷份额之和，已装份额是空冷存量；
    2060 年退役份额变大，已装存量（只增不减）高于运行份额。逐路径的减排量与捕集量含空冷背压的排放与捕集，逐厂相加
    等于求解器的值。"""
    from coal_retrofit.optimization.results_plant import _build_pathway_table, _build_plant_detail_table

    monkeypatch.setattr(builders_water, "_assign_basin_codes", _toy_basin_codes)
    monkeypatch.setattr(builders_water_quota, "calibrated_withdrawal_intensities", _toy_withdrawal)
    scenario, prepared, solution = _solve(_toy_paths(tmp_path, basin_caps=True), "TEST-WATER-AIR")
    assert solution["status"] == "optimal"
    operating = [k for k, pathway in enumerate(PATHWAYS) if pathway != "retire"]
    for year in YEARS:
        ys = solution["year_solutions"][year]
        air_operating = float(ys["air_share"][0, operating].sum())
        assert air_operating > 1e-3, "前提：流域上限逼出空冷"
        detail = _build_plant_detail_table(
            prepared, scenario, year, ys["share"], ys["captured_mt_by_plant"], ys["biomass_use_gj"],
            ys["ammonia_use_kg"], ys["water_use_m3"], ys["blend_level_b"], ys["blend_level_a"], ys["air_share"],
            year_data=ys["year_data"], plant_reduction_mt=ys["plant_reduction_mt"],
            biomass_blend_x_share=ys["biomass_blend_x_share"], beccs_blend_x_share=ys["beccs_blend_x_share"],
            ammonia_blend_x_share=ys["ammonia_blend_x_share"], air_installed=ys["air_installed"],
        )
        assert float(detail["air_operating_share"].iloc[0]) == pytest.approx(air_operating, rel=1e-12)
        assert float(detail["air_installed_share"].iloc[0]) == pytest.approx(float(ys["air_installed"][0]), rel=1e-12)
        assert float(ys["air_installed"][0]) >= air_operating - 1e-6
        pathways = _build_pathway_table(
            prepared, scenario, year, ys["share"],
            ys["biomass_blend_x_share"], ys["beccs_blend_x_share"], ys["ammonia_blend_x_share"],
            year_data=ys["year_data"], air_share=ys["air_share"],
        )
        assert float(pathways["abatement_mt"].sum()) == pytest.approx(float(ys["plant_reduction_mt"][0]), rel=1e-6)
        assert float(pathways["captured_mt"].sum()) == pytest.approx(float(ys["captured_mt_by_plant"][0]), rel=1e-6)
    last = solution["year_solutions"][YEARS[-1]]
    assert float(last["air_installed"][0]) > float(last["air_share"][0, operating].sum()) + 1e-3, "前提：末年运行份额低于已装存量"
