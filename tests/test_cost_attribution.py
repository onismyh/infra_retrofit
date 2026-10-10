"""逐实体成本表的对账（`results_costs`，2026-10-10 起）：拆到煤电 hub、工业 hub、管段、封存汇与系统项的成本，按年按类别
相加等于求解器的 `cost_breakdown_cny`，全部相加等于目标函数值。

三个 toy 各覆盖一组成本类别：
- 残值 + 工业：碳价、水泥目标在 2050 年收紧（工业捕集岛 capex 与运行费）、管道固定运维、期末残值；这一例另用九张表
  核对：`sources`、`network`、`sinks` 的成本合计加上系统项等于目标函数值；
- 掺烧 + 重建：除退役外全部路径开放、有碳价、机组分批到期（生物质、氨、掺烧升级、重建 capex、搁浅资产）；
- 水：节点余量低于电厂耗水，超出部分落在节点松弛上（取水成本、松弛惩罚）；另一例给电厂加空冷耗水列，
  缺的水靠空冷改造补上（空冷 capex、背压能耗）。
都要 Gurobi。
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.results_costs import build_costs_table, build_system_table, cost_closure
from coal_retrofit.optimization.results_tables import build_result_tables
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.solver import _solve_joint_multi_period
from coal_retrofit.run_controls import initial_state
from toy_inputs import _write_targets, _write_toy_inputs

pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")


def _check_closure(prepared, solution, scenario, assumptions) -> tuple:
    assert solution["status"] == "optimal"
    costs = build_costs_table(prepared, solution, scenario, assumptions)
    system = build_system_table(solution, costs)
    assert cost_closure(system).max() < 1e-6, system[
        (system["attributed_discounted_cny"] - system["cost_discounted_cny"]).abs() > 1.0
    ]
    breakdown = sum(float(v) for ys in solution["year_solutions"].values() for v in ys["cost_breakdown_cny"].values())
    assert costs["cost_discounted_cny"].sum() == pytest.approx(breakdown, rel=1e-9, abs=1.0)
    assert costs["cost_discounted_cny"].sum() == pytest.approx(solution["objective_cny"], rel=1e-6)
    return costs, system


def test_salvage_industry_and_pipe_om_close(tmp_path) -> None:
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    years = (2030, 2040, 2050)
    _write_targets(paths, dict(zip(years, (1.0, 1.0, 0.5), strict=True)),
                   dict(zip(years, (1.0, 1.0, 0.5), strict=True)))
    scenario = OptimizationScenario(
        experiment_id="TEST-ATTRIBUTION", description="toy", planning_years=years, sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 50.0, 100.0),
        electricity_price_cny_per_mwh_by_year=(400.0, 440.0, 490.0),
        pathway_disable=("retire",), solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(end_of_horizon_salvage=True, pipe_fixed_om_fraction=0.02)
    prepared = prepare_inputs(paths, scenario, assumptions)
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, years, initial_state(prepared))
    costs, _ = _check_closure(prepared, solution, scenario, assumptions)
    # 九张表层面：源、管段、汇各自的成本合计加上系统项（松弛、残值）就是目标函数值；对账检查行全部通过。
    node_province = {str(node): "Shanxi" for node in prepared.network.nodes["node_id"]}
    tables = build_result_tables(prepared, solution, scenario, assumptions, node_province, initial_state(prepared))
    entities = sum(float(tables[name]["cost_discounted_cny"].sum()) for name in ("sources", "network", "sinks"))
    system_items = float(costs.loc[costs["entity_type"] == "system", "cost_discounted_cny"].sum())
    assert entities + system_items == pytest.approx(solution["objective_cny"], rel=1e-6)
    closure = tables["checks"][tables["checks"]["check_name"] == "cost_attribution_closure"]
    assert len(closure) == len(years) and (closure["status"] == "pass").all()
    present = set(zip(costs["entity_type"], costs["category"], strict=True))
    for key in (("industry", "industry_capex"), ("industry", "industry_cost"), ("industry", "carbon_cost"),
                ("coal", "carbon_cost"), ("coal", "ccs_retrofit_capex"), ("edge", "pipe_capex"),
                ("edge", "transport_opex"), ("sink", "storage_cost"), ("system", "salvage_credit")):
        assert key in present, key
    assert (costs["item"] == "pipe_fixed_om").any()


def test_blend_and_rebuild_close(tmp_path) -> None:
    from test_blend_ratios import _solve_blend_toy

    scenario, assumptions, prepared, solution = _solve_blend_toy(
        tmp_path, ("retire",), {2050: 0.45, 2060: 0.2}, carbon=(300.0, 600.0),
        units=((500.0, 2005), (300.0, 2015), (200.0, 2022)),
    )
    costs, _ = _check_closure(prepared, solution, scenario, assumptions)
    assert (costs["category"] == "rebuild_capex").any()
    assert (costs["category"] == "blend_upgrade_capex").any()


@pytest.mark.parametrize("air", [False, True])
def test_water_air_cooling_and_node_slack_close(tmp_path, monkeypatch, air: bool) -> None:
    """节点余量低于电厂耗水。`air` 为真时给电厂加空冷耗水列（取缺省的空冷耗水强度，带捕集的再乘捕集耗水倍率），
    缺的水靠空冷改造补上。"""
    import test_water_constraints as water

    monkeypatch.setattr(water.builders_water, "_assign_basin_codes", water._toy_basin_codes)
    paths = water._toy_paths(tmp_path, basin_caps=True, dry_use_m3=1.0e8)
    if air:
        plants = pd.read_csv(paths.inputs_dir / "plants.csv")
        defaults = OptimizationAssumptions()
        plants["air_consumption_intensity_m3_per_mwh"] = defaults.cooling_air_water_intensity_m3_per_mwh
        plants["air_consumption_ccs_intensity_m3_per_mwh"] = (
            defaults.cooling_air_water_intensity_m3_per_mwh * defaults.ccs_water_multiplier)
        plants.to_csv(paths.inputs_dir / "plants.csv", index=False)
    scenario, prepared, solution = water._solve(paths, "TEST-ATTRIBUTION-WATER", apply_basin_cap=False)
    costs, _ = _check_closure(prepared, solution, scenario, replace(OptimizationAssumptions(), apply_basin_cap=False))
    assert (costs["category"] == "water_cost").any()
    if air:
        assert (costs["category"] == "air_retrofit_capex").any()
        assert (costs["item"] == "air_cooling_backpressure").any()
    else:
        assert ((costs["entity_type"] == "system") & (costs["item"] == "water_node")).any()
