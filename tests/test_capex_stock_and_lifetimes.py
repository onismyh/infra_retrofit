"""一次性 capex 的计费基数与资产寿命（2026-09-23 的模型改动），每项一组 toy 回归测试。

(a) 工业 capex 计在能力存量的增量上，不计在路线份额的增量上。
(b) BECCS 的捕集岛按 CCS 计价，生物质改造只走掺烧档位 capex，各收一次。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.optimization._shared import PATHWAY_INDEX, SolveState  # noqa: E402
from coal_retrofit.optimization.data_prep import prepare_inputs  # noqa: E402
from coal_retrofit.optimization.industry import (  # noqa: E402
    CCS,
    IndustryInputs,
    add_industry_monotonicity,
    add_industry_year,
    industry_capex_expr,
    industry_year_data,
)
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario  # noqa: E402
from coal_retrofit.optimization.solver import _solve_joint_multi_period  # noqa: E402
from test_multiperiod_investment_logic import _write_targets, _write_toy_inputs  # noqa: E402


# ---------------------------------------------------------- (a) 工业 capex 按能力存量计 ---
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


def _capex_by_year(industry: IndustryInputs, ccs_share: dict[int, float]) -> tuple[dict[int, float], dict]:
    """固定各年 CCS 份额，最小化一次性 capex 之和，返回每年的 capex 与逐年系数。"""
    scenario = OptimizationScenario(experiment_id="T", description="toy")
    assumptions = OptimizationAssumptions()
    model = gp.Model()
    model.Params.OutputFlag = 0
    years = sorted(ccs_share)
    payloads, data = [], {}
    for year in years:
        data[year] = industry_year_data(industry, scenario, assumptions, year)
        payload = add_industry_year(model, industry, data[year], str(year))
        model.addConstr(payload.share[0, CCS] == ccs_share[year], name=f"fix_ccs_{year}")
        payloads.append(payload)
    add_industry_monotonicity(model, payloads, 1)
    exprs = [
        industry_capex_expr(payload, payloads[i - 1] if i else None, routes=(CCS,))
        for i, payload in enumerate(payloads)
    ]
    model.setObjective(gp.quicksum(exprs))
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    values = {year: float(expr.getValue()) for year, expr in zip(years, exprs)}
    return values, data


def test_industry_capex_charges_output_growth_at_a_constant_share() -> None:
    """份额两年都是 1，产量 1.0 -> 1.5：多出的 50% 捕集能力要付 capex。
    按份额增量计时 2040 年的增量为 0，这部分能力从来不付钱。"""
    industry = _cement_hub({("cement", 2030): 1.0, ("cement", 2040): 1.5})
    capex, data = _capex_by_year(industry, {2030: 1.0, 2040: 1.0})
    v30 = data[2030].capacity_mt_per_share[0, CCS]
    v40 = data[2040].capacity_mt_per_share[0, CCS]
    assert v40 == pytest.approx(1.5 * v30, rel=1e-12)
    assert capex[2030] == pytest.approx(data[2030].capex_cny_per_mt[0, CCS] * v30, rel=1e-9)
    assert capex[2040] == pytest.approx(data[2040].capex_cny_per_mt[0, CCS] * (v40 - v30), rel=1e-9)
    assert capex[2040] > 0.0


def test_industry_capex_not_recharged_on_idle_capacity_of_a_declining_sector() -> None:
    """产量 1.0 -> 0.5，份额 0.5 -> 0.8：2040 年只需 0.8 x 0.5 = 0.40 份 2030 年能力，
    2030 年已建 0.50 份，不必新建。按份额增量计会再收 0.3 x 0.5 份的钱。"""
    industry = _cement_hub({("cement", 2030): 1.0, ("cement", 2040): 0.5})
    capex, data = _capex_by_year(industry, {2030: 0.5, 2040: 0.8})
    v30 = data[2030].capacity_mt_per_share[0, CCS]
    assert capex[2030] == pytest.approx(data[2030].capex_cny_per_mt[0, CCS] * 0.5 * v30, rel=1e-9)
    assert capex[2040] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------- (b) BECCS 只收一次生物质改造费 ---
def _solve_toy(paths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> dict:
    prepared = prepare_inputs(paths, scenario, assumptions)
    state = SolveState(
        edge_added_stock_mtpa=np.zeros(len(prepared.network.edges), dtype=np.float64),
        remaining_storage_mt=prepared.storages["available_capacity_mt"].astype(float).to_numpy(),
    )
    solution = _solve_joint_multi_period(prepared, scenario, assumptions, scenario.planning_years, state)
    assert solution["status"] == "optimal"
    return solution


def test_beccs_pays_the_capture_island_once_and_the_biomass_conversion_once(tmp_path) -> None:
    """只开放 BECCS，2050 年电力目标为零排放：必须掺生物质。捕集岛按 CCS capex 计一次，
    生物质改造按掺烧档位 capex 计一次；2026-09-23 前捕集岛之上还要再付 +1 000 CNY/kW。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    bio = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv")
    bio["longitude"], bio["latitude"], bio["province_name"], bio["available_gj"] = 112.05, 37.0, "Shanxi", 1.0e9
    bio.to_csv(paths.inputs_dir / "biomass_supply_curve.csv", index=False)
    _write_targets(paths, {2030: 1.0, 2040: 1.0, 2050: 0.0, 2060: 0.0})
    scenario = OptimizationScenario(
        experiment_id="TEST-BECCS", description="toy", planning_years=(2050, 2060),
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=("retire", "ccs", "biomass", "ammonia"),
        solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0))
    y50 = _solve_toy(paths, scenario, assumptions)["year_solutions"][2050]
    capacity_mw = 1000.0
    beccs_share = float(y50["share"][0, PATHWAY_INDEX["beccs"]])
    blend_level = float(y50["blend_level_b"][0])
    assert beccs_share > 0.5 and blend_level >= 1.0 - 1e-6
    assert y50["retrofit_installed"].shape == (1, 1)
    assert float(y50["retrofit_installed"][0, 0]) == pytest.approx(beccs_share, rel=1e-6)
    df = 1.0 / (1.0 + scenario.discount_rate) ** (2050 - scenario.discount_base_year)
    island = assumptions.ccs_retrofit_capex_cny_per_kw * 1000.0 * capacity_mw * assumptions.ccs_learning_factor(2050)
    costs = y50["cost_breakdown_cny"]
    assert costs["ccs_retrofit_capex"] == pytest.approx(island * beccs_share * df, rel=1e-6)
    blend = assumptions.biomass_upgrade_capex_cny_per_mw_per_level * capacity_mw * blend_level
    assert costs["blend_upgrade_capex"] == pytest.approx(blend * df, rel=1e-6)
    assert not hasattr(assumptions, "beccs_retrofit_capex_cny_per_kw")
