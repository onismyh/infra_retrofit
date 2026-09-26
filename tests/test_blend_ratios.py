"""连续 hub 下的掺烧比例换算与逐厂碳成本。

连续 hub 下一个 hub 可以把不同份额改造到不同档位，`blend_level = Σ l·select` 只是档位下标的加权和：
一半第 1 档、一半第 3 档记作 2，按档位读成 0.25，实际是 0.30。结果表改按约束里的 Σβ_l·z_l 除以
路径份额换算（`results_plant._blend_ratios`）；`blend_level_to_ratio` 只认整数档位；成本表的碳成本
与目标函数同式，用求解器的逐厂减排量。末一条求解 toy，需要 Gurobi，其余不依赖。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.optimization._shared import PATHWAY_INDEX, PreparedInputs
from coal_retrofit.optimization.emissions import blend_level_to_ratio
from coal_retrofit.optimization.results_plant import (
    _blend_ratios,
    _build_pathway_table,
    _build_plant_cost_table,
    _build_plant_detail_table,
)
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.year_types import YearData

SCENARIO = OptimizationScenario(experiment_id="T", description="toy")
LEVELS_B = SCENARIO.biomass_blend_levels  # (0.10, 0.25, 0.50, 0.75, 1.00)
BIO, BECCS, AMM = PATHWAY_INDEX["biomass"], PATHWAY_INDEX["beccs"], PATHWAY_INDEX["ammonia"]


def _three_hubs() -> dict[str, np.ndarray]:
    """三个 hub 的份额与 Σβ·z，档位混合各不相同。

    hub 0：全部生物质，改造容量一半第 1 档（0.10）、一半第 3 档（0.50）。档位下标 2.0，比例 0.30。
    hub 1：生物质 0.4 用第 4 档（0.75），BECCS 0.6 用第 2 档（0.25）。档位下标 2.8，两条路径比例不同。
    hub 2：掺氨 0.5 用第 5 档（0.50），其余未改造。档位下标 2.5，比例 0.50。
    """
    share = np.zeros((3, len(PATHWAYS)))
    share[0, BIO] = 1.0
    share[1, BIO], share[1, BECCS] = 0.4, 0.6
    share[2, AMM], share[2, PATHWAY_INDEX["unabated"]] = 0.5, 0.5
    return {
        "share": share,
        "bio_xs": np.array([0.5 * 0.10 + 0.5 * 0.50, 0.4 * 0.75, 0.0]),
        "beccs_xs": np.array([0.0, 0.6 * 0.25, 0.0]),
        "amm_xs": np.array([0.0, 0.0, 0.5 * 0.50]),
        "level_b": np.array([0.5 * 1 + 0.5 * 3, 0.4 * 4 + 0.6 * 2, 0.0]),
        "level_a": np.array([0.0, 0.0, 0.5 * 5]),
    }


def _prepared(n: int) -> PreparedInputs:
    plants = pd.DataFrame({
        "plant_id": [f"P{i}" for i in range(n)], "province_name": ["Shanxi"] * n,
        "total_capacity_mw": [1000.0] * n, "centroid_longitude": [112.0] * n,
        "centroid_latitude": [37.0] * n, "retirement_year": [2060] * n,
    })
    network = SimpleNamespace(
        edges=pd.DataFrame(columns=["from_node_id", "to_node_id", "length_km"]),
        storage_node_ids={}, plant_node_ids={},
    )
    return cast(PreparedInputs, SimpleNamespace(plants=plants, network=network))


def test_effective_ratio_is_sum_beta_z_over_the_pathway_share() -> None:
    hubs = _three_hubs()
    ratios = _blend_ratios(hubs["share"], hubs["bio_xs"], hubs["beccs_xs"], hubs["amm_xs"])
    np.testing.assert_allclose(ratios["biomass"], [0.30, 0.75, 0.0], rtol=1e-12)
    np.testing.assert_allclose(ratios["beccs"], [0.0, 0.25, 0.0], rtol=1e-12)
    np.testing.assert_allclose(ratios["ammonia"], [0.0, 0.0, 0.50], rtol=1e-12)
    # 份额为零（或在可行性容差之内）的路径记 0，不做 0/0。
    tiny = np.zeros((1, len(PATHWAYS)))
    tiny[0, BIO] = 5e-7
    assert _blend_ratios(tiny, np.array([5e-7]), np.zeros(1), np.zeros(1))["biomass"][0] == 0.0


def test_blend_level_to_ratio_reads_integer_levels_only() -> None:
    assert blend_level_to_ratio(None, LEVELS_B) == 0.0
    assert blend_level_to_ratio(0.0, LEVELS_B) == 0.0
    assert blend_level_to_ratio(2.0, LEVELS_B) == 0.25
    assert blend_level_to_ratio(5.0 + 5e-5, LEVELS_B) == 1.0  # 二元档位的整数容差之内
    # hub 0 的下标恰为整数，按档位只能读成第 2 档：这正是它不能用于连续 hub 的原因。
    assert blend_level_to_ratio(_three_hubs()["level_b"][0], LEVELS_B) == 0.25
    for bad in (2.5, 2.8, 6.0, -1.0):
        with pytest.raises(ValueError, match="blend_ratio"):
            blend_level_to_ratio(bad, LEVELS_B)


def test_detail_and_pathway_tables_use_the_effective_ratios() -> None:
    hubs = _three_hubs()
    n = len(hubs["share"])
    prepared = _prepared(n)
    emissions = np.full(n, 10.0)
    year_data = cast(YearData, SimpleNamespace(generation=np.full(n, 5.0e6), emissions_mt=emissions))
    reduction = np.array([3.0, 8.0, 2.5])
    zeros = np.zeros(n)

    detail = _build_plant_detail_table(
        prepared, SCENARIO, 2040, hubs["share"], zeros, zeros, zeros, zeros,
        hubs["level_b"], hubs["level_a"], np.zeros((n, len(PATHWAYS))),
        year_data=year_data, plant_reduction_mt=reduction,
        biomass_blend_x_share=hubs["bio_xs"], beccs_blend_x_share=hubs["beccs_xs"],
        ammonia_blend_x_share=hubs["amm_xs"],
    )
    np.testing.assert_allclose(detail["biomass_blend_ratio"], [0.30, 0.75, 0.0], rtol=1e-12)
    np.testing.assert_allclose(detail["beccs_blend_ratio"], [0.0, 0.25, 0.0], rtol=1e-12)
    np.testing.assert_allclose(detail["ammonia_blend_ratio"], [0.0, 0.0, 0.50], rtol=1e-12)
    # 档位下标照原样保留（连续 hub 下是加权下标）。
    np.testing.assert_allclose(detail["biomass_blend_level"], hubs["level_b"], rtol=1e-12)

    pathways = _build_pathway_table(
        prepared, SCENARIO, 2040, hubs["share"], zeros,
        hubs["bio_xs"], hubs["beccs_xs"], hubs["amm_xs"],
        year_data=year_data, plant_reduction_mt=reduction,
    )
    hub1 = pathways[pathways["plant_id"] == "P1"].set_index("pathway")["abatement_mt"]
    # 厂合计锚定到求解器的减排量；生物质与 BECCS 按各自的比例拆分：
    # 生物质 0.75 × 0.4 = 0.30，BECCS (0.90 + 0.25) × 0.6 = 0.69。此前把档位下标 2.8 当比例拆，生物质分到 1.12 / 3.34。
    assert hub1.sum() == pytest.approx(8.0, rel=1e-12)
    assert hub1["biomass"] == pytest.approx(8.0 * 0.30 / 0.99, rel=1e-12)
    assert hub1["beccs"] == pytest.approx(8.0 * 0.69 / 0.99, rel=1e-12)


def test_plant_cost_carbon_cost_is_the_objective_expression() -> None:
    """碳成本 = 碳价 × 1e6 × (基线排放 − 求解器逐厂减排量)，与 `model_costs._operating_costs` 同式；
    与份额、掺烧档位无关（`year_data` 里故意没有 `emissions_retrofit_mt`：旧的近似式要用它）。"""
    n = 2
    zeros_path = np.zeros((n, len(PATHWAYS)))
    share = np.zeros((n, len(PATHWAYS)))
    share[0, PATHWAY_INDEX["ccs"]] = 1.0
    share[1, PATHWAY_INDEX["unabated"]] = 1.0

    def table(carbon_price: float) -> pd.DataFrame:
        year_data = cast(YearData, SimpleNamespace(
            emissions_mt=np.array([10.0, 4.0]), carbon_price=carbon_price,
            retrofit_stock_capex=np.zeros((n, 1)), baseline_net_matrix=zeros_path,
            biomass_flow_scale=1.0, coal_savings_per_gj=np.zeros(n), fixed_cost_matrix=zeros_path,
            energy_penalty_matrix=zeros_path, ccs_om_matrix=zeros_path, stranded_per_plant=np.zeros(n),
        ))
        return _build_plant_cost_table(
            _prepared(n), SCENARIO, OptimizationAssumptions(), 2040, year_data, share,
            np.zeros(n), np.zeros(n), np.zeros(n),
            # 第 2 个 hub 的减排为负：惩罚燃料使排放高于基线，碳成本随之高于基线排放的碳价。
            plant_reduction_mt=np.array([7.5, -0.2]),
            retrofit_installed=np.zeros((n, 1)), capex_pathway_indices=(PATHWAY_INDEX["ccs"],),
        )

    priced = table(100.0)
    np.testing.assert_allclose(priced["carbon_cost_cny"], [100.0 * 1e6 * 2.5, 100.0 * 1e6 * 4.2], rtol=1e-12)
    np.testing.assert_allclose(priced["total_plant_cost_cny"], priced["carbon_cost_cny"], rtol=1e-12)
    assert (table(0.0)["carbon_cost_cny"] == 0.0).all()


def test_solved_blend_x_share_is_the_quantity_the_constraints_use(tmp_path) -> None:
    """求解后提取的 Σβ·z 与约束里用的是同一个量。只开放 BECCS、2050 年电力目标为零排放（同
    `test_capex_stock_and_lifetimes` 的 BECCS 用例）：生物质与氨两列为 0；生物质用量约束
    `biomass_use_gj = 热耗 × (G_bio·Σβz_bio + G_beccs·Σβz_beccs)` 成立；有效比例落在档位之内。"""
    pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")
    from test_capex_stock_and_lifetimes import _solve_toy
    from toy_inputs import _write_targets, _write_toy_inputs

    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    bio = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv")
    bio["longitude"], bio["latitude"], bio["province_name"], bio["available_gj"] = 112.05, 37.0, "Shanxi", 1.0e9
    bio.to_csv(paths.inputs_dir / "biomass_supply_curve.csv", index=False)
    _write_targets(paths, {2030: 1.0, 2040: 1.0, 2050: 0.0, 2060: 0.0})
    scenario = OptimizationScenario(
        experiment_id="TEST-BLEND", description="toy", planning_years=(2050, 2060),
        sector_target_source="toy",
        carbon_price_cny_per_t_by_year=(0.0, 0.0),
        electricity_price_cny_per_mwh_by_year=(490.0, 550.0),
        pathway_disable=("retire", "ccs", "biomass", "ammonia"),
        solver_time_limit=300,
    )
    assumptions = OptimizationAssumptions(storage_deployment_fraction_by_year=(1.0, 1.0, 1.0, 1.0))
    y50 = _solve_toy(paths, scenario, assumptions)["year_solutions"][2050]

    bio_xs, beccs_xs = float(y50["biomass_blend_x_share"][0]), float(y50["beccs_blend_x_share"][0])
    assert bio_xs == pytest.approx(0.0, abs=1e-9)
    assert float(y50["ammonia_blend_x_share"][0]) == pytest.approx(0.0, abs=1e-9)
    assert beccs_xs > 0.0
    gen = y50["year_data"].generation_by_pathway[0]
    expected_gj = float(y50["year_data"].heat_rate_eff[0]) * (gen[BIO] * bio_xs + gen[BECCS] * beccs_xs)
    assert float(y50["biomass_use_gj"][0]) == pytest.approx(expected_gj, rel=1e-6)
    ratios = _blend_ratios(
        y50["share"], y50["biomass_blend_x_share"], y50["beccs_blend_x_share"], y50["ammonia_blend_x_share"],
    )
    assert min(LEVELS_B) - 1e-6 <= float(ratios["beccs"][0]) <= max(LEVELS_B) + 1e-6
