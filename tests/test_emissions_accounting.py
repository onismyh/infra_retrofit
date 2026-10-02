"""路径成本缺省值；逐路径拆分与闭合检查的两条补测（不需要 Gurobi）。

拆分的手算用例与求解器对拍在 `test_blend_ratios.py`。那边的 toy 设计退役年都是 9999（E_op = E_p），也没有
拆分对不上的结果，这里补上：未改造路径按到期原址重建后的运行排放 E_op 计；`pathway_split_closure` 行在拆分
与求解器对不上时记 warn。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from coal_retrofit.optimization._shared import PATHWAY_INDEX
from coal_retrofit.optimization.results import _build_sanity_checks
from coal_retrofit.optimization.results_plant import _build_pathway_table
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions
from coal_retrofit.optimization.year_types import YearData
from test_blend_ratios import SCENARIO, _prepared


def test_pathway_and_retirement_cost_defaults_are_publication_ready() -> None:
    assumptions = OptimizationAssumptions()

    assert PATHWAYS[0] == "unabated"
    assert assumptions.fixed_cost_cny_per_mwh("unabated") == 0.0
    assert assumptions.fixed_cost_cny_per_mwh("retire") > 0.0


def test_pathway_split_unabated_uses_the_operating_emissions() -> None:
    """到期原址重建后效率比 0.9（E_op = 9 Mt，E_p = 10 Mt）：未改造 0.6 按 E_op 排放，少排的 (E_p − E_op) × 0.6 = 0.6
    记在 unabated 上；退役 0.4 减 E_p × 0.4 = 4.0。未改造路径若按 E_p 算，unabated 一栏为零。"""
    n, k = 1, len(PATHWAYS)
    share = np.zeros((n, k))
    share[0, PATHWAY_INDEX["unabated"]], share[0, PATHWAY_INDEX["retire"]] = 0.6, 0.4
    zeros_k = np.zeros((n, k))
    year_data = cast(YearData, SimpleNamespace(
        generation=np.full(n, 5.0e6), emissions_mt=np.full(n, 10.0), emissions_operating_mt=np.full(n, 9.0),
        emissions_retrofit_mt=np.full(n, 10.35), generation_by_pathway=np.full((n, k), 5.0e6),
        ccs_penalty_emissions_matrix=zeros_k, ccs_penalty_captured_matrix=zeros_k,
        air_penalty_emissions_matrix=zeros_k, air_penalty_captured_matrix=zeros_k,
        biomass_penalty_emissions_coeff_per_level=np.zeros(n), beccs_penalty_emissions_coeff_per_level=np.zeros(n),
        beccs_penalty_captured_coeff_per_level=np.zeros(n), rebuilt_deltas=(),
    ))
    no_rebuilt = np.zeros((0, n, k))
    table = _build_pathway_table(
        _prepared(n), SCENARIO, 2050, share, np.zeros(n), np.zeros(n), np.zeros(n),
        year_data=year_data, air_share=zeros_k, rebuilt_share=no_rebuilt, rebuilt_blend_x_share=no_rebuilt,
    ).set_index("pathway")
    assert table.loc["unabated", "abatement_mt"] == pytest.approx(0.6, rel=1e-12)
    assert table.loc["retire", "abatement_mt"] == pytest.approx(4.0, rel=1e-12)


SLACKS = {
    "target_shortfall_mt": 0.0, "target_shortfall_by_group": {}, "biomass_slack_gj": np.zeros(0),
    "ammonia_slack_kg": np.zeros(0), "water_slack_m3": np.zeros(0), "injectivity_slack_mtpa": np.zeros(0),
    "storage_slack_mt": np.zeros(0), "edge_slack_mtpa": np.zeros(0),
}


@pytest.mark.parametrize(("reduction", "captured", "expected"), [
    ((4.0, 0.0), (4.5, 0.0), "pass"),
    ((4.0 + 2e-4, 0.0), (4.5, 0.0), "warn"),   # 2e-4 ÷ 10 Mt = 2e-5 > 1e-5
    ((4.0 + 5e-5, 0.0), (4.5, 0.0), "pass"),   # 5e-5 ÷ 10 Mt = 5e-6 < 1e-5（不除以基线排放就是 5e-5，记 warn）
    ((4.0, 0.0), (4.5 - 2e-4, 0.0), "warn"),   # 只差在捕集一侧
    ((4.0, 5e-6), (4.5, 0.0), "pass"),         # 0.2 Mt 的 hub 按 1 Mt 计：5e-6 < 1e-5
])
def test_split_closure_flags_a_split_that_misses_the_solver(reduction, captured, expected) -> None:
    """拆分逐厂相加与求解器的逐厂减排、捕集对不上（差额除以该厂基线排放，不足 1 Mt 按 1 Mt）超过 1e-5 记 warn，
    减排、捕集两侧都查。"""
    pathways = pd.DataFrame({
        "plant_id": ["P0", "P0", "P1"], "province_name": "Shanxi", "pathway": ["ccs", "unabated", "unabated"],
        "annual_generation_mwh": 1.0, "baseline_emissions_mt": [5.0, 5.0, 0.2],
        "abatement_mt": [4.0, 0.0, 0.0], "captured_mt": [4.5, 0.0, 0.0],
    })
    checks = _build_sanity_checks(
        2050, SLACKS, pathways, pathways, plant_reduction_mt=np.array(reduction), captured_mt=np.array(captured),
    ).set_index("check_name")
    assert checks.loc["pathway_split_closure", "status"] == expected
