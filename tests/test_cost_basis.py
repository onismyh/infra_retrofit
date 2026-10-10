"""煤电与工业同一增量口径（2026-10-10 起）：煤电基线净运行成本进目标的是它与参照"全部维持不改造运行"之差。

参照是未改造列（逐 hub 一个常数），份额和为 1，目标只差一个常数、最优解不变；参照另记 `baseline_reference_cny`。
不需要 Gurobi。
"""
from __future__ import annotations

import numpy as np
import pytest

from coal_retrofit.optimization._shared import PATHWAY_INDEX
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.plant_matrices import _plant_operating_matrices
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from toy_inputs import _write_targets, _write_toy_inputs


def test_coal_operating_cost_is_relative_to_running_unretrofitted(tmp_path) -> None:
    """未改造列为零，退役列为省下的参照，参照 = 发电量 x (毛热耗 x 煤价 + 运维 − 电价)（容量电价缺省 0）。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    _write_targets(paths, {2030: 1.0})
    scenario = OptimizationScenario(experiment_id="T", description="toy", sector_target_source="toy")
    assumptions = OptimizationAssumptions()
    prepared = prepare_inputs(paths, scenario, assumptions)
    m = _plant_operating_matrices(prepared, scenario, assumptions, 2030)
    reference = m["baseline_reference_cny"]
    margin = (
        m["heat_rate_eff"] * m["coal_price_per_plant"] + assumptions.baseline_om_cost_cny_per_mwh
        - scenario.electricity_price_for_year(2030)
    )
    np.testing.assert_allclose(reference, m["generation"] * margin)
    assert np.all(m["baseline_net_matrix"][:, PATHWAY_INDEX["unabated"]] == 0.0)
    np.testing.assert_allclose(m["baseline_net_matrix"][:, PATHWAY_INDEX["retire"]], -reference)
    boost = scenario.retrofit_cf_boost
    assert m["baseline_net_matrix"][0, PATHWAY_INDEX["ccs"]] == pytest.approx((boost - 1.0) * reference[0])
