"""工业氢路线的减排比例（2026-09-23）：长流程钢 0.85 → 0.95。

不求解，单独成文件以免被 Gurobi 门控（见 `test_discount_rate.py` 的说明）。
"""
from __future__ import annotations

import pytest

from coal_retrofit.optimization.industry import H2, industry_year_data
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from test_h2_route_multiplier import _steel_hub


def test_steel_h2_route_removes_95_percent_of_hub_co2() -> None:
    """长流程钢 hub 全部转氢后减排 = hub CO2 × 0.95。

    0.95 由 MPP 钢铁模型 100% 绿氢 DRI-EAF 的直接排放 0.0816 t/t 钢（`mpp_steel`）推得：1 - 0.0816/1.8 = 0.955，
    1.8 是点源表长流程的排放因子；两者按同一口径（直接排放）相比，这一前提未核，见 `constants_industry.py` 的注释。
    2026-09-23 前为 0.85，把 EAF 用的网电排放也算作残余。
    """
    data = industry_year_data(
        _steel_hub(), OptimizationScenario(experiment_id="T", description="toy"), OptimizationAssumptions(), 2030,
    )
    assert data.route_available[0, H2]
    assert data.reduction_mt[0, H2] == pytest.approx(0.95 * 2.0, rel=1e-12)
