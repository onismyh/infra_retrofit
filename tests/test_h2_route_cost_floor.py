"""氢路线年度成本的下限（`model_industry.add_industry_year`）：`h2_route_cost` 不低于"年度项 + 买氢"，
变量下界为 0。年度项（`opex_cny[:, H2]`）= 路线固定运维 + 由锚点反推的非氢运行差额，后者为负；
氢价低时"年度项 + 买氢"仍为负，下限 0 起作用，氢路线不会以负成本进目标。

与 `test_capex_stock_and_lifetimes._capex_by_year` 同法，直接建一个规划年的工业块，不经 toy 输入。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

gp = pytest.importorskip("gurobipy", reason="gurobipy is required for solver integration tests")

from coal_retrofit.constants import AMMONIA_FLOW_SCALE  # noqa: E402
from coal_retrofit.optimization.industry import H2, add_industry_year, industry_year_data  # noqa: E402
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario  # noqa: E402
from test_h2_route_multiplier import _steel_hub  # noqa: E402


def _h2_route_cost(price_cny_per_kg: float) -> tuple[float, float]:
    """唯一的长流程钢 hub 只接一条氢链路、氢路线份额钉在 1，最小化氢路线成本。

    返回（求解得到的 `h2_route_cost`，年度项 + 买氢）。
    """
    industry = _steel_hub()
    ydata = industry_year_data(
        industry, OptimizationScenario(experiment_id="T", description="toy"), OptimizationAssumptions(), 2030,
    )
    model = gp.Model()
    model.Params.OutputFlag = 0
    payload = add_industry_year(
        model, industry, ydata, "2030",
        h2_link_cost=np.array([price_cny_per_kg * AMMONIA_FLOW_SCALE]),  # CNY / 缩放后的 kg
        h2_hub_membership=sparse.csr_matrix(np.ones((1, 1))),
    )
    model.addConstr(payload.share[0, H2] == 1.0, name="fix_h2_share")
    model.setObjective(payload.h2_route_cost.sum())
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    purchase = price_cny_per_kg * float(ydata.h2_demand_kg_per_share[0])
    return float(payload.h2_route_cost.X[0]), float(ydata.opex_cny[0, H2]) + purchase


def test_h2_route_cost_floors_at_zero_when_the_annual_term_is_negative() -> None:
    """氢价为 0：年度项为负，氢路线成本停在下限 0，不是负数。"""
    cost, annual = _h2_route_cost(0.0)
    assert annual < 0.0
    assert cost == pytest.approx(0.0, abs=1e-6)


def test_h2_route_cost_equals_the_annual_term_plus_purchase_when_positive() -> None:
    """氢价 30 CNY/kg：年度项加买氢为正，氢路线成本就等于它，下限不起作用。"""
    cost, annual = _h2_route_cost(30.0)
    assert annual > 0.0
    assert cost == pytest.approx(annual, rel=1e-9)
