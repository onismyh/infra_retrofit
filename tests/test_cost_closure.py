"""成本对账 `results_costs.cost_closure` 的容差，不求解：绝对差不到 1 元的类别记 0（2026-10-14 起），其余按相对差报。"""
from __future__ import annotations

import pandas as pd
import pytest

from coal_retrofit.optimization.results_costs import CLOSURE_ABS_TOL_CNY, cost_closure
from coal_retrofit.optimization.results_tables import CLOSURE_TOL, _closure_checks


def _system(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    """(年, 求解器的折现合计, 拆到实体的折现合计) 各一行，类别名不影响对账。"""
    return pd.DataFrame(
        [{"year": year, "category": f"c{i}", "cost_discounted_cny": solver, "attributed_discounted_cny": attributed}
         for i, (year, solver, attributed) in enumerate(rows)]
    )


def test_sub_yuan_rounding_on_a_near_zero_category_passes() -> None:
    # 分段 14 的 ST_BASE：2060 年搁浅资产求解器合计 −1.4e-5 元、拆到 hub 的合计 −6e-7 元；此前分母封底 1 元，相对差 1.4e-5 > 1e-6 判不过
    system = _system([(2060, -1.42e-5, -6.2e-7), (2050, 1e12, 1e12 + 0.5), (2060, 0.0, 0.99)])
    assert CLOSURE_ABS_TOL_CNY == 1.0
    assert cost_closure(system).tolist() == [0.0, 0.0]
    assert (_closure_checks(system)["status"] == "pass").all()


def test_gaps_of_a_yuan_or_more_are_reported_relative_to_the_solver_total() -> None:
    system = _system([(2050, 1e3, 1e3 + 1.0), (2060, 0.0, 2.0), (2060, 1e12, 1e12 + 3.0)])
    gaps = cost_closure(system)
    assert gaps.loc[2050] == pytest.approx(1e-3)
    assert gaps.loc[2060] == pytest.approx(2.0)   # 合计为零的类别分母封底 1 元
    checks = _closure_checks(system)
    assert checks["status"].tolist() == ["fail", "fail"] and (checks["threshold"] == CLOSURE_TOL).all()
