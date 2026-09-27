"""连续 hub 下路径份额恰好分摊到各掺烧档位上（`Σ_l z = s`；2026-09-27 前是 `≤`，实现说明 §9.8）。

`z[p,l]` 是路径份额里落在档位 l 的部分。连续 hub 下 McCormick 只给 z 上下界，`≤` 允许份额有一部分不落在任何档位上：
改造记在生物质 / BECCS / 掺氨名下，却不掺烧。改成等式后，份额为正就至少按最低档掺烧：Σβ·z ≥ 最低档 × 份额。
三条都需要 Gurobi：两条求解 toy（实现说明 §9.8 记的两种情形），一条只建掺烧档位这组约束，三条路径各测一次。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from coal_retrofit.optimization._shared import PATHWAY_INDEX
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions
from coal_retrofit.optimization.year_types import YearData
from test_blend_ratios import LEVELS_A, LEVELS_B, SCENARIO, _solve_blend_toy

GUROBI = "gurobipy is required for solver integration tests"


def _below_lowest_level(ys) -> dict[str, tuple[float, float]]:
    """份额为正而 Σβ·z 低于"最低档 × 份额"的掺烧路径：{路径: (份额, Σβ·z)}。BECCS 用生物质的档位表。"""
    below = {}
    for pathway, levels in (("biomass", LEVELS_B), ("beccs", LEVELS_B), ("ammonia", LEVELS_A)):
        share = float(ys["share"][0, PATHWAY_INDEX[pathway]])
        x_share = float(ys[f"{pathway}_blend_x_share"][0])
        if x_share < min(levels) * share - 1e-6:
            below[pathway] = (share, x_share)
    return below


# 两条 toy 测试 gap 都取 0：改前的约束下，最优解正是这样的解；gap 放宽后可能停在别的可行解上，测试就拦不住改前的约束。
def test_beccs_without_ccs_blends_at_least_the_lowest_level(tmp_path) -> None:
    """关掉 CCS、BECCS 仍可用，电力上限 0.3：BECCS 份额为正，有效掺烧比例不低于最低档（0.10）。
    改前最优解 BECCS 份额 0.737、有效比例 0.066：约 1/3 的份额没掺生物质，等于借 BECCS 的名义做 CCS。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    _, _, _, solution = _solve_blend_toy(tmp_path, ("retire", "ccs", "ammonia"), {2050: 0.3, 2060: 0.3}, mip_gap=0.0)
    for ys in solution["year_solutions"].values():
        assert float(ys["share"][0, PATHWAY_INDEX["beccs"]]) > 0.0
        assert _below_lowest_level(ys) == {}


def test_cheap_coal_does_not_retrofit_without_blending(tmp_path) -> None:
    """路径全开、山西煤价 10 元/GJ、电力上限 0.8：没有份额为正却不掺烧的掺烧路径。改造路径的发电量乘 1.15
    （`retrofit_cf_boost`），煤价低时多发的电有利可图。改前 2060 年"生物质"份额 0.5、生物质用量为 0，
    多出的排放靠 CCS 份额由 0.229 加到 0.314 抵掉。"""
    pytest.importorskip("gurobipy", reason=GUROBI)
    _, _, _, solution = _solve_blend_toy(tmp_path, (), {2050: 0.8, 2060: 0.8}, coal=10.0, mip_gap=0.0)
    for ys in solution["year_solutions"].values():
        assert _below_lowest_level(ys) == {}


@pytest.mark.parametrize("pathway", ["biomass", "beccs", "ammonia"])
def test_a_positive_share_cannot_skip_blending(pathway) -> None:
    """只建一个 hub 的掺烧档位约束：路径份额固定为 0.5 时可行；再令它的 Σβ·z = 0（改造了不掺烧），就不可行。
    改前 z 全取 0 即可行。掺氨那一处靠这条测：toy 上没找到改前会"改造了不掺氨"的情形，两条 toy 测试拦不住它。"""
    gp = pytest.importorskip("gurobipy", reason=GUROBI)
    from coal_retrofit.optimization.constraints import _add_blend_level_constraints

    year_data = cast(YearData, SimpleNamespace(
        emissions_retrofit_mt=np.ones(1), heat_rate_eff=np.full(1, 9.0),
        generation_by_pathway=np.ones((1, len(PATHWAYS))),
        biomass_penalty_coeff_per_level=0.0, biomass_penalty_emissions_coeff_per_level=0.0,
        beccs_penalty_emissions_coeff_per_level=0.0, beccs_penalty_captured_coeff_per_level=0.0,
    ))
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.DualReductions = 0  # 不可行时报 INFEASIBLE，不报 INF_OR_UNBD
    share = model.addMVar((1, len(PATHWAYS)), lb=0.0, ub=1.0)
    model.addConstr(share[0, PATHWAY_INDEX[pathway]] == 0.5)
    blocks = _add_blend_level_constraints(
        model, share, 1, SCENARIO, OptimizationAssumptions(hub_decisions_continuous=True), year_data, "",
    )
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    x_share = dict(zip(("biomass", "beccs", "ammonia"), blocks[-3:]))[pathway][0]
    model.addConstr(x_share == 0.0)
    model.optimize()
    assert model.Status == gp.GRB.INFEASIBLE
