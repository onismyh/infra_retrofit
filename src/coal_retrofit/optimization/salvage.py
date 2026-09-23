"""一次性 capex 的期末残值抵扣（作者决定 2026-09-22）。

目标函数一次性计入的每一笔 capex——煤电捕集岛、掺烧升级、空冷改造、管道、原址重建、
工业捕集与氢路线——买到的都是有经济寿命的资产。在止于 2070 年的规划期里，2060 年建成的
改造到模型不再往后看时，只用掉了 30 年寿命的三分之一；计入其全部 capex 会让最后一期投资
不足，改用平准化又会带回作者已否决的资本回收假设。修正采用标准做法：在经济寿命内直线
折旧，未折旧的余值在规划期末抵回，并从期末折现。

    credit = df(T_end) * sum_t sum_items  max(0, 1 - (T_end - t) / L_item) * capex_item(t)

其中 `T_end = last planning year + its interval`（在 2030/40/50/60 网格上为 2070）。抵扣额
永远不会超过它所对应的 capex，而且比那笔支出折现得更远，所以它本身不可能让建设变得有利
可图；它只是让规划期末端不再惩罚晚期投资。它作为一条负的 `salvage_credit` 项记在最后一个
payload 上（其余 payload 上为零，使各年成本分项的表结构相同）。
`end_of_horizon_salvage=False` 时根本不写入这个键，因此复现 2026-09-22 之前目标函数的
运行，也会复现当时的 `cost_breakdown.csv` 表结构。

已知简化：经济寿命取资产自身的，而不是所在机组的。一台 2065 年外生退役的机组上，2060 年
做的空冷改造到 2070 年仍能收回一半 capex；2030 年的捕集岛（20 a）一直运行到 2070 年，
却不计更换 capex（相比之下，管道在求解器里确实会在 `pipeline_lifetime_years` 到期）。
两种效应在整个机组群层面都是二阶的，且在煤电与工业之间对称。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, Sequence

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover
    gp = None

from ._shared import _discount_factor

if TYPE_CHECKING:  # pragma: no cover
    from .scenario import OptimizationAssumptions, OptimizationScenario
    from .year_types import YearPayload

logger = logging.getLogger(__name__)


class _PlanningPeriod(Protocol):
    """`horizon_end_year` 从 payload 读取的内容。"""

    year: int
    interval_years: int


def horizon_end_year(year_payloads: Sequence[_PlanningPeriod]) -> int:
    """最后一个规划年加上它所代表的间隔年数。"""
    last = year_payloads[-1]
    return int(last.year) + int(last.interval_years)


def remaining_fraction(build_year: int, life_years: int, end_year: int) -> float:
    """在 `build_year` 建成的资产到规划期末时尚未折旧的比例。"""
    life = max(1, int(life_years))
    served = max(0, int(end_year) - int(build_year))
    return max(0.0, 1.0 - served / life)


def _add_salvage_credit(
    year_payloads: list["YearPayload"],
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    cost_scale: float,
) -> None:
    """向每个 payload 的 `cost_exprs` 追加 `salvage_credit`，并刷新目标函数。

    Args:
        year_payloads: 按规划年顺序排列的求解器 payload；每个都带 `salvage_ledger`，
            即 `(name, undiscounted capex expr, life_years)` 的列表。
        scenario: `OptimizationScenario`；取其贴现率与基年。
        assumptions: `OptimizationAssumptions`；取其 `end_of_horizon_salvage` 开关。
        cost_scale: 求解器的目标函数缩放（`_COST_SCALE`），与其他所有成本表达式一样
            也作用于抵扣额。
    """
    if not bool(assumptions.end_of_horizon_salvage):
        return  # objective_expr 组装时本就不含该键
    if gp is None:  # pragma: no cover
        raise ImportError("gurobipy is required to build the salvage credit")
    for payload in year_payloads:
        payload.cost_exprs["salvage_credit"] = 0.0
    end_year = horizon_end_year(year_payloads)
    df_end = _discount_factor(end_year, scenario.discount_base_year, scenario.discount_rate)
    terms = []
    for payload in year_payloads:
        build_year = int(payload.year)
        for name, expr, life in payload.salvage_ledger:
            frac = remaining_fraction(build_year, life, end_year)
            if frac <= 0.0:
                continue
            if isinstance(expr, (int, float)):
                if float(expr) != 0.0:
                    raise ValueError(f"salvage ledger item {name!r} in {build_year} is a constant {expr}")
                continue  # 构建函数返回 0.0：本年这一类资产都建不了
            terms.append(frac * expr)
    if terms:
        year_payloads[-1].cost_exprs["salvage_credit"] = -df_end * gp.quicksum(terms) / float(cost_scale)
    for payload in year_payloads:
        payload.objective_expr = gp.quicksum(list(payload.cost_exprs.values()))
    logger.info("salvage credit at horizon end %d (df=%.4f) on %d capex items", end_year, df_end, len(terms))
