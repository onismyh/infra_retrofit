"""工业年块：路线份额变量、年内约束、跨年单调性与一次性 capex 表达式。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover
    gp = None

from ..constants import AMMONIA_FLOW_SCALE, WATER_FLOW_SCALE
from ..constants_industry import INDUSTRY_ROUTES
from .industry_inputs import IndustryInputs
from .industry_matrices import CCS, H2, IndustryYearData
from .year_types import GrbExpr, GrbMVar


@dataclass(frozen=True)
class IndustryPayload:
    """`add_industry_year` 的输出：一年的工业变量与派生表达式。

    求解器把它们接进 CO2 管网（`captured_by_hub`）、流域取水上限（`withdrawal_by_hub_scaled`）、
    部门目标（`residual_by_group`）、氢节点（`h2_flow_kg`）与目标函数（`annual_cost_expr`，
    一次性 capex 另由 `industry_capex_expr` 从 `share` 与 `year_data.capex_cny` 构造）。
    """

    share: GrbMVar                            # (hub_count, len(INDUSTRY_ROUTES))
    year_suffix: str
    captured_by_hub: list[GrbExpr]            # Mt/yr
    reduction_by_hub: list[GrbExpr]           # Mt/yr
    total_reduction_mt: GrbExpr
    residual_by_group: dict[str, GrbExpr]     # 部门目标组 -> 残余排放，Mt/yr
    withdrawal_by_hub_scaled: list[GrbExpr]   # 取水，与煤电流域表达式同一缩放单位（WATER_FLOW_SCALE）
    # CNY/yr，未折现未缩放：CCS 路线年度成本 + 氢路线带地板的年度成本（含按链路买氢）。
    annual_cost_expr: GrbExpr
    h2_flow_kg: GrbMVar                       # 每条氢链路，kg / AMMONIA_FLOW_SCALE
    h2_route_cost: GrbMVar                    # (hub_count,)，氢路线年度成本（地板变量）
    year_data: IndustryYearData


def add_industry_year(
    model,
    industry: IndustryInputs,
    ydata: IndustryYearData,
    year_suffix: str,
    h2_link_cost: np.ndarray | None = None,
    h2_hub_membership=None,
) -> IndustryPayload:
    """添加一个规划年的工业变量与年内约束。

    Args:
        model: 正在构建的 Gurobi 模型。
        industry: 准备好的工业输入。
        ydata: 本年的 `industry_year_data` 输出。
        year_suffix: 附加在每个变量名与约束名后面的年份字符串。
        h2_link_cost: 每条链路的到厂氢成本，单位 CNY / 缩放后的 kg（None：没有链路）。
        h2_hub_membership: 氢链路的稀疏 (hub, link) 关联矩阵。

    Returns:
        payload：路线份额变量，以及求解器接入 CO2 管网、流域上限、排放目标、氢节点与
        目标函数的各个派生表达式。
    """
    if gp is None:  # pragma: no cover
        raise RuntimeError("gurobipy is required to build the industrial block.")
    n = len(industry.hubs)
    n_routes = len(INDUSTRY_ROUTES)
    share = model.addMVar((n, n_routes), lb=0.0, ub=1.0, name=f"ind_share_{year_suffix}")
    model.addConstrs(
        (share[hub, :].sum() == 1.0 for hub in range(n)), name=f"ind_share_sum_{year_suffix}"
    )
    unavailable = np.argwhere(~ydata.route_available)
    if len(unavailable):
        model.addConstrs(
            (share[int(hub), int(route)] == 0.0 for hub, route in unavailable),
            name=f"ind_route_unavailable_{year_suffix}",
        )
    captured = ydata.captured_mt
    captured_by_hub = [
        gp.quicksum(float(captured[hub, r]) * share[hub, r] for r in range(n_routes))
        for hub in range(n)
    ]
    reduction = ydata.reduction_mt
    baseline = ydata.baseline_emissions_mt
    reduction_by_hub = [
        gp.quicksum(float(reduction[hub, r]) * share[hub, r] for r in range(n_routes))
        for hub in range(n)
    ]
    total_reduction = gp.quicksum(reduction_by_hub)
    groups = ydata.target_groups
    residual_by_group = {
        str(group): gp.quicksum(
            float(baseline[hub]) - reduction_by_hub[hub] for hub in np.flatnonzero(groups == group)
        )
        for group in sorted(set(str(g) for g in groups))
    }
    water = ydata.water_m3
    # 缩放到煤电侧流域表达式所用的同一 Mm3 单位，这样两者可以在同一条约束里相加，
    # 不会暗藏单位换算。
    withdrawal_by_hub = [
        gp.quicksum(float(water[hub, r]) / WATER_FLOW_SCALE * share[hub, r] for r in range(n_routes))
        for hub in range(n)
    ]
    opex = ydata.opex_cny
    annual_cost = gp.quicksum(
        float(opex[hub, r]) * share[hub, r] for hub in range(n) for r in (CCS,)
    )

    # 氢：逐链路采购、hub 平衡，以及带地板的氢路线年度成本。
    h2_demand = ydata.h2_demand_kg_per_share
    link_count = 0 if h2_link_cost is None else int(len(h2_link_cost))
    h2_flow_kg = model.addMVar(link_count, lb=0.0, name=f"ind_h2_flow_kg_{year_suffix}")
    h2_route_cost = model.addMVar(n, lb=0.0, name=f"ind_h2_cost_{year_suffix}")
    if link_count:
        hub_draw = h2_hub_membership @ h2_flow_kg
        purchase_by_hub = [None] * n
        # 先按 hub 把链路成本归组一次，这样每个 hub 的采购只是一个短表达式。
        coo = h2_hub_membership.tocoo()
        links_of_hub: dict[int, list[int]] = {}
        for hub_idx, link_idx in zip(coo.row.tolist(), coo.col.tolist()):
            links_of_hub.setdefault(int(hub_idx), []).append(int(link_idx))
        for hub in range(n):
            hub_links = links_of_hub.get(hub, [])
            purchase_by_hub[hub] = gp.quicksum(
                float(h2_link_cost[link]) * h2_flow_kg[link] for link in hub_links
            ) if hub_links else gp.LinExpr()
            model.addConstr(
                hub_draw[hub] == float(h2_demand[hub]) / AMMONIA_FLOW_SCALE * share[hub, H2],
                name=f"ind_h2_balance_{hub}_{year_suffix}",
            )
            if float(h2_demand[hub]) > 0.0:
                model.addConstr(
                    h2_route_cost[hub] >= float(opex[hub, H2]) * share[hub, H2] + purchase_by_hub[hub],
                    name=f"ind_h2_cost_lb_{hub}_{year_suffix}",
                )
    else:
        # 完全没有氢链路：这时只有需氢量为零的 hub 能走这条路线，而这些 hub 的路线已被
        # `industry_year_data` 关闭。这里仍显式强制平衡。
        for hub in range(n):
            if float(h2_demand[hub]) > 0.0:
                model.addConstr(share[hub, H2] == 0.0, name=f"ind_h2_no_supply_{hub}_{year_suffix}")
    annual_cost = annual_cost + h2_route_cost.sum()

    return IndustryPayload(
        share=share,
        year_suffix=year_suffix,
        captured_by_hub=captured_by_hub,
        reduction_by_hub=reduction_by_hub,
        total_reduction_mt=total_reduction,
        residual_by_group=residual_by_group,
        withdrawal_by_hub_scaled=withdrawal_by_hub,
        annual_cost_expr=annual_cost,
        h2_flow_kg=h2_flow_kg,
        h2_route_cost=h2_route_cost,
        year_data=ydata,
    )


def add_industry_monotonicity(model, payloads: list[IndustryPayload], hub_count: int) -> None:
    """减排路线不可逆：`ccs` 与 `h2` 的份额跨年从不下降。

    份额之和为 1，因此这同时禁止了把捕集换成氢——没有人会拆掉捕集岛去建 DRI 竖炉——
    也正是它让一次性资本费用可以按份额的增量收取：增量就是新建存量。

    Args:
        model: Gurobi 模型。
        payloads: 每个规划年一个 payload，按时间顺序排列。
        hub_count: 工业 hub 个数。
    """
    for index in range(1, len(payloads)):
        current = payloads[index].share
        previous = payloads[index - 1].share
        suffix = payloads[index].year_suffix
        for route in (CCS, H2):
            model.addConstrs(
                (current[hub, route] >= previous[hub, route] for hub in range(hub_count)),
                name=f"ind_mono_{INDUSTRY_ROUTES[route]}_{suffix}",
            )


def industry_capex_expr(
    payload: IndustryPayload, previous: IndustryPayload | None, routes: tuple[int, ...] = (CCS, H2)
) -> GrbExpr:
    """本年的一次性资本费用：capex 系数 x 路线份额增量。

    份额单调（`add_industry_monotonicity`），所以 `share_t - share_{t-1}` 就是新建存量，
    且从不为负；第一年整个份额都是新建的。

    Args:
        payload: 本年的工业 payload（`share` 变量与 `year_data`）。
        previous: 上一年的 payload；第一年为 None。
        routes: 要计入的路线下标；求解器分别请求 CCS 与 H2，好让两者在残值抵扣里
            各带自己的经济寿命。
    """
    share = payload.share
    capex = payload.year_data.capex_cny
    n = share.shape[0]
    terms = []
    for hub in range(n):
        for route in routes:
            coeff = float(capex[hub, route])
            if coeff <= 0.0:
                continue
            if previous is None:
                terms.append(coeff * share[hub, route])
            else:
                terms.append(coeff * (share[hub, route] - previous.share[hub, route]))
    return gp.quicksum(terms) if terms else 0.0
