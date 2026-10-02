"""工业年块：路线份额变量、年内约束、跨年单调性、分代能力与一次性 capex 表达式。"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover
    gp = None

from ..constants import AMMONIA_FLOW_SCALE, WATER_FLOW_SCALE
from ..constants_industry import INDUSTRY_ROUTES
from .industry_inputs import IndustryInputs
from .industry_matrices import CCS, H2, UNABATED, IndustryYearData
from .vintage import StockYear, add_vintage_stock
from .year_types import GrbExpr, GrbMVar


@dataclass(frozen=True)
class IndustryPayload:
    """`add_industry_year` 的输出：一年的工业变量与派生表达式。

    求解器把它们接进 CO2 管网（`captured_by_hub`）、流域取水上限（`withdrawal_by_hub_scaled`）、
    部门目标（`residual_by_group`）、氢节点（`h2_flow_kg`）与目标函数（`annual_cost_expr`，
    一次性 capex 另由 `industry_capex_expr` 从 `new_capacity_mt` 与 `year_data.capex_cny_per_mt` 构造）。
    在役能力、到寿命退出与捕集的固定运维在 `add_industry_capacity`。
    """

    share: GrbMVar                            # (hub_count, len(INDUSTRY_ROUTES))
    year_suffix: str
    captured_by_hub: list[GrbExpr]            # Mt/yr
    reduction_by_hub: list[GrbExpr]           # Mt/yr
    total_reduction_mt: GrbExpr
    residual_by_group: dict[str, GrbExpr]     # 部门目标组 -> 残余排放，Mt/yr
    withdrawal_by_hub_scaled: list[GrbExpr]   # 取水，与煤电流域表达式同一缩放单位（WATER_FLOW_SCALE）
    # CNY/yr，未折现未缩放：CCS 路线的能耗与耗材 + 氢路线带地板的年度成本（含按链路买氢）；
    # 捕集的固定运维按在役且在用的能力另计（`vintage`）。
    annual_cost_expr: GrbExpr
    h2_flow_kg: GrbMVar                       # 每条氢链路，kg / AMMONIA_FLOW_SCALE
    h2_route_cost: GrbMVar                    # (hub_count,)，氢路线年度成本（地板变量）
    # (hub_count, len(INDUSTRY_ROUTES))，本年新建的路线能力，Mt/yr（CCS 为捕集能力，氢路线为产能）；
    # 一次性 capex 计在它上面，建成后在役到寿命（`vintage`）。unabated 列恒为 0。
    new_capacity_mt: GrbMVar
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

    # 本年新建能力（一次性 capex 的计费基数）。在役能力 >= 份额为 1 时所需能力 x 份额，所需能力随产量指数
    # 变化，所以份额不变而产量增长时要新建能力，产量萎缩后闲置的在役能力可以接住份额的上升；这些约束要用到
    # 往年的新建量，在所有年块建完后加（`add_industry_capacity`）。
    capacity_ub = np.full((n, n_routes), gp.GRB.INFINITY)
    capacity_ub[:, UNABATED] = 0.0
    new_capacity_mt = model.addMVar(
        (n, n_routes), lb=0.0, ub=capacity_ub, name=f"ind_new_capacity_mt_{year_suffix}"
    )

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
        new_capacity_mt=new_capacity_mt,
        year_data=ydata,
    )


def add_industry_monotonicity(model, payloads: list[IndustryPayload], hub_count: int) -> None:
    """减排路线不可逆：`ccs` 与 `h2` 的份额跨年都不下降。

    份额之和为 1，因此这同时禁止了把捕集换成氢——没有人会拆掉捕集岛去建 DRI 竖炉。
    能力不再跨年单调（2026-09-30 起）：建成的能力到寿命退出，份额仍在就得重建（`vintage`）。

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


def add_industry_capacity(
    model, years: Sequence[int], payloads: list[IndustryPayload], salvage_end_year: int | None = None
) -> tuple[list[StockYear], list[StockYear]]:
    """捕集与氢路线能力按建设年分代（`vintage.add_vintage_stock`）：在役 >= 所需能力 x 份额，到寿命退出。

    所需能力按铭牌产能定（`IndustryYearData.capacity_mt_per_share`），寿命取 `capex_lifetime_years`。
    捕集的固定运维按建设年的单价（`fixed_om_cny_per_mt`）；氢路线 capex 不随年份变，固定运维按当年单价
    留在年度项里（`industry_matrices`），这里不单列。*salvage_end_year* 不为 None 时建期末在用量，残值按它计。

    Returns:
        逐年的（捕集，氢路线）在役能力、固定运维与期末在用量，每个 hub 一项。
    """
    hub_count = int(payloads[0].share.shape[0])
    lives = payloads[0].year_data.capex_lifetime_years

    def required(route: int) -> list[list[GrbExpr]]:
        return [
            [float(p.year_data.capacity_mt_per_share[h, route]) * p.share[h, route] for h in range(hub_count)]
            for p in payloads
        ]

    ccs = add_vintage_stock(
        model, "ind_ccs_capacity", years,
        new=[p.new_capacity_mt[:, CCS] for p in payloads], required=required(CCS), life=int(lives[CCS]),
        om_unit=[np.asarray(p.year_data.fixed_om_cny_per_mt[:, CCS], dtype=np.float64) for p in payloads],
        salvage_end_year=salvage_end_year,
    )
    h2 = add_vintage_stock(
        model, "ind_h2_capacity", years,
        new=[p.new_capacity_mt[:, H2] for p in payloads], required=required(H2), life=int(lives[H2]),
        salvage_end_year=salvage_end_year,
    )
    return ccs, h2


def industry_capex_expr(payload: IndustryPayload, routes: tuple[int, ...] = (CCS, H2)) -> GrbExpr:
    """本年的一次性资本费用：本年单位能力 capex x 本年新建能力。

    与煤电捕集岛同法：按新建能力而不是按份额计费（2026-09-30 前按单调存量的增量计，存量永不退出）。

    Args:
        payload: 本年的工业 payload（`new_capacity_mt` 变量与 `year_data`）。
        routes: 要计入的路线下标；求解器分别请求 CCS 与 H2，好让两者在残值抵扣里
            各带自己的经济寿命。
    """
    new_capacity = payload.new_capacity_mt
    unit = payload.year_data.capex_cny_per_mt
    terms = [
        float(unit[hub, route]) * new_capacity[hub, route]
        for hub in range(new_capacity.shape[0]) for route in routes
        if float(unit[hub, route]) > 0.0
    ]
    return gp.quicksum(terms) if terms else 0.0
