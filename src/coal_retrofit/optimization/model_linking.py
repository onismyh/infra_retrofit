"""跨期约束（不可逆性、单调性、分代能力的在役与退出）与逐年管网/封存容量约束。"""
from __future__ import annotations

import numpy as np

from ._shared import PATHWAY_INDEX, SolveState, gp
from .model_industry import add_industry_capacity
from .scenario import OptimizationAssumptions, OptimizationScenario
from .vintage import add_vintage_stock
from .year_types import YearPayload


def add_inter_period_constraints(
    model,
    year_payloads: list[YearPayload],
    scenario: OptimizationScenario,
    plant_count: int,
    edge_count: int,
    industry_hub_count: int,
) -> None:
    """所有年块建完后加的跨期约束。多年才有意义的项在单年模型下自动跳过。"""
    from .industry import add_industry_monotonicity

    # 掺烧档位单调：CDF 形式 Σ_{l'≤l} select[p,l',t+1] ≤ Σ_{l'≤l} select[p,l',t]。掺烧设备不可逆。
    if len(year_payloads) > 1:
        n_opts_b = len(scenario.biomass_blend_levels) + 1
        n_opts_a = len(scenario.ammonia_blend_levels) + 1
        for yi in range(1, len(year_payloads)):
            sel_b_curr = year_payloads[yi].select_b
            sel_b_prev = year_payloads[yi - 1].select_b
            sel_a_curr = year_payloads[yi].select_a
            sel_a_prev = year_payloads[yi - 1].select_a
            year_sfx = str(year_payloads[yi].year)
            for p in range(plant_count):
                cdf_b_curr = gp.LinExpr()
                cdf_b_prev = gp.LinExpr()
                for level in range(n_opts_b - 1):
                    cdf_b_curr += sel_b_curr[p, level]
                    cdf_b_prev += sel_b_prev[p, level]
                    model.addConstr(cdf_b_curr <= cdf_b_prev, name=f"mono_b_{p}_{level}_{year_sfx}")
                cdf_a_curr = gp.LinExpr()
                cdf_a_prev = gp.LinExpr()
                for level in range(n_opts_a - 1):
                    cdf_a_curr += sel_a_curr[p, level]
                    cdf_a_prev += sel_a_prev[p, level]
                    model.addConstr(cdf_a_curr <= cdf_a_prev, name=f"mono_a_{p}_{level}_{year_sfx}")

    # 工业减排不可逆。
    if len(year_payloads) > 1:
        add_industry_monotonicity(
            model, [payload.industry for payload in year_payloads], industry_hub_count,
        )

    # 退役单调：退了不能重启。捕集份额锁定：捕集岛投运后持续运行，份额只能通过退役离开捕集路径
    # （2026-09-10 前只有 capex 存量单调，运行份额可零成本归零）。捕集岛到寿命退出后份额仍锁定，得重建。
    retire_idx = PATHWAY_INDEX["retire"]
    ccs_idx, beccs_idx = PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]
    if len(year_payloads) > 1:
        for yi in range(1, len(year_payloads)):
            share_curr = year_payloads[yi].share
            share_prev = year_payloads[yi - 1].share
            yr_sfx = str(year_payloads[yi].year)
            model.addConstrs(
                (share_curr[p, retire_idx] >= share_prev[p, retire_idx] for p in range(plant_count)),
                name=f"retire_mono_{yr_sfx}",
            )
            model.addConstrs(
                (
                    share_curr[p, ccs_idx] + share_curr[p, beccs_idx]
                    >= share_prev[p, ccs_idx] + share_prev[p, beccs_idx]
                    - (share_curr[p, retire_idx] - share_prev[p, retire_idx])
                    for p in range(plant_count)
                ),
                name=f"capture_share_lock_{yr_sfx}",
            )

    # 管道建成不可逆。
    if len(year_payloads) > 1:
        for yi in range(1, len(year_payloads)):
            be_curr = year_payloads[yi].build_edge
            be_prev = year_payloads[yi - 1].build_edge
            yr_sfx = str(year_payloads[yi].year)
            model.addConstrs(
                (be_curr[e] >= be_prev[e] for e in range(edge_count)),
                name=f"build_irreversible_{yr_sfx}",
            )

    # build_edge[e,t] = 1 当且仅当 t 或之前某期 add_cap 置 1。
    for yi, payload in enumerate(year_payloads):
        yr_sfx = str(payload.year)
        model.addConstrs(
            (
                payload.build_edge[edge_idx]
                <= gp.quicksum(year_payloads[pi].add_cap[edge_idx] for pi in range(yi + 1))
                for edge_idx in range(edge_count)
            ),
            name=f"build_flag_tie_{yr_sfx}",
        )

    # 原址重建不可逆。
    if len(year_payloads) > 1:
        for yi in range(1, len(year_payloads)):
            rb_curr = year_payloads[yi].rebuild
            rb_prev = year_payloads[yi - 1].rebuild
            yr_sfx = str(year_payloads[yi].year)
            model.addConstrs(
                (rb_curr[p] >= rb_prev[p] for p in range(plant_count)),
                name=f"rebuild_irreversible_{yr_sfx}",
            )


def add_capacity_vintages(
    model, year_payloads: list[YearPayload], assumptions: OptimizationAssumptions
) -> None:
    """三类改造能力按建设年分代（`vintage.add_vintage_stock`）：在役 >= 当年所需、到寿命退出、固定运维按建设年的单价。

    - 煤电捕集岛（CCS 与 BECCS 共用，占装机的份额）：所需 = share_ccs + share_beccs，寿命 `ccs_retrofit_lifetime_years`；
    - 工业捕集能力与氢路线能力：`model_industry.add_industry_capacity`。
    结果写到各 payload 的 `ccs_island`、`industry_ccs`、`industry_h2`，供成本与结果提取。
    """
    years = [int(payload.year) for payload in year_payloads]
    ccs_k, beccs_k = PATHWAY_INDEX["ccs"], PATHWAY_INDEX["beccs"]
    plant_count = int(year_payloads[0].share.shape[0])
    islands = add_vintage_stock(
        model, "ccs_island", years,
        new=[payload.retrofit_new[:, 0] for payload in year_payloads],
        required=[
            [payload.share[p, ccs_k] + payload.share[p, beccs_k] for p in range(plant_count)]
            for payload in year_payloads
        ],
        life=int(assumptions.ccs_retrofit_lifetime_years),
        om_unit=[np.asarray(payload.year_data.retrofit_stock_om, dtype=np.float64)[:, 0] for payload in year_payloads],
    )
    industry_ccs, industry_h2 = add_industry_capacity(model, years, [payload.industry for payload in year_payloads])
    for payload, island, ccs, h2 in zip(year_payloads, islands, industry_ccs, industry_h2, strict=True):
        payload.ccs_island, payload.industry_ccs, payload.industry_h2 = island, ccs, h2


def add_capacity_constraints(
    model,
    payload: YearPayload,
    year_payloads: list[YearPayload],
    year_position: int,
    assumptions: OptimizationAssumptions,
    state: SolveState,
    edge_base_stock: np.ndarray,
    edge_max_new_total: np.ndarray,
    edge_count: int,
    storage_count: int,
) -> None:
    """一年的管道新增/流量容量与封存累计容量约束。"""
    year_suffix = str(payload.year)
    build_edge = payload.build_edge
    add_cap = payload.add_cap
    pipe_count = payload.pipe_count
    new_cap_mtpa = payload.new_cap_mtpa
    edge_flow_mtpa = payload.edge_flow_mtpa
    edge_slack_mtpa = payload.edge_slack_mtpa
    storage_slack_mt = payload.storage_slack_mt
    n_tiers = int(pipe_count.shape[1])
    edge_buildable = (edge_max_new_total > 1e-9).astype(float)

    model.addConstrs((build_edge[edge_idx] <= edge_buildable[edge_idx] for edge_idx in range(edge_count)), name=f"edge_buildable_{year_suffix}")
    # 容量上限挂在 add_cap（本期新增）上而非锁存的 build_edge：只有 add_cap=1 才能铺管，
    # 铺了管 add_cap 才能为 1，标志精确。
    model.addConstrs(
        (new_cap_mtpa[edge_idx] <= edge_max_new_total[edge_idx] * add_cap[edge_idx] for edge_idx in range(edge_count)),
        name=f"edge_new_cap_limit_{year_suffix}",
    )
    model.addConstrs(
        (
            gp.quicksum(pipe_count[edge_idx, k] for k in range(n_tiers)) >= add_cap[edge_idx]
            for edge_idx in range(edge_count)
        ),
        name=f"edge_add_lays_pipe_{year_suffix}",
    )
    model.addConstrs(
        (
            gp.quicksum(pipe_count[edge_idx, k] for k in range(n_tiers))
            <= float(assumptions.max_parallel_pipes) * add_cap[edge_idx]
            for edge_idx in range(edge_count)
        ),
        name=f"edge_pipes_need_add_{year_suffix}",
    )
    model.addConstrs(
        (add_cap[edge_idx] <= build_edge[edge_idx] for edge_idx in range(edge_count)),
        name=f"edge_add_implies_build_{year_suffix}",
    )

    current_year = int(payload.year)
    lifetime = assumptions.pipeline_lifetime_years
    # 只累计仍在寿命内的往期新增容量：流量上限与累计新增上限都只数在役的管，到寿命的管可在
    # 原址重建。2026-09-23 前累计新增上限把已到寿命的管也算进去，边一旦铺满就再也不能重建。
    alive_indices = [
        past_idx for past_idx in range(year_position + 1)
        if current_year - int(year_payloads[past_idx].year) < lifetime
    ]
    model.addConstrs(
        (
            gp.quicksum(year_payloads[pi].new_cap_mtpa[edge_idx] for pi in alive_indices)
            <= edge_max_new_total[edge_idx]
            for edge_idx in range(edge_count)
        ),
        name=f"edge_total_new_cap_limit_{year_suffix}",
    )
    model.addConstrs(
        (
            edge_flow_mtpa[edge_idx]
            <= edge_base_stock[edge_idx]
            + gp.quicksum(year_payloads[pi].new_cap_mtpa[edge_idx] for pi in alive_indices)
            + edge_slack_mtpa[edge_idx]
            for edge_idx in range(edge_count)
        ),
        name=f"edge_capacity_limit_{year_suffix}",
    )
    model.addConstrs(
        (
            gp.quicksum(
                year_payloads[past_idx].storage_use_mtpa[storage_idx] * int(year_payloads[past_idx].interval_years)
                for past_idx in range(year_position + 1)
            )
            <= float(state.remaining_storage_mt[storage_idx]) + storage_slack_mt[storage_idx]
            for storage_idx in range(storage_count)
        ),
        name=f"storage_capacity_limit_{year_suffix}",
    )
