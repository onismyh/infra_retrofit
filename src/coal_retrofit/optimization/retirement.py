"""煤电退役核算：提前退役与重建后退役的存量、每期新增、自愿退役速率上限，期末已退役重建装机的残值扣回。

hub 装机按份额分五部分：未到期在运行、未到期就关停（提前退役，存量 U）、到期未重建（正常寿终）、原址重建在运行（R）、
重建后又关停（存量 Q）。自愿退役是 U 的新增（扣除按比例到期的部分）与 R 的净减少（2026-10-02 起，此前按退役份额的增量计）。
记 f 为到期份额、x 为退役份额、ρ 为重建份额、Σr 为重建部分（各重建热耗类之和）的运行份额（`constraints._add_rebuilt_split`）：

    f = 0：U = x，Q = 0；  0 < f < 1：U = x - f + Σr，Q = ρ - Σr；  f = 1：U = 0，Q = x - 1 + ρ。

模型不追踪逐台机组，提前退役按未到期机组等比例摊：与未重建部分取未到期机组的平均毛热耗（`data_prep._with_expiry`）、
搁浅资产取未到期装机的平均剩余账面份额（`plant_matrices`）是同一个假设。于是本期到期的装机里早已关停的比例是
U_{t-1} / (1 - f_{t-1})，上期的 U 留到本期的是 c = (1 - f_t) / (1 - f_{t-1}) 倍。
"""
from __future__ import annotations

from ._shared import PATHWAY_INDEX, gp
from .salvage import horizon_end_year, remaining_fraction
from .scenario import OptimizationAssumptions, OptimizationScenario
from .year_types import GrbExpr, YearPayload

_RETIRE = PATHWAY_INDEX["retire"]


def _retired_stocks(payload: YearPayload, p: int) -> tuple[GrbExpr, GrbExpr]:
    """hub p 的提前退役存量 U 与重建后退役存量 Q（份额，见模块说明）；到期约束与拆分约束保证两者 >= 0。"""
    expired = float(payload.year_data.expired_share[p])
    retired = payload.share[p, _RETIRE]
    if expired <= 0.0:
        return retired, 0.0
    if expired >= 1.0:
        return 0.0, retired - 1.0 + payload.rebuild[p]
    rebuilt_running = gp.quicksum(var for part in payload.rebuilt_share[p].values() for var in part.values())
    return retired - expired + rebuilt_running, payload.rebuild[p] - rebuilt_running


def retirement_flows(
    model, payload: YearPayload, prev_payload: YearPayload | None, plant_count: int
) -> tuple[list[GrbExpr], list[GrbExpr]]:
    """逐 hub 本期新增的提前退役 n^o 与重建后退役 n^r（份额）。

    n^o = U_t - c x U_{t-1}，约束 >= 0：关停的机组不能重启。n^r = max(0, R_{t-1} - R_t)，R = ρ - Q 为在运行的重建装机，
    只计它的净减少：本期新重建又没运行的不算，整数 hub 的 ρ 只能取 0 或 1，到期当期退掉的那部分是正常寿终。
    另加 Q 不减（重建后关停的不能重启）。首期 n^o = U、n^r = 0。只在可能出负值的 hub 上加约束与辅助变量。
    """
    if prev_payload is None:
        return [_retired_stocks(payload, p)[0] for p in range(plant_count)], [0.0] * plant_count
    yr_sfx = str(payload.year)
    early_new: list[GrbExpr] = []
    rebuilt_new: list[GrbExpr] = []
    for p in range(plant_count):
        expired = float(payload.year_data.expired_share[p])
        prev_expired = float(prev_payload.year_data.expired_share[p])
        early, retired_rebuilt = _retired_stocks(payload, p)
        prev_early, prev_retired_rebuilt = _retired_stocks(prev_payload, p)
        if expired >= 1.0:
            early_new.append(0.0)  # U = 0 且 c = 0
        else:
            new = early - (1.0 - expired) / (1.0 - prev_expired) * prev_early
            if expired > 0.0:  # f = 0 时 n^o 是退役份额的增量，退役单调已保证 >= 0
                model.addConstr(new >= 0.0, name=f"early_retire_new_lb_{p}_{yr_sfx}")
            early_new.append(new)
        if prev_expired <= 0.0:
            rebuilt_new.append(0.0)  # 上期还没有重建装机
            continue
        running_drop = (prev_payload.rebuild[p] - prev_retired_rebuilt) - (payload.rebuild[p] - retired_rebuilt)
        if prev_expired >= 1.0:
            rebuilt_new.append(running_drop)  # 两期 f 都是 1：R = 1 - x，净减少就是退役份额的增量
            continue
        model.addConstr(retired_rebuilt >= prev_retired_rebuilt, name=f"rebuilt_retired_mono_{p}_{yr_sfx}")
        new = model.addVar(lb=0.0, name=f"rebuilt_retire_new_{p}_{yr_sfx}")
        model.addConstr(new >= running_drop, name=f"rebuilt_retire_new_lb_{p}_{yr_sfx}")
        rebuilt_new.append(new)
    return early_new, rebuilt_new


def add_retirement_rate_limit(
    model,
    payload: YearPayload,
    scenario: OptimizationScenario,
    early_new: list[GrbExpr],
    rebuilt_new: list[GrbExpr],
    plant_count: int,
) -> None:
    """自愿退役速率上限：本期新增的提前退役与重建后退役（`retirement_flows`）按发电份额加权，不超过
    `max_new_retirement_share_per_period`。两边除以总发电量，系数为发电份额，避免 4e7 量级系数。"""
    generation = payload.year_data.generation
    total_gen = float(generation.sum())
    if scenario.max_new_retirement_share_per_period <= 0 or total_gen <= 0:
        return
    model.addConstr(
        gp.quicksum(float(generation[p]) / total_gen * (early_new[p] + rebuilt_new[p]) for p in range(plant_count))
        <= scenario.max_new_retirement_share_per_period,
        name=f"max_retire_rate_{payload.year}",
    )


def add_retired_rebuild_offset(
    model,
    year_payloads: list[YearPayload],
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    plant_count: int,
) -> None:
    """期末重建后已关停的装机（Q_T）不计残值：在残值台账上按建设年扣回（`salvage._add_salvage_credit`）。

    关停的是哪一年重建的不追踪，按先关最早建成的计（残值比例随建设年递增，求解器本就这样选）：建设年 τ 的
    0 <= q_τ <= 当年新增重建，Σ q_τ >= Q_T - 残值为零那几年的累计重建，τ 年的台账记 -重建单价 x q_τ。
    关停的重建装机不计搁浅资产：它的 capex 已在目标函数里，损失就是这笔残值。没有残值时不加。
    """
    if not bool(assumptions.end_of_horizon_salvage):
        return
    life = int(assumptions.rebuild_lifetime_years)
    end_year = horizon_end_year(year_payloads)
    cost_per_mw = assumptions.stranded_asset_base_cny_per_kw * scenario.rebuild_capex_fraction * 1000.0
    last = year_payloads[-1]
    offsets: list[list[GrbExpr]] = [[] for _ in year_payloads]
    for p in range(plant_count):
        if float(last.year_data.expired_share[p]) <= 0.0:
            continue  # 规划期内没有到期装机，不能重建
        written_off: GrbExpr = 0.0
        vintages = []
        for i, payload in enumerate(year_payloads):
            if remaining_fraction(int(payload.year), life, end_year) <= 0.0:
                written_off = payload.rebuild[p]  # 残值比例随建设年递增，为零的是开头几年
                continue
            if float(payload.year_data.expired_share[p]) <= 0.0:
                continue  # 本年不能重建
            q = model.addVar(lb=0.0, name=f"rebuild_retired_{p}_{payload.year}")
            model.addConstr(
                q <= payload.rebuild[p] - (year_payloads[i - 1].rebuild[p] if i else 0.0),
                name=f"rebuild_retired_le_new_{p}_{payload.year}",
            )
            vintages.append(q)
            offsets[i].append(float(last.year_data.capacity_mw[p]) * cost_per_mw * q)
        if vintages:
            model.addConstr(
                gp.quicksum(vintages) >= _retired_stocks(last, p)[1] - written_off,
                name=f"rebuild_retired_by_vintage_{p}",
            )
    for payload, items in zip(year_payloads, offsets, strict=True):
        if items:
            payload.salvage_ledger.append(("rebuild_retired", -gp.quicksum(items), life))
