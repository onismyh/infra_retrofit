"""按建设年分代的改造能力（作者决定 2026-09-30）：到寿命退出，固定运维按建设年的单价。

煤电捕集岛、工业捕集能力、工业氢路线能力三类共用这一套：

- 规划年 t 的新建量 b_t ≥ 0，一次性 capex = t 年单价 × b_t（`model_costs`；期末按未折旧部分计残值，`salvage`）；
- v 年建成的能力在 t − v < L 的规划年在役，L 为经济寿命，与管道 `pipeline_lifetime_years` 同一口径；到寿命退出，
  仍要用就得重建；
- 在用量 y_{v,t} ≤ b_v，Σ_v y_{v,t} ≥ 当年所需能力；固定运维 = Σ_v 运维单价_v × y_{v,t}，单价按建设年定。
  所需能力降下来（退役、减产）时多出的那部分不再付固定运维，capex 不退。不单列固定运维的一类不建 y，
  直接要求 Σ_v b_v ≥ 当年所需能力。

2026-09-30 之前：存量跨期单调、永不退出；固定运维按当年单价计（煤电按路径份额，工业按当年捕集量）。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._shared import gp
from .year_types import GrbExpr, GrbMVar


@dataclass(frozen=True)
class StockYear:
    """一类分代能力在一个规划年的派生量，每个单元（电厂或 hub）一项。"""

    alive: list[GrbExpr]     # 在役能力：未到寿命的历年新建量之和
    fixed_om: list[GrbExpr]  # 固定运维，CNY/yr，未折现；不单列固定运维时为 0.0


def alive_vintages(years: Sequence[int], position: int, life: int) -> list[int]:
    """`years[position]` 年仍在役的建设年（在 `years` 里的下标）：t − v < life。"""
    year = int(years[position])
    return [v for v in range(position + 1) if year - int(years[v]) < int(life)]


def add_vintage_stock(
    model,
    name: str,
    years: Sequence[int],
    new: Sequence[GrbMVar],
    required: Sequence[Sequence[GrbExpr]],
    life: int,
    om_unit: Sequence[np.ndarray] | None = None,
) -> list[StockYear]:
    """加一类分代能力的在役约束，返回逐年的在役能力与固定运维表达式。

    Args:
        model: Gurobi 模型。
        name: 变量与约束名的前缀。
        years: 规划年，升序。
        new: 每个规划年的新建量，一维 MVar，每个单元一项。
        required: 每个规划年每个单元所需的在役能力，与 `new` 同单位。
        life: 经济寿命（年）。
        om_unit: 每个规划年建成的每单位能力每年的固定运维（CNY/yr），每个单元一项；None 时不单列固定运维。
    """
    stocks = []
    for position, year in enumerate(years):
        vintages = alive_vintages(years, position, life)
        count = len(required[position])
        alive = [gp.quicksum(new[v][i] for v in vintages) for i in range(count)]
        if om_unit is None:
            model.addConstrs(
                (alive[i] >= required[position][i] for i in range(count)), name=f"{name}_need_{year}"
            )
            stocks.append(StockYear(alive=alive, fixed_om=[0.0] * count))
            continue
        in_use = {v: model.addMVar(count, lb=0.0, name=f"{name}_in_use_{years[v]}_{year}") for v in vintages}
        model.addConstrs(
            (in_use[v][i] <= new[v][i] for v in vintages for i in range(count)), name=f"{name}_built_{year}"
        )
        model.addConstrs(
            (gp.quicksum(in_use[v][i] for v in vintages) >= required[position][i] for i in range(count)),
            name=f"{name}_need_{year}",
        )
        fixed_om = [
            gp.quicksum(float(om_unit[v][i]) * in_use[v][i] for v in vintages) for i in range(count)
        ]
        stocks.append(StockYear(alive=alive, fixed_om=fixed_om))
    return stocks
