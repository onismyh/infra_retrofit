"""按建设年分代的改造能力（作者决定 2026-09-30）：到寿命退出，固定运维按建设年的单价。

煤电捕集岛、空冷改造、生物质与氨掺烧升级（后三者 2026-10-02 起）、工业捕集能力、工业氢路线能力共用这一套：

- 规划年 t 的新建量 b_t ≥ 0，一次性 capex = t 年单价 × b_t（`model_costs`）；
- v 年建成的能力在 t − v < L 的规划年在役，L 为经济寿命，与管道 `pipeline_lifetime_years` 同一口径；到寿命退出，
  仍要用就得重建；
- 在用量 y_{v,t} ≤ b_v，Σ_v y_{v,t} ≥ 当年所需能力；固定运维 = Σ_v 运维单价_v × y_{v,t}，单价按建设年定。
  所需能力降下来（退役、减产）时多出的那部分不再付固定运维，capex 不退。不单列固定运维的一类不建 y，
  直接要求 Σ_v b_v ≥ 当年所需能力；
- 期末残值只计最后一个规划年仍在用的部分（2026-10-02 起，`salvage`）：最后一年 Σ_v y_v ≤ 当年所需（单列固定运维的
  一类即取等号），残值按 y_v 计；不单列固定运维的一类只为期末残值比例 > 0 的建设年在最后一年建 y_v。

2026-09-30 之前：存量跨期单调、永不退出；固定运维按当年单价计（煤电按路径份额，工业按当年捕集量）。
2026-10-02 之前：空冷改造与掺烧升级不分代、永不到期；残值按建成量计，期末闲置的也计。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ._shared import gp
from .salvage import remaining_fraction
from .year_types import GrbExpr, GrbMVar


@dataclass(frozen=True)
class StockYear:
    """一类分代能力在一个规划年的派生量，每个单元（电厂或 hub）一项。"""

    alive: list[GrbExpr]     # 在役能力：未到寿命的历年新建量之和
    fixed_om: list[GrbExpr]  # 固定运维，CNY/yr，未折现；不单列固定运维时为 0.0
    # 本年建成、规划期最后一年仍在用的能力，一维 MVar，期末残值按它计（`model_costs`）；不计残值或期末残值比例为零时为 None。
    end_in_use: GrbMVar | None = None


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
    salvage_end_year: int | None = None,
) -> list[StockYear]:
    """加一类分代能力的在役约束，返回逐年的在役能力、固定运维与期末在用量。

    Args:
        model: Gurobi 模型。
        name: 变量与约束名的前缀。
        years: 规划年，升序。
        new: 每个规划年的新建量，一维 MVar，每个单元一项。
        required: 每个规划年每个单元所需的在役能力，与 `new` 同单位。
        life: 经济寿命（年）。
        om_unit: 每个规划年建成的每单位能力每年的固定运维（CNY/yr），每个单元一项；None 时不单列固定运维。
        salvage_end_year: 计残值的期末年（`salvage.horizon_end_year`）；None 时不计残值，不建期末在用量。
    """
    stocks = []
    in_use: dict[int, GrbMVar] = {}
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
    if salvage_end_year is None:
        return stocks
    end_in_use = _end_in_use(
        model, name, years, new, required[-1], life, in_use if om_unit is not None else None, salvage_end_year
    )
    return [replace(stock, end_in_use=end_in_use.get(v)) for v, stock in enumerate(stocks)]


def _end_in_use(
    model,
    name: str,
    years: Sequence[int],
    new: Sequence[GrbMVar],
    required: Sequence[GrbExpr],
    life: int,
    in_use: dict[int, GrbMVar] | None,
    end_year: int,
) -> dict[int, GrbMVar]:
    """最后一个规划年仍在用的各代能力，{建设年下标: MVar}，只含期末残值比例 > 0 的建设年；Σ <= 当年所需 *required*。

    *in_use* 是单列固定运维的一类在最后一年的在用量（各在役代都有），加上这条后 Σ = 当年所需；为 None 时只为
    有残值的建设年新建在用量，<= 当年新建量。
    """
    last = len(years) - 1
    salvaged = [v for v in alive_vintages(years, last, life) if remaining_fraction(int(years[v]), life, end_year) > 0.0]
    if not salvaged:
        return {}
    count = len(required)
    if in_use is None:
        in_use = {v: model.addMVar(count, lb=0.0, name=f"{name}_end_in_use_{years[v]}") for v in salvaged}
        model.addConstrs(
            (in_use[v][i] <= new[v][i] for v in salvaged for i in range(count)), name=f"{name}_end_built"
        )
    model.addConstrs(
        (gp.quicksum(in_use[v][i] for v in in_use) <= required[i] for i in range(count)), name=f"{name}_end_in_use"
    )
    return {v: in_use[v] for v in salvaged}
