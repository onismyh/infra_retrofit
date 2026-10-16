"""求解辅助：LP 松弛解取整做 MIP 起点、增量解回调。只改搜索路径，不改模型。"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from ._shared import GRB
from .assumptions import OptimizationAssumptions

logger = logging.getLogger(__name__)


def _cheapest_cover(need: float, tiers: tuple[float, ...], costs: tuple[float, ...]) -> tuple[int, ...]:
    """覆盖 `need`（Mt/a）的最便宜整根组合，各档根数不设上限，按 `tiers` 的顺序返回各档根数。

    从大档到小档枚举根数（每档最多 ceil(余量/档容量) 根，多了只会更贵），最小一档补足余量；大档在前，枚举量很小。
    """
    if need <= 1e-6:
        return tuple(0 for _ in tiers)
    order = sorted(range(len(tiers)), key=lambda k: -tiers[k])

    def search(position: int, remaining: float) -> tuple[float, tuple[int, ...]]:
        k = order[position]
        upper = max(0, int(np.ceil(remaining / tiers[k] - 1e-9)))
        if position == len(order) - 1:
            return upper * costs[k], (upper,)
        best: tuple[float, tuple[int, ...]] | None = None
        for count in range(upper + 1):
            rest_cost, rest = search(position + 1, max(0.0, remaining - count * tiers[k]))
            candidate = (count * costs[k] + rest_cost, (count, *rest))
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        return best

    counts = dict(zip(order, search(0, need)[1]))
    return tuple(counts[k] for k in range(len(tiers)))


def _apply_rounded_start(model, sol_path: Path, assumptions: OptimizationAssumptions) -> None:
    """用 Gurobi .sol 文件给整数变量设 MIP 起点。

    管道：每条边每年松弛解的 `new_cap_mtpa` 用整根组合覆盖，组合取目标函数里 `pipe_count` 系数（已含折现 capex、固定运维
    与期末残值）合计最小的，根数不设上限（2026-10-16 起；此前每边每期最多 `max_parallel_pipes` 根、在役合计不超过边上限，
    按不折现的每 km capex 选组合）。逐年覆盖的整根容量都不小于松弛解，所以在役容量每年都不小于松弛解。
    掺烧档位选择行（`sel_b*` / `sel_a*`）取有正值的最高档；其余整数变量向上取整；连续变量不设起点。
    """
    values: dict[str, float] = {}
    with open(sol_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            name, _, text = line.partition(" ")
            values[name.strip()] = float(text)
    model.update()
    by_name = {var.VarName: var for var in model.getVars() if var.VType != GRB.CONTINUOUS}
    tiers = tuple(float(t) for t in assumptions.pipe_capacity_tiers_mtpa)
    years = sorted({int(m.group(1)) for m in (re.match(r"pipe_count_(\d+)\[", n) for n in by_name) if m})
    edge_ids = sorted({int(m.group(1)) for m in (re.match(r"pipe_count_\d+\[(\d+),", n) for n in by_name) if m})
    n_set = 0
    for e in edge_ids:
        for year in years:
            need = values.get(f"new_cap_mtpa_{year}[{e}]", 0.0)
            pipes = [by_name[f"pipe_count_{year}[{e},{k}]"] for k in range(len(tiers))]
            counts = _cheapest_cover(need, tiers, tuple(float(var.Obj) for var in pipes))
            for var, count in zip(pipes, counts):
                var.Start = float(count)
            n_set += len(counts)
    selector_rows: dict[str, list[tuple[int, Any, float]]] = {}
    for name, var in by_name.items():
        if name.startswith("pipe_count_"):
            continue
        value = values.get(name)
        if value is None:
            continue
        if name.startswith(("sel_b", "sel_a")) and "[" in name:
            row, _, col = name[:-1].partition("[")
            plant, level = col.split(",")
            selector_rows.setdefault(f"{row}[{plant}", []).append((int(level), var, value))
            continue
        var.Start = float(np.ceil(value - 1e-6)) if value > 1e-6 else 0.0
        n_set += 1
    for entries in selector_rows.values():
        touched = [level for level, _, value in entries if value > 1e-6]
        chosen = max(touched) if touched else 0
        for level, var, _ in entries:
            var.Start = 1.0 if level == chosen else 0.0
            n_set += 1
    logger.warning("MIP start: %d integer variables seeded from %s", n_set, sol_path)
