"""求解辅助：LP 松弛解取整做 MIP 起点、增量解回调。只改搜索路径，不改模型。"""
from __future__ import annotations

import logging
import re
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from ._shared import GRB
from .scenario import OptimizationAssumptions
from .year_types import YearPayload

logger = logging.getLogger(__name__)


def _pipe_combo_for(capacity: float, tiers: tuple[float, ...], capex: tuple[float, ...],
                    max_pipes: int, cap_limit: float) -> tuple[int, ...]:
    """在 `cap_limit` 内、不超过 `max_pipes` 根的前提下，覆盖 `capacity` 的最便宜管径组合。"""
    if capacity <= 1e-6:
        return tuple(0 for _ in tiers)
    best: tuple[float, tuple[int, ...]] | None = None
    fallback: tuple[float, tuple[int, ...]] | None = None
    for n in range(1, max_pipes + 1):
        for combo in combinations_with_replacement(range(len(tiers)), n):
            cap = sum(tiers[k] for k in combo)
            cost = sum(capex[k] for k in combo)
            counts = tuple(combo.count(k) for k in range(len(tiers)))
            if cap > cap_limit + 1e-9:
                continue
            if cap + 1e-9 >= capacity:
                if best is None or cost < best[0]:
                    best = (cost, counts)
            elif fallback is None or cap > sum(tiers[k] * c for k, c in zip(range(len(tiers)), fallback[1])):
                fallback = (cap, counts)
    if best is not None:
        return best[1]
    return fallback[1] if fallback is not None else tuple(0 for _ in tiers)


def _apply_rounded_start(model, sol_path: Path, assumptions: OptimizationAssumptions) -> None:
    """用 Gurobi .sol 文件给整数变量设 MIP 起点。

    管道：每条边每年松弛解的 `new_cap_mtpa` 用最便宜的整根组合覆盖（在役新增累计不超过边上限，
    到寿命的管不再占额度，与 `edge_total_new_cap_limit` 一致），`add_cap` / `build_edge` 随之确定；
    掺烧档位选择行（`sel_b*` / `sel_a*`）取有正值的最高档；其余整数变量向上取整；连续变量不设起点。
    """
    values: dict[str, float] = {}
    with open(sol_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            name, _, value = line.partition(" ")
            values[name.strip()] = float(value)
    model.update()
    by_name = {var.VarName: var for var in model.getVars() if var.VType != GRB.CONTINUOUS}
    tiers = tuple(float(t) for t in assumptions.pipe_capacity_tiers_mtpa)
    capex = tuple(float(c) for c in assumptions.pipe_capex_cny_per_km_by_tier)
    max_pipes = int(assumptions.max_parallel_pipes)
    cap_total = float(assumptions.standard_pipe_capacity_mtpa) * max_pipes
    lifetime = int(assumptions.pipeline_lifetime_years)
    years = sorted({int(m.group(1)) for m in (re.match(r"pipe_count_(\d+)\[", n) for n in by_name) if m})
    edge_ids = sorted({int(m.group(1)) for m in (re.match(r"pipe_count_\d+\[(\d+),", n) for n in by_name) if m})
    n_set = 0
    for e in edge_ids:
        added_by_year: dict[int, float] = {}
        built = False
        for year in years:
            need = values.get(f"new_cap_mtpa_{year}[{e}]", 0.0)
            used = sum(cap for built_year, cap in added_by_year.items() if year - built_year < lifetime)
            counts = _pipe_combo_for(need, tiers, capex, max_pipes, cap_total - used)
            added = sum(t * c for t, c in zip(tiers, counts))
            added_by_year[year] = added
            built = built or added > 0
            for k, c in enumerate(counts):
                by_name[f"pipe_count_{year}[{e},{k}]"].Start = float(c)
            by_name[f"add_cap_{year}[{e}]"].Start = 1.0 if added > 0 else 0.0
            by_name[f"build_edge_{year}[{e}]"].Start = 1.0 if built else 0.0
            n_set += len(counts) + 2
    selector_rows: dict[str, list[tuple[int, object, float]]] = {}
    for name, var in by_name.items():
        if name.startswith(("pipe_count_", "add_cap_", "build_edge_")):
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


def _incumbent_logger(year_payloads: list[YearPayload]):
    """MIPSOL 回调：打印每个新增量解的各组目标缺口与物理松弛量。只读，不改搜索。"""
    slack_keys = ("injectivity_slack_mtpa", "storage_slack_mt", "edge_slack_mtpa", "biomass_slack_gj")

    def _callback(model, where) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        parts: list[str] = []
        for payload in year_payloads:
            groups = {
                str(group): float(model.cbGetSolution(var))
                for group, var in payload.target_shortfall_by_group.items()
            }
            slacks = {
                key: float(np.sum(model.cbGetSolution(getattr(payload, key).tolist())))
                for key in slack_keys
                if getattr(payload, key) is not None and int(getattr(payload, key).shape[0]) > 0
            }
            short_txt = " ".join(f"{g}={v:.1f}" for g, v in groups.items() if v > 1e-6) or "none"
            slack_txt = " ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in slacks.items() if v > 1e-6) or "none"
            parts.append(f"{payload.year}: shortfall[{short_txt}] slack[{slack_txt}]")
        print(f"INCUMBENT obj={obj:.1f} bound={bound:.1f} | " + " ; ".join(parts), flush=True)

    return _callback
