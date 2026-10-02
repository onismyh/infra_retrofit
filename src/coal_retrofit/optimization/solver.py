"""多期联合 MILP 的编排：建索引 → 逐年变量与约束 → 跨期约束与分代能力 → 容量与成本 → 残值 → 求解 → 提取。

各步的实现在 `model_index` / `model_year` / `model_linking` / `model_costs` / `retirement` / `salvage` /
`solver_extract`；本文件只负责顺序与求解器参数。变量与约束的创建顺序决定 Gurobi 指纹，
改动顺序会使新旧结果不可比。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._shared import (
    _COST_SCALE,
    GRB,
    PreparedInputs,
    SolveState,
    _extract_solver_status,
    _model_obj_value,
    _new_gurobi_model,
    gp,
)
from .model_costs import add_year_costs
from .model_index import build_model_index
from .model_linking import add_capacity_constraints, add_capacity_vintages, add_inter_period_constraints
from .model_year import add_year_block
from .retirement import add_retired_rebuild_offset
from .salvage import _add_salvage_credit
from .scenario import OptimizationAssumptions, OptimizationScenario
from .solver_extract import empty_year_solutions, extract_year_solutions
from .solver_provenance import _optional_model_attr, _solver_quality
from .solver_start import _apply_rounded_start, _incumbent_logger
from .year_types import SolveResult, YearPayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveControls:
    """只改搜索路径或只做诊断的开关，不改模型；要相减的求解必须用同一套（实现说明 §9.7）。

    - `relax`：把全部整数变量改成连续变量再求解（LP 松弛，热启动第 1 步）。
    - `start_sol`：按这个 .sol 给整数变量设 MIP start（热启动第 2 步，`solver_start._apply_rounded_start`）。
    - `write_sol`：有解时把解写到这里。路径须是 ASCII：Gurobi 在中文路径下写文件会失败。
    - `log_incumbents`：每个新可行解打一行 INCUMBENT。

    由运行器给：情景 `warm_start = "lp_relax"` 时两步各一套；此外读兼容的环境变量（`run_controls.env_controls`），
    此前由 `_solve_joint_multi_period` 自己读 COAL_RETROFIT_LP_RELAX / START_SOL / WRITE_SOL / LOG_INCUMBENTS。
    `_solve_joint_multi_period` 的 *controls* 缺省（None）即全关。
    """

    relax: bool = False
    start_sol: Path | None = None
    write_sol: Path | None = None
    log_incumbents: bool = False


def _solve_joint_multi_period(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    years: tuple[int, ...],
    state: SolveState,
    controls: SolveControls | None = None,
) -> SolveResult:
    controls = controls if controls is not None else SolveControls()
    # 线程数、MIPFocus、seed 都属于（模型, 参数, 线程数）：要相减的两次求解必须一致，溯源记录里有它们。
    model = _new_gurobi_model(
        "joint_multi_period",
        threads=scenario.solver_threads,
        time_limit=scenario.solver_time_limit,
        seed=scenario.solver_seed,
        mip_focus=scenario.mip_focus,
    )
    model.Params.MIPGap = scenario.mip_gap
    logger.info("MIPGap %.4f, MIPFocus %d, Seed %d", scenario.mip_gap, scenario.mip_focus, scenario.solver_seed)

    idx = build_model_index(prepared, scenario, assumptions)
    year_payloads: list[YearPayload] = [
        add_year_block(model, prepared, scenario, assumptions, idx, years, year_index, state)
        for year_index in range(len(years))
    ]
    add_inter_period_constraints(
        model, year_payloads, scenario, idx.plant_count, idx.edge_count, len(prepared.industry.hubs),
    )
    add_capacity_vintages(model, year_payloads, assumptions)

    first_year_data = year_payloads[0].year_data
    edge_base_stock = np.asarray(first_year_data.edge_base_stock_mtpa, dtype=np.float64)
    edge_max_new_total = np.asarray(first_year_data.edge_max_new_mtpa, dtype=np.float64)
    for year_position, payload in enumerate(year_payloads):
        add_capacity_constraints(
            model, payload, year_payloads, year_position, assumptions, state,
            edge_base_stock, edge_max_new_total, idx.edge_count, idx.storage_count,
        )
        add_year_costs(
            model, payload, year_payloads, year_position, prepared, scenario, assumptions,
            idx.plant_count, idx.edge_count, idx.storage_count,
        )

    add_retired_rebuild_offset(model, year_payloads, scenario, assumptions, idx.plant_count)
    _add_salvage_credit(year_payloads, scenario, assumptions, _COST_SCALE)
    model.setObjective(gp.quicksum(payload.objective_expr for payload in year_payloads), GRB.MINIMIZE)

    if controls.relax:
        model.update()
        for var in model.getVars():
            if var.VType != GRB.CONTINUOUS:
                var.VType = GRB.CONTINUOUS
        logger.warning("solving the LP relaxation, not the MIP")
    if controls.start_sol is not None:
        _apply_rounded_start(model, Path(controls.start_sol), assumptions)
    if controls.log_incumbents:
        model.optimize(_incumbent_logger(year_payloads))
    else:
        model.optimize()
    if controls.write_sol is not None and int(_optional_model_attr(model, "SolCount") or 0) > 0:
        model.write(str(controls.write_sol))
        logger.warning("solution written to %s", controls.write_sol)

    status = _extract_solver_status(model)
    solver_quality = _solver_quality(model, status, prepared.input_digest)
    has_solution = bool(solver_quality.get("solution_count") or 0)
    acceptable_status = status in ("optimal", "suboptimal", "solution_limit") or (status == "time_limit" and has_solution)
    if not acceptable_status:
        logger.error("Solver returned status '%s' — results will be zero-filled.", status)
        return {
            "status": status,
            "objective_cny": float("nan"),
            "solver_quality": solver_quality,
            "capex_pathway_indices": tuple(idx.capex_pathway_indices),
            "year_solutions": empty_year_solutions(prepared, idx, year_payloads, status),
        }
    return {
        "status": status,
        "objective_cny": _model_obj_value(model) * _COST_SCALE,
        "solver_quality": solver_quality,
        "capex_pathway_indices": tuple(idx.capex_pathway_indices),
        "year_solutions": extract_year_solutions(prepared, idx, year_payloads, status),
    }
