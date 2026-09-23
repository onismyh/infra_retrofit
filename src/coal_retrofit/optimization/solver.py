"""多期联合 MILP 的编排：建索引 → 逐年变量与约束 → 跨期约束 → 容量与成本 → 残值 → 求解 → 提取。

各步的实现在 `model_index` / `model_year` / `model_linking` / `model_costs` / `salvage` /
`solver_extract`；本文件只负责顺序与求解器参数。变量与约束的创建顺序决定 Gurobi 指纹，
改动顺序会使新旧结果不可比。
"""
from __future__ import annotations

import logging
import os
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
from .model_linking import add_capacity_constraints, add_inter_period_constraints
from .model_year import add_year_block
from .salvage import _add_salvage_credit
from .scenario import OptimizationAssumptions, OptimizationScenario
from .solver_extract import empty_year_solutions, extract_year_solutions
from .solver_provenance import _optional_model_attr, _solver_quality
from .solver_start import _apply_rounded_start, _incumbent_logger
from .year_types import YearPayload

logger = logging.getLogger(__name__)


def _solve_joint_multi_period(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    years: tuple[int, ...],
    state: SolveState,
) -> dict[str, object]:
    if scenario.solve_mode != "joint":
        raise ValueError(f"Unsupported solve_mode {scenario.solve_mode!r}: only 'joint' is implemented.")
    model = _new_gurobi_model(
        "joint_multi_period",
        threads=scenario.solver_threads,
        time_limit=scenario.solver_time_limit,
    )
    model.Params.MIPGap = scenario.mip_gap
    logger.info("MIPGap set to %.4f", scenario.mip_gap)
    # MIPFocus 从环境变量读，整个求解批次统一设置。它属于 (模型, 参数, 线程) 元组，
    # 要相减的两次求解必须一致，溯源记录里有它。
    _focus = os.environ.get("COAL_RETROFIT_MIPFOCUS")
    if _focus:
        model.Params.MIPFocus = int(_focus)
        logger.info("MIPFocus set to %s", _focus)

    idx = build_model_index(prepared, scenario, assumptions)
    year_payloads: list[YearPayload] = [
        add_year_block(model, prepared, scenario, assumptions, idx, years, year_index, state)
        for year_index in range(len(years))
    ]
    add_inter_period_constraints(
        model, year_payloads, scenario, idx.plant_count, idx.edge_count, len(prepared.industry.hubs),
    )

    first_year_data = year_payloads[0].year_data
    edge_base_stock = np.asarray(first_year_data.edge_base_stock_mtpa, dtype=np.float64)
    edge_max_new_total = np.asarray(first_year_data.edge_max_new_mtpa, dtype=np.float64)
    for year_position, payload in enumerate(year_payloads):
        add_capacity_constraints(
            model, payload, year_payloads, year_position, scenario, assumptions, state,
            edge_base_stock, edge_max_new_total, idx.edge_count, idx.storage_count,
        )
        add_year_costs(
            model, payload, year_payloads, year_position, prepared, scenario, assumptions,
            idx.retirement_years, idx.plant_count, idx.edge_count, idx.storage_count,
        )

    _add_salvage_credit(year_payloads, scenario, assumptions, _COST_SCALE)
    model.setObjective(gp.quicksum(payload.objective_expr for payload in year_payloads), GRB.MINIMIZE)

    # 以下环境变量只用于诊断或热启动，改搜索路径不改模型；要相减的求解必须用同一套。
    if os.environ.get("COAL_RETROFIT_LP_RELAX"):
        model.update()
        for var in model.getVars():
            if var.VType != GRB.CONTINUOUS:
                var.VType = GRB.CONTINUOUS
        logger.warning("COAL_RETROFIT_LP_RELAX set: solving the LP relaxation, not the MIP")
    start_sol = os.environ.get("COAL_RETROFIT_START_SOL")
    if start_sol:
        _apply_rounded_start(model, Path(start_sol), assumptions)
    if os.environ.get("COAL_RETROFIT_LOG_INCUMBENTS"):
        model.optimize(_incumbent_logger(year_payloads))
    else:
        model.optimize()
    write_sol = os.environ.get("COAL_RETROFIT_WRITE_SOL")
    if write_sol and int(_optional_model_attr(model, "SolCount") or 0) > 0:
        model.write(write_sol)
        logger.warning("solution written to %s", write_sol)

    status = _extract_solver_status(model)
    solver_quality = _solver_quality(model, status, prepared.inputs_dir)
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
