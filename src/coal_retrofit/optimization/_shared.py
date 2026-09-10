from __future__ import annotations

import os

from dataclasses import dataclass

import numpy as np
import pandas as pd
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    gp = None
    GRB = None

from .network import RuntimeNetwork
from .scenario import PATHWAYS


PATHWAY_INDEX = {name: index for index, name in enumerate(PATHWAYS)}
GUROBI_STATUS_NAMES = {
    1: "loaded",
    2: "optimal",
    3: "infeasible",
    4: "infeasible_or_unbounded",
    5: "unbounded",
    6: "cutoff",
    7: "iteration_limit",
    8: "node_limit",
    9: "time_limit",
    10: "solution_limit",
    11: "interrupted",
    12: "numeric",
    13: "suboptimal",
    14: "inprogress",
    15: "user_objective_limit",
}


@dataclass(frozen=True)
class PreparedInputs:
    plants: pd.DataFrame
    storages: pd.DataFrame
    biomass: pd.DataFrame
    biomass_links: pd.DataFrame
    ammonia_supply: pd.DataFrame
    ammonia_links: pd.DataFrame
    water_nodes: pd.DataFrame
    water_links: pd.DataFrame
    water_availability: pd.DataFrame
    # Official 用水总量控制指标 per basin per planning year, empty when the file has not
    # been built. Only read when `assumptions.water_budget == 'official_quota'`.
    water_basin_caps: pd.DataFrame
    network: RuntimeNetwork
    available_ammonia_years: tuple[int, ...]
    # Industrial hubs as decision agents, None unless `assumptions.include_industry`.
    # Defaulted so that with industry off `PreparedInputs` is constructed exactly as it
    # was before industry existed.
    industry: object | None = None
    # Per-sector residual caps (fractions of each group's own 2030 baseline); empty unless
    # `scenario.sector_target_source` is set.
    sector_targets: pd.DataFrame | None = None
    # (year, hub_id, ammonia_node_id, distance_km, lcoh_usd_per_kg): industrial hydrogen links
    # onto the shared green-ammonia nodes. Empty unless industry is on.
    industry_h2_links: pd.DataFrame | None = None


@dataclass
class SolveState:
    edge_added_stock_mtpa: np.ndarray
    remaining_storage_mt: np.ndarray

    def clone(self) -> "SolveState":
        return SolveState(
            edge_added_stock_mtpa=self.edge_added_stock_mtpa.copy(),
            remaining_storage_mt=self.remaining_storage_mt.copy(),
        )


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise RuntimeError("gurobipy is required to run the optimization model.")


def _new_gurobi_model(name: str, threads: int = 0, time_limit: int = 36000):
    _require_gurobi()
    try:
        model = gp.Model(name)
    except gp.GurobiError as exc:  # pragma: no cover
        raise RuntimeError("Could not initialize Gurobi. Please verify the local license environment.") from exc
    model.Params.OutputFlag = 1
    model.Params.MIPGap = 0.01    # 1% gap for publication quality
    model.Params.MIPFocus = 1     # Focus on finding good feasible solutions quickly
    model.Params.Presolve = 2     # Aggressive presolve
    model.Params.TimeLimit = time_limit
    model.Params.Heuristics = 0.3        # More heuristic effort for better incumbents
    model.Params.NumericFocus = 1        # Better numeric handling for large coefficient ranges
    model.Params.ScaleFlag = 2           # Aggressive scaling for large coefficient models
    if threads > 0:
        model.Params.Threads = threads
    # Diagnostic only, and deliberately NOT a model parameter: varying the seed changes the
    # search path while leaving the model, the parameters and the feasible set bit-identical.
    # That is the only way to measure this model's degeneracy. Gurobi is deterministic for a
    # fixed (model, params, threads), so re-solving without changing the seed measures nothing,
    # and perturbing any physical input measures physics rather than solver arbitrariness --
    # which is exactly the error the `*_nobias` "floor" made (bias factor 0.412 in Hai means
    # that switching bias correction off multiplies its availability by 2.43x, in the basin
    # that binds). Unset by default, so default behaviour is unchanged.
    seed = os.environ.get("COAL_RETROFIT_GUROBI_SEED")
    if seed:
        model.Params.Seed = int(seed)
    return model


def _nearest_year(target_year: int, available_years: tuple[int, ...]) -> int:
    if not available_years:
        return target_year
    if target_year in available_years:
        return target_year
    return min(available_years, key=lambda year: abs(year - target_year))


def _var_value(var, shape: tuple[int, ...] | int) -> np.ndarray:
    try:
        values = np.asarray(var.X, dtype=np.float64)
    except (AttributeError, Exception) if gp is None else (AttributeError, gp.GurobiError):
        return np.zeros(shape, dtype=np.float64)
    return values.reshape(shape)


def _expr_value(expr, default: float = 0.0) -> float:
    try:
        return float(expr.getValue())
    except (AttributeError, Exception) if gp is None else (AttributeError, gp.GurobiError):
        return default


def _model_obj_value(model, default: float = 0.0) -> float:
    try:
        return float(model.ObjVal)
    except (AttributeError, Exception) if gp is None else (AttributeError, gp.GurobiError):
        return default


def _var_scalar_value(var, default: float = 0.0) -> float:
    try:
        return float(var.X)
    except (AttributeError, Exception) if gp is None else (AttributeError, gp.GurobiError):
        return default


def _extract_solver_status(model) -> str:
    try:
        return GUROBI_STATUS_NAMES.get(int(model.Status), f"unknown_{model.Status}")
    except Exception:
        return "error"


def _year_objective_weight(interval_years: int, rate: float = 0.0) -> float:
    """Annuity factor: NPV of 1 unit/year for interval_years at discount rate."""
    n = max(1, interval_years)
    if rate <= 1e-9:
        return float(n)
    return (1.0 - (1.0 + rate) ** (-n)) / rate


def _discount_factor(year: int, base_year: int, rate: float) -> float:
    """Present-value discount factor: 1 / (1 + r)^(t - t0)."""
    return 1.0 / (1.0 + rate) ** max(0, year - base_year)
