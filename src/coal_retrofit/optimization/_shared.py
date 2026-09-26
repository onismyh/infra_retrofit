from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from .industry_inputs import IndustryInputs


PATHWAY_INDEX = {name: index for index, name in enumerate(PATHWAYS)}
# 目标函数以十亿元计，缩小系数范围；成本表达式进目标前除以它，提取时乘回。
_COST_SCALE = 1e9
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
    # 各流域、各规划年的官方用水总量控制指标；文件尚未构建时为空。
    # 只在有水约束（`scenario.water_mode` 不为 no_water）时读取。
    water_basin_caps: pd.DataFrame
    network: RuntimeNetwork
    available_ammonia_years: tuple[int, ...]
    # 工业点源，与煤电同在一个目标函数里决策。
    industry: IndustryInputs
    # 部门残余排放上限：sector_group, planning_year, cap_fraction_of_2030。
    sector_targets: pd.DataFrame
    # 工业氢路线到共享绿氨节点的候选链路：year, hub_id, ammonia_node_id, distance_km, lcoh_usd_per_kg。
    industry_h2_links: pd.DataFrame
    # 本次实际读取的输入目录（`ProjectPaths.inputs_dir`），供溯源摘要按真实文件计算。
    inputs_dir: Path


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
    model.Params.MIPGap = 0.01    # 1% gap，达到发表质量
    model.Params.MIPFocus = 1     # 侧重于尽快找到好的可行解
    model.Params.Presolve = 2     # 激进预求解
    model.Params.TimeLimit = time_limit
    model.Params.Heuristics = 0.3        # 加大启发式力度，以得到更好的当前最优可行解（incumbent）
    model.Params.NumericFocus = 1        # 针对大系数范围加强数值处理
    model.Params.ScaleFlag = 2           # 对大系数模型做激进缩放
    if threads > 0:
        model.Params.Threads = threads
    # 仅用于诊断，并且有意不作为模型参数：改变 seed 会改变搜索路径，同时模型、参数与可行集
    # 保持逐位相同。这是度量本模型简并度的唯一办法。Gurobi 在（模型、参数、线程数）固定时
    # 是确定性的，所以不换 seed 重解什么也测不到；而扰动任何物理输入，测到的是物理而不是
    # 求解器的随意性——这正是 v9 的 `*_nobias` 那个"地板"犯的错（Hai 的偏差因子为 0.412，意味着
    # 在这个起约束作用的流域里，关掉偏差校正会把其可用量乘以 2.43x）。默认不设置，
    # 因此默认行为不变。
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
    """年金系数：按贴现率计，interval_years 年内每年 1 单位的 NPV。"""
    n = max(1, interval_years)
    if rate <= 1e-9:
        return float(n)
    return (1.0 - (1.0 + rate) ** (-n)) / rate


def _discount_factor(year: int, base_year: int, rate: float) -> float:
    """现值折现因子：1 / (1 + r)^(t - t0)。"""
    return 1.0 / (1.0 + rate) ** max(0, year - base_year)
