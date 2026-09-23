"""求解质量与溯源记录：指纹、规模、线程、种子、输入哈希。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ._shared import _COST_SCALE, _model_obj_value


def _optional_model_attr(model, attr_name: str) -> float | int | str | None:
    try:
        return getattr(model, attr_name)
    except Exception:
        return None


def _scale_optional_cost(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * _COST_SCALE
    except (TypeError, ValueError):
        return None


def _solver_quality(model, status: str, inputs_dir: Path) -> dict[str, float | int | str | None]:
    """目标值、界、gap、耗时，外加 `_run_provenance` 的溯源字段。"""
    objective = _model_obj_value(model, default=float("nan"))
    sol_count = _optional_model_attr(model, "SolCount")
    try:
        sol_count_out = int(sol_count) if sol_count is not None else None
    except (TypeError, ValueError):
        sol_count_out = None
    return {
        "status": status,
        "objective_cny": objective * _COST_SCALE,
        "objective_bound_cny": _scale_optional_cost(_optional_model_attr(model, "ObjBound")),
        "mip_gap": _optional_model_attr(model, "MIPGap"),
        "runtime_seconds": _optional_model_attr(model, "Runtime"),
        "node_count": _optional_model_attr(model, "NodeCount"),
        "solution_count": sol_count_out,
        **_run_provenance(model, inputs_dir),
    }


# 摘要覆盖的输入表：前四张定义右端项；后五张决定这次解跑在哪个输入版本上——仓库根
# `inputs/`（v7 管网）与求解树 `_indtree/inputs/`（2026-09-12 重建的管网）恰在这几张上不同。
_DIGEST_FILES = (
    ("water_availability", "water_availability.csv"),
    ("water_nodes", "water_nodes.csv"),
    ("water_links", "water_supply_links.csv"),
    ("plants", "plants.csv"),
    ("pipeline_nodes", "pipeline_nodes.csv"),
    ("pipeline_edges", "pipeline_candidate_edges.csv"),
    ("storage_hubs", "storage_hubs.csv"),
    ("industry_hubs", "industry_hubs.csv"),
    ("water_basin_caps", "water_basin_caps.csv"),
)


def _input_digest(inputs_dir: Path) -> dict:
    """本次求解实际读取的输入表的 SHA-256 前缀，外加输入目录。

    列名与形状已由指纹覆盖，这里补数值的变化。`inputs_dir` 来自 `PreparedInputs`，即
    `prepare_inputs` 真正读的目录；此前按源码位置推仓库根，求解树里的运行会记下仓库根
    文件的摘要。`input_dir` 在仓库内时记相对路径（如 `_indtree/inputs`）。
    """
    repo_root = Path(__file__).resolve().parents[3]
    inputs = Path(inputs_dir).resolve()
    try:
        shown = inputs.relative_to(repo_root).as_posix()
    except ValueError:
        shown = inputs.as_posix()
    out: dict[str, str | None] = {"input_dir": shown}
    for key, name in _DIGEST_FILES:
        try:
            out[f"digest_{key}"] = hashlib.sha256((inputs / name).read_bytes()).hexdigest()[:12]
        except OSError:
            out[f"digest_{key}"] = None
    return out


def _run_provenance(model, inputs_dir: Path) -> dict[str, object]:
    """标识模型与求解，使两个结果可判断是否可比。

    Gurobi 只在 (模型, 参数, 线程数) 三者不变时确定性可复现：`fingerprint` 是模型哈希，
    只换种子的两次求解必须一致；`threads_param` 为 0 表示自动，不算固定；`mip_focus`
    属于参数元组，要相减的两次求解必须相同；输入数值的改动指纹看不到，由 `_input_digest` 补。
    """
    try:
        threads_out = int(model.Params.Threads)
    except Exception:
        threads_out = None
    fingerprint = _optional_model_attr(model, "Fingerprint")
    try:
        fingerprint_out = hex(int(fingerprint) & 0xFFFFFFFF) if fingerprint is not None else None
    except (TypeError, ValueError):
        fingerprint_out = None
    seed_env = os.environ.get("COAL_RETROFIT_GUROBI_SEED")
    return {
        "fingerprint": fingerprint_out,
        "num_vars": _optional_model_attr(model, "NumVars"),
        "num_constrs": _optional_model_attr(model, "NumConstrs"),
        "num_nonzeros": _optional_model_attr(model, "NumNZs"),
        "threads_param": threads_out,
        "threads_pinned": bool(threads_out),
        "seed": int(seed_env) if seed_env else 0,
        "mip_focus": int(os.environ.get("COAL_RETROFIT_MIPFOCUS") or 0),
        **_input_digest(inputs_dir),
        "host_cpu_count": os.cpu_count(),
    }
