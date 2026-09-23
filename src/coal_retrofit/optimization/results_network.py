"""CO2 管网与封存结果表：逐边容量与流量、逐汇注入与剩余容量、逐边流向。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import PreparedInputs, SolveState
from .scenario import OptimizationAssumptions


def _build_edge_table(
    prepared: PreparedInputs,
    year: int,
    edge_flow_mtpa: np.ndarray,
    build_edge: np.ndarray,
    new_cap_mtpa: np.ndarray,
    state_before: SolveState,
    assumptions: OptimizationAssumptions,
    pipe_count: np.ndarray | None = None,
    pipe_tiers: tuple[float, ...] = (),
) -> pd.DataFrame:
    edges = prepared.network.edges.copy()
    edges["year"] = year
    edges["available_stock_before_mtpa"] = (
        edges["existing_corridor_flag"].fillna(0).astype(float) * assumptions.existing_corridor_capacity_mtpa
        + state_before.edge_added_stock_mtpa
    )
    edges["edge_flow_mtpa"] = edge_flow_mtpa
    edges["build_selected"] = np.rint(np.asarray(build_edge, dtype=np.float64)).astype(int)
    edges["new_capacity_mtpa"] = new_cap_mtpa
    edges["total_capacity_mtpa"] = edges["available_stock_before_mtpa"] + edges["new_capacity_mtpa"]
    edges["edge_active"] = ((edges["edge_flow_mtpa"] > 1e-6) | (edges["new_capacity_mtpa"] > 1e-6)).astype(int)
    edges["num_pipe_new"] = edges["new_capacity_mtpa"] / assumptions.standard_pipe_capacity_mtpa
    edges["num_pipe_stock"] = edges["total_capacity_mtpa"] / assumptions.standard_pipe_capacity_mtpa
    # Whole pipes laid this year by diameter tier, e.g. "2x2|1x20" -- what was actually built.
    if pipe_count is not None and len(pipe_tiers):
        counts = np.rint(np.asarray(pipe_count, dtype=np.float64)).astype(int)
        edges["pipes_new_by_tier"] = [
            "|".join(f"{int(counts[e, k])}x{pipe_tiers[k]:g}" for k in range(len(pipe_tiers)) if counts[e, k] > 0)
            for e in range(len(edges))
        ]
    else:
        edges["pipes_new_by_tier"] = ""
    return edges[
        [
            "year",
            "edge_id",
            "from_node_id",
            "to_node_id",
            "length_km",
            "corridor_type",
            "existing_corridor_flag",
            "edge_class",
            "source",
            "year_basis",
            "available_stock_before_mtpa",
            "build_selected",
            "edge_active",
            "edge_flow_mtpa",
            "new_capacity_mtpa",
            "total_capacity_mtpa",
            "num_pipe_new",
            "num_pipe_stock",
            "pipes_new_by_tier",
        ]
    ]


def _build_storage_table(
    prepared: PreparedInputs,
    year: int,
    storage_use_mtpa: np.ndarray,
    state_before: SolveState,
    interval_years: int,
    injectivity_mtpa: np.ndarray | None = None,
) -> pd.DataFrame:
    table = prepared.storages.copy()
    table["year"] = year
    if injectivity_mtpa is not None:
        # The year's deployed rate (buildable rate x ramp), which is what the constraint used.
        table["injectivity_mtpa"] = np.asarray(injectivity_mtpa, dtype=np.float64)
    table["storage_use_mtpa"] = storage_use_mtpa
    table["remaining_capacity_before_mt"] = state_before.remaining_storage_mt
    table["remaining_capacity_after_mt"] = np.maximum(0.0, state_before.remaining_storage_mt - storage_use_mtpa * interval_years)
    table["injectivity_utilization"] = np.where(
        table["injectivity_mtpa"].astype(float) > 0,
        storage_use_mtpa / table["injectivity_mtpa"].astype(float),
        0.0,
    )
    return table[
        [
            "year",
            "storage_hub_id",
            "province",
            "injectivity_mtpa",
            "available_capacity_mt",
            "remaining_capacity_before_mt",
            "storage_use_mtpa",
            "remaining_capacity_after_mt",
            "injectivity_utilization",
            "source",
            "year_basis",
        ]
    ]


def _build_co2_flow_direction_table(
    prepared: PreparedInputs,
    year: int,
    co2_flow_fwd: np.ndarray,
    co2_flow_bwd: np.ndarray,
) -> pd.DataFrame:
    """CO2 flow direction on each active edge: resolved to source→sink with magnitude."""
    edges = prepared.network.edges
    fwd = np.asarray(co2_flow_fwd, dtype=np.float64)
    bwd = np.asarray(co2_flow_bwd, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for i in range(len(edges)):
        net = fwd[i] - bwd[i]
        total = fwd[i] + bwd[i]
        if total < 1e-9:
            continue
        edge = edges.iloc[i]
        if net >= 0:
            source_node, sink_node = str(edge["from_node_id"]), str(edge["to_node_id"])
        else:
            source_node, sink_node = str(edge["to_node_id"]), str(edge["from_node_id"])
        rows.append({
            "year": year, "edge_id": str(edge["edge_id"]),
            "source_node": source_node, "sink_node": sink_node,
            "flow_fwd_mtpa": float(fwd[i]), "flow_bwd_mtpa": float(bwd[i]),
            "net_flow_mtpa": float(abs(net)), "length_km": float(edge["length_km"]),
            "corridor_type": str(edge.get("corridor_type", "")),
            "edge_class": str(edge.get("edge_class", "")),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["year", "edge_id", "source_node", "sink_node", "flow_fwd_mtpa", "flow_bwd_mtpa", "net_flow_mtpa", "length_km", "corridor_type", "edge_class"])
