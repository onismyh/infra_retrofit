"""候选管网两张表（`pipeline_nodes.csv`、`pipeline_candidate_edges.csv`）的构建流程 `build_network_tables`：
既有油气干线走廊（`network_corridors`）与煤电、封存、工业点源的接入支线（`network_branches`）合成一张图，加源—汇
三角剖分候选边（`network_triangulation`），剔西藏、去交叉、缝合、并片（`network_repair`），再补源到汇的专线，最后量每条边的
海上段（`network_offshore`）。`--no-corridors` 时只有煤电与封存端点，没有走廊与支线。列序与边记录在 `network_edges`。"""
from __future__ import annotations

import logging

import pandas as pd

from ..artifacts import write_csv
from ..paths import ProjectPaths
from ..sink_inputs import sink_input_path
from .network_branches import build_attachment_tables
from .network_corridors import build_corridor_tables, load_corridor_layer
from .network_edges import EDGE_COLUMNS, NODE_COLUMNS, next_edge_index
from .network_offshore import offshore_length_km
from .network_repair import (
    _drop_edges_over_excluded_region,
    _merge_components,
    _remove_crossing_edges,
    _unreached_terminals,
)
from .network_triangulation import build_direct_sink_edges, build_triangulation_candidate_edges

logger = logging.getLogger(__name__)


def _merge_network_tables(
    main_nodes: pd.DataFrame,
    main_edges: pd.DataFrame,
    branch_nodes: pd.DataFrame,
    branch_edges: pd.DataFrame,
    corridor_degree_additions: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """走廊表与支线表合成一张图：走廊节点的度数加上接入的支线数并据此重定节点类型，两表按 `NODE_COLUMNS` / `EDGE_COLUMNS`
    补齐列并按 id 排序。"""
    nodes = main_nodes.copy()
    if corridor_degree_additions:
        added_degree = nodes["node_id"].map(corridor_degree_additions).fillna(0).astype(int)
        nodes["degree"] = nodes["degree"].astype(int) + added_degree
        nodes["node_type"] = nodes["degree"].map(
            lambda degree: "corridor_junction" if int(degree) > 1 else "corridor_endpoint"
        )

    if not branch_nodes.empty:
        nodes = pd.concat([nodes, branch_nodes], ignore_index=True, sort=False)

    edges = main_edges.copy()
    if not branch_edges.empty:
        edges = pd.concat([edges, branch_edges], ignore_index=True, sort=False)

    for column in NODE_COLUMNS:
        if column not in nodes.columns:
            nodes[column] = pd.NA
    for column in EDGE_COLUMNS:
        if column not in edges.columns:
            edges[column] = pd.NA

    return (
        nodes[list(NODE_COLUMNS)].sort_values(["node_id"]).reset_index(drop=True),
        edges[list(EDGE_COLUMNS)].sort_values(["edge_id"]).reset_index(drop=True),
    )


def build_network_tables(
    paths: ProjectPaths, use_corridors: bool = True, sink_variant: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """候选管网两张表；汇取汇变体 *sink_variant* 的汇表（`sink_inputs`，缺省 `storage_hubs.csv`）。"""
    if use_corridors:
        gas_layer, gas_path, _ = load_corridor_layer(paths, "gas_pipelines.shp", "gas")
        oil_layer, oil_path, _ = load_corridor_layer(paths, "oil_pipeline.shp", "oil")
        main_nodes, main_edges = build_corridor_tables(paths, [(gas_layer, gas_path), (oil_layer, oil_path)])

        corridor_nodes = main_nodes.loc[
            main_nodes["node_type"].isin(["corridor_junction", "corridor_endpoint"]),
            ["node_id", "lon", "lat"],
        ].copy()
        branch_nodes, branch_edges, corridor_degree_additions = build_attachment_tables(
            paths=paths,
            corridor_nodes=corridor_nodes,
            main_edges=main_edges,
            first_node_index=len(main_nodes) + 1,
            first_edge_index=next_edge_index(main_edges),
            sink_variant=sink_variant,
        )
        nodes, edges = _merge_network_tables(main_nodes, main_edges, branch_nodes, branch_edges, corridor_degree_additions)
    else:
        # 只含端点的网络：电厂 + 封存 hub，没有走廊
        nodes, edges = _build_terminal_only_nodes(paths, sink_variant)

    triangulation_edges = build_triangulation_candidate_edges(nodes, edges)
    if not triangulation_edges.empty:
        edges = pd.concat([edges, triangulation_edges], ignore_index=True, sort=False)
        edges = edges.sort_values(["edge_id"]).reset_index(drop=True)
    # 西藏不参与减排：先把穿过西藏的三角剖分候选边去掉，再去交叉。
    edges = _drop_edges_over_excluded_region(paths, nodes, edges)
    # 优先保证平面性，但绝不以断开任何端点为代价
    edges = _remove_crossing_edges(nodes, edges)
    # 专线（源 -> 最近的 k 个汇）在去交叉之后补回，但逐条做相交检验：穿过已有管网的
    # 一律丢弃。既保住"可以为一个源单建一条专线"这个真实选项（2.8 倍造价溢价），
    # 又不会再出现那些横穿全图的直线。
    # 先把去交叉切出来的孤片并回一张网，再补专线（专线要对着合并后的图做相交检验）
    edges = _merge_components(paths, nodes, edges)
    direct_edges = build_direct_sink_edges(nodes, edges)
    if not direct_edges.empty:
        edges = pd.concat([edges, direct_edges], ignore_index=True, sort=False)
        edges = _drop_edges_over_excluded_region(paths, nodes, edges)
        edges = edges.sort_values(["edge_id"]).reset_index(drop=True)
    duplicates = int(edges["edge_id"].duplicated().sum())
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate edge_id values; downstream code keys on edge_id and would "
            f"drop them silently"
        )
    stranded = _unreached_terminals(nodes, edges)
    if stranded:
        raise ValueError(
            f"{len(stranded)} plant nodes still cannot reach a storage hub: {sorted(stranded)[:8]}"
        )
    edges["offshore_length_km"] = offshore_length_km(paths, nodes, edges)
    logger.info(
        "candidate network: %d nodes, %d edges (%s)",
        len(nodes), len(edges),
        ", ".join(f"{k} {v}" for k, v in edges["edge_class"].value_counts().items()),
    )
    return nodes, edges


def _build_terminal_only_nodes(paths: ProjectPaths, sink_variant: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """只用电厂与封存 hub 构建节点表（不含走廊节点）。"""
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
    storage = pd.read_csv(sink_input_path(paths.inputs_dir, "storage_hubs", sink_variant))

    node_rows: list[dict] = []
    node_index = 1
    for _, row in plants.iterrows():
        node_rows.append({
            "node_id": f"node_{node_index:05d}",
            "lon": row["centroid_longitude"],
            "lat": row["centroid_latitude"],
            "node_type": "plant",
            "degree": 0,
            "source": row.get("source", ""),
            "year_basis": row.get("year_basis", ""),
            "plant_id": row["plant_id"],
            "plant_index": row["plant_index"],
            "storage_hub_id": pd.NA,
            "storage_hub_index": pd.NA,
            "province": row.get("province_mode", ""),
        })
        node_index += 1
    for _, row in storage.iterrows():
        node_rows.append({
            "node_id": f"node_{node_index:05d}",
            "lon": row["longitude"],
            "lat": row["latitude"],
            "node_type": "storage_hub",
            "degree": 0,
            "source": row.get("source", ""),
            "year_basis": row.get("year_basis", ""),
            "plant_id": pd.NA,
            "plant_index": pd.NA,
            "storage_hub_id": row["storage_hub_id"],
            "storage_hub_index": row["storage_hub_index"],
            "province": row.get("province", ""),
        })
        node_index += 1

    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(columns=list(EDGE_COLUMNS))
    return nodes, edges


def write_network_inputs(
    paths: ProjectPaths, use_corridors: bool = True, sink_variant: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """建候选管网并写出：缺省 `pipeline_nodes.csv`、`pipeline_candidate_edges.csv`，汇变体写带后缀的两张（`sink_inputs`）。"""
    nodes, edges = build_network_tables(paths, use_corridors=use_corridors, sink_variant=sink_variant)
    write_csv(nodes, sink_input_path(paths.inputs_dir, "pipeline_nodes", sink_variant))
    write_csv(edges, sink_input_path(paths.inputs_dir, "pipeline_candidate_edges", sink_variant))
    return nodes, edges
