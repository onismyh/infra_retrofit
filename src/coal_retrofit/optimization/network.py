from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ..frames import iter_rows
from ..paths import ProjectPaths
from ..sink_inputs import sink_input_path


@dataclass(frozen=True)
class RuntimeNetwork:
    nodes: pd.DataFrame
    # 列包括：node_id, lon, lat, node_type, plant_id, storage_hub_id, ...

    edges: pd.DataFrame
    # 列包括：edge_id, from_node_id, to_node_id, length_km, offshore_length_km,
    #         existing_corridor_flag, edge_class, source, year_basis

    incidence: np.ndarray
    # 形状：(n_nodes, n_edges)
    # incidence[n, e] = +1.0  若边 e 从节点 n 出发（from_node_id == node）
    # incidence[n, e] = -1.0  若边 e 到达节点 n（to_node_id  == node）
    # incidence[n, e] =  0.0  其他情况

    plant_node_ids: dict[str, str]
    # 映射 plant_id -> node_id，覆盖每个将参与求解的电厂

    storage_node_ids: dict[str, str]
    # 映射 storage_hub_id -> node_id，覆盖每个将参与求解的封存 hub

    industry_node_ids: dict[str, str]
    # 映射工业 hub_id -> node_id，覆盖每个将参与求解的工业 hub


def _build_base_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in iter_rows(nodes):
        graph.add_node(str(row.node_id), lon=float(row.lon), lat=float(row.lat), node_type=str(row.node_type))
    for row in iter_rows(edges):
        edge_id = str(row.edge_id)
        length_km = float(row.length_km)
        existing_flag = int(row.existing_corridor_flag)
        edge_class = str(row.edge_class)
        graph.add_edge(
            str(row.from_node_id),
            str(row.to_node_id),
            edge_id=edge_id,
            length_km=length_km,
            corridor_type=str(row.corridor_type),
            existing_corridor_flag=existing_flag,
            edge_class=edge_class,
            source=str(row.source),
            year_basis=str(row.year_basis),
        )
    return graph


def _terminal_nodes(nodes: pd.DataFrame, id_column: str) -> dict[str, str]:
    """管网节点表里源或汇的 ID -> 节点 ID（该列为空的节点不是这一类）；没有这一列时为空。"""
    if id_column not in nodes.columns:
        return {}
    return {str(getattr(row, id_column)): str(row.node_id) for row in iter_rows(nodes.loc[nodes[id_column].notna()])}


def _require_sinks_where_the_table_puts_them(
    nodes: pd.DataFrame, storages: pd.DataFrame, storage_node_ids: dict[str, str], nodes_path: Path, rebuild: str,
    tol_deg: float = 1e-6,
) -> None:
    """模型给每个参与求解的汇用的节点（*storage_node_ids*）坐标须与汇表的一致（建网时照抄汇表坐标，`builders.network_branches`）。

    汇 id 按序号编（S001 起），不同的汇表（如两种汇变体，`sink_inputs`）用的是同一批 id：换了汇表而没在它上面重建管网，
    汇 id 都找得到也会把汇接到别处（2026-10-16 起查）。节点号重复时与关联矩阵一样取最后一行。
    """
    at_node = nodes.assign(node_id=nodes["node_id"].astype(str)).drop_duplicates("node_id", keep="last").set_index("node_id")
    ids = storages["storage_hub_id"].astype(str)
    placed = at_node.loc[[storage_node_ids[sid] for sid in ids], ["lon", "lat"]].to_numpy(dtype=float)
    table = storages[["longitude", "latitude"]].to_numpy(dtype=float)
    off = np.abs(placed - table).max(axis=1) if len(table) else np.zeros(0)
    moved = ids[~(off <= tol_deg)].tolist()
    if moved:
        more = " …" if len(moved) > 6 else ""
        raise ValueError(
            f"{len(moved)} 个汇在 {nodes_path.name} 里的坐标与汇表不同（{', '.join(moved[:6])}{more}）："
            f"管网不是在这张汇表上建的，重新运行 {rebuild}"
        )


def _require_sources_reach_sinks(
    graph: nx.Graph, sources: dict[str, str], sinks: set[str], inputs_dir: Path
) -> None:
    """每个参与求解的源（煤电 + 工业）都必须沿候选网络到达至少一个汇，否则报错。

    builders 建网时已对每个源做过这项检查（`builders.network_repair._unreached_terminals`）。运行期
    再查一次，拦的是代码与管网输入不配套：2026-09-12 之前的管网（例如 `a303f05` 里仓库根的 v7 输入，
    35 汇 / 923 边）不含工业节点；运行期直连弧去掉后，挂上去的工业点源会有一批到不了任何汇，
    CCS 通路静默不可行，求解却照常完成。

    Args:
        graph: 候选网络。
        sources: 标签（如 "industry H012"）-> 节点 ID。
        sinks: 参与求解的封存汇节点 ID。
        inputs_dir: 管网输入目录，只用于报错信息。
    """
    component = {node: i for i, part in enumerate(nx.connected_components(graph)) for node in part}
    with_sink = {component[s] for s in sinks if s in component}
    stranded = sorted(label for label, node in sources.items() if component.get(node) not in with_sink)
    if stranded:
        more = " …" if len(stranded) > 6 else ""
        raise ValueError(
            f"{len(stranded)} 个参与求解的源到不了任何封存汇（{', '.join(stranded[:6])}{more}）。"
            f"{inputs_dir} 的管网多半不是本代码对应的版本：2026-09-12 重建后的管网把工业点源作为 "
            "industry_hub 节点收进 pipeline_nodes.csv，并保证每个源都可达；v7 输入（a303f05 的仓库根 inputs/）不满足这一点。"
        )


def build_runtime_network(
    paths: ProjectPaths,
    plants: pd.DataFrame,
    storages: pd.DataFrame,
    industry_hubs: pd.DataFrame | None = None,
    sink_variant: str = "",
) -> RuntimeNetwork:
    """读候选管网两张表（汇变体 *sink_variant* 的，`sink_inputs.sink_input_path`），核海上段长度与源汇都在网里且可达，
    建有向关联矩阵。"""
    nodes_path = sink_input_path(paths.inputs_dir, "pipeline_nodes", sink_variant)
    edges_path = sink_input_path(paths.inputs_dir, "pipeline_candidate_edges", sink_variant)
    rebuild = "scripts/build_network_inputs.py" + (f" --sinks {sink_variant}" if sink_variant else "")
    base_nodes = pd.read_csv(nodes_path)
    base_edges = pd.read_csv(edges_path)
    if "offshore_length_km" not in base_edges.columns:
        raise ValueError(
            f"{edges_path} 没有 offshore_length_km 列（每条边的海上段长度，"
            f"2026-10-10 起按它计海上倍率）：先运行 {rebuild} --offshore-only"
        )
    offshore = base_edges["offshore_length_km"].astype(float)
    if not (np.isfinite(offshore).all() and (offshore >= 0.0).all() and (offshore <= base_edges["length_km"] + 1e-3).all()):
        raise ValueError(
            f"{edges_path.name} 的 offshore_length_km 须无缺失、在 0 与 length_km 之间："
            f"重新运行 {rebuild} --offshore-only"
        )
    graph = _build_base_graph(base_nodes, base_edges)
    nodes = base_nodes.copy()
    # 源与汇都必须已是管网输入里的节点（`scripts/build_network_inputs.py` 把煤电 hub、工业 hub 与汇一起建网）。
    # 2026-10-13 前这里给缺节点的源与汇补一条接最近节点的运行期支线，v9.2 输入用不到，删去。
    plant_node_ids = _terminal_nodes(base_nodes, "plant_id")
    storage_node_ids = _terminal_nodes(base_nodes, "storage_hub_id")
    industry_node_ids = _terminal_nodes(base_nodes, "industry_hub_id")
    missing = [
        *(f"plant {pid}" for pid in plants["plant_id"].astype(str) if pid not in plant_node_ids),
        *(f"storage {sid}" for sid in storages["storage_hub_id"].astype(str) if sid not in storage_node_ids),
        *(f"industry {hid}" for hid in (industry_hubs["hub_id"].astype(str) if industry_hubs is not None else [])
          if hid not in industry_node_ids),
    ]
    if missing:
        more = " …" if len(missing) > 6 else ""
        raise ValueError(
            f"{len(missing)} 个参与求解的源或汇不在 {nodes_path} 里"
            f"（{', '.join(missing[:6])}{more}）：管网输入与厂址、汇、工业 hub 表不配套，重新运行 {rebuild}"
        )
    _require_sinks_where_the_table_puts_them(base_nodes, storages, storage_node_ids, nodes_path, rebuild)

    solved_sources = {f"plant {pid}": plant_node_ids[pid] for pid in plants["plant_id"].astype(str)}
    if industry_hubs is not None and len(industry_hubs):
        solved_sources.update(
            {f"industry {hid}": industry_node_ids[hid] for hid in industry_hubs["hub_id"].astype(str)}
        )
    _require_sources_reach_sinks(
        graph,
        sources=solved_sources,
        sinks={storage_node_ids[sid] for sid in storages["storage_hub_id"].astype(str)},
        inputs_dir=paths.inputs_dir,
    )

    edges = base_edges.drop_duplicates(subset=["edge_id"]).reset_index(drop=True)

    # 构造有向关联矩阵：形状 (n_nodes, n_edges)
    # incidence[n, e] = +1 若边 e 从节点 n 出发
    # incidence[n, e] = -1 若边 e 到达节点 n
    final_edges = edges.reset_index(drop=True)
    final_nodes = nodes.reset_index(drop=True)
    node_idx_map = {str(row.node_id): i for i, row in enumerate(iter_rows(final_nodes))}
    n_nodes = len(final_nodes)
    n_edges = len(final_edges)
    incidence = np.zeros((n_nodes, n_edges), dtype=np.float64)
    for e_idx, edge in enumerate(iter_rows(final_edges)):
        from_id = str(edge.from_node_id)
        to_id = str(edge.to_node_id)
        if from_id not in node_idx_map:
            raise ValueError(
                f"Edge {getattr(edge, 'edge_id', e_idx)} references unknown from_node_id={from_id!r}"
            )
        if to_id not in node_idx_map:
            raise ValueError(
                f"Edge {getattr(edge, 'edge_id', e_idx)} references unknown to_node_id={to_id!r}"
            )
        incidence[node_idx_map[from_id], e_idx] = 1.0
        incidence[node_idx_map[to_id], e_idx] = -1.0

    return RuntimeNetwork(
        nodes=final_nodes,
        edges=final_edges,
        incidence=incidence,
        plant_node_ids=plant_node_ids,
        storage_node_ids=storage_node_ids,
        industry_node_ids=industry_node_ids,
    )
