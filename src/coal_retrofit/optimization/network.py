from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ..constants import NETWORK_DETOUR_FACTOR
from ..paths import ProjectPaths
from ..spatial import geodesic_length_km
from .scenario import OptimizationAssumptions, OptimizationScenario

# 模型计价的最短支线。有三个工业 hub 正压在走廊节点上（大地线距离 0.000 km），
# 它们的零长度支线在 2030 年被"免费"按 20 Mtpa 建成。
# 以一公里的厂内管道为下限。
MIN_BRANCH_LENGTH_KM = 1.0


def _routed_branch_km(straight_km: float) -> float:
    """支线直线长度 -> 路由长度：先乘绕行系数，再套下限。

    CLAUDE.md 1.3：每条直线候选都是 `haversine x 1.136`。运行期支线（电厂、封存、工业）
    原先用的是原始大地线距离；`builders/network.py` 建的三角化边与直连边已带该系数。
    """
    return max(float(straight_km) * NETWORK_DETOUR_FACTOR, MIN_BRANCH_LENGTH_KM)


@dataclass(frozen=True)
class RuntimeNetwork:
    nodes: pd.DataFrame
    # 列包括：node_id, lon, lat, node_type, plant_id, storage_hub_id, ...

    edges: pd.DataFrame
    # 列包括：edge_id, from_node_id, to_node_id, length_km,
    #         existing_corridor_flag, edge_class, source, year_basis, capex_multiplier

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


def _build_base_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(str(row.node_id), lon=float(row.lon), lat=float(row.lat), node_type=str(row.node_type))
    for row in edges.itertuples(index=False):
        edge_id = str(row.edge_id)
        length_km = float(row.length_km)
        existing_flag = int(row.existing_corridor_flag)
        edge_class = str(row.edge_class)
        weight = length_km
        if existing_flag and edge_class == "existing_main_corridor":
            weight = length_km * scenario.corridor_prior_strength
        graph.add_edge(
            str(row.from_node_id),
            str(row.to_node_id),
            edge_id=edge_id,
            length_km=length_km,
            weight=weight,
            corridor_type=str(row.corridor_type),
            existing_corridor_flag=existing_flag,
            edge_class=edge_class,
            source=str(row.source),
            year_basis=str(row.year_basis),
        )
    return graph


def _haversine_km(lon: float, lat: float, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    lon0, lat0 = np.radians(lon), np.radians(lat)
    lons_r, lats_r = np.radians(lons), np.radians(lats)
    a = np.sin((lats_r - lat0) / 2.0) ** 2 + np.cos(lat0) * np.cos(lats_r) * np.sin((lons_r - lon0) / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _nearest_corridor_node(nodes: pd.DataFrame, lon: float, lat: float) -> tuple[str, float]:
    best_id = ""
    best_distance = float("inf")
    for row in nodes.itertuples(index=False):
        distance = geodesic_length_km([(lon, lat), (float(row.lon), float(row.lat))])
        if distance < best_distance:
            best_id = str(row.node_id)
            best_distance = float(distance)
    if not best_id:
        raise ValueError("Could not find corridor node for runtime connector.")
    return best_id, best_distance


def _append_runtime_edge(
    edge_rows: list[dict[str, object]],
    graph: nx.Graph,
    edge_id: str,
    from_node_id: str,
    to_node_id: str,
    length_km: float,
    edge_class: str,
    source: str,
    year_basis: str,
    capex_multiplier: float,
) -> None:
    edge_rows.append(
        {
            "edge_id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "length_km": round(length_km, 3),
            "corridor_type": "runtime",
            "existing_corridor_flag": 0,
            "edge_class": edge_class,
            "source": source,
            "year_basis": year_basis,
            "capex_multiplier": capex_multiplier,
        }
    )
    graph.add_edge(
        from_node_id,
        to_node_id,
        edge_id=edge_id,
        length_km=length_km,
        weight=length_km,
        corridor_type="runtime",
        existing_corridor_flag=0,
        edge_class=edge_class,
        source=source,
        year_basis=year_basis,
    )


def _require_sources_reach_sinks(
    graph: nx.Graph, sources: dict[str, str], sinks: set[str], inputs_dir: Path
) -> None:
    """每个参与求解的源（煤电 + 工业）都必须沿候选网络到达至少一个汇，否则报错。

    builders 建网时已对每个源做过这项检查（`builders.network_repair._unreached_terminals`）。运行期
    再查一次，拦的是代码与管网输入不配套：2026-09-12 之前的管网（例如仓库根的 v7 输入，
    35 汇 / 923 边）不含工业节点；运行期直连弧去掉后，挂上去的工业点源会有一批到不了任何汇，
    CCS 通路静默不可行，求解却照常完成。

    Args:
        graph: 含运行期接入边的完整候选网络。
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
            "industry_hub 节点收进 pipeline_nodes.csv，并保证每个源都可达；仓库根的 v7 输入不满足这一点。"
        )


def build_runtime_network(
    paths: ProjectPaths,
    plants: pd.DataFrame,
    storages: pd.DataFrame,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    industry_hubs: pd.DataFrame | None = None,
) -> RuntimeNetwork:
    base_nodes = pd.read_csv(paths.inputs_dir / "pipeline_nodes.csv")
    base_edges = pd.read_csv(paths.inputs_dir / "pipeline_candidate_edges.csv")
    graph = _build_base_graph(base_nodes, base_edges, scenario, assumptions)
    nodes = base_nodes.copy()
    runtime_edge_rows: list[dict[str, object]] = []
    existing_plant_nodes = {}
    existing_storage_nodes = {}
    if "plant_id" in base_nodes.columns:
        existing_plant_nodes = {
            str(row.plant_id): str(row.node_id)
            for row in base_nodes.loc[base_nodes["plant_id"].notna()].itertuples(index=False)
        }
    if "storage_hub_id" in base_nodes.columns:
        existing_storage_nodes = {
            str(row.storage_hub_id): str(row.node_id)
            for row in base_nodes.loc[base_nodes["storage_hub_id"].notna()].itertuples(index=False)
        }

    for row in plants.itertuples(index=False):
        if str(row.plant_id) in existing_plant_nodes:
            continue
        runtime_node_id = f"plant::{row.plant_id}"
        graph.add_node(runtime_node_id, lon=float(row.centroid_longitude), lat=float(row.centroid_latitude), node_type="plant")
        nearest_node_id, length_km = _nearest_corridor_node(nodes, float(row.centroid_longitude), float(row.centroid_latitude))
        length_km = _routed_branch_km(length_km)
        _append_runtime_edge(
            edge_rows=runtime_edge_rows,
            graph=graph,
            edge_id=f"edge_runtime_plant_{row.plant_id}",
            from_node_id=runtime_node_id,
            to_node_id=nearest_node_id,
            length_km=length_km,
            edge_class="runtime_plant_branch",
            source="runtime_short_link_rule",
            year_basis="runtime",
            capex_multiplier=assumptions.branch_capex_multiplier,
        )
        nodes = pd.concat(
            [nodes, pd.DataFrame([{"node_id": runtime_node_id, "lon": float(row.centroid_longitude), "lat": float(row.centroid_latitude), "node_type": "plant", "degree": 1, "source": "runtime_short_link_rule", "year_basis": "runtime"}])],
            ignore_index=True,
        )
        existing_plant_nodes[str(row.plant_id)] = runtime_node_id

    storage_nodes = storages
    for row in storage_nodes.itertuples(index=False):
        if str(row.storage_hub_id) in existing_storage_nodes:
            continue
        runtime_node_id = f"storage::{row.storage_hub_id}"
        graph.add_node(runtime_node_id, lon=float(row.longitude), lat=float(row.latitude), node_type="storage_hub")
        nearest_node_id, length_km = _nearest_corridor_node(nodes, float(row.longitude), float(row.latitude))
        length_km = _routed_branch_km(length_km)
        _append_runtime_edge(
            edge_rows=runtime_edge_rows,
            graph=graph,
            edge_id=f"edge_runtime_storage_{row.storage_hub_id}",
            from_node_id=runtime_node_id,
            to_node_id=nearest_node_id,
            length_km=length_km,
            edge_class="runtime_storage_branch",
            source="runtime_short_link_rule",
            year_basis="runtime",
            capex_multiplier=assumptions.branch_capex_multiplier,
        )
        nodes = pd.concat(
            [nodes, pd.DataFrame([{"node_id": runtime_node_id, "lon": float(row.longitude), "lat": float(row.latitude), "node_type": "storage_hub", "degree": 1, "source": "runtime_short_link_rule", "year_basis": "runtime"}])],
            ignore_index=True,
        )
        existing_storage_nodes[str(row.storage_hub_id)] = runtime_node_id

    # 工业 hub 与煤电 hub 进入同一张图：其捕集的 CO2 争用同样的边容量和同样的汇。
    # 支线规则完全相同，因此两类源谁都不会得到对方没有的接入优势。
    #
    # 最近节点可能是煤电厂节点或封存节点，因为 `nodes` 随支线的加入而增长。这不是缺陷：
    # 节点平衡固定的是节点的净流出，令其等于自身捕集量（在汇处则为自身注入量），所以过境的
    # 工业流量使流出恰好增加它带进来的量。质量守恒，而共享一个汇集点正是这里"共享基础设施"
    # 的含义。
    existing_industry_nodes: dict[str, str] = {}
    if "industry_hub_id" in base_nodes.columns:
        existing_industry_nodes = {
            str(row.industry_hub_id): str(row.node_id)
            for row in base_nodes.loc[base_nodes["industry_hub_id"].notna()].itertuples(index=False)
        }
    if industry_hubs is not None and len(industry_hubs):
        # 工业点源已作为 industry_hub 节点进入 inputs/pipeline_*.csv，并与煤电、封存汇
        # 一起做过联合三角剖分与去交叉；到汇的可达性由 builders 的缝合步骤保证。因此这里
        # 不再追加 `runtime_direct_fallback` 直连弧——那是图上唯一还会穿越管网的一类边。
        for row in industry_hubs.itertuples(index=False):
            if str(row.hub_id) in existing_industry_nodes:
                continue
            runtime_node_id = f"industry::{row.hub_id}"
            graph.add_node(runtime_node_id, lon=float(row.longitude), lat=float(row.latitude), node_type="industry_hub")
            nearest_node_id, length_km = _nearest_corridor_node(nodes, float(row.longitude), float(row.latitude))
            length_km = _routed_branch_km(length_km)
            _append_runtime_edge(
                edge_rows=runtime_edge_rows,
                graph=graph,
                edge_id=f"edge_runtime_industry_{row.hub_id}",
                from_node_id=runtime_node_id,
                to_node_id=nearest_node_id,
                length_km=length_km,
                edge_class="runtime_industry_branch",
                source="runtime_short_link_rule",
                year_basis="runtime",
                capex_multiplier=assumptions.branch_capex_multiplier,
            )
            nodes = pd.concat(
                [nodes, pd.DataFrame([{"node_id": runtime_node_id, "lon": float(row.longitude), "lat": float(row.latitude), "node_type": "industry_hub", "degree": 1, "source": "runtime_short_link_rule", "year_basis": "runtime"}])],
                ignore_index=True,
            )
            existing_industry_nodes[str(row.hub_id)] = runtime_node_id

    solved_sources = {f"plant {pid}": existing_plant_nodes[pid] for pid in plants["plant_id"].astype(str)}
    if industry_hubs is not None and len(industry_hubs):
        solved_sources.update(
            {f"industry {hid}": existing_industry_nodes[hid] for hid in industry_hubs["hub_id"].astype(str)}
        )
    _require_sources_reach_sinks(
        graph,
        sources=solved_sources,
        sinks={existing_storage_nodes[sid] for sid in storages["storage_hub_id"].astype(str)},
        inputs_dir=paths.inputs_dir,
    )

    edges = base_edges.copy()
    if "capex_multiplier" not in edges.columns:
        edges["capex_multiplier"] = np.where(
            edges["edge_class"].eq("existing_main_corridor"),
            1.0,
            assumptions.branch_capex_multiplier,
        )
    if runtime_edge_rows:
        edges = pd.concat([edges, pd.DataFrame(runtime_edge_rows)], ignore_index=True, sort=False)

    edges = edges.drop_duplicates(subset=["edge_id"]).reset_index(drop=True)

    # 由现有映射构造 plant_node_ids 与 storage_node_ids
    plant_node_ids = {
        str(plant_id): str(node_id)
        for plant_id, node_id in existing_plant_nodes.items()
    }
    storage_node_ids = {
        str(storage_hub_id): str(node_id)
        for storage_hub_id, node_id in existing_storage_nodes.items()
    }
    industry_node_ids = {
        str(hub_id): str(node_id)
        for hub_id, node_id in existing_industry_nodes.items()
    }

    # 构造有向关联矩阵：形状 (n_nodes, n_edges)
    # incidence[n, e] = +1 若边 e 从节点 n 出发
    # incidence[n, e] = -1 若边 e 到达节点 n
    final_edges = edges.reset_index(drop=True)
    final_nodes = nodes.reset_index(drop=True)
    node_idx_map = {str(row.node_id): i for i, row in enumerate(final_nodes.itertuples(index=False))}
    n_nodes = len(final_nodes)
    n_edges = len(final_edges)
    incidence = np.zeros((n_nodes, n_edges), dtype=np.float64)
    for e_idx, edge in enumerate(final_edges.itertuples(index=False)):
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
