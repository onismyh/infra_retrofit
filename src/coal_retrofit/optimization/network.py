from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from ..paths import ProjectPaths
from ..spatial import geodesic_length_km
from .scenario import OptimizationAssumptions, OptimizationScenario


@dataclass(frozen=True)
class RuntimeNetwork:
    nodes: pd.DataFrame
    # columns include: node_id, lon, lat, node_type, plant_id, storage_hub_id, ...

    edges: pd.DataFrame
    # columns include: edge_id, from_node_id, to_node_id, length_km,
    #                  existing_corridor_flag, edge_class, source, year_basis, capex_multiplier

    incidence: np.ndarray
    # shape: (n_nodes, n_edges)
    # incidence[n, e] = +1.0  if edge e departs from node n (from_node_id == node)
    # incidence[n, e] = -1.0  if edge e arrives at node n  (to_node_id  == node)
    # incidence[n, e] =  0.0  otherwise

    plant_node_ids: dict[str, str]
    # maps plant_id  -> node_id  for every plant that will be solved

    storage_node_ids: dict[str, str]
    # maps storage_hub_id -> node_id  for every storage hub that will be solved

    industry_node_ids: dict[str, str] = field(default_factory=dict)
    # maps industry hub_id -> node_id, EMPTY unless `include_industry` is on. Defaulted so
    # that with industry off the network is constructed exactly as it was before industry
    # existed -- the runs solved up to 2026-09-08 have to stay reproducible.


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

    # Industrial hubs join the SAME graph as the coal hubs: their captured CO2 competes for
    # the same edge capacity and the same sinks. Identical branch rule, so neither source group
    # gets a connection advantage the other does not have.
    #
    # The nearest node can be a coal-plant or storage node, because `nodes` grows as branches
    # are added. That is not a defect: a node's balance fixes its NET outflow to its own
    # capture (or, at a sink, to its own injection), so a transiting industrial flow raises the
    # outflow by exactly what it brought in. Mass is conserved, and sharing a collection point
    # is precisely what "shared infrastructure" means here.
    existing_industry_nodes: dict[str, str] = {}
    if industry_hubs is not None and len(industry_hubs):
        for row in industry_hubs.itertuples(index=False):
            runtime_node_id = f"industry::{row.hub_id}"
            graph.add_node(runtime_node_id, lon=float(row.longitude), lat=float(row.latitude), node_type="industry_hub")
            nearest_node_id, length_km = _nearest_corridor_node(nodes, float(row.longitude), float(row.latitude))
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

    # Build plant_node_ids and storage_node_ids from the existing maps
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

    # Build directed incidence matrix: shape (n_nodes, n_edges)
    # incidence[n, e] = +1 if edge e departs from node n
    # incidence[n, e] = -1 if edge e arrives at node n
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
