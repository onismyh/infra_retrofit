"""源到汇的 CO2 输送量：按节点充分混合的比例追踪，供结果工作簿的输送矩阵（`results_workbook`）。

几个源的 CO2 在管网里汇流以后，哪个源送到哪个汇没有唯一答案。这里按节点充分混合追踪：每个节点流出的 CO2（流向下游
各边的，与在本节点注入封存的）按流入它的各来源的比例分配。先把每对节点间两个方向的流量相抵，再抵消有向环流（环流不改变
任何节点的净流出），余下的流量无环，按拓扑序逐节点算来源构成。节点守恒成立时，每个源分出去的量之和等于它的注入量，
每个汇分到的量之和等于它的封存量。
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

import networkx as nx

logger = logging.getLogger(__name__)

# 不超过它的流量、注入与封存量（Mt/yr，即 1 t/yr）记为零：求解器的可行性容差在 1e-6 量级，这以内的是数值噪声。
FLOW_TOL = 1e-6


def trace_sources_to_sinks(
    arcs: Mapping[tuple[str, str], float], supply: Mapping[str, float], demand: Mapping[str, float]
) -> dict[tuple[str, str], float]:
    """按节点充分混合把各汇的封存量分回各源。

    Args:
        arcs: {(上游节点, 下游节点): 流量}，Mt/yr；同一对节点两个方向都有流量时先相抵。
        supply: 源节点注入管网的量（捕集量），Mt/yr。
        demand: 汇节点流出管网的量（封存量），Mt/yr。

    Returns:
        {(源节点, 汇节点): Mt/yr}，只含正值。
    """
    graph = nx.DiGraph()
    graph.add_nodes_from([*supply, *demand])
    for (u, v), flow in arcs.items():
        net = float(flow) - float(arcs.get((v, u), 0.0))
        if net > FLOW_TOL:
            graph.add_edge(u, v, flow=net)
    cancelled = 0.0
    while True:
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            break
        delta = min(graph.edges[u, v]["flow"] for u, v in cycle)
        for u, v in cycle:
            graph.edges[u, v]["flow"] -= delta
            if graph.edges[u, v]["flow"] <= FLOW_TOL:
                graph.remove_edge(u, v)
        cancelled += delta
    if cancelled > 0.0:
        logger.info("输送追踪：抵消有向环流 %.6g Mt/yr", cancelled)

    composition: dict[str, dict[str, float]] = {}
    traced: dict[tuple[str, str], float] = {}
    for node in nx.topological_sort(graph):
        mix: dict[str, float] = {}
        own = float(supply.get(node, 0.0))
        if own > FLOW_TOL:
            mix[node] = own
        for upstream in graph.predecessors(node):
            flow = graph.edges[upstream, node]["flow"]
            for source, share in composition[upstream].items():
                mix[source] = mix.get(source, 0.0) + flow * share
        total = sum(mix.values())
        composition[node] = {source: amount / total for source, amount in mix.items()} if total > FLOW_TOL else {}
        stored = float(demand.get(node, 0.0))
        if stored > FLOW_TOL:
            for source, share in composition[node].items():
                traced[(source, node)] = traced.get((source, node), 0.0) + stored * share
    return traced
