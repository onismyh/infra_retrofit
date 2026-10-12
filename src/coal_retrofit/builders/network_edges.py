"""候选管网两张表的列序与边记录工厂。

`NODE_COLUMNS` / `EDGE_COLUMNS` 是 `pipeline_nodes.csv` / `pipeline_candidate_edges.csv` 的列序（`offshore_length_km`
在建网末尾另加）。六处写边记录的地方（既有干线走廊 `network_corridors`、接入支线 `network_branches`、三角剖分候选边与
源到汇的专线 `network_triangulation`、连通性缝合边与连通片合并边 `network_repair`）都走 `make_edge_record`：
`length_km = direct_length_km × tortuosity`，直线边的 `tortuosity` 取绕行系数 `NETWORK_DETOUR_FACTOR`，
走廊边按路由折线的大地线长计、`tortuosity` 为 1。
"""
from __future__ import annotations

import pandas as pd

from ..constants import NETWORK_DETOUR_FACTOR, STATIC_LAYER_YEAR_BASIS

NODE_COLUMNS = (
    "node_id",
    "lon",
    "lat",
    "node_type",
    "degree",
    "source",
    "year_basis",
    "plant_id",
    "plant_index",
    "storage_hub_id",
    "storage_hub_index",
    "industry_hub_id",
    "industry_index",
    "province",
)
EDGE_COLUMNS = (
    "edge_id",
    "feature_index",
    "part_index",
    "from_node_id",
    "to_node_id",
    "length_km",
    "direct_length_km",
    "tortuosity",
    "geometry_wkt",
    "corridor_type",
    "existing_corridor_flag",
    "edge_class",
    "source",
    "year_basis",
)


def make_edge_record(
    edge_id: str,
    feature_index: int,
    from_node_id: str,
    to_node_id: str,
    direct_length_km: float,
    *,
    corridor_type: str,
    edge_class: str,
    source: str,
    geometry_wkt: str | None,
    tortuosity: float = NETWORK_DETOUR_FACTOR,
    part_index: int = 1,
    existing_corridor_flag: int = 0,
    year_basis: str = STATIC_LAYER_YEAR_BASIS,
) -> dict[str, object]:
    """一条候选边的记录，键序即 `EDGE_COLUMNS`，长度保留三位小数。

    `geometry_wkt` 为 None（接入支线不存几何）时写缺失值，几何由 `network_repair.build_edge_geometry` 按两端节点坐标补成直线。
    """
    return {
        "edge_id": edge_id,
        "feature_index": feature_index,
        "part_index": part_index,
        "from_node_id": from_node_id,
        "to_node_id": to_node_id,
        "length_km": round(direct_length_km * tortuosity, 3),
        "direct_length_km": round(direct_length_km, 3),
        "tortuosity": tortuosity,
        "geometry_wkt": pd.NA if geometry_wkt is None else geometry_wkt,
        "corridor_type": corridor_type,
        "existing_corridor_flag": existing_corridor_flag,
        "edge_class": edge_class,
        "source": source,
        "year_basis": year_basis,
    }


def unordered_edge_key(from_node_id: object, to_node_id: object) -> tuple[str, str]:
    """无向边的键：两端 id 按字串排序。三角剖分与专线用它判断"这对节点之间已有边"。"""
    a, b = str(from_node_id), str(to_node_id)
    return (a, b) if a <= b else (b, a)


def next_edge_index(edges: pd.DataFrame) -> int:
    """`edge_id` 中最大数字后缀加一。

    不是 `len(edges) + 1`：去交叉过滤删掉若干行之后，表中最大的 id 会超过表长，按表长编号
    会悄悄重发已在使用的 id。`prepare_inputs` 以 `edge_id` 为键，所以重复项会被丢弃而不是
    报错——早先有一次建网就这样丢了 410 条候选边，连一条警告都没有。

    表为空或没有 `edge_id` 列时返回 1；有 `edge_id` 但没有一个以数字结尾时返回 `len(edges) + 1`：
    调用方的新 id 形如 `edge_{n:05d}`，以数字结尾，撞不上这些旧 id。
    """
    if edges.empty or "edge_id" not in edges.columns:
        return 1
    suffixes = (
        edges["edge_id"].astype(str).str.extract(r"(\d+)$", expand=False).dropna().astype(int)
    )
    return int(suffixes.max()) + 1 if len(suffixes) else len(edges) + 1
