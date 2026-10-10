"""结果表的省与分区：管网节点归省，省归 ChinaCCS 的六大区。

供 `network.csv` 的起止省（`results_network._build_network_table`）、`sinks.csv` 的省（`_build_sinks_table`）与结果工作簿
（`results_workbook`）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from ..constants import TARGET_GEO_CRS
from ..spatial import load_provinces
from ._shared import PreparedInputs
from .scenario import PROVINCE_NAME_ALIASES

OFFSHORE = "Offshore"
# 省名（输入表与分省煤价表的写法）-> (中文简称, 分区号)，顺序即结果表的行序。省到区的对应取自 ChinaCCS.xlsm 的
# Data_Source 与 Data_Sink 两表（每省只对应一个区）；西藏不在其中，按六大区的惯例归西南。
PROVINCES: dict[str, tuple[str, int]] = {
    "Heilongjiang": ("黑龙江", 1), "Jilin": ("吉林", 1), "Liaoning": ("辽宁", 1),
    "Beijing": ("北京", 2), "Tianjin": ("天津", 2), "Hebei": ("河北", 2), "Shanxi": ("山西", 2),
    "Inner Mongolia": ("内蒙古", 2),
    "Shanghai": ("上海", 3), "Jiangsu": ("江苏", 3), "Zhejiang": ("浙江", 3), "Anhui": ("安徽", 3),
    "Fujian": ("福建", 3), "Jiangxi": ("江西", 3), "Shandong": ("山东", 3),
    "Shaanxi": ("陕西", 4), "Gansu": ("甘肃", 4), "Qinghai": ("青海", 4), "Ningxia": ("宁夏", 4),
    "Xinjiang": ("新疆", 4),
    "Chongqing": ("重庆", 5), "Sichuan": ("四川", 5), "Guizhou": ("贵州", 5), "Yunnan": ("云南", 5),
    "Xizang": ("西藏", 5),
    "Henan": ("河南", 6), "Hubei": ("湖北", 6), "Hunan": ("湖南", 6), "Guangdong": ("广东", 6),
    "Guangxi": ("广西", 6), "Hainan": ("海南", 6),
}
# 7 海上（海上封存汇与落在省界之外的管网节点）是本模型加的，8 跨地理分区只用于边（`edge_region`）。ChinaCCS 没有海上区，
# 海上汇归沿海的区，跨地理分区记 7：按 ChinaCCS 的编号读本模型的表会把海上当成跨区。
OFFSHORE_REGION = 7
CROSS_REGION = 8
REGIONS: dict[int, str] = {
    1: "东北", 2: "华北", 3: "华东", 4: "西北", 5: "西南", 6: "中南", OFFSHORE_REGION: "海上", CROSS_REGION: "跨地理分区",
}


def province_region(province: str) -> int:
    """省名（`PROVINCES` 的写法或 `OFFSHORE`）所在的分区号。"""
    return OFFSHORE_REGION if province == OFFSHORE else PROVINCES[province][1]


def edge_region(from_region: int, to_region: int) -> int:
    """边的分区：两端同区记该区；一端在海上的记陆上那一端的区（ChinaCCS 把海上汇归到沿海的分区，接海上汇的管道
    因此不算跨区）；两端在不同的陆上分区记跨地理分区。"""
    if from_region == to_region:
        return from_region
    if OFFSHORE_REGION in (from_region, to_region):
        return to_region if from_region == OFFSHORE_REGION else from_region
    return CROSS_REGION


def node_provinces(prepared: PreparedInputs, provinces_path: Path) -> dict[str, str]:
    """管网每个节点的省名（`PROVINCES` 的写法或 `OFFSHORE`）。

    依次取第一个有的：源的省（煤电 `province_name`、工业 `province`）；本情景纳入的封存汇（`prepared.storages`）在
    `offshore` 为真时记海上（与封存成本的海上倍率同一标记，`data_prep._prepare_storages`），否则取
    `storage_hubs.csv` 的省名；`pipeline_nodes.csv` 自带的省名。拼法统一换成 `PROVINCES` 的写法（`PROVINCE_NAME_ALIASES`）。
    都没有的（走廊节点、没填省名的封存汇）按经纬度落在省界图层哪个省，落在所有省之外记海上，与 builders 标记海上封存汇
    同一判据（`builders.storage._flag_offshore`）；只有这时才读图层。
    """
    network = prepared.network
    plants = dict(zip(prepared.plants["plant_id"].astype(str), prepared.plants["province_name"], strict=True))
    hubs = prepared.industry.hubs
    industry = dict(zip(hubs["hub_id"].astype(str), hubs["province"], strict=True))
    storages = prepared.storages.set_index(prepared.storages["storage_hub_id"].astype(str))
    offshore = storages["offshore"].astype(bool) if "offshore" in storages.columns else pd.Series(False, storages.index)
    nodes = network.nodes.set_index(network.nodes["node_id"].astype(str))
    by_node: dict[str, str] = {}

    def offer(node: object, value: object) -> None:  # 先给的算数：源、封存汇、节点表自带的省名
        name = _given(value)
        if name is not None:
            by_node.setdefault(str(node), name)

    for plant_id, node in network.plant_node_ids.items():
        offer(node, plants[str(plant_id)])
    for hub_id, node in network.industry_node_ids.items():
        offer(node, industry[str(hub_id)])
    for hub_id, node in network.storage_node_ids.items():
        if str(hub_id) not in storages.index:
            continue  # 节点表里有、本情景不纳入的汇（`storage_scope = "dsa_only"` 去掉的 EOR 汇），与走廊节点同法定省
        offer(node, OFFSHORE if offshore[str(hub_id)] else storages.get("province", pd.Series()).get(str(hub_id)))
    for node_id, value in nodes.get("province", pd.Series()).items():
        offer(node_id, value)
    unplaced = [node for node in nodes.index if node not in by_node]
    if unplaced:
        if not Path(provinces_path).exists():
            raise FileNotFoundError(f"{provinces_path} 不存在：{len(unplaced)} 个管网节点没有省名，要按省界图层定省")
        by_node.update(_locate(nodes.loc[unplaced, ["lon", "lat"]], provinces_path))
    unknown = sorted({name for name in by_node.values() if name != OFFSHORE and name not in PROVINCES})
    if unknown:
        raise ValueError(f"管网节点的省名 {unknown} 不在 results_regions.PROVINCES 里")
    return by_node


def _given(value: Any) -> str | None:
    """输入表里的省名换成 `PROVINCES` 的写法（`PROVINCE_NAME_ALIASES`）；空值返回 None。"""
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    name = str(value).strip()
    return PROVINCE_NAME_ALIASES.get(name, name)


def _locate(points: pd.DataFrame, provinces_path: Path) -> dict[str, str]:
    """按经纬度（`lon`、`lat` 两列，索引为节点号）把节点落到省界图层的省；压在省界上的取 `province_id` 小的那个，
    落在所有省之外的记海上。"""
    layer = load_provinces(provinces_path).to_crs(TARGET_GEO_CRS)
    located = gpd.GeoDataFrame(
        index=points.index, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs=TARGET_GEO_CRS
    )
    joined = gpd.sjoin(located, layer[["province_id", "province_name", "geometry"]], how="left", predicate="intersects")
    joined = joined.sort_values("province_id", na_position="last")
    first = joined.loc[~joined.index.duplicated(keep="first"), "province_name"]
    return {str(node): OFFSHORE if pd.isna(name) else _english(str(name)) for node, name in first.items()}


def _english(full_name: str) -> str:
    """省界图层的中文全称换成 `PROVINCES` 的写法：中文简称是全称的前缀（内蒙古 / 内蒙古自治区），同
    `builders.water_quota._short_province_name`。港澳台不在 ChinaCCS 的分区里，落到那里就报错。"""
    matches = [name for name, (short, _) in PROVINCES.items() if full_name.startswith(short)]
    if len(matches) != 1:
        raise ValueError(f"省界图层的 {full_name!r} 对不上 results_regions.PROVINCES 里的省")
    return matches[0]
