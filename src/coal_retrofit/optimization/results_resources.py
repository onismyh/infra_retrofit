"""资源结果表：生物质 / 氨 / 水（含流域指标）的节点用量与可用量，及逐链路流量。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import AMMONIA_FLOW_SCALE, WATER_FLOW_SCALE
from ._shared import PreparedInputs
from .year_types import YearData


def _build_supply_table(
    prepared: PreparedInputs,
    year: int,
    year_data: YearData,
    biomass_flow_gj: np.ndarray,
    ammonia_flow_kg: np.ndarray,
    water_flow_m3: np.ndarray,
    basin_use_m3: np.ndarray | None = None,
) -> pd.DataFrame:
    biomass_links = prepared.biomass_links[["biomass_node_id"]].copy()
    biomass_links["used"] = np.asarray(biomass_flow_gj, dtype=np.float64)
    biomass_grouped = biomass_links.groupby("biomass_node_id", as_index=False)["used"].sum()
    biomass_table = prepared.biomass[["biomass_node_id", "province_name", "available_gj"]].copy()
    biomass_table = biomass_table.merge(biomass_grouped, on="biomass_node_id", how="left").fillna({"used": 0.0})
    biomass_table["year"] = year
    biomass_table["resource_type"] = "biomass"
    biomass_table["region"] = biomass_table["biomass_node_id"]
    biomass_table["available"] = biomass_table["available_gj"].astype(float)
    biomass_table["competition_scope"] = "shared_biomass_node"
    biomass_table["unit"] = "GJ/yr"

    ammonia_links = year_data.ammonia_links[["ammonia_node_id"]].copy()
    ammonia_links["used"] = np.asarray(ammonia_flow_kg, dtype=np.float64)
    ammonia_grouped = ammonia_links.groupby("ammonia_node_id", as_index=False)["used"].sum()
    ammonia_table = year_data.ammonia_nodes.copy()
    # 求解器返回的 `used` 是物理单位，而可用量向量仍是求解器的缩放单位，
    # 所以必须还原缩放，否则利用率会读成 1e6。
    ammonia_table["available"] = np.asarray(year_data.ammonia_available_kg, dtype=np.float64) * AMMONIA_FLOW_SCALE
    ammonia_table = ammonia_table.merge(ammonia_grouped, on="ammonia_node_id", how="left").fillna({"used": 0.0})
    ammonia_table["year"] = year
    ammonia_table["resource_type"] = "ammonia"
    ammonia_table["region"] = ammonia_table["ammonia_node_id"]
    ammonia_table["competition_scope"] = "shared_ammonia_node"
    ammonia_table["unit"] = "kg/yr"

    water_links = year_data.water_links[["water_node_id"]].copy()
    water_links["used"] = np.asarray(water_flow_m3, dtype=np.float64)
    water_grouped = water_links.groupby("water_node_id", as_index=False)["used"].sum()
    water_table = year_data.water_nodes.copy()
    water_table["available"] = np.asarray(year_data.water_available_m3, dtype=np.float64) * WATER_FLOW_SCALE
    water_table = water_table.merge(water_grouped, on="water_node_id", how="left").fillna({"used": 0.0})
    water_table["year"] = year
    water_table["resource_type"] = "water"
    water_table["region"] = water_table["water_node_id"]
    water_table["competition_scope"] = "shared_water_node"
    water_table["unit"] = "m3/yr"

    # 官方指标流域上限：制度半边，口径是取水。无水约束或关掉流域上限时
    # 为空。单列为一种 resource_type，免得有任何汇总把它与上面按耗水口径的
    # `water` 行合到一起——两者是不同的计量，相加毫无意义。
    frames = [
        biomass_table[["year", "resource_type", "region", "province_name", "competition_scope", "used", "available", "unit"]],
        ammonia_table[["year", "resource_type", "region", "province_name", "competition_scope", "used", "available", "unit"]],
        water_table[["year", "resource_type", "region", "province_name", "competition_scope", "used", "available", "unit"]],
    ]
    basin_codes = list(year_data.water_basin_codes or [])
    if basin_use_m3 is not None and len(basin_codes):
        used = np.asarray(basin_use_m3, dtype=np.float64)
        available = np.asarray(year_data.water_basin_available_m3, dtype=np.float64)
        frames.append(pd.DataFrame({
            "year": year,
            "resource_type": "water_basin_quota",
            "region": basin_codes[:len(used)],
            "province_name": "",
            "competition_scope": "basin_withdrawal_cap",
            "used": used,
            "available": available[:len(used)],
            "unit": "m3/yr",
        }))
    output = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )
    output["utilization"] = np.where(output["available"] > 0, output["used"] / output["available"], 0.0)
    return output


def _build_biomass_flow_table(
    prepared: PreparedInputs,
    year: int,
    biomass_flow_gj: np.ndarray,
) -> pd.DataFrame:
    """逐链路生物质流量：哪个节点给哪个厂供应多少。"""
    links = prepared.biomass_links.copy()
    links["year"] = year
    links["flow_gj"] = np.asarray(biomass_flow_gj, dtype=np.float64)
    # 只保留活跃链路
    active = links[links["flow_gj"] > 1e-3].copy()
    # 附上电厂位置，供绘制地图
    plant_loc = prepared.plants[["plant_id", "centroid_longitude", "centroid_latitude", "province_name"]].copy()
    plant_loc["plant_id"] = plant_loc["plant_id"].astype(str)
    active["plant_id"] = active["plant_id"].astype(str)
    active = active.merge(plant_loc, on="plant_id", how="left", suffixes=("", "_plant"))
    # 附上节点位置
    node_loc = prepared.biomass[["biomass_node_id", "longitude", "latitude", "province_name"]].copy()
    node_loc.columns = ["biomass_node_id", "node_longitude", "node_latitude", "node_province"]
    node_loc["biomass_node_id"] = node_loc["biomass_node_id"].astype(str)
    active["biomass_node_id"] = active["biomass_node_id"].astype(str)
    active = active.merge(node_loc, on="biomass_node_id", how="left")
    return active


def _build_ammonia_flow_table(
    year_data: YearData,
    year: int,
    ammonia_flow_kg: np.ndarray,
    plants: pd.DataFrame,
) -> pd.DataFrame:
    """逐链路氨流量：哪个节点给哪个厂供应多少。"""
    links = year_data.ammonia_links.copy()
    links["year"] = year
    links["flow_kg"] = np.asarray(ammonia_flow_kg, dtype=np.float64)
    active = links[links["flow_kg"] > 1e-3].copy()
    if active.empty:
        return active
    plant_loc = plants[["plant_id", "centroid_longitude", "centroid_latitude", "province_name"]].copy()
    plant_loc["plant_id"] = plant_loc["plant_id"].astype(str)
    active["plant_id"] = active["plant_id"].astype(str)
    active = active.merge(plant_loc, on="plant_id", how="left", suffixes=("", "_plant"))
    return active


def _build_water_flow_table(
    year_data: YearData,
    year: int,
    water_flow_m3: np.ndarray,
    plants: pd.DataFrame,
) -> pd.DataFrame:
    """逐链路水流量：哪个水节点给哪个厂供应多少。"""
    links = year_data.water_links.copy()
    links["year"] = year
    links["flow_m3"] = np.asarray(water_flow_m3, dtype=np.float64)
    active = links[links["flow_m3"] > 1e-3].copy()
    if active.empty:
        return active
    plant_loc = plants[["plant_id", "centroid_longitude", "centroid_latitude", "province_name"]].copy()
    plant_loc["plant_id"] = plant_loc["plant_id"].astype(str)
    active["plant_id"] = active["plant_id"].astype(str)
    active = active.merge(plant_loc, on="plant_id", how="left", suffixes=("", "_plant"))
    return active
