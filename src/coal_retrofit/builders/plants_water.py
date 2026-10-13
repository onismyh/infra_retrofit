"""煤电 hub 的用水强度与海水冷却分类：逐机组按蒸汽参数与冷却方式查 Wang (2023) 表并按装机加权（`hub_water_intensities`），
按到海岸的距离判定直流冷却是否取海水（`add_seawater_classification`），再为海水 hub 选淡水变体并算定额口径
（`finalize_water_intensities`）。机组表读入、聚类与毛热耗在 `plants`，冷却方式归类在 `constants.cooling_class`。
"""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd

from ..constants import (
    CHINA_WATER_QUOTA_M3_PER_MWH,
    COMBUSTION_CLASS_MAP,
    COOLING_WATER_INTENSITY_M3_PER_MWH,
    TARGET_GEO_CRS,
    WATER_INTENSITY_BY_TECH_M3_PER_MWH,
    cooling_class,
    quota_capacity_band,
)
from ..paths import ProjectPaths
from ..spatial import distance_to_lines_km

WATER_KEYS = ("withdrawal", "consumption", "withdrawal_ccs", "consumption_ccs")


def combustion_class(label: str) -> str:
    """把 GEM 的燃烧技术标签映射到用水强度表的三个蒸汽参数等级。"""
    text = str(label).strip().lower().split("/")[0]
    return COMBUSTION_CLASS_MAP.get(text, "subcritical")


def hub_water_intensities(units: pd.DataFrame, total_capacity: float) -> dict[str, float]:
    """单个 hub 按装机容量加权的用水强度，两种淡水变体都给出。

    同时返回中国取水定额口径的强度，它不需要淡水变体：定额本身已不含凝汽器冷却水量，
    剩下的部分（锅炉补给水与杂用水，直流冷却为 0.35-0.72 m3/MWh）无论凝汽器取的是
    河水还是海水，都从淡水系统取用。
    """
    out = {f"{key}_intensity_{variant}": 0.0 for key in WATER_KEYS for variant in ("all", "fresh")}
    out["quota_intensity_m3_per_mwh"] = 0.0
    for basis in ("consumption", "consumption_ccs", "withdrawal", "withdrawal_ccs"):
        out[f"air_{basis}_intensity_m3_per_mwh"] = 0.0
    out["already_air_share"] = 0.0
    if total_capacity <= 0:
        return out
    combustion = units["combustion"].map(combustion_class)
    cooling = units["cooling_technology"].map(cooling_class)
    capacity = units["capacity_mw"].astype(float)
    for group_key, index in units.groupby([combustion, cooling]).groups.items():
        comb, cool = cast(tuple[str, str], group_key)
        entry = WATER_INTENSITY_BY_TECH_M3_PER_MWH.get((comb, cool))
        if entry is None:
            continue
        share = float(capacity.loc[index].sum()) / total_capacity
        for key in WATER_KEYS:
            out[f"{key}_intensity_all"] += share * entry[key]
            if cool != "once-through":
                out[f"{key}_intensity_fresh"] += share * entry[key]
    for cooling_key, index in units.groupby(cooling).groups.items():
        unit_caps = capacity.loc[index]
        for unit_capacity in unit_caps:
            quota = CHINA_WATER_QUOTA_M3_PER_MWH.get((str(cooling_key), quota_capacity_band(float(unit_capacity))))
            if quota is not None:
                out["quota_intensity_m3_per_mwh"] += (float(unit_capacity) / total_capacity) * quota
    # 若该 hub 的凝汽器改为空冷，其用水强度会是多少。蒸汽参数不变，取同一张表的
    # 空冷行，因此差值只归因于冷却方式。四种口径都算，并按每台机组自己的蒸汽参数
    # 加权——若在下游以 `air_consumption x ratio(dominant_combustion)` 推出取水那一对，
    # 得到的不是同一个数，而且曾把五个本已空冷的 hub 推到其自身基准取水量之上。
    for combustion_key, index in units.groupby(combustion).groups.items():
        entry = WATER_INTENSITY_BY_TECH_M3_PER_MWH.get((str(combustion_key), "air"))
        if entry is None:
            continue
        share = float(capacity.loc[index].sum()) / total_capacity
        for basis in ("consumption", "consumption_ccs", "withdrawal", "withdrawal_ccs"):
            out[f"air_{basis}_intensity_m3_per_mwh"] += share * entry[basis]
    out["already_air_share"] = float(capacity[cooling == "air"].sum()) / total_capacity
    return {k: round(v, 4) for k, v in out.items()}


def finalize_water_intensities(plants: pd.DataFrame) -> pd.DataFrame:
    """为海水冷却 hub 选用淡水变体，并删去临时列。

    海水凝汽器不取河水：按取水口径，这就是 241 GW 装机上 ~100 m3/MWh 的虚构取水量
    与完全不取水之间的差别。

    带捕集的定额是基准定额加上捕集装置额外的耗水性补给水。定额表早于 CCS，没有对应的
    行，但它本应计量的增量，恰好就是电厂必须额外购入的水量，即 Wang (2023) 表中的
    耗水增量。
    """
    plants = plants.copy()
    seawater = plants["seawater_cooled"].astype(bool)
    for key in WATER_KEYS:
        column = f"{key}_intensity_m3_per_mwh"
        plants[column] = plants[f"{key}_intensity_all"].where(~seawater, plants[f"{key}_intensity_fresh"])
    plants["quota_ccs_intensity_m3_per_mwh"] = (
        plants["quota_intensity_m3_per_mwh"]
        + (plants["consumption_ccs_intensity_m3_per_mwh"] - plants["consumption_intensity_m3_per_mwh"]).clip(lower=0.0)
    ).round(4)
    # 单独拆出淡水直流冷却的凝汽器水量。`_intensity_all` 对所有冷却类别求和，
    # `_intensity_fresh` 对除直流冷却外的所有类别求和，所以二者之差正是直流冷却这一项——
    # 逐机组、按各机组自己的蒸汽参数，精确成立。海水 hub 取零，因为它们对外发布的列
    # 本来就是 fresh 变体。
    #
    # 保留这一项，是因为它是取水列中唯一需要重新标定的部分：WATER_INTENSITY_BY_TECH_M3_PER_MWH
    # 的直流冷却行是 Macknick (2011) 的美国数值，与水资源公报约成 2 倍关系，而循环冷却与空冷行
    # 吻合到 5-14%。见 `builders/water_quota`。在下游由 `dominant_combustion` x
    # `capacity_mw_once_through` 重建它并不等价，曾在全机队合计上差了 24%：hub 的主导
    # 蒸汽参数往往不是其直流冷却机组的蒸汽参数。
    for key in ("withdrawal", "withdrawal_ccs"):
        plants[f"once_through_{key}_intensity_m3_per_mwh"] = (
            (plants[f"{key}_intensity_all"] - plants[f"{key}_intensity_fresh"])
            .where(~seawater, 0.0).clip(lower=0.0).round(4)
        )
    return plants.drop(columns=[f"{k}_intensity_{v}" for k in WATER_KEYS for v in ("all", "fresh")])


# 海水冷却分类。
#
# GEM 只记 `once-through`，不说明凝汽器取的是海水还是河水，而二者在淡水预算中的表现
# 完全不同：沿海机组一点也不争用淡水，内陆机组则要引走 ~85-115 m3/MWh。二者的划分
# 由到海岸线的距离推断。
#
# 海岸线取自 data/ChinaMap/boundary.shp 的 GBCODE 26* 要素：26010 是大陆海岸（54 段，18 137 km）与沿海岛屿
# 轮廓（889 个闭合环，7 416 km），26100 是南海诸岛等岛屿的岸线段（261 段，2 068 km），26080 是南海岛礁
# （92 段，846 km），合计 28 467 km（大地线长，2026-10-13 实测；此前写的 26 085 km 只是 26010 在 EPSG:2380 下的长度）。
# 陆地国界与九段线同在 61010，不算海岸。早先的版本量的是到整条国界的距离，把鸭绿江与中越陆地边界也混同为海，
# 只好再加一张沿海省份白名单来弥补。
#
# 按到真实海岸的距离，直流冷却机组被分成两群，中间几乎是空带：0-20 km 有 240.2 GW，
# 20-50 km 有 2.0 GW，50 km 以外有 222.9 GW。阈值取在这段空档的中间，因此分类结果
# 对阈值不敏感。
SEAWATER_COAST_DISTANCE_KM = 30.0


def add_seawater_classification(paths: ProjectPaths, plants: pd.DataFrame) -> pd.DataFrame:
    """按到海岸的距离，把直流冷却装机拆分为海水与淡水两部分。

    海岸线取 `boundary.shp` 的 GBCODE 26* 要素，距离按球面算（`spatial.distance_to_lines_km`；
    2026-10-11 前在 EPSG:2380 下算，该投影的长度比例在岸线顶点处最大 1.045–1.049、离岸 100 km 内的 hub 处最大 1.034，
    不改变任何 hub 的海水标志）。
    """
    boundary = gpd.read_file(paths.find_data_file("boundary.shp")).to_crs(TARGET_GEO_CRS)
    coast_lines = boundary[boundary["GBCODE"].astype(str).str.startswith("26")]
    if coast_lines.empty:
        raise ValueError("boundary.shp has no GBCODE 26* coastline features")
    distance_km = distance_to_lines_km(
        plants["centroid_longitude"].to_numpy(float), plants["centroid_latitude"].to_numpy(float), coast_lines.geometry
    )

    once_through = plants.get("capacity_mw_once_through", pd.Series(0.0, index=plants.index)).astype(float)
    plants = plants.copy()
    plants["coast_distance_km"] = distance_km.round(2)
    plants["seawater_cooled"] = (once_through > 0) & (distance_km <= SEAWATER_COAST_DISTANCE_KM)

    total = plants["total_capacity_mw"].astype(float).replace(0.0, np.nan)
    freshwater = pd.Series(0.0, index=plants.index)
    for tech_key, intensity in COOLING_WATER_INTENSITY_M3_PER_MWH.items():
        column = f"capacity_mw_{tech_key.replace('-', '_')}"
        if column not in plants.columns:
            continue
        capacity = plants[column].astype(float)
        if tech_key == "once-through":
            capacity = capacity.where(~plants["seawater_cooled"], 0.0)
        freshwater += capacity / total * intensity
    plants["freshwater_intensity_m3_per_mwh"] = freshwater.fillna(0.0).round(4)
    plants["requires_water_supply"] = plants["freshwater_intensity_m3_per_mwh"] > 0
    return plants
