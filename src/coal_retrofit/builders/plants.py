from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from ..artifacts import write_csv
from ..constants import (
    CHINA_WATER_QUOTA_M3_PER_MWH,
    COMBUSTION_CLASS_MAP,
    COOLING_WATER_INTENSITY_M3_PER_MWH,
    PLANT_YEAR_BASIS,
    WATER_INTENSITY_BY_TECH_M3_PER_MWH,
    quota_capacity_band,
)
from ..paths import ProjectPaths

# 空间 hub 的默认个数
DEFAULT_N_HUBS = 350


COLUMN_MAP = {
    "unit_id": ["unit ID", "Unit ID", "unit_id"],
    "plant_site": ["Plant name (local)", "Plant name", "plant_site"],
    "capacity_mw": ["Capacity (MW)", "Capacity", "capacity_mw"],
    "commission_year": ["Year", "Commission Year", "commission_year"],
    "combustion": ["Combustion", "combustion"],
    "cooling_technology": ["Cooling Technology", "Cooling", "cooling_technology"],
    "province": ["Province", "province"],
    "latitude": ["Latitude", "lat", "latitude"],
    "longitude": ["Longitude", "lon", "longitude"],
    "status": ["Status", "status"],
}


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing required column. Tried: {candidates}")


def build_plants_unit_dataframe(paths: ProjectPaths) -> pd.DataFrame:
    source_file = paths.data_dir / "GEM_with_Cooling_Technology_July2025.xlsx"
    df = pd.read_excel(source_file)

    col_unit_id = pick_column(df, COLUMN_MAP["unit_id"])
    col_plant_site = pick_column(df, COLUMN_MAP["plant_site"])
    col_capacity = pick_column(df, COLUMN_MAP["capacity_mw"])
    col_year = pick_column(df, COLUMN_MAP["commission_year"])
    col_combustion = pick_column(df, COLUMN_MAP["combustion"])
    col_cooling = pick_column(df, COLUMN_MAP["cooling_technology"])
    col_province = pick_column(df, COLUMN_MAP["province"])
    col_lat = pick_column(df, COLUMN_MAP["latitude"])
    col_lon = pick_column(df, COLUMN_MAP["longitude"])
    col_status = pick_column(df, COLUMN_MAP["status"])

    status = df[col_status].astype(str).str.strip().str.lower()
    # 只纳入在运与在建机组（沿用文献中"存量机组"（existing fleet）的惯例）。
    # 已宣布 / 已获许可 / 许可前（announced/permitted/pre-permit）的机组
    # （GEM Jul-2025 中 ~257 GW）被排除在外，使基准与在运机组一致。
    valid_statuses = ["operating", "construction"]
    df = df.loc[status.isin(valid_statuses)].copy()

    out = pd.DataFrame(
        {
            "unit_id": df[col_unit_id],
            "plant_site": df[col_plant_site],
            "capacity_mw": pd.to_numeric(df[col_capacity], errors="coerce"),
            "commission_year": pd.to_numeric(df[col_year], errors="coerce").astype("Int64"),
            "combustion": df[col_combustion],
            "cooling_technology": df[col_cooling],
            "province": df[col_province],
            "latitude": pd.to_numeric(df[col_lat], errors="coerce"),
            "longitude": pd.to_numeric(df[col_lon], errors="coerce"),
        }
    )
    out["source"] = source_file.name
    out["year_basis"] = PLANT_YEAR_BASIS
    return out[
        [
            "unit_id",
            "plant_site",
            "capacity_mw",
            "commission_year",
            "combustion",
            "cooling_technology",
            "province",
            "latitude",
            "longitude",
            "source",
            "year_basis",
        ]
    ]


def most_common_string(values: pd.Series) -> str:
    cleaned = [str(v) for v in values.dropna().tolist() if str(v)]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def mix_string(values: pd.Series, max_items: int = 4) -> str:
    cleaned = [str(v) for v in values.dropna().tolist() if str(v)]
    if not cleaned:
        return ""
    total = len(cleaned)
    return ";".join(f"{name}:{count / total:.3f}" for name, count in Counter(cleaned).most_common(max_items))


def build_plant_dataframe(
    plants: pd.DataFrame,
    source_label: str,
    year_basis: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    plants_run = plants.copy()
    plants_run["capacity_mw"] = pd.to_numeric(plants_run["capacity_mw"], errors="coerce").fillna(0.0)

    for plant_index, (plant_site, frame) in enumerate(
        plants_run.groupby("plant_site", sort=True), start=1
    ):
        plant_id = f"P{plant_index:04d}"
        centroid_lat = float(frame["latitude"].mean())
        centroid_lon = float(frame["longitude"].mean())

        cooling_tech = frame["cooling_technology"].astype(str).str.strip().str.lower()
        capacity = frame["capacity_mw"]
        total_cap = capacity.sum()

        cooling_capacities = {
            f"capacity_mw_{tech_key.replace('-', '_')}": 0.0
            for tech_key in COOLING_WATER_INTENSITY_M3_PER_MWH.keys()
        }
        weighted_intensity = 0.0
        has_water_cooling = False

        if total_cap > 0:
            for tech_key, intensity in COOLING_WATER_INTENSITY_M3_PER_MWH.items():
                tech_mask = cooling_tech.str.contains(tech_key, case=False, na=False)
                tech_capacity = capacity[tech_mask].sum()
                cooling_capacities[f"capacity_mw_{tech_key.replace('-', '_')}"] = float(tech_capacity)
                weighted_intensity += (tech_capacity / total_cap) * intensity
                if tech_key in ["once-through", "recirculating"] and tech_capacity > 0:
                    has_water_cooling = True

        mean_commission = float(frame["commission_year"].mean())
        # 设计寿命 40 年：Fan et al. 2023（`fan2023cofiring`）SI Table 10（煤电 40 年）；
        # Wang et al. 2025（`wang2025reducing`）正文与 SI Table 3（40 年，区间 25-40）。
        retirement_year = int(mean_commission + 40)

        # Wang (2023) 用水强度表：逐机组按蒸汽参数（steam cycle）与冷却方式查表，
        # 再按装机容量加权到 hub。之所以产出两个变体，是因为沿海直流冷却 hub 是否取
        # 海水，要等其质心定位之后（add_seawater_classification）才知道：`_all` 把
        # 每台直流冷却机组都计入淡水，`_fresh` 把它们置零。
        water_intensities = _hub_water_intensities(frame, total_cap)

        rows.append(
            {
                "plant_id": plant_id,
                "plant_index": plant_index,
                "plant_site": str(plant_site),
                "unit_count": int(len(frame)),
                "total_capacity_mw": float(total_cap),
                "mean_commission_year": mean_commission,
                "retirement_year": retirement_year,
                "province_mode": most_common_string(frame["province"]),
                "dominant_combustion": most_common_string(frame["combustion"]),
                "dominant_cooling_technology": most_common_string(frame["cooling_technology"]),
                "combustion_mix": mix_string(frame["combustion"]),
                **cooling_capacities,
                "weighted_water_intensity_m3_per_mwh": round(weighted_intensity, 4),
                **water_intensities,
                "requires_water_supply": has_water_cooling,
                "centroid_latitude": centroid_lat,
                "centroid_longitude": centroid_lon,
                "source": source_label,
                "year_basis": year_basis,
            }
        )

    return pd.DataFrame(rows).sort_values("plant_index").reset_index(drop=True)


WATER_KEYS = ("withdrawal", "consumption", "withdrawal_ccs", "consumption_ccs")


def _hub_water_intensities(units: pd.DataFrame, total_capacity: float) -> dict[str, float]:
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
    combustion = units["combustion"].map(_combustion_class)
    cooling = units["cooling_technology"].map(_cooling_class)
    capacity = units["capacity_mw"].astype(float)
    for (comb, cool), index in units.groupby([combustion, cooling]).groups.items():
        entry = WATER_INTENSITY_BY_TECH_M3_PER_MWH.get((comb, cool))
        if entry is None:
            continue
        share = float(capacity.loc[index].sum()) / total_capacity
        for key in WATER_KEYS:
            out[f"{key}_intensity_all"] += share * entry[key]
            if cool != "once-through":
                out[f"{key}_intensity_fresh"] += share * entry[key]
    for cool, index in units.groupby(cooling).groups.items():
        unit_caps = capacity.loc[index]
        for unit_capacity in unit_caps:
            quota = CHINA_WATER_QUOTA_M3_PER_MWH.get((cool, quota_capacity_band(float(unit_capacity))))
            if quota is not None:
                out["quota_intensity_m3_per_mwh"] += (float(unit_capacity) / total_capacity) * quota
    # 若该 hub 的凝汽器改为空冷，其用水强度会是多少。蒸汽参数不变，取同一张表的
    # 空冷行，因此差值只归因于冷却方式。四种口径都算，并按每台机组自己的蒸汽参数
    # 加权——若在下游以 `air_consumption x ratio(dominant_combustion)` 推出取水那一对，
    # 得到的不是同一个数，而且曾把五个本已空冷的 hub 推到其自身基准取水量之上。
    for comb, index in units.groupby(combustion).groups.items():
        entry = WATER_INTENSITY_BY_TECH_M3_PER_MWH.get((comb, "air"))
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


def _combustion_class(label: str) -> str:
    """把 GEM 的燃烧技术标签映射到用水强度表的三个蒸汽参数等级。"""
    text = str(label).strip().lower().split("/")[0]
    return COMBUSTION_CLASS_MAP.get(text, "subcritical")


def _cooling_class(label: str) -> str:
    text = str(label).strip().lower()
    if "once" in text:
        return "once-through"
    if "air" in text or "dry" in text:
        return "air"
    return "recirculating"


# 海水冷却分类。
#
# GEM 只记 `once-through`，不说明凝汽器取的是海水还是河水，而二者在淡水预算中的表现
# 完全不同：沿海机组一点也不争用淡水，内陆机组则要引走 ~85-115 m3/MWh。二者的划分
# 由到海岸线的距离推断。
#
# 海岸线取自 data/ChinaMap/boundary.shp，其 GBCODE 字段把海岸（26010 开阔海岸，
# 26080/26100 岛屿海岸，合计 26 085 km）与陆地国界（61010）区分开。早先的版本量的是
# 到整条国界的距离，把鸭绿江与中越陆地边界也混同为海，只好再加一张沿海省份白名单
# 来弥补。
#
# 按到真实海岸的距离，直流冷却机组被分成两群，中间几乎是空带：0-20 km 有 240.2 GW，
# 20-50 km 有 2.0 GW，50 km 以外有 222.9 GW。阈值取在这段空档的中间，因此分类结果
# 对阈值不敏感。
SEAWATER_COAST_DISTANCE_KM = 30.0


def add_seawater_classification(paths: ProjectPaths, plants: pd.DataFrame) -> pd.DataFrame:
    """按到海岸的距离，把直流冷却装机拆分为海水与淡水两部分。"""
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import unary_union

    boundary = gpd.read_file(paths.find_data_file("boundary.shp")).to_crs("EPSG:2380")
    coast_lines = boundary[boundary["GBCODE"].astype(str).str.startswith("26")]
    if coast_lines.empty:
        raise ValueError("boundary.shp has no GBCODE 26* coastline features")
    coastline = unary_union(coast_lines.geometry.tolist())

    points = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(plants["centroid_longitude"], plants["centroid_latitude"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:2380")
    distance_km = points.distance(coastline) / 1000.0

    once_through = plants.get("capacity_mw_once_through", pd.Series(0.0, index=plants.index)).astype(float)
    plants = plants.copy()
    plants["coast_distance_km"] = distance_km.round(2).to_numpy()
    plants["seawater_cooled"] = (once_through > 0) & (distance_km.to_numpy() <= SEAWATER_COAST_DISTANCE_KM)

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


def write_plants_unit(paths: ProjectPaths) -> pd.DataFrame:
    plants = build_plants_unit_dataframe(paths)
    write_csv(plants, paths.inputs_dir / "plants_unit.csv")
    return plants


def _haversine_distance_matrix(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """两两之间的 Haversine 距离（km）。"""
    lon_r = np.radians(lons)
    lat_r = np.radians(lats)
    dlon = lon_r[:, None] - lon_r[None, :]
    dlat = lat_r[:, None] - lat_r[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _cluster_plants_to_hubs(units: pd.DataFrame, n_hubs: int = DEFAULT_N_HUBS) -> pd.DataFrame:
    """把逐机组数据按空间聚类为 *n_hubs* 个 hub。

    对机组坐标做凝聚层次聚类（agglomerative clustering）。每个 hub 继承其簇内的
    全部机组；汇总（装机、冷却方式等）由 build_plant_dataframe 借助合成的
    'plant_site' 标签完成。
    """
    valid = units.dropna(subset=["latitude", "longitude"]).copy()
    if len(valid) <= n_hubs:
        return valid

    dist = _haversine_distance_matrix(
        valid["longitude"].to_numpy(), valid["latitude"].to_numpy()
    )
    clustering = AgglomerativeClustering(
        n_clusters=n_hubs,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist)
    valid["plant_site"] = [f"H{n_hubs}_{int(label):03d}" for label in labels]
    return valid


def write_plants(paths: ProjectPaths, n_hubs: int = DEFAULT_N_HUBS) -> None:
    plants_path = paths.inputs_dir / "plants_unit.csv"
    plants = pd.read_csv(plants_path)
    clustered = _cluster_plants_to_hubs(plants, n_hubs=n_hubs)
    plant_sites = build_plant_dataframe(
        plants=clustered,
        source_label=paths.rel(plants_path),
        year_basis=PLANT_YEAR_BASIS,
    )
    plant_sites = add_seawater_classification(paths, plant_sites)
    plant_sites = finalize_water_intensities(plant_sites)
    write_csv(plant_sites, paths.inputs_dir / "plants.csv")
