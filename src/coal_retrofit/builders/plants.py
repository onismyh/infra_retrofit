"""煤电机组表与 hub 表的构建：读 GEM 机组表（`build_plants_unit_dataframe`）、按球面距离聚类为 hub（`_cluster_plants_to_hubs`，
`spatial.cluster_labels`）、hub 汇总（`build_plant_dataframe`）、逐台毛热耗与重建上限（`unit_heat_rate_gj_per_mwh`、
`unit_rebuild_heat_rate_cap_gj_per_mwh`）。用水强度与海水冷却分类在 `plants_water`，冷却方式归类在 `constants.cooling_class`。
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from ..artifacts import write_csv
from ..constants import (
    AIR_COOLED_SUPPLY_COAL_RATE_ADDER_G_PER_KWH,
    COAL_DESIGN_LIFE_YEARS,
    COAL_STATION_SERVICE_RATE,
    COOLING_CLASSES,
    GEM_NAMED_CFB_UNIT_IDS,
    GJ_PER_MWH,
    PLANT_YEAR_BASIS,
    STANDARD_COAL_GJ_PER_KG,
    SUPPLY_COAL_RATE_CLASS_MIN_MW,
    SUPPLY_COAL_RATE_G_PER_KWH,
    cooling_class,
)
from ..paths import ProjectPaths
from ..spatial import cluster_labels
from .plants_water import add_seawater_classification, finalize_water_intensities, hub_water_intensities

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
    out["combustion"] = mark_named_cfb(out)
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


def is_cfb(labels: pd.Series) -> pd.Series:
    """机型标签按 `/` 分段（去首尾空格、小写）后有一段是 `cfb`：本体机型为 CFB（`CFB`、`CFB/CCS`），或带 `/CFB` 标记
    （`mark_named_cfb`）。缺失值为否。"""
    return labels.map(lambda label: "cfb" in {part.strip() for part in str(label).lower().split("/")}).astype(bool)


def mark_named_cfb(units: pd.DataFrame) -> pd.Series:
    """`constants.GEM_NAMED_CFB_UNIT_IDS` 里的机组在机型标签后加 `/CFB`（已有 `cfb` 段的不再加，可重复调用），其余不变。

    本体机型（`/` 前的部分）不变，毛热耗与用水强度照旧按它查表；hub 的 CFB 装机份额按 `is_cfb` 计。
    """
    combustion = units["combustion"].copy()
    named = units["unit_id"].astype(str).isin(GEM_NAMED_CFB_UNIT_IDS) & ~is_cfb(combustion)
    combustion[named] = combustion[named].astype(str) + "/CFB"
    return combustion


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
            f"capacity_mw_{tech_key.replace('-', '_')}": float(
                capacity[cooling_tech.str.contains(tech_key, case=False, na=False)].sum()
            )
            for tech_key in COOLING_CLASSES
        }

        mean_commission = float(frame["commission_year"].mean())
        # 整个 hub 在平均投产年 + 设计寿命那年一起到期：整数 hub 用它；连续 hub 按机组逐台到期（`data_prep`）。
        retirement_year = int(mean_commission + COAL_DESIGN_LIFE_YEARS)
        # 按装机加权的逐台毛热耗；不用 `dominant_combustion`，那只是机组数的众数机型。
        heat_rate = float("nan")
        if total_cap > 0:
            heat_rate = float((unit_heat_rate_gj_per_mwh(frame) * capacity).sum() / total_cap)

        # Wang (2023) 用水强度表：逐机组按蒸汽参数（steam cycle）与冷却方式查表，
        # 再按装机容量加权到 hub。之所以产出两个变体，是因为沿海直流冷却 hub 是否取
        # 海水，要等其质心定位之后（add_seawater_classification）才知道：`_all` 把
        # 每台直流冷却机组都计入淡水，`_fresh` 把它们置零。
        water_intensities = hub_water_intensities(frame, total_cap)

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
                "heat_rate_gj_per_mwh": round(heat_rate, 4),
                **cooling_capacities,
                **water_intensities,
                "centroid_latitude": centroid_lat,
                "centroid_longitude": centroid_lon,
                "source": source_label,
                "year_basis": year_basis,
            }
        )

    return pd.DataFrame(rows).sort_values("plant_index").reset_index(drop=True)


def unit_heat_rate_gj_per_mwh(units: pd.DataFrame) -> pd.Series:
    """逐台毛热耗（GJ/MWh）= 分档供电煤耗（空冷 +15）x (1 - 厂用电率) x 标准煤热值，见 `constants`。

    带 `/CCS` 后缀的机组按本体机型归档：后缀是改造状态，这些机组在模型里按未改造计。`/CFB` 后缀（`mark_named_cfb`）
    标的是炉型，同样按本体机型归档。
    """
    rates = [
        _supply_coal_rate_g_per_kwh(combustion, float(capacity), cooling_class(cooling))
        for combustion, capacity, cooling in zip(units["combustion"], units["capacity_mw"], units["cooling_technology"])
    ]
    return pd.Series(rates, index=units.index, dtype=float) * (1.0 - COAL_STATION_SERVICE_RATE) * STANDARD_COAL_GJ_PER_KG


def unit_rebuild_heat_rate_cap_gj_per_mwh(units: pd.DataFrame, rebuild_efficiency: float) -> pd.Series:
    """到期机组原址重建后的毛热耗上限：3.6 / 重建效率，空冷机组另加空冷的 +15 g/kWh（重建效率与分档煤耗一样按湿冷
    机组计，原址重建不换冷却方式）。重建机组取 min(原机组毛热耗, 上限)，见 `optimization.data_prep_plants._with_expiry`。
    """
    air = units["cooling_technology"].map(cooling_class).eq("air").astype(float)
    adder = AIR_COOLED_SUPPLY_COAL_RATE_ADDER_G_PER_KWH * (1.0 - COAL_STATION_SERVICE_RATE) * STANDARD_COAL_GJ_PER_KG
    return GJ_PER_MWH / rebuild_efficiency + adder * air


def _supply_coal_rate_g_per_kwh(combustion: str, capacity_mw: float, cooling: str) -> float:
    """机型、铭牌容量与冷却方式 -> 供电煤耗基准水平（g/kWh）。"""
    base = str(combustion).strip().lower().split("/")[0]
    level = ""
    if base in SUPPLY_COAL_RATE_CLASS_MIN_MW:
        threshold, upper, lower = SUPPLY_COAL_RATE_CLASS_MIN_MW[base]
        level = upper if capacity_mw >= threshold else lower
    if (base, level) not in SUPPLY_COAL_RATE_G_PER_KWH:
        raise ValueError(f"no supply coal rate for combustion label {combustion!r}")
    adder = AIR_COOLED_SUPPLY_COAL_RATE_ADDER_G_PER_KWH if cooling == "air" else 0.0
    return SUPPLY_COAL_RATE_G_PER_KWH[(base, level)] + adder


def write_plants_unit(paths: ProjectPaths) -> pd.DataFrame:
    plants = build_plants_unit_dataframe(paths)
    write_csv(plants, paths.inputs_dir / "plants_unit.csv")
    return plants


def _cluster_plants_to_hubs(units: pd.DataFrame, n_hubs: int = DEFAULT_N_HUBS) -> pd.DataFrame:
    """把逐机组数据按空间聚类为 *n_hubs* 个 hub。

    对机组坐标做凝聚层次聚类（`spatial.cluster_labels`，与工业点源同一算法）。每个 hub 继承其簇内的
    全部机组；汇总（装机、冷却方式等）由 build_plant_dataframe 借助合成的
    'plant_site' 标签完成。
    """
    valid = units.dropna(subset=["latitude", "longitude"]).copy()
    if len(valid) <= n_hubs:
        return valid
    labels = cluster_labels(valid["longitude"].to_numpy(), valid["latitude"].to_numpy(), n_hubs)
    valid["plant_site"] = [f"H{n_hubs}_{int(label):03d}" for label in labels]
    return valid


def write_plants(paths: ProjectPaths, n_hubs: int = DEFAULT_N_HUBS) -> None:
    """写 `plants.csv`（hub 表）与 `plants_unit_hub.csv`（机组到 hub 的映射，连续 hub 的机组级到期要用）。"""
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
    # 机组表的原有列（`plant_site` 是原厂名，不是聚类标签），加所属 hub 与逐台毛热耗。逐台毛热耗取 4 位，只供复核、
    # 模型不读；hub 值按未取整的逐台值加权，用这一列重算差在 1e-4 以内。
    units = plants.loc[clustered.index].copy()
    units["plant_id"] = clustered["plant_site"].map(dict(zip(plant_sites["plant_site"], plant_sites["plant_id"])))
    units["heat_rate_gj_per_mwh"] = unit_heat_rate_gj_per_mwh(units).round(4)
    write_csv(units, paths.inputs_dir / "plants_unit_hub.csv")
