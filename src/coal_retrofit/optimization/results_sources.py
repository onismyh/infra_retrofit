"""源表 `sources.csv` 与逐路线表 `source_routes.csv`（2026-10-10 起）：煤电 hub 与工业 hub 用同一套列。

`sources.csv` 每年每个源一行。共用列在前：源类型（`coal` / `industry`）、部门（煤电记 `coal`）、目标组、省、流域、
经纬度、活动量与其单位（煤电发电量 MWh/yr、工业产量 kt/yr）、基线排放、减排、剩余排放、捕集量、各路线份额（不适用的
路线记 0，如煤电的 `share_h2`、工业的 `share_retire`）、主导路线、生物质、氨、绿氢、耗水与取水量；其后是只对一类源
有意义的列，另一类记空。成本不在这里拆分：逐项成本在 `costs.csv`（`results_costs`），本表只带每个源的合计（不折现与
折现各一列，`results_tables.build_result_tables` 并入）。

- `h2_kg`：从绿氢节点取的氢。煤电是用氨折成的氢（氨 x `NH3_H2_RATIO`，与节点约束同一系数，`model_resources`），工业是
  链路上买的氢。
- `water_consumption_m3`：耗水，节点生态流量约束的口径；只有煤电有，工业记空（模型只按取水计工业用水）。
- `water_withdrawal_m3`：取水，流域取水指标的口径。工业是路线取水量；煤电按取水强度与解出的份额、空冷份额重算
  （与 `model_resources.add_water_balances` 的流域约束同式），只在建了流域取水约束时有，否则记空。

`source_routes.csv` 每年每个源每条路线一行：份额、活动量、基线排放、减排、捕集量。煤电按 `results_plant._pathway_split`
逐项拆到路径上。工业的减排与捕集量 = 该路线的系数 x 份额；活动量：改造路线 = 产量 x 可改造产量比例
（`abatable_production_share`）x 份额，其余产量记在未改造上；工业的份额作用于可改造的点源，基线排放不按份额拆，记空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import NH3_H2_RATIO
from ..constants_industry import INDUSTRY_ROUTES, POWER_TARGET_GROUP
from ._shared import PreparedInputs
from .industry import UNABATED
from .year_types import YearSolution

# 煤电六条路径（`scenario.PATHWAYS`）与工业三条路线（`constants_industry.INDUSTRY_ROUTES`）的并集。
ROUTES = ("unabated", "retire", "ccs", "biomass", "beccs", "ammonia", "h2")
SOURCE_COLUMNS = [
    "year", "source_type", "source_id", "sector", "target_group", "province", "basin_code", "longitude", "latitude",
    "activity", "activity_unit", "baseline_co2_mtpa", "reduction_co2_mtpa", "residual_co2_mtpa", "captured_co2_mtpa",
    *[f"share_{route}" for route in ROUTES], "dominant_route",
    "biomass_gj", "ammonia_kg", "h2_kg", "water_consumption_m3", "water_withdrawal_m3",
    # 煤电
    "capacity_mw", "retirement_year", "cooling", "rebuild_share", "air_operating_share", "air_installed_share",
    "already_air_share", "biomass_blend_level", "ammonia_blend_level", "biomass_blend_ratio", "beccs_blend_ratio",
    "ammonia_blend_ratio", "min_distance_to_storage_km",
    # 工业
    "output_index", "abatable_production_share", "process_co2_mtpa", "capacity_ccs_mtpa", "capacity_h2_mtpa",
    "new_capacity_ccs_mtpa", "new_capacity_h2_mtpa", "water_base_m3", "water_capture_increment_m3",
    "h2_price_paid_cny_per_kg", "h2_price_national_mean_cny_per_kg",
]
ROUTE_COLUMNS = ["year", "source_type", "source_id", "route", "share", "activity", "activity_unit",
                 "baseline_co2_mtpa", "reduction_co2_mtpa", "captured_co2_mtpa", "enabled"]
_COAL_ONLY = ["capacity_mw", "retirement_year", "air_operating_share", "air_installed_share", "already_air_share",
              "biomass_blend_level", "ammonia_blend_level", "biomass_blend_ratio", "beccs_blend_ratio",
              "ammonia_blend_ratio", "min_distance_to_storage_km"]
# 只对工业有意义的列：源表列名 -> 工业明细（`results_industry`）的列名；明细把年量写作 `_mt`，源表统一写 `_mtpa`。
_INDUSTRY_ONLY = {
    "output_index": "output_index", "abatable_production_share": "abatable_production_share",
    "process_co2_mtpa": "process_co2_mt", "capacity_ccs_mtpa": "capacity_ccs_mt", "capacity_h2_mtpa": "capacity_h2_mt",
    "new_capacity_ccs_mtpa": "new_capacity_ccs_mt", "new_capacity_h2_mtpa": "new_capacity_h2_mt",
    "water_base_m3": "water_base_m3", "water_capture_increment_m3": "water_capture_increment_m3",
    "h2_price_paid_cny_per_kg": "h2_price_paid_cny_per_kg",
    "h2_price_national_mean_cny_per_kg": "h2_price_national_mean_cny_per_kg",
}


def _build_sources_table(
    prepared: PreparedInputs, ys: YearSolution, plant_detail: pd.DataFrame, industry_detail: pd.DataFrame
) -> pd.DataFrame:
    """一年的源表：`plant_detail` 与 `industry_detail` 是本年的煤电明细（`results_plant._build_plant_detail_table`）与
    工业明细（`results_industry._build_industry_detail_table`），换成统一的列。"""
    coal = pd.DataFrame({
        "year": plant_detail["year"], "source_type": "coal", "source_id": plant_detail["plant_id"].astype(str),
        "sector": "coal", "target_group": POWER_TARGET_GROUP, "province": plant_detail["province_name"],
        "basin_code": prepared.plants["basin_code"].astype(str).to_numpy() if "basin_code" in prepared.plants else "",
        "longitude": plant_detail["centroid_longitude"], "latitude": plant_detail["centroid_latitude"],
        "activity": plant_detail["annual_generation_mwh"], "activity_unit": "MWh/yr",
        "baseline_co2_mtpa": plant_detail["baseline_emissions_mt"], "reduction_co2_mtpa": plant_detail["reduction_mt"],
        "residual_co2_mtpa": plant_detail["baseline_emissions_mt"] - plant_detail["reduction_mt"],
        "captured_co2_mtpa": plant_detail["captured_mt"],
        **{f"share_{route}": plant_detail.get(f"share_{route}", 0.0) for route in ROUTES},
        "dominant_route": plant_detail["dominant_pathway"],
        "biomass_gj": plant_detail["biomass_use_gj"], "ammonia_kg": plant_detail["ammonia_use_kg"],
        "h2_kg": plant_detail["ammonia_use_kg"] * NH3_H2_RATIO,
        "water_consumption_m3": plant_detail["water_use_m3"], "water_withdrawal_m3": _coal_withdrawal_m3(ys),
        "cooling": plant_detail["dominant_cooling"], "rebuild_share": np.asarray(ys["rebuild"], dtype=np.float64),
        **{column: plant_detail[column] for column in _COAL_ONLY},
    })
    routes = [f"share_{route}" for route in INDUSTRY_ROUTES]
    industry = pd.DataFrame({
        "year": industry_detail["year"], "source_type": "industry", "source_id": industry_detail["hub_id"].astype(str),
        "sector": industry_detail["sector"], "target_group": industry_detail["target_group"],
        "province": industry_detail["province"], "basin_code": industry_detail["basin_code"],
        "longitude": industry_detail["longitude"], "latitude": industry_detail["latitude"],
        "activity": industry_detail["production_kt_per_year"], "activity_unit": "kt/yr",
        "baseline_co2_mtpa": industry_detail["baseline_co2_mt"], "reduction_co2_mtpa": industry_detail["reduction_mt"],
        "residual_co2_mtpa": industry_detail["residual_mt"], "captured_co2_mtpa": industry_detail["captured_mt"],
        **{f"share_{route}": industry_detail.get(f"share_{route}", 0.0) for route in ROUTES},
        "dominant_route": [INDUSTRY_ROUTES[k] for k in np.argmax(industry_detail[routes].to_numpy(), axis=1)]
        if len(industry_detail) else [],
        "biomass_gj": 0.0, "ammonia_kg": 0.0, "h2_kg": industry_detail["h2_kg"],
        "water_consumption_m3": np.nan, "water_withdrawal_m3": industry_detail["water_m3"],
        **{column: industry_detail[detail] for column, detail in _INDUSTRY_ONLY.items()},
    })
    table = _stack((coal, industry), SOURCE_COLUMNS)
    table["retirement_year"] = table["retirement_year"].astype("Int64")  # 工业行为空，整数列不变成浮点
    return table


def _coal_withdrawal_m3(ys: YearSolution) -> np.ndarray:
    """逐厂取水，m3/yr：Σ_路径 发电量 x (取水强度 x 份额 − (取水强度 − 空冷取水强度) x 空冷份额)，与流域约束同式
    （`model_resources.add_water_balances`）。没建流域取水约束时（无水约束、关掉流域上限）取水强度矩阵不生成，记空。"""
    yd = ys["year_data"]
    share = np.asarray(ys["share"], dtype=np.float64)
    if yd.withdrawal_intensity is None:
        return np.full(len(share), np.nan)
    gen = np.asarray(yd.generation_by_pathway, dtype=np.float64)
    withdrawal = np.asarray(yd.withdrawal_intensity, dtype=np.float64)
    use = gen * withdrawal * share
    if bool(yd.allow_air_cooling_retrofit) and yd.air_withdrawal_intensity is not None:
        use = use - gen * (withdrawal - np.asarray(yd.air_withdrawal_intensity, dtype=np.float64)) * ys["air_share"]
    return use.sum(axis=1)


def _build_source_route_table(
    prepared: PreparedInputs, ys: YearSolution, year: int, pathways: pd.DataFrame
) -> pd.DataFrame:
    """一年的逐路线表；`pathways` 是本年的煤电逐厂 x 路径表（`results_plant._build_pathway_table`）。"""
    coal = pd.DataFrame({
        "year": pathways["year"], "source_type": "coal", "source_id": pathways["plant_id"].astype(str),
        "route": pathways["pathway"], "share": pathways["share"], "activity": pathways["annual_generation_mwh"],
        "activity_unit": "MWh/yr", "baseline_co2_mtpa": pathways["baseline_emissions_mt"],
        "reduction_co2_mtpa": pathways["abatement_mt"], "captured_co2_mtpa": pathways["captured_mt"],
        "enabled": pathways["enabled"],
    })
    iy = ys["year_data"].industry
    hubs = prepared.industry.hubs
    share = np.asarray(ys["industry_share"], dtype=np.float64)
    production = hubs["production_kt_per_year"].astype(float).to_numpy() * np.asarray(iy.output_scale, dtype=np.float64)
    abatable = (hubs["abatable_production_share"].astype(float).to_numpy()
                if "abatable_production_share" in hubs else np.ones(len(hubs)))
    activity = production[:, None] * abatable[:, None] * share
    activity[:, UNABATED] = production - np.delete(activity, UNABATED, axis=1).sum(axis=1)
    count = len(INDUSTRY_ROUTES)
    industry = pd.DataFrame({
        "year": year, "source_type": "industry", "source_id": np.repeat(hubs["hub_id"].astype(str).to_numpy(), count),
        "route": np.tile(INDUSTRY_ROUTES, len(hubs)), "share": share.ravel(), "activity": activity.ravel(),
        "activity_unit": "kt/yr", "baseline_co2_mtpa": np.nan,
        "reduction_co2_mtpa": (np.asarray(iy.reduction_mt) * share).ravel(),
        "captured_co2_mtpa": (np.asarray(iy.captured_mt) * share).ravel(),
        "enabled": np.asarray(iy.route_available, dtype=bool).ravel(),
    })
    return _stack((coal, industry), ROUTE_COLUMNS)


def _stack(frames: tuple[pd.DataFrame, ...], columns: list[str]) -> pd.DataFrame:
    """煤电与工业两块上下拼接、按 `columns` 排列；空块不拼（没有工业 hub 的情景），都空时是列齐全的空表。"""
    kept = [frame for frame in frames if len(frame)]
    if not kept:
        return pd.DataFrame(columns=columns)
    return pd.concat(kept, ignore_index=True, sort=False).reindex(columns=columns)

