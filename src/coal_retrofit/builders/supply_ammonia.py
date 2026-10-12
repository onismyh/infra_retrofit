"""绿氨供给曲线：`data/H2/<年>/` 下逐 (资源, 电解技术) 的绿氢产量、LCOH、LCOE 栅格按省与 `AMMONIA_NODE_AGGREGATION_DEGREES`
网格聚合成节点（`build_ammonia_supply_dataframe`），到厂氨价 = 产量加权 LCOH × `NH3_H2_RATIO` + 合成岛电耗 + 合成岛 capex 年金 +
储存 + 运输；求解时 `reprice_h2_component` / `reprice_hb_capex` 按当前氢耗与情景贴现率换掉表里的两项。
生物质供给与写盘入口在 `supply`。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ..constants import (
    AMMONIA_NODE_AGGREGATION_DEGREES,
    BIOMASS_MATCH_BUFFER_KM,
    DEFAULT_DISCOUNT_RATE,
    NH3_ELECTROLYSIS_TECHS,
    NH3_H2_RATIO,
    NH3_HB_CAPEX_LIFETIME_YEARS,
    NH3_HB_CAPEX_USD_PER_TONNE_YEAR,
    NH3_HB_POWER_KWH_PER_KG,
    NH3_STORAGE_ADDER_USD_PER_KG,
    NH3_TRANSPORT_ADDER_USD_PER_KG,
)
from ..constants_industry_capture import capital_recovery_factor
from ..paths import ProjectPaths
from ..spatial import aggregate_to_cells, load_provinces, pixel_centers_lonlat, province_lookup, rasterize_provinces_to_match

logger = logging.getLogger(__name__)

H2_PATTERN = re.compile(r"(?P<resource>.+)_h2_production_(?P<tech>.+)\.tif")

AMMONIA_COLUMNS = (
    "ammonia_node_id",
    "province_id",
    "province_name",
    "year",
    "resource",
    "electrolysis_tech",
    "longitude",
    "latitude",
    "h2_supply_kg_per_year",
    "weighted_lcoh_usd_per_kg_h2",
    "weighted_lcoe_usd_per_mwh",
    "nh3_supply_kg_per_year",
    "nh3_h2_cost_component_usd_per_kg",
    "hb_power_need_kwh_per_kg_nh3",
    "nh3_hb_power_cost_usd_per_kg",
    "nh3_hb_capex_usd_per_kg",
    "nh3_storage_adder_usd_per_kg",
    "nh3_transport_adder_usd_per_kg",
    "nh3_cost_lb_usd_per_kg",
    "match_rule",
    "competition_scope",
    "aggregation_cell_deg",
    "buffer_km",
    "source",
    "year_basis",
)


def hb_capex_annuity_usd_per_kg(discount_rate: float) -> float:
    """Haber-Bosch + ASU 的年化资本成本，以每生产 1 kg NH3 的 USD 计。

    构建输入时按 `DEFAULT_DISCOUNT_RATE` 写进 `ammonia_supply_curve.csv`；求解时
    `reprice_hb_capex` 再按情景的 `discount_rate` 算一次，换掉 CSV 里的那一份。

    Args:
        discount_rate: 年贴现率。

    Returns:
        按满负荷、`NH3_HB_CAPEX_LIFETIME_YEARS` 年折算的每 kg NH3 年金（USD）。
    """
    crf = capital_recovery_factor(discount_rate, NH3_HB_CAPEX_LIFETIME_YEARS)
    return NH3_HB_CAPEX_USD_PER_TONNE_YEAR * crf / 1000.0


def reprice_h2_component(ammonia: pd.DataFrame) -> pd.DataFrame:
    """把氨价里的氢成本换成按当前 `NH3_H2_RATIO` 折算的，返回新表，不改动传入的表。

    `nh3_cost_lb_usd_per_kg` 减去表里的 `nh3_h2_cost_component_usd_per_kg`、加上 `weighted_lcoh_usd_per_kg_h2 x NH3_H2_RATIO`，
    两列随之改写：表里那一份是构建输入时按当时的比例算的（2026-10-10 前的输入按 0.176）。没有这两列的表
    （toy 测试的输入）原样使用。由 `optimization.data_prep._prepare_ammonia_supply` 在求解时调用。

    Raises:
        ValueError: 两列有缺失值或非数值时，免得 NaN 价格进入目标函数。
    """
    repriced = ammonia.copy()
    if not {"nh3_h2_cost_component_usd_per_kg", "weighted_lcoh_usd_per_kg_h2"} <= set(repriced.columns):
        return repriced
    cost = pd.to_numeric(repriced["nh3_cost_lb_usd_per_kg"], errors="coerce")
    old_h2 = pd.to_numeric(repriced["nh3_h2_cost_component_usd_per_kg"], errors="coerce")
    lcoh = pd.to_numeric(repriced["weighted_lcoh_usd_per_kg_h2"], errors="coerce")
    bad = cost.isna() | old_h2.isna() | lcoh.isna()
    if bad.any():
        nodes = repriced.loc[bad, "ammonia_node_id"] if "ammonia_node_id" in repriced.columns else repriced.index[bad]
        raise ValueError(
            "ammonia supply curve has missing or non-numeric nh3_cost_lb / nh3_h2_cost_component / weighted_lcoh "
            f"in {int(bad.sum())} rows; ammonia_node_id: {sorted({str(node) for node in nodes})[:10]}"
        )
    new_h2 = lcoh * NH3_H2_RATIO
    repriced["nh3_cost_lb_usd_per_kg"] = cost - old_h2 + new_h2
    repriced["nh3_h2_cost_component_usd_per_kg"] = new_h2
    return repriced


def reprice_hb_capex(ammonia: pd.DataFrame, discount_rate: float) -> pd.DataFrame:
    """把氨供给曲线里的合成岛年金换成按 `discount_rate` 算的，返回新表，不改动传入的表。

    `nh3_cost_lb_usd_per_kg` 减去表里的 `nh3_hb_capex_usd_per_kg`、加上新年金，该列随之改写。
    表里那一份是构建输入时算的（2026-09-23 前的输入按 8%、20 年），模型自己折现与折年金的地方则都用情景贴现率。
    由 `optimization.data_prep._prepare_ammonia_supply` 在求解时调用。没有这一列的表
    （toy 测试的输入）不拆分成本，原样使用并记一条告警日志。

    Args:
        ammonia: 读入的 `ammonia_supply_curve.csv`。
        discount_rate: 情景贴现率。

    Returns:
        换过年金的副本；缺这一列时是原表的副本。

    Raises:
        ValueError: 年金列或成本列有缺失值或非数值时，列出涉及的节点，免得 NaN 价格进入目标函数。
    """
    repriced = ammonia.copy()
    if "nh3_hb_capex_usd_per_kg" not in repriced.columns:
        logger.warning(
            "ammonia supply curve has no nh3_hb_capex_usd_per_kg column; "
            "using nh3_cost_lb_usd_per_kg as is (no re-annuitisation at the scenario discount rate)"
        )
        return repriced
    cost = pd.to_numeric(repriced["nh3_cost_lb_usd_per_kg"], errors="coerce")
    old_annuity = pd.to_numeric(repriced["nh3_hb_capex_usd_per_kg"], errors="coerce")
    bad = cost.isna() | old_annuity.isna()
    if bad.any():
        nodes = repriced.loc[bad, "ammonia_node_id"] if "ammonia_node_id" in repriced.columns else repriced.index[bad]
        raise ValueError(
            "ammonia supply curve has missing or non-numeric nh3_cost_lb_usd_per_kg / nh3_hb_capex_usd_per_kg "
            f"in {int(bad.sum())} rows; ammonia_node_id: {sorted({str(node) for node in nodes})[:10]}"
        )
    annuity = hb_capex_annuity_usd_per_kg(discount_rate)
    repriced["nh3_cost_lb_usd_per_kg"] = cost - old_annuity + annuity
    repriced["nh3_hb_capex_usd_per_kg"] = annuity
    return repriced


def parse_h2_name(name: str) -> tuple[str, str] | None:
    match = H2_PATTERN.fullmatch(name)
    if not match:
        return None
    return match.group("resource"), match.group("tech")


def _pixel_values(path: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """栅格在像元 (rows, cols) 上的值，NaN 与无穷记 0。"""
    with rasterio.open(path) as src:
        values = np.nan_to_num(src.read(1), nan=0.0, posinf=0.0, neginf=0.0)
    return values[rows, cols].astype(np.float64)


def _h2_pixels(
    prod_path: Path, lcoh_path: Path, lcoe_path: Path, provinces: pd.DataFrame, zone_cache: dict[str, np.ndarray]
) -> pd.DataFrame | None:
    """一张产量栅格上有产量且落在省界内的像元：省号、经纬度、氢产量与 LCOH / LCOE × 产量（加权用）；没有这样的像元时为 None。

    省界栅格化按 (CRS, 高, 宽, 仿射) 缓存在 `zone_cache` 里，同一网格的多张栅格只栅格化一次。
    """
    with rasterio.open(prod_path) as prod_src:
        prod = prod_src.read(1).astype(np.float64)
        transform, crs = prod_src.transform, prod_src.crs
        cache_key = f"{crs.to_wkt() if crs else 'none'}|{prod_src.height}|{prod_src.width}|{transform}"
    if cache_key not in zone_cache:
        zone_cache[cache_key], _ = rasterize_provinces_to_match(provinces, prod_path)
    zone_grid = zone_cache[cache_key]
    prod = np.nan_to_num(prod, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (prod > 0.0) & (zone_grid > 0)
    if not np.any(valid):
        return None
    rows, cols = np.where(valid)
    # 栅格单位为 kg H₂/yr/km²（Albers 1km 投影）：每个像元本身就代表 1 km²，无需再乘面积。
    h2_supply = prod[rows, cols]
    lcoh = _pixel_values(lcoh_path, rows, cols)
    # Haber-Bosch 合成回路与 ASU 的用电按同一地点的 LCOE（USD/MWh）购入，而不是像之前的成本下界那样假定免费。
    if lcoe_path.exists():
        lcoe = _pixel_values(lcoe_path, rows, cols)
    else:
        logger.warning("Missing LCOE raster %s; HB power cost set to zero", lcoe_path.name)
        lcoe = np.zeros_like(lcoh)
    lons, lats = pixel_centers_lonlat(transform, crs, rows, cols)
    return pd.DataFrame(
        {
            "province_id": zone_grid[rows, cols].astype(np.int32),
            "longitude": lons,
            "latitude": lats,
            "h2_supply_kg_per_year": h2_supply,
            "weighted_lcoh_component": lcoh * h2_supply,
            "weighted_lcoe_component": lcoe * h2_supply,
        }
    )


def _price_nodes(nodes: pd.DataFrame, year_dir: Path, resource: str, tech: str, lookup: dict[int, str], source: str) -> pd.DataFrame:
    """就地给聚合后的节点加列：产量加权的 LCOH / LCOE、折成的氨量与到厂氨价各项。"""
    supply = nodes["h2_supply_kg_per_year"]
    nodes["weighted_lcoh_usd_per_kg_h2"] = np.where(supply > 0, nodes["weighted_lcoh_component"] / supply, np.nan)
    nodes["weighted_lcoe_usd_per_mwh"] = np.where(supply > 0, nodes["weighted_lcoe_component"] / supply, np.nan)
    nodes["province_name"] = nodes["province_id"].map(lookup)
    nodes["year"] = int(year_dir.name)
    nodes["resource"] = resource
    nodes["electrolysis_tech"] = tech
    nodes["nh3_supply_kg_per_year"] = supply / NH3_H2_RATIO
    nodes["nh3_h2_cost_component_usd_per_kg"] = nodes["weighted_lcoh_usd_per_kg_h2"] * NH3_H2_RATIO
    nodes["hb_power_need_kwh_per_kg_nh3"] = NH3_HB_POWER_KWH_PER_KG
    # Haber-Bosch 用电，按同址可再生能源的 LCOE 计价。
    nodes["nh3_hb_power_cost_usd_per_kg"] = NH3_HB_POWER_KWH_PER_KG * nodes["weighted_lcoe_usd_per_mwh"].fillna(0.0) / 1000.0
    nodes["nh3_hb_capex_usd_per_kg"] = hb_capex_annuity_usd_per_kg(DEFAULT_DISCOUNT_RATE)
    nodes["nh3_storage_adder_usd_per_kg"] = NH3_STORAGE_ADDER_USD_PER_KG
    nodes["nh3_transport_adder_usd_per_kg"] = NH3_TRANSPORT_ADDER_USD_PER_KG
    # 完整到岸成本：H2 原料 + HB/ASU 电力 + HB/ASU 资本 + 储存。（为兼容下游而保留该列名；它已不再是下界。）
    nodes["nh3_cost_lb_usd_per_kg"] = (
        nodes["nh3_h2_cost_component_usd_per_kg"]
        + nodes["nh3_hb_power_cost_usd_per_kg"]
        + nodes["nh3_hb_capex_usd_per_kg"]
        + NH3_STORAGE_ADDER_USD_PER_KG
        + NH3_TRANSPORT_ADDER_USD_PER_KG
    )
    nodes["source"] = source
    nodes["year_basis"] = f"cost_year_{year_dir.name}"
    return nodes


def build_ammonia_supply_dataframe(paths: ProjectPaths) -> pd.DataFrame:
    """`data/H2/<年>/` 下每张 `NH3_ELECTROLYSIS_TECHS` 路线、且有配套 LCOH 栅格的产量栅格各聚合成一组节点，
    按 (年, 省, 资源, 技术, 网格) 排序后统一编号；一张都没有时返回只有列名的空表。"""
    provinces = load_provinces(paths.data_dir / "ChinaMap" / "provinces.shp")
    h2_root = paths.data_dir / "H2"
    year_dirs = sorted(path for path in h2_root.iterdir() if path.is_dir() and path.name.isdigit())
    lookup = province_lookup(provinces)
    zone_cache: dict[str, np.ndarray] = {}
    frames: list[pd.DataFrame] = []

    for year_dir in year_dirs:
        for prod_path in sorted(year_dir.glob("*_h2_production_*.tif")):
            parsed = parse_h2_name(prod_path.name)
            if parsed is None:
                continue
            resource, tech = parsed
            lcoh_path = year_dir / f"{resource}_lcoh_{tech}.tif"
            if tech not in NH3_ELECTROLYSIS_TECHS or not lcoh_path.exists():
                continue
            pixels = _h2_pixels(prod_path, lcoh_path, year_dir / f"{resource}_lcoe.tif", provinces, zone_cache)
            if pixels is None:
                continue
            nodes = aggregate_to_cells(pixels, "h2_supply_kg_per_year", AMMONIA_NODE_AGGREGATION_DEGREES)
            frames.append(_price_nodes(nodes, year_dir, resource, tech, lookup, paths.rel(prod_path)))

    if not frames:
        return pd.DataFrame(columns=list(AMMONIA_COLUMNS))
    ammonia = pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["year", "province_id", "resource", "electrolysis_tech", "lon_bin", "lat_bin"]
    ).reset_index(drop=True)
    ammonia["ammonia_node_id"] = [f"A{index:06d}" for index in range(1, len(ammonia) + 1)]
    ammonia["match_rule"] = "point_source_competition"
    ammonia["competition_scope"] = "shared_ammonia_node"
    ammonia["aggregation_cell_deg"] = AMMONIA_NODE_AGGREGATION_DEGREES
    ammonia["buffer_km"] = BIOMASS_MATCH_BUFFER_KM
    return ammonia[list(AMMONIA_COLUMNS)].reset_index(drop=True)
