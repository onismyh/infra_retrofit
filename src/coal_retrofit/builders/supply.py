"""供给侧输入表：生物质供给曲线（`build_biomass_supply_dataframe`）与写盘入口 `write_supply_inputs`；绿氨供给曲线及其求解时重定价在
`supply_ammonia`。栅格像元中心与按省-网格聚合的工具在 `spatial`。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from ..artifacts import write_csv
from ..constants import (
    BIOMASS_COST_BASE,
    BIOMASS_COST_HIGH,
    BIOMASS_COST_LOW,
    BIOMASS_GJ_PER_TONNE,
    BIOMASS_NODE_AGGREGATION_DEGREES,
)
from ..paths import ProjectPaths
from ..spatial import (
    aggregate_to_cells,
    latlon_row_areas,
    load_provinces,
    pixel_centers,
    province_lookup,
    rasterize_provinces_to_match,
)
from .supply_ammonia import build_ammonia_supply_dataframe

BIOMASS_COLUMNS = (
    "biomass_node_id",
    "province_id",
    "province_name",
    "longitude",
    "latitude",
    "available_gj",
    "biomass_supply_gj",
    "biomass_supply_tonnes_equiv",
    "unit",
    "mass_conversion_gj_per_tonne",
    "base_cost_cny_per_gj",
    "low_cost_cny_per_gj",
    "high_cost_cny_per_gj",
    "match_rule",
    "competition_scope",
    "aggregation_cell_deg",
    "land_competition_included",
    "source",
    "year_basis",
)


def build_biomass_supply_dataframe(paths: ProjectPaths) -> pd.DataFrame:
    """Wang 2023 生物质栅格（GJ/km²）按省与 `BIOMASS_NODE_AGGREGATION_DEGREES` 网格聚合成供给节点，坐标按资源量加权。"""
    provinces = load_provinces(paths.data_dir / "ChinaMap" / "provinces.shp")
    # Wang et al. 2023（Sci Data 10:384）的情景 4：农林残余物，不含能源作物，与采购成本（残余物）同口径。
    # 2026-10-11 之前读情景 3（`bioenergy_s3_AFE_abandon.tif`，另含废弃耕地上的能源作物）。
    biomass_tif = paths.find_data_file("bioenergy_s4_AF_max.tif")

    with rasterio.open(biomass_tif) as src:
        biomass = src.read(1).astype(np.float64)
        row_areas = latlon_row_areas(src.transform, src.height)
        pixel_area = np.broadcast_to(row_areas[:, None], biomass.shape)
        zone_grid, _ = rasterize_provinces_to_match(provinces, biomass_tif)
        # 栅格单位为 GJ/km²（Wang et al. 2023, Sci Data）；
        # pixel_area 的单位是 m²。换算：GJ = (GJ/km²) × (m² / 1e6)。
        total_gj_grid = np.nan_to_num(biomass, nan=0.0, posinf=0.0, neginf=0.0) * pixel_area / 1e6
        valid_mask = (total_gj_grid > 0.0) & (zone_grid > 0)
        rows, cols = np.where(valid_mask)
        lons, lats = pixel_centers(src.transform, rows, cols)

    points = pd.DataFrame(
        {
            "province_id": zone_grid[rows, cols].astype(np.int32),
            "longitude": lons,
            "latitude": lats,
            "available_gj": total_gj_grid[rows, cols].astype(np.float64),
        }
    )
    grouped = aggregate_to_cells(points, "available_gj", BIOMASS_NODE_AGGREGATION_DEGREES)
    grouped["province_name"] = grouped["province_id"].map(province_lookup(provinces))
    grouped["biomass_node_id"] = [f"B{index:05d}" for index in range(1, len(grouped) + 1)]
    grouped["biomass_supply_gj"] = grouped["available_gj"]
    grouped["biomass_supply_tonnes_equiv"] = grouped["available_gj"] / BIOMASS_GJ_PER_TONNE
    grouped["unit"] = "GJ/yr"
    grouped["mass_conversion_gj_per_tonne"] = BIOMASS_GJ_PER_TONNE
    grouped["base_cost_cny_per_gj"] = BIOMASS_COST_BASE
    grouped["low_cost_cny_per_gj"] = BIOMASS_COST_LOW
    grouped["high_cost_cny_per_gj"] = BIOMASS_COST_HIGH
    grouped["match_rule"] = "buffered_point_source_competition"
    grouped["competition_scope"] = "shared_biomass_node"
    grouped["aggregation_cell_deg"] = BIOMASS_NODE_AGGREGATION_DEGREES
    grouped["land_competition_included"] = False
    grouped["source"] = paths.rel(biomass_tif)
    grouped["year_basis"] = "static_resource_layer"

    return grouped[list(BIOMASS_COLUMNS)].sort_values("biomass_node_id").reset_index(drop=True)


def write_supply_inputs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    biomass_nodes = build_biomass_supply_dataframe(paths)
    ammonia_nodes = build_ammonia_supply_dataframe(paths)
    write_csv(biomass_nodes, paths.inputs_dir / "biomass_supply_curve.csv")
    write_csv(ammonia_nodes, paths.inputs_dir / "ammonia_supply_curve.csv")
    return biomass_nodes, ammonia_nodes
