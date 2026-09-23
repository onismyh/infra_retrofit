from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from ..artifacts import write_csv
from ..constants import (
    AMMONIA_NODE_AGGREGATION_DEGREES,
    BIOMASS_COST_BASE,
    BIOMASS_COST_HIGH,
    BIOMASS_COST_LOW,
    BIOMASS_GJ_PER_TONNE,
    BIOMASS_MATCH_BUFFER_KM,
    BIOMASS_NODE_AGGREGATION_DEGREES,
    NH3_ELECTROLYSIS_TECHS,
    NH3_H2_RATIO,
    NH3_HB_CAPEX_DISCOUNT_RATE,
    NH3_HB_CAPEX_LIFETIME_YEARS,
    NH3_HB_CAPEX_USD_PER_TONNE_YEAR,
    NH3_HB_POWER_KWH_PER_KG,
    NH3_STORAGE_ADDER_USD_PER_KG,
    NH3_TRANSPORT_ADDER_USD_PER_KG,
)
from ..paths import ProjectPaths
from ..spatial import latlon_row_areas, load_provinces, rasterize_provinces_to_match


logger = logging.getLogger(__name__)

H2_PATTERN = re.compile(r"(?P<resource>.+)_h2_production_(?P<tech>.+)\.tif")


def province_lookup(provinces) -> dict[int, str]:
    return dict(zip(provinces["province_id"], provinces["province_name"], strict=True))


def _bin_coordinates(values: np.ndarray, cell_degrees: float, offset: float) -> np.ndarray:
    return np.floor((values + offset) / cell_degrees).astype(np.int32)


def _pixel_centers(transform, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = transform.c + (cols.astype(np.float64) + 0.5) * transform.a
    lat = transform.f + (rows.astype(np.float64) + 0.5) * transform.e
    return lon, lat


def _pixel_centers_lonlat(
    transform,
    crs,
    rows: np.ndarray,
    cols: np.ndarray,
    transformer_cache: dict[str, Transformer] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = _pixel_centers(transform, rows, cols)
    if crs is None or str(crs).upper() == "EPSG:4326":
        return x, y

    cache_key = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
    transformer = None
    if transformer_cache is not None:
        transformer = transformer_cache.get(cache_key)
    if transformer is None:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        if transformer_cache is not None:
            transformer_cache[cache_key] = transformer
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _hb_capex_annuity_usd_per_kg() -> float:
    """Annualised Haber-Bosch + ASU capital cost, USD per kg of NH3 produced."""
    rate = NH3_HB_CAPEX_DISCOUNT_RATE
    years = NH3_HB_CAPEX_LIFETIME_YEARS
    crf = rate / (1.0 - (1.0 + rate) ** (-years)) if rate > 0 else 1.0 / years
    return NH3_HB_CAPEX_USD_PER_TONNE_YEAR * crf / 1000.0


def _haversine_distances_km(
    origin_lon: float,
    origin_lat: float,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
) -> np.ndarray:
    origin_lon_rad = np.radians(origin_lon)
    origin_lat_rad = np.radians(origin_lat)
    target_lons_rad = np.radians(target_lons.astype(np.float64))
    target_lats_rad = np.radians(target_lats.astype(np.float64))
    delta_lon = target_lons_rad - origin_lon_rad
    delta_lat = target_lats_rad - origin_lat_rad
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(origin_lat_rad) * np.cos(target_lats_rad) * np.sin(delta_lon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_biomass_supply_dataframe(paths: ProjectPaths) -> pd.DataFrame:
    provinces = load_provinces(paths.data_dir / "ChinaMap" / "provinces.shp")
    biomass_tif = paths.find_data_file("bioenergy_s3_AFE_abandon.tif")
    lookup = province_lookup(provinces)

    with rasterio.open(biomass_tif) as src:
        biomass = src.read(1).astype(np.float64)
        row_areas = latlon_row_areas(src.transform, src.height)
        pixel_area = np.broadcast_to(row_areas[:, None], biomass.shape)
        zone_grid, _ = rasterize_provinces_to_match(provinces, biomass_tif)
        # Raster unit is GJ/km²  (Wang et al. 2023, Sci Data);
        # pixel_area is in m².  Convert: GJ = (GJ/km²) × (m² / 1e6).
        total_gj_grid = np.nan_to_num(biomass, nan=0.0, posinf=0.0, neginf=0.0) * pixel_area / 1e6
        valid_mask = (total_gj_grid > 0.0) & (zone_grid > 0)
        rows, cols = np.where(valid_mask)
        lons, lats = _pixel_centers(src.transform, rows, cols)

    frame = pd.DataFrame(
        {
            "province_id": zone_grid[rows, cols].astype(np.int32),
            "longitude": lons,
            "latitude": lats,
            "available_gj": total_gj_grid[rows, cols].astype(np.float64),
        }
    )
    frame["lon_bin"] = _bin_coordinates(frame["longitude"].to_numpy(), BIOMASS_NODE_AGGREGATION_DEGREES, 180.0)
    frame["lat_bin"] = _bin_coordinates(frame["latitude"].to_numpy(), BIOMASS_NODE_AGGREGATION_DEGREES, 90.0)

    weighted = frame.assign(
        weighted_lon=frame["longitude"] * frame["available_gj"],
        weighted_lat=frame["latitude"] * frame["available_gj"],
    )
    grouped = (
        weighted.groupby(["province_id", "lon_bin", "lat_bin"], as_index=False)[
            ["available_gj", "weighted_lon", "weighted_lat"]
        ]
        .sum()
        .sort_values(["province_id", "lon_bin", "lat_bin"])
        .reset_index(drop=True)
    )
    grouped["longitude"] = np.where(
        grouped["available_gj"] > 0,
        grouped["weighted_lon"] / grouped["available_gj"],
        0.0,
    )
    grouped["latitude"] = np.where(
        grouped["available_gj"] > 0,
        grouped["weighted_lat"] / grouped["available_gj"],
        0.0,
    )
    grouped["province_name"] = grouped["province_id"].map(lookup)
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
    grouped["buffer_km"] = BIOMASS_MATCH_BUFFER_KM
    grouped["aggregation_cell_deg"] = BIOMASS_NODE_AGGREGATION_DEGREES
    grouped["land_competition_included"] = False
    grouped["source"] = paths.rel(biomass_tif)
    grouped["year_basis"] = "static_resource_layer"

    return grouped[
        [
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
            "buffer_km",
            "aggregation_cell_deg",
            "land_competition_included",
            "source",
            "year_basis",
        ]
    ].sort_values("biomass_node_id").reset_index(drop=True)


def build_biomass_link_dataframe(
    paths: ProjectPaths,
    biomass_nodes: pd.DataFrame,
) -> pd.DataFrame:
    plants = pd.read_csv(paths.inputs_dir / "plants.csv").copy()
    plant_count = len(plants)
    node_lons = biomass_nodes["longitude"].astype(float).to_numpy()
    node_lats = biomass_nodes["latitude"].astype(float).to_numpy()
    rows: list[dict[str, object]] = []

    for plant in plants.itertuples(index=False):
        distances_km = _haversine_distances_km(
            float(plant.centroid_longitude),
            float(plant.centroid_latitude),
            node_lons,
            node_lats,
        )
        matched_indices = np.flatnonzero(distances_km <= BIOMASS_MATCH_BUFFER_KM)
        if matched_indices.size == 0:
            continue
        ranked_indices = matched_indices[np.argsort(distances_km[matched_indices])]
        for rank, node_idx in enumerate(ranked_indices, start=1):
            node = biomass_nodes.iloc[int(node_idx)]
            rows.append(
                {
                    "plant_count": plant_count,
                    "plant_id": str(plant.plant_id),
                    "plant_index": int(plant.plant_index),
                    "plant_province_name": str(plant.province_mode),
                    "biomass_node_id": str(node["biomass_node_id"]),
                    "biomass_node_province_name": str(node["province_name"]),
                    "distance_km": round(float(distances_km[node_idx]), 3),
                    "distance_rank": rank,
                    "base_cost_cny_per_gj": float(node["base_cost_cny_per_gj"]),
                    "low_cost_cny_per_gj": float(node["low_cost_cny_per_gj"]),
                    "high_cost_cny_per_gj": float(node["high_cost_cny_per_gj"]),
                    "base_delivered_cost_cny_per_gj": float(node["base_cost_cny_per_gj"]),
                    "match_rule": "plant_buffer_intersects_biomass_node",
                    "buffer_km": BIOMASS_MATCH_BUFFER_KM,
                    "source": str(node["source"]),
                    "year_basis": str(node["year_basis"]),
                }
            )

    columns = [
        "plant_count",
        "plant_id",
        "plant_index",
        "plant_province_name",
        "biomass_node_id",
        "biomass_node_province_name",
        "distance_km",
        "distance_rank",
        "base_cost_cny_per_gj",
        "low_cost_cny_per_gj",
        "high_cost_cny_per_gj",
        "base_delivered_cost_cny_per_gj",
        "match_rule",
        "buffer_km",
        "source",
        "year_basis",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["plant_id", "distance_km", "biomass_node_id"]
    ).reset_index(drop=True)


def parse_h2_name(name: str) -> tuple[str, str] | None:
    match = H2_PATTERN.fullmatch(name)
    if not match:
        return None
    return match.group("resource"), match.group("tech")


def build_ammonia_supply_dataframe(paths: ProjectPaths) -> pd.DataFrame:
    provinces = load_provinces(paths.data_dir / "ChinaMap" / "provinces.shp")
    h2_root = paths.data_dir / "H2"
    year_dirs = sorted(path for path in h2_root.iterdir() if path.is_dir() and path.name.isdigit())
    lookup = province_lookup(provinces)
    zone_cache: dict[str, np.ndarray] = {}
    transformer_cache: dict[str, Transformer] = {}
    grouped_frames: list[pd.DataFrame] = []

    for year_dir in year_dirs:
        for prod_path in sorted(year_dir.glob("*_h2_production_*.tif")):
            parsed = parse_h2_name(prod_path.name)
            if parsed is None:
                continue
            resource, tech = parsed
            if NH3_ELECTROLYSIS_TECHS and tech not in NH3_ELECTROLYSIS_TECHS:
                continue
            lcoh_path = year_dir / f"{resource}_lcoh_{tech}.tif"
            if not lcoh_path.exists():
                continue
            lcoe_path = year_dir / f"{resource}_lcoe.tif"
            with rasterio.open(prod_path) as prod_src:
                prod = prod_src.read(1).astype(np.float64)
                prod_transform = prod_src.transform
                prod_crs = prod_src.crs
                cache_key = (
                    f"{prod_src.crs.to_wkt() if prod_src.crs else 'none'}"
                    f"|{prod_src.height}|{prod_src.width}|{prod_src.transform}"
                )
            if cache_key not in zone_cache:
                zone_cache[cache_key], _ = rasterize_provinces_to_match(provinces, prod_path)
            zone_grid = zone_cache[cache_key]
            prod = np.nan_to_num(prod, nan=0.0, posinf=0.0, neginf=0.0)
            valid = (prod > 0.0) & (zone_grid > 0)
            if not np.any(valid):
                continue
            rows, cols = np.where(valid)
            # Raster unit is kg H₂/yr/km² (Albers 1km projection);
            # pixel_area_m2 = 1e6 m² = 1 km².  No area multiplication needed
            # since each pixel already represents 1 km².
            h2_supply = prod[rows, cols]  # kg/yr per pixel (= per km²)
            if h2_supply.size == 0:
                continue
            with rasterio.open(lcoh_path) as lcoh_src:
                lcoh = np.nan_to_num(lcoh_src.read(1), nan=0.0, posinf=0.0, neginf=0.0)
                lcoh_values = lcoh[rows, cols].astype(np.float64)
            # Electricity for the Haber-Bosch loop and the ASU is bought at the same site's
            # LCOE (USD/MWh) rather than assumed free, as the previous cost lower bound did.
            if lcoe_path.exists():
                with rasterio.open(lcoe_path) as lcoe_src:
                    lcoe = np.nan_to_num(lcoe_src.read(1), nan=0.0, posinf=0.0, neginf=0.0)
                    lcoe_values = lcoe[rows, cols].astype(np.float64)
            else:
                logger.warning("Missing LCOE raster %s; HB power cost set to zero", lcoe_path.name)
                lcoe_values = np.zeros_like(lcoh_values)
            lons, lats = _pixel_centers_lonlat(
                prod_transform,
                prod_crs,
                rows,
                cols,
                transformer_cache=transformer_cache,
            )
            province_ids = zone_grid[rows, cols].astype(np.int32)
            lon_bin = _bin_coordinates(lons, AMMONIA_NODE_AGGREGATION_DEGREES, 180.0)
            lat_bin = _bin_coordinates(lats, AMMONIA_NODE_AGGREGATION_DEGREES, 90.0)
            frame = pd.DataFrame(
                {
                    "province_id": province_ids,
                    "lon_bin": lon_bin,
                    "lat_bin": lat_bin,
                    "h2_supply_kg_per_year": h2_supply.astype(np.float64),
                    "weighted_lcoh_component": (lcoh_values * h2_supply).astype(np.float64),
                    "weighted_lcoe_component": (lcoe_values * h2_supply).astype(np.float64),
                    "weighted_lon": (lons * h2_supply).astype(np.float64),
                    "weighted_lat": (lats * h2_supply).astype(np.float64),
                }
            )
            grouped = (
                frame.groupby(["province_id", "lon_bin", "lat_bin"], as_index=False)[
                    ["h2_supply_kg_per_year", "weighted_lcoh_component", "weighted_lcoe_component",
                     "weighted_lon", "weighted_lat"]
                ]
                .sum()
                .reset_index(drop=True)
            )
            grouped["longitude"] = np.where(
                grouped["h2_supply_kg_per_year"] > 0,
                grouped["weighted_lon"] / grouped["h2_supply_kg_per_year"],
                0.0,
            )
            grouped["latitude"] = np.where(
                grouped["h2_supply_kg_per_year"] > 0,
                grouped["weighted_lat"] / grouped["h2_supply_kg_per_year"],
                0.0,
            )
            grouped["weighted_lcoh_usd_per_kg_h2"] = np.where(
                grouped["h2_supply_kg_per_year"] > 0,
                grouped["weighted_lcoh_component"] / grouped["h2_supply_kg_per_year"],
                np.nan,
            )
            grouped["province_name"] = grouped["province_id"].map(lookup)
            grouped["year"] = int(year_dir.name)
            grouped["resource"] = resource
            grouped["electrolysis_tech"] = tech
            grouped["weighted_lcoe_usd_per_mwh"] = np.where(
                grouped["h2_supply_kg_per_year"] > 0,
                grouped["weighted_lcoe_component"] / grouped["h2_supply_kg_per_year"],
                np.nan,
            )
            grouped["nh3_supply_kg_per_year"] = grouped["h2_supply_kg_per_year"] / NH3_H2_RATIO
            grouped["nh3_h2_cost_component_usd_per_kg"] = grouped["weighted_lcoh_usd_per_kg_h2"] * NH3_H2_RATIO
            grouped["hb_power_need_kwh_per_kg_nh3"] = NH3_HB_POWER_KWH_PER_KG
            # Haber-Bosch electricity, priced at the co-located renewable LCOE.
            grouped["nh3_hb_power_cost_usd_per_kg"] = (
                NH3_HB_POWER_KWH_PER_KG * grouped["weighted_lcoe_usd_per_mwh"].fillna(0.0) / 1000.0
            )
            grouped["nh3_hb_capex_usd_per_kg"] = _hb_capex_annuity_usd_per_kg()
            grouped["nh3_storage_adder_usd_per_kg"] = NH3_STORAGE_ADDER_USD_PER_KG
            grouped["nh3_transport_adder_usd_per_kg"] = NH3_TRANSPORT_ADDER_USD_PER_KG
            # Full landed cost: H2 feedstock + HB/ASU electricity + HB/ASU capital + storage.
            # (Column name kept for downstream compatibility; it is no longer a lower bound.)
            grouped["nh3_cost_lb_usd_per_kg"] = (
                grouped["nh3_h2_cost_component_usd_per_kg"]
                + grouped["nh3_hb_power_cost_usd_per_kg"]
                + grouped["nh3_hb_capex_usd_per_kg"]
                + NH3_STORAGE_ADDER_USD_PER_KG
                + NH3_TRANSPORT_ADDER_USD_PER_KG
            )
            grouped["source"] = paths.rel(prod_path)
            grouped["year_basis"] = f"cost_year_{year_dir.name}"
            grouped["buffer_km"] = BIOMASS_MATCH_BUFFER_KM
            grouped_frames.append(grouped)

    if not grouped_frames:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    ammonia_nodes = pd.concat(grouped_frames, ignore_index=True, sort=False)
    ammonia_nodes = ammonia_nodes.sort_values(
        ["year", "province_id", "resource", "electrolysis_tech", "lon_bin", "lat_bin"]
    ).reset_index(drop=True)
    ammonia_nodes["ammonia_node_id"] = [f"A{index:06d}" for index in range(1, len(ammonia_nodes) + 1)]
    ammonia_nodes["match_rule"] = "point_source_competition"
    ammonia_nodes["competition_scope"] = "shared_ammonia_node"
    ammonia_nodes["aggregation_cell_deg"] = AMMONIA_NODE_AGGREGATION_DEGREES
    ammonia_nodes["buffer_km"] = BIOMASS_MATCH_BUFFER_KM
    return ammonia_nodes[
        [
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
        ]
    ].reset_index(drop=True)


def build_ammonia_link_dataframe(
    paths: ProjectPaths,
    ammonia_nodes: pd.DataFrame,
) -> pd.DataFrame:
    plants = pd.read_csv(paths.inputs_dir / "plants.csv").copy()
    plant_count = len(plants)
    rows: list[dict[str, object]] = []
    for year, year_nodes in ammonia_nodes.groupby("year", sort=True):
        if year_nodes.empty:
            continue
        year_nodes = year_nodes.reset_index(drop=True)
        node_lons = year_nodes["longitude"].astype(float).to_numpy()
        node_lats = year_nodes["latitude"].astype(float).to_numpy()
        buffer_km = float(year_nodes["buffer_km"].iloc[0]) if "buffer_km" in year_nodes.columns else BIOMASS_MATCH_BUFFER_KM
        for plant in plants.itertuples(index=False):
            distances_km = _haversine_distances_km(
                float(plant.centroid_longitude),
                float(plant.centroid_latitude),
                node_lons,
                node_lats,
            )
            matched_indices = np.flatnonzero(distances_km <= buffer_km)
            if matched_indices.size == 0:
                continue
            ranked_indices = matched_indices[np.argsort(distances_km[matched_indices])]
            for rank, node_idx in enumerate(ranked_indices, start=1):
                node = year_nodes.iloc[int(node_idx)]
                rows.append(
                    {
                        "plant_count": plant_count,
                        "year": int(year),
                        "plant_id": str(plant.plant_id),
                        "plant_index": int(plant.plant_index),
                        "plant_province_name": str(plant.province_mode),
                        "ammonia_node_id": str(node["ammonia_node_id"]),
                        "ammonia_node_province_name": str(node["province_name"]),
                        "distance_km": round(float(distances_km[node_idx]), 3),
                        "distance_rank": rank,
                        "base_cost_usd_per_kg": float(node["nh3_cost_lb_usd_per_kg"]),
                        "source": str(node["source"]),
                        "year_basis": str(node["year_basis"]),
                        "match_rule": "plant_buffer_intersects_ammonia_node",
                        "competition_scope": "shared_ammonia_node",
                        "buffer_km": buffer_km,
                    }
                )
    columns = [
        "plant_count",
        "year",
        "plant_id",
        "plant_index",
        "plant_province_name",
        "ammonia_node_id",
        "ammonia_node_province_name",
        "distance_km",
        "distance_rank",
        "base_cost_usd_per_kg",
        "source",
        "year_basis",
        "match_rule",
        "competition_scope",
        "buffer_km",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["year", "plant_id", "distance_km", "ammonia_node_id"]
    ).reset_index(drop=True)


def write_supply_inputs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    biomass_nodes = build_biomass_supply_dataframe(paths)
    ammonia_nodes = build_ammonia_supply_dataframe(paths)
    write_csv(biomass_nodes, paths.inputs_dir / "biomass_supply_curve.csv")
    biomass_links = build_biomass_link_dataframe(paths, biomass_nodes)
    ammonia_links = build_ammonia_link_dataframe(paths, ammonia_nodes)
    write_csv(biomass_links, paths.inputs_dir / "biomass_supply_links.csv")
    write_csv(ammonia_links, paths.inputs_dir / "ammonia_supply_links.csv")
    write_csv(ammonia_nodes, paths.inputs_dir / "ammonia_supply_curve.csv")
    return biomass_nodes, ammonia_nodes
