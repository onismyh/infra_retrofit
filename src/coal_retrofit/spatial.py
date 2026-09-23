from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

try:
    from pyproj import Geod
except ImportError:  # pragma: no cover
    Geod = None


def load_provinces(provinces_path: Path) -> gpd.GeoDataFrame:
    provinces = gpd.read_file(provinces_path)[["OBJECTID", "NAME", "geometry"]].copy()
    provinces = provinces.rename(columns={"OBJECTID": "province_id", "NAME": "province_name"})
    provinces["province_id"] = provinces["province_id"].astype(int)
    return provinces.sort_values("province_id").reset_index(drop=True)


def rasterize_provinces_to_match(
    provinces: gpd.GeoDataFrame,
    raster_path: Path,
) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    with rasterio.open(raster_path) as src:
        provinces_in_crs = provinces.to_crs(src.crs)
        shapes = (
            (geom, province_id)
            for geom, province_id in zip(provinces_in_crs.geometry, provinces_in_crs["province_id"], strict=True)
        )
        zone_grid = rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="int32",
        )
    return zone_grid, provinces_in_crs


def latlon_row_areas(transform, height: int) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    x_left = transform.c
    x_right = transform.c + transform.a
    row_areas = np.zeros(height, dtype=np.float64)
    for row in range(height):
        y_top = transform.f + row * transform.e
        y_bottom = transform.f + (row + 1) * transform.e
        lons = [x_left, x_right, x_right, x_left]
        lats = [y_top, y_top, y_bottom, y_bottom]
        area, _ = geod.polygon_area_perimeter(lons, lats)
        row_areas[row] = abs(area)
    return row_areas


def geometry_parts(geom) -> list:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


def geodesic_length_km(coords: Iterable[tuple[float, float]]) -> float:
    points = list(coords)
    if len(points) < 2:
        return 0.0
    if Geod is not None:
        geod = Geod(ellps="WGS84")
        lons = [pt[0] for pt in points]
        lats = [pt[1] for pt in points]
        return float(geod.line_length(lons, lats) / 1000.0)

    total_km = 0.0
    for start, end in zip(points[:-1], points[1:]):
        lon1, lat1 = start
        lon2, lat2 = end
        dlon = radians(lon2 - lon1)
        dlat = radians(lat2 - lat1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        total_km += 6371.0088 * (2 * asin(sqrt(a)))
    return total_km
