"""空间工具：省界与流域的读取/栅格化、球面与大地线距离。

距离与面积一律在地理坐标下算（球面 haversine 或 WGS84 大地线），不在投影坐标下算（CLAUDE.md §4.1）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from numpy.typing import ArrayLike
from pyproj import Geod
from rasterio.features import rasterize

EARTH_RADIUS_KM = 6371.0088   # IUGG 平均半径，全仓库 haversine 共用


def haversine_km(lon_a: ArrayLike, lat_a: ArrayLike, lon_b: ArrayLike, lat_b: ArrayLike) -> np.ndarray:
    """球面大圆距离（km）。四个参数按 numpy 广播规则配对，单位为度。"""
    lon_a, lat_a, lon_b, lat_b = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lon_a, lat_a, lon_b, lat_b))
    h = np.sin((lat_b - lat_a) / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def haversine_matrix_km(lons: ArrayLike, lats: ArrayLike) -> np.ndarray:
    """各点两两之间的球面距离矩阵（km），形状 (n, n)。"""
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    return haversine_km(lons[:, None], lats[:, None], lons[None, :], lats[None, :])


def within_radius(
    lon: float, lat: float, lons: ArrayLike, lats: ArrayLike, radius_km: float, *, ranked: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) 周围 `radius_km` 以内的点：返回 (下标, 对应的 haversine 距离 km)。

    缺省按下标升序；`ranked=True` 按距离升序（`np.argsort` 缺省算法，并列的先后不保证）。厂址到生物质 / 氨 / 水节点、
    工业 hub 到氢节点的候选链路都由它配对，各处的列、四舍五入位数与排序要求留在调用方。
    """
    distances = haversine_km(lon, lat, lons, lats)
    matched = np.flatnonzero(distances <= radius_km)
    if ranked:
        matched = matched[np.argsort(distances[matched])]
    return matched, distances[matched]


def _densified_vertices(geometries: Iterable, max_segment_deg: float) -> np.ndarray:
    """把线（或多边形边界）按 *max_segment_deg* 加密后的全部顶点，(n, 2) 的 (lon, lat)。"""
    dense = shapely.segmentize(np.asarray(list(geometries), dtype=object), max_segment_deg)
    return shapely.get_coordinates(dense)


def _require_geographic(layer: gpd.GeoSeries, what: str) -> None:
    """*layer* 须为非空的地理坐标图层，且不含 None 或空几何（`isna` 只认 None，空多边形要另查 `is_empty`）。"""
    crs = layer.crs
    if crs is None or not crs.is_geographic:
        raise ValueError(f"{what} 须为地理坐标（经纬度），不在投影坐标下量距离（CLAUDE.md §4.1）；得到 {crs}")
    if len(layer) == 0 or layer.isna().any() or layer.is_empty.any():
        raise ValueError(f"{what} 为空或含空几何，无法量距离")


def _finite_points(lons: ArrayLike, lats: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """点坐标转成等长的一维 float64 数组（标量算一个点）；含 NaN / inf、个数不一致或超出经纬度范围就报错，
    不让投影坐标或写反后越界的经纬度静默落到错误的归属上（写反后仍在范围内的，如 80°E 40°N 写成 (40, 80)，抓不到）。"""
    lons = np.atleast_1d(np.asarray(lons, dtype=np.float64))
    lats = np.atleast_1d(np.asarray(lats, dtype=np.float64))
    if lons.ndim != 1 or lats.ndim != 1:
        raise ValueError(f"点坐标须为标量或一维数组，得到形状 {lons.shape} 与 {lats.shape}")
    if lons.shape != lats.shape:
        raise ValueError(f"经度与纬度个数不一致：{lons.shape} 对 {lats.shape}")
    if not (np.isfinite(lons).all() and np.isfinite(lats).all()):
        raise ValueError("点坐标含 NaN 或 inf，无法量距离")
    if (np.abs(lons) > 180.0).any() or (np.abs(lats) > 90.0).any():
        raise ValueError("点坐标超出经纬度范围（|经度| <= 180、|纬度| <= 90）：是不是传了投影坐标，或经纬度写反了")
    return lons, lats


def distance_to_lines_km(
    lons: ArrayLike, lats: ArrayLike, lines: gpd.GeoSeries, max_segment_deg: float = 0.01
) -> np.ndarray:
    """每个点到一组线要素的最小球面距离（km）。

    线按 *max_segment_deg*（缺省 0.01°，约 1 km）加密顶点后取到顶点的最小 haversine 距离，
    比到线段的真实距离最多偏大半个加密步长。*lines* 须为非空的地理坐标图层；点坐标含 NaN / inf 或超出经纬度范围报错。
    """
    _require_geographic(lines, "lines")
    vertices = _densified_vertices(lines.geometry, max_segment_deg)
    lons, lats = _finite_points(lons, lats)
    return np.array([haversine_km(lon, lat, vertices[:, 0], vertices[:, 1]).min() for lon, lat in zip(lons, lats)])


def nearest_polygon_index(
    lons: ArrayLike, lats: ArrayLike, polygons: gpd.GeoSeries, max_segment_deg: float = 0.01
) -> np.ndarray:
    """每个点落在哪个多边形里，返回 *polygons* 的位置序号。

    先按包含关系配对，同时落在多个多边形里的点取序号最小的；落在所有多边形之外的点（海岸、多边形之间的
    缝隙）取边界球面距离最近的多边形，边界按 *max_segment_deg* 加密后量到顶点。*polygons* 须为非空的地理坐标
    图层；点坐标含 NaN / inf 或超出经纬度范围报错。
    """
    _require_geographic(polygons, "polygons")
    lons, lats = _finite_points(lons, lats)
    frame = gpd.GeoDataFrame({"_polygon": np.arange(len(polygons))}, geometry=polygons.geometry.to_numpy(), crs=polygons.crs)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs=polygons.crs)
    joined = gpd.sjoin(points, frame, how="left", predicate="within")
    # 落在多个多边形里的点取序号最小的；没落进任何多边形的为 NaN。
    index = np.array(joined["_polygon"].groupby(level=0).min().reindex(points.index), dtype=np.float64)
    outside = np.flatnonzero(np.isnan(index))
    if outside.size:
        boundaries = [_densified_vertices([geom.boundary], max_segment_deg) for geom in polygons.geometry]
        for i in outside:
            index[i] = np.argmin([haversine_km(lons[i], lats[i], b[:, 0], b[:, 1]).min() for b in boundaries])
    return index.astype(int)


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
    geod = Geod(ellps="WGS84")
    lons = [pt[0] for pt in points]
    lats = [pt[1] for pt in points]
    return float(geod.line_length(lons, lats) / 1000.0)
