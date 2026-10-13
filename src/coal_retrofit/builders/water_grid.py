"""水输入构建共用的网格与流域工具：ISIMIP 0.5° 网格的面积、时间窗、枯水期窗口，省界与一级流域的读取、栅格化与点归属。

`water_availability`（节点与可用量）、`water`（粗化与链路）、`water_bias`（偏差校正）、`water_use`（生活与灌溉耗水）、
`water_quota`（流域取水指标）与
`scripts/validate_basin_runoff.py` 都从这里取，保证各表在同一网格、同一枯水期窗口与同一流域归属上算。
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd

from ..constants import TARGET_GEO_CRS
from ..paths import ProjectPaths
from ..spatial import EARTH_RADIUS_KM, nearest_polygon_index
from ..spatial import load_provinces as load_provinces_layer

WATER_WINDOW_BASIS = {
    2030: (2021, 2030),
    2040: (2031, 2040),
    2050: (2041, 2050),
    2060: (2051, 2060),
}
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def _resolve_source_path(paths: ProjectPaths, source: str) -> str:
    relative = Path(str(source))
    if relative.exists():
        return str(relative)
    cwd_candidate = Path.cwd() / relative
    if cwd_candidate.exists():
        return str(cwd_candidate)
    root_candidate = paths.root / relative
    if root_candidate.exists():
        return str(root_candidate)
    return str(relative)


def _nc_path(path: Path) -> str:
    """netCDF4 能打开的路径：其 C 层在 Windows 上拒绝含非 ASCII 字符的绝对路径。"""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        import os

        return os.path.relpath(path, Path.cwd())


def _clean_series(series: np.ndarray, fill_value: float | None) -> np.ndarray:
    data = np.asarray(np.ma.filled(series, np.nan), dtype=np.float64)
    if fill_value is not None:
        data = np.where(data == fill_value, np.nan, data)
    return data


def _clean_window(ds: netCDF4.Dataset, variable: str, mask: np.ndarray) -> np.ndarray:
    """变量 `variable` 在时间掩膜 `mask` 内的逐月序列，填充值换成 NaN。"""
    var = ds.variables[variable]
    return _clean_series(np.asarray(var[mask]), getattr(var, "_FillValue", None))


def _file_years(ds: netCDF4.Dataset) -> np.ndarray:
    """逐个时间步的公历年份。"""
    time_var = ds.variables["time"]
    dates = netCDF4.num2date(time_var[:], time_var.units, getattr(time_var, "calendar", "standard"))
    return np.array([int(dt.year) for dt in np.asarray(dates)])


def _cell_area_m2(lat: np.ndarray, cell_degrees: float = 0.5) -> np.ndarray:
    """规则经纬网格上每个纬度带的网格单元面积（m²），球面半径取 `spatial.EARTH_RADIUS_KM`（2026-10-13 前用 6 371 km）。"""
    half = cell_degrees / 2.0
    band = (
        (EARTH_RADIUS_KM * 1000.0) ** 2
        * np.deg2rad(cell_degrees)
        * np.abs(np.sin(np.deg2rad(lat + half)) - np.sin(np.deg2rad(lat - half)))
    )
    return band


def _annual_m3(series: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    """逐月序列（kg m-2 s-1）的多年均值换成每个网格的年水量，m3/yr（水的密度 1000 kg m-3）。"""
    return np.nan_to_num(np.nanmean(series, axis=0)) * area_m2 * SECONDS_PER_YEAR / 1000.0


def _annual_and_seasonal(window: np.ndarray, area_m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """整年的逐月序列（kg m-2 s-1，从 1 月起）的逐网格年均水量与 12 个候选三个月窗口的年化水量，m3/yr。

    第 m 个窗口是从第 m 个日历月（0 = 1 月）起的连续三个月，取多年月均的年循环。径流与耗水共用，
    所以两者的枯水期是同一组月份。
    """
    annual = _annual_m3(window, area_m2)
    months = np.arange(window.shape[0]) % 12
    monthly_clim = np.stack(
        [np.nan_to_num(np.nanmean(window[months == m], axis=0)) for m in range(12)]
    )
    rolling = np.stack([monthly_clim[np.arange(m, m + 3) % 12].mean(axis=0) for m in range(12)])
    return annual, rolling * area_m2 * SECONDS_PER_YEAR / 1000.0


def _basin_dry_window(rolling_volume: np.ndarray, cells: np.ndarray) -> tuple[int, float]:
    """流域的枯水季：12 个候选三个月窗口中流域径流合计最小的那个，返回 (窗口序号, 年化水量 m3/yr)。

    在流域合计上选，不逐网格选，理由见 `water_availability.build_water_availability_dataframe` 里选取处的注释块。
    """
    basin_rolling = rolling_volume[:, cells].sum(axis=1)
    dry = int(np.argmin(basin_rolling))
    return dry, float(basin_rolling[dry])


def load_provinces(paths: ProjectPaths) -> gpd.GeoDataFrame:
    provinces = load_provinces_layer(paths.find_data_file("provinces.shp"))
    if provinces.crs is None:
        raise ValueError("Province layer has no CRS defined")
    if provinces.crs.to_string() != TARGET_GEO_CRS:
        provinces = provinces.to_crs(TARGET_GEO_CRS)
    return provinces.sort_values("province_id").reset_index(drop=True)


def load_basins(paths: ProjectPaths) -> gpd.GeoDataFrame:
    """水资源一级区，即中国公布官方水资源总量所用的单元。"""
    path = paths.data_dir / "ChinaBasins" / "basin_l1.gpkg"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在。该文件入库在仓库根 data/ChinaBasins/，求解树经 <tree>/data 链接读它"
            "（README 的 `_indtree/data` 一行）；列为 `code`（A-K）、`name`、`name_en`，EPSG:4326。"
        )
    return gpd.read_file(path).to_crs(TARGET_GEO_CRS)


def assign_basin_codes(paths: ProjectPaths, nodes: pd.DataFrame) -> np.ndarray:
    """按所在（或最近的）多边形，给每个点（`longitude` / `latitude` 列）分配一级流域代码。

    先按包含关系配对；落在所有多边形之外的点（海岸、多边形之间的缝隙）取边界最近的流域，否则这些网格会被
    所有流域预算漏掉，其水量被悄悄删除。距离按球面算（`spatial.nearest_polygon_index`），2026-10-11 前在 EPSG:2380 下算。
    """
    basins = load_basins(paths)
    index = nearest_polygon_index(nodes["longitude"].to_numpy(float), nodes["latitude"].to_numpy(float), basins.geometry)
    return basins["code"].astype(str).to_numpy()[index]


def _province_zone_grid(paths: ProjectPaths, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """把省界多边形栅格化到气候网格上；0 = 中国境外。"""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    provinces = load_provinces(paths).to_crs("EPSG:4326")
    cell = float(abs(lat[1] - lat[0]))
    transform = from_origin(float(lon.min() - cell / 2.0), float(lat.max() + cell / 2.0), cell, cell)
    zones = rasterize(
        ((geom, idx + 1) for idx, geom in enumerate(provinces.geometry)),
        out_shape=(len(lat), len(lon)),
        transform=transform,
        fill=0,
        dtype="int32",
    )
    name_col = "province_name" if "province_name" in provinces.columns else provinces.columns[0]
    return zones, [str(v) for v in provinces[name_col]]


def _basin_zone_grid(paths: ProjectPaths, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """把一级流域多边形栅格化到气候网格上；0 = 不在任何流域内。"""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    basins = load_basins(paths)
    cell = float(abs(lat[1] - lat[0]))
    transform = from_origin(float(lon.min() - cell / 2.0), float(lat.max() + cell / 2.0), cell, cell)
    zones = rasterize(
        ((geom, idx + 1) for idx, geom in enumerate(basins.geometry)),
        out_shape=(len(lat), len(lon)),
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return zones, [str(v) for v in basins["code"]]
