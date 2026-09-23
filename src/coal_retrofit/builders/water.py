from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd

from ..artifacts import write_csv
from ..constants import (
    BIAS_BASELINE_WINDOW,
    OFFICIAL_BASIN_WATER_1E8_M3,
    PLANNING_YEARS,
    TARGET_GEO_CRS,
    WATER_COARSE_GRID_DEGREES,
    WATER_MATCH_BUFFER_KM,
)
from ..paths import ProjectPaths
from ..spatial import load_provinces as load_provinces_layer


WATER_PATTERN = re.compile(
    r"(?P<hydrology>[^_]+)_(?P<gcm>.+?)_w5e5_(?P<ssp>ssp\d+)_"
    r"(?P<socioeconomics>.+?)_default_(?P<variable>[^_]+)_global_monthly_"
    r"(?P<start_year>\d{4})_(?P<end_year>\d{4})\.nc"
)
WATER_WINDOW_BASIS = {
    2030: (2021, 2030),
    2040: (2031, 2040),
    2050: (2041, 2050),
    2060: (2051, 2060),
}
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def _scenario_family(ssp: str) -> str:
    if ssp == "ssp126":
        return "baseline"
    if ssp == "ssp370":
        return "high_pressure"
    return "other"


def parse_water_file(paths: ProjectPaths, filename: str) -> dict[str, object]:
    match = WATER_PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f"Could not parse water scenario filename: {filename}")
    scenario_id = "|".join([match.group("hydrology"), match.group("gcm"), match.group("ssp")])
    path = paths.data_dir / "water" / filename
    return {
        "scenario_id": scenario_id,
        "hydrology_model": match.group("hydrology"),
        "gcm": match.group("gcm"),
        "ssp": match.group("ssp"),
        "scenario_family": _scenario_family(match.group("ssp")),
        "socioeconomics": match.group("socioeconomics"),
        "variable": match.group("variable"),
        "frequency": "monthly",
        "start_year": int(match.group("start_year")),
        "end_year": int(match.group("end_year")),
        "is_base_candidate": match.group("ssp") == "ssp126",
        "source": paths.rel(path),
        "year_basis": f"{match.group('start_year')}-{match.group('end_year')}",
    }


def build_water_scenarios_dataframe(paths: ProjectPaths) -> pd.DataFrame:
    """只收情景成员。`historical` 运行与它们同在一个目录，但不是成员：
    它们是 `basin_bias_factors` 估计各模型偏差时所对照的基准。"""
    files = sorted(
        path.name for path in (paths.data_dir / "water").glob("*.nc")
        if WATER_PATTERN.fullmatch(path.name)
    )
    if not files:
        raise FileNotFoundError(f"No scenario .nc files matching WATER_PATTERN in {paths.data_dir / 'water'}")
    scenarios = pd.DataFrame(parse_water_file(paths, filename) for filename in files)
    return scenarios.sort_values(["hydrology_model", "gcm", "ssp"]).reset_index(drop=True)


def build_water_base_dataframe(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in scenarios[scenarios["is_base_candidate"]].iterrows():
        for planning_year in PLANNING_YEARS:
            rows.append(
                {
                    "planning_year": planning_year,
                    "scenario_id": row["scenario_id"],
                    "hydrology_model": row["hydrology_model"],
                    "gcm": row["gcm"],
                    "ssp": row["ssp"],
                    "variable": row["variable"],
                    "selection_rule": "ssp126_ensemble_baseline",
                    "selection_status": "grid_node_extract_ready",
                    "source": row["source"],
                    "year_basis": row["year_basis"],
                }
            )
    return pd.DataFrame(rows).sort_values(["planning_year", "hydrology_model", "gcm"]).reset_index(drop=True)


def load_provinces(paths: ProjectPaths) -> gpd.GeoDataFrame:
    provinces = load_provinces_layer(paths.find_data_file("provinces.shp"))
    if provinces.crs is None:
        raise ValueError("Province layer has no CRS defined")
    if provinces.crs.to_string() != TARGET_GEO_CRS:
        provinces = provinces.to_crs(TARGET_GEO_CRS)
    return provinces.sort_values("province_id").reset_index(drop=True)


def _clean_series(series: np.ndarray, fill_value: float | None) -> np.ndarray:
    data = np.asarray(np.ma.filled(series, np.nan), dtype=np.float64)
    if fill_value is not None:
        data = np.where(data == fill_value, np.nan, data)
    return data


def _representative_scenario_rows(scenarios: pd.DataFrame) -> dict[str, pd.Series]:
    baseline_candidates = scenarios[scenarios["ssp"] == "ssp126"].sort_values(["hydrology_model", "gcm"]).reset_index(drop=True)
    if baseline_candidates.empty:
        raise ValueError("No ssp126 water scenarios available for baseline proxy")
    baseline_row = baseline_candidates.iloc[0]
    high_candidates = scenarios[
        (scenarios["ssp"] == "ssp370")
        & (scenarios["hydrology_model"] == baseline_row["hydrology_model"])
        & (scenarios["gcm"] == baseline_row["gcm"])
    ].sort_values(["hydrology_model", "gcm"]).reset_index(drop=True)
    if high_candidates.empty:
        high_candidates = scenarios[scenarios["ssp"] == "ssp370"].sort_values(["hydrology_model", "gcm"]).reset_index(drop=True)
    if high_candidates.empty:
        raise ValueError("No ssp370 water scenarios available for high-pressure proxy")
    return {
        "baseline": baseline_row,
        "high_pressure": high_candidates.iloc[0],
    }


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


def _candidate_grid_points(
    provinces: gpd.GeoDataFrame,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = provinces.total_bounds
    lon_mask = (lon_values >= (min_lon - 0.5)) & (lon_values <= (max_lon + 0.5))
    lat_mask = (lat_values >= (min_lat - 0.5)) & (lat_values <= (max_lat + 0.5))
    lon_indices = np.flatnonzero(lon_mask)
    lat_indices = np.flatnonzero(lat_mask)
    lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
    frame = pd.DataFrame(
        {
            "lon_index": lon_grid.ravel().astype(np.int32),
            "lat_index": lat_grid.ravel().astype(np.int32),
        }
    )
    frame["longitude"] = lon_values[frame["lon_index"].to_numpy()].astype(np.float64)
    frame["latitude"] = lat_values[frame["lat_index"].to_numpy()].astype(np.float64)
    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs=TARGET_GEO_CRS,
    )
    joined = gpd.sjoin(
        points,
        provinces[["province_id", "province_name", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    joined = (
        joined.sort_values(["lat_index", "lon_index", "province_id"])
        .drop_duplicates(subset=["lat_index", "lon_index"])
        .reset_index(drop=True)
    )
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def build_water_nodes_dataframe(paths: ProjectPaths, scenarios: pd.DataFrame) -> pd.DataFrame:
    provinces = load_provinces(paths)
    reference_row = _representative_scenario_rows(scenarios)["baseline"]
    reference_path = _resolve_source_path(paths, str(reference_row["source"]))
    with netCDF4.Dataset(str(reference_path), "r") as ds:
        lon_values = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        lat_values = np.asarray(ds.variables["lat"][:], dtype=np.float64)

    nodes = _candidate_grid_points(provinces, lon_values, lat_values)
    nodes["basin_code"] = _assign_basin_codes(paths, nodes)
    lon_step = float(np.median(np.abs(np.diff(lon_values)))) if len(lon_values) > 1 else 0.5
    lat_step = float(np.median(np.abs(np.diff(lat_values)))) if len(lat_values) > 1 else 0.5
    nodes["water_node_id"] = [f"W{index:05d}" for index in range(1, len(nodes) + 1)]
    nodes["grid_cell_width_deg"] = lon_step
    nodes["grid_cell_height_deg"] = lat_step
    nodes["competition_scope"] = "shared_water_node"
    nodes["match_rule"] = "china_hydrology_grid_cell_center"
    nodes["source"] = str(reference_row["source"])
    nodes["year_basis"] = "static_hydrology_grid"
    return nodes[
        [
            "water_node_id",
            "province_id",
            "province_name",
            "basin_code",
            "longitude",
            "latitude",
            "lon_index",
            "lat_index",
            "grid_cell_width_deg",
            "grid_cell_height_deg",
            "competition_scope",
            "match_rule",
            "source",
            "year_basis",
        ]
    ].sort_values("water_node_id").reset_index(drop=True)


EARTH_RADIUS_M = 6_371_000.0


def _cell_area_m2(lat: np.ndarray, cell_degrees: float = 0.5) -> np.ndarray:
    """规则经纬网格上每个纬度带的网格单元面积（m²）。"""
    half = cell_degrees / 2.0
    band = (
        EARTH_RADIUS_M ** 2
        * np.deg2rad(cell_degrees)
        * np.abs(np.sin(np.deg2rad(lat + half)) - np.sin(np.deg2rad(lat - half)))
    )
    return band


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


def _assign_basin_codes(paths: ProjectPaths, nodes: pd.DataFrame) -> np.ndarray:
    """按最近的多边形，给每个网格节点分配一级流域代码。

    用最近邻而不是严格包含：否则海岸上以及多边形之间缝隙里的网格会被所有流域预算
    漏掉，其水量被悄悄删除。距离在 EPSG:2380 下计算，因此以米为单位。
    """
    basins = load_basins(paths).to_crs("EPSG:2380")
    points = gpd.GeoDataFrame(
        nodes[["longitude", "latitude"]],
        geometry=gpd.points_from_xy(nodes["longitude"], nodes["latitude"]),
        crs=TARGET_GEO_CRS,
    ).to_crs("EPSG:2380")
    joined = gpd.sjoin_nearest(points, basins[["code", "geometry"]], how="left")
    joined = joined[~joined.index.duplicated()]
    return joined["code"].astype(str).to_numpy()


def load_basins(paths: ProjectPaths) -> gpd.GeoDataFrame:
    """水资源一级区，即中国公布官方水资源总量所用的单元。"""
    path = paths.data_dir / "ChinaBasins" / "basin_l1.gpkg"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it from data/basins_raw with scripts/validate_basin_runoff.py's "
            "layer conventions: columns `code` (A-K) and `name`, EPSG:4326."
        )
    return gpd.read_file(path).to_crs(TARGET_GEO_CRS)


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


def _historical_source(paths: ProjectPaths, hydrology_model: str, gcm: str) -> Path | None:
    """某个 (水文模型, GCM) 组合的 `historical` qtot 运行文件（若已下载）。

    两个键缺一不可。ISIMIP3b 用每个 GCM 分别驱动每个水文模型，所以 `historical` 运行是
    按组合存在的，而不是按水文模型。只按水文模型匹配、取排序后的第一个命中，曾对每个
    GCM 都悄悄返回 gfdl-esm4 的运行（它排在最前），这本会把一个 GCM 的偏差因子套用到
    所有 GCM 上，恰好抹平集合本来要度量的 GCM 间离散。
    """
    matches = sorted(
        p for p in (paths.data_dir / "water").glob("*historical*qtot*.nc")
        if p.name.startswith(f"{hydrology_model}_{gcm}_")
    )
    return matches[0] if matches else None


def _basin_runoff_from_file(
    path: Path, window: tuple[int, int], zones: np.ndarray, codes: list[str], area_m2: np.ndarray
) -> dict[str, float]:
    """在已栅格化的网格上，逐流域求 `window` 内的总径流，m3/yr。"""
    # netCDF4 的 C 层在 Windows 上打不开含非 ASCII 字符的绝对路径。
    with netCDF4.Dataset(_nc_path(path), "r") as ds:
        time_var = ds.variables["time"]
        years = np.array(
            [
                int(dt.year)
                for dt in netCDF4.num2date(
                    time_var[:], time_var.units, getattr(time_var, "calendar", "standard")
                )
            ]
        )
        mask = (years >= window[0]) & (years <= window[1])
        if not mask.any():
            raise ValueError(f"{path.name} covers {years.min()}-{years.max()}, not {window}")
        runoff_var = ds.variables["qtot"]
        series = _clean_series(np.asarray(runoff_var[mask]), getattr(runoff_var, "_FillValue", None))
        annual = np.nan_to_num(np.nanmean(series, axis=0)) * area_m2 * SECONDS_PER_YEAR / 1000.0
    return {code: float(annual[zones == idx].sum()) for idx, code in enumerate(codes, start=1)}


def basin_bias_factors(
    paths: ProjectPaths,
    hydrology_model: str,
    gcm: str,
    zones: np.ndarray,
    codes: list[str],
    area_m2: np.ndarray,
) -> dict[str, float]:
    """乘性校正因子：把某个模型的流域径流拉到官方基准上。

    0.5 deg 的全球水文模型即使全国总量正确，也带有很大的区域偏差：在 1956-2014 年，
    WaterGAP2-2e 复现中国总量的误差在 1.2% 以内，却在海河流域偏湿 2.43x、在淮河偏湿
    1.69x——这两个流域承载着 358 GW 煤电，也正是可用水量约束真正起作用的地方。若不
    校正，这一偏差就会被当作水文模型不确定性报告出来。

    该因子对每个 (水文模型, GCM) 组合只估计一次，取自该组合自己的 `historical` 运行，
    并施加到该组合的每一年、每个 SSP 上，因此每个成员的变化率被精确保留，只替换绝对
    水平。按组合而不是按水文模型估计，对集合很重要：历史偏差既是水文模型的属性，同样
    也是驱动 GCM 的属性，把一个 GCM 的因子套用到其他 GCM 上会抹掉真实存在的 GCM 间
    离散。若该组合的历史运行尚未下载，则对每个流域都返回 1.0。
    """
    source = _historical_source(paths, hydrology_model, gcm)
    if source is None:
        return {code: 1.0 for code in codes}
    modelled = _basin_runoff_from_file(source, BIAS_BASELINE_WINDOW, zones, codes, area_m2)
    factors: dict[str, float] = {}
    for code in codes:
        official_m3 = OFFICIAL_BASIN_WATER_1E8_M3.get(code, 0.0) * 1e8
        model_m3 = modelled.get(code, 0.0)
        factors[code] = official_m3 / model_m3 if model_m3 > 0 and official_m3 > 0 else 1.0
    return factors


def build_water_availability_dataframe(
    paths: ProjectPaths,
    scenarios: pd.DataFrame,
    water_nodes: pd.DataFrame,
) -> pd.DataFrame:
    """到达每个节点的可再生水量，取自本地径流（ISIMIP `qtot`）。

    旧实现读取每个节点所在网格的 `dis`（经汇流演算的河道流量）。流量是累积量——下游
    网格承载其整个上游集水区的来水——所以对 293 个节点求和会把同一份水重复计算几十次：
    全国总量达到 430 766 x10^8 m3/yr，是中国实际可再生水资源量的 15x，由此得到的约束
    永远不会起作用（需求/供给 = 0.083%）。

    `qtot` 是本地产生的径流（kg m-2 s-1），因此对网格求和就是真实的水量预算。按省界
    掩膜后，它复现出 26 291 x10^8 m3/yr，对照官方多年平均 ~28 000，相差 6%。

    可用水量按水资源一级区做预算，再按各节点本地径流的比例分配到该流域的节点上，因此
    对所有节点求和，每个流域的可再生水量恰好只计一次：

        basin_runoff   = sum(qtot x cell_area x seconds_per_year) x bias_factor   [m3/yr]
        node_available = basin_runoff x node_runoff / sum(node_runoff in basin)

    这里以流域而不是省为单元，理由有二：水是流域量（黄河流经九个省），而流域也是全球
    水文模型率定所用的尺度、中国公布官方总量所用的尺度。按省做预算还会使与官方统计之比
    落在 0.34x 到 4.00x 之间，其中大部分是用省界切割 0.5 deg 网格造成的假象；按流域时
    离散范围为 0.74x-2.43x，而 `basin_bias_factors` 连这一点也消除了。

    环境流量与存量取水不在这里施加——它们是求解时才施加的政策假设（见
    `WATER_EXTRACTABLE_FRACTION` 与 `OptimizationAssumptions.existing_withdrawal_share`），
    因此不必重建输入就能跑它们的敏感性分析。

    同时生成年均列和枯水期列（最低的连续三个月，年化）：火电受限发生在低流量时段，
    而不是在年均水平上，且源数据是逐月的。

    枯水季是在流域汇总量上选取的——对流域径流总和的 12 个候选 3 个月窗口取最小值——
    而不是逐网格选取。逐网格选取再求和得到的是 sum(min) 而不是 min(sum)，会把流域
    枯水期流量在海河低估 2.25x、在西北诸河低估 2.13x。见选取处的注释块。

    这一列不代表什么。`qtot` 是未经汇流演算的产流量，所以枯水期数值是流域最枯一季的
    产水速率——而不是河道中可用水的速率。水库调节体现在汇流方案里，而本流程刻意不用
    汇流（见开头一段）。因此把它当作保证供水量（firm yield）预算使用，就等于假设零调蓄；
    这在海河、黄河和淮河是很严格的假设，因为那里的枯水期流量很大程度上是水库放水决策
    的结果。在此写明，是因为由此得到的约束是本研究最紧的约束，其严格程度既来自水文，
    同样也来自这一选择。
    """
    rows: list[dict[str, object]] = []
    lat_indices = water_nodes["lat_index"].astype(int).to_numpy()
    lon_indices = water_nodes["lon_index"].astype(int).to_numpy()
    node_ids = water_nodes["water_node_id"].astype(str).to_numpy()
    node_basins = water_nodes["basin_code"].astype(str).to_numpy()
    bias_cache: dict[tuple[str, str], dict[str, float]] = {}

    for _, scenario_row in scenarios.iterrows():
        scenario_path = _resolve_source_path(paths, str(scenario_row["source"]))
        with netCDF4.Dataset(str(scenario_path), "r") as ds:
            time_var = ds.variables["time"]
            years = np.array(
                [
                    int(dt.year)
                    for dt in netCDF4.num2date(
                        time_var[:], time_var.units, getattr(time_var, "calendar", "standard")
                    )
                ]
            )
            if "qtot" not in ds.variables:
                raise ValueError(
                    f"{scenario_path} has no 'qtot' variable; routed discharge ('dis') is not a "
                    "valid water budget — see this function's docstring."
                )
            runoff_var = ds.variables["qtot"]
            fill_value = getattr(runoff_var, "_FillValue", None)
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            area_m2 = _cell_area_m2(lat)[:, None] * np.ones((1, len(lon)))
            basin_zones, basin_codes = _basin_zone_grid(paths, lat, lon)
            hydrology_model = str(scenario_row["hydrology_model"])
            gcm = str(scenario_row["gcm"])
            # 按组合缓存：每个 (水文模型, GCM) 一套因子，各 SSP 共用。
            bias_key = (hydrology_model, gcm)
            if bias_key not in bias_cache:
                bias_cache[bias_key] = basin_bias_factors(
                    paths, hydrology_model, gcm, basin_zones, basin_codes, area_m2
                )
            bias_factors = bias_cache[bias_key]

            for planning_year in PLANNING_YEARS:
                window_start, window_end = WATER_WINDOW_BASIS[planning_year]
                mask = (years >= window_start) & (years <= window_end)
                if not mask.any():
                    continue
                window = _clean_series(np.asarray(runoff_var[mask]), fill_value)
                # kg m-2 s-1 -> 每个网格的 m3/yr（水的密度 1000 kg m-3）
                annual = np.nan_to_num(np.nanmean(window, axis=0)) * area_m2 * SECONDS_PER_YEAR / 1000.0
                # 枯水期：气候态年循环中最低的连续 3 个日历月
                months = np.arange(window.shape[0]) % 12
                monthly_clim = np.stack(
                    [np.nan_to_num(np.nanmean(window[months == m], axis=0)) for m in range(12)]
                )
                rolling = np.stack([monthly_clim[np.arange(m, m + 3) % 12].mean(axis=0) for m in range(12)])
                # 12 个候选 3 个月窗口各自的每网格水量。流域的枯水季只选一次，
                # 且是在流域总量上选——见下面的循环。
                rolling_volume = rolling * area_m2 * SECONDS_PER_YEAR / 1000.0

                node_annual = annual[lat_indices, lon_indices]

                # 流域预算：先做偏差校正，再分配到该流域的各节点。
                node_annual_out = np.zeros(len(node_ids), dtype=np.float64)
                node_dry_out = np.zeros(len(node_ids), dtype=np.float64)
                for code in set(node_basins):
                    members = node_basins == code
                    zone_idx = basin_codes.index(code) + 1 if code in basin_codes else 0
                    cells = basin_zones == zone_idx
                    total_annual = float(annual[cells].sum()) * bias_factors.get(code, 1.0)
                    # 季节性沿用模型自身的；只校正水平。
                    modelled_annual = float(annual[cells].sum())
                    # 枯水期是流域量，不是网格量。
                    # 旧实现取 `rolling.min(axis=0)`——逐网格取最小值，每个网格各自挑选
                    # 自己的枯水季——再把这些互相独立的最小值在流域内求和。那是 sum(min)，
                    # 而约束需要的是 min(sum)：流域在其自身枯水季内的总流量。由 Jensen
                    # 不等式，只要各网格在不同月份见底，两者就不相等，且方向始终相同；差距
                    # 最大的恰恰是本研究判为超限的那些流域（2021-2030 窗口，
                    # cwatm|gfdl-esm4|ssp126，比值 min(sum)/sum(min)）：
                    #     C 海河 2.25x   K 西北诸河 2.13x   D 黄河 1.47x   E 淮河 1.19x
                    #     H 珠江 1.03x   F 长江 1.06x   G 东南诸河 1.11x
                    # 逐网格的相位信息无论如何都保留不下来：往下四行，流域就坍缩成一个
                    # `dry_share`，再按年径流权重重新分配到节点，所以单个网格在哪个月
                    # 见底，在它被使用之后的下一条语句就被丢弃了。
                    basin_rolling = rolling_volume[:, cells].sum(axis=1)
                    dry_share = float(basin_rolling.min()) / modelled_annual if modelled_annual > 0 else 0.0
                    total_dry = total_annual * dry_share
                    weights = node_annual[members]
                    weight_sum = float(weights.sum())
                    if weight_sum > 0:
                        share = weights / weight_sum
                    else:  # 该流域的节点上没有模拟径流：均分
                        share = np.full(int(members.sum()), 1.0 / max(1, int(members.sum())))
                    node_annual_out[members] = total_annual * share
                    node_dry_out[members] = total_dry * share

                for idx in range(len(node_ids)):
                    rows.append(
                        {
                            "water_node_id": str(node_ids[idx]),
                            "planning_year": int(planning_year),
                            "scenario_family": str(scenario_row["scenario_family"]),
                            "scenario_id": str(scenario_row["scenario_id"]),
                            "hydrology_model": str(scenario_row["hydrology_model"]),
                            "gcm": str(scenario_row["gcm"]),
                            "ssp": str(scenario_row["ssp"]),
                            "available_water_m3_per_year": max(0.0, float(node_annual_out[idx])),
                            "dry_season_water_m3_per_year": max(0.0, float(node_dry_out[idx])),
                            "local_runoff_m3_per_year": max(0.0, float(node_annual[idx])),
                            "basin_code": str(node_basins[idx]),
                            "bias_factor": round(float(bias_factors.get(str(node_basins[idx]), 1.0)), 4),
                            "proxy_type": "basin_budgeted_bias_corrected_local_runoff",
                            "unit": "m3/yr",
                            "source": str(scenario_row["source"]),
                            "year_basis": f"{window_start}-{window_end}",
                        }
                    )

    return pd.DataFrame(rows).sort_values(
        ["water_node_id", "planning_year", "scenario_family"]
    ).reset_index(drop=True)


def coarsen_water_inputs(
    fine_nodes: pd.DataFrame,
    fine_availability: pd.DataFrame,
    cell_degrees: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把水文网格聚合到更粗的网格上，供优化使用。

    网格按 (流域, 经度分箱, 纬度分箱) 分组，而不是只按分箱分组。只按分箱分组时，一个
    ~200 km 的网格会横跨多个预算单元，并被整体划给其最大节点所在的单元，从而把水量
    搬过单元边界：在早先的分省预算下，这使天津 +347%、北京 +107%、宁夏 +50%，而陕西
    减少 40%、山东减少 33%、河北减少 25%。由于可用水量预算是按流域构建的，这会悄悄
    改写模型本来要检验的那条约束。按流域拆分每个网格，可使粗化前后的流域总量保持一致。
    """
    nodes = fine_nodes.copy()
    nodes["_ci"] = np.floor(nodes["longitude"].astype(float) / cell_degrees).astype(int)
    nodes["_cj"] = np.floor(nodes["latitude"].astype(float) / cell_degrees).astype(int)
    nodes["_bid"] = nodes["basin_code"].astype(str) if "basin_code" in nodes.columns else "X"
    nodes["coarse_id"] = [
        f"WC_{ci:04d}_{cj:04d}_{bid}"
        for ci, cj, bid in zip(nodes["_ci"], nodes["_cj"], nodes["_bid"])
    ]

    weight = (
        fine_availability.groupby("water_node_id")["available_water_m3_per_year"].mean().rename("_w")
    )
    nodes = nodes.merge(weight, left_on="water_node_id", right_index=True, how="left")
    nodes["_w"] = nodes["_w"].fillna(0.0)

    rows: list[dict[str, object]] = []
    for coarse_id, group in nodes.groupby("coarse_id"):
        total = float(group["_w"].sum())
        if total > 0:
            lon = float((group["_w"] * group["longitude"]).sum() / total)
            lat = float((group["_w"] * group["latitude"]).sum() / total)
        else:
            lon = float(group["longitude"].mean())
            lat = float(group["latitude"].mean())
        rows.append(
            {
                "water_node_id": str(coarse_id),
                "province_name": str(group["province_name"].iloc[0]),
                "basin_code": str(group["_bid"].iloc[0]),
                "longitude": lon,
                "latitude": lat,
                "source": "data/water/*_qtot_*.nc",
                "year_basis": "static_hydrology_grid",
            }
        )
    coarse_nodes = pd.DataFrame(rows).sort_values("water_node_id").reset_index(drop=True)

    mapping = dict(zip(nodes["water_node_id"], nodes["coarse_id"]))
    availability = fine_availability.copy()
    availability["water_node_id"] = availability["water_node_id"].map(mapping)
    # scenario_id 必须留在分组键里：否则同一情景族（family）的所有气候成员会被加总
    # 成一个节点总量。
    keys = ["water_node_id", "planning_year", "scenario_family", "scenario_id",
            "hydrology_model", "gcm", "ssp", "basin_code"]
    keys = [key for key in keys if key in availability.columns]
    coarse_availability = availability.groupby(keys, as_index=False).agg(
        available_water_m3_per_year=("available_water_m3_per_year", "sum"),
        dry_season_water_m3_per_year=("dry_season_water_m3_per_year", "sum"),
        local_runoff_m3_per_year=("local_runoff_m3_per_year", "sum"),
        bias_factor=("bias_factor", "first"),
    )
    coarse_availability["proxy_type"] = "basin_budgeted_bias_corrected_local_runoff"
    coarse_availability["unit"] = "m3/yr"
    coarse_availability["source"] = "data/water/*_qtot_*.nc"
    coarse_availability["year_basis"] = "decadal window"
    return coarse_nodes, coarse_availability


def build_water_link_dataframe(
    paths: ProjectPaths,
    water_nodes: pd.DataFrame,
) -> pd.DataFrame:
    plants = pd.read_csv(paths.inputs_dir / "plants.csv").copy()
    plant_count = len(plants)
    node_lons = water_nodes["longitude"].astype(float).to_numpy()
    node_lats = water_nodes["latitude"].astype(float).to_numpy()
    rows: list[dict[str, object]] = []

    for plant in plants.itertuples(index=False):
        distances_km = _haversine_distances_km(
            float(plant.centroid_longitude),
            float(plant.centroid_latitude),
            node_lons,
            node_lats,
        )
        matched_indices = np.flatnonzero(distances_km <= WATER_MATCH_BUFFER_KM)
        if matched_indices.size == 0:
            continue
        ranked_indices = matched_indices[np.argsort(distances_km[matched_indices])]
        for rank, node_idx in enumerate(ranked_indices, start=1):
            node = water_nodes.iloc[int(node_idx)]
            rows.append(
                {
                    "plant_count": int(plant_count),
                    "plant_id": str(plant.plant_id),
                    "plant_index": int(plant.plant_index),
                    "plant_province_name": str(plant.province_mode),
                    "water_node_id": str(node["water_node_id"]),
                    "water_node_province_name": str(node["province_name"]),
                    "distance_km": round(float(distances_km[node_idx]), 3),
                    "distance_rank": int(rank),
                    "match_rule": "plant_buffer_intersects_water_node",
                    "competition_scope": "shared_water_node",
                    "buffer_km": WATER_MATCH_BUFFER_KM,
                    "source": str(node["source"]),
                    "year_basis": str(node["year_basis"]),
                }
            )

    columns = [
        "plant_count",
        "plant_id",
        "plant_index",
        "plant_province_name",
        "water_node_id",
        "water_node_province_name",
        "distance_km",
        "distance_rank",
        "match_rule",
        "competition_scope",
        "buffer_km",
        "source",
        "year_basis",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["plant_id", "distance_km", "water_node_id"]
    ).reset_index(drop=True)


def write_water_inputs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = build_water_scenarios_dataframe(paths)
    scenarios = scenarios[scenarios["ssp"].astype(str).str.startswith("ssp")].reset_index(drop=True)
    base = build_water_base_dataframe(scenarios)
    water_nodes = build_water_nodes_dataframe(paths, scenarios)
    water_availability = build_water_availability_dataframe(paths, scenarios, water_nodes)
    # 粗化以前是在构建之后手工施加的，所以已提交的输入无法从这个入口重新生成。
    # 它应当放在这里。
    if WATER_COARSE_GRID_DEGREES > 0:
        water_nodes, water_availability = coarsen_water_inputs(
            water_nodes, water_availability, WATER_COARSE_GRID_DEGREES
        )

    active_node_ids = set(
        water_availability.loc[
            water_availability["available_water_m3_per_year"].astype(float) > 0.0,
            "water_node_id",
        ].astype(str)
    )
    if active_node_ids:
        water_nodes = water_nodes[water_nodes["water_node_id"].astype(str).isin(active_node_ids)].reset_index(drop=True)
        water_availability = water_availability[
            water_availability["water_node_id"].astype(str).isin(active_node_ids)
        ].reset_index(drop=True)

    write_csv(scenarios, paths.inputs_dir / "water_scenarios.csv")
    write_csv(base, paths.inputs_dir / "water_base.csv")
    write_csv(water_nodes, paths.inputs_dir / "water_nodes.csv")
    write_csv(water_availability, paths.inputs_dir / "water_availability.csv")
    water_links = build_water_link_dataframe(paths, water_nodes)
    write_csv(water_links, paths.inputs_dir / "water_supply_links.csv")
    return scenarios, base, water_nodes, water_availability
