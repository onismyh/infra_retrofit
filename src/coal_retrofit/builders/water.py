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
    """Scenario members only. The `historical` runs share the directory but are not members:
    they are the baseline `basin_bias_factors` estimates each model's bias against."""
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


def _annualized_window_value(series: np.ndarray, years: np.ndarray, window: tuple[int, int]) -> float:
    start_year, end_year = window
    mask = (years >= start_year) & (years <= end_year)
    if not mask.any():
        return float("nan")
    window_values = series[mask]
    if not np.isfinite(window_values).any():
        return float("nan")
    return float(np.nanmean(window_values) * SECONDS_PER_YEAR)


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
    """A path netCDF4 can open: its C layer rejects non-ASCII absolute paths on Windows."""
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
    """Area of each latitude band's grid cell on a regular lat-lon grid (m²)."""
    half = cell_degrees / 2.0
    band = (
        EARTH_RADIUS_M ** 2
        * np.deg2rad(cell_degrees)
        * np.abs(np.sin(np.deg2rad(lat + half)) - np.sin(np.deg2rad(lat - half)))
    )
    return band


def _province_zone_grid(paths: ProjectPaths, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Rasterise province polygons onto the climate grid; 0 = outside China."""
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
    """Level-1 basin code for each grid node, by nearest polygon.

    Nearest rather than strict containment: cells on the coast and in the gaps between
    polygons would otherwise be dropped from every basin budget, silently deleting their
    water. Distances are computed in EPSG:2380 so they are metric.
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
    """Level-1 water-resource regions, the unit China publishes official water totals for."""
    path = paths.data_dir / "ChinaBasins" / "basin_l1.gpkg"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it from data/basins_raw with scripts/validate_basin_runoff.py's "
            "layer conventions: columns `code` (A-K) and `name`, EPSG:4326."
        )
    return gpd.read_file(path).to_crs(TARGET_GEO_CRS)


def _basin_zone_grid(paths: ProjectPaths, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Rasterise level-1 basin polygons onto the climate grid; 0 = outside every basin."""
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
    """The `historical` qtot run for one (hydrology model, GCM) pair, if downloaded.

    Both keys are required. ISIMIP3b drives each hydrology model with every GCM separately,
    so `historical` runs exist per pair, not per hydrology model. Matching on the hydrology
    model alone and taking the first sorted hit silently returned the gfdl-esm4 run for every
    GCM (it sorts first), which would have applied one GCM's bias factor to all of them and
    flattened exactly the GCM spread the ensemble is built to measure.
    """
    matches = sorted(
        p for p in (paths.data_dir / "water").glob("*historical*qtot*.nc")
        if p.name.startswith(f"{hydrology_model}_{gcm}_")
    )
    return matches[0] if matches else None


def _basin_runoff_from_file(
    path: Path, window: tuple[int, int], zones: np.ndarray, codes: list[str], area_m2: np.ndarray
) -> dict[str, float]:
    """Total runoff per basin over `window`, m3/yr, on an already-rasterised grid."""
    # netCDF4's C layer cannot open absolute paths with non-ASCII characters on Windows.
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
    """Multiplicative correction bringing a model's basin runoff onto the official baseline.

    Global hydrological models at 0.5 deg carry large regional biases even when their national
    total is right: over 1956-2014, WaterGAP2-2e reproduces China's total to 1.2% while
    running 2.43x too wet in the Hai basin and 1.69x too wet in the Huai -- the two basins
    that carry 358 GW of coal and where the availability constraint actually binds. Left
    uncorrected, that bias is reported as if it were hydrological-model uncertainty.

    The factor is estimated once per (hydrology model, GCM) pair from that pair's `historical`
    run and applied to every year and SSP of the pair, so each member's rate of change is
    preserved exactly and only the absolute level is replaced. Estimating it per pair rather
    than per hydrology model matters for the ensemble: the historical bias is a property of the
    forcing GCM as much as of the hydrology model, and reusing one GCM's factor across the
    others would remove genuine between-GCM spread. Returns 1.0 for every basin when that
    pair's historical run has not been downloaded.
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
    """Renewable water reaching each node, from LOCAL RUNOFF (ISIMIP `qtot`).

    The previous implementation read `dis` (routed river discharge) at each node's cell.
    Discharge is cumulative — a downstream cell carries all of its upstream catchment —
    so summing 293 nodes counted the same water tens of times: the national total came to
    430 766 x10^8 m3/yr, 15x China's actual renewable water resources, and the resulting
    constraint could never bind (demand/supply = 0.083%).

    `qtot` is locally generated runoff (kg m-2 s-1), so a grid-cell sum is a genuine water
    budget. Province-masked, this reproduces 26 291 x10^8 m3/yr against the official
    multi-year mean of ~28 000 — a 6% difference.

    Availability is budgeted per LEVEL-1 WATER-RESOURCE REGION and then split across that
    basin's nodes in proportion to their local runoff, so summing all nodes returns the
    basin's renewable water exactly once:

        basin_runoff   = sum(qtot x cell_area x seconds_per_year) x bias_factor   [m3/yr]
        node_available = basin_runoff x node_runoff / sum(node_runoff in basin)

    The basin, not the province, is the unit here for two reasons: water is a basin quantity
    (the Yellow River crosses nine provinces), and it is the scale at which global hydrology
    models are calibrated and at which China publishes official totals. Budgeting by province
    also produced ratios against official statistics from 0.34x to 4.00x, most of which was
    an artefact of cutting a 0.5 deg grid with provincial boundaries; by basin the spread is
    0.74x-2.43x, and `basin_bias_factors` removes even that.

    Environmental flow and existing withdrawals are NOT applied here — they are policy
    assumptions applied at solve time (see `WATER_EXTRACTABLE_FRACTION` and
    `OptimizationAssumptions.existing_withdrawal_share`), so their sensitivity can be run
    without rebuilding inputs.

    Both an annual mean and a dry-season (lowest three consecutive months, annualised)
    column are produced: thermal power is curtailed in low-flow periods, not at the annual
    mean, and the source data is monthly.

    The low-flow quarter is selected on the BASIN AGGREGATE -- min over the 12 candidate
    3-month windows of the basin's summed runoff -- not per grid cell. Selecting per cell
    and summing gives sum(min) rather than min(sum), which understates the basin's dry-season
    flow by 2.25x in the Hai and 2.13x in the Northwest Interior. See the comment block at the
    selection itself.

    WHAT THIS COLUMN IS NOT. `qtot` is unrouted runoff GENERATION, so the dry-season number is
    the rate at which water is produced in the basin's driest quarter -- not the rate at which
    it is available in the river. Reservoir regulation lives in the routing scheme, which this
    pipeline deliberately does not use (see the opening paragraph). Using this as a firm-yield
    budget therefore assumes ZERO storage, which is a strict assumption in the Hai, the Yellow
    and the Huai, where dry-season flow is largely a reservoir-release decision. It is stated
    here because the resulting constraint is the study's tightest, and its severity comes from
    this choice as much as from hydrology.
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
            # Cache on the pair: one factor set per (hydrology model, GCM), shared across SSPs.
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
                # kg m-2 s-1 -> m3/yr per cell (water density 1000 kg m-3)
                annual = np.nan_to_num(np.nanmean(window, axis=0)) * area_m2 * SECONDS_PER_YEAR / 1000.0
                # Dry season: lowest 3 consecutive calendar months of the climatological cycle
                months = np.arange(window.shape[0]) % 12
                monthly_clim = np.stack(
                    [np.nan_to_num(np.nanmean(window[months == m], axis=0)) for m in range(12)]
                )
                rolling = np.stack([monthly_clim[np.arange(m, m + 3) % 12].mean(axis=0) for m in range(12)])
                # Volume per cell for each of the 12 candidate 3-month windows. The basin's
                # low-flow quarter is chosen ONCE, on the basin total -- see the loop below.
                rolling_volume = rolling * area_m2 * SECONDS_PER_YEAR / 1000.0

                node_annual = annual[lat_indices, lon_indices]

                # Basin budgets, bias-corrected, then distributed across that basin's nodes.
                node_annual_out = np.zeros(len(node_ids), dtype=np.float64)
                node_dry_out = np.zeros(len(node_ids), dtype=np.float64)
                for code in set(node_basins):
                    members = node_basins == code
                    zone_idx = basin_codes.index(code) + 1 if code in basin_codes else 0
                    cells = basin_zones == zone_idx
                    total_annual = float(annual[cells].sum()) * bias_factors.get(code, 1.0)
                    # Seasonality stays the model's own; only the level is corrected.
                    modelled_annual = float(annual[cells].sum())
                    # THE DRY SEASON IS A BASIN QUANTITY, NOT A CELL QUANTITY.
                    # The previous implementation took `rolling.min(axis=0)` -- a minimum per
                    # grid cell, each cell free to pick its own low-flow quarter -- and summed
                    # those independent minima over the basin. That is sum(min), when the
                    # constraint needs min(sum): the basin's total flow during the basin's own
                    # low-flow quarter. By Jensen the two differ whenever cells bottom out in
                    # different months, always in the same direction, and the gap was largest
                    # in precisely the basins this study calls over-limit (2021-2030 window,
                    # cwatm|gfdl-esm4|ssp126, ratio min(sum)/sum(min)):
                    #     C Hai 2.25x   K Northwest 2.13x   D Yellow 1.47x   E Huai 1.19x
                    #     H Pearl 1.03x   F Yangtze 1.06x   G Southeast 1.11x
                    # The per-cell phase information could not survive in any case: four lines
                    # below, the basin collapses to ONE `dry_share` redistributed to nodes by
                    # ANNUAL runoff weights, so which month a single cell bottomed out in is
                    # discarded one statement after it is used.
                    basin_rolling = rolling_volume[:, cells].sum(axis=1)
                    dry_share = float(basin_rolling.min()) / modelled_annual if modelled_annual > 0 else 0.0
                    total_dry = total_annual * dry_share
                    weights = node_annual[members]
                    weight_sum = float(weights.sum())
                    if weight_sum > 0:
                        share = weights / weight_sum
                    else:  # basin with no modelled runoff at its nodes: split evenly
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
    """Aggregate the hydrology grid onto coarser cells for the optimisation.

    Cells are grouped by (basin, lon bin, lat bin) rather than by bin alone. Grouping on the
    bin only lets one ~200 km cell straddle several budget units and be assigned wholly to
    whichever unit holds its largest node, which moves water across their borders: under the
    earlier provincial budget that gave Tianjin +347%, Beijing +107% and Ningxia +50% while
    Shaanxi lost 40%, Shandong 33% and Hebei 25%. Since the availability budget is built per
    basin, that would silently rewrite the very constraint the model is meant to test.
    Splitting each cell by basin keeps basin totals identical before and after coarsening.
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
    # scenario_id must stay in the key: without it every climate member of a family would be
    # summed into one node total.
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
    # Coarsening used to be applied by hand after the build, so the committed inputs could
    # not be regenerated from this entry point. It belongs here.
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
