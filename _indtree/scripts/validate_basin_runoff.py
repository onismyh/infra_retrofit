"""Validate modelled runoff against official statistics at the water-resource-region scale.

The provincial check (see plan/nature_water_storyline_and_figures.md v3) found ratios from
0.34x (Tianjin) to 4.00x (Ningxia). Part of that spread is a genuine model bias in arid
regions, part is an artefact of cutting a 0.5 deg grid with provincial boundaries: a small
province spans only a handful of cells, and Ningxia's water is a Yellow River allocation
rather than local runoff, which `qtot` cannot represent by construction.

Global hydrological models are calibrated and evaluated against BASIN discharge, so the
water-resource region (codes A-K) is the scale where the comparison is meaningful and where
China publishes matching official totals.

Two windows are compared, because they answer different questions:

  BASELINE     model 1956-2014 (historical run) vs the Third National Water Resources
               Survey and Evaluation, 1956-2016. Pure model bias, no forcing signal.
               Requires a `historical` qtot file; only CWatM has one locally.
  CONTEMPORARY model 2015-2024 (scenario runs) vs the bulletin series 2015-2024. SSP1-2.6
               and SSP3-7.0 have barely diverged this early, so this window gives a bias
               estimate for every member including those without a historical file.

Usage:  python scripts/validate_basin_runoff.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from coal_retrofit.builders.water import SECONDS_PER_YEAR, _cell_area_m2, _clean_series

ROOT = Path(__file__).resolve().parents[1]
BASIN_L1 = ROOT / "data" / "ChinaBasins" / "basin_l1.gpkg"
WATER_DIR = ROOT / "data" / "water"

# Official water resources by water-resource region, 10^8 m3/yr.
#   baseline    Third National Water Resources Survey and Evaluation, 1956-2016
#   recent      China Water Resources Bulletin series, 2015-2024 mean
# Both transcribed from Wang G. et al. (2025) Advances in Water Science 36(6), Table 1
# (reference/water/wangGQ2025_adv_water_sci_36_948.pdf). Internally consistent with that
# paper's stated aggregates: north six 5221, south four 23078, national 28299.
# The boundary file merges Songhua and Liao into one polygon (code A), so they are summed.
OFFICIAL_1E8_M3: dict[str, tuple[float, float]] = {
    "A": (1469.2 + 483.4, 1972.1 + 463.1),   # Songhua + Liao
    "C": (327.6, 353.5),                     # Hai
    "D": (702.8, 753.1),                     # Yellow
    "E": (928.3, 953.4),                     # Huai
    "F": (9871.2, 10451.5),                  # Yangtze
    "G": (2694.5, 2090.2),                   # Southeast rivers
    "H": (4758.6, 4908.6),                   # Pearl
    "J": (5753.8, 5528.5),                   # Southwest rivers
    "K": (1310.1, 1413.1),                   # Northwest interior
}

BASELINE_WINDOW = (1956, 2014)      # historical run stops at 2014; official ends 2016
CONTEMPORARY_WINDOW = (2015, 2024)  # matches the bulletin mean quoted above


def _basin_zone_grid(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    """Rasterise level-1 basins onto the climate grid; 0 = outside China."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    basins = gpd.read_file(BASIN_L1).to_crs("EPSG:4326")
    cell = float(abs(lat[1] - lat[0]))
    transform = from_origin(float(lon.min() - cell / 2.0), float(lat.max() + cell / 2.0), cell, cell)
    zones = rasterize(
        ((geom, idx + 1) for idx, geom in enumerate(basins.geometry)),
        out_shape=(len(lat), len(lon)),
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return zones, list(zip(basins["code"], basins["name"], basins["name_en"]))


def basin_runoff(nc_path: Path, window: tuple[int, int]) -> pd.DataFrame:
    """Annual-mean and dry-season runoff per level-1 basin over `window`, 10^8 m3/yr."""
    # The netCDF4 C library cannot open absolute paths containing non-ASCII characters on
    # Windows, and this repository lives under one. A cwd-relative path avoids the issue.
    with netCDF4.Dataset(os.path.relpath(nc_path, Path.cwd()), "r") as ds:
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
            raise ValueError(f"{nc_path.name} covers {years.min()}-{years.max()}, not {window}")

        runoff_var = ds.variables["qtot"]
        fill_value = getattr(runoff_var, "_FillValue", None)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        area = _cell_area_m2(lat)[:, None] * np.ones((1, len(lon)))
        zones, labels = _basin_zone_grid(lat, lon)

        series = _clean_series(np.asarray(runoff_var[mask]), fill_value)
        annual = np.nan_to_num(np.nanmean(series, axis=0)) * area * SECONDS_PER_YEAR / 1000.0
        months = np.arange(series.shape[0]) % 12
        clim = np.stack([np.nan_to_num(np.nanmean(series[months == m], axis=0)) for m in range(12)])
        rolling = np.stack([clim[np.arange(m, m + 3) % 12].mean(axis=0) for m in range(12)])
        # min(sum), not sum(min): the basin's low-flow quarter is a basin quantity.
        # See builders/water.py for why the per-cell minimum understates it (2.25x in
        # the Hai). This script previously carried the same defect and therefore
        # validated the annual total while blessing a dry-season total that was wrong.
        rolling_volume = rolling * area * SECONDS_PER_YEAR / 1000.0

    return pd.DataFrame(
        [
            {
                "code": code,
                "name": name,
                "name_en": name_en,
                "model": float(annual[zones == idx].sum()) / 1e8,
                "model_dry": float(rolling_volume[:, zones == idx].sum(axis=1).min()) / 1e8,
            }
            for idx, (code, name, name_en) in enumerate(labels, start=1)
        ]
    ).set_index("code")


def _member_tag(path: Path) -> str:
    parts = path.name.split("_")
    return parts[0].replace("watergap2-2e", "wgap") + "|" + parts[3].replace("ssp", "")


def coal_capacity_by_basin() -> pd.Series:
    plants = pd.read_csv(ROOT / "inputs" / "plants.csv")
    points = gpd.GeoDataFrame(
        plants,
        geometry=gpd.points_from_xy(plants["centroid_longitude"], plants["centroid_latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2380")
    basins = gpd.read_file(BASIN_L1).to_crs("EPSG:2380")
    joined = gpd.sjoin_nearest(points, basins[["code", "geometry"]], how="left")
    joined = joined[~joined.index.duplicated()]
    return joined.groupby("code")["total_capacity_mw"].sum() / 1000.0


def _report(title: str, frames: dict[str, pd.DataFrame], official: pd.Series) -> pd.DataFrame:
    base = next(iter(frames.values()))
    table = base[["name"]].copy()
    for tag, frame in frames.items():
        table[tag] = (frame["model"] / official).round(2)
    table["official"] = official.round(1)
    table["coal_GW"] = coal_capacity_by_basin().round(0)
    table["dry_share"] = (base["model_dry"] / base["model"]).round(2)

    print(f"\n{'=' * 92}\n  {title}\n{'=' * 92}")
    print(table.to_string())
    for tag, frame in frames.items():
        print(f"  national ratio {tag:>16s}  {frame['model'].sum() / official.sum():.3f}")
    return table


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    baseline = pd.Series({k: v[0] for k, v in OFFICIAL_1E8_M3.items()})
    recent = pd.Series({k: v[1] for k, v in OFFICIAL_1E8_M3.items()})

    hist_files = sorted(WATER_DIR.glob("*historical*qtot*.nc"))
    if hist_files:
        frames = {_member_tag(p): basin_runoff(p, BASELINE_WINDOW) for p in hist_files}
        _report(
            f"BASELINE  model {BASELINE_WINDOW[0]}-{BASELINE_WINDOW[1]} vs "
            "Third National Survey 1956-2016",
            frames,
            baseline,
        )
    else:
        print("no historical qtot files - baseline comparison skipped")

    scen_files = sorted(p for p in WATER_DIR.glob("*qtot*.nc") if "historical" not in p.name)
    frames = {_member_tag(p): basin_runoff(p, CONTEMPORARY_WINDOW) for p in scen_files}
    _report(
        f"CONTEMPORARY  model {CONTEMPORARY_WINDOW[0]}-{CONTEMPORARY_WINDOW[1]} vs "
        "bulletin series 2015-2024",
        frames,
        recent,
    )


if __name__ == "__main__":
    main()
