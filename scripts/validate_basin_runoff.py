"""Validate modelled runoff against official statistics at the water-resource-region scale.

The provincial check (research note plan/nature_water_storyline_and_figures.md v3, not in
the repository) found ratios from 0.34x (Tianjin) to 4.00x (Ningxia). Part of that spread is
a genuine model bias in arid regions, part is an artefact of cutting a 0.5 deg grid with
provincial boundaries: a small province spans only a handful of cells, and Ningxia's water
is a Yellow River allocation rather than local runoff, which `qtot` cannot represent by
construction.

Global hydrological models are calibrated and evaluated against BASIN discharge, so the
water-resource region (codes A-K) is the scale where the comparison is meaningful and where
China publishes matching official totals.

Two windows are compared, because they answer different questions:

  BASELINE     model 1956-2014 (historical run) vs the Third National Water Resources
               Survey and Evaluation, 1956-2016. Pure model bias, no forcing signal.
               Needs the `historical` qtot run of each (hydrology model, GCM) pair:
               `download_isimip_extra_gcms.py` fetches them for the four extra GCMs, the
               gfdl-esm4 `historical` runs have no download script. data/water is not tracked,
               so a checkout without them skips this block; the bias factors in
               `_indtree/inputs/water_availability.csv` (all != 1) show the build machine had
               all ten.
  CONTEMPORARY model 2015-2024 (scenario runs) vs the bulletin series 2015-2024. SSP1-2.6
               and SSP3-7.0 have barely diverged this early, so this window gives a bias
               estimate for every member including those without a historical file.

Every member is one column, keyed `<hydrology model>|<GCM>|<SSP or historical>`; through
`d519525` the key dropped the GCM, so the five GCMs of one model and SSP collapsed into a
single column (the last one sorted won).

Basin rasterisation, cell areas, annual and three-month-window volumes and the plant-to-basin
assignment are the builder's own functions (`coal_retrofit.builders.water`), so this check
cannot drift from what `build_water_inputs.py` writes.

Usage:  python scripts/validate_basin_runoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from _bootstrap import ROOT
from coal_retrofit.constants import BIAS_BASELINE_WINDOW, OFFICIAL_BASIN_WATER_1E8_M3
from coal_retrofit.builders.water import (
    _annual_and_seasonal,
    _assign_basin_codes,
    _basin_dry_window,
    _basin_zone_grid,
    _cell_area_m2,
    _clean_series,
    _file_years,
    _nc_path,
    load_basins,
)
from coal_retrofit.paths import ProjectPaths

PATHS = ProjectPaths(ROOT)
WATER_DIR = PATHS.data_dir / "water"

# Official water resources by water-resource region, 10^8 m3/yr. The baseline column (Third
# National Water Resources Survey and Evaluation, 1956-2016) is the model's own bias reference,
# `constants.OFFICIAL_BASIN_WATER_1E8_M3`, and is not copied here; the recent column is the China
# Water Resources Bulletin series, 2015-2024 mean. Both are transcribed from Wang G. et al. (2025)
# Advances in Water Science 36(6), Table 1 (reference/water/wangGQ2025_adv_water_sci_36_948.pdf),
# internally consistent with that paper's aggregates: north six 5221, south four 23078, national
# 28299. The boundary file merges Songhua and Liao into one polygon (code A), so they are summed.
BULLETIN_2015_2024_1E8_M3: dict[str, float] = {
    "A": 1972.1 + 463.1,   # Songhua + Liao
    "C": 353.5,            # Hai
    "D": 753.1,            # Yellow
    "E": 953.4,            # Huai
    "F": 10451.5,          # Yangtze
    "G": 2090.2,           # Southeast rivers
    "H": 4908.6,           # Pearl
    "J": 5528.5,           # Southwest rivers
    "K": 1413.1,           # Northwest interior
}

BASELINE_WINDOW = BIAS_BASELINE_WINDOW   # (1956, 2014): historical run stops at 2014; official ends 2016
CONTEMPORARY_WINDOW = (2015, 2024)       # matches the bulletin mean quoted above


def basin_runoff(nc_path: Path, window: tuple[int, int]) -> pd.DataFrame:
    """Annual-mean and dry-season runoff per level-1 basin over `window`, 10^8 m3/yr.

    The dry season is the basin's own low-flow quarter (`_basin_dry_window`: min over the 12
    windows of the basin sum), not the sum of per-cell minima, which understates it (2.25x in
    the Hai). An earlier version of this script carried that defect and therefore validated
    the annual total while blessing a dry-season total that was wrong.
    """
    with netCDF4.Dataset(_nc_path(nc_path), "r") as ds:
        years = _file_years(ds)
        mask = (years >= window[0]) & (years <= window[1])
        if not mask.any():
            raise ValueError(f"{nc_path.name} covers {years.min()}-{years.max()}, not {window}")
        runoff_var = ds.variables["qtot"]
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        series = _clean_series(np.asarray(runoff_var[mask]), getattr(runoff_var, "_FillValue", None))
    area = _cell_area_m2(lat)[:, None] * np.ones((1, len(lon)))
    zones, codes = _basin_zone_grid(PATHS, lat, lon)
    annual, rolling_volume = _annual_and_seasonal(series, area)
    basins = load_basins(PATHS)
    names = dict(zip(basins["code"].astype(str), zip(basins["name"], basins["name_en"], strict=True), strict=True))
    rows = []
    for idx, code in enumerate(codes, start=1):
        cells = zones == idx
        rows.append({
            "code": code,
            "name": names[code][0],
            "name_en": names[code][1],
            "model": float(annual[cells].sum()) / 1e8,
            "model_dry": _basin_dry_window(rolling_volume, cells)[1] / 1e8,
        })
    return pd.DataFrame(rows).set_index("code")


def _member_tag(path: Path) -> str:
    """`<hydrology model>|<GCM>|<SSP or historical>`, e.g. `cwatm|gfdl-esm4|126`."""
    hydro, gcm, _, run = path.name.split("_")[:4]
    return f"{hydro.replace('watergap2-2e', 'wgap')}|{gcm}|{run.replace('ssp', '')}"


def _frames_by_member(files: list[Path], window: tuple[int, int]) -> dict[str, pd.DataFrame]:
    """One basin table per file, keyed by `_member_tag`; two files on one key would silently drop one."""
    tags = [_member_tag(p) for p in files]
    collided = sorted(p.name for p, tag in zip(files, tags, strict=True) if tags.count(tag) > 1)
    if collided:
        raise ValueError(f"member tags collided, files would share a column: {collided}")
    return {tag: basin_runoff(p, window) for tag, p in zip(tags, files, strict=True)}


def coal_capacity_by_basin() -> pd.Series:
    """各一级流域的煤电装机（GW），hub 按所在或最近的流域多边形归属（`builders.water._assign_basin_codes`）。"""
    plants = pd.read_csv(PATHS.inputs_dir / "plants.csv")
    points = plants.rename(columns={"centroid_longitude": "longitude", "centroid_latitude": "latitude"})
    plants["code"] = _assign_basin_codes(PATHS, points)
    return plants.groupby("code")["total_capacity_mw"].sum() / 1000.0


def _report(title: str, frames: dict[str, pd.DataFrame], official: pd.Series, coal_gw: pd.Series) -> pd.DataFrame:
    base = next(iter(frames.values()))
    table = base[["name"]].copy()
    for tag, frame in frames.items():
        table[tag] = (frame["model"] / official).round(2)
    table["official"] = official.round(1)
    table["coal_GW"] = coal_gw.reindex(table.index).fillna(0.0)   # 没有煤电 hub 的流域（西南诸河）记 0
    table[f"dry_share[{next(iter(frames))}]"] = (base["model_dry"] / base["model"]).round(2)

    print(f"\n{'=' * 92}\n  {title}\n{'=' * 92}")
    print(table.to_string())
    for tag, frame in frames.items():
        print(f"  national ratio {tag:>28s}  {frame['model'].sum() / official.sum():.3f}")
    return table


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    baseline = pd.Series(OFFICIAL_BASIN_WATER_1E8_M3)
    recent = pd.Series(BULLETIN_2015_2024_1E8_M3)
    coal_gw = coal_capacity_by_basin().round(0)

    hist_files = sorted(WATER_DIR.glob("*historical*qtot*.nc"))
    if hist_files:
        frames = _frames_by_member(hist_files, BASELINE_WINDOW)
        _report(
            f"BASELINE  model {BASELINE_WINDOW[0]}-{BASELINE_WINDOW[1]} vs "
            "Third National Survey 1956-2016",
            frames,
            baseline,
            coal_gw,
        )
    else:
        print("no historical qtot files - baseline comparison skipped")

    scen_files = sorted(p for p in WATER_DIR.glob("*qtot*.nc") if "historical" not in p.name)
    frames = _frames_by_member(scen_files, CONTEMPORARY_WINDOW)
    _report(
        f"CONTEMPORARY  model {CONTEMPORARY_WINDOW[0]}-{CONTEMPORARY_WINDOW[1]} vs "
        "bulletin series 2015-2024",
        frames,
        recent,
        coal_gw,
    )


if __name__ == "__main__":
    main()
