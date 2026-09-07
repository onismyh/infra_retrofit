"""Basin water budget on the official 用水总量控制指标 basis.

Replaces the runoff-derived budget (`qtot x 0.20 x (1 - 0.85)`) with China's own allocation
instrument. See `constants_water_quota` for why and for every source. The basis is 用水量 as the
水资源公报 defines it -- gross intake, once-through condenser flow INCLUDED -- so the model term
these caps bind is plant withdrawal, not the 定额 volume in `quota_intensity_m3_per_mwh`, which
excludes that flow and would understate the fleet's claim roughly fivefold.

THE APPORTIONMENT PROBLEM. 国办发〔2013〕2号 caps water by PROVINCE; the model constrains by
level-1 water-resource region. No official basin table exists (附件1 is a closed provincial
partition with zero residual), and provinces do not nest inside basins. So each province's cap
must be split across the basins it spans, and the split needs a spatial key.

Three keys were considered:

    area            wrong: water use is nowhere near uniform in space. Inner Mongolia's area is
                    mostly in the Northwest region but most of its use is on the Yellow.
    population      wrong for the same reason inverted: agriculture is 62% of national use.
    gridded demand  used here. ISIMIP `ptotuse` is spatially explicit and already on the model's
                    own 0.5-degree grid.

The known level bias in ISIMIP demand (it over-states north China irrigation by 2-3x against the
水资源公报) does NOT propagate: the field enters only as a WITHIN-PROVINCE share, so any factor
common to a province cancels exactly. What would propagate is a bias that varies across basins
*inside one province*; that is a real residual uncertainty and is why `basin_caps` can be re-run
with `weight="area"` as a sensitivity.

The weight field is deliberately fixed (one SSP, one decade) so the apportionment key is a piece
of geography, not a scenario variable. Changing scenario must move the water budget through the
climate signal only, never by silently redrawing the province-to-basin split.
"""
from __future__ import annotations

import logging
from typing import Final, Literal

import netCDF4
import numpy as np
import pandas as pd

from ..constants import COMBUSTION_CLASS_MAP, WATER_INTENSITY_BY_TECH_M3_PER_MWH
from ..constants_water_quota import (
    BASIN_NAMES_ZH,
    BASIN_WITHDRAWAL_2025_1E8_M3,
    NATIONAL_ONCE_THROUGH_WITHDRAWAL_1E8_M3,
    PROVINCE_WATER_CAP_1E8_M3,
    PROVINCES_WITHOUT_CAP,
    province_cap,
)
from ..paths import ProjectPaths
from .water import (
    SECONDS_PER_YEAR,
    _basin_zone_grid,
    _cell_area_m2,
    _clean_series,
    _nc_path,
    _province_zone_grid,
)

logger = logging.getLogger(__name__)

# Fixed apportionment key: one hydrology model, one SSP, one decade. See module docstring.
WEIGHT_FILE: Final[str] = (
    "cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_"
    "ptotuse_global_monthly_2015_2100.nc"
)
WEIGHT_VARIABLE: Final[str] = "ptotuse"
WEIGHT_WINDOW: Final[tuple[int, int]] = (2021, 2030)

WeightKind = Literal["demand", "area"]


def _short_province_name(full_name: str) -> str | None:
    """Map a `provinces.shp` full name to the short form used by the cap table.

    Every short name in 附件1 is a prefix of its shapefile full name (内蒙古自治区 -> 内蒙古,
    新疆维吾尔自治区 -> 新疆), so longest-prefix matching is exact. Returns None for the three
    regions that carry no cap.
    """
    candidates = [name for name in PROVINCE_WATER_CAP_1E8_M3 if full_name.startswith(name)]
    if candidates:
        return max(candidates, key=len)
    if any(full_name.startswith(name) for name in PROVINCES_WITHOUT_CAP):
        return None
    raise KeyError(f"province {full_name!r} matches no entry in PROVINCE_WATER_CAP_1E8_M3")


def _weight_grid(paths: ProjectPaths, kind: WeightKind) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (weight, lat, lon) for the apportionment key, on the ISIMIP 0.5-degree grid."""
    path = paths.data_dir / "water" / WEIGHT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_isimip_water_use.py first."
        )
    with netCDF4.Dataset(_nc_path(path), "r") as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        area = _cell_area_m2(lat)[:, None] * np.ones((1, len(lon)))
        if kind == "area":
            return area, lat, lon
        time_var = dataset.variables["time"]
        years = np.array([
            date.year for date in netCDF4.num2date(
                time_var[:], time_var.units, getattr(time_var, "calendar", "standard")
            )
        ])
        window = (years >= WEIGHT_WINDOW[0]) & (years <= WEIGHT_WINDOW[1])
        variable = dataset.variables[WEIGHT_VARIABLE]
        series = _clean_series(np.asarray(variable[window]), getattr(variable, "_FillValue", None))
        weight = np.nan_to_num(np.nanmean(series, axis=0)) * area * SECONDS_PER_YEAR / 1000.0
    return np.clip(weight, 0.0, None), lat, lon


def province_basin_shares(
    paths: ProjectPaths, kind: WeightKind = "demand"
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Share of each province's water use falling in each basin.

    Cells inside a province but outside every basin polygon (coastal fringe, polygon gaps) carry
    no basin and are dropped before normalising, which redistributes them across that province's
    basins in proportion -- the alternative, dropping their water outright, would silently shrink
    the national cap.

    Returns:
        (shares, unassigned) where `shares[province][basin]` sums to 1.0 over basins for every
        province that has any weight, and `unassigned` is the per-province weight fraction that
        fell outside all basins, for reporting.
    """
    weight, lat, lon = _weight_grid(paths, kind)
    province_zones, province_names = _province_zone_grid(paths, lat, lon)
    basin_zones, basin_codes = _basin_zone_grid(paths, lat, lon)

    shares: dict[str, dict[str, float]] = {}
    unassigned = np.zeros(len(province_names), dtype=np.float64)
    for index, full_name in enumerate(province_names, start=1):
        short = _short_province_name(full_name)
        if short is None:
            continue
        mask = province_zones == index
        total = float(weight[mask].sum())
        if total <= 0.0:
            logger.warning("province %s has zero apportionment weight; falling back to area", short)
            area_weight, _, _ = _weight_grid(paths, "area")
            weight_here, total = area_weight, float(area_weight[mask].sum())
        else:
            weight_here = weight
        inside = mask & (basin_zones > 0)
        unassigned[index - 1] = (total - float(weight_here[inside].sum())) / total if total else 0.0
        assigned = float(weight_here[inside].sum())
        if assigned <= 0.0:
            raise ValueError(f"province {short} has no cell inside any basin polygon")
        by_basin = {
            code: float(weight_here[inside & (basin_zones == basin_index)].sum()) / assigned
            for basin_index, code in enumerate(basin_codes, start=1)
        }
        # Provinces split across the 松花江/辽河 boundary already land in the merged code A
        # because `basin_l1.gpkg` stores them merged; nothing to fold here.
        shares[short] = {code: value for code, value in by_basin.items() if value > 0.0}
    return shares, unassigned


def basin_caps(
    paths: ProjectPaths, year: int, kind: WeightKind = "demand"
) -> dict[str, float]:
    """Official 用水总量控制指标 apportioned to level-1 basins, 亿 m3/yr, withdrawal basis."""
    shares, _ = province_basin_shares(paths, kind)
    caps: dict[str, float] = {code: 0.0 for code in BASIN_NAMES_ZH}
    for province, by_basin in shares.items():
        cap = province_cap(province, year)
        for code, share in by_basin.items():
            caps[code] = caps.get(code, 0.0) + cap * share
    return caps


def once_through_withdrawal(plants: pd.DataFrame, generation_mwh: pd.Series) -> pd.Series:
    """Freshwater once-through withdrawal per hub, m3/yr, on the model's raw (US) intensities.

    Split out from the hub's blended withdrawal because only the once-through part needs
    recalibrating: the recirculating and air rows of `WATER_INTENSITY_BY_TECH_M3_PER_MWH` were
    already cross-checked against Macknick to within 5-14% and agree with Chinese practice, while
    the once-through rows are the ones that disagree with the 水资源公报 by a factor of two.

    Seawater-cooled hubs return zero: the 公报 excludes 海水直接利用量, and `builders/plants.py`
    already zeroes their condenser flow, so both sides of the calibration exclude the same fleet.
    """
    once_through_rows = {
        combustion: values["withdrawal"]
        for (combustion, cooling), values in WATER_INTENSITY_BY_TECH_M3_PER_MWH.items()
        if cooling == "once-through"
    }
    # `dominant_combustion` carries raw GEM labels: "CFB", and "<class>/CCS" where the suffix
    # describes the retrofit state, not the steam cycle. COMBUSTION_CLASS_MAP is the same
    # normalisation `builders/plants.py` applies before looking up the intensity table.
    normalised = (
        plants["dominant_combustion"].astype(str)
        .str.split("/").str[0].str.strip().str.lower()
        .map(COMBUSTION_CLASS_MAP)
    )
    intensity = normalised.map(once_through_rows)
    if intensity.isna().any():
        unmapped = sorted(plants.loc[intensity.isna(), "dominant_combustion"].astype(str).unique())
        raise ValueError(f"combustion classes with no once-through intensity row: {unmapped}")
    share = plants["capacity_mw_once_through"].astype(float) / plants["total_capacity_mw"].astype(float)
    freshwater = ~plants["seawater_cooled"].astype(bool)
    return generation_mwh * share * intensity * freshwater.astype(float)


def once_through_calibration(plants: pd.DataFrame, generation_mwh: pd.Series) -> float:
    """Factor rescaling modelled once-through withdrawal onto the 水资源公报 baseline.

    Same construction as `water.basin_bias_factors`: official total over modelled total, applied
    multiplicatively. One national factor rather than nine basin factors -- the discrepancy is a
    property of the intensity table (a US condenser design assumption), not of any basin, and the
    bulletin publishes 直流火(核)电 by region but not by fuel, so a per-basin factor would be
    fitting coal's share of each region's nuclear and gas fleet, which is not observed.
    """
    modelled = float(once_through_withdrawal(plants, generation_mwh).sum())
    if modelled <= 0.0:
        raise ValueError("no freshwater once-through withdrawal in the fleet; nothing to calibrate")
    return NATIONAL_ONCE_THROUGH_WITHDRAWAL_1E8_M3 * 1e8 / modelled


def basin_reserved_withdrawal(exclude_all_industry: bool = False) -> dict[str, float]:
    """Withdrawal already committed to users the model does not decide, 亿 m3/yr.

    The 水资源公报 reports 工业用水 with thermal power folded in: 直流火(核)电 is broken out as a
    sub-column, but closed-cycle and air-cooled thermal cooling sit inside the industry residual
    with no sub-column and cannot be recovered from this source. So there are two defensible
    reservations and they bracket the truth:

        exclude_all_industry=False  reserve 生活 + 农业 + 生态 + (工业 - 直流火(核)电).
                                    Closed-cycle thermal stays reserved AND is then re-spent by
                                    the model, so the same water is charged twice. Conservative:
                                    it under-states what is available to power.
        exclude_all_industry=True   reserve 生活 + 农业 + 生态 only. All industry -- including
                                    the non-power part the model does not decide -- is handed to
                                    the model. Optimistic: it over-states availability.

    The headline budget uses the conservative form; the optimistic form is the sensitivity that
    bounds how much the un-separable closed-cycle term can matter.
    """
    reserved: dict[str, float] = {}
    for code, row in BASIN_WITHDRAWAL_2025_1E8_M3.items():
        base = row["domestic"] + row["agriculture"] + row["ecology"]
        if not exclude_all_industry:
            base += row["industry"] - row["thermal_once_through"]
        reserved[code] = base
    return reserved
