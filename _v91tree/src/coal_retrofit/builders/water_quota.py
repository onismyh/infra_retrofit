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


def _once_through_component(plants: pd.DataFrame, capture: bool) -> pd.Series:
    """Freshwater once-through withdrawal intensity per hub, m3/MWh, before calibration.

    Read straight off the column `builders/plants.finalize_water_intensities` writes, which is
    the exact per-unit difference `_intensity_all - _intensity_fresh`. Do NOT reconstruct it
    from `dominant_combustion` x `capacity_mw_once_through`: a hub's dominant steam cycle is
    often not the steam cycle of its once-through units, and that reconstruction was wrong by
    24% on the fleet total (404 vs 530 1e8 m3/yr), silently, because the error is one-sided
    only after clipping.
    """
    column = ("once_through_withdrawal_ccs_intensity_m3_per_mwh" if capture
              else "once_through_withdrawal_intensity_m3_per_mwh")
    if column not in plants.columns:
        raise KeyError(
            f"plants.csv lacks {column}. Rebuild it: python scripts/build_plant_inputs.py --hubs"
        )
    return plants[column].astype(float)


def once_through_withdrawal(plants: pd.DataFrame, generation_mwh: pd.Series) -> pd.Series:
    """Freshwater once-through withdrawal per hub, m3/yr, on the model's raw (US) intensities.

    Split out from the hub's blended withdrawal because only the once-through part needs
    recalibrating: the recirculating and air rows of `WATER_INTENSITY_BY_TECH_M3_PER_MWH` were
    already cross-checked against Macknick to within 5-14% and agree with Chinese practice, while
    the once-through rows are the ones that disagree with the 水资源公报 by a factor of two.

    Seawater-cooled hubs return zero: the 公报 excludes 海水直接利用量, and `builders/plants.py`
    already zeroes their condenser flow, so both sides of the calibration exclude the same fleet.
    """
    return generation_mwh * _once_through_component(plants, capture=False)


def calibrated_withdrawal_intensities(
    plants: pd.DataFrame, generation_mwh: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, float]:
    """Per-hub withdrawal intensities on the 水资源公报 scale, m3/MWh.

    Returns (base, capture, air_base, air_capture, factor). Since

        calibrated = blended + once_through_component x (k - 1)

    the calibration is generation-free once `k` is known, which is what lets it be applied to
    intensities and hence to a per-pathway matrix.

    The two air-cooled variants carry NO once-through term at all: a converted condenser has no
    pass-through flow, so on the withdrawal basis dry conversion removes ~90 m3/MWh rather than
    the ~1 m3/MWh it removes on the consumption basis. That gap is the whole reason the basin cap
    has to see a per-pathway matrix and not a per-hub ratio, and it is why every column here is
    read off `plants.csv` -- weighted by each unit's own steam cycle -- rather than rebuilt from
    the hub's dominant one.
    """
    factor = once_through_calibration(plants, generation_mwh)
    base = (plants["withdrawal_intensity_m3_per_mwh"].astype(float)
            + _once_through_component(plants, capture=False) * (factor - 1.0)).clip(lower=0.0)
    capture = (plants["withdrawal_ccs_intensity_m3_per_mwh"].astype(float)
               + _once_through_component(plants, capture=True) * (factor - 1.0)).clip(lower=0.0)
    for column in ("air_withdrawal_intensity_m3_per_mwh", "air_withdrawal_ccs_intensity_m3_per_mwh"):
        if column not in plants.columns:
            raise KeyError(
                f"plants.csv lacks {column}. Rebuild it: python scripts/build_plant_inputs.py --hubs"
            )
    air_base = plants["air_withdrawal_intensity_m3_per_mwh"].astype(float)
    air_capture = plants["air_withdrawal_ccs_intensity_m3_per_mwh"].astype(float)
    return base, capture, air_base, air_capture, factor


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


def modelled_industry_withdrawal(paths: ProjectPaths) -> dict[str, float]:
    """Withdrawal of the industrial point sources the model decides, 亿 m3/yr, by basin.

    Those sources sit inside the 公报's 工业用水 figure, so once industry becomes a decision
    agent its water would be charged twice: once in `basin_reserved_withdrawal`, once by the
    model re-spending it. This carve-out is what makes the two consistent.

    Returns an empty mapping when `inputs/industry_sources.csv` has not been built, which is the
    coal-only configuration and needs no carve-out.
    """
    path = paths.inputs_dir / "industry_sources.csv"
    if not path.exists():
        logger.info("industry_sources.csv absent; no industrial carve-out from the reservation")
        return {}
    from .water import _assign_basin_codes

    sources = pd.read_csv(path)
    sources["basin_code"] = _assign_basin_codes(paths, sources)
    by_basin = sources.groupby("basin_code")["water_m3_per_year"].sum() / 1e8
    return {str(code): float(value) for code, value in by_basin.items()}


def write_basin_caps(paths: ProjectPaths, kind: WeightKind = "demand") -> pd.DataFrame:
    """Write `inputs/water_basin_caps.csv`: the institutional half of the water constraint.

    One row per (basin, planning year). `residual_m3_per_year` is what the model's own sources
    -- coal hubs, and industrial point sources once they enter -- may withdraw between them:

        residual = 用水总量控制指标 - (生活 + 农业 + 生态 + 非电工业) + 已建模工业的现状取水

    Held flat after 2030: 国办发〔2013〕2号 sets no later target and extrapolating it would be
    inventing policy.

    OVER-SUBSCRIBED BASINS. The Northwest already withdraws more (729亿 m3 in 2025) than its
    apportioned 2030 cap (641亿), so `cap - reserved` is NEGATIVE there. Writing that through
    would make the constraint unsatisfiable no matter what the model does, and since the basin
    slack is penalised at the same big-M as the node slack, an ~85亿 m3 permanent violation would
    dominate the objective and drive decisions everywhere else in the country for a reason that
    has nothing to do with them.

    Neither is clipping at zero honest on its own -- it would silently delete the finding. What
    the red-line policy actually implies is that an over-cap basin must shrink, and that everyone
    in it shrinks, not just power. So the reservation is scaled PRO RATA by `cap / actual`
    wherever `actual > cap`, which is equivalent to holding every user to the same compliance
    ratio. The unscaled figure survives in `residual_uncapped_1e8_m3` and the ratio in
    `compliance_ratio`, so the over-subscription is reported rather than hidden.
    """
    from ..constants import PLANNING_YEARS

    caps_by_year = {year: basin_caps(paths, year, kind) for year in PLANNING_YEARS}
    reserved = basin_reserved_withdrawal()
    carve_out = modelled_industry_withdrawal(paths)
    rows = []
    for year in PLANNING_YEARS:
        for code in sorted(BASIN_NAMES_ZH):
            cap = caps_by_year[year][code]
            actual = BASIN_WITHDRAWAL_2025_1E8_M3[code]["total"]
            industry = carve_out.get(code, 0.0)
            uncapped = cap - reserved[code] + industry
            ratio = min(1.0, cap / actual) if actual > 0 else 1.0
            residual = cap - reserved[code] * ratio + industry * ratio
            if ratio < 1.0:
                logger.warning(
                    "basin %s (%s) withdraws %.0f against a cap of %.0f 1e8 m3; reservation "
                    "scaled pro rata by %.3f, raw residual %.1f -> %.1f",
                    code, BASIN_NAMES_ZH[code], actual, cap, ratio, uncapped, residual,
                )
            rows.append({
                "basin_code": code,
                "basin_name": BASIN_NAMES_ZH[code],
                "planning_year": year,
                "cap_1e8_m3": round(cap, 3),
                "actual_2025_1e8_m3": round(actual, 3),
                "reserved_1e8_m3": round(reserved[code], 3),
                "modelled_industry_1e8_m3": round(industry, 3),
                "compliance_ratio": round(ratio, 4),
                "residual_uncapped_1e8_m3": round(uncapped, 3),
                "residual_1e8_m3": round(residual, 3),
                "residual_m3_per_year": residual * 1e8,
                "apportionment_weight": kind,
                "source": "国办发〔2013〕2号 附件1 + 2025年中国水资源公报 表9",
                "basis": "withdrawal (用水量, 新水取用量)",
            })
    frame = pd.DataFrame(rows)
    destination = paths.inputs_dir / "water_basin_caps.csv"
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    logger.info("wrote %d rows to %s", len(frame), destination)
    return frame
