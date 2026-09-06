"""Build the industrial point-source inventory from the plant-level source library.

Source: `D:\\6. Transfer\\PhD_tht\\point_source\\` — six workbooks covering seven sectors,
each with WGS84 coordinates, capacity, output, CO2 and (except EAF) a hydrogen-demand column.

THE UNITS IN THAT LIBRARY ARE NOT UNIFORM, and getting them wrong silently rescales the
whole industrial sector. Verified against national totals before writing this module:

    steel     production in kt      CO2 in Mt   H2_DMD in kt    (939 Mt crude steel)
    others    production in 万吨    CO2 in Mt   H2_DMD in 万吨  (52 Mt NH3, 678 Mt crude,
                                                                 1 269 Mt clinker)
    cement    capacity in t/d (not 万吨) — 9 000 t/d x 300 d = 2.7 Mt/yr matches its own
              production column, which is the check that pinned this down.

Everything is normalised to kt/yr for capacity and output, Mt/yr for CO2, kt/yr for hydrogen.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from ..artifacts import write_csv
from ..constants import PLANT_YEAR_BASIS
from ..constants_industry import (
    ASSET_LIFETIME_YEARS,
    DEFAULT_SECTOR_HUB_COUNTS,
    INDUSTRY_CONSUMPTION_SHARE,
    SECTOR_AMMONIA,
    SECTOR_CEMENT,
    SECTOR_COAL_CHEM,
    SECTOR_HAS_H2_ROUTE,
    SECTOR_LABELS_ZH,
    SECTOR_METHANOL,
    SECTOR_REFINERY,
    SECTOR_STEEL_BF,
    SECTOR_STEEL_EAF,
    SECTORS_WITH_UNIFORM_RETIREMENT,
    water_quota,
)
from ..paths import ProjectPaths
# Reuse the coal fleet's distance matrix rather than writing a second one: the two hub
# families must be clustered on identical geometry or their hub sizes are not comparable.
from .plants import _haversine_distance_matrix

logger = logging.getLogger(__name__)

POINT_SOURCE_DIR = Path(r"D:\6. Transfer\PhD_tht\point_source")

WAN_TO_KT = 10.0        # 万吨 -> kt
CEMENT_KILN_DAYS = 300  # t/d -> kt/yr, the operating-day convention implied by the source

CANONICAL_COLUMNS = [
    "source_id", "sector", "sector_zh", "plant_name", "province",
    "longitude", "latitude", "capacity_kt_per_year", "production_kt_per_year",
    "co2_mt_per_year", "process_co2_mt_per_year", "h2_demand_kt_per_year",
    "has_h2_route", "feedstock", "commission_year", "commission_year_observed",
    "asset_life_years", "water_intensity_m3_per_t", "water_m3_per_year", "status",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numeric view of a column, or zeros if the column is absent."""
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _text(frame: pd.DataFrame, column: str, default: str) -> pd.Series:
    """String view of a column, or a constant series if the column is absent.

    `DataFrame.get(col, default)` returns the bare default, not a Series, so it cannot be
    used here — that mismatch is silent until `.astype` is called on it.
    """
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].astype(str).fillna(default)


def _year(frame: pd.DataFrame, column: str) -> pd.Series:
    """Commissioning year as a nullable int; 'unknown' and blanks become NA."""
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Int64")
    raw = pd.to_numeric(frame[column], errors="coerce")
    raw = raw.where((raw >= 1900) & (raw <= 2100))
    return raw.astype("Int64")


def _read(sheet_file: str, sheet: str) -> pd.DataFrame:
    path = POINT_SOURCE_DIR / sheet_file
    if not path.exists():
        raise FileNotFoundError(f"point-source workbook not found: {path}")
    return pd.read_excel(path, sheet_name=sheet)


def _load_steel() -> pd.DataFrame:
    """Blast-furnace/BOF and EAF units. Two sheets, deliberately kept as two sectors.

    They are different technologies with different water quotas, different emission factors
    (1.8 vs 0.4 tCO2/t crude steel) and only one hydrogen route between them, so merging them
    into a single "steel" sector would make every downstream per-tonne number a blend of two
    incomparable things.
    """
    frames = []
    bof = _read("steel_all.xlsx", "China_ope_cons_BOF")
    frames.append(pd.DataFrame({
        "source_id": "BOF_" + bof["ID"].astype(str),
        "sector": SECTOR_STEEL_BF,
        # `Name` carries only 60 distinct values across 705 furnace rows and 237 distinct
        # coordinates, so it cannot identify a site; `GEM Plant ID` can, and matches the key
        # the EAF sheet uses.
        "plant_name": _text(bof, "GEM Plant ID", "unknown"),
        "province": bof["Province"].astype(str),
        "longitude": _num(bof, "lon"), "latitude": _num(bof, "lat"),
        "capacity_kt_per_year": _num(bof, "Current Capacity (ttpa)"),
        "production_kt_per_year": _num(bof, "BOF steel production"),
        "co2_mt_per_year": _num(bof, "CO2_Emi"),
        "process_co2_mt_per_year": 0.0,
        "h2_demand_kt_per_year": _num(bof, "H2_DMD"),
        "feedstock": "default",
        "commission_year": _year(bof, "Start Date"),
        "status": _text(bof, "Unit Status", "operating"),
    }))
    eaf = _read("steel_all.xlsx", "China_ope_cons_EAF")
    frames.append(pd.DataFrame({
        "source_id": "EAF_" + eaf["GEM Unit ID"].astype(str),
        "sector": SECTOR_STEEL_EAF,
        "plant_name": eaf["GEM Plant ID"].astype(str),
        "province": eaf["Province"].astype(str),
        "longitude": _num(eaf, "lon"), "latitude": _num(eaf, "lat"),
        "capacity_kt_per_year": _num(eaf, "Current Capacity (ttpa)"),
        "production_kt_per_year": _num(eaf, "EAF steel production"),
        "co2_mt_per_year": _num(eaf, "CO2_Emi"),
        "process_co2_mt_per_year": 0.0,
        # EAF has no H2_DMD column in the source, which matches SECTOR_HAS_H2_ROUTE: an
        # electric arc furnace running on scrap has no fossil reductant to displace.
        "h2_demand_kt_per_year": 0.0,
        "feedstock": "default",
        "commission_year": _year(eaf, "Start Date"),
        "status": _text(eaf, "Unit Status", "operating"),
    }))
    return pd.concat(frames, ignore_index=True)


def _load_cement() -> pd.DataFrame:
    df = _read("Cement_all.xlsx", "水泥厂")
    return pd.DataFrame({
        "source_id": "CEM_" + df["ID"].astype(str),
        "sector": SECTOR_CEMENT,
        "plant_name": df["Name"].astype(str),
        "province": df["Province"].astype(str),
        "longitude": _num(df, "lon"), "latitude": _num(df, "lat"),
        # t/d -> kt/yr. The only sector whose capacity is a daily rate.
        "capacity_kt_per_year": _num(df, "Clinker Capacity") * CEMENT_KILN_DAYS / 1000.0,
        "production_kt_per_year": _num(df, "Clinker Production") * WAN_TO_KT,
        "co2_mt_per_year": _num(df, "CO2 Emission"),
        # Calcination share of the 0.839 tCO2/t clinker factor. Kept explicit because it is
        # the part no fuel switch can touch (see constants_industry.SECTOR_HAS_H2_ROUTE).
        "process_co2_mt_per_year": _num(df, "CO2 Emission") * 0.63,
        "h2_demand_kt_per_year": 0.0,   # no hydrogen route; source column is fuel-heat only
        "feedstock": "default",
        "commission_year": _year(df, "Time"),
        "status": "operating",
    })


def _load_ammonia_methanol() -> pd.DataFrame:
    frames = []
    for sheet, sector, prefix in (("合成氨厂", SECTOR_AMMONIA, "NH3"),
                                  ("甲醇厂", SECTOR_METHANOL, "MEOH")):
        df = _read("Ammonia_Methanol.xlsx", sheet)
        frames.append(pd.DataFrame({
            "source_id": f"{prefix}_" + df["ID"].astype(str),
            "sector": sector,
            "plant_name": df["Company"].astype(str),
            "province": df["Province"].astype(str),
            "longitude": _num(df, "lon"), "latitude": _num(df, "lat"),
            "capacity_kt_per_year": _num(df, "Capacity") * WAN_TO_KT,
            "production_kt_per_year": _num(df, "Production") * WAN_TO_KT,
            "co2_mt_per_year": _num(df, "Emission"),
            "process_co2_mt_per_year": 0.0,
            "h2_demand_kt_per_year": _num(df, "H2_DMD") * WAN_TO_KT,
            "feedstock": _text(df, "Feedstock", "Coal"),
            "commission_year": pd.Series(pd.NA, index=df.index, dtype="Int64"),
            "status": "operating",
        }))
    return pd.concat(frames, ignore_index=True)


def _load_refinery() -> pd.DataFrame:
    df = _read("Refinery_all.xlsx", "China")
    return pd.DataFrame({
        "source_id": "REF_" + df["ID"].astype(str),
        "sector": SECTOR_REFINERY,
        "plant_name": df["Name"].astype(str),
        "province": df["Province"].astype(str),
        "longitude": _num(df, "lon"), "latitude": _num(df, "lat"),
        "capacity_kt_per_year": _num(df, "Crude cap") * WAN_TO_KT,
        "production_kt_per_year": _num(df, "Crude input") * WAN_TO_KT,
        "co2_mt_per_year": _num(df, "Emission"),
        "process_co2_mt_per_year": 0.0,
        "h2_demand_kt_per_year": _num(df, "H2_dmd") * WAN_TO_KT,
        "feedstock": "default",
        "commission_year": pd.Series(pd.NA, index=df.index, dtype="Int64"),
        "status": "operating",
    })


def _load_coal_chemical() -> pd.DataFrame:
    """Modern coal chemicals. Read the four process sheets, not the `All` summary.

    `All` drops capacity and output, which the water calculation needs; the process sheets
    also split process from fuel CO2, which matters because only the fuel part is displaced
    by a hydrogen blend.
    """
    frames = []
    for sheet in ("煤制天然气", "煤制烯烃", "煤制油", "煤制乙二醇"):
        df = _read("Coal-to-chemicals.xlsx", sheet)
        frames.append(pd.DataFrame({
            "source_id": f"CTC_{sheet}_" + df["ID"].astype(str),
            "sector": SECTOR_COAL_CHEM,
            "plant_name": df["Project"].astype(str),
            "province": df["Province"].astype(str),
            "longitude": _num(df, "Longtitude"), "latitude": _num(df, "Latitude"),
            "capacity_kt_per_year": _num(df, "Capacity") * WAN_TO_KT,
            "production_kt_per_year": _num(df, "Production") * WAN_TO_KT,
            "co2_mt_per_year": _num(df, "Total Direct Emission"),
            "process_co2_mt_per_year": _num(df, "CO2 Emission-Process"),
            "h2_demand_kt_per_year": _num(df, "H2_DMD") * WAN_TO_KT,
            "feedstock": _text(df, "Feedstock", "Coal"),
            "commission_year": pd.Series(pd.NA, index=df.index, dtype="Int64"),
            "status": "operating",
        }))
    return pd.concat(frames, ignore_index=True)


def _assign_uniform_ages(frame: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """Fill missing commissioning years by spreading ages uniformly over the asset life.

    Sectors without observed years (ammonia, methanol, refinery, coal chemicals) get ages
    spread evenly across [0, life] by a deterministic quantile ladder, ordered by source_id.
    This is the maximum-entropy assumption given no age data, and it is deterministic rather
    than sampled so two runs are bit-identical (`.claude/rules/experiment-reproducibility.md`).

    `commission_year_observed` stays False for every row filled this way, so any figure that
    reads plant age can exclude them instead of quietly treating a synthetic year as data.
    """
    frame = frame.copy()
    frame["commission_year_observed"] = frame["commission_year"].notna()
    for sector, group in frame.groupby("sector", sort=True):
        life = ASSET_LIFETIME_YEARS[str(sector)]
        missing = group.index[group["commission_year"].isna()]
        if len(missing) == 0:
            continue
        # Sectors in SECTORS_WITH_UNIFORM_RETIREMENT have NO observed years at all; steel and
        # cement are partially observed and only their gaps are filled. Both cases use the
        # same flat uniform ladder, as instructed — filling a partially observed sector from
        # its own observed age distribution instead would use more information, but it is a
        # different assumption and is not made here without being asked for.
        logger.info("%s: filling %d/%d commissioning years uniformly over %d-year life",
                    sector, len(missing), len(group), life)
        ordered = frame.loc[missing].sort_values("source_id").index
        # i/(n-1) over [0, 1] -> ages 0..life, so the oldest cohort is exactly at end of life.
        fractions = np.linspace(0.0, 1.0, num=len(ordered)) if len(ordered) > 1 else np.array([0.5])
        frame.loc[ordered, "commission_year"] = (
            base_year - np.rint(fractions * life)
        ).astype("int64")
    frame["commission_year"] = frame["commission_year"].astype("Int64")
    frame["asset_life_years"] = frame["sector"].map(ASSET_LIFETIME_YEARS).astype(int)
    return frame


def _apply_water(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the GB/T 18916 unit-product water intake and the resulting annual volume."""
    frame = frame.copy()
    intensities = []
    for sector, feedstock in zip(frame["sector"], frame["feedstock"], strict=True):
        try:
            intensities.append(water_quota(str(sector), str(feedstock)))
        except ValueError:
            intensities.append(np.nan)
    frame["water_intensity_m3_per_t"] = intensities
    # production is kt/yr; kt -> t is x1000.
    frame["water_m3_per_year"] = (
        frame["water_intensity_m3_per_t"] * frame["production_kt_per_year"] * 1000.0
        * INDUSTRY_CONSUMPTION_SHARE
    )
    unsourced = sorted(frame.loc[frame["water_intensity_m3_per_t"].isna(), "sector"].unique())
    if unsourced:
        logger.warning(
            "no water intake quota sourced for %s — %d plants carry NaN water use. "
            "Fill WATER_INTAKE_QUOTA_M3_PER_T from GB/T 18916 before solving.",
            unsourced, int(frame["water_intensity_m3_per_t"].isna().sum()),
        )
    return frame


def build_industry_sources(base_year: int = int(PLANT_YEAR_BASIS)) -> pd.DataFrame:
    """Normalised plant-level industrial point sources across all seven sectors."""
    frame = pd.concat(
        [_load_steel(), _load_cement(), _load_ammonia_methanol(),
         _load_refinery(), _load_coal_chemical()],
        ignore_index=True,
    )
    before = len(frame)
    frame = frame[
        frame["latitude"].between(3.0, 54.0) & frame["longitude"].between(73.0, 136.0)
    ].reset_index(drop=True)
    if len(frame) < before:
        logger.warning("dropped %d rows with missing or out-of-China coordinates",
                       before - len(frame))
    frame["sector_zh"] = frame["sector"].map(SECTOR_LABELS_ZH)
    frame["has_h2_route"] = frame["sector"].map(SECTOR_HAS_H2_ROUTE)
    frame.loc[~frame["has_h2_route"], "h2_demand_kt_per_year"] = 0.0
    frame = _assign_uniform_ages(frame, base_year)
    frame = _apply_water(frame)
    return frame[CANONICAL_COLUMNS]


def cluster_sector_hubs(
    sources: pd.DataFrame,
    hub_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Cluster point sources into hubs SEPARATELY WITHIN EACH SECTOR.

    Same algorithm as the coal fleet (`builders/plants._cluster_plants_to_hubs`):
    agglomerative, average linkage, precomputed haversine distances. Clustering per sector
    keeps a hub technologically homogeneous — a hub that mixed a cement kiln with an ammonia
    plant would have no meaningful water intensity, hydrogen basis or capture cost.
    """
    counts = dict(DEFAULT_SECTOR_HUB_COUNTS if hub_counts is None else hub_counts)
    labelled = []
    for sector, group in sources.groupby("sector", sort=True):
        n_hubs = min(int(counts.get(str(sector), 30)), len(group))
        group = group.copy()
        if len(group) <= n_hubs:
            group["hub_id"] = [f"{sector}_{i:03d}" for i in range(len(group))]
        else:
            distances = _haversine_distance_matrix(
                group["longitude"].to_numpy(), group["latitude"].to_numpy()
            )
            labels = AgglomerativeClustering(
                n_clusters=n_hubs, metric="precomputed", linkage="average"
            ).fit_predict(distances)
            group["hub_id"] = [f"{sector}_{int(x):03d}" for x in labels]
        labelled.append(group)
    return pd.concat(labelled, ignore_index=True)


def aggregate_hubs(labelled: pd.DataFrame) -> pd.DataFrame:
    """Collapse labelled point sources to hub rows, output-weighting the intensities."""
    rows = []
    for hub_id, group in labelled.groupby("hub_id", sort=True):
        output = float(group["production_kt_per_year"].sum())
        weights = group["production_kt_per_year"].to_numpy(dtype=float)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            weights = np.ones(len(group))
            weight_sum = float(len(group))
        intensity = group["water_intensity_m3_per_t"].to_numpy(dtype=float)
        finite = np.isfinite(intensity)
        rows.append({
            "hub_id": str(hub_id),
            "sector": str(group["sector"].iloc[0]),
            "sector_zh": str(group["sector_zh"].iloc[0]),
            "n_plants": int(len(group)),
            "province": group["province"].mode().iat[0] if not group["province"].empty else "",
            "longitude": float(np.average(group["longitude"], weights=weights)),
            "latitude": float(np.average(group["latitude"], weights=weights)),
            "capacity_kt_per_year": float(group["capacity_kt_per_year"].sum()),
            "production_kt_per_year": output,
            "co2_mt_per_year": float(group["co2_mt_per_year"].sum()),
            "process_co2_mt_per_year": float(group["process_co2_mt_per_year"].sum()),
            "h2_demand_kt_per_year": float(group["h2_demand_kt_per_year"].sum()),
            "has_h2_route": bool(group["has_h2_route"].iloc[0]),
            "water_intensity_m3_per_t": (
                float(np.average(intensity[finite], weights=weights[finite]))
                if finite.any() else np.nan
            ),
            "water_m3_per_year": float(group["water_m3_per_year"].sum(min_count=1)),
            "mean_commission_year": float(group["commission_year"].astype(float).mean()),
            "share_year_observed": float(group["commission_year_observed"].mean()),
        })
    return pd.DataFrame(rows)


def write_industry_inputs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write `industry_sources.csv` and `industry_hubs.csv` into `inputs/`."""
    paths.ensure_inputs_dir()
    sources = build_industry_sources()
    labelled = cluster_sector_hubs(sources)
    hubs = aggregate_hubs(labelled)
    write_csv(labelled, paths.inputs_dir / "industry_sources.csv")
    write_csv(hubs, paths.inputs_dir / "industry_hubs.csv")
    return labelled, hubs
