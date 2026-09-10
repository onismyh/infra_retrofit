"""Unit-resolved plant data for the Extended Data figures.

WHY THIS MODULE EXISTS. The v6 Extended Data set was drawn almost entirely at PROVINCE
resolution from the `BASE` scenario, and province resolution is the wrong unit for this paper
twice over. First, the constraint is a BASIN constraint: provinces straddle basin divides, so a
provincial mean averages over the very boundary the study is about. Second, a choropleth encodes
by AREA, and area is close to inversely related to where the coal is -- Tibet and Taiwan carried
the most conspicuous colour on the old fleet-age map while holding no fleet at all, because
`plot_extended.ed_fig5_fleet_age_map` filled unmatched provinces with a literal 2040 and drew it
indistinguishably from measured values.

The underlying data is far richer than that. `inputs/plants.csv` carries 350 sites, 3,623
generating units and 1,416 GW, with each site's capacity split across once-through (465 GW),
recirculating (703 GW) and air (248 GW) cooling, a mean commission year per site, coastal
distance and a seawater flag, and four separate water intensities on three accounting bases.
`results/<scenario>/plant_detail.csv` then carries, for every site and planning year, the
realised pathway shares, generation, water use, capture, biomass draw, blend levels and distance
to the nearest storage sink. None of that was reaching the appendix.

This module assembles it once, joins each site to its level-1 basin through the water supply
network, and hands the figure scripts a single tidy frame. Every figure that follows is drawn on
sites, units and capacity rather than on provincial polygons.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"

# Level-1 basin names. J hosts no coal capacity and is kept only so a missing join is visible
# rather than silently dropped.
BASIN_NAMES = {
    "A": "Northeast Rivers", "B": "Liao River", "C": "Hai River", "D": "Yellow River",
    "E": "Huai River", "F": "Yangtze River", "G": "Southeast Rivers", "H": "Pearl River",
    "J": "Southwest Rivers", "K": "Northwest Interior",
}

# The four basins whose dry-season allowance the standing fleet already exceeds at s = 0.85.
CONSTRAINED_BASINS = ("C", "D", "E", "K")


@lru_cache(maxsize=1)
def unit_to_site() -> pd.DataFrame:
    """Every generating unit, tagged with the hub it was clustered into.

    `inputs/plants_unit.csv` keeps `plant_site` as the plant's real name -- the H350 hub label
    exists only inside `builders.plants.write_plants` and is never written back, so the
    unit-to-hub map is not on disk. The clustering is deterministic (agglomerative, average
    linkage, precomputed haversine, fixed k), so re-running it reproduces the labels exactly;
    the assertions below fail loudly if that ever stops being true.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from coal_retrofit.builders.plants import _cluster_plants_to_hubs

    units = pd.read_csv(INPUTS / "plants_unit.csv", encoding="utf-8-sig")
    clustered = _cluster_plants_to_hubs(units).copy()
    sites = pd.read_csv(INPUTS / "plants.csv", encoding="utf-8-sig")[
        ["plant_id", "plant_site", "unit_count", "total_capacity_mw"]
    ]
    counts = clustered.groupby("plant_site").agg(
        n=("capacity_mw", "size"), mw=("capacity_mw", "sum")).reset_index()
    check = sites.merge(counts, on="plant_site", how="left")
    if check["n"].isna().any() or not np.allclose(check["n"], check["unit_count"]):
        raise RuntimeError("re-clustering no longer reproduces inputs/plants.csv hub labels")
    if not np.allclose(check["mw"], check["total_capacity_mw"]):
        raise RuntimeError("re-clustered hub capacity does not match inputs/plants.csv")
    return clustered.merge(sites[["plant_id", "plant_site"]], on="plant_site", how="left")


@lru_cache(maxsize=1)
def capacity_mode_labels() -> pd.DataFrame:
    """Province and cooling technology of each site by CAPACITY, not by unit count.

    `builders.plants.most_common_string` takes a plain mode over units, so a hub of many small
    units outvotes the big ones that carry the capacity. Measured over the 350 sites that
    rebooks 87.9 GW into a province it is not in -- including the largest hub in the model,
    27.9 GW standing on Shanghai's coordinates and booked to Jiangsu -- and mislabels the
    cooling technology of 262.7 GW. The physics is unaffected (the capacity split columns and
    the intensities derived from them are exact); only the labels move, so these are the labels
    the figures group and colour by.
    """
    units = unit_to_site()
    out = []
    for column, name in (("province", "province_by_capacity"),
                         ("cooling_technology", "cooling_by_capacity")):
        totals = units.groupby(["plant_id", column])["capacity_mw"].sum().reset_index()
        winner = (totals.sort_values("capacity_mw", ascending=False)
                  .drop_duplicates("plant_id")[["plant_id", column]]
                  .rename(columns={column: name}))
        out.append(winner.set_index("plant_id"))
    return pd.concat(out, axis=1).reset_index()


def plant_basin() -> pd.DataFrame:
    """Assign each plant site to one level-1 basin, with the ambiguity made explicit.

    A plant inside the 200 km supply radius can reach several basins -- 25% of the 1,569 links
    cross a basin divide -- so "the plant's basin" is a choice, not a fact. The rule here is the
    NEAREST reachable water node, which is the only assignment that does not depend on the
    solution. `basin_share` records how much of that plant's reachable node set sits in the
    assigned basin, so a figure can show where the assignment is weak instead of hiding it.
    """
    links = pd.read_csv(INPUTS / "water_supply_links.csv")
    nodes = pd.read_csv(INPUTS / "water_nodes.csv").set_index("water_node_id")["basin_code"]
    links["basin_code"] = links["water_node_id"].map(nodes)
    links = links.dropna(subset=["basin_code"])

    nearest = (links.sort_values("distance_km")
               .groupby("plant_id", as_index=False)
               .first()[["plant_id", "basin_code", "distance_km"]]
               .rename(columns={"distance_km": "nearest_node_km"}))

    share = (links.groupby(["plant_id", "basin_code"]).size()
             .rename("n").reset_index())
    total = share.groupby("plant_id")["n"].transform("sum")
    share["frac"] = share["n"] / total
    nearest = nearest.merge(
        share.rename(columns={"basin_code": "_b"}),
        left_on=["plant_id", "basin_code"], right_on=["plant_id", "_b"], how="left")
    nearest["basin_share"] = nearest["frac"].fillna(1.0)
    nearest["basin_name"] = nearest["basin_code"].map(BASIN_NAMES)
    nearest["constrained"] = nearest["basin_code"].isin(CONSTRAINED_BASINS)
    return nearest[["plant_id", "basin_code", "basin_name", "constrained",
                    "nearest_node_km", "basin_share"]]


def fleet() -> pd.DataFrame:
    """Static, unit-resolved site attributes joined to basin.

    Returns one row per site with unit count, capacity split by cooling technology, mean
    commission year, coastal geometry and the three water-accounting intensities.
    """
    p = pd.read_csv(INPUTS / "plants.csv")
    p.columns = [c.lstrip("﻿") for c in p.columns]
    keep = ["plant_id", "plant_site", "unit_count", "total_capacity_mw", "mean_commission_year",
            "retirement_year", "province_mode", "dominant_combustion",
            "dominant_cooling_technology", "capacity_mw_once_through",
            "capacity_mw_recirculating", "capacity_mw_air", "already_air_share",
            "centroid_latitude", "centroid_longitude", "coast_distance_km", "seawater_cooled",
            "weighted_water_intensity_m3_per_mwh", "quota_intensity_m3_per_mwh",
            "consumption_intensity_m3_per_mwh", "withdrawal_intensity_m3_per_mwh"]
    p = p[[c for c in keep if c in p.columns]].copy()
    p["capacity_gw"] = p["total_capacity_mw"] / 1e3
    p["mean_unit_mw"] = p["total_capacity_mw"] / p["unit_count"].clip(lower=1)
    for tech in ("once_through", "recirculating", "air"):
        p[f"share_{tech}"] = (p[f"capacity_mw_{tech}"] / p["total_capacity_mw"]).fillna(0.0)
    p = p.merge(plant_basin(), on="plant_id", how="left")
    return p.merge(capacity_mode_labels(), on="plant_id", how="left")


def outcomes(scenario: str, year: int | None = None) -> pd.DataFrame:
    """Realised per-site outcomes for one scenario, joined to the static fleet.

    `share_*` columns are STOCKS and monotone in year -- they are shares of the site's capacity
    already committed to a pathway, not annual increments. They must never be accumulated across
    years, which is the single most common way this table has been misread.
    """
    path = RESULTS / scenario / "plant_detail.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} — solve {scenario} first")
    d = pd.read_csv(path)
    if year is not None:
        d = d[d["year"] == year].copy()
    base = fleet().drop(columns=["retirement_year", "centroid_latitude",
                                 "centroid_longitude"], errors="ignore")
    out = d.merge(base, on="plant_id", how="left", suffixes=("", "_static"))

    # Dry cooling actually OPERATING this year, net of the units that were built dry. The
    # distinction matters: `air_cooled_share` is an operating fraction, and by 2060 a large part
    # of the installed dry stock has reverted to wet operation.
    already = out.get("already_air_share_static", out.get("already_air_share", 0.0))
    out["converted_share"] = (out["air_cooled_share"].fillna(0.0)
                              * (1.0 - pd.Series(already).fillna(0.0).to_numpy()))
    out["converted_gw"] = out["converted_share"] * out["capacity_mw"] / 1e3
    out["surviving_gw"] = (1.0 - out["share_retire"].fillna(0.0)) * out["capacity_mw"] / 1e3
    out["retired_gw"] = out["share_retire"].fillna(0.0) * out["capacity_mw"] / 1e3
    gen = out["annual_generation_mwh"].replace(0.0, np.nan)
    out["water_m3_per_mwh"] = out["water_use_m3"] / gen
    out["age_2030"] = 2030 - out["mean_commission_year"]
    return out


def pair(control: str, treatment: str, year: int) -> pd.DataFrame:
    """Per-site response to making the water reservation bind.

    The whole paper is a difference between two runs, so the appendix should be able to show
    that difference at the unit where the decision is taken. Columns suffixed `_c` and `_t` are
    control and treatment; `d_*` are treatment minus control.
    """
    c = outcomes(control, year).set_index("plant_id")
    t = outcomes(treatment, year).set_index("plant_id")
    cols = ["converted_gw", "retired_gw", "captured_mt", "biomass_use_gj", "water_use_m3",
            "surviving_gw", "annual_generation_mwh"]
    out = c[["province_name", "capacity_mw", "basin_code", "basin_name", "constrained",
             "centroid_longitude", "centroid_latitude", "unit_count", "mean_commission_year",
             "min_distance_to_storage_km", "dominant_cooling", "share_once_through",
             "share_recirculating", "share_air", "seawater_cooled"]].copy()
    for col in cols:
        out[f"{col}_c"] = c[col]
        out[f"{col}_t"] = t[col]
        out[f"d_{col}"] = t[col] - c[col]
    return out.reset_index()


__all__ = ["BASIN_NAMES", "CONSTRAINED_BASINS", "plant_basin", "fleet", "outcomes", "pair"]
