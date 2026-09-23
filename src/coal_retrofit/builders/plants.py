from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from ..artifacts import write_csv
from ..constants import (
    CHINA_WATER_QUOTA_M3_PER_MWH,
    COMBUSTION_CLASS_MAP,
    COOLING_WATER_INTENSITY_M3_PER_MWH,
    PLANT_YEAR_BASIS,
    WATER_INTENSITY_BY_TECH_M3_PER_MWH,
    quota_capacity_band,
)
from ..paths import ProjectPaths

# Default number of spatial hubs
DEFAULT_N_HUBS = 350


COLUMN_MAP = {
    "unit_id": ["unit ID", "Unit ID", "unit_id"],
    "plant_site": ["Plant name (local)", "Plant name", "plant_site"],
    "capacity_mw": ["Capacity (MW)", "Capacity", "capacity_mw"],
    "commission_year": ["Year", "Commission Year", "commission_year"],
    "combustion": ["Combustion", "combustion"],
    "cooling_technology": ["Cooling Technology", "Cooling", "cooling_technology"],
    "province": ["Province", "province"],
    "latitude": ["Latitude", "lat", "latitude"],
    "longitude": ["Longitude", "lon", "longitude"],
    "status": ["Status", "status"],
}


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing required column. Tried: {candidates}")


def build_plants_unit_dataframe(paths: ProjectPaths) -> pd.DataFrame:
    source_file = paths.data_dir / "GEM_with_Cooling_Technology_July2025.xlsx"
    df = pd.read_excel(source_file)

    col_unit_id = pick_column(df, COLUMN_MAP["unit_id"])
    col_plant_site = pick_column(df, COLUMN_MAP["plant_site"])
    col_capacity = pick_column(df, COLUMN_MAP["capacity_mw"])
    col_year = pick_column(df, COLUMN_MAP["commission_year"])
    col_combustion = pick_column(df, COLUMN_MAP["combustion"])
    col_cooling = pick_column(df, COLUMN_MAP["cooling_technology"])
    col_province = pick_column(df, COLUMN_MAP["province"])
    col_lat = pick_column(df, COLUMN_MAP["latitude"])
    col_lon = pick_column(df, COLUMN_MAP["longitude"])
    col_status = pick_column(df, COLUMN_MAP["status"])

    status = df[col_status].astype(str).str.strip().str.lower()
    # Include operating and under-construction units only (the "existing fleet"
    # literature convention). Announced/permitted/pre-permit units (~257 GW in
    # GEM Jul-2025) are excluded so the baseline matches the operating fleet.
    valid_statuses = ["operating", "construction"]
    df = df.loc[status.isin(valid_statuses)].copy()

    out = pd.DataFrame(
        {
            "unit_id": df[col_unit_id],
            "plant_site": df[col_plant_site],
            "capacity_mw": pd.to_numeric(df[col_capacity], errors="coerce"),
            "commission_year": pd.to_numeric(df[col_year], errors="coerce").astype("Int64"),
            "combustion": df[col_combustion],
            "cooling_technology": df[col_cooling],
            "province": df[col_province],
            "latitude": pd.to_numeric(df[col_lat], errors="coerce"),
            "longitude": pd.to_numeric(df[col_lon], errors="coerce"),
        }
    )
    out["source"] = source_file.name
    out["year_basis"] = PLANT_YEAR_BASIS
    return out[
        [
            "unit_id",
            "plant_site",
            "capacity_mw",
            "commission_year",
            "combustion",
            "cooling_technology",
            "province",
            "latitude",
            "longitude",
            "source",
            "year_basis",
        ]
    ]


def most_common_string(values: pd.Series) -> str:
    cleaned = [str(v) for v in values.dropna().tolist() if str(v)]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def mix_string(values: pd.Series, max_items: int = 4) -> str:
    cleaned = [str(v) for v in values.dropna().tolist() if str(v)]
    if not cleaned:
        return ""
    total = len(cleaned)
    return ";".join(f"{name}:{count / total:.3f}" for name, count in Counter(cleaned).most_common(max_items))


def build_plant_dataframe(
    plants: pd.DataFrame,
    source_label: str,
    year_basis: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    plants_run = plants.copy()
    plants_run["capacity_mw"] = pd.to_numeric(plants_run["capacity_mw"], errors="coerce").fillna(0.0)

    for plant_index, (plant_site, frame) in enumerate(
        plants_run.groupby("plant_site", sort=True), start=1
    ):
        plant_id = f"P{plant_index:04d}"
        centroid_lat = float(frame["latitude"].mean())
        centroid_lon = float(frame["longitude"].mean())

        cooling_tech = frame["cooling_technology"].astype(str).str.strip().str.lower()
        capacity = frame["capacity_mw"]
        total_cap = capacity.sum()

        cooling_capacities = {
            f"capacity_mw_{tech_key.replace('-', '_')}": 0.0
            for tech_key in COOLING_WATER_INTENSITY_M3_PER_MWH.keys()
        }
        weighted_intensity = 0.0
        has_water_cooling = False

        if total_cap > 0:
            for tech_key, intensity in COOLING_WATER_INTENSITY_M3_PER_MWH.items():
                tech_mask = cooling_tech.str.contains(tech_key, case=False, na=False)
                tech_capacity = capacity[tech_mask].sum()
                cooling_capacities[f"capacity_mw_{tech_key.replace('-', '_')}"] = float(tech_capacity)
                weighted_intensity += (tech_capacity / total_cap) * intensity
                if tech_key in ["once-through", "recirculating"] and tech_capacity > 0:
                    has_water_cooling = True

        mean_commission = float(frame["commission_year"].mean())
        retirement_year = int(mean_commission + 40)

        # Wang (2023) water table, looked up per unit by (steam cycle, cooling system) and
        # capacity-weighted to the hub. Two variants are produced because whether a coastal
        # once-through hub draws seawater is only known after its centroid is placed
        # (add_seawater_classification): `_all` counts every once-through unit against
        # freshwater, `_fresh` zeroes them.
        water_intensities = _hub_water_intensities(frame, total_cap)

        rows.append(
            {
                "plant_id": plant_id,
                "plant_index": plant_index,
                "plant_site": str(plant_site),
                "unit_count": int(len(frame)),
                "total_capacity_mw": float(total_cap),
                "mean_commission_year": mean_commission,
                "retirement_year": retirement_year,
                "province_mode": most_common_string(frame["province"]),
                "dominant_combustion": most_common_string(frame["combustion"]),
                "dominant_cooling_technology": most_common_string(frame["cooling_technology"]),
                "combustion_mix": mix_string(frame["combustion"]),
                **cooling_capacities,
                "weighted_water_intensity_m3_per_mwh": round(weighted_intensity, 4),
                **water_intensities,
                "requires_water_supply": has_water_cooling,
                "centroid_latitude": centroid_lat,
                "centroid_longitude": centroid_lon,
                "source": source_label,
                "year_basis": year_basis,
            }
        )

    return pd.DataFrame(rows).sort_values("plant_index").reset_index(drop=True)


WATER_KEYS = ("withdrawal", "consumption", "withdrawal_ccs", "consumption_ccs")


def _hub_water_intensities(units: pd.DataFrame, total_capacity: float) -> dict[str, float]:
    """Capacity-weighted water intensities for one hub, in both freshwater variants.

    Also returns the Chinese abstraction-quota intensity, which needs no freshwater variant:
    the quota already excludes condenser flow, so what remains (boiler make-up and service
    water, 0.35-0.72 m3/MWh for once-through) is drawn from the freshwater system whether the
    condenser is fed by a river or by the sea.
    """
    out = {f"{key}_intensity_{variant}": 0.0 for key in WATER_KEYS for variant in ("all", "fresh")}
    out["quota_intensity_m3_per_mwh"] = 0.0
    for basis in ("consumption", "consumption_ccs", "withdrawal", "withdrawal_ccs"):
        out[f"air_{basis}_intensity_m3_per_mwh"] = 0.0
    out["already_air_share"] = 0.0
    if total_capacity <= 0:
        return out
    combustion = units["combustion"].map(_combustion_class)
    cooling = units["cooling_technology"].map(_cooling_class)
    capacity = units["capacity_mw"].astype(float)
    for (comb, cool), index in units.groupby([combustion, cooling]).groups.items():
        entry = WATER_INTENSITY_BY_TECH_M3_PER_MWH.get((comb, cool))
        if entry is None:
            continue
        share = float(capacity.loc[index].sum()) / total_capacity
        for key in WATER_KEYS:
            out[f"{key}_intensity_all"] += share * entry[key]
            if cool != "once-through":
                out[f"{key}_intensity_fresh"] += share * entry[key]
    for cool, index in units.groupby(cooling).groups.items():
        unit_caps = capacity.loc[index]
        for unit_capacity in unit_caps:
            quota = CHINA_WATER_QUOTA_M3_PER_MWH.get((cool, quota_capacity_band(float(unit_capacity))))
            if quota is not None:
                out["quota_intensity_m3_per_mwh"] += (float(unit_capacity) / total_capacity) * quota
    # What this hub's intensity would be if its condensers were converted to dry cooling.
    # Same steam cycles, air rows of the same table, so the difference is attributable to
    # the cooling system alone. All four bases, weighted by each unit's OWN steam cycle --
    # deriving the withdrawal pair downstream as `air_consumption x ratio(dominant_combustion)`
    # is not the same number and put five already-dry hubs above their own base withdrawal.
    for comb, index in units.groupby(combustion).groups.items():
        entry = WATER_INTENSITY_BY_TECH_M3_PER_MWH.get((comb, "air"))
        if entry is None:
            continue
        share = float(capacity.loc[index].sum()) / total_capacity
        for basis in ("consumption", "consumption_ccs", "withdrawal", "withdrawal_ccs"):
            out[f"air_{basis}_intensity_m3_per_mwh"] += share * entry[basis]
    out["already_air_share"] = float(capacity[cooling == "air"].sum()) / total_capacity
    return {k: round(v, 4) for k, v in out.items()}


def finalize_water_intensities(plants: pd.DataFrame) -> pd.DataFrame:
    """Pick the freshwater variant for seawater-cooled hubs and drop the scratch columns.

    Seawater condensers draw no river water: on a withdrawal basis that is the difference
    between ~100 m3/MWh of imagined abstraction and none at all, for 241 GW of capacity.

    The with-capture quota is the base quota plus the capture unit's extra consumptive
    make-up water. The quota schedule predates CCS and has no row for it, but the increment
    it would meter is exactly the additional water the plant must buy in, which is the
    consumption increment from the Wang (2023) table.
    """
    plants = plants.copy()
    seawater = plants["seawater_cooled"].astype(bool)
    for key in WATER_KEYS:
        column = f"{key}_intensity_m3_per_mwh"
        plants[column] = plants[f"{key}_intensity_all"].where(~seawater, plants[f"{key}_intensity_fresh"])
    plants["quota_ccs_intensity_m3_per_mwh"] = (
        plants["quota_intensity_m3_per_mwh"]
        + (plants["consumption_ccs_intensity_m3_per_mwh"] - plants["consumption_intensity_m3_per_mwh"]).clip(lower=0.0)
    ).round(4)
    # Freshwater once-through condenser flow, isolated. `_intensity_all` sums every cooling
    # class and `_intensity_fresh` sums all but once-through, so the difference IS the
    # once-through term -- exactly, per unit, at each unit's own steam cycle. Seawater hubs get
    # zero because their published column already is the fresh variant.
    #
    # It is kept because it is the only part of the withdrawal column that has to be
    # recalibrated: the once-through rows of WATER_INTENSITY_BY_TECH_M3_PER_MWH are Macknick
    # (2011) US values and disagree with the 水资源公报 by a factor of two, while the
    # recirculating and air rows agree to 5-14%. See `builders/water_quota`. Reconstructing it
    # downstream from `dominant_combustion` x `capacity_mw_once_through` is NOT equivalent and
    # was wrong by 24% on the fleet total: the hub's dominant steam cycle is often not the
    # steam cycle of its once-through units.
    for key in ("withdrawal", "withdrawal_ccs"):
        plants[f"once_through_{key}_intensity_m3_per_mwh"] = (
            (plants[f"{key}_intensity_all"] - plants[f"{key}_intensity_fresh"])
            .where(~seawater, 0.0).clip(lower=0.0).round(4)
        )
    return plants.drop(columns=[f"{k}_intensity_{v}" for k in WATER_KEYS for v in ("all", "fresh")])


def _combustion_class(label: str) -> str:
    """Map a GEM combustion label onto the three steam-cycle classes of the water table."""
    text = str(label).strip().lower().split("/")[0]
    return COMBUSTION_CLASS_MAP.get(text, "subcritical")


def _cooling_class(label: str) -> str:
    text = str(label).strip().lower()
    if "once" in text:
        return "once-through"
    if "air" in text or "dry" in text:
        return "air"
    return "recirculating"


# Seawater classification.
#
# GEM records `once-through` without saying whether the condenser draws sea or river water,
# and the two behave completely differently in a freshwater budget: a coastal unit competes
# for none of it, an inland one diverts ~85-115 m3/MWh. The split is inferred from distance
# to the coastline.
#
# The coastline comes from data/ChinaMap/boundary.shp, whose GBCODE field separates coast
# (26010 open coast, 26080/26100 island coasts, 26 085 km in total) from the land national
# boundary (61010). The earlier version measured distance to the national boundary as a
# whole, which conflated the Yalu and Vietnam land borders with the sea and needed a
# maritime-province whitelist to compensate.
#
# Distance to the true coast splits the once-through fleet in two with an almost empty band
# between: 0-20 km holds 240.2 GW, 20-50 km holds 2.0 GW, beyond 50 km holds 222.9 GW. The
# threshold sits in the middle of that gap, so the classification is insensitive to it.
SEAWATER_COAST_DISTANCE_KM = 30.0


def add_seawater_classification(paths: ProjectPaths, plants: pd.DataFrame) -> pd.DataFrame:
    """Split once-through capacity into seawater and freshwater by distance to the coast."""
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import unary_union

    boundary = gpd.read_file(paths.find_data_file("boundary.shp")).to_crs("EPSG:2380")
    coast_lines = boundary[boundary["GBCODE"].astype(str).str.startswith("26")]
    if coast_lines.empty:
        raise ValueError("boundary.shp has no GBCODE 26* coastline features")
    coastline = unary_union(coast_lines.geometry.tolist())

    points = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(plants["centroid_longitude"], plants["centroid_latitude"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:2380")
    distance_km = points.distance(coastline) / 1000.0

    once_through = plants.get("capacity_mw_once_through", pd.Series(0.0, index=plants.index)).astype(float)
    plants = plants.copy()
    plants["coast_distance_km"] = distance_km.round(2).to_numpy()
    plants["seawater_cooled"] = (once_through > 0) & (distance_km.to_numpy() <= SEAWATER_COAST_DISTANCE_KM)

    total = plants["total_capacity_mw"].astype(float).replace(0.0, np.nan)
    freshwater = pd.Series(0.0, index=plants.index)
    for tech_key, intensity in COOLING_WATER_INTENSITY_M3_PER_MWH.items():
        column = f"capacity_mw_{tech_key.replace('-', '_')}"
        if column not in plants.columns:
            continue
        capacity = plants[column].astype(float)
        if tech_key == "once-through":
            capacity = capacity.where(~plants["seawater_cooled"], 0.0)
        freshwater += capacity / total * intensity
    plants["freshwater_intensity_m3_per_mwh"] = freshwater.fillna(0.0).round(4)
    plants["requires_water_supply"] = plants["freshwater_intensity_m3_per_mwh"] > 0
    return plants


def write_plants_unit(paths: ProjectPaths) -> pd.DataFrame:
    plants = build_plants_unit_dataframe(paths)
    write_csv(plants, paths.inputs_dir / "plants_unit.csv")
    return plants


def _haversine_distance_matrix(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Pairwise Haversine distances (km)."""
    lon_r = np.radians(lons)
    lat_r = np.radians(lats)
    dlon = lon_r[:, None] - lon_r[None, :]
    dlat = lat_r[:, None] - lat_r[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _cluster_plants_to_hubs(units: pd.DataFrame, n_hubs: int = DEFAULT_N_HUBS) -> pd.DataFrame:
    """Spatially cluster unit-level data into *n_hubs* hubs.

    Uses agglomerative clustering on unit coordinates. Each hub inherits
    all units in its cluster; aggregation (capacity, cooling, etc.) is
    done by build_plant_dataframe via a synthetic 'plant_site' label.
    """
    valid = units.dropna(subset=["latitude", "longitude"]).copy()
    if len(valid) <= n_hubs:
        return valid

    dist = _haversine_distance_matrix(
        valid["longitude"].to_numpy(), valid["latitude"].to_numpy()
    )
    clustering = AgglomerativeClustering(
        n_clusters=n_hubs,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist)
    valid["plant_site"] = [f"H{n_hubs}_{int(label):03d}" for label in labels]
    return valid


def write_plants(paths: ProjectPaths, n_hubs: int = DEFAULT_N_HUBS) -> None:
    plants_path = paths.inputs_dir / "plants_unit.csv"
    plants = pd.read_csv(plants_path)
    clustered = _cluster_plants_to_hubs(plants, n_hubs=n_hubs)
    plant_sites = build_plant_dataframe(
        plants=clustered,
        source_label=paths.rel(plants_path),
        year_basis=PLANT_YEAR_BASIS,
    )
    plant_sites = add_seawater_classification(paths, plant_sites)
    plant_sites = finalize_water_intensities(plant_sites)
    write_csv(plant_sites, paths.inputs_dir / "plants.csv")
