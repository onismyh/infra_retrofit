from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage
from sklearn.cluster import AgglomerativeClustering

from ..artifacts import write_csv
from ..paths import ProjectPaths
from ..spatial import load_provinces

logger = logging.getLogger(__name__)

# Bridge small within-basin gaps (2 pixels = 10 km at 5 km resolution)
_DILATION_PIXELS = 2
# Drop noise components below both thresholds
_MIN_STORAGE_MT = 10.0
_MIN_INJECTIVITY_MTPA = 0.1


def _extract_storage_nodes(
    storage_path: object,
    injectivity_path: object,
    storage_type: str,
) -> pd.DataFrame:
    """Extract injection nodes from a pair of rasters via connected-component analysis.

    Each connected component represents a geological formation (basin or oil field).
    The injection centre is the storage-potential-weighted centroid of all valid pixels
    in the component, reprojected to WGS84.

    Args:
        storage_path: Path to storage potential raster (Mt CO2 per pixel).
        injectivity_path: Path to injection rate raster (Mt/a per pixel).
        storage_type: ``"dsa"`` or ``"eor"``.

    Returns:
        DataFrame with one row per retained component.
    """
    with rasterio.open(storage_path) as src:
        storage = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    with rasterio.open(injectivity_path) as src:
        injectivity = src.read(1).astype(np.float64)
        inj_nodata = src.nodata

    # Replace nodata with zero so sums are safe
    if nodata is not None:
        storage_nodata = storage == nodata
    else:
        storage_nodata = np.zeros_like(storage, dtype=bool)
    if inj_nodata is not None:
        injectivity[injectivity == inj_nodata] = 0.0

    valid = ~storage_nodata & (storage > 0)
    if not valid.any():
        return pd.DataFrame()

    # Dilate to bridge small intra-basin gaps, then label connected components
    dilated = ndimage.binary_dilation(valid, iterations=_DILATION_PIXELS)
    labeled, _ = ndimage.label(dilated)

    # Pixel centre coordinates in projected CRS
    rows_idx, cols_idx = np.where(valid)
    x_proj = transform.c + (cols_idx + 0.5) * transform.a
    y_proj = transform.f + (rows_idx + 0.5) * transform.e

    storage_vals = storage[valid]
    inj_vals = injectivity[valid]
    comp_ids = labeled[valid]

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    result_rows = []
    for cid in np.unique(comp_ids):
        mask = comp_ids == cid
        sv = storage_vals[mask]
        iv = inj_vals[mask]

        total_storage = float(sv.sum())
        total_injectivity = float(iv.sum())

        if total_storage < _MIN_STORAGE_MT:
            continue

        # Storage-weighted centroid → injection centre at the richest part of the basin
        total_weight = sv.sum()
        if total_weight > 0:
            weights = sv / total_weight
        else:
            weights = np.ones(len(sv)) / len(sv)
        cx = float((x_proj[mask] * weights).sum())
        cy = float((y_proj[mask] * weights).sum())
        lon, lat = transformer.transform(cx, cy)

        result_rows.append(
            {
                "storage_type": storage_type,
                "storage_dsa_mt": total_storage if storage_type == "dsa" else 0.0,
                "storage_eor_mt": total_storage if storage_type == "eor" else 0.0,
                "storage_all_mt": total_storage,
                "injectivity_dsa_avg_mtpa": total_injectivity if storage_type == "dsa" else 0.0,
                "injectivity_eor_avg_mtpa": total_injectivity if storage_type == "eor" else 0.0,
                "pixel_count": int(mask.sum()),
                "longitude": round(lon, 4),
                "latitude": round(lat, 4),
            }
        )

    return pd.DataFrame(result_rows)


def _haversine_distance_matrix(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Compute pairwise Haversine distances (km) between points."""
    lon_r = np.radians(lons)
    lat_r = np.radians(lats)
    dlon = lon_r[:, None] - lon_r[None, :]
    dlat = lat_r[:, None] - lat_r[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


_CLUSTER_DISTANCE_KM = 200.0


def _cluster_storage_nodes(df: pd.DataFrame, max_distance_km: float = _CLUSTER_DISTANCE_KM) -> pd.DataFrame:
    """Merge nearby storage nodes so that no two centres are within *max_distance_km*.

    DSA and EOR are clustered **jointly** — hubs in the same basin are merged
    into a single node regardless of storage type.  Within each cluster the
    injection centre is the storage-weighted centroid; capacities and
    injectivities are summed (DSA and EOR components tracked separately).

    A post-merge pass iteratively merges any remaining centroids closer than
    *max_distance_km*, guaranteeing the final minimum inter-hub distance.
    """
    if len(df) <= 1:
        return df

    dist = _haversine_distance_matrix(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=max_distance_km,
        metric="precomputed",
        linkage="complete",
    )
    labels = clustering.fit_predict(dist)

    rows: list[dict] = []
    for cid in np.unique(labels):
        mask = labels == cid
        group = df[mask].reset_index(drop=True)
        rows.append(_merge_group(group))

    result = pd.DataFrame(rows)

    # Post-merge: iteratively merge closest centroids until min distance >= threshold
    while len(result) > 1:
        d = _haversine_distance_matrix(result["longitude"].to_numpy(), result["latitude"].to_numpy())
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] >= max_distance_km:
            break
        merged = _merge_group(result.iloc[[i, j]])
        keep = [k for k in range(len(result)) if k not in (i, j)]
        result = pd.concat([result.iloc[keep], pd.DataFrame([merged])], ignore_index=True)

    return result


def _merge_group(group: pd.DataFrame) -> dict:
    """Merge a group of storage nodes into a single hub row."""
    weights = group["storage_all_mt"].to_numpy().astype(float)
    total_w = weights.sum()
    w = weights / total_w if total_w > 0 else np.ones(len(weights)) / len(weights)

    dsa_mt = float(group["storage_dsa_mt"].astype(float).sum())
    eor_mt = float(group["storage_eor_mt"].astype(float).sum())
    stype = "dsa" if dsa_mt >= eor_mt else "eor"

    return {
        "storage_type": stype,
        "storage_dsa_mt": dsa_mt,
        "storage_eor_mt": eor_mt,
        "storage_all_mt": float(total_w),
        "injectivity_dsa_avg_mtpa": float(group["injectivity_dsa_avg_mtpa"].astype(float).sum()),
        "injectivity_eor_avg_mtpa": float(group["injectivity_eor_avg_mtpa"].astype(float).sum()),
        "pixel_count": int(group["pixel_count"].sum()),
        "longitude": round(float((group["longitude"].to_numpy() * w).sum()), 4),
        "latitude": round(float((group["latitude"].to_numpy() * w).sum()), 4),
    }


def _flag_offshore(paths: ProjectPaths, hubs: pd.DataFrame) -> pd.Series:
    """Flag hubs whose injection centre falls outside China's land provinces.

    About 46% of the DSA storage raster lies outside the provincial land boundaries — the
    offshore basins (East China Sea, Pearl River Mouth, Bohai, Beibu Gulf). Those hubs need
    subsea pipelines and platforms, so the optimizer scales their transport cost
    (OptimizationAssumptions.offshore_transport_multiplier).
    """
    from shapely.geometry import Point

    provinces = load_provinces(paths.data_dir / "ChinaMap" / "provinces.shp").to_crs("EPSG:4326")
    land = provinces.union_all() if hasattr(provinces, "union_all") else provinces.unary_union
    return pd.Series(
        [not land.contains(Point(float(lon), float(lat))) for lon, lat in zip(hubs["longitude"], hubs["latitude"])],
        index=hubs.index,
        dtype=bool,
    )


def _offshore_basin_labels(
    storage_path: object, hubs: pd.DataFrame, dilation_px: int
) -> np.ndarray:
    """Basin label for every hub, from a *dilation_px* dilation of the aquifer raster.

    Bodies are delineated with a 2-pixel dilation (`_DILATION_PIXELS`). Growing the same mask
    further fuses bodies of one sedimentary basin into a single connected component while
    leaving distinct basins apart. The sweep behind the default (scratch `offshore_basin_sweep.py`,
    2026-09-03) gave, for the 25 offshore bodies:

        10 km -> 9 basins   20 km -> 8   **30 km -> 7**   40 km -> 5   60 km -> 4   100 km -> 3

    30 km is the smallest radius at which the count equals the **7 offshore basins the source
    assessment itself evaluates**; the next step (40 km) fuses the Pearl River Mouth with the
    Beibu Gulf and the Bohai with the North Yellow Sea, which are different basins. The number
    therefore comes from the data set, not from a tuning knob. The EOR raster sits on a
    different grid, so basins are drawn on the saline-aquifer raster alone and every body
    (DSA or EOR) snaps to the nearest labelled pixel.
    """
    with rasterio.open(storage_path) as src:
        storage = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    valid = np.isfinite(storage) & (storage > 0)
    if nodata is not None:
        valid &= storage != nodata
    labeled, _ = ndimage.label(ndimage.binary_dilation(valid, iterations=int(dilation_px)))

    to_raster = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = to_raster.transform(hubs["longitude"].to_numpy(float), hubs["latitude"].to_numpy(float))
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.clip(np.asarray(rows), 0, labeled.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, labeled.shape[1] - 1)
    labels = labeled[rows, cols]
    if (labels == 0).any():
        _, (ri, ci) = ndimage.distance_transform_edt(labeled == 0, return_indices=True)
        labels = np.where(labels == 0, labeled[ri[rows, cols], ci[rows, cols]], labels)
    return labels.astype(int)


def _merge_offshore_by_basin(
    paths: ProjectPaths, hubs: pd.DataFrame, storage_path: object, dilation_px: int
) -> pd.DataFrame:
    """Merge offshore bodies of one basin and one storage type into a single hub.

    Onshore bodies are untouched. DSA and EOR stay separate inside a basin so the EOR credit
    is never averaged away (the pricing argument in `build_storage_hub_dataframe`). Capacities,
    injectivities and pixel counts add; the injection centre is the storage-weighted centroid.
    """
    offshore = _flag_offshore(paths, hubs).to_numpy()
    if not offshore.any():
        return hubs
    sea = hubs[offshore].reset_index(drop=True)
    sea["_basin"] = _offshore_basin_labels(storage_path, sea, dilation_px)
    merged = [
        _merge_group(group.drop(columns="_basin").reset_index(drop=True))
        for _, group in sea.groupby(["_basin", "storage_type"], sort=True)
    ]
    out = pd.concat([hubs[~offshore], pd.DataFrame(merged)], ignore_index=True)
    logger.info(
        "offshore basin merge (%d px dilation): %d offshore bodies -> %d hubs in %d basins",
        dilation_px, int(offshore.sum()), len(merged), int(sea["_basin"].nunique()),
    )
    return out


def _validate_against_source_dataset(frame: pd.DataFrame) -> None:
    """Check the extracted totals against the published national figures.

    The rasters are the 5 km fine-grid assessment of 69 saline reservoirs and ~1 150 oilfields
    across 24 major sedimentary basins (17 onshore, 7 offshore). Its published national storage
    potential is 1 506-2 265 Gt for deep saline aquifers (90% interval) and 8.0-9.1 Gt for
    oilfields. Extracting the rasters here must land inside those, or the extraction is wrong.
    """
    dsa_gt = float(frame.loc[frame["storage_type"] == "dsa", "storage_all_mt"].sum()) / 1e3
    eor_gt = float(frame.loc[frame["storage_type"] == "eor", "storage_all_mt"].sum()) / 1e3
    for name, value, lo, hi in (("DSA", dsa_gt, 1506.0, 2265.0), ("EOR", eor_gt, 8.0, 9.1)):
        inside = lo <= value <= hi
        logger.info(
            "storage extraction %s: %.1f Gt vs published %.0f-%.1f Gt -- %s",
            name, value, lo, hi, "OK" if inside else "OUT OF RANGE",
        )
        if not inside:
            logger.warning("%s storage total %.1f Gt falls outside the published range", name, value)


def build_storage_hub_dataframe(
    paths: ProjectPaths,
    merge_distance_km: float | None = None,
    offshore_basin_dilation_px: int | None = None,
) -> pd.DataFrame:
    """Build the sink table.

    *offshore_basin_dilation_px* merges offshore bodies basin-by-basin (and type-by-type);
    see `_offshore_basin_labels` for why 6 px (30 km) is the defensible value. Onshore bodies
    are never merged by it. Default None keeps every body, i.e. the 103-sink table.

    Each retained connected component of the storage raster is one sink -- a contiguous
    geological storage body as delineated by the source assessment. DSA and EOR bodies stay
    SEPARATE even where they overlie one another, because they carry different costs: EOR earns
    `eor_credit_cny_per_t` against the base storage cost and deep saline storage does not.

    *merge_distance_km* reinstates the old behaviour of merging bodies whose centres fall within
    that distance. It defaults to None (no merging). The 200 km merge that used to be
    unconditional had no geological basis: it collapsed 103 bodies into 35, relabelled 7 232 Mt
    of EOR capacity as DSA -- deleting its revenue credit -- and lengthened the capacity-weighted
    haul from a plant to its nearest sink from 200 km to 234 km.
    """
    base = paths.data_dir / "封存汇图层-Fan"

    dsa = _extract_storage_nodes(
        storage_path=base / "DSA-storage potential.tif",
        injectivity_path=base / "DSA-injection-AVG-Recommended.tif",
        storage_type="dsa",
    )

    eor = _extract_storage_nodes(
        storage_path=base / "EOR-storage potential.tif",
        injectivity_path=base / "EOR-injectionl.tif",
        storage_type="eor",
    )

    combined = pd.concat([dsa, eor], ignore_index=True)
    _validate_against_source_dataset(combined)
    if merge_distance_km is not None:
        combined = _cluster_storage_nodes(combined, max_distance_km=merge_distance_km)
        logger.info("merged storage bodies within %.0f km: %d hubs", merge_distance_km, len(combined))
    else:
        logger.info("no distance merge: %d geological storage bodies retained", len(combined))
    if offshore_basin_dilation_px is not None:
        combined = _merge_offshore_by_basin(
            paths, combined, base / "DSA-storage potential.tif", offshore_basin_dilation_px
        )
    combined = combined.sort_values(
        ["storage_type", "storage_all_mt"], ascending=[True, False]
    ).reset_index(drop=True)
    combined["storage_hub_index"] = range(1, len(combined) + 1)
    combined["storage_hub_id"] = combined["storage_hub_index"].map(lambda i: f"S{int(i):03d}")
    combined["province"] = ""
    combined["offshore"] = _flag_offshore(paths, combined)
    combined["source"] = str(base.name)
    combined["year_basis"] = "Fan_raster_5km"

    columns = [
        "storage_hub_id",
        "storage_hub_index",
        "storage_type",
        "province",
        "storage_dsa_mt",
        "storage_eor_mt",
        "storage_all_mt",
        "injectivity_dsa_avg_mtpa",
        "injectivity_eor_avg_mtpa",
        "pixel_count",
        "offshore",
        "longitude",
        "latitude",
        "source",
        "year_basis",
    ]
    return combined[columns].sort_values("storage_hub_index").reset_index(drop=True)


def write_storage_hubs(
    paths: ProjectPaths,
    merge_distance_km: float | None = None,
    offshore_basin_dilation_px: int | None = None,
) -> pd.DataFrame:
    storage = build_storage_hub_dataframe(
        paths,
        merge_distance_km=merge_distance_km,
        offshore_basin_dilation_px=offshore_basin_dilation_px,
    )
    write_csv(storage, paths.inputs_dir / "storage_hubs.csv")
    return storage
