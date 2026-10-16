"""封存栅格按距离聚类成汇：`builders.storage.build_storage_hub_dataframe(cluster_km=...)` 的可选建法（2026-10-16 起）。

缺省建法每个连通域一个汇（`storage._extract_storage_nodes`）：大盆地（鄂尔多斯、塔里木、东海）一个汇横跨上千公里，
零散油田又各自成汇。按距离聚类把同一封存类型（DSA / EOR）、同在陆上或同在海上的格按距离并成簇，簇内逐格的封存潜力
与注入能力相加（汇的累计封存上限与年注入上限都是这两个和，`optimization.data_prep._prepare_storages`），注入中心取
按封存量加权的质心。两种划分都保留（作者 2026-10-16 要求，由作者选一种重建管网）：

    complete  簇内任意两块相距不超过阈值（簇的直径不超过阈值）；
    average   两簇块与块之间的平均距离不超过阈值才合并。

聚类前先把 5 km 格并成 25 km 块（5 x 5 格，块的位置取块内按封存量加权的质心）：全国有效格数以万计，两两距离矩阵放不进
内存。所以"任意两块"量的是块的质心，同簇的两个格最多再远约 2 x 17.7 km（块的半对角线）。距离按球面 haversine
（`spatial.haversine_matrix_km`）。陆上 / 海上按格的落省判（`spatial.containing_polygon_index`，与
`storage._flag_offshore` 同一判据），海上的格与陆上的格不并成一簇。
"""
from __future__ import annotations

from typing import Final, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..spatial import containing_polygon_index, haversine_matrix_km, pixel_centers

ClusterLinkage = Literal["complete", "average"]
CLUSTER_LINKAGES: Final[tuple[str, ...]] = ("complete", "average")
# 聚类前并块的边长（格）：Fan 2025 栅格 5 km 一格，5 格即 25 km
BLOCK_PIXELS: Final[int] = 5


def _block_table(
    valid: np.ndarray, storage: np.ndarray, injectivity: np.ndarray, transform, provinces: gpd.GeoSeries,
    to_lonlat: Transformer,
) -> pd.DataFrame:
    """有效格并成 25 km 块：每块的封存潜力、注入能力、格数，与封存量加权的投影坐标之和（`wx`、`wy`）；陆上与海上的格分块。"""
    rows, cols = np.nonzero(valid)
    x, y = pixel_centers(transform, rows, cols)
    lon, lat = to_lonlat.transform(x, y)
    weight = storage[valid]
    pixels = pd.DataFrame({
        "offshore": np.isnan(containing_polygon_index(lon, lat, provinces)),
        "block_row": rows // BLOCK_PIXELS,
        "block_col": cols // BLOCK_PIXELS,
        "storage": weight,
        "injectivity": injectivity[valid],
        "wx": x * weight,
        "wy": y * weight,
        "pixels": 1,
    })
    sums = ["storage", "injectivity", "wx", "wy", "pixels"]
    return pixels.groupby(["offshore", "block_row", "block_col"], sort=True)[sums].sum().reset_index()


def distance_clusters(
    valid: np.ndarray,
    storage: np.ndarray,
    injectivity: np.ndarray,
    transform,
    crs,
    provinces: gpd.GeoSeries,
    cluster_km: float,
    method: ClusterLinkage,
) -> pd.DataFrame:
    """一对栅格（同一封存类型）的有效格按距离聚成簇，每簇一行：`storage_mt`、`injectivity_mtpa`、`pixel_count`、
    `longitude`、`latitude`（按封存量加权的质心）、`offshore`（簇的格在省界之外）。不丢小簇，由调用方按储量筛。

    `provinces` 是 EPSG:4326 的省界多边形；`cluster_km` 是切树的阈值（km），`method` 为 complete 或 average（见模块说明）。
    """
    if method not in CLUSTER_LINKAGES:
        raise ValueError(f"linkage 须为 {CLUSTER_LINKAGES} 之一，得到 {method!r}")
    if not cluster_km > 0:
        raise ValueError(f"cluster_km 须 > 0，得到 {cluster_km}")
    columns = ["storage_mt", "injectivity_mtpa", "pixel_count", "longitude", "latitude", "offshore"]
    if not valid.any():
        return pd.DataFrame(columns=columns)
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    blocks = _block_table(valid, storage, injectivity, transform, provinces, to_lonlat)
    block_lon, block_lat = to_lonlat.transform(
        (blocks["wx"] / blocks["storage"]).to_numpy(), (blocks["wy"] / blocks["storage"]).to_numpy()
    )
    blocks["cluster"] = 0
    for _, group in blocks.groupby("offshore", sort=True):
        index = group.index.to_numpy()
        if len(index) == 1:
            labels = np.ones(1, dtype=int)
        else:
            distances = squareform(haversine_matrix_km(block_lon[index], block_lat[index]), checks=False)
            labels = fcluster(linkage(distances, method=method), t=float(cluster_km), criterion="distance")
        blocks.loc[index, "cluster"] = labels
    clusters = blocks.groupby(["offshore", "cluster"], sort=True)[["storage", "injectivity", "wx", "wy", "pixels"]].sum()
    lon, lat = to_lonlat.transform(
        (clusters["wx"] / clusters["storage"]).to_numpy(), (clusters["wy"] / clusters["storage"]).to_numpy()
    )
    return pd.DataFrame({
        "storage_mt": clusters["storage"].to_numpy(dtype=float),
        "injectivity_mtpa": clusters["injectivity"].to_numpy(dtype=float),
        "pixel_count": clusters["pixels"].to_numpy(dtype=int),
        "longitude": np.asarray(lon, dtype=float),
        "latitude": np.asarray(lat, dtype=float),
        "offshore": clusters.index.get_level_values("offshore").to_numpy(dtype=bool),
    })
