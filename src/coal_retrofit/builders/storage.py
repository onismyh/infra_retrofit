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

# 弥合盆地内部的小间隙（5 km 分辨率下 2 个像素 = 10 km）
_DILATION_PIXELS = 2
# 丢弃同时低于两个阈值的噪声连通域
_MIN_STORAGE_MT = 10.0
_MIN_INJECTIVITY_MTPA = 0.1


def _extract_storage_nodes(
    storage_path: object,
    injectivity_path: object,
    storage_type: str,
) -> pd.DataFrame:
    """用连通域分析从一对栅格中提取注入节点。

    每个连通域代表一个地质构造（盆地或油田）。
    注入中心是该连通域内全部有效像素按封存潜力加权的质心，
    并重投影到 WGS84。

    Args:
        storage_path: 封存潜力栅格的路径（每像素 Mt CO2）。
        injectivity_path: 注入速率栅格的路径（每像素 Mt/a）。
        storage_type: ``"dsa"`` 或 ``"eor"``。

    Returns:
        每个保留下来的连通域占一行的 DataFrame。
    """
    with rasterio.open(storage_path) as src:
        storage = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    with rasterio.open(injectivity_path) as src:
        injectivity = src.read(1).astype(np.float64)
        inj_nodata = src.nodata

    # 把 nodata 替换为零，保证求和安全
    if nodata is not None:
        storage_nodata = storage == nodata
    else:
        storage_nodata = np.zeros_like(storage, dtype=bool)
    if inj_nodata is not None:
        injectivity[injectivity == inj_nodata] = 0.0

    valid = ~storage_nodata & (storage > 0)
    if not valid.any():
        return pd.DataFrame()

    # 先膨胀以弥合盆地内部的小间隙，再标记连通域
    dilated = ndimage.binary_dilation(valid, iterations=_DILATION_PIXELS)
    labeled, _ = ndimage.label(dilated)

    # 像素中心在投影 CRS 下的坐标
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

        # 按封存量加权的质心 → 注入中心落在盆地最富集的部分
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
    """计算各点两两之间的 Haversine 距离（km）。"""
    lon_r = np.radians(lons)
    lat_r = np.radians(lats)
    dlon = lon_r[:, None] - lon_r[None, :]
    dlat = lat_r[:, None] - lat_r[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


_CLUSTER_DISTANCE_KM = 200.0


def _cluster_storage_nodes(df: pd.DataFrame, max_distance_km: float = _CLUSTER_DISTANCE_KM) -> pd.DataFrame:
    """合并相近的封存节点，使任意两个中心的距离都不在 *max_distance_km* 以内。

    DSA 与 EOR **联合**聚类——同一盆地内的 hub 不论封存类型都合并为
    单个节点。每个簇内的注入中心取按封存量加权的质心；容量和
    注入能力相加（DSA 与 EOR 分量分别记录）。

    合并后再做一轮迭代，把仍然近于 *max_distance_km* 的质心继续合并，
    从而保证最终 hub 间的最小距离。
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

    # 合并后处理：迭代合并最近的两个质心，直到最小距离 >= 阈值
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
    """把一组封存节点合并成一行 hub 记录。"""
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
    """标记注入中心落在中国陆地省份之外的 hub。

    DSA 封存栅格约有 46% 位于省级陆地边界之外——即海域盆地（东海、珠江口、渤海、
    北部湾）。这些 hub 需要海底管道和平台，因此优化器会按系数缩放其运输成本
    （OptimizationAssumptions.offshore_transport_multiplier）。
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
    """基于对含水层栅格做 *dilation_px* 像素膨胀的结果，给出每个 hub 的盆地标签。

    封存体按 2 像素膨胀（`_DILATION_PIXELS`）划定。把同一掩膜继续膨胀，会把
    同一沉积盆地内的封存体融成一个连通域，而不同盆地仍保持分开。默认值背后的
    扫描（临时脚本 `offshore_basin_sweep.py`，2026-09-03）对 25 个海上封存体给出：

        10 km -> 9 个盆地   20 km -> 8   **30 km -> 7**   40 km -> 5   60 km -> 4   100 km -> 3

    30 km 是盆地数恰好等于**来源评估本身所评价的 7 个海域盆地**的最小半径；
    下一档（40 km）会把珠江口与北部湾、渤海与北黄海融在一起，而它们是不同的
    盆地。因此这个数取自数据集，而不是来自调参旋钮。EOR 栅格在另一套网格上，
    所以盆地只在咸水层栅格上划分，每个封存体（DSA 或 EOR）都吸附到最近的
    已标记像素。
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
    """把同一盆地、同一封存类型的海上封存体合并为单个 hub。

    陆上封存体不动。盆地内 DSA 与 EOR 仍然分开，这样 EOR 抵扣就不会被平均掉
    （计价理由见 `build_storage_hub_dataframe`）。容量、注入能力和像素数
    相加；注入中心取按封存量加权的质心。
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
    """用已发表的全国数字核对提取出的总量。

    这些栅格来自对 24 个主要沉积盆地（陆上 17 个、海域 7 个）中 69 个咸水储层和
    约 1 150 个油田所做的 5 km 精细网格评估。其发表的全国封存潜力为：深部咸水层
    1 506-2 265 Gt（90% 区间），油田 8.0-9.1 Gt。这里对栅格的提取结果必须落在
    这些区间内，否则就是提取出错了。
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
    """构建汇表。

    *offshore_basin_dilation_px* 逐盆地（并逐类型）合并海上封存体；为什么 6 px（30 km）
    是站得住的取值，见 `_offshore_basin_labels`。陆上封存体从不因它而合并。默认 None
    保留每个封存体，即 103 个汇的表。

    封存栅格中每个保留下来的连通域就是一个汇——即来源评估所划定的一个连续的
    地质封存体。DSA 与 EOR 封存体即使上下叠置也保持分开，因为二者成本不同：
    EOR 可按 `eor_credit_cny_per_t` 抵减基础封存成本，深部咸水层封存则不能。

    *merge_distance_km* 恢复旧行为：把中心相距在该距离以内的封存体合并。默认为
    None（不合并）。过去无条件执行的 200 km 合并没有地质依据：它把 103 个封存体
    压成 35 个，把 7 232 Mt 的 EOR 容量改标为 DSA——删掉了其收益抵扣——并把
    按容量加权的电厂到最近汇的运距从 200 km 拉长到 234 km。
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
