"""汇表构建的入参处理：海上盆地膨胀像素数在读 Fan 栅格之前就查（2026-10-11 起缺省 6），0 为不合并、不进盆地合并；
海上标记的落省判据；按距离聚类的可选建法（2026-10-16 起，complete / average linkage）。"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from coal_retrofit.builders import storage
from coal_retrofit.builders.storage import build_storage_hub_dataframe
from coal_retrofit.paths import ProjectPaths


def test_negative_dilation_is_rejected_before_any_raster_is_read(tmp_path) -> None:
    # 临时目录里没有栅格：报的是入参错误而不是打不开文件，说明校验在读栅格之前
    with pytest.raises(ValueError, match="须 >= 0"):
        build_storage_hub_dataframe(ProjectPaths(tmp_path), offshore_basin_dilation_px=-1)


def _toy_bodies(storage_type: str) -> pd.DataFrame:
    dsa = storage_type == "dsa"
    return pd.DataFrame({
        "storage_type": [storage_type],
        "storage_dsa_mt": [100.0 if dsa else 0.0],
        "storage_eor_mt": [0.0 if dsa else 10.0],
        "storage_all_mt": [100.0 if dsa else 10.0],
        "injectivity_dsa_avg_mtpa": [1.0 if dsa else 0.0],
        "injectivity_eor_avg_mtpa": [0.0 if dsa else 0.1],
        "pixel_count": [4],
        "longitude": [120.0],
        "latitude": [30.0],
    })


def test_zero_dilation_skips_the_offshore_basin_merge(monkeypatch, tmp_path) -> None:
    """0 不得进 `_merge_offshore_by_basin`：scipy 的 binary_dilation 对 iterations <= 0 是"膨胀到不再变化"，会并成一个盆地。"""
    calls: list[int] = []

    def fake_merge(paths: ProjectPaths, hubs: pd.DataFrame, storage_path: object, dilation_px: int) -> pd.DataFrame:
        calls.append(dilation_px)
        return hubs

    monkeypatch.setattr(storage, "_extract_storage_nodes",
                        lambda storage_path, injectivity_path, storage_type: _toy_bodies(storage_type))
    monkeypatch.setattr(storage, "_merge_offshore_by_basin", fake_merge)
    monkeypatch.setattr(storage, "_flag_offshore", lambda paths, hubs: pd.Series(False, index=hubs.index))
    paths = ProjectPaths(tmp_path)
    kept = build_storage_hub_dataframe(paths, offshore_basin_dilation_px=0)
    assert calls == []
    assert kept["storage_hub_id"].tolist() == ["S001", "S002"]   # dsa、eor 各一个封存体，原样保留
    build_storage_hub_dataframe(paths, offshore_basin_dilation_px=6)
    assert calls == [6]


def test_offshore_flag_counts_points_on_the_coastline_as_land_and_rejects_nan(monkeypatch, tmp_path) -> None:
    """压在省界上的算陆上（intersects，2026-10-13 前按 contains 算海上），省界之外算海上，NaN 坐标报错（此前记海上）。"""
    provinces = gpd.GeoDataFrame({"province_id": [1], "province_name": ["甲"]},
                                 geometry=[box(100.0, 30.0, 101.0, 31.0)], crs="EPSG:4326")
    monkeypatch.setattr(storage, "load_provinces", lambda path: provinces)
    paths = ProjectPaths(tmp_path)
    hubs = pd.DataFrame({"longitude": [100.5, 101.0, 102.0], "latitude": [30.5, 30.5, 30.5]}, index=[3, 5, 7])
    flags = storage._flag_offshore(paths, hubs)
    assert flags.dtype == bool
    assert flags.to_dict() == {3: False, 5: False, 7: True}
    with pytest.raises(ValueError, match="NaN"):
        storage._flag_offshore(paths, pd.DataFrame({"longitude": [np.nan], "latitude": [30.5]}))


def _three_cells() -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """一行三个有效格，北纬 30 度上东经 100.0、101.5、103.2（地理坐标栅格，0.05 度一格）：A-B 144 km、B-C 164 km、A-C 308 km，
    各在不同的 25 km 块里。"""
    from rasterio.transform import from_origin

    storage_raster = np.zeros((1, 70))
    injectivity = np.zeros((1, 70))
    for col, stored, injected in ((0, 100.0, 1.0), (30, 300.0, 2.0), (64, 50.0, 4.0)):
        storage_raster[0, col], injectivity[0, col] = stored, injected
    return storage_raster > 0, storage_raster, injectivity, from_origin(99.975, 30.025, 0.05, 0.05)


@pytest.mark.parametrize(("method", "groups"), [("complete", [400.0, 50.0]), ("average", [450.0])])
def test_distance_clusters_complete_and_average_linkage(method, groups) -> None:
    """complete 下 C 与 A 相距 308 km > 300，单独成簇；average 下 {A, B} 与 C 的平均距离 (308 + 164) / 2 = 236 km，并成一簇。
    簇内封存潜力、注入能力与格数相加，质心按封存量加权。"""
    from coal_retrofit.builders.storage_clusters import distance_clusters

    valid, stored, injected, transform = _three_cells()
    land = gpd.GeoSeries([box(90.0, 20.0, 110.0, 40.0)], crs="EPSG:4326")
    clusters = distance_clusters(valid, stored, injected, transform, "EPSG:4326", land, 300.0, method)
    assert sorted(clusters["storage_mt"], reverse=True) == groups
    assert clusters["injectivity_mtpa"].sum() == 7.0 and clusters["pixel_count"].sum() == 3
    assert not clusters["offshore"].any()
    first = clusters.loc[clusters["storage_mt"].idxmax()]
    lon = (100.0 * 100.0 + 101.5 * 300.0 + (103.2 * 50.0 if method == "average" else 0.0)) / first["storage_mt"]
    assert first["longitude"] == pytest.approx(lon, abs=1e-9)


def test_distance_clusters_keep_land_and_sea_apart() -> None:
    """省界只盖住 A、B：C 在海上，average 下也不与陆上的 A、B 并簇（陆上、海上分开聚），海上标记取格的落省。"""
    from coal_retrofit.builders.storage_clusters import distance_clusters

    valid, stored, injected, transform = _three_cells()
    land = gpd.GeoSeries([box(99.0, 29.0, 102.0, 31.0)], crs="EPSG:4326")
    clusters = distance_clusters(valid, stored, injected, transform, "EPSG:4326", land, 300.0, "average")
    assert clusters.set_index("storage_mt")["offshore"].to_dict() == {400.0: False, 50.0: True}


def test_distance_clustering_arguments_are_checked_and_skip_the_basin_merge(monkeypatch, tmp_path) -> None:
    """阈值不为正、linkage 不认识都在读栅格之前报错；距离聚类不做海上盆地合并，海上标记沿用簇的（不按质心重判）。"""
    paths = ProjectPaths(tmp_path)
    with pytest.raises(ValueError, match="cluster_km"):
        build_storage_hub_dataframe(paths, cluster_km=0.0)
    with pytest.raises(ValueError, match="cluster_linkage"):
        build_storage_hub_dataframe(paths, cluster_km=300.0, cluster_linkage="single")  # type: ignore[arg-type]

    def fail(*args: object) -> None:
        raise AssertionError("distance clustering must not call this")

    def fake_clusters(paths, storage_path, injectivity_path, storage_type, cluster_km, cluster_linkage) -> pd.DataFrame:
        return _toy_bodies(storage_type).assign(offshore=storage_type == "eor")

    monkeypatch.setattr(storage, "_extract_distance_clusters", fake_clusters)
    monkeypatch.setattr(storage, "_extract_storage_nodes", fail)
    monkeypatch.setattr(storage, "_merge_offshore_by_basin", fail)
    monkeypatch.setattr(storage, "_flag_offshore", fail)
    hubs = build_storage_hub_dataframe(paths, cluster_km=300.0, cluster_linkage="average")
    assert hubs.set_index("storage_type")["offshore"].to_dict() == {"dsa": False, "eor": True}
    assert set(hubs["year_basis"]) == {"Fan_raster_5km_average_300km"}
