"""`coal_retrofit.spatial` 的球面距离与地理归属（CLAUDE.md §4.1：距离不在投影坐标下算）。"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from coal_retrofit.spatial import (
    EARTH_RADIUS_KM,
    cluster_labels,
    distance_to_lines_km,
    haversine_km,
    haversine_matrix_km,
    nearest_polygon_index,
)

BEIJING = (116.4074, 39.9042)
SHANGHAI = (121.4737, 31.2304)


def _reference_km(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    """教科书写法的大圆距离，作为 `haversine_km` 的对照。"""
    phi_a, phi_b = np.radians(lat_a), np.radians(lat_b)
    d_lambda = np.radians(lon_b - lon_a)
    cos_c = np.sin(phi_a) * np.sin(phi_b) + np.cos(phi_a) * np.cos(phi_b) * np.cos(d_lambda)
    return float(EARTH_RADIUS_KM * np.arccos(np.clip(cos_c, -1.0, 1.0)))


def test_haversine_matches_reference_and_is_symmetric() -> None:
    d = float(haversine_km(*BEIJING, *SHANGHAI))
    assert d == pytest.approx(_reference_km(*BEIJING, *SHANGHAI), rel=1e-9)
    assert 1_050.0 < d < 1_080.0
    assert float(haversine_km(*SHANGHAI, *BEIJING)) == pytest.approx(d, rel=1e-12)
    assert float(haversine_km(*BEIJING, *BEIJING)) == 0.0
    # 对跖点：半个大圆（h 恰为 1，clip 不触发；随机对跖点 h 最多超 1 一个 ulp，开方后仍为 1）
    assert float(haversine_km(0.0, 0.0, 180.0, 0.0)) == pytest.approx(np.pi * EARTH_RADIUS_KM, rel=1e-9)


def test_haversine_broadcasts_and_matrix_is_symmetric() -> None:
    lons = np.array([116.4, 121.5, 113.3])
    lats = np.array([39.9, 31.2, 23.1])
    from_beijing = haversine_km(BEIJING[0], BEIJING[1], lons, lats)
    assert from_beijing.shape == (3,)
    assert from_beijing[1] == pytest.approx(float(haversine_km(*BEIJING, lons[1], lats[1])))
    matrix = haversine_matrix_km(lons, lats)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] == pytest.approx(float(haversine_km(lons[0], lats[0], lons[1], lats[1])))


def _squares() -> gpd.GeoSeries:
    # 0：[0,1]×[0,1]；1：[2,3]×[0,1]；2：[0.5,1.5]×[0,1]，与 0 重叠。
    return gpd.GeoSeries([box(0, 0, 1, 1), box(2, 0, 3, 1), box(0.5, 0, 1.5, 1)], crs="EPSG:4326")


def test_nearest_polygon_index_inside_overlap_and_outside() -> None:
    polygons = _squares()
    lons = np.array([0.25, 0.75, 1.7, 5.0])
    lats = np.array([0.5, 0.5, 0.5, 5.0])
    index = nearest_polygon_index(lons, lats, polygons)
    # 只落在 0 里；同时落在 0 与 2 里取序号最小的 0；1.7 离 2 的东边界 0.2°、离 1 的西边界 0.3°；(5, 5) 离 1 最近。
    assert index.tolist() == [0, 0, 2, 1]
    assert index.dtype.kind == "i"


def test_nearest_polygon_index_measures_on_the_sphere() -> None:
    # 60°N：A 在东边 1.2°（球面 66.7 km），B 在北边 0.8°（89.0 km）。按度数的平面度量会选 B，球面距离选 A。
    polygons = gpd.GeoSeries([box(1.2, 59.9, 2.0, 60.1), box(-0.1, 60.8, 0.1, 61.5)], crs="EPSG:4326")
    assert nearest_polygon_index([0.0], [60.0], polygons).tolist() == [0]


def test_nearest_polygon_index_rejects_projected_crs_and_nan() -> None:
    projected = _squares().to_crs("EPSG:2380")
    with pytest.raises(ValueError, match="地理坐标"):
        nearest_polygon_index([0.5], [0.5], projected)
    with pytest.raises(ValueError, match="NaN"):
        nearest_polygon_index([np.nan], [0.5], _squares())
    with pytest.raises(ValueError, match="不一致"):
        nearest_polygon_index([0.25, 0.3], [0.5], _squares())
    with pytest.raises(ValueError, match="为空"):
        nearest_polygon_index([0.25], [0.5], gpd.GeoSeries([], crs="EPSG:4326"))


def test_distance_to_lines_km_on_and_off_the_line() -> None:
    lines = gpd.GeoSeries([LineString([(100.0, 30.0), (101.0, 30.0)])], crs="EPSG:4326")
    endpoint, north = distance_to_lines_km([100.0, 100.5], [30.0, 31.0], lines)
    assert endpoint < 1e-6   # 端点一定是加密后的顶点；不假定 segmentize 把 1° 恰好分成 100 段
    # 线上任一点到最近顶点不超过半个加密步长（0.005°，30°N 上约 0.48 km），这就是加密带来的最大偏差；
    # 沿线扫 193 个点（步距 0.98/192°，相对顶点间距有 96 个相位，其中两个恰在两顶点正中、取到半步），
    # 只测一个点抓不到加密步长变粗
    half_step = float(haversine_km(100.5, 30.0, 100.505, 30.0))
    along = distance_to_lines_km(np.linspace(100.01, 100.99, 193), np.full(193, 30.0), lines)
    assert 0.9 * half_step < along.max() <= half_step * (1.0 + 1e-9)
    # 线以北 1° 的点：最近顶点离 (100.5, 30) 不超过半步，距离与到该点的球面距离只差万分之一以内
    assert north == pytest.approx(float(haversine_km(100.5, 31.0, 100.5, 30.0)), rel=1e-4)
    with pytest.raises(ValueError, match="地理坐标"):
        distance_to_lines_km([100.5], [30.0], lines.to_crs("EPSG:2380"))


def test_point_coordinates_must_be_lon_lat_in_range() -> None:
    lines = gpd.GeoSeries([LineString([(100.0, 30.0), (101.0, 30.0)])], crs="EPSG:4326")
    with pytest.raises(ValueError, match="经纬度范围"):
        nearest_polygon_index([500_000.0], [3_300_000.0], _squares())   # 投影坐标
    with pytest.raises(ValueError, match="经纬度范围"):
        distance_to_lines_km([30.0], [100.5], lines)                    # 经纬度写反
    with pytest.raises(ValueError, match="一维"):
        nearest_polygon_index([[0.25]], [[0.5]], _squares())                # 列向量
    # 标量算一个点
    assert nearest_polygon_index(0.25, 0.5, _squares()).tolist() == [0]
    assert distance_to_lines_km(100.0, 30.0, lines).shape == (1,)


def test_layers_with_empty_geometries_are_rejected() -> None:
    polygons = gpd.GeoSeries([Polygon(), box(2, 0, 3, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="空几何"):
        nearest_polygon_index([0.5], [0.5], polygons)


def test_cluster_labels_groups_nearby_points() -> None:
    """两团各三点、相距约 1 000 km：聚成 2 簇时同团同号、异团异号，簇号为 0 与 1。"""
    lons = np.array([100.0, 100.1, 100.2, 110.0, 110.1, 110.2])
    lats = np.array([30.0, 30.1, 30.0, 35.0, 35.1, 35.0])
    labels = cluster_labels(lons, lats, 2)
    assert labels.shape == (6,) and set(labels.tolist()) == {0, 1}
    assert len(set(labels[:3].tolist())) == 1 and len(set(labels[3:].tolist())) == 1 and labels[0] != labels[3]
