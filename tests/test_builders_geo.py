"""建输入时的两处地理归属按真实函数（不打桩）走一遍：点归一级流域（`builders.water_grid._assign_basin_codes`）与海水冷却判定
（`builders.plants.add_seawater_classification`），toy 图层写进临时 data/ 目录。"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from coal_retrofit.builders.plants import SEAWATER_COAST_DISTANCE_KM, add_seawater_classification
from coal_retrofit.builders.water_grid import _assign_basin_codes, load_basins
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.spatial import haversine_km


def _paths_with_basins(tmp_path) -> ProjectPaths:
    paths = ProjectPaths(tmp_path)
    target = paths.data_dir / "ChinaBasins" / "basin_l1.gpkg"
    target.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"code": ["A", "B"], "name": ["甲", "乙"], "name_en": ["Alpha", "Beta"]},
        geometry=[box(100, 30, 101, 31), box(102, 30, 103, 31)], crs="EPSG:4326",
    ).to_file(target, driver="GPKG")
    return paths


def test_assign_basin_codes_maps_points_inside_and_in_gaps_to_codes(tmp_path) -> None:
    # 读 longitude / latitude 列，按包含或最近边界归到流域代码；包含优先于最近边界的次序由 test_spatial 的重叠方块覆盖
    paths = _paths_with_basins(tmp_path)
    nodes = pd.DataFrame({"longitude": [100.5, 102.5, 101.2, 103.9], "latitude": [30.5, 30.5, 30.5, 30.5]})
    # 前两点各在一个多边形内；101.2 在缝隙里、离 A 的东边界 0.2°、离 B 的西边界 0.8°；103.9 在 B 以东
    assert _assign_basin_codes(paths, nodes).tolist() == ["A", "B", "A", "B"]
    assert load_basins(paths).crs.is_geographic


def test_load_basins_names_the_checked_in_location_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="仓库根"):   # 路径本身就含 data/ChinaBasins，要钉的是指引文字
        load_basins(ProjectPaths(tmp_path))


def test_seawater_classification_by_spherical_coast_distance(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    shp = paths.data_dir / "ChinaMap" / "boundary.shp"
    shp.parent.mkdir(parents=True)
    # 一条经线上的"海岸线"（GBCODE 26010）加一条不算海岸的国界（61010）
    gpd.GeoDataFrame(
        {"GBCODE": [26010, 61010]},
        geometry=[LineString([(120.0, 30.0), (120.0, 32.0)]), LineString([(110.0, 30.0), (110.0, 32.0)])],
        crs="EPSG:4326",
    ).to_file(shp)
    # 31°N 上：离岸约 0.1°（9.5 km）、约 0.5°（47.6 km）；第三个厂紧挨国界但远离海岸
    plants = pd.DataFrame({
        "centroid_longitude": [120.1, 120.5, 110.05], "centroid_latitude": [31.0, 31.0, 31.0],
        "total_capacity_mw": [1000.0, 1000.0, 1000.0],
        "capacity_mw_once_through": [1000.0, 1000.0, 1000.0],
    })
    out = add_seawater_classification(paths, plants)
    expected = [float(haversine_km(lon, 31.0, 120.0, 31.0)) for lon in (120.1, 120.5, 110.05)]
    assert expected[0] < SEAWATER_COAST_DISTANCE_KM < expected[1]
    np.testing.assert_allclose(out["coast_distance_km"][:2], expected[:2], atol=0.02)
    # 10° 开外：球面上到经线的垂足偏向高纬，最近顶点比同纬度的 (120, 31) 还近一点（约 1 km）
    assert 900.0 < float(out["coast_distance_km"][2]) <= expected[2]
    assert out["seawater_cooled"].tolist() == [True, False, False]
    # 海水直流不计淡水强度；淡水直流按 0.35 m3/MWh
    assert out["freshwater_intensity_m3_per_mwh"].tolist() == [0.0, 0.35, 0.35]
    assert out["requires_water_supply"].tolist() == [False, True, True]
