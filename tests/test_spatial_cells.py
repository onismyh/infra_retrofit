"""`spatial` 的栅格像元中心与按省-网格聚合（生物质、绿氨供给曲线与封存汇都走这几个函数）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine

from coal_retrofit.spatial import aggregate_to_cells, bin_index, pixel_centers, pixel_centers_lonlat, province_lookup

ALBERS = rasterio.crs.CRS.from_proj4("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +units=m")


def test_pixel_centers_and_lonlat() -> None:
    transform = Affine(0.5, 0.0, 100.0, 0.0, -0.5, 40.0)  # 0.5° 格，左上角 (100°E, 40°N)
    rows, cols = np.array([0, 1]), np.array([0, 3])
    lons, lats = pixel_centers(transform, rows, cols)
    assert lons.tolist() == [100.25, 101.75] and lats.tolist() == [39.75, 39.25]
    for crs in (None, rasterio.crs.CRS.from_epsg(4326)):  # 地理坐标的栅格原样返回
        got = pixel_centers_lonlat(transform, crs, rows, cols)
        assert got[0].tolist() == lons.tolist() and got[1].tolist() == lats.tolist()

    projected = Affine(10_000.0, 0.0, 500_000.0, 0.0, -10_000.0, 4_000_000.0)  # 10 km 格的 Albers 栅格
    x, y = pixel_centers(projected, rows, cols)
    lon, lat = pixel_centers_lonlat(projected, ALBERS, rows, cols)
    want_lon, want_lat = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True).transform(x, y)
    assert lon.tolist() == list(want_lon) and lat.tolist() == list(want_lat)
    assert lon.dtype == np.float64 and 100.0 < lon.min() and lon.max() < 120.0 and 30.0 < lat.min() < 40.0
    # 没有 `to_wkt` 的 CRS 按字串交给 pyproj（EPSG:2380 的中央经线 105°E 在 x = 500 km 处）
    lon, lat = pixel_centers_lonlat(projected, "EPSG:2380", rows, cols)
    want_lon, want_lat = Transformer.from_crs("EPSG:2380", "EPSG:4326", always_xy=True).transform(x, y)
    assert lon.tolist() == list(want_lon) and lat.tolist() == list(want_lat) and abs(lon[0] - 105.0) < 0.1


def test_aggregate_to_cells_sums_and_weights_coordinates() -> None:
    points = pd.DataFrame(
        {
            "province_id": np.array([1, 1, 1, 2, 2], dtype=np.int32),
            "longitude": [100.1, 100.2, 100.9, 100.1, 100.1],
            "latitude": [30.1, 30.1, 30.1, 30.1, 30.6],
            "supply": [1.0, 3.0, 2.0, 0.0, 5.0],
            "cost_x_supply": [10.0, 30.0, 20.0, 0.0, 50.0],
        }
    )
    cells = aggregate_to_cells(points, "supply", 0.5)
    # 省 1 的前两点同格（0.5° 格），第三点另一格；省 2 的两点经度同格、纬度不同格。按 (省, 经箱, 纬箱) 升序。
    assert cells["province_id"].tolist() == [1, 1, 2, 2]
    assert cells["lon_bin"].tolist() == [560, 561, 560, 560] and cells["lat_bin"].tolist() == [240, 240, 240, 241]
    assert cells["lon_bin"].dtype == np.int32
    assert cells["supply"].tolist() == [4.0, 2.0, 0.0, 5.0]
    assert cells["cost_x_supply"].tolist() == [40.0, 20.0, 0.0, 50.0]
    assert cells["longitude"].tolist() == [pytest.approx((100.1 + 3 * 100.2) / 4), 100.9, 0.0, 100.1]
    assert cells["latitude"].tolist() == [pytest.approx(30.1), 30.1, 0.0, 30.6]  # 权重合计 0 的格坐标记 0
    assert "weighted_lon" in cells.columns and "weighted_lat" in cells.columns
    # 输入行序不影响结果：打乱后聚合，与按序输入逐值相同
    shuffled = aggregate_to_cells(points.sample(frac=1.0, random_state=0), "supply", 0.5)
    pd.testing.assert_frame_equal(shuffled, cells)


def test_province_lookup() -> None:
    provinces = pd.DataFrame({"province_id": [3, 1], "province_name": ["Hebei", "Beijing"]})
    assert province_lookup(provinces) == {3: "Hebei", 1: "Beijing"}


def test_bin_index_floors_with_an_optional_offset() -> None:
    # 粗化水节点不移（格号写进节点 id）；供给节点把经度 +180 移成非负。
    bins = bin_index(np.array([-0.1, 0.0, 0.49, 0.5, 103.7]), 0.5)
    assert bins.tolist() == [-1, 0, 0, 1, 207] and bins.dtype == np.int32
    assert bin_index(np.array([-180.0, 179.9]), 0.5, 180.0).tolist() == [0, 719]
