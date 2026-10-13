"""水输入构建的小工具（`builders/water_grid`、`water_availability`、`water`）在合成数据上：读窗口序列、年化、网格面积、
网格落省、参照成员与供水链路。"""
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from coal_retrofit.builders.water import build_water_link_dataframe
from coal_retrofit.builders.water_availability import _candidate_grid_points, _reference_scenario_row
from coal_retrofit.builders.water_grid import SECONDS_PER_YEAR, _annual_m3, _cell_area_m2, _clean_window, _file_years
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.spatial import EARTH_RADIUS_KM


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")  # 全填充值的网格，生产代码同样如此
def test_clean_window_file_years_and_annual_m3(tmp_path) -> None:
    months = [dt.datetime(year, month, 15) for year in (2020, 2021) for month in range(1, 13)]
    path = tmp_path / "toy.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 2)
        time = ds.createVariable("time", "f8", ("time",))
        time.units, time.calendar = "days since 1601-1-1 00:00:00", "standard"
        time[:] = netCDF4.date2num(months, time.units, time.calendar)
        var = ds.createVariable("qtot", "f4", ("time", "lat", "lon"), fill_value=np.float32(1e20))
        field = np.zeros((len(months), 1, 2), dtype=np.float32)
        field[:12, 0, 0] = 1.0  # 2020 年第一格恒为 1
        field[12:, 0, 0] = 3.0  # 2021 年第一格恒为 3
        field[:, 0, 1] = 1e20  # 第二格全是填充值
        var[:] = field
    with netCDF4.Dataset(path, "r") as ds:
        years = _file_years(ds)
        assert years.tolist() == [2020] * 12 + [2021] * 12
        window = _clean_window(ds, "qtot", years == 2021)
    assert window.shape == (12, 1, 2)
    assert np.all(window[:, 0, 0] == 3.0)
    assert np.all(np.isnan(window[:, 0, 1]))  # 填充值换成 NaN
    area = np.array([[2.0, 5.0]])
    annual = _annual_m3(window, area)
    assert annual[0, 0] == pytest.approx(3.0 * 2.0 * SECONDS_PER_YEAR / 1000.0)
    assert annual[0, 1] == 0.0  # 全 NaN 的网格年水量计 0


def test_cell_area_sums_to_the_sphere_with_the_shared_radius() -> None:
    """全球 0.5° 网格的面积之和等于半径 `EARTH_RADIUS_KM` 的球面积（2026-10-13 前半径另写 6 371 km）。"""
    lat = np.arange(-89.75, 90.0, 0.5)
    total = _cell_area_m2(lat).sum() * 720
    assert total == pytest.approx(4.0 * np.pi * (EARTH_RADIUS_KM * 1000.0) ** 2, rel=1e-12)


def test_candidate_grid_points_take_the_lower_province_id_on_a_border_and_drop_points_outside() -> None:
    """网格中心压在两省公共边上的归 `province_id` 小的省，落在所有省之外的不作节点，按 (lat_index, lon_index) 排序。"""
    provinces = gpd.GeoDataFrame(
        {"province_id": [1, 2], "province_name": ["甲", "乙"]},
        geometry=[box(100.0, 30.0, 101.0, 31.0), box(101.0, 30.0, 102.0, 30.7)],
        crs="EPSG:4326",
    )
    points = _candidate_grid_points(provinces, np.array([100.5, 101.0, 101.5]), np.array([30.5, 30.9]))
    # (101.0, 30.5) 压在甲乙公共边上 -> 甲；(101.0, 30.9) 压在甲的东边上 -> 甲；(101.5, 30.9) 在两省之外 -> 去掉
    assert list(zip(points["lon_index"], points["lat_index"], points["province_id"])) == [
        (0, 0, 1), (1, 0, 1), (2, 0, 2), (0, 1, 1), (1, 1, 1),
    ]
    assert points["province_name"].tolist() == ["甲", "甲", "乙", "甲", "甲"]
    assert points["longitude"].tolist() == [100.5, 101.0, 101.5, 100.5, 101.0]


def test_reference_scenario_row_picks_first_ssp126_and_requires_both_ssps() -> None:
    members = pd.DataFrame([
        {"hydrology_model": "watergap2-2e", "gcm": "gfdl-esm4", "ssp": "ssp126", "source": "wg"},
        {"hydrology_model": "cwatm", "gcm": "mri-esm2-0", "ssp": "ssp126", "source": "cw-mri"},
        {"hydrology_model": "cwatm", "gcm": "gfdl-esm4", "ssp": "ssp370", "source": "cw-gfdl-370"},
        {"hydrology_model": "cwatm", "gcm": "gfdl-esm4", "ssp": "ssp126", "source": "cw-gfdl"},
    ])
    assert _reference_scenario_row(members)["source"] == "cw-gfdl"
    with pytest.raises(ValueError, match="No ssp126"):
        _reference_scenario_row(members[members["ssp"] == "ssp370"])
    with pytest.raises(ValueError, match="No ssp370"):
        _reference_scenario_row(members[members["ssp"] == "ssp126"])


def test_water_links_rank_nodes_within_buffer(tmp_path) -> None:
    paths = ProjectPaths(root=tmp_path)
    paths.ensure_inputs_dir()
    pd.DataFrame([
        {"plant_id": "P1", "plant_index": 0, "province_mode": "甲", "centroid_longitude": 110.0, "centroid_latitude": 30.0},
        {"plant_id": "P2", "plant_index": 1, "province_mode": "乙", "centroid_longitude": 100.0, "centroid_latitude": 20.0},
    ]).to_csv(paths.inputs_dir / "plants.csv", index=False)
    nodes = pd.DataFrame({
        "water_node_id": ["WA", "WB", "WC"],
        "province_name": ["甲", "甲", "丙"],
        "longitude": [111.0, 110.0, 115.0],  # 距 P1 约 96 km、约 111 km、远在 200 km 外
        "latitude": [30.0, 31.0, 30.0],
        "source": "s",
        "year_basis": "y",
    })
    links = build_water_link_dataframe(paths, nodes)
    assert links["plant_id"].tolist() == ["P1", "P1"]  # P2 在所有节点的 200 km 以外
    assert links["water_node_id"].tolist() == ["WA", "WB"]
    assert links["distance_rank"].tolist() == [1, 2]
    assert links["distance_km"].tolist() == [pytest.approx(96.3, abs=0.1), pytest.approx(111.2, abs=0.1)]
    assert (links["plant_count"] == 2).all() and (links["buffer_km"] == 200.0).all()

    nodes_far = nodes.assign(longitude=nodes["longitude"] + 30.0)
    empty = build_water_link_dataframe(paths, nodes_far)
    assert empty.empty and list(empty.columns) == list(links.columns)
    assert empty.index.equals(pd.RangeIndex(0))
