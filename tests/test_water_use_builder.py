"""耗水表构建（`builders/water_use.build_water_use_dataframe`）在合成的 ISIMIP 文件上，不求解。

4 x 4 的 0.5 度网格分成左右两个流域：B1 的径流在 12-2 月最低；B2 在 CWatM 成员里 6-8 月最低、在 WaterGAP 成员里
9-11 月最低，两个成员的枯水期窗口要各按自己的 qtot 选。核对：生活与灌溉的全年值和枯水期年化值（CWatM 的灌溉由
总耗水减其余三项得出，WaterGAP 直接读 pirruse）、枯水期起始月，以及耗水表的枯水期窗口与 `water_availability.csv`
枯水期列（`build_water_availability_dataframe`）是同一个。
"""
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import coal_retrofit.builders.water as builders_water
from coal_retrofit.builders.water import SECONDS_PER_YEAR, _cell_area_m2, build_water_scenarios_dataframe
from coal_retrofit.builders.water_use import build_water_use_dataframe, water_use_file, write_water_use
from coal_retrofit.paths import ProjectPaths

LAT = np.array([31.75, 31.25, 30.75, 30.25])
LON = np.array([100.25, 100.75, 101.25, 101.75])
MONTHS = [dt.datetime(year, month, 15) for year in range(2015, 2061) for month in range(1, 13)]
# 各成员、各流域径流最低的三个月
LOW = {"cwatm": {"B1": (12, 1, 2), "B2": (6, 7, 8)}, "watergap2-2e": {"B1": (12, 1, 2), "B2": (9, 10, 11)}}
DOMESTIC, INDUSTRY, LIVESTOCK = 1.0e-7, 2.0e-7, 0.5e-7  # kg m-2 s-1，全年不变


def _irrigation(month: int) -> float:
    return 4.0e-7 if month in (6, 7, 8) else 1.0e-7


def _field(variable: str, hydro: str) -> np.ndarray:
    """(time, lat, lon) 的合成场，左两列是 B1、右两列是 B2。"""
    data = np.zeros((len(MONTHS), len(LAT), len(LON)), dtype=np.float32)
    for t, date in enumerate(MONTHS):
        irrigation = _irrigation(date.month)
        for basin, columns in (("B1", slice(0, 2)), ("B2", slice(2, 4))):
            value = {
                "qtot": 0.2e-5 if date.month in LOW[hydro][basin] else 1.0e-5,
                "pdomuse": DOMESTIC, "pinduse": INDUSTRY, "pliveuse": LIVESTOCK,
                "ptotuse": DOMESTIC + INDUSTRY + LIVESTOCK + irrigation,
                "pirruse": irrigation,
            }[variable]
            data[t, :, columns] = value
    return data


def _write_nc(path, variable: str, hydro: str) -> None:
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", len(LAT))
        ds.createDimension("lon", len(LON))
        time = ds.createVariable("time", "f8", ("time",))
        time.units, time.calendar = "days since 1601-1-1 00:00:00", "standard"
        time[:] = netCDF4.date2num(MONTHS, time.units, time.calendar)
        ds.createVariable("lat", "f4", ("lat",))[:] = LAT
        ds.createVariable("lon", "f4", ("lon",))[:] = LON
        ds.createVariable(variable, "f4", ("time", "lat", "lon"), fill_value=np.float32(1e20))[:] = _field(variable, hydro)


MEMBERS = {"cwatm": ("qtot", "pdomuse", "pinduse", "pliveuse", "ptotuse"),
           "watergap2-2e": ("qtot", "pdomuse", "pirruse")}


def _qtot_name(hydro: str) -> str:
    return f"{hydro}_toy-gcm_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_monthly_2015_2060.nc"


@pytest.fixture
def paths(tmp_path, monkeypatch) -> ProjectPaths:
    water = tmp_path / "data" / "water"
    water.mkdir(parents=True)
    for hydro, variables in MEMBERS.items():
        for variable in variables:
            _write_nc(water / water_use_file(_qtot_name(hydro), variable), variable, hydro)
    (tmp_path / "inputs").mkdir()
    pd.DataFrame([
        {"scenario_id": f"{hydro}|toy-gcm|ssp126", "hydrology_model": hydro, "gcm": "toy-gcm", "ssp": "ssp126",
         "source": f"data/water/{_qtot_name(hydro)}"}
        for hydro in MEMBERS
    ]).to_csv(tmp_path / "inputs" / "water_scenarios.csv", index=False)
    basins = gpd.GeoDataFrame({"code": ["B1", "B2"], "name": ["左", "右"]},
                              geometry=[box(100.0, 30.0, 101.0, 32.0), box(101.0, 30.0, 102.0, 32.0)], crs="EPSG:4326")
    monkeypatch.setattr(builders_water, "load_basins", lambda _paths: basins)
    return ProjectPaths(root=tmp_path)


def _volume(flux: float) -> float:
    """一个流域（8 个网格）上恒定通量的年化水量，m3/yr。"""
    area = _cell_area_m2(LAT)[:, None] * np.ones((1, 2))
    return float((flux * area * SECONDS_PER_YEAR / 1000.0).sum())


def test_basin_use_annual_dry_and_window(paths) -> None:
    frame = write_water_use(paths)
    assert len(frame) == 2 * 4 * 2  # 成员 x 规划年 x 流域
    assert (paths.inputs_dir / "water_basin_use.csv").exists()
    annual_irrigation = np.mean([_irrigation(month) for month in range(1, 13)])
    for row in frame.itertuples(index=False):
        basin = row.basin_code
        low = LOW[row.hydrology_model][basin]
        dry_irrigation = np.mean([_irrigation(month) for month in low])
        assert row.dry_season_start_month == low[0]
        assert row.domestic_m3_per_year == pytest.approx(_volume(DOMESTIC), rel=1e-6)
        assert row.dry_season_domestic_m3_per_year == pytest.approx(_volume(DOMESTIC), rel=1e-6)
        assert row.irrigation_m3_per_year == pytest.approx(_volume(annual_irrigation), rel=1e-5)
        assert row.dry_season_irrigation_m3_per_year == pytest.approx(_volume(dry_irrigation), rel=1e-5)
    assert set(frame["year_basis"]) == {"2021-2030", "2031-2040", "2041-2050", "2051-2060"}


def test_dry_window_is_the_one_behind_the_dry_season_runoff(paths) -> None:
    """`water_availability.csv` 的枯水期列 = 全年径流 x 流域最枯三个月的份额；这个份额与耗水表选的窗口一致。"""
    scenarios = build_water_scenarios_dataframe(paths)
    assert list(scenarios["variable"]) == ["qtot", "qtot"]  # 同目录的耗水文件不算成员
    nodes = pd.DataFrame([
        {"water_node_id": f"W{i}{j}", "lat_index": i, "lon_index": j, "basin_code": "B1" if j < 2 else "B2"}
        for i in range(len(LAT)) for j in range(len(LON))
    ])
    availability = builders_water.build_water_availability_dataframe(paths, scenarios, nodes)
    use = build_water_use_dataframe(paths, pd.read_csv(paths.inputs_dir / "water_scenarios.csv"))
    totals = availability.groupby(["scenario_id", "planning_year", "basin_code"])[
        ["available_water_m3_per_year", "dry_season_water_m3_per_year"]].sum()
    for row in use.itertuples(index=False):
        annual, dry = totals.loc[(row.scenario_id, row.planning_year, row.basin_code)]
        low_months = [(row.dry_season_start_month - 1 + k) % 12 + 1 for k in range(3)]
        flux = [0.2e-5 if month in LOW[row.hydrology_model][row.basin_code] else 1.0e-5 for month in low_months]
        assert dry / annual == pytest.approx(np.mean(flux) / np.mean([0.2e-5] * 3 + [1.0e-5] * 9), rel=1e-6)


def test_missing_water_use_file_raises(paths) -> None:
    (paths.data_dir / "water" / water_use_file(_qtot_name("watergap2-2e"), "pirruse")).unlink()
    with pytest.raises(FileNotFoundError, match="download_isimip_water_use.py"):
        write_water_use(paths)
