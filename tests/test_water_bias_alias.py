"""CWatM historical 文件对应表（`builders/water_bias.HISTORICAL_FILE_GCM`）与成员均值补全（`member_bias_factors`），在合成文件上。

ISIMIP issue #34：CWatM 的 historical qtot 文件在 GCM 之间错位一格（标 ipsl 的是 GFDL 的模拟、标 mpi 的是 IPSL、标 mri 的是
MPI、标 ukesm 的是 MRI，真正的 UKESM 没有）。合成五份 cwatm 历史文件（gfdl、ipsl、mpi、mri、ukesm 标签，径流水平各不同）与
一份 watergap2-2e 的 ipsl 文件，核对：成员取到的是实际驱动它的文件、因子等于官方总量除以该文件的流域径流、cwatm|ukesm 取
四个真实成员的均值、没下载文件的成员保持 1.0 且不进均值（并有告警）、对应表指向的文件缺失时该成员同样 1.0、没有捐赠成员时
ukesm 也是 1.0、WaterGAP 不受对应表影响；粗化按情景保留 `bias_source`、坐标按模拟径流而不是校正后的可用量加权。
"""
from __future__ import annotations

import datetime as dt
import logging

import netCDF4
import numpy as np
import pandas as pd
import pytest

from coal_retrofit.builders.water import coarsen_water_inputs
from coal_retrofit.builders.water_bias import _historical_source, has_genuine_history, member_bias_factors
from coal_retrofit.builders.water_grid import SECONDS_PER_YEAR, _cell_area_m2
from coal_retrofit.constants import OFFICIAL_BASIN_WATER_1E8_M3
from coal_retrofit.paths import ProjectPaths

LAT = np.array([40.25, 39.75])
LON = np.array([116.25, 116.75])
MONTHS = [dt.datetime(year, month, 15) for year in range(2010, 2015) for month in range(1, 13)]
# 文件名里的 GCM 标签 -> 该文件的径流水平（kg m-2 s-1，全场常数）
LEVEL = {"gfdl-esm4": 1.0e-5, "ipsl-cm6a-lr": 1.0e-5, "mpi-esm1-2-hr": 2.0e-5, "mri-esm2-0": 3.0e-5, "ukesm1-0-ll": 4.0e-5}
GCMS = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
ZONES = np.array([[1, 1], [2, 2]])   # 上一行流域 C，下一行流域 F
CODES = ["C", "F"]
AREA = _cell_area_m2(LAT)[:, None] * np.ones((1, len(LON)))


def _name(hydro: str, gcm: str) -> str:
    return f"{hydro}_{gcm}_w5e5_historical_histsoc_default_qtot_global_monthly_1850_2014.nc"


def _write_nc(path, level: float) -> None:
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", len(LAT))
        ds.createDimension("lon", len(LON))
        time = ds.createVariable("time", "f8", ("time",))
        time.units, time.calendar = "days since 1601-1-1 00:00:00", "standard"
        time[:] = netCDF4.date2num(MONTHS, time.units, time.calendar)
        ds.createVariable("lat", "f4", ("lat",))[:] = LAT
        ds.createVariable("lon", "f4", ("lon",))[:] = LON
        qtot = ds.createVariable("qtot", "f4", ("time", "lat", "lon"), fill_value=np.float32(1e20))
        qtot[:] = np.full((len(MONTHS), len(LAT), len(LON)), level, dtype=np.float32)


@pytest.fixture
def paths(tmp_path) -> ProjectPaths:
    water = tmp_path / "data" / "water"
    water.mkdir(parents=True)
    for gcm, level in LEVEL.items():
        _write_nc(water / _name("cwatm", gcm), level)
    _write_nc(water / _name("watergap2-2e", "ipsl-cm6a-lr"), 5.0e-5)
    return ProjectPaths(tmp_path)


def _expected(code: str, level: float) -> float:
    """官方总量 ÷（常数径流 × 流域格面积 × 一年秒数 / 1000）。"""
    row = 0 if code == "C" else 1
    return OFFICIAL_BASIN_WATER_1E8_M3[code] * 1e8 / (level * AREA[row].sum() * SECONDS_PER_YEAR / 1000.0)


def test_has_genuine_history_only_denies_cwatm_ukesm() -> None:
    assert not has_genuine_history("cwatm", "ukesm1-0-ll")
    assert has_genuine_history("cwatm", "mri-esm2-0")
    assert has_genuine_history("watergap2-2e", "ukesm1-0-ll")
    assert has_genuine_history("cwatm", "toy-gcm")


def test_historical_source_follows_the_actual_driving_gcm(paths: ProjectPaths) -> None:
    assert _historical_source(paths, "cwatm", "gfdl-esm4").name == _name("cwatm", "gfdl-esm4")
    assert _historical_source(paths, "cwatm", "ipsl-cm6a-lr").name == _name("cwatm", "mpi-esm1-2-hr")
    assert _historical_source(paths, "cwatm", "mpi-esm1-2-hr").name == _name("cwatm", "mri-esm2-0")
    assert _historical_source(paths, "cwatm", "mri-esm2-0").name == _name("cwatm", "ukesm1-0-ll")
    assert _historical_source(paths, "cwatm", "ukesm1-0-ll") is None
    # WaterGAP2-2e 的文件没有错位，按文件名直接取。
    assert _historical_source(paths, "watergap2-2e", "ipsl-cm6a-lr").name == _name("watergap2-2e", "ipsl-cm6a-lr")
    assert _historical_source(paths, "watergap2-2e", "gfdl-esm4") is None


def test_member_factors_use_the_remapped_file_and_mean_for_ukesm(paths: ProjectPaths) -> None:
    members = member_bias_factors(paths, "cwatm", [*GCMS, "toy-gcm"], ZONES, CODES, AREA)
    # 实际驱动：ipsl <- 标 mpi 的文件（2e-5）、mpi <- 标 mri 的（3e-5）、mri <- 标 ukesm 的（4e-5）
    genuine = {"gfdl-esm4": 1.0e-5, "ipsl-cm6a-lr": 2.0e-5, "mpi-esm1-2-hr": 3.0e-5, "mri-esm2-0": 4.0e-5}
    for gcm, level in genuine.items():
        factors, source = members[gcm]
        assert source == _name("cwatm", {"gfdl-esm4": "gfdl-esm4", "ipsl-cm6a-lr": "mpi-esm1-2-hr",
                                          "mpi-esm1-2-hr": "mri-esm2-0", "mri-esm2-0": "ukesm1-0-ll"}[gcm])
        for code in CODES:
            assert factors[code] == pytest.approx(_expected(code, level), rel=1e-5)
    # 没有真实历史运行的 ukesm：四个真实成员的流域均值，来源写明均值与成员
    ukesm, source = members["ukesm1-0-ll"]
    assert source == "mean of cwatm gfdl-esm4, ipsl-cm6a-lr, mpi-esm1-2-hr, mri-esm2-0"
    for code in CODES:
        assert ukesm[code] == pytest.approx(np.mean([_expected(code, level) for level in genuine.values()]), rel=1e-5)
    # 没下载文件、也不在对应表里的成员：1.0、"none"，且没有被拉进均值
    assert members["toy-gcm"] == ({"C": 1.0, "F": 1.0}, "none")


def test_mapped_member_without_its_file_stays_uncorrected(paths: ProjectPaths, caplog) -> None:
    # mri 的真实历史运行是标 ukesm 的文件；删掉它，mri 就是"文件未下载"：1.0、"none"、告警，且不进 ukesm 的均值
    (paths.data_dir / "water" / _name("cwatm", "ukesm1-0-ll")).unlink()
    with caplog.at_level(logging.WARNING, logger="coal_retrofit.builders.water_bias"):
        members = member_bias_factors(paths, "cwatm", GCMS, ZONES, CODES, AREA)
    assert members["mri-esm2-0"] == ({"C": 1.0, "F": 1.0}, "none")
    assert any("cwatm|mri-esm2-0" in record.getMessage() for record in caplog.records)
    ukesm, source = members["ukesm1-0-ll"]
    assert source == "mean of cwatm gfdl-esm4, ipsl-cm6a-lr, mpi-esm1-2-hr"
    for code in CODES:
        assert ukesm[code] == pytest.approx(np.mean([_expected(code, level) for level in (1.0e-5, 2.0e-5, 3.0e-5)]), rel=1e-5)


def test_member_without_genuine_run_and_without_donors_is_none(paths: ProjectPaths) -> None:
    members = member_bias_factors(paths, "cwatm", ["ukesm1-0-ll"], ZONES, CODES, AREA)
    assert members == {"ukesm1-0-ll": ({"C": 1.0, "F": 1.0}, "none")}


def test_watergap_member_is_not_remapped(paths: ProjectPaths) -> None:
    members = member_bias_factors(paths, "watergap2-2e", ["gfdl-esm4", "ipsl-cm6a-lr"], ZONES, CODES, AREA)
    assert members["ipsl-cm6a-lr"][1] == _name("watergap2-2e", "ipsl-cm6a-lr")
    assert members["ipsl-cm6a-lr"][0]["C"] == pytest.approx(_expected("C", 5.0e-5), rel=1e-5)
    # gfdl 的文件没下载：不是"没有真实运行"，保持 1.0 而不是取均值
    assert members["gfdl-esm4"] == ({"C": 1.0, "F": 1.0}, "none")


def test_coarsening_keeps_bias_source() -> None:
    nodes = pd.DataFrame({"water_node_id": ["N1", "N2"], "province_name": ["北京市", "北京市"], "basin_code": ["C", "C"],
                          "longitude": [116.25, 116.75], "latitude": [40.25, 40.75]})
    source = {"ukesm1-0-ll": "mean of cwatm gfdl-esm4, ipsl-cm6a-lr, mpi-esm1-2-hr, mri-esm2-0",
              "gfdl-esm4": _name("cwatm", "gfdl-esm4")}
    availability = pd.concat([
        pd.DataFrame({
            "water_node_id": ["N1", "N2"], "planning_year": [2030, 2030], "scenario_family": ["baseline"] * 2,
            "scenario_id": [f"cwatm|{gcm}|ssp126"] * 2, "hydrology_model": ["cwatm"] * 2, "gcm": [gcm] * 2,
            "ssp": ["ssp126"] * 2, "basin_code": ["C", "C"], "available_water_m3_per_year": [1.0e6, 2.0e6],
            "dry_season_water_m3_per_year": [1.0e5, 2.0e5], "local_runoff_m3_per_year": [1.0e6, 2.0e6],
            "bias_factor": [0.85, 0.85], "bias_source": [source[gcm]] * 2,
        })
        for gcm in source
    ], ignore_index=True)
    _, coarse = coarsen_water_inputs(nodes, availability, 2.0)
    # 两个节点并成一个粗节点，两个情景各一行，各带自己的来源
    assert len(coarse) == 2
    assert coarse["available_water_m3_per_year"].tolist() == pytest.approx([3.0e6, 3.0e6])
    assert dict(zip(coarse["gcm"], coarse["bias_source"], strict=True)) == source


def test_coarsening_weights_coordinates_by_local_runoff() -> None:
    nodes = pd.DataFrame({"water_node_id": ["N1", "N2"], "province_name": ["北京市", "北京市"], "basin_code": ["C", "C"],
                          "longitude": [116.0, 117.0], "latitude": [40.0, 41.0]})
    availability = pd.DataFrame({
        "water_node_id": ["N1", "N2"], "planning_year": [2030, 2030], "scenario_family": ["baseline"] * 2,
        "scenario_id": ["cwatm|gfdl-esm4|ssp126"] * 2, "hydrology_model": ["cwatm"] * 2, "gcm": ["gfdl-esm4"] * 2,
        "ssp": ["ssp126"] * 2, "basin_code": ["C", "C"],
        # 校正后可用量相同、模拟径流 1 : 3——坐标只跟径流走，与偏差因子无关
        "available_water_m3_per_year": [1.0e6, 1.0e6], "dry_season_water_m3_per_year": [1.0e5, 1.0e5],
        "local_runoff_m3_per_year": [1.0e6, 3.0e6], "bias_factor": [1.0, 1.0 / 3.0], "bias_source": ["f", "f"],
    })
    coarse, _ = coarsen_water_inputs(nodes, availability, 2.0)
    assert len(coarse) == 1
    assert coarse["longitude"].iloc[0] == pytest.approx(116.75)
    assert coarse["latitude"].iloc[0] == pytest.approx(40.75)
