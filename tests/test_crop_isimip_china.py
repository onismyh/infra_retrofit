"""`scripts/crop_isimip_china.py`：合成的 0.5° 全球文件裁到中国框，数值、属性与 `_FillValue` 原样，已裁的跳过，
失败时不留临时文件、不动原文件。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import netCDF4
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FILL = np.float32(1e20)


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    alias = "crop_isimip_china_under_test"
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / "crop_isimip_china.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    return module


def _write_global(path: Path, lat_north_first: bool = True) -> np.ndarray:
    data = np.arange(3 * 360 * 720, dtype=np.float32).reshape(3, 360, 720)
    data[:, 100, 550] = FILL
    lat = np.arange(89.75, -90, -0.5) if lat_north_first else np.arange(-89.75, 90, 0.5)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("institution", "test")
        for name, size in (("time", None), ("lon", 720), ("lat", 360)):
            ds.createDimension(name, size)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "months since 1661-01-01 00:00:00"
        time[:] = [0, 1, 2]
        ds.createVariable("lon", "f8", ("lon",))[:] = np.arange(-179.75, 180, 0.5)
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        var = ds.createVariable("pdomuse", "f4", ("time", "lat", "lon"), fill_value=FILL)
        var.units = "kg m-2 s-1"
        var.set_auto_maskandscale(False)
        var[:] = data
    return data


def test_crop_keeps_values_attributes_and_fill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = _load(monkeypatch)
    source, dest = tmp_path / "global.nc", tmp_path / "out" / "china"   # 输出目录不存在时自动建
    data = _write_global(source)
    assert script.crop(source, dest) is True
    with netCDF4.Dataset(dest / "global.nc") as ds:
        var = ds.variables["pdomuse"]
        var.set_auto_maskandscale(False)
        assert ds.data_model == "NETCDF4_CLASSIC"
        assert ds.getncattr("institution") == "test"
        assert ds.getncattr("history_crop") == script.CROP_NOTE
        assert var.units == "kg m-2 s-1" and var.getncattr("_FillValue") == FILL
        assert np.array_equal(var[:], data[:, 66:180, 500:640])
        assert (ds.variables["lat"][0], ds.variables["lat"][-1]) == (56.75, 0.25)
        assert (ds.variables["lon"][0], ds.variables["lon"][-1]) == (70.25, 139.75)
        assert list(ds.variables["time"][:]) == [0, 1, 2]
    assert script.crop(dest / "global.nc", dest) is False   # 已是中国框
    assert not list(dest.glob("*.cropping"))


def test_crop_rejects_a_south_first_grid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = _load(monkeypatch)
    source = tmp_path / "global.nc"
    _write_global(source, lat_north_first=False)
    with pytest.raises(SystemExit, match="不是 ISIMIP 的 0.5° 网格"):
        script.crop(source, tmp_path)


def test_crop_failure_leaves_no_temporary_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = _load(monkeypatch)
    source = tmp_path / "global.nc"
    _write_global(source)
    original = source.read_bytes()

    def broken_write(src: netCDF4.Dataset, name: str, tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(script, "_write_crop", broken_write)
        with pytest.raises(OSError, match="disk full"):
            script.crop(source, tmp_path)
    monkeypatch.setattr(script, "_first_mismatch", lambda source, tmp, name: 1)
    with pytest.raises(SystemExit, match="第 1 个月裁后与原文件不同"):
        script.crop(source, tmp_path)
    assert not list(tmp_path.glob("*.cropping"))
    assert source.read_bytes() == original   # 原地裁剪失败，原文件不动
