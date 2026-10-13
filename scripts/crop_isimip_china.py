#!/usr/bin/env python3
"""把 ISIMIP3b 全球 0.5° 逐月文件裁到中国框：lat[66:180]、lon[500:640]（56.75–0.25°N、70.25–139.75°E，114 × 140 格）。

`data/water/` 里的文件都是这样裁过的（文件属性 `history_crop`），两个下载脚本（`download_isimip_water_use.py`、
`download_isimip_qtot.py`）下的是全球文件，下载后用本脚本裁。不改数值：数据变量按原 dtype 与 `_FillValue`
原样拷，全局与变量属性照抄，另加 `history_crop`；容器为 NETCDF4_CLASSIC，zlib 5 + shuffle，一个月一块。写到临时文件、
逐月与原文件逐格比对相同后才换成正式文件名，中途出错就删掉临时文件，所以缺省的原地裁剪（输出目录即输入所在目录）
失败时原文件还在。已裁过的文件（网格已是 114 × 140）跳过，给了 `--dest` 也不拷过去。

用法：
  python scripts/crop_isimip_china.py <文件或目录>... [--dest 输出目录]
目录取其中全部 `*.nc`；不给 `--dest` 就原地替换。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import netCDF4
import numpy as np

LAT, LON = slice(66, 180), slice(500, 640)   # 0.5° 全球网格上的中国框
CROP_NOTE = "cropped to lat[66:180] lon[500:640] of the global file (China box)"
GLOBAL_SHAPE = (360, 720)
COORDINATES = ("time", "lon", "lat")
MONTHS_PER_BLOCK = 120   # 一次读写 120 个月的中国框：120 × 114 × 140 × 4 B ≈ 7.7 MB

logger = logging.getLogger(__name__)


def _data_variable(ds: netCDF4.Dataset) -> str:
    """文件里唯一的非坐标变量（qtot、pdomuse 等）。"""
    names = [name for name in ds.variables if name not in COORDINATES]
    if len(names) != 1:
        raise SystemExit(f"{ds.filepath()} 应只有一个数据变量，实有 {names}")
    return names[0]


def _copy_attributes(src: netCDF4.Variable | netCDF4.Dataset, dst: netCDF4.Variable | netCDF4.Dataset) -> None:
    for key in src.ncattrs():
        if key != "_FillValue":   # 只能在建变量时给
            dst.setncattr(key, src.getncattr(key))


def _write_crop(src: netCDF4.Dataset, name: str, tmp: Path) -> None:
    """把 `src` 的中国框写到 `tmp`：坐标、数据变量与全部属性，数据按原始值（不掩膜、不缩放）逐块拷。"""
    var = src.variables[name]
    with netCDF4.Dataset(tmp, "w", format="NETCDF4_CLASSIC") as dst:
        _copy_attributes(src, dst)
        dst.setncattr("history_crop", CROP_NOTE)
        dst.createDimension("time", None)
        dst.createDimension("lon", LON.stop - LON.start)
        dst.createDimension("lat", LAT.stop - LAT.start)
        for coord, part in (("time", slice(None)), ("lon", LON), ("lat", LAT)):
            out = dst.createVariable(coord, src.variables[coord].dtype, (coord,))
            _copy_attributes(src.variables[coord], out)
            out[:] = src.variables[coord][part]
        out = dst.createVariable(name, var.dtype, ("time", "lat", "lon"), zlib=True, complevel=5, shuffle=True,
                                 chunksizes=(1, LAT.stop - LAT.start, LON.stop - LON.start),
                                 fill_value=getattr(var, "_FillValue", None))
        _copy_attributes(var, out)
        var.set_auto_maskandscale(False)
        out.set_auto_maskandscale(False)
        for start in range(0, var.shape[0], MONTHS_PER_BLOCK):
            block = var[start:start + MONTHS_PER_BLOCK, LAT, LON]
            out[start:start + block.shape[0]] = block


def _first_mismatch(source: Path, tmp: Path, name: str) -> int | None:
    """`tmp` 与 `source` 的中国框逐块比对（原始值），返回第一个不同的月序号，全同返回 None。"""
    with netCDF4.Dataset(source) as src, netCDF4.Dataset(tmp) as dst:
        before, after = src.variables[name], dst.variables[name]
        before.set_auto_maskandscale(False)
        after.set_auto_maskandscale(False)
        if after.shape[0] != before.shape[0]:
            return min(after.shape[0], before.shape[0])
        for start in range(0, before.shape[0], MONTHS_PER_BLOCK):
            a = before[start:start + MONTHS_PER_BLOCK, LAT, LON]
            b = after[start:start + MONTHS_PER_BLOCK]
            if not np.array_equal(a, b, equal_nan=True):
                differs = ~((a == b) | (np.isnan(a) & np.isnan(b)))
                return start + int(np.flatnonzero(differs.reshape(len(a), -1).any(axis=1))[0])
    return None


def crop(source: Path, dest_dir: Path) -> bool:
    """裁一个文件到 `dest_dir/<同名>`；已是中国框的跳过并返回 False。"""
    with netCDF4.Dataset(source) as src:
        name = _data_variable(src)
        shape = src.variables[name].shape[1:]
        if shape != GLOBAL_SHAPE:
            if shape == (LAT.stop - LAT.start, LON.stop - LON.start):
                logger.info("%s 已是中国框，跳过", source.name)
                return False
            raise SystemExit(f"{source.name} 的网格 {shape} 不是 0.5° 全球网格 {GLOBAL_SHAPE}")
        lat, lon = src.variables["lat"][:], src.variables["lon"][:]
        if (float(lat[LAT][0]), float(lat[LAT][-1])) != (56.75, 0.25) or (
                float(lon[LON][0]), float(lon[LON][-1])) != (70.25, 139.75):
            raise SystemExit(f"{source.name} 的经纬度不是 ISIMIP 的 0.5° 网格（自北向南、自西向东）")
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest_dir / (source.name + ".cropping")
        try:
            _write_crop(src, name, tmp)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        month = _first_mismatch(source, tmp, name)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if month is not None:
        tmp.unlink()
        raise SystemExit(f"{source.name} 第 {month} 个月裁后与原文件不同")
    tmp.replace(dest_dir / source.name)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="netCDF 文件或含 *.nc 的目录")
    parser.add_argument("--dest", type=Path, help="输出目录；缺省原地替换")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    files = [f for p in args.paths for f in (sorted(p.glob("*.nc")) if p.is_dir() else [p])]
    cropped = 0
    for source in files:
        dest_dir = args.dest or source.parent
        if crop(source, dest_dir):
            cropped += 1
            logger.info("%s 已裁剪，%.1f MB", source.name, (dest_dir / source.name).stat().st_size / 1e6)
    print(f"裁剪 {cropped} 个，跳过 {len(files) - cropped} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
