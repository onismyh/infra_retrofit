#!/usr/bin/env python3
"""把 `data/water/` 里的每个 qtot 文件归到实际驱动它的 GCM：径流异常与各 GCM 自己的降水异常做格点合并相关。

为什么需要：ISIMIP issue #34（data.isimip.org/issues/34）——CWatM 的 `historical` qtot 文件在 GCM 之间错位一格，文件名里的
GCM 不是实际驱动它的 GCM，ISIMIP 只重发了未来文件，historical 没有重发。偏差因子按 (水文模型, GCM) 从各自的历史运行估计
（`builders/water_bias.basin_bias_factors`），文件标错就把一个 GCM 的偏差套到了另一个成员上。2026-10-12 的核定结果写在
`builders/water_bias.HISTORICAL_FILE_GCM`：标 ipsl 的是 GFDL 的模拟（qtot 数据逐格相同，只差文件头的 GCM 名）、标 mpi 的是 IPSL、标 mri 的是 MPI、
标 ukesm 的是 MRI，真正的 UKESM 历史运行没有；WaterGAP2-2e 的 historical 与两家的 ssp126 / ssp370 未来文件全部对号（相关 0.86–0.90，
其余列 |r| < 0.17）。

方法：在中国框（0.5° 全球网格的 lat[66:180]、lon[500:640]，与 `data/water/` 的裁剪一致）内，取窗口内逐月径流与逐月降水
（日值按月平均），逐网格扣去各自日历月的多年均值得到异常，在"有径流的陆地格 × 月"上合并算相关。径流只跟随自己的强迫，
另一 GCM 的降水是独立的实现，所以同一行里只有一列相关高。行是 qtot 文件，列是降水所属的 GCM，`*` 标最高列，
最高相关低于 0.5 记"无匹配"。

降水强迫是 ISIMIP3b 的 W5E5 偏差订正日值（每个文件约 0.8–1.3 GB，不入库），放在任一目录传给本脚本：
  https://files.isimip.org/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/<period>/<GCM>/
    <gcm>_<member>_w5e5_<period>_pr_global_daily_<y0>_<y1>.nc
  period 为 historical（2011_2014）或 ssp126 / ssp370（2015_2020）；member 一般是 r1i1p1f1，UKESM1-0-LL 是 r1i1p1f2。
用法：
  python scripts/attribute_isimip_forcing.py <pr_dir> [--period historical|ssp126|ssp370] [--hydrology cwatm|watergap2-2e]
                                              [--years Y0 Y1]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import netCDF4
import numpy as np
from _bootstrap import REPO_ROOT

WATER_DIR = REPO_ROOT / "data" / "water"
LAT, LON = slice(66, 180), slice(500, 640)   # 0.5° 全球网格上的中国框
DEFAULT_YEARS = {"historical": (2011, 2014)}  # 其余（ssp*）取 2015–2020
MATCH_THRESHOLD = 0.5
_GCM_OF_QTOT = re.compile(r"(?:cwatm|watergap2-2e)_([a-z0-9-]+)_w5e5_")
_GCM_OF_PR = re.compile(r"([a-z0-9-]+)_r\d+i\d+p\d+f\d+_w5e5_")


def _gcm_in_name(pattern: re.Pattern[str], name: str) -> str:
    """文件名里的 GCM；不合 ISIMIP 命名就报错，而不是静默跳过。"""
    match = pattern.match(name)
    if match is None:
        raise SystemExit(f"文件名不合 ISIMIP 命名，认不出 GCM：{name}")
    return match.group(1)


def _year_month(ds: netCDF4.Dataset) -> list[tuple[int, int]]:
    time = ds.variables["time"]
    dates = netCDF4.num2date(time[:], time.units, getattr(time, "calendar", "standard"))
    return [(int(d.year), int(d.month)) for d in dates]


def monthly_precipitation(path: Path, keys: list[tuple[int, int]]) -> np.ndarray:
    """日值降水按月平均，裁到中国框，(月, lat, lon)。"""
    with netCDF4.Dataset(str(path)) as ds:
        year_month = _year_month(ds)
        var = ds.variables["pr"]
        var.set_auto_maskandscale(False)
        out = np.full((len(keys), LAT.stop - LAT.start, LON.stop - LON.start), np.nan)
        for k, key in enumerate(keys):
            days = [i for i, ym in enumerate(year_month) if ym == key]
            if not days:
                raise SystemExit(f"{path.name} 没有 {key[0]}-{key[1]:02d}")
            block = np.asarray(var[days[0]:days[-1] + 1, LAT, LON], dtype=np.float64)
            block[block > 1e19] = np.nan
            out[k] = block.mean(axis=0)
    return out


def monthly_runoff(path: Path, keys: list[tuple[int, int]]) -> np.ndarray:
    """逐月径流，(月, lat, lon)；全球文件裁到中国框，已裁剪的文件原样读。"""
    with netCDF4.Dataset(str(path)) as ds:
        year_month = _year_month(ds)
        rows = [year_month.index(key) for key in keys]
        var = ds.variables["qtot"]
        var.set_auto_maskandscale(False)
        box = (LAT, LON) if var.shape[1:] == (360, 720) else ()
        data = np.asarray(var[(slice(rows[0], rows[-1] + 1), *box)], dtype=np.float64)
    data[data > 1e19] = np.nan
    return data


def anomalies(field: np.ndarray) -> np.ndarray:
    """逐网格扣去各自日历月在窗口内的均值。"""
    out = np.empty_like(field)
    for month in range(12):
        out[month::12] = field[month::12] - field[month::12].mean(axis=0)
    return out


def attribute(pr_dir: Path, period: str, hydrology: str, years: tuple[int, int]) -> None:
    keys = [(year, month) for year in range(years[0], years[1] + 1) for month in range(1, 13)]
    pr_files = sorted(pr_dir.glob(f"*_w5e5_{period}_pr_global_daily_*.nc"))
    qtot_files = sorted(WATER_DIR.glob(f"{hydrology}_*_{period}_*qtot*.nc"))
    if not pr_files or not qtot_files:
        raise SystemExit(f"文件不足：降水 {len(pr_files)} 个、{hydrology} {period} qtot {len(qtot_files)} 个")
    gcms = [_gcm_in_name(_GCM_OF_PR, p.name) for p in pr_files]
    pr_anomaly = {gcm: anomalies(monthly_precipitation(p, keys)) for gcm, p in zip(gcms, pr_files, strict=True)}
    runoff = {p.name: monthly_runoff(p, keys) for p in qtot_files}
    # 陆地格：所有径流文件与所有降水文件都没有缺测、且平均有径流的格，这样每一行的相关都在同一组格上算。
    land = np.nanmean(next(iter(runoff.values())), axis=0) > 1e-7
    for field in (*runoff.values(), *pr_anomaly.values()):
        land &= np.isfinite(field).all(axis=0)
    print(f"\n{hydrology} {period} {years[0]}–{years[1]}（{len(keys)} 个月，{int(land.sum())} 个有径流的陆地格）；"
          f"行 = qtot 文件，列 = 降水所属 GCM，* = 最高列")
    print(" " * 72 + "".join(f"{gcm[:12]:>14s}" for gcm in gcms))
    for name, field in runoff.items():
        anomaly = anomalies(field)[:, land].ravel()
        corr = [float(np.corrcoef(pr_anomaly[gcm][:, land].ravel(), anomaly)[0, 1]) for gcm in gcms]
        if not np.all(np.isfinite(corr)):
            raise RuntimeError(f"{name}: 相关为 NaN（某列异常方差为零或仍有缺测）")
        best = int(np.argmax(corr))
        labelled = _gcm_in_name(_GCM_OF_QTOT, name)
        if corr[best] < MATCH_THRESHOLD:
            verdict = "无匹配"
        else:
            verdict = "对号" if gcms[best] == labelled else f"标错 -> 实际是 {gcms[best]}"
        cells = "".join(f"{value:13.3f}{'*' if i == best else ' '}" for i, value in enumerate(corr))
        print(f"{name[:70]:70s}  {cells}   {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pr_dir", type=Path, help="ISIMIP3b 偏差订正日值降水文件所在目录")
    parser.add_argument("--period", default="historical", choices=["historical", "ssp126", "ssp370"])
    parser.add_argument("--hydrology", choices=["cwatm", "watergap2-2e"], help="缺省两个水文模型都算")
    parser.add_argument("--years", nargs=2, type=int, metavar=("Y0", "Y1"),
                        help="窗口首末年，缺省 historical 2011 2014、ssp 2015 2020")
    args = parser.parse_args()
    years = tuple(args.years) if args.years else DEFAULT_YEARS.get(args.period, (2015, 2020))
    for hydrology in ([args.hydrology] if args.hydrology else ["cwatm", "watergap2-2e"]):
        attribute(args.pr_dir, args.period, hydrology, (int(years[0]), int(years[1])))


if __name__ == "__main__":
    main()
