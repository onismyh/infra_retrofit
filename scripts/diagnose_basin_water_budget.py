"""逐流域核算：径流 x 0.20  vs  生活耗水 + 灌溉耗水。诊断脚本，不写入 inputs/。

流域索引用 enumerate(codes, start=1)，与 builders/water._basin_runoff_from_file 一致；
zones==0 是流域外（含海洋），必须排除。
"""
from __future__ import annotations

import sys
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(r"D:\000. Paper work\煤电改造claude")
sys.path.insert(0, str(ROOT / "src"))

from coal_retrofit.builders.water import (  # noqa: E402
    SECONDS_PER_YEAR, WATER_WINDOW_BASIS, _basin_zone_grid, _cell_area_m2, _clean_series,
    _nc_path, basin_bias_factors,
)
from coal_retrofit.constants import (  # noqa: E402
    OFFICIAL_BASIN_WATER_1E8_M3, WATER_EXTRACTABLE_FRACTION,
)
from coal_retrofit.paths import ProjectPaths  # noqa: E402

NAMES = {"A": "东北诸河", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
         "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北诸河"}
W = ROOT / "data" / "water"
paths = ProjectPaths(ROOT)


def fn(hydro: str, ssp: str, var: str) -> Path:
    return W / (f"{hydro}_gfdl-esm4_w5e5_{ssp}_2015soc-from-histsoc_default_"
                f"{var}_global_monthly_2015_2100.nc")


def basin_total(path: Path, var: str, zones, codes, area, y0: int, y1: int) -> dict[str, float]:
    """流域年总量，1e8 m3/yr，取 [y0,y1] 窗口月均年化。"""
    with netCDF4.Dataset(_nc_path(path), "r") as ds:
        tv = ds.variables["time"]
        yrs = np.array([d.year for d in netCDF4.num2date(
            tv[:], tv.units, getattr(tv, "calendar", "standard"))])
        m = (yrs >= y0) & (yrs <= y1)
        v = ds.variables[var]
        series = _clean_series(np.asarray(v[m]), getattr(v, "_FillValue", None))
        annual = np.nan_to_num(np.nanmean(series, axis=0)) * area * SECONDS_PER_YEAR / 1000.0
    return {c: float(annual[zones == i].sum()) / 1e8
            for i, c in enumerate(codes, start=1)}


def main() -> None:
    with netCDF4.Dataset(_nc_path(fn("cwatm", "ssp126", "qtot")), "r") as ds:
        lat = np.asarray(ds.variables["lat"][:], float)
        lon = np.asarray(ds.variables["lon"][:], float)
    area = _cell_area_m2(lat)[:, None] * np.ones((1, len(lon)))
    zones, codes = _basin_zone_grid(paths, lat, lon)
    y0, y1 = WATER_WINDOW_BASIS[2050]
    ssp = "ssp126"
    print(f"窗口 {y0}-{y1}  {ssp}  GFDL-ESM4   单位 1e8 m3/yr\n")

    for hydro in ("cwatm", "watergap2-2e"):
        q = basin_total(fn(hydro, ssp, "qtot"), "qtot", zones, codes, area, y0, y1)
        bias = basin_bias_factors(paths, hydro, "gfdl-esm4", zones, codes, area)
        dom = basin_total(fn(hydro, ssp, "pdomuse"), "pdomuse", zones, codes, area, y0, y1)
        if hydro == "watergap2-2e":
            irr = basin_total(fn(hydro, ssp, "pirruse"), "pirruse", zones, codes, area, y0, y1)
        else:
            tot = basin_total(fn(hydro, ssp, "ptotuse"), "ptotuse", zones, codes, area, y0, y1)
            ind = basin_total(fn(hydro, ssp, "pinduse"), "pinduse", zones, codes, area, y0, y1)
            liv = basin_total(fn(hydro, ssp, "pliveuse"), "pliveuse", zones, codes, area, y0, y1)
            irr = {c: max(tot[c] - dom[c] - ind[c] - liv[c], 0.0) for c in codes}

        print(f"===== {hydro} =====")
        print(f"{'流域':<12}{'官方':>7}{'订正径流':>9}{'x0.20':>7}{'生活':>6}{'灌溉':>7}"
              f"{'可用':>8}{'占可提取':>9}  {'现行0.03口径':>12}")
        print("-" * 96)
        s_avail = s_ext = s_old = 0.0
        for c in sorted(codes):
            if c not in OFFICIAL_BASIN_WATER_1E8_M3:
                continue
            qc = q[c] * bias.get(c, 1.0)
            ext = qc * WATER_EXTRACTABLE_FRACTION
            avail = ext - dom[c] - irr[c]
            old = qc * WATER_EXTRACTABLE_FRACTION * (1 - 0.85)
            s_avail += avail; s_ext += ext; s_old += old
            pct = (dom[c] + irr[c]) / ext * 100 if ext > 0 else float("nan")
            flag = "  <-- 负" if avail < 0 else ""
            print(f"{c + ' ' + NAMES.get(c, ''):<12}{OFFICIAL_BASIN_WATER_1E8_M3[c]:>7.0f}"
                  f"{qc:>9.0f}{ext:>7.0f}{dom[c]:>6.0f}{irr[c]:>7.0f}{avail:>8.0f}"
                  f"{pct:>8.0f}%{old:>13.0f}{flag}")
        print(f"{'全国':<12}{sum(OFFICIAL_BASIN_WATER_1E8_M3.values()):>7.0f}"
              f"{'':>9}{s_ext:>7.0f}{'':>6}{'':>7}{s_avail:>8.0f}{'':>9}{s_old:>13.0f}\n")


if __name__ == "__main__":
    main()
