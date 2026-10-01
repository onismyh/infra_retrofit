"""节点余量要扣的其他用户耗水：各气候成员、各规划年、各一级流域的生活与灌溉耗水（ISIMIP3b）。

节点可用量 = max(径流 x 0.20 − 生活耗水 − 灌溉耗水, 煤电存量)（`optimization/water_access._water_available_by_node`，
docs/工业部门参数溯源.md §六 第 1 条、§七）。本模块只求这两项耗水的流域合计，写 `inputs/water_basin_use.csv`；
`water_availability.csv` 不动。

变量。两个水文模型发布的变量不同（`scripts/download_isimip_water_use.py`）：

    CWatM         pdomuse、pinduse、pliveuse、ptotuse，没有 pirruse：灌溉 = ptotuse − pdomuse − pinduse − pliveuse
    WaterGAP2-2e  pdomuse、pirruse

都是潜在耗水（p 前缀），取自 2015soc-from-histsoc 运行：社会经济驱动冻结在 2015 年，生活耗水基本不随年份变，
灌溉需水随气候变。不扣工业（它是决策主体，取水进流域取水指标），也不扣畜牧（两个模型同口径，只扣生活与灌溉）。
不做偏差校正：没有官方的分流域耗水可对（水资源公报只给全国分部门耗水率）。已知 ISIMIP 的华北灌溉需水比公报
高 2-3 倍（`water_quota` 模块说明），所以华北的扣减偏大、余量偏紧。

季节。枯水期耗水取与径流枯水期同一个三个月窗口（`water._basin_dry_window`，按该成员自己的 qtot 在流域合计上选），
年化后与 `water_availability.csv` 的 `dry_season_water_m3_per_year` 相减；全年口径取十年均值。偏差因子只乘在径流上，
不改变窗口的选择，所以不需要历史期文件。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import netCDF4
import numpy as np
import pandas as pd

from ..artifacts import write_csv
from ..constants import PLANNING_YEARS
from ..paths import ProjectPaths
from .water import (
    WATER_WINDOW_BASIS,
    _annual_and_seasonal,
    _basin_dry_window,
    _basin_zone_grid,
    _cell_area_m2,
    _clean_series,
    _file_years,
    _nc_path,
    _resolve_source_path,
)

logger = logging.getLogger(__name__)

# 每个水文模型要读的耗水变量；第一个总是生活耗水。
WATER_USE_VARIABLES: Final[dict[str, tuple[str, ...]]] = {
    "cwatm": ("pdomuse", "ptotuse", "pinduse", "pliveuse"),
    "watergap2-2e": ("pdomuse", "pirruse"),
}
COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id", "hydrology_model", "gcm", "ssp", "planning_year", "basin_code",
    "domestic_m3_per_year", "irrigation_m3_per_year",
    "dry_season_domestic_m3_per_year", "dry_season_irrigation_m3_per_year",
    "dry_season_start_month", "unit", "source", "year_basis",
)
_QTOT_MARKER = "_default_qtot_global_"


def water_use_file(qtot_file: str, variable: str) -> str:
    """成员 qtot 文件名换成同一成员 `variable` 的文件名（两者只差变量名）。"""
    if _QTOT_MARKER not in qtot_file:
        raise ValueError(f"{qtot_file} is not a qtot member file")
    return qtot_file.replace(_QTOT_MARKER, f"_default_{variable}_global_")


def _window(ds: netCDF4.Dataset, variable: str, years: np.ndarray, planning_year: int) -> np.ndarray:
    """`planning_year` 对应十年窗口的逐月序列（kg m-2 s-1），填充值换成 NaN。"""
    start, end = WATER_WINDOW_BASIS[planning_year]
    mask = (years >= start) & (years <= end)
    if not mask.any():
        raise ValueError(f"{variable} covers {years.min()}-{years.max()}, not {start}-{end}")
    var = ds.variables[variable]
    return _clean_series(np.asarray(var[mask]), getattr(var, "_FillValue", None))


def build_water_use_dataframe(paths: ProjectPaths, scenarios: pd.DataFrame) -> pd.DataFrame:
    """每个 (成员, 规划年, 一级流域) 一行：生活、灌溉耗水的全年值与枯水期年化值（m3/yr），及枯水期起始月（1-12）。

    `scenarios` 是 `water_scenarios.csv` 的成员表（`source` 列为该成员的 qtot 文件）。缺任何一个耗水文件就报错，
    不拿别的 GCM 顶替。
    """
    rows: list[dict[str, object]] = []
    for member in scenarios.itertuples(index=False):
        hydrology = str(member.hydrology_model)
        if hydrology not in WATER_USE_VARIABLES:
            raise ValueError(f"no water-use variables known for hydrology model {hydrology!r}")
        variables = WATER_USE_VARIABLES[hydrology]
        qtot_path = Path(_resolve_source_path(paths, str(member.source)))
        use_paths = {v: qtot_path.with_name(water_use_file(qtot_path.name, v)) for v in variables}
        missing = sorted(p.name for p in use_paths.values() if not p.exists())
        if missing:
            raise FileNotFoundError(
                f"{member.scenario_id}: missing {missing} in {qtot_path.parent}; run scripts/download_isimip_water_use.py"
            )

        # 枯水期窗口按本成员的径流在流域合计上选，与 `water_availability.csv` 的枯水期列同一段代码。
        with netCDF4.Dataset(_nc_path(qtot_path), "r") as ds:
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            area_m2 = _cell_area_m2(lat)[:, None] * np.ones((1, len(lon)))
            zones, codes = _basin_zone_grid(paths, lat, lon)
            cells = {code: zones == idx for idx, code in enumerate(codes, start=1)}
            years = _file_years(ds)
            dry: dict[tuple[int, str], int] = {}
            for planning_year in PLANNING_YEARS:
                _, rolling = _annual_and_seasonal(_window(ds, "qtot", years, planning_year), area_m2)
                for code in codes:
                    dry[(planning_year, code)] = _basin_dry_window(rolling, cells[code])[0]

        # volumes[变量][(规划年, 流域)] = (全年, 枯水期年化)，m3/yr
        volumes: dict[str, dict[tuple[int, str], tuple[float, float]]] = {}
        for variable, path in use_paths.items():
            with netCDF4.Dataset(_nc_path(path), "r") as ds:
                same_grid = (np.array_equal(np.asarray(ds.variables["lat"][:], dtype=np.float64), lat)
                             and np.array_equal(np.asarray(ds.variables["lon"][:], dtype=np.float64), lon))
                if not same_grid:
                    raise ValueError(f"{path.name} is not on the grid of {qtot_path.name}")
                years = _file_years(ds)
                per_basin: dict[tuple[int, str], tuple[float, float]] = {}
                for planning_year in PLANNING_YEARS:
                    annual, rolling = _annual_and_seasonal(_window(ds, variable, years, planning_year), area_m2)
                    for code in codes:
                        dry_volume = rolling[dry[(planning_year, code)]]
                        per_basin[(planning_year, code)] = (
                            float(annual[cells[code]].sum()), float(dry_volume[cells[code]].sum())
                        )
                volumes[variable] = per_basin

        source = str(member.source).replace(_QTOT_MARKER, f"_default_{{{','.join(variables)}}}_global_")
        for planning_year in PLANNING_YEARS:
            start, end = WATER_WINDOW_BASIS[planning_year]
            for code in codes:
                key = (planning_year, code)
                domestic = volumes["pdomuse"][key]
                if "pirruse" in volumes:
                    irrigation = volumes["pirruse"][key]
                else:  # CWatM：流域合计后用总耗水减去其余三项
                    rest = [volumes[v][key] for v in ("pdomuse", "pinduse", "pliveuse")]
                    raw = tuple(volumes["ptotuse"][key][i] - sum(r[i] for r in rest) for i in (0, 1))
                    if min(raw) < 0.0:
                        logger.warning("%s %d %s: ptotuse below its components by %.3g m3/yr; irrigation set to 0",
                                       member.scenario_id, planning_year, code, -min(raw))
                    irrigation = (max(raw[0], 0.0), max(raw[1], 0.0))
                rows.append({
                    "scenario_id": str(member.scenario_id),
                    "hydrology_model": hydrology,
                    "gcm": str(member.gcm),
                    "ssp": str(member.ssp),
                    "planning_year": int(planning_year),
                    "basin_code": code,
                    "domestic_m3_per_year": domestic[0],
                    "irrigation_m3_per_year": irrigation[0],
                    "dry_season_domestic_m3_per_year": domestic[1],
                    "dry_season_irrigation_m3_per_year": irrigation[1],
                    "dry_season_start_month": dry[key] + 1,
                    "unit": "m3/yr",
                    "source": source,
                    "year_basis": f"{start}-{end}",
                })
    return pd.DataFrame(rows, columns=list(COLUMNS))


def write_water_use(paths: ProjectPaths) -> pd.DataFrame:
    """按 `inputs/water_scenarios.csv` 的成员写 `inputs/water_basin_use.csv`，成员与 `water_availability.csv` 同一套。"""
    scenarios = pd.read_csv(paths.inputs_dir / "water_scenarios.csv")
    frame = build_water_use_dataframe(paths, scenarios)
    write_csv(frame, paths.inputs_dir / "water_basin_use.csv")
    return frame
