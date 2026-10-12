"""流域偏差校正：各 (水文模型, GCM) 组合的 `historical` 运行对照官方流域水资源总量得到的乘性因子（`basin_bias_factors`），
以及同一水文模型各成员的因子与来源（`member_bias_factors`，没有真实历史运行的成员取其余成员的均值）。

`water_availability.build_water_availability_dataframe` 把因子乘到流域径流上；`scripts/validate_basin_runoff.py` 用同一套函数核对。
"""
from __future__ import annotations

import logging
from pathlib import Path

import netCDF4
import numpy as np

from ..constants import BIAS_BASELINE_WINDOW, OFFICIAL_BASIN_WATER_1E8_M3
from ..paths import ProjectPaths
from .water_grid import _annual_m3, _clean_window, _file_years, _nc_path

logger = logging.getLogger(__name__)

# ISIMIP issue #34（data.isimip.org/issues/34，2023-02）：CWatM 的 `historical` qtot 文件在 GCM 之间错位一格，文件名里的
# GCM 不是实际驱动它的 GCM。ISIMIP 只重发了未来文件（2023-02-16 版，本地即此版），historical 至今是 2022-12-15 的错位版。
# 2026-10-12 用各 GCM 自己的 W5E5 降水异常与径流异常做格点合并相关核定（`scripts/attribute_isimip_forcing.py`；WaterGAP2-2e
# 的 historical 与两家的 ssp126 / ssp370 未来文件全部对号，相关 0.86–0.90）：标 ipsl 的文件是 GFDL 的模拟（qtot 数据与 gfdl
# 文件逐格相同，只差文件头的 GCM 名）、标 mpi 的是 IPSL、标 mri 的是 MPI、标 ukesm 的是 MRI，真正的 UKESM 历史运行服务器上没有。
# 键 (水文模型, 实际驱动 GCM) → 文件名里写的 GCM；None 表示没有真实的历史运行，这时 `member_bias_factors` 取同一水文模型
# 其余成员因子的流域均值（作者决定 2026-10-12）。不在表里的组合按文件名直接取。
HISTORICAL_FILE_GCM: dict[tuple[str, str], str | None] = {
    ("cwatm", "ipsl-cm6a-lr"): "mpi-esm1-2-hr",
    ("cwatm", "mpi-esm1-2-hr"): "mri-esm2-0",
    ("cwatm", "mri-esm2-0"): "ukesm1-0-ll",
    ("cwatm", "ukesm1-0-ll"): None,
}


def has_genuine_history(hydrology_model: str, gcm: str) -> bool:
    """该 (水文模型, GCM) 组合有没有真实的 `historical` 运行（与文件是否已下载无关）。"""
    return HISTORICAL_FILE_GCM.get((hydrology_model, gcm), gcm) is not None


def _historical_source(paths: ProjectPaths, hydrology_model: str, gcm: str) -> Path | None:
    """某个 (水文模型, GCM) 组合的 `historical` qtot 运行文件（若已下载）；没有真实历史运行的组合返回 None。

    两个键缺一不可。ISIMIP3b 用每个 GCM 分别驱动每个水文模型，所以 `historical` 运行是
    按组合存在的，而不是按水文模型。只按水文模型匹配、取排序后的第一个命中，曾对每个
    GCM 都悄悄返回 gfdl-esm4 的运行（它排在最前），这本会把一个 GCM 的偏差因子套用到
    所有 GCM 上，恰好抹平集合本来要度量的 GCM 间离散。文件名里的 GCM 按 `HISTORICAL_FILE_GCM`
    换成实际驱动该 GCM 的那份，因为 CWatM 的文件在服务器上就标错了。
    """
    if not has_genuine_history(hydrology_model, gcm):
        return None
    file_gcm = HISTORICAL_FILE_GCM.get((hydrology_model, gcm), gcm)
    matches = sorted(
        p for p in (paths.data_dir / "water").glob("*historical*qtot*.nc")
        if p.name.startswith(f"{hydrology_model}_{file_gcm}_")
    )
    return matches[0] if matches else None


def _basin_runoff_from_file(
    path: Path, window: tuple[int, int], zones: np.ndarray, codes: list[str], area_m2: np.ndarray
) -> dict[str, float]:
    """在已栅格化的网格上，逐流域求 `window` 内的总径流，m3/yr。"""
    # netCDF4 的 C 层在 Windows 上打不开含非 ASCII 字符的绝对路径。
    with netCDF4.Dataset(_nc_path(path), "r") as ds:
        years = _file_years(ds)
        mask = (years >= window[0]) & (years <= window[1])
        if not mask.any():
            raise ValueError(f"{path.name} covers {years.min()}-{years.max()}, not {window}")
        annual = _annual_m3(_clean_window(ds, "qtot", mask), area_m2)
    return {code: float(annual[zones == idx].sum()) for idx, code in enumerate(codes, start=1)}


def basin_bias_factors(
    paths: ProjectPaths,
    hydrology_model: str,
    gcm: str,
    zones: np.ndarray,
    codes: list[str],
    area_m2: np.ndarray,
) -> dict[str, float]:
    """乘性校正因子：把某个模型的流域径流拉到官方基准上。

    0.5 deg 的全球水文模型即使全国总量正确，也带有很大的区域偏差：在 1956-2014 年，
    WaterGAP2-2e 复现中国总量的误差在 1.2% 以内，却在海河流域偏湿 2.43x、在淮河偏湿
    1.69x——这两个流域承载着 358 GW 煤电，也正是可用水量约束真正起作用的地方。若不
    校正，这一偏差就会被当作水文模型不确定性报告出来。

    该因子对每个 (水文模型, GCM) 组合只估计一次，取自该组合自己的 `historical` 运行，
    并施加到该组合的每一年、每个 SSP 上，因此每个成员的变化率被精确保留，只替换绝对
    水平。按组合而不是按水文模型估计，对集合很重要：历史偏差既是水文模型的属性，同样
    也是驱动 GCM 的属性，把一个 GCM 的因子套用到其他 GCM 上会抹掉真实存在的 GCM 间
    离散。若该组合没有历史运行文件（未下载，或 `HISTORICAL_FILE_GCM` 记为没有真实运行），则对每个流域都返回 1.0；
    用其余成员的均值补全在 `member_bias_factors` 里做。
    """
    source = _historical_source(paths, hydrology_model, gcm)
    if source is None:
        return {code: 1.0 for code in codes}
    modelled = _basin_runoff_from_file(source, BIAS_BASELINE_WINDOW, zones, codes, area_m2)
    factors: dict[str, float] = {}
    for code in codes:
        official_m3 = OFFICIAL_BASIN_WATER_1E8_M3.get(code, 0.0) * 1e8
        model_m3 = modelled.get(code, 0.0)
        factors[code] = official_m3 / model_m3 if model_m3 > 0 and official_m3 > 0 else 1.0
    return factors


def member_bias_factors(
    paths: ProjectPaths,
    hydrology_model: str,
    gcms: list[str],
    zones: np.ndarray,
    codes: list[str],
    area_m2: np.ndarray,
) -> dict[str, tuple[dict[str, float], str]]:
    """同一水文模型各 GCM 的流域偏差因子及其来源：历史文件名、均值说明或 "none"。

    有真实历史运行的成员按自己的运行算（`basin_bias_factors`）。`HISTORICAL_FILE_GCM` 记为没有真实运行的成员
    （cwatm|ukesm1-0-ll），取其余有真实运行成员因子的流域均值：GCM 专属的那部分偏差无从得知，均值是对"该水文模型
    配任意 GCM"偏差的估计，且全集合都在校正状态下（作者决定 2026-10-12；取 1.0 会把它单独留在未校正状态，全国
    可用量比均值低 15%）。文件未下载的成员保持 1.0、记 "none"，也不参与均值，缺文件不被均值掩盖。
    """
    factors: dict[str, tuple[dict[str, float], str]] = {}
    for gcm in gcms:
        source = _historical_source(paths, hydrology_model, gcm)
        if source is not None:
            factors[gcm] = (basin_bias_factors(paths, hydrology_model, gcm, zones, codes, area_m2), source.name)
    donors = [gcm for gcm in gcms if gcm in factors]
    for gcm in gcms:
        if gcm in factors:
            continue
        if not has_genuine_history(hydrology_model, gcm) and donors:
            mean = {code: float(np.mean([factors[donor][0][code] for donor in donors])) for code in codes}
            factors[gcm] = (mean, f"mean of {hydrology_model} {', '.join(donors)}")
        else:
            logger.warning("%s|%s: no historical qtot file, bias factor left at 1.0 (uncorrected)", hydrology_model, gcm)
            factors[gcm] = ({code: 1.0 for code in codes}, "none")
    return factors
