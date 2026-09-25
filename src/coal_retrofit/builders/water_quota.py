"""按官方用水总量控制指标口径的流域水量预算。

用中国自己的水量分配工具，取代由径流推出的预算（`qtot x 0.20 x (1 - 0.85)`）。原因与全部
出处见 `constants_water_quota`。口径是水资源公报所定义的用水量——毛取水量，直流冷却凝汽器
过流也计在内——所以这些上限约束的模型项是电厂取水，而不是 `quota_intensity_m3_per_mwh`
中的定额水量；后者不含这部分过流，会把机队对上限的占用低估约五倍。

分摊问题。国办发〔2013〕2号按省设定用水上限；模型按水资源一级区施加约束。不存在官方的
流域表（附件1 是一个残差为零的闭合省级划分），而且省并不嵌套在流域之内。因此每个省的
上限必须拆分到它所跨的各流域上，而拆分需要一个空间分摊键。

考虑过三种分摊键：

    面积            不对：用水在空间上远非均匀。内蒙古的面积大部分在西北诸河区，但其
                    用水大部分在黄河。
    人口            出于同一原因反过来也不对：农业占全国用水的 62%。
    网格化需求      本模块采用。ISIMIP `ptotuse` 是空间显式的，且已在模型自己的 0.5 度
                    网格上。

ISIMIP 需求已知的水平偏差（相对水资源公报，它把华北灌溉高估了 2-3x）不会传递过来：该场
只以省内份额的形式进入，所以一个省内共同的任何因子都会被精确抵消。会传递过来的是
*在同一个省内部*随流域变化的偏差；这是真实存在的残余不确定性，也正因此 `basin_caps`
可以用 `weight="area"` 重跑，作为敏感性分析。

权重场被刻意固定（一个 SSP、一个十年），使分摊键是一项地理属性，而不是情景变量。更换情景
只能通过气候信号改变水量预算，绝不能靠悄悄重画省到流域的拆分来改变。
"""
from __future__ import annotations

import logging
from typing import Final, Literal

import netCDF4
import numpy as np
import pandas as pd


from ..constants_water_quota import (
    BASIN_NAMES_ZH,
    BASIN_WITHDRAWAL_2025_1E8_M3,
    NATIONAL_ONCE_THROUGH_WITHDRAWAL_1E8_M3,
    PROVINCE_WATER_CAP_1E8_M3,
    PROVINCES_WITHOUT_CAP,
    province_cap,
)
from ..paths import ProjectPaths
from .water import (
    SECONDS_PER_YEAR,
    _basin_zone_grid,
    _cell_area_m2,
    _clean_series,
    _nc_path,
    _province_zone_grid,
)

logger = logging.getLogger(__name__)

# 固定的分摊键：一个水文模型、一个 SSP、一个十年。见模块 docstring。
WEIGHT_FILE: Final[str] = (
    "cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_"
    "ptotuse_global_monthly_2015_2100.nc"
)
WEIGHT_VARIABLE: Final[str] = "ptotuse"
WEIGHT_WINDOW: Final[tuple[int, int]] = (2021, 2030)

WeightKind = Literal["demand", "area"]


def _short_province_name(full_name: str) -> str | None:
    """把 `provinces.shp` 的全称映射到上限表所用的简称。

    附件1 中的每个简称都是其 shapefile 全称的前缀（内蒙古自治区 -> 内蒙古，
    新疆维吾尔自治区 -> 新疆），所以最长前缀匹配是精确的。对没有上限的三个地区返回 None。
    """
    candidates = [name for name in PROVINCE_WATER_CAP_1E8_M3 if full_name.startswith(name)]
    if candidates:
        return max(candidates, key=len)
    if any(full_name.startswith(name) for name in PROVINCES_WITHOUT_CAP):
        return None
    raise KeyError(f"province {full_name!r} matches no entry in PROVINCE_WATER_CAP_1E8_M3")


def _weight_grid(paths: ProjectPaths, kind: WeightKind) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回分摊键的 (weight, lat, lon)，位于 ISIMIP 0.5 度网格上。"""
    path = paths.data_dir / "water" / WEIGHT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_isimip_water_use.py first."
        )
    with netCDF4.Dataset(_nc_path(path), "r") as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        area = _cell_area_m2(lat)[:, None] * np.ones((1, len(lon)))
        if kind == "area":
            return area, lat, lon
        time_var = dataset.variables["time"]
        years = np.array([
            date.year for date in netCDF4.num2date(
                time_var[:], time_var.units, getattr(time_var, "calendar", "standard")
            )
        ])
        window = (years >= WEIGHT_WINDOW[0]) & (years <= WEIGHT_WINDOW[1])
        variable = dataset.variables[WEIGHT_VARIABLE]
        series = _clean_series(np.asarray(variable[window]), getattr(variable, "_FillValue", None))
        weight = np.nan_to_num(np.nanmean(series, axis=0)) * area * SECONDS_PER_YEAR / 1000.0
    return np.clip(weight, 0.0, None), lat, lon


def province_basin_shares(
    paths: ProjectPaths, kind: WeightKind = "demand"
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """每个省的用水落在各流域中的份额。

    位于省内、但不在任何流域多边形内的网格（沿海边缘、多边形缝隙）没有所属流域，在归一化
    之前被剔除，这相当于把它们按比例重新分配到该省的各流域上——另一种做法是直接丢掉它们的
    水量，那会悄悄缩小全国上限。

    Returns:
        (shares, unassigned)：对每个有任何权重的省，`shares[province][basin]` 在各流域上
        求和为 1.0；`unassigned` 是各省落在所有流域之外的权重比例，用于报告。
    """
    weight, lat, lon = _weight_grid(paths, kind)
    province_zones, province_names = _province_zone_grid(paths, lat, lon)
    basin_zones, basin_codes = _basin_zone_grid(paths, lat, lon)

    shares: dict[str, dict[str, float]] = {}
    unassigned = np.zeros(len(province_names), dtype=np.float64)
    for index, full_name in enumerate(province_names, start=1):
        short = _short_province_name(full_name)
        if short is None:
            continue
        mask = province_zones == index
        total = float(weight[mask].sum())
        if total <= 0.0:
            logger.warning("province %s has zero apportionment weight; falling back to area", short)
            area_weight, _, _ = _weight_grid(paths, "area")
            weight_here, total = area_weight, float(area_weight[mask].sum())
        else:
            weight_here = weight
        inside = mask & (basin_zones > 0)
        unassigned[index - 1] = (total - float(weight_here[inside].sum())) / total if total else 0.0
        assigned = float(weight_here[inside].sum())
        if assigned <= 0.0:
            raise ValueError(f"province {short} has no cell inside any basin polygon")
        by_basin = {
            code: float(weight_here[inside & (basin_zones == basin_index)].sum()) / assigned
            for basin_index, code in enumerate(basin_codes, start=1)
        }
        # 跨松花江/辽河分界的省已经落在合并后的代码 A 中，因为 `basin_l1.gpkg`
        # 存储的就是合并后的流域；这里无需再归并。
        shares[short] = {code: value for code, value in by_basin.items() if value > 0.0}
    return shares, unassigned


def basin_caps(
    paths: ProjectPaths, year: int, kind: WeightKind = "demand"
) -> dict[str, float]:
    """分摊到一级流域的官方用水总量控制指标，亿 m3/yr，取水口径。"""
    shares, _ = province_basin_shares(paths, kind)
    caps: dict[str, float] = {code: 0.0 for code in BASIN_NAMES_ZH}
    for province, by_basin in shares.items():
        cap = province_cap(province, year)
        for code, share in by_basin.items():
            caps[code] = caps.get(code, 0.0) + cap * share
    return caps


def _once_through_component(plants: pd.DataFrame, capture: bool) -> pd.Series:
    """每个 hub 的淡水直流冷却取水强度，m3/MWh，标定前。

    直接读取 `builders/plants.finalize_water_intensities` 写出的列，即逐机组的精确差值
    `_intensity_all - _intensity_fresh`。不要用 `dominant_combustion` x
    `capacity_mw_once_through` 重建它：hub 的主导蒸汽参数往往不是其直流冷却机组的蒸汽
    参数，那种重建在全机队合计上错了 24%（404 vs 530 1e8 m3/yr），而且悄无声息，因为
    误差只有在截断（clip）之后才是单向的。
    """
    column = ("once_through_withdrawal_ccs_intensity_m3_per_mwh" if capture
              else "once_through_withdrawal_intensity_m3_per_mwh")
    if column not in plants.columns:
        raise KeyError(
            f"plants.csv lacks {column}. Rebuild it: python scripts/build_plant_inputs.py --hubs"
        )
    return plants[column].astype(float)


def once_through_withdrawal(plants: pd.DataFrame, generation_mwh: pd.Series) -> pd.Series:
    """每个 hub 的淡水直流冷却取水量，m3/yr，按模型原始（美国）强度计。

    从 hub 的综合取水中单独拆出来，是因为只有直流冷却部分需要重新标定：
    `WATER_INTENSITY_BY_TECH_M3_PER_MWH` 的循环冷却与空冷行已与 Macknick 交叉核对，吻合到
    5-14% 以内，并与中国实际相符；与水资源公报相差两倍的正是直流冷却行。

    海水冷却 hub 返回零：公报不含海水直接利用量，`builders/plants.py` 也已把它们的凝汽器
    过流置零，所以标定两侧排除的是同一批机组。
    """
    return generation_mwh * _once_through_component(plants, capture=False)


def calibrated_withdrawal_intensities(
    plants: pd.DataFrame, generation_mwh: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, float]:
    """水资源公报尺度上的逐 hub 取水强度，m3/MWh。

    返回 (base, capture, air_base, air_capture, factor)。由于

        calibrated = blended + once_through_component x (k - 1)

    一旦 `k` 已知，标定就与发电量无关，正是这一点使它能作用于强度，进而作用于逐路径矩阵。

    两种空冷变体完全不含直流冷却项：改造后的凝汽器没有过流，所以按取水口径，空冷改造去掉
    的是 ~90 m3/MWh，而按耗水口径只去掉 ~1 m3/MWh。这个差距正是流域上限必须看到逐路径
    矩阵、而不能只看逐 hub 比值的全部原因，也是这里每一列都从 `plants.csv` 读取（按各机组
    自己的蒸汽参数加权），而不是按 hub 的主导蒸汽参数重建的原因。
    """
    factor = once_through_calibration(plants, generation_mwh)
    base = (plants["withdrawal_intensity_m3_per_mwh"].astype(float)
            + _once_through_component(plants, capture=False) * (factor - 1.0)).clip(lower=0.0)
    capture = (plants["withdrawal_ccs_intensity_m3_per_mwh"].astype(float)
               + _once_through_component(plants, capture=True) * (factor - 1.0)).clip(lower=0.0)
    for column in ("air_withdrawal_intensity_m3_per_mwh", "air_withdrawal_ccs_intensity_m3_per_mwh"):
        if column not in plants.columns:
            raise KeyError(
                f"plants.csv lacks {column}. Rebuild it: python scripts/build_plant_inputs.py --hubs"
            )
    air_base = plants["air_withdrawal_intensity_m3_per_mwh"].astype(float)
    air_capture = plants["air_withdrawal_ccs_intensity_m3_per_mwh"].astype(float)
    return base, capture, air_base, air_capture, factor


def once_through_calibration(plants: pd.DataFrame, generation_mwh: pd.Series) -> float:
    """把模拟的直流冷却取水重新缩放到水资源公报基准上的因子。

    构造与 `water.basin_bias_factors` 相同：官方总量除以模拟总量，以乘法施加。用一个全国
    因子而不是九个流域因子——这一差异是强度表（一个美国凝汽器设计假设）的属性，而不是任何
    流域的属性；并且公报按区域公布直流火(核)电，但不按燃料公布，所以逐流域因子拟合的将是
    各区域煤电相对于核电与燃气机队的份额，而这是观测不到的。
    """
    modelled = float(once_through_withdrawal(plants, generation_mwh).sum())
    if modelled <= 0.0:
        raise ValueError("no freshwater once-through withdrawal in the fleet; nothing to calibrate")
    return NATIONAL_ONCE_THROUGH_WITHDRAWAL_1E8_M3 * 1e8 / modelled


def basin_reserved_withdrawal(exclude_all_industry: bool = False) -> dict[str, float]:
    """已承诺给模型不做决策的用户的取水量，亿 m3/yr。

    水资源公报报告的工业用水把火电也并在里面：直流火(核)电作为子列单独列出，但闭式循环与
    空冷火电的冷却用水留在工业剩余项里，没有子列，无法从这一来源中还原。所以有两种站得住脚
    的预留口径，它们把真实值夹在中间：

        exclude_all_industry=False  预留 生活 + 农业 + 生态 + (工业 - 直流火(核)电)。
                                    闭式循环火电既保持预留，又被模型再次使用，所以同一份
                                    水被计了两次。保守：低估了电力可用的水量。
        exclude_all_industry=True   只预留 生活 + 农业 + 生态。全部工业——包括模型不做
                                    决策的非电部分——都交给模型。乐观：高估了可用水量。

    主线预算用保守形式；乐观形式是敏感性分析，用来界定这项无法分离的闭式循环项最多能有
    多大影响。
    """
    reserved: dict[str, float] = {}
    for code, row in BASIN_WITHDRAWAL_2025_1E8_M3.items():
        base = row["domestic"] + row["agriculture"] + row["ecology"]
        if not exclude_all_industry:
            base += row["industry"] - row["thermal_once_through"]
        reserved[code] = base
    return reserved


def modelled_industry_withdrawal(paths: ProjectPaths) -> dict[str, float]:
    """模型做决策的工业点源的取水量，亿 m3/yr，按流域。

    这些点源包含在公报的工业用水数字里，所以一旦工业成为决策主体，它的水就会被计两次：
    一次在 `basin_reserved_withdrawal` 里，一次是模型再次使用它。正是这一扣出（carve-out）
    使两者一致。

    若尚未构建 `inputs/industry_sources.csv`，则返回空映射；那是只含煤电的配置，不需要扣出。
    """
    path = paths.inputs_dir / "industry_sources.csv"
    if not path.exists():
        logger.info("industry_sources.csv absent; no industrial carve-out from the reservation")
        return {}
    from .water import _assign_basin_codes

    sources = pd.read_csv(path)
    sources["basin_code"] = _assign_basin_codes(paths, sources)
    by_basin = sources.groupby("basin_code")["water_m3_per_year"].sum() / 1e8
    return {str(code): float(value) for code, value in by_basin.items()}


def write_basin_caps(paths: ProjectPaths, kind: WeightKind = "demand") -> pd.DataFrame:
    """写出 `inputs/water_basin_caps.csv`：水约束的制度半边。

    每个 (流域, 规划年) 一行。`residual_m3_per_year` 是模型自己的源——煤电 hub，以及进入
    模型之后的工业点源——之间合计可取的水量：

        residual = 用水总量控制指标 - (生活 + 农业 + 生态 + 非电工业) + 已建模工业的现状取水

    2030 年之后保持不变：国办发〔2013〕2号没有设定更晚的目标，外推它就等于编造政策。

    超额占用的流域。西北诸河的取水量（2025 年 729亿 m3）已超过分摊给它的 2030 年上限
    （641亿），所以那里的 `cap - reserved` 为负。原样写入会使约束无论模型怎么做都无法
    满足；而且由于流域松弛与节点松弛按同一个 big-M 惩罚，~85亿 m3 的永久违约会主导目标
    函数，并以一个与之毫不相干的理由左右全国其他各地的决策。

    单独在零处截断也不诚实——它会悄悄删掉这一发现。红线政策真正隐含的是：超上限的流域
    必须压缩，而且其中所有用户都要压缩，不只是电力。所以凡 `actual > cap` 之处，预留量
    都按 `cap / actual` 等比例缩放，这等价于让每个用户都满足同一个达标比例。未缩放的数字
    保留在 `residual_uncapped_1e8_m3` 中，比例保留在 `compliance_ratio` 中，因此超额占用
    会被报告出来，而不是被隐藏。
    """
    from ..constants import PLANNING_YEARS

    caps_by_year = {year: basin_caps(paths, year, kind) for year in PLANNING_YEARS}
    reserved = basin_reserved_withdrawal()
    carve_out = modelled_industry_withdrawal(paths)
    rows = []
    for year in PLANNING_YEARS:
        for code in sorted(BASIN_NAMES_ZH):
            cap = caps_by_year[year][code]
            actual = BASIN_WITHDRAWAL_2025_1E8_M3[code]["total"]
            industry = carve_out.get(code, 0.0)
            uncapped = cap - reserved[code] + industry
            ratio = min(1.0, cap / actual) if actual > 0 else 1.0
            residual = cap - reserved[code] * ratio + industry * ratio
            if ratio < 1.0:
                logger.warning(
                    "basin %s (%s) withdraws %.0f against a cap of %.0f 1e8 m3; reservation "
                    "scaled pro rata by %.3f, raw residual %.1f -> %.1f",
                    code, BASIN_NAMES_ZH[code], actual, cap, ratio, uncapped, residual,
                )
            rows.append({
                "basin_code": code,
                "basin_name": BASIN_NAMES_ZH[code],
                "planning_year": year,
                "cap_1e8_m3": round(cap, 3),
                "actual_2025_1e8_m3": round(actual, 3),
                "reserved_1e8_m3": round(reserved[code], 3),
                "modelled_industry_1e8_m3": round(industry, 3),
                "compliance_ratio": round(ratio, 4),
                "residual_uncapped_1e8_m3": round(uncapped, 3),
                "residual_1e8_m3": round(residual, 3),
                "residual_m3_per_year": residual * 1e8,
                "apportionment_weight": kind,
                "source": "国办发〔2013〕2号 附件1 + 2025年中国水资源公报 表9",
                "basis": "withdrawal (用水量, 新水取用量)",
            })
    frame = pd.DataFrame(rows)
    destination = paths.inputs_dir / "water_basin_caps.csv"
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    logger.info("wrote %d rows to %s", len(frame), destination)
    return frame
