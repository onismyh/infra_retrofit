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
from .water_grid import (
    _annual_m3,
    _basin_zone_grid,
    _cell_area_m2,
    _clean_window,
    _file_years,
    _nc_path,
    _province_zone_grid,
)
from .water_quota_carveout import modelled_coal_closed_loop_withdrawal, modelled_industry_withdrawal

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
        years = _file_years(dataset)
        window = (years >= WEIGHT_WINDOW[0]) & (years <= WEIGHT_WINDOW[1])
        weight = _annual_m3(_clean_window(dataset, WEIGHT_VARIABLE, window), area)
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

    直接读取 `builders/plants_water.finalize_water_intensities` 写出的列，即逐机组的精确差值
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
    """每个 hub 的淡水直流冷却取水量，m3/yr，按模型原始强度（Wang 2023 Table 1）计。

    从 hub 的综合取水中单独拆出来，是因为只有直流冷却部分需要重新标定：
    `WATER_INTENSITY_BY_TECH_M3_PER_MWH` 的循环冷却行已与 Macknick 交叉核对，相差
    3-15%（空冷行 Macknick 没有煤电的值，不带 CCS 时取水只有 0.15-0.23 m3/MWh）；与水资源公报相差两倍的正是直流冷却行。

    海水冷却 hub 返回零：公报不含海水直接利用量，`builders/plants_water.py` 也已把它们的凝汽器
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

    构造与 `water_bias.basin_bias_factors` 相同：官方总量除以模拟总量，以乘法施加。用一个全国
    因子而不是九个流域因子——这一差异是强度表（文献里的凝汽器设计取水强度）的属性，而不是任何
    流域的属性；并且公报按区域公布直流火(核)电，但不按燃料公布，所以逐流域因子拟合的将是
    各区域煤电相对于核电与燃气机队的份额，而这是观测不到的。

    模拟总量只数公报年（`BULLETIN_YEAR`）及以前投产的机组：逐 hub 乘 `once_through_bulletin_share`
    （`optimization.data_prep_plants._with_expiry` 按机组算）。2026-10-16 前此后投产的机组也算在内，因子偏小约 9%。
    因子只保证全国合计对得上（公报小数位舍入内：453.8 对九流域直流子列之和 453.9），分流域不相等（GEM 给内陆北方若干厂
    标了直流，模型在海河、黄河的直流取水远大于公报子列的 0.1、0.0 亿 m3）。公报子列只含凝汽器冷却水；直流机组的锅炉补给水等
    淡水留在公报工业余项里，模型两侧都不计（海水直流 hub 的直流分量为 0，内陆直流只有按本因子标定的 Wang 直流整行）。
    """
    if "once_through_bulletin_share" not in plants.columns:
        raise KeyError("plants lacks once_through_bulletin_share; prepare it with optimization.data_prep_plants._with_expiry")
    bulletin_share = plants["once_through_bulletin_share"].astype(float)
    modelled = float((once_through_withdrawal(plants, generation_mwh) * bulletin_share).sum())
    if modelled <= 0.0:
        raise ValueError("no freshwater once-through withdrawal in the fleet; nothing to calibrate")
    return NATIONAL_ONCE_THROUGH_WITHDRAWAL_1E8_M3 * 1e8 / modelled


def basin_reserved_withdrawal() -> dict[str, float]:
    """公报口径下除直流火(核)电以外的全部取水，亿 m3/yr：生活 + 农业 + 生态 + (工业 - 直流火(核)电)。

    水资源公报的工业用水把火电也并在里面：直流火(核)电单列为子列，闭式循环与空冷火电的冷却用水留在工业
    剩余项里，没有子列。所以这里的预留还含两块模型自己做决策的用水——已建模工业点源（`modelled_industry_withdrawal`）
    与煤电的闭式循环与空冷取水（`modelled_coal_closed_loop_withdrawal`），`write_basin_caps` 把二者扣出、交还给模型。
    2026-10-16 前只扣出工业，煤电闭式循环取水既留在预留里、又由模型再次使用，同一份水计了两次。
    """
    return {
        code: row["domestic"] + row["agriculture"] + row["ecology"] + row["industry"] - row["thermal_once_through"]
        for code, row in BASIN_WITHDRAWAL_2025_1E8_M3.items()
    }


def write_basin_caps(paths: ProjectPaths, kind: WeightKind = "demand") -> pd.DataFrame:
    """写出 `inputs/water_basin_caps.csv`：水约束的制度半边。

    每个 (流域, 规划年) 一行。`residual_m3_per_year` 是模型自己的源——煤电 hub，以及进入
    模型之后的工业点源——之间合计可取的水量：

        residual = 用水总量控制指标 - (生活 + 农业 + 生态 + 工业 - 直流火(核)电)
                   + 已建模工业的现状取水 + 煤电闭式循环与空冷的现状取水

    （超额占用的流域，括号与两项扣出一起乘达标比例，见下。公报的工业含闭式循环与空冷火电，不是"非电工业"。）

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

    扣出的是模型的两类决策主体在公报年（`constants_water_quota.BULLETIN_YEAR`）的取水：已建模工业点源与煤电的闭式循环与空冷取水
    （`water_quota_carveout`；2026-10-16 前只扣工业，煤电这份水计了两次；2026-10-17 起工业也只计公报年在运、且属于求解侧建模 hub（剔西藏）的点源，
    煤电的利用小时改按 hub 所在省取，与求解侧同法）。二者合计超过公报工业减直流火(核)电就说明模型的强度或冷却方式与公报对不上，直接报错
    （不截断：截了会让保留量低于不建模用户之和，再按达标比例缩放后余量虚高）；当前输入最接近的是西北诸河，占六成。
    """
    from ..constants import PLANNING_YEARS

    caps_by_year = {year: basin_caps(paths, year, kind) for year in PLANNING_YEARS}
    reserved = basin_reserved_withdrawal()
    carve_out = modelled_industry_withdrawal(paths)
    coal_carve_out = modelled_coal_closed_loop_withdrawal(paths)
    unknown = (set(carve_out) | set(coal_carve_out)) - set(BASIN_NAMES_ZH)
    if unknown:
        raise ValueError(f"carve-out basin codes {sorted(unknown)} are not in BASIN_NAMES_ZH; check assign_basin_codes")
    for code, row in BASIN_WITHDRAWAL_2025_1E8_M3.items():
        closed_loop_industry = row["industry"] - row["thermal_once_through"]
        industry, coal = carve_out.get(code, 0.0), coal_carve_out.get(code, 0.0)
        if industry + coal > closed_loop_industry + 1e-9:  # 容差：567.3 − 394.4 在浮点里是 172.8999…
            raise ValueError(
                f"basin {code} ({BASIN_NAMES_ZH[code]}): modelled industry {industry:.2f} + coal closed-loop {coal:.2f} = "
                f"{industry + coal:.2f} 1e8 m3 exceeds the bulletin's industry less once-through thermal "
                f"{closed_loop_industry:.2f}; the model's intensities or cooling types disagree with the bulletin"
            )
    rows = []
    for year in PLANNING_YEARS:
        for code in sorted(BASIN_NAMES_ZH):
            cap = caps_by_year[year][code]
            actual = BASIN_WITHDRAWAL_2025_1E8_M3[code]["total"]
            industry = carve_out.get(code, 0.0)
            coal = coal_carve_out.get(code, 0.0)
            uncapped = cap - reserved[code] + industry + coal
            ratio = min(1.0, cap / actual) if actual > 0 else 1.0
            residual = cap - (reserved[code] - industry - coal) * ratio
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
                "modelled_coal_closed_loop_1e8_m3": round(coal, 3),
                "compliance_ratio": round(ratio, 4),
                "residual_uncapped_1e8_m3": round(uncapped, 3),
                "residual_1e8_m3": round(residual, 3),
                # 取整到 m3：不取整时末位浮点随平台的求和顺序变（2026-10-13 前不取整，重建的表与库内表差 4e-15）。
                "residual_m3_per_year": round(residual * 1e8),
                "apportionment_weight": kind,
                "source": "国办发〔2013〕2号 附件1 + 2025年中国水资源公报 表9",
                "basis": "withdrawal (用水量, 新水取用量)",
            })
    frame = pd.DataFrame(rows)
    destination = paths.inputs_dir / "water_basin_caps.csv"
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    logger.info("wrote %d rows to %s", len(frame), destination)
    return frame
