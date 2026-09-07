"""Official Chinese water-allocation figures: the 用水总量控制指标 basis.

WHY THIS MODULE EXISTS. The basin water budget used to be built from modelled natural runoff:

    available = qtot x WATER_EXTRACTABLE_FRACTION (0.20) x (1 - existing_withdrawal_share (0.85))

`optimization/data_prep.py` documents that those last two factors are ALIASED -- the solver
only ever sees their product (0.03) -- so no experiment in the study can say whether the
environmental-flow standard or the allocation rule is what binds. Worse, the denominator is
*local natural runoff*, which for the north China basins is simply the wrong quantity: the Hai
basin physically consumes more than its own runoff every year, financed by inter-basin transfer
(South-North Water Transfer) and by groundwater overdraft. Deducting real domestic + irrigation
demand from 20% of local runoff therefore returns a NEGATIVE allowance for Hai, Yellow, Huai and
Northwest -- an artefact of the denominator, not a finding.

China's own water-allocation instrument does not have that defect. The 最严格水资源管理制度
("three red lines") caps 用水总量 per province, and those caps are written against supply as
actually delivered, transfers and groundwater included. Using them makes numerator and
denominator come from the same official statistical system.

BASIS -- AND THE TRAP IN IT. Everything in this module is 用水量 as defined by the
中国水资源公报 编写说明 2.(7): gross intake (毛用水量) on a 新水取用量 basis, excluding in-plant
recirculation. It is NOT the consumption basis the old constraint used.

It is also NOT the 取水定额 basis of `constants.CHINA_WATER_QUOTA_M3_PER_MWH`, and conflating
the two is the easy mistake here. The 定额 standard (水节约〔2019〕373号) deliberately excludes
once-through condenser flow -- which is why its once-through allowance, 0.35-0.72 m3/MWh, is the
LOWEST of the three cooling systems. The 公报 does the opposite: it reports 直流火(核)电冷却用水
as a named sub-column INSIDE 工业用水 (453.8 of 953.7亿 m3 nationally in 2025). The caps in this
module are enforced against the 公报 quantity, so the model term that must be held under them is
plant WITHDRAWAL, not the metered 定额 volume. Pricing the fleet at 定额 would understate its
claim on the cap by roughly 5x.

The model's withdrawal column is itself not on the 公报's scale: its once-through intensities are
Macknick (2011) US values (82.8-116.5 m3/MWh), and applying them to China's freshwater
once-through fleet returns 1034亿 m3/yr for coal alone -- more than the bulletin's entire national
industrial withdrawal. `builders/water_quota.once_through_calibration` rescales them onto the
公报 baseline, exactly as `builders/water.basin_bias_factors` rescales modelled runoff onto the
official water-resource baseline.

WHAT IS DELIBERATELY ABSENT.
  * Per-basin caps. 国办发 2013 no.2 attachment 1 is a closed provincial partition (the 31 rows
    sum to 6350/6700/7000 exactly, zero residual); no parallel basin table exists, and
    流域 != 一级区 anyway. Basin caps must be derived by apportionment -- see
    `builders/water_quota.py`.
  * The 长江 2348亿 m3 figure. The 长江流域综合规划 (2012-2030 revision) was never publicly
    released; the number has no citable public source and must not enter the paper.
  * Per-basin consumption (耗水量). The 水资源公报 表11 has no consumption column; only the
    national sectoral 耗水率 is published. Applying it per basin would be an assumption.
"""
from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------------------
# 1. Provincial 用水总量控制指标, 亿 m3/yr, withdrawal basis.
#    国务院办公厅. 关于印发实行最严格水资源管理制度考核办法的通知 (国办发〔2013〕2号), 2013-01-02,
#    附件1 "各省、自治区、直辖市用水总量控制目标".
#    https://www.gov.cn/gongbao/content/2013/content_2313188.htm
#    Province names are the short forms used by `data/ChinaMap/provinces.shp`.
# --------------------------------------------------------------------------------------
PROVINCE_WATER_CAP_1E8_M3: Final[dict[str, tuple[float, float, float]]] = {
    # 省份:            (2015,   2020,   2030)
    "北京": (40.00, 46.58, 51.56),
    "天津": (27.50, 38.00, 42.20),
    "河北": (217.80, 221.00, 246.00),
    "山西": (76.40, 93.00, 99.00),
    "内蒙古": (199.00, 211.57, 236.25),
    "辽宁": (158.00, 160.60, 164.58),
    "吉林": (141.55, 165.49, 178.35),
    "黑龙江": (353.00, 353.34, 370.05),
    "上海": (122.07, 129.35, 133.52),
    "江苏": (508.00, 524.15, 527.68),
    "浙江": (229.49, 244.40, 254.67),
    "安徽": (273.45, 270.84, 276.75),
    "福建": (215.00, 223.00, 233.00),
    "江西": (250.00, 260.00, 264.63),
    "山东": (250.60, 276.59, 301.84),
    "河南": (260.00, 282.15, 302.78),
    "湖北": (315.51, 365.91, 368.91),
    "湖南": (344.00, 359.75, 359.77),
    "广东": (457.61, 456.04, 450.18),
    "广西": (304.00, 309.00, 314.00),
    "海南": (49.40, 50.30, 56.00),
    "重庆": (94.06, 97.13, 105.58),
    "四川": (273.14, 321.64, 339.43),
    "贵州": (117.35, 134.39, 143.33),
    "云南": (184.88, 214.63, 226.82),
    "西藏": (35.79, 36.89, 39.77),
    "陕西": (102.00, 112.92, 125.51),
    "甘肃": (124.80, 114.15, 125.63),
    "青海": (37.00, 37.95, 47.54),
    "宁夏": (73.00, 73.27, 87.93),
    "新疆": (515.60, 515.97, 526.74),
}
# The document's own totals; the 31 rows reproduce them with zero residual.
PROVINCE_CAP_NATIONAL_1E8_M3: Final[dict[int, float]] = {2015: 6350.0, 2020: 6700.0, 2030: 7000.0}

# Hong Kong, Macau and Taiwan carry no cap in 附件1 and no coal hub in `inputs/plants.csv`.
PROVINCES_WITHOUT_CAP: Final[frozenset[str]] = frozenset({"香港", "澳门", "台湾"})

# --------------------------------------------------------------------------------------
# 2. Actual withdrawal by level-1 water resource region, 亿 m3/yr, withdrawal basis.
#    2025年中国水资源公报 表9 "2025年水资源一级区供水量和用水量", 水利部, 2026.
#    http://www.mwr.gov.cn/sj/tjgb/szygb/202606/t20260629_2131924.html
#    Keys are this repo's basin codes; A = 松花江区 + 辽河区 merged to match the geometry.
#    `thermal_once_through` is the bulletin's 〔其中: 直流火(核)电〕 sub-column: a COMPONENT of
#    `industry`, not an addition. Subtract it to get non-power industry; do not add it to the
#    total. Closed-cycle thermal cooling is inside the industry residual and is NOT separable
#    in this source -- which is why `builders/water_quota.py` hands the whole power-sector
#    withdrawal back to the model rather than trying to net it out here.
# --------------------------------------------------------------------------------------
BASIN_WITHDRAWAL_2025_1E8_M3: Final[dict[str, dict[str, float]]] = {
    "A": {"domestic": 58.3, "industry": 38.5, "thermal_once_through": 6.6,
          "agriculture": 478.2, "ecology": 34.5, "total": 609.5},
    "C": {"domestic": 73.4, "industry": 40.0, "thermal_once_through": 0.1,
          "agriculture": 187.8, "ecology": 77.1, "total": 378.3},
    "D": {"domestic": 58.4, "industry": 51.8, "thermal_once_through": 0.0,
          "agriculture": 247.5, "ecology": 34.2, "total": 391.8},
    "E": {"domestic": 106.1, "industry": 67.3, "thermal_once_through": 5.9,
          "agriculture": 404.1, "ecology": 42.4, "total": 619.8},
    "F": {"domestic": 352.0, "industry": 567.3, "thermal_once_through": 394.4,
          "agriculture": 1041.3, "ecology": 88.3, "total": 2049.0},
    "G": {"domestic": 74.6, "industry": 58.3, "thermal_once_through": 11.6,
          "agriculture": 140.8, "ecology": 17.5, "total": 291.3},
    "H": {"domestic": 176.1, "industry": 106.3, "thermal_once_through": 35.3,
          "agriculture": 453.6, "ecology": 35.6, "total": 771.7},
    "J": {"domestic": 12.9, "industry": 4.7, "thermal_once_through": 0.0,
          "agriculture": 85.1, "ecology": 1.4, "total": 104.1},
    "K": {"domestic": 24.8, "industry": 19.4, "thermal_once_through": 0.0,
          "agriculture": 645.4, "ecology": 39.4, "total": 729.0},
}
BASIN_WITHDRAWAL_2025_NATIONAL_1E8_M3: Final[float] = 5944.5

# National 直流火(核)电冷却用水量, 亿 m3/yr -- the calibration target for the model's once-through
# withdrawal (see `builders/water_quota.once_through_calibration`). Freshwater only: the bulletin
# states 用水量 "不包括海水直接利用量", and the model already zeroes seawater condensers, so the
# two sides exclude the same thing. It covers all thermal AND nuclear; China's nuclear fleet is
# entirely coastal seawater-cooled and therefore contributes ~0 here, and gas once-through is
# negligible, so attributing the whole figure to coal is a small over-attribution that makes the
# calibrated intensity -- and hence the constraint -- conservative rather than loose.
NATIONAL_ONCE_THROUGH_WITHDRAWAL_1E8_M3: Final[float] = 453.8

BASIN_NAMES_ZH: Final[dict[str, str]] = {
    "A": "东北诸河", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
    "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北诸河",
}

# --------------------------------------------------------------------------------------
# 3. Yellow River: the one basin with a published allocation of its own.
#    "八七分水" (国办发〔1987〕61号) allocated 370亿 m3/yr of the 580亿 multi-year mean, leaving
#    210亿 as in-stream ecological flow. NOTE THE BASIS: the 1987 table header reads 年耗水量 --
#    it is a CONSUMPTION allocation of surface water only, so it is NOT interchangeable with the
#    withdrawal caps above and is kept here for cross-checking, not for the budget. The current
#    plan revises it to 332.8亿; groundwater is capped separately at 123.7亿.
# --------------------------------------------------------------------------------------
YELLOW_RIVER_CONSUMPTION_CAP_1E8_M3: Final[float] = 332.8
YELLOW_RIVER_CONSUMPTION_CAP_1987_1E8_M3: Final[float] = 370.0
YELLOW_RIVER_GROUNDWATER_CAP_1E8_M3: Final[float] = 123.7

# --------------------------------------------------------------------------------------
# 4. National sectoral consumption ratios (耗水率), 2025年中国水资源公报 用水消耗量 section.
#    Published for the NATION ONLY -- the bulletin has no per-basin consumption table. Used to
#    translate between bases when a cross-check demands it, never to build the budget.
# --------------------------------------------------------------------------------------
NATIONAL_CONSUMPTION_RATIO: Final[dict[str, float]] = {
    "agriculture": 0.669, "industry": 0.239, "domestic": 0.383, "ecology": 0.593,
}
NATIONAL_CONSUMPTION_RATIO_ALL: Final[float] = 0.550


def province_cap(province: str, year: int) -> float:
    """Cap for `province` in 亿 m3/yr, interpolating between the three fixed years.

    Years beyond 2030 are held flat: 国办发〔2013〕2号 sets no post-2030 target, and extending
    the trend would be inventing policy.

    Raises:
        KeyError: when the province carries no cap in 附件1.
    """
    caps = PROVINCE_WATER_CAP_1E8_M3.get(province)
    if caps is None:
        raise KeyError(f"no 用水总量控制指标 for province {province!r}")
    if year <= 2015:
        return caps[0]
    if year >= 2030:
        return caps[2]
    if year <= 2020:
        return caps[0] + (caps[1] - caps[0]) * (year - 2015) / 5.0
    return caps[1] + (caps[2] - caps[1]) * (year - 2020) / 10.0
