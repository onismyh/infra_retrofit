"""工业点源参数：部门键、取水定额、氢路线有无、资产寿命、hub 数与路线集合；CCS 捕集参数在 `constants_industry_capture`，氢路线在 `constants_industry_h2`。

这里的每个数字要么附出处，要么标 `⚠ 假设`。沿用 `constants.py` 的两条规则：

1. **水量口径。** 工业点源的用水只进流域取水上限（`optimization/model_resources.add_basin_withdrawal_cap`，
   有水约束且流域上限打开时才有）；作用于耗水的生态流量那一档只含煤电。
   定额（GB/T 18916 单位产品取水量）与流域指标（国办发〔2013〕2号 附件1 用水总量控制指标）都是取水口径，
   所以工业用水按取水计，`INDUSTRY_CONSUMPTION_SHARE` = 1.0 是定义性取值。它不是"取水≈耗水"的判断，
   那一说没有数据支撑：全国工业耗水/取水比只有 0.23（Jin et al. 2022，`jin2022climate`），但全国工业
   用水里约 37% 是直流火（核）电冷却（2016 年水资源公报），不能直接当点源耗水率；钢铁等部门级数据只见
   检索摘要，在 0.34-0.87 之间。若将来把工业接入耗水档，这里要换成分部门耗水率。
2. **出处与中国匹配。** 所有定额都取自 GB/T 18916《取水定额》系列——与
   `CHINA_WATER_QUOTA_M3_PER_MWH` 中已有的煤电定额同属一族（GB/T 18916.1，经由
   水利部《钢铁等十八项工业用水定额》水节约〔2019〕373号）。若把中国的电力定额与
   （比如）欧洲的工业用水强度混用，就会破坏水约束所依赖的同口径比较。

定额分级：GB 标准给出通用值（现有企业，日常管理）和先进值（新建 / 改建企业，用于
取水许可审批）。因此存量工厂取通用值；实施改造的工厂取先进值。
"""
from __future__ import annotations

from typing import Final


# --- 部门键 ------------------------------------------------------------------------------
SECTOR_STEEL_BF: Final = "steel_bf_bof"
SECTOR_STEEL_EAF: Final = "steel_eaf"
SECTOR_CEMENT: Final = "cement"
SECTOR_AMMONIA: Final = "ammonia"
SECTOR_METHANOL: Final = "methanol"
SECTOR_REFINERY: Final = "refinery"
SECTOR_COAL_CHEM: Final = "coal_chemical"

# 炼化与现代煤化工暂不在建模范围内（作者决定，2026-09-06）。
# 决定性的原因是数据：GB/T 18916.3-2022（石油炼制）与煤化工定额表都未能取得，
# 因此它们的用水强度只能靠猜——而水正是本研究的紧约束。它们的读取器与键仍然
# 保留，这样只要补上这两项定额，就能把它们重新启用。
INDUSTRY_SECTORS: Final[tuple[str, ...]] = (
    SECTOR_STEEL_BF,
    SECTOR_STEEL_EAF,
    SECTOR_CEMENT,
    SECTOR_AMMONIA,
    SECTOR_METHANOL,
)

SECTORS_OUT_OF_SCOPE: Final[frozenset[str]] = frozenset({SECTOR_REFINERY, SECTOR_COAL_CHEM})

# 部门 -> 目标组。部门上限（scripts/build_sector_targets.py）从 China TIMES 读取，
# 所取层级就是 TIMES 核算 CO2 的层级：钢铁、建材、化工。煤电是第四组 "power"，
# 它不是工业部门，所以不在这张映射里。
SECTOR_TARGET_GROUP: Final[dict[str, str]] = {
    SECTOR_STEEL_BF: "steel",
    SECTOR_STEEL_EAF: "steel",
    SECTOR_CEMENT: "cement",
    SECTOR_AMMONIA: "chemicals",
    SECTOR_METHANOL: "chemicals",
}
POWER_TARGET_GROUP: Final[str] = "power"

SECTOR_LABELS_ZH: Final[dict[str, str]] = {
    SECTOR_STEEL_BF: "钢铁（高炉-转炉）",
    SECTOR_STEEL_EAF: "钢铁（电炉）",
    SECTOR_CEMENT: "水泥",
    SECTOR_AMMONIA: "合成氨",
    SECTOR_METHANOL: "甲醇",
    SECTOR_REFINERY: "炼化",
    SECTOR_COAL_CHEM: "现代煤化工",
}


# --- 取水定额，m3/t 产品 ------------------------------------------------------------------
# 出处：GB/T 18916《取水定额》系列国家标准，各部门所用分册见下；甲醇用河北省地方标准。
#
# 按 (部门, 原料/路线) 给出 (通用值, 先进值)。`None` 表示尚未找到出处——builder 会
# 报错，而不是拿猜测值顶替，因为编造的用水强度会直接流进本研究的紧约束。
#
# 钢铁 GB/T 18916.2-2022《取水定额 第2部分：钢铁联合企业》，m3/t 粗钢：
#     含焦化+含冷轧 4.8/3.1 ; 含焦化+不含冷轧 4.5/2.4
#     不含焦化+含冷轧 4.2/2.2 ; 不含焦化+不含冷轧 3.6/2.1
#   中国的钢铁联合企业绝大多数同时带焦化和冷轧，所以存量默认取
#   "含焦化+含冷轧"这一行。
# 电炉在 GB/T 18916.2 中没有全厂行；它由该标准的工序行组合而成
#   （电炉炼钢 1.74/0.55 + 棒材轧钢 0.70/0.34），因为短流程钢厂恰好就是这两道工序。
#   是组合值而非原文引用——特此标明。
WATER_INTAKE_QUOTA_M3_PER_T: Final[dict[tuple[str, str], tuple[float, float] | None]] = {
    (SECTOR_STEEL_BF, "default"): (4.8, 3.1),
    (SECTOR_STEEL_EAF, "default"): (2.44, 0.89),   # 1.74+0.70 / 0.55+0.34，组合值
    # 水泥 GB/T 18916.62-2022《取水定额 第62部分：水泥》，m3/t 熟料
    (SECTOR_CEMENT, "default"): (0.510, 0.225),
    # 合成氨 GB/T 18916.8《取水定额 第8部分：合成氨》，m3/t 氨，按原料分档——与合成氨
    # 点源表的 `Feedstock` 列一一对应。
    (SECTOR_AMMONIA, "Coal"): (12.0, 8.0),          # 烟煤
    (SECTOR_AMMONIA, "Gas"): (9.0, 5.5),            # 天然气
    (SECTOR_AMMONIA, "Oil"): (12.0, 8.0),           # GB 无油原料行；借用煤原料行，仅 1 家厂
    (SECTOR_AMMONIA, "Anthracite"): (11.0, 7.5),    # 无烟块煤（型煤）
    # 原料为空 / 未映射时的回退：取煤路线，占 198 家中的 160 家。
    (SECTOR_AMMONIA, "default"): (12.0, 8.0),
    # 甲醇 DB13/T 5448.7-2021《工业取水定额 第7部分：煤化工行业》（河北），m3/t 甲醇
    # 这是省级标准；之所以采用，是因为未找到 GB/T 18916 中甲醇的国家标准部分。
    (SECTOR_METHANOL, "Coal"): (15.0, 11.0),
    (SECTOR_METHANOL, "Coke oven gas"): (9.03, 6.34),
    (SECTOR_METHANOL, "Gas"): (9.03, 6.34),         # 天然气路线套用焦炉煤气那一行
    # 矿热炉尾气 / 电石炉尾气是副产尾气路线，与焦炉煤气同属一类。
    (SECTOR_METHANOL, "矿热炉尾气"): (9.03, 6.34),
    (SECTOR_METHANOL, "电石炉尾气"): (9.03, 6.34),
    (SECTOR_METHANOL, "default"): (15.0, 11.0),      # 回退：煤路线，占 221 家中的 148 家
    # 未找到出处。GB/T 18916.3-2022《取水定额 第3部分：石油炼制》确实存在，也是对应的
    # 标准，但没能取得其数值表；煤制烯烃/煤制油/煤制乙二醇各行同样如此
    # （GB/T 18916 有一个第36部分：煤制乙二醇）。这些值不要猜。
    (SECTOR_REFINERY, "default"): None,
    (SECTOR_COAL_CHEM, "default"): None,
}

# 工业点源用水计入流域取水上限时乘的系数（`builders/industry.py` 用它乘定额 x 产量）。1.0 = 按取水计，
# 是定义性取值：定额（GB/T 18916）与流域指标（国办发〔2013〕2号 附件1）都是取水口径，见模块 docstring 第 1 条。
INDUSTRY_CONSUMPTION_SHARE: Final[float] = 1.0


# --- 氢路线 -------------------------------------------------------------------------------
# 哪些部门究竟能用绿氢替代，以及替代的是什么。
#
# 水泥 = False。已按要求对照文献核查。氢路线在物理上存在——H2 可在窑头燃烧器中
# 掺烧——但它只能替代燃料燃烧；CaCO3 煅烧是化学分解，不论用什么加热都会排放 CO2。
# 文献报告的上限：100% 以氢为燃料的熟料工艺使 CO2 强度降低 27.6%
# （Energy Conversion & Management 2023, "Decarbonisation pathways of the cement production
# process via hydrogen and oxy-combustion"），而水泥部门 >60% 的排放来自煅烧。
# 点源表中水泥的 `H2_DMD`（全国 1 396 万吨 H2/yr）是燃料热值当量口径的数字；把它当作
# 减排基数，会把水泥靠氢可达的减排量高估到约两倍。因此这里水泥只走 CCS，而这个
# 标志是留给敏感性运行的开关，并非硬编码的排除。
SECTOR_HAS_H2_ROUTE: Final[dict[str, bool]] = {
    SECTOR_STEEL_BF: True,      # H2-DRI 替代高炉的焦炭还原
    SECTOR_STEEL_EAF: False,    # 本来就是废钢+电力；没有可替代的化石还原剂
    SECTOR_CEMENT: False,       # 见上文
    SECTOR_AMMONIA: True,       # 绿氢替代灰氢原料，按化学计量
    SECTOR_METHANOL: True,      # 绿氢调节煤制甲醇的 H/C 比
    SECTOR_REFINERY: True,      # 绿氢替代厂内 SMR/副产氢
    SECTOR_COAL_CHEM: True,     # 绿氢替代合成气变换所制的氢
}


# --- 资产寿命，用于退役 --------------------------------------------------------------------
# 寿命取值：⚠ 假设（无出处）。缺投产年份时分两种情形填补，因为各部门已知的信息不同：
#
#   部分观测（水泥 99.6%，钢铁 BOF 55% / EAF 35%）-> 缺口按该部门自身的观测年份分布、
#     以确定性分位数取值来填补。已观测的那一半是关于该部门年龄结构的真实信息，
#     平铺阶梯会把它扔掉，使存量偏老或偏新。
#   完全无观测（合成氨、甲醇）-> 年龄在 [0, life] 上均匀铺开，于是每年恒有 1/life
#     的产能到寿。这是没有年龄数据时的最大熵选择；它不是对真实年龄结构的断言，
#     也不得被这样解读。
#
# 两种填补都是确定性的，从不随机抽样，所以两次运行逐位一致。
ASSET_LIFETIME_YEARS: Final[dict[str, int]] = {
    SECTOR_STEEL_BF: 30,
    SECTOR_STEEL_EAF: 30,
    SECTOR_CEMENT: 35,
    SECTOR_AMMONIA: 30,
    SECTOR_METHANOL: 30,
    SECTOR_REFINERY: 40,
    SECTOR_COAL_CHEM: 30,
}

# 退役时间表基于均匀年龄假设、而非观测年份的部门。
SECTORS_WITH_UNIFORM_RETIREMENT: Final[frozenset[str]] = frozenset(
    {SECTOR_AMMONIA, SECTOR_METHANOL}
)


# --- 各部门 hub 数 --------------------------------------------------------------------------
# 煤电把 3 623 台机组聚成 350 个 hub（~10 台/hub）。工业按部门分别聚类，这样一个 hub
# 绝不会是（比如）水泥窑与合成氨厂的混合：它们的用水强度、氢口径和捕集成本毫无
# 共同之处，混合的 hub 无法解释。各部门 hub 数大致保持煤电的每 hub 机组数比例，
# 并设下限，以免小部门被压成寥寥几个点。⚠ 假设（设定值）。
DEFAULT_SECTOR_HUB_COUNTS: Final[dict[str, int]] = {
    SECTOR_STEEL_BF: 80,
    SECTOR_STEEL_EAF: 40,
    SECTOR_CEMENT: 150,
    SECTOR_AMMONIA: 60,
    SECTOR_METHANOL: 60,
    SECTOR_REFINERY: 50,
    SECTOR_COAL_CHEM: 30,
}


def water_quota(sector: str, feedstock: str, advanced: bool = False) -> float:
    """给定部门与原料的单位产品取水量，m3/t。

    Args:
        sector: `INDUSTRY_SECTORS` 之一。
        feedstock: 原料 / 路线键；没有原料之分的部门用 "default"。
        advanced: 为 True 时返回先进值（新建 / 改造后的工厂），为 False 时返回通用值。

    Returns:
        每吨产品的取水量，m3/t。

    Raises:
        KeyError: 部门 / 原料组合不在定额表中。
        ValueError: 组合在表中，但其值尚未找到出处。
    """
    key = (sector, feedstock)
    if key not in WATER_INTAKE_QUOTA_M3_PER_T:
        key = (sector, "default")
    value = WATER_INTAKE_QUOTA_M3_PER_T[key]
    if value is None:
        raise ValueError(
            f"water intake quota for {key} has not been sourced; refusing to guess. "
            "Fill it from GB/T 18916 before running any scenario that touches this sector."
        )
    general, advanced_value = value
    return advanced_value if advanced else general


# =========================================================================================
# 减排选项
# =========================================================================================
# 每个 hub 三条路线，即作者 2026-09-07 选定的集合：`unabated`、`ccs`、`h2`。
# 没有退役路线——这里工业需求是外生的，一个停产的 hub 会悄无声息地丢掉它的产量，
# 而不是把产量转移到别处。工业产量究竟能否下降，是本模型不回答的需求问题。

INDUSTRY_ROUTES: Final[tuple[str, ...]] = ("unabated", "ccs", "h2")
