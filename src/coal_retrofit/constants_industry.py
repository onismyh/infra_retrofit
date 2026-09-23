"""工业点源参数：取水定额、氢路线、资产寿命。

这里的每个数字要么附出处，要么标 `⚠ 假设`。沿用 `constants.py` 的两条规则：

1. **水量口径。** 工业点源的用水只进流域取水上限（`optimization/model_resources.add_basin_withdrawal_cap`，
   `water_budget="official_quota"` 且流域上限打开时才有）；作用于耗水的生态流量那一档只含煤电。
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

from .constants import DEFAULT_DISCOUNT_RATE

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

# 各点源表 `H2_DMD` 列的确切含义。它们都是"若该厂可由 H2 替代的需求全部由氢满足，
# 所需的氢量"，但可替代的量因部门而异，所以掺烧档位必须对照正确的基数来解读。
H2_DEMAND_BASIS: Final[dict[str, str]] = {
    SECTOR_STEEL_BF: "full_h2_dri",          # 0.081 t H2 / t 粗钢
    SECTOR_CEMENT: "fuel_heat_equivalent",   # 不使用：没有氢路线
    SECTOR_AMMONIA: "stoichiometric_feed",   # 0.178 t H2 / t NH3
    SECTOR_METHANOL: "stoichiometric_feed",
    SECTOR_REFINERY: "current_h2_use",       # 原油加工量的 ~1.5%（按质量）
    SECTOR_COAL_CHEM: "stoichiometric_feed",
}


# --- 资产寿命，用于退役 --------------------------------------------------------------------
# 分两种情形，因为各部门已知的信息不同：
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
# 煤电把 3 541 台机组聚成 350 个 hub（~10 台/hub）。工业按部门分别聚类，这样一个 hub
# 绝不会是（比如）水泥窑与合成氨厂的混合：它们的用水强度、氢口径和捕集成本毫无
# 共同之处，混合的 hub 无法解释。各部门 hub 数大致保持煤电的每 hub 机组数比例，
# 并设下限，以免小部门被压成寥寥几个点。
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
#
# 成本口径。以下一切都按吨计，与 Tang et al. (2023, iScience 26:106347) 一致，本模型
# 扩展的正是其公式：Z_cap(i) = sum_g [ CAPEX_g(a_g_i) * CRF + OPEX_g(a_g_i) ]，
# 其中 a_g_i 为 hub i 上部门 g 的年捕集量。本仓库的煤电侧则按 MW 和 MWh 计；两边从不
# 混用，只在共享的 CO2 管网、共享的流域水量上限和唯一的联合排放目标处交汇。
#
# 包含与不包含的内容。下面的捕集参数只覆盖捕集岛——分离、压缩和厂内处理。运输与
# 封存不在其中：它们来自管网与封存 hub，与煤电完全相同。把文献里的"全链条 CCUS
# 成本"放进来会重复计算这两部分。
#
# 成本口径（作者 2026-09-22 决定，煤电与工业相同）：每个减排选项都按一次性改造 capex
# + 固定运维（每年为 capex 的一个比例）+ 实际使用的能源与耗材计价，能源按模型自己的
# 煤价和电价计，期末对未折旧的 capex 计残值。平准化的每吨捕集成本（ACCA21 式的
# CNY/t CO2）不再用于目标函数：平准化成本内含资本回收假设，而动态模型要自己决定何时
# 建设、资产随后在规划期内运行多久，两者不可能一致。ACCA21 的区间只保留作交叉核对，
# 用来检验下面参数的隐含值（`levelised_capture_cost_cny_per_t`）。

INDUSTRY_ROUTES: Final[tuple[str, ...]] = ("unabated", "ccs", "h2")

# --- CO2 捕集：改造 capex，单位为每吨年捕集能力的 CNY -------------------------------------
# 基准年 2030（由煤电改造同样使用的 `ccs_learning_factor` 对其缩放），含压缩至管输压力。
# 只计捕集岛。水泥、长流程钢为中国项目备案；电炉钢为 ⚠ 假设，合成氨 / 甲醇由全成本反推：
#   cement   中联水泥青州 20 万 t/a 全氧燃烧耦合碳捕集示范线 2.56 亿元 -> 1 280 元/(t·a)
#            （中国建材，2023 开工）；中联 20 万 t/a 捕集提纯项目 1.98 亿元 -> 990；
#            海螺白马山 5 万 t/a 5 500 万元 -> 1 100（2018 投运，含食品级提纯）。
#            0.05-0.2 Mt/a 规模下区间为 990-1 280；取 1 150。
#   steel    包钢 200 万 t CCUS 一期 50 万 t/a 6.14 亿元 -> 1 228（全产业链一期，含部分输送，
#            上界）；日照钢铁 18 万 t/a 1.35 亿元 -> 750（2025 开工，捕集+资源化）；
#            宝武案例（PKU CCUS 2020，Baowu Zhanjiang slip-stream，胺法）：捕集装置
#            CNY 360 M + 7% 业主费 + 20 M 流动资金 = 407 M，对应 0.5 Mt/a -> 814；
#            IEAGHG 2013/04 Table 6, Case 2A：捕集装置 US$(2010) 679 M，对应捕集量
#            ~4.7 Mt/a -> ~145 USD/(t·a) ~ 1 000 元/(t·a)。中值取 1 000。
#   steel_eaf ⚠ 假设（无出处）。尾气稀薄且间歇，即难度高于水泥窑，所以取水泥的值。
#            EAF 占模型内工业 CO2 的 1.7%。
#   ammonia / methanol   高浓度（>95%）气化尾气：无需吸收，只需脱水 + 压缩/液化。
#            延长石油榆林煤化 30 万 t/a（2022）报告捕集全成本 105 元/t；扣除 ~110 kWh/t
#            压缩电耗（按 ~0.45 元/kWh 计）后，剩 ~55 元/t 用于资本 + 固定运维，在
#            CRF(6%, 20 a) + 5%/a 下折得 capex 约 405 元/(t·a)（要 4% 才约 450）。⚠ 假设：取 450，
#            比按上列输入推得的值高 11%；没有备案给出投资额本身（齐鲁石化 100 万 t/a 未公布投资额）。
INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 1000.0,
    SECTOR_STEEL_EAF: 1150.0,
    SECTOR_CEMENT: 1150.0,
    SECTOR_AMMONIA: 450.0,
    SECTOR_METHANOL: 450.0,
}
# 捕集岛的固定运维，按每年占（经学习曲线调整后的）capex 的比例计。与煤电侧所用的
# 5%/a 相同（An et al. 2025 Nat Commun SI Table 7：固定运维 / 投资 = 5.4%/a）；
# PKU/Baowu 案例为 407 M 投资对应 12 M/a = 2.9%/a，IEAGHG 2013/04 钢厂维护费为安装成本的
# 142/3 928 = 3.6%/a。5% 取的是保守一端。
INDUSTRY_CCS_FIXED_OM_FRACTION: Final[float] = 0.05
# 每吨捕集 CO2 的捕集能耗。在 `industry_year_data` 中按模型自己的煤价（蒸汽，经锅炉
# 产生）和情景电价计价——而不是按冻结的文献价格——这样工业侧与煤电侧的能耗惩罚同步变动。
#   胺法燃烧后捕集（钢铁高炉煤气 / 热风炉烟气、水泥窑烟气、EAF 尾气）：
#     再沸器蒸汽 2.8 GJ/t：IEAGHG 2013/04 MDEA/Pz 2.3 GJ/t（CSIRO 综述：2.5-2.7
#     可达），国能锦界第 2 代吸收剂 2.35 GJ/t（中国 CCUS 进展报告 2025 p.18），
#     MEA 3.0-3.5 GJ/t；对 2030 年的改造而言，2.8 是这一分布的中心。
#     电耗 130 kWh/t，含压缩：PKU/Baowu 总出力损失 142 kWh/t
#     （蒸汽 + 电力，Table 9）；Gardarsdottir et al. 2019 水泥 MEA ~130 kWh/t。
#   高浓度流股（合成氨、甲醇）：无再沸器；只有压缩 + 脱水，~110 kWh/t（0.1 -> 11 MPa 密相）。
#     DEA 401 压缩与脱水 0.1 MWh/t（2025 年，区间 0.09-0.11，压至 150 bar、脱水至 <50 ppmv；`dea_ccts`）；
#     NPC 2019 合成氨捕集改造 0.1 MWh/t（`npc2019dualchallenge`，经 `pypsa_techdata`）。110 在 DEA 区间上端。
INDUSTRY_CCS_STEAM_GJ_PER_T_CO2: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 2.8,
    SECTOR_STEEL_EAF: 2.8,
    SECTOR_CEMENT: 2.8,
    SECTOR_AMMONIA: 0.0,
    SECTOR_METHANOL: 0.0,
}
INDUSTRY_CCS_ELECTRICITY_KWH_PER_T_CO2: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 130.0,
    SECTOR_STEEL_EAF: 130.0,
    SECTOR_CEMENT: 130.0,
    SECTOR_AMMONIA: 110.0,
    SECTOR_METHANOL: 110.0,
}
# 溶剂补充、废胺处置、水：PKU/Baowu 胺 40 000 元/t、水 6.5 元/t CO2；
# 胺法系统取 ~15 元/t CO2：DEA 401 可变运维 2.5（1.5-3.5）€2020/t，按 7.8 元/EUR 折 19.5（11.7-27.3）元/t，
# 胺补充 0.2-0.3 kg/t（`dea_ccts`）。仅压缩的系统取 5：⚠ 假设（无出处；NPC 2019 与 PyPSA 把这类非能源运维
# 都放在固定运维里，本模型的固定运维已按 capex 的 5% 另计）。
INDUSTRY_CCS_CONSUMABLES_CNY_PER_T_CO2: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 15.0,
    SECTOR_STEEL_EAF: 15.0,
    SECTOR_CEMENT: 15.0,
    SECTOR_AMMONIA: 5.0,
    SECTOR_METHANOL: 5.0,
}
# 再沸器蒸汽由厂内燃煤锅炉产生：每 GJ 蒸汽的耗煤量 = 1 / 效率。
# 其 CO2 直接排空（改造捕集的是工艺流股，不是辅助锅炉——PKU/Baowu 案例也这样计，
# 辅助电厂按 743 g/kWh），所以 CCS 路线的净减排是捕集量减去这部分蒸汽 CO2。
# 2026-09-22 之前，模型把 ACCA21 单位成本视为已含能耗，且什么都不排空。
# 效率 0.88：DEA 311.1a 燃煤蒸汽锅炉年均净效率 89%（2030 年区间 87-90.8；`dea_iph`），是为捕集新建锅炉的口径；
# 若蒸汽取自存量工业锅炉，运行效率低得多（检索摘要称 60-72%，未核原文）。
INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY: Final[float] = 0.88
# 捕集岛的经济寿命 20 a：NPC 2019 的钢铁、水泥、合成氨、乙醇捕集改造都取 20 a（`npc2019dualchallenge`，经
# `pypsa_techdata`）；DEA 401 的技术寿命为 25 a（`dea_ccts`），可作敏感性。PKU/Baowu 假定 25 a；煤电侧的改造
# 捕集岛依附于剩余 15-25 a 的机组；两侧统一取 20 a，使残值规则对捕集岛一视同仁，无论它建在哪里。
INDUSTRY_CAPTURE_LIFETIME_YEARS: Final[int] = 20

# ACCA21 / China Energy News 的平准化捕集成本，CNY/吨，仅作交叉核对（2026-09-22 起不进
# 目标函数）。水泥 305-730（ACCA21；China Energy News 430-650），钢铁 348-560，煤电
# 300-450，高浓度煤化工 <100（ACCA21 105-250）。按 38 元/GJ 煤价、0.40 元/kWh 电价和
# 6%/20 a 计，上面的参数隐含：水泥 ~345，钢铁 ~325，高浓度 ~110 元/t——都在各自区间的
# 低端（钢铁比 ACCA21 的 348 低 7%），因为 capex 锚点是中国项目备案而非欧洲首台套
# （FOAK）估算，且能源按模型的煤价与电价计价，而不是按文献价格。
INDUSTRY_CAPTURE_COST_REFERENCE_CNY_PER_T: Final[dict[str, tuple[float, float]]] = {
    SECTOR_STEEL_BF: (348.0, 560.0),
    SECTOR_STEEL_EAF: (305.0, 730.0),
    SECTOR_CEMENT: (305.0, 730.0),
    SECTOR_AMMONIA: (105.0, 250.0),
    SECTOR_METHANOL: (105.0, 250.0),
}

# --- 工业捕集的用水惩罚，单位为每吨捕集 CO2 的 m3 ------------------------------------------
# 胺法捕集需要冷却与水洗。中国工业改造为 1.60-1.69 m3/t CO2
# （Wang, Wen & Xu, Nat Commun 16:4251, 2025）；取 1.65。
# 这一项叠加在该部门 GB/T 18916 单位产品取水量之上，正是它让工业 CCS 与煤电冷却
# 争夺同一份流域定额。
INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2: Final[float] = 1.65

# --- 氢替代路线 -----------------------------------------------------------------------------
# 氢路线能去除的 hub CO2 比例。没有一处取 1.0：每条路线都留有氢触及不到的残余。
#   steel_bf_bof  H2-DRI + EAF 替代焦炭还原。MPP 钢铁模型里 100% 绿氢 DRI-EAF 的直接排放为 0.0816 t/t 钢
#                 （"Emissivity wout CCS" 行，`mpp_steel`），按点源表长流程的 1.8 t/t 算，减排比例为
#                 1 - 0.0816/1.8 = 0.955，取 0.95（推导）。两条前提都未核：一是点源表的 1.8 为直接排放口径——
#                 同库电炉取 0.4 t/t，是 MPP 电炉直接排放 0.1632 的 2.5 倍，说明该库至少电炉因子很可能含了用电
#                 排放；1.8 与 MPP 同一行里 BAT 高炉-转炉的 1.794 相近，但数值相近说明不了口径。二是 TIMES 钢铁
#                 部门的 CO2 为直接排放口径（`scripts/build_sector_targets.py` 文件头写能源 + 过程，TIMES 文档未核）。
#                 2026-09-23 前取 0.85，把 EAF 用的网电排放也算作残余；按直接排放口径，那属于电力部门。
#   ammonia       绿氢完全替代气化 + 水煤气变换；公用工程和空分装置仍在。⚠ 假设（无直接出处）：
#                 只去掉过程排放时约 0.67，连公用工程一起电气化时接近 1.0（只见检索摘要），0.95 取上端。
#   methanol      绿氢调节 H/C 比，并不去除碳原料，所以残余比合成氨大。⚠ 假设（无直接出处）：
#                 绿氢耦合煤制甲醇的案例减排约 70%-98%（化工学报 2022 等，只见检索摘要）。
INDUSTRY_H2_ABATEMENT_FRACTION: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 0.95,
    SECTOR_AMMONIA: 0.95,
    SECTOR_METHANOL: 0.90,
}

# 氢路线的净增量成本，单位为每吨产品的 CNY，对应旁边给出的参考氢价。"净" = 绿氢采购
# + 新路线 CAPEX + 运维 - 省下的化石原料/还原剂。模型按求解年份的氢价重新计价：
#
#     premium(P) = premium_ref + h2_intensity_t_per_t * 1000 * (P - P_ref)   [CNY / t product]
#
# 即在文献点附近做一阶展开。在氢价维度上精确——氢价是主导项（据 NER 2024，占绿色
# 甲醇成本的 ~70%）——其余各方面保持不变。这里 `h2_intensity` 不是常数：它取自点源表
# 自身的 h2_demand_kt_per_year / production_kt_per_year（每吨钢 / 氨 / 甲醇分别为
# 81 / 180 / 190 kg H2），所以斜率逐 hub 不同。
#
# 锚点，均为中国：
#   steel     氢价 5 USD/kg H2 时绿色溢价 ~225 USD/t 粗钢（Transition Asia / Global
#             Efficiency Intelligence, Green Steel Economics 2024）-> 按本仓库的
#             7.0 CNY/USD 折合 1575 CNY/t，参考价 35.0 CNY/kg。
#   ammonia   绿氢合成氨 ~2870 CNY/t NH3，对比煤制合成氨 2380-2560 CNY/t
#             （China Energy News 2023）-> 溢价 ~400 CNY/t，对应该案例风光假设所隐含的
#             ~12 CNY/kg 氢价。
#   methanol  氢价 15-18 CNY/kg 时绿氢甲醇 4500-5500 CNY/t，对比化石路线 ~2000 CNY/t
#             （NER 2024）-> 溢价 ~3000 CNY/t，参考价 16.5 CNY/kg。
# 三个锚点有意落在三个不同的参考氢价上：每个都按其来源自身假设的氢价引用，再由展开式
# 各自移到模型的氢价。先把它们平均到同一参考价只会丢信息，而不会增加信息。
INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT: Final[dict[str, tuple[float, float]]] = {
    # 部门: (premium_cny_per_t_product, reference_h2_price_cny_per_kg)
    SECTOR_STEEL_BF: (1575.0, 35.0),
    SECTOR_AMMONIA: (400.0, 12.0),
    SECTOR_METHANOL: (3000.0, 16.5),
}

# 氢路线的改造 capex，单位为每吨年产能的 CNY；在能力存量的增量上一次性计入（2026-09-23 前
# 计在路线份额的增量上），并为残值规则按直线折旧。
#   steel_bf_bof  在现有厂址用 H2-DRI 竖炉 + EAF 替代 BF-BOF（烧结、焦炉、BF 和 BOF
#                 弃用；铸造和轧制保留）：
#                 竖炉：宝钢湛江百万吨级氢基竖炉 总投资 18.9 亿元，对应 1.0 Mt/a DRI
#                 （中国钢铁新闻网 2022-02-17；2023-12 投产）-> 1 890 元/(t DRI·a)，
#                 ×1.08 t DRI/t 粗钢 = 2 040；EAF 184 EUR/(t·a)（Vogl, Åhman & Nilsson
#                 2018, J Clean Prod 203:736, cost assumptions）~ 1 430 元/(t·a)，按
#                 7.8 元/EUR 折算（约为 2018 年、2024 年的年均汇率 7.81、7.79，美联储 H.10
#                 交叉汇率，`fred_h10`）——未找到中国 EAF 的每吨投资备案（"80 t 电炉 5 000 万元"
#                 这类数字只含炉体）。合计 ~3 470；取 3 500。
#   ammonia       现有煤制合成氨厂改用外购绿氢：Haber-Bosch 回路与空分装置保留，
#                 气化炉、变换和净化工段停用。新增：H2 接收/压缩、N2 接入、控制系统。
#                 ⚠ 假设（无出处）：500 元/(t·a) ~ 绿地合成岛的 8%（875 USD/(t·a)，
#                 `constants.NH3_HB_CAPEX_USD_PER_TONNE_YEAR`）。未找到中国的改造
#                 备案；Yara Pilbara（2022-24）只公布了电解槽一侧。
#   methanol      绿氢耦合煤制甲醇：氢替代变换段来调节 H/C 比，合成回路保留。接入范围
#                 与合成氨相同；⚠ 假设（无出处）：500 元/(t·a)。
INDUSTRY_H2_ROUTE_CAPEX_CNY_PER_T_PRODUCT_YR: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 3500.0,
    SECTOR_AMMONIA: 500.0,
    SECTOR_METHANOL: 500.0,
}
# 新路线设备的固定运维，按每年占 capex 的比例计。IEAGHG 2013/04 Tables 6-7：
# 钢厂维护费 142 M$/a，安装成本 3 928 M$，即 3.6%/a；取 3.5%。
INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION: Final[float] = 0.035
# 重建路线（DRI 竖炉、EAF、合成接入）的经济寿命：25 a（PKU/Baowu 对捕集改造用
# 25 a；DRI/EAF 模块是比它更长寿的资产）。⚠ 假设（无直接出处）：开源数据给出 20-40 a，即 MPP 钢铁模型的
# 投资周期 20 a、钢厂寿命 40 a（`mpp_steel`），DEA 合成氨与甲醇的技术寿命 30 a（`dea_renewable_fuels`）。
INDUSTRY_H2_LIFETIME_YEARS: Final[int] = 25

# 上面的文献溢价锚点是平准化的（其中含路线自身的资本回收）。现在 capex 已显式给出，
# 锚点被分解而不是丢弃：
#     premium_ref = k * P_ref + capex * (CRF(r, life) + fom) + opex_delta_nonH2
# 因此 `opex_delta_nonH2`——相对现有化石路线的非氢运行差额（EAF/压缩机用电、省下的
# 焦炭或煤、省下的化石路线运维）——就是锚点扣除其中的氢和资本后所隐含的值。
# 钢铁的这一项为负（省下的焦炭与 BF opex 超过 EAF 电费），这是锚点自身算术所迫；
# 模型的下限（固定运维 + opex_delta + H2 采购 >= 0，capex 年金始终支付）依然成立。

# 走氢路线的 hub 的厂内用水：取该部门的先进值，而不是通用值。这不是猜——先进值按定义
# 就是适用于新建与改建工厂的定额，而氢路线就是一次改建。GB/T 18916 没有 H2-DRI 行，
# 另一条路是自己编一行。设为 False 则厂内用水改为保持通用值。
INDUSTRY_H2_USES_ADVANCED_QUOTA: Final[bool] = True

# 未建模，而且这一点必须保持醒目：电解制氢的原水需求为 10-22 L/kg H2
# （Arup, "Water for Hydrogen" 2022；Energy UK 2022 review），它落在电解槽所在的流域，
# 而不是工业 hub 所在的流域。煤电侧的掺氨同样不计这部分水，所以不计它至少在两边是
# 一致的——但按工业全面替代所需的 103 Mt/yr 氢计，它是 1.0-2.3e9 m3/yr，即 10-23
# 亿 m3，是流域 K 全部被执行余量的 5-11 倍。以后无论怎么补上，它都属于供给节点，
# 而且两类用氢方必须同时计费。
INDUSTRY_ELECTROLYSIS_WATER_L_PER_KG_H2: Final[tuple[float, float]] = (10.0, 22.0)


def capital_recovery_factor(rate: float, life_years: int) -> float:
    """资本回收系数，即年金系数的倒数。

    Args:
        rate: 年贴现率。
        life_years: 经济寿命（年），下限为 1。

    Returns:
        按 `rate` 在 `life_years` 年内还清单位资本所需的每年支付额。
    """
    n = max(1, int(life_years))
    if rate <= 1e-9:
        return 1.0 / n
    return rate / (1.0 - (1.0 + rate) ** (-n))


def _require(table: dict[str, float], sector: str, what: str) -> float:
    if sector not in table:
        raise KeyError(
            f"no {what} sourced for sector {sector!r}; refusing to guess. "
            f"Sourced sectors: {sorted(table)}"
        )
    return float(table[sector])


def capture_capex_cny_per_t_yr(sector: str) -> float:
    """捕集岛的改造 capex，单位为每吨年捕集能力的 CNY。

    Args:
        sector: `INDUSTRY_SECTORS` 之一。

    Returns:
        2030 基准年的 capex，未计学习曲线与情景乘子。

    Raises:
        KeyError: 该部门的捕集 capex 尚无出处。
    """
    return _require(INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR, sector, "capture capex")


def capture_variable_cost_cny_per_t(
    sector: str, coal_price_cny_per_gj: float, electricity_price_cny_per_mwh: float
) -> float:
    """按给定燃料与电力价格计的每吨捕集 CO2 的能源与耗材成本。

    Args:
        sector: `INDUSTRY_SECTORS` 之一。
        coal_price_cny_per_gj: 用于产生再沸器蒸汽的煤的到厂价。
        electricity_price_cny_per_mwh: 所计价年份的电价。

    Returns:
        每吨捕集 CO2 的 CNY：蒸汽用煤 + 电力 + 耗材。
    """
    steam_gj = _require(INDUSTRY_CCS_STEAM_GJ_PER_T_CO2, sector, "capture steam duty")
    kwh = _require(INDUSTRY_CCS_ELECTRICITY_KWH_PER_T_CO2, sector, "capture electricity")
    consumables = _require(INDUSTRY_CCS_CONSUMABLES_CNY_PER_T_CO2, sector, "capture consumables")
    steam_coal = steam_gj / INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY * float(coal_price_cny_per_gj)
    return steam_coal + kwh / 1000.0 * float(electricity_price_cny_per_mwh) + consumables


def capture_steam_co2_t_per_t(sector: str, coal_emission_factor_t_per_gj: float) -> float:
    """产生再沸器蒸汽所排空的 CO2，单位为每吨捕集 CO2 对应的吨数。"""
    steam_gj = _require(INDUSTRY_CCS_STEAM_GJ_PER_T_CO2, sector, "capture steam duty")
    return steam_gj / INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY * float(coal_emission_factor_t_per_gj)


def levelised_capture_cost_cny_per_t(
    sector: str,
    discount_rate: float,
    coal_price_cny_per_gj: float,
    electricity_price_cny_per_mwh: float,
    learning: float = 1.0,
) -> float:
    """capex + 运维 + 能耗参数折合成的平准化成本，单位为每吨捕集量的 CNY。

    只作报告与交叉核对（对照 `INDUSTRY_CAPTURE_COST_REFERENCE_CNY_PER_T`）；
    目标函数从不使用平准化数值。
    """
    capex = capture_capex_cny_per_t_yr(sector) * float(learning)
    crf = capital_recovery_factor(discount_rate, INDUSTRY_CAPTURE_LIFETIME_YEARS)
    return capex * (crf + INDUSTRY_CCS_FIXED_OM_FRACTION) + capture_variable_cost_cny_per_t(
        sector, coal_price_cny_per_gj, electricity_price_cny_per_mwh
    )


def h2_route_capex_cny_per_t_yr(sector: str) -> float:
    """氢路线的改造 capex，单位为每吨年产能的 CNY。"""
    return _require(INDUSTRY_H2_ROUTE_CAPEX_CNY_PER_T_PRODUCT_YR, sector, "H2-route capex")


def h2_route_annual_capital_cny_per_t(sector: str, discount_rate: float) -> float:
    """氢路线的 capex 年金加固定运维，单位为每吨产品每年的 CNY。

    即锚点分解时作为资本从 `premium_ref` 中扣除的部分。注意：求解器的下限只把年金放在
    `max` 之外（固定运维属于年度项）；见 `h2_premium_cny_per_t`。
    """
    capex = h2_route_capex_cny_per_t_yr(sector)
    crf = capital_recovery_factor(discount_rate, INDUSTRY_H2_LIFETIME_YEARS)
    return capex * (crf + INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION)


def h2_route_opex_delta_cny_per_t(sector: str, h2_intensity_t_per_t: float, discount_rate: float) -> float:
    """氢路线相对现有路线的非氢运行差额，CNY/t 产品。

    由文献溢价锚点扣除其中的氢（按锚点自身的参考价）以及显式的 capex 年金 + 固定运维
    后反推得到。当省下的化石原料与现有路线运维超过新路线的电费时为负。

    与情景的 `industry_h2_cost_multiplier` 无关：该乘子只乘路线 capex 与随之的固定运维
    （与两侧 CCS 的乘子同口径），这个差额固定在乘子为 1 时反推的值。2026-09-23 前乘子也乘
    这里的路线自身成本，使锚点参考价下的平准化溢价恰为 `multiplier x premium_ref`。

    Args:
        sector: `INDUSTRY_SECTORS` 之一，且其 `SECTOR_HAS_H2_ROUTE` 条目为 true。
        h2_intensity_t_per_t: 该 hub 每吨产品所需 H2 的吨数。
        discount_rate: 情景贴现率，用于 capex 年金。

    Raises:
        KeyError: 该部门的氢溢价锚点或 capex 尚无出处。
    """
    if sector not in INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT:
        raise KeyError(
            f"no H2 premium anchor sourced for sector {sector!r}; refusing to guess. "
            f"Sourced sectors: {sorted(INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT)}"
        )
    premium_ref, price_ref = INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT[sector]
    hydrogen_at_ref = float(h2_intensity_t_per_t) * 1000.0 * float(price_ref)
    own_cost_at_ref = float(premium_ref) - h2_route_annual_capital_cny_per_t(sector, discount_rate)
    return own_cost_at_ref - hydrogen_at_ref


def h2_premium_cny_per_t(
    sector: str,
    h2_price_cny_per_kg: float,
    h2_intensity_t_per_t: float,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    multiplier: float = 1.0,
) -> float:
    """给定氢价下氢路线的平准化净溢价，单位为每吨产品的 CNY。

    只作报告（用于图的面板）；求解器对 capex 只计一次，并按链路买氢。与求解器自身的
    算术一致：capex 年金 + max(固定运维 + opex 差额 + 氢, 0)。这个下限就是
    `add_industry_year` 所施加的那个（含固定运维的年度成本 + 买氢 >= 0），capex 年金在
    它之外：无论氢变得多便宜，转换都不可能比继续运行现有路线已沉没的资产更便宜，而新
    路线的资本总是要付的。

    Args:
        sector: `INDUSTRY_SECTORS` 之一，且其 `SECTOR_HAS_H2_ROUTE` 条目为 true。
        h2_price_cny_per_kg: 所计价年份的绿氢到厂价。
        h2_intensity_t_per_t: 该 hub 每吨产品所需 H2 的吨数。
        discount_rate: 情景贴现率，用于 capex 年金。
        multiplier: 情景的 `industry_h2_cost_multiplier`，只乘路线 capex（年金与固定运维随之）；
            在锚点参考价下（地板不起作用时）溢价为
            `premium_ref + (multiplier - 1) x capex x (CRF + 固定运维比例)`。
    """
    capex = h2_route_capex_cny_per_t_yr(sector) * float(multiplier)
    annuity = capex * capital_recovery_factor(discount_rate, INDUSTRY_H2_LIFETIME_YEARS)
    fixed_om = capex * INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION
    opex_delta = h2_route_opex_delta_cny_per_t(sector, h2_intensity_t_per_t, discount_rate)
    hydrogen = float(h2_intensity_t_per_t) * 1000.0 * float(h2_price_cny_per_kg)
    return annuity + max(fixed_om + opex_delta + hydrogen, 0.0)
