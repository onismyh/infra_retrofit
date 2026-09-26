from __future__ import annotations


TARGET_GEO_CRS = "EPSG:4326"
PLANT_YEAR_BASIS = "2025"
STATIC_LAYER_YEAR_BASIS = "static_source_layer"
NETWORK_EDGE_CLASS_MAIN = "existing_main_corridor"
NETWORK_EDGE_CLASS_TRIANGULATION = "triangulation_candidate"
NETWORK_COORD_DECIMALS = 6
NETWORK_TRIANGULATION_CRS = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
NETWORK_TRIANGULATION_MAX_EDGE_KM = 500.0
NETWORK_TRIANGULATION_MAX_STRETCH = 1.2
NETWORK_EDGE_CLASS_DIRECT = "runtime_direct_fallback"
# 直线候选边的绕行系数。在本模型自己的走廊图层上实测：
# 62 段既有油气干线的直线距离为 21 418 km，实际走线长 24 338 km，
# 按长度加权的系数为 1.136（中位数 1.036，p90 1.306）。管道技术经济分析
# 通常借用国际线路的 1.2-1.4；中国已建干线网比这更直，因此以本国自己的
# 基础设施为锚更合适，而且与同一张表里的走廊边内部一致。既有走廊边
# 已带有其路由折线长度，不乘这个系数。
NETWORK_DETOUR_FACTOR = 1.136
# 每个厂都有通往其最近 k 个汇的直连候选弧，因此候选网络永远不会
# 留下到不了封存的源。SimCCS 保证的是同一个不变量：即使成本面经过
# 加权以抑制穿越，每个源和汇也必须至少由一条可行走廊连通。
NETWORK_DIRECT_SINK_TOP_K = 5

BIOMASS_GJ_PER_TONNE = 15.0
# 收购成本锚点：Wang & Cai 2024 (Nat Commun, SI Table 3) 10.3 USD/MWh × 7.0 / 3.6 ≈ 20.0。
# LOW/HIGH 是围绕基准值的情景区间，不是换算得来的。
BIOMASS_COST_BASE = 20.0
BIOMASS_COST_LOW = 18.0
BIOMASS_COST_HIGH = 26.0
BIOMASS_MATCH_BUFFER_KM = 150.0
BIOMASS_NODE_AGGREGATION_DEGREES = 0.25

NH3_H2_RATIO = 0.176
# 合成岛用电（合成回路 + 空分 + 辅助），kWh/kg NH3，按同址风光 LCOE 计价：⚠ 假设（无单一出处）。"合成回路 + 空分"
# 口径的开源数据在 0.55-1.17 之间：下端 0.55 = DEA 103 合成回路 0.34 + DEA 空分 0.25 MWh/t N2 x 0.839 t N2/t NH3
# （后一项 0.21；`dea_renewable_fuels`，经 `pypsa_techdata`）；上端是 Joule 2018
# （doi:10.1016/j.joule.2018.04.017）Table 13，经 PyPSA-Eur-Sec 配置转引。
NH3_HB_POWER_KWH_PER_KG = 0.74
# 液氨罐储存附加，USD/kg 通过量，定值（不随情景贴现率重算）：⚠ 假设（储存天数无出处）。罐价与寿命可查：
# Morgan 2013 的 9 kt 罐 8 M$(2010)、寿命 20 a（`morgan2013ammonia`，经 `pypsa_techdata`，后者另估固定运维 2%/a）。
# 按这些数与 6% 贴现，0.017 相当于常年保有约 47-65 天储量（罐价按 2025 年或 2010 年币值计）；保有 30 天约为 0.008-0.011。
NH3_STORAGE_ADDER_USD_PER_KG = 0.017
NH3_TRANSPORT_ADDER_USD_PER_KG = 0.0
# 纳入氨供给曲线的电解路线。中国已装机的电解槽中
# >90% 是碱性的，而 data/H2 中 PEM 图层的近期成本是 AE 图层的 2-3 倍
# （2025 年按产量加权的均值：AE 4.36 vs PEM 10.87 USD/kg H2）。两者都纳入
# 还曾把同一份风 / 光资源重复计算。只保留 AE，让曲线落在中国实际建设的
# 路线上；风光联合 H2/NH3 数据集到位后再复核。
NH3_ELECTROLYSIS_TECHS = ("AE",)
# Haber-Bosch 合成回路 + 空分装置 + 全厂辅助系统（balance of plant），按每吨年
# NH3 产能计。电解槽及其可再生能源供给已包含在 LCOH 中，所以这里只加
# 合成岛：新建绿氨 CAPEX 为 1300-2000 USD/(t·yr)，其中电解槽占直接成本的
# 40-50%，其余部分为 650-1100（取中点）。⚠ 假设（出处不具体：区间与电解槽占比都没有列出文献）。
# 按 30 年折成年金计入氨价：DEA 103 绿氨合成装置（不含电解与空分）的技术寿命为 30 a，PyPSA technology-data 把这一值
# 同时用于合成回路与空分（`dea_renewable_fuels`，经 `pypsa_techdata`）；2026-09-23 前取 20 a，无出处。30 a 是技术寿命，
# 按经济寿命折的线索指向 25 a（J. Cleaner Production 2026 一文转引 IEA 2021，只见检索摘要）；工业捕集岛取的是
# NPC 2019 的 20 a，不是 DEA 401 的技术寿命 25 a，两处口径不同。贴现率与模型其余部分相同：构建输入时按
# DEFAULT_DISCOUNT_RATE，求解时再按情景的 `discount_rate` 重算（`builders/supply.reprice_hb_capex`）。
# 2026-09-23 前这里单独用 8%。
NH3_HB_CAPEX_USD_PER_TONNE_YEAR = 875.0
NH3_HB_CAPEX_LIFETIME_YEARS = 30
AMMONIA_NODE_AGGREGATION_DEGREES = 0.5

# 水网格粗化：构建输入时只做一次
# （`builders.water.write_water_inputs` -> `coarsen_water_inputs`），求解时从不做。
WATER_COARSE_GRID_DEGREES = 2.0      # 从 0.5° 粗化到 2.0°

WATER_MATCH_BUFFER_KM = 200.0  # 200km 供水半径
# 参考文献键：richter2012。一种推定性的环境流量标准：保护 80% 的
# 日流量即可维持生态完整性，因此名义上有 20% 可取用。这里把它作为
# 单一全局系数施加于每个流域，这是方法部分必须写明的简化——Richter 的
# 标准是筛查级的默认值，而不是针对具体流域的分配；在针对具体流域的
# 研究中，海河与长江不会取同一个值。
WATER_EXTRACTABLE_FRACTION = 0.20  # 天然径流的 20% 可取用于工业冷却（80% 为环境流量）

# 求解器单位缩放：缩小系数范围，提高数值稳定性
# 流量变量用缩放后的单位；成本相应调整
BIOMASS_FLOW_SCALE = 1e6   # 求解器流量变量用 TJ 而非 GJ（÷1e6）
AMMONIA_FLOW_SCALE = 1e6   # 求解器流量变量用 kt 而非 kg（÷1e6）
WATER_FLOW_SCALE = 1e6     # 求解器流量变量用 Mm³ 而非 m³（÷1e6）

# 参考文献键：ndrc2015no9。
COOLING_WATER_INTENSITY_M3_PER_MWH = {
    "once-through": 0.35,   # 耗水口径，NDRC 2015 No.9：0.29-0.41
    "recirculating": 1.85,
    "air": 0.37,
}

# --- 按燃烧技术与冷却方式分列的取水与耗水 ------------------------------------------------
# 参考文献键：wang2023waterintensity（主要来源）、macknick2011（交叉核对）。
# 这两篇在此处和一幅图注中都有引用，但在 v5 之前一直没有收进 references.bib；
# 对本研究中分量最重的这张输入表来说，这种状态很不妥：每幅图里的
# 每个水量数字都由它得出。
#
# Wang F., Wang P., Xu M. (2023) Water 15:1167, Table 1——针对中国，也是找到的
# 唯一一个同时给出取水与耗水、有捕集与无捕集，并按燃烧技术和冷却方式
# 双重细分的来源。单位 m3/MWh。
#
# 与 Macknick et al. (2011) NREL/TP-6A20-50900 中的美国机组数据交叉核对
# （reference/water/macknick_osti1009674.pdf，1 m3 = 264.172 gal）：
#     冷却塔 亚临界          取水 2.01 (US) vs 2.31 (CN);  耗水 1.78 vs 2.01
#     冷却塔 超临界+CCS           4.25      vs 4.14;            3.20 vs 3.06
#     直流冷却 亚临界           102.54      vs 116.48
# 两个独立来源、不同国家，取水量相差 5-14%。
#
# 注意直流冷却的两种口径相差多远：取水约 100 m3/MWh，而耗水约
# 1 m3/MWh，因为凝汽器冷却水回到了河里。中国自己的
# 取水定额完全不计这部分回流，所以直流冷却的定额
# （0.19-0.72 m3/MWh）在三种冷却方式中最低，而不是最高。
WATER_INTENSITY_BY_TECH_M3_PER_MWH: dict[tuple[str, str], dict[str, float]] = {
    # (燃烧类别, 冷却类别)：取水 / 耗水，基准值与带捕集值
    ("subcritical", "once-through"):   {"withdrawal": 116.48, "consumption": 1.240,
                                        "withdrawal_ccs": 199.11, "consumption_ccs": 1.770},
    ("supercritical", "once-through"): {"withdrawal": 88.90, "consumption": 0.690,
                                        "withdrawal_ccs": 161.49, "consumption_ccs": 0.850},
    ("ultra-supercritical", "once-through"): {"withdrawal": 82.80, "consumption": 0.228,
                                              "withdrawal_ccs": 143.20, "consumption_ccs": 0.344},
    ("subcritical", "recirculating"):   {"withdrawal": 2.31, "consumption": 2.01,
                                         "withdrawal_ccs": 4.51, "consumption_ccs": 3.65},
    ("supercritical", "recirculating"): {"withdrawal": 2.19, "consumption": 1.61,
                                         "withdrawal_ccs": 4.14, "consumption_ccs": 3.06},
    ("ultra-supercritical", "recirculating"): {"withdrawal": 1.58, "consumption": 1.26,
                                               "withdrawal_ccs": 3.44, "consumption_ccs": 2.53},
    ("subcritical", "air"):   {"withdrawal": 0.23, "consumption": 0.20,
                               "withdrawal_ccs": 0.45, "consumption_ccs": 0.36},
    ("supercritical", "air"): {"withdrawal": 0.21, "consumption": 0.16,
                               "withdrawal_ccs": 0.41, "consumption_ccs": 0.31},
    ("ultra-supercritical", "air"): {"withdrawal": 0.15, "consumption": 0.12,
                                     "withdrawal_ccs": 0.34, "consumption_ccs": 0.25},
}

# --- 中国取水定额：实际计收水费所依据的口径 ---------------------------------------------
# 参考文献键：mwr_quota_2019。
# 《水利部关于印发钢铁等十八项工业用水定额的通知》, 燃煤发电 单位发电量取水量 基准值,
# m3/MWh，按冷却方式和机组容量分档。武汉产业能效指南 (2025) pp.88-89 有转载。
#
# 这是第三种口径，有别于 WATER_INTENSITY_BY_TECH_M3_PER_MWH 里的两种：
#   耗水   流域实际损失的水量     -> 可用水量约束
#   取水   全部引出的水量，含回流 -> 只报告，从不进约束
#   定额   中国计量并收费的水量   -> 水费
# 定额有意不计直流冷却的凝汽器水量，所以其直流冷却值
# （0.35-0.72）在三种冷却方式中最低，而不是最高者的约 100 倍。
CHINA_WATER_QUOTA_M3_PER_MWH: dict[tuple[str, str], float] = {
    ("recirculating", "<300"):    3.20,
    ("recirculating", "300"):     2.70,
    ("recirculating", "600"):     2.35,
    ("recirculating", ">=1000"):  2.00,
    ("once-through", "<300"):     0.72,
    ("once-through", "300"):      0.49,
    ("once-through", "600"):      0.42,
    ("once-through", ">=1000"):   0.35,
    ("air", "<300"):              0.80,
    ("air", "300"):               0.57,
    ("air", "600"):               0.49,
    ("air", ">=1000"):            0.42,
}


def quota_capacity_band(capacity_mw: float) -> str:
    """把机组铭牌容量映射到定额表的容量分档。"""
    if capacity_mw >= 1000.0:
        return ">=1000"
    if capacity_mw >= 600.0:
        return "600"
    if capacity_mw >= 300.0:
        return "300"
    return "<300"


# --- 各水资源一级区的官方水资源量，10^8 m3/yr --------------------------------------------
# 第三次全国水资源调查评价，1956-2016 年多年平均，转引自
# Wang G. et al. (2025) Advances in Water Science 36(6) Table 1
# （reference/water/wangGQ2025_adv_water_sci_36_948.pdf）。与该文的
# 汇总值内部一致：北方六区 5221，南方四区 23078，全国 28299。
#
# 用于逐流域对模拟径流做偏差校正：全球水文模型是按流域流量率定的，
# 在 0.5 deg 下即使全国总量看起来没问题，也带有很大的区域偏差
# （WaterGAP2-2e 复现中国总量的误差为 1.2%，却把海河流域高估到 2.43 倍）。
# 校正保留每个集合成员的变化率，只替换绝对水平。
#
# 编码 A 合并了松花江与辽河，因为 data/ChinaBasins/basin_l1.gpkg 就是这样合并的。
OFFICIAL_BASIN_WATER_1E8_M3: dict[str, float] = {
    "A": 1469.2 + 483.4,   # 东北诸河区 (松花江 + 辽河)
    "C": 327.6,            # 海河区
    "D": 702.8,            # 黄河区
    "E": 928.3,            # 淮河区
    "F": 9871.2,           # 长江区
    "G": 2694.5,           # 东南诸河区
    "H": 4758.6,           # 珠江区
    "J": 5753.8,           # 西南诸河区
    "K": 1310.1,           # 西北诸河区
}
# 相对官方基准估计模型偏差所用的时间窗。ISIMIP3b 的历史期（historical）
# 模拟止于 2014 年；官方序列止于 2016 年。
BIAS_BASELINE_WINDOW = (1956, 2014)

# GEM 燃烧技术标签 -> 上表中的三个类别。CFB 机组按蒸汽参数属于亚临界；
# IGCC（全国 0.2 GW）在来源中没有对应行，映射为超临界。
# `.../CCS` 后缀描述的是改造状态，而不是蒸汽循环。
COMBUSTION_CLASS_MAP = {
    "subcritical": "subcritical",
    "supercritical": "supercritical",
    "ultra-supercritical": "ultra-supercritical",
    "cfb": "subcritical",
    "igcc": "supercritical",
}

PLANNING_YEARS = [2030, 2040, 2050, 2060]
# 模型的贴现率缺省值（`OptimizationScenario.discount_rate`）。模型自己折现与折年金的地方都用
# 情景的这一个值，需要缺省贴现率时引用这里、不另写数字；氨价里的合成岛年金也按它重算。
# 外生的 LCOH（`data/H2`，工业买氢也用它）内含该数据集自己的资本成本率，仓库里没有记录，不随情景贴现率变；
# 缺省贴现率下，按仓库根氨供给曲线四个规划年供给量加权的总额算，它占氨价的 77-86%（README §0.1 (e)）。
# 取值 6%：⚠ 假设（待核规范原文）。《建设项目经济评价方法与参数（第三版）》（发改投资〔2006〕1325号）的社会折现率
# 为 8%，受益期长的项目可降低但不低于 6%（只见检索摘要，条款与后续修订未核）；Wang et al. 2025 取 5%（敏感性
# 3-8%），`reference/CCS_Network_Optimize.gms` 取 8%。
DEFAULT_DISCOUNT_RATE = 0.06
