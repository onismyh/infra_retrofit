"""工业 CO2 捕集（CCS 路线）参数与计价函数：可捕集份额、捕集岛改造 capex、固定运维、蒸汽与电耗、耗材、用水惩罚、寿命，
以及按模型煤价电价计的每吨捕集成本（资本回收系数在 `constants.capital_recovery_factor`）。成本口径（作者 2026-09-22 决定）见下方注释；每个数字要么附出处，
要么标 `⚠ 假设`。部门键与取水定额在 `constants_industry`，氢路线在 `constants_industry_h2`。
"""
from __future__ import annotations

from typing import Final

from .constants import capital_recovery_factor
from .constants_industry import (
    SECTOR_AMMONIA,
    SECTOR_CEMENT,
    SECTOR_METHANOL,
    SECTOR_STEEL_BF,
    SECTOR_STEEL_EAF,
)

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

# --- CO2 捕集：可捕集份额 ----------------------------------------------------------------------
# CCS 能捕集的 CO2 占排放的份额，按 (部门, 原料) 给；hub 的份额按成员点源的 CO2 加权（`industry_inputs.prepare_industry`）。
# 捕集量 = hub CO2 x 份额 x 捕集率，份额为 0 的 hub 不开放 CCS（2026-10-02 起；此前都按 1.0，化工的纯流股参数用在全厂 CO2 上）。
#   合成氨、甲醇  只有原料制氢（气化 + 变换）经低温甲醇洗脱出的高浓度流股按下文的纯流股参数（无再沸器蒸汽，450 元/(t·a)）
#                 捕集；燃料燃烧与公用工程的烟气不开放，本轮不加胺法路线（作者 2026-10-02）。B 级、待核：煤头合成氨 0.75
#                 （IEA 2021 Ammonia Technology Roadmap，检索摘要：原料流股约 75%、燃料燃烧 25%；另一摘要 65.2%–77.7%，来源未指认），
#                 气头合成氨 0.67（同一摘要，约 2/3）；煤制甲醇 0.57（检索摘要 53.5%–60%，来源未能唯一指认）。油头合成氨与电石炉、
#                 矿热炉尾气制甲醇按煤头取；焦炉煤气、天然气制甲醇取 0：这两类没有排放 CO2 的纯流股（推断，未见文献）。
#   长流程钢、水泥  1.0：胺法捕集全厂烟气，即原口径。
#   电炉钢        0：不开放 CCS（作者 2026-10-02）。它的捕集 capex 借用水泥的值、没有出处，点源表的 0.4 t/t 很可能含外购电
#                 排放（MPP 废钢电炉的直接排放为 0.163 t/t，`mpp_steel`）。
# 合成氨、甲醇的氢路线同样只替代原料制氢，减排比例也取这个份额（`SECTORS_H2_ABATES_CAPTURABLE_SHARE`）；混合原料的 hub 里，
# 两条路线都只作用于份额为正的点源（`industry_matrices`：能力的铭牌系数，氢路线的产量、需氢与取水）。
INDUSTRY_CAPTURABLE_SHARE: Final[dict[tuple[str, str], float]] = {
    (SECTOR_STEEL_BF, "default"): 1.0,
    (SECTOR_STEEL_EAF, "default"): 0.0,
    (SECTOR_CEMENT, "default"): 1.0,
    (SECTOR_AMMONIA, "Coal"): 0.75,
    (SECTOR_AMMONIA, "Gas"): 0.67,
    (SECTOR_AMMONIA, "Oil"): 0.75,
    (SECTOR_AMMONIA, "default"): 0.75,       # 原料为空 / 未映射时按煤头，同取水定额
    (SECTOR_METHANOL, "Coal"): 0.57,
    (SECTOR_METHANOL, "Coke oven gas"): 0.0,
    (SECTOR_METHANOL, "Gas"): 0.0,
    (SECTOR_METHANOL, "矿热炉尾气"): 0.57,
    (SECTOR_METHANOL, "电石炉尾气"): 0.57,
    (SECTOR_METHANOL, "default"): 0.57,      # 回退：煤头
}


def capturable_share(sector: str, feedstock: str = "default") -> float:
    """CCS 可捕集的 CO2 占排放的份额；表里没有的原料按该部门的 default 行。

    Raises:
        KeyError: 该部门在表里没有 default 行。
    """
    key = (sector, feedstock) if (sector, feedstock) in INDUSTRY_CAPTURABLE_SHARE else (sector, "default")
    if key not in INDUSTRY_CAPTURABLE_SHARE:
        raise KeyError(f"no capturable share for sector {sector!r}; refusing to guess")
    return float(INDUSTRY_CAPTURABLE_SHARE[key])


# --- CO2 捕集：改造 capex，单位为每吨年捕集能力的 CNY -------------------------------------
# 基准年 2030（由煤电改造同样使用的 `ccs_learning_factor` 对其缩放），含压缩至管输压力，只计捕集岛。水泥参照
# 三个中国项目公告；长流程钢参照两个项目公告、北大宝武案例与 IEAGHG 各一个值；电炉钢与合成氨 / 甲醇为 ⚠ 假设：
#   cement   中联水泥青州 20 万 t/a 全氧燃烧耦合碳捕集示范线 2.56 亿元 -> 1 280 元/(t·a)
#            （中国建材，2023 开工）；中联 20 万 t/a 捕集提纯项目 1.98 亿元 -> 990；
#            海螺白马山 5 万 t/a 5 000-5 500 万元 -> 1 000-1 100（2018 投运，含食品级提纯）。
#            0.05-0.2 Mt/a 规模下区间为 990-1 280；取 1 150（取法未记录；区间中点 1 135，
#            三个值的中位数 1 000-1 100）。
#   steel    包钢 200 万 t CCUS 一期 50 万 t/a 6.14 亿元 -> 1 228（全产业链一期，含部分输送，
#            上界）；日照钢铁 18 万 t/a 1.35 亿元 -> 750（2025 开工，捕集+资源化）；
#            宝武案例（PKU CCUS 2020，宝武湛江侧线装置，胺法）：捕集装置
#            CNY 360 M + 7% 业主费 + 20 M 流动资金 = 407 M，对应 0.5 Mt/a -> 814；
#            IEAGHG 2013/04 Table 6, Case 2A：捕集装置 US$(2010) 679 M，对应捕集量
#            ~4.7 Mt/a -> ~145 USD/(t·a) ~ 1 000 元/(t·a)。取 1 000 作中心值（四个值的中位数约 910）。
#   steel_eaf ⚠ 假设（无出处）。尾气稀薄且间歇，即难度高于水泥窑，所以取水泥的值。
#            EAF 占模型内工业 CO2 的 1.7%。2026-10-02 起电炉不开放 CCS（可捕集份额 0），这个值只在份额改回正数时用。
#   ammonia / methanol   高浓度（>95%）气化尾气：无需吸收，只需脱水 + 压缩/液化。
#            延长石油榆林煤化 30 万 t/a（2022）报告捕集全成本 105 元/t；扣除 ~110 kWh/t
#            压缩电耗（按 ~0.45 元/kWh 计）与 5 元/t 耗材（下文 `INDUSTRY_CCS_CONSUMABLES_CNY_PER_T_CO2`）
#            后，剩 50.5 元/t 用于资本 + 固定运维，在 CRF(6%, 20 a) + 5%/a 下折得 capex 约 368 元/(t·a)
#            （贴现率 4% 时约 409）。⚠ 假设：取 450，比按上列输入推得的值高 22%（105 元/t 是否含耗材原文
#            未核；不扣耗材约 405，高 11%）。按 2030 年情景电价 0.40 元/kWh（低于反推所用的 0.45），450 的
#            平准化成本约 111 元/t，比 105 高约 5%。仍未找到只含捕集岛的投资额：齐鲁石化 100 万 t/a 的
#            380 元/(t·a)（Lu 等 2025）含压缩、提纯、制冷、球罐装车、公用工程，及建设期利息与流动资金。
INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 1000.0,
    SECTOR_STEEL_EAF: 1150.0,
    SECTOR_CEMENT: 1150.0,
    SECTOR_AMMONIA: 450.0,
    SECTOR_METHANOL: 450.0,
}
# 捕集岛的固定运维，按每年占（经学习曲线调整后的）capex 的比例计，按建设年的 capex 与在役且在用的能力计（2026-09-30 起，
# `vintage`；此前按当年 capex x 当年捕集量）。与煤电侧所用的
# 5%/a 相同（An et al. 2025 Nat Commun SI Table 7：固定运维 / 投资 = 5.4%/a）；
# PKU/Baowu 案例为 407 M 投资对应 12 M/a = 2.9%/a，IEAGHG 2013/04 钢厂维护费为安装成本的
# 142/3 928 = 3.6%/a。5% 取的是保守一端。
INDUSTRY_CCS_FIXED_OM_FRACTION: Final[float] = 0.05
# 每吨捕集 CO2 的捕集能耗。在 `industry_year_data` 中按模型自己的煤价（蒸汽，经锅炉
# 产生）和情景电价计价——而不是按冻结的文献价格——这样工业侧与煤电侧的能耗惩罚同步变动。
#   胺法燃烧后捕集（钢铁高炉煤气 / 热风炉烟气、水泥窑烟气、EAF 尾气）：
#     再沸器蒸汽 2.8 GJ/t：IEAGHG 2013/04 MDEA/Pz 2.3 GJ/t（CSIRO 综述：2.5-2.7
#     可达），国能锦界第 2 代吸收剂 2.35 GJ/t（中国 CCUS 进展报告 2025 p.18），
#     MEA 3.0-3.5 GJ/t；对 2030 年的改造取 2.8（取法未记录；所列值区间 2.3-3.5 的中点 2.9，中位数约 2.4-2.5）。
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
# 蒸汽可有一部分取自窑 / 炉余热（2026-10-02 起）：份额按部门由 `OptimizationAssumptions.industry_capture_waste_heat_share`
# 给（水泥缺省 0.3，2026-10-07 起；此前各部门 0），蒸汽用煤与蒸汽 CO2 都乘 (1 − 份额)，少发的余热电见下。
INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY: Final[float] = 0.88
# 余热改供再生蒸汽后每 GJ 余热蒸汽少发的余热电，kWh（2026-10-07 起计入，按情景电价买电补上；此前不计）；有再沸器蒸汽而没有这个值的
# 部门（钢铁）设了正的余热份额时报错（`capture_variable_cost_cny_per_t`）。水泥 55（47-63），推算（B）：新型干法窑纯低温余热发电
# 30-40 kWh/t 熟料（检索摘要列有 30-38、32-40、30-40），用的是 3-5 bara 低压汽能取的那段废气热，按 0.3 x 2.8 GJ/t x 0.755 t 捕集量/t
# 熟料 = 0.634 GJ/t 熟料折算：35 / 0.634 = 55。份额 0.3 时整套余热发电都让给捕集，更高的份额线性外推会超过余热发电总量（0.53 时约 62 kWh/t 熟料）。
INDUSTRY_WASTE_HEAT_LOST_POWER_KWH_PER_GJ: Final[dict[str, float]] = {
    SECTOR_CEMENT: 55.0,
}
# 捕集岛的经济寿命 20 a：NPC 2019 的钢铁、水泥、合成氨、乙醇捕集改造都取 20 a（`npc2019dualchallenge`，经
# `pypsa_techdata`）；DEA 401 的技术寿命为 25 a（`dea_ccts`），可作敏感性。PKU/Baowu 假定 25 a；煤电侧的改造
# 捕集岛依附于剩余 15-25 a 的机组；两侧统一取 20 a，使残值规则对捕集岛一视同仁，无论它建在哪里。捕集能力按这个
# 寿命退出，份额仍在就得重建（2026-09-30 起，`vintage`）。
INDUSTRY_CAPTURE_LIFETIME_YEARS: Final[int] = 20

# ACCA21 / China Energy News 的平准化捕集成本，CNY/吨，仅作交叉核对（2026-09-22 起不进
# 目标函数）。水泥 305-730（ACCA21；China Energy News 430-650），钢铁 348-560，煤电
# 300-450，高浓度煤化工 <100（ACCA21 105-250）。按缺省煤价 38.2 元/GJ、0.40 元/kWh 电价和
# 6%/20 a 计，上面的参数隐含：水泥 ~346，钢铁 ~326，高浓度 ~111 元/t。水泥与高浓度落在 ACCA21
# 区间的低端，钢铁比其下限 348 低 6%。能源按模型的煤价与电价计价，而不是按文献价格；
# capex 的取值与出处见上方 `INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR` 的注释。
INDUSTRY_CAPTURE_COST_REFERENCE_CNY_PER_T: Final[dict[str, tuple[float, float]]] = {
    SECTOR_STEEL_BF: (348.0, 560.0),
    SECTOR_STEEL_EAF: (305.0, 730.0),
    SECTOR_CEMENT: (305.0, 730.0),
    SECTOR_AMMONIA: (105.0, 250.0),
    SECTOR_METHANOL: (105.0, 250.0),
}

# --- 工业捕集的用水惩罚，单位为每吨捕集 CO2 的 m3 ------------------------------------------
# 胺法捕集需要冷却与水洗。⚠ 出处待核：此前标注的"1.60-1.69 m3/t（Wang, Wen & Xu, Nat Commun 16:4251, 2025）"
# 经 2026-09-30 审查不成立（该文 SI Table 9 是 2.25-3.47 m3/t 的生命周期耗水）。量级佐证：Rosa 等 2021 的 CCS 用水综述
# （Renew. Sustain. Energy Rev. 138:110511）给 CCS 附加用水 0.74-575 m3/t（含各类捕集与冷却方式；范围取自摘要与检索摘录，
# 全文 2026-10-11 未能打开），胺法燃烧后捕集约 1.71 m3/t 只见二手转引。1.65 落在该量级内，维持（`docs/工业联合减排实现说明.md` 的补注）。
# 这一项叠加在该部门 GB/T 18916 单位产品取水量之上，正是它让工业 CCS 与煤电冷却
# 争夺同一份流域定额。
INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2: Final[float] = 1.65


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
    sector: str, coal_price_cny_per_gj: float, electricity_price_cny_per_mwh: float, waste_heat_share: float = 0.0
) -> float:
    """按给定燃料与电力价格计的每吨捕集 CO2 的能源与耗材成本。

    Args:
        sector: `INDUSTRY_SECTORS` 之一。
        coal_price_cny_per_gj: 用于产生再沸器蒸汽的煤的到厂价。
        electricity_price_cny_per_mwh: 所计价年份的电价。
        waste_heat_share: 再沸器蒸汽中由窑 / 炉余热供给的份额，这部分不烧煤，但少发余热电
            （`INDUSTRY_WASTE_HEAT_LOST_POWER_KWH_PER_GJ`），按同一电价买电补上。

    Returns:
        每吨捕集 CO2 的 CNY：蒸汽用煤 + 电力（含少发的余热电）+ 耗材。

    Raises:
        KeyError: 部门缺参数；有再沸器蒸汽的部门设了正的余热份额、却没有少发余热电的值时也报错。
    """
    steam_gj = _require(INDUSTRY_CCS_STEAM_GJ_PER_T_CO2, sector, "capture steam duty")
    kwh = _require(INDUSTRY_CCS_ELECTRICITY_KWH_PER_T_CO2, sector, "capture electricity")
    consumables = _require(INDUSTRY_CCS_CONSUMABLES_CNY_PER_T_CO2, sector, "capture consumables")
    waste_heat_gj = steam_gj * float(waste_heat_share)
    if waste_heat_gj > 0.0:
        kwh += waste_heat_gj * _require(INDUSTRY_WASTE_HEAT_LOST_POWER_KWH_PER_GJ, sector, "waste-heat lost power")
    steam_coal = steam_gj * (1.0 - float(waste_heat_share)) / INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY * float(coal_price_cny_per_gj)
    return steam_coal + kwh / 1000.0 * float(electricity_price_cny_per_mwh) + consumables


def capture_steam_co2_t_per_t(sector: str, coal_emission_factor_t_per_gj: float, waste_heat_share: float = 0.0) -> float:
    """产生再沸器蒸汽所排空的 CO2，单位为每吨捕集 CO2 对应的吨数；余热供给的那部分蒸汽（`waste_heat_share`）不排。"""
    steam_gj = _require(INDUSTRY_CCS_STEAM_GJ_PER_T_CO2, sector, "capture steam duty")
    return steam_gj * (1.0 - float(waste_heat_share)) / INDUSTRY_CCS_STEAM_BOILER_EFFICIENCY * float(coal_emission_factor_t_per_gj)


def levelised_capture_cost_cny_per_t(
    sector: str,
    discount_rate: float,
    coal_price_cny_per_gj: float,
    electricity_price_cny_per_mwh: float,
    learning: float = 1.0,
) -> float:
    """capex + 运维 + 能耗参数折合成的平准化成本，单位为每吨捕集量的 CNY。

    只作报告与交叉核对（对照 `INDUSTRY_CAPTURE_COST_REFERENCE_CNY_PER_T`，按余热份额 0 计）；
    目标函数从不使用平准化数值。
    """
    capex = capture_capex_cny_per_t_yr(sector) * float(learning)
    crf = capital_recovery_factor(discount_rate, INDUSTRY_CAPTURE_LIFETIME_YEARS)
    return capex * (crf + INDUSTRY_CCS_FIXED_OM_FRACTION) + capture_variable_cost_cny_per_t(
        sector, coal_price_cny_per_gj, electricity_price_cny_per_mwh
    )
