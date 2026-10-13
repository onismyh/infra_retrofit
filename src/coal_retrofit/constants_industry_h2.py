"""工业氢替代路线参数与计价函数：各部门氢路线的减排比例、需氢强度、文献溢价锚点、改造 capex、固定运维、寿命与取水定额档位，
以及氢路线的年金化 capex 与运行成本差。成本口径同 `constants_industry_capture` 模块头的说明；每个数字要么附出处，要么标 `⚠ 假设`。
"""
from __future__ import annotations

from typing import Final

from .constants import capital_recovery_factor
from .constants_industry import SECTOR_AMMONIA, SECTOR_METHANOL, SECTOR_STEEL_BF
from .constants_industry_capture import _require

# --- 氢替代路线 -----------------------------------------------------------------------------
# 氢路线能去除的 hub CO2 比例。没有一处取 1.0：每条路线都留有氢触及不到的残余。
#   steel_bf_bof  H2-DRI + EAF 替代焦炭还原。MPP 钢铁模型里 100% 绿氢 DRI-EAF 的直接排放为 0.0816 t/t 钢
#                 （"Emissivity wout CCS" 行，`mpp_steel`），按点源表长流程的 1.8 t/t 算，减排比例为
#                 1 - 0.0816/1.8 = 0.955，取 0.95（推导）。两条前提都未核：一是点源表的 1.8 为直接排放口径——
#                 同库电炉取 0.4 t/t，是 MPP 电炉直接排放 0.1632 的 2.5 倍，说明该库至少电炉因子很可能含了用电
#                 排放；1.8 与 MPP 同一行里 BAT 高炉-转炉的 1.794 相近，但数值相近说明不了口径。二是 TIMES 钢铁
#                 部门的 CO2 为直接排放口径（`scripts/build_sector_targets.py` 文件头写能源 + 过程，TIMES 文档未核）。
#                 2026-09-23 前取 0.85，把 EAF 用的网电排放也算作残余；按直接排放口径，那属于电力部门。
#   合成氨、甲醇  2026-10-02 起不在这张表里：绿氢只替代原料制氢（气化 + 水煤气变换），燃料燃烧与公用工程的 CO2 不变，
#                 减排比例取 hub 的可捕集份额（`INDUSTRY_CAPTURABLE_SHARE`：煤头合成氨 0.75、气头 0.67、煤制甲醇 0.57，B 级），
#                 份额为 0 的 hub 不开放氢路线。此前氨 0.95、甲醇 0.90（⚠ 假设），按上述份额推算，隐含另外消除约 80%、77% 的
#                 燃料燃烧 CO2，模型没有那部分的成本。甲醇路线是绿氢耦合煤制甲醇：氢替代变换段调节 H/C 比，碳原料不变。
INDUSTRY_H2_ABATEMENT_FRACTION: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 0.95,
}
# 氢路线减排比例取可捕集份额的部门。
SECTORS_H2_ABATES_CAPTURABLE_SHARE: Final[frozenset[str]] = frozenset({SECTOR_AMMONIA, SECTOR_METHANOL})

# 氢路线每吨产品的需氢量，t H2/t 产品，覆盖点源表的 h2_demand / production（2026-10-02 起）；不在表里的部门照用点源表。
#   steel_bf_bof  0.063：PyPSA technology-data（`pypsa_techdata`）outputs/costs_2030.csv 的氢直接还原竖炉 hydrogen-input
#                 2.1 MWh H2/t HBI，按 LHV 33.33 MWh/t 折 63 kg/t，按 1 t HBI/t 钢折成吨钢；PyPSA 注明 MPP 文档为 63 kg/t，MPP
#                 原始输入的 73 "probably incorrect"。此前取点源表的 81（口径不明，可能含循环与加热用氢）；Vogl 2018 与 PyPSA-Eur
#                 为 51，GCAM 2025 年 61，化学计量下限 54。由下文钢铁溢价锚点的数字反推的斜率约 62.5（B，推算），与 63 一致。
INDUSTRY_H2_INTENSITY_T_PER_T: Final[dict[str, float]] = {
    SECTOR_STEEL_BF: 0.063,
}

# 氢路线的净增量成本，单位为每吨产品的 CNY，对应旁边给出的参考氢价。"净" = 绿氢采购
# + 新路线 CAPEX + 运维 - 省下的化石原料/还原剂。模型按求解年份的氢价重新计价：
#
#     premium(P) = premium_ref + h2_intensity_t_per_t * 1000 * (P - P_ref)   [CNY / t 产品]
#
# 即在文献点附近做一阶展开。在氢价维度上精确——氢价是主导项（据 NER 2024，占绿色
# 甲醇成本的 ~70%）——其余各方面保持不变。这里的 `h2_intensity` 即每吨产品的需氢量：长流程钢取上面
# `INDUSTRY_H2_INTENSITY_T_PER_T` 的 63 kg/t（2026-10-02 前取点源表的 81），合成氨、甲醇取点源表自身的
# h2_demand_kt_per_year / production_kt_per_year（180 / 190 kg/t，表里同一部门逐厂相同）。
#
# 锚点，均为中国：
#   steel     氢价 5 USD/kg H2 时绿色溢价 ~225 USD/t 粗钢（Transition Asia / Global
#             Efficiency Intelligence, Green Steel Economics 2024）-> 按本仓库的
#             7.0 CNY/USD 折合 1575 CNY/t，参考价 35.0 CNY/kg。
#   ammonia   绿氢合成氨 ~2870 CNY/t NH3，对比煤制合成氨 2380-2560 CNY/t
#             （China Energy News 2023）-> 溢价 ~400 CNY/t，对应该案例风光假设所隐含的
#             ~12 CNY/kg 氢价。
#   methanol  氢价 15-18 CNY/kg 时绿氢甲醇 4500-5500 CNY/t，对比化石路线 ~2000 CNY/t
#             （NER 2024）-> 溢价 ~3000 CNY/t，参考价 16.5 CNY/kg。这是全绿氢 CO2 加氢甲醇的成本，点源表的 190 kg/t 也是
#             CO2 加氢（CO2 + 3H2）的化学计量；模型的路线却是绿氢耦合煤制甲醇（减排 = 可捕集份额），按化学计量只需外购约
#             63–126 kg/t（随粗煤气的 H2/CO 比，推算）。两者口径不一，本轮未改（作者 2026-10-02），需氢量与成本很可能偏高。
# 三个锚点有意落在三个不同的参考氢价上：每个都按其来源自身假设的氢价引用，再由展开式
# 各自移到模型的氢价。先把它们平均到同一参考价只会丢信息，而不会增加信息。
INDUSTRY_H2_PREMIUM_CNY_PER_T_PRODUCT: Final[dict[str, tuple[float, float]]] = {
    # 部门: (premium_cny_per_t_product, reference_h2_price_cny_per_kg)
    SECTOR_STEEL_BF: (1575.0, 35.0),
    SECTOR_AMMONIA: (400.0, 12.0),
    SECTOR_METHANOL: (3000.0, 16.5),
}

# 氢路线的改造 capex，单位为每吨年产能的 CNY；在本年新建能力上一次性计入，能力按铭牌产能定规模、到寿命退出
# （2026-09-23 前计在路线份额的增量上，此后到 09-30 计在按产量定规模、永不退出的能力存量的增量上），并为残值规则
# 按直线折旧。
#   steel_bf_bof  在现有厂址用 H2-DRI 竖炉 + EAF 替代 BF-BOF（烧结、焦炉、BF 和 BOF
#                 弃用；铸造和轧制保留）：
#                 竖炉：宝钢湛江百万吨级氢基竖炉 总投资 18.9 亿元，对应 1.0 Mt/a DRI
#                 （中国钢铁新闻网 2022-02-17；2023-12 投产）-> 1 890 元/(t DRI·a)，
#                 ×1.08 t DRI/t 粗钢 = 2 040；EAF 184 EUR/(t·a)（Vogl, Åhman & Nilsson
#                 2018, J Clean Prod 203:736，文中的成本假设）~ 1 430 元/(t·a)，按
#                 7.8 元/EUR 折算（约为 2018 年、2024 年的年均汇率 7.81、7.79，美联储 H.10
#                 交叉汇率，`fred_h10`）——未找到中国 EAF 含辅助系统的全套每吨投资数据
#                 （"80 t 电炉 5 000 万元"这类数字只含炉体）。合计 ~3 470；取 3 500。
#   ammonia       现有煤制合成氨厂改用外购绿氢：Haber-Bosch 回路与空分装置保留，
#                 气化炉、变换和净化工段停用。新增：H2 接收/压缩、N2 接入、控制系统。
#                 ⚠ 假设（无出处）：500 元/(t·a) ~ 绿地合成岛的 8%（875 USD/(t·a)，
#                 `constants.NH3_HB_CAPEX_USD_PER_TONNE_YEAR`）。未找到中国改造项目只含
#                 接入部分的投资额；Yara Pilbara（2022-24）只公布了电解槽一侧。
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
# 重建路线（DRI 竖炉、EAF、合成接入）的经济寿命，也是在役年限（到寿命退出，`vintage`）：25 a（PKU/Baowu 对捕集改造用
# 25 a；DRI/EAF 模块是比它更长寿的资产）。⚠ 假设（无直接出处）：开源数据给出 20-40 a，即 MPP 钢铁模型的
# 投资周期 20 a、钢厂寿命 40 a（`mpp_steel`），DEA 合成氨与甲醇的技术寿命 30 a（`dea_renewable_fuels`）。
INDUSTRY_H2_LIFETIME_YEARS: Final[int] = 25

# 上面的文献溢价锚点是平准化的（其中含路线自身的资本回收）。现在 capex 已显式给出，
# 锚点被分解而不是丢弃：
#     premium_ref = k * P_ref + capex * (CRF(r, life) + fom) + opex_delta_nonH2
# 因此 `opex_delta_nonH2`——相对现有化石路线的非氢运行差额（EAF/压缩机用电、省下的
# 焦炭或煤、省下的化石路线运维）——就是锚点扣除其中的氢和资本后所隐含的值。
# 钢铁的这一项为负（省下的焦炭与 BF opex 超过 EAF 电费），这是锚点自身算术所迫；
# 求解器计入目标的仍是 max(0, 固定运维 + opex_delta_nonH2 + H2 采购)；capex 按本年新建能力一次计入，在 max 之外。

# 走氢路线的 hub 的厂内用水：取该部门的先进值，而不是通用值。这不是猜——先进值按定义
# 就是适用于新建与改建工厂的定额，而氢路线就是一次改建。GB/T 18916 没有 H2-DRI 行，
# 另一条路是自己编一行。设为 False 则厂内用水改为保持通用值。
INDUSTRY_H2_USES_ADVANCED_QUOTA: Final[bool] = True

# 未建模，而且这一点必须保持醒目：电解制氢的原水需求为 10-22 L/kg H2
# （Arup, "Water for Hydrogen" 2022；Energy UK 2022 review），它落在电解槽所在的流域，
# 而不是工业 hub 所在的流域。煤电侧的掺氨同样不计这部分水，所以不计它至少在两边是
# 一致的——但按工业全面替代所需的约 83 Mt/yr 氢计（2030 年产量；长流程钢 63 kg/t，合成氨、甲醇只计有纯流股的点源，
# 2026-10-02 起；按点源表的需氢量为 103 Mt/yr），它是 0.8-1.8e9 m3/yr，即 8-18 亿 m3，是流域 K 全部被执行余量的
# 4-9 倍（按 103 Mt/yr 为 10-23 亿 m3、5-11 倍）。以后无论怎么补上，它都属于供给节点，
# 而且两类用氢方必须同时计费。


def h2_route_capex_cny_per_t_yr(sector: str) -> float:
    """氢路线的改造 capex，单位为每吨年产能的 CNY。"""
    return _require(INDUSTRY_H2_ROUTE_CAPEX_CNY_PER_T_PRODUCT_YR, sector, "H2-route capex")


def h2_route_annual_capital_cny_per_t(sector: str, discount_rate: float) -> float:
    """氢路线的 capex 年金加固定运维，单位为每吨产品每年的 CNY。

    即锚点分解时作为资本从 `premium_ref` 中扣除的部分。注意：求解器里 capex 按本年新建能力
    一次计入、在 `max` 之外，固定运维属于年度项、在 `max` 之内；见 `add_industry_year`。
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
