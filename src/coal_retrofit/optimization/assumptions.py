"""求解器的技术经济假设 `OptimizationAssumptions`；分省煤价、分省利用小时的缺省表与输入表省名的别名表在 `constants_province`，
情景层的 `OptimizationScenario` 在 `scenario`。
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..constants import COOLING_WATER_INTENSITY_M3_PER_MWH, PLANNING_YEARS, cooling_class
from ..constants_province import PROVINCE_COAL_COST_CNY_PER_GJ, PROVINCE_NAME_ALIASES, PROVINCE_OPERATING_HOURS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationAssumptions:
    capacity_factor: float = 0.55
    # 煤的排放因子，t CO2/GJ：等于 2026-10-02 前全国一个数的写法 0.82 t/MWh ÷ 8.5714 GJ/MWh（= 3.6 / 0.42），
    # 95.7 kgCO2/GJ，介于 IPCC 2006 其他烟煤与次烟煤的缺省值（94.6、96.1）之间。煤电基线排放 = 发电量 x hub 毛热耗（`plants.csv` 的
    # `heat_rate_gj_per_mwh`，逐台分档煤耗按装机加权，`builders.plants.unit_heat_rate_gj_per_mwh`）x 此因子；
    # 工业捕集再生蒸汽的排放用同一因子。
    coal_emission_factor_t_per_gj: float = 0.82 / 8.5714
    # 工业捕集的再生蒸汽中由窑 / 炉余热供给的份额，按部门（键为 `constants_industry.INDUSTRY_SECTORS`，没写的部门为 0），
    # 蒸汽用煤与蒸汽的 CO2 都乘 (1 − 份额)（`industry_matrices`，2026-10-02 起），少发的余热电按情景电价买电补上
    # （`constants_industry_capture.INDUSTRY_WASTE_HEAT_LOST_POWER_KWH_PER_GJ`，2026-10-07 起）。水泥缺省 0.3（2026-10-07 起；此前各部门 0，
    # 即全部由燃煤锅炉供）：3-5 bara 低压汽只能把约 350 °C 的窑废气冷到约 155 °C，可取 0.58-0.67 GJ/t 熟料，合 90% 捕集所需热的
    # 0.27-0.32；CEMCAP 的最高情景为 30%（B，推算与检索摘要，方法论附录 B.5）。敏感性 0 与 0.53（0.4-0.5 t 低压汽/t 熟料推算的中点，
    # 热平衡上偏高，少发电也偏高），如 `--set 'assumptions.industry_capture_waste_heat_share={cement=0}' --as …`。
    industry_capture_waste_heat_share: dict[str, float] = field(default_factory=lambda: {"cement": 0.3})
    nh3_lhv_gj_per_kg: float = 0.0186
    # 美元汇率取整，约为 2023 年年均：美联储 H.10 年均 7.0809（`fred_h10`）；国家统计局 2023 年统计公报
    # 7.0467（只见检索摘要，未核原文）。模型里的美元参数分属不同价格年，都按这一个汇率折算，没有价格指数。
    usd_to_cny: float = 7.0
    retire_cost_cny_per_mwh: float = 450.0  # ⚠ 假设（无出处）
    # CCS/BECCS 捕集岛运维按每年 ccs_om_fraction x 改造 CAPEX 计（见下方 ccs_om_fraction），生物质与掺氨的掺烧运维按每年
    # biomass_upgrade_om_fraction / ammonia_upgrade_om_fraction x 掺烧能力的 capex 计（生物质 2026-10-02 起、掺氨 2026-10-10
    # 起），都不另设每 MWh 附加项：两者并存会把同一笔成本重复计算。此前生物质与 BECCS 的掺烧运维各 30 CNY/MWh，按改造路径的
    # 全部发电量收，与掺烧比例无关：Wang & Cai 2024 SI Table 3 的 γ2 18.85 $/kW/yr ~ 132 CNY/kW/yr 按约 4 400 h 折算（这个
    # 小时数无出处）；掺氨此前 80 CNY/MWh（无出处）。
    ccs_fixed_cost_cny_per_mwh: float = 0.0
    biomass_fixed_cost_cny_per_mwh: float = 0.0
    beccs_fixed_cost_cny_per_mwh: float = 0.0
    ammonia_fixed_cost_cny_per_mwh: float = 0.0
    # 学习参考年（2030）的捕集岛改造 CAPEX，含压缩，在既有 300-1000 MW 机组上做 90% 胺法
    # 捕集。文献综述 2026-09-10（docs/工业联合减排实现说明.md §9.6）：中值 3 500 CNY/kW，
    # 区间 2 700-4 400（区间的出处没有记录；主引、上沿与国能锦界原文未核，本地 PDF 的对照值见 README §0.2）。
    #   Yuan J-H et al. 2022, 气候变化研究进展 18(6) 764-776, Table 2：分省 3 318-3 925 CNY/kW，
    #     改造口径（主引）；
    #   Lockwood 2018, IEA Clean Coal Centre for CIAB, Table 4：4 121 CNY(2016)/kW，1000 MW
    #     USC，90%，捕集 + 压缩，2025-2030 年改造，作者自评偏保守（上沿）；
    #   An K et al. 2025, Nat Commun 16:2311, SI Table 7：2025 / 2030 年 381.9 / 305.5 $/kW
    #     （~2 670 / 2 140 CNY/kW），学习曲线下沿；
    #   国能锦界 4 Mt/a 全烟气 CCUS，600 MW 级机组上 19.9 亿元（2024 年备案）：
    #     ~3 000-3 300 CNY/kW，真实改造项目，含试点封存。
    # 2026-09-10 之前为 4 000（无出处）。BECCS 的捕集岛与 CCS 相同，capex 与固定运维都按此计；
    # 它的生物质改造只走掺烧档位 capex（`biomass_upgrade_capex_cny_per_mw_per_level`）。
    # 2026-09-23 之前另有 `beccs_retrofit_capex_cny_per_kw` = 4 500，在捕集岛之上再收
    # +1 000 CNY/kW 的"生物质改造增量"，与档位 capex 重复计费，已删除。
    ccs_retrofit_capex_cny_per_kw: float = 3500.0
    biomass_efficiency_penalty_per_ratio: float = 0.0373  # 15% 掺烧时效率下降 0.56%（Fan et al. 2023）
    coal_fuel_cost_cny_per_gj: float = 38.2             # 查不到省名时用；⚠ 假设（设定值，见 README §0.1）
    # 捕集岛固定运维，按每年占（经学习曲线调整的）改造 CAPEX 的比例计；2026-09-30 起按建设年的 CAPEX、只计在役且在用的捕集岛。
    # An et al. 2025 (Nat Commun) SI Table 7 给出煤电 CCS 改造在每个预测年的
    # 固定运维 / 投资 = 20.7/381.9 = 5.4%；此处取 5%。
    ccs_om_fraction: float = 0.05
    # CCS 能耗惩罚，表示为单位产出所需的额外燃料（无量纲，= 额外 GJ / 基线 GJ）。模型保持发电量不变、买入额外的煤，
    # 所以进入成本与排放的是额外燃料比。An et al. 2025 SI Table 7（PDF p20-21）的煤电能耗惩罚 2030/2040/2050/2060 年为
    # 22.2 / 15.6 / 13.3 / 11.1 %，是出力损失 p；净出力不变时要多烧 p/(1-p) 的煤，即下面四个数。2026-10-02 前是
    # (0.150, 0.1054, 0.0899, 0.075)：2030 年的 15% 无出处，之后按 An 的相对下降缩放。
    ccs_energy_penalty_ratio_by_year: tuple[float, ...] = (0.2853, 0.1848, 0.1534, 0.1249)

    # 分省煤价，CNY/GJ，缺省为 `constants_province.PROVINCE_COAL_COST_CNY_PER_GJ` 的副本（出处见该处）。
    province_coal_cost_cny_per_gj: dict[str, float] = field(default_factory=PROVINCE_COAL_COST_CNY_PER_GJ.copy)
    # 煤价乘子（2026-10-02 起，缺省 1.0，只作敏感性）：乘在上表与缺省煤价上，煤电与工业捕集蒸汽的用煤同时变。上表是 2023 价格年，
    # 2024–2025 年发电集团口径低 2%–11%、2025 年秦皇岛港口价低 14%（B），敏感性取 0.89、0.86，保留分省结构、整体平移
    # （docs/参数调研_20261001.md §2.3）。
    coal_price_multiplier: float = 1.0

    def province_coal_cost(self, province_name: str) -> float:
        """取某省的燃煤成本（CNY/GJ），已乘 `coal_price_multiplier`。

        省名用本表的写法（读入时已由 `canonical_provinces` 换过）；查不到时退回
        `coal_fuel_cost_cny_per_gj`，读入时已告警。
        """
        if not 0.0 < self.coal_price_multiplier < float("inf"):
            raise ValueError(f"coal_price_multiplier must be positive and finite, got {self.coal_price_multiplier}")
        price = self.province_coal_cost_cny_per_gj.get(province_name, self.coal_fuel_cost_cny_per_gj)
        return price * self.coal_price_multiplier

    def canonical_provinces(self, province_names: Iterable[object], source: str) -> list[str]:
        """把输入表的省名逐个换成分省煤价表的写法（`PROVINCE_NAME_ALIASES`），按原顺序返回。

        换过之后仍不在煤价表里的省名告警一次：这些行由 `province_coal_cost` 按
        `coal_fuel_cost_cny_per_gj` 计价。`source` 只进告警文字，指明是哪张表。
        """
        names = [PROVINCE_NAME_ALIASES.get(str(name), str(name)) for name in province_names]
        missing = sorted(set(names) - set(self.province_coal_cost_cny_per_gj))
        if missing:
            logger.warning(
                "%s: %d row(s) in province(s) %s have no coal price; priced at the default %.1f CNY/GJ",
                source, sum(name in missing for name in names), missing, self.coal_fuel_cost_cny_per_gj * self.coal_price_multiplier,
            )
        return names
    # 技术学习曲线（外生 Wright 定律）。
    # LR=15%，按累计捕集容量每翻一番计（文献中的捕集岛学习率：中国 IGCC+CC 为 9.6-20.2%，
    # Li et al. 2012 Appl. Energy；全链条 ~10%，DNV/Gassnova 2020），每 5.6 年翻一番——
    # 与中国 CCS 规模从目前 ~4 Mt/yr 扩大到 2060 年数百 Mt/yr 相符（35 年内 ~6 次翻番）。
    # 由此得到的随日历年的下降，按对数线性（过 2030 年）拟合到 An et al. 2025 SI Table 7
    # 的煤电 CCS 改造 CAPEX 轨迹（305.5 -> 142.5 $/kW，2030->2060）：
    #   模型因子 1.000 / 0.748 / 0.560 / 0.419   vs   来源 1.000 / 0.640 / 0.519 / 0.466
    # 即模型在本世纪中叶略偏保守，在 2060 年略偏乐观。
    ccs_learning_rate: float = 0.15
    ccs_deployment_doubling_years: float = 5.6
    ccs_learning_reference_year: int = 2030
    # 系统净成本框架
    # 基线燃煤运行的非燃料运维（2026-10-10 起；此前 80，无出处）：Wang & Cai 2024 SI Table 3 的固定运维 β2 41.66
    # (29.3-54.0) $/kW 与可变运维 β3 31.35 (24.0-38.7) $/kW（表里单位 $/kW，按每年读）合 511 CNY/kW/yr，按该 SI 数据处理
    # 说明 c 的全国平均利用小时 4 611.83 h 折成 110.8 (80.9-140.7) CNY/MWh，取 110。模型按发电量收，固定运维部分实际随
    # 利用小时变（作者决定 2026-10-10）。
    baseline_om_cost_cny_per_mwh: float = 110.0
    # 计算搁浅资产用的新建成本。对照 Fan et al. 2023（`fan2023cofiring`）SI Table 15 的煤电初始投资
    # 3 636 000 CNY/MW（取值比原文低 3.7%）；电规总院 2020 年水平 660-1 000 MW 超超临界 3 309-3 636 元/kW
    # （经《中国能源报》2023-04-24 转述，该文即按 3 500 元/kW 计；未核原文）。
    stranded_asset_base_cny_per_kw: float = 3500.0
    stranded_asset_accounting_life: int = 20           # 折旧年限：Wang et al. 2025 SI Table 3 折旧率 5%（3-8%），按每年直线折旧推得 20 a
    # 陆上封存成本，取 An et al. 2025 SI Table 7：5.0 (3.0-8.5) $/t = 35 (21-60) CNY/t（2026-10-02 前为 32）。每个汇的单价
    # = 本值 x 海上倍率 − EOR 容量份额 x EOR 抵扣（`data_prep._prepare_storages`）。
    storage_cost_cny_per_t: float = 35.0
    eor_credit_cny_per_t: float = 12.0  # EOR 每吨抵扣；⚠ 假设（无出处）
    # 海上汇的封存成本倍率（2026-10-02 起；此前海上、陆上同价）：REMIND `core/input/generisdata_tech.prn` 的海上、陆上注入
    # 投资 525 / 350（单位按表头推断，待核），固定运维占投资 0.12 / 0.06 每年，寿命都是 40 年；按模型贴现率 6% 年化后
    # (CRF + 0.12) x 525 / ((CRF + 0.06) x 350) = 2.21（贴现率 5%–8% 为 2.13–2.26；建设期 5 / 3 年按年中均匀投入计息、运维按隔夜投资计为 2.23），取 2.2。
    # 只按投资比为 1.5。与管道的 offshore_transport_multiplier 分开设，设 1.0 即不分陆海。
    offshore_storage_multiplier: float = 2.2
    # 管道运维（2026-10-02 起）：每年 pipe_fixed_om_fraction x 在役管道的 capex（建设年的单价，含类别与海上倍率），在役与
    # 流量上限同一判据，含到寿命后原址重铺的那一代，闲置的管也付（`model_costs._transport_storage_costs`）。4% 取 Fan et al.
    # 2023 SI p.12（PDF 第 13 页）式 (S29) 后：单位管长运维 = 单位管长建造成本（式 S26）x 4%，引其文献 18，原文未核；DEA 陆上 0.9% 作敏感性。
    pipe_fixed_om_fraction: float = 0.04
    # 按流量计的运输运维，CNY/(t·km)。2026-10-02 前为 0.15、管道没有固定运维：An et al. 2025 SI Table 7 的 CO2 运输全口径
    # 成本 0.026 (0.020-0.036) $/(t·km) = 0.182 CNY/(t·km) 减去 20 Mt 档满负荷的 capex 年化 ~0.029；满负荷时相当于每年
    # capex 的 15% / 21% / 38%（类别倍率 1.0 的边，2 / 5 / 20 Mt 档）。设回 0.15、pipe_fixed_om_fraction 设 0 即复原旧口径，两项同开会重复计费。
    route_opex_cny_per_t_km: float = 0.0
    # 海底管道单位长度造价相对陆上的倍数（作者定 1.5，2026-10-10）：只乘在每条边的海上段上（`offshore_length_km`，
    # 边的几何落在陆地省界之外的部分，`builders.network_offshore`；2026-10-10 前是凡接海上汇的边整条乘），capex 与按 capex
    # 比例计的固定运维随之同乘。文献（来源索引）：[R15] 转述 USAID 2002、ZEP 2011、Knoope 2014 为 1.6（1.5–2.0），
    # [A19] PyPSA technology-data（转录 DEA）海底管道投资为陆上 2.0 倍；1.5 在区间低端。
    offshore_transport_multiplier: float = 1.5
    # 沿既有油气干线的 62 条候选边（`existing_corridor_flag`）所记的免费 CO2 容量。
    # 2026-09-10 之前为 20 Mtpa（无出处）；文献综述（docs/工业联合减排实现说明.md §9.6）
    # 的结论相反：复用的输气管只能在降压下输送小流量（IEAGHG 2013/18：Longannet
    # 1.8 Mt/yr，OCAP 降压 56 -> 21 bar），而 20 Mt/yr、>150 km 的密相 CO2 因输气管壁厚
    # 被明确排除（Smith et al. 2021，经 Drax DR1907-6 转引；Anvari et al. 2025）。这里的
    # 走廊边平均 393 km（最短 22，最长 1 349），所以主线中任何复用抵扣都站不住。
    # 复用敏感性可设为 2.0（IEAGHG 案例规模）。
    existing_corridor_capacity_mtpa: float = 0.0
    standard_pipe_capacity_mtpa: float = 20.0
    max_parallel_pipes: int = 2
    pipeline_lifetime_years: int = 30
    # 管径分档。2026-09-10 之前每条边只有一种规格，即 20-Mtpa 干线，1-Mt/yr 的支线也要按
    # 20 Mtpa 付费：旧情景 `IND_BASE_t95`（已不在登记表）建成的 344 条边里有 222 条正好是 20；
    # 任何小到不值得建干线的流量，走 big-M "容量松弛"都比铺管便宜——于是有 11 条边在建成容量
    # 为零的情况下输送了 CO2。带规模经济的三档管径补上了这个缺口。各档每 km capex 由 20-Mtpa 干线费率
    # （0.4e6 CNY/(Mtpa·km) x 20 = 8.0e6 CNY/km，出处见下）乘 (cap/20)^0.6 缩放，
    # 这是 CO2 管道常用的管径-成本指数（Knoope et al. 2013, IJGGC 16:241, Table 4 拟合为
    # 0.5-0.7）。交叉核对：2-Mtpa 档的 2.0e6 CNY/km 低于 1.7-Mtpa 齐鲁-胜利管线的
    # 3.1e6 CNY/km，而后者含压缩站，所以小档若有偏差也是偏便宜。类别乘数（支线 1.35、
    # 直连 2.8、走廊 0.97）照旧叠加在上面。
    # 干线费率 0.4e6 CNY/(Mtpa·km) 是干线规模（>=20 Mtpa）的系数法估算：
    # MIT Smith et al. 2021 (IJGGC) $52,892/(in·mi) → 20-in 干线 ≈4.6M CNY/km ≈0.23e6 CNY/(Mtpa·km)；
    # ADB 中国系数 57,124 USD/(km·in) → ≈0.4e6；小规模实际项目（齐鲁-胜利，1.7 Mtpa，含站场）
    # 3.1e6。（原值 18,000 是单位错误，约低了 100 倍。）
    # ⚠ 假设（出处不具体）：取的是 ADB 系数，但没能定位它出自哪份 ADB 报告，同处的 Smith 系数也没核到原文。
    # 对照：按 Fan et al. 2023 SI 式 (S26)-(S28) 的材料法自算，20 Mt/a 时 0.12e6-0.16e6 CNY/(Mtpa·km)（式中 r 按
    # 半径读；原文称其为直径，按直径读是 0.47e6-0.65e6，原式待核；壁厚与保温层为自设）；吉林石化—吉林油田 CO2
    # 管道一期（13.67 亿元、282 km、3.3 Mt/a；只见环评公示的检索摘要）按 0.6 规模指数放大到 20 Mt/a 是 0.71e6。
    # 这一费率原先是字段 `pipe_capex_cny_per_mtpa_km`；分档之后它只流向一个没人读的报告系数，已删除。
    pipe_capacity_tiers_mtpa: tuple[float, ...] = (2.0, 5.0, 20.0)
    pipe_capex_cny_per_km_by_tier: tuple[float, ...] = (2.0e6, 3.5e6, 8.0e6)
    # 封存部署进度：全国各汇合计的年注入量上限（Mt/a，各规划年一个；空元组即不设），各汇另受自身年注入上限
    # （`injectivity_mtpa`，Fan 2025 逐格注入能力之和，不随年份变）约束（`model_year._add_injectivity_limit`）。取 ACCA21 (2021)
    # CCUS 路线图的区间中点：2030 年 0.2-4.08 亿 t（中点 2.1）、2050 年 6-14.5（10.2）、2060 年 10-18.2（14.1），2040 年在 2035 与
    # 2050 年的区间之间插值（~7.5）。作者定（2026-10-10）：部署进度直接作全国速率，各汇不再按进度缩放；此前是各汇可建速率 x 比例
    # (0.17, 0.58, 0.80, 1.00)，比例 = 上面的中点 ÷ 旧的全国可建速率 12.8 亿 t/a、以 1 封顶。
    storage_national_injection_mtpa_by_year: tuple[float, ...] = (210.0, 750.0, 1020.0, 1410.0)
    # 绿氨供给爬坡：各节点的绿氢供给逐年乘这一比例（`resource_access`）。供给曲线只是技术潜力（2025 年 8 551 Mt NH3/yr，
    # 而全机组 50% 掺烧只需 ~1 600 Mt），所以没有爬坡时节点上限永不绑定。默认全为 1，因为仓库里
    # 还没有有出处的中国绿氨建设轨迹；有了再设。
    ammonia_supply_deployment_fraction_by_year: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # 工业用氢由长管拖车从供给节点运送。临时值：取中国长管拖车成本的量级（每 100 km
    # ~3 CNY/kg，中国氢能联盟白皮书 2019 区间 2-4），待作者给出有出处的数值。作用同煤电侧
    # 的 `ammonia_transport_cost_cny_per_kg_km`。⚠ 假设（临时值；白皮书区间未核到原文）。
    h2_transport_cost_cny_per_kg_km: float = 0.03
    # 缺逐机组用水表时按冷却方式取的统一耗水值，与 `constants.COOLING_WATER_INTENSITY_M3_PER_MWH` 同一组数（耗水口径：
    # 直流 0.29-0.41 m³/MWh，依据 NDRC et al. 2015 No.9 清洁生产基准，中位数 ≈0.35；原值 1.0 混淆了取水口径与耗水口径）。
    cooling_once_through_water_intensity_m3_per_mwh: float = COOLING_WATER_INTENSITY_M3_PER_MWH["once-through"]
    cooling_recirculating_water_intensity_m3_per_mwh: float = COOLING_WATER_INTENSITY_M3_PER_MWH["recirculating"]
    cooling_air_water_intensity_m3_per_mwh: float = COOLING_WATER_INTENSITY_M3_PER_MWH["air"]
    ccs_water_multiplier: float = 1.82
    biomass_water_multiplier: float = 1.00
    ammonia_water_multiplier: float = 1.01  # 掺氨使电厂取水、耗水都 +1%，与掺氨档位无关；⚠ 假设（无出处）
    # 水预算：`OptimizationScenario.water_mode` 不为 "no_water" 时生效，只有 v9.1 起的官方指标口径一种。
    # 两条规则是两个口径不同的独立约束，各自约束其条文实际所针对的量：
    #    node  <= max(qtot x 0.20 - 生活与灌溉耗水, 煤电存量)   生态流量，作用于耗水（耗减规则）
    #    basin <= 用水总量控制指标 - 非电既有取水              分配规则，作用于取水（公报计量的正是它）
    # 分配规则直接从国办发〔2013〕2号读取，而不是靠假设；流域上限来自 `inputs/water_basin_caps.csv`
    # （`scripts/build_water_basin_caps.py`）。v9 及更早的 runoff 口径（available = qtot x 0.20 x
    # (1 - existing_withdrawal_share)，两个因子被别名化，无从区分生态流量标准与分配规则）已删除，
    # v9 结果按 CLAUDE.md §1.5 在 `892c877^` 里复现。
    #
    # 在优化模型中约束电力部门流域用水、且完全不设生态流量份额的先例：Zhang, He,
    # Johnston & Zhong (2021) J. Clean. Prod. 329:129765 (SWITCH-China)。电力占中国取水的
    # ~8%，却只占其耗水的 ~1%（Zhang et al. 2017, JCLP 161:1171-1179）。
    #
    # 流域上限开关。关闭时只剩生态流量的节点上限，这正是 v9.1 设计中的对照组：
    #
    #   BASE                 完全没有水约束
    #   *_oq_envonly         只有生态流量          （本开关为 False）
    #   *_oq                 生态流量 + 分配规则   （本开关为 True）
    #
    # 这一阶梯正是去别名化换来的：BASE->envonly 给生态流量标准定价，envonly->oq 给分配规则定价，
    # 各自独立。
    apply_basin_cap: bool = True
    # 向求解器提供水量时施加流域偏差校正因子。这些因子（builders/water_bias.py 中的
    # `.basin_bias_factors`，由各模型 `historical` 试验对照第三次水资源调查的流域总量估计）
    # 已固化进 water_availability.csv 的 `available_water_m3_per_year`；
    # `local_runoff_m3_per_year` 存的是同一个数的未校正值。设为 False 时提供未经校正的
    # 模拟径流，以量化约束对校正的敏感性（"校正开/关"稳健性检验）。作为假设项而不是重建
    # 开关保留，这样两次求解读的是同一个输入文件。
    apply_bias_correction: bool = True
    # --- 湿冷改空冷 ---------------------------------------------------------------------
    # 水账上最大的单一杠杆，也是中国电厂在河流来水不足时真正会动用的手段：把循环冷却
    # 凝汽器改为直接空冷，耗水从 ~1.3-2.0 降到 ~0.12-0.20 m3/MWh。加装捕集后差距更大
    # （2.5-3.7 vs 0.25-0.36），这就是为什么冷却方式能使 m3/tCO2 数值变动 ~5 倍，而改造
    # 路径只能使其变动 ~1.6 倍。冷却方式固定不变，会迫使所有对水的响应都经由提前退役，
    # 从而高估缺水的成本。
    #
    # 中国文献中的 capex 跨度很大（在既有机组上更换凝汽器与 ACC 岛大约 200-400 CNY/kW）；
    # 300 取中点，并做敏感性。⚠ 假设（出处不具体：没有列出具体文献或项目）。
    #
    # 空冷真正的主导成本是效率惩罚，而不是 capex。在已求解的算例中，能耗惩罚项是空冷改造
    # capex 的 11-23 倍，所以这个数远比上面的 300 CNY/kW 重要。其单位是净效率的百分点：
    # 空冷抬高汽轮机背压并增加风机厂用电负荷，所以每供出 1 MWh 要多烧煤、多排 CO2。
    #
    # 取 2.0 pp，不是以前用的 1.5 pp。旧值取的是 "1-2 pp" 区间的中点，但三个独立来源都把
    # 真实分布放在该区间的上半段或更高，1.5 比其中每一个都低：
    #
    #   * 同参数的中国机组，实测煤耗。600 MW 亚临界：湿冷 288 g/kWh vs 空冷 301（发电口径）；
    #     200 MW：315 vs 333。厂用电率从 8.2% 升到 8.8%，所以按净（供电）口径，600 MW 这一对
    #     是 313.7 -> 330.0 g/kWh。按 7000 kcal/kg 标准煤折算，即净效率 39.2% -> 37.2%
    #     = 1.93 pp（200 MW：2.11 pp）。
    #   * Zhang, Anadon, Mo, Zhao & Liu (2014), "Water-Carbon Trade-off in China's Coal Power
    #     Industry", Environ. Sci. Technol. 48(19), 11082。2012 年中国 ~130 GW 空冷机组比
    #     同等湿冷机组多排放 24.3-31.9 Mt CO2。按 4500-5000 运行小时与 820 g CO2/kWh 的基线
    #     摊到这批机组上，反推得 1.9-2.8 pp。
    #   * Qin et al. (2023), "Global assessment of the carbon-water tradeoff of dry cooling for
    #     thermal power generation", Nature Water 1(8), 682-693。机组级全球评估：空冷的能耗
    #     与 CO2 惩罚为发电出力的 1-15%，随地点与气候而异。2.0 pp 按 hub 效率（0.38-0.45）多耗 4.4%-5.2%
    #     燃料，落在该区间内；旧的 1.5 pp 在 η = 0.42 时为 3.6%，处在其下沿。
    #
    # 已知简化，且会让空冷显得偏便宜。这里的惩罚是常数。Qin et al. 发现惩罚的恶化快于环境
    # 温度的上升，而在中国北方枯水季与高温季重合——所以本处低估惩罚的时段恰恰是水最紧缺的
    # 时段。在把惩罚做成随温度变化之前，把 `air_retrofit_efficiency_penalty_pp` 设为 0.028
    # （v9 的 `SA_air_penalty_high`）是对此的粗略替代；做成随温度变化属于模型改动，而不是参数改动。
    allow_air_cooling_retrofit: bool = True
    air_retrofit_capex_cny_per_kw: float = 300.0
    air_retrofit_efficiency_penalty_pp: float = 0.020
    air_retrofit_lifetime_years: int = 20
    # --- 期末残值（作者决定 2026-09-22） ---
    # 目标函数中每一笔一次性 capex（煤电 CCS/BECCS 改造捕集岛、掺烧升级、空冷改造、管道、
    # 原址重建、工业捕集与氢路线）都按下面的经济寿命直线折旧；到规划期末（最后一个规划年
    # + 其区间长度，标准年份网格上为 2070）尚未折旧的部分返还抵扣，从该年折现。没有这一项
    # 时，2060 年的改造要为十年的使用付全部 capex，模型在最后一期投资不足；有了它，规划期
    # 内实际承担的 capex 就是资产服役年份内的折旧，这是与 capex 只计一次相一致的唯一核算
    # 方式。搁浅资产核销不计残值（它是损失，不是资产）。寿命：煤电捕集岛 20 a（同
    # `INDUSTRY_CAPTURE_LIFETIME_YEARS`；2026-09-30 起兼作在役年限，到寿命退出）；掺烧燃烧器升级 20 a；空冷 20 a（见上方
    # `air_retrofit_lifetime_years`；这两项 2026-10-02 起同样兼作在役年限）；管道 30 a（`pipeline_lifetime_years`）；
    # 重建的厂址 30 a（只作折旧年限：重建机组按设计寿命 40 a 服役，规划期内不到期）。这些寿命都是 ⚠ 假设（设定值，无文献）。
    # 工业捕集岛的 20 a 有出处（NPC 2019 的工业捕集改造，见 `constants_industry_capture.INDUSTRY_CAPTURE_LIFETIME_YEARS`），煤电捕集岛只是沿用同值，那份出处不含煤电。
    # 设 `end_of_horizon_salvage=False` 只去掉残值项；09-22 以来目标函数还有别的改动（README §0.1），复现不了更早的求解。
    end_of_horizon_salvage: bool = True
    ccs_retrofit_lifetime_years: int = 20
    blend_upgrade_lifetime_years: int = 20
    rebuild_lifetime_years: int = 30
    water_extraction_cost_cny_per_m3: float = 4.0       # 煤电取水单价（只用于煤电取水链路）；⚠ 假设（无出处）
    water_transport_cost_cny_per_m3_km: float = 0.05     # 水的管道 / 罐车运输；⚠ 假设（无出处）
    # 直连弧与支线的 capex 乘数：⚠ 假设（无出处）。
    direct_fallback_capex_multiplier: float = 2.8
    branch_capex_multiplier: float = 1.35
    # 在既有管道走廊内新铺管道：节省的是路权（以及未量化的审批时间）。NETL 2013
    # （DOE/NETL-2013/1614，转载于 IEAGHG 2013/18 Table 20）：ROW = 51 200 + 1.28 L
    # (577 D + 29 788) USD，即 24-40 英寸管道在 100 英里长度上 capex 的 2-3%。
    # 0.97 = 去掉这部分 ROW 份额。原为 0.4（无出处）；据称一篇德国拓扑论文用了 10% 的
    # 走廊折扣，但未能打开（ScienceDirect S2772656826001004）——须经作者核实后才可用 0.9。
    corridor_capex_multiplier: float = 0.97
    slack_penalty_cny_per_unit: float = 5_000_000_000.0
    sparse_interval_years: int = 10
    # 动态资源调配参数
    resource_match_radius_km: float = 200.0
    # 煤电机组掺烧生物质的全国上限，单位为每年 GJ（16 EJ；作者拍板，2026-09-10）。0.25 度
    # 节点层合计 ~18 EJ/yr 农林残余物（2026-10-11 前含能源作物 ~30 EJ），是与所有其他生物质用户共享的技术潜力；
    # 此上限是本研究允许全机组使用的份额。逐规划年施加，叠加在节点上限之上。
    # 0 或负值即关闭。2026-10-11 供给表改读情景 4 后作者决定暂不改。
    biomass_national_cap_gj_per_year: float = 16.0e9
    # hub 级的重建与掺烧档位决策取连续份额（作者拍板，2026-09-10）。一个 hub 聚合 ~10 台
    # 机组，所以"hub 的一部分重建"和"hub 的一部分改到掺烧档位 l"是真实的自由度；做成
    # one-hot 二元变量时松弛太弱，部门上限 MIP 跑 10 h 后仍停在 4-15% gap（下界一直停在
    # 根节点 LP 上）。管道保持整数。False 恢复二元形式。
    hub_decisions_continuous: bool = True
    # 2026-10-10 之前另有一条煤电掺烧绿氨的全国上限 `ammonia_fleet_cap_mt_by_year`（2 / 12 / 47 / 55 Mt NH3，2030-2060），
    # 已删：氨在节点上由绿氢合成，节点与全国都只设绿氢上限，煤电用氨按 NH3_H2_RATIO 折成氢计入（作者决定 2026-10-10）。
    # 所有用户从共享电解节点取用的绿氢全国上限（机组用氨按 NH3_H2_RATIO 折成 H2
    # + 工业用氢），每个规划年的 Mt H2。2030 与 2060 取自中国氢能联盟
    # 《中国氢能技术发展路线图研究》(2024-12)：2030 年可再生氢 3.5-6.5 Mt，2060 年
    # 75-162 Mt（分别取上沿 / 中值）；2040 与 2050 取自水电水利规划设计总院（澎湃 2024，
    # "据预测"，二手）：69 / 91 Mt。
    # 空元组 = 关闭。
    green_h2_national_cap_mt_by_year: tuple[float, ...] = (6.5, 69.0, 91.0, 120.0)
    # 掺烧比例升级的资本成本（每 MW 电厂容量、每升一个掺烧档位的 CNY）。生物质每档 175 元/kW：Fan et al. 2023 SI
    # 式 (S56) 的 500 MW 锅炉 15% 掺烧 50 USD/kW = 350 元/kW（Fan 转引 IEA 2019，未核）定在第 2 档（15%），三档依次
    # 175 / 350 / 525 元/kW；不用 (S56) 的规模指数 0.79（写法有歧义）。逐档线性这个形状：⚠ 假设（无出处）。Yuan et al.
    # 2022 的 10% / 15% / 20% 为 573 / 790 / 991 元/kW（未核），不是逐档线性。2026-10-02 前每档 500 元/kW、五档
    # （0.10–1.00），满档 2 500 元/kW 落在 Wang & Cai 2024 SI Table 3 的整包 985–3 596 元/kW 之内。
    biomass_upgrade_capex_cny_per_mw_per_level: float = 175_000.0
    # 生物质掺烧能力的固定运维，每年占其 capex 的比例：Fan et al. 2023 SI 式 (S57) 取 3%（不含燃料）。按在用的掺烧
    # 能力（各档位层之和）计，闲置的不付；单价不随年份变，按当年单价与按建设年单价计相同（2026-10-02 起，此前见上方
    # `biomass_fixed_cost_cny_per_mwh`）。
    biomass_upgrade_om_fraction: float = 0.03
    # 掺生物质比例的炉型上限（2026-10-02 起）：煤粉炉不超过 0.15（Fan et al. 2023 的基准掺烧比例），CFB 不超过 0.30
    # （调研稿 B 级，原文未取得：⚠ 假设）。hub 内落在煤粉炉上限以上档位的在用份额不超过该 hub 的 CFB 装机份额，
    # 落在 CFB 上限以上的为零（`constraints_blend.add_blend_level_constraints`）。都设 1.0 即不限。
    biomass_blend_max_pulverized: float = 0.15
    biomass_blend_max_cfb: float = 0.30
    ammonia_upgrade_capex_cny_per_mw_per_level: float = 25_000.0  # 最高档（50% 掺烧）≈125 CNY/kW。
    # 中国全改造文献：90 CNY/kW（CNERI 2025，100% 改造）+ 储存；MIT 供应系统 ≈163 CNY/kW
    # （Deng et al. 2024）；仅燃烧器 121.6 CNY/kW（Li & Li 2022，未确认）。三者口径各异，中位数
    # 121.6，取 125（每档 25 的整数倍），对应第 5 档；每档线性 25 CNY/kW，这个形状：⚠ 假设（无出处）。
    # （原值 800,000 是中国文献集中区间的 ~30 倍。）
    # 掺氨能力的固定运维，每年占其 capex 的比例，按在用的掺氨能力计，与生物质同口径（作者决定 2026-10-10；掺氨没有同口径
    # 的出处）。此前见上方 `ammonia_fixed_cost_cny_per_mwh`。
    ammonia_upgrade_om_fraction: float = 0.03
    # 生物质到厂成本构成（Wang et al. 2024, Nat Commun）
    # 收购价 base_cost_cny_per_gj 在供给曲线 CSV 里（`constants.BIOMASS_COST_BASE` = 20 元/GJ）。
    biomass_pretreatment_cost_cny_per_gj: float = 11.96     # γ₅: 6.15 $/MWh × 7.0 / 3.6 (Wang et al. 2024, Nat Commun)
    biomass_transport_fixed_cost_cny_per_gj: float = 13.13    # γ₆: 6.75 $/MWh × 7.0 / 3.6
    biomass_transport_variable_cost_cny_per_gj_km: float = 0.126  # γ₇: 0.065 $/(MWh·km) × 7.0 / 3.6
    # 氨运输成本（卡车，Hydrogen Council & McKinsey 2022；IEA GHR 2023）
    # 0.12 USD/(t·km) = 0.00012 USD/(kg·km) × 7.0 = 0.00084 CNY/(kg·km)
    ammonia_transport_cost_cny_per_kg_km: float = 0.00084

    # 分省年运行小时数（计算发电量时替代统一的 capacity_factor），缺省为 `constants_province.PROVINCE_OPERATING_HOURS` 的副本（出处见该处）。
    province_operating_hours: dict[str, float] = field(default_factory=PROVINCE_OPERATING_HOURS.copy)

    def province_cf(self, province_name: str) -> float:
        """由运行小时数求某省的容量因子。"""
        hours = self.province_operating_hours.get(province_name, self.capacity_factor * 8760.0)
        return hours / 8760.0

    def fixed_cost_cny_per_mwh(self, pathway: str) -> float:
        return {
            "unabated": 0.0,
            "retire": self.retire_cost_cny_per_mwh,
            "ccs": self.ccs_fixed_cost_cny_per_mwh,
            "biomass": self.biomass_fixed_cost_cny_per_mwh,
            "beccs": self.beccs_fixed_cost_cny_per_mwh,
            "ammonia": self.ammonia_fixed_cost_cny_per_mwh,
        }[pathway]

    def ccs_learning_factor(self, year: int) -> float:
        """按 Wright 定律学习曲线给出的外生成本下降因子。"""
        n_doublings = max(0, year - self.ccs_learning_reference_year) / self.ccs_deployment_doubling_years
        return (1.0 - self.ccs_learning_rate) ** n_doublings

    def ccs_energy_penalty_ratio(self, year: int) -> float:
        """CCS/BECCS 电厂在 `year` 的单位产出额外燃料（无量纲）。

        数值按 `constants.PLANNING_YEARS` 列表给出；不在网格上的年份取最近的列表点，
        与碳价、电价的查表方式一致。
        """
        if not self.ccs_energy_penalty_ratio_by_year:
            return 0.0
        return self._fraction_for_year(self.ccs_energy_penalty_ratio_by_year, year)

    def _fraction_for_year(self, values: tuple[float, ...], year: int) -> float:
        """按 `PLANNING_YEARS` 列表查 `year` 的值；不在网格上的年份取最近的列表点，空元组为 1.0。"""
        if not values:
            return 1.0
        mapping = dict(zip(PLANNING_YEARS, values, strict=False))
        if year in mapping:
            return float(mapping[year])
        nearest = min(mapping, key=lambda candidate: abs(candidate - year))
        return float(mapping[nearest])

    def storage_national_injection_mtpa(self, year: int) -> float:
        """`year` 全国各汇合计的年注入量上限（Mt/a）；没有设（空元组）时为 inf。"""
        if not self.storage_national_injection_mtpa_by_year:
            return float("inf")
        return self._fraction_for_year(self.storage_national_injection_mtpa_by_year, year)

    def ammonia_supply_deployment_fraction(self, year: int) -> float:
        """绿氨技术潜力中，在 `year` 已建成的份额。"""
        return self._fraction_for_year(self.ammonia_supply_deployment_fraction_by_year, year)

    def green_h2_national_cap_mt(self, year: int) -> float:
        """所有节点用户在 `year` 的全国绿氢上限，Mt H2（0 = 关闭）。"""
        if not self.green_h2_national_cap_mt_by_year:
            return 0.0
        return self._fraction_for_year(self.green_h2_national_cap_mt_by_year, year)

    def cooling_baseline_water_intensity(self, cooling_label: str) -> float:
        """缺逐机组用水表时的统一耗水值（m³/MWh），冷却方式按 `constants.cooling_class` 归类。"""
        return {
            "once-through": self.cooling_once_through_water_intensity_m3_per_mwh,
            "air": self.cooling_air_water_intensity_m3_per_mwh,
            "recirculating": self.cooling_recirculating_water_intensity_m3_per_mwh,
        }[cooling_class(cooling_label)]
