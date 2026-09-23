from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import DEFAULT_DISCOUNT_RATE, PLANNING_YEARS


PATHWAYS = ("unabated", "retire", "ccs", "biomass", "beccs", "ammonia")


@dataclass(frozen=True)
class OptimizationAssumptions:
    capacity_factor: float = 0.55
    # 全机组平均排放强度。保持 0.82 t/MWh：配合下面按效率锚定的热耗率，折合
    # 95.7 kgCO2/GJ，落在 IPCC 烟煤区间（~94.6-96.1）内。
    coal_emission_factor_t_per_mwh: float = 0.82
    # 由效率锚点推得：3.6 / 0.42 = 8.5714 GJ/MWh（原为 9.0，隐含 40% 效率，
    # 与 coal_plant_base_efficiency 不一致）。
    heat_rate_gj_per_mwh: float = 8.5714
    nh3_lhv_gj_per_kg: float = 0.0186
    # 美元汇率取整，约为 2023 年年均：美联储 H.10 年均 7.0809（`fred_h10`）；国家统计局 2023 年统计公报
    # 7.0467（只见检索摘要，未核原文）。模型里的美元参数分属不同价格年，都按这一个汇率折算，没有价格指数。
    usd_to_cny: float = 7.0
    retire_cost_cny_per_mwh: float = 450.0  # ⚠ 假设（无出处）
    # CCS/BECCS 捕集岛运维按每年 ccs_om_fraction x 改造 CAPEX 计（见下方 ccs_om_fraction），
    # 因此不再另设每 MWh 附加项：两者并存会把同一笔成本重复计算。BECCS 的每 MWh 项只保留
    # 生物质掺烧运维，与纯生物质路径承担的 30 CNY/MWh 相同（Wang & Cai 2024 SI Table 3，
    # gamma2 18.85 $/kW/yr ~ 132 CNY/kW/yr，按全机组平均运行小时数计；只计 gamma2，同表可变运维 gamma3 未计）。
    ccs_fixed_cost_cny_per_mwh: float = 0.0
    biomass_fixed_cost_cny_per_mwh: float = 30.0
    beccs_fixed_cost_cny_per_mwh: float = 30.0
    ammonia_fixed_cost_cny_per_mwh: float = 80.0  # 掺氨运维；⚠ 假设（无出处）
    # 学习参考年（2030）的捕集岛改造 CAPEX，含压缩，在既有 300-1000 MW 机组上做 90% 胺法
    # 捕集。文献综述 2026-09-10（docs/工业联合减排实现说明.md §9.6）：中值 3 500 CNY/kW，
    # 区间 2 700-4 400。
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
    # 全机组基准发电效率，低位热值口径，全期取常数。Fan et al. 2023 SI 式 (S42) 设煤电效率由 2020 年 0.4 平滑升至
    # 2060 年 0.5，按线性插值 0.42 约为其 2028 年值；Wang et al. 2025 SI Table 1（引 NDRC 2022 基准水平，
    # 285-323 gce/kWh）各机型折 0.380-0.431。文献值应是供电（净）口径，模型的发电量按利用小时计、应属毛口径
    # （两者都是推断，未核），差一个厂用电率，未修正。
    coal_plant_base_efficiency: float = 0.42
    coal_fuel_cost_cny_per_gj: float = 38.2             # 全国均值，按省查表时被覆盖
    # 捕集岛固定运维，按每年占（经学习曲线调整的）改造 CAPEX 的比例计。
    # An et al. 2025 (Nat Commun) SI Table 7 给出煤电 CCS 改造在每个预测年的
    # 固定运维 / 投资 = 20.7/381.9 = 5.4%；此处取 5%。
    ccs_om_fraction: float = 0.05
    # CCS 能耗惩罚，表示为单位产出所需的额外燃料（无量纲，= 额外 GJ / 基线 GJ）。
    # 取代旧的绝对效率惩罚点值：模型保持发电量不变、买入额外的煤，所以额外燃料比
    # 才是真正进入成本与排放的量。水平锚定在 2030 年的 15%（⚠ 假设：15% 这个水平无出处）；下降路径沿用
    # An et al. 2025 SI Table 7（煤电能耗惩罚 2030/2040/2050/2060 年为
    # 22.2 / 15.6 / 13.3 / 11.1 %，即归一化后 1.000 / 0.703 / 0.599 / 0.500）。
    ccs_energy_penalty_ratio_by_year: tuple[float, ...] = (0.1500, 0.1054, 0.0899, 0.0750)

    # 分省煤价，CNY/GJ = An et al. 2025（Nat Commun）SI Table 2 的 USD/GJ × 7。原表北京无煤价（9.92 $/GJ 是气价，
    # 2026-09-23 前误记为 69.4），京津两行气价、生物质价与潜力相同，取天津 5.51（38.6）；内蒙古取东、西两行 2.63 / 2.49 的均值。
    province_coal_cost_cny_per_gj: dict[str, float] = field(default_factory=lambda: {
        "Anhui": 42.8, "Beijing": 38.6, "Chongqing": 42.6,
        "Fujian": 40.0, "Gansu": 37.0, "Guangdong": 41.5,
        "Guangxi": 47.9, "Guizhou": 34.6, "Hainan": 33.6,
        "Hebei": 30.8, "Heilongjiang": 35.6, "Henan": 42.1,
        "Hubei": 46.2, "Hunan": 48.5, "Inner Mongolia": 17.9,
        "Jiangsu": 40.7, "Jiangxi": 48.2, "Jilin": 39.1,
        "Liaoning": 37.5, "Ningxia": 27.4, "Qinghai": 33.9,
        "Shaanxi": 31.0, "Shandong": 41.5, "Shanghai": 38.1,
        "Shanxi": 28.2, "Sichuan": 45.4, "Tianjin": 38.6,
        "Xinjiang": 16.8, "Yunnan": 29.6, "Zhejiang": 39.4,
    })

    def province_coal_cost(self, province_name: str) -> float:
        """取某省的燃煤成本（CNY/GJ）。查不到时退回全国均值。"""
        return self.province_coal_cost_cny_per_gj.get(province_name, self.coal_fuel_cost_cny_per_gj)
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
    baseline_om_cost_cny_per_mwh: float = 80.0       # 基线燃煤运行的非燃料运维；⚠ 假设（无出处）
    # 计算搁浅资产用的新建成本。对照 Fan et al. 2023（`fan2023cofiring`）SI Table 15 的煤电初始投资
    # 3 636 000 CNY/MW（取值比原文低 3.7%）；电规总院 2020 年水平 660-1 000 MW 超超临界 3 309-3 636 元/kW
    # （经《中国能源报》2023-04-24 转述，该文即按 3 500 元/kW 计；未核原文）。
    stranded_asset_base_cny_per_kw: float = 3500.0
    stranded_asset_accounting_life: int = 20           # 折旧年限；⚠ 假设（无出处）
    # 封存成本 32 CNY/t，对照 An et al. 2025 SI Table 7：5.0 (3.0-8.5) $/t = 35 (21-60) CNY/t。
    storage_cost_cny_per_t: float = 32.0
    eor_credit_cny_per_t: float = 12.0  # EOR 每吨抵扣；⚠ 假设（无出处）
    # --- hub 注入速率，由候选场址密度推得 ------------------------------------------------
    # 源栅格（Fan 5 km 网格；见 data/封存汇图层-Fan）按每个 5x5 km 格子存的是地层能接受的
    # 地质注入速率（全国均值 8.0，单格最大 134 Mt/a）——已用数据集自带的分省表核对，
    # 其 Max 列逐省复现了栅格的单格最大值。因此把一个盆地内的格子加总没有工程意义
    # （全国合计 555 Gt/yr）。
    # 可建速率改为如下推得：
    #     hub 注入能力 = (hub 内格子数 / storage_site_block_pixels) x 单场址项目速率
    # 即每个 50x50 km 区块（=100 个 5 km 格子；按压力干扰定的间距）一个封存项目，每个项目
    # 取真实项目规模（齐鲁-胜利 1 Mt/yr、Gorgon 设计 4 Mt/yr -> 中值 2 Mt/yr）。全国合计
    # = 1 350 Mt/yr，落在 ACCA21/CAEP 2060 年 CCUS 部署区间 1 000-1 800 Mt/yr 内。
    # max_hub_injectivity_mtpa 现在只是给个别异常 hub 的上限。三者都在
    # injectivity_multiplier 之前施加。
    storage_site_block_pixels: float = 100.0
    storage_site_project_rate_mtpa: float = 2.0
    max_hub_injectivity_mtpa: float = 200.0
    # 管道运维。An et al. 2025 SI Table 7 给出 CO2 运输全口径成本 0.026 (0.020-0.036)
    # $/(t·km) = 0.182 (0.14-0.25) CNY/(t·km)。本模型单独计管道 CAPEX（满负荷年化
    # ~0.029 CNY/(t·km)），所以 0.15 opex + 0.029 capex = 0.179 CNY/(t·km)（平准化）
    # 复现了来源的中值。
    route_opex_cny_per_t_km: float = 0.15
    # 海上盆地（东海、珠江口、渤海、北部湾）要承担海底管道与平台成本：凡与海上封存 hub
    # 相连的边，运输 CAPEX 与 OPEX 都乘以此系数。⚠ 假设（无出处）。
    offshore_transport_multiplier: float = 1.5
    pipe_capex_cny_per_mtpa_km: float = 400_000.0  # 干线规模（>=20 Mtpa）的系数法估算：
    # MIT Smith et al. 2021 (IJGGC) $52,892/(in·mi) → 20-in 干线 ≈4.6M CNY/km ≈0.23e6 CNY/(Mtpa·km)；
    # ADB 中国系数 57,124 USD/(km·in) → ≈0.4e6；小规模实际项目（齐鲁-胜利，1.7 Mtpa，含站场）
    # 3.1e6。基准费率对应标准的 20-Mtpa 管道；更小的管道另带支线 / 直连乘数。
    # （原值 18,000 是单位错误，约低了 100 倍。）
    # ⚠ 假设（出处不具体）：取的是 ADB 系数，但没能定位它出自哪份 ADB 报告，同处的 Smith 系数也没核到原文。
    # 对照：按 Fan et al. 2023 SI 式 (S26)-(S28) 的材料法自算，20 Mt/a 时 0.12e6-0.16e6 CNY/(Mtpa·km)（式中 r 按
    # 半径读；原文称其为直径，按直径读是 0.47e6-0.65e6，原式待核；壁厚与保温层为自设）；吉林石化—吉林油田 CO2
    # 管道一期（13.67 亿元、282 km、3.3 Mt/a；只见环评公示的检索摘要）按 0.6 规模指数放大到 20 Mt/a 是 0.71e6。
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
    # 20 Mtpa 付费：IND_BASE_t95 建成的 344 条边里有 222 条正好是 20；任何小到不值得建干线
    # 的流量，走 big-M "容量松弛"都比铺管便宜——于是有 11 条边在建成容量为零的情况下
    # 输送了 CO2。带规模经济的三档管径补上了这个缺口。各档每 km capex 由 20-Mtpa 干线费率
    # （400 000 x 20 = 8.0e6 CNY/km，见 pipe_capex_cny_per_mtpa_km）乘 (cap/20)^0.6 缩放，
    # 这是 CO2 管道常用的管径-成本指数（Knoope et al. 2013, IJGGC 16:241, Table 4 拟合为
    # 0.5-0.7）。交叉核对：2-Mtpa 档的 2.0e6 CNY/km 低于 1.7-Mtpa 齐鲁-胜利管线的
    # 3.1e6 CNY/km，而后者含压缩站，所以小档若有偏差也是偏便宜。类别乘数（支线 1.35、
    # 直连 2.8、走廊 0.97）照旧叠加在上面。
    pipe_capacity_tiers_mtpa: tuple[float, ...] = (2.0, 5.0, 20.0)
    pipe_capex_cny_per_km_by_tier: tuple[float, ...] = (2.0e6, 3.5e6, 8.0e6)
    # 封存部署爬坡。`injectivity_mtpa` 是 2060 年规模的可建速率（按 ACCA21 的 2060 年区间
    # 标定，见 storage_site_project_rate_mtpa）。2030 年就全部开放时，IND_BASE_t95 仅凭碳价
    # 就在 2040 年注入了 1 265 Mt/yr，而目前全国注入量为 ~4 Mt/yr。各规划年的可用比例取
    # ACCA21 (2021) CCUS 路线图的区间中点：2030 年 0.2-4.08 亿 t（中点 2.1），2050 年 6-14.5
    # （10.2），2060 年 10-18.2（14.1）；2040 年在 2035 年与 2050 年的区间之间插值（~7.5）。
    # 再除以全国可建速率 12.8 亿 t/yr，并以 1 封顶。
    storage_deployment_fraction_by_year: tuple[float, ...] = (0.17, 0.58, 0.80, 1.00)
    # 绿氨供给爬坡，同一机制。供给曲线只是技术潜力（2025 年 8 551 Mt NH3/yr，而全机组
    # 50% 掺烧只需 ~1 600 Mt），所以没有爬坡时节点上限永不绑定。默认全为 1，因为仓库里
    # 还没有有出处的中国绿氨建设轨迹；有了再设。
    ammonia_supply_deployment_fraction_by_year: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # 工业用氢由长管拖车从供给节点运送。临时值：取中国长管拖车成本的量级（每 100 km
    # ~3 CNY/kg，中国氢能联盟白皮书 2019 区间 2-4），待作者给出有出处的数值。作用同煤电侧
    # 的 `ammonia_transport_cost_cny_per_kg_km`。⚠ 假设（临时值；白皮书区间未核到原文）。
    h2_transport_cost_cny_per_kg_km: float = 0.03
    cooling_once_through_water_intensity_m3_per_mwh: float = 0.35  # 耗水口径（耗水量）：
    # 0.29-0.41 m³/MWh，依据 NDRC et al. 2015 No.9 清洁生产基准（中位数 ≈0.35）。
    # （原值 1.0 混淆了取水口径与耗水口径。）
    cooling_recirculating_water_intensity_m3_per_mwh: float = 1.85
    cooling_air_water_intensity_m3_per_mwh: float = 0.37
    ccs_water_multiplier: float = 1.82
    biomass_water_multiplier: float = 1.00
    beccs_water_multiplier: float = 1.82
    ammonia_water_multiplier: float = 1.01  # 掺氨使电厂取水、耗水都 +1%，与掺氨档位无关；⚠ 假设（无出处）
    # 供水成本（grid_supply 模式用）
    # 一个省的可再生水资源中已被农业、生活和其他工业占用的份额，在向电厂提供任何水量之前
    # 先扣除。0 保留原始的物理可用量；取水数据载入后按水资源公报设定。
    # 一个省的可再生水资源中已被农业、生活和其他工业占用的份额，在向电厂提供任何水量之前
    # 先扣除。
    #
    # 这种余量结构是 Richter et al. (2012) River Res. Applic. 28(8):1312-1321 本身的规定，
    # 而不是类比：天然月均流量的 20% "can be allocated for consumptive use"（可分配给
    # 消耗性用水），且按 "when added to already-existing water uses"（叠加在既有用水之上）
    # 评估（他的 Table II 标题为 "Cumulative allowable depletion"，即累计允许耗减量）。
    # 用 Smakhtin et al. (2004) Eq.(1) 的术语，这个参数就是水压力指标（water stress
    # indicator），`utilizable x (1 - WSI)` 就是他的余量。
    #
    # 0.85 有实测支持。黄河 1987 年"八七分水"方案是按耗水计的指标（各行标题为年耗水量，
    # 只含地表水），所以与本约束作用的耗水口径同口径可比。对照这 370x10^8 m3 的指标，
    # YRCC 2024 年公报（地表耗水 307.09）给出的非电份额实测值为 0.807-0.830（扣除电力后
    # 为 0.776-0.786）。0.85 略高于该区间，即略偏保守。
    #
    # 须在方法部分披露：近似之处在于锚定方式，而不在数值。上述实测份额是相对于中国自己
    # 允许的径流 52.6% 而言的，代码却把它乘在 Richter 的 20% 上——而 Richter 的严格程度是
    # 中国水法的 2.63 倍。结果是有意在两套各自自洽的口径之间取的中间值，这也是北方四个
    # 流域突破上限的原因：一个完全依法合规的北方流域仍达 Richter 限值的 ~2 倍。
    #
    # 在优化模型中约束电力部门流域用水、且完全不设生态流量份额的先例：Zhang, He,
    # Johnston & Zhong (2021) J. Clean. Prod. 329:129765 (SWITCH-China)。电力占中国取水的
    # ~8%，却只占其耗水的 ~1%（Zhang et al. 2017, JCLP 161:1171-1179）。
    existing_withdrawal_share: float = 0.0
    # 可用水量约束按哪一种水预算构建。
    #
    #   "runoff"          v9 及更早。available = qtot x 0.20 x (1 - existing_withdrawal_
    #                     share)。两个因子被别名化（见 `_water_available_by_node`），因此
    #                     无从区分生态流量标准与分配规则；而且对华北各流域，分母——当地
    #                     天然径流——比实际用水还小，实际用水靠跨流域调水和地下水支撑。
    #   "official_quota"  v9.1 起。两条规则变成两个口径不同的独立约束，各自约束其条文
    #                     实际所针对的量：
    #                        node  <= qtot x 0.20          生态流量，作用于耗水
    #                                                      （耗减规则）
    #                        basin <= 用水总量控制指标 - 非电既有取水
    #                                                      分配规则，作用于取水
    #                                                      （公报计量的正是它）
    #                     此时 `existing_withdrawal_share` 不再使用：分配规则直接从
    #                     国办发〔2013〕2号 读取，而不是靠假设。
    #                     流域上限来自 `inputs/water_basin_caps.csv`
    #                     （`scripts/build_water_basin_caps.py`）。
    water_budget: str = "runoff"
    # 流域上限开关，只在 water_budget='official_quota' 下生效。关闭时只剩生态流量的节点
    # 上限，这正是 v9.1 设计中的对照组：
    #
    #   BASE                 完全没有水约束
    #   *_oq_envonly         只有生态流量          （本开关为 False）
    #   *_oq                 生态流量 + 分配规则   （本开关为 True）
    #
    # 这一阶梯正是去别名化换来的。在 'runoff' 下两条规则是同一个乘积，任何实验都无法把
    # 它们分开；这里 BASE->envonly 给生态流量标准定价，envonly->oq 给分配规则定价，
    # 各自独立。
    apply_basin_cap: bool = True
    # 向求解器提供水量时施加流域偏差校正因子。这些因子（builders/water.py 中的
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
    #     与 CO2 惩罚为发电出力的 1-15%，随地点与气候而异。2.0 pp = 多耗 4.8% 燃料，落在
    #     该区间内；旧的 1.5 pp = 3.6%，处在其下沿。
    #
    # 已知简化，且会让空冷显得偏便宜。这里的惩罚是常数。Qin et al. 发现惩罚的恶化快于环境
    # 温度的上升，而在中国北方枯水季与高温季重合——所以本处低估惩罚的时段恰恰是水最紧缺的
    # 时段。在把惩罚做成随温度变化之前，`SA_air_penalty_high`（2.8 pp）是对此的粗略替代；
    # 做成随温度变化属于模型改动，而不是参数改动。
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
    # `INDUSTRY_CAPTURE_LIFETIME_YEARS`）；掺烧燃烧器升级 20 a；空冷 20 a（见上方
    # `air_retrofit_lifetime_years`）；管道 30 a（`pipeline_lifetime_years`）；重建的厂址
    # 30 a。这些寿命都是 ⚠ 假设（设定值，无文献）。工业捕集岛的 20 a 有出处（NPC 2019 的工业捕集改造，见
    # `constants_industry.INDUSTRY_CAPTURE_LIFETIME_YEARS`），煤电捕集岛只是沿用同值，那份出处不含煤电。
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
    # (577 D + 29 788) USD，即 24-40 inch 管道在 100 miles 长度上 capex 的 2-3%。
    # 0.97 = 去掉这部分 ROW 份额。原为 0.4（无出处）；据称一篇德国拓扑论文用了 10% 的
    # 走廊折扣，但未能打开（ScienceDirect S2772656826001004）——须经作者核实后才可用 0.9。
    corridor_capex_multiplier: float = 0.97
    top_k_storage_pairs: int = 5
    slack_penalty_cny_per_unit: float = 5_000_000_000.0
    sparse_interval_years: int = 10
    # 动态资源调配参数
    resource_match_radius_km: float = 200.0
    # 煤电机组掺烧生物质的全国上限，单位为每年 GJ（16 EJ；作者拍板，2026-09-10）。0.25 度
    # 节点层合计 ~30 EJ/yr 可收集的残余物，但那是与所有其他生物质用户共享的技术潜力；
    # 此上限是本研究允许全机组使用的份额。逐规划年施加，叠加在节点上限之上。
    # 0 或负值即关闭。
    biomass_national_cap_gj_per_year: float = 16.0e9
    # hub 级的重建与掺烧档位决策取连续份额（作者拍板，2026-09-10）。一个 hub 聚合 ~10 台
    # 机组，所以"hub 的一部分重建"和"hub 的一部分改到掺烧档位 l"是真实的自由度；做成
    # one-hot 二元变量时松弛太弱，部门上限 MIP 跑 10 h 后仍停在 4-15% gap（下界一直停在
    # 根节点 LP 上）。管道保持整数。False 恢复二元形式。
    hub_decisions_continuous: bool = True
    # 煤电机组掺烧绿氨的全国上限，每个规划年（2030/2040/2050/2060）的 Mt NH3。节点层是
    # 电解制氨潜力（2030 年 ~8 800 Mt NH3），永不绑定；2026-09-10 的文献综述（见
    # docs/工业联合减排实现说明.md §9.6）给出的可供电力使用的氨为：
    #   2050  47 Mt  -- Xiong et al. 2022, 储能科学与技术 11(12)，掺氨发电渗透率 30%
    #                  （DOI 10.19799/j.cnki.2095-4239.2022.0364），可直接引用；
    #   2030   2 Mt  -- 示范规模：2030 年全国绿氨产能 4.5 Mt
    #                  （中国化工节能技术协会，经中国能源报 2025-09-01 转引），化肥优先；
    #   2040  12 Mt  -- 在 Xiong 的 2035 年掺烧需求（5.4 Mt）与 2050 年之间内插；
    #   2060  55 Mt  -- Xiong 2060 年氨总量 120 Mt，可再生比例 >97%，其中约一半为能源用途
    #                  （RMI/CPCIF 2024）；取 50-60 Mt 区间的中点。
    # 2030/2040/2060 是推导值，不是引文原值——已为作者标出。空元组 = 关闭。
    ammonia_fleet_cap_mt_by_year: tuple[float, ...] = (2.0, 12.0, 47.0, 55.0)
    # 所有用户从共享电解节点取用的绿氢全国上限（机组用氨按 NH3_H2_RATIO 折成 H2
    # + 工业用氢），每个规划年的 Mt H2。2030 与 2060 取自中国氢能联盟
    # 《中国氢能技术发展路线图研究》(2024-12)：2030 年可再生氢 3.5-6.5 Mt，2060 年
    # 75-162 Mt（分别取上沿 / 中值）；2040 与 2050 取自水电水利规划设计总院（澎湃 2024，
    # "据预测"，二手）：69 / 91 Mt。
    # 空元组 = 关闭。
    green_h2_national_cap_mt_by_year: tuple[float, ...] = (6.5, 69.0, 91.0, 120.0)
    # 掺烧比例升级的资本成本（每 MW 电厂容量、每升一个掺烧档位的 CNY）。生物质每档 500 元/kW，
    # 满档（5 档）2 500 元/kW，落在 Wang et al. 2025（`wang2025reducing`，即上文的 Wang & Cai 2024）SI Table 3
    # 的掺烧改造 capex 327.2（140.7-513.7）$/kW = 2 290（985-3 596）元/kW 之内；第 1 档（10%）500 元/kW 高于
    # Fan et al. 2023 SI 式 (S56) 的 500 MW 锅炉 15% 掺烧 50 USD/kW = 350 元/kW（Fan 转引 IEA 2019，未核）。
    # 逐档线性这个形状：⚠ 假设（无出处）。
    biomass_upgrade_capex_cny_per_mw_per_level: float = 500_000.0
    ammonia_upgrade_capex_cny_per_mw_per_level: float = 25_000.0  # 最高档（50% 掺烧）≈125 CNY/kW。
    # 中国全改造文献：90 CNY/kW（CNERI 2025，100% 改造）+ 储存；MIT 供应系统 ≈163 CNY/kW
    # （Deng et al. 2024）；仅燃烧器 121.6 CNY/kW（Li & Li 2022，未确认）。中位数
    # ≈125 CNY/kW 对应第 5 档；每档线性 25 CNY/kW。
    # （原值 800,000 是中国文献集中区间的 ~30 倍。）
    # 生物质到厂成本构成（Wang et al. 2024, Nat Commun）
    # 收购价 base_cost_cny_per_gj 在供给曲线 CSV 里（`constants.BIOMASS_COST_BASE` = 20 元/GJ）。
    biomass_pretreatment_cost_cny_per_gj: float = 11.96     # γ₅: 6.15 $/MWh × 7.0 / 3.6 (Wang et al. 2024, Nat Commun)
    biomass_transport_fixed_cost_cny_per_gj: float = 13.13    # γ₆: 6.75 $/MWh × 7.0 / 3.6
    biomass_transport_variable_cost_cny_per_gj_km: float = 0.126  # γ₇: 0.065 $/(MWh·km) × 7.0 / 3.6
    # 氨运输成本（卡车，Hydrogen Council & McKinsey 2022；IEA GHR 2023）
    # 0.12 USD/(t·km) = 0.00012 USD/(kg·km) × 7.0 = 0.00084 CNY/(kg·km)
    ammonia_transport_cost_cny_per_kg_km: float = 0.00084
    # 运行时网格粗化（单位：度；0 = 不粗化，磁盘上的数据已粗化）
    biomass_coarse_grid_degrees: float = 0.0
    ammonia_coarse_grid_degrees: float = 0.0
    water_coarse_grid_degrees: float = 0.0

    # 分省年运行小时数（计算发电量时替代统一的 capacity_factor）
    # 来源：中国电力企业联合会（China Electricity Council）统计，分省煤电平均利用小时数
    province_operating_hours: dict[str, float] = field(default_factory=lambda: {
        "Inner Mongolia": 5031.5, "Zhejiang": 5535.3, "Xinjiang": 5281.2,
        "Jiangsu": 5015.5, "Fujian": 5126.2, "Anhui": 5082.1,
        "Hainan": 5030.9, "Ningxia": 4970.9, "Shanghai": 4954.8,
        "Sichuan": 4996.4, "Chongqing": 4794.2, "Guangdong": 4831.7,
        "Guangxi": 4786.6, "Jiangxi": 4766.4, "Guizhou": 4747.1,
        "Shanxi": 4629.5, "Shaanxi": 4538.0, "Yunnan": 4482.2,
        "Hebei": 4455.1, "Shandong": 4361.1, "Tianjin": 4347.5,
        "Gansu": 4189.6, "Qinghai": 4097.0, "Hunan": 3906.9,
        "Hubei": 3749.6, "Jilin": 3621.4, "Henan": 3599.6,
        "Heilongjiang": 3478.5, "Liaoning": 3100.7, "Beijing": 779.2,
    })

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
        ratios = self.ccs_energy_penalty_ratio_by_year
        if not ratios:
            return 0.0
        mapping = dict(zip(PLANNING_YEARS, ratios, strict=False))
        if year in mapping:
            return float(mapping[year])
        nearest = min(mapping, key=lambda candidate: abs(candidate - year))
        return float(mapping[nearest])

    def _fraction_for_year(self, values: tuple[float, ...], year: int) -> float:
        if not values:
            return 1.0
        mapping = dict(zip(PLANNING_YEARS, values, strict=False))
        if year in mapping:
            return float(mapping[year])
        nearest = min(mapping, key=lambda candidate: abs(candidate - year))
        return float(mapping[nearest])

    def storage_deployment_fraction(self, year: int) -> float:
        """2060 年规模的注入速率中，在 `year` 已建成可用的份额。"""
        return self._fraction_for_year(self.storage_deployment_fraction_by_year, year)

    def ammonia_supply_deployment_fraction(self, year: int) -> float:
        """绿氨技术潜力中，在 `year` 已建成的份额。"""
        return self._fraction_for_year(self.ammonia_supply_deployment_fraction_by_year, year)

    def ammonia_fleet_cap_mt(self, year: int) -> float:
        """煤电掺氨在 `year` 的全国绿氨上限，Mt NH3（0 = 关闭）。"""
        if not self.ammonia_fleet_cap_mt_by_year:
            return 0.0
        return self._fraction_for_year(self.ammonia_fleet_cap_mt_by_year, year)

    def green_h2_national_cap_mt(self, year: int) -> float:
        """所有节点用户在 `year` 的全国绿氢上限，Mt H2（0 = 关闭）。"""
        if not self.green_h2_national_cap_mt_by_year:
            return 0.0
        return self._fraction_for_year(self.green_h2_national_cap_mt_by_year, year)

    def cooling_baseline_water_intensity(self, cooling_label: str) -> float:
        label = str(cooling_label).strip().lower()
        if "once" in label:
            return self.cooling_once_through_water_intensity_m3_per_mwh
        if "air" in label or "dry" in label:
            return self.cooling_air_water_intensity_m3_per_mwh
        return self.cooling_recirculating_water_intensity_m3_per_mwh


@dataclass(frozen=True)
class OptimizationScenario:
    experiment_id: str
    description: str
    planning_years: tuple[int, ...] = field(default_factory=lambda: tuple(PLANNING_YEARS))
    # 燃烧后捕集率，全期不变：Fan et al. 2023 SI p.43-44（取 90% 并设研究期内不变）；An et al. 2025 SI p.20；
    # Wang et al. 2025 SI Table 5。三处都是煤电；工业 CCS 也用这个值（`industry_matrices`），未另找工业侧出处。
    capture_rate: float = 0.90
    biomass_blend_levels: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
    ammonia_blend_levels: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
    # 部门碳目标来源：读 `inputs/sector_targets_<source>.csv`（scripts/build_sector_targets.py）。
    # 每组每个规划年一条上限：residual_g(y) <= cap_fraction_g(y) x baseline_g(2030) + shortfall_g(y)，
    # g 属于 {power, steel, cement, chemicals}；baseline_g(2030) 是本模型自身的 2030 冻结技术排放，
    # 即 TIMES 轨迹给形状、模型给水平。煤电整体为 power 组。
    sector_target_source: str = "times_cn60"
    # 各规划年的煤电机组利用小时，全国按容量加权。为空时四个年份都冻结在分省统计值上
    # （2026-09-10 之前的行为，此时机组 2060 年发电 6 576 TWh，与 2030 年相同）。设定后，
    # 每个 hub 所在省的小时数乘以 hours_y / 全机组当前平均小时数（4 643 h），从而保留省间
    # 差异，只移动水平。作者的指示（2026-09-10）是一条粗略的利用小时轨迹，而不是调度模型；
    # TIMES CN60 给出 2030 年 3 594 h、2040 年 3 092 h（TIMES 里 2040 年后煤电是剩余项），
    # 之后各点是按机组留作灵活性电源取的整数。
    coal_operating_hours_by_year: tuple[float, ...] = ()
    # 外生的分行业、分年份工业产量指数，来自 `inputs/industry_output_index_<source>.csv`；
    # 为空表示每年产量都保持在 2025 年水平（2026-09-10 之前的行为）。设了
    # `sector_target_source` 而本项未设时默认取前者，这样以 TIMES 为锚的上限总是配上推导它
    # 时所用的 TIMES 产量路径。
    industry_output_index_source: str = ""
    storage_scope: str = "dsa_eor"
    injectivity_multiplier: float = 1.0
    biomass_supply_multiplier: float = 1.0
    biomass_cost_multiplier: float = 1.0
    # 煤电捕集成本乘子：只乘捕集岛 capex 与随之的固定运维（`ccs_om_fraction`），不乘能耗惩罚与
    # BECCS 的掺烧运维（`plant_matrices`）。2026-09-23 前乘的是 capex 与每 MWh 附加项，固定运维不乘。
    ccs_cost_multiplier: float = 1.0
    # 工业捕集成本乘子：只乘捕集 capex（固定运维随之），不乘每吨捕集的能耗、耗材成本
    # （`industry_matrices.industry_year_data`），与煤电同口径；2026-09-23 前也乘能耗与耗材。
    # 2026-09-22 起不再乘 ACCA21 平准化成本。
    industry_cost_multiplier: float = 1.0
    # 工业氢路线成本乘子：只乘路线 capex 与随之的固定运维，与上面两个 CCS 乘子同口径；不乘买氢，
    # 也不乘由锚点反推的非氢运行差额（固定在乘子为 1 时的值）。因此作用有限：锚点氢价下乘子取 2
    # 只让平准化溢价增加钢铁 25%、合成氨 14%、甲醇 2%。2026-09-23 前还乘该差额里路线自身的成本，
    # 锚点价下溢价恰为乘子 x 锚点溢价（乘子取 2 即 +100%）。与上面的乘子分开，因为氢锚点带有已知的
    # 向下偏差——它是新建对新建的比较，却套用在现有资本已沉没的存量工厂上——所以其采用量是上界，
    # 需要单独的调节参数。见 `optimization/industry.py` 模块说明中氢路线一段。
    industry_h2_cost_multiplier: float = 1.0
    ammonia_cost_multiplier: float = 1.0
    ammonia_transport_adder_usd_per_kg: float = 0.0
    water_mode: str = "no_water"
    # 驱动可用水量的气候成员，例如 "cwatm|gfdl-esm4|ssp370"。
    # 为空时选 water_mode 所隐含的那一族中的第一个成员。
    water_scenario_id: str = ""
    # 不再有口径开关：可用水量约束总是作用于耗水，水价总是按中国取水定额计，取水只报告、
    # 从不约束。为什么拿取水去对照生态流量允许量是范畴错误，见 `data_prep._prepare_plants`。
    # "annual" 用十年均值；"dry" 用最低的连续三个月，火电厂恰恰在这段时间真正被限发。
    water_season: str = "annual"
    water_multiplier: float = 1.0
    # 对输送到电厂的每 m3 水附加的参数化收费，叠加在 water_supply_links.csv 中已有的取水与
    # 输水成本之上。扫描它就能描出水-碳前沿：由于模型是 MIP，Gurobi 无法为可用水量约束
    # 返回可靠的对偶值，所以改用参数化定价来还原水的影子价格——每一点上的附加费就是该点的
    # 影子价格。
    water_price_adder_cny_per_m3: float = 0.0
    forced_cooling_technology: str = ""
    ccs_water_multiplier_adjustment: float = 1.0
    beccs_water_multiplier_adjustment: float = 1.0
    dense_time_grid: bool = False
    carry_state_between_years: bool = True
    pathway_disable: tuple[str, ...] = ()
    forced_pathways: tuple[str, ...] = ()
    min_forced_path_share: float = 0.0
    corridor_prior_strength: float = 0.70
    solve_mode: str = "joint"           # 只实现了 "joint"；其他取值会报错
    mip_gap: float = 0.01               # MIP 最优性间隙（默认 1%；探索性求解可放宽）
    solver_threads: int = 0       # 0 = 由 Gurobi 自动检测
    solver_time_limit: int = 36000  # 秒（默认 10h）
    discount_rate: float = DEFAULT_DISCOUNT_RATE
    discount_base_year: int = 2025
    # 系统净成本参数
    # 碳价，元/t CO2，2030/2040/2050/2060 年：⚠ 假设（情景设定，无出处），每 10 年加 380 元的直线；模型没有价格指数，
    # 按不变价用。登记表里只有 `ST_CP_BASE` 用它，另两个 `ST_` 情景置零（`scripts/run_single.py`）。对照（都只见
    # 检索摘要，未核原文）：ICF 2022 中国碳价调查对 2030 年的预期 130 元/t；C-GEM（张希良等 2022，管理世界 38(1)）
    # 2030 年 100 以上、2060 年 2 700 以上；Zhang & Chen 2022（`zhang2022probabilistic`）2060 年中位数 168-1 096 USD/t。
    carbon_price_cny_per_t_by_year: tuple[float, ...] = (120.0, 500.0, 880.0, 1260.0)
    # 电价路径。2030 年的 400 元/MWh 对照 Wang et al. 2025（`wang2025reducing`）SI Table 3：
    # 0.06（0.048-0.072）$/kWh = 420（336-504）元/MWh。逐年上涨的路径：⚠ 假设（无出处）。
    electricity_price_cny_per_mwh_by_year: tuple[float, ...] = (400.0, 440.0, 490.0, 550.0)
    # 每个规划期新增的自愿退役（未到设计寿命的机组）不超过当年总发电量的 15%，约合每年 1.5%：⚠ 假设（无出处）。
    # 对照：REMIND 的提前退役上限，中国落在缺省组 2%/年，常规煤电再乘 1.2，即 2.4%/年（`remind`，按装机计）；
    # An et al. 2025 不设速率上限，其 Base 情景 2030-2040 年仅比 Flex 情景多出的提前退役就有 302.8-397.1 GW。
    # v9 的 `*_noair` 结果正好卡在这个上限上（`scripts/plot_fig5_pathway_succession.py`）。
    max_new_retirement_share_per_period: float = 0.15
    # 改造后 CF 提升：改造过的电厂（CCS/生物质/BECCS/氨）发电量相对基线乘以此系数，是必然多发，不是上限。
    # ⚠ 假设（用法无出处）：数值与 Fan et al. 2023 SI Table 11 里 CCS 类机组与未改造煤电的最大容量因子之比
    # 0.69/0.60 = 1.15 相同，但那是上限之比，且只有 CCS 类；生物质、掺氨两列没有出处。
    retrofit_cf_boost: float = 1.15
    # 原址重建参数：到期电厂可按新建成本的 70% 重建
    rebuild_capex_fraction: float = 0.70  # stranded_asset_base_cny_per_kw 的 70%；⚠ 假设（无出处）
    # 重建电厂取超超临界效率（基线为 0.42）：Wang et al. 2025 SI Table 1 引 NDRC 2022 标准，"Ultra-supercritical/ccs"
    # 一行为 270 gce/kWh，按低位热值折 0.455，取值低 1.1%。该行名原文如此、含义有歧义；NDRC 原文未核。
    rebuild_efficiency: float = 0.45
    notes: str = ""

    def _interpolate_year_tuple(self, values: tuple[float, ...], year: int) -> float:
        mapping = dict(zip(self.planning_years, values, strict=False))
        if year in mapping:
            return float(mapping[year])
        if not mapping:
            return float(values[0]) if values else 0.0
        nearest = min(mapping, key=lambda y: abs(y - year))
        return float(mapping[nearest])

    def carbon_price_for_year(self, year: int) -> float:
        return self._interpolate_year_tuple(self.carbon_price_cny_per_t_by_year, year)

    @property
    def effective_output_index_source(self) -> str:
        chosen = str(self.industry_output_index_source).strip()
        return chosen or str(self.sector_target_source).strip()

    def operating_hours_scale(self, year: int, fleet_hours_now: float) -> float:
        """`year` 年施加在每个 hub 当前所在省小时数上的乘数（未设定时为 1.0）。"""
        if not self.coal_operating_hours_by_year:
            return 1.0
        if fleet_hours_now <= 0:
            raise ValueError("fleet_hours_now must be positive to scale operating hours")
        target = self._interpolate_year_tuple(self.coal_operating_hours_by_year, year)
        return float(target) / float(fleet_hours_now)

    def electricity_price_for_year(self, year: int) -> float:
        return self._interpolate_year_tuple(self.electricity_price_cny_per_mwh_by_year, year)

    def effective_years(self, available_years: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if not self.dense_time_grid:
            return tuple(self.planning_years)
        filtered = [year for year in sorted(set(available_years)) if year <= max(self.planning_years)]
        return tuple(filtered or self.planning_years)

    def interval_years(self, years: tuple[int, ...], index: int, assumptions: OptimizationAssumptions) -> int:
        if len(years) <= 1:
            return assumptions.sparse_interval_years
        if index < len(years) - 1:
            return max(1, years[index + 1] - years[index])
        return max(1, years[index] - years[index - 1])

    def path_enabled(self, pathway: str) -> bool:
        return pathway not in set(self.pathway_disable)
