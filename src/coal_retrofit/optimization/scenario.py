"""情景定义 `OptimizationScenario` 与煤电路径集合 `PATHWAYS`；技术经济假设 `OptimizationAssumptions` 在 `assumptions`。"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import DEFAULT_DISCOUNT_RATE, HOURS_PER_YEAR, PLANNING_YEARS, part_load_heat_rate_factor
from .assumptions import OptimizationAssumptions

PATHWAYS = ("unabated", "retire", "ccs", "biomass", "beccs", "ammonia")


@dataclass(frozen=True)
class OptimizationScenario:
    experiment_id: str
    description: str
    planning_years: tuple[int, ...] = field(default_factory=lambda: tuple(PLANNING_YEARS))
    # 燃烧后捕集率，全期不变：Fan et al. 2023 SI p.43-44（取 90% 并设研究期内不变）；An et al. 2025 SI p.20；
    # Wang et al. 2025 SI Table 5。三处都是煤电；工业 CCS 也用这个值（`industry_matrices`），未另找工业侧出处。
    capture_rate: float = 0.90
    # 掺生物质档位（热量比例，升序）：煤粉炉直燃 10%–15%（Fan et al. 2023 基准 15%），0.20 档只对 CFB 开放
    # （`OptimizationAssumptions.biomass_blend_max_pulverized`）。2026-10-02 前为 0.10 / 0.25 / 0.50 / 0.75 / 1.00，
    # 0.25 以上没有煤粉炉工程证据，0.75、1.00 实为改烧。
    biomass_blend_levels: tuple[float, ...] = (0.10, 0.15, 0.20)
    ammonia_blend_levels: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
    # 部门碳目标来源：读 `inputs/sector_targets_<source>.csv`（scripts/build_sector_targets.py）。
    # 每组每个规划年一条上限：residual_g(y) <= cap_fraction_g(y) x baseline_g(2030) + shortfall_g(y)，
    # g 属于 {power, steel, cement, chemicals}；baseline_g(2030) 是本模型自身的 2030 冻结技术排放，
    # 即 TIMES 轨迹给形状、模型给水平。煤电整体为 power 组。
    sector_target_source: str = "times_cn60"
    # 碳目标的两个入口（作者决定 2026-10-10）："sector" 四组各一条上限（上式）；"total" 电力加工业合计一条总量上限，
    # 额度 = 四组额度之和，sum_g residual_g(y) <= sum_g cap_fraction_g(y) x baseline_g(2030) + shortfall(y)，部门间可以调剂
    # （`model_year._add_sector_targets`）。
    sector_target_mode: str = "sector"
    # 各规划年的煤电机组利用小时，全国按容量加权。为空时四个年份都冻结在分省统计值上
    # （2026-09-10 之前的行为，此时机组 2060 年发电 6 576 TWh，与 2030 年相同）。设定后，
    # 每个 hub 所在省的小时数乘以 hours_y / 全机组当前平均小时数（4 643 h），从而保留省间
    # 差异，只移动水平。作者的指示（2026-09-10）是一条粗略的利用小时轨迹，而不是调度模型；
    # TIMES CN60 给出 2030 年 3 594 h、2040 年 3 092 h（TIMES 里 2040 年后煤电是剩余项），
    # 之后各点是按机组留作灵活性电源取的整数。
    coal_operating_hours_by_year: tuple[float, ...] = ()
    # 部分负荷修正（2026-10-02 起，缺省 0 即关，只作敏感性）：设为机组全年在线小时数时，各规划年全国一个系数
    # κ = `constants.part_load_heat_rate_factor`(利用小时 ÷ 在线小时)，乘在煤电各部分的毛热耗与基线排放上，燃料与掺烧用量、排放、各项能耗
    # 惩罚、空冷背压与 2030 年电力基线随之放大（`plant_matrices`、`model_index`）；耗水强度不乘。利用小时取上一项，没设时取全机组
    # 当前平均小时数。ST 的小时数下降是"整体降负荷"才该加，"整台停运、其余满发"则不该加，主线不开（docs/archive/参数调研_20261001.md
    # §2.3）。在线 7 500 h 时 ST 四年 κ = 1.110 / 1.150 / 1.353 / 1.609，6 500 h 作低档（1.000 / 1.111 / 1.268 / 1.466）。
    coal_part_load_online_hours: float = 0.0
    # 外生的分行业、分年份工业产量指数，来自 `inputs/industry_output_index_<source>.csv`；
    # 为空表示每年产量都保持在 2025 年水平（2026-09-10 之前的行为）。设了
    # `sector_target_source` 而本项未设时默认取前者，这样以 TIMES 为锚的上限总是配上推导它
    # 时所用的 TIMES 产量路径。
    industry_output_index_source: str = ""
    storage_scope: str = "dsa_eor"
    # 封存汇与候选管网的输入变体（2026-10-16 起，`coal_retrofit.sink_inputs`）：空串读 `storage_hubs.csv` 与在它上面建的管网
    # （连通域 + 海上盆地合并，89 个汇）；"300km_complete" / "300km_average" 读同名加后缀的三张表（300 km 距离聚类，
    # 簇内任意两块 / 簇间平均距离不超过 300 km）。命令行 `run --sinks`。
    sink_variant: str = ""
    injectivity_multiplier: float = 1.0
    biomass_supply_multiplier: float = 1.0
    biomass_cost_multiplier: float = 1.0
    # 煤电捕集成本乘子：只乘捕集岛 capex 与随之的固定运维（`ccs_om_fraction`），不乘能耗惩罚与
    # BECCS 的掺烧运维（`plant_matrices`；2026-10-02 起按掺烧能力计，此前按每 MWh）。2026-09-23 前乘的是 capex 与
    # 每 MWh 附加项，固定运维不乘。
    ccs_cost_multiplier: float = 1.0
    # 工业捕集成本乘子：只乘捕集 capex（固定运维随之），不乘每吨捕集的能耗、耗材成本
    # （`industry_matrices.industry_year_data`），与煤电同口径；2026-09-23 前也乘能耗与耗材。
    # 2026-09-22 起不再乘 ACCA21 平准化成本。
    industry_cost_multiplier: float = 1.0
    # 工业氢路线成本乘子：只乘路线 capex 与随之的固定运维，与上面两个 CCS 乘子同口径；不乘买氢，
    # 也不乘由锚点反推的非氢运行差额（固定在乘子为 1 时的值）。因此作用有限：锚点氢价下乘子取 2
    # 只让平准化溢价增加 ν x 路线年资本项（ν 为铭牌系数，ν = 1 的 hub 为钢铁 25%、合成氨 14%、甲醇 2%）。
    # 2026-09-23 前还乘该差额里路线自身的成本，锚点价下溢价恰为乘子 x 锚点溢价（乘子取 2 即 +100%）。与上面的
    # 乘子分开，因为氢锚点带有已知的向下偏差——它是新建对新建的比较，却套用在现有资本已沉没的存量工厂上——
    # 所以其采用量是上界，需要单独的调节参数。见 `optimization/industry.py` 模块说明中氢路线一段。
    industry_h2_cost_multiplier: float = 1.0
    ammonia_cost_multiplier: float = 1.0
    ammonia_transport_adder_usd_per_kg: float = 0.0
    water_mode: str = "no_water"
    # 驱动可用水量的气候成员，例如 "cwatm|gfdl-esm4|ssp370"。
    # 为空时选 water_mode 所隐含的那一族中的第一个成员。
    water_scenario_id: str = ""
    # 没有口径开关：节点可用水量（生态流量）约束作用于耗水，水价按中国取水定额计，取水只进
    # 流域用水总量指标约束（`OptimizationAssumptions.apply_basin_cap`）。为什么拿取水去对照生态流量
    # 允许量是范畴错误，见 `data_prep_plants._prepare_plants`。
    # "annual" 用十年均值；"dry" 用最低的连续三个月，火电厂恰恰在这段时间真正被限发。
    water_season: str = "annual"
    water_multiplier: float = 1.0
    # 对输送到电厂的每 m3 水附加的参数化收费，叠加在 `data_prep._prepare_water` 运行时按距离
    # 算出的取水与输水成本之上。扫描它就能描出水-碳前沿：由于模型是 MIP，Gurobi 无法为可用水量约束
    # 返回可靠的对偶值，所以改用参数化定价来还原水的影子价格——每一点上的附加费就是该点的
    # 影子价格。
    water_price_adder_cny_per_m3: float = 0.0
    pathway_disable: tuple[str, ...] = ()
    mip_gap: float = 0.01               # MIP 最优性间隙（默认 1%；探索性求解可放宽）
    solver_threads: int = 0       # 0 = 由 Gurobi 自动检测
    solver_time_limit: int = 36000  # 秒（默认 10h）
    # Gurobi 的 Seed 与 MIPFocus，与线程数同属（模型, 参数, 线程数）：对照的两次求解必须相同，seed 族只换 solver_seed。
    # 0 是 Gurobi 的缺省种子；MIPFocus 1 侧重尽快找到好的可行解。2026-09-28 起是情景字段，此前只能用环境变量
    # COAL_RETROFIT_GUROBI_SEED / COAL_RETROFIT_MIPFOCUS 设（2026-10-13 起不再读）。
    solver_seed: int = 0
    mip_focus: int = 1
    # 热启动（实现说明 §9.7）："lp_relax" 时运行器先把全部整数变量松弛后求解、写 .sol（时限 warm_start_time_limit 秒），
    # 再重新建模，按 .sol 给整数变量设 MIP start 求 MIP。只改搜索路径，不改模型；同一族必须同法。
    warm_start: str = "none"
    warm_start_time_limit: int = 1800
    discount_rate: float = DEFAULT_DISCOUNT_RATE
    discount_base_year: int = 2025
    # 系统净成本参数
    # 碳价，元/t CO2，2030/2040/2050/2060 年：⚠ 假设（情景设定，无出处），每 10 年加 380 元的直线；模型没有价格指数，
    # 按不变价用。登记表里只有 `ST_CP_BASE` 用它，另三个 `ST_` 情景置零（`scenarios/st.toml`）。对照：Wu 等 2026 SI
    # Table 7 的情景 2055 年低 / 中 / 高 300 / 600 / 1 000 元/t（已核）；只见检索摘要的：ICF 2022 中国碳价调查对 2030 年的预期
    # 130 元/t，C-GEM（张希良等 2022，管理世界 38(1)）2030 年 100 以上、2060 年 2 700 以上；另有一条出处未明的：2060 年中位数
    # 168-1 096 USD/t（BEF60 / AFT60 两组情景；此前记为 Zhang & Chen 2022，其 SI 文字与表、Zenodo 数据包里都没有碳价，正文未取得）。
    carbon_price_cny_per_t_by_year: tuple[float, ...] = (120.0, 500.0, 880.0, 1260.0)
    # 电价路径。2030 年的 400 元/MWh 对照 Wang et al. 2025（`wang2025reducing`）SI Table 3：
    # 0.06（0.048-0.072）$/kWh = 420（336-504）元/MWh。逐年上涨的路径：⚠ 假设（无出处）。
    electricity_price_cny_per_mwh_by_year: tuple[float, ...] = (400.0, 440.0, 490.0, 550.0)
    # 煤电容量电价，元/(kW·a)（2026-10-02 起，缺省 0 即单一制电价，只作敏感性）：按未退役装机收，煤电的电量电价同时减去
    # 容量电价 ÷ 全机组现状利用小时，按现状小时算的总收入不变，不与上面的单一制电价重复计收（`plant_matrices`）；工业买电仍按
    # 上面的电价。2024 年起煤电两部制，每年 100 或 165 元/kW（B，政策摘要；docs/archive/参数调研_20261001.md §2.3）。
    coal_capacity_price_cny_per_kw_yr: float = 0.0
    # 每个规划期新增的自愿退役（未到设计寿命的机组）不超过当年总发电量的 15%，约合每年 1.5%：⚠ 假设（无出处）。
    # 对照：REMIND 的提前退役上限，中国落在缺省组 2%/年，常规煤电再乘 1.2，即 2.4%/年（`remind`，按装机计）；
    # An et al. 2025 不设速率上限，其 Base 情景 2030-2040 年仅比 Flex 情景多出的提前退役就有 302.8-397.1 GW。
    # v9 的 `*_noair` 结果正好卡在这个上限上（`a303f05` 里的 `_v9tree/scripts/plot_fig5_pathway_succession.py`）。
    max_new_retirement_share_per_period: float = 0.15
    # 改造后 CF 提升：改造过的电厂（CCS/生物质/BECCS/氨）发电量相对基线乘以此系数，是必然多发，不是上限。
    # ⚠ 假设（用法无出处）：数值与 Fan et al. 2023 SI Table 11 里 CCS 类机组与未改造煤电的最大容量因子之比
    # 0.69/0.60 = 1.15 相同，但那是上限之比，且只有 CCS 类；生物质、掺氨两列没有出处。
    retrofit_cf_boost: float = 1.15
    # 原址重建参数：到期电厂可按新建成本的 70% 重建
    rebuild_capex_fraction: float = 0.70  # stranded_asset_base_cny_per_kw 的 70%；⚠ 假设（无出处）
    # 重建机组取超超临界效率：逐台毛热耗取 min(机组毛热耗, 3.6 / 0.45 = 8.0 GJ/MWh)，不劣于原值；空冷机组原址重建仍为
    # 空冷，上限加空冷的 +15 g/kWh（折毛 0.418，即 8.418；`data_prep_plants._with_expiry`）。
    # Wang et al. 2025 SI Table 1 引 NDRC 2022 标准，"Ultra-supercritical/ccs" 一行为 270 gce/kWh（供电），按低位热值折
    # 0.455，取值低 1.1%；该行名原文如此、含义有歧义，NDRC 原文未核。这里把 0.45 直接当毛效率用，比按 5% 厂用电率折毛
    # （270 -> 7.52 GJ/MWh）保守。
    rebuild_efficiency: float = 0.45

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

    def part_load_factor(self, year: int, fleet_hours_now: float) -> float:
        """`year` 年的部分负荷修正系数 κ（`coal_part_load_online_hours` 为 0 时为 1.0）。"""
        online = float(self.coal_part_load_online_hours)
        if online == 0.0:
            return 1.0
        if not 0.0 < online <= HOURS_PER_YEAR:
            raise ValueError(f"coal_part_load_online_hours must be 0 (off) or in (0, {HOURS_PER_YEAR:g}], got {online}")
        hours = (
            self._interpolate_year_tuple(self.coal_operating_hours_by_year, year)
            if self.coal_operating_hours_by_year else float(fleet_hours_now)
        )
        if not 0.0 < hours <= online:
            raise ValueError(f"utilisation hours {hours} in {year} must lie in (0, coal_part_load_online_hours = {online}]")
        return part_load_heat_rate_factor(hours / online)

    def electricity_price_for_year(self, year: int) -> float:
        return self._interpolate_year_tuple(self.electricity_price_cny_per_mwh_by_year, year)

    def interval_years(self, years: tuple[int, ...], index: int, assumptions: OptimizationAssumptions) -> int:
        if len(years) <= 1:
            return assumptions.sparse_interval_years
        if index < len(years) - 1:
            return max(1, years[index + 1] - years[index])
        return max(1, years[index] - years[index - 1])

    def path_enabled(self, pathway: str) -> bool:
        return pathway not in set(self.pathway_disable)
