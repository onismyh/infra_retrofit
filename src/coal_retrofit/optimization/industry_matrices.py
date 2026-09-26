"""工业点源的逐年系数：每个 hub、每条路线的减排、捕集、取水、年度成本与一次性 capex。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants_industry import (
    INDUSTRY_CAPTURE_LIFETIME_YEARS,
    INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2,
    INDUSTRY_CCS_FIXED_OM_FRACTION,
    INDUSTRY_H2_ABATEMENT_FRACTION,
    INDUSTRY_H2_LIFETIME_YEARS,
    INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION,
    INDUSTRY_H2_USES_ADVANCED_QUOTA,
    INDUSTRY_ROUTES,
    SECTOR_HAS_H2_ROUTE,
    capture_capex_cny_per_t_yr,
    capture_steam_co2_t_per_t,
    capture_variable_cost_cny_per_t,
    h2_route_capex_cny_per_t_yr,
    h2_route_opex_delta_cny_per_t,
    water_quota,
)
from .industry_inputs import IndustryInputs

ROUTE_INDEX = {name: index for index, name in enumerate(INDUSTRY_ROUTES)}
UNABATED, CCS, H2 = ROUTE_INDEX["unabated"], ROUTE_INDEX["ccs"], ROUTE_INDEX["h2"]

# 读取各行业先进值 / 通用值定额比时用的原料键。hub 表带的是成员原料按产能加权混合后的
# 结果，不带原料构成本身，所以比值取该行业的 default 行，再作用到 hub 自己的混合强度上。
# 各原料之间的差别很小（三个氢路线行业为 0.61-0.73），不可能改变结果；这里写明而不隐去。
_DEFAULT_FEEDSTOCK = "default"


@dataclass(frozen=True)
class IndustryYearData:
    """`industry_year_data` 的输出：一个规划年里每个工业 hub、每条路线的系数。

    路线矩阵形状 (hub_count, len(INDUSTRY_ROUTES))、列序同 INDUSTRY_ROUTES；其余形状在旁注明。
    成本分两部分：`opex_cny` 是年度部分，每个运行年按路线份额计；一次性改造 capex 是
    `capex_cny_per_mt` x 新增能力，计在能力存量的增量上（`model_industry.industry_capex_expr`），
    能力存量 >= `capacity_mt_per_share` x 份额。氢路线买氢不在 `opex_cny` 里，由求解器按链路采购。
    """

    hub_ids: list[str]
    sectors: np.ndarray                  # (hub_count,)
    target_groups: np.ndarray            # (hub_count,)，部门目标组
    output_scale: np.ndarray             # (hub_count,)，产量指数，2030 = 1
    baseline_emissions_mt: np.ndarray    # (hub_count,)，已乘产量指数
    route_available: np.ndarray          # bool
    reduction_mt: np.ndarray             # CCS 列已扣除放空的再生蒸汽 CO2
    captured_mt: np.ndarray
    opex_cny: np.ndarray                 # 年度：固定运维 + 能耗 + 耗材 + 氢路线非氢运行差额
    capacity_mt_per_share: np.ndarray    # 份额为 1 时所需能力，Mt/yr：CCS 为捕集量，氢路线为产量
    capex_cny_per_mt: np.ndarray         # 一次性：每 Mt/yr 新增能力的改造 capex，本年价
    h2_demand_kg_per_share: np.ndarray   # (hub_count,)，氢路线份额为 1 时的年需氢量，kg
    water_m3: np.ndarray
    h2_price_cny_per_kg: float           # 全国供给加权均价，只作报告
    capture_learning_factor: float
    capex_lifetime_years: dict[int, int]  # 路线下标 -> 经济寿命（年），供期末残值


def _h2_price_for_year(prices: dict[int, float], year: int) -> float:
    """`year` 年的氢价；表中没有该年时退回到最近的已列年份。

    与 `scenario.carbon_price_for_year` 处理网格外年份的方式一致。
    """
    if year in prices:
        return float(prices[year])
    if not prices:
        raise ValueError("no hydrogen prices available")
    nearest = min(prices, key=lambda candidate: abs(candidate - year))
    return float(prices[nearest])


def _advanced_quota_ratio(sector: str) -> float:
    """某行业取水定额的先进值 / 通用值之比（两者都取自 GB/T 18916）。"""
    general = water_quota(sector, _DEFAULT_FEEDSTOCK, advanced=False)
    advanced = water_quota(sector, _DEFAULT_FEEDSTOCK, advanced=True)
    if general <= 0:
        raise ValueError(f"sector {sector!r} has a non-positive general water quota")
    return float(advanced) / float(general)


def _output_scale(industry: IndustryInputs, sectors: np.ndarray, year: int) -> np.ndarray:
    if not industry.output_index:
        return np.ones(len(sectors), dtype=np.float64)
    available_years = sorted({y for _, y in industry.output_index})
    nearest = year if year in available_years else min(available_years, key=lambda y: abs(y - year))
    return np.array(
        [float(industry.output_index[(str(sector), nearest)]) for sector in sectors], dtype=np.float64
    )


def industry_year_data(
    industry: IndustryInputs, scenario, assumptions, year: int
) -> IndustryYearData:
    """每个工业 hub、每条路线的逐年成本、排放与用水系数。

    除另有注明外，数组形状均为 `(hub_count, len(INDUSTRY_ROUTES))`。成本分成年度部分
    （`opex_cny` = 固定运维 + 能耗 + 耗材 + 非氢运行差额，每个运行年按路线份额计）与一次性
    部分（`capex_cny_per_mt` x 能力存量的增量；份额为 1 时所需能力是 `capacity_mt_per_share`，
    随产量指数变化）。氢路线的买氢不在 `opex_cny` 里：它在求解器里按链路购买。
    `reduction_mt[:, CCS]` 已扣除放空的再生蒸汽 CO2。

    Args:
        industry: 准备好的工业输入。
        scenario: `OptimizationScenario`；从中读取捕集率、两个工业成本乘数、贴现率与本年电价。
        assumptions: `OptimizationAssumptions`；从中读取 CCS 学习曲线、分省煤价、燃煤排放因子与热耗率。
        year: 规划年。

    Returns:
        系数数组，外加求解器需要的标量。
    """
    hubs = industry.hubs
    sectors = hubs["sector"].astype(str).to_numpy()
    n = len(hubs)
    output_scale = _output_scale(industry, sectors, int(year))
    co2_mt = hubs["co2_mt_per_year"].astype(float).to_numpy() * output_scale
    production_t = hubs["production_kt_per_year"].astype(float).to_numpy() * 1_000.0 * output_scale
    base_water_m3 = hubs["water_m3_per_year"].astype(float).to_numpy() * output_scale
    production_now = hubs["production_kt_per_year"].astype(float).to_numpy()
    h2_intensity_t_per_t = np.divide(
        hubs["h2_demand_kt_per_year"].astype(float).to_numpy(),
        np.where(production_now > 0, production_now, np.nan),
    )
    h2_intensity_t_per_t = np.nan_to_num(h2_intensity_t_per_t, nan=0.0)

    capture_rate = float(scenario.capture_rate)
    # 与煤电改造用同一条外生学习曲线，两个部门的捕集成本一起下降。给工业另用一条曲线，
    # 部门间的分工就会取决于一个任意的建模选择，而不是技术本身。
    learning = float(assumptions.ccs_learning_factor(year))
    cost_multiplier = float(scenario.industry_cost_multiplier)
    h2_multiplier = float(scenario.industry_h2_cost_multiplier)
    h2_price_mean = _h2_price_for_year(industry.h2_price_cny_per_kg, int(year))
    rate = float(scenario.discount_rate)
    # 捕集能耗按模型自己的价格计价：再生蒸汽用 hub 所在省的煤价（与煤电 hub 用同一个查表），
    # 压缩与辅机用情景电价。这样煤价或电价的敏感性分析会让两个部门的捕集成本一起变动，
    # 而不是让工业停在一个冻结的文献价格上。
    provinces = hubs["province"].astype(str).to_numpy() if "province" in hubs.columns else np.array([""] * n)
    coal_price_gj = np.array([float(assumptions.province_coal_cost(p)) for p in provinces], dtype=np.float64)
    elec_price_mwh = float(scenario.electricity_price_for_year(int(year)))
    emission_factor_t_per_gj = float(assumptions.coal_emission_factor_t_per_mwh) / float(assumptions.heat_rate_gj_per_mwh)

    route_available = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=bool)
    route_available[:, UNABATED] = True
    route_available[:, CCS] = True
    reduction_mt = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    captured_mt = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    opex_cny = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    capacity_mt = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    capex_cny_per_mt = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    water_m3 = np.zeros((n, len(INDUSTRY_ROUTES)), dtype=np.float64)
    h2_demand_kg = np.zeros(n, dtype=np.float64)  # 氢路线每单位份额的需氢量，kg H2

    # unabated：hub 按现状运行。无减排、无额外成本，取水保持现状。
    water_m3[:, UNABATED] = base_water_m3

    # ccs：按 `capture_rate` 捕集 hub 的全部排放，燃烧排放与工艺排放一视同仁——这对水泥
    # 恰恰是关键：水泥 63% 的排放来自煅烧，任何燃料替代都碰不到它们。
    captured_mt[:, CCS] = co2_mt * capture_rate
    captured_t = captured_mt[:, CCS] * 1e6
    # 捕集岛改造 capex 按捕集能力定规模（每 t/a 能力的 CNY x 新增的捕集能力 t/a），与煤电改造
    # 一样做学习调整；固定运维取该 capex 的一个比例，与能耗、耗材一样按当年捕集量计，不按能力存量计，
    # 随产量升降（煤电 `ccs_om_matrix` 则按改造 MW x 份额计，与利用小时无关）。成本乘子只乘 capex
    # （固定运维随之），能耗与耗材按模型价格计、不乘，与煤电 `ccs_cost_multiplier` 同口径（2026-09-23 前也乘）。
    capex_unit = np.array([capture_capex_cny_per_t_yr(s) for s in sectors], dtype=np.float64)
    capex_unit = capex_unit * learning * cost_multiplier
    variable_unit = np.array(
        [capture_variable_cost_cny_per_t(s, float(c), elec_price_mwh) for s, c in zip(sectors, coal_price_gj)],
        dtype=np.float64,
    )
    capacity_mt[:, CCS] = captured_mt[:, CCS]
    capex_cny_per_mt[:, CCS] = capex_unit * 1e6
    opex_cny[:, CCS] = captured_t * (capex_unit * INDUSTRY_CCS_FIXED_OM_FRACTION + variable_unit)
    # 再生蒸汽由燃煤锅炉产生，其 CO2 直接放空，所以该路线的净减排是捕集量减去这部分蒸汽 CO2
    # （只需压缩的化工气流为零）。与煤电侧能耗惩罚排放的处理口径相同。
    steam_co2_per_t = np.array(
        [capture_steam_co2_t_per_t(s, emission_factor_t_per_gj) for s in sectors], dtype=np.float64
    )
    reduction_mt[:, CCS] = captured_mt[:, CCS] * (1.0 - steam_co2_per_t)
    water_m3[:, CCS] = base_water_m3 + captured_t * INDUSTRY_CAPTURE_WATER_M3_PER_T_CO2

    # h2：只在该行业确有氢路线时开放。
    quota_ratio = {s: _advanced_quota_ratio(s) for s in set(sectors) if SECTOR_HAS_H2_ROUTE.get(s, False)}
    for hub_idx in range(n):
        sector = sectors[hub_idx]
        if not SECTOR_HAS_H2_ROUTE.get(sector, False):
            continue
        if h2_intensity_t_per_t[hub_idx] <= 0.0 or production_t[hub_idx] <= 0.0:
            # 所在行业有氢路线、但点源表里没有需氢量的 hub 无法定价。
            # 宁可让该路线保持关闭，也不按零价计。
            continue
        route_available[hub_idx, H2] = True
        reduction_mt[hub_idx, H2] = co2_mt[hub_idx] * float(INDUSTRY_H2_ABATEMENT_FRACTION[sector])
        k_kg_per_t = float(h2_intensity_t_per_t[hub_idx]) * 1000.0
        # 按 hub 的全部产量计重建路线的 capex，在其上计固定运维，再加由文献锚点反推的
        # 非氢运行差额（见 `industry.py` 的模块 docstring）。买氢之前的年度部分可以为负
        # （锚点把省下的化石原料计为收益）；求解器计入目标的是 max(0, 年度部分 + 购氢费)。
        # `h2_multiplier` 只乘路线 capex（固定运维随之），与两侧 CCS 的乘子同口径；不乘氢，也不乘
        # 反推的非氢运行差额——后者固定在乘子为 1 时的值。2026-09-23 前乘子还乘差额里路线自身的
        # 成本，使锚点价下的平准化溢价恰为乘子 x 锚点溢价。
        route_capex_unit = h2_route_capex_cny_per_t_yr(sector) * h2_multiplier
        opex_delta_unit = h2_route_opex_delta_cny_per_t(sector, float(h2_intensity_t_per_t[hub_idx]), rate)
        capacity_mt[hub_idx, H2] = production_t[hub_idx] / 1e6
        capex_cny_per_mt[hub_idx, H2] = route_capex_unit * 1e6
        opex_cny[hub_idx, H2] = production_t[hub_idx] * (
            route_capex_unit * INDUSTRY_H2_ROUTE_FIXED_OM_FRACTION + opex_delta_unit
        )
        h2_demand_kg[hub_idx] = k_kg_per_t * production_t[hub_idx]
        ratio = quota_ratio[sector] if INDUSTRY_H2_USES_ADVANCED_QUOTA else 1.0
        water_m3[hub_idx, H2] = base_water_m3[hub_idx] * ratio

    # 注意：`scenario.water_multiplier` 缩放的是可用水量而不是需水量，所以这里故意不乘——
    # 它只在 `_basin_cap_data` 里对流域余量乘一次。
    return IndustryYearData(
        hub_ids=hubs["hub_id"].astype(str).tolist(),
        sectors=sectors,
        target_groups=hubs["target_group"].astype(str).to_numpy(),
        output_scale=output_scale,
        baseline_emissions_mt=co2_mt,
        route_available=route_available,
        reduction_mt=reduction_mt,
        captured_mt=captured_mt,
        opex_cny=opex_cny,
        capacity_mt_per_share=capacity_mt,
        capex_cny_per_mt=capex_cny_per_mt,
        h2_demand_kg_per_share=h2_demand_kg,
        water_m3=water_m3,
        h2_price_cny_per_kg=h2_price_mean,
        capture_learning_factor=learning,
        # 各路线的经济寿命，供求解器的期末残值抵扣读取。
        capex_lifetime_years={CCS: INDUSTRY_CAPTURE_LIFETIME_YEARS, H2: INDUSTRY_H2_LIFETIME_YEARS},
    )


def basin_membership(industry: IndustryInputs, basin_codes: list[str]) -> np.ndarray:
    """(n_basins, n_hubs) 指示矩阵：每个工业 hub 位于哪个流域。

    与煤电的成员矩阵分开存放而不拼接：两套下标空间保持独立，下游就不可能悄悄把 hub 下标
    当成电厂下标来读。

    Args:
        industry: 准备好的工业输入，带 `basin_code`。
        basin_codes: 流域码，顺序与上限约束所用的一致。

    Returns:
        指示矩阵。

    Raises:
        ValueError: 有 hub 不在上限表的任何一个流域里。
    """
    if "basin_code" not in industry.hubs.columns:
        raise ValueError("industry hubs carry no basin_code; prepare_industry ran with assign_basins=False")
    hub_basins = industry.hubs["basin_code"].astype(str).to_numpy()
    membership = np.zeros((len(basin_codes), len(hub_basins)), dtype=np.float64)
    for row, code in enumerate(basin_codes):
        membership[row, :] = (hub_basins == str(code)).astype(np.float64)
    unmatched = int(len(hub_basins) - membership.sum())
    if unmatched:
        raise ValueError(f"{unmatched} industrial hubs fell outside every basin in water_basin_caps.csv")
    return membership
