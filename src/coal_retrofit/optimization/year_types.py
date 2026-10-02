"""逐年容器：`YearData` 是一个规划年的全部系数，`YearPayload` 是这一年的变量与表达式。

两者此前都是 `dict[str, object]`：键名拼错要到运行到那一行才报错，读者也看不出有哪些键。
字段名与原字典键一一对应。工业侧（两者的 `industry` 字段）同样是 dataclass：
`industry_matrices.IndustryYearData` 与 `model_industry.IndustryPayload`。

求解后的结果仍是字典（`ys["share"]` 的读法不变），键由文件末的 TypedDict 声明：
`SolveResult` 是 `_solve_joint_multi_period` 的返回值，`YearSolution` 是其中每年一项，
`SolveSlacks` 是每年的松弛量。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import gurobipy as gp
    from scipy import sparse

    from .industry_matrices import IndustryYearData
    from .model_industry import IndustryPayload
    from .vintage import StockYear

# gurobipy 13 的存根把 MVar 的标量下标 `x[i, j]` 与 `x.sum()` 标成 MVar / MLinExpr，而 `quicksum`、
# `LinExpr.__iadd__` 的存根只收 `float | Var | LinExpr`；运行时它们是 0 维对象，照常参与求和。
# 字段若直接标 `gp.MVar`，mypy 会在几十处正确的建模代码上报假阳性，所以按 Any 放行，别名只说明字段是什么。
GrbMVar = Any
GrbExpr = Any  # LinExpr、0 维 MLinExpr 或 float
# 原址重建部分的份额：{厂: {重建热耗类: {路径列: 变量}}}，只含拆出重建部分的 hub（`constraints._add_rebuilt_split`）。
RebuiltShares = dict[int, dict[int, dict[int, Any]]]


@dataclass(frozen=True)
class RebuiltDelta:
    """一类原址重建部分与未重建部分的系数之差（各按自己的毛热耗算，`plant_matrices`），每个重建热耗类一份。

    字段与 `YearData` 的同名字段同形状、同单位，乘该类重建部分的份额进约束与成本（`constraints._add_rebuilt_split`）。
    全部到期的 hub 没有未重建部分，系数取各类重建热耗的装机加权平均，差为该类与平均之差；没有该类的 hub 差为零。
    空冷背压的排放与捕集按 hub 毛热耗计，没有差。
    """

    heat_rate_eff: np.ndarray
    emissions_operating_mt: np.ndarray
    emissions_retrofit_mt: np.ndarray
    energy_penalty_matrix: np.ndarray
    biomass_penalty_coeff_per_level: np.ndarray
    ccs_penalty_emissions_matrix: np.ndarray
    ccs_penalty_captured_matrix: np.ndarray
    biomass_penalty_emissions_coeff_per_level: np.ndarray
    beccs_penalty_emissions_coeff_per_level: np.ndarray
    beccs_penalty_captured_coeff_per_level: np.ndarray
    baseline_net_matrix: np.ndarray
    air_penalty_cost_matrix: np.ndarray


@dataclass(frozen=True)
class YearData:
    """`_build_year_matrices` 的输出。煤电厂侧的路径矩阵形状 (plant_count, len(PATHWAYS))、列序同 PATHWAYS；
    其余形状不同的字段在旁边注明。"""

    # --- 煤电厂侧（`plant_matrices._plant_operating_matrices`）。随毛热耗变的系数按未重建部分的毛热耗
    #     `heat_rate_eff`（未到期机组；全部到期的 hub 为各类重建热耗的平均）算，各类重建部分的差在 `rebuilt_deltas` ---
    hours_scale: float
    generation: np.ndarray
    generation_by_pathway: np.ndarray
    emissions_mt: np.ndarray
    emissions_operating_mt: np.ndarray
    emissions_retrofit_mt: np.ndarray
    heat_rate_eff: np.ndarray
    expired_share: np.ndarray  # (plant_count,) 本年已到期的装机份额 f（`data_prep._with_expiry`）
    # (plant_count, 类数) 本年已到期的装机里各重建热耗类的份额 f_c，按类相加为 f；类按重建热耗升序，各年同序。
    rebuilt_class_share: np.ndarray
    rebuilt_deltas: tuple[RebuiltDelta, ...]  # 每类一份，与 `rebuilt_class_share` 的列同序
    capacity_mw: np.ndarray
    fixed_cost_matrix: np.ndarray
    energy_penalty_matrix: np.ndarray
    biomass_penalty_coeff_per_level: np.ndarray
    ccs_penalty_emissions_matrix: np.ndarray
    ccs_penalty_captured_matrix: np.ndarray
    biomass_penalty_emissions_coeff_per_level: np.ndarray
    beccs_penalty_emissions_coeff_per_level: np.ndarray
    beccs_penalty_captured_coeff_per_level: np.ndarray
    ccs_retrofit_capex_matrix: np.ndarray
    baseline_net_matrix: np.ndarray
    stranded_per_plant: np.ndarray
    # 本年建成的捕集岛 (plant_count, 1)：只有一列，按 CCS capex 计（见 `model_year`）。capex 是一次性的；
    # 固定运维是它在役且在用的每一年都付的数，按本年（建设年）的单价（`vintage`）。
    retrofit_stock_capex: np.ndarray
    retrofit_stock_om: np.ndarray

    # --- 价格、封存部署、部门上限 ---
    carbon_price: float
    coal_savings_per_gj: np.ndarray
    coal_savings_per_kg_nh3: np.ndarray
    storage_injectivity_mtpa: np.ndarray
    storage_deployment_fraction: float
    sector_cap_fraction: dict[str, float]

    # --- 工业氢链路（与煤电氨共用节点；本年输入没有链路时两个关联矩阵为 None，
    #     有链路但节点或厂址都对不上时为 0 列稀疏矩阵）---
    industry_h2_links: pd.DataFrame
    industry_h2_hub_membership: sparse.csr_matrix | None
    industry_h2_node_membership: sparse.csr_matrix | None
    industry_h2_link_cost_cny_per_kg: np.ndarray

    # --- 生物质 ---
    biomass_link_hub_membership: sparse.csr_matrix
    biomass_link_node_membership: sparse.csr_matrix
    biomass_available: np.ndarray
    biomass_link_cost_cny_per_gj: np.ndarray
    biomass_flow_scale: float

    # --- 氨 ---
    ammonia_nodes: pd.DataFrame
    ammonia_links: pd.DataFrame
    ammonia_link_hub_membership: sparse.csr_matrix
    ammonia_link_node_membership: sparse.csr_matrix
    ammonia_available_kg: np.ndarray
    ammonia_link_cost_cny_per_kg: np.ndarray

    # --- 水：节点与链路、耗水与取水强度、流域指标（未激活时为 None / 空）---
    water_nodes: pd.DataFrame
    water_links: pd.DataFrame
    water_link_hub_membership: np.ndarray
    water_link_node_membership: np.ndarray
    water_available_m3: np.ndarray | None
    water_link_cost_cny_per_m3: np.ndarray
    water_intensity: np.ndarray
    air_water_intensity: np.ndarray
    withdrawal_intensity: np.ndarray | None
    air_withdrawal_intensity: np.ndarray | None
    water_basin_membership: np.ndarray | None
    water_basin_available_m3: np.ndarray | None
    water_basin_codes: list[str]

    # --- 工业（`industry_matrices.industry_year_data` 的输出）与其流域成员矩阵 (n_basins, n_industry_hubs) ---
    industry: IndustryYearData
    industry_basin_membership: np.ndarray | None

    # --- 湿冷→空冷改造（`plant_matrices._air_cooling_matrices`）---
    air_retrofit_capex_per_plant: np.ndarray
    air_penalty_emissions_matrix: np.ndarray
    air_penalty_captured_matrix: np.ndarray
    air_penalty_cost_matrix: np.ndarray
    allow_air_cooling_retrofit: bool

    # --- 求解器缩放单位 ---
    ammonia_flow_scale: float
    water_flow_scale: float

    # --- 管网边（`year_matrices._edge_matrices`）---
    edge_base_stock_mtpa: np.ndarray
    edge_max_new_mtpa: np.ndarray
    pipe_tiers_mtpa: tuple[float, ...]
    edge_tier_capex: np.ndarray
    edge_route_opex_coeff: np.ndarray


@dataclass
class YearPayload:
    """`add_year_block` 的输出：一年的变量、表达式与系数。

    `cost_exprs`、`cost_weights`、`salvage_ledger`、`objective_expr` 由 `add_year_costs` 写入，
    `retirement.add_retired_rebuild_offset` 往台账补已退役重建装机的扣回项，`_add_salvage_credit` 再补残值项并重算目标，
    所以这个类不冻结。
    """

    year: int
    interval_years: int
    year_data: YearData
    share: GrbMVar
    co2_flow_fwd: GrbMVar
    co2_flow_bwd: GrbMVar
    build_edge: GrbMVar
    add_cap: GrbMVar
    pipe_count: GrbMVar
    rebuild: GrbMVar
    # (plant_count, 类数) 各重建热耗类的重建份额 ρ_c <= f_c，Σ_c ρ_c = rebuild，跨期不减。
    rebuild_class: GrbMVar
    retrofit_new: GrbMVar
    # 本年新建的空冷改造（占仍湿冷装机的份额）与掺烧能力（占装机的份额，按档位分层，(plant_count, 档位数)，第 j 列是第 j 层，
    # 生物质、氨各一组），按建设年分代（`model_linking.add_capacity_vintages`）。
    air_new: GrbMVar
    blend_new_b: GrbMVar
    blend_new_a: GrbMVar
    new_cap_mtpa: GrbMVar
    biomass_flow_gj: GrbMVar
    ammonia_flow_kg: GrbMVar
    water_flow_m3: GrbMVar
    target_shortfall_mt: gp.Var
    target_shortfall_by_group: dict[str, gp.Var]
    biomass_slack_gj: GrbMVar
    ammonia_slack_kg: GrbMVar
    water_slack_m3: GrbMVar
    # 流域指标未激活时为 None。
    water_basin_slack_m3: GrbMVar | None
    water_basin_use_m3: GrbMVar | None
    injectivity_slack_mtpa: GrbMVar
    storage_slack_mt: GrbMVar
    edge_slack_mtpa: GrbMVar
    edge_flow_mtpa: GrbMVar
    storage_use_mtpa: GrbMVar
    captured_mt_by_plant: GrbMVar
    biomass_use_gj: GrbMVar
    ammonia_use_kg: GrbMVar
    water_use_m3: GrbMVar
    air_share: GrbMVar
    select_b: GrbMVar
    select_a: GrbMVar
    # 在用的掺烧能力 Σ_l (档位下标) x z_l，生物质（含 BECCS）、氨各一份，供结果表
    # （`constraints._add_blend_level_constraints`）。独热档位下是所选档位 x 路径份额；连续 hub 下非整数时对应不到任何一档。
    blend_level_b: GrbMVar
    blend_level_a: GrbMVar
    # 在用的掺烧能力按档位分层，[厂][j] = 落在第 j+1 档及以上的份额 Σ_{l>=j} z_l（j、l 从 0 起），各层相加即上面的 blend_level；
    # 各层是掺烧能力分代的所需能力。
    blend_layers_b: list[list[GrbExpr]]
    blend_layers_a: list[list[GrbExpr]]
    plant_reduction_exprs: list[GrbExpr]
    # 逐厂 Σβ_l·z_l（掺烧比例 × 路径份额），生物质、BECCS、氨各一份；只供结果表换算有效掺烧比例。
    biomass_blend_x_share: list[GrbExpr]
    beccs_blend_x_share: list[GrbExpr]
    ammonia_blend_x_share: list[GrbExpr]
    # 拆出原址重建部分的 hub（`constraints._add_rebuilt_split`），只含这些 hub：各运行路径份额与空冷份额里由各类重建机组
    # 承担的部分，{厂: {类: {路径列: 变量}}}；掺烧三条路径上各类重建部分的 Σβ_l·rz_l，{(厂, 类, 路径列): 表达式}。
    rebuilt_share: RebuiltShares
    rebuilt_air_share: RebuiltShares
    rebuilt_blend_x_share: dict[tuple[int, int, int], GrbExpr]
    total_reduction_mt: GrbExpr
    # 逐厂随掺烧档位变的生物质效率惩罚燃料费（CNY/yr，未折现），含重建部分的差；目标函数的 energy_penalty_cost 含其合计。
    bio_penalty_by_plant: list[GrbExpr]
    # `model_industry.add_industry_year` 的输出。
    industry: IndustryPayload
    # 成本类别 -> 折现并缩放后的表达式（碳价为零时碳成本是 0.0）。
    cost_exprs: dict[str, GrbExpr] = field(default_factory=dict)
    # 成本类别 -> (类别, 折现权重)：年度项 annual 乘折现 x 年金权重，一次性项 one_off 乘折现，期末残值 horizon_end 乘期末
    # 折现（`add_year_costs`、`_add_salvage_credit` 写入）；结果表除以它得本年不折现的值。
    cost_weights: dict[str, tuple[str, float]] = field(default_factory=dict)
    # (名称, 未折现 capex 表达式, 经济寿命年)，供期末残值；期末已退役重建装机的扣回项（`rebuild_retired`）为负。
    salvage_ledger: list[tuple[str, GrbExpr, int]] = field(default_factory=list)
    # `add_year_costs` 之前为 None。
    objective_expr: GrbExpr = None
    # 逐厂搁浅资产（CNY，未折现未缩放），`add_year_costs` 写入；结果表的逐厂成本取它的解值。
    stranded_by_plant: list[GrbExpr] = field(default_factory=list)
    # 分代能力本年的在役能力、固定运维与期末在用量（`model_linking.add_capacity_vintages` 写入，此前为 None）：
    # 煤电捕集岛、空冷改造、生物质与氨掺烧能力（每厂一项；掺烧按档位分层，每层一个），工业捕集与氢路线能力（每 hub 一项）。
    ccs_island: StockYear | None = None
    air_cooling: StockYear | None = None
    biomass_blend: list[StockYear] | None = None
    ammonia_blend: list[StockYear] | None = None
    industry_ccs: StockYear | None = None
    industry_h2: StockYear | None = None


class SolveSlacks(TypedDict):
    """每年的松弛量，已乘回物理单位；流域指标未激活时流域三项为空。"""

    target_shortfall_mt: float
    target_shortfall_by_group: dict[str, float]
    biomass_slack_gj: np.ndarray
    ammonia_slack_kg: np.ndarray
    water_slack_m3: np.ndarray
    water_basin_slack_m3: np.ndarray
    water_basin_use_m3: np.ndarray
    water_basin_codes: list[str]
    injectivity_slack_mtpa: np.ndarray
    storage_slack_mt: np.ndarray
    edge_slack_mtpa: np.ndarray


class YearSolution(TypedDict):
    """`solver_extract.extract_year_solutions` 的每年一项（求解失败时由 `empty_year_solutions` 零填充）。

    资源流量已乘回 GJ、kg、m3；`cost_breakdown_cny` 与 `objective_cny` 已乘回元。
    """

    status: str
    objective_cny: float
    share: np.ndarray
    build_edge: np.ndarray
    rebuild: np.ndarray
    rebuild_class: np.ndarray  # (plant_count, 类数)
    new_cap_mtpa: np.ndarray
    edge_flow_mtpa: np.ndarray
    co2_flow_fwd: np.ndarray
    co2_flow_bwd: np.ndarray
    storage_use_mtpa: np.ndarray
    water_use_m3: np.ndarray
    water_flow_m3: np.ndarray
    biomass_use_gj: np.ndarray
    biomass_flow_gj: np.ndarray
    ammonia_use_kg: np.ndarray
    ammonia_flow_kg: np.ndarray
    captured_mt_by_plant: np.ndarray
    air_share: np.ndarray
    # 在役的空冷改造（寿命内历年新建之和，占仍湿冷装机的份额）；2026-10-02 前是只增不减的已装存量。
    air_installed: np.ndarray
    # 在用的掺烧能力（档位下标 x 份额，`YearPayload.blend_level_b`）；2026-10-02 前是 Σ 档位下标 x 改造到该档的容量份额。
    blend_level_b: np.ndarray
    blend_level_a: np.ndarray
    # 捕集岛：本年新建 (plant_count, 1)、在役 (plant_count,)、按建设年单价的固定运维 (plant_count,)，CNY/yr。
    retrofit_new: np.ndarray
    retrofit_alive: np.ndarray
    ccs_om_by_plant: np.ndarray
    # 逐厂搁浅资产，CNY（未折现）：新增提前退役 x 每单位的剩余账面价值（`retirement.retirement_flows`）。
    stranded_by_plant: np.ndarray
    pipe_count: np.ndarray
    industry_share: np.ndarray
    # 工业路线能力，Mt/yr (hub_count, len(INDUSTRY_ROUTES))：本年新建、在役；捕集的固定运维 (hub_count,)，CNY/yr。
    industry_new_capacity_mt: np.ndarray
    industry_capacity_mt: np.ndarray
    industry_ccs_om_by_hub: np.ndarray
    industry_h2_flow_kg: np.ndarray
    plant_reduction_mt: np.ndarray
    # `YearPayload` 同名字段的值：逐厂 Σβ_l·z_l。
    biomass_blend_x_share: np.ndarray
    beccs_blend_x_share: np.ndarray
    ammonia_blend_x_share: np.ndarray
    # `YearPayload` 同名字段的值，(类数, plant_count, len(PATHWAYS))，其余 hub 与列为零：各路径份额与空冷份额里各类的
    # 重建部分；掺烧三列的 Σβ_l·rz_l。
    rebuilt_share: np.ndarray
    rebuilt_air_share: np.ndarray
    rebuilt_blend_x_share: np.ndarray
    # 逐厂生物质效率惩罚燃料费，CNY/yr（`YearPayload.bio_penalty_by_plant` 的值）。
    bio_penalty_by_plant: np.ndarray
    total_reduction_mt: float
    cost_breakdown_cny: dict[str, float]
    # `YearPayload.cost_weights`；残值台账 (名称, 未折现 capex 的值, 经济寿命年)，扣回项为负。
    cost_weights: dict[str, tuple[str, float]]
    salvage_ledger: list[tuple[str, float, int]]
    slacks: SolveSlacks
    year_data: YearData


class SolveResult(TypedDict):
    """`solver._solve_joint_multi_period` 的返回值。"""

    status: str
    objective_cny: float
    solver_quality: dict[str, float | int | str | None]
    capex_pathway_indices: tuple[int, ...]
    year_solutions: dict[int, YearSolution]
