from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    gp = None
    GRB = None

from .scenario import OptimizationAssumptions, OptimizationScenario, PATHWAYS
from ._shared import PATHWAY_INDEX
from .year_types import YearData, YearPayload

# 导入流量缩放系数，用于数值稳定
from ..constants import AMMONIA_FLOW_SCALE, WATER_FLOW_SCALE, BIOMASS_FLOW_SCALE

# 运行路径（退役之外）的列：部分到期 hub 的份额在这些路径上拆出重建部分（`_add_rebuilt_split`）。
_RUNNING = tuple(k for k, pathway in enumerate(PATHWAYS) if pathway != "retire")


def _add_vector_equality(model, lhs, rhs, length: int, name: str) -> None:
    model.addConstrs((lhs[idx] == rhs[idx] for idx in range(length)), name=name)


def _add_vector_upper_bound(model, lhs, rhs, length: int, name: str) -> None:
    model.addConstrs((lhs[idx] <= rhs[idx] for idx in range(length)), name=name)


def _add_mccormick_product(model, share_var, binary_var, name: str):
    """线性化 z = share_var * binary_var，其中 binary 取值 {0,1}，share 取值 [0,1]。

    连续 hub 下 binary 是改造到该档位的容量份额，z 是路径份额里落在该档位的部分，不等于两者之积：
    这三条只给上下界，怎样分摊由调用方的 `Σ_l z = share` 与 `z_bio + z_beccs ≤ select` 这几条约束决定。
    有了 `Σ_l z = share`，`_u1`、`_lo` 两条可由它、`_u2`、各档位份额合计为 1 与变量下界 0 推出（两种形式下都是）；
    留着不删：删了可行域不变，但模型指纹与求解路径都会变。
    """
    z = model.addVar(lb=0.0, ub=1.0, name=name)
    model.addConstr(z <= share_var,                      name=f"{name}_u1")
    model.addConstr(z <= binary_var,                     name=f"{name}_u2")
    model.addConstr(z >= share_var - (1.0 - binary_var), name=f"{name}_lo")
    return z


def _add_rebuilt_split(
    model, share, rebuild, air_share, expired_share, plant_count: int, allow_air: bool, sfx: str
) -> tuple[dict[int, dict[int, gp.Var]], dict[int, dict[int, gp.Var]]]:
    """部分到期 hub（0 < f < 1）把运行路径的份额拆出原址重建机组承担的部分；两部分毛热耗不同（`data_prep._with_expiry`）。

    每条运行路径 k：0 <= r_k <= x_k；Σr <= 重建份额 ρ（重建机组至多全部运行）；Σr + 退役份额 >= f（未重建部分
    Σ(x_k - r_k) 不超过未到期装机 1 - f）。空冷份额同样拆：ra_k <= a_k、ra_k <= r_k、a_k - ra_k <= x_k - r_k。
    掺烧各档的拆分在 `_add_blend_level_constraints`。哪几台机组运行、退役、改空冷由求解器定，模型不追踪。
    f = 0 或 1 的 hub 两部分毛热耗相同，不拆。返回 r、ra 两份 {厂: {路径列: 变量}}，只含部分到期的 hub。
    """
    retire = PATHWAY_INDEX["retire"]
    rebuilt_share: dict[int, dict[int, gp.Var]] = {}
    rebuilt_air_share: dict[int, dict[int, gp.Var]] = {}
    for p in range(plant_count):
        expired = float(expired_share[p])
        if not 0.0 < expired < 1.0:
            continue
        r = {k: model.addVar(lb=0.0, ub=1.0, name=f"rebuilt_share_{p}_{PATHWAYS[k]}{sfx}") for k in _RUNNING}
        for k, var in r.items():
            model.addConstr(var <= share[p, k], name=f"rebuilt_le_share_{p}_{PATHWAYS[k]}{sfx}")
        model.addConstr(gp.quicksum(r.values()) <= rebuild[p], name=f"rebuilt_le_rebuild_{p}{sfx}")
        model.addConstr(
            gp.quicksum(r.values()) + share[p, retire] >= expired, name=f"unexpired_running_le_cap_{p}{sfx}"
        )
        rebuilt_share[p] = r
        if not allow_air:
            continue
        ra = {k: model.addVar(lb=0.0, ub=1.0, name=f"rebuilt_air_share_{p}_{PATHWAYS[k]}{sfx}") for k in _RUNNING}
        for k, var in ra.items():
            name = f"{p}_{PATHWAYS[k]}{sfx}"
            model.addConstr(var <= air_share[p, k], name=f"rebuilt_air_le_air_{name}")
            model.addConstr(var <= r[k], name=f"rebuilt_air_le_rebuilt_{name}")
            model.addConstr(air_share[p, k] - var <= share[p, k] - r[k], name=f"unexpired_air_le_unexpired_{name}")
        rebuilt_air_share[p] = ra
    return rebuilt_share, rebuilt_air_share


def _add_rebuilt_part(model, z, name: str) -> gp.Var:
    """z 里由原址重建机组承担的部分 rz：0 <= rz <= z（`_add_rebuilt_split`）。"""
    rz = model.addVar(lb=0.0, ub=1.0, name=name)
    model.addConstr(rz <= z, name=f"{name}_le")
    return rz


def _rebuilt_dot(row, part: dict[int, gp.Var]):
    """Σ_k row[k] x part[k]：路径系数行（两部分之差）乘重建部分的份额（`_add_rebuilt_split`）。"""
    return gp.quicksum(float(row[k]) * var for k, var in part.items())


class _BlendCoeffs(NamedTuple):
    """一个厂掺烧项随毛热耗变的系数：`YearData` 的第 p 行，或重建部分的差（`YearData.rebuilt_delta` 的第 p 行）。"""

    heat_rate: float
    emissions_retrofit: float
    penalty: float
    penalty_emissions: float
    beccs_penalty_emissions: float
    beccs_penalty_captured: float


def _blend_coeffs(source, p: int) -> _BlendCoeffs:
    def at(values) -> float:
        return float(values[p]) if np.ndim(values) else float(values)  # 惩罚系数可以是标量（测试里）

    return _BlendCoeffs(
        at(source.heat_rate_eff),
        at(source.emissions_retrofit_mt),
        at(source.biomass_penalty_coeff_per_level),
        at(source.biomass_penalty_emissions_coeff_per_level),
        at(source.beccs_penalty_emissions_coeff_per_level),
        at(source.beccs_penalty_captured_coeff_per_level),
    )


def _build_air_retrofit_capex(
    model,
    year_data: YearData,
    payload: YearPayload,
    plant_count: int,
    yr_sfx: str,
    prev_payload: YearPayload | None = None,
):
    """湿冷凝汽器改为空冷的一次性 capex。

    按已装存量的增量计费，而不是按本期改造份额的增量：空冷凝汽器一旦建成就一直在
    （不像 CCS 捕集岛按寿命退出），所以空冷份额在某一期下降、下一期回升的 hub
    不能付两次钱。存量只设下限（>= 本期改造份额，>= 上期存量），
    这就够了，因为目标是最小化且系数为正。
    """
    if not bool(year_data.allow_air_cooling_retrofit):
        return 0.0
    capex_per_plant = year_data.air_retrofit_capex_per_plant
    if capex_per_plant is None:
        return 0.0
    installed = payload.air_installed
    terms = []
    for plant_idx in range(plant_count):
        coeff = float(capex_per_plant[plant_idx])
        if coeff <= 0:
            continue
        if prev_payload is None:
            terms.append(coeff * installed[plant_idx])
        else:
            previous = prev_payload.air_installed[plant_idx]
            model.addConstr(
                installed[plant_idx] >= previous, name=f"air_installed_mono_{plant_idx}_{yr_sfx}"
            )
            terms.append(coeff * (installed[plant_idx] - previous))
    return gp.quicksum(terms) if terms else 0.0


def _build_blend_upgrade_capex(
    model,
    capacity_mw,
    blend_level_b,
    blend_level_a,
    assumptions: "OptimizationAssumptions",
    plant_count: int,
    sfx: str = "",
    prev_blend_level_b=None,
    prev_blend_level_a=None,
):
    """掺烧升级 capex 表达式。prev_* 为 None 时，视为从档位 0 起步。"""
    cap = np.asarray(capacity_mw, dtype=np.float64)
    coeff_b = cap * assumptions.biomass_upgrade_capex_cny_per_mw_per_level
    coeff_a = cap * assumptions.ammonia_upgrade_capex_cny_per_mw_per_level
    if prev_blend_level_b is None:
        return coeff_b @ blend_level_b + coeff_a @ blend_level_a
    # 用辅助变量实现 max(0, delta)，防止退役电厂把掺烧档位
    # 重置为 0 时出现负 CAPEX
    delta_b_pos = model.addMVar(plant_count, lb=0.0, name=f"blend_delta_b_pos{sfx}")
    delta_a_pos = model.addMVar(plant_count, lb=0.0, name=f"blend_delta_a_pos{sfx}")
    model.addConstr(delta_b_pos >= blend_level_b - prev_blend_level_b, name=f"blend_delta_b_lb{sfx}")
    model.addConstr(delta_a_pos >= blend_level_a - prev_blend_level_a, name=f"blend_delta_a_lb{sfx}")
    return coeff_b @ delta_b_pos + coeff_a @ delta_a_pos


def _add_blend_level_constraints(
    model,
    share,
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_data: YearData,
    sfx: str,
    rebuilt_share: dict[int, dict[int, gp.Var]] | None = None,
) -> tuple:
    """为每个厂添加掺烧档位变量（独热二元变量，或连续 hub 下改造到各档位的容量份额）、线性化用的 zeta 以及用量约束。

    部分到期 hub（`rebuilt_share` 里有的厂，`_add_rebuilt_split`）把各档的 z 再拆出重建部分 rz：0 <= rz_l <= z_l，
    Σ_l rz_l = 该路径的重建份额；用量、减排、惩罚各项按同式加一遍重建部分，系数换成两部分之差（`YearData.rebuilt_delta`）。

    Returns
    -------
    select_b : MVar (plant_count, L_b+1)  — 档位 0 = 不掺生物质
    select_a : MVar (plant_count, L_a+1)  — 档位 0 = 不掺氨
    blend_level_b : MVar (plant_count,)   — 数值档位索引（Σ l·select_b）
    blend_level_a : MVar (plant_count,)
    biomass_use_gj : MVar (plant_count,)
    ammonia_use_kg : MVar (plant_count,)
    bio_red_exprs   : list[LinExpr]  — E_retrofit · Σ β_b[l] · zeta_bio[p,l]，每厂一项
    beccs_blend_red_exprs : list[LinExpr]  — E_retrofit · Σ β_b[l]·zeta_beccs[p,l]，每厂一项
        （只含掺烧替代部分；η·share 的捕集部分由调用方加上）
    amm_red_exprs   : list[LinExpr]  — E_retrofit · Σ β_a[l] · zeta_amm[p,l]，每厂一项
    bio_penalty_exprs : list[LinExpr] — 每个厂随掺烧档位变化的能耗惩罚成本
    bio_penalty_emissions_exprs : list[LinExpr] — 惩罚燃料额外排放的 CO2（Mt）
    beccs_penalty_captured_exprs : list[LinExpr] — BECCS 惩罚燃料中被捕集的 CO2（Mt）
    （以上六项含重建部分）
    rebuilt_blend_x_share : dict[(厂, 路径列), LinExpr] — 重建部分的 Σ β·rz，部分到期 hub 的生物质、BECCS、氨三列
    bio_blend_x_share   : list[LinExpr] — Σ β_b[l] · zeta_bio[p,l]，每厂一项（生物质路径的掺烧比例 × 份额）
    beccs_blend_x_share : list[LinExpr] — Σ β_b[l] · zeta_beccs[p,l]，每厂一项
    amm_blend_x_share   : list[LinExpr] — Σ β_a[l] · zeta_amm[p,l]，每厂一项
        （后三项只供结果表换算有效掺烧比例：不加约束，不进目标。连续 hub 下 `blend_level` 是
        档位下标的加权和，换算不出比例）
    """
    blend_b = np.asarray(scenario.biomass_blend_levels, dtype=np.float64)
    blend_a = np.asarray(scenario.ammonia_blend_levels, dtype=np.float64)
    L_b = len(blend_b)
    L_a = len(blend_a)
    lhv = float(assumptions.nh3_lhv_gj_per_kg)

    # select_b/select_a：第 0 列 = "不掺烧"，第 l+1 列 = "掺烧档位 l"
    # 独热二元变量（一个 hub 选一个档位），或 hub 容量中改造到各档位的
    # 连续份额；见 `OptimizationAssumptions.hub_decisions_continuous`。
    sel_vtype = GRB.CONTINUOUS if assumptions.hub_decisions_continuous else GRB.BINARY
    select_b = model.addMVar((plant_count, L_b + 1), lb=0.0, ub=1.0, vtype=sel_vtype, name=f"sel_b{sfx}")
    select_a = model.addMVar((plant_count, L_a + 1), lb=0.0, ub=1.0, vtype=sel_vtype, name=f"sel_a{sfx}")
    blend_level_b = model.addMVar(plant_count, lb=0.0, ub=float(L_b), name=f"blv_b{sfx}")
    blend_level_a = model.addMVar(plant_count, lb=0.0, ub=float(L_a), name=f"blv_a{sfx}")
    biomass_use_gj = model.addMVar(plant_count, lb=0.0, name=f"biomass_use_gj{sfx}")
    ammonia_use_kg = model.addMVar(plant_count, lb=0.0, name=f"ammonia_use_kg{sfx}")

    bio_red_exprs: list[object] = []
    beccs_blend_red_exprs: list[object] = []
    amm_red_exprs: list[object] = []
    bio_penalty_exprs: list[object] = []  # 每个厂随掺烧档位变化的能耗惩罚
    bio_penalty_emissions_exprs: list[object] = []  # 惩罚燃料额外产生的 CO2（Mt）
    # BECCS 下，掺烧惩罚燃料与被捕集的烟气在同一台锅炉里燃烧，所以只有
    # 未捕集的份额排放（Fan et al. 2023 SI eq. S42）——而被捕集的份额是
    # 实实在在的吨数，必须经管道输送并封存。
    beccs_penalty_captured_exprs: list[object] = []
    bio_blend_x_share: list[object] = []
    beccs_blend_x_share: list[object] = []
    amm_blend_x_share: list[object] = []
    rebuilt_blend_x_share: dict[tuple[int, int], gp.LinExpr] = {}
    rebuilt_share = rebuilt_share or {}
    bio_idx, beccs_idx, amm_idx = PATHWAY_INDEX["biomass"], PATHWAY_INDEX["beccs"], PATHWAY_INDEX["ammonia"]

    for p in range(plant_count):
        # 改造后的运行带有效率比（重建电厂）和改造后的
        # CF 提升；燃料用量还要再按电厂热耗率缩放。
        base = _blend_coeffs(year_data, p)
        # 部分到期 hub：每档的各项按同式再加一遍重建部分，z 换成 rz、系数换成两部分之差。
        r = rebuilt_share.get(p)
        delta = _blend_coeffs(year_data.rebuilt_delta, p) if r is not None else None
        G_bp = year_data.generation_by_pathway[p, :]
        G_bio = float(G_bp[bio_idx])
        G_beccs = float(G_bp[beccs_idx])
        G_amm = float(G_bp[amm_idx])
        s_bio = share[p, bio_idx]
        s_beccs = share[p, beccs_idx]
        s_amm = share[p, amm_idx]

        # --- 每类路径恰好选一个掺烧档位（连续 hub 下：各档位的容量份额合计为 1） ---
        model.addConstr(select_b[p, :].sum() == 1.0, name=f"sel_b_{p}{sfx}")
        model.addConstr(select_a[p, :].sum() == 1.0, name=f"sel_a_{p}{sfx}")

        # --- 若选档位 0（不掺烧），相关份额必须为 0（连续 hub 下：份额不超过改造了的容量份额 1 − select[0]）。
        # 下面的 `Σ_l z = s` 加上 `z_bio + z_beccs ≤ select_b`、`z_amm ≤ select_a` 与上面的档位份额合计为 1 已蕴含这两条
        # （两种形式下都是），留着不删，理由同 `_add_mccormick_product` ---
        model.addConstr(s_bio + s_beccs <= 1.0 - select_b[p, 0], name=f"nb_b_{p}{sfx}")
        model.addConstr(s_amm <= 1.0 - select_a[p, 0], name=f"nb_a_{p}{sfx}")

        # --- 数值掺烧档位（用于计算升级成本） ---
        model.addConstr(
            blend_level_b[p] == gp.quicksum(level * select_b[p, level] for level in range(L_b + 1)),
            name=f"blv_b_{p}{sfx}",
        )
        model.addConstr(
            blend_level_a[p] == gp.quicksum(level * select_a[p, level] for level in range(L_a + 1)),
            name=f"blv_a_{p}{sfx}",
        )

        # --- 线性化：独热档位下 z = select_b[p, l+1] × share；连续 hub 下 z 是路径份额里落在该档位的部分
        # （见 `_add_mccormick_product`） ---
        bio_use_expr = gp.LinExpr()
        bio_red = gp.LinExpr()
        beccs_blend_red = gp.LinExpr()
        bio_penalty = gp.LinExpr()  # 随掺烧档位变化的能耗惩罚（成本）
        bio_penalty_emissions = gp.LinExpr()  # 同一份惩罚燃料，折为排放的 CO2（Mt）
        beccs_penalty_captured = gp.LinExpr()  # BECCS 惩罚燃料中被捕集的份额（Mt）
        # Σ β·z（blend × share）：只供结果表换算有效掺烧比例，不加约束，不进目标。
        bio_bxs = gp.LinExpr()
        beccs_bxs = gp.LinExpr()

        rebuilt_bio_bxs = gp.LinExpr()
        rebuilt_beccs_bxs = gp.LinExpr()
        z_bio_all: list[object] = []
        z_beccs_all: list[object] = []
        rz_bio_all: list[gp.Var] = []
        rz_beccs_all: list[gp.Var] = []
        for level, beta_b in enumerate(blend_b):
            bin_b = select_b[p, level + 1]
            z_bio = _add_mccormick_product(model, s_bio, bin_b, f"zb_{p}_{level}{sfx}")
            z_beccs = _add_mccormick_product(model, s_beccs, bin_b, f"zbc_{p}_{level}{sfx}")
            # 两条路径共用改造到该档位的容量。独热形式下此式自动成立；
            # 档位取份额时它才真正起约束作用，也才有物理含义。
            model.addConstr(z_bio + z_beccs <= bin_b, name=f"zlvl_b_{p}_{level}{sfx}")
            z_bio_all.append(z_bio)
            z_beccs_all.append(z_beccs)
            bio_bxs += beta_b * z_bio
            beccs_bxs += beta_b * z_beccs
            parts = [(base, z_bio, z_beccs)]
            if delta is not None:
                rz_bio = _add_rebuilt_part(model, z_bio, f"rzb_{p}_{level}{sfx}")
                rz_beccs = _add_rebuilt_part(model, z_beccs, f"rzbc_{p}_{level}{sfx}")
                rz_bio_all.append(rz_bio)
                rz_beccs_all.append(rz_beccs)
                rebuilt_bio_bxs += beta_b * rz_bio
                rebuilt_beccs_bxs += beta_b * rz_beccs
                parts.append((delta, rz_bio, rz_beccs))
            for c, x_bio, x_beccs in parts:
                bio_use_expr += c.heat_rate * beta_b * (G_bio * x_bio + G_beccs * x_beccs) / BIOMASS_FLOW_SCALE
                bio_red += c.emissions_retrofit * beta_b * x_bio
                beccs_blend_red += c.emissions_retrofit * beta_b * x_beccs
                # 能耗惩罚：β_b × coeff × (G_bio·z_bio + G_beccs·z_beccs)
                bio_penalty += beta_b * c.penalty * (G_bio * x_bio + G_beccs * x_beccs)
                bio_penalty_emissions += beta_b * (
                    c.penalty_emissions * G_bio * x_bio + c.beccs_penalty_emissions * G_beccs * x_beccs
                )
                beccs_penalty_captured += beta_b * c.beccs_penalty_captured * G_beccs * x_beccs

        # 一条路径的份额恰好分摊到各档位上。上界：多个档位份额为正时，减排量不会被重复计算；
        # 下界：份额为正就至少按最低档掺烧。独热档位下 McCormick 精确，加上 `nb_b` / `nb_a`（选档位 0 时份额为 0），
        # 等式自动成立；连续 hub 下 McCormick 只给 z 上下界，2026-09-27 前这里是 `<=`，份额可以有一部分不落在任何档位上
        # （BECCS 不掺生物质就是 CCS；"生物质"份额不烧生物质，也能拿改造路径的 CF 提升）。
        model.addConstr(gp.quicksum(z_bio_all) == s_bio, name=f"zsum_bio_{p}{sfx}")
        model.addConstr(gp.quicksum(z_beccs_all) == s_beccs, name=f"zsum_beccs_{p}{sfx}")
        if r is not None:
            # 重建份额同样恰好分摊到各档位上。
            model.addConstr(gp.quicksum(rz_bio_all) == r[bio_idx], name=f"rzsum_bio_{p}{sfx}")
            model.addConstr(gp.quicksum(rz_beccs_all) == r[beccs_idx], name=f"rzsum_beccs_{p}{sfx}")
            rebuilt_blend_x_share[(p, bio_idx)] = rebuilt_bio_bxs
            rebuilt_blend_x_share[(p, beccs_idx)] = rebuilt_beccs_bxs
        model.addConstr(biomass_use_gj[p] == bio_use_expr, name=f"bu_{p}{sfx}")

        amm_use_expr = gp.LinExpr()
        amm_red = gp.LinExpr()
        amm_bxs = gp.LinExpr()
        rebuilt_amm_bxs = gp.LinExpr()
        z_amm_all: list[object] = []
        rz_amm_all: list[gp.Var] = []
        for level, beta_a in enumerate(blend_a):
            bin_a = select_a[p, level + 1]
            z_amm = _add_mccormick_product(model, s_amm, bin_a, f"za_{p}_{level}{sfx}")
            z_amm_all.append(z_amm)
            amm_bxs += beta_a * z_amm
            amm_parts = [(base, z_amm)]
            if delta is not None:
                rz_amm = _add_rebuilt_part(model, z_amm, f"rza_{p}_{level}{sfx}")
                rz_amm_all.append(rz_amm)
                rebuilt_amm_bxs += beta_a * rz_amm
                amm_parts.append((delta, rz_amm))
            for c, x_amm in amm_parts:
                amm_use_expr += G_amm * c.heat_rate / lhv * beta_a * x_amm / AMMONIA_FLOW_SCALE
                amm_red += c.emissions_retrofit * beta_a * x_amm

        # 同生物质：份额恰好分摊到各档位上。
        model.addConstr(gp.quicksum(z_amm_all) == s_amm, name=f"zsum_amm_{p}{sfx}")
        if r is not None:
            model.addConstr(gp.quicksum(rz_amm_all) == r[amm_idx], name=f"rzsum_amm_{p}{sfx}")
            rebuilt_blend_x_share[(p, amm_idx)] = rebuilt_amm_bxs
        model.addConstr(ammonia_use_kg[p] == amm_use_expr, name=f"au_{p}{sfx}")

        bio_red_exprs.append(bio_red)
        beccs_blend_red_exprs.append(beccs_blend_red)
        amm_red_exprs.append(amm_red)
        bio_penalty_exprs.append(bio_penalty)
        bio_penalty_emissions_exprs.append(bio_penalty_emissions)
        beccs_penalty_captured_exprs.append(beccs_penalty_captured)
        bio_blend_x_share.append(bio_bxs)
        beccs_blend_x_share.append(beccs_bxs)
        amm_blend_x_share.append(amm_bxs)

    return (
        select_b, select_a,
        blend_level_b, blend_level_a,
        biomass_use_gj, ammonia_use_kg,
        bio_red_exprs, beccs_blend_red_exprs, amm_red_exprs,
        bio_penalty_exprs, bio_penalty_emissions_exprs,
        beccs_penalty_captured_exprs, rebuilt_blend_x_share,
        bio_blend_x_share, beccs_blend_x_share, amm_blend_x_share,
    )


def _add_plant_path_constraints(
    model,
    share,
    rebuild,
    year_data: YearData,
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_suffix: str = "",
):
    pathway_count = len(PATHWAYS)
    sfx = f"_{year_suffix}" if year_suffix else ""
    captured_mt_by_plant = model.addMVar(plant_count, lb=0.0, name=f"captured_mt_by_plant{sfx}")
    water_use_m3 = model.addMVar(plant_count, lb=0.0, name=f"water_use_m3{sfx}")

    # 湿冷改空冷。`air_share[p, k]` 是 hub p 发电量中既走路径 k 又采用空冷的部分，
    # 因此 0 <= air_share <= share，hub 的改造比例为 sum_k air_share[p, k]。
    # 按路径分列可让每一项都保持线性：另一种写法是用单一空冷比例去乘
    # 各路径份额，那是双线性的，需要二元变量或 McCormick 包络。它还对应
    # 一个真实的自由度：一个 hub 聚合了约 10 台机组，运营方要选择其中
    # 哪几台同时配捕集岛和空冷凝汽器。
    allow_air = bool(year_data.allow_air_cooling_retrofit) and year_data.air_water_intensity is not None
    air_share = model.addMVar((plant_count, pathway_count), lb=0.0, ub=1.0, name=f"air_share{sfx}")
    # 已装存量：capex 按历史最高值计费，已改造 hub 的空冷份额在某一期下降、
    # 下一期回升时不会重复计费；没有它，空冷凝汽器可能被付两次钱。存量不到期（CCS 捕集岛
    # 2026-09-30 前也是这一手法，此后按建设年分代、到寿命退出，`vintage`）。
    air_installed = model.addMVar(plant_count, lb=0.0, ub=1.0, name=f"air_installed{sfx}")
    if not allow_air:
        model.addConstr(air_share == 0.0, name=f"air_share_off{sfx}")
        model.addConstr(air_installed == 0.0, name=f"air_installed_off{sfx}")
    else:
        model.addConstrs(
            (
                air_installed[plant_idx] >= gp.quicksum(
                    air_share[plant_idx, path_idx] for path_idx in range(pathway_count)
                )
                for plant_idx in range(plant_count)
            ),
            name=f"air_installed_ge_share{sfx}",
        )

    # 部分到期 hub 的份额拆出原址重建部分（两部分毛热耗不同）。
    rebuilt_share, rebuilt_air_share = _add_rebuilt_split(
        model, share, rebuild, air_share, year_data.expired_share, plant_count, allow_air, sfx
    )
    (
        select_b, select_a,
        blend_level_b, blend_level_a,
        biomass_use_gj, ammonia_use_kg,
        bio_red_exprs, beccs_blend_red_exprs, amm_red_exprs,
        bio_penalty_exprs, bio_penalty_emissions_exprs,
        beccs_penalty_captured_exprs, rebuilt_blend_x_share,
        bio_blend_x_share, beccs_blend_x_share, amm_blend_x_share,
    ) = _add_blend_level_constraints(model, share, plant_count, scenario, assumptions, year_data, sfx, rebuilt_share)

    eta = float(scenario.capture_rate)
    plant_reduction_exprs: list[object] = []
    ccs_penalty_captured = year_data.ccs_penalty_captured_matrix
    air_penalty_captured = year_data.air_penalty_captured_matrix
    un, ccs, bio, beccs, amm = (PATHWAY_INDEX[k] for k in ("unabated", "ccs", "biomass", "beccs", "ammonia"))

    for plant_idx in range(plant_count):
        E_p = float(year_data.emissions_mt[plant_idx])               # 基准
        E_op = float(year_data.emissions_operating_mt[plant_idx])    # 经效率调整
        E_rt = float(year_data.emissions_retrofit_mt[plant_idx])     # 效率 × CF 提升
        G_bp = year_data.generation_by_pathway[plant_idx, :]
        s_un = share[plant_idx, PATHWAY_INDEX["unabated"]]
        s_ccs = share[plant_idx, PATHWAY_INDEX["ccs"]]
        s_bio = share[plant_idx, PATHWAY_INDEX["biomass"]]
        s_beccs = share[plant_idx, PATHWAY_INDEX["beccs"]]
        s_amm = share[plant_idx, PATHWAY_INDEX["ammonia"]]

        # 用水量（按 WATER_FLOW_SCALE 缩放以保证数值稳定）；各路径发电量
        # （含改造提升，退役 = 0）乘以该路径的用水强度，其中各路径的
        # 空冷部分改按空冷强度计。
        air_intensity = year_data.air_water_intensity if allow_air else None
        water_expr = gp.quicksum(
            float(G_bp[path_idx])
            * (
                float(year_data.water_intensity[plant_idx, path_idx]) * share[plant_idx, path_idx]
                - (
                    float(
                        year_data.water_intensity[plant_idx, path_idx]
                        - air_intensity[plant_idx, path_idx]
                    )
                    * air_share[plant_idx, path_idx]
                    if air_intensity is not None
                    else 0.0
                )
            )
            / WATER_FLOW_SCALE
            for path_idx in range(pathway_count)
        )
        model.addConstr(water_use_m3[plant_idx] == water_expr, name=f"water_use_{plant_idx}{sfx}")
        if allow_air:
            for path_idx in range(pathway_count):
                model.addConstr(
                    air_share[plant_idx, path_idx] <= share[plant_idx, path_idx],
                    name=f"air_share_le_share_{plant_idx}_{path_idx}{sfx}",
                )

        # 物理捕集的 CO2 不同于 BECCS 的净减排。除锅炉自身烟气的 eta x 之外，
        # 捕集装置还会捕获同一锅炉里燃烧的每一份惩罚燃料的 eta
        # （CCS 能耗惩罚、BECCS 掺烧惩罚、空冷背压）；这些吨数在残余排放里
        # 按 (1-eta) 计了排放，但从未计入这里，于是它们在账面上被减掉了，
        # 却没有经过输送和封存。
        captured_expr = E_rt * eta * (s_ccs + s_beccs) + beccs_penalty_captured_exprs[plant_idx]
        if ccs_penalty_captured is not None:
            captured_expr = captured_expr + gp.quicksum(
                float(ccs_penalty_captured[plant_idx, path_idx]) * share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
                if float(ccs_penalty_captured[plant_idx, path_idx]) != 0.0
            )
        if allow_air and air_penalty_captured is not None:
            captured_expr = captured_expr + gp.quicksum(
                float(air_penalty_captured[plant_idx, path_idx]) * air_share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
                if float(air_penalty_captured[plant_idx, path_idx]) != 0.0
            )

        # 先算各路径下的实际（残余）排放，再算相对基准的减排量。
        # CF 提升与效率比都取 1 时，这恰好化简为经典的
        # 各路径减排比例乘以基准排放。
        # 效率惩罚燃料（CCS 固定部分 + 随掺烧档位变化部分）与其他煤一样燃烧并
        # 排放，因此其排放属于残余排放的一部分。
        residual_expr = (
            E_op * s_un
            + E_rt * (1.0 - eta) * s_ccs
            + (E_rt * s_bio - bio_red_exprs[plant_idx])
            + (E_rt * (1.0 - eta) * s_beccs - beccs_blend_red_exprs[plant_idx])
            + (E_rt * s_amm - amm_red_exprs[plant_idx])
            + gp.quicksum(
                float(year_data.ccs_penalty_emissions_matrix[plant_idx, path_idx]) * share[plant_idx, path_idx]
                for path_idx in range(pathway_count)
            )
            + bio_penalty_emissions_exprs[plant_idx]
            # 空冷的背压惩罚：与其他煤一样燃烧并排放。
            + (
                gp.quicksum(
                    float(year_data.air_penalty_emissions_matrix[plant_idx, path_idx])
                    * air_share[plant_idx, path_idx]
                    for path_idx in range(pathway_count)
                )
                if allow_air
                else 0.0
            )
        )
        r = rebuilt_share.get(plant_idx)
        if r is not None:
            # 部分到期 hub 的重建部分：乘路径份额的各项按同式再加一遍，份额换成重建部分 r、系数换成两部分之差
            # （乘 Σβz 的掺烧项已含重建部分；空冷背压按 hub 毛热耗计，没有差）。
            delta = year_data.rebuilt_delta
            d_op = float(delta.emissions_operating_mt[plant_idx])
            d_rt = float(delta.emissions_retrofit_mt[plant_idx])
            captured_expr = captured_expr + (
                d_rt * eta * (r[ccs] + r[beccs]) + _rebuilt_dot(delta.ccs_penalty_captured_matrix[plant_idx], r)
            )
            residual_expr = residual_expr + (
                d_op * r[un]
                + d_rt * (1.0 - eta) * (r[ccs] + r[beccs])
                + d_rt * (r[bio] + r[amm])
                + _rebuilt_dot(delta.ccs_penalty_emissions_matrix[plant_idx], r)
            )
        model.addConstr(captured_mt_by_plant[plant_idx] == captured_expr, name=f"captured_balance_{plant_idx}{sfx}")
        plant_red = E_p - residual_expr
        plant_reduction_exprs.append(plant_red)

    total_reduction_mt = gp.quicksum(plant_reduction_exprs)
    return (
        captured_mt_by_plant, biomass_use_gj, ammonia_use_kg, water_use_m3, total_reduction_mt,
        select_b, select_a, blend_level_b, blend_level_a,
        bio_penalty_exprs, plant_reduction_exprs, air_share, air_installed,
        bio_blend_x_share, beccs_blend_x_share, amm_blend_x_share,
        rebuilt_share, rebuilt_air_share, rebuilt_blend_x_share,
    )
