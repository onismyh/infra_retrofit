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
from .year_types import RebuiltShares, YearData

# 导入流量缩放系数，用于数值稳定
from ..constants import AMMONIA_FLOW_SCALE, WATER_FLOW_SCALE, BIOMASS_FLOW_SCALE

# 运行路径（退役之外）的列：有到期装机的 hub 的份额在这些路径上拆出重建部分（`_add_rebuilt_split`），
# 空冷改造的在用量是这些路径上空冷份额之和（`model_linking.add_capacity_vintages`）。
RUNNING = tuple(k for k, pathway in enumerate(PATHWAYS) if pathway != "retire")


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
    model, share, rebuild_class, air_share, expired_share, class_share, plant_count: int, allow_air: bool, sfx: str
) -> tuple[RebuiltShares, RebuiltShares]:
    """有到期装机的 hub 把运行路径的份额拆出原址重建机组承担的部分，按重建毛热耗分类 c（`data_prep._with_expiry`），
    各类与未重建部分的毛热耗不同。

    每条运行路径 k：r_{c,k} >= 0，Σ_c r_{c,k} <= x_k；Σ_k r_{c,k} <= 该类的重建份额 ρ_c（重建机组至多全部运行）；
    Σ_c Σ_k r_{c,k} + 退役份额 >= f（未重建部分 Σ_k (x_k - Σ_c r_{c,k}) 不超过未到期装机 1 - f；f = 1 时各路径全由
    重建部分承担）。空冷份额同样拆：ra_{c,k} <= r_{c,k}、Σ_c ra_{c,k} <= a_k、a_k - Σ_c ra_{c,k} <= x_k - Σ_c r_{c,k}。
    掺烧各档的拆分在 `_add_blend_level_constraints`。哪几台机组运行、退役、改空冷由求解器定，模型不追踪。
    只拆两部分毛热耗不同的 hub：部分到期（0 < f < 1），或全部到期且已到期的有两类以上（后者 2026-10-02 起；此前
    重建部分只有一类，取到期机组重建热耗的装机加权平均）。返回 r、ra 两份 {厂: {类: {路径列: 变量}}}，只含拆分的 hub。
    """
    retire = PATHWAY_INDEX["retire"]
    rebuilt_share: RebuiltShares = {}
    rebuilt_air_share: RebuiltShares = {}
    for p in range(plant_count):
        expired = float(expired_share[p])
        classes = [c for c in range(class_share.shape[1]) if float(class_share[p, c]) > 0.0]
        if expired <= 0.0 or (expired >= 1.0 and len(classes) < 2):
            continue
        r = {
            c: {k: model.addVar(lb=0.0, ub=1.0, name=f"rebuilt_share_{p}_{c}_{PATHWAYS[k]}{sfx}") for k in RUNNING}
            for c in classes
        }
        for k in RUNNING:
            model.addConstr(
                gp.quicksum(r[c][k] for c in classes) <= share[p, k], name=f"rebuilt_le_share_{p}_{PATHWAYS[k]}{sfx}"
            )
        for c in classes:
            model.addConstr(gp.quicksum(r[c].values()) <= rebuild_class[p, c], name=f"rebuilt_le_rebuild_{p}_{c}{sfx}")
        model.addConstr(
            gp.quicksum(var for part in r.values() for var in part.values()) + share[p, retire] >= expired,
            name=f"unexpired_running_le_cap_{p}{sfx}",
        )
        rebuilt_share[p] = r
        if not allow_air:
            continue
        ra = {
            c: {k: model.addVar(lb=0.0, ub=1.0, name=f"rebuilt_air_share_{p}_{c}_{PATHWAYS[k]}{sfx}") for k in RUNNING}
            for c in classes
        }
        for k in RUNNING:
            name = f"{p}_{PATHWAYS[k]}{sfx}"
            rebuilt_air = gp.quicksum(ra[c][k] for c in classes)
            model.addConstrs((ra[c][k] <= r[c][k] for c in classes), name=f"rebuilt_air_le_rebuilt_{name}")
            model.addConstr(rebuilt_air <= air_share[p, k], name=f"rebuilt_air_le_air_{name}")
            model.addConstr(
                air_share[p, k] - rebuilt_air <= share[p, k] - gp.quicksum(r[c][k] for c in classes),
                name=f"unexpired_air_le_unexpired_{name}",
            )
        rebuilt_air_share[p] = ra
    return rebuilt_share, rebuilt_air_share


def _add_rebuilt_parts(model, z, classes, name: str) -> dict[int, gp.Var]:
    """z 里由各类原址重建机组承担的部分 rz_c：rz_c >= 0，Σ_c rz_c <= z（`_add_rebuilt_split`）。"""
    rz = {c: model.addVar(lb=0.0, ub=1.0, name=f"{name}_{c}") for c in classes}
    model.addConstr(gp.quicksum(rz.values()) <= z, name=f"{name}_le")
    return rz


def _rebuilt_dot(row, part: dict[int, gp.Var]):
    """Σ_k row[k] x part[k]：路径系数行（两部分之差）乘重建部分的份额（`_add_rebuilt_split`）。"""
    return gp.quicksum(float(row[k]) * var for k, var in part.items())


class _BlendCoeffs(NamedTuple):
    """一个厂掺烧项随毛热耗变的系数：`YearData` 的第 p 行，或某一类重建部分的差（`YearData.rebuilt_deltas` 的第 p 行）。"""

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


def _add_blend_level_constraints(
    model,
    share,
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_data: YearData,
    sfx: str,
    rebuilt_share: RebuiltShares | None = None,
) -> tuple:
    """为每个厂添加掺烧档位变量（独热二元变量，或连续 hub 下改造到各档位的容量份额）、线性化用的 zeta 以及用量约束。

    拆出重建部分的 hub（`rebuilt_share` 里有的厂，`_add_rebuilt_split`）把各档的 z 再按重建热耗类拆出重建部分 rz_c：
    rz_{c,l} >= 0，Σ_c rz_{c,l} <= z_l，Σ_l rz_{c,l} = 该路径该类的重建份额；用量、减排、惩罚各项按同式逐类加一遍重建部分，
    系数换成该类与未重建部分之差（`YearData.rebuilt_deltas`）。

    掺生物质比例的炉型上限（2026-10-02 起，BECCS 同）：落在高于 `biomass_blend_max_pulverized` 的档位上的份额不超过 hub 的
    CFB 装机份额（`YearData.cfb_share`），落在高于 `biomass_blend_max_cfb` 的档位上的为零；约束加在 `bio_layers` 的对应层上。
    档位须在 (0, 1] 内严格升序（分层与炉型上限都按升序取），煤粉炉上限须为正且不高于 CFB 上限，否则报错。

    Returns
    -------
    select_b : MVar (plant_count, L_b+1)  — 档位 0 = 不掺生物质
    select_a : MVar (plant_count, L_a+1)  — 档位 0 = 不掺氨
    blend_level_b : MVar (plant_count,)   — 在用的生物质掺烧能力 Σ_l (l+1)·(z_bio[p,l] + z_beccs[p,l])，即档位下标 x
        落在该档的份额（生物质与 BECCS 共用档位能力），供结果表与掺烧能力的固定运维（`model_costs._operating_costs`，
        2026-10-02 起）。2026-10-02 前是 Σ l·select_b，capex 计在它的增量上
    blend_level_a : MVar (plant_count,)   — 在用的掺氨能力 Σ_l (l+1)·z_amm[p,l]，只供结果表
    bio_layers : list[list[LinExpr]]      — [厂][j] 在用掺烧能力的第 j+1 层 Σ_{l>=j} (z_bio[p,l] + z_beccs[p,l])（j、l 从 0
        起，即落在第 j+1 档及以上的份额），各层相加即 blend_level_b；掺烧能力按层分代、计 capex（`model_linking.add_capacity_vintages`）
    amm_layers : list[list[LinExpr]]      — 同上，Σ_{l>=j} z_amm[p,l]
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
    rebuilt_blend_x_share : dict[(厂, 类, 路径列), LinExpr] — 各类重建部分的 Σ β·rz，生物质、BECCS、氨三列
    bio_blend_x_share   : list[LinExpr] — Σ β_b[l] · zeta_bio[p,l]，每厂一项（生物质路径的掺烧比例 × 份额）
    beccs_blend_x_share : list[LinExpr] — Σ β_b[l] · zeta_beccs[p,l]，每厂一项
    amm_blend_x_share   : list[LinExpr] — Σ β_a[l] · zeta_amm[p,l]，每厂一项
        （后三项只供结果表换算有效掺烧比例：不加约束，不进目标。连续 hub 下 `blend_level` 是
        档位下标的加权和，换算不出比例）
    """
    blend_b = np.asarray(scenario.biomass_blend_levels, dtype=np.float64)
    blend_a = np.asarray(scenario.ammonia_blend_levels, dtype=np.float64)
    for name, levels in (("biomass_blend_levels", blend_b), ("ammonia_blend_levels", blend_a)):
        if not (np.all(np.diff(levels) > 0.0) and np.all((levels > 0.0) & (levels <= 1.0))):
            raise ValueError(f"{name} must be strictly ascending within (0, 1], got {tuple(levels.tolist())}")
    if not 0.0 < assumptions.biomass_blend_max_pulverized <= assumptions.biomass_blend_max_cfb:
        raise ValueError(
            f"need 0 < biomass_blend_max_pulverized ({assumptions.biomass_blend_max_pulverized}) "
            f"<= biomass_blend_max_cfb ({assumptions.biomass_blend_max_cfb})"
        )
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
    rebuilt_blend_x_share: dict[tuple[int, int, int], gp.LinExpr] = {}
    bio_layers: list[list[gp.LinExpr]] = []
    amm_layers: list[list[gp.LinExpr]] = []
    rebuilt_share = rebuilt_share or {}
    bio_idx, beccs_idx, amm_idx = PATHWAY_INDEX["biomass"], PATHWAY_INDEX["beccs"], PATHWAY_INDEX["ammonia"]
    # 第一个高于煤粉炉上限、高于 CFB 上限的生物质档位（下标从 0 起；没有为 None）。
    above_pc = next((j for j, beta in enumerate(blend_b) if beta > assumptions.biomass_blend_max_pulverized), None)
    above_cfb = next((j for j, beta in enumerate(blend_b) if beta > assumptions.biomass_blend_max_cfb), None)

    for p in range(plant_count):
        # 改造后的运行带有效率比（重建电厂）和改造后的
        # CF 提升；燃料用量还要再按电厂热耗率缩放。
        base = _blend_coeffs(year_data, p)
        # 拆出重建部分的 hub：每档的各项按同式逐类再加一遍重建部分，z 换成 rz_c、系数换成该类与未重建部分之差。
        classes = rebuilt_share.get(p, {})
        deltas = {c: _blend_coeffs(year_data.rebuilt_deltas[c], p) for c in classes}
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
        bio_in_use = gp.LinExpr()  # 在用的掺烧能力：Σ (档位下标) x z，生物质与 BECCS 共用档位能力

        rebuilt_bio_bxs = {c: gp.LinExpr() for c in classes}
        rebuilt_beccs_bxs = {c: gp.LinExpr() for c in classes}
        z_bio_all: list[gp.Var] = []
        z_beccs_all: list[gp.Var] = []
        rz_bio_all: dict[int, list[gp.Var]] = {c: [] for c in classes}
        rz_beccs_all: dict[int, list[gp.Var]] = {c: [] for c in classes}
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
            bio_in_use += (level + 1) * (z_bio + z_beccs)
            parts = [(base, z_bio, z_beccs)]
            if classes:
                rz_bio = _add_rebuilt_parts(model, z_bio, classes, f"rzb_{p}_{level}{sfx}")
                rz_beccs = _add_rebuilt_parts(model, z_beccs, classes, f"rzbc_{p}_{level}{sfx}")
                for c in classes:
                    rz_bio_all[c].append(rz_bio[c])
                    rz_beccs_all[c].append(rz_beccs[c])
                    rebuilt_bio_bxs[c] += beta_b * rz_bio[c]
                    rebuilt_beccs_bxs[c] += beta_b * rz_beccs[c]
                    parts.append((deltas[c], rz_bio[c], rz_beccs[c]))
            for coeffs, x_bio, x_beccs in parts:
                bio_use_expr += coeffs.heat_rate * beta_b * (G_bio * x_bio + G_beccs * x_beccs) / BIOMASS_FLOW_SCALE
                bio_red += coeffs.emissions_retrofit * beta_b * x_bio
                beccs_blend_red += coeffs.emissions_retrofit * beta_b * x_beccs
                # 能耗惩罚：β_b × coeff × (G_bio·z_bio + G_beccs·z_beccs)
                bio_penalty += beta_b * coeffs.penalty * (G_bio * x_bio + G_beccs * x_beccs)
                bio_penalty_emissions += beta_b * (
                    coeffs.penalty_emissions * G_bio * x_bio + coeffs.beccs_penalty_emissions * G_beccs * x_beccs
                )
                beccs_penalty_captured += beta_b * coeffs.beccs_penalty_captured * G_beccs * x_beccs

        # 一条路径的份额恰好分摊到各档位上。上界：多个档位份额为正时，减排量不会被重复计算；
        # 下界：份额为正就至少按最低档掺烧。独热档位下 McCormick 精确，加上 `nb_b` / `nb_a`（选档位 0 时份额为 0），
        # 等式自动成立；连续 hub 下 McCormick 只给 z 上下界，2026-09-27 前这里是 `<=`，份额可以有一部分不落在任何档位上
        # （BECCS 不掺生物质就是 CCS；"生物质"份额不烧生物质，也能拿改造路径的 CF 提升）。
        model.addConstr(gp.quicksum(z_bio_all) == s_bio, name=f"zsum_bio_{p}{sfx}")
        model.addConstr(gp.quicksum(z_beccs_all) == s_beccs, name=f"zsum_beccs_{p}{sfx}")
        for c, r in classes.items():
            # 各类的重建份额同样恰好分摊到各档位上。
            model.addConstr(gp.quicksum(rz_bio_all[c]) == r[bio_idx], name=f"rzsum_bio_{p}_{c}{sfx}")
            model.addConstr(gp.quicksum(rz_beccs_all[c]) == r[beccs_idx], name=f"rzsum_beccs_{p}_{c}{sfx}")
            rebuilt_blend_x_share[(p, c, bio_idx)] = rebuilt_bio_bxs[c]
            rebuilt_blend_x_share[(p, c, beccs_idx)] = rebuilt_beccs_bxs[c]
        model.addConstr(biomass_use_gj[p] == bio_use_expr, name=f"bu_{p}{sfx}")
        model.addConstr(blend_level_b[p] == bio_in_use, name=f"blv_b_{p}{sfx}")
        # 在用的掺烧能力按档位分层：第 j+1 层（j 从 0 起）是落在第 j+1 档及以上的份额。能力按层分代，闲置的低档能力不能顶替高档。
        bio_layers.append([gp.quicksum(z_bio_all[j:]) + gp.quicksum(z_beccs_all[j:]) for j in range(L_b)])
        # 炉型上限（2026-10-02 起）：高于煤粉炉上限的档位只有 CFB 能用，高于 CFB 上限的档位谁都不能用。
        if above_pc is not None:
            model.addConstr(bio_layers[p][above_pc] <= float(year_data.cfb_share[p]), name=f"cfb_b_{p}{sfx}")
        if above_cfb is not None:
            model.addConstr(bio_layers[p][above_cfb] <= 0.0, name=f"cfb_max_b_{p}{sfx}")

        amm_use_expr = gp.LinExpr()
        amm_red = gp.LinExpr()
        amm_bxs = gp.LinExpr()
        amm_in_use = gp.LinExpr()
        rebuilt_amm_bxs = {c: gp.LinExpr() for c in classes}
        z_amm_all: list[gp.Var] = []
        rz_amm_all: dict[int, list[gp.Var]] = {c: [] for c in classes}
        for level, beta_a in enumerate(blend_a):
            bin_a = select_a[p, level + 1]
            z_amm = _add_mccormick_product(model, s_amm, bin_a, f"za_{p}_{level}{sfx}")
            z_amm_all.append(z_amm)
            amm_bxs += beta_a * z_amm
            amm_in_use += (level + 1) * z_amm
            amm_parts = [(base, z_amm)]
            if classes:
                rz_amm = _add_rebuilt_parts(model, z_amm, classes, f"rza_{p}_{level}{sfx}")
                for c in classes:
                    rz_amm_all[c].append(rz_amm[c])
                    rebuilt_amm_bxs[c] += beta_a * rz_amm[c]
                    amm_parts.append((deltas[c], rz_amm[c]))
            for coeffs, x_amm in amm_parts:
                amm_use_expr += G_amm * coeffs.heat_rate / lhv * beta_a * x_amm / AMMONIA_FLOW_SCALE
                amm_red += coeffs.emissions_retrofit * beta_a * x_amm

        # 同生物质：份额恰好分摊到各档位上。
        model.addConstr(gp.quicksum(z_amm_all) == s_amm, name=f"zsum_amm_{p}{sfx}")
        for c, r in classes.items():
            model.addConstr(gp.quicksum(rz_amm_all[c]) == r[amm_idx], name=f"rzsum_amm_{p}_{c}{sfx}")
            rebuilt_blend_x_share[(p, c, amm_idx)] = rebuilt_amm_bxs[c]
        model.addConstr(ammonia_use_kg[p] == amm_use_expr, name=f"au_{p}{sfx}")
        model.addConstr(blend_level_a[p] == amm_in_use, name=f"blv_a_{p}{sfx}")
        amm_layers.append([gp.quicksum(z_amm_all[j:]) for j in range(L_a)])

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
        blend_level_b, blend_level_a, bio_layers, amm_layers,
        biomass_use_gj, ammonia_use_kg,
        bio_red_exprs, beccs_blend_red_exprs, amm_red_exprs,
        bio_penalty_exprs, bio_penalty_emissions_exprs,
        beccs_penalty_captured_exprs, rebuilt_blend_x_share,
        bio_blend_x_share, beccs_blend_x_share, amm_blend_x_share,
    )


def _add_plant_path_constraints(
    model,
    share,
    rebuild_class,
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
    # 退役路径不发电，空冷份额的上界为零（2026-10-02 前由已装存量 >= 各路径空冷份额之和压低，存量有富余时可任取）。
    air_ub = np.ones((plant_count, pathway_count))
    air_ub[:, PATHWAY_INDEX["retire"]] = 0.0
    air_share = model.addMVar((plant_count, pathway_count), lb=0.0, ub=air_ub, name=f"air_share{sfx}")
    # 本年新建的空冷改造（占仍湿冷装机的份额）：在役（寿命内历年新建之和）>= 运行路径上的空冷份额之和，到寿命退出，
    # capex 计在新建上（`model_linking.add_capacity_vintages`，2026-10-02 起；此前是只增不减、永不到期的已装存量，
    # capex 计在它的增量上）。
    air_new = model.addMVar(plant_count, lb=0.0, name=f"air_new{sfx}")
    if not allow_air:
        model.addConstr(air_share == 0.0, name=f"air_share_off{sfx}")
        model.addConstr(air_new == 0.0, name=f"air_new_off{sfx}")

    # 有到期装机的 hub 的份额按重建热耗类拆出原址重建部分（与未重建部分毛热耗不同）。
    rebuilt_share, rebuilt_air_share = _add_rebuilt_split(
        model, share, rebuild_class, air_share, year_data.expired_share, year_data.rebuilt_class_share,
        plant_count, allow_air, sfx,
    )
    (
        select_b, select_a,
        blend_level_b, blend_level_a, blend_layers_b, blend_layers_a,
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
        for c, r in rebuilt_share.get(plant_idx, {}).items():
            # 重建部分（逐类）：乘路径份额的各项按同式再加一遍，份额换成该类的重建部分 r、系数换成该类与未重建部分之差
            # （乘 Σβz 的掺烧项已含重建部分；空冷背压按 hub 毛热耗计，没有差）。
            delta = year_data.rebuilt_deltas[c]
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
        select_b, select_a, blend_level_b, blend_level_a, blend_layers_b, blend_layers_a,
        bio_penalty_exprs, plant_reduction_exprs, air_share, air_new,
        bio_blend_x_share, beccs_blend_x_share, amm_blend_x_share,
        rebuilt_share, rebuilt_air_share, rebuilt_blend_x_share,
    )
