"""掺烧档位：路径份额在生物质（含 BECCS）与氨各档掺烧比例上的分摊 z（`Σ_l z = 路径份额`），整数 hub 另有每厂选一个档位的
独热二元变量；由 z 组成燃料用量、减排、能耗惩罚与在用掺烧能力（`add_blend_level_constraints`）。
由 `constraints._add_plant_path_constraints` 调用。"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..constants import AMMONIA_FLOW_SCALE, BIOMASS_FLOW_SCALE
from ._shared import GRB, PATHWAY_INDEX, gp
from .assumptions import OptimizationAssumptions
from .scenario import OptimizationScenario
from .year_types import GrbExpr, GrbMVar, RebuiltShares, YearData


def _add_rebuilt_parts(model, z, classes, name: str) -> dict[int, gp.Var]:
    """z 里由各类原址重建机组承担的部分 rz_c：rz_c >= 0，Σ_c rz_c <= z（`constraints_rebuilt.add_rebuilt_split`）。"""
    rz = {c: model.addVar(lb=0.0, ub=1.0, name=f"{name}_{c}") for c in classes}
    model.addConstr(gp.quicksum(rz.values()) <= z, name=f"{name}_le")
    return rz


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


class _LevelShares:
    """一个厂共用一组掺烧档位的路径（生物质与 BECCS 共用生物质档位，氨单独一组）在各档位上的份额。

    每档每条路径一个 z >= 0（路径份额里落在该档的部分）；拆出重建部分的 hub（`rebuilt` 非空）再按重建热耗类拆出 rz_c
    （`_add_rebuilt_parts`）。另记结果表用的 Σβ·z（`x_share`）、各类的 Σβ·rz（`rebuilt_x_share`）与在用掺烧能力
    Σ (档位下标) x z（`in_use`，组内各路径共用档位能力）。`paths` 是 (约束名里的路径名, z 的变量名前缀, 路径列)；
    变量与约束按 `paths` 的顺序加。
    """

    def __init__(self, model, paths, share, p: int, sfx: str, base: _BlendCoeffs,
                 deltas: dict[int, _BlendCoeffs], rebuilt: dict[int, dict[int, gp.Var]]) -> None:
        self.model, self.paths, self.p, self.sfx = model, paths, p, sfx
        self.base, self.deltas, self.rebuilt = base, deltas, rebuilt
        self.shares = [share[p, k] for _, _, k in paths]
        self.z: list[list[gp.Var]] = [[] for _ in paths]                                    # [路径][档位]
        self.rz: list[dict[int, list[gp.Var]]] = [{c: [] for c in rebuilt} for _ in paths]  # [路径][类][档位]
        self.x_share = [gp.LinExpr() for _ in paths]
        self.rebuilt_x_share = [{c: gp.LinExpr() for c in rebuilt} for _ in paths]
        self.in_use = gp.LinExpr()

    def add_level(self, level: int, beta: float, select, coupling: str) -> list[tuple[_BlendCoeffs, list]]:
        """第 level 档（掺烧比例 beta）：各路径的 z 与各类的 rz。整数 hub 另加 Σ_路径 z <= select（约束名前缀 coupling；
        select 是该档的独热变量，连续 hub 为 None，不加）。
        返回 [(系数, 各路径的份额)]：先是未重建部分 (base, z)，再是各类重建部分 (该类与未重建部分之差, rz_c)。"""
        name = f"_{self.p}_{level}{self.sfx}"
        prefixes = [prefix for _, prefix, _ in self.paths]
        zs = [self.model.addVar(lb=0.0, ub=1.0, name=f"{prefix}{name}") for prefix in prefixes]
        if select is not None:
            self.model.addConstr(gp.quicksum(zs) <= select, name=f"{coupling}{name}")
        self.in_use += (level + 1) * gp.quicksum(zs)
        for i, z in enumerate(zs):
            self.z[i].append(z)
            self.x_share[i] += beta * z
        parts: list[tuple[_BlendCoeffs, list]] = [(self.base, zs)]
        if self.rebuilt:
            rzs = [_add_rebuilt_parts(self.model, z, self.rebuilt, f"r{prefix}{name}") for z, prefix in zip(zs, prefixes)]
            for c in self.rebuilt:
                for i, rz in enumerate(rzs):
                    self.rz[i][c].append(rz[c])
                    self.rebuilt_x_share[i][c] += beta * rz[c]
                parts.append((self.deltas[c], [rz[c] for rz in rzs]))
        return parts

    def add_partition(self, out: dict[tuple[int, int, int], GrbExpr]) -> None:
        """一条路径的份额恰好分摊到各档位上（Σ_l z_l = 路径份额），各类的重建份额同样（Σ_l rz_{c,l} = r_c）。上界：多个档位份额
        为正时，减排量不会被重复计算；下界：份额为正就至少按最低档掺烧。2026-09-27 前这里是 `<=`，连续 hub 下份额可以有一部分
        不落在任何档位上（BECCS 不掺生物质就是 CCS；"生物质"份额不烧生物质，也能拿改造路径的 CF 提升）。各类的 Σβ·rz 记进
        out[(厂, 类, 路径列)]。"""
        for (label, _, _), s, z in zip(self.paths, self.shares, self.z):
            self.model.addConstr(gp.quicksum(z) == s, name=f"zsum_{label}_{self.p}{self.sfx}")
        for c, r in self.rebuilt.items():
            for i, (label, _, k) in enumerate(self.paths):
                self.model.addConstr(gp.quicksum(self.rz[i][c]) == r[k], name=f"rzsum_{label}_{self.p}_{c}{self.sfx}")
                out[(self.p, c, k)] = self.rebuilt_x_share[i][c]

    def layers(self) -> list[GrbExpr]:
        """在用掺烧能力按档位分层：第 j+1 层（j 从 0 起）是组内各路径落在第 j+1 档及以上的份额 Σ_路径 Σ_{l>=j} z_l。"""
        return [gp.quicksum(var for z in self.z for var in z[j:]) for j in range(len(self.z[0]))]


class BlendTerms(NamedTuple):
    """`add_blend_level_constraints` 的变量与表达式；表达式的列表每厂一项。用量、减排与惩罚各项含重建部分。"""

    # 在用的生物质掺烧能力 Σ_l (l+1)·(z_bio[p,l] + z_beccs[p,l])，即档位下标 x 落在该档的份额（生物质与 BECCS 共用档位能力），
    # 供结果表与掺烧能力的固定运维（`model_costs._operating_costs`，2026-10-02 起）。2026-10-02 前是 Σ l·select_b，capex 计在它的增量上
    blend_level_b: GrbMVar
    blend_level_a: GrbMVar              # 在用的掺氨能力 Σ_l (l+1)·z_amm[p,l]，供结果表与掺氨能力的固定运维（2026-10-10 起）
    # [厂][j] 在用掺烧能力的第 j+1 层 Σ_{l>=j} (z_bio[p,l] + z_beccs[p,l])（j、l 从 0 起，即落在第 j+1 档及以上的份额），各层相加即
    # blend_level_b；掺烧能力按层分代、计 capex（`model_linking.add_capacity_vintages`）
    blend_layers_b: list[list[GrbExpr]]
    blend_layers_a: list[list[GrbExpr]]  # 同上，Σ_{l>=j} z_amm[p,l]
    biomass_use_gj: GrbMVar
    ammonia_use_kg: GrbMVar
    bio_red_exprs: list[GrbExpr]          # E_retrofit · Σ β_b[l] · z_bio[p,l]
    beccs_blend_red_exprs: list[GrbExpr]  # E_retrofit · Σ β_b[l]·z_beccs[p,l]，只含掺烧替代部分；η·share 的捕集部分由调用方加上
    amm_red_exprs: list[GrbExpr]          # E_retrofit · Σ β_a[l] · z_amm[p,l]
    bio_penalty_by_plant: list[GrbExpr]   # 随掺烧档位变化的能耗惩罚成本
    bio_penalty_emissions_exprs: list[GrbExpr]   # 惩罚燃料额外排放的 CO2（Mt）
    beccs_penalty_captured_exprs: list[GrbExpr]  # BECCS 惩罚燃料中被捕集的 CO2（Mt）
    rebuilt_blend_x_share: dict[tuple[int, int, int], GrbExpr]  # (厂, 类, 路径列) -> 各类重建部分的 Σ β·rz，生物质、BECCS、氨三列
    # Σ β·z，生物质、BECCS、氨各一份（路径的掺烧比例 × 份额）：只供结果表换算有效掺烧比例，不加约束，不进目标。连续 hub 下
    # `blend_level` 是档位下标的加权和，换算不出比例
    biomass_blend_x_share: list[GrbExpr]
    beccs_blend_x_share: list[GrbExpr]
    ammonia_blend_x_share: list[GrbExpr]


def add_blend_level_constraints(
    model,
    share,
    plant_count: int,
    scenario: "OptimizationScenario",
    assumptions: "OptimizationAssumptions",
    year_data: YearData,
    sfx: str,
    rebuilt_share: RebuiltShares | None = None,
) -> BlendTerms:
    """为每个厂添加各档的路径份额 z（整数 hub 另有每厂选一个档位的独热二元变量）以及用量约束。

    连续 hub（`hub_decisions_continuous`，缺省）只有 z：z >= 0，Σ_l z_l = 路径份额，hub 内不同机组可以掺到不同档位。整数 hub 另有
    独热变量 select（第 0 列 = 不掺烧，第 l+1 列 = 档位 l，每厂合计为 1），每档 Σ_路径 z_l <= select_l：生物质与 BECCS 共用一组
    （`zlvl_b`），氨一组（`zlvl_a`），一个 hub 一年只掺一个档位。2026-10-16 前连续 hub 也有 select（改造到各档的容量份额，跨期
    CDF 不增，即掺烧比例不许下降），另有 McCormick 三条（z <= 份额、z <= select、z >= 份额 - (1 - select)）与"选档位 0 则份额为 0"
    （`nb_b` / `nb_a`）：后几条都由 Σ_l z = 份额、z <= select 与 Σ select = 1 蕴含；跨期单调删去后，连续 hub 的 select 对 z 不再有约束
    （任意 z 都能取 select_l = Σ_路径 z_l），一并删去。

    拆出重建部分的 hub（`rebuilt_share` 里有的厂，`constraints_rebuilt.add_rebuilt_split`）把各档的 z 再按重建热耗类拆出重建部分
    rz_c：rz_{c,l} >= 0，Σ_c rz_{c,l} <= z_l，Σ_l rz_{c,l} = 该路径该类的重建份额；用量、减排、惩罚各项按同式逐类加一遍重建部分，
    系数换成该类与未重建部分之差（`YearData.rebuilt_deltas`）。

    掺生物质比例的炉型上限（2026-10-02 起，BECCS 同）：落在高于 `biomass_blend_max_pulverized` 的档位上的份额不超过 hub 的
    CFB 装机份额（`YearData.cfb_share`），落在高于 `biomass_blend_max_cfb` 的档位上的为零；约束加在 `blend_layers_b` 的对应层上。
    档位须在 (0, 1] 内严格升序（分层与炉型上限都按升序取），煤粉炉上限须为正且不高于 CFB 上限，否则报错。
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

    # select_b/select_a（只有整数 hub）：第 0 列 = "不掺烧"，第 l+1 列 = "掺烧档位 l"，独热二元变量（一个 hub 选一个档位）。
    one_hot = not assumptions.hub_decisions_continuous
    select_b = select_a = None
    if one_hot:
        select_b = model.addMVar((plant_count, L_b + 1), vtype=GRB.BINARY, name=f"sel_b{sfx}")
        select_a = model.addMVar((plant_count, L_a + 1), vtype=GRB.BINARY, name=f"sel_a{sfx}")
    blend_level_b = model.addMVar(plant_count, lb=0.0, ub=float(L_b), name=f"blv_b{sfx}")
    blend_level_a = model.addMVar(plant_count, lb=0.0, ub=float(L_a), name=f"blv_a{sfx}")
    biomass_use_gj = model.addMVar(plant_count, lb=0.0, name=f"biomass_use_gj{sfx}")
    ammonia_use_kg = model.addMVar(plant_count, lb=0.0, name=f"ammonia_use_kg{sfx}")

    bio_red_exprs: list[GrbExpr] = []
    beccs_blend_red_exprs: list[GrbExpr] = []
    amm_red_exprs: list[GrbExpr] = []
    bio_penalty_exprs: list[GrbExpr] = []
    bio_penalty_emissions_exprs: list[GrbExpr] = []
    beccs_penalty_captured_exprs: list[GrbExpr] = []
    bio_blend_x_share: list[GrbExpr] = []
    beccs_blend_x_share: list[GrbExpr] = []
    amm_blend_x_share: list[GrbExpr] = []
    rebuilt_blend_x_share: dict[tuple[int, int, int], GrbExpr] = {}
    bio_layers: list[list[GrbExpr]] = []
    amm_layers: list[list[GrbExpr]] = []
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
        rebuilt = rebuilt_share.get(p, {})
        deltas = {c: _blend_coeffs(year_data.rebuilt_deltas[c], p) for c in rebuilt}
        G_bp = year_data.generation_by_pathway[p, :]
        G_bio = float(G_bp[bio_idx])
        G_beccs = float(G_bp[beccs_idx])
        G_amm = float(G_bp[amm_idx])

        # --- 整数 hub：每类路径恰好选一个掺烧档位；选档位 0（不掺烧）时份额为 0 由 Σ_l z = 份额与 z <= select 蕴含 ---
        if one_hot:
            model.addConstr(select_b[p, :].sum() == 1.0, name=f"sel_b_{p}{sfx}")
            model.addConstr(select_a[p, :].sum() == 1.0, name=f"sel_a_{p}{sfx}")

        # --- z 是路径份额里落在该档位的部分；整数 hub 下 z = select[p, l+1] × 份额 ---
        bio = _LevelShares(model, (("bio", "zb", bio_idx), ("beccs", "zbc", beccs_idx)), share, p, sfx, base, deltas, rebuilt)
        bio_use_expr = gp.LinExpr()
        bio_red = gp.LinExpr()
        beccs_blend_red = gp.LinExpr()
        bio_penalty = gp.LinExpr()  # 随掺烧档位变化的能耗惩罚（成本）
        bio_penalty_emissions = gp.LinExpr()  # 同一份惩罚燃料，折为排放的 CO2（Mt）
        # BECCS 下，掺烧惩罚燃料与被捕集的烟气在同一台锅炉里燃烧，所以只有未捕集的份额排放（Fan et al. 2023 SI eq. S42）
        # ——而被捕集的份额是实实在在的吨数，必须经管道输送并封存。
        beccs_penalty_captured = gp.LinExpr()
        for level, beta_b in enumerate(blend_b):
            # 生物质与 BECCS 两条路径共用改造到该档位的容量（`zlvl_b`）。
            select = select_b[p, level + 1] if one_hot else None
            for coeffs, (x_bio, x_beccs) in bio.add_level(level, beta_b, select, "zlvl_b"):
                bio_use_expr += coeffs.heat_rate * beta_b * (G_bio * x_bio + G_beccs * x_beccs) / BIOMASS_FLOW_SCALE
                bio_red += coeffs.emissions_retrofit * beta_b * x_bio
                beccs_blend_red += coeffs.emissions_retrofit * beta_b * x_beccs
                # 能耗惩罚：β_b × coeff × (G_bio·z_bio + G_beccs·z_beccs)
                bio_penalty += beta_b * coeffs.penalty * (G_bio * x_bio + G_beccs * x_beccs)
                bio_penalty_emissions += beta_b * (
                    coeffs.penalty_emissions * G_bio * x_bio + coeffs.beccs_penalty_emissions * G_beccs * x_beccs
                )
                beccs_penalty_captured += beta_b * coeffs.beccs_penalty_captured * G_beccs * x_beccs
        bio.add_partition(rebuilt_blend_x_share)
        model.addConstr(biomass_use_gj[p] == bio_use_expr, name=f"bu_{p}{sfx}")
        model.addConstr(blend_level_b[p] == bio.in_use, name=f"blv_b_{p}{sfx}")
        # 在用的掺烧能力按档位分层。能力按层分代，闲置的低档能力不能顶替高档。
        bio_layers.append(bio.layers())
        # 炉型上限（2026-10-02 起）：高于煤粉炉上限的档位只有 CFB 能用，高于 CFB 上限的档位谁都不能用。
        if above_pc is not None:
            model.addConstr(bio_layers[p][above_pc] <= float(year_data.cfb_share[p]), name=f"cfb_b_{p}{sfx}")
        if above_cfb is not None:
            model.addConstr(bio_layers[p][above_cfb] <= 0.0, name=f"cfb_max_b_{p}{sfx}")

        amm = _LevelShares(model, (("amm", "za", amm_idx),), share, p, sfx, base, deltas, rebuilt)
        amm_use_expr = gp.LinExpr()
        amm_red = gp.LinExpr()
        for level, beta_a in enumerate(blend_a):
            select = select_a[p, level + 1] if one_hot else None
            for coeffs, (x_amm,) in amm.add_level(level, beta_a, select, "zlvl_a"):
                amm_use_expr += G_amm * coeffs.heat_rate / lhv * beta_a * x_amm / AMMONIA_FLOW_SCALE
                amm_red += coeffs.emissions_retrofit * beta_a * x_amm
        amm.add_partition(rebuilt_blend_x_share)
        model.addConstr(ammonia_use_kg[p] == amm_use_expr, name=f"au_{p}{sfx}")
        model.addConstr(blend_level_a[p] == amm.in_use, name=f"blv_a_{p}{sfx}")
        amm_layers.append(amm.layers())

        bio_red_exprs.append(bio_red)
        beccs_blend_red_exprs.append(beccs_blend_red)
        amm_red_exprs.append(amm_red)
        bio_penalty_exprs.append(bio_penalty)
        bio_penalty_emissions_exprs.append(bio_penalty_emissions)
        beccs_penalty_captured_exprs.append(beccs_penalty_captured)
        bio_blend_x_share.append(bio.x_share[0])
        beccs_blend_x_share.append(bio.x_share[1])
        amm_blend_x_share.append(amm.x_share[0])

    return BlendTerms(
        blend_level_b=blend_level_b, blend_level_a=blend_level_a, blend_layers_b=bio_layers, blend_layers_a=amm_layers,
        biomass_use_gj=biomass_use_gj, ammonia_use_kg=ammonia_use_kg,
        bio_red_exprs=bio_red_exprs, beccs_blend_red_exprs=beccs_blend_red_exprs, amm_red_exprs=amm_red_exprs,
        bio_penalty_by_plant=bio_penalty_exprs, bio_penalty_emissions_exprs=bio_penalty_emissions_exprs,
        beccs_penalty_captured_exprs=beccs_penalty_captured_exprs, rebuilt_blend_x_share=rebuilt_blend_x_share,
        biomass_blend_x_share=bio_blend_x_share, beccs_blend_x_share=beccs_blend_x_share,
        ammonia_blend_x_share=amm_blend_x_share,
    )
