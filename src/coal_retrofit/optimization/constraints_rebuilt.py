"""原址重建部分的份额：有到期装机的 hub 把各运行路径的份额（与空冷份额）按重建毛热耗类拆出原址重建机组承担的部分
（`add_rebuilt_split`），以及路径系数行乘重建部分份额的点积 `rebuilt_dot`。由 `constraints._add_plant_path_constraints` 调用，
`model_costs` 用 `rebuilt_dot` 计重建部分的成本差。"""
from __future__ import annotations

from ._shared import PATHWAY_INDEX, gp
from .scenario import PATHWAYS
from .year_types import RebuiltShares

# 运行路径（退役之外）的列：有到期装机的 hub 的份额在这些路径上拆出重建部分（`add_rebuilt_split`），
# 空冷改造的在用量是这些路径上空冷份额之和（`model_linking.add_capacity_vintages`）。
RUNNING = tuple(k for k, pathway in enumerate(PATHWAYS) if pathway != "retire")


def add_rebuilt_split(
    model, share, rebuild_class, air_share, expired_share, class_share, plant_count: int, allow_air: bool, sfx: str
) -> tuple[RebuiltShares, RebuiltShares]:
    """有到期装机的 hub 把运行路径的份额拆出原址重建机组承担的部分，按重建毛热耗分类 c（`data_prep_plants._with_expiry`），
    各类与未重建部分的毛热耗不同。

    每条运行路径 k：r_{c,k} >= 0，Σ_c r_{c,k} <= x_k；Σ_k r_{c,k} <= 该类的重建份额 ρ_c（重建机组至多全部运行）；
    Σ_c Σ_k r_{c,k} + 退役份额 >= f（未重建部分 Σ_k (x_k - Σ_c r_{c,k}) 不超过未到期装机 1 - f；f = 1 时各路径全由
    重建部分承担）。空冷份额同样拆：ra_{c,k} <= r_{c,k}、Σ_c ra_{c,k} <= a_k、a_k - Σ_c ra_{c,k} <= x_k - Σ_c r_{c,k}。
    掺烧各档的拆分在 `constraints_blend.add_blend_level_constraints`。哪几台机组运行、退役、改空冷由求解器定，模型不追踪。
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


def rebuilt_dot(row, part: dict[int, gp.Var]):
    """Σ_k row[k] x part[k]：路径系数行（两部分之差）乘重建部分的份额（`add_rebuilt_split`）。"""
    return gp.quicksum(float(row[k]) * var for k, var in part.items())
