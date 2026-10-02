"""煤电厂侧结果表：路径份额（逐厂 x 路径）、省级汇总、逐厂明细、逐厂成本分解。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import PATHWAY_INDEX, PreparedInputs
from .scenario import PATHWAYS, OptimizationScenario
from .year_types import YearData

# 路径份额不超过它时，有效掺烧比例记 0：份额在求解器可行性容差（1e-6）量级时，Σβ·z / 份额只是噪声。
_SHARE_EPS = 1e-6


def _blend_ratios(
    share_values: np.ndarray,
    biomass_blend_x_share: np.ndarray,
    beccs_blend_x_share: np.ndarray,
    ammonia_blend_x_share: np.ndarray,
    *,
    biomass_levels: tuple[float, ...],
    ammonia_levels: tuple[float, ...],
) -> dict[str, np.ndarray]:
    """逐厂有效掺烧比例 = Σβ_l·z_l / 路径份额，键 biomass、beccs、ammonia。

    连续 hub 下一个 hub 可以把不同份额改造到不同档位；`blend_level`（在用的掺烧能力 Σ 档位下标 x 落在该档的份额）
    只是档位下标的加权和：非整数时对应不到任何一档，恰为整数时也可能是几档的混合（一半第 1 档、
    一半第 3 档记作 2）。模型的减排、燃料用量与惩罚都按 Σβ_l·z_l 计（`constraints._add_blend_level_constraints`），
    这里除以同一条路径的份额。生物质与 BECCS 共用档位容量与生物质档位表，但各自的 z 不同，比例分开算。
    结果截到 [0, 最高档]：份额只比 `_SHARE_EPS` 略大时，分子分母都在求解器容差量级，商可能越出档位范围。
    """
    shares = np.asarray(share_values, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for pathway, blend_x_share, levels in (
        ("biomass", biomass_blend_x_share, biomass_levels),
        ("beccs", beccs_blend_x_share, biomass_levels),
        ("ammonia", ammonia_blend_x_share, ammonia_levels),
    ):
        share = shares[:, PATHWAY_INDEX[pathway]]
        ratio = np.divide(
            np.asarray(blend_x_share, dtype=np.float64), share,
            out=np.zeros(len(share)), where=share > _SHARE_EPS,
        )
        out[pathway] = np.clip(ratio, 0.0, max(levels))
    return out


def _pathway_split(
    scenario: OptimizationScenario,
    year_data: YearData,
    share_values: np.ndarray,
    air_share: np.ndarray,
    biomass_blend_x_share: np.ndarray,
    beccs_blend_x_share: np.ndarray,
    ammonia_blend_x_share: np.ndarray,
    rebuilt_share: np.ndarray,
    rebuilt_blend_x_share: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """逐厂 x 路径的残余排放与物理捕集量（Mt），返回 (residual, captured)。

    逐项照搬 `constraints._add_plant_path_constraints` 的 `residual_expr` 与 `captured_expr`：每一项记到它所乘的
    份额（路径份额、该路径的 Σβ·z 或空冷份额）所在的路径上，按路径相加就是求解器的逐厂残余排放与捕集量。
    各类重建部分（`rebuilt_share`、`rebuilt_blend_x_share`，求解器的同名值，形状 (类数, plant_count, len(PATHWAYS))）
    按同式逐类再加一遍，系数换成该类与未重建部分之差；没有拆出重建部分的 hub 时两者全为零。
    """
    gen = np.asarray(year_data.generation_by_pathway, dtype=np.float64)
    eta = float(scenario.capture_rate)
    un, ccs, bio, beccs, amm = (PATHWAY_INDEX[k] for k in ("unabated", "ccs", "biomass", "beccs", "ammonia"))

    def split(coeffs, s: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按份额 s 与掺烧三列的 Σβz（xs，形状同 s）计的各项，系数取 `YearData` 或其 `rebuilt_deltas` 的一类。"""
        e_op = np.asarray(coeffs.emissions_operating_mt, dtype=np.float64)
        e_rt = np.asarray(coeffs.emissions_retrofit_mt, dtype=np.float64)
        # CCS 能耗惩罚燃料按路径份额计，捕集路径上按 1 − η 排放、按 η 捕集。
        residual = coeffs.ccs_penalty_emissions_matrix * s
        captured = coeffs.ccs_penalty_captured_matrix * s
        residual[:, un] += e_op * s[:, un]
        residual[:, ccs] += e_rt * (1.0 - eta) * s[:, ccs]
        # 掺烧替代的 E_rt·Σβz 不排放；掺烧惩罚燃料照样排放，BECCS 上只排 1 − η，η 被捕集。
        residual[:, bio] += (
            e_rt * (s[:, bio] - xs[:, bio]) + coeffs.biomass_penalty_emissions_coeff_per_level * gen[:, bio] * xs[:, bio]
        )
        residual[:, beccs] += (
            e_rt * ((1.0 - eta) * s[:, beccs] - xs[:, beccs])
            + coeffs.beccs_penalty_emissions_coeff_per_level * gen[:, beccs] * xs[:, beccs]
        )
        residual[:, amm] += e_rt * (s[:, amm] - xs[:, amm])
        captured[:, ccs] += e_rt * eta * s[:, ccs]
        captured[:, beccs] += (
            e_rt * eta * s[:, beccs] + coeffs.beccs_penalty_captured_coeff_per_level * gen[:, beccs] * xs[:, beccs]
        )
        return residual, captured

    s = np.asarray(share_values, dtype=np.float64)
    xs = np.zeros_like(s)
    xs[:, bio], xs[:, beccs], xs[:, amm] = biomass_blend_x_share, beccs_blend_x_share, ammonia_blend_x_share
    residual, captured = split(year_data, s, xs)
    # 空冷背压燃料按空冷份额计，按 hub 毛热耗、不随重建变。
    air = np.asarray(air_share, dtype=np.float64)
    residual += year_data.air_penalty_emissions_matrix * air
    captured += year_data.air_penalty_captured_matrix * air
    rebuilt = np.asarray(rebuilt_share, dtype=np.float64)
    rebuilt_xs = np.asarray(rebuilt_blend_x_share, dtype=np.float64)
    for c, delta in enumerate(year_data.rebuilt_deltas):
        if rebuilt[c].any():
            rebuilt_residual, rebuilt_captured = split(delta, rebuilt[c], rebuilt_xs[c])
            residual += rebuilt_residual
            captured += rebuilt_captured
    return residual, captured


def _build_pathway_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    year: int,
    share_values: np.ndarray,
    biomass_blend_x_share: np.ndarray,
    beccs_blend_x_share: np.ndarray,
    ammonia_blend_x_share: np.ndarray,
    year_data: YearData,
    air_share: np.ndarray,
    *,
    rebuilt_share: np.ndarray,
    rebuilt_blend_x_share: np.ndarray,
) -> pd.DataFrame:
    """单年的逐厂 x 路径份额、发电量、基线排放、减排量与捕集量。

    发电量与基线排放取 `year_data` 里该年的值（已套用利用小时轨迹），按份额分到各路径。`abatement_mt` 是
    基线排放 × 份额 − 该路径的残余排放，`captured_mt` 是该路径的物理捕集量，都按约束逐项拆分（`_pathway_split`）：
    按路径相加就是求解器的逐厂减排量与捕集量（`sanity_checks.csv` 的 `pathway_split_closure` 行核对）。
    改造路径的发电量带 CF 提升（`retrofit_cf_boost`），低比例掺烧的减排可以为负，就是这条路径净增排；未改造一栏也不一定
    为零：部分到期 hub 的未到期机组与原址重建机组按各自的毛热耗排放，比 hub 毛热耗少排的记为正、多排的记为负，空冷背压
    多排的记为负。此前按经典减排比例（η、β，不含 CF 提升与惩罚燃料）
    拆分、再缩放到求解器的逐厂合计：合计对，各路径的值与符号可能不对；捕集量按 CCS 与 BECCS 的份额比例分摊。
    """
    gen_year = np.asarray(year_data.generation, dtype=np.float64)
    em_year = np.asarray(year_data.emissions_mt, dtype=np.float64)
    residual, captured = _pathway_split(
        scenario, year_data, share_values, air_share,
        biomass_blend_x_share, beccs_blend_x_share, ammonia_blend_x_share,
        rebuilt_share, rebuilt_blend_x_share,
    )
    rows: list[dict[str, object]] = []
    for plant_idx, plant in enumerate(prepared.plants.itertuples(index=False)):
        for path_idx, pathway in enumerate(PATHWAYS):
            share = float(share_values[plant_idx, path_idx])
            baseline_mt = float(em_year[plant_idx]) * share
            rows.append(
                {
                    "year": year,
                    "plant_id": plant.plant_id,
                    "province_name": plant.province_name,
                    "pathway": pathway,
                    "share": share,
                    "annual_generation_mwh": float(gen_year[plant_idx]) * share,
                    "baseline_emissions_mt": baseline_mt,
                    "abatement_mt": baseline_mt - float(residual[plant_idx, path_idx]),
                    "captured_mt": float(captured[plant_idx, path_idx]),
                    "enabled": scenario.path_enabled(pathway),
                }
            )
    return pd.DataFrame(rows)


def _build_province_table(pathways: pd.DataFrame) -> pd.DataFrame:
    grouped = pathways.groupby(["year", "province_name", "pathway"], as_index=False)[
        ["annual_generation_mwh", "baseline_emissions_mt", "abatement_mt", "captured_mt"]
    ].sum()
    totals = grouped.groupby(["year", "province_name"], as_index=False)["annual_generation_mwh"].sum().rename(
        columns={"annual_generation_mwh": "province_generation_mwh"}
    )
    merged = grouped.merge(totals, on=["year", "province_name"], how="left")
    merged["generation_share"] = np.where(
        merged["province_generation_mwh"] > 0,
        merged["annual_generation_mwh"] / merged["province_generation_mwh"],
        0.0,
    )
    return merged


def _build_plant_detail_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    year: int,
    share_values: np.ndarray,
    captured_mt: np.ndarray,
    biomass_use_gj: np.ndarray,
    ammonia_use_kg: np.ndarray,
    water_use_m3: np.ndarray,
    blend_level_b: np.ndarray,
    blend_level_a: np.ndarray,
    air_share: np.ndarray,
    year_data: YearData,
    plant_reduction_mt: np.ndarray,
    *,
    biomass_blend_x_share: np.ndarray,
    beccs_blend_x_share: np.ndarray,
    ammonia_blend_x_share: np.ndarray,
    air_installed: np.ndarray,
) -> pd.DataFrame:
    """逐厂高分辨率明细：路径份额、资源用量、空冷份额、掺烧档位与有效掺烧比例、与封存汇的邻近程度。"""
    plants = prepared.plants
    operating = [k for k, pathway in enumerate(PATHWAYS) if pathway != "retire"]
    gen_year = np.asarray(year_data.generation, dtype=np.float64)
    em_year = np.asarray(year_data.emissions_mt, dtype=np.float64)
    ratios = _blend_ratios(
        share_values, biomass_blend_x_share, beccs_blend_x_share, ammonia_blend_x_share,
        biomass_levels=scenario.biomass_blend_levels, ammonia_levels=scenario.ammonia_blend_levels,
    )

    # 预先计算各厂经管网边到封存汇的最小距离
    plant_to_min_storage_km: dict[str, float] = {}
    edges = prepared.network.edges
    storage_node_set = set(prepared.network.storage_node_ids.values())
    for _, edge in edges.iterrows():
        from_id, to_id = str(edge["from_node_id"]), str(edge["to_node_id"])
        length = float(edge["length_km"])
        # 检查边的任一端是否为封存节点
        for plant_id, node_id in prepared.network.plant_node_ids.items():
            if node_id == from_id and to_id in storage_node_set:
                plant_to_min_storage_km[plant_id] = min(plant_to_min_storage_km.get(plant_id, 1e9), length)
            elif node_id == to_id and from_id in storage_node_set:
                plant_to_min_storage_km[plant_id] = min(plant_to_min_storage_km.get(plant_id, 1e9), length)

    rows: list[dict[str, object]] = []
    for p in range(len(plants)):
        plant = plants.iloc[p]
        pid = str(plant["plant_id"])
        dominant_path_idx = int(np.argmax(share_values[p, :]))
        rows.append({
            "year": year,
            "plant_id": pid,
            "province_name": plant["province_name"],
            "capacity_mw": float(plant["total_capacity_mw"]),
            "centroid_longitude": float(plant["centroid_longitude"]),
            "centroid_latitude": float(plant["centroid_latitude"]),
            "retirement_year": int(plant["retirement_year"]),
            "dominant_cooling": str(plant.get("dominant_cooling_technology", "")),
            "annual_generation_mwh": float(gen_year[p]),
            "baseline_emissions_mt": float(em_year[p]),
            "reduction_mt": float(plant_reduction_mt[p]),
            # 路径份额
            "share_unabated": float(share_values[p, PATHWAY_INDEX["unabated"]]),
            "share_retire": float(share_values[p, PATHWAY_INDEX["retire"]]),
            "share_ccs": float(share_values[p, PATHWAY_INDEX["ccs"]]),
            "share_biomass": float(share_values[p, PATHWAY_INDEX["biomass"]]),
            "share_beccs": float(share_values[p, PATHWAY_INDEX["beccs"]]),
            "share_ammonia": float(share_values[p, PATHWAY_INDEX["ammonia"]]),
            "dominant_pathway": PATHWAYS[dominant_path_idx],
            # 资源消耗
            "captured_mt": float(captured_mt[p]),
            "biomass_use_gj": float(biomass_use_gj[p]),
            "ammonia_use_kg": float(ammonia_use_kg[p]),
            "water_use_m3": float(water_use_m3[p]),
            # 空冷两列都是仍湿冷那部分的转换进度，乘 1 − already_air_share 才是全厂份额；该改造未启用或不值其 capex 时为零。
            # 已全空冷的 hub capex 与背压惩罚系数为 0，两列不保证为零，乘上式后为零。
            # air_operating_share：当年在运行路径上以空冷运行的份额（2026-10-02 起退役列的空冷份额恒为零）。
            # air_installed_share：在役的空冷能力，各代新建之和（capex 计在新建上），不小于运行份额、可含闲置的；
            # 2026-10-02 起建成 `air_retrofit_lifetime_years` 年后退出（`vintage`），此前只增不减、含此后退役的容量。
            "air_operating_share": float(air_share[p, operating].sum()),
            "air_installed_share": float(air_installed[p]),
            "already_air_share": float(plant.get("already_air_share", 0.0)),
            # 在用的掺烧能力：各档位层之和 Σ (档位下标) x 落在该档的份额（生物质列含 BECCS）；capex 按层分代计，
            # 生物质的掺烧运维按它计。
            # 独热档位下是所选档位 x 路径份额；连续 hub 下只是加权下标，不能换算成比例。2026-10-02 前是 Σ 档位下标 x 改造到该档的容量份额。
            "biomass_blend_level": float(blend_level_b[p]),
            "ammonia_blend_level": float(blend_level_a[p]),
            # 有效掺烧比例：Σβ_l·z_l / 该路径份额（`_blend_ratios`），份额不超过 `_SHARE_EPS` 时记 0，
            # 其余截到 [0, 该路径最高档]。
            "biomass_blend_ratio": float(ratios["biomass"][p]),
            "beccs_blend_ratio": float(ratios["beccs"][p]),
            "ammonia_blend_ratio": float(ratios["ammonia"][p]),
            # 空间信息
            "min_distance_to_storage_km": plant_to_min_storage_km.get(pid, float("nan")),
        })
    return pd.DataFrame(rows)


def _build_plant_cost_table(
    prepared: PreparedInputs,
    year: int,
    year_data: YearData,
    share_values: np.ndarray,
    biomass_use_gj: np.ndarray,
    *,
    plant_reduction_mt: np.ndarray,
    retrofit_new: np.ndarray,
    ccs_om_by_plant: np.ndarray,
    stranded_by_plant: np.ndarray,
    capex_pathway_indices: tuple[int, ...],
    rebuilt_share: np.ndarray,
    air_share: np.ndarray,
    rebuilt_air_share: np.ndarray,
    bio_penalty_by_plant: np.ndarray,
    blend_level_b: np.ndarray,
) -> pd.DataFrame:
    """逐厂成本分解：由求解得到的变量值计算。

    未折现的逐年口径。一次性 CAPEX 列与模型一致：搁浅资产取求解器的逐厂值 `stranded_by_plant`（计在新增提前退役上，
    `retirement.retirement_flows`），CCS 改造 CAPEX 计在本年新建的捕集岛 `retrofit_new` 上，并含学习
    曲线成本系数。捕集岛固定运维取求解器按在用的各代与建设年单价算的 `ccs_om_by_plant`（`vintage`）。增量运维含生物质
    掺烧能力的固定运维：求解器的在用掺烧能力 `blend_level_b` x 每单位运维（2026-10-02 起）。
    碳成本与目标函数同式，用求解器的逐厂减排量 `plant_reduction_mt`。基线净成本、CCS 能耗惩罚与空冷背压燃料含各类
    重建部分的差（`rebuilt_share`、`rebuilt_air_share`，求解器的同名值，形状 (类数, plant_count, len(PATHWAYS))；
    没有拆出重建部分的 hub 时全为零）。

    能耗惩罚分三列：`energy_penalty_cny` 是 CCS 与 BECCS 的额外燃料；`air_penalty_cny` 是空冷背压、
    `biomass_penalty_cny` 是生物质掺烧效率损失多烧的煤（2026-10-02 起另列，后者取求解器的逐厂值
    `bio_penalty_by_plant`）。三列逐厂相加即目标函数的 `energy_penalty_cost`（未折现）。
    `total_plant_cost_cny` 的口径不变，不含后两列。
    """
    plants = prepared.plants
    n = len(plants)
    emissions = np.asarray(year_data.emissions_mt, dtype=np.float64)
    capacity_mw = plants["total_capacity_mw"].astype(float).to_numpy()
    carbon_price = float(year_data.carbon_price)
    # 本年建成的捕集岛的系数（只有一列，见 `model_year._add_retrofit_new`）。
    stock_coeff = year_data.retrofit_stock_capex
    air_cost_on = year_data.air_penalty_cost_matrix is not None and bool(year_data.allow_air_cooling_retrofit)

    rows: list[dict[str, object]] = []
    for p in range(n):
        share = share_values[p]
        e = float(emissions[p])
        cap = float(capacity_mw[p])

        # 基线净成本：逐路径（燃料 + 运维 - 电）矩阵行 × 份额
        # （改造列含 CF 提升，退役列为零）
        baseline_net = sum(
            float(year_data.baseline_net_matrix[p, k]) * float(share[k])
            for k in range(len(PATHWAYS))
        )
        # 能耗惩罚
        energy_pen = sum(float(year_data.energy_penalty_matrix[p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        for delta, rebuilt in zip(year_data.rebuilt_deltas, rebuilt_share[:, p], strict=True):
            baseline_net += float(delta.baseline_net_matrix[p] @ rebuilt)
            energy_pen += float(delta.energy_penalty_matrix[p] @ rebuilt)
        # 碳成本：碳价 × (基线排放 − 求解器逐厂减排量)，与目标函数同式（`model_costs._operating_costs`）；
        # 减排量取约束本身的表达式，含效率比、CF 提升、全部惩罚燃料与连续 hub 下的掺烧份额。
        # 此前是近似式：未减排部分按基线排放计、不含惩罚燃料，掺烧比例按档位换算（连续 hub 下换算错）。
        carbon_cost = carbon_price * 1e6 * (e - float(plant_reduction_mt[p])) if carbon_price > 0 else 0.0
        # 节煤（coal_savings_per_gj 已缩放为 CNY/PJ；biomass_use_gj 是未缩放的 GJ）
        _bio_scale = float(year_data.biomass_flow_scale)
        _cspg = year_data.coal_savings_per_gj
        _cspg_val = float(_cspg[p]) if hasattr(_cspg, '__getitem__') and not isinstance(_cspg, (int, float)) else float(_cspg)
        coal_savings = (_cspg_val / _bio_scale) * float(biomass_use_gj[p])
        # 增量运维：每 MWh 附加项 + 生物质掺烧能力的固定运维（在用的掺烧能力，与目标函数同式）
        incr_om = sum(float(year_data.fixed_cost_matrix[p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        incr_om += float(year_data.biomass_blend_om_per_level[p]) * float(blend_level_b[p])
        # 捕集岛固定运维：在役且在用的捕集岛 x 建设年单价，取求解器的值
        ccs_om = float(ccs_om_by_plant[p])
        # 搁浅资产：取求解器的值（新增提前退役 n^o x 每单位的剩余账面价值，`retirement.retirement_flows`）
        stranded = float(stranded_by_plant[p])
        # CCS 改造 CAPEX（与模型一致：本年单价 x 本年新建的捕集岛；coeff 已含学习系数）。
        ccs_capex = sum(
            float(stock_coeff[p, j]) * float(retrofit_new[p, j]) for j in range(len(capex_pathway_indices))
        )
        # 空冷背压多烧的煤，与目标函数同式（`model_costs._operating_costs`）：按空冷份额，重建部分加两部分之差。
        air_pen = float(year_data.air_penalty_cost_matrix[p] @ air_share[p]) if air_cost_on else 0.0
        for delta, rebuilt_air in zip(year_data.rebuilt_deltas, rebuilt_air_share[:, p], strict=True):
            air_pen += float(delta.air_penalty_cost_matrix[p] @ rebuilt_air)

        rows.append({
            "year": year,
            "plant_id": plants.iloc[p]["plant_id"],
            "province_name": plants.iloc[p]["province_name"],
            "capacity_mw": cap,
            "baseline_net_cost_cny": baseline_net,
            "carbon_cost_cny": carbon_cost,
            "coal_savings_cny": -coal_savings,
            "incremental_om_cny": incr_om,
            "energy_penalty_cny": energy_pen,
            "ccs_om_cny": ccs_om,
            "stranded_capex_cny": stranded,
            "ccs_retrofit_capex_cny": ccs_capex,
            "total_plant_cost_cny": baseline_net + carbon_cost - coal_savings + incr_om + energy_pen + ccs_om + stranded + ccs_capex,
            "air_penalty_cny": air_pen,
            "biomass_penalty_cny": float(bio_penalty_by_plant[p]),
        })
    return pd.DataFrame(rows)
