"""煤电厂侧结果表：路径份额（逐厂 x 路径）、省级汇总、逐厂明细、逐厂成本分解。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import PATHWAY_INDEX, PreparedInputs
from .emissions import blend_level_to_ratio, reduction_fraction
from .scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from .year_types import YearData


def _build_pathway_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    year: int,
    share_values: np.ndarray,
    captured_mt_by_plant: np.ndarray,
    blend_level_b: np.ndarray,
    blend_level_a: np.ndarray,
    year_data: YearData,
    plant_reduction_mt: np.ndarray,
) -> pd.DataFrame:
    """单年的逐厂 x 路径份额、发电量、排放与减排量。

    发电量与基线排放取 `year_data` 里该年的值（已套用利用小时轨迹）。`abatement_mt` 锚定到
    求解器自己的逐厂减排量 `plant_reduction_mt`：经典的逐路径减排比例只决定一个厂在各路径之间
    怎么拆分，厂合计则重新缩放到约束中的值，该值含 CF 提升与全部惩罚燃料。2026-09-10 之前
    这一列是未锚定的经典公式，曾出现合计达到基线 111.7% 的情况。
    """
    rows: list[dict[str, object]] = []
    gen_year = np.asarray(year_data.generation, dtype=np.float64)
    em_year = np.asarray(year_data.emissions_mt, dtype=np.float64)
    for plant_idx, plant in enumerate(prepared.plants.itertuples(index=False)):
        baseline_emissions_mt = float(em_year[plant_idx])
        # 求解器给出的实际捕集量
        actual_captured = float(captured_mt_by_plant[plant_idx])
        bio_blend = blend_level_to_ratio(blend_level_b[plant_idx], scenario.biomass_blend_levels)
        amm_blend = blend_level_to_ratio(blend_level_a[plant_idx], scenario.ammonia_blend_levels)
        classic = np.array([
            baseline_emissions_mt
            * reduction_fraction(pathway, scenario.capture_rate, bio_blend, amm_blend)
            * float(share_values[plant_idx, path_idx])
            for path_idx, pathway in enumerate(PATHWAYS)
        ], dtype=np.float64)
        classic_total = float(classic.sum())
        if abs(classic_total) > 1e-9:
            abatement = classic * (float(plant_reduction_mt[plant_idx]) / classic_total)
        else:
            # 按经典口径没有任何减排（全部未改造）：把求解器的值（此时是惩罚燃料修正量）
            # 记到 unabated 列上，使合计正确。
            abatement = np.zeros(len(PATHWAYS))
            abatement[PATHWAY_INDEX["unabated"]] = float(plant_reduction_mt[plant_idx])
        for path_idx, pathway in enumerate(PATHWAYS):
            share = float(share_values[plant_idx, path_idx])
            # 实际捕集量按份额比例分摊到 CCS/BECCS 路径
            if pathway in ("ccs", "beccs"):
                ccs_share = float(share_values[plant_idx, PATHWAY_INDEX["ccs"]])
                beccs_share = float(share_values[plant_idx, PATHWAY_INDEX["beccs"]])
                total_capture_share = ccs_share + beccs_share
                captured_mt = actual_captured * (share / total_capture_share) if total_capture_share > 1e-12 else 0.0
            else:
                captured_mt = 0.0
            rows.append(
                {
                    "year": year,
                    "plant_id": plant.plant_id,
                    "province_name": plant.province_name,
                    "pathway": pathway,
                    "share": share,
                    "annual_generation_mwh": float(gen_year[plant_idx]) * share,
                    "baseline_emissions_mt": baseline_emissions_mt * share,
                    "abatement_mt": float(abatement[path_idx]),
                    "captured_mt": captured_mt,
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
) -> pd.DataFrame:
    """逐厂高分辨率明细：路径份额、资源用量、掺烧档位、与封存汇的邻近程度。"""
    plants = prepared.plants
    gen_year = np.asarray(year_data.generation, dtype=np.float64)
    em_year = np.asarray(year_data.emissions_mt, dtype=np.float64)

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
            # 本 hub 发电量中，转换后以空冷运行的比例。
            # 该改造未启用或不值其 capex 时，各处都为零。
            "air_cooled_share": float(air_share[p, :].sum()),
            "already_air_share": float(plant.get("already_air_share", 0.0)),
            # 掺烧档位
            "biomass_blend_level": float(blend_level_b[p]),
            "ammonia_blend_level": float(blend_level_a[p]),
            # 空间信息
            "min_distance_to_storage_km": plant_to_min_storage_km.get(pid, float("nan")),
        })
    return pd.DataFrame(rows)


def _build_plant_cost_table(
    prepared: PreparedInputs,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    year: int,
    year_data: YearData,
    share_values: np.ndarray,
    captured_mt: np.ndarray,
    biomass_use_gj: np.ndarray,
    water_use_m3: np.ndarray,
    blend_level_b: np.ndarray,
    blend_level_a: np.ndarray,
    prev_share_values: np.ndarray | None = None,
    *,
    retrofit_installed: np.ndarray,
    prev_retrofit_installed: np.ndarray | None = None,
    capex_pathway_indices: tuple[int, ...],
) -> pd.DataFrame:
    """逐厂成本分解：由求解得到的变量值计算。

    未折现的逐年口径。一次性 CAPEX 列与模型一致：搁浅资产计在新增退役份额上，
    CCS 改造 CAPEX 计在已装存量（历史最高份额）的增量上，并含学习曲线成本系数——
    上一年的份额 / 已装值须经 prev_share_values / prev_retrofit_installed 传入
    （首年为 None）。
    """
    plants = prepared.plants
    n = len(plants)
    emissions = np.asarray(year_data.emissions_mt, dtype=np.float64)
    capacity_mw = plants["total_capacity_mw"].astype(float).to_numpy()
    carbon_price = float(year_data.carbon_price)
    retire_idx = PATHWAY_INDEX["retire"]
    # 改造存量的系数（只有捕集岛一列，见 `model_year._add_retrofit_stock`）。
    stock_coeff = year_data.retrofit_stock_capex

    rows: list[dict[str, object]] = []
    for p in range(n):
        share = share_values[p]
        e = float(emissions[p])
        cap = float(capacity_mw[p])
        bio_blend = blend_level_to_ratio(blend_level_b[p], scenario.biomass_blend_levels)
        amm_blend = blend_level_to_ratio(blend_level_a[p], scenario.ammonia_blend_levels)

        # 基线净成本：逐路径（燃料 + 运维 - 电）矩阵行 × 份额
        # （改造列含 CF 提升，退役列为零）
        baseline_net = sum(
            float(year_data.baseline_net_matrix[p, k]) * float(share[k])
            for k in range(len(PATHWAYS))
        )
        # 碳成本：碳价 × 残余排放（近似的报告口径：改造路径用效率 × CF 提升的排放基数，
        # 退役避免的是基线排放）
        e_rt = float(year_data.emissions_retrofit_mt[p])
        reduction_mt = (
            e * float(share[retire_idx])
            + sum(
                e_rt
                * reduction_fraction(pathway, scenario.capture_rate, bio_blend, amm_blend)
                * float(share[k])
                for k, pathway in enumerate(PATHWAYS)
                if k != retire_idx
            )
        )
        residual_mt = e - reduction_mt
        carbon_cost = carbon_price * 1e6 * residual_mt if carbon_price > 0 else 0.0
        # 节煤（coal_savings_per_gj 已缩放为 CNY/TJ；biomass_use_gj 是未缩放的 GJ）
        _bio_scale = float(year_data.biomass_flow_scale)
        _cspg = year_data.coal_savings_per_gj
        _cspg_val = float(_cspg[p]) if hasattr(_cspg, '__getitem__') and not isinstance(_cspg, (int, float)) else float(_cspg)
        coal_savings = (_cspg_val / _bio_scale) * float(biomass_use_gj[p])
        # 增量运维
        incr_om = sum(float(year_data.fixed_cost_matrix[p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # 能耗惩罚
        energy_pen = sum(float(year_data.energy_penalty_matrix[p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # CCS 运维
        ccs_om = sum(float(year_data.ccs_om_matrix[p, k]) * float(share[k]) for k in range(len(PATHWAYS)))
        # 搁浅资产（与模型一致：计在新增退役份额上）
        prev_retire = float(prev_share_values[p, retire_idx]) if prev_share_values is not None else 0.0
        stranded = float(year_data.stranded_per_plant[p]) * max(0.0, float(share[retire_idx]) - prev_retire)
        # CCS 改造 CAPEX（与模型一致：计在已装捕集岛存量的增量上，即 ccs + beccs 的历史最高份额；
        # coeff 已含学习系数）。
        ccs_capex = 0.0
        for j, _k in enumerate(capex_pathway_indices):
            coeff = float(stock_coeff[p, j])
            if coeff <= 0:
                continue
            inst = float(retrofit_installed[p, j])
            prev_inst = float(prev_retrofit_installed[p, j]) if prev_retrofit_installed is not None else 0.0
            ccs_capex += coeff * max(0.0, inst - prev_inst)

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
        })
    return pd.DataFrame(rows)
