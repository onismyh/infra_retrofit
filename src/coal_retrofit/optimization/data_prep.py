"""读入 `inputs/` 各表并按情景加工成 `PreparedInputs`：机组、封存、生物质、氨、水、工业、部门目标、管网。

机组表的加工在 `data_prep_plants.py`，逐年系数矩阵在 `year_matrices.py`，资源/水链路矩阵在 `resource_access.py` /
`water_access.py`。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from ..constants import WATER_MATCH_BUFFER_KM
from ..frames import iter_rows
from ..paths import ProjectPaths
from ..sink_inputs import require_sink_inputs, sink_input_path
from ..spatial import within_radius
from ._shared import PreparedInputs
from .assumptions import OptimizationAssumptions
from .data_prep_plants import _prepare_plants, _with_expiry
from .input_tables import read_input_csv
from .network import build_runtime_network
from .scenario import OptimizationScenario
from .solver_provenance import _input_digest

logger = logging.getLogger(__name__)


def _prepare_basin_caps(paths: ProjectPaths, scenario: OptimizationScenario) -> pd.DataFrame:
    """读入按流域、按规划年的官方用水总量控制指标余量（`water_basin_caps.csv`）；无水约束时返回空表。"""
    if scenario.water_mode == "no_water":
        return pd.DataFrame(columns=["basin_code", "planning_year", "residual_m3_per_year"])
    return read_input_csv(
        paths.inputs_dir / "water_basin_caps.csv", ("basin_code", "planning_year", "residual_m3_per_year"),
        not_found=". The water constraint (water_mode != 'no_water') needs it; run scripts/build_water_basin_caps.py.",
    )


_BASIN_USE_COLUMNS = (
    "scenario_id", "planning_year", "basin_code", "domestic_m3_per_year", "irrigation_m3_per_year",
    "dry_season_domestic_m3_per_year", "dry_season_irrigation_m3_per_year",
)


def _prepare_basin_use(paths: ProjectPaths, scenario: OptimizationScenario) -> pd.DataFrame:
    """读入各成员、各规划年、各流域的生活与灌溉耗水（`water_basin_use.csv`），节点余量要扣；无水约束时返回空表。"""
    if scenario.water_mode == "no_water":
        return pd.DataFrame(columns=list(_BASIN_USE_COLUMNS))
    path = paths.inputs_dir / "water_basin_use.csv"
    use = read_input_csv(
        path, _BASIN_USE_COLUMNS,
        not_found=". The water constraint (water_mode != 'no_water') needs it; run scripts/build_water_use.py.",
    )
    # 空值求和时会被当成 0（等于不扣），重复键在扣减时对不上：都报错。
    if use[list(_BASIN_USE_COLUMNS[3:])].isna().any().any() or use.duplicated(list(_BASIN_USE_COLUMNS[:3])).any():
        raise ValueError(f"{path} has empty values or duplicate (scenario_id, planning_year, basin_code) rows")
    return use


def _prepare_sector_targets(paths: ProjectPaths, scenario: OptimizationScenario) -> pd.DataFrame:
    """部门残余排放上限，各组自身 2030 基线的比例。"""
    columns = ["sector_group", "planning_year", "cap_fraction_of_2030"]
    table = read_input_csv(
        paths.inputs_dir / f"sector_targets_{scenario.sector_target_source}.csv", columns,
        not_found=f" for sector_target_source={scenario.sector_target_source!r}; run scripts/build_sector_targets.py",
    )
    return table[columns].copy()


def _prepare_output_index(paths: ProjectPaths, scenario: OptimizationScenario) -> dict[tuple[str, int], float]:
    """{(sector, year): 产量指数, 2030 = 1}；空 dict 表示产量保持不变。"""
    source = scenario.effective_output_index_source
    if not source:
        return {}
    table = read_input_csv(
        paths.inputs_dir / f"industry_output_index_{source}.csv", ("sector", "planning_year", "output_index"),
        not_found=f" for industry_output_index_source={source!r}; run scripts/build_sector_targets.py",
    )
    return {
        (str(row.sector), int(row.planning_year)): float(row.output_index)
        for row in iter_rows(table)
    }


def _prepare_storages(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    # 汇变体（`scenario.sink_variant`，2026-10-16 起）：缺省读 storage_hubs.csv，距离聚类的变体读带后缀的表（`sink_inputs`）。
    require_sink_inputs(paths.inputs_dir, scenario.sink_variant)
    storages = pd.read_csv(sink_input_path(paths.inputs_dir, "storage_hubs", scenario.sink_variant))
    if scenario.storage_scope == "dsa_only":
        storages = storages[storages["storage_type"].astype(str) == "dsa"].copy()
    storages["available_capacity_mt"] = storages["storage_all_mt"].astype(float).clip(lower=0.0)
    # 累计封存上限与年注入上限都是汇内（建汇时的聚类范围，`builders.storage`）Fan 2025 的逐格之和：前者是封存潜力，后者是注入
    # 能力（DSA 与 EOR 两列相加，再乘 injectivity_multiplier，不随年份变；`model_year._add_injectivity_limit`）。没有其他来源的上限：
    # 2026-10-10 至 10-14 全国合计另受逐年部署上限（ACCA21 区间中点）约束，已删；2026-10-10 前按每 100 格一个 2 Mt/a 项目推算、
    # 单汇封顶 200 Mt/a（`storage_site_block_pixels` 等三个参数，已删），当时逐格之和只作地质上限，各年再乘逐汇的部署进度。
    storages["injectivity_mtpa"] = (
        storages["injectivity_dsa_avg_mtpa"].astype(float) + storages["injectivity_eor_avg_mtpa"].astype(float)
    ).clip(lower=0.0) * scenario.injectivity_multiplier
    # 每吨封存成本 = 基准封存成本 x 海上倍率（`offshore` 为真的汇，2026-10-02 起）− EOR 容量份额 x EOR 抵扣（credit）。
    # 按容量拆分计价，而不是按 `storage_type`，因为只要一个 hub 两者兼有，这个标签就只是
    # 多数表决的结果。不做距离合并建出的汇是纯的，份额非 0 即 1，此式与旧规则完全一致；
    # 在合并情形下，它能防止 7 232 Mt 的 EOR 容量因在表决中被压过而悄悄丢掉抵扣。
    # 扣抵扣之前的单价另存一列，结果工作簿按它与扣后单价之差记 EOR 抵扣（`results_workbook_network.cost_lines`）。
    dsa_mt = storages["storage_dsa_mt"].astype(float).clip(lower=0.0)
    eor_mt = storages["storage_eor_mt"].astype(float).clip(lower=0.0)
    eor_share = (eor_mt / (dsa_mt + eor_mt).replace(0.0, np.nan)).fillna(0.0)
    offshore = storages["offshore"].astype(bool) if "offshore" in storages.columns else False
    storages["storage_cost_before_credit_cny_per_t"] = assumptions.storage_cost_cny_per_t * np.where(
        offshore, assumptions.offshore_storage_multiplier, 1.0
    )
    storages["storage_cost_cny_per_t"] = (
        storages["storage_cost_before_credit_cny_per_t"] - eor_share * assumptions.eor_credit_cny_per_t
    )
    storages = storages.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    storages = storages[(storages["available_capacity_mt"] > 0) | (storages["injectivity_mtpa"] > 0)].reset_index(drop=True)
    return storages


def _prepare_biomass(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions, plants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    biomass = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv")
    biomass["available_gj"] = biomass["available_gj"].astype(float) * scenario.biomass_supply_multiplier
    biomass["cost_cny_per_gj"] = biomass["base_cost_cny_per_gj"].astype(float) * scenario.biomass_cost_multiplier

    node_lons = biomass["longitude"].astype(float).to_numpy()
    node_lats = biomass["latitude"].astype(float).to_numpy()
    link_rows = []
    for plant in iter_rows(plants):
        matched, distances_km = within_radius(
            float(plant.centroid_longitude), float(plant.centroid_latitude),
            node_lons, node_lats, assumptions.resource_match_radius_km,
        )
        for node_idx, distance_km in zip(matched, distances_km, strict=True):
            node = biomass.iloc[int(node_idx)]
            link_rows.append({
                "plant_id": str(plant.plant_id),
                "biomass_node_id": str(node["biomass_node_id"]),
                "distance_km": round(float(distance_km), 3),
                "cost_cny_per_gj": float(node["cost_cny_per_gj"]),
            })
    biomass_links = pd.DataFrame(link_rows, columns=["plant_id", "biomass_node_id", "distance_km", "cost_cny_per_gj"])
    logger.info("Biomass links: %d", len(biomass_links))
    return biomass, biomass_links


def _prepare_ammonia_supply(
    paths: ProjectPaths,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    plants: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ammonia = pd.read_csv(paths.inputs_dir / "ammonia_supply_curve.csv")
    ammonia = ammonia.replace([np.inf, -np.inf], np.nan)
    ammonia = ammonia.dropna(subset=["ammonia_node_id", "year", "h2_supply_kg_per_year", "nh3_cost_lb_usd_per_kg"])
    # 到厂氨价 = 绿氢 x NH3_H2_RATIO + 合成岛电耗 + 合成岛 capex 年金 + 储存（+ 运输）。氢成本按当前折算比例、合成岛年金按
    # 情景贴现率重算（CSV 里那一份是构建输入时算的）。
    from ..builders.supply_ammonia import reprice_h2_component, reprice_hb_capex

    ammonia = reprice_hb_capex(reprice_h2_component(ammonia), float(scenario.discount_rate))
    ammonia["cost_cny_per_kg"] = (
        (ammonia["nh3_cost_lb_usd_per_kg"].astype(float) + scenario.ammonia_transport_adder_usd_per_kg)
        * assumptions.usd_to_cny
        * scenario.ammonia_cost_multiplier
    )

    link_rows = []
    for year, year_nodes in ammonia.groupby("year", sort=True):
        year_nodes = year_nodes.reset_index(drop=True)
        node_lons = year_nodes["longitude"].astype(float).to_numpy()
        node_lats = year_nodes["latitude"].astype(float).to_numpy()
        for plant in iter_rows(plants):
            matched, distances_km = within_radius(
                float(plant.centroid_longitude), float(plant.centroid_latitude),
                node_lons, node_lats, assumptions.resource_match_radius_km,
            )
            for node_idx, distance_km in zip(matched, distances_km, strict=True):
                node = year_nodes.iloc[int(node_idx)]
                link_rows.append({
                    "year": int(cast(int, year)),
                    "plant_id": str(plant.plant_id),
                    "ammonia_node_id": str(node["ammonia_node_id"]),
                    "distance_km": round(float(distance_km), 3),
                    "cost_cny_per_kg": float(node["cost_cny_per_kg"]),
                })
    ammonia_links = pd.DataFrame(
        link_rows, columns=["year", "plant_id", "ammonia_node_id", "distance_km", "cost_cny_per_kg"]
    )
    logger.info("Ammonia links: %d (across all years)", len(ammonia_links))
    return ammonia.sort_values(["year", "ammonia_node_id"]).reset_index(drop=True), ammonia_links


def _prepare_water(paths: ProjectPaths, plants: pd.DataFrame, assumptions: OptimizationAssumptions) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    water_nodes = pd.read_csv(paths.inputs_dir / "water_nodes.csv")
    water_availability = pd.read_csv(paths.inputs_dir / "water_availability.csv")

    node_lons = water_nodes["longitude"].astype(float).to_numpy()
    node_lats = water_nodes["latitude"].astype(float).to_numpy()
    link_rows = []
    for plant in iter_rows(plants):
        # 按距离排名（`distance_rank`），最近的节点另有用处（`water_access`）。
        ranked, distances_km = within_radius(
            float(plant.centroid_longitude), float(plant.centroid_latitude),
            node_lons, node_lats, WATER_MATCH_BUFFER_KM, ranked=True,
        )
        for rank, (node_idx, dist) in enumerate(zip(ranked, distances_km, strict=True), 1):
            node = water_nodes.iloc[int(node_idx)]
            dist = float(dist)
            delivered_cost = assumptions.water_extraction_cost_cny_per_m3 + assumptions.water_transport_cost_cny_per_m3_km * dist
            link_rows.append({
                "plant_id": str(plant.plant_id),
                "water_node_id": str(node["water_node_id"]),
                "distance_km": round(dist, 2),
                "distance_rank": rank,
                "delivered_cost_cny_per_m3": round(delivered_cost, 4),
            })
    water_links = pd.DataFrame(
        link_rows, columns=["plant_id", "water_node_id", "distance_km", "distance_rank", "delivered_cost_cny_per_m3"]
    )
    # 厂侧水平衡（供水流量 == 耗水）在所有情景里都建（`model_resources.add_resource_balances`，无水约束时也建）：
    # 没有一条链路的 hub 耗水被迫为 0，只剩退役一条路径可行，而且不报错。半径内没有水节点的 hub 直接拒绝。
    unlinked = sorted(set(plants["plant_id"].astype(str)) - set(water_links["plant_id"].astype(str)))
    if unlinked:
        more = " …" if len(unlinked) > 6 else ""
        raise ValueError(
            f"{len(unlinked)} 个煤电 hub 在 {WATER_MATCH_BUFFER_KM:g} km 内没有水节点（{', '.join(unlinked[:6])}{more}）："
            "厂侧水平衡会把它们的耗水压成 0、只剩退役可行；核对 water_nodes.csv 与 plants.csv 的坐标，或重建水节点表"
        )
    logger.info("Water links: %d", len(water_links))
    return water_nodes, water_links, water_availability


def _input_files(paths: ProjectPaths, scenario: OptimizationScenario) -> dict[str, Path]:
    """本情景 `prepare_inputs` 实际读取的文件，{逻辑名: 路径}，供溯源摘要用（`solver_provenance._input_digest`）。

    条件与各读取处一致；增删读取处时同步改这里，`tests/test_input_digest.py` 在 toy 上核对两者。
    逻辑名固定，不随情景变（部门目标与产量指数的文件名带来源名、汇与管网的带汇变体，键不带）。
    """
    inputs = paths.inputs_dir
    files = {
        "plants": inputs / "plants.csv",
        "plants_unit_hub": inputs / "plants_unit_hub.csv",  # 机组级到期与两部分毛热耗（`_with_expiry`）
        "storage_hubs": sink_input_path(inputs, "storage_hubs", scenario.sink_variant),
        "biomass_supply_curve": inputs / "biomass_supply_curve.csv",
        "ammonia_supply_curve": inputs / "ammonia_supply_curve.csv",  # 煤电掺氨与工业氢价共用
        "water_nodes": inputs / "water_nodes.csv",
        "water_availability": inputs / "water_availability.csv",
        "sector_targets": inputs / f"sector_targets_{scenario.sector_target_source}.csv",
        "industry_hubs": inputs / "industry_hubs.csv",
        "pipeline_nodes": sink_input_path(inputs, "pipeline_nodes", scenario.sink_variant),
        "pipeline_edges": sink_input_path(inputs, "pipeline_candidate_edges", scenario.sink_variant),
    }
    # 工业 hub 的可捕集份额与份额为正的点源所占比例由点源表算（`industry_inputs._hub_source_shares`）；
    # 没有点源表时（toy 输入）不读。
    sources = inputs / "industry_sources.csv"
    if sources.exists():
        files["industry_sources"] = sources
    source = scenario.effective_output_index_source
    if source:
        files["industry_output_index"] = inputs / f"industry_output_index_{source}.csv"
    if scenario.water_mode != "no_water":
        files["water_basin_caps"] = inputs / "water_basin_caps.csv"
        files["water_basin_use"] = inputs / "water_basin_use.csv"
        # 电厂与工业 hub 按厂址归一级流域（`builders.water_grid.load_basins`），读的是 data/ 而不是 inputs/。
        files["basin_polygons"] = paths.data_dir / "ChinaBasins" / "basin_l1.gpkg"
    return files


def prepare_inputs(
    paths: ProjectPaths,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
) -> PreparedInputs:
    plants = _with_expiry(paths, scenario, assumptions, _prepare_plants(paths, scenario, assumptions))
    storages = _prepare_storages(paths, scenario, assumptions)
    biomass, biomass_links = _prepare_biomass(paths, scenario, assumptions, plants)
    ammonia_supply, ammonia_links = _prepare_ammonia_supply(paths, scenario, assumptions, plants)
    water_nodes, water_links, water_availability = _prepare_water(paths, plants, assumptions)
    water_basin_caps = _prepare_basin_caps(paths, scenario)
    water_basin_use = _prepare_basin_use(paths, scenario)
    sector_targets = _prepare_sector_targets(paths, scenario)
    available_ammonia_years = tuple(sorted(ammonia_supply["year"].astype(int).unique().tolist()))
    from .industry import prepare_industry, prepare_industry_h2_links

    industry = prepare_industry(
        paths, assumptions, output_index=_prepare_output_index(paths, scenario),
        assign_basins=scenario.water_mode != "no_water",
    )
    industry_h2_links = prepare_industry_h2_links(
        industry.hubs, ammonia_supply, float(assumptions.resource_match_radius_km)
    )
    network = build_runtime_network(
        paths, plants, storages, industry_hubs=industry.hubs, sink_variant=scenario.sink_variant,
    )
    return PreparedInputs(
        plants=plants,
        storages=storages,
        biomass=biomass,
        biomass_links=biomass_links,
        ammonia_supply=ammonia_supply,
        ammonia_links=ammonia_links,
        water_nodes=water_nodes,
        water_links=water_links,
        water_availability=water_availability,
        water_basin_caps=water_basin_caps,
        water_basin_use=water_basin_use,
        network=network,
        available_ammonia_years=available_ammonia_years,
        industry=industry,
        sector_targets=sector_targets,
        industry_h2_links=industry_h2_links,
        input_digest=_input_digest(paths.inputs_dir, _input_files(paths, scenario)),
    )

