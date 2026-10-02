"""读入 `inputs/` 各表并按情景加工成 `PreparedInputs`：机组、封存、生物质、氨、水、工业、部门目标、管网。

逐年系数矩阵在 `year_matrices.py`，资源/水链路矩阵在 `resource_access.py` / `water_access.py`。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import COAL_DESIGN_LIFE_YEARS
from ..paths import ProjectPaths
from ._shared import PreparedInputs
from .network import build_runtime_network
from .resource_access import _haversine_distances_km
from .scenario import OptimizationAssumptions, OptimizationScenario
from .solver_provenance import _input_digest

logger = logging.getLogger(__name__)


def _prepare_plants(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    plants = pd.read_csv(paths.inputs_dir / "plants.csv").copy()
    # hub 毛热耗（逐台分档煤耗按装机加权，`builders.plants.unit_heat_rate_gj_per_mwh`）：基线排放在下面，
    # 燃料与各项惩罚在 `plant_matrices`。2026-10-02 前全国一个数（8.5714 GJ/MWh），没有这一列。
    if "heat_rate_gj_per_mwh" not in plants.columns or not (plants["heat_rate_gj_per_mwh"].astype(float) > 0).all():
        raise ValueError(
            f"{paths.inputs_dir / 'plants.csv'} lacks positive heat_rate_gj_per_mwh values; "
            "rebuild it with scripts/build_plant_inputs.py --hubs"
        )
    # 省名换成分省煤价表的写法；仍查不到的告警，按缺省煤价与缺省利用小时计。
    plants["province_name"] = assumptions.canonical_provinces(plants["province_mode"], "plants")
    # 发电量按分省利用小时数计算
    plants["province_cf"] = plants["province_name"].map(
        lambda prov: assumptions.province_cf(prov)
    )
    plants["annual_generation_mwh"] = plants["total_capacity_mw"].astype(float) * plants["province_cf"] * 8760.0
    # 按装机容量加权的当前机组利用小时数，即
    # `scenario.operating_hours_scale` 的分母。每一行都存一份，这样 `_build_year_matrices`
    # 读取时无需重算加权。
    capacity = plants["total_capacity_mw"].astype(float)
    plants["fleet_hours_now"] = float(
        (capacity * plants["province_cf"] * 8760.0).sum() / max(float(capacity.sum()), 1e-9)
    )
    plants["baseline_emissions_mt"] = (
        plants["annual_generation_mwh"].astype(float) * plants["heat_rate_gj_per_mwh"].astype(float)
        * assumptions.coal_emission_factor_t_per_gj / 1_000_000.0
    )
    plants["effective_cooling_technology"] = (
        str(scenario.forced_cooling_technology)
        if scenario.forced_cooling_technology
        else plants["dominant_cooling_technology"].astype(str)
    )
    # 必须来自淡水系统的水量。
    # `builders/plants.py` 依据 Wang (2023) 表为每个厂址写出四个按装机容量加权的强度：
    # 逐机组按蒸汽参数（steam cycle）与冷却方式查表，沿海的海水凝汽器已置零。没有这些列
    # 的旧输入退回到按冷却方式取的统一耗水值。
    # 三种口径，三种用途。它们不可互换，模型三种都用：
    #
    #   consumption  流域实际损失的水量               -> 节点可用水量（生态流量）约束
    #   quota        中国计量并收费的水量             -> 水价
    #   withdrawal   全部引水量，含回流到下游的部分   -> 流域用水总量指标约束（开流域上限时）
    #
    # 曾尝试按取水量设生态流量约束，后来放弃：生态流量预留是关于耗减（depletion）的规则，把它
    # 用到直流冷却的凝汽器水量上——这些水在下游几公里处就回到河里——仅因长江流域 45.6% 的
    # 煤电是直流冷却，就让该流域超出限额 155%。真正限制直流冷却取水的是取水口处的瞬时
    # 河道流量，这需要河段尺度的河道演算流量（`dis`），模型里没有。取水只进流域尺度的用水总量
    # 指标约束：那条分配规则本身按取水计量（`water_access._withdrawal_matrices`）。
    base_column = "consumption_intensity_m3_per_mwh"
    ccs_column = "consumption_ccs_intensity_m3_per_mwh"
    has_table = base_column in plants.columns and ccs_column in plants.columns
    if has_table and not scenario.forced_cooling_technology:
        plants["baseline_water_intensity_m3_per_mwh"] = plants[base_column].astype(float)
        plants["capture_water_intensity_m3_per_mwh"] = plants[ccs_column].astype(float)
    else:
        if not has_table:
            logger.warning("plants.csv lacks the per-technology water table; using flat cooling values")
        flat = plants["effective_cooling_technology"].map(assumptions.cooling_baseline_water_intensity)
        plants["baseline_water_intensity_m3_per_mwh"] = flat
        plants["capture_water_intensity_m3_per_mwh"] = flat * assumptions.ccs_water_multiplier
    # 计费比：水价按计量的定额水量征收，但求解器跟踪的变量是耗水量，所以到厂水价要逐厂
    # 乘以 quota/consumption。捕集带来的增量按基准比计费——这是对一个占系统成本 ~5% 的项
    # 所做的 <=25% 的近似。凡没有定额列之处，退回 1.0。
    if "quota_intensity_m3_per_mwh" in plants.columns:
        consumption = plants["baseline_water_intensity_m3_per_mwh"].astype(float)
        ratio = plants["quota_intensity_m3_per_mwh"].astype(float) / consumption.where(consumption > 0)
        plants["water_charge_ratio"] = ratio.fillna(1.0).clip(lower=0.0, upper=20.0)
    else:
        logger.warning("plants.csv lacks quota_intensity_m3_per_mwh; charging water at the consumption volume")
        plants["water_charge_ratio"] = 1.0
    # 取水强度只在有水约束且开流域上限时读（`builders.water_quota.calibrated_withdrawal_intensities`）。
    for column in ("withdrawal_intensity_m3_per_mwh", "withdrawal_ccs_intensity_m3_per_mwh"):
        if column not in plants.columns:
            plants[column] = 0.0
    # hub 自身所在位置的一级流域，用于官方指标上限（有水约束时）。按所在位置归属，而不是按 hub 取水
    # 节点所在的流域归属，因为取水许可就是这样核发的。只在这里算一次：它是一次空间连接
    # （spatial join），若按每个规划年、每个情景重做，耗时会超过其余全部准备工作之和。
    if scenario.water_mode != "no_water":
        from ..builders.water import _assign_basin_codes

        located = plants.rename(columns={"centroid_latitude": "latitude",
                                         "centroid_longitude": "longitude"})
        plants["basin_code"] = _assign_basin_codes(paths, located)
    return plants


_UNIT_HUB_COLUMNS = ["plant_id", "capacity_mw", "commission_year", "combustion", "cooling_technology"]
_UNIT_HUB_HINT = "rebuild plants.csv and plants_unit_hub.csv together with scripts/build_plant_inputs.py --hubs"


def _read_unit_hub_map(paths: ProjectPaths, plants: pd.DataFrame) -> pd.DataFrame:
    """机组到 hub 的映射（`plants_unit_hub.csv`）；核对它与 `plants.csv` 出自同一次聚类：hub 相同、各 hub 装机相等。"""
    path = paths.inputs_dir / "plants_unit_hub.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; {_UNIT_HUB_HINT}.")
    units = pd.read_csv(path)
    missing = set(_UNIT_HUB_COLUMNS) - set(units.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    units = units[_UNIT_HUB_COLUMNS]
    hub_capacity = plants["total_capacity_mw"].astype(float).set_axis(plants["plant_id"].astype(str))
    unit_capacity = units.groupby(units["plant_id"].astype(str))["capacity_mw"].sum()
    if (
        units.isna().any().any()
        or not (units["capacity_mw"] > 0).all()
        or set(unit_capacity.index) != set(hub_capacity.index)
        or not np.allclose(unit_capacity.reindex(hub_capacity.index), hub_capacity, rtol=1e-9, atol=1e-6)
    ):
        raise ValueError(
            f"{path} does not match plants.csv (empty values, other hubs or other hub capacities); {_UNIT_HUB_HINT}."
        )
    return units.assign(plant_id=units["plant_id"].astype(str))


def _with_expiry(
    paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions, plants: pd.DataFrame
) -> pd.DataFrame:
    """加逐规划年的列（`plant_matrices` 用）：到期装机份额 `expired_share_{年}`（f）、剩余账面份额
    `remaining_life_fraction_{年}`、未重建部分的毛热耗 `heat_rate_unexpired_{年}`；重建部分按重建热耗分类 j，
    每类一列不随年变的重建热耗 `heat_rate_rebuilt_c{j}` 与逐年的到期份额 `expired_share_c{j}_{年}`（f_j，按类相加为 f）。
    另加不随年变的 CFB 装机份额 `cfb_share`（掺生物质比例的炉型上限，2026-10-02 起），不随机组到期与重建变。

    按机组计（`plants_unit_hub.csv`）：连续 hub（缺省）逐台在投产年 + 设计寿命到期；整数 hub 整个 hub 在
    `retirement_year`（平均投产年 + 设计寿命）一起到期，f 只取 0 或 1。2026-10-02 前两种模式都按整个 hub 计。
    剩余账面份额按装机加权 min(1, 剩余设计寿命 / 会计寿命)，到期装机为 0，搁浅资产按它计。
    重建机组逐台取 min(机组毛热耗, 3.6 / rebuild_efficiency，空冷机组加空冷的 +15 g/kWh)，不比它替换的那台差
    （`builders.plants.unit_rebuild_heat_rate_cap_gj_per_mwh`）；规划期内到期的机组按这个值（六位小数）分类，
    类按重建热耗升序编号，没有该类的 hub 填 hub 毛热耗、f_j 为 0。哪一类重建由求解器定（`model_year`）。
    未重建部分取 (hub 毛热耗 - f x 到期机组的毛热耗) / (1 - f)；f = 0 时取 hub 毛热耗，f = 1 时没有未重建部分，取到期机组
    重建热耗的装机加权平均（`constraints._add_rebuilt_split` 只拆两类以上的这种 hub，拆了与它无关）。逐台毛热耗与
    hub 毛热耗同一算法（`builders.plants.unit_heat_rate_gj_per_mwh`）。按机组细化的 `4d5ce30`（2026-10-02）里重建部分
    只有一类，取到期机组重建热耗的装机加权平均；再之前全国一个热耗 8.5714，整个 hub 自 `retirement_year` 起乘
    0.42 / rebuild_efficiency，不论到期装机是重建还是退役。
    """
    from ..builders.plants import unit_heat_rate_gj_per_mwh, unit_rebuild_heat_rate_cap_gj_per_mwh

    units = _read_unit_hub_map(paths, plants)
    out = plants.copy()
    plant_ids = out["plant_id"].astype(str)
    by_hub = units["plant_id"]
    capacity = units["capacity_mw"].astype(float)
    if assumptions.hub_decisions_continuous:
        end_year = units["commission_year"].astype(int) + COAL_DESIGN_LIFE_YEARS
    else:
        end_year = by_hub.map(out["retirement_year"].astype(int).set_axis(plant_ids))
    unit_heat_rate = unit_heat_rate_gj_per_mwh(units)
    rebuilt_heat_rate = pd.Series(np.minimum(
        unit_heat_rate, unit_rebuild_heat_rate_cap_gj_per_mwh(units, max(float(scenario.rebuild_efficiency), 1e-9))
    ), index=unit_heat_rate.index)
    hub_heat_rate = out["heat_rate_gj_per_mwh"].astype(float).to_numpy()
    life = max(1, int(assumptions.stranded_asset_accounting_life))

    def by_plant(values: pd.Series) -> np.ndarray:
        return plant_ids.map(values.groupby(by_hub).sum()).to_numpy()

    total = by_plant(capacity)
    # CFB 装机份额：机型标签同 `builders.plants` 归一（去首尾空格、小写，带 `/CCS` 后缀的按本体机型）。
    cfb = units["combustion"].astype(str).str.strip().str.lower().str.split("/").str[0].eq("cfb")
    out["cfb_share"] = by_plant(capacity.where(cfb, 0.0)) / total
    # 部分到期 hub 的未重建部分由 hub 毛热耗反推，误差放大 1 / (1 - f) 倍，甚至为负：这些 hub 的 hub 值须与机组表按装机
    # 加权的值一致（plants.csv 取四位小数），改了分档煤耗等常量而没重建输入时报错。
    stale = np.abs(by_plant(capacity * unit_heat_rate) / total - hub_heat_rate) > 1e-4
    # 重建热耗类：规划期内到期的机组按重建热耗（六位小数）分，hub 内升序编号（dense 名次从 1 起），不到期的机组为 NaN；
    # 类的重建热耗取类内机组按装机加权（同一类的机组取值相同，只差浮点误差）。
    rebuilt_class = (
        rebuilt_heat_rate.round(6).where(end_year <= max(scenario.planning_years)).groupby(by_hub).rank(method="dense")
        - 1.0
    )
    class_count = int(np.nan_to_num(rebuilt_class.max(), nan=0.0)) + 1
    for j in range(class_count):
        class_capacity = by_plant(capacity.where(rebuilt_class == j, 0.0))
        out[f"heat_rate_rebuilt_c{j}"] = np.divide(
            by_plant((capacity * rebuilt_heat_rate).where(rebuilt_class == j, 0.0)), class_capacity,
            out=hub_heat_rate.copy(), where=class_capacity > 0.0,
        )
    for year in scenario.planning_years:
        expired_capacity = capacity.where(end_year <= year, 0.0)
        expired_total = by_plant(expired_capacity)
        # 全部到期时分子分母是同一组数、同序求和，f 恰为 1.0，与整数 hub 加同一条约束（`model_year._add_expiry_rules`）。
        expired = expired_total / total
        some = expired > 0.0
        expired_heat_rate = np.divide(
            by_plant(expired_capacity * unit_heat_rate), expired_total, out=hub_heat_rate.copy(), where=some
        )
        rebuilt = np.divide(
            by_plant(expired_capacity * rebuilt_heat_rate), expired_total, out=hub_heat_rate.copy(), where=some
        )
        unexpired = np.where(expired >= 1.0, rebuilt, hub_heat_rate)
        part = some & (expired < 1.0)
        if np.any(part & stale):
            raise ValueError(
                f"heat_rate_gj_per_mwh in plants.csv differs from the capacity-weighted unit heat rates for "
                f"{int(np.sum(part & stale))} partly expired hubs, e.g. {list(plant_ids[part & stale][:5])}; "
                f"{_UNIT_HUB_HINT}."
            )
        unexpired[part] = (hub_heat_rate[part] - expired[part] * expired_heat_rate[part]) / (1.0 - expired[part])
        book = capacity * np.minimum(1.0, np.maximum(0, end_year - year) / life)
        out[f"expired_share_{year}"] = expired
        out[f"remaining_life_fraction_{year}"] = by_plant(book) / total
        out[f"heat_rate_unexpired_{year}"] = unexpired
        for j in range(class_count):
            out[f"expired_share_c{j}_{year}"] = by_plant(expired_capacity.where(rebuilt_class == j, 0.0)) / total
    return out


def _prepare_basin_caps(paths: ProjectPaths, scenario: OptimizationScenario) -> pd.DataFrame:
    """读入按流域、按规划年的官方用水总量控制指标余量（`water_basin_caps.csv`）；无水约束时返回空表。"""
    if scenario.water_mode == "no_water":
        return pd.DataFrame(columns=["basin_code", "planning_year", "residual_m3_per_year"])
    path = paths.inputs_dir / "water_basin_caps.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The water constraint (water_mode != 'no_water') needs it; run "
            "scripts/build_water_basin_caps.py."
        )
    caps = pd.read_csv(path)
    missing = {"basin_code", "planning_year", "residual_m3_per_year"} - set(caps.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    return caps


_BASIN_USE_COLUMNS = (
    "scenario_id", "planning_year", "basin_code", "domestic_m3_per_year", "irrigation_m3_per_year",
    "dry_season_domestic_m3_per_year", "dry_season_irrigation_m3_per_year",
)


def _prepare_basin_use(paths: ProjectPaths, scenario: OptimizationScenario) -> pd.DataFrame:
    """读入各成员、各规划年、各流域的生活与灌溉耗水（`water_basin_use.csv`），节点余量要扣；无水约束时返回空表。"""
    if scenario.water_mode == "no_water":
        return pd.DataFrame(columns=list(_BASIN_USE_COLUMNS))
    path = paths.inputs_dir / "water_basin_use.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The water constraint (water_mode != 'no_water') needs it; run "
            "scripts/build_water_use.py."
        )
    use = pd.read_csv(path)
    missing = set(_BASIN_USE_COLUMNS) - set(use.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    # 空值求和时会被当成 0（等于不扣），重复键在扣减时对不上：都报错。
    if use[list(_BASIN_USE_COLUMNS[3:])].isna().any().any() or use.duplicated(list(_BASIN_USE_COLUMNS[:3])).any():
        raise ValueError(f"{path} has empty values or duplicate (scenario_id, planning_year, basin_code) rows")
    return use


def _prepare_sector_targets(paths: ProjectPaths, scenario: OptimizationScenario) -> pd.DataFrame:
    """部门残余排放上限，各组自身 2030 基线的比例。"""
    columns = ["sector_group", "planning_year", "cap_fraction_of_2030"]
    path = paths.inputs_dir / f"sector_targets_{scenario.sector_target_source}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found for sector_target_source={scenario.sector_target_source!r}; "
            "run scripts/build_sector_targets.py"
        )
    table = pd.read_csv(path)
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    return table[columns].copy()


def _prepare_output_index(paths: ProjectPaths, scenario: OptimizationScenario) -> dict[tuple[str, int], float]:
    """{(sector, year): 产量指数, 2030 = 1}；空 dict 表示产量保持不变。"""
    source = scenario.effective_output_index_source
    if not source:
        return {}
    path = paths.inputs_dir / f"industry_output_index_{source}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found for industry_output_index_source={source!r}; "
            "run scripts/build_sector_targets.py"
        )
    table = pd.read_csv(path)
    missing = {"sector", "planning_year", "output_index"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    return {
        (str(row.sector), int(row.planning_year)): float(row.output_index)
        for row in table.itertuples(index=False)
    }


def _prepare_storages(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    storages = pd.read_csv(paths.inputs_dir / "storage_hubs.csv").copy()
    if scenario.storage_scope == "dsa_only":
        storages = storages[storages["storage_type"].astype(str) == "dsa"].copy()
    storages["available_capacity_mt"] = storages["storage_all_mt"].astype(float).clip(lower=0.0)
    # 可建注入速率按候选场址密度推算，而不是把栅格逐格的地质速率加总
    # （见 OptimizationAssumptions.storage_site_block_pixels）。
    # 由 5 km 栅格组成的每个 50x50 km 区块算一个项目，按真实项目规模计。栅格加总值仍作为
    # 地质上限施加，使 hub 永远不会超过地层能接受的量。以上全部在乘 injectivity_multiplier
    # 之前定下，这样对它的敏感性（v9 的 `SA_injectivity_half`）依然起作用。
    geological_ceiling = (
        storages["injectivity_dsa_avg_mtpa"].astype(float) + storages["injectivity_eor_avg_mtpa"].astype(float)
    ).clip(lower=0.0)
    if "pixel_count" in storages.columns:
        site_count = storages["pixel_count"].astype(float) / max(1e-9, assumptions.storage_site_block_pixels)
        buildable = (site_count * assumptions.storage_site_project_rate_mtpa).clip(
            lower=0.0, upper=assumptions.max_hub_injectivity_mtpa
        )
    else:
        # 没有栅格像元计数的输入（例如手写的测试夹具）退回到
        # 上限之内的原始速率。
        logger.warning("storage_hubs.csv has no pixel_count column; using raw injectivity under the hub ceiling")
        buildable = geological_ceiling.clip(upper=assumptions.max_hub_injectivity_mtpa)
    storages["injectivity_mtpa"] = (
        np.minimum(buildable, geological_ceiling) * scenario.injectivity_multiplier
    )
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
    biomass = pd.read_csv(paths.inputs_dir / "biomass_supply_curve.csv").copy()
    biomass["available_gj"] = biomass["available_gj"].astype(float) * scenario.biomass_supply_multiplier
    biomass["cost_cny_per_gj"] = biomass["base_cost_cny_per_gj"].astype(float) * scenario.biomass_cost_multiplier

    node_lons = biomass["longitude"].astype(float).to_numpy()
    node_lats = biomass["latitude"].astype(float).to_numpy()
    link_rows = []
    for plant in plants.itertuples(index=False):
        distances_km = _haversine_distances_km(
            float(plant.centroid_longitude), float(plant.centroid_latitude),
            node_lons, node_lats,
        )
        matched = np.flatnonzero(distances_km <= assumptions.resource_match_radius_km)
        for node_idx in matched:
            node = biomass.iloc[int(node_idx)]
            link_rows.append({
                "plant_id": str(plant.plant_id),
                "biomass_node_id": str(node["biomass_node_id"]),
                "distance_km": round(float(distances_km[node_idx]), 3),
                "cost_cny_per_gj": float(node["cost_cny_per_gj"]),
            })
    biomass_links = pd.DataFrame(link_rows) if link_rows else pd.DataFrame(
        columns=["plant_id", "biomass_node_id", "distance_km", "cost_cny_per_gj"]
    )
    logger.info("Biomass links: %d", len(biomass_links))
    return biomass, biomass_links


def _prepare_ammonia_supply(
    paths: ProjectPaths,
    scenario: OptimizationScenario,
    assumptions: OptimizationAssumptions,
    plants: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ammonia = pd.read_csv(paths.inputs_dir / "ammonia_supply_curve.csv").copy()
    ammonia = ammonia.replace([np.inf, -np.inf], np.nan)
    ammonia = ammonia.dropna(subset=["ammonia_node_id", "year", "nh3_supply_kg_per_year", "nh3_cost_lb_usd_per_kg"])
    # 合成岛 capex 年金按情景贴现率重算（CSV 里那一份是构建输入时算的）。
    from ..builders.supply import reprice_hb_capex

    ammonia = reprice_hb_capex(ammonia, float(scenario.discount_rate))
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
        for plant in plants.itertuples(index=False):
            distances_km = _haversine_distances_km(
                float(plant.centroid_longitude), float(plant.centroid_latitude),
                node_lons, node_lats,
            )
            for node_idx in np.flatnonzero(distances_km <= assumptions.resource_match_radius_km):
                node = year_nodes.iloc[int(node_idx)]
                link_rows.append({
                    "year": int(year),
                    "plant_id": str(plant.plant_id),
                    "ammonia_node_id": str(node["ammonia_node_id"]),
                    "distance_km": round(float(distances_km[node_idx]), 3),
                    "cost_cny_per_kg": float(node["cost_cny_per_kg"]),
                })
    ammonia_links = pd.DataFrame(link_rows) if link_rows else pd.DataFrame(
        columns=["year", "plant_id", "ammonia_node_id", "distance_km", "cost_cny_per_kg"]
    )
    logger.info("Ammonia links: %d (across all years)", len(ammonia_links))
    return ammonia.sort_values(["year", "ammonia_node_id"]).reset_index(drop=True), ammonia_links


def _prepare_water(paths: ProjectPaths, plants: pd.DataFrame, assumptions: OptimizationAssumptions) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from ..constants import WATER_MATCH_BUFFER_KM
    water_nodes = pd.read_csv(paths.inputs_dir / "water_nodes.csv").copy()
    water_availability = pd.read_csv(paths.inputs_dir / "water_availability.csv").copy()

    node_lons = water_nodes["longitude"].astype(float).to_numpy()
    node_lats = water_nodes["latitude"].astype(float).to_numpy()
    link_rows = []
    for plant in plants.itertuples(index=False):
        distances_km = _haversine_distances_km(
            float(plant.centroid_longitude), float(plant.centroid_latitude),
            node_lons, node_lats,
        )
        matches = np.flatnonzero(distances_km <= WATER_MATCH_BUFFER_KM)
        sorted_indices = matches[np.argsort(distances_km[matches])]
        for rank, node_idx in enumerate(sorted_indices, 1):
            node = water_nodes.iloc[int(node_idx)]
            dist = float(distances_km[int(node_idx)])
            delivered_cost = assumptions.water_extraction_cost_cny_per_m3 + assumptions.water_transport_cost_cny_per_m3_km * dist
            link_rows.append({
                "plant_id": str(plant.plant_id),
                "water_node_id": str(node["water_node_id"]),
                "distance_km": round(dist, 2),
                "distance_rank": rank,
                "delivered_cost_cny_per_m3": round(delivered_cost, 4),
            })
    water_links = pd.DataFrame(link_rows) if link_rows else pd.DataFrame(
        columns=["plant_id", "water_node_id", "distance_km", "distance_rank", "delivered_cost_cny_per_m3"]
    )
    logger.info("Water links: %d", len(water_links))
    return water_nodes, water_links, water_availability


def _input_files(paths: ProjectPaths, scenario: OptimizationScenario) -> dict[str, Path]:
    """本情景 `prepare_inputs` 实际读取的文件，{逻辑名: 路径}，供溯源摘要用（`solver_provenance._input_digest`）。

    条件与各读取处一致；增删读取处时同步改这里，`tests/test_input_digest.py` 在 toy 上核对两者。
    逻辑名固定，不随情景变（部门目标与产量指数的文件名带来源名，键不带）。
    """
    inputs = paths.inputs_dir
    files = {
        "plants": inputs / "plants.csv",
        "plants_unit_hub": inputs / "plants_unit_hub.csv",  # 机组级到期与两部分毛热耗（`_with_expiry`）
        "storage_hubs": inputs / "storage_hubs.csv",
        "biomass_supply_curve": inputs / "biomass_supply_curve.csv",
        "ammonia_supply_curve": inputs / "ammonia_supply_curve.csv",  # 煤电掺氨与工业氢价共用
        "water_nodes": inputs / "water_nodes.csv",
        "water_availability": inputs / "water_availability.csv",
        "sector_targets": inputs / f"sector_targets_{scenario.sector_target_source}.csv",
        "industry_hubs": inputs / "industry_hubs.csv",
        "pipeline_nodes": inputs / "pipeline_nodes.csv",
        "pipeline_edges": inputs / "pipeline_candidate_edges.csv",
    }
    source = scenario.effective_output_index_source
    if source:
        files["industry_output_index"] = inputs / f"industry_output_index_{source}.csv"
    if scenario.water_mode != "no_water":
        files["water_basin_caps"] = inputs / "water_basin_caps.csv"
        files["water_basin_use"] = inputs / "water_basin_use.csv"
        # 电厂与工业 hub 按厂址归一级流域（`builders.water.load_basins`），读的是 data/ 而不是 inputs/。
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
        paths, plants, storages, scenario, industry_hubs=industry.hubs,
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

