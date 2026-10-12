"""`plants.csv` 与 `plants_unit_hub.csv` 的读入与加工：省名、发电量、基线排放、水强度与计费比、所在流域，
以及按机组到期的逐规划年列（`_with_expiry`）。`prepare_inputs`（`data_prep`）先后调用 `_prepare_plants`、`_with_expiry`。"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import COAL_DESIGN_LIFE_YEARS
from ..paths import ProjectPaths
from .assumptions import OptimizationAssumptions
from .input_tables import read_input_csv
from .scenario import OptimizationScenario

logger = logging.getLogger(__name__)


def _prepare_plants(paths: ProjectPaths, scenario: OptimizationScenario, assumptions: OptimizationAssumptions) -> pd.DataFrame:
    plants = pd.read_csv(paths.inputs_dir / "plants.csv")
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
    # `builders/plants_water.py` 依据 Wang (2023) 表为每个厂址写出四个按装机容量加权的强度：
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
        from ..builders.water_grid import _assign_basin_codes

        located = plants.rename(columns={"centroid_latitude": "latitude",
                                         "centroid_longitude": "longitude"})
        plants["basin_code"] = _assign_basin_codes(paths, located)
    return plants


_UNIT_HUB_COLUMNS = ["plant_id", "capacity_mw", "commission_year", "combustion", "cooling_technology"]
_UNIT_HUB_HINT = "rebuild plants.csv and plants_unit_hub.csv together with scripts/build_plant_inputs.py --hubs"


def _read_unit_hub_map(paths: ProjectPaths, plants: pd.DataFrame) -> pd.DataFrame:
    """机组到 hub 的映射（`plants_unit_hub.csv`）；核对它与 `plants.csv` 出自同一次聚类：hub 相同、各 hub 装机相等。"""
    path = paths.inputs_dir / "plants_unit_hub.csv"
    units = read_input_csv(path, _UNIT_HUB_COLUMNS, not_found=f"; {_UNIT_HUB_HINT}.")[_UNIT_HUB_COLUMNS]
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
    重建热耗的装机加权平均（`constraints_rebuilt.add_rebuilt_split` 只拆两类以上的这种 hub，拆了与它无关）。逐台毛热耗与
    hub 毛热耗同一算法（`builders.plants.unit_heat_rate_gj_per_mwh`）。按机组细化的 `4d5ce30`（2026-10-02）里重建部分
    只有一类，取到期机组重建热耗的装机加权平均；再之前全国一个热耗 8.5714，整个 hub 自 `retirement_year` 起乘
    0.42 / rebuild_efficiency，不论到期装机是重建还是退役。
    """
    from ..builders.plants import is_cfb, unit_heat_rate_gj_per_mwh, unit_rebuild_heat_rate_cap_gj_per_mwh

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
    # CFB 装机份额：机型标签按 `/` 分段后有一段是 cfb（本体机型为 CFB，或厂名写明 CFB 的 `/CFB` 标记，2026-10-07 起）。
    out["cfb_share"] = by_plant(capacity.where(is_cfb(units["combustion"]), 0.0)) / total
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
