"""流域取水指标余量里的两项扣出（carve-out）：建模工业与建模煤电在公报年份的现状取水，亿 m3/yr，按流域。

两者都在水资源公报的"工业"项里（`constants_water_quota.BASIN_WITHDRAWAL_2025_1E8_M3` 的注释）：一旦成为决策主体，它们的水会被
计两次——一次留在保留量里（`water_quota.basin_reserved_withdrawal`），一次被模型重用。`water_quota.write_basin_caps` 把这两项从
保留量里刨出、再按达标比例回补（工业项 2026-09-10 起，煤电项 2026-10-16 起，PR #68）。只算公报年（`BULLETIN_YEAR`）已在运的部分：
在建的工业点源（`status == "construction"`，2026-10-17 起不计）与之后投产的机组那一年没有取水，不在公报里，不能从保留量里刨；
它们日后的取水要在余量里挤。
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import cooling_class
from ..constants_water_quota import BULLETIN_YEAR
from ..paths import ProjectPaths

logger = logging.getLogger(__name__)

# `industry_sources.csv` 的 `status` 里不在运的那一类（其余为 operating、operating pre-retirement）。
INDUSTRY_STATUS_UNDER_CONSTRUCTION = "construction"
REBUILD_HINT = "python scripts/build_plant_inputs.py --hubs, then python scripts/build_water_basin_caps.py"
INDUSTRY_REBUILD_HINT = "python scripts/build_industry_inputs.py, then python scripts/build_water_basin_caps.py"


def _read_columns(path: Path, needed: Iterable[str], hint: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Rebuild it: {hint}")
    frame = pd.read_csv(path)
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} lacks {missing}. Rebuild it: {hint}")
    return frame


def modelled_industry_withdrawal(paths: ProjectPaths) -> dict[str, float]:
    """模型做决策的工业点源在公报年份的取水量，亿 m3/yr，按点源所在流域。

    若尚未构建 `inputs/industry_sources.csv`，返回空映射；那是只含煤电的配置，不需要扣出。
    在建点源（`status == "construction"`，当前输入 74 个、3.04 亿 m3）不计：它们公报年还没有取水（2026-10-17 起；此前全算，
    PR #68 复核发现与煤电侧只计公报年及以前投产机组的口径不对称）。点源按自己的坐标归流域，而求解按枢纽坐标归
    （`optimization.industry_inputs`）：47 个枢纽的点源跨流域，各流域最多差 0.7 亿 m3、全国合计相同，见方法论 §7.3。

    Raises:
        KeyError: 表缺 `status`、`water_m3_per_year` 或经纬度列。
    """
    path = paths.inputs_dir / "industry_sources.csv"
    if not path.exists():
        logger.info("industry_sources.csv absent; no industrial carve-out from the reservation")
        return {}
    from .water_grid import assign_basin_codes

    sources = _read_columns(path, ("status", "water_m3_per_year", "longitude", "latitude"), INDUSTRY_REBUILD_HINT)
    status = sources["status"].astype(str).str.strip().str.lower()
    under_construction = status.eq(INDUSTRY_STATUS_UNDER_CONSTRUCTION)
    if under_construction.any():
        logger.info(
            "industry carve-out leaves out %d sources under construction (%.2f 1e8 m3): not withdrawing in %d",
            int(under_construction.sum()), float(sources.loc[under_construction, "water_m3_per_year"].sum()) / 1e8,
            BULLETIN_YEAR,
        )
    sources = sources.loc[~under_construction].copy()
    sources["basin_code"] = assign_basin_codes(paths, sources)
    by_basin = sources.groupby("basin_code")["water_m3_per_year"].sum() / 1e8
    return {str(code): float(value) for code, value in by_basin.items()}


def _check_units_match_hubs(path: Path, units: pd.DataFrame, plants: pd.DataFrame) -> None:
    """机组表与 hub 表须出自同一次聚类（与 `optimization.data_prep_plants._read_unit_hub_map` 同一核对）：没有空值、装机为正、
    hub 集合相同、各 hub 装机相等。否则扣出会静默偏低（空装机被 groupby 跳过、多余机组被丢、没有机组的 hub 取 0），表照样写出。"""
    needed = ["plant_id", "capacity_mw", "commission_year", "combustion", "cooling_technology"]
    hub_capacity = plants["total_capacity_mw"].astype(float).set_axis(plants["plant_id"].astype(str))
    unit_capacity = units.groupby(units["plant_id"].astype(str))["capacity_mw"].sum()
    if (
        units[needed].isna().any().any()
        or not (units["capacity_mw"].astype(float) > 0).all()
        or set(unit_capacity.index) != set(hub_capacity.index)
        or not np.allclose(unit_capacity.reindex(hub_capacity.index), hub_capacity, rtol=1e-9, atol=1e-6)
    ):
        raise ValueError(
            f"{path} does not match plants.csv (empty values, non-positive capacity, other hubs or other hub capacities). "
            f"Rebuild both: {REBUILD_HINT}"
        )


def modelled_coal_closed_loop_withdrawal(paths: ProjectPaths) -> dict[str, float]:
    """模型里煤电在公报年的闭式循环与空冷取水，亿 m3/yr，按 hub 所在流域。

    这份水在公报的工业剩余项里（`water_quota.basin_reserved_withdrawal`），煤电是模型的决策主体，所以要从预留里扣出、交还给模型，
    与工业点源的扣出（`modelled_industry_withdrawal`）同理。逐机组算（`plants_unit_hub.csv`）：
    只数公报年（`BULLETIN_YEAR`）及以前投产的机组——此后投产的 136.7 GW 不在公报里；公报年当年投产的 94.5 GW 按全年计，其中混有
    GEM 2025 年 7 月快照里计划当年投产的在建机组（机组表没带 status，分不开；这一队列的取水 3.4 亿 m3，西北诸河 0.85，改按半年计
    K 余量少 0.37、不计少 0.75 亿 m3）；只数非直流冷却（直流冷却已随公报的直流火(核)电子列整列扣出，直流机组的补给水等淡水两侧都不计，
    见 `water_quota.once_through_calibration`）；取水强度按 Wang (2023) 表逐台查（`plants_water.unit_withdrawal_intensities`，不带 CCS）；
    发电量按 hub 所在省（`plants.csv` 的 `province_mode` 经 `OptimizationAssumptions.canonical_provinces`、`province_hours`）的现状利用小时，
    与求解侧 `optimization.data_prep_plants._prepare_plants` 同法（2026-10-17 起；PR #68 按机组自己的省查常量表，与求解侧差 0.02%、
    省名别名会报错）；机组按其 hub 的质心归流域，与约束里 hub 的归属一致。

    Raises:
        FileNotFoundError: 两张表缺一张。
        KeyError: 表缺所需列。
        ValueError: 机组表有空值、装机不为正，或与 `plants.csv` 不是同一次聚类（`_check_units_match_hubs`）。
        三种都先 `python scripts/build_plant_inputs.py --hubs`，再重跑 `python scripts/build_water_basin_caps.py`。
    """
    from ..optimization.assumptions import OptimizationAssumptions
    from .plants_water import commissioned_by, unit_withdrawal_intensities
    from .water_grid import assign_basin_codes

    plants = _read_columns(
        paths.inputs_dir / "plants.csv",
        ("plant_id", "province_mode", "total_capacity_mw", "centroid_longitude", "centroid_latitude"), REBUILD_HINT,
    )
    unit_path = paths.inputs_dir / "plants_unit_hub.csv"
    units = _read_columns(unit_path, ("plant_id", "capacity_mw", "commission_year", "combustion", "cooling_technology"), REBUILD_HINT)
    units = units.assign(
        plant_id=units["plant_id"].astype(str), commission_year=pd.to_numeric(units["commission_year"], errors="coerce")
    )
    _check_units_match_hubs(unit_path, units, plants)
    hub_ids = plants["plant_id"].astype(str)
    assumptions = OptimizationAssumptions()
    provinces = assumptions.canonical_provinces(plants["province_mode"], "plants")
    hours_by_hub = pd.Series([assumptions.province_hours(name) for name in provinces], index=hub_ids, dtype=float)
    located = plants.rename(columns={"centroid_latitude": "latitude", "centroid_longitude": "longitude"})
    basin_by_hub = pd.Series(assign_basin_codes(paths, located), index=hub_ids).astype(str)
    in_service = units[commissioned_by(units, BULLETIN_YEAR) & (units["cooling_technology"].map(cooling_class) != "once-through")]
    left_out_mw = float(units.loc[~commissioned_by(units, BULLETIN_YEAR), "capacity_mw"].astype(float).sum())
    if left_out_mw > 0:
        logger.info("coal carve-out leaves out %.0f MW commissioned after %d", left_out_mw, BULLETIN_YEAR)
    withdrawal_m3 = (
        in_service["capacity_mw"].astype(float) * in_service["plant_id"].map(hours_by_hub)
        * unit_withdrawal_intensities(in_service)
    )
    by_basin = withdrawal_m3.groupby(in_service["plant_id"].map(basin_by_hub)).sum() / 1e8
    return {str(code): float(value) for code, value in by_basin.items()}
