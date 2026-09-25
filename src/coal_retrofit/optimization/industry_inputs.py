"""工业点源的输入准备：读 hub 表、全国氢均价（只作报告）、hub 到氨节点的候选氢链路。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..constants_industry import (
    INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR,
    INDUSTRY_SECTORS,
    SECTOR_HAS_H2_ROUTE,
    SECTOR_TARGET_GROUP,
)
from ..paths import ProjectPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndustryInputs:
    """为优化准备好的工业 hub，以及氢价路径。"""

    hubs: pd.DataFrame
    # 按 INDUSTRY_SECTORS 筛选后的 industry_hubs.csv，并加上 `basin_code`。
    h2_price_cny_per_kg: dict[int, float]
    # 每个规划年的全国供给加权 LCOH，自 2026-09-10 起只作报告：
    # 模型按链路、以各节点自己的价格买氢。
    output_index: dict[tuple[str, int], float] = field(default_factory=dict)
    # {(sector, year): 产量指数，2030 = 1}；为空时产量保持不变。


def _national_h2_price(paths: ProjectPaths, usd_to_cny: float) -> dict[int, float]:
    """逐年的供给加权平均 LCOH（CNY/kg），取自本仓库自己的氢供给曲线。

    与逐链路价格并列报告，让读者看出模型实际支付的边际价格离全部潜力的均值有多远。

    Raises:
        FileNotFoundError: 供给曲线还没有构建。
        ValueError: 供给曲线缺少这里需要的列。
    """
    path = paths.inputs_dir / "ammonia_supply_curve.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; industry's H2 route is priced from it. Build it with "
            "scripts/build_ammonia_supply.py, or disable industry."
        )
    curve = pd.read_csv(path, usecols=["year", "h2_supply_kg_per_year", "weighted_lcoh_usd_per_kg_h2"])
    missing = {"year", "h2_supply_kg_per_year", "weighted_lcoh_usd_per_kg_h2"} - set(curve.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    prices: dict[int, float] = {}
    for year, block in curve.groupby(curve["year"].astype(int)):
        weight = block["h2_supply_kg_per_year"].astype(float)
        total = float(weight.sum())
        if total <= 0:
            raise ValueError(f"{path}: zero hydrogen supply in {year}, cannot weight LCOH")
        lcoh_usd = float((block["weighted_lcoh_usd_per_kg_h2"].astype(float) * weight).sum() / total)
        prices[int(year)] = lcoh_usd * float(usd_to_cny)
    return prices


def prepare_industry(
    paths: ProjectPaths, assumptions, output_index: dict[tuple[str, int], float] | None = None
) -> IndustryInputs:
    """读取并准备工业 hub。

    Args:
        paths: 项目路径。
        assumptions: `OptimizationAssumptions`；读取其中的 `usd_to_cny` 与 `water_budget`，
            并用 `canonical_provinces` 把省名换成分省煤价表的写法。
        output_index: {(sector, year): index}；为 None 或为空时产量保持不变。

    Returns:
        准备好的工业输入。

    Raises:
        FileNotFoundError: `industry_hubs.csv` 还没有构建。
        ValueError: 有 hub 所属的行业在本模块里没有参数。
    """
    path = paths.inputs_dir / "industry_hubs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with `builders.industry.write_industry_inputs`."
        )
    hubs = pd.read_csv(path)
    # 西藏不参与减排：备选管网里已经没有西藏点源（`builders.network_branches`
    # 同一张排除表），优化侧必须用同一个点源集合，否则会给一个没有管网接入的点源
    # 派任务。部门碳目标是"各组自身 2030 基线的比例"，基线随点源集合一起缩放，
    # 因此剔除既不放松也不收紧目标。
    from ..builders.network_branches import EXCLUDED_PROVINCES as _EXCLUDED_PROVINCES

    dropped = hubs["province"].astype(str).str.strip().str.lower().isin(_EXCLUDED_PROVINCES)
    if bool(dropped.any()):
        logger.info("industry: dropped %d hub(s) in non-abating provinces", int(dropped.sum()))
        hubs = hubs.loc[~dropped].reset_index(drop=True)
    hubs = hubs[hubs["sector"].astype(str).isin(INDUSTRY_SECTORS)].reset_index(drop=True)
    if hubs.empty:
        raise ValueError(f"{path} has no rows in the in-scope sectors {sorted(INDUSTRY_SECTORS)}")
    unknown = sorted(set(hubs["sector"].astype(str)) - set(INDUSTRY_CCS_CAPEX_CNY_PER_T_CO2_YR))
    if unknown:
        raise ValueError(f"no capture capex sourced for sector(s) {unknown}; refusing to guess")
    hubs["target_group"] = hubs["sector"].astype(str).map(SECTOR_TARGET_GROUP)
    if hubs["target_group"].isna().any():
        raise ValueError("a hub's sector has no entry in SECTOR_TARGET_GROUP")
    # 省名换成分省煤价表的写法；仍查不到的告警，其捕集蒸汽按缺省煤价计。
    hubs["province"] = assumptions.canonical_provinces(hubs["province"], "industry hubs")

    # 按 hub 自身所在位置归流域，与 `_prepare_plants` 对煤电 hub 的归属方式一致：
    # 取水许可跟着厂址走，而不是跟着取水口走。只有流域上限需要它。
    if str(assumptions.water_budget) == "official_quota":
        from ..builders.water import _assign_basin_codes

        # industry_hubs.csv 的列名本来就是 `latitude`/`longitude`，正是 `_assign_basin_codes`
        # 所要的；无需改名（plants.csv 需要改名，这里不需要）。
        hubs["basin_code"] = _assign_basin_codes(paths, hubs)

    prices = _national_h2_price(paths, float(assumptions.usd_to_cny))
    index = dict(output_index or {})
    if index:
        sectors_missing = sorted(
            {str(s) for s in hubs["sector"]} - {sector for sector, _ in index}
        )
        if sectors_missing:
            raise ValueError(f"industry output index has no rows for sector(s) {sectors_missing}")
    logger.info(
        "industry: %d hubs, %.0f Mt CO2/yr baseline, %.1f 亿 m3/yr water; national mean H2 price %s CNY/kg; output index %s",
        len(hubs),
        float(hubs["co2_mt_per_year"].sum()),
        float(hubs["water_m3_per_year"].sum()) / 1e8,
        {y: round(p, 1) for y, p in sorted(prices.items()) if y in (2030, 2060)},
        "on" if index else "flat",
    )
    return IndustryInputs(hubs=hubs, h2_price_cny_per_kg=prices, output_index=index)


def prepare_industry_h2_links(
    hubs: pd.DataFrame, ammonia_supply: pd.DataFrame, radius_km: float
) -> pd.DataFrame:
    """逐供给年列出 `radius_km` 以内的候选氢链路（hub, 氨节点）。

    只有所在行业有氢路线的 hub 才有链路。节点就是煤电侧所用的同一批 0.5 度绿氨节点；
    `lcoh_usd_per_kg` 是节点在 Haber-Bosch 合成之前的出厂氢成本，也就是取氢的工业 hub
    所付的价格。

    Args:
        hubs: 准备好的工业 hub（需要 `hub_id`、`sector`、`longitude`、`latitude`）。
        ammonia_supply: 由 `_prepare_ammonia_supply` 读入的 `ammonia_supply_curve.csv`。
        radius_km: 匹配半径，即煤电侧的 `resource_match_radius_km`。

    Returns:
        列为 `year, hub_id, ammonia_node_id, distance_km, lcoh_usd_per_kg` 的表。
    """
    from .resource_access import _haversine_distances_km

    columns = ["year", "hub_id", "ammonia_node_id", "distance_km", "lcoh_usd_per_kg"]
    eligible = hubs[hubs["sector"].astype(str).map(lambda s: bool(SECTOR_HAS_H2_ROUTE.get(s, False)))]
    if eligible.empty or ammonia_supply.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for year, nodes in ammonia_supply.groupby("year", sort=True):
        nodes = nodes.reset_index(drop=True)
        lons = nodes["longitude"].astype(float).to_numpy()
        lats = nodes["latitude"].astype(float).to_numpy()
        node_ids = nodes["ammonia_node_id"].astype(str).to_numpy()
        lcoh = nodes["weighted_lcoh_usd_per_kg_h2"].astype(float).to_numpy()
        for hub in eligible.itertuples(index=False):
            distances = _haversine_distances_km(float(hub.longitude), float(hub.latitude), lons, lats)
            for node_idx in np.flatnonzero(distances <= radius_km):
                rows.append({
                    "year": int(year),
                    "hub_id": str(hub.hub_id),
                    "ammonia_node_id": node_ids[node_idx],
                    "distance_km": round(float(distances[node_idx]), 3),
                    "lcoh_usd_per_kg": float(lcoh[node_idx]),
                })
    links = pd.DataFrame(rows, columns=columns)
    unreachable = sorted(set(eligible["hub_id"].astype(str)) - set(links["hub_id"].astype(str)))
    if unreachable:
        logger.warning(
            "industry: %d H2-capable hubs have no hydrogen node within %.0f km; their H2 route "
            "is closed by the balance (first few: %s)", len(unreachable), radius_km, unreachable[:5],
        )
    logger.info("Industry H2 links: %d (across all years)", len(links))
    return links
