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
    """Industrial hubs prepared for optimisation, plus the hydrogen price path."""

    hubs: pd.DataFrame
    # industry_hubs.csv filtered to INDUSTRY_SECTORS, with `basin_code` added.
    h2_price_cny_per_kg: dict[int, float]
    # National supply-weighted LCOH per planning year, REPORTING ONLY since 2026-09-10: the
    # model buys hydrogen per link at the node's own price.
    output_index: dict[tuple[str, int], float] = field(default_factory=dict)
    # {(sector, year): production index, 2030 = 1}; empty holds output flat.


def _national_h2_price(paths: ProjectPaths, usd_to_cny: float) -> dict[int, float]:
    """Supply-weighted mean LCOH per year, CNY/kg, from the repo's own hydrogen supply curve.

    Reported beside the per-link prices so a reader can see how far the marginal price the
    model pays sits from the mean of the whole potential.

    Raises:
        FileNotFoundError: The supply curve has not been built.
        ValueError: The curve lacks the columns this needs.
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
    """Load and prepare the industrial hubs.

    Args:
        paths: Project paths.
        assumptions: `OptimizationAssumptions`; `usd_to_cny` and `water_budget` are read.
        output_index: {(sector, year): index}; None or empty holds output flat.

    Returns:
        Prepared industrial inputs.

    Raises:
        FileNotFoundError: `industry_hubs.csv` has not been built.
        ValueError: A hub carries a sector this module has no parameters for.
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

    # Basin of the hub's own location, matching how `_prepare_plants` attributes coal hubs:
    # the withdrawal permit follows the site, not the intake. Only needed for the basin cap.
    if str(assumptions.water_budget) == "official_quota":
        from ..builders.water import _assign_basin_codes

        # industry_hubs.csv already names the columns `latitude`/`longitude`, which is what
        # `_assign_basin_codes` expects; no rename needed (plants.csv needs one, this does not).
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
    """Candidate (hub, ammonia node) hydrogen links per supply year, within `radius_km`.

    Only hubs whose sector has an H2 route get links. Nodes are the same 0.5-degree green
    ammonia nodes the coal side draws on; `lcoh_usd_per_kg` is the node's plant-gate hydrogen
    cost before Haber-Bosch, which is what an industrial hub taking hydrogen pays.

    Args:
        hubs: Prepared industrial hubs (needs `hub_id`, `sector`, `longitude`, `latitude`).
        ammonia_supply: `ammonia_supply_curve.csv` as loaded by `_prepare_ammonia_supply`.
        radius_km: Matching radius, the coal side's `resource_match_radius_km`.

    Returns:
        Columns `year, hub_id, ammonia_node_id, distance_km, lcoh_usd_per_kg`.
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
