"""工业结果表：逐 hub 的路线份额、减排、捕集、用水、用氢与成本。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import AMMONIA_FLOW_SCALE
from ..constants_industry import INDUSTRY_ROUTES
from ._shared import PreparedInputs
from .industry import CCS as _CCS, H2 as _H2, UNABATED as _UNABATED
from .year_types import YearData


def _build_industry_detail_table(
    prepared: PreparedInputs,
    year: int,
    industry_year_data: dict | None,
    share_values: np.ndarray | None,
    prev_share_values: np.ndarray | None = None,
    h2_flow_kg: np.ndarray | None = None,
    year_data: YearData | None = None,
) -> pd.DataFrame:
    """One row per industrial hub per year: routes chosen, abatement, capture, water, cost.

    Costs follow the model's own split: `cost_annual_cny` is the fixed O&M, capture energy
    and consumables (H2 route: the non-hydrogen operating delta) plus the hydrogen actually
    bought on the hub's links this year, `cost_capital_cny` the one-time retrofit capex on the
    route-share increment (whole share in the first year). The end-of-horizon salvage credit
    is not attributed per hub; it is the `salvage_credit` row of `cost_breakdown.csv`.

    Args:
        prepared: Prepared inputs; `prepared.industry` carries the hub frame.
        year: Planning year.
        industry_year_data: The year's industrial coefficient block, or None when industry off.
        share_values: Solved route shares, shape (hub_count, len(INDUSTRY_ROUTES)).
        prev_share_values: Previous year's shares (None in the first year).
        h2_flow_kg: Solved hydrogen flow per link, kg.
        year_data: The year's matrices, for the hydrogen link costs and incidence.

    Returns:
        Empty frame with the right columns when industry is off, so downstream readers get a
        frame either way.
    """
    columns = [
        "year", "hub_id", "sector", "target_group", "province", "longitude", "latitude", "basin_code",
        "output_index", "production_kt_per_year", "baseline_co2_mt", "process_co2_mt",
        "share_unabated", "share_ccs", "share_h2",
        "reduction_mt", "residual_mt", "captured_mt", "h2_kg",
        "water_m3", "water_base_m3", "water_capture_increment_m3",
        "cost_cny", "cost_capital_cny", "cost_annual_cny", "cost_h2_purchase_cny",
        "h2_price_paid_cny_per_kg", "h2_price_national_mean_cny_per_kg",
    ]
    if industry_year_data is None or share_values is None:
        return pd.DataFrame(columns=columns)
    hubs = prepared.industry.hubs
    reduction = industry_year_data["reduction_mt"]
    baseline = industry_year_data["baseline_emissions_mt"]
    captured = industry_year_data["captured_mt"]
    water = industry_year_data["water_m3"]
    opex = industry_year_data["opex_cny"]
    capex = industry_year_data["capex_cny"]
    output_scale = industry_year_data.get("output_scale", np.ones(len(hubs)))
    h2_price_mean = float(industry_year_data["h2_price_cny_per_kg"])
    n_hubs = len(hubs)
    # Hydrogen bought per hub: link flows x link costs, folded onto hubs with the incidence.
    h2_kg_by_hub = np.zeros(n_hubs)
    h2_cost_by_hub = np.zeros(n_hubs)
    if (
        h2_flow_kg is not None and len(h2_flow_kg) and year_data is not None
        and year_data.industry_h2_hub_membership is not None
    ):
        incidence = year_data.industry_h2_hub_membership
        flows = np.asarray(h2_flow_kg, dtype=np.float64)
        unit_cost = np.asarray(year_data.industry_h2_link_cost_cny_per_kg, dtype=np.float64) / AMMONIA_FLOW_SCALE
        h2_kg_by_hub = np.asarray(incidence @ flows).ravel()
        h2_cost_by_hub = np.asarray(incidence @ (flows * unit_cost)).ravel()
    rows: list[dict[str, object]] = []
    for hub_idx, hub in enumerate(hubs.itertuples(index=False)):
        share = share_values[hub_idx]
        prev = prev_share_values[hub_idx] if prev_share_values is not None else np.zeros_like(share)
        annual_ccs = float(opex[hub_idx, _CCS] * share[_CCS])
        annual_h2 = max(0.0, float(opex[hub_idx, _H2] * share[_H2]) + float(h2_cost_by_hub[hub_idx]))
        capital = float(
            capex[hub_idx, _CCS] * max(0.0, share[_CCS] - prev[_CCS])
            + capex[hub_idx, _H2] * max(0.0, share[_H2] - prev[_H2])
        )
        base_water = float(water[hub_idx, _UNABATED])
        total_water = float(sum(water[hub_idx, r] * share[r] for r in range(len(INDUSTRY_ROUTES))))
        red = float(sum(reduction[hub_idx, r] * share[r] for r in range(len(INDUSTRY_ROUTES))))
        rows.append({
            "year": year,
            "hub_id": str(hub.hub_id),
            "sector": str(hub.sector),
            "target_group": str(getattr(hub, "target_group", "")),
            "province": str(hub.province),
            "longitude": float(hub.longitude),
            "latitude": float(hub.latitude),
            "basin_code": str(getattr(hub, "basin_code", "")),
            "output_index": float(output_scale[hub_idx]),
            "production_kt_per_year": float(hub.production_kt_per_year) * float(output_scale[hub_idx]),
            "baseline_co2_mt": float(baseline[hub_idx]),
            "process_co2_mt": float(hub.process_co2_mt_per_year) * float(output_scale[hub_idx]),
            "share_unabated": float(share[_UNABATED]),
            "share_ccs": float(share[_CCS]),
            "share_h2": float(share[_H2]),
            "reduction_mt": red,
            "residual_mt": float(baseline[hub_idx]) - red,
            "captured_mt": float(captured[hub_idx, _CCS] * share[_CCS]),
            "h2_kg": float(h2_kg_by_hub[hub_idx]),
            "water_m3": total_water,
            "water_base_m3": base_water,
            "water_capture_increment_m3": float(
                (water[hub_idx, _CCS] - base_water) * share[_CCS]
            ),
            "cost_cny": annual_ccs + annual_h2 + capital,
            "cost_capital_cny": capital,
            "cost_annual_cny": annual_ccs + annual_h2,
            "cost_h2_purchase_cny": float(h2_cost_by_hub[hub_idx]),
            "h2_price_paid_cny_per_kg": (
                float(h2_cost_by_hub[hub_idx] / h2_kg_by_hub[hub_idx]) if h2_kg_by_hub[hub_idx] > 1e-6 else float("nan")
            ),
            "h2_price_national_mean_cny_per_kg": h2_price_mean,
        })
    return pd.DataFrame(rows, columns=columns)
