"""工业结果表：逐 hub 的路线份额、减排、捕集、用水、用氢与成本。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import AMMONIA_FLOW_SCALE
from ..constants_industry import INDUSTRY_ROUTES
from ._shared import PreparedInputs
from .industry import CCS as _CCS, H2 as _H2, UNABATED as _UNABATED, IndustryYearData
from .year_types import YearData


def _build_industry_detail_table(
    prepared: PreparedInputs,
    year: int,
    industry_year_data: IndustryYearData,
    share_values: np.ndarray,
    h2_flow_kg: np.ndarray | None = None,
    year_data: YearData | None = None,
    *,
    capacity_mt: np.ndarray,
    new_capacity_mt: np.ndarray,
    ccs_fixed_om_cny: np.ndarray,
) -> pd.DataFrame:
    """每个工业 hub 每年一行：所选路线、减排、捕集、用水、成本。

    成本沿用模型自己的拆分：`cost_annual_cny` 是捕集能耗与耗材、在役且在用的捕集能力按建设年单价的固定运维
    （氢路线为固定运维与非氢运行差额），再加本年在该 hub 链路上实际买的氢；`cost_capital_cny` 是按本年新建
    能力计的一次性改造 capex。末尾四列（2026-10-02 起）把两者按捕集与氢路线分开，两两相加即这两列。
    `abatable_production_share`（2026-10-02 起）是可捕集份额为正的点源占 hub 产量的比例：两条路线只改造这些点源
    （混合原料的甲醇 hub 里份额为 0 的点源不改造，电炉钢为 0），没有点源表时为 1；按产量汇总路线构成时路线份额乘它，其余计入未改造。
    期末残值抵扣不分摊到各 hub，它是 `cost_breakdown.csv` 里的 `salvage_credit` 一行。

    Args:
        prepared: 准备好的输入；`prepared.industry` 带 hub 表。
        year: 规划年。
        industry_year_data: 本年的工业系数块。
        share_values: 求解得到的路线份额，形状 (hub_count, len(INDUSTRY_ROUTES))。
        h2_flow_kg: 求解得到的每条链路氢流量，kg。
        year_data: 本年的矩阵，用于取氢链路成本与关联矩阵。
        capacity_mt: 求解得到的在役路线能力，Mt/yr，形状同 `share_values`。
        new_capacity_mt: 本年新建的路线能力，Mt/yr，形状同 `share_values`。
        ccs_fixed_om_cny: 每个 hub 捕集能力的固定运维，CNY/yr（`vintage`）。

    Returns:
        每个 hub 一行，列序固定；没有工业 hub 时是列齐全的空表。
    """
    columns = [
        "year", "hub_id", "sector", "target_group", "province", "longitude", "latitude", "basin_code",
        "output_index", "production_kt_per_year", "abatable_production_share", "baseline_co2_mt", "process_co2_mt",
        "share_unabated", "share_ccs", "share_h2", "capacity_ccs_mt", "capacity_h2_mt",
        "reduction_mt", "residual_mt", "captured_mt", "h2_kg",
        "water_m3", "water_base_m3", "water_capture_increment_m3",
        "cost_cny", "cost_capital_cny", "cost_annual_cny", "cost_h2_purchase_cny",
        "h2_price_paid_cny_per_kg", "h2_price_national_mean_cny_per_kg",
        "cost_capital_ccs_cny", "cost_capital_h2_cny", "cost_annual_ccs_cny", "cost_annual_h2_cny",
    ]
    hubs = prepared.industry.hubs
    reduction = industry_year_data.reduction_mt
    baseline = industry_year_data.baseline_emissions_mt
    captured = industry_year_data.captured_mt
    water = industry_year_data.water_m3
    opex = industry_year_data.opex_cny
    unit_capex = industry_year_data.capex_cny_per_mt
    output_scale = industry_year_data.output_scale
    h2_price_mean = float(industry_year_data.h2_price_cny_per_kg)
    n_hubs = len(hubs)
    # 每个 hub 买的氢：链路流量 x 链路成本，用关联矩阵归到各 hub。
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
        cap = capacity_mt[hub_idx]
        new = new_capacity_mt[hub_idx]
        annual_ccs = float(opex[hub_idx, _CCS] * share[_CCS]) + float(ccs_fixed_om_cny[hub_idx])
        # 氢路线年度费有地板 0（求解器的 `h2_route_cost`，`model_industry.add_industry_year`）。
        annual_h2 = max(0.0, float(opex[hub_idx, _H2] * share[_H2]) + float(h2_cost_by_hub[hub_idx]))
        capital_ccs = float(unit_capex[hub_idx, _CCS] * new[_CCS])
        capital_h2 = float(unit_capex[hub_idx, _H2] * new[_H2])
        capital = capital_ccs + capital_h2
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
            "abatable_production_share": float(getattr(hub, "abatable_production_share", 1.0)),
            "baseline_co2_mt": float(baseline[hub_idx]),
            "process_co2_mt": float(hub.process_co2_mt_per_year) * float(output_scale[hub_idx]),
            "share_unabated": float(share[_UNABATED]),
            "share_ccs": float(share[_CCS]),
            "share_h2": float(share[_H2]),
            "capacity_ccs_mt": float(cap[_CCS]),
            "capacity_h2_mt": float(cap[_H2]),
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
            "cost_capital_ccs_cny": capital_ccs,
            "cost_capital_h2_cny": capital_h2,
            "cost_annual_ccs_cny": annual_ccs,
            "cost_annual_h2_cny": annual_h2,
        })
    return pd.DataFrame(rows, columns=columns)
