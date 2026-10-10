"""资源结果表：`resources.csv`（生物质、绿氢、水节点与流域取水指标的用量、可用量与松弛）与 `resource_flows.csv`
（逐链路流量与采购费）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import AMMONIA_FLOW_SCALE, NH3_H2_RATIO, WATER_FLOW_SCALE
from ._shared import PreparedInputs
from .year_types import YearSolution

RESOURCE_COLUMNS = ["year", "resource_type", "node_id", "province", "longitude", "latitude", "competition_scope",
                    "used", "available", "slack", "unit", "utilization"]
FLOW_COLUMNS = ["year", "resource_type", "node_id", "source_type", "source_id", "distance_km", "flow", "unit",
                "h2_kg", "price_cny_per_unit", "cost_cny"]
_FLOW_TOL = 1e-3  # 只列流量大于它的链路


def _build_resource_table(prepared: PreparedInputs, year: int, ys: YearSolution) -> pd.DataFrame:
    """各节点一行；`used`、`available`、`slack` 的单位见 `unit`，`slack` 是该节点（流域）约束的松弛（应为零）。

    绿氢节点（`resource_type` = green_h2，2026-10-10 起按绿氢计）的用量 = 煤电氨流量 x NH3_H2_RATIO + 工业氢流量，
    与节点约束 `model_resources.add_resource_balances` 同口径。水节点（water）是耗水口径的生态流量上限；流域取水指标
    （water_basin_quota）是取水口径，单列一种，两者不能相加。无水约束或关掉流域上限时没有流域行；无水约束时水节点
    没有上限，可用量为空。利用率 = 用量 / 可用量；可用量 ≤ 0 而用量 > 1e-6（用了松弛）记 inf，与图 7 的"无余量"
    同一规则（`scripts/plot_fig7_water.utilization`），可用量 ≤ 0 而没有用量记 0。经纬度：生物质节点取自节点表，绿氢与水
    节点按节点号从输入表（`prepared.ammonia_supply`、`prepared.water_nodes`）补上，流域行为空。
    """
    yd = ys["year_data"]
    slacks = ys["slacks"]
    biomass = prepared.biomass
    h2_used = _node_sum(yd.ammonia_links["ammonia_node_id"], np.asarray(ys["ammonia_flow_kg"]) * NH3_H2_RATIO,
                        yd.ammonia_nodes["ammonia_node_id"])
    membership = getattr(yd, "industry_h2_node_membership", None)
    if membership is not None and len(ys["industry_h2_flow_kg"]):
        h2_used = h2_used + np.asarray(membership @ np.asarray(ys["industry_h2_flow_kg"], dtype=np.float64)).ravel()
    frames = [
        _nodes(year, "biomass", biomass, "biomass_node_id", "shared_biomass_node", "GJ/yr",
               _node_sum(prepared.biomass_links["biomass_node_id"], ys["biomass_flow_gj"], biomass["biomass_node_id"]),
               biomass["available_gj"].astype(float).to_numpy(), slacks["biomass_slack_gj"]),
        # 求解器的可用量是缩放单位，乘回物理单位，否则利用率会差 1e6 倍。
        _nodes(year, "green_h2", _with_coordinates(yd.ammonia_nodes, "ammonia_node_id", prepared.ammonia_supply),
               "ammonia_node_id", "shared_green_h2_node", "kg H2/yr", h2_used,
               np.asarray(yd.h2_available_kg, dtype=np.float64) * AMMONIA_FLOW_SCALE, slacks["h2_slack_kg"]),
        _nodes(year, "water", _with_coordinates(yd.water_nodes, "water_node_id", prepared.water_nodes),
               "water_node_id", "shared_water_node", "m3/yr",
               _node_sum(yd.water_links["water_node_id"], ys["water_flow_m3"], yd.water_nodes["water_node_id"]),
               np.asarray(yd.water_available_m3, dtype=np.float64) * WATER_FLOW_SCALE, slacks["water_slack_m3"]),
    ]
    codes = list(yd.water_basin_codes or [])
    if codes:
        used = np.asarray(slacks["water_basin_use_m3"], dtype=np.float64)
        frames.append(pd.DataFrame({
            "year": year, "resource_type": "water_basin_quota", "node_id": codes[:len(used)], "province": "",
            "competition_scope": "basin_withdrawal_cap", "used": used,
            "available": np.asarray(yd.water_basin_available_m3, dtype=np.float64)[:len(used)],
            "slack": np.asarray(slacks["water_basin_slack_m3"], dtype=np.float64)[:len(used)], "unit": "m3/yr",
        }))
    table = pd.concat(frames, ignore_index=True, sort=False)
    used, available = table["used"].to_numpy(float), table["available"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["utilization"] = np.where(
            available > 0, used / available, np.where((available <= 0) & (used > 1e-6), np.inf, 0.0)
        )
    return table.reindex(columns=RESOURCE_COLUMNS)


def _with_coordinates(nodes: pd.DataFrame, id_column: str, source: pd.DataFrame) -> pd.DataFrame:
    """年度节点表只带节点号与省，经纬度按节点号从输入表 *source*（逐年重复的取第一行）补上。"""
    xy = source.assign(node=source[id_column].astype(str)).drop_duplicates("node").set_index("node")
    ids = nodes[id_column].astype(str)
    return nodes.assign(longitude=ids.map(xy["longitude"]).to_numpy(), latitude=ids.map(xy["latitude"]).to_numpy())


def _node_sum(link_nodes: pd.Series, flows: np.ndarray, nodes: pd.Series) -> np.ndarray:
    """逐链路的量按链路的节点号加到各节点上，顺序同 `nodes`。"""
    grouped = pd.Series(np.asarray(flows, dtype=np.float64), index=link_nodes.astype(str).to_numpy()).groupby(level=0).sum()
    return nodes.astype(str).map(grouped).fillna(0.0).to_numpy(dtype=np.float64)


def _nodes(year: int, resource: str, nodes: pd.DataFrame, id_column: str, scope: str, unit: str,
           used: np.ndarray, available: np.ndarray, slack: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "year": year, "resource_type": resource, "node_id": nodes[id_column].astype(str).to_numpy(),
        "province": nodes.get("province_name", pd.Series("", index=nodes.index)).to_numpy(),
        "longitude": nodes.get("longitude", pd.Series(np.nan, index=nodes.index)).to_numpy(),
        "latitude": nodes.get("latitude", pd.Series(np.nan, index=nodes.index)).to_numpy(),
        "competition_scope": scope, "used": used, "available": available,
        "slack": np.asarray(slack, dtype=np.float64), "unit": unit,
    })


def _build_resource_flow_table(prepared: PreparedInputs, year: int, ys: YearSolution) -> pd.DataFrame:
    """逐链路一行，只列流量大于 1e-3 的。`price_cny_per_unit` 是目标函数所用的到厂价（每 `unit` 的一个单位），
    `cost_cny` = 到厂价 x 流量，CNY/yr。

    - biomass：煤电从生物质节点买的生物质，GJ/yr；采购费即 `costs.csv` 的 biomass_cost。
    - ammonia：煤电从绿氢节点买的氨，kg NH3/yr，`h2_kg` 是它折成的氢（x NH3_H2_RATIO）；采购费即 ammonia_cost。
    - green_h2：工业 hub 从同一批节点买的氢，kg H2/yr；采购费计在 `costs.csv` 工业的 h2_route 里（氢路线年度费
      = 非氢运行差额 + 买氢，下限 0，`model_industry.add_industry_year`）。
    - water：煤电从水节点取的水，m3/yr；取水费即 water_cost。
    """
    yd = ys["year_data"]
    frames = [
        _links(year, "biomass", prepared.biomass_links, "biomass_node_id", "coal", "plant_id", ys["biomass_flow_gj"],
               "GJ/yr", np.asarray(yd.biomass_link_cost_cny_per_gj) / float(yd.biomass_flow_scale)),
        _links(year, "ammonia", yd.ammonia_links, "ammonia_node_id", "coal", "plant_id", ys["ammonia_flow_kg"],
               "kg NH3/yr", np.asarray(yd.ammonia_link_cost_cny_per_kg) / float(yd.ammonia_flow_scale), NH3_H2_RATIO),
        _links(year, "green_h2", yd.industry_h2_links, "ammonia_node_id", "industry", "hub_id",
               ys["industry_h2_flow_kg"], "kg H2/yr",
               np.asarray(yd.industry_h2_link_cost_cny_per_kg) / AMMONIA_FLOW_SCALE, 1.0),
        _links(year, "water", yd.water_links, "water_node_id", "coal", "plant_id", ys["water_flow_m3"], "m3/yr",
               np.asarray(yd.water_link_cost_cny_per_m3) / float(yd.water_flow_scale)),
    ]
    kept = [frame for frame in frames if len(frame)]
    if not kept:
        return pd.DataFrame(columns=FLOW_COLUMNS)
    return pd.concat(kept, ignore_index=True, sort=False).reindex(columns=FLOW_COLUMNS)


def _links(year: int, resource: str, links: pd.DataFrame, node_column: str, source_type: str, source_column: str,
           flows: np.ndarray, unit: str, price: np.ndarray, h2_per_unit: float | None = None) -> pd.DataFrame:
    flow = np.asarray(flows, dtype=np.float64)
    if not len(flow):
        return pd.DataFrame(columns=FLOW_COLUMNS)
    # 没有单价的链路（水价向量为空时目标函数不计水费，`model_costs._resource_costs`）记 0。
    price = np.asarray(price, dtype=np.float64) if np.size(price) else np.zeros_like(flow)
    keep = flow > _FLOW_TOL
    links = links.reset_index(drop=True).loc[keep]
    return pd.DataFrame({
        "year": year, "resource_type": resource, "node_id": links[node_column].astype(str).to_numpy(),
        "source_type": source_type, "source_id": links[source_column].astype(str).to_numpy(),
        "distance_km": links["distance_km"].astype(float).to_numpy(), "flow": flow[keep], "unit": unit,
        "h2_kg": flow[keep] * h2_per_unit if h2_per_unit is not None else np.nan,
        "price_cny_per_unit": price[keep], "cost_cny": price[keep] * flow[keep],
    })
