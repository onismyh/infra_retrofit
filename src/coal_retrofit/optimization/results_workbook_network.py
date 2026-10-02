"""结果工作簿里管道与成本的各表：num_pipe_stock、num_pipes_total、CO2_flow_stock、GIS_info、taransport_analysis、
Total_CO2_Pipeline_Length、Cost_Capture / Transport / Storage、Revenue_EOR、Cost_Analysis（入口在 `results_workbook`）。"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._shared import PreparedInputs
from .resource_access import _haversine_distances_km
from .results_regions import CROSS_REGION, REGIONS, edge_region, province_region
from .results_tracing import FLOW_TOL
from .results_workbook_sources import MILLION, Block, blank
from .salvage import remaining_fraction

# 一项成本：(行名, 折现类别 annual / one_off, 残值台账里的同一项或空串, 各年不折现的 CNY)。
CostLine = tuple[str, str, str, dict[int, float]]
# Cost_Analysis 的四个环节：(行名, 成本表, 符号)；EOR 抵扣冲减成本，记负。
STAGES = (("捕集环节", "Cost_Capture", 1.0), ("运输环节", "Cost_Transport", 1.0), ("封存环节", "Cost_Storage", 1.0),
          ("EOR 抵扣", "Revenue_EOR", -1.0))


def edge_years(tables: Mapping[str, pd.DataFrame], tiers: Sequence[float]) -> pd.DataFrame:
    """每年每条边一行：起止节点、管长、各管径档的在役根数（`pipes_<档>` 列，取自 `pipes_in_service_by_tier`）与合计
    （`pipes`）、净流量（`net`，Mt/yr，起点到终点为正，取自 `co2_flow_direction.csv`）。"""
    labels = [f"{tier:g}" for tier in tiers]
    edges = tables["network_edges.csv"][
        ["year", "edge_id", "from_node_id", "to_node_id", "length_km", "pipes_in_service_by_tier"]
    ].copy()
    edges["edge_id"] = edges["edge_id"].astype(str)  # 与 `co2_flow_direction.csv`、`pipe_sheets` 的边号同为字符串
    counts = np.array(
        [_tier_counts(text, labels) for text in edges["pipes_in_service_by_tier"]], dtype=np.int64
    ).reshape(len(edges), len(labels))
    for k, label in enumerate(labels):
        edges[f"pipes_{label}"] = counts[:, k]
    edges["pipes"] = edges[[f"pipes_{label}" for label in labels]].sum(axis=1)
    flows = tables["co2_flow_direction.csv"]
    net = flows.assign(edge_id=flows["edge_id"].astype(str), net=flows["flow_fwd_mtpa"] - flows["flow_bwd_mtpa"])[
        ["year", "edge_id", "net"]]
    edges = edges.merge(net, on=["year", "edge_id"], how="left", validate="one_to_one")
    edges["net"] = edges["net"].fillna(0.0)
    return edges


def _tier_counts(text: object, labels: Sequence[str]) -> list[int]:
    """"1x2|1x20" 这样的串（`results_network._tier_strings`）拆成各档根数，档序同 *labels*。"""
    counts = dict.fromkeys(labels, 0)
    if isinstance(text, str) and text:
        for part in text.split("|"):
            number, tier = part.split("x")
            counts[tier] += int(number)
    return list(counts.values())


def pipe_sheets(
    edges: pd.DataFrame, years: Sequence[int], tiers: Sequence[float], prepared: PreparedInputs,
    node_province: Mapping[str, str],
) -> dict[str, list[Block]]:
    """管道各表。只列有一年有在役管或有流量的边，行序同候选边表；*edges* 是 `edge_years` 的结果。"""
    labels = [f"{tier:g}" for tier in tiers]
    by = edges.set_index(["edge_id", "year"])
    used = set(edges.loc[(edges["pipes"] > 0) | (edges["net"].abs() > FLOW_TOL), "edge_id"])
    nodes = prepared.network.nodes.set_index(prepared.network.nodes["node_id"].astype(str))
    static = prepared.network.edges.assign(edge_id=prepared.network.edges["edge_id"].astype(str))
    region: dict[str, int] = {}
    info: dict[str, list[Any]] = {}
    for row in static.itertuples(index=False):
        edge, start, end = str(row.edge_id), str(row.from_node_id), str(row.to_node_id)
        regions = (province_region(node_province[start]), province_region(node_province[end]))
        region[edge] = edge_region(*regions)
        if edge in used:
            lon0, lat0 = float(nodes.at[start, "lon"]), float(nodes.at[start, "lat"])
            lon1, lat1 = float(nodes.at[end, "lon"]), float(nodes.at[end, "lat"])
            distance = float(_haversine_distances_km(lon0, lat0, np.array([lon1]), np.array([lat1]))[0])
            info[edge] = [start, end, distance, float(row.length_km), node_province[start], node_province[end],
                          *regions, region[edge], lon0, lat0, lon1, lat1]
    order = [edge for edge in static["edge_id"] if edge in used]

    def series(edge: str, column: str) -> list[float]:
        return [float(by.at[(edge, y), column]) for y in years]

    stock: list[list[Any]] = [["edge_id", "Node1", "Node2", "档（Mtpa）", *years, "Length_Weighted(km)",
                               *(f"{y}长度" for y in years)]]
    total: list[list[Any]] = [["edge_id", "Node1", "Node2", *years, "Distance(km)", "Length_Weighted(km)",
                               "Province_from", "Province_to", "region_from", "region_to", "region",
                               *(f"{y}长度" for y in years), *(f"{y}flow" for y in years)]]
    flow: list[list[Any]] = [["edge_id", "Node1", "Node2", *years]]
    gis: list[list[Any]] = [["edge_id", "Node 1", "Node 2", *(f"CO2_flow/Mt_{y}" for y in years), "Distance/km",
                             "Start_lon", "Start_lat", "End_lon", "End_lat"]]
    for edge in order:
        start, end, distance, length, *place, lon0, lat0, lon1, lat1 = info[edge]
        for tier, label in zip(tiers, labels):
            counts = series(edge, f"pipes_{label}")
            if any(counts):
                stock.append([edge, start, end, tier, *map(blank, counts), length,
                              *(blank(count * length) for count in counts)])
        counts, nets = series(edge, "pipes"), series(edge, "net")
        total.append([edge, start, end, *map(blank, counts), distance, length, *place,
                      *(blank(count * length) for count in counts), *map(blank, nets)])
        if any(abs(net) > FLOW_TOL for net in nets):
            flow.append([edge, start, end, *map(blank, nets)])
        gis.append([edge, start, end, *nets, distance, lon0, lat0, lon1, lat1])
    return {
        "num_pipe_stock": [(1, 1, stock)],
        "num_pipes_total": [(1, 1, total)],
        "CO2_flow_stock": [(1, 1, flow)],
        "GIS_info": [(1, 1, gis)],
        "taransport_analysis": _transport_analysis(edges.assign(region=edges["edge_id"].map(region)), years, labels),
        "Total_CO2_Pipeline_Length": [(1, 1, [
            ["口径", *years],
            ["每条边只算一次（km）", *(float(edges.loc[(edges["year"] == y) & (edges["pipes"] > 0), "length_km"].sum())
                                  for y in years)],
            ["按根累计（km）", *(float((edges["pipes"] * edges["length_km"])[edges["year"] == y].sum()) for y in years)],
        ])],
    }


def _transport_analysis(edges: pd.DataFrame, years: Sequence[int], labels: Sequence[str]) -> list[Block]:
    """taransport_analysis（表名照 ChinaCCS 的拼写）：上面各管径档的在役根数与按根累计的管长，下面按分区汇总。"""
    tiers: list[list[Any]] = [[None, None, *(f"{label} Mtpa" for label in labels), "总"]]
    for y in years:
        year = edges[edges["year"] == y]
        counts = [int(year[f"pipes_{label}"].sum()) for label in labels]
        lengths = [float((year[f"pipes_{label}"] * year["length_km"]).sum()) for label in labels]
        tiers += [["数量", y, *counts, sum(counts)], ["长度（km）", y, *lengths, sum(lengths)]]
    region_counts = edges.groupby(["region", "year"])["pipes"].sum()
    region_lengths = (edges["pipes"] * edges["length_km"]).groupby([edges["region"], edges["year"]]).sum()
    pad = [None] * (len(years) - 1)
    regions: list[list[Any]] = [[None, "管道数量", *pad, "管道长度（km）", *pad], ["地区", *years, *years, "地区"]]
    for code in range(1, CROSS_REGION + 1):
        regions.append([code, *(int(region_counts.get((code, y), 0)) for y in years),
                        *(float(region_lengths.get((code, y), 0.0)) for y in years), REGIONS[code]])
    return [(1, 1, tiers), (len(tiers) + 3, 1, regions)]


def cost_lines(
    tables: Mapping[str, pd.DataFrame], sinks: pd.DataFrame, years: Sequence[int]
) -> dict[str, list[CostLine]]:
    """Cost_Capture、Cost_Transport、Cost_Storage、Revenue_EOR 各项各年不折现的 CNY。

    捕集取逐厂、逐 hub 的值（`plant_cost.csv`、`industry_detail.csv`），运输取 `cost_breakdown.csv`；封存费按封存量 x
    汇的扣抵扣前单价（含海上倍率），EOR 抵扣 = 封存量 x（扣前单价 − 扣后单价），两者之差即目标函数的 `storage_cost`。
    2026-10-02 前扣前单价取全国一个 `storage_cost_cny_per_t`，海上汇加价后会被记出负的抵扣，改为逐汇取。
    """
    def per_year(frame: pd.DataFrame, column: str) -> dict[int, float]:
        return {y: float(frame.loc[frame["year"] == y, column].sum()) for y in years}

    plant, hubs, breakdown = tables["plant_cost.csv"], tables["industry_detail.csv"], tables["cost_breakdown.csv"]
    before = sinks["cost_before_credit_cny_per_t"]
    stored = sinks.assign(before=sinks["use_mt"] * MILLION * before,
                          credit=sinks["use_mt"] * MILLION * (before - sinks["cost_cny_per_t"]))
    return {
        "Cost_Capture": [
            ("煤电捕集岛投资", "one_off", "ccs_retrofit_capex", per_year(plant, "ccs_retrofit_capex_cny")),
            ("工业捕集投资", "one_off", "industry_ccs_capex", per_year(hubs, "cost_capital_ccs_cny")),
            ("煤电捕集岛固定运维", "annual", "", per_year(plant, "ccs_om_cny")),
            ("煤电 CCS 额外燃料", "annual", "", per_year(plant, "energy_penalty_cny")),
            ("工业捕集年度费", "annual", "", per_year(hubs, "cost_annual_ccs_cny")),
        ],
        "Cost_Transport": [
            ("管道投资", "one_off", "pipe_capex", per_year(breakdown[breakdown["category"] == "pipe_capex"],
                                                           "cost_undiscounted_cny")),
            ("运输运维", "annual", "", per_year(breakdown[breakdown["category"] == "transport_opex"],
                                               "cost_undiscounted_cny")),
        ],
        "Cost_Storage": [("封存费（扣 EOR 抵扣前）", "annual", "", per_year(stored, "before"))],
        "Revenue_EOR": [("EOR 抵扣", "annual", "", per_year(stored, "credit"))],
    }


def ledger_salvage(solution: Mapping[str, Any], years: Sequence[int], end_year: int) -> dict[str, float]:
    """残值台账各项的期末残值抵扣（CNY，已按期末折现），与 `salvage._add_salvage_credit` 同式，各项相加即最后一年
    `salvage_credit` 的相反数；不计残值（`end_of_horizon_salvage=False`）时为空。"""
    last = solution["year_solutions"][years[-1]]["cost_weights"].get("salvage_credit")
    if last is None:
        return {}
    credit: dict[str, float] = defaultdict(float)
    for y in years:
        for name, value, life in solution["year_solutions"][y]["salvage_ledger"]:
            credit[name] += float(last[1]) * remaining_fraction(y, life, end_year) * float(value)
    return dict(credit)


def cost_sheets(
    lines: Mapping[str, list[CostLine]], captured: Mapping[int, float], edges: pd.DataFrame,
    solution: Mapping[str, Any], years: Sequence[int], end_year: int,
) -> dict[str, list[Block]]:
    """Cost_Capture / Transport / Storage、Revenue_EOR（各项各年不折现，百万元）与 Cost_Analysis。

    *captured* 是各年的总捕集量（Mt/yr），*edges* 是 `edge_years` 的结果。全期每吨成本 = 各环节成本的现值 ÷ 捕集量的
    现值：每项乘目标函数里同类成本的折现权重（`cost_weights`），投资减去期末残值（`ledger_salvage`）；捕集量乘年度项的
    权重（折现 x 年金）。
    """
    sheets: dict[str, list[Block]] = {}
    for sheet, items in lines.items():
        rows: list[list[Any]] = [["百万元", *years]]
        rows += [[label, *(values[y] / MILLION for y in years)] for label, _, _, values in items]
        if len(items) > 1:
            rows.append(["合计", *(sum(values[y] for *_, values in items) / MILLION for y in years)])
        sheets[sheet] = [(1, 1, rows)]

    weights = {y: {kind: w for kind, w in solution["year_solutions"][y]["cost_weights"].values()} for y in years}
    salvage = ledger_salvage(solution, years, end_year)
    present = {
        sheet: sum(weights[y][kind] * values[y] for _, kind, _, values in items for y in years)
        - sum(salvage.get(item, 0.0) for _, _, item, _ in items if item)
        for sheet, items in lines.items()
    }
    present_captured = sum(weights[y]["annual"] * captured[y] for y in years)
    current = {sheet: {y: sum(values[y] for *_, values in items) for y in years} for sheet, items in lines.items()}
    running = {sheet: {y: sum(values[y] for _, kind, _, values in items if kind == "annual") for y in years}
               for sheet, items in lines.items()}
    tonne_km = {y: MILLION * float((edges["net"].abs() * edges["length_km"])[edges["year"] == y].sum()) for y in years}

    def per_tonne(value: float, captured_mt: float) -> float | None:  # 捕集量（Mt）在 FLOW_TOL 以内是数值噪声，记空
        return value / (captured_mt * MILLION) if captured_mt > FLOW_TOL else None

    analysis: list[list[Any]] = [["当期成本（本年投资 + 本年运行费，不折现）"], ["百万元", *years]]
    analysis += [[label, *(sign * current[sheet][y] / MILLION for y in years)] for label, sheet, sign in STAGES]
    analysis += [["合计", *(sum(sign * current[sheet][y] for _, sheet, sign in STAGES) / MILLION for y in years)], []]
    analysis += [["每吨运行费（本年运行费 ÷ 本年捕集量）"], ["元/t", *years]]
    analysis += [[label, *(per_tonne(sign * running[sheet][y], captured[y]) for y in years)]
                 for label, sheet, sign in STAGES]
    analysis += [["全流程", *(per_tonne(sum(sign * running[sheet][y] for _, sheet, sign in STAGES), captured[y])
                             for y in years)], []]
    analysis += [["全期每吨成本（成本现值 ÷ 捕集量现值）"], ["元/t", "全期"]]
    analysis += [[label, per_tonne(sign * present[sheet], present_captured)] for label, sheet, sign in STAGES]
    analysis += [["全流程", per_tonne(sum(sign * present[sheet] for _, sheet, sign in STAGES), present_captured)], []]
    analysis += [["平均运程（Σ |净流量| x 管长 ÷ 捕集量）"], ["km", *years],
                 ["平均运程", *(per_tonne(tonne_km[y], captured[y]) for y in years)]]
    side: list[list[Any]] = [["CO2 捕集量"], ["Mt/yr", *years], ["捕集量", *(captured[y] for y in years)]]
    sheets["Cost_Analysis"] = [(1, 1, analysis), (1, len(years) + 3, side)]
    return sheets
