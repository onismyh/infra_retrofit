"""结果工作簿里源、汇与输送的各表：Source_Results、Sink_Results、Transmission_Matrix、捕集与注入的 *_stock 表，
以及本模型另加的 Plant_Pathways、Industry_Routes（入口在 `results_workbook`）。"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..constants_industry import (
    INDUSTRY_ROUTES,
    SECTOR_AMMONIA,
    SECTOR_CEMENT,
    SECTOR_METHANOL,
    SECTOR_STEEL_BF,
    SECTOR_STEEL_EAF,
)
from ._shared import PreparedInputs
from .results_regions import CROSS_REGION, OFFSHORE, PROVINCES, REGIONS, province_region
from .results_tracing import FLOW_TOL, trace_sources_to_sinks
from .scenario import PATHWAYS

# 一张表的一块：(起始行, 起始列, 各行的值)，行列号从 1 起。
Block = tuple[int, int, list[list[Any]]]
MILLION = 1e6

# 工业部门 -> ChinaCCS 的部门组（isi 钢铁、cem 水泥、nh3 合成氨、meoh 甲醇）；煤电是 power。
SECTOR_GROUP = {
    SECTOR_STEEL_BF: "isi", SECTOR_STEEL_EAF: "isi", SECTOR_CEMENT: "cem", SECTOR_AMMONIA: "nh3", SECTOR_METHANOL: "meoh",
}
# Source_Results 的四个部门（电力、钢铁、水泥、化工）各含哪些部门组。
SOURCE_SECTORS = {"power": ("power",), "isi": ("isi",), "cement": ("cem",), "chemical": ("nh3", "meoh")}
SOURCE_SECTOR_ZH = {"power": "电力", "isi": "钢铁", "cement": "水泥", "chemical": "化工"}
# CO2_capture_<键>_stock 各表含哪些部门组。
CAPTURE_STOCKS = {
    "power": ("power",), "isi": ("isi",), "cem": ("cem",), "nh3": ("nh3",), "meoh": ("meoh",),
    "chemical": ("nh3", "meoh"), "total": ("power", "isi", "cem", "nh3", "meoh"),
}
SOURCE_HEADER = [
    None, "CO2_emit_rate", "CAPEX_Capture", "OPEX_Capture_p", "OPEX_Capture_c", "OPEX_Capture_i", "OPEX_Capture_a",
    "Region", "province", "CO2_power", "CO2_isi", "CO2_cement", "CO2_chemical", "Total Capture",
    "Power_Capture", "ISI_Capture", "Cement_Capture", "Chemical_Capture",
]
OPEX_ORDER = ("power", "cement", "isi", "chemical")  # OPEX_Capture_p、_c、_i、_a 的顺序
# Plant_Pathways 的份额列：各路径份额，及湿冷改空冷的装机里当年在运行的、已建成的各占全厂的份额（`plant_pathways`）。
PLANT_SHARES = {**{pw: f"share_{pw}" for pw in PATHWAYS}, "air_retrofit_operating": "air_retrofit_operating",
                "air_retrofit_installed": "air_retrofit_installed"}


def sources_frame(tables: Mapping[str, pd.DataFrame], prepared: PreparedInputs) -> pd.DataFrame:
    """每年每个源一行：id、部门组、Source_Results 的部门、省、区、所在节点、本年基线排放（Mt/yr）、本年新建捕集能力的
    投资（CNY）、本年捕集运行费（CNY/yr）、捕集量（Mt/yr）。煤电的捕集运行费 = 捕集岛固定运维 + CCS 额外燃料
    （`plant_cost.csv` 的 `ccs_om_cny`、`energy_penalty_cny`），工业的 = 捕集路线的年度费（`cost_annual_ccs_cny`）。"""
    plant = tables["plant_detail.csv"].merge(
        tables["plant_cost.csv"][["year", "plant_id", "ccs_retrofit_capex_cny", "ccs_om_cny", "energy_penalty_cny"]],
        on=["year", "plant_id"], how="left", validate="one_to_one",
    )
    hubs = tables["industry_detail.csv"]
    network = prepared.network
    frame = pd.concat([
        pd.DataFrame({
            "year": plant["year"], "id": plant["plant_id"].astype(str), "group": "power",
            "province": plant["province_name"], "emit_mt": plant["baseline_emissions_mt"],
            "capex_cny": plant["ccs_retrofit_capex_cny"], "opex_cny": plant["ccs_om_cny"] + plant["energy_penalty_cny"],
            "captured_mt": plant["captured_mt"], "node": plant["plant_id"].astype(str).map(network.plant_node_ids),
        }),
        pd.DataFrame({
            "year": hubs["year"], "id": hubs["hub_id"].astype(str), "group": hubs["sector"].map(SECTOR_GROUP),
            "province": hubs["province"], "emit_mt": hubs["baseline_co2_mt"],
            "capex_cny": hubs["cost_capital_ccs_cny"], "opex_cny": hubs["cost_annual_ccs_cny"],
            "captured_mt": hubs["captured_mt"], "node": hubs["hub_id"].astype(str).map(network.industry_node_ids),
        }),
    ], ignore_index=True)
    if frame[["group", "node"]].isna().to_numpy().any():
        raise ValueError("结果表里有部门不在 SECTOR_GROUP 里、或不在管网上的源")
    frame["sector"] = frame["group"].map({group: s for s, groups in SOURCE_SECTORS.items() for group in groups})
    frame["region"] = frame["province"].map(province_region)
    return frame


def captured_by_year(sources: pd.DataFrame, years: Sequence[int]) -> dict[int, float]:
    """各年的总捕集量（Mt/yr），煤电与工业的源都算，即式 (37) 分母里逐年的捕集量；*sources* 是 `sources_frame` 的结果。"""
    return {y: float(sources.loc[sources["year"] == y, "captured_mt"].sum()) for y in years}


def sinks_frame(
    tables: Mapping[str, pd.DataFrame], prepared: PreparedInputs, node_province: Mapping[str, str]
) -> pd.DataFrame:
    """每年每个封存汇一行：id、DSA/EOR、所在节点、省、区、封存量（Mt/yr）、扣 EOR 抵扣后的每吨封存成本（CNY/t，
    即目标函数用的 `storage_cost_cny_per_t`）。"""
    use = tables["storage_utilization.csv"]
    storages = prepared.storages.set_index(prepared.storages["storage_hub_id"].astype(str))
    ids = use["storage_hub_id"].astype(str)
    node = ids.map(prepared.network.storage_node_ids)
    frame = pd.DataFrame({
        "year": use["year"], "id": ids, "category": ids.map(storages["storage_type"]).astype(str).str.upper(),
        "node": node, "province": node.map(node_province), "use_mt": use["storage_use_mtpa"],
        "cost_cny_per_t": ids.map(storages["storage_cost_cny_per_t"]).astype(float),
    })
    frame["region"] = frame["province"].map(province_region)
    return frame


def source_results(sources: pd.DataFrame) -> list[Block]:
    """Source_Results_<年>：左边每个源一行（成本为百万元），右边分区与分省的捕集量。*sources* 只含这一年。"""
    left: list[list[Any]] = [SOURCE_HEADER]
    for row in sources.itertuples(index=False):
        left.append([
            row.id, row.emit_mt, row.capex_cny / MILLION,
            *(row.opex_cny / MILLION if row.sector == sector else None for sector in OPEX_ORDER),
            row.region, row.province,
            *(row.emit_mt if row.sector == sector else 0.0 for sector in SOURCE_SECTORS),
            row.captured_mt,
            *(row.captured_mt if row.sector == sector else 0.0 for sector in SOURCE_SECTORS),
        ])
    sector_zh = [SOURCE_SECTOR_ZH[sector] for sector in SOURCE_SECTORS]
    by_region = sources.groupby(["region", "sector"])["captured_mt"].sum()
    regions: list[list[Any]] = [["分区捕集量（Mt/yr）"], ["Region", "地区", "总捕集", *sector_zh]]
    for code in range(1, 7):
        values = [float(by_region.get((code, sector), 0.0)) for sector in SOURCE_SECTORS]
        regions.append([code, REGIONS[code], sum(values), *values])
    by_province = sources.groupby(["province", "sector"])["captured_mt"].sum()
    provinces: list[list[Any]] = [["分省捕集量（Mt/yr）"], ["province", "省", *sector_zh, "总"]]
    for name, (short, _) in PROVINCES.items():
        values = [float(by_province.get((name, sector), 0.0)) for sector in SOURCE_SECTORS]
        provinces.append([name, short, *values, sum(values)])
    column = len(SOURCE_HEADER) + 2
    return [(1, 1, left), (1, column, regions), (len(regions) + 2, column, provinces)]


def sink_results(sinks: pd.DataFrame, years: Sequence[int]) -> list[Block]:
    """Sink_Results：左边每个汇一行、各年注入量，右边分区（含 7 海上）与分省（含海上）各年的注入量。"""
    wide = wide_by_year(sinks, "id", "use_mt", years)
    info = sinks.drop_duplicates("id").set_index("id")
    left: list[list[Any]] = [[None, "Sink_Category", "Region", "Province", *(f"Storage_{y}" for y in years)]]
    for sink_id, row in wide.iterrows():
        left.append([sink_id, info.at[sink_id, "category"], info.at[sink_id, "region"], info.at[sink_id, "province"],
                     *row])
    by_region = sinks.groupby(["region", "year"])["use_mt"].sum()
    regions: list[list[Any]] = [["分区封存量（Mt/yr）"], ["Region", "地区", *years]]
    for code in range(1, CROSS_REGION):
        regions.append([code, REGIONS[code], *(float(by_region.get((code, y), 0.0)) for y in years)])
    by_province = sinks.groupby(["province", "year"])["use_mt"].sum()
    provinces: list[list[Any]] = [["分省封存量（Mt/yr）"], ["Province", "省", *years]]
    for name, short in [*((name, short) for name, (short, _) in PROVINCES.items()), (OFFSHORE, "海上")]:
        provinces.append([name, short, *(float(by_province.get((name, y), 0.0)) for y in years)])
    column = len(left[0]) + 2
    return [(1, 1, left), (1, column, regions), (len(regions) + 2, column, provinces)]


def transmission_matrix(
    flows: pd.DataFrame, sources: pd.DataFrame, sinks: pd.DataFrame, node_province: Mapping[str, str]
) -> list[list[Any]]:
    """Transmission_Matrix_<年>：源省 x 汇省的输送量（Mt/yr），末行末列为合计，零格留空。

    *flows* 是这一年的 `co2_flow_direction.csv`（逐边净流量与方向），*sources*、*sinks* 也只含这一年。按节点充分
    混合把各汇的封存量分回各源（`results_tracing`），源与汇再按所在节点的省归并。
    """
    arcs: dict[tuple[str, str], float] = defaultdict(float)
    for row in flows.itertuples(index=False):
        arcs[(str(row.source_node), str(row.sink_node))] += float(row.net_flow_mtpa)
    supply = {str(node): float(mt) for node, mt in sources.groupby("node")["captured_mt"].sum().items()}
    demand = {str(node): float(mt) for node, mt in sinks.groupby("node")["use_mt"].sum().items()}
    cells: dict[tuple[str, str], float] = defaultdict(float)
    for (source, sink), amount in trace_sources_to_sinks(arcs, supply, demand).items():
        cells[(node_province[source], node_province[sink])] += amount
    columns = [*PROVINCES, OFFSHORE]
    out: list[list[Any]] = [["源/汇", *columns, "合计"]]
    for origin in PROVINCES:
        values = [cells.get((origin, target), 0.0) for target in columns]
        out.append([origin, *map(blank, values), blank(sum(values))])
    totals = [sum(cells.get((origin, target), 0.0) for origin in PROVINCES) for target in columns]
    out.append(["合计", *map(blank, totals), blank(sum(totals))])
    return out


def stock_table(frame: pd.DataFrame, value: str, years: Sequence[int]) -> list[list[Any]]:
    """ChinaCCS 的 *_stock 表：id x 年份，只列有一年不为零的行，零格留空。"""
    wide = wide_by_year(frame, "id", value, years)
    out: list[list[Any]] = [[None, *years]]
    for key, row in wide.iterrows():
        if (row.abs() > FLOW_TOL).any():
            out.append([key, *map(blank, row)])
    return out


def share_table(
    frame: pd.DataFrame, info: Mapping[str, str], shares: Mapping[str, str], years: Sequence[int]
) -> list[list[Any]]:
    """Plant_Pathways、Industry_Routes：每个 hub 一行，*info* 的说明列（表头 -> 列名），然后各份额各年一列，表头为
    <份额>_<年>。*frame* 带 `id` 与 `year` 两列。"""
    order = list(dict.fromkeys(frame["id"]))
    first = frame.drop_duplicates("id").set_index("id")
    wides = {
        label: frame.pivot(index="id", columns="year", values=column).reindex(index=order, columns=list(years))
        for label, column in shares.items()
    }
    out: list[list[Any]] = [[None, *info, *(f"{label}_{y}" for label in shares for y in years)]]
    for key in order:
        out.append([key, *(first.at[key, column] for column in info.values()),
                    *(wides[label].at[key, y] for label in shares for y in years)])
    return out


def plant_pathways(tables: Mapping[str, pd.DataFrame], years: Sequence[int]) -> list[list[Any]]:
    """Plant_Pathways：煤电 hub 各年各路径份额与空冷改造份额（`plant_detail.csv`）。

    `plant_detail.csv` 的 `air_operating_share`、`air_installed_share` 是仍湿冷那部分的改造进度，乘 1 − already_air_share
    才是占全厂的份额（已全空冷的 hub 两列不保证为零，乘上后为零）；原本就是空冷的份额另列 `already_air`。
    """
    plants = tables["plant_detail.csv"]
    wet = 1.0 - plants["already_air_share"]
    frame = plants.assign(
        id=plants["plant_id"].astype(str), region=plants["province_name"].map(province_region),
        air_retrofit_operating=plants["air_operating_share"] * wet,
        air_retrofit_installed=plants["air_installed_share"] * wet,
    )
    info = {"Region": "region", "province": "province_name", "already_air": "already_air_share"}
    return share_table(frame, info, PLANT_SHARES, years)


def industry_routes(tables: Mapping[str, pd.DataFrame], years: Sequence[int]) -> list[list[Any]]:
    """Industry_Routes：工业 hub 各年路线份额（`industry_detail.csv`）。"""
    hubs = tables["industry_detail.csv"]
    frame = hubs.assign(id=hubs["hub_id"].astype(str), region=hubs["province"].map(province_region))
    shares = {route: f"share_{route}" for route in INDUSTRY_ROUTES}
    return share_table(frame, {"sector": "sector", "Region": "region", "province": "province"}, shares, years)


def wide_by_year(frame: pd.DataFrame, key: str, value: str, years: Sequence[int]) -> pd.DataFrame:
    """*key* x 年份的宽表，行序为 *key* 首次出现的顺序，缺的格为 0。"""
    order = list(dict.fromkeys(frame[key]))
    if frame.empty:
        return pd.DataFrame(0.0, index=order, columns=list(years))
    wide = frame.groupby([key, "year"])[value].sum().unstack("year")
    return wide.reindex(index=order, columns=list(years)).fillna(0.0)


def blank(value: float) -> float | None:
    """绝对值不超过 `FLOW_TOL` 的记空格。"""
    return None if abs(float(value)) <= FLOW_TOL else float(value)
