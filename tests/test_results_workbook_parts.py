"""结果工作簿的几个部件，不求解：源到汇的充分混合追踪（`results_tracing`）与输送矩阵的行列、管网节点归省与分区
（`results_regions`，落点定省要仓库里的 data/ChinaMap/provinces.shp）、在役根数串的拆分（`results_workbook_network._tier_counts`）、
工业明细按路线分的四列、源表与成本各表只取捕集的列、全期每吨成本、Plant_Pathways 的空冷份额。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from coal_retrofit.constants import AMMONIA_FLOW_SCALE
from coal_retrofit.constants_industry import SECTOR_STEEL_BF
from coal_retrofit.optimization.industry import CCS, H2, industry_year_data
from coal_retrofit.optimization.results_industry import _build_industry_detail_table
from coal_retrofit.optimization.results_network import _tier_strings
from coal_retrofit.optimization.results_regions import (
    CROSS_REGION,
    OFFSHORE,
    OFFSHORE_REGION,
    PROVINCES,
    REGIONS,
    _english,
    edge_region,
    node_provinces,
    province_region,
)
from coal_retrofit.optimization.results_tracing import trace_sources_to_sinks
from coal_retrofit.optimization.results_workbook_network import _tier_counts, cost_lines, cost_sheets
from coal_retrofit.optimization.results_workbook_sources import (
    SOURCE_HEADER,
    captured_by_year,
    plant_pathways,
    source_results,
    sources_frame,
    transmission_matrix,
)
from coal_retrofit.optimization.salvage import remaining_fraction
from coal_retrofit.optimization.scenario import PATHWAYS, OptimizationAssumptions, OptimizationScenario
from coal_retrofit.spatial import load_provinces
from test_h2_route_multiplier import _steel_hub

PROVINCES_SHP = Path(__file__).resolve().parents[1] / "data" / "ChinaMap" / "provinces.shp"


def _approx(traced: dict[tuple[str, str], float]) -> dict[tuple[str, str], object]:
    return {key: pytest.approx(value) for key, value in traced.items()}


def test_tracing_merge_split_and_mixing() -> None:
    """两个源在节点 J 汇流后分给两个汇：每个汇按 J 的来源构成分（各一半）；单源分流时按流量分。"""
    arcs = {("A", "J"): 2.0, ("B", "J"): 2.0, ("J", "S1"): 1.0, ("J", "S2"): 3.0}
    traced = trace_sources_to_sinks(arcs, {"A": 2.0, "B": 2.0}, {"S1": 1.0, "S2": 3.0})
    assert traced == _approx({("A", "S1"): 0.5, ("B", "S1"): 0.5, ("A", "S2"): 1.5, ("B", "S2"): 1.5})
    traced = trace_sources_to_sinks({("A", "J"): 3.0, ("J", "S1"): 1.0, ("J", "S2"): 2.0}, {"A": 3.0},
                                    {"S1": 1.0, "S2": 2.0})
    assert traced == _approx({("A", "S1"): 1.0, ("A", "S2"): 2.0})


def test_tracing_nets_opposite_arcs_and_cancels_cycles() -> None:
    """同一对节点两个方向的流量先相抵；有向环流（X→Y→Z→X 0.5）抵消后不影响结果；流经一个汇的 CO2 只有它在这里
    封存的那部分记给它。"""
    arcs = {("A", "X"): 3.0, ("X", "A"): 2.0, ("X", "Y"): 1.5, ("Y", "Z"): 0.5, ("Z", "X"): 0.5,
            ("Y", "S1"): 1.0, ("B", "S1"): 2.0, ("S1", "S2"): 1.0}
    traced = trace_sources_to_sinks(arcs, {"A": 1.0, "B": 2.0}, {"S1": 2.0, "S2": 1.0})
    # S1 流入 A 1、B 2，留下 2、送走 1，两者都按 1:2 分。
    assert traced == _approx({("A", "S1"): 2 / 3, ("B", "S1"): 4 / 3, ("A", "S2"): 1 / 3, ("B", "S2"): 2 / 3})
    by_source = {s: sum(v for (src, _), v in traced.items() if src == s) for s in "AB"}
    by_sink = {t: sum(v for (_, sink), v in traced.items() if sink == t) for t in ("S1", "S2")}
    assert by_source == _approx({"A": 1.0, "B": 2.0}) and by_sink == _approx({"S1": 2.0, "S2": 1.0})


def test_transmission_matrix_rows_are_source_provinces() -> None:
    """山西 2、河北 1 在 J 汇流，海上汇与山东汇各封存 1.5：行是源所在的省、列是汇所在的省，各格按 J 的来源构成
    （山西 2/3、河北 1/3）分；山东只有汇，那一行全空。"""
    flows = pd.DataFrame({"source_node": ["a", "b", "j", "j"], "sink_node": ["j", "j", "s1", "s2"],
                          "net_flow_mtpa": [2.0, 1.0, 1.5, 1.5]})
    sources = pd.DataFrame({"node": ["a", "b"], "captured_mt": [2.0, 1.0]})
    sinks = pd.DataFrame({"node": ["s1", "s2"], "use_mt": [1.5, 1.5]})
    province = {"a": "Shanxi", "b": "Hebei", "j": "Hebei", "s1": OFFSHORE, "s2": "Shandong"}
    rows = transmission_matrix(flows, sources, sinks, province)
    cells = {row[0]: dict(zip(rows[0][1:], row[1:])) for row in rows[1:]}
    assert cells["Shanxi"][OFFSHORE] == pytest.approx(1.0) and cells["Shanxi"]["Shandong"] == pytest.approx(1.0)
    assert cells["Hebei"][OFFSHORE] == pytest.approx(0.5) and cells["Hebei"]["Shandong"] == pytest.approx(0.5)
    assert cells["Shanxi"]["合计"] == pytest.approx(2.0) and cells["Hebei"]["合计"] == pytest.approx(1.0)
    assert cells["合计"][OFFSHORE] == pytest.approx(1.5) and cells["合计"]["合计"] == pytest.approx(3.0)
    assert all(value is None for value in cells["Shandong"].values())


def test_edge_region_rules() -> None:
    """两端同区记该区；一端在海上记陆上那一端；两端在不同的陆上分区记跨地理分区。"""
    assert edge_region(2, 2) == 2 and edge_region(OFFSHORE_REGION, OFFSHORE_REGION) == OFFSHORE_REGION
    assert edge_region(3, OFFSHORE_REGION) == 3 and edge_region(OFFSHORE_REGION, 6) == 6
    assert edge_region(2, 3) == CROSS_REGION


def test_region_codes() -> None:
    """分区号：ChinaCCS 的六大区（西藏归西南），本模型另设 7 海上，跨地理分区顺延为 8（ChinaCCS 的跨区是 7）。"""
    assert REGIONS == {1: "东北", 2: "华北", 3: "华东", 4: "西北", 5: "西南", 6: "中南", 7: "海上", 8: "跨地理分区"}
    assert (OFFSHORE_REGION, CROSS_REGION) == (7, 8)
    assert province_region(OFFSHORE) == 7 and province_region("Xizang") == 5 and province_region("Shandong") == 3


@pytest.mark.skipif(not PROVINCES_SHP.exists(), reason="needs data/ChinaMap/provinces.shp")
def test_province_layer_names_map_to_the_31_provinces() -> None:
    """省界图层的 34 个省级单位里，31 个对上 `PROVINCES`（各对一个），港澳台报错。"""
    names = load_provinces(PROVINCES_SHP)["province_name"].astype(str).tolist()
    excluded = {"香港特别行政区", "澳门特别行政区", "台湾省"}
    mapped = {name: _english(name) for name in names if name not in excluded}
    assert sorted(mapped.values()) == sorted(PROVINCES) and len(mapped) == 31
    assert mapped["内蒙古自治区"] == "Inner Mongolia" and mapped["西藏自治区"] == "Xizang"
    for name in excluded & set(names):
        with pytest.raises(ValueError, match="对不上"):
            _english(name)


def _prepared(nodes: list[tuple[str, float, float, object]]) -> SimpleNamespace:
    """node_provinces 读的那几样：煤电 P、工业 I、海上汇 S_off、填了省的汇 S_on、没填省的陆上汇 S_geo，加 *nodes*。"""
    fixed = [("n_P", 115.5, 38.9, None), ("n_I", 111.7, 40.8, None), ("n_S_off", 125.0, 30.0, None),
             ("n_S_on", 118.0, 37.5, None), ("n_S_geo", 125.0, 46.6, None)]
    table = pd.DataFrame([*fixed, *nodes], columns=["node_id", "lon", "lat", "province"])
    return SimpleNamespace(
        network=SimpleNamespace(nodes=table, plant_node_ids={"P": "n_P"}, industry_node_ids={"I": "n_I"},
                                storage_node_ids={"S_off": "n_S_off", "S_on": "n_S_on", "S_geo": "n_S_geo"}),
        plants=pd.DataFrame({"plant_id": ["P"], "province_name": ["Shanxi"]}),
        industry=SimpleNamespace(hubs=pd.DataFrame({"hub_id": ["I"], "province": ["Neimenggu"]})),
        storages=pd.DataFrame({"storage_hub_id": ["S_off", "S_on", "S_geo"], "province": [np.nan, "Shandong", np.nan],
                               "offshore": [True, False, False]}),
    )


@pytest.mark.skipif(not PROVINCES_SHP.exists(), reason="needs data/ChinaMap/provinces.shp")
def test_node_provinces_attributes_first_then_location() -> None:
    """源取源的省（落点在河北的山西 hub 仍记山西，拼音写法统一），海上汇记海上，汇与走廊节点有省名的取省名，
    其余按落点：北京、呼和浩特、拉萨、大庆各归其省，东海里的点记海上。"""
    prepared = _prepared([("c_attr", 116.4, 39.9, "Jiangsu"), ("c_bj", 116.4, 39.9, None),
                          ("c_sea", 125.0, 30.0, None), ("c_xz", 91.1, 29.65, None), ("c_nmg", 111.7, 40.8, None)])
    assert node_provinces(prepared, PROVINCES_SHP) == {  # type: ignore[arg-type]
        "n_P": "Shanxi", "n_I": "Inner Mongolia", "n_S_off": OFFSHORE, "n_S_on": "Shandong",
        "n_S_geo": "Heilongjiang", "c_attr": "Jiangsu", "c_bj": "Beijing", "c_sea": OFFSHORE, "c_xz": "Xizang",
        "c_nmg": "Inner Mongolia",
    }


def test_node_provinces_reads_the_layer_only_when_needed(tmp_path) -> None:
    """节点都有省名时不读图层；要按落点定省而图层不在就报错；省名不在 `PROVINCES` 里也报错。"""
    missing = tmp_path / "provinces.shp"

    def named(province: str) -> SimpleNamespace:
        prepared = _prepared([("c1", 116.4, 39.9, province)])
        prepared.storages.loc[2, "province"] = "Heilongjiang"
        return prepared

    assert node_provinces(named("Beijing"), missing)["n_S_geo"] == "Heilongjiang"  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Atlantis"):
        node_provinces(named("Atlantis"), missing)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="1 个管网节点没有省名"):
        node_provinces(_prepared([("c1", 116.4, 39.9, "Beijing")]), missing)  # type: ignore[arg-type]


def test_node_provinces_skips_sinks_the_scenario_does_not_use(tmp_path) -> None:
    """节点表里有、本情景不纳入的汇（`storage_scope = "dsa_only"` 去掉的 EOR 汇）不在 `prepared.storages` 里：与走廊
    节点同法定省，不报 KeyError。"""
    prepared = _prepared([("n_S_eor", 117.0, 38.0, "Hebei")])
    prepared.network.storage_node_ids["S_eor"] = "n_S_eor"
    prepared.storages.loc[2, "province"] = "Heilongjiang"
    assert node_provinces(prepared, tmp_path / "provinces.shp")["n_S_eor"] == "Hebei"  # type: ignore[arg-type]


@pytest.mark.skipif(not PROVINCES_SHP.exists(), reason="needs data/ChinaMap/provinces.shp")
def test_unused_sinks_without_a_province_are_located() -> None:
    """本情景不纳入、节点表也没有省名的汇按落点定省：东海里的记海上，北京的记北京。真实数据里 dsa_only 去掉的 58 个
    EOR 汇都走这条路，其中带海上标记的 4 个落在省界外，记海上。"""
    prepared = _prepared([("n_S_sea", 125.0, 30.0, None), ("n_S_land", 116.4, 39.9, None)])
    prepared.network.storage_node_ids.update(S_sea="n_S_sea", S_land="n_S_land")
    province = node_provinces(prepared, PROVINCES_SHP)  # type: ignore[arg-type]
    assert (province["n_S_sea"], province["n_S_land"]) == (OFFSHORE, "Beijing")


def test_tier_counts_round_trip() -> None:
    """在役根数写成 "根数x档容量" 串（`_tier_strings`）再拆回来，与原数相同；空串是零根。"""
    counts = np.array([[1, 0, 2], [0, 0, 0], [0, 3, 0]], dtype=float)
    labels = ["2", "5", "20"]
    strings = _tier_strings(counts, (2.0, 5.0, 20.0))
    assert strings == ["1x2|2x20", "", "3x5"]
    assert [_tier_counts(text, labels) for text in strings] == counts.astype(int).tolist()
    assert _tier_counts(float("nan"), labels) == [0, 0, 0]  # 读回的 CSV 里空串是 NaN


def test_industry_cost_columns_split_by_route() -> None:
    """工业明细的四列各归各的路线：捕集列是 CCS 路线的投资与年度费（能耗耗材 + 在用能力的固定运维），氢列是氢路线的
    投资与年度费（运行差额 + 本年买的氢）；两两相加即 cost_capital_cny、cost_annual_cny。"""
    industry = _steel_hub()
    data = industry_year_data(industry, OptimizationScenario(experiment_id="T", description="toy"),
                              OptimizationAssumptions(), 2030)
    share, new = np.array([[0.2, 0.3, 0.5]]), np.array([[0.0, 0.4, 0.6]])
    link_cost, flow = 30.0, 4.0e7  # CNY/kg、kg
    year_data = SimpleNamespace(industry_h2_hub_membership=sparse.csr_matrix(np.ones((1, 1))),
                                industry_h2_link_cost_cny_per_kg=np.array([link_cost * AMMONIA_FLOW_SCALE]))
    row = _build_industry_detail_table(
        SimpleNamespace(industry=industry), 2030, data, share, np.array([flow]), year_data,  # type: ignore[arg-type]
        capacity_mt=new, new_capacity_mt=new, ccs_fixed_om_cny=np.array([7.0e6]),
    ).iloc[0]
    annual_h2 = data.opex_cny[0, H2] * 0.5 + flow * link_cost
    assert annual_h2 > 0.0 and data.opex_cny[0, CCS] > 0.0  # 前提：两条路线的年度费都不为零
    assert row["cost_capital_ccs_cny"] == pytest.approx(data.capex_cny_per_mt[0, CCS] * 0.4)
    assert row["cost_capital_h2_cny"] == pytest.approx(data.capex_cny_per_mt[0, H2] * 0.6)
    assert row["cost_annual_ccs_cny"] == pytest.approx(data.opex_cny[0, CCS] * 0.3 + 7.0e6)
    assert row["cost_annual_h2_cny"] == pytest.approx(annual_h2)
    assert row["cost_capital_cny"] == pytest.approx(row["cost_capital_ccs_cny"] + row["cost_capital_h2_cny"])
    assert row["cost_annual_cny"] == pytest.approx(row["cost_annual_ccs_cny"] + row["cost_annual_h2_cny"])


def test_capture_costs_take_the_ccs_columns_and_the_industry_salvage() -> None:
    """Source_Results 与 Cost_Capture 的工业部分只取捕集路线的列，不含氢路线；各年捕集量是煤电与工业之和；全期每吨
    成本按定义复算：投资乘一次性权重、运行费乘年度权重，扣煤电捕集岛与工业捕集两项的期末残值（不扣氢路线的），除以
    捕集量乘年度权重之和。平均运程取净流量的绝对值。"""
    years, end_year = [2050, 2060], 2070
    plant_detail = pd.DataFrame({"year": years, "plant_id": "P", "province_name": "Shanxi",
                                 "baseline_emissions_mt": 10.0, "captured_mt": [4.0, 5.0]})
    plant_cost = pd.DataFrame({"year": years, "plant_id": "P", "ccs_retrofit_capex_cny": [3e9, 1e9],
                               "ccs_om_cny": 2e8, "energy_penalty_cny": 1e8})
    hubs = pd.DataFrame({"year": years, "hub_id": "C", "sector": SECTOR_STEEL_BF, "province": "Hebei",
                         "baseline_co2_mt": 2.0, "captured_mt": [1.0, 1.5],
                         "cost_capital_ccs_cny": [0.0, 8e8], "cost_capital_h2_cny": [5e8, 3e8],
                         "cost_annual_ccs_cny": [6e7, 9e7], "cost_annual_h2_cny": 4e7})
    hubs = hubs.assign(cost_capital_cny=hubs["cost_capital_ccs_cny"] + hubs["cost_capital_h2_cny"],
                       cost_annual_cny=hubs["cost_annual_ccs_cny"] + hubs["cost_annual_h2_cny"])
    breakdown = pd.DataFrame({"year": years * 2, "category": ["pipe_capex"] * 2 + ["transport_opex"] * 2,
                              "cost_undiscounted_cny": [7e8, 0.0, 5e7, 6e7]})
    tables = {"plant_detail.csv": plant_detail, "plant_cost.csv": plant_cost, "industry_detail.csv": hubs,
              "cost_breakdown.csv": breakdown}
    prepared = SimpleNamespace(network=SimpleNamespace(plant_node_ids={"P": "n_P"}, industry_node_ids={"C": "n_C"}))
    sources = sources_frame(tables, prepared)  # type: ignore[arg-type]
    left = source_results(sources[sources["year"] == 2060])[0][2]
    by_id = {row[0]: dict(zip(SOURCE_HEADER[1:], row[1:])) for row in left[1:]}
    assert by_id["C"]["CAPEX_Capture"] == pytest.approx(800.0) and by_id["C"]["OPEX_Capture_i"] == pytest.approx(90.0)
    assert by_id["P"]["CAPEX_Capture"] == pytest.approx(1000.0) and by_id["P"]["OPEX_Capture_p"] == pytest.approx(300.0)

    sinks = pd.DataFrame({"year": years, "use_mt": [5.0, 6.5], "cost_before_credit_cny_per_t": 35.0,
                          "cost_cny_per_t": [35.0, 29.0]})
    lines = cost_lines(tables, sinks, years)
    weights = {2050: (3.0, 0.5), 2060: (2.0, 0.3)}  # (年度项, 一次性项)
    df_end = 0.2
    solution = {"year_solutions": {
        y: {"cost_weights": {"opex": ("annual", a), "capex": ("one_off", o), "salvage_credit": ("horizon_end", df_end)},
            "salvage_ledger": [("ccs_retrofit_capex", plant_cost.loc[k, "ccs_retrofit_capex_cny"], 20),
                               ("industry_ccs_capex", hubs.loc[k, "cost_capital_ccs_cny"], 20),
                               ("industry_h2_capex", hubs.loc[k, "cost_capital_h2_cny"], 25)]}
        for k, (y, (a, o)) in enumerate(weights.items())
    }}
    edges = pd.DataFrame({"year": years, "net": [5.0, -6.5], "length_km": 100.0})
    captured = captured_by_year(sources, years)
    assert captured == pytest.approx({2050: 5.0, 2060: 6.5})  # 煤电 4.0、5.0 加工业 1.0、1.5
    sheets = cost_sheets(lines, captured, edges, solution, years, end_year)
    capture = {row[0]: row[1:] for row in sheets["Cost_Capture"][0][2][1:]}
    assert capture["工业捕集投资"] == pytest.approx([0.0, 800.0]) and capture["工业捕集年度费"] == pytest.approx([60.0, 90.0])

    present = sum(o * (plant_cost.loc[k, "ccs_retrofit_capex_cny"] + hubs.loc[k, "cost_capital_ccs_cny"])
                  + a * (3e8 + hubs.loc[k, "cost_annual_ccs_cny"]) for k, (a, o) in enumerate(weights.values()))
    present -= df_end * remaining_fraction(2060, 20, end_year) * (1e9 + 8e8)  # 2050 年建的到 2070 年已折完
    tonnes = sum(a * captured[y] * 1e6 for y, (a, _) in weights.items())
    analysis = sheets["Cost_Analysis"][0][2]
    start = next(i for i, row in enumerate(analysis) if row and str(row[0]).startswith("全期每吨成本"))
    horizon = {row[0]: row[1] for row in analysis[start + 2:start + 6]}
    assert horizon["捕集环节"] == pytest.approx(present / tonnes, rel=1e-12)
    assert next(row for row in analysis if row and row[0] == "平均运程")[1:] == pytest.approx([100.0, 100.0])

    # 捕集量在 FLOW_TOL（Mt）以内是数值噪声：该年每吨的数记空，不拿 0.5 t 去除。
    noise = cost_sheets(lines, {2050: 5e-7, 2060: 6.5}, edges, solution, years, end_year)["Cost_Analysis"][0][2]
    assert next(row for row in noise if row and row[0] == "全流程")[1] is None
    assert next(row for row in noise if row and row[0] == "平均运程")[1] is None


def test_plant_pathways_air_shares_are_whole_plant() -> None:
    """Plant_Pathways 的空冷两列是占全厂的份额：`plant_detail.csv` 的改造进度乘 1 − already_air_share；已全空冷的 hub
    为零（明细表的这两列在那里是求解器任取的数）。"""
    plants = pd.DataFrame({"year": 2050, "plant_id": ["P1", "P2"], "province_name": ["Shanxi", "Hebei"],
                           "already_air_share": [0.25, 1.0], "air_operating_share": [0.4, 0.7],
                           "air_installed_share": [0.6, 0.9], **{f"share_{pw}": 0.0 for pw in PATHWAYS}})
    rows = plant_pathways({"plant_detail.csv": plants}, [2050])
    table = {row[0]: dict(zip(rows[0][1:], row[1:])) for row in rows[1:]}
    assert table["P1"]["already_air"] == 0.25
    assert table["P1"]["air_retrofit_operating_2050"] == pytest.approx(0.3)
    assert table["P1"]["air_retrofit_installed_2050"] == pytest.approx(0.45)
    assert table["P2"]["air_retrofit_operating_2050"] == 0.0 and table["P2"]["air_retrofit_installed_2050"] == 0.0
