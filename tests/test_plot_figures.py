"""出图脚本（scripts/plot_fig1 … plot_fig10 与共用的 plot_style、plot_map、plot_io）的数据函数与出图前自检：不画整张图、不求解，不需要 Gurobi。

各图的自检是"图上要画的量与模型记的量对不上就报错、不出图"，这里用最小的表造出对得上与对不上两种情形；
另查共用模块的五件事：旧模型分段的结果读不进来，同值的圆点、菱形、方块面积相同，本机没有所用字体时 save_fig 拒绝出图，没有文件头的脚本
也能建命令行，plot_io 不导入 matplotlib；最后一条查底图：南海小图不压台湾与大陆沿海，图例压到国土会报错（要 geopandas 与仓库里的 data/ChinaMapTHT）。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib", reason="出图脚本依赖 matplotlib")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_fig1_checks_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺坐标或排放的源、不认识的部门、不认识的封存汇类型都拒绝出图（缺值的点会被 scatter 静默丢掉）。"""
    fig1 = _load(monkeypatch, "plot_fig1_sources_sinks")
    data = {"sources": pd.DataFrame({"sector": ["coal", "cement"], "lon": [110.0, 115.0], "lat": [30.0, 35.0],
                                     "co2_mt": [5.0, 1.0]}),
            "sinks": pd.DataFrame({"storage_type": ["dsa", "eor"], "longitude": [100.0, 120.0],
                                   "latitude": [40.0, 38.0]})}
    fig1.check(data)
    with pytest.raises(ValueError, match="缺坐标"):
        fig1.check({**data, "sources": data["sources"].assign(lat=[30.0, np.nan])})
    with pytest.raises(ValueError, match="SECTOR_ORDER"):
        fig1.check({**data, "sources": data["sources"].assign(sector=["coal", "glass"])})
    with pytest.raises(ValueError, match="dsa / eor"):
        fig1.check({**data, "sinks": data["sinks"].assign(storage_type=["dsa", "ecbm"])})


def test_fig2_edge_classes_and_industry_sectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """四类边各归各类，不认识的 edge_class 报错（否则少画而图例条数照样对得上）；工业 hub 配不上部门也报错。
    边色取 Blues，不与部门色相撞。"""
    fig2 = _load(monkeypatch, "plot_fig2_candidate_network")
    edges = pd.DataFrame({"edge_class": ["existing_main_corridor", "hub_to_corridor_branch",
                                         "corridor_to_storage_branch", "runtime_direct_fallback",
                                         "triangulation_candidate"]})
    data = {"edges": edges, "storage": pd.DataFrame({"storage_type": ["dsa", "eor"]}),
            "industry": pd.DataFrame({"sector": ["cement", "methanol"]})}
    groups = fig2.classify(edges)
    assert {k: len(v) for k, v in groups.items()} == {"triangulation": 1, "branch": 2, "direct": 1, "corridor": 1}
    fig2.check(data, groups)
    odd = pd.concat([edges, pd.DataFrame({"edge_class": ["ferry"]})], ignore_index=True)
    with pytest.raises(ValueError, match="ferry"):
        fig2.check({**data, "edges": odd}, fig2.classify(odd))
    with pytest.raises(ValueError, match="部门"):
        fig2.check({**data, "industry": pd.DataFrame({"sector": ["cement", np.nan]})}, groups)
    edge_colors = {style[1] for style in fig2.EDGE_STYLE.values()}
    assert not edge_colors & set(fig2.SECTOR_COLORS.values())


def test_fig2_load_takes_industry_sector_from_hub_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """工业 hub 的部门按管网节点表的 industry_hub_id 从 industry_hubs.csv 取；同一个 hub 在部门表里出现两次就报错。"""
    fig2 = _load(monkeypatch, "plot_fig2_candidate_network")
    tables = {
        "pipeline_nodes": pd.DataFrame({"node_id": ["n1", "n2", "n3"], "lon": [110.0, 115.0, 120.0],
                                        "lat": [30.0, 35.0, 38.0], "node_type": ["industry_hub", "plant", "storage_hub"],
                                        "industry_hub_id": ["cement_000", np.nan, np.nan],
                                        "storage_hub_id": [np.nan, np.nan, "S1"]}),
        "pipeline_candidate_edges": pd.DataFrame({"edge_class": ["triangulation_candidate"]}),
        "storage_hubs": pd.DataFrame({"storage_hub_id": ["S1"], "storage_type": ["dsa"]}),
        "industry_hubs": pd.DataFrame({"hub_id": ["cement_000", "steel_eaf_000"], "sector": ["cement", "steel_eaf"]}),
    }
    monkeypatch.setattr(fig2, "read_input", lambda name, sinks="": tables[name].copy())
    data = fig2.load()
    assert data["industry"]["sector"].tolist() == ["cement"]
    tables["industry_hubs"] = pd.concat([tables["industry_hubs"]] * 2, ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        fig2.load()


def test_fig3_capacity_split_and_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    fig3 = _load(monkeypatch, "plot_fig3_power_pathways")
    shares = {"unabated": 0.5, "ccs": 0.3, "biomass": 0.0, "beccs": 0.0, "ammonia": 0.0, "retire": 0.2}
    detail = pd.DataFrame([{"year": 2030, "source_id": "P1", "capacity_mw": 2000.0, "reduction_co2_mtpa": 1.5,
                            **{f"share_{k}": v for k, v in shares.items()}}])
    pathways = pd.DataFrame([{"year": 2030, "source_id": "P1", "route": k, "reduction_co2_mtpa": a}
                             for k, a in (("unabated", 0.0), ("ccs", 0.9), ("retire", 0.6))])
    sanity = pd.DataFrame({"check_name": ["pathway_split_closure"], "status": ["pass"]})
    data = {"detail": detail, "pathways": pathways, "sanity": sanity,
            "result": {"years": {"2030": {"coal_reduction_mt": 1.5}}}}
    fig3.check(data)
    capacity = fig3.capacity_by_pathway(detail)
    assert capacity.loc[2030, "ccs"] == pytest.approx(0.6)
    assert capacity.loc[2030].sum() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="份额之和"):
        fig3.check({**data, "detail": detail.assign(share_ccs=0.4)})
    with pytest.raises(ValueError, match="reduction_co2_mtpa"):
        fig3.check({**data, "pathways": pathways.assign(reduction_co2_mtpa=pathways["reduction_co2_mtpa"] * 2)})
    with pytest.raises(ValueError, match="重解"):          # 旧结果：拆分是旧口径，没有闭合行
        fig3.check({**data, "sanity": sanity.iloc[:0]})
    with pytest.raises(ValueError, match="对不上"):
        fig3.check({**data, "sanity": sanity.assign(status="warn")})


def test_fig4_counts_sources_left_unrebuilt_as_unabated(monkeypatch: pytest.MonkeyPatch) -> None:
    """混合原料的甲醇 hub 全部转氢时只有份额为正的点源改造（产量的 60%），其余计入未改造；结果表没有这一列
    （2026-10-02 前的结果）时按 1；比例缺失或不在 [0, 1] 内时自检不通过。"""
    fig4 = _load(monkeypatch, "plot_fig4_industry_routes")
    detail = pd.DataFrame({
        "year": [2030, 2030], "sector": ["methanol", "methanol"], "activity": [1000.0, 500.0],
        "abatable_production_share": [0.6, 1.0],
        "share_unabated": [0.0, 1.0], "share_ccs": [0.0, 0.0], "share_h2": [1.0, 0.0],
    })
    fig4.check(detail)
    for bad in ([np.nan, 1.0], [1.5, 1.0]):
        with pytest.raises(ValueError, match="产量比例"):
            fig4.check(detail.assign(abatable_production_share=bad))
    shares = fig4.route_shares(detail)["methanol"].loc[2030]
    assert shares[["unabated", "ccs", "h2"]].tolist() == pytest.approx([60.0, 0.0, 40.0])
    legacy = fig4.route_shares(detail.drop(columns="abatable_production_share"))["methanol"].loc[2030]
    assert legacy[["unabated", "ccs", "h2"]].tolist() == pytest.approx([100.0 / 3.0, 0.0, 200.0 / 3.0])


def _fig5_data(residual_2060: float, cap_fraction: float, shortfall: float, years=(2030, 2060)) -> dict:
    """电力一组：冻结技术排放 2030 年 10、2060 年 6（利用小时下降），残余 2030 年 9。"""
    residual = {2030: 9.0, 2060: residual_2060}
    baseline = {2030: 10.0, 2060: 6.0}
    plants = pd.DataFrame([{"year": y, "baseline_co2_mtpa": baseline[y], "reduction_co2_mtpa": baseline[y] - residual[y]}
                           for y in years])
    industry = pd.DataFrame(columns=["year", "target_group", "baseline_co2_mtpa", "residual_co2_mtpa"])
    result = {"years": {str(y): {
        "coal_baseline_mt": baseline[y], "coal_residual_mt": residual[y],
        "sector_cap_fraction": {"power": 1.0 if y == 2030 else cap_fraction},
        "target_shortfall_by_group_mt": {"power": 0.0 if y == 2030 else shortfall},
        "industry": {"baseline_mt": 0.0, "residual_by_group_mt": {}},
    } for y in years}}
    return {"plants": plants, "industry": industry, "result": result}


def test_fig5_cap_is_fraction_of_2030_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """上限 = 比例 × 2030 年冻结技术排放（不是当年的：2060 年按当年算是 1.8）；目标缺口 = max(0, 残余 − 上限)，
    上限算低（缺口不够）、算高（有缺口而残余在上限以内）都报错；没有 2030 年算不出上限。"""
    fig5 = _load(monkeypatch, "plot_fig5_sector_targets")
    data = _fig5_data(residual_2060=4.0, cap_fraction=0.3, shortfall=1.0)
    table = fig5.sector_table(data)
    assert table.set_index("year").loc[2060, "cap"] == pytest.approx(3.0)
    fig5.check(data, table)
    unmet = _fig5_data(residual_2060=4.0, cap_fraction=0.3, shortfall=0.0)
    with pytest.raises(ValueError, match="目标缺口"):
        fig5.check(unmet, fig5.sector_table(unmet))
    slack = _fig5_data(residual_2060=4.0, cap_fraction=0.5, shortfall=1.0)
    with pytest.raises(ValueError, match="目标缺口"):
        fig5.check(slack, fig5.sector_table(slack))
    with pytest.raises(ValueError, match="2030"):
        fig5.sector_table(_fig5_data(residual_2060=4.0, cap_fraction=0.3, shortfall=1.0, years=(2060,)))
    pooled = _fig5_data(residual_2060=4.0, cap_fraction=0.3, shortfall=1.0)   # 合计总量上限：缺口只记在 total 下
    for year in pooled["result"]["years"].values():
        year["target_shortfall_by_group_mt"] = {"total": year["target_shortfall_by_group_mt"]["power"]}
    with pytest.raises(ValueError, match="合计总量上限"):
        fig5.sector_table(pooled)


def test_fig6_mass_balance_and_legend_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    fig6 = _load(monkeypatch, "plot_fig6_co2_network")
    data = {"edges": pd.DataFrame({"year": [2040], "edge_id": ["E1"], "flow_mtpa": [3.0]}),
            "plants": pd.DataFrame({"year": [2040], "captured_co2_mtpa": [2.0]}),
            "industry": pd.DataFrame({"year": [2040], "sector": ["cement"], "captured_co2_mtpa": [1.0]}),
            "storage": pd.DataFrame({"year": [2040], "storage_type": ["dsa"], "injected_mtpa": [3.0]})}
    fig6.check(data, (2040,))
    with pytest.raises(ValueError, match="守恒"):
        fig6.check({**data, "storage": data["storage"].assign(injected_mtpa=2.5)}, (2040,))
    with pytest.raises(ValueError, match="dsa / eor"):        # 封存汇类型来自 sinks.csv，不认识的画不出颜色
        fig6.check({**data, "storage": data["storage"].assign(storage_type="ccs")}, (2040,))
    with pytest.raises(ValueError, match="2060"):
        fig6.check(data, (2040, 2060))
    with pytest.raises(ValueError, match="SECTOR_ORDER"):     # 不认识的部门画不出来，守恒照样对得上
        fig6.check({**data, "industry": data["industry"].assign(sector="glass")}, (2040,))
    assert fig6.nice_levels(87.0) == [1.0, 10.0, 50.0]


def test_fig6_draws_idle_in_service_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    """当年无流量、但在役容量为正的管段另成一层（浅蓝细线）；没有容量也没有流量的候选边不画。"""
    fig6 = _load(monkeypatch, "plot_fig6_co2_network")
    edges = pd.DataFrame({"year": 2040, "edge_id": ["E1", "E2", "E3", "E4"], "flow_mtpa": [3.0, 0.0, 0.0, 5e-4],
                          "total_capacity_mtpa": [4.0, 2.0, 0.0, 2.0]})
    empty = pd.DataFrame({"year": [], "captured_co2_mtpa": [], "injected_mtpa": []})
    candidates = pd.DataFrame({"edge_id": ["E1", "E2", "E4"],
                               "geometry_wkt": ["LINESTRING (110 30, 111 31)", "LINESTRING (112 32, 113 33)", ""]})
    layer = fig6.year_layers({"edges": edges, "candidates": candidates, "plants": empty, "industry": empty,
                              "storage": empty}, 2040)
    assert layer["edges"]["edge_id"].tolist() == ["E1"]
    assert layer["idle"]["edge_id"].tolist() == ["E2", "E4"]          # 流量低于 FLOW_MIN 的在役管段也算闲置
    assert layer["idle"]["geometry_wkt"].tolist()[0].startswith("LINESTRING (112")   # 闲置管段也按 WKT 折线画


def test_fig8_industry_abatement_by_sector_and_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """逐 hub 各路线的减排与捕集按部门汇总；各路线之和对不上 sources.csv、合计对不上 result.json、未改造路线有减排
    都报错。"""
    fig8 = _load(monkeypatch, "plot_fig8_industry_abatement")
    hubs = pd.DataFrame({"year": 2040, "source_id": ["A1", "C1"], "sector": ["ammonia", "cement"],
                         "reduction_co2_mtpa": [3.0, 2.0], "captured_co2_mtpa": [1.5, 2.4]})
    routes = pd.DataFrame({"year": 2040, "source_id": ["A1", "A1", "A1", "C1", "C1", "C1"],
                           "route": ["unabated", "ccs", "h2"] * 2,
                           "reduction_co2_mtpa": [0.0, 1.0, 2.0, 0.0, 2.0, 0.0],
                           "captured_co2_mtpa": [0.0, 1.5, 0.0, 0.0, 2.4, 0.0]})
    result = {"years": {"2040": {"industry": {"reduction_mt": 5.0, "captured_mt": 3.9}}}}
    data = {"hubs": hubs, "routes": routes, "result": result}
    fig8.check(data)
    sectors = fig8.by_sector(data)
    assert list(sectors) == ["cement", "ammonia"]                        # SECTOR_ORDER 的顺序
    assert sectors["ammonia"]["reduction"].loc[2040].tolist() == pytest.approx([1.0, 2.0])   # CCS、氢路线
    assert sectors["ammonia"]["captured"].loc[2040] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="reduction_co2_mtpa"):
        fig8.check({**data, "hubs": hubs.assign(reduction_co2_mtpa=[3.5, 2.0])})
    with pytest.raises(ValueError, match="captured_mt"):
        fig8.check({**data, "result": {"years": {"2040": {"industry": {"reduction_mt": 5.0, "captured_mt": 3.0}}}}})
    leaky = routes.assign(reduction_co2_mtpa=[0.5, 0.5, 2.0, 0.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="未改造"):
        fig8.check({**data, "routes": leaky})
    with pytest.raises(ValueError, match="captured_co2_mtpa"):
        fig8.check({**data, "hubs": hubs.assign(captured_co2_mtpa=[1.5, 2.0])})
    with pytest.raises(ValueError, match="ROUTE_ORDER"):
        fig8.check({**data, "routes": routes.assign(route=["unabated", "ccs", "biochar"] * 2)})
    stray = pd.concat([routes, routes.iloc[:1].assign(source_id="X9")], ignore_index=True)
    with pytest.raises(ValueError, match="sources.csv 里没有"):
        fig8.check({**data, "routes": stray})
    with pytest.raises(ValueError, match="没有工业 hub"):
        fig8.check({**data, "hubs": hubs.iloc[:0], "routes": routes.iloc[:0]})


def test_fig9_storage_by_type_and_offshore(monkeypatch: pytest.MonkeyPatch) -> None:
    """年注入与累计封存按汇型 × 陆海汇总（累计量换成 Gt）；注入不等于捕集、海上标记不认识都报错（全国部署上限 2026-10-14 删，
    不再查）。"""
    fig9 = _load(monkeypatch, "plot_fig9_storage")
    sinks = pd.DataFrame({"year": 2040, "storage_type": ["dsa", "dsa", "eor"], "offshore": [False, True, False],
                          "injected_mtpa": [10.0, 4.0, 1.0], "cumulative_injected_mt": [100.0, 40.0, 10.0]})
    data = {"sinks": sinks, "captured": pd.Series({2040: 15.0})}
    fig9.check(data)
    tables = fig9.summary(sinks)
    assert tables["annual"].loc[2040].tolist() == pytest.approx([10.0, 4.0, 1.0, 0.0])
    assert tables["cumulative"].loc[2040, "dsa_offshore"] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="守恒"):
        fig9.check({**data, "captured": pd.Series({2040: 14.0})})
    with pytest.raises(ValueError, match="offshore"):
        fig9.check({**data, "sinks": sinks.assign(offshore=["no", "yes", "no"])})


def test_fig10_cost_groups_and_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    """成本类别归成十类：年度费用与一次性投资按不折现值，折现值合计等于目标函数；类别没归类、成本对账没通过、合计对不上
    目标函数都报错。"""
    fig10 = _load(monkeypatch, "plot_fig10_costs")
    system = pd.DataFrame({
        "year": [2030, 2030, 2030, 2030, 2060],
        "category": ["coal_operating_delta", "coal_savings_credit", "ccs_retrofit_capex", "storage_cost",
                     "salvage_credit"],
        "kind": ["annual", "annual", "one_off", "annual", "horizon_end"],
        "cost_cny": [5e9, -2e9, 8e9, 1e9, -4e9],
        "cost_discounted_cny": [40e9, -16e9, 8e9, 8e9, -1e9],
    })
    checks = pd.DataFrame({"check_name": ["cost_attribution_closure"], "status": ["pass"]})
    data = {"system": system, "checks": checks, "result": {"global_objective_cny": 39e9}}
    fig10.check(data)
    tables = fig10.grouped(system)
    assert tables["annual"].loc[2030].to_dict() == pytest.approx({"coal_operating": 3.0, "storage": 1.0})
    assert tables["annual"].loc[2060].sum() == 0.0                       # 期末残值（horizon_end）不进年度费用
    assert tables["one_off"].loc[2030, "coal_capex"] == pytest.approx(8.0)
    assert tables["present"]["salvage"] == pytest.approx(-1.0)
    assert float(tables["present"].sum()) == pytest.approx(39.0)
    assert set(fig10.CATEGORY_GROUP.values()) == set(fig10.COST_ORDER)
    with pytest.raises(ValueError, match="归类"):
        fig10.check({**data, "system": system.assign(category=["mystery", *system["category"][1:]])})
    with pytest.raises(ValueError, match="cost_attribution_closure"):
        fig10.check({**data, "checks": checks.assign(status="fail")})
    with pytest.raises(ValueError, match="global_objective_cny"):
        fig10.check({**data, "result": {"global_objective_cny": 40e9}})


def test_fig7_utilization_and_converted_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """余量 ≤ 0 而仍有取水记为无余量（inf），按 used / available 重算、不读结果表的 utilization（旧结果表
    在这里记 0，夹具即按旧表写）；超出余量的取水要与 resources 的 slack 列对上；超了但四舍五入是 100% 的格子写 >100%；
    空冷量只算当年在运行、仍湿冷的部分。"""
    fig7 = _load(monkeypatch, "plot_fig7_water")
    basins = pd.DataFrame({"year": 2030, "node_id": ["D", "C", "E"], "used": [9.0, 5.0e6, 0.0],
                           "available": [10.0, -1.0e6, 0.0], "slack": [0.0, 6.0e6, 0.0],
                           "utilization": [0.9, 0.0, 0.0]})
    table = fig7.utilization(basins)
    assert list(table.index) == ["C", "D", "E"]
    assert np.isinf(table.loc["C", 2030])
    assert table.loc["D", 2030] == pytest.approx(0.9)
    assert table.loc["E", 2030] == 0.0
    text = fig7.TEXT["en"]
    assert [fig7.cell_label(v, text) for v in (0.904, 1.002, 1.2, np.inf, np.nan)] == \
        ["90%", ">100%", "120%", text["none"], text["nan"]]
    plants = pd.DataFrame({"year": [2040], "capacity_mw": [1000.0], "air_operating_share": [0.5],
                           "already_air_share": [0.4]})
    assert fig7.converted_gw(plants).loc[2040] == pytest.approx(0.3)
    fig7.check({"basins": basins, "plants": plants})
    with pytest.raises(ValueError, match="松弛"):
        fig7.check({"basins": basins.assign(slack=0.0), "plants": plants})
    with pytest.raises(ValueError, match="BASIN_ORDER"):
        fig7.check({"basins": basins.assign(node_id="B1"), "plants": plants})
    old = plants.rename(columns={"air_operating_share": "air_cooled_share"})   # 2026-09-30 之前落盘的结果
    with pytest.raises(ValueError, match="重解"):
        fig7.check({"basins": basins, "plants": old})


def test_same_value_gives_same_marker_area(monkeypatch: pytest.MonkeyPatch) -> None:
    """scatter 画出来的面积 = s × marker 路径自身的面积：圆点路径面积 π/4，方块、菱形是 1。`area_scale` 按形状换算后，
    同值的三种点面积相同（此前菱形、方块比圆点大 4/π ≈ 1.27 倍）；尺寸图例用同一换算。"""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import plot_style
    from matplotlib.markers import MarkerStyle

    def drawn_area(marker: str, size: float) -> float:
        style = MarkerStyle(marker)
        xy = style.get_path().transformed(style.get_transform()).to_polygons()[0]
        return size * 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))

    areas = {m: drawn_area(m, float(plot_style.area_scale([3.0], 10.0, marker=m)[0])) for m in ("o", "D", "s")}
    assert areas["D"] == pytest.approx(areas["o"], rel=2e-3)
    assert areas["s"] == pytest.approx(areas["o"], rel=2e-3)
    handle = plot_style.size_legend([3.0], 10.0, "D", "grey")[0]
    assert handle.get_markersize() ** 2 == pytest.approx(float(plot_style.area_scale([3.0], 10.0, marker="D")[0]))


def test_results_of_an_old_model_segment_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`--scenario` 给路径时读那里；result.json 没有记模型分段号或分段号不是当前代码的，读结果表之前就拒绝。"""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import json

    import plot_io
    from coal_retrofit.segment import MODEL_SEGMENT, StaleResultError

    (tmp_path / "R").mkdir()
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "R" / "t.csv", index=False, encoding="utf-8-sig")
    for segment, ok in ((None, False), (MODEL_SEGMENT - 1, False), (MODEL_SEGMENT, True)):
        resolved = {} if segment is None else {"model_segment": segment}
        (tmp_path / "R.json").write_text(json.dumps({"resolved": resolved}), encoding="utf-8")
        plot_io.read_result_json.cache_clear()
        if ok:
            assert plot_io.read_result(str(tmp_path / "R"), "t").columns.tolist() == ["a"]
        else:
            with pytest.raises(StaleResultError, match="重解"):
                plot_io.read_result(str(tmp_path / "R"), "t")
    assert plot_io.result_dir("ST_BASE") == plot_io.SOLVED_DIR / "ST_BASE"


def test_plot_io_reads_the_sink_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """汇表与管网按汇变体读带后缀的文件，别的表给了变体就报错；结果图的变体取求解时记的 `resolved.scenario.sink_variant`
    （这一项之前落盘的结果没有，即缺省汇表）；图名加 `_<变体>`。"""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import json

    import plot_io
    from coal_retrofit.segment import MODEL_SEGMENT

    monkeypatch.setattr(plot_io, "INPUTS_DIR", tmp_path)
    pd.DataFrame({"v": ["default"]}).to_csv(tmp_path / "storage_hubs.csv", index=False)
    pd.DataFrame({"v": ["average"]}).to_csv(tmp_path / "storage_hubs_300km_average.csv", index=False)
    assert plot_io.read_input("storage_hubs")["v"].tolist() == ["default"]
    assert plot_io.read_input("storage_hubs", "300km_average")["v"].tolist() == ["average"]
    with pytest.raises(ValueError, match="industry_hubs 不随汇变体变"):
        plot_io.read_input("industry_hubs", "300km_average")
    assert plot_io.variant_name("fig2", "") == "fig2" and plot_io.variant_name("fig2", "300km_average") == "fig2_300km_average"
    for scenario, expected in (({}, ""), ({"sink_variant": "300km_complete"}, "300km_complete")):
        resolved = {"model_segment": MODEL_SEGMENT, "scenario": scenario}
        (tmp_path / "R.json").write_text(json.dumps({"resolved": resolved}), encoding="utf-8")
        plot_io.read_result_json.cache_clear()
        assert plot_io.result_sink_variant(str(tmp_path / "R")) == expected
    plot_io.read_result_json.cache_clear()


def test_plot_io_does_not_import_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """`plot_io` 不导入 matplotlib（`render_version.py` 归档时只从它取 `SOLVED_DIR`），在新进程里查；它的 `LANGS` 与
    `plot_style.FONTS` 的语言一一对应。"""
    code = "import sys, plot_io; assert not [m for m in sys.modules if m.startswith('matplotlib')]"
    subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS, check=True)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import plot_io
    import plot_style

    assert set(plot_io.LANGS) == set(plot_style.FONTS)


def test_save_fig_refuses_a_missing_font(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """图里文字用的字体本机没有时 save_fig 报错、一个文件都不写（matplotlib 自己会静默换成 DejaVu Sans 照常出图）；
    字体在就照常出图。没有文件头的脚本也能建命令行。"""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import matplotlib.pyplot as plt
    import plot_io
    import plot_style

    def figure(family: str):
        with plt.rc_context({"font.family": family}):
            fig, ax = plt.subplots(figsize=(80 * plot_style.MM, 50 * plot_style.MM))
            ax.set_xlabel("x")
        return fig

    with pytest.raises(RuntimeError, match="没有字体"):
        plot_style.save_fig(figure("No Such Font"), "t", "en", out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    written = plot_style.save_fig(figure("DejaVu Sans"), "t", "en", out_dir=tmp_path)
    assert sorted(p.name for p in written) == ["t_en.pdf", "t_en.png"]
    assert plot_io.figure_cli(None).parse_args([]).lang == "both"


def test_save_fig_refuses_small_text_with_scripts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """带 mathtext 上下标的文字小于 7 pt 时 save_fig 报错（上下标按 0.7 缩放会低于 5 pt）；7 pt 的照常出图。"""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import matplotlib.pyplot as plt
    import plot_style

    def figure(size: float):
        with plt.rc_context({"font.family": "DejaVu Sans"}):
            fig, ax = plt.subplots(figsize=(80 * plot_style.MM, 50 * plot_style.MM))
            ax.set_xlabel(plot_style.MT_CO2_YR, fontsize=size)
        return fig

    assert plot_style._has_script(plot_style.MT_CO2_YR) and not plot_style._has_script("$\\leq$ 50%")
    # `$` 不成对或被转义时 matplotlib 不按 mathtext 处理，`_` / `^` 只是普通字符
    assert not plot_style._has_script("价格 $5, x_y")
    assert not plot_style._has_script(r"\$a_b\$")
    assert plot_style._has_script(r"价格 \$5 CO$_2$")
    assert not plot_style._has_script("$x$ $y_z")
    with pytest.raises(RuntimeError, match="上下标"):
        plot_style.save_fig(figure(6), "s", "en", out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert len(plot_style.save_fig(figure(7), "s", "en", out_dir=tmp_path)) == 2


def test_scs_inset_leaves_taiwan_uncovered_and_legends_off_land(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("geopandas", reason="底图需要 geopandas")
    from shapely.geometry import box

    monkeypatch.syspath_prepend(str(SCRIPTS))
    import matplotlib.pyplot as plt
    import plot_map

    fig, ax = plt.subplots()
    plot_map.mainland_extent(ax)
    inset = plot_map.add_scs_inset(ax)
    fig.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(inset.get_window_extent().get_points())
    land = plot_map.map_layer("country_main").geometry
    assert len(land) == 3                                  # 大陆、台湾、海南（面积按 shp 原生的等积投影算）
    assert not any(part.intersects(box(x0, y0, x1, y1)) for part in land)
    taiwan = min(land, key=lambda part: abs(part.centroid.x - plot_map.to_map_xy([121.0], [23.7])[0][0]))
    assert ax.get_xlim()[1] > taiwan.bounds[2]            # 台湾在主图范围内
    label = ax.text(0.5, 0.5, "legend", transform=ax.transAxes)   # 主图正中，压在国土上
    with pytest.raises(RuntimeError, match="国土"):
        plot_map.check_off_land(ax, label)
    plt.close(fig)
