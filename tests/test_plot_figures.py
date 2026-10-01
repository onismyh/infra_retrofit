"""出图脚本（scripts/plot_fig*.py、plot_style）的数据函数与出图前自检：不画整张图、不求解，不需要 Gurobi。

各图的自检是"图上要画的量与模型记的量对不上就报错、不出图"，这里用最小的表造出对得上与对不上两种情形；
另查 plot_style 的三件事：同值的圆点、菱形、方块面积相同，本机没有所用字体时 save_fig 拒绝出图，没有文件头的脚本
也能建命令行；最后一条查底图：南海小图不压台湾与大陆沿海，图例压到国土会报错（要 geopandas 与仓库里的 data/ChinaMapTHT）。
"""
from __future__ import annotations

import importlib.util
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


def test_fig3_capacity_split_and_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    fig3 = _load(monkeypatch, "plot_fig3_power_pathways")
    shares = {"unabated": 0.5, "ccs": 0.3, "biomass": 0.0, "beccs": 0.0, "ammonia": 0.0, "retire": 0.2}
    detail = pd.DataFrame([{"year": 2030, "plant_id": "P1", "capacity_mw": 2000.0, "reduction_mt": 1.5,
                            **{f"share_{k}": v for k, v in shares.items()}}])
    pathways = pd.DataFrame([{"year": 2030, "plant_id": "P1", "pathway": k, "abatement_mt": a}
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
    with pytest.raises(ValueError, match="reduction_mt"):
        fig3.check({**data, "pathways": pathways.assign(abatement_mt=pathways["abatement_mt"] * 2)})
    with pytest.raises(ValueError, match="重解"):          # 旧结果：拆分是旧口径，没有闭合行
        fig3.check({**data, "sanity": sanity.iloc[:0]})
    with pytest.raises(ValueError, match="对不上"):
        fig3.check({**data, "sanity": sanity.assign(status="warn")})


def _fig5_data(residual_2060: float, cap_fraction: float, shortfall: float, years=(2030, 2060)) -> dict:
    """电力一组：冻结技术排放 2030 年 10、2060 年 6（利用小时下降），残余 2030 年 9。"""
    residual = {2030: 9.0, 2060: residual_2060}
    baseline = {2030: 10.0, 2060: 6.0}
    plants = pd.DataFrame([{"year": y, "baseline_emissions_mt": baseline[y], "reduction_mt": baseline[y] - residual[y]}
                           for y in years])
    industry = pd.DataFrame(columns=["year", "target_group", "baseline_co2_mt", "residual_mt"])
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


def test_fig6_mass_balance_and_legend_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    fig6 = _load(monkeypatch, "plot_fig6_co2_network")
    data = {"edges": pd.DataFrame({"year": [2040], "edge_id": ["E1"], "edge_flow_mtpa": [3.0]}),
            "plants": pd.DataFrame({"year": [2040], "captured_mt": [2.0]}),
            "industry": pd.DataFrame({"year": [2040], "sector": ["cement"], "captured_mt": [1.0]}),
            "storage": pd.DataFrame({"year": [2040], "storage_use_mtpa": [3.0]}),
            "sinks": pd.DataFrame({"storage_type": ["dsa"]})}
    fig6.check(data, (2040,))
    with pytest.raises(ValueError, match="守恒"):
        fig6.check({**data, "storage": data["storage"].assign(storage_use_mtpa=2.5)}, (2040,))
    with pytest.raises(ValueError, match="2060"):
        fig6.check(data, (2040, 2060))
    with pytest.raises(ValueError, match="SECTOR_ORDER"):     # 不认识的部门画不出来，守恒照样对得上
        fig6.check({**data, "industry": data["industry"].assign(sector="glass")}, (2040,))
    assert fig6.nice_levels(87.0) == [1.0, 10.0, 50.0]


def test_fig7_utilization_and_converted_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """余量 ≤ 0 而仍有取水记为无余量（inf），按 used / available 重算、不读结果表的 utilization（旧结果表
    在这里记 0，夹具即按旧表写）；超出余量的取水要与 slack_detail 的流域松弛对上；超了但四舍五入是 100% 的格子写 >100%；
    空冷量只算当年在运行、仍湿冷的部分。"""
    fig7 = _load(monkeypatch, "plot_fig7_water")
    basins = pd.DataFrame({"year": 2030, "region": ["D", "C", "E"], "used": [9.0, 5.0e6, 0.0],
                           "available": [10.0, -1.0e6, 0.0], "utilization": [0.9, 0.0, 0.0]})
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
    slack = pd.DataFrame({"year": [2030], "constraint_type": ["water_basin_quota"], "node_id": ["C"],
                          "slack_value": [6.0e6]})
    fig7.check({"basins": basins, "plants": plants, "slack": slack})
    with pytest.raises(ValueError, match="松弛"):
        fig7.check({"basins": basins, "plants": plants, "slack": slack.iloc[:0]})
    with pytest.raises(ValueError, match="BASIN_ORDER"):
        fig7.check({"basins": basins.assign(region="B1"), "plants": plants, "slack": slack})
    old = plants.rename(columns={"air_operating_share": "air_cooled_share"})   # 2026-09-30 之前落盘的结果
    with pytest.raises(ValueError, match="重解"):
        fig7.check({"basins": basins, "plants": old, "slack": slack})


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


def test_save_fig_refuses_a_missing_font(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """图里文字用的字体本机没有时 save_fig 报错、一个文件都不写（matplotlib 自己会静默换成 DejaVu Sans 照常出图）；
    字体在就照常出图。没有文件头的脚本也能建命令行。"""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import matplotlib.pyplot as plt
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
    assert plot_style.figure_cli(None).parse_args([]).lang == "both"


def test_scs_inset_leaves_taiwan_uncovered_and_legends_off_land(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("geopandas", reason="底图需要 geopandas")
    from shapely.geometry import box

    monkeypatch.syspath_prepend(str(SCRIPTS))
    import matplotlib.pyplot as plt
    import plot_style

    fig, ax = plt.subplots()
    plot_style.mainland_extent(ax)
    inset = plot_style.add_scs_inset(ax)
    fig.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(inset.get_window_extent().get_points())
    land = plot_style.map_layer("country_main").geometry
    assert len(land) == 3                                  # 大陆、台湾、海南（面积按 shp 原生的等积投影算）
    assert not any(part.intersects(box(x0, y0, x1, y1)) for part in land)
    taiwan = min(land, key=lambda part: abs(part.centroid.x - plot_style.to_map_xy([121.0], [23.7])[0][0]))
    assert ax.get_xlim()[1] > taiwan.bounds[2]            # 台湾在主图范围内
    label = ax.text(0.5, 0.5, "legend", transform=ax.transAxes)   # 主图正中，压在国土上
    with pytest.raises(RuntimeError, match="国土"):
        plot_style.check_off_land(ax, label)
    plt.close(fig)
