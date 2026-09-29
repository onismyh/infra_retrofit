"""出图脚本（scripts/plot_fig*.py、plot_style）的数据函数与出图前自检：不画整张图、不求解，不需要 Gurobi。

各图的自检是"图上要画的量与模型记的量对不上就报错、不出图"，这里用最小的表造出对得上与对不上两种情形；
最后一条查底图：南海小图不压台湾与大陆沿海，图例压到国土会报错（要 geopandas 与仓库里的 data/ChinaMapTHT）。
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
    data = {"detail": detail, "pathways": pathways, "result": {"years": {"2030": {"coal_reduction_mt": 1.5}}}}
    fig3.check(data)
    capacity = fig3.capacity_by_pathway(detail)
    assert capacity.loc[2030, "ccs"] == pytest.approx(0.6)
    assert capacity.loc[2030].sum() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="份额之和"):
        fig3.check({**data, "detail": detail.assign(share_ccs=0.4)})
    with pytest.raises(ValueError, match="reduction_mt"):
        fig3.check({**data, "pathways": pathways.assign(abatement_mt=pathways["abatement_mt"] * 2)})


def _fig5_data(residual_2060: float, cap_fraction: float, shortfall: float, years=(2030, 2060)) -> dict:
    residual = {2030: 9.0, 2060: residual_2060}
    plants = pd.DataFrame([{"year": y, "baseline_emissions_mt": 10.0, "reduction_mt": 10.0 - residual[y]}
                           for y in years])
    industry = pd.DataFrame(columns=["year", "target_group", "baseline_co2_mt", "residual_mt"])
    result = {"years": {str(y): {
        "coal_baseline_mt": 10.0, "coal_residual_mt": residual[y],
        "sector_cap_fraction": {"power": 1.0 if y == 2030 else cap_fraction},
        "target_shortfall_by_group_mt": {"power": 0.0 if y == 2030 else shortfall},
        "industry": {"baseline_mt": 0.0, "residual_by_group_mt": {}},
    } for y in years}}
    return {"plants": plants, "industry": industry, "result": result}


def test_fig5_cap_is_fraction_of_2030_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """上限 = 比例 × 2030 年冻结技术排放；残余 − 上限不得大于目标缺口；没有 2030 年算不出上限。"""
    fig5 = _load(monkeypatch, "plot_fig5_sector_targets")
    data = _fig5_data(residual_2060=4.0, cap_fraction=0.3, shortfall=1.0)
    table = fig5.sector_table(data)
    assert table.set_index("year").loc[2060, "cap"] == pytest.approx(3.0)
    fig5.check(data, table)
    unmet = _fig5_data(residual_2060=4.0, cap_fraction=0.3, shortfall=0.0)
    with pytest.raises(ValueError, match="目标缺口"):
        fig5.check(unmet, fig5.sector_table(unmet))
    with pytest.raises(ValueError, match="2030"):
        fig5.sector_table(_fig5_data(residual_2060=4.0, cap_fraction=0.3, shortfall=1.0, years=(2060,)))


def test_fig6_mass_balance_and_legend_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    fig6 = _load(monkeypatch, "plot_fig6_co2_network")
    data = {"edges": pd.DataFrame({"year": [2040], "edge_id": ["E1"], "edge_flow_mtpa": [3.0]}),
            "plants": pd.DataFrame({"year": [2040], "captured_mt": [2.0]}),
            "industry": pd.DataFrame({"year": [2040], "captured_mt": [1.0]}),
            "storage": pd.DataFrame({"year": [2040], "storage_use_mtpa": [3.0]}),
            "sinks": pd.DataFrame({"storage_type": ["dsa"]})}
    fig6.check(data, (2040,))
    with pytest.raises(ValueError, match="守恒"):
        fig6.check({**data, "storage": data["storage"].assign(storage_use_mtpa=2.5)}, (2040,))
    with pytest.raises(ValueError, match="2060"):
        fig6.check(data, (2040, 2060))
    assert fig6.nice_levels(87.0) == [1.0, 10.0, 50.0]


def test_fig7_utilization_and_converted_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """余量 ≤ 0 而仍有取水记为无余量（inf），不能按结果表的 0 读成"没用"；空冷量只算仍湿冷的部分。"""
    fig7 = _load(monkeypatch, "plot_fig7_water")
    basins = pd.DataFrame({"year": 2030, "region": ["D", "C", "E"], "used": [9.0, 5.0, 0.0],
                           "available": [10.0, -1.0, 0.0], "utilization": [0.9, 0.0, 0.0]})
    table = fig7.utilization(basins)
    assert list(table.index) == ["C", "D", "E"]
    assert np.isinf(table.loc["C", 2030])
    assert table.loc["D", 2030] == pytest.approx(0.9)
    assert table.loc["E", 2030] == 0.0
    plants = pd.DataFrame({"year": [2040], "capacity_mw": [1000.0], "air_cooled_share": [0.5],
                           "already_air_share": [0.4]})
    assert fig7.converted_gw(plants).loc[2040] == pytest.approx(0.3)
    fig7.check({"basins": basins, "plants": plants})
    with pytest.raises(ValueError, match="BASIN_ORDER"):
        fig7.check({"basins": basins.assign(region="B1"), "plants": plants})


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
    assert not any(part.intersects(box(x0, y0, x1, y1)) for part in land)
    taiwan = min(land, key=lambda part: abs(part.centroid.x - plot_style.to_map_xy([121.0], [23.7])[0][0]))
    assert ax.get_xlim()[1] > taiwan.bounds[2]            # 台湾在主图范围内
    label = ax.text(0.5, 0.5, "legend", transform=ax.transAxes)   # 主图正中，压在国土上
    with pytest.raises(RuntimeError, match="国土"):
        plot_style.check_off_land(ax, label)
    plt.close(fig)
