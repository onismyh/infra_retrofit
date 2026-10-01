"""图 1  排放源与封存汇（输入）

图含义：
  a  地图：进模型的全部煤电厂址（圆点）与工业 hub（菱形，颜色按部门，与 b 同色），点面积随现状 CO2 排放线性增大
     （有下限，小点才看得见），煤电与工业用同一尺度、同值同面积；封存汇（方块）按类型分深部咸水层与驱油封存，
     只标位置，不按容量缩放（容量跨四个数量级，按面积画会让驱油汇看不见）。
  b  各部门现状排放合计（Mt CO2/yr），括号里是源的个数。
读图注意：排放是现状值（煤电 = 装机 × 分省利用小时 × 排放因子，工业取 industry_hubs.csv），不是求解结果；
          规划年的排放会按利用小时轨迹与产量指数缩放。
数据：煤电用模型自己的读入函数 `data_prep._prepare_plants`，工业用 `industry_inputs.prepare_industry`
      （已按模型剔除西藏 hub），封存汇取 `_indtree/inputs/storage_hubs.csv`。
自检：坐标与排放没有缺值（缺值的点会被静默丢掉）；部门都在 `SECTOR_ORDER` 里；封存汇类型只有 dsa / eor。
输出：_indtree/results/figures/fig1_sources_sinks{,_en}.{pdf,png}
用法：python scripts/plot_fig1_sources_sinks.py [--lang zh|en|both]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from _bootstrap import ROOT
from plot_style import (
    INK, MAP_LEGEND, MARKERS, MM, MT_CO2_YR, SECTOR_COLORS, SECTOR_ORDER, SINK_COLORS,
    add_scs_inset, apply_style, area_scale, check_off_land, draw_china_basemap, figure_cli, figure_label,
    labels, langs, mainland_extent, read_input, save_fig, size_legend, to_map_xy,
)

NAME = "fig1_sources_sinks"
TEXT = {
    "zh": {"xlabel": f"现状排放（{MT_CO2_YR}）", "size": "点面积随排放线性增大（" + MT_CO2_YR + "）",
           "count": "{name}（{n}）", "industry": "工业 hub"},
    "en": {"xlabel": f"Current emissions ({MT_CO2_YR})", "size": "Area grows linearly with emissions (" + MT_CO2_YR + ")",
           "count": "{name} ({n})", "industry": "Industrial hubs"},
}
SIZE_TICKS = (1.0, 10.0, 50.0)
SMAX = 70.0                  # 排放最大的源（煤电与工业合在一起取最大）的点面积，pt²


def load() -> dict[str, pd.DataFrame]:
    """煤电厂址、工业 hub、封存汇，都是进模型的那一套。"""
    from coal_retrofit.optimization.data_prep import _prepare_plants
    from coal_retrofit.optimization.industry_inputs import prepare_industry
    from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
    from coal_retrofit.paths import ProjectPaths

    paths = ProjectPaths(ROOT)
    assumptions = OptimizationAssumptions()
    plants = _prepare_plants(paths, OptimizationScenario(experiment_id="plot", description="plot"), assumptions)
    coal = pd.DataFrame({"sector": "coal", "lon": plants["centroid_longitude"],
                         "lat": plants["centroid_latitude"], "co2_mt": plants["baseline_emissions_mt"]})
    hubs = prepare_industry(paths, assumptions).hubs
    industry = pd.DataFrame({"sector": hubs["sector"], "lon": hubs["longitude"],
                             "lat": hubs["latitude"], "co2_mt": hubs["co2_mt_per_year"]})
    sinks = read_input("storage_hubs")[["storage_type", "longitude", "latitude"]]
    return {"sources": pd.concat([coal, industry], ignore_index=True), "sinks": sinks}


def check(data: dict[str, pd.DataFrame]) -> None:
    sources, sinks = data["sources"], data["sinks"]
    if sources[["lon", "lat", "co2_mt"]].isna().any().any():
        raise ValueError("自检不通过：有排放源缺坐标或排放，不出图")
    unknown = set(sources["sector"]) - set(SECTOR_ORDER)
    if unknown:
        raise ValueError(f"自检不通过：部门 {sorted(unknown)} 不在 SECTOR_ORDER 里，不出图")
    if not sinks["storage_type"].isin(list(SINK_COLORS)).all():
        raise ValueError("自检不通过：封存汇类型只应是 dsa / eor（CLAUDE.md §1.3），不出图")


def draw(data: dict[str, pd.DataFrame], lang: str):
    lab, text = labels(lang), TEXT[lang]
    sources, sinks = data["sources"], data["sinks"]
    vmax = float(sources["co2_mt"].max())
    sx, sy = to_map_xy(sources["lon"], sources["lat"])
    kx, ky = to_map_xy(sinks["longitude"], sinks["latitude"])
    sources = sources.assign(x=sx, y=sy)
    sinks = sinks.assign(x=kx, y=ky)

    def layers(ax, k: float = 1.0) -> None:
        for sector in SECTOR_ORDER:            # 煤电先画、压在下面，工业小点在上
            part = sources[sources["sector"] == sector]
            marker = MARKERS["coal" if sector == "coal" else "industry"]
            ax.scatter(part["x"], part["y"], s=area_scale(part["co2_mt"], vmax, smax=SMAX, marker=marker) * k,
                       c=SECTOR_COLORS[sector], marker=marker,
                       alpha=0.8, edgecolors="white", linewidths=0.25 * k, zorder=3)
        for kind, colour in SINK_COLORS.items():
            part = sinks[sinks["storage_type"] == kind]
            ax.scatter(part["x"], part["y"], s=6 * k, c=colour, marker=MARKERS["sink"],
                       edgecolors="#08519C", linewidths=0.3 * k, zorder=4)

    fig = plt.figure(figsize=(183 * MM, 98 * MM))
    ax = fig.add_axes((0.0, 0.0, 0.64, 0.97))
    draw_china_basemap(ax)
    layers(ax)
    mainland_extent(ax)
    add_scs_inset(ax, draw=lambda a: layers(a, 0.5))
    counts = sources["sector"].value_counts()
    n_industry = int((sources["sector"] != "coal").sum())
    # 工业 hub 一行画五个部门色的菱形，颜色与 b 的柱对应
    industry = tuple(Line2D([], [], marker=MARKERS["industry"], linestyle="none", markersize=3,
                            markerfacecolor=SECTOR_COLORS[s], markeredgecolor="white", markeredgewidth=0.3)
                     for s in SECTOR_ORDER if s != "coal")
    handles: list = [Line2D([], [], marker=MARKERS["coal"], linestyle="none", markersize=4,
                            markerfacecolor=SECTOR_COLORS["coal"], markeredgecolor="white", markeredgewidth=0.3),
                     industry]
    names = [text["count"].format(name=lab["sector"]["coal"], n=counts["coal"]),
             text["count"].format(name=text["industry"], n=n_industry)]
    for kind, colour in SINK_COLORS.items():
        handles.append(Line2D([], [], marker=MARKERS["sink"], linestyle="none", markersize=3,
                              markerfacecolor=colour, markeredgecolor="#08519C", markeredgewidth=0.3))
        names.append(text["count"].format(name=lab["sink"][kind], n=int((sinks["storage_type"] == kind).sum())))
    size = ax.legend(handles=size_legend(SIZE_TICKS, vmax, "o", "#B0B0B0", smax=SMAX), loc="lower left",
                     bbox_to_anchor=(0.0, 0.0), ncol=len(SIZE_TICKS), title=text["size"], alignment="left",
                     **MAP_LEGEND)
    ax.add_artist(size)
    kinds = ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0.0, 0.085), handlelength=3.2,
                      handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, **MAP_LEGEND)
    check_off_land(ax, size, kinds)

    bx = fig.add_axes((0.775, 0.16, 0.205, 0.72))
    totals = sources.groupby("sector")["co2_mt"].sum().reindex(SECTOR_ORDER)
    y = np.arange(len(SECTOR_ORDER))[::-1]
    bx.barh(y, totals.to_numpy(), height=0.62, color=[SECTOR_COLORS[s] for s in SECTOR_ORDER],
            edgecolor="none", zorder=3)
    for yi, value in zip(y, totals.to_numpy()):
        bx.text(value + totals.max() * 0.03, yi, f"{value:.0f}", va="center", ha="left", fontsize=6, color=INK)
    bx.set_yticks(y)
    bx.set_yticklabels([text["count"].format(name=lab["sector"][s], n=counts[s]) for s in SECTOR_ORDER])
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(0, totals.max() * 1.3)
    bx.set_xlabel(text["xlabel"])
    figure_label(fig, "a", 0.005, 0.975)
    figure_label(fig, "b", 0.655, 0.975)
    return fig


def main(argv: list[str] | None = None) -> None:
    args = figure_cli(__doc__, scenario=False).parse_args(argv)
    data = load()
    check(data)
    for lang in langs(args.lang):
        apply_style(lang)
        save_fig(draw(data, lang), NAME, lang)


if __name__ == "__main__":
    main()
