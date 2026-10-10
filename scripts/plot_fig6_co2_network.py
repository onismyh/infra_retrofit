"""图 6  CO2 管网与封存（求解结果）

图含义：两个规划年（缺省 2040 与 2060）的 CO2 捕集、输送与封存。蓝线是有流量的管段，线宽 ∝ √年流量（正反两向
  之和，细管段有下限）；灰色圆点是有捕集的煤电厂址，菱形是有捕集的工业 hub（颜色按部门，同图 1），面积随捕集量
  线性增大；方块是有注入的封存汇（深蓝深部咸水层、浅蓝驱油封存），面积随注入量线性增大。点面积有下限，同值的
  圆点、菱形、方块面积相同。两年用同一尺度，左上角是当年封存总量。
读图注意：画的是年流量，不是建成的管道能力；管段按候选管网的 WKT 折线画，运行期补的短连接（runtime_*，不在候选表里）
  按两端直线画。
数据：results/solved/<情景>/network.csv、sources.csv（煤电、工业取 source_type 为 coal、industry 的行）、sinks.csv，
  _indtree/inputs/pipeline_candidate_edges.csv、pipeline_nodes.csv、storage_hubs.csv。
自检：每年 煤电捕集 + 工业捕集 = 封存注入（管网节点守恒，model_year 的三条 co2 节点约束）；有流量的管段端点都找得到；
  封存汇类型只有 dsa / eor；工业 hub 的部门都在 SECTOR_ORDER 里（否则图上漏画而守恒照样对得上）。
输出：_indtree/results/figures/fig6_co2_network{,_en}.{pdf,png}
用法：python scripts/plot_fig6_co2_network.py [--scenario 情景] [--years 2040 2060] [--lang zh|en|both]
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from plot_style import (
    INK, MARKERS, MM, MT_CO2_YR, PIPE_COLOR, SECTOR_COLORS, SECTOR_ORDER, SINK_COLORS,
    add_scs_inset, apply_style, area_scale, check_close, check_off_land, draw_china_basemap, edge_lines, figure_cli,
    figure_label, fmt_number, labels, langs, mainland_extent, read_input, read_result, save_fig, size_legend,
    to_map_xy,
)

NAME = "fig6_co2_network"
DEFAULT_YEARS = (2040, 2060)
FLOW_MIN = 1e-3              # Mt/yr；流量、捕集、注入低于它不画（求解器容差量级）
LW_MAX = 2.8                 # 最大流量管段的线宽（pt）
SMAX = 40.0                  # 最大捕集 / 注入处圆点的 s（pt²，外接正方形面积）；画出的面积是 π/4·SMAX，各形状同值同面积
TEXT = {
    "zh": {"total": "{year} 年封存 {v} ", "flow": "管段年流量（" + MT_CO2_YR + "）",
           "size": "捕集量 / 注入量（" + MT_CO2_YR + "）", "industry": "工业 hub"},
    "en": {"total": "Stored in {year}: {v} ", "flow": "Pipeline flow (" + MT_CO2_YR + ")",
           "size": "Captured / injected (" + MT_CO2_YR + ")", "industry": "Industrial hubs"},
}


def load(scenario: str) -> dict[str, pd.DataFrame]:
    sources = read_result(scenario, "sources").astype({"source_id": str})
    return {
        "edges": read_result(scenario, "network"),
        "plants": sources[sources["source_type"] == "coal"],
        "industry": sources[sources["source_type"] == "industry"],
        "storage": read_result(scenario, "sinks").astype({"sink_id": str}),
        "candidates": read_input("pipeline_candidate_edges"),
        "nodes": read_input("pipeline_nodes"),
        "sinks": read_input("storage_hubs").astype({"storage_hub_id": str}),
    }


def node_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """管网节点坐标：输入的节点表，加上运行期补的节点（network.py：plant::、storage::、industry:: 前缀）。"""
    plants = data["plants"].drop_duplicates("source_id")
    hubs = data["industry"].drop_duplicates("source_id")
    sinks = data["sinks"]
    return pd.concat([
        data["nodes"][["node_id", "lon", "lat"]],
        pd.DataFrame({"node_id": "plant::" + plants["source_id"], "lon": plants["longitude"],
                      "lat": plants["latitude"]}),
        pd.DataFrame({"node_id": "storage::" + sinks["storage_hub_id"], "lon": sinks["longitude"],
                      "lat": sinks["latitude"]}),
        pd.DataFrame({"node_id": "industry::" + hubs["source_id"], "lon": hubs["longitude"], "lat": hubs["latitude"]}),
    ], ignore_index=True)


def year_layers(data: dict[str, pd.DataFrame], year: int) -> dict[str, pd.DataFrame]:
    """某年要画的管段、捕集源与封存汇（低于 FLOW_MIN 的不画）。"""
    edges = data["edges"]
    edges = edges[(edges["year"] == year) & (edges["flow_mtpa"] > FLOW_MIN)]
    if "geometry_wkt" in data["candidates"].columns:
        edges = edges.merge(data["candidates"][["edge_id", "geometry_wkt"]], on="edge_id", how="left")
    plants = data["plants"][(data["plants"]["year"] == year) & (data["plants"]["captured_co2_mtpa"] > FLOW_MIN)]
    industry = data["industry"][(data["industry"]["year"] == year) & (data["industry"]["captured_co2_mtpa"] > FLOW_MIN)]
    storage = data["storage"][(data["storage"]["year"] == year) & (data["storage"]["injected_mtpa"] > FLOW_MIN)]
    return {"edges": edges, "plants": plants, "industry": industry, "storage": storage}


def check(data: dict[str, pd.DataFrame], years: tuple[int, ...]) -> None:
    available = sorted(int(y) for y in data["edges"]["year"].unique())
    missing = [y for y in years if y not in available]
    if missing:
        raise ValueError(f"结果里没有 {missing} 年（有 {available}），用 --years 指定")
    if not data["storage"]["storage_type"].isin(list(SINK_COLORS)).all():
        raise ValueError("自检不通过：封存汇类型只应是 dsa / eor（CLAUDE.md §1.3），不出图")
    unknown = set(data["industry"]["sector"]) - set(SECTOR_ORDER[1:])
    if unknown:
        raise ValueError(f"自检不通过：部门 {sorted(unknown)} 不在 SECTOR_ORDER 里，图上会漏画，不出图")
    for year in years:
        captured = (data["plants"].loc[data["plants"]["year"] == year, "captured_co2_mtpa"].sum()
                    + data["industry"].loc[data["industry"]["year"] == year, "captured_co2_mtpa"].sum())
        stored = data["storage"].loc[data["storage"]["year"] == year, "injected_mtpa"].sum()
        check_close(f"{year} 年 煤电捕集 + 工业捕集 应等于 封存注入（管网节点守恒）", captured, stored, atol=1e-3)


def nice_levels(vmax: float) -> list[float]:
    """图例的三个代表值：不超过 vmax 的最大"1、2、5 × 10^k"，及其约 1/5 与 1/50。"""
    def floor_nice(v: float) -> float:
        exp = np.floor(np.log10(v))
        return max(m * 10 ** exp for m in (1, 2, 5) if m * 10 ** exp <= v)
    top = floor_nice(vmax)
    return sorted({floor_nice(top / 50), floor_nice(top / 5), top})


def draw(data: dict[str, pd.DataFrame], years: tuple[int, ...], lines: dict, lang: str):
    lab, text = labels(lang), TEXT[lang]
    by_year = {y: year_layers(data, y) for y in years}
    fmax = max([float(v["edges"]["flow_mtpa"].max()) for v in by_year.values() if len(v["edges"])] + [FLOW_MIN])
    smax = max([float(v[k][c].max()) for v in by_year.values()
                for k, c in (("plants", "captured_co2_mtpa"), ("industry", "captured_co2_mtpa"), ("storage", "injected_mtpa"))
                if len(v[k])] + [FLOW_MIN])

    def width(flow):
        return np.maximum(0.35, LW_MAX * np.sqrt(np.asarray(flow, dtype=float) / fmax))

    def layers(ax, year: int, k: float = 1.0) -> None:
        layer = by_year[year]
        ax.add_collection(LineCollection(lines[year], colors=PIPE_COLOR,
                                         linewidths=width(layer["edges"]["flow_mtpa"]) * k,
                                         capstyle="round", joinstyle="round", alpha=0.9, zorder=3))
        px, py = to_map_xy(layer["plants"]["longitude"], layer["plants"]["latitude"])
        ax.scatter(px, py, s=area_scale(layer["plants"]["captured_co2_mtpa"], smax, smax=SMAX, marker=MARKERS["coal"]) * k,
                   c=SECTOR_COLORS["coal"], marker=MARKERS["coal"], edgecolors="white", linewidths=0.3 * k, zorder=4)
        for sector in SECTOR_ORDER[1:]:
            part = layer["industry"][layer["industry"]["sector"] == sector]
            ix, iy = to_map_xy(part["longitude"], part["latitude"])
            ax.scatter(ix, iy, s=area_scale(part["captured_co2_mtpa"], smax, smax=SMAX, marker=MARKERS["industry"]) * k,
                       c=SECTOR_COLORS[sector], marker=MARKERS["industry"], edgecolors="white", linewidths=0.3 * k,
                       zorder=4)
        for kind, colour in SINK_COLORS.items():
            part = layer["storage"][layer["storage"]["storage_type"] == kind]
            sx, sy = to_map_xy(part["longitude"], part["latitude"])
            ax.scatter(sx, sy, s=area_scale(part["injected_mtpa"], smax, smax=SMAX, marker=MARKERS["sink"]) * k,
                       c=colour, marker=MARKERS["sink"], edgecolors="#08519C", linewidths=0.4 * k, zorder=5)

    fig = plt.figure(figsize=(183 * MM, 85 * MM))
    for i, year in enumerate(years):
        ax = fig.add_axes((0.505 * i, 0.1, 0.495, 0.87))
        draw_china_basemap(ax)
        layers(ax, year)
        mainland_extent(ax)
        add_scs_inset(ax, draw=lambda a, y=year: layers(a, y, 0.5))
        stored = float(by_year[year]["storage"]["injected_mtpa"].sum())
        # 单位带 mathtext 的花括号，不能进 str.format，拼在格式化之后
        label = ax.text(0.02, 0.98, text["total"].format(year=year, v=fmt_number(stored)) + MT_CO2_YR,
                        transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK)
        check_off_land(ax, label)
        figure_label(fig, "ab"[i], 0.505 * i + 0.003, 0.995)

    flow_handles = [Line2D([], [], color=PIPE_COLOR, linewidth=float(width(v)), label=f"{v:g}")
                    for v in nice_levels(fmax)]
    size_handles = size_legend(nice_levels(smax), smax, "o", "#B0B0B0", fmt="{:g}", smax=SMAX)
    industry = tuple(Line2D([], [], marker=MARKERS["industry"], linestyle="none", markersize=3,
                            markerfacecolor=SECTOR_COLORS[s], markeredgecolor="white", markeredgewidth=0.3)
                     for s in SECTOR_ORDER[1:])
    # 含元组句柄（HandlerTuple 画成多色菱形）；Figure.legend 的类型存根只认单个 Artist，所以标成 Any
    kinds: list[Any] = [Line2D([], [], marker=MARKERS["coal"], linestyle="none", markersize=3.5,
                               markerfacecolor=SECTOR_COLORS["coal"], markeredgecolor="white", markeredgewidth=0.3),
                        industry]
    kinds += [Line2D([], [], marker=MARKERS["sink"], linestyle="none", markersize=3.5, markerfacecolor=colour,
                     markeredgecolor="#08519C", markeredgewidth=0.4) for colour in SINK_COLORS.values()]
    names = [lab["sector"]["coal"], text["industry"]] + [lab["sink"][k] for k in SINK_COLORS]
    common = {"frameon": False, "borderaxespad": 0.3}
    fig.add_artist(fig.legend(handles=flow_handles, title=text["flow"], loc="lower left", bbox_to_anchor=(0.01, 0.0),
                              ncol=3, alignment="left", **common))
    fig.add_artist(fig.legend(handles=size_handles, title=text["size"], loc="lower center",
                              bbox_to_anchor=(0.46, 0.0), ncol=3, alignment="left", **common))
    fig.legend(kinds, names, loc="lower right", bbox_to_anchor=(0.995, 0.0), ncol=2, handlelength=3.2,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, **common)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = figure_cli(__doc__)
    parser.add_argument("--years", type=int, nargs=2, default=DEFAULT_YEARS, help="画哪两个规划年（缺省 2040 2060）")
    args = parser.parse_args(argv)
    years = tuple(args.years)
    data = load(args.scenario)
    check(data, years)
    nodes = node_table(data)
    lines = {y: edge_lines(year_layers(data, y)["edges"], nodes) for y in years}
    for lang in langs(args.lang):
        apply_style(lang)
        save_fig(draw(data, years, lines, lang), NAME, lang)


if __name__ == "__main__":
    main()
