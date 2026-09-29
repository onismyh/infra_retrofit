"""图 2  候选 CO2 管网（输入）

图含义：模型可以选用的全部候选管段，按管网输入表的 edge_class 分四类着色——既有油气干线走廊、支线（源或汇接到
  干线）、源到汇的直连弧、三角化候选边（含连通片合并边与缝合边）；图例里是各类条数。灰色圆点是煤电厂址、
  灰色菱形是工业 hub，方块是封存汇（深蓝深部咸水层、浅蓝驱油封存）。
读图注意：这是候选集合，不是求解出的管网（求解结果见图 6）；既有干线按实际路由画，其余按 WKT 折线或两端直线画。
数据：_indtree/inputs/pipeline_candidate_edges.csv、pipeline_nodes.csv、storage_hubs.csv。
自检：edge_class 都认识（各类条数之和 = 总边数）；每条边都画得出来（端点都在节点表里）；封存汇类型只有 dsa / eor。
输出：_indtree/results/figures/fig2_candidate_network{,_en}.{pdf,png}
用法：python scripts/plot_fig2_candidate_network.py [--lang zh|en|both]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.typing import LineStyleType

from plot_style import (
    MAP_LEGEND, MARKERS, MM, SECTOR_COLORS, SINK_COLORS,
    add_scs_inset, apply_style, check_off_land, draw_china_basemap, edge_lines, figure_cli, labels, langs,
    mainland_extent, read_input, save_fig, to_map_xy,
)

NAME = "fig2_candidate_network"
# 四类边：键 -> (edge_class, 颜色, 线宽, 透明度, 线型)。按字典顺序画，后画的压在上面，干线最后。
EDGE_STYLE: dict[str, tuple[tuple[str, ...], str, float, float, LineStyleType]] = {
    "triangulation": (("triangulation_candidate",), "#B09C85", 0.35, 0.7, "-"),
    "branch": (("hub_to_corridor_branch", "corridor_to_storage_branch"), "#8491B4", 0.4, 0.85, "--"),
    "direct": (("runtime_direct_fallback",), "#4DBBD5", 0.45, 0.85, "-"),
    "corridor": (("existing_main_corridor",), "#E64B35", 0.8, 0.9, "-"),
}
INDUSTRY_GREY = "#969696"
EDGE_NAMES = {
    "zh": {"corridor": "既有油气干线", "direct": "源到汇直连弧", "branch": "支线（接既有干线）",
           "triangulation": "三角化候选边"},
    "en": {"corridor": "Existing oil & gas trunkline", "direct": "Direct source-sink arc",
           "branch": "Branch to trunkline", "triangulation": "Triangulation candidate"},
}
TEXT = {
    "zh": {"plant": "煤电厂址", "industry": "工业 hub", "count": "{name}（{n}）", "title": "候选边共 {n} 条"},
    "en": {"plant": "Coal plants", "industry": "Industrial hubs", "count": "{name} ({n})",
           "title": "{n} candidate edges"},
}


def load() -> dict[str, pd.DataFrame]:
    nodes = read_input("pipeline_nodes")
    edges = read_input("pipeline_candidate_edges")
    storage = nodes[nodes["node_type"] == "storage_hub"].merge(
        read_input("storage_hubs")[["storage_hub_id", "storage_type"]],
        on="storage_hub_id", how="left", validate="one_to_one")
    return {"nodes": nodes, "edges": edges, "storage": storage}


def classify(edges: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {key: edges[edges["edge_class"].isin(classes)] for key, (classes, *_) in EDGE_STYLE.items()}


def check(data: dict[str, pd.DataFrame], groups: dict[str, pd.DataFrame]) -> None:
    drawn = sum(len(frame) for frame in groups.values())
    if drawn != len(data["edges"]):
        unknown = sorted(set(data["edges"]["edge_class"]) - {c for classes, *_ in EDGE_STYLE.values() for c in classes})
        raise ValueError(f"自检不通过：edge_class {unknown} 不认识，{len(data['edges']) - drawn} 条边没有归类，不出图")
    if not data["storage"]["storage_type"].isin(list(SINK_COLORS)).all():
        raise ValueError("自检不通过：封存汇类型只应是 dsa / eor（CLAUDE.md §1.3），不出图")


def draw(data: dict[str, pd.DataFrame], groups: dict[str, pd.DataFrame], lines: dict, lang: str):
    lab, text, edge_names = labels(lang), TEXT[lang], EDGE_NAMES[lang]
    nodes = data["nodes"]
    x, y = to_map_xy(nodes["lon"], nodes["lat"])
    nodes = nodes.assign(x=x, y=y)
    plants = nodes[nodes["node_type"] == "plant"]
    industry = nodes[nodes["node_type"] == "industry_hub"]
    sx, sy = to_map_xy(data["storage"]["lon"], data["storage"]["lat"])
    storage = data["storage"].assign(x=sx, y=sy)

    def layers(ax, k: float = 1.0) -> None:
        for z, (key, (_, colour, lw, alpha, ls)) in enumerate(EDGE_STYLE.items()):
            ax.add_collection(LineCollection(lines[key], colors=colour, linewidths=lw * k, alpha=alpha,
                                             linestyles=ls, capstyle="round", zorder=2 + 0.1 * z))
        ax.scatter(plants["x"], plants["y"], s=5 * k, c=SECTOR_COLORS["coal"], marker=MARKERS["coal"],
                   edgecolors="white", linewidths=0.25 * k, zorder=4)
        ax.scatter(industry["x"], industry["y"], s=4 * k, c=INDUSTRY_GREY, marker=MARKERS["industry"],
                   edgecolors="white", linewidths=0.25 * k, zorder=4)
        for kind, colour in SINK_COLORS.items():
            part = storage[storage["storage_type"] == kind]
            ax.scatter(part["x"], part["y"], s=12 * k, c=colour, marker=MARKERS["sink"],
                       edgecolors="#08519C", linewidths=0.3 * k, zorder=5)

    fig = plt.figure(figsize=(183 * MM, 143 * MM))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    draw_china_basemap(ax)
    layers(ax)
    mainland_extent(ax)
    add_scs_inset(ax, draw=lambda a: layers(a, 0.6))

    handles = [Line2D([], [], color=EDGE_STYLE[key][1], linewidth=max(EDGE_STYLE[key][2], 0.9),
                      linestyle=EDGE_STYLE[key][4],
                      label=text["count"].format(name=edge_names[key], n=len(groups[key])))
               for key in reversed(EDGE_STYLE)]
    handles += [
        Line2D([], [], marker=MARKERS["coal"], linestyle="none", markersize=3.5, markerfacecolor=SECTOR_COLORS["coal"],
               markeredgecolor="white", markeredgewidth=0.3,
               label=text["count"].format(name=text["plant"], n=len(plants))),
        Line2D([], [], marker=MARKERS["industry"], linestyle="none", markersize=3.2, markerfacecolor=INDUSTRY_GREY,
               markeredgecolor="white", markeredgewidth=0.3,
               label=text["count"].format(name=text["industry"], n=len(industry))),
    ]
    handles += [Line2D([], [], marker=MARKERS["sink"], linestyle="none", markersize=3.5, markerfacecolor=colour,
                       markeredgecolor="#08519C", markeredgewidth=0.3,
                       label=text["count"].format(name=lab["sink"][kind],
                                                  n=int((storage["storage_type"] == kind).sum())))
                for kind, colour in SINK_COLORS.items()]
    legend = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0),
                       title=text["title"].format(n=len(data["edges"])), alignment="left", **MAP_LEGEND)
    check_off_land(ax, legend)
    return fig


def main(argv: list[str] | None = None) -> None:
    args = figure_cli(__doc__, scenario=False).parse_args(argv)
    data = load()
    groups = classify(data["edges"])
    check(data, groups)
    lines = {key: edge_lines(frame, data["nodes"]) for key, frame in groups.items()}
    for lang in langs(args.lang):
        apply_style(lang)
        save_fig(draw(data, groups, lines, lang), NAME, lang)


if __name__ == "__main__":
    main()
