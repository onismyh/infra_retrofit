"""候选 CO2 管网图（输入诊断）：既有油气干线、支线、直连弧、三角化候选边，以及煤电厂址、工业源、封存汇。

底图、字体与南海小图都走 `plot_style`（CLAUDE.md §3、§4）：EPSG:2380，省界 0.20，国界与九段线 0.75，
主图裁到 17°N，九段线主体在南海小图里；字体由 `apply_style()` 统一设置，本脚本不自设。
无坐标轴、无网格、无标题，各类的条数写在图例里；存图走 `save_fig`（缺字拒绝出图、宽度守卫，§4.6）。

输出：_indtree/inputs/figures/candidate_network.png + .pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

from _bootstrap import ROOT
from plot_style import MM, add_scs_inset, apply_style, draw_china_basemap, mainland_extent, save_fig, to_map_xy

# 离散类别用 CLAUDE.md §3.3 的 NPG 色；煤电厂址与 plot_ed_source_atlas 同色，封存汇按 §3.3 分 DSA / EOR。
C_GAS, C_OIL, C_BRANCH, C_DIRECT = "#E64B35", "#7E6148", "#8491B4", "#F39B7F"
C_TRI_SOURCE_SINK, C_TRI_OTHER = "#4DBBD5", "#B09C85"
C_PLANT, C_INDUSTRY, C_DSA, C_EOR = "#636363", "#00A087", "#3182BD", "#9ECAE1"


def _edge_xy(edge: pd.Series, coord: dict) -> tuple:
    """边的 EPSG:2380 坐标：有 WKT 折线的用折线，否则连两端节点。"""
    wkt_str = edge.get("geometry_wkt", "")
    if pd.notna(wkt_str) and str(wkt_str).startswith("LINESTRING"):
        lon, lat = np.asarray(shapely_wkt.loads(str(wkt_str)).coords).T
        return to_map_xy(lon, lat)
    f, t = coord.get(edge["from_node_id"]), coord.get(edge["to_node_id"])
    if f is None or t is None:
        raise RuntimeError(f"边 {edge['edge_id']} 的端点不在节点表里，图上会少画而图例照算")
    return (f[0], t[0]), (f[1], t[1])


def main() -> None:
    apply_style()
    nodes = pd.read_csv(ROOT / "inputs" / "pipeline_nodes.csv")
    edges = pd.read_csv(ROOT / "inputs" / "pipeline_candidate_edges.csv")
    x, y = to_map_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    nodes = nodes.assign(x=x, y=y)
    coord = dict(zip(nodes["node_id"], zip(nodes["x"], nodes["y"])))
    plants = nodes[nodes["node_type"] == "plant"]
    industry = nodes[nodes["node_type"] == "industry_hub"]
    storage = nodes[nodes["node_type"] == "storage_hub"].merge(
        pd.read_csv(ROOT / "inputs" / "storage_hubs.csv", usecols=["storage_hub_id", "storage_type"]),
        on="storage_hub_id", how="left", validate="one_to_one",
    )
    if not storage["storage_type"].isin(["dsa", "eor"]).all():
        raise RuntimeError("封存汇的 storage_type 只应是 dsa 或 eor（CLAUDE.md §1.3 两类分开）")

    # 2026-09-12 起工业源也进了节点表，源汇之间的三角化边两端可以是煤电厂址或工业源。
    sources = set(plants["node_id"]) | set(industry["node_id"])
    sinks = set(storage["node_id"])
    tri = edges[edges["edge_class"] == "triangulation_candidate"]
    source_sink = np.array([(f in sources and t in sinks) or (f in sinks and t in sources)
                            for f, t in zip(tri["from_node_id"], tri["to_node_id"])], dtype=bool)
    corridor = edges[edges["edge_class"] == "existing_main_corridor"]
    # (边, 颜色, 线宽, 透明度, 线型, zorder, 图例名)；先画的压在下面
    groups = [
        (tri[~source_sink], C_TRI_OTHER, 0.35, 0.6, "-", 2, "三角化候选：其他"),
        (tri[source_sink], C_TRI_SOURCE_SINK, 0.45, 0.8, "-", 2, "三角化候选：源汇之间"),
        (edges[edges["edge_class"] == "runtime_direct_fallback"], C_DIRECT, 0.45, 0.8, "-", 2, "源到汇直连弧"),
        (edges[edges["edge_class"].isin(["hub_to_corridor_branch", "corridor_to_storage_branch"])],
         C_BRANCH, 0.45, 0.8, "--", 2, "支线"),
        (corridor[corridor["corridor_type"] == "oil"], C_OIL, 0.7, 0.8, "-", 3, "既有石油干线"),
        (corridor[corridor["corridor_type"] == "gas"], C_GAS, 0.7, 0.8, "-", 3, "既有天然气干线"),
    ]
    drawn = sum(len(frame) for frame, *_ in groups)
    if drawn != len(edges):
        raise RuntimeError(f"{len(edges) - drawn} 条边没有归入任何图例类（edge_class 或 corridor_type 不认识）")
    lines = [[_edge_xy(e, coord) for _, e in frame.iterrows()] for frame, *_ in groups]
    points = [(plants, C_PLANT, "o", 5, 4, "煤电厂址"), (industry, C_INDUSTRY, "s", 5, 4, "工业源"),
              (storage[storage["storage_type"] == "dsa"], C_DSA, "D", 18, 5, "深部咸水层封存汇"),
              (storage[storage["storage_type"] == "eor"], C_EOR, "D", 18, 5, "驱油封存汇")]

    def draw_layers(ax) -> None:
        for (_, colour, lw, alpha, ls, z, _), xys in zip(groups, lines):
            for xy in xys:
                ax.plot(*xy, color=colour, linewidth=lw, alpha=alpha, linestyle=ls,
                        solid_capstyle="round", zorder=z)
        for frame, colour, marker, size, z, _ in points:
            ax.scatter(frame["x"], frame["y"], s=size, c=colour, marker=marker,
                       edgecolors="white", linewidths=0.3, zorder=z)

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    draw_china_basemap(ax, facecolor="#F7F8F9")
    draw_layers(ax)
    for _, row in storage.iterrows():
        ax.annotate(str(row["storage_hub_id"]), xy=(row["x"], row["y"]), xytext=(2, 2),
                    textcoords="offset points", fontsize=5, color=C_DSA, zorder=6)
    mainland_extent(ax)
    ax.set_axis_off()
    add_scs_inset(fig, ax, draw=draw_layers)

    handles = [Line2D([], [], color=colour, linewidth=max(lw, 0.8), linestyle=ls, label=f"{name}（{len(frame)}）")
               for frame, colour, lw, _, ls, _, name in reversed(groups)]
    handles += [Line2D([], [], marker=marker, color="none", markerfacecolor=colour, markeredgecolor="white",
                       markeredgewidth=0.3, markersize=4.5 if marker == "D" else 3.5,
                       label=f"{name}（{len(frame)}）")
                for frame, colour, marker, _, _, name in points]
    ax.legend(handles=handles, loc="lower left", fontsize=6, frameon=False,
              title=f"候选边共 {len(edges)} 条", title_fontsize=6)

    save_fig(fig, "candidate_network", out_dir=ROOT / "inputs" / "figures")


if __name__ == "__main__":
    main()
