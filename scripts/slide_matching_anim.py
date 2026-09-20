# -*- coding: utf-8 -*-
"""幻灯片用：源汇匹配格局的跨期演变（2030/2040/2050/2060）逐期单帧 + 动图。

不改论文脚本，只复用读数函数。配色重做：暖色底 + 琥珀管网 + 酒红差异段，全图无蓝色；
机组按主导改造技术分形状与颜色（技术色沿用 plot_style.PATHWAY_COLORS，与 p48/p50 一致）。
输出 results/figures/slides/anim/。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import plot_style                                          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
plot_style.RESULTS_DIR = ROOT / "results" / "figures" / "v9.1" / "data"   # 必须早于下面的 import

from plot_style import (                                   # noqa: E402
    PATHWAY_COLORS, add_scs_inset, apply_style, draw_china_basemap, mainland_extent, to_map_xy,
)
from plot_ed_water_abatement import network                # noqa: E402
from plot_ed_source_sink_matching import (                 # noqa: E402
    ACTIVE, FLOW_REF, GEOM, HUBS, SINK_REF, BASE, TREAT, edge_flows, hub_frame, sink_injection,
)

OUT = ROOT / "results" / "figures" / "slides" / "anim"
YEARS = (2030, 2040, 2050, 2060)
MAIN, CTRL = "5C3A6E", "8A6520"                 # 两种口径的标识色（非蓝、与数据色不撞）
CASES = ((TREAT, "water", "考虑水约束 · 主情景", "#" + MAIN),
         (BASE, "nowater", "不考虑水约束 · 对照", "#" + CTRL))
LAND, PROV, BORDER = "#F6F1E9", "#E2D9CC", "#3F3730"
PIPE, PIPE_X, SINK_E = "#C4471C", "#7D1538", "#2F2A3D"
INK, MUTE, HAIR = "#2B2723", "#8A8079", "#DCD4CA"
FIG_W, FIG_H, DPI = 5.80, 3.62, 190
PX = 0.735                                      # 右栏左边界
MAP_RECT = [0.0, 0.070, 0.715, 0.915]
# 机组：主导改造技术 -> (颜色, 形状, 相对大小, 图例名)
TECH = [("unabated", PATHWAY_COLORS["unabated"], "o", 0.58, 0.55, "未改造燃煤"),
        ("retire", PATHWAY_COLORS["retire"], "o", 0.52, 0.55, "提前退役"),
        ("biomass", PATHWAY_COLORS["biomass"], "o", 0.85, 0.92, "生物质掺烧"),
        ("beccs", PATHWAY_COLORS["beccs"], "^", 1.00, 0.95, "BECCS"),
        ("ccs", PATHWAY_COLORS["ccs"], "s", 0.90, 0.95, "CCS 改造")]


def draw_layers(a, flows, exclusive, inj, plants, k=1.0):
    """一帧的业务图层：管网 → 机组（按技术）→ 封存汇。"""
    for only, colour, z in ((False, PIPE, 3.0), (True, PIPE_X, 3.4)):
        segs, widths = [], []
        for eid, f in flows.items():
            if (eid in exclusive) != only or GEOM.get(eid) is None:
                continue
            segs.append(GEOM[eid])
            widths.append(k * (0.85 + 3.30 * np.sqrt(min(f / FLOW_REF, 1.0))))
        if segs:
            a.add_collection(LineCollection(segs, colors=colour, linewidths=widths,
                                            capstyle="round", alpha=0.92, zorder=z))
    for key, colour, marker, rel, alpha, _ in TECH:
        sub = plants[plants["dominant_pathway"] == key]
        if sub.empty:
            continue
        px, py = to_map_xy(sub["centroid_longitude"].to_numpy(), sub["centroid_latitude"].to_numpy())
        a.scatter(px, py, s=k * rel * (1.6 + 3.4 * sub["cap_gw"].to_numpy()), marker=marker,
                  facecolor=colour, edgecolor="#5A5248", linewidth=k * 0.16, alpha=alpha,
                  zorder=2.0 + rel * 0.2)
    keys = list(inj.index)
    if not keys:
        return
    hx, hy = to_map_xy(np.array([float(HUBS["longitude"][q]) for q in keys]),
                       np.array([float(HUBS["latitude"][q]) for q in keys]))
    off = np.array([bool(HUBS["offshore"].get(q, False)) for q in keys])
    size = k * (14.0 + 150.0 * np.sqrt(inj.to_numpy() / SINK_REF))
    for m, sel in (("o", ~off), ("H", off)):
        if sel.any():
            a.scatter(hx[sel], hy[sel], s=size[sel], marker=m, facecolor="white",
                      edgecolor=SINK_E, linewidth=k * 1.05, zorder=5)


def lg_row(fig, y, marker, colour, text, edge=None, lw=0.0, ms=6.0, line=False):
    if line:
        fig.add_artist(Line2D([PX + 0.005, PX + 0.045], [y, y], color=colour, lw=3.0,
                              solid_capstyle="round", transform=fig.transFigure))
    else:
        fig.add_artist(Line2D([PX + 0.025], [y], marker=marker, ms=ms, ls="none",
                              markerfacecolor=colour, markeredgecolor=edge or "#5A5248",
                              mew=lw or 0.4, transform=fig.transFigure))
    fig.text(PX + 0.062, y, text, fontsize=8.4, color=INK, ha="left", va="center")


def hairline(fig, y):
    fig.add_artist(Line2D([PX, 0.995], [y, y], color=HAIR, lw=0.9, transform=fig.transFigure))


def progress(fig, year, accent):
    x0, x1 = 0.045, 0.660
    fig.add_artist(Line2D([x0, x1], [0.042, 0.042], color=HAIR, lw=1.4,
                          transform=fig.transFigure))
    for i, y in enumerate(YEARS):
        x = x0 + i * (x1 - x0) / (len(YEARS) - 1.0)
        on = y == year
        fig.add_artist(Line2D([x], [0.042], marker="o", ms=9.0 if on else 5.0, ls="none",
                              markerfacecolor=accent if on else "#FFFFFF",
                              markeredgecolor=accent if on else HAIR, mew=1.4,
                              transform=fig.transFigure))
        fig.text(x, 0.0, "%d" % y, fontsize=8.6, ha="center", va="bottom",
                 color=accent if on else MUTE, fontweight="bold" if on else "normal")


def frame(scen, other, year, label, accent):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes(MAP_RECT)

    flows = edge_flows(scen, year)
    flows = flows[flows > ACTIVE]
    other_flows = edge_flows(other, year)
    exclusive = set(flows.index) - set(other_flows[other_flows > ACTIVE].index)
    inj = sink_injection(scen, year)
    inj = inj[inj > ACTIVE]
    plants = hub_frame(scen, year)

    def business(a, k=1.0):
        draw_layers(a, flows, exclusive, inj, plants, k)

    draw_china_basemap(ax, province_lw=0.28, country_lw=0.80, facecolor=LAND)
    for coll, colour in zip(ax.collections, (None, PROV, BORDER)):
        if colour:
            coll.set_color(colour)
    business(ax)
    mainland_extent(ax)
    ax.set_axis_off()
    for art in list(ax.collections) + list(ax.lines) + list(ax.patches):
        art.set_clip_on(True)
        art.set_clip_box(ax.bbox)
    scs = add_scs_inset(fig, ax, draw=lambda a: business(a, 0.55),
                        axes_rect=[0.012, 0.02, 0.155, 0.27])
    for s in scs.spines.values():
        s.set_visible(True)
        s.set_edgecolor(HAIR)

    fig.text(PX, 0.930, "%d" % year, fontsize=30, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(PX + 0.175, 0.918, "年", fontsize=13, color=MUTE, ha="left", va="center")
    fig.add_artist(FancyBboxPatch((PX, 0.812), 0.252, 0.052, boxstyle="round,pad=0.004,rounding_size=0.012",
                                  facecolor=accent, edgecolor="none", transform=fig.transFigure))
    fig.text(PX + 0.126, 0.838, label, fontsize=8.8, color="white", ha="center", va="center",
             fontweight="bold")
    hairline(fig, 0.782)
    cap, n_sink = float(plants["captured_mt"].sum()), len(inj)
    km = float(network(scen, year)["length_km"].sum())
    rows = ([("捕集", "尚未出现"), ("封存汇", "尚未启用"), ("管网", "尚未建设")] if n_sink == 0 else
            [("捕集", "%.0f Mt/年" % cap), ("封存汇", "%d 个" % n_sink),
             ("管网", "%.2f 万 km" % (km / 1e4))])
    for i, (k, v) in enumerate(rows):
        y = 0.735 - i * 0.062
        fig.text(PX, y, k, fontsize=9.6, color=MUTE, ha="left", va="center")
        fig.text(0.995, y, v, fontsize=11.6, color=INK, ha="right", va="center", fontweight="bold")
    hairline(fig, 0.522)
    fig.text(PX, 0.487, "机组 · 按主导改造技术", fontsize=8.4, color=MUTE, ha="left", va="center")
    order = [TECH[2], TECH[3], TECH[4], TECH[0], TECH[1]]      # 图例按重要性排，不按画序
    for i, (_, colour, marker, rel, _a, name) in enumerate(order):
        lg_row(fig, 0.437 - i * 0.048, marker, colour, name, ms=4.6 + 2.2 * rel)
    fig.text(PX, 0.158, "CO2 管网与封存", fontsize=8.4, color=MUTE, ha="left", va="center")
    lg_row(fig, 0.110, None, PIPE, "两种口径共有管段", line=True)
    lg_row(fig, 0.062, None, PIPE_X, "仅本口径启用的管段", line=True)
    lg_row(fig, 0.014, "o", "white", "封存汇（六边形为海上）", edge=SINK_E, lw=1.1, ms=7.0)
    progress(fig, year, accent)
    return fig


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for scen, key, label, accent in CASES:
        other = BASE if scen == TREAT else TREAT
        paths = []
        for year in YEARS:
            fig = frame(scen, other, year, label, accent)
            p = OUT / ("match_%s_%d.png" % (key, year))
            fig.savefig(p, dpi=DPI, facecolor="white")
            plt.close(fig)
            paths.append(p)
            print("  frame", p.name)
        ims = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=220) for p in paths]
        gif = OUT / ("match_%s.gif" % key)
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=[1150, 1150, 1150, 1550],
                    loop=0, disposal=2, optimize=True)
        print("GIF", gif.name, "%.2f MB" % (gif.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
