# -*- coding: utf-8 -*-
"""幻灯片 p51：ED14 的 a–c 三张地图单独排成一行（燃料 / 水 / CO2 三条链在同一批机组上）。

和论文里的 ED14 共用同一批面板函数，只换版式与字号：论文版是 7 个面板、183 mm 幅面，
放到 13.3 in 的幻灯片上字会小到看不清。这里只取三张地图 + 三条图例，按 2106 x 792 px
出图，正好填满原 p51 的图位。底图走统一后的 plot_style（2023 版省界 + 九段线单独成层）。
输出 results/figures/slides/multihop_row.png / .pdf。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colorbar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import plot_ed_multihop_matching as MH                    # noqa: E402
from plot_style import PATHWAY_COLORS, PATHWAY_LABELS, apply_style, panel_label  # noqa: E402

OUT = HERE.parent.parent / "results" / "figures" / "slides"
FIG_W, FIG_H, DPI = 10.53, 3.96, 200
LG = 6.6            # 图例字号：幻灯片上要比论文版的 4.9 大一圈
TITLE_PT = 8.8


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    left, right, gap = 0.030, 0.985, 0.018
    cell_w = (right - left - 2 * gap) / 3.0
    map_bottom, map_h = 0.300, 0.650
    inset = [0.795, 0.020, 0.150, 0.215]
    records = {}
    for i, fn in enumerate((MH.panel_co2, MH.panel_fuel, MH.panel_water)):
        ax = fig.add_axes([left + i * (cell_w + gap), map_bottom, cell_w, map_h])
        records[fn.__name__] = fn(fig, ax, MH.TREAT, MH.MAP_YEAR, inset)
        ax.title.set_fontsize(TITLE_PT)
        panel_label(ax, "abc"[i], x=0.005, y=1.015, fontsize=11.0)

    lg_y = map_bottom - 0.006
    handles_a = [
        Line2D([], [], color=MH.PIPE_C, lw=2.0,
               label=f"管段（线宽 ∝ 流量，满标度 {MH.FLOW_REF:.0f} Mt yr$^{{-1}}$；三角 = 流向）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MH.SINK_EDGE,
               mew=0.6, markersize=4.6, label="在用封存汇（面积 ∝ 注入量；□ 海上）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MH.SINK_IDLE_C,
               mew=0.5, markersize=3.4, label="未启用的封存汇"),
        Line2D([], [], marker="o", color="none", markerfacecolor=PATHWAY_COLORS["ccs"],
               markeredgecolor="#1F2933", mew=0.4, markersize=4.4, label="机组（面积 ∝ 装机；描边 = 已接入）"),
    ]
    fig.legend(handles=handles_a, loc="upper left", bbox_to_anchor=(left, lg_y), fontsize=LG,
               frameon=False, handlelength=1.5, handletextpad=0.45, labelspacing=0.32, ncol=1)
    fig.legend(handles=[Patch(facecolor=PATHWAY_COLORS[k], label=PATHWAY_LABELS[k])
                        for k in ("unabated", "biomass", "ccs", "beccs", "retire")],
               loc="upper left", bbox_to_anchor=(left + cell_w + gap, lg_y), fontsize=LG,
               frameon=False, handlelength=1.1, handletextpad=0.4, labelspacing=0.32, ncol=2,
               columnspacing=1.0, title="a、b 机组填色 = 主导路径", title_fontsize=LG)
    handles_b = [Line2D([], [], color=MH.BIO_C, lw=1.6, label="生物质供给辐条（线宽 ∝ 供给量）")]
    if records["panel_fuel"]["nh3_kt"] > 0:
        handles_b.append(Line2D([], [], color=MH.NH3_C, lw=1.6, label="氨供给辐条"))
    fig.legend(handles=handles_b, loc="upper left", bbox_to_anchor=(left + cell_w + gap, lg_y - 0.175),
               fontsize=LG, frameon=False, handlelength=1.5, handletextpad=0.45, labelspacing=0.32)
    handles_c = [Line2D([], [], color=MH.WATER_C, lw=1.4, label="取水辐条（线宽 ∝ 取水量）"),
                 Patch(facecolor=MH.WET_C, label="湿冷取水机组"),
                 Patch(facecolor=MH.AIR_C, label="已转空冷（含既有）"),
                 Patch(facecolor=MH.RET_C, label="退役")]
    fig.legend(handles=handles_c, loc="upper left", bbox_to_anchor=(left + 2 * (cell_w + gap), lg_y),
               fontsize=LG, frameon=False, handlelength=1.3, handletextpad=0.45, labelspacing=0.32,
               ncol=2, columnspacing=1.0)
    cax = fig.add_axes([left + 2 * (cell_w + gap) + 0.012, lg_y - 0.185, cell_w - 0.03, 0.022])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=MH.RATIO_CMAP, norm=MH.RATIO_NORM,
                                          orientation="horizontal", spacing="uniform",
                                          ticks=MH.RATIO_BOUNDS)
    cb.ax.set_xticklabels([f"{b:g}" for b in MH.RATIO_BOUNDS[:-1]] + [">1.5"], fontsize=LG - 0.4)
    cb.ax.tick_params(length=1.8, pad=1.2)
    cb.set_label("流域电力取水 / 配额（配额 = 枯水期径流 × 0.20 × (1 - 0.85)）", fontsize=LG - 0.2,
                 labelpad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("multihop_row." + ext), dpi=DPI, facecolor="white")
    plt.close(fig)
    print("multihop_row %d 年（%s）｜%d x %d px" % (MH.MAP_YEAR, MH.TREAT,
                                                  int(FIG_W * DPI), int(FIG_H * DPI)))
    for k, v in records.items():
        print("  %-12s %s" % (k, {kk: (round(vv, 1) if isinstance(vv, float) else vv)
                                  for kk, vv in v.items()}))


if __name__ == "__main__":
    main()
