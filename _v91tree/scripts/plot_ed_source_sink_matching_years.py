# -*- coding: utf-8 -*-
"""ED14: source-sink matching in 2040, 2050, and 2060 under two water cases."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from plot_style import apply_style, panel_label, save_fig, DOUBLE_COL  # noqa: E402
from plot_ed_source_sink_matching import (  # noqa: E402
    BASE,
    TREAT,
    PIPE_C,
    EXCL_C,
    SINK_EDGE,
    SRC_C,
    SRC_OFF_C,
    FLOW_REF,
    draw_matching,
)


YEARS = (2040, 2050, 2060)
ROWS = (
    (BASE, "不考虑水\n（BASE）"),
    (TREAT, "考虑水\n（s = 0.85）"),
)


def build_figure():
    """Build the two-water-state by three-year comparison grid."""
    apply_style()
    fig_w, fig_h = DOUBLE_COL[0], 5.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("源汇匹配格局随时间演变", fontsize=9.0, y=0.988)

    left, right = 0.055, 0.985
    col_gap = 0.020
    cell_w = (right - left - 2 * col_gap) / 3.0
    map_h = 0.355
    row_bottoms = (0.565, 0.155)
    records = []

    for row, ((scenario, row_label), bottom) in enumerate(zip(ROWS, row_bottoms)):
        fig.text(0.018, bottom + map_h / 2, row_label, rotation=90,
                 ha="center", va="center", fontsize=7.0, fontweight="bold",
                 linespacing=1.25)
        other = TREAT if scenario == BASE else BASE
        for col, year in enumerate(YEARS):
            x0 = left + col * (cell_w + col_gap)
            ax = fig.add_axes([x0, bottom, cell_w, map_h])
            title = f"{year} 年" if row == 0 else ""
            result = draw_matching(
                fig,
                ax,
                scenario,
                title,
                [0.795, 0.020, 0.150, 0.215],
                other=other,
                year=year,
            )
            records.append(result)
            panel_label(ax, chr(ord("a") + row * len(YEARS) + col),
                        x=0.005, y=1.015, fontsize=8.5)
            ax.text(0.018, 0.025,
                    f"{result['n_edge']} 条管段｜{result['n_sink']} 个封存汇",
                    transform=ax.transAxes, fontsize=5.4, color="#000000",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="square,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.78),
                    zorder=8)

    handles = [
        Line2D([], [], color=PIPE_C, lw=1.65,
               label=f"两种口径共有管段（线宽 ∝ 流量；满标度 {FLOW_REF:.0f} Mt yr$^{{-1}}$）"),
        Line2D([], [], color=EXCL_C, lw=1.65, label="同年仅该口径启用的管段"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", mew=0.42,
               markeredgecolor=SINK_EDGE, markersize=3.4, label="陆上封存汇（面积 ∝ 注入量）"),
        Line2D([], [], marker="s", color="none", markerfacecolor="white", mew=0.42,
               markeredgecolor=SINK_EDGE, markersize=3.2, label="海上封存汇"),
        Line2D([], [], marker="o", color="none", markerfacecolor=SRC_C,
               markeredgecolor="none", markersize=1.9, label="接入源汇网络的厂址"),
        Line2D([], [], marker="o", color="none", markerfacecolor=SRC_OFF_C,
               markeredgecolor="none", markersize=1.5, label="未接入的厂址"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.066),
               fontsize=5.5, frameon=False, handlelength=1.6, handletextpad=0.5,
               ncol=3, columnspacing=1.25, labelspacing=0.35)
    fig.text(
        0.055, 0.018,
        "注：每一列比较同一年份；六图共用管段线宽和封存汇点面积标度。红色表示同年仅该用水口径启用的管段。",
        fontsize=5.6, color="#4A4A4A", ha="left", va="bottom",
    )
    return fig, records


def main() -> None:
    fig, records = build_figure()
    save_fig(fig, "ed_fig14_source_sink_matching_years", subdir="extended")
    print("ED14 - source-sink matching by year")
    for record in records:
        label = "BASE" if record["scenario"] == BASE else "s=0.85"
        print(f"  {record['year']} {label:6}  {record['n_edge']:3d} edges  "
              f"{record['n_sink']:2d} sinks  {record['flow']:.1f} Mt flow")


if __name__ == "__main__":
    main()
