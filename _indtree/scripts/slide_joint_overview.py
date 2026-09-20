# -*- coding: utf-8 -*-
"""幻灯片用：电力 + 工业的系统级总览（部门碳目标算例 ST_，v9 管网重建后）。

一句话：电力可以靠退役与生物质掺烧先减，工业除了捕集几乎没有别的手段，
所以 CO2 管网在 2040 年是被工业拉起来的；水约束只咬住 2040 年的煤电。

三个面板都只用求解结果里的量，不做任何外推：
  a  排放去向 = 残余 + 各类减排手段，按"是否需要 CO2 管网"分组（柱高即当年基线）
  b  捕集量分部门 + 在用管网长度（谁把管网拉起来）
  c  水约束的相对影响（考虑水 vs 不考虑水，按不考虑水的量取百分比）
输出 results/figures/slides/joint_overview.png / .pdf。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from plot_style import apply_style                         # noqa: E402

TREE = HERE.parent
RES = TREE / "results"
OUT = TREE.parent / "results" / "figures" / "slides"
MAIN = "ST_WA_cwatm_126_dry_oq"
CTRL = "ST_BASE"
YEARS = (2030, 2040, 2050, 2060)
ACTIVE = 0.01

INK, MUTE, HAIR, GRID = "#2B2723", "#8A8079", "#DCD4CA", "#EFEAE3"
# 颜色与 p52 的动图一致：电力走绿 / 深灰，工业走紫 / 赭
C_REST = "#E3DED7"          # 残余排放
C_P_OFF = "#7FD79A"         # 电力：不依赖管网（提前退役 + 生物质掺烧）
C_P_NET = "#12A54F"         # 电力：依赖管网（BECCS + CCS）
C_I_NET = "#8B2FD0"         # 工业：依赖管网（CCS）
C_I_OFF = "#D9911F"         # 工业：不依赖管网（氢替代）
NETMARK = "///"             # 依赖管网的手段统一加斜纹
PIPE = "#121212"


def read(case: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RES / case / name)


def table(case: str) -> pd.DataFrame:
    """每个规划期一行：基线、四类减排手段、残余、捕集、管网、在用汇。"""
    ps = read(case, "pathway_shares.csv")
    pl = read(case, "plant_detail.csv")
    ind = read(case, "industry_detail.csv")
    net = read(case, "network_edges.csv")
    sto = read(case, "storage_utilization.csv")
    ab = ps.groupby(["year", "pathway"])["abatement_mt"].sum().unstack(fill_value=0.0)
    rows = []
    for y in YEARS:
        p, i = pl[pl.year == y], ind[ind.year == y]
        p_base, p_red = p["baseline_emissions_mt"].sum(), p["reduction_mt"].sum()
        p_net = float(ab.loc[y].get("beccs", 0.0) + ab.loc[y].get("ccs", 0.0))
        i_cap = i["captured_mt"].sum()
        rows.append({
            "year": y,
            "base": p_base + i["baseline_co2_mt"].sum(),
            "p_off": p_red - p_net, "p_net": p_net,
            "i_net": i_cap, "i_off": i["reduction_mt"].sum() - i_cap,
            "rest": (p_base - p_red) + i["residual_mt"].sum(),
            "coal_cap": p["captured_mt"].sum(), "ind_cap": i_cap,
            "km": net[(net.year == y) & (net.edge_flow_mtpa > ACTIVE)]["length_km"].sum(),
            "sinks": int((sto[(sto.year == y) & (sto.storage_use_mtpa > ACTIVE)]).shape[0]),
        })
    return pd.DataFrame(rows).set_index("year")


def panel_a(ax, d):
    x = np.arange(len(YEARS))
    segs = [("rest", C_REST, None, "残余排放"),
            ("p_off", C_P_OFF, None, "电力：提前退役 + 生物质掺烧"),
            ("p_net", C_P_NET, NETMARK, "电力：BECCS + CCS"),
            ("i_net", C_I_NET, NETMARK, "工业：CCS"),
            ("i_off", C_I_OFF, None, "工业：氢替代")]
    bottom = np.zeros(len(YEARS))
    for key, colour, hatch, _ in segs:
        v = d[key].to_numpy()
        ax.bar(x, v, width=0.56, bottom=bottom, color=colour, hatch=hatch,
               edgecolor="white", linewidth=0.5, zorder=3)
        bottom = bottom + v
    for xi, total in zip(x, d["base"].to_numpy()):
        ax.text(xi, total + 90, "%.0f" % total, ha="center", va="bottom",
                fontsize=7.4, color=INK, fontweight="bold")
    ax.text(x[0], d["base"].iloc[0] + 430, "柱高 = 当年基线排放", fontsize=7.0, color=MUTE,
            ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEARS])
    ax.set_ylabel(r"Mt CO$_2$ yr$^{-1}$", fontsize=7.6)
    ax.set_ylim(0, d["base"].max() * 1.20)
    ax.set_title("a   排放去向：谁用什么手段减", loc="left", pad=6, fontsize=8.6, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=t) for _, c, h, t in segs]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.0,
              handlelength=1.5, handleheight=1.0, labelspacing=0.35, borderaxespad=0.2)


def panel_b(ax, d):
    x = np.arange(len(YEARS))
    ax.bar(x, d["i_net"], width=0.52, color=C_I_NET, edgecolor="white", linewidth=0.5,
           hatch=NETMARK, label="工业捕集", zorder=3)
    ax.bar(x, d["coal_cap"], width=0.52, bottom=d["i_net"], color=C_P_NET, edgecolor="white",
           linewidth=0.5, hatch=NETMARK, label="煤电捕集", zorder=3)
    for xi, y0, y1, n in zip(x, d["i_net"], d["coal_cap"], d["sinks"]):
        if y0 + y1 > 1:
            ax.text(xi, y0 + y1 + 22, "%d 个汇" % n, ha="center", va="bottom",
                    fontsize=6.8, color=MUTE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEARS])
    ax.set_ylabel(r"捕集量  Mt CO$_2$ yr$^{-1}$", fontsize=7.6)
    ax.set_ylim(0, max(d["i_net"] + d["coal_cap"]) * 1.30)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("b   管网由谁拉起来", loc="left", pad=6, fontsize=8.6, color=INK)
    tw = ax.twinx()
    tw.plot(x, d["km"] / 1e4, color=PIPE, lw=1.5, marker="o", ms=4.0, zorder=5,
            markerfacecolor="white", markeredgewidth=1.2)
    tw.set_ylabel("在用管网  万 km", fontsize=7.6, color=PIPE)
    tw.set_ylim(0, max(d["km"] / 1e4) * 1.30)
    tw.spines["right"].set_visible(True)
    tw.spines["right"].set_color(PIPE)
    tw.tick_params(axis="y", colors=PIPE, labelsize=7)
    # 工业占当年捕集量的比例直接写在紫色柱里：比拉一条注释线稳，也顺带让"煤电追上来"这件事看得见。
    for xi, y0, y1 in zip(x, d["i_net"], d["coal_cap"]):
        if y0 + y1 > 1:
            ax.text(xi, y0 * 0.5, "%.0f%%" % (100.0 * y0 / (y0 + y1)), ha="center",
                    va="center", fontsize=8.0, color="white", fontweight="bold", zorder=6,
                    bbox=dict(facecolor=C_I_NET, edgecolor="none", pad=1.6))
    ax.legend(loc="upper left", frameon=False, fontsize=7.0, handlelength=1.5,
              labelspacing=0.3, borderaxespad=0.2)


def panel_c(ax, wa, base):
    """水约束的相对影响：(考虑水 − 不考虑水) / 不考虑水。2030 全零，不画。"""
    items = [("coal_cap", C_P_NET, "煤电捕集"), ("ind_cap", C_I_NET, "工业捕集"),
             ("km", PIPE, "在用管网")]
    years = [y for y in YEARS if y != 2030]
    x = np.arange(len(years))
    width = 0.26
    for k, (key, colour, label) in enumerate(items):
        vals = []
        for y in years:
            b = float(base.loc[y, key])
            vals.append(100.0 * (float(wa.loc[y, key]) - b) / b if abs(b) > 1e-9 else 0.0)
        pos = x + (k - 1) * width
        ax.bar(pos, vals, width=width * 0.9, color=colour, edgecolor="white", linewidth=0.4,
               label=label, zorder=3)
        for xi, v in zip(pos, vals):
            txt = "0" if abs(v) < 0.5 else "%.0f" % v          # 避免打出负零
            ax.text(xi, v - 2.4 if v < 0 else v + 2.4, txt, ha="center",
                    va="top" if v < 0 else "bottom", fontsize=6.8, color=colour)
    ax.axhline(0.0, color=INK, lw=0.7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("相对不考虑水的变化  %", fontsize=7.6)
    ax.set_ylim(-104, 26)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("c   水约束只咬住 2040 年的煤电", loc="left", pad=6, fontsize=8.6, color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=7.0, handlelength=1.4,
              labelspacing=0.3, borderaxespad=0.2)


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    wa, base = table(MAIN), table(CTRL)
    fig = plt.figure(figsize=(12.40, 4.10), dpi=200)
    fig.patch.set_facecolor("white")
    panel_a(fig.add_axes([0.043, 0.135, 0.372, 0.760]), wa)
    panel_b(fig.add_axes([0.508, 0.135, 0.185, 0.760]), wa)
    panel_c(fig.add_axes([0.790, 0.135, 0.185, 0.760]), wa, base)
    fig.add_artist(Line2D([0.455, 0.455], [0.10, 0.93], color=HAIR, lw=0.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.742, 0.742], [0.10, 0.93], color=HAIR, lw=0.8,
                          transform=fig.transFigure))
    fig.text(0.043, 0.022, "主情景 ST_WA_cwatm_126_dry_oq（考虑水约束）；面板 c 的对照为 ST_BASE。"
                           "斜纹 = 必须依托 CO2 管网的手段。", fontsize=6.8, color=MUTE,
             ha="left", va="bottom")
    for ext in ("png", "pdf"):
        p = OUT / ("joint_overview.%s" % ext)
        fig.savefig(p, dpi=200, facecolor="white")
        print("saved", p)
    plt.close(fig)
    print(wa.round(1).to_string())


if __name__ == "__main__":
    main()
