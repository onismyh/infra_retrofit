"""Supplementary: the transition province by province, and what water changes about it.

WHY PROVINCE, WHEN THE CONSTRAINT IS A BASIN CONSTRAINT. Everywhere else in this appendix the unit
is the site or the basin, deliberately: provinces straddle basin divides, so a provincial mean
averages across the boundary the water argument is about. But the transition itself -- who retires,
who co-fires, who takes capture, and when -- is decided and financed provincially. A reader who
wants to act on this paper needs it in the unit the policy is written in, so these two figures give
it, with the caveat below attached rather than hidden.

CAVEAT ON PROVINCIAL ATTRIBUTION. The 350 modelled objects are CLUSTERS of units, not single
plants: the largest holds 27.9 GW and 70 units. Each is booked here to the province holding most
of its CAPACITY. The model's own `province_mode` is a mode over UNITS, which lets many small
units outvote the big ones -- it books that 27.9 GW cluster, standing on Shanghai's coordinates,
entirely to Jiangsu and leaves Shanghai with no capacity at all. Six sites carrying 54.0 GW move
under the capacity rule. Provincial totals remain exact for the model and approximate for the
country, because a cluster still cannot be split across a provincial boundary.

WHAT THE TWO PANELS SHOW.
Figure 8 is the transition itself: each province's capacity split across unabated, biomass
co-firing, BECCS, CCS retrofit and retirement, 2030 to 2060, with dry-cooling conversion drawn over
it as a line. Pathway shares sum to exactly 1.000000 in all 1,400 plant-years, so the stacks are a
partition and not a selection.

Figure 9 is what the water reservation changes, provincially: the treatment minus the control. The
national +227 GW of extra dry-cooling conversion is not spread across the country. Shandong alone
takes +60 GW, and the top three -- Shandong, Henan, Hebei -- take 56% of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ed_plant_data as EP
from plot_style import MM, save_fig, PROV_ZH

CONTROL, TREAT = "WA_cwatm_126_dry_oq_envonly", "WA_cwatm_126_dry_oq"
YEARS = (2030, 2040, 2050, 2060)
# 配色改用 .claude/CLAUDE.md §3.3 的同族同色系表：Greys 浅端=未改造、深端=CCS；
# Greens 浅端=生物质掺烧、深端=BECCS；退役用中性灰。原先的 Tol bright 系已停用。
STACK = [("share_unabated", "未改造燃煤", "#969696"),
         ("share_biomass", "生物质掺烧", "#74C476"),
         ("share_beccs", "BECCS", "#31A354"),
         ("share_ccs", "CCS 改造", "#636363"),
         ("share_retire", "提前退役", "#D8DCE0")]
CONV_C = "#CC3311"


def provincial(scenario: str) -> pd.DataFrame:
    """Capacity by province, year and pathway, in GW.

    `share_*` are stocks and shares of the site's capacity; they sum to 1 exactly, so multiplying
    by capacity and summing gives a partition of the province's fleet.

    Sites are booked to the province holding most of their CAPACITY, not the one holding most of
    their UNITS. `plant_detail.csv` inherits `province_mode`, a plain mode over units, which lets
    many small units outvote the big ones: six sites carrying 54.0 GW move, the largest of them
    the 27.9 GW hub standing on Shanghai's coordinates and booked to Jiangsu -- under the model's
    own label Shanghai holds no coal capacity at all. Nothing physical changes; the optimisation
    used `province_mode` only to look up a provincial capacity factor and was solved that way.
    """
    d = pd.read_csv(EP.RESULTS / scenario / "plant_detail.csv")
    labels = EP.capacity_mode_labels().set_index("plant_id")["province_by_capacity"]
    remapped = d["plant_id"].map(labels)
    if remapped.isna().any():
        raise RuntimeError("capacity-mode province label missing for some plant_id")
    d["province_name"] = remapped
    d["cap_gw"] = d["capacity_mw"] / 1e3
    # `air_cooled_share` is the dry-operating fraction of the STILL-WET capacity (plant_matrices._air_cooling_matrices)
    d["conv_gw"] = (d["air_cooled_share"].fillna(0.0)
                    * (1.0 - d["already_air_share"].fillna(0.0)) * d["cap_gw"])
    for col, _, _ in STACK:
        d[col + "_gw"] = d[col].fillna(0.0) * d["cap_gw"]
    agg = {col + "_gw": (col + "_gw", "sum") for col, _, _ in STACK}
    agg.update(cap_gw=("cap_gw", "sum"), conv_gw=("conv_gw", "sum"),
               captured_mt=("captured_mt", "sum"))
    return d.groupby(["province_name", "year"]).agg(**agg).reset_index()


def figure_transition(t: pd.DataFrame) -> None:
    order = (t[t["year"] == 2030].set_index("province_name")["cap_gw"]
             .sort_values(ascending=False))
    show = list(order.index[:20])
    ncol, nrow = 5, 4
    # Panels were touching: at gapx = 0.010 the "60" of one panel and the "30" of the next
    # rendered as "6030", and the header sat on top of the first row of panel titles.
    fig = plt.figure(figsize=(183 * MM, 148 * MM))
    left, bottom, w, h = 0.050, 0.075, 0.170, 0.1725
    gapx, gapy = 0.022, 0.048

    for i, prov in enumerate(show):
        r, c = divmod(i, ncol)
        ax = fig.add_axes([left + c * (w + gapx), bottom + (nrow - 1 - r) * (h + gapy), w, h])
        sub = t[t["province_name"] == prov].set_index("year").reindex(YEARS)
        x = np.arange(len(YEARS))
        base = np.zeros(len(YEARS))
        for col, _, colour in STACK:
            v = sub[col + "_gw"].to_numpy(dtype=float)
            ax.fill_between(x, base, base + v, color=colour, lw=0.25, edgecolor="white",
                            zorder=3)
            base = base + v
        ax.plot(x, sub["conv_gw"].to_numpy(dtype=float), color=CONV_C, lw=1.2, zorder=6)

        cap = float(order[prov])
        ax.set_xlim(0, len(YEARS) - 1)
        ax.set_ylim(0, cap * 1.06)
        ax.set_xticks(x)
        ax.set_xticklabels(["30", "40", "50", "60"], fontsize=4.6)
        ax.tick_params(labelsize=4.6, length=1.4, pad=1.0)
        ax.set_title(f"{PROV_ZH.get(prov, prov)}  {cap:.0f} GW", fontsize=5.6, pad=2.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if c == 0:
            ax.set_ylabel("GW", fontsize=5.4)

    handles = [Patch(facecolor=col, label=lab) for _, lab, col in STACK]
    handles.append(Line2D([], [], color=CONV_C, lw=1.4, label="空冷改造"))
    fig.legend(handles=handles, fontsize=5.8, frameon=False, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, 0.005), handlelength=1.3,
               handletextpad=0.5, columnspacing=1.4)
    fig.text(0.050, 0.995,
             "水资源约束（生态流量 ＋ 用水总量控制指标）下的分省转型路径，按容量排序的前 20 个省。"
             "堆叠为完全划分："
             "\n每个厂址-年份的各路径份额之和恰为 1.000000；横轴为 2030–2060 年；各面板纵轴独立。",
             fontsize=7.0, linespacing=1.3, va="top")
    save_fig(fig, "ed_fig8_province_transition", subdir="extended")
    return show


def figure_response(t: pd.DataFrame, c: pd.DataFrame, show: list) -> None:
    m = t.merge(c, on=["province_name", "year"], suffixes=("_t", "_c"))
    y30 = m[m["year"] == 2030].copy()
    y60 = m[m["year"] == 2060].copy()
    y30["d_conv"] = y30["conv_gw_t"] - y30["conv_gw_c"]
    y60["d_ret"] = y60["share_retire_gw_t"] - y60["share_retire_gw_c"]
    y60["d_capt"] = y60["captured_mt_t"] - y60["captured_mt_c"]

    d = (y30[["province_name", "d_conv"]]
         .merge(y60[["province_name", "d_ret", "d_capt"]], on="province_name"))
    d = d[d[["d_conv", "d_ret", "d_capt"]].abs().max(axis=1) > 0.15]
    d = d.sort_values("d_conv")

    # 104 -> 116 mm: the caution note under (b) and (c) needs three lines of clearance.
    fig = plt.figure(figsize=(183 * MM, 116 * MM))
    specs = [("d_conv", "新增空冷改造，2030 年（GW）", CONV_C, 0.075),
             ("d_ret", "新增提前退役，2060 年（GW）", "#6B7683", 0.400),
             ("d_capt", "捕集量变化，2060 年（Mt CO$_2$ yr$^{-1}$）", "#3182BD", 0.725)]
    yy = np.arange(len(d))
    for k, (col, label, colour, x0) in enumerate(specs):
        ax = fig.add_axes([x0, 0.205, 0.245, 0.650])
        v = d[col].to_numpy(dtype=float)
        ax.barh(yy, v, height=0.66, color=np.where(v >= 0, colour, "#BBBBBB"),
                edgecolor="white", lw=0.3, zorder=3)
        ax.axvline(0, color="#333333", lw=0.7, zorder=4)
        if k == 0:
            ax.set_yticks(yy)
            ax.set_yticklabels([PROV_ZH.get(p, p) for p in d["province_name"]], fontsize=5.4)
        else:
            ax.set_yticks([])
        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.set_xlabel(label, fontsize=6.2)
        ax.tick_params(labelsize=5.4, length=1.6)
        ax.grid(axis="x", lw=0.3, alpha=0.30)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        top = d.nlargest(1, col).iloc[0]
        ax.annotate(f"{PROV_ZH.get(top['province_name'], top['province_name'])} {top[col]:+.0f}",
                    xy=(top[col], float(np.where(d["province_name"] == top["province_name"])[0][0])),
                    xytext=(-3, 8), textcoords="offset points", fontsize=5.2,
                    color=colour, ha="right", fontweight="bold")
        # Labels sat at 1.055 of the axes, which put them inside the two-line header block.
        ax.text(-0.02 if k == 0 else -0.06, 1.025, "abc"[k], transform=ax.transAxes,
                fontsize=8.0, fontweight="bold", va="bottom", ha="left")

    tot = d["d_conv"].sum()
    top3 = d.nlargest(3, "d_conv")
    fig.text(0.075, 0.995,
             f"让水资源预留真正起约束作用，分省会改变什么。全国新增的 "
             f"+{tot:.0f} GW 空冷改造高度集中：\n"
             f"{'、'.join(PROV_ZH.get(p, p) for p in top3['province_name'])} 三省占其中的 "
             f"{100 * top3['d_conv'].sum() / tot:.0f}%。",
             fontsize=7.2, linespacing=1.3, va="top")
    # READ (a) ONLY. Panels (b) and (c) are two orders of magnitude smaller than (a) and sit
    # inside capacity-level solver degeneracy: re-solving the same contrast on a repaired
    # source-sink network moves the national retirement difference from -1.5 to +7.9 to -3.3 GW
    # and the capture difference from -1.2 to +10.9 to -10.7 Mt/yr, while the objective effect
    # moves by 0.002 pp. Only the conversion response is stable across those three families.
    # 1/50 是 v9 的测量值。比值从本图画出的同三列自己算，口径一换它没有理由仍然成立。
    _mag = {c: float(d[c].abs().sum()) for c in ("d_conv", "d_ret", "d_capt")}
    _big = max(_mag["d_ret"], _mag["d_capt"])
    _r = _mag["d_conv"] / _big if _big > 0 else float("inf")
    _rtxt = f"1/{_r:.0f}" if np.isfinite(_r) else "可忽略"
    fig.text(0.400, 0.120,
             f"面板 b 与 c 的量级只有面板 a 的 {_rtxt}，且未被分辨出来："
             "同一对比在修复后的源汇网络上\n重新求解会翻转符号。不要从中读出任何机理。"
             "面板 a 的空冷改造响应在所有\n已测试的输入版本下都稳定。",
             fontsize=5.4, linespacing=1.35, va="top", color="#8A3324")
    save_fig(fig, "ed_fig9_province_response", subdir="extended")

    print("  provincial response, 2030 conversion (GW):")
    for r in d.sort_values("d_conv", ascending=False).head(8).itertuples():
        print(f"    {r.province_name:18s} {r.d_conv:+7.1f}")
    print(f"    national total {tot:+.1f}")


def main() -> None:
    t, c = provincial(TREAT), provincial(CONTROL)
    show = figure_transition(t)
    figure_response(t, c, show)


if __name__ == "__main__":
    main()
