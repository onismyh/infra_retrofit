# -*- coding: utf-8 -*-
"""附录图：源汇匹配格局本身 —— 有水约束 vs 无水约束。

ED12 用汇个数、注入量、管长等标量回答"匹配变了没有"。这张图把匹配格局**画出来**，
并把"变了多少"拆成两个层级分别量化。结论是这两个层级的行为完全不同：

    越靠近"存到哪"，水约束的影响越小；越靠近"怎么运过去"，影响越大。

本图固定采用 2060 年横截面。所有分歧、相关性和零假设区间都在运行时从该年结果
重新计算，并写入面板标题、图注和终端报告，不在说明中保留容易过期的硬编码数值。

也就是说：**CO2 最终存进哪些地质体，基本不受水约束影响；但它走哪条管廊过去，
是明确被水约束改写的。** 这与 fig4 在对照-处理对上的发现一致（那里能分辨的是
"共用管段输送量"与"交集/并集输送量"，不是管道总长或汇的个数），本图是在更强的
"有水约束 vs 完全无水约束"这一对上复现同一件事。

口径与 ED11 / ED12 相同（`scripts/run_single.py` 的三档）：

    BASE                    water_mode="no_water"                  完全不考虑水
    WA_*_dry                水约束在，existing_withdrawal_share=0   流域可取用量全给电力
    WA_*_dry_wd085          同上，但 s=0.85，电力只留 15%           处理组

两张地图用**完全相同**的线宽标度与点面积标度，否则并排看没有意义。
散点面板里的灰点是零假设：同一口径仅换随机种子的两两配对，它给出"什么都不改
能漂多远"的参照 —— 没有这层参照，任何 1:1 图上的散布都无法判定。

数据源：重建输入版本（103 个汇、连通性修复网络），与 v9 其余图同源。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from plot_style import (  # noqa: E402
    apply_style,
    save_fig,
    panel_label,
    draw_china_basemap,
    mainland_extent,
    add_scs_inset,
    to_map_xy,
    safe_log_axis,
    cjk_fill,
    DOUBLE_COL,
)
from plot_ed_water_on_off import (  # noqa: E402
    BASE, ARMS, SEED_ARM, floor_of, seed_scenarios, hub_frame,
)
from plot_ed_water_abatement import injection, network, HUBS, INPUTS  # noqa: E402

TREAT = f"{SEED_ARM}_wd085"
ANALYSIS_YEAR = 2060

# --- 配色（CLAUDE.md §3.3）----------------------------------------------------------------
PIPE_C = "#2171B5"        # CO2 管网：Blues 深端
SINK_EDGE = "#123F6B"     # 封存汇：白心 + 深蓝描边，与实心的管段分通道
SRC_C = "#9AA3AB"         # 接入源汇网络的厂址（有捕集）：中性灰
SRC_OFF_C = "#D5DADE"     # 未接入的厂址：更淡一档。350 个 hub 里有 224-229 个不捕集，
                          # 画上去是机队底图 —— 没有它，读者会以为管网覆盖了全部煤电
EXCL_C = "#CC3311"        # 只在本情景启用的管段：强调红，与 c/d/e 里"差异"的红同义
C_BASE = "#969696"
C_S0 = "#6BAED6"
C_S085 = "#CC3311"
NULL_C = "#BDBDBD"
GRID = "#9AA0A6"
SPINE = "#5A5A5A"

ACTIVE = 0.01             # Mt/yr，低于此视为未启用（与 ED12 一致）
FLOW_REF = 40.0           # 管段流量的标度参考（上限即单管容量）


# =========================================================================================
# 读数
# =========================================================================================
def edge_geometry() -> dict:
    """候选管廊的真实中心线，键为 edge_id，值为 EPSG:2380 下的 (x, y) 数组。

    直接走数字化路由而不是端点直线：管长本来就按 haversine x 1.136 的绕行系数算，
    画成直线会和长度口径对不上（CLAUDE.md §1.3）。
    """
    from shapely import wkt

    cand = pd.read_csv(INPUTS / "pipeline_candidate_edges.csv")
    out = {}
    for eid, geom in zip(cand["edge_id"].astype(str), cand["geometry_wkt"]):
        if not isinstance(geom, str):
            continue
        try:
            coords = np.asarray(wkt.loads(geom).coords)
        except Exception:  # noqa: BLE001 - 一行坏几何只丢一条边，不该拖垮整张图
            continue
        x, y = to_map_xy(coords[:, 0], coords[:, 1])
        out[eid] = np.column_stack([x, y])
    return out


GEOM = edge_geometry()


def edge_flows(scen: str, year: int = ANALYSIS_YEAR) -> pd.Series:
    n = network(scen, year)
    return n.groupby(n["edge_id"].astype(str))["edge_flow_mtpa"].sum()


def sink_injection(scen: str, year: int = ANALYSIS_YEAR) -> pd.Series:
    """Storage-hub injection for this figure's common analysis year."""
    return injection(scen, year)


def disagreement(a: pd.Series, b: pd.Series) -> dict:
    """两次求解在同一组对象上的分歧程度。"""
    idx = a.index.union(b.index)
    a, b = a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0)
    tot = max(float(a.sum()), float(b.sum()), 1e-9)
    return {"l1": float(np.abs(a - b).sum()) / tot * 100.0,
            "r2": float(np.corrcoef(a, b)[0, 1] ** 2),
            "flip": int(((a > ACTIVE) ^ (b > ACTIVE)).sum())}


def null_pairs(getter, stem: str) -> list[dict]:
    """零假设：同一口径仅换随机种子的两两配对。"""
    S = [getter(x) for x in seed_scenarios(stem)]
    return [disagreement(x, y) for x, y in itertools.combinations(S, 2)]


# =========================================================================================
# 面板 a / b：两张匹配格局图
# =========================================================================================
def draw_matching(fig, ax, scen: str, title: str, inset_rect, other: str,
                  year: int = ANALYSIS_YEAR) -> dict:
    flows = edge_flows(scen, year)
    flows = flows[flows > ACTIVE]
    # 只在本情景启用的管段。两张图的汇几乎重合、管段流量却差 53%，
    # 若两图都画成同一种蓝，这个差异在 85 mm 的幅面上肉眼读不出来。
    other_flows = edge_flows(other, year)
    exclusive = set(flows.index) - set(other_flows[other_flows > ACTIVE].index)
    inj = sink_injection(scen, year)
    inj = inj[inj > ACTIVE]
    hubs_all = hub_frame(scen, year)
    src = hubs_all[hubs_all["captured_mt"] > 0.01]
    src_off = hubs_all[hubs_all["captured_mt"] <= 0.01]

    def business(a):
        # 共有的先画、独有的后画：独有的是本图要看的东西，不能被压在下面。
        for only, colour, z in ((False, PIPE_C, 3), (True, EXCL_C, 3.5)):
            segs, widths = [], []
            for eid, f in flows.items():
                if (eid in exclusive) != only:
                    continue
                g = GEOM.get(eid)
                if g is None:
                    continue
                segs.append(g)
                # 线宽只编码流量。两张图共用同一条标度，否则并排比较没有意义。
                widths.append(0.30 + 1.35 * np.sqrt(min(f / FLOW_REF, 1.0)))
            if segs:
                a.add_collection(LineCollection(segs, colors=colour, linewidths=widths,
                                                capstyle="round", alpha=0.85, zorder=z))
        # 源：两级。未接入源汇的在下、更淡更小；接入的在上。两者都很小 ——
        # 它们是管网的上游语境，不是本图的判定对象，点一大就会盖住管段和汇。
        for frame, colour, size, z in ((src_off, SRC_OFF_C, 0.9, 1.6),
                                       (src, SRC_C, 1.5, 2.0)):
            if frame.empty:
                continue
            px, py = to_map_xy(frame["centroid_longitude"].to_numpy(),
                               frame["centroid_latitude"].to_numpy())
            a.scatter(px, py, s=size, color=colour, linewidth=0, zorder=z)
        # 汇：白心 + 深蓝描边，面积正比注入量。同样共用标度。
        keys = list(inj.index)
        hx, hy = to_map_xy(np.array([float(HUBS["longitude"][k]) for k in keys]),
                           np.array([float(HUBS["latitude"][k]) for k in keys]))
        off = np.array([bool(HUBS["offshore"].get(k, False)) for k in keys])
        size = 2.0 + 26.0 * np.sqrt(inj.to_numpy() / SINK_REF)
        for m, sel in (("o", ~off), ("s", off)):
            if not sel.any():
                continue
            a.scatter(hx[sel], hy[sel], s=size[sel], marker=m, facecolor="white",
                      edgecolor=SINK_EDGE, linewidth=0.42, zorder=5)

    draw_china_basemap(ax, province_lw=0.12, country_lw=0.42, facecolor="#F1F3F5")
    business(ax)
    mainland_extent(ax)
    ax.set_axis_off()
    ax.set_title(title, fontsize=6.6, pad=2.0)
    scs_ax = add_scs_inset(fig, ax, draw=business, axes_rect=inset_rect)
    for spine in scs_ax.spines.values():
        spine.set_visible(True)
    return {"scenario": scen, "year": year,
            "n_src": len(src), "n_src_off": len(src_off),
            "n_edge": len(flows), "flow": float(flows.sum()), "n_sink": len(inj),
            "inj": float(inj.sum()),
            "km": float(network(scen, year)["length_km"].sum()),
            "n_excl": len(exclusive),
            "flow_excl": float(flows[list(exclusive)].sum()) if exclusive else 0.0}


SINK_REF = 200.0          # Mt/yr，跨情景共用的点面积标度参考值


# =========================================================================================
# 面板 c / d：1:1 散点
# =========================================================================================
def scatter_panel(ax, getter, log: bool, label: str, unit: str) -> dict:
    b = getter(BASE)
    t = getter(TREAT)
    idx = b.index.union(t.index)
    b, t = b.reindex(idx).fillna(0.0), t.reindex(idx).fillna(0.0)

    # 零假设先画、画在底下：它是判定的参照，不是结论本身。
    lo = ACTIVE
    for x, y in itertools.combinations(seed_scenarios(TREAT), 2):
        sx, sy = getter(x), getter(y)
        i2 = sx.index.union(sy.index)
        sx, sy = sx.reindex(i2).fillna(0.0), sy.reindex(i2).fillna(0.0)
        keep = (sx > ACTIVE) | (sy > ACTIVE)
        ax.scatter(np.maximum(sx[keep], lo), np.maximum(sy[keep], lo), s=3.0,
                   color=NULL_C, alpha=0.55, linewidth=0, zorder=2)

    keep = (b > ACTIVE) | (t > ACTIVE)
    ax.scatter(np.maximum(b[keep], lo), np.maximum(t[keep], lo), s=5.5,
               facecolor="none", edgecolor=C_S085, linewidth=0.55, zorder=4)

    hi = float(max(b.max(), t.max())) * 1.6
    ax.plot([lo, hi], [lo, hi], color=SPINE, lw=0.6, ls=(0, (3, 2)), zorder=3)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        # log 轴的科学计数刻度会带 U+2212，SimHei 没有这个字形（CLAUDE.md §3.1）。
        safe_log_axis(ax, "x")
        safe_log_axis(ax, "y")
        ax.set_xlim(lo * 0.7, hi)
        ax.set_ylim(lo * 0.7, hi)
    else:
        ax.set_xlim(-hi * 0.02, hi)
        ax.set_ylim(-hi * 0.02, hi)
    ax.set_aspect("equal", adjustable="box")

    d = disagreement(b, t)
    nulls = null_pairs(getter, TREAT) + null_pairs(getter, SEED_ARM)
    n_l1 = [x["l1"] for x in nulls]
    ax.set_xlabel(f"不考虑水的{label}（{unit}）", fontsize=5.8, labelpad=1.5)
    ax.set_ylabel(f"考虑水 s = 0.85（{unit}）", fontsize=5.8, labelpad=1.5)
    ax.tick_params(labelsize=5.0, length=1.6, pad=1.2)
    ax.grid(lw=0.22, alpha=0.18, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
        ax.spines[s].set_color(SPINE)
    # 标题只报数字，不下判定。这里画的是单一水文臂，最终判定必须看面板 e 的三臂口径。
    ax.set_title(f"{label}：$R^2$ = {d['r2']:.2f}，$\\Sigma|\\Delta|$ = {d['l1']:.0f}%"
                 f"（零假设 {min(n_l1):.0f}–{max(n_l1):.0f}%）", fontsize=6.0, pad=3.0)
    return {"d": d, "null_l1": (min(n_l1), max(n_l1))}


# =========================================================================================
# 面板 e：两个层级的汇总
# =========================================================================================
def panel_e(ax) -> pd.DataFrame:
    levels = [("汇注入量\n（存到哪）", sink_injection),
              ("管段流量\n（怎么运）", edge_flows)]
    rows = []
    for i, (name, getter) in enumerate(levels):
        B = getter(BASE)
        ctrl = [disagreement(B, getter(a))["l1"] for a in ARMS]
        treat = [disagreement(B, getter(f"{a}_wd085"))["l1"] for a in ARMS]
        nulls = [x["l1"] for x in null_pairs(getter, TREAT) + null_pairs(getter, SEED_ARM)]
        y = len(levels) - 1 - i
        ax.barh(y, max(nulls) - min(nulls), left=min(nulls), height=0.42,
                color=NULL_C, alpha=0.65, zorder=2,
                label="仅换随机种子的零假设" if i == 0 else None)
        ax.scatter(ctrl, [y + 0.16] * 3, s=11, facecolor=C_S0, edgecolor="white",
                   linewidth=0.35, zorder=4, label="考虑水，s = 0" if i == 0 else None)
        ax.scatter(treat, [y - 0.16] * 3, s=11, facecolor=C_S085, edgecolor="white",
                   linewidth=0.35, zorder=4, label="考虑水，s = 0.85" if i == 0 else None)
        rows.append({"level": name.replace("\n", ""), "ctrl": ctrl, "treat": treat,
                     "null_lo": min(nulls), "null_hi": max(nulls),
                     "clears": min(treat) > max(nulls)})

    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([n for n, _ in levels][::-1], fontsize=5.4)
    ax.set_ylim(-0.55, len(levels) - 0.45)
    ax.set_xlim(0, 60)
    ax.set_xlabel(r"与不考虑水的分歧 $\Sigma|\Delta|$ / 总量（%）", fontsize=6.0, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.25, alpha=0.20, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(SPINE)
    # 图例放右上：汇注入量那一行的右半边（10-60%）本来就是空的，
    # 而右下正好是管段流量处理组三个点落的位置。
    ax.legend(fontsize=4.8, frameon=False, loc="upper right", handlelength=1.1,
              handletextpad=0.5, labelspacing=0.26, borderpad=0.2)
    ax.set_title("同一次求解的两个层级，判定相反", fontsize=6.6, pad=4.0)
    return pd.DataFrame(rows)


# =========================================================================================
def main() -> None:
    apply_style()
    fig_w, fig_h = DOUBLE_COL[0], 6.30
    fig = plt.figure(figsize=(fig_w, fig_h))

    map_w = 0.455
    map_h = (map_w * fig_w / 1.177) / fig_h
    map_top = 0.945
    ax_a = fig.add_axes([0.020, map_top - map_h, map_w, map_h])
    ax_b = fig.add_axes([0.520, map_top - map_h, map_w, map_h])

    # 共用图例是两行（ncol=3），占到 map_top - map_h - 0.045 附近；
    # 下排整体再让出 0.030，否则 c/d/e 的面板标号会压在图例第二行上。
    row_top = map_top - map_h - 0.115
    # 图注是 6 行（6 x 5.2 pt x 1.5 行距 = 0.103 画布高，顶到 y=0.115）。
    # 面板高 0.245 时 c/d 的横轴标题落在 0.111，正好相撞；收到 0.200 后抬到 0.156。
    # c/d 是 aspect="equal"，收高度只是把正方形画小一点，不会变形。
    ax_c = fig.add_axes([0.075, row_top - 0.200, 0.215, 0.200])
    ax_d = fig.add_axes([0.395, row_top - 0.200, 0.215, 0.200])
    ax_e = fig.add_axes([0.715, row_top - 0.160, 0.265, 0.160])

    a = draw_matching(fig, ax_a, BASE, f"不考虑水（BASE），{ANALYSIS_YEAR} 年",
                      [0.775, 0.015, 0.170, 0.230], other=TREAT)
    b = draw_matching(fig, ax_b, TREAT, f"考虑水（s = 0.85），{ANALYSIS_YEAR} 年",
                      [0.775, 0.015, 0.170, 0.230], other=BASE)
    c = scatter_panel(ax_c, sink_injection, True, "汇注入量", r"Mt yr$^{-1}$")
    d = scatter_panel(ax_d, edge_flows, False, "管段流量", r"Mt yr$^{-1}$")
    e = panel_e(ax_e)

    # 两张地图共用的标度图例，放在地图之间的下方，说明两图可比。
    fig.legend(handles=[
        Line2D([], [], color=PIPE_C, lw=1.65,
               label=f"两种口径共有的管段（线宽 ∝ 流量，满标度 {FLOW_REF:.0f} Mt yr$^{{-1}}$）"),
        Line2D([], [], color=EXCL_C, lw=1.65, label="仅本图这一口径启用的管段"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", mew=0.42,
               markeredgecolor=SINK_EDGE, markersize=3.4, label="陆上封存汇（面积 ∝ 注入量）"),
        Line2D([], [], marker="s", color="none", markerfacecolor="white", mew=0.42,
               markeredgecolor=SINK_EDGE, markersize=3.2, label="海上封存汇"),
        Line2D([], [], marker="o", color="none", markerfacecolor=SRC_C,
               markeredgecolor="none", markersize=1.9, label="接入源汇网络的厂址"),
        Line2D([], [], marker="o", color="none", markerfacecolor=SRC_OFF_C,
               markeredgecolor="none", markersize=1.5, label="未接入的厂址"),
    ], loc="upper center", bbox_to_anchor=(0.50, map_top - map_h - 0.012), fontsize=5.4,
        frameon=False, handlelength=1.6, handletextpad=0.5, ncol=3, columnspacing=1.4)

    panel_label(ax_a, "a", x=0.010, y=1.020)
    panel_label(ax_b, "b", x=0.010, y=1.020)
    panel_label(ax_c, "c", x=-0.290, y=1.140)
    panel_label(ax_d, "d", x=-0.290, y=1.140)
    panel_label(ax_e, "e", x=-0.235, y=1.175)

    fig.text(0.030, 0.012, caption(a, b, c, d, e), fontsize=5.2, color="#555555",
             va="bottom", ha="left", linespacing=1.5)

    save_fig(fig, "ed_fig13_source_sink_matching", subdir="extended")
    report(a, b, c, d, e)


def caption(a, b, c, d, e) -> str:
    sink_row = e[e["level"].str.startswith("汇注入量")].iloc[0]
    pipe_row = e[e["level"].str.startswith("管段流量")].iloc[0]
    bits = (
        f"口径与 ED11、ED12 相同，由 run_single.py 定义；a–e 均采用 {ANALYSIS_YEAR} 年结果。",
        f"a、b 展示该年的源汇匹配格局，",
        f"**两图共用同一条线宽标度与同一条点面积标度**，否则并排比较没有意义：",
        f"不考虑水时 {a['n_edge']} 条活跃管段、{a['n_sink']} 个在用封存汇；",
        f"s = 0.85 时 {b['n_edge']} 条、{b['n_sink']} 个。",
        f"灰点是全部 350 个煤电厂址：深灰接入源汇网络（{a['n_src']} / {b['n_src']} 个）、"
        f"浅灰未接入（{a['n_src_off']} / {b['n_src_off']} 个）—— 管网覆盖的只是机队的一部分。",
        f"**红色是只在本图这一口径下启用的管段** —— a 有 {a['n_excl']} 条（承担 {a['flow_excl']:.0f} Mt），",
        f"b 有 {b['n_excl']} 条（{b['flow_excl']:.0f} Mt）；两图的汇几乎重合而红色管段分布明显不同，",
        f"这就是 d 里 {d['d']['l1']:.0f}% 分歧在地图上的样子。c、d 把同一批对象逐个做 1:1 对比，",
        f"虚线是 y = x（完全不变）。**灰点是零假设** —— 同一口径仅换随机种子的两两配对，",
        f"它给出“什么都不改能漂多远”的参照；没有这层参照，1:1 图上的散布无法判定。",
        f"汇注入量 $R^2$ = {c['d']['r2']:.3f}、$\\Sigma|\\Delta|$ = {c['d']['l1']:.1f}%，",
        f"而零假设本身就有 {c['null_l1'][0]:.1f}–{c['null_l1'][1]:.1f}%；",
        f"管段流量 $R^2$ = {d['d']['r2']:.2f}、$\\Sigma|\\Delta|$ = {d['d']['l1']:.1f}%，",
        f"零假设 {d['null_l1'][0]:.1f}–{d['null_l1'][1]:.1f}%。e 把三个水文臂都放上去：",
        f"汇注入量一侧处理三臂 {min(sink_row['treat']):.1f}–{max(sink_row['treat']):.1f}%，",
        f"与零假设带{'不重叠' if sink_row['clears'] else '重叠'}；",
        f"管段流量一侧 {min(pipe_row['treat']):.1f}–{max(pipe_row['treat']):.1f}%，",
        f"{'三臂全部超出零假设' if pipe_row['clears'] else '与零假设重叠'}。",
        f"c 的两轴下限 {ACTIVE} 就是“启用”的判定阈值，钉在轴边缘的点表示该汇只在一侧被启用。",
        f"结论：CO$_2$ 最终存进哪些地质体基本不受水约束影响，但它走哪条管廊过去是被改写的。",
        f"这与 fig4 在对照–处理对上的发现一致，本图是在“有水约束 vs 完全无水约束”这一对上复现。",
        f"落在零假设内不等于无效应，只是本组求解分辨不出。",
    )
    return cjk_fill(" ".join(bits), width=178)


def report(a, b, c, d, e) -> None:
    print("=" * 80)
    print(f"面板 a / b  {ANALYSIS_YEAR} 年源汇匹配格局")
    print(f"  {'':22}{'活跃管段':>10}{'管道总长(km)':>14}{'输送量(Mt)':>12}"
          f"{'在用汇':>8}{'注入量(Mt)':>12}")
    for lab, s in (("不考虑水 BASE", a), ("考虑水 s=0.85", b)):
        print(f"  {lab:22}{s['n_edge']:>10}{s['km']:>14.0f}{s['flow']:>12.1f}"
              f"{s['n_sink']:>8}{s['inj']:>12.1f}   仅本口径启用 "
              f"{s['n_excl']:>3} 条 / {s['flow_excl']:.1f} Mt"
              f"   厂址 接入 {s['n_src']} / 未接入 {s['n_src_off']}")

    print()
    print("面板 c / d  1:1 对比（不考虑水 vs s=0.85）")
    for name, r in (("汇注入量", c), ("管段流量", d)):
        print(f"  {name:10} R² {r['d']['r2']:.4f}   Σ|Δ|/总量 {r['d']['l1']:5.1f}%   "
              f"开关翻转 {r['d']['flip']:3d}   [零假设 Σ|Δ| {r['null_l1'][0]:.1f}-{r['null_l1'][1]:.1f}%]")

    print()
    print("面板 e  两个层级 × 三个水文臂")
    for _, r in e.iterrows():
        print(f"  {r['level']:10} 对照 {min(r['ctrl']):5.1f}-{max(r['ctrl']):5.1f}%   "
              f"处理 {min(r['treat']):5.1f}-{max(r['treat']):5.1f}%   "
              f"零假设 {r['null_lo']:5.1f}-{r['null_hi']:5.1f}%   "
              f"{'处理组全部超出零假设 -> 可分辨' if r['clears'] else '与零假设重叠 -> 未分辨'}")


if __name__ == "__main__":
    main()
