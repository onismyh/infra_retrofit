# -*- coding: utf-8 -*-
"""附录图：考虑水 vs 不考虑水 —— 减排与源汇匹配这一侧。

ED11 讲的是水约束改变了什么（冷却方式与取水）。这张图讲它**没有**改变什么，
而"没有改变"在这里恰恰是主结论：

    满足水资源约束，不以少减排为代价。代价是把机组改成空冷。

十项减排与源汇统计量按 fig4 的口径做两组极差检验（对照三臂的极差 vs 处理三臂的极差，
间隙除以简并度地板），**只有"退役容量"一项越过（2.03 倍，且落在最小的一块地板上，
按 §2.4 属于容差敏感），其余九项全部未越过，最高 0.07 倍**；作为对照，同一框架下
空冷改造 14.9 倍、取水量 15.9 倍。也就是说水约束会逼出一部分提前退役，
但被退役的那部分捕集量由别处补上了 —— 捕集总量、源汇配对、管网规模都读不出差别。

源汇匹配的骨架同样不动：承担 99.7% 注入量的 55 个封存汇
在两种口径下完全相同，边缘上翻转的十几个汇，其数量与仅换随机种子造成的翻转同量级。

口径与 ED11 完全一致（`scripts/run_single.py` 定义的三档）：

    BASE                    water_mode="no_water"                  完全不考虑水
    WA_*_dry                水约束在，existing_withdrawal_share=0   流域可取用量全给电力
    WA_*_dry_wd085          同上，但 s=0.85，电力只留 15%           处理组

为什么用"两组极差的间隙 / 地板"而不是"点差值 / 地板"：处理组内部有三个水文臂，
对照组也有三个，如果两组的取值范围本来就重叠，那么"处理组比对照组高"这句话
连方向都立不住。间隙取 max(min(t)-max(c), min(c)-max(t), 0)，重叠时直接为 0。
这比 ED11 面板 e 的点差值检验更严格，两者都报，不冲突 —— ED11 问"处理组偏离 BASE 多远"，
本图问"处理组与对照组能不能分开"。

数据源：重建输入版本（103 个汇、连通性修复网络），与 v9 其余图同源。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
    cjk_fill,
    DOUBLE_COL,
    RESULTS_DIR,
    PATHWAY_COLORS,
)
from plot_ed_water_on_off import (  # noqa: E402
    BASE, ARMS, SEED_ARM, YEARS, PEAK, hub_frame, floor_of, seed_scenarios,
)

# 面板 b 只画 2040 起：2030 年的排放目标不折减，三档口径的捕集量都恰好是 0，
# 把这一点画进去，y 轴就被 0-1250 的跳变主导，2040-2060 的三条线全挤在顶端一条缝里。
CAP_YEARS = [y for y in YEARS if y >= 2040]

RES = RESULTS_DIR
INPUTS = Path(__file__).resolve().parents[1] / "inputs"

# --- 配色 ---------------------------------------------------------------------------------
C_BASE = "#969696"
C_S0 = "#6BAED6"
C_S085 = "#CC3311"
GRID = "#9AA0A6"
SPINE = "#5A5A5A"
# 汇的一致性：被几个处理臂用到。0/3 -> 3/3，离散四档，显式 BoundaryNorm（§3.3(3)）。
# 用 Blues 族因为封存汇在 §3.3 里就是 Blues（DSA #3182BD / EOR #9ECAE1）。
AGREE_STEPS = ["#D9D9D9", "#C6DBEF", "#6BAED6", "#08519C"]
AGREE_CMAP = ListedColormap(AGREE_STEPS)
AGREE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], AGREE_CMAP.N)
# 路径构成用 §3.3 的技术路径色表
STACK = [("ccs_gw", "CCS 改造", PATHWAY_COLORS["ccs"]),
         ("beccs_gw", "BECCS", PATHWAY_COLORS["beccs"]),
         ("bio_gw", "生物质掺烧", PATHWAY_COLORS["biomass"]),
         ("ret_gw", "退役", PATHWAY_COLORS["retire"])]


# =========================================================================================
# 读数
# =========================================================================================
def _hubs() -> pd.DataFrame:
    h = pd.read_csv(INPUTS / "storage_hubs.csv")
    # 这个文件带 UTF-8 BOM，首列名会变成 "﻿storage_hub_id"，直接取列会 KeyError。
    h.columns = [c.lstrip("﻿") for c in h.columns]
    return h.set_index(h["storage_hub_id"].astype(str))


HUBS = _hubs()


def injection(scen: str, year: int = PEAK) -> pd.Series:
    d = pd.read_csv(RES / scen / "storage_utilization.csv")
    d = d[d["year"] == year]
    return d.groupby(d["storage_hub_id"].astype(str))["storage_use_mtpa"].sum()


def network(scen: str, year: int = PEAK) -> pd.DataFrame:
    n = pd.read_csv(RES / scen / "network_edges.csv")
    return n[(n["year"] == year) & (n["edge_active"].astype(bool))]


def sink_stats(scen: str, year: int = PEAK) -> dict:
    """一次求解的源汇与减排统计量。全是决策变量的直接聚合。"""
    inj = injection(scen, year)
    act = inj[inj > 0.01]
    ne = network(scen, year)
    p = hub_frame(scen, year)
    offshore = act.index.map(lambda k: bool(HUBS["offshore"].get(k, False)))
    return {
        "捕集量（Mt）": float(p["captured_mt"].sum()),
        "在用封存汇数": float(len(act)),
        "海上注入占比（%）": float(act[offshore].sum()) / max(float(act.sum()), 1e-9) * 100.0,
        "流量加权运距（km）": float((ne["length_km"] * ne["edge_flow_mtpa"]).sum()
                             / max(ne["edge_flow_mtpa"].sum(), 1e-9)),
        "管道总长（km）": float(ne["length_km"].sum()),
        "活跃管段数": float(len(ne)),
        "CCS 容量（GW）": float(p["ccs_gw"].sum()),
        "BECCS 容量（GW）": float(p["beccs_gw"].sum()),
        "生物质容量（GW）": float(p["bio_gw"].sum()),
        "退役容量（GW）": float(p["ret_gw"].sum()),
        # 下面两项是对照组，不是减排量。放进同一根轴，是为了让"减排侧全在噪声内"
        # 这句话有个可比的参照，而不是孤零零一列小数。
        "空冷改造容量（GW）": float(p["air_gw"].sum()),
        # 走 mathtext：SimHei 没有 U+00B3，字面 ³ 会渲染成方框（CLAUDE.md §3.1）。
        r"取水量（$10^8$ m$^3$）": float(p["wat_e8"].sum()),
    }


CONTRAST = ("空冷改造容量（GW）", r"取水量（$10^8$ m$^3$）")


def two_group_test(year: int = PEAK) -> pd.DataFrame:
    """fig4 口径：两组极差的间隙 / 简并度地板。重叠时间隙为 0。"""
    ctrl = [sink_stats(a, year) for a in ARMS]
    treat = [sink_stats(f"{a}_wd085", year) for a in ARMS]
    sc = [sink_stats(s, year) for s in seed_scenarios(SEED_ARM)]
    st = [sink_stats(s, year) for s in seed_scenarios(f"{SEED_ARM}_wd085")]
    rows = []
    for k in sink_stats(BASE, year):
        c = [x[k] for x in ctrl]
        t = [x[k] for x in treat]
        f = max(floor_of([x[k] for x in sc]), floor_of([x[k] for x in st]))
        gap = max(min(t) - max(c), min(c) - max(t), 0.0)
        rows.append({"stat": k, "c_lo": min(c), "c_hi": max(c), "t_lo": min(t), "t_hi": max(t),
                     "gap": gap, "floor": f, "ratio": gap / f if f > 1e-9 else 0.0,
                     "contrast": k in CONTRAST})
    return pd.DataFrame(rows)


# =========================================================================================
# 面板 a：源汇匹配的地图
# =========================================================================================
def panel_a(fig, ax, cax, legend_anchor) -> dict:
    b = injection(BASE)
    arms = pd.DataFrame({a: injection(f"{a}_wd085") for a in ARMS})
    idx = b.index.union(arms.index)
    b = b.reindex(idx).fillna(0.0)
    arms = arms.reindex(idx).fillna(0.0)
    n_arm = (arms > 0.01).sum(axis=1)          # 0..3：几个处理臂用了这个汇
    on_base = b > 0.01

    core = on_base & (n_arm == 3)
    core_share = float(b[core].sum()) / float(b[on_base].sum()) * 100.0
    flip = int(((n_arm > 0) & (n_arm < 3)).sum())

    # 种子翻转数：同一口径仅换随机种子，有多少汇会开关。这是 flip 的零假设参照。
    seed_flip = {}
    for stem, lab in ((SEED_ARM, "s0"), (f"{SEED_ARM}_wd085", "s085")):
        S = pd.DataFrame({x: injection(x).reindex(idx).fillna(0.0)
                          for x in seed_scenarios(stem)})
        na = (S > 0.01).sum(axis=1)
        seed_flip[lab] = int(((na > 0) & (na < len(S.columns))).sum())

    def business(a):
        # 汇点：大小 = 不考虑水时的注入量（没用到的给一个最小可见尺寸），
        # 颜色 = 三个处理臂里有几个也在用它。两个通道各管一件事。
        vis = idx[(b > 0.01) | (n_arm > 0)]
        lon = [float(HUBS["longitude"].get(k, np.nan)) for k in vis]
        lat = [float(HUBS["latitude"].get(k, np.nan)) for k in vis]
        x, y = to_map_xy(np.array(lon), np.array(lat))
        size = 3.0 + 30.0 * np.sqrt(np.maximum(b[vis].to_numpy(), 0.0) / max(b.max(), 1e-9))
        a.scatter(x, y, s=size, c=n_arm[vis].to_numpy(), cmap=AGREE_CMAP, norm=AGREE_NORM,
                  edgecolor="white", linewidth=0.30, zorder=5)
        # 只在一侧出现的汇：用形状通道标出来，不再借颜色。
        only_b = idx[on_base & (n_arm == 0)]
        only_t = idx[(~on_base) & (n_arm == 3)]
        for keys, marker in ((only_b, "x"), (only_t, "P")):
            if len(keys) == 0:
                continue
            xx, yy = to_map_xy(np.array([float(HUBS["longitude"][k]) for k in keys]),
                               np.array([float(HUBS["latitude"][k]) for k in keys]))
            a.scatter(xx, yy, s=11, marker=marker, color="#67000D", linewidth=0.7, zorder=6)

    draw_china_basemap(ax, province_lw=0.13, country_lw=0.45, facecolor="#F1F3F5")
    business(ax)
    mainland_extent(ax)
    ax.set_axis_off()
    ax.set_title(f"承担 {core_share:.1f}% 注入量的 {int(core.sum())} 个封存汇，两种口径下完全相同",
                 fontsize=6.6, pad=2.0)
    add_scs_inset(fig, ax, draw=business, axes_rect=[0.795, 0.015, 0.155, 0.215])

    bar = plt.colorbar(plt.cm.ScalarMappable(norm=AGREE_NORM, cmap=AGREE_CMAP), cax=cax,
                       orientation="horizontal", ticks=[0, 1, 2, 3], spacing="uniform")
    cax.set_xticklabels(["0/3", "1/3", "2/3", "3/3"], fontsize=4.8)
    cax.tick_params(length=1.2, pad=1.2)
    bar.outline.set_linewidth(0.3)
    cax.set_title("三个水文臂中有几个用到该封存汇（点面积 ∝ 不考虑水时的注入量）",
                  fontsize=5.4, pad=2.5)

    fig.legend(handles=[
        Line2D([], [], marker="x", color="none", markeredgecolor="#67000D", mew=0.7,
               markersize=3.4, label=f"仅不考虑水时使用（{int((on_base & (n_arm == 0)).sum())} 个，"
                                     f"共 {float(b[on_base & (n_arm == 0)].sum()):.1f} Mt）"),
        Line2D([], [], marker="P", color="none", markeredgecolor="#67000D", mew=0.7,
               markersize=3.4, label=f"仅考虑水时使用（{int(((~on_base) & (n_arm == 3)).sum())} 个，"
                                     f"共 {float(arms[(~on_base) & (n_arm == 3)].median(axis=1).sum()):.1f} Mt）"),
        Patch(facecolor="none", edgecolor="none",
              label=f"三臂之间自相矛盾的 {flip} 个；仅换种子就翻转 {_span(seed_flip)} 个"),
    ], loc="center left", bbox_to_anchor=legend_anchor, fontsize=5.4, frameon=False,
        handlelength=1.5, handletextpad=0.5, labelspacing=0.34)

    return {"core_n": int(core.sum()), "core_share": core_share, "flip": flip,
            "seed_flip": seed_flip, "only_b": int((on_base & (n_arm == 0)).sum()),
            "only_t": int(((~on_base) & (n_arm == 3)).sum()),
            "n_base": int(on_base.sum())}


# =========================================================================================
# 面板 b：捕集量逐年
# =========================================================================================
def panel_b(ax) -> dict:
    base = [hub_frame(BASE, y)["captured_mt"].sum() for y in CAP_YEARS]
    ctrl = np.array([[hub_frame(a, y)["captured_mt"].sum() for y in CAP_YEARS] for a in ARMS])
    treat = np.array([[hub_frame(f"{a}_wd085", y)["captured_mt"].sum() for y in CAP_YEARS]
                      for a in ARMS])
    seeds = np.array([[hub_frame(s, y)["captured_mt"].sum() for y in CAP_YEARS]
                      for s in seed_scenarios(f"{SEED_ARM}_wd085")])

    # 种子包络先画、画在最下：它是"什么都不改、只换随机种子"能漂多远的标尺。
    ax.fill_between(CAP_YEARS, seeds.min(axis=0), seeds.max(axis=0), facecolor="#D9D9D9",
                    alpha=0.75, lw=0, zorder=2, label="仅换随机种子的包络（4 次复现）")
    ax.plot(CAP_YEARS, base, color=C_BASE, lw=1.5, marker="o", ms=2.6, zorder=5, label="不考虑水")
    for arr, colour, name in ((ctrl, C_S0, "考虑水，s = 0"),
                              (treat, C_S085, "考虑水，s = 0.85")):
        ax.fill_between(CAP_YEARS, arr.min(axis=0), arr.max(axis=0), facecolor=colour,
                        alpha=0.20, lw=0, zorder=3)
        ax.plot(CAP_YEARS, np.median(arr, axis=0), color=colour, lw=1.6, marker="o", ms=2.6,
                zorder=5, label=name)

    ax.set_xticks(CAP_YEARS)
    ax.set_ylabel(r"捕集量（Mt CO$_2$ yr$^{-1}$）", fontsize=6.2, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.grid(axis="y", lw=0.25, alpha=0.22, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
        ax.spines[s].set_color(SPINE)
    ax.legend(fontsize=5.0, frameon=False, loc="lower right", handlelength=1.4,
              handletextpad=0.5, labelspacing=0.28, borderpad=0.2)
    i = CAP_YEARS.index(PEAK)
    ax.set_title("三档口径的捕集量落在同一条种子包络里", fontsize=6.6, pad=4.0)
    return {"base": base, "ctrl": ctrl, "treat": treat, "seeds": seeds, "i": i}


# =========================================================================================
# 面板 c：减排路径构成
# =========================================================================================
def panel_c(ax) -> pd.DataFrame:
    groups = [("不考虑水", [BASE]), ("s = 0", ARMS), ("s = 0.85", [f"{a}_wd085" for a in ARMS])]
    rows = []
    for lab, scens in groups:
        vals = {}
        for key, _, _ in STACK:
            vals[key] = float(np.median([hub_frame(s, PEAK)[key].sum() for s in scens]))
        vals["label"] = lab
        rows.append(vals)
    tab = pd.DataFrame(rows).set_index("label")

    y = np.arange(len(tab))[::-1]
    left = np.zeros(len(tab))
    for key, name, colour in STACK:
        ax.barh(y, tab[key], left=left, height=0.55, color=colour, edgecolor="white",
                lw=0.4, zorder=3, label=name)
        left = left + tab[key].to_numpy()

    ax.set_yticks(y)
    ax.set_yticklabels(tab.index, fontsize=5.4)
    ax.set_ylim(-0.6, len(tab) - 0.4)
    ax.set_xlabel(f"{PEAK} 年各路径的煤电容量（GW）", fontsize=6.0, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.25, alpha=0.20, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(SPINE)
    ax.set_title("路径构成也没有换：三档口径的堆叠几乎重合", fontsize=6.6, pad=4.0)
    return tab


# =========================================================================================
# 面板 d：两组极差检验
# =========================================================================================
def panel_d(ax) -> pd.DataFrame:
    tab = two_group_test().sort_values("ratio").reset_index(drop=True)
    y = np.arange(len(tab))
    # 对照组（空冷、取水）用强调红，减排/源汇用中性灰蓝 —— 判定靠位置（越没越过 1.0），
    # 颜色只用来分"这是被检验的量"还是"这是拿来比的参照"。
    colours = [C_S085 if c else "#7FA8C9" for c in tab["contrast"]]
    stub = 0.012 * max(tab["ratio"].max(), 1.6)
    ax.barh(y, [max(v, stub) for v in tab["ratio"]], height=0.58, color=colours,
            edgecolor="white", lw=0.3, zorder=3)
    for yi, v in zip(y, tab["ratio"]):
        ax.text(v + 0.012 * tab["ratio"].max() + stub, yi,
                f"{v:.2f}×" if v < 1.0 else f"{v:.1f}×",
                va="center", ha="left", fontsize=4.8, color="#4A4A4A")

    ax.axvline(1.0, color=SPINE, lw=0.7, ls=(0, (3, 2)), zorder=4)
    ax.text(1.0, len(tab) - 0.30, "简并度地板", fontsize=5.0, color=SPINE,
            ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(tab["stat"], fontsize=5.2)
    ax.set_ylim(-0.7, len(tab) - 0.20)
    ax.set_xlabel("两组极差的间隙 / 该统计量自身的简并度地板（倍）", fontsize=6.0, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.25, alpha=0.20, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(SPINE)
    ax.legend(handles=[Patch(facecolor="#7FA8C9", label="减排与源汇统计量"),
                       Patch(facecolor=C_S085, label="对照：冷却与取水")],
              loc="lower right", fontsize=4.8, frameon=False, handlelength=1.1,
              handletextpad=0.5, labelspacing=0.26, borderpad=0.2)
    n = int((tab["ratio"] > 1.0).sum())
    n_ab = int(((~tab["contrast"]) & (tab["ratio"] > 1.0)).sum())
    ax.set_title(f"{int((~tab['contrast']).sum())} 项减排与源汇统计量里 {n_ab} 项越过地板",
                 fontsize=6.6, pad=4.0)
    return tab


# =========================================================================================
def _span(d: dict) -> str:
    """两个种子族的翻转数相同时写成一个数，不写成 "8-8" 这种假区间。"""
    lo, hi = min(d.values()), max(d.values())
    return f"{lo}" if lo == hi else f"{lo}–{hi}"


LEGEND_ANCHOR = (0.055, 0.180)


def main() -> None:
    apply_style()
    fig_w, fig_h = DOUBLE_COL[0], 6.40
    fig = plt.figure(figsize=(fig_w, fig_h))

    LEFT, RIGHT = 0.055, 0.982
    map_w = 0.475
    map_h = (map_w * fig_w / 1.177) / fig_h
    map_top = 0.930
    ax_a = fig.add_axes([LEFT - 0.012, map_top - map_h, map_w, map_h])
    cax = fig.add_axes([LEFT + 0.025, map_top - map_h - 0.052, map_w * 0.52, 0.010])

    r_left, r_w = 0.595, RIGHT - 0.595
    ax_b = fig.add_axes([r_left, 0.660, r_w, 0.225])
    ax_c = fig.add_axes([r_left, 0.475, r_w, 0.105])
    ax_d = fig.add_axes([r_left, 0.180, r_w, 0.215])

    sinks = panel_a(fig, ax_a, cax, LEGEND_ANCHOR)
    cap = panel_b(ax_b)
    stack = panel_c(ax_c)
    # c 的路径图例：放在 c 的横轴标题与 d 的标题之间那条窄带上，用图幅坐标定位。
    fig.legend(handles=[Patch(facecolor=col, label=name) for _, name, col in STACK],
               loc="upper center", bbox_to_anchor=(0.79, 0.437), fontsize=4.8,
               frameon=False, handlelength=1.1, handletextpad=0.5, ncol=4,
               columnspacing=1.0, borderpad=0.0)
    test = panel_d(ax_d)

    panel_label(ax_a, "a", x=0.010, y=1.010)
    panel_label(ax_b, "b", x=-0.145, y=1.150)
    panel_label(ax_c, "c", x=-0.145, y=1.300)
    panel_label(ax_d, "d", x=-0.145, y=1.150)

    fig.text(0.055, 0.012, caption(sinks, cap, stack, test), fontsize=5.2, color="#555555",
             va="bottom", ha="left", linespacing=1.5)

    save_fig(fig, "ed_fig12_water_abatement", subdir="extended")
    report(sinks, cap, stack, test)


def caption(sinks, cap, stack, test) -> str:
    i = cap["i"]
    b = cap["base"][i]
    t = float(np.median(cap["treat"], axis=0)[i])
    sd = cap["seeds"][:, i]
    ab = test[~test["contrast"]]
    worst = ab.loc[ab["ratio"].idxmax()]
    hit = ab[ab["ratio"] > 1.0].sort_values("ratio", ascending=False)
    rest = ab[ab["ratio"] <= 1.0]
    if hit.empty:
        clear_txt = (f"没有一项越过地板，最高的{worst['stat']}也只有 "
                     f"{worst['ratio']:.2f} 倍。")
    else:
        # 越过地板的那几项要点名，并且要说清它们的地板有多小 —— 落在最小地板上的
        # "可分辨"按 §2.4 属于容差敏感，不能与几十倍的那两项等同看待。
        named = "、".join(f"{r['stat']}（{r['ratio']:.2f} 倍）" for _, r in hit.iterrows())
        clear_txt = (f"只有{named}越过地板，且它落在这批统计量里最小的一块地板上"
                     f"（{float(hit.iloc[0]['floor']):.1f}），按 §2.4 属容差敏感；"
                     f"其余 {len(rest)} 项全部未越过，最高 "
                     f"{float(rest['ratio'].max()):.2f} 倍。")
    ctr = test[test["contrast"]].sort_values("ratio", ascending=False)
    bits = (
        f"口径与 ED11 相同，由 run_single.py 定义：不考虑水（BASE）；考虑水但流域可取用量",
        f"全部给电力（s = 0）；考虑水且电力只留 15% 存量取水权（s = 0.85）。",
        f"a 的点是封存汇，面积正比于不考虑水时的注入量，颜色是三个水文臂",
        f"（CWatM SSP1-2.6 / CWatM SSP3-7.0 / WaterGAP SSP1-2.6）中有几个也用到它。",
        f"{sinks['core_n']} 个汇被两种口径共同使用并承担 {sinks['core_share']:.1f}% 的注入量；",
        f"只在一侧出现的汇共 {sinks['only_b'] + sinks['only_t']} 个、合计不到 1 Mt；",
        f"三臂彼此就不一致的有 {sinks['flip']} 个，而**什么都不改、只换随机种子**",
        f"就能翻转 {_span(sinks['seed_flip'])} 个 —— 两者同量级，",
        f"所以边缘上的汇进汇出读不出水的信号。b 中 {PEAK} 年捕集量 {b:.0f}（不考虑水）对",
        f"{t:.0f} Mt（s = 0.85），而仅换种子的复现族当年就横跨 {sd.min():.0f}–{sd.max():.0f} Mt。",
        f"d 用的检验比 ED11 面板 e 更严：横轴是**对照三臂与处理三臂两组取值范围的间隙**",
        f"除以该统计量的简并度地板（1.96·√2·极差/d2(k)，极差取自仅换种子的 4 次复现），",
        f"两组范围重叠时间隙记为 0。{len(ab)} 项减排与源汇统计量里，{clear_txt}",
        f"作为参照，同一框架下",
        f"{ctr.iloc[0]['stat']}是 {ctr.iloc[0]['ratio']:.0f} 倍、{ctr.iloc[1]['stat']}是 {ctr.iloc[1]['ratio']:.0f} 倍。",
        f"结论是水约束的代价落在冷却方式上，不落在减排量上；未越过地板不等于无效应，",
        f"只是本组求解分辨不出。",
    )
    return cjk_fill(" ".join(bits), width=178)


def report(sinks, cap, stack, test) -> None:
    print("=" * 84)
    print("面板 a  源汇匹配")
    print(f"  不考虑水在用封存汇 {sinks['n_base']} 个")
    print(f"  两种口径共同使用的核心 {sinks['core_n']} 个，承担 {sinks['core_share']:.2f}% 的注入量")
    print(f"  仅不考虑水使用 {sinks['only_b']} 个；仅考虑水使用 {sinks['only_t']} 个")
    print(f"  三臂之间自相矛盾 {sinks['flip']} 个；仅换随机种子翻转 "
          f"{sinks['seed_flip']['s0']}（s=0）/ {sinks['seed_flip']['s085']}（s=0.85）个")

    print()
    print("面板 b  捕集量（Mt/yr），三臂中位数")
    print(f"  {'年份':<8}{'不考虑水':>12}{'s=0':>12}{'s=0.85':>12}{'种子包络':>20}")
    for j, y in enumerate(CAP_YEARS):
        sd = cap["seeds"][:, j]
        print(f"  {y:<8}{cap['base'][j]:>12.1f}{float(np.median(cap['ctrl'], axis=0)[j]):>12.1f}"
              f"{float(np.median(cap['treat'], axis=0)[j]):>12.1f}"
              f"{f'{sd.min():.1f}-{sd.max():.1f}':>20}")

    print()
    print(f"面板 c  {PEAK} 年路径构成（GW）")
    print(stack.round(1).to_string())

    print()
    print("面板 d  两组极差检验（对照三臂 vs 处理三臂）")
    print(f"  {'统计量':<20}{'对照极差':>21}{'处理极差':>21}{'间隙':>9}{'地板':>10}{'倍数':>8}  判定")
    for _, r in test.sort_values("ratio", ascending=False).iterrows():
        tag = "对照量" if r["contrast"] else ("可分辨" if r["ratio"] > 1.0 else "未分辨")
        print(f"  {r['stat']:<20}[{r['c_lo']:9.1f},{r['c_hi']:9.1f}]"
              f"[{r['t_lo']:9.1f},{r['t_hi']:9.1f}]{r['gap']:>9.1f}{r['floor']:>10.2f}"
              f"{r['ratio']:>8.2f}  {tag}")


if __name__ == "__main__":
    main()
