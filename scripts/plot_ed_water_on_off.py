# -*- coding: utf-8 -*-
"""附录图：考虑水 vs 不考虑水 —— 影响落在哪些区域，以及是否鲁棒。

这张图回答一个问题：把水资源约束放进模型，和完全不放，差别到底在哪。

口径不是我定的，是 `scripts/run_single.py` 里写死的三档：

    BASE                    water_mode="no_water"                  完全不考虑水
    WA_*_dry                水约束在，existing_withdrawal_share=0   流域可取用量全给电力
    WA_*_dry_wd085          同上，但 s=0.85，电力只留 15%           处理组

第二档的存在是这张图的关键。run_single.py 自己的注释就写着"at 0.0 ... `WA_*_dry`
lands within the MIP gap of BASE" —— 也就是说，**"考虑水有多大影响"这个问题没有单一答案，
它完全取决于你假设电力能占用多少存量取水权。** 这正是本图的主线：

    a  成本：s=0 与不考虑水的可证区间跨零（分辨不出）；s=0.85 为 [+1.00, +5.44]%
    b  空冷改造逐年：效应是前置的，2040 年从 46 GW 推到 356 GW（与 fig3/fig5 同口径）
    c  空间：309 GW 的增量里 97.5% 落在海河、黄河、西北内陆河、淮河四个北方流域
    d  取水净减 36%（70.6 -> 45.0 亿 m3），减量全部落在五个北方流域，南方几乎不动
    e  鲁棒性：上述结论逐项对各自的简并度地板检验；捕集量与各路径容量全在噪声内

所有差值都配了简并度地板（CLAUDE.md §2.4）：floor = 1.96*sqrt(2)*range/d2(k)，
range 来自同一模型仅换随机种子的复现族。没有越过地板的量，本图一律标为"未分辨"，
而不是"无效应"——这两件事不一样。

成本的区间不用种子族，用可证边界（§2.3）：lo = (LB_t - INC_c)/INC_c，
hi = (INC_t - LB_c)/LB_c。因为 BASE 的 gap（1.70%）与处理组（2.55%）不同，
点估计相减没有意义，只有边界相减是可证的。

数据源：重建输入版本（103 个汇、连通性修复网络），与 v9 其余图同源。
"""
from __future__ import annotations

import json
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
    load_basins,
    assign_basin,
    cjk_fill,
    BASIN_NAMES_ZH,
    DOUBLE_COL,
    RESULTS_DIR,
)

# --- 口径 --------------------------------------------------------------------------------
BASE = "BASE"
# 三个水文臂。同一个处理组开关（s=0.85）配三套水文强迫，用来看结论是否依赖水文模型的选择。
ARMS = ["WA_cwatm_126_dry", "WA_cwatm_370_dry", "WA_wgap_126_dry"]
SEED_ARM = "WA_cwatm_126_dry"          # 只有这一臂跑了种子复现族
SEEDS = (2, 3, 4)
YEARS = [2030, 2040, 2050, 2060]
PEAK = 2040                            # 效应峰值年，横截面面板都用它
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534}

# --- 配色（CLAUDE.md §3.3）----------------------------------------------------------------
# 三档口径各一色。处理组用强调红 #CC3311 —— §3.3 把这个色分配给"空冷改造"，
# 而本图的主角正好就是空冷改造，所以是同一件事，不是借色。
C_BASE = "#969696"      # 不考虑水：中性灰
C_S0 = "#6BAED6"        # s=0：取水蓝（浅）
C_S085 = "#CC3311"      # s=0.85：强调红
C_GAIN = "#CC3311"      # d 面板：取水增加
C_LOSS = "#08519C"      # d 面板：取水减少（Blues 深端 = 耗水色）
GRID = "#9AA0A6"
SPINE = "#5A5A5A"

# c 面板流域填充：Δ空冷改造是单向正量，用顺序色带 + 显式 BoundaryNorm（§3.3(3)）。
# 与 #CC3311 同族的 Oranges/Reds 中高段，浅端不从近白起（底图是 #F1F3F5）。
DELTA_BOUNDS = [0.0, 5.0, 20.0, 50.0, 100.0, 150.0, 200.0]
DELTA_STEPS = ["#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"]
DELTA_CMAP = ListedColormap(DELTA_STEPS)
DELTA_NORM = BoundaryNorm(DELTA_BOUNDS, DELTA_CMAP.N, clip=True)

RES = RESULTS_DIR


# =========================================================================================
# 读数
# =========================================================================================
def _plants(scen: str) -> pd.DataFrame:
    return pd.read_csv(RES / scen / "plant_detail.csv")


def hub_frame(scen: str, year: int) -> pd.DataFrame:
    """机组级表，附上流域码与三个派生容量列。

    流域用 plot_style.assign_basin（最近多边形，覆盖全部 350 个 hub），
    而不是"取水量最大的水节点所属流域"——后者只覆盖 338 个（12 个 hub 当年不取水），
    两者在共同覆盖的 338 个上一致率 93.5%，差异都在流域交界带。
    """
    p = _plants(scen)
    p = p[p["year"] == year].copy()
    p["basin"] = assign_basin(p)
    cap = p["capacity_mw"] / 1000.0
    p["cap_gw"] = cap
    # air_cooled_share 是"相对该 hub 尚未改造的湿冷容量的改造进度"，不是总空冷份额
    # （plot_fig3_mechanism.py:248 的口径）。直接乘容量会把 248 GW 已空冷的底数混进来，
    # 2030 年 BASE 会从 45 GW 虚报成 153 GW。ED12 / ED13 都从这里导入，改这一处即可。
    p["air_gw"] = p["capacity_mw"] / 1000.0 * (1.0 - p["already_air_share"]) * p["air_cooled_share"]
    p["ret_gw"] = p["share_retire"] * cap
    p["ccs_gw"] = p["share_ccs"] * cap
    p["beccs_gw"] = p["share_beccs"] * cap
    p["bio_gw"] = p["share_biomass"] * cap
    p["wat_e8"] = p["water_use_m3"] / 1e8
    return p


def national(scen: str, year: int) -> dict:
    p = hub_frame(scen, year)
    return {
        "空冷改造容量": float(p["air_gw"].sum()),
        "取水量": float(p["wat_e8"].sum()),
        "退役容量": float(p["ret_gw"].sum()),
        "捕集量": float(p["captured_mt"].sum()),
        "CCS 容量": float(p["ccs_gw"].sum()),
        "BECCS 容量": float(p["beccs_gw"].sum()),
        "生物质容量": float(p["bio_gw"].sum()),
    }


def basin_series(scen: str, year: int, col: str) -> pd.Series:
    return hub_frame(scen, year).groupby("basin")[col].sum()


def floor_of(vals) -> float:
    """简并度地板：1.96*sqrt(2)*range/d2(k)。k 是复现族成员数。"""
    vals = list(vals)
    k = len(vals)
    if k < 2:
        return float("nan")
    return 1.96 * np.sqrt(2.0) * (max(vals) - min(vals)) / D2[k]


def seed_scenarios(stem: str) -> list[str]:
    return [stem] + [f"{stem}_seed{i}" for i in SEEDS]


def certified(treat: str, ctrl: str) -> tuple[float, float]:
    """可证区间（%）。点估计相减在 gap 不同的两次求解之间没有意义。"""
    def q(n):
        d = json.loads((RES / f"{n}.json").read_text(encoding="utf-8"))["solver_quality"]
        return d["objective_cny"], d["objective_bound_cny"]
    it, lt = q(treat)
    ic, lc = q(ctrl)
    return (lt - ic) / ic * 100.0, (it - lc) / lc * 100.0


# =========================================================================================
# 面板 a：成本，两个可证区间
# =========================================================================================
def panel_a(ax) -> list[tuple]:
    rows = [
        ("考虑水，流域可取用量\n全部给电力（s = 0）", SEED_ARM, C_S0),
        ("考虑水，电力只留 15%\n存量取水权（s = 0.85）", f"{SEED_ARM}_wd085", C_S085),
    ]
    out = []
    for i, (label, scen, colour) in enumerate(rows):
        lo, hi = certified(scen, BASE)
        y = len(rows) - 1 - i
        crosses = lo <= 0.0 <= hi
        ax.plot([lo, hi], [y, y], color=colour, lw=4.2, solid_capstyle="butt",
                alpha=0.35 if crosses else 1.0, zorder=3)
        # 两端的端帽：区间是可证边界，端点本身有意义，不能画成渐隐。
        for v in (lo, hi):
            ax.plot([v, v], [y - 0.16, y + 0.16], color=colour, lw=1.0, zorder=4)
        ax.text(hi + 0.22, y, f"[{lo:+.2f}, {hi:+.2f}]%", va="center", ha="left",
                fontsize=5.4, color=colour if not crosses else "#7A7A7A")
        out.append((label, lo, hi, crosses))

    ax.axvline(0.0, color=SPINE, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(0.0, len(rows) - 0.42, "不考虑水（BASE）", fontsize=5.2, color=SPINE,
            ha="center", va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=5.4)
    ax.set_ylim(-0.55, len(rows) - 0.05)
    ax.set_xlim(-2.6, 8.4)
    ax.set_xlabel("相对不考虑水的总成本变化（%，可证区间）", fontsize=6.0, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.25, alpha=0.20, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(SPINE)
    ax.set_title("“考虑水”值多少钱，取决于给电力留多少取水权", fontsize=6.6, pad=4.0)
    return out


# =========================================================================================
# 面板 b：空冷改造逐年
# =========================================================================================
def panel_b(ax) -> dict:
    base = [national(BASE, y)["空冷改造容量"] for y in YEARS]
    ctrl = np.array([[national(a, y)["空冷改造容量"] for y in YEARS] for a in ARMS])
    treat = np.array([[national(f"{a}_wd085", y)["空冷改造容量"] for y in YEARS] for a in ARMS])

    ax.plot(YEARS, base, color=C_BASE, lw=1.5, marker="o", ms=2.6,
            label="不考虑水", zorder=5)
    for arr, colour, name in ((ctrl, C_S0, "考虑水，s = 0"),
                              (treat, C_S085, "考虑水，s = 0.85")):
        # 带 = 三个水文臂的极差。结论不该依赖于选了哪个水文模型，这条带就是在给这个说法作证。
        ax.fill_between(YEARS, arr.min(axis=0), arr.max(axis=0), facecolor=colour,
                        alpha=0.20, lw=0, zorder=3)
        ax.plot(YEARS, np.median(arr, axis=0), color=colour, lw=1.6, marker="o", ms=2.6,
                label=name, zorder=5)

    i = YEARS.index(PEAK)
    ratio = np.median(treat, axis=0)[i] / base[i]
    # 三条曲线把画面切成的空隙都很窄，斜引线放哪都会压到某条线上。改用竖直差值标记：
    # 在 2040 处从"不考虑水"拉到"s=0.85"，文字放右下那片真正空的三角区。
    # 差值本来就是竖直方向的量，这样标注也更贴切。
    peak_t = float(np.median(treat, axis=0)[i])
    ax.annotate("", xy=(PEAK, peak_t), xytext=(PEAK, base[i]),
                arrowprops=dict(arrowstyle="<->", color=C_S085, lw=0.7,
                                shrinkA=1.5, shrinkB=1.5), zorder=6)
    # 2030-2040 之间红线（~315-356）与蓝线（~95）之间是整片空白，注释放那里；
    # 右上角是图例，右下的红线下降段正好穿过 2050 的点，两处都放不下。
    ax.text(2031.0, 0.58 * peak_t, f"{PEAK} 年 {ratio:.1f} 倍\n{base[i]:.0f} → {peak_t:.0f} GW",
            fontsize=5.4, color=C_S085, ha="left", va="center", zorder=6)

    ax.set_xticks(YEARS)
    ax.set_ylabel("累计空冷改造容量（GW）", fontsize=6.2, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.grid(axis="y", lw=0.25, alpha=0.22, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
        ax.spines[s].set_color(SPINE)
    ax.legend(fontsize=5.2, frameon=False, loc="upper right", handlelength=1.4,
              handletextpad=0.5, labelspacing=0.3, borderpad=0.2)
    ax.set_title("效应是前置的：2040 年前后最大，之后随机组退役收敛", fontsize=6.6, pad=4.0)
    return {"base": base, "ctrl": ctrl, "treat": treat}


# =========================================================================================
# 面板 c：地图
# =========================================================================================
def panel_c(fig, ax, cax) -> pd.DataFrame:
    b_air = basin_series(BASE, PEAK, "air_gw")
    t_air = pd.concat([basin_series(f"{a}_wd085", PEAK, "air_gw") for a in ARMS],
                      axis=1).median(axis=1)
    seeds_c = [basin_series(s, PEAK, "air_gw") for s in seed_scenarios(SEED_ARM)]
    seeds_t = [basin_series(s, PEAK, "air_gw") for s in seed_scenarios(f"{SEED_ARM}_wd085")]

    codes = sorted(set(b_air.index) | set(t_air.index))
    rows = []
    for k in codes:
        d = float(t_air.get(k, 0.0)) - float(b_air.get(k, 0.0))
        f = max(floor_of([float(s.get(k, 0.0)) for s in seeds_c]),
                floor_of([float(s.get(k, 0.0)) for s in seeds_t]))
        rows.append({"code": k, "base": float(b_air.get(k, 0.0)),
                     "treat": float(t_air.get(k, 0.0)), "delta": d, "floor": f,
                     "ratio": (abs(d) / f if f > 1e-9 else 0.0)})
    tab = pd.DataFrame(rows).set_index("code")

    basins = load_basins()
    basins = basins.merge(tab, left_on="code", right_index=True, how="left")

    def business(a):
        # 流域填充 = Δ空冷改造。越过地板的流域加一圈深色描边 —— 判定用形状通道，
        # 不用颜色通道，因为颜色已经被 Δ 的大小占住了。
        basins.plot(ax=a, column="delta", cmap=DELTA_CMAP, norm=DELTA_NORM,
                    edgecolor="none", zorder=2)
        sep = basins[basins["ratio"] > 1.0]
        if not sep.empty:
            sep.boundary.plot(ax=a, edgecolor="#67000D", linewidth=0.9, zorder=3)
        basins[basins["ratio"] <= 1.0].boundary.plot(
            ax=a, edgecolor="#B9BFC5", linewidth=0.35, zorder=2.5)

    draw_china_basemap(ax, province_lw=0.0, country_lw=0.45, facecolor="#F1F3F5")
    business(ax)

    # 机组级：在不考虑水时完全没有空冷、在处理组下转了的 hub。地图上的点回答
    # "是哪些厂址在动"，流域填充回答"总量在哪个流域"。
    hb = hub_frame(BASE, PEAK).set_index("plant_id")
    ht = hub_frame(f"{SEED_ARM}_wd085", PEAK).set_index("plant_id")
    both = hb.index.intersection(ht.index)
    newly = ht.loc[both][(hb.loc[both, "air_cooled_share"] < 0.01)
                         & (ht.loc[both, "air_cooled_share"] > 0.50)]
    if not newly.empty:
        hx, hy = to_map_xy(newly["centroid_longitude"].to_numpy(),
                           newly["centroid_latitude"].to_numpy())
        ax.scatter(hx, hy, s=2.0 + 16.0 * np.sqrt(newly["cap_gw"] / newly["cap_gw"].max()),
                   facecolor="none", edgecolor="#3D0A08", linewidth=0.4, zorder=6)

    mainland_extent(ax)
    ax.set_axis_off()
    n_hit = int((tab["ratio"] > 1.0).sum())
    hit_share = tab.loc[tab["ratio"] > 1.0, "delta"].sum() / tab["delta"].sum() * 100.0
    ax.set_title(f"增量的 {hit_share:.1f}% 集中在 {n_hit} 个北方流域", fontsize=6.6, pad=2.0)
    add_scs_inset(fig, ax, draw=business, axes_rect=[0.795, 0.015, 0.155, 0.215])

    # spacing="uniform" 而不是 §3.3(3) 举例用的 "proportional"：那个例子的分档是等距的，
    # 两者没差别。这里分档不等距（0-5 只占量程的 2.5%），proportional 会把头两个刻度
    # 压进 2.5% 的宽度里，标签必然重叠。分档色标本来就是等宽色块，uniform 才是它的画法。
    bar = plt.colorbar(plt.cm.ScalarMappable(norm=DELTA_NORM, cmap=DELTA_CMAP), cax=cax,
                       orientation="horizontal", ticks=DELTA_BOUNDS,
                       spacing="uniform")
    cax.set_xticklabels([f"{int(v)}" for v in DELTA_BOUNDS[:-1]] + [f">{int(DELTA_BOUNDS[-2])}"],
                        fontsize=4.8)
    cax.tick_params(length=1.2, pad=1.2)
    bar.outline.set_linewidth(0.3)
    cax.set_title(f"{PEAK} 年空冷改造容量的增量（GW）：考虑水与不考虑水之差",
                  fontsize=5.4, pad=2.5)

    n_sep = int((tab["ratio"] > 1.0).sum())
    share = tab.loc[tab["ratio"] > 1.0, "delta"].sum() / tab["delta"].sum() * 100.0
    fig.legend(handles=[
        Patch(facecolor="none", edgecolor="#67000D", lw=0.9,
              label=f"越过简并度地板的流域（{n_sep} 个，占增量 {share:.1f}%）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none", mew=0.4,
               markeredgecolor="#3D0A08", markersize=3.2,
               label=f"由未改造转为空冷的厂址（{len(newly)} 个，点面积 ∝ 容量）"),
    ], loc="center left", bbox_to_anchor=CBAR_LEGEND_ANCHOR, fontsize=5.4, frameon=False,
        handlelength=1.5, handletextpad=0.5, labelspacing=0.34)
    return tab


# =========================================================================================
# 面板 d：取水的跨流域再配置
# =========================================================================================
def panel_d(ax) -> pd.DataFrame:
    b_w = basin_series(BASE, PEAK, "wat_e8")
    t_w = pd.concat([basin_series(f"{a}_wd085", PEAK, "wat_e8") for a in ARMS],
                    axis=1).median(axis=1)
    seeds_c = [basin_series(s, PEAK, "wat_e8") for s in seed_scenarios(SEED_ARM)]
    seeds_t = [basin_series(s, PEAK, "wat_e8") for s in seed_scenarios(f"{SEED_ARM}_wd085")]

    rows = []
    for k in sorted(set(b_w.index) | set(t_w.index)):
        d = float(t_w.get(k, 0.0)) - float(b_w.get(k, 0.0))
        f = max(floor_of([float(s.get(k, 0.0)) for s in seeds_c]),
                floor_of([float(s.get(k, 0.0)) for s in seeds_t]))
        rows.append({"code": k, "delta": d, "floor": f,
                     "ratio": (abs(d) / f if f > 1e-9 else 0.0)})
    tab = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)

    y = np.arange(len(tab))
    colours = [C_GAIN if v > 0 else C_LOSS for v in tab["delta"]]
    alphas = [1.0 if r > 1.0 else 0.30 for r in tab["ratio"]]
    for yi, v, cl, al in zip(y, tab["delta"], colours, alphas):
        ax.barh(yi, v, height=0.60, color=cl, alpha=al, edgecolor="white", lw=0.3, zorder=3)
    for yi, v, r in zip(y, tab["delta"], tab["ratio"]):
        if r <= 1.0:
            continue
        ax.text(v + (0.30 if v > 0 else -0.30), yi, f"{r:.0f}×", va="center",
                ha="left" if v > 0 else "right", fontsize=4.8, color="#4A4A4A")

    b_tot, t_tot = float(b_w.sum()), float(t_w.sum())
    net = (t_tot - b_tot) / b_tot * 100.0
    ax.axvline(0.0, color=SPINE, lw=0.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([BASIN_NAMES_ZH[c] for c in tab["code"]], fontsize=5.2)
    ax.set_ylim(-0.7, len(tab) - 0.3)
    # 收到实际数据范围：原来给到 +6.5，而最大正值是 +0.10，右半边整片是空的。
    lo, hi = float(tab["delta"].min()), float(tab["delta"].max())
    ax.set_xlim(lo * 1.30, max(hi * 1.30, abs(lo) * 0.16))
    ax.set_xlabel(r"取水量变化（$10^8$ m$^3$ yr$^{-1}$）：考虑水与不考虑水之差",
                  fontsize=6.0, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.25, alpha=0.20, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(SPINE)
    ax.legend(handles=[
        Patch(facecolor=C_LOSS, label="取水减少"),
        Patch(facecolor=C_GAIN, label="取水增加"),
        Patch(facecolor="#BDBDBD", alpha=0.45, label="未越过地板"),
    ], loc="upper left", fontsize=4.8, frameon=False, handlelength=1.1,
        handletextpad=0.5, labelspacing=0.26, borderpad=0.2, ncol=1)
    # 标题必须照数据写。曾写成"是搬了家：北方减、长江增"，那是探查阶段单臂 +
    # 取水节点归属的产物；三臂中位数 + 几何归属下长江 +0.01（0.05x 地板）、
    # 珠江 +0.10（0.73x），都在噪声里。真实结果是净减少，不是再配置。
    ax.set_title(f"取水净减 {abs(net):.0f}%，减量全部落在北方", fontsize=6.6, pad=4.0)
    return tab


# =========================================================================================
# 面板 e：鲁棒性汇总
# =========================================================================================
def panel_e(ax) -> pd.DataFrame:
    keys = list(national(BASE, PEAK).keys())
    b = national(BASE, PEAK)
    c = {k: np.median([national(a, PEAK)[k] for a in ARMS]) for k in keys}
    t = {k: np.median([national(f"{a}_wd085", PEAK)[k] for a in ARMS]) for k in keys}
    sc = [national(s, PEAK) for s in seed_scenarios(SEED_ARM)]
    st = [national(s, PEAK) for s in seed_scenarios(f"{SEED_ARM}_wd085")]

    rows = []
    for k in keys:
        f = max(floor_of([x[k] for x in sc]), floor_of([x[k] for x in st]))
        r0 = abs(c[k] - b[k]) / f if f > 1e-9 else 0.0
        r85 = abs(t[k] - b[k]) / f if f > 1e-9 else 0.0
        rows.append({"stat": k, "floor": f, "r_s0": r0, "r_s085": r85,
                     "d_s0": c[k] - b[k], "d_s085": t[k] - b[k]})
    tab = pd.DataFrame(rows).sort_values("r_s085", ascending=True).reset_index(drop=True)

    y = np.arange(len(tab))
    h = 0.34
    ax.barh(y + h / 2 + 0.02, tab["r_s085"], height=h, color=C_S085,
            edgecolor="white", lw=0.3, zorder=3, label="s = 0.85 相对不考虑水")
    ax.barh(y - h / 2 - 0.02, tab["r_s0"], height=h, color=C_S0,
            edgecolor="white", lw=0.3, zorder=3, label="s = 0 相对不考虑水")
    for yi, v in zip(y, tab["r_s085"]):
        # 只标越过地板的。未越过的柱子长度 0.2-0.8，标签起点会正好压在 1.0 的地板虚线上；
        # 而它们的具体倍数不承载结论 —— "未分辨"就是全部信息，图注已说明它不等于无效应。
        if v <= 1.0:
            continue
        ax.text(v + 0.30, yi + h / 2 + 0.02, f"{v:.1f}×", va="center", ha="left",
                fontsize=4.8, color="#4A4A4A")

    ax.axvline(1.0, color=SPINE, lw=0.7, ls=(0, (3, 2)), zorder=4)
    ax.text(1.0, len(tab) - 0.35, "简并度地板", fontsize=5.0, color=SPINE,
            ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(tab["stat"], fontsize=5.2)
    ax.set_ylim(-0.7, len(tab) - 0.25)
    ax.set_xlabel("差值 / 该统计量自身的简并度地板（倍）", fontsize=6.0, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.25, alpha=0.20, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(SPINE)
    ax.legend(fontsize=4.8, frameon=False, loc="lower right", handlelength=1.1,
              handletextpad=0.5, labelspacing=0.26, borderpad=0.2)
    n = int((tab["r_s085"] > 1.0).sum())
    ax.set_title(f"{len(tab)} 项里 {n} 项越过地板；未越过 ≠ 无效应", fontsize=6.6, pad=4.0)
    return tab


# =========================================================================================
CBAR_LEGEND_ANCHOR = (0.055, 0.185)


def main() -> None:
    apply_style()
    fig_w, fig_h = DOUBLE_COL[0], 6.85
    fig = plt.figure(figsize=(fig_w, fig_h))

    LEFT, RIGHT = 0.055, 0.982
    # 上排：a 窄、b 宽。a 只有两根横条，给宽了就是一片空白。
    ax_a = fig.add_axes([0.155, 0.790, 0.235, 0.145])
    ax_b = fig.add_axes([0.565, 0.790, 0.400, 0.145])

    # 地图轴的高度按数据的真实宽高比给（mainland_extent 在 EPSG:2380 下实测 1.177），
    # 否则 aspect="equal" 会在轴内上下留出空白带。
    map_w = 0.470
    map_h = (map_w * fig_w / 1.177) / fig_h
    map_top = 0.720
    ax_c = fig.add_axes([LEFT - 0.010, map_top - map_h, map_w, map_h])
    cax = fig.add_axes([LEFT + 0.030, map_top - map_h - 0.048, map_w * 0.52, 0.010])

    r_left, r_w = 0.590, RIGHT - 0.590
    # e 的下沿要给 8 行图注让出净空：图注 8 x 5.2 pt x 1.5 行距 = 0.127 画布高，
    # 顶到 y=0.139，而 BOT=0.145 时 e 的横轴标题落在 y≈0.115，正好相撞。
    ax_d = fig.add_axes([r_left, 0.455, r_w, 0.205])
    ax_e = fig.add_axes([r_left, 0.190, r_w, 0.175])

    cost = panel_a(ax_a)
    curves = panel_b(ax_b)
    air = panel_c(fig, ax_c, cax)
    water = panel_d(ax_d)
    robust = panel_e(ax_e)

    panel_label(ax_a, "a", x=-0.62, y=1.30)
    panel_label(ax_b, "b", x=-0.150, y=1.30)
    panel_label(ax_c, "c", x=0.010, y=1.010)
    panel_label(ax_d, "d", x=-0.185, y=1.16)
    panel_label(ax_e, "e", x=-0.185, y=1.18)

    fig.text(0.055, 0.012, caption(cost, curves, air, water, robust), fontsize=5.2,
             color="#555555", va="bottom", ha="left", linespacing=1.5)

    save_fig(fig, "ed_fig11_water_on_off", subdir="extended")
    report(cost, curves, air, water, robust)


def caption(cost, curves, air, water, robust) -> str:
    i = YEARS.index(PEAK)
    base_peak = curves["base"][i]
    treat_peak = float(np.median(curves["treat"], axis=0)[i])
    sep = air[air["ratio"] > 1.0]
    share = sep["delta"].sum() / air["delta"].sum() * 100.0
    names = "、".join(BASIN_NAMES_ZH[c] for c in sep.sort_values("delta", ascending=False).index)
    weak = sep["ratio"].idxmin()
    weak_name, weak_ratio = BASIN_NAMES_ZH[weak], float(sep.loc[weak, "ratio"])
    # 证据强度这句话要照数据写：最弱的那个流域低于 2 倍才叫"勉强"，否则直接报最低倍数。
    if weak_ratio < 2.0:
        strength_txt = (f"这 {len(sep)} 个流域的证据强度并不齐平 —— {weak_name}只有 "
                        f"{weak_ratio:.1f} 倍地板，是唯一勉强越线的；")
    else:
        strength_txt = f"{len(sep)} 个流域里最弱的{weak_name}也有 {weak_ratio:.1f} 倍地板；"
    wat_b = national(BASE, PEAK)["取水量"]
    wat_t = float(np.median([national(f"{a}_wd085", PEAK)["取水量"] for a in ARMS]))
    n_rob = int((robust["r_s085"] > 1.0).sum())
    n_rob0 = int((robust["r_s0"] > 1.0).sum())
    bits = (
        f"三档口径由 run_single.py 定义：不考虑水（BASE，water_mode=no_water）；",
        f"考虑水但把流域可取用量全部给电力（s = 0）；考虑水且电力只留 15% 存量取水权（s = 0.85）。",
        f"面板 a 的区间是可证边界 —— BASE 的 MIP gap 是 1.70%，处理组是 2.55%，",
        f"两次求解的点估计相减没有意义，只有 lo=(LB_t-INC_c)/INC_c、hi=(INC_t-LB_c)/LB_c 可证。",
        f"s = 0 那一档跨零，也就是说把整个流域配额都交给电力时，加不加水约束分辨不出；",
        f"这不是水不重要，是这个配额假设让约束不咬。b–e 的“考虑水”取三个水文臂",
        f"（CWatM SSP1-2.6 / CWatM SSP3-7.0 / WaterGAP SSP1-2.6）的中位数，带为三臂极差。",
        f"b 中 {PEAK} 年空冷改造从 {base_peak:.0f} 升到 {treat_peak:.0f} GW；",
        f"c、d、e 都取 {PEAK} 年这个峰值年的横截面。c 的填充是流域尺度的增量，",
        f"深色描边表示越过简并度地板（{len(sep)} 个流域：{names}，占全国增量 {share:.1f}%）；"
        f"{strength_txt}",
        f"圈点是在不考虑水时完全没有空冷、在 s = 0.85 下转过一半以上的厂址。",
        f"d 显示水是被净减掉的，不是搬到南方：全国 {wat_b:.1f} 降到 {wat_t:.1f} 亿 m3，"
        f"减量 {wat_b - wat_t:.1f} 全部落在北方五流域，长江与珠江的变化都在地板之内。"
        f"e 把 {len(robust)} 项全国统计量放在同一个无量纲轴上：",
        f"横轴是差值除以该统计量自身的简并度地板（1.96·√2·极差/d2(k)，极差取自仅换随机种子的",
        f"4 次复现）。s = 0.85 一侧 {n_rob} 项越过，s = 0 一侧 {n_rob0} 项越过。",
        f"未越过地板不等于无效应，只是本组求解分辨不出。",
    )
    return cjk_fill(" ".join(bits), width=178)


def report(cost, curves, air, water, robust) -> None:
    """把进入图里的每个数字都打出来，便于核对图注。"""
    print("=" * 78)
    print("面板 a  成本的可证区间（%）")
    for label, lo, hi, crosses in cost:
        print(f"  {label.replace(chr(10), ' '):<38} [{lo:+.2f}, {hi:+.2f}]"
              f"  {'跨零 -> 分辨不出' if crosses else '不跨零 -> 可分辨'}")

    print()
    print("面板 b  空冷改造容量（GW），三臂中位数")
    print(f"  {'年份':<8}{'不考虑水':>12}{'s=0':>12}{'s=0.85':>12}{'倍数':>10}")
    for j, y in enumerate(YEARS):
        b = curves["base"][j]
        c = float(np.median(curves["ctrl"], axis=0)[j])
        t = float(np.median(curves["treat"], axis=0)[j])
        print(f"  {y:<8}{b:>12.1f}{c:>12.1f}{t:>12.1f}{(t / b if b > 0 else float('nan')):>10.1f}")

    print()
    print(f"面板 c  {PEAK} 年空冷改造增量，按流域（GW）")
    print(f"  {'流域':<14}{'不考虑水':>10}{'s=0.85':>10}{'Δ':>10}{'地板':>9}{'倍数':>8}  判定")
    for k, r in air.sort_values("delta", ascending=False).iterrows():
        print(f"  {BASIN_NAMES_ZH[k]:<14}{r['base']:>10.1f}{r['treat']:>10.1f}"
              f"{r['delta']:>10.1f}{r['floor']:>9.1f}{r['ratio']:>8.2f}"
              f"  {'可分辨' if r['ratio'] > 1.0 else '未分辨'}")
    sep = air[air["ratio"] > 1.0]
    print(f"  全国增量 {air['delta'].sum():.1f} GW，其中越过地板的 {len(sep)} 个流域占"
          f" {sep['delta'].sum() / air['delta'].sum() * 100:.1f}%")

    print()
    print(f"面板 d  {PEAK} 年取水量变化，按流域（10^8 m3/yr）")
    for _, r in water.iterrows():
        print(f"  {BASIN_NAMES_ZH[r['code']]:<14}Δ={r['delta']:>8.2f}"
              f"  地板={r['floor']:>7.2f}  倍数={r['ratio']:>7.2f}"
              f"  {'可分辨' if r['ratio'] > 1.0 else '未分辨'}")

    print()
    print(f"面板 e  {PEAK} 年全国统计量对地板的倍数")
    print(f"  {'统计量':<14}{'不考虑水':>11}{'Δ(s=0)':>11}{'Δ(s=0.85)':>12}"
          f"{'地板':>10}{'倍数(0)':>10}{'倍数(0.85)':>12}")
    b = national(BASE, PEAK)
    for _, r in robust.sort_values("r_s085", ascending=False).iterrows():
        print(f"  {r['stat']:<14}{b[r['stat']]:>11.1f}{r['d_s0']:>11.1f}"
              f"{r['d_s085']:>12.1f}{r['floor']:>10.2f}{r['r_s0']:>10.2f}{r['r_s085']:>12.2f}")


if __name__ == "__main__":
    main()
