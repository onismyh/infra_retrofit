# -*- coding: utf-8 -*-
"""ind_fig3 — 水约束对联合减排做了什么：把空冷提前、把工业转氢提前 30 年。

这是本批唯一一对**可以相减**的运行：同一模型、同一联合目标（0.95）、同一 v9 管网，
只差 `apply_basin_cap` 一个旋钮（`_indtree/README.md` §四.3）。

    IND_BASE_t95                  水约束全关
    IND_WA_cwatm_126_dry_oq_t95   + 官方用水总量控制指标（取水口径）的流域上限

四个面板对应四条彼此独立的响应链：

    a  湿冷转空冷改造装机    —— 水约束的主要响应，2030 年 54.7 -> 170.8 GW
    b  水量：煤电耗水、工业取水  —— 响应的结果（两条**不同的表**，见下）
    c  工业减排量            —— 无水时工业到 2060 才动，有水时 2030 就被推去转氢
    d  目标函数差的分解      —— 其中 54% 是 big-M 罚金，不是价格

**面板 a 的口径（曾经画错过，改前请读完）。** `air_cooled_share` 是"仍是湿冷那部分的
转换进度"，不是全厂份额——`data_prep.py` 的原话是"driving it to 1 converts only the part
that is still wet-cooled"，capex 与背压惩罚都按 `1 - already_air_share` 缩放。
直接乘装机会把已经是空冷的 248 GW 再数一遍（实测 `air_cooled_share + already_air_share`
最大 1.989），2030 年会从 170.8 GW 虚报成 316.9 GW。本仓库其余六个脚本
（`build_numbers_ledger`、`ed_plant_data`、`plot_ed_province_transition`、`plot_ed_reversion`、
`plot_fig3_attribution`、`plot_fig3_mechanism`）都乘了 `(1 - already_air_share)`，这里也必须乘。

**面板 b 的两条序列不是同一个表，不可相加。** 煤电的 `plant_detail.water_use_m3` 走的是
**耗水**矩阵（`data_prep._prepare_plants` 的 `water_intensity`，来自
`baseline_water_intensity_m3_per_mwh`）；工业的 `industry_detail.water_m3` 才是进入流域上限的
**取水**量（`industry.add_industry_year` 直接用它构造 `withdrawal_by_hub`）。煤电的取水
是另一套矩阵（`_withdrawal_matrices`），**没有落进任何产物 CSV**，所以这里拿不到。
两条并排画、各自标各自的表，绝不堆叠求和。

**面板 d 存在的理由是纪律而不是结果。** 两次求解的目标函数差 +11.03%，读起来像是
"流域指标的代价"；但有水那一侧在 2030-2050 年付了 7 995 亿元的流域 K 违约罚金
（外加 2060 年 72 亿元的管道容量罚金），罚参数 5×10⁹ 元/单位是我们自己设的惩罚强度。
扣掉它剩 +5.07%，而且那仍是**下界**——模型并没有守住 K 的指标。
两个运行都是 1% gap，所以差值必须按 CLAUDE.md §二.3 给可证区间，
且**单个成本类别的差值只要小于 1% 的目标函数（±1 352 亿元）就落在简并噪声里**（§二.4），
面板 d 用淡色把这一档标出来。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from plot_style import (  # noqa: E402
    MM,
    RESULTS_DIR,
    apply_style,
    cjk_fill,
    panel_label,
    save_fig,
)

RUN_OFF = "IND_BASE_t95"                     # 水约束全关
RUN_ON = "IND_WA_cwatm_126_dry_oq_t95"       # + 流域用水总量指标
LABEL_OFF = "水约束关"
LABEL_ON = "水约束开（官方指标）"
EXPECTED_SINKS = 89                          # v9 管网的汇数，用来把树认出来

# --- 配色（CLAUDE.md §3.3，与 ed_water_* 系列同源）------------------------------------
# 一个情景在四个面板里必须是同一个颜色：对照臂中性灰、处理臂强调红。
# 曾经把面板 c 的处理臂画成紫色（因为那一格全是绿氢），读者会以为那是第三个情景。
C_OFF = "#969696"          # 对照臂：中性灰
C_ON = "#CC3311"           # 处理臂：强调红（本仓库 official-quota 臂固定这个色）
C_COAL_W = "#08519C"       # 煤电耗水：Blues 深端（§3.3 把深端给耗水，正好对上）
C_IND_W = "#6BAED6"        # 工业取水：Blues 浅端（§3.3 把浅端给取水，也对上）
C_UP = "#B2492E"           # 面板 d 增项：与 C_ON 同色相、暗一档，避免和处理臂混
C_DOWN = "#2C7C9B"         # 面板 d 减项
C_NOISE = "#C8CDD2"        # 落在求解容差内的类别
GRID = "#E4E7EA"

# 面板 d：类别中文名。缺一个会在图上画出英文键名，所以这里对完整性有要求——
# `_check_categories` 会在画之前把没登记的类别报出来。
CATEGORY_ZH = {
    "slack_penalty": "松弛罚金（big-M）",
    "incremental_om": "增量运维",
    "industry_cost": "工业改造运维",
    # 2026-09-10 起 industry_cost 只留年度运维，投资单列为 industry_capex
    "industry_capex": "工业改造投资",
    "baseline_net_cost": "基准净成本",
    "stranded_capex": "搁浅资产",
    "coal_savings_credit": "燃料节约抵扣",
    "air_retrofit_capex": "空冷改造投资",
    "energy_penalty_cost": "能耗惩罚",
    "rebuild_capex": "重建投资",
    "carbon_cost": "碳成本",
    "water_cost": "水成本",
    "biomass_cost": "生物质",
    "ccs_retrofit_capex": "CCS 改造投资",
    "ccs_om_cost": "CCS 运维",
    "storage_cost": "封存",
    "pipe_capex": "管网投资",
    "transport_opex": "输送运维",
    "blend_upgrade_capex": "掺烧改造投资",
    "ammonia_cost": "氨",
}
TOP_N = 8                  # 面板 d 单列的类别数，其余并入"其他"


# =========================================================================================
# 读数
# =========================================================================================
def assert_v9_tree(run: str) -> int:
    """确认脚本正跑在 v9 求解树上，而不是仓库根的 v7 结果上。

    两棵树里都有同名目录 `IND_WA_cwatm_126_dry_oq_t95`，`RESULTS_DIR` 只跟着脚本自己的位置
    走（`plot_style.py`：`__file__.parent.parent / "results"`）。在仓库根跑会画出 35 汇的
    v7 结果，而图注里"v9 管网"是写死的字符串——图看上去完全正常。这一条就是防那个。
    """
    sinks = pd.read_csv(RESULTS_DIR / run / "storage_utilization.csv")["storage_hub_id"].nunique()
    if sinks != EXPECTED_SINKS:
        raise RuntimeError(
            f"{RESULTS_DIR} 不是 v9 求解树：{run} 只有 {sinks} 个汇（应为 {EXPECTED_SINKS}）。"
            f"请在 _indtree/ 下运行本脚本。")
    return int(sinks)


def meta(run: str) -> dict:
    return json.loads((RESULTS_DIR / f"{run}.json").read_text(encoding="utf-8"))


def yearly(run: str) -> pd.DataFrame:
    """逐年的空冷转换量、两侧水量与工业减排。全部是决策变量的直接聚合。"""
    plants = pd.read_csv(RESULTS_DIR / run / "plant_detail.csv")
    ind = pd.read_csv(RESULTS_DIR / run / "industry_detail.csv")
    rows = []
    for year in sorted(plants["year"].unique()):
        p = plants[plants["year"] == year]
        i = ind[ind["year"] == year]
        cap_gw = p["capacity_mw"] / 1000.0
        still_wet = (1.0 - p["already_air_share"].fillna(0.0)).clip(0.0, 1.0)
        rows.append({
            "year": int(year),
            # 见模块开头："本次改造转空冷"的容量 = 进度 x 仍是湿冷的份额 x 装机。
            # 不乘 still_wet 会把存量空冷再数一遍。
            "air_gw": float((p["air_cooled_share"] * still_wet * cap_gw).sum()),
            "already_air_gw": float((p["already_air_share"].fillna(0.0) * cap_gw).sum()),
            # 耗水，不是取水（见模块开头）。
            "coal_consumption": float(p["water_use_m3"].sum()) / 1e8,
            # 取水，就是进流域上限的那个量。
            "ind_withdrawal": float(i["water_m3"].sum()) / 1e8,
            "ind_reduction": float(i["reduction_mt"].sum()),
        })
    return pd.DataFrame(rows).set_index("year")


def aligned(off: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:
    """把处理臂按年份对齐到对照臂。两臂年份集若不同，宁可抛错也不要位置错配。"""
    out = on.reindex(off.index)
    if out.isna().any().any():
        missing = list(out.index[out.isna().any(axis=1)])
        raise RuntimeError(f"两臂年份不一致，处理臂缺 {missing}；不能按位置配对。")
    return out


def cost_delta() -> pd.Series:
    """处理臂减对照臂，按成本类别。cost_breakdown 逐年求和后恰好等于目标函数。"""
    off = pd.read_csv(RESULTS_DIR / RUN_OFF / "cost_breakdown.csv").groupby("category")["cost_cny"].sum()
    on = pd.read_csv(RESULTS_DIR / RUN_ON / "cost_breakdown.csv").groupby("category")["cost_cny"].sum()
    index = off.index.union(on.index)
    unmapped = [k for k in index if k not in CATEGORY_ZH]
    if unmapped:
        raise RuntimeError(f"成本类别未登记中文名，会在图上画出英文键：{unmapped}")
    return on.reindex(index).fillna(0.0) - off.reindex(index).fillna(0.0)


def certified_band() -> dict[str, float]:
    """两次求解之差的可证区间（CLAUDE.md §二.3）。两臂都是 1% gap，点值不足以下结论。"""
    c, t = meta(RUN_OFF), meta(RUN_ON)
    inc_c, inc_t = c["global_objective_cny"], t["global_objective_cny"]
    lb_c = c["solver_quality"]["objective_bound_cny"]
    lb_t = t["solver_quality"]["objective_bound_cny"]
    return {
        "point": (inc_t - inc_c) / inc_c,
        "lo": (lb_t - inc_c) / inc_c,
        "hi": (inc_t - lb_c) / lb_c,
        "base": inc_c,
        # 单个类别的差值小于这个就分辨不出来：两臂各自的 incumbent 都可以在 1% 内重排类别。
        "floor_cny": 0.01 * inc_c,
    }


def slack_split() -> dict[str, float]:
    """处理臂的罚金按年份拆开，并核对哪几年是流域 K、哪几年是别的约束。"""
    cost = pd.read_csv(RESULTS_DIR / RUN_ON / "cost_breakdown.csv")
    per_year = cost[cost["category"] == "slack_penalty"].set_index("year")["cost_cny"]
    detail = pd.read_csv(RESULTS_DIR / RUN_ON / "slack_detail.csv")
    basin_years = sorted(detail[detail["constraint_type"] == "water_basin_quota"]["year"].unique())
    other_years = [y for y in per_year.index if y not in basin_years]
    return {
        "total": float(per_year.sum()),
        "basin": float(per_year.reindex(basin_years).sum()),
        "other": float(per_year.reindex(other_years).sum()),
        "basin_years": basin_years,
        "other_years": other_years,
    }


# =========================================================================================
# 面板 a / b / c：逐年响应
# =========================================================================================
def _year_axis(ax, years: list[int]) -> np.ndarray:
    xs = np.arange(len(years), dtype=float)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(y) for y in years], fontsize=5.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=5.8)
    return xs


def panel_a(ax, off: pd.DataFrame, on: pd.DataFrame) -> None:
    years = list(off.index)
    xs = _year_axis(ax, years)
    w = 0.36
    ax.bar(xs - w / 2, off["air_gw"], width=w, color=C_OFF, linewidth=0, zorder=3,
           label=LABEL_OFF)
    ax.bar(xs + w / 2, on["air_gw"], width=w, color=C_ON, linewidth=0, zorder=3,
           label=LABEL_ON)
    # 逐年都标差值：水约束把整条轨迹抬高，只标一年会被读成"一次性提前"。
    # ylim 显式留 22% 顶空——自动范围正好卡在最高柱上，标注会被画到轴外。
    top = float(max(off["air_gw"].max(), on["air_gw"].max()))
    ax.set_ylim(0, top * 1.22)
    for k in range(len(years)):
        gap = float(on["air_gw"].iloc[k] - off["air_gw"].iloc[k])
        if abs(gap) < 1.0:
            continue
        ax.text(xs[k] + w / 2, float(on["air_gw"].iloc[k]) + top * 0.022,
                f"+{gap:.0f}", fontsize=5.4, color=C_ON, ha="center", va="bottom")
    already = float(off["already_air_gw"].iloc[0])
    ax.text(0.985, 0.985, f"不含存量空冷 {already:.0f} GW（两情景相同）",
            transform=ax.transAxes, fontsize=5.2, color="#8A9199", ha="right", va="top")
    ax.set_ylabel("本次改造转空冷（GW）", fontsize=6.2, labelpad=2.0)
    ax.legend(fontsize=5.4, loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False,
              handlelength=0.9, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)


def panel_b(ax, off: pd.DataFrame, on: pd.DataFrame) -> None:
    """两条序列并排，不堆叠：一条是耗水表，一条是取水表，加起来没有意义。"""
    years = list(off.index)
    xs = _year_axis(ax, years)
    w = 0.20
    series = (("coal_consumption", C_COAL_W, "煤电耗水"),
              ("ind_withdrawal", C_IND_W, "工业取水"))
    for s_idx, (col, colour, _label) in enumerate(series):
        base = (s_idx - 0.5) * 2 * w          # 两组序列左右分开
        for shift, frame, hatch in ((-w / 2, off, None), (w / 2, on, "///")):
            ax.bar(xs + base + shift, frame[col], width=w, color=colour, linewidth=0,
                   zorder=3, hatch=hatch, edgecolor="white")
    ax.set_ylabel(r"水量（$10^8$ m$^3$ yr$^{-1}$）", fontsize=6.2, labelpad=2.0)
    ax.set_ylim(0, float(max(off[["coal_consumption", "ind_withdrawal"]].max().max(),
                             on[["coal_consumption", "ind_withdrawal"]].max().max())) * 1.42)
    # 两个维度叠在一根柱上（哪张表 / 哪个情景），颜色管前者、斜纹管后者，图例分两组建。
    # 情景那两块用中性灰而不是与序列同色，否则图例里会出现两个一模一样的深蓝方块。
    handles = [Patch(facecolor=C_COAL_W, edgecolor="none", label="煤电耗水（耗水表）"),
               Patch(facecolor=C_IND_W, edgecolor="none", label="工业取水（取水表）"),
               Patch(facecolor="#9AA0A6", edgecolor="white", label=LABEL_OFF),
               Patch(facecolor="#9AA0A6", edgecolor="white", hatch="///", label=LABEL_ON)]
    ax.legend(handles=handles, fontsize=5.2, loc="upper center", frameon=False, ncol=2,
              handlelength=0.9, handletextpad=0.4, labelspacing=0.3, columnspacing=0.8,
              borderaxespad=0.2)
    # 贴底放会被 2060 年的工业柱压住（那根到 0.52 轴高）。图例在上、柱顶在 0.7 以下，
    # 右侧 0.72 这一条是空的。
    ax.text(0.985, 0.72, "两条表不同口径，不可相加", transform=ax.transAxes,
            fontsize=5.2, color="#8A9199", ha="right", va="bottom")


def panel_c(ax, off: pd.DataFrame, on: pd.DataFrame) -> int | None:
    years = list(off.index)
    xs = _year_axis(ax, years)
    w = 0.36
    ax.bar(xs - w / 2, off["ind_reduction"], width=w, color=C_OFF, linewidth=0, zorder=3,
           label=LABEL_OFF)
    ax.bar(xs + w / 2, on["ind_reduction"], width=w, color=C_ON, linewidth=0, zorder=3,
           label=LABEL_ON)

    # "提前多少年"必须从数据里数出来。写成 years[-1]-years[0] 的话，无论结果是什么
    # 都会印出"30 年"——那不是测量，是把结论贴在图上。
    def first_action(frame: pd.DataFrame) -> int | None:
        acted = frame.index[frame["ind_reduction"] > 0.5]
        return int(acted[0]) if len(acted) else None

    y_off, y_on = first_action(off), first_action(on)
    lead = (y_off - y_on) if (y_off is not None and y_on is not None) else None
    if lead:
        k = years.index(y_on)
        ax.annotate(
            f"工业提前 {lead} 年开始动作\n（{y_on} 年 {on['ind_reduction'].loc[y_on]:.0f} 对 "
            f"{off['ind_reduction'].loc[y_on]:.0f} Mt）",
            xy=(xs[k] + w / 2, on["ind_reduction"].loc[y_on]),
            xytext=(xs[k] + 0.32, float(on["ind_reduction"].max()) * 0.42),
            fontsize=5.4, color=C_ON, ha="left", va="bottom", linespacing=1.35,
            arrowprops=dict(arrowstyle="-", lw=0.4, color=C_ON, shrinkA=1.5, shrinkB=1.0))
    # 末年的量级压平了前三年的柱。数值必须写出来，否则"提前动作"这句话在图上找不到对应物；
    # 线性轴是对的——量级差本身就是结论，补标签就够了。
    for k, year in enumerate(years[:-1]):
        value = float(on["ind_reduction"].loc[year])
        if value <= 0.5:
            continue
        ax.text(xs[k] + w / 2, value + float(on["ind_reduction"].max()) * 0.018,
                f"{value:.0f}", fontsize=5.2, color=C_ON, ha="center", va="bottom")
    ax.set_ylabel(r"工业减排量（Mt CO$_2$ yr$^{-1}$）", fontsize=6.2, labelpad=2.0)
    ax.legend(fontsize=5.4, loc="upper left", frameon=False, handlelength=0.9,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    return lead


# =========================================================================================
# 面板 d：目标函数差的分解
# =========================================================================================
def panel_d(ax, delta: pd.Series, band: dict[str, float], slack: dict[str, float]) -> None:
    order = delta.reindex(delta.abs().sort_values(ascending=False).index)
    head = order.iloc[:TOP_N]
    rest = float(order.iloc[TOP_N:].sum())
    values = list(head.to_numpy()) + ([rest] if abs(rest) > 1e6 else [])
    labels = [CATEGORY_ZH[k] for k in head.index] + (["其他"] if abs(rest) > 1e6 else [])

    floor = float(band["floor_cny"])
    ys = np.arange(len(values))[::-1]
    # 小于求解容差的类别画成灰色：两臂各自的 incumbent 都可以在 1% 内重排类别，
    # 所以这几条的方向和大小都不可读（CLAUDE.md §二.4）。不删掉，是因为删掉会让
    # 各条之和对不上合计。
    colours = [C_NOISE if abs(v) < floor else (C_UP if v > 0 else C_DOWN) for v in values]
    ax.barh(ys, np.asarray(values) / 1e9, height=0.60, color=colours, linewidth=0, zorder=3)
    for y, v in zip(ys, values):
        pad = 22.0 if v > 0 else -22.0
        ax.text(v / 1e9 + pad, y, f"{v / 1e9:+,.0f}", fontsize=5.4, va="center",
                ha="left" if v > 0 else "right",
                color=("#9AA0A6" if abs(v) < floor else "#333333"))

    ax.axvline(0.0, color="#5A5A5A", lw=0.6, zorder=4)
    ax.axvspan(-floor / 1e9, floor / 1e9, color=C_NOISE, alpha=0.30, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xlabel("目标函数变化（十亿元，处理臂减对照臂）", fontsize=6.2, labelpad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=5.8)
    lim = max(abs(min(values)), abs(max(values))) / 1e9 * 1.34
    ax.set_xlim(-lim * 0.55, lim)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    # 合计、可证区间与罚金占比放在图注里，不放在面板上：这三行摊开有 46 mm 宽，
    # 而面板只有 65 mm，无论贴哪个角都会压到某几条柱上（右下压搁浅资产与其他，
    # 左下压碳成本与水成本）。面板上只留一条读图必需的说明——灰带是什么。
    # 左上那块（x < 0，前四条都朝右）是这张图里唯一真正空的地方。
    ax.text(0.015, 0.965, f"灰带 ±{floor / 1e9:,.0f} 十亿元\n= 1% 求解容差\n带内不可读",
            transform=ax.transAxes, fontsize=5.2, ha="left", va="top",
            color="#7A8188", linespacing=1.4)


# =========================================================================================
def main() -> None:
    apply_style()
    for run in (RUN_OFF, RUN_ON):
        assert_v9_tree(run)
    off = yearly(RUN_OFF)
    on = aligned(off, yearly(RUN_ON))
    delta = cost_delta()
    band = certified_band()
    slack = slack_split()

    fig = plt.figure(figsize=(183 * MM, 112 * MM))
    ax_a = fig.add_axes([0.070, 0.610, 0.375, 0.345])
    ax_b = fig.add_axes([0.590, 0.610, 0.375, 0.345])
    ax_c = fig.add_axes([0.070, 0.170, 0.375, 0.335])
    ax_d = fig.add_axes([0.610, 0.170, 0.355, 0.335])

    panel_a(ax_a, off, on)
    panel_b(ax_b, off, on)
    lead = panel_c(ax_c, off, on)
    panel_d(ax_d, delta, band, slack)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        panel_label(ax, letter, x=-0.16, y=1.06)

    basin_years = "-".join(str(y) for y in (slack["basin_years"][0], slack["basin_years"][-1]))
    # 2026-09-12 重建的管网不再出现管道容量 big-M 松弛，这一句要能说没有。
    if slack["other_years"]:
        other_txt = (f" {slack['other'] / 1e9:,.0f} 十亿元是 "
                     f"{slack['other_years'][0]} 年的管道容量罚金；")
    else:
        other_txt = "无其他类别的罚金；"
    fig.text(0.004, 0.004, cjk_fill(
        f"{RUN_OFF} 对 {RUN_ON}：同一模型、同一联合目标（0.95）、同一 v9 管网，"
        f"只差流域取水上限一个旋钮，因此这一对可以相减（v9.2 管网，2026-09-12 重建）。"
        f"(a) 只计本次改造转空冷的容量，按 1 减去存量空冷份额折算，不含"
        f" {off['already_air_gw'].iloc[0]:.0f} GW 的存量空冷（两情景相同）。"
        f"(b) 煤电走耗水表、工业走取水表，煤电的取水量不在任何产物 CSV 里，两条不可相加。"
        f"(d) 的合计 {float(delta.sum()) / 1e9:+,.0f} 十亿元里有 {slack['basin'] / 1e9:,.0f}"
        f" 十亿元是流域 K 在 {basin_years} 年的 big-M 违约罚金，另有"
        f"{other_txt}"
        f"罚参数为设定值而非价格，因此 {band['point']:+.2%}"
        f"（两臂均 1% gap，可证区间 {band['lo']:+.2%} 到 {band['hi']:+.2%}）"
        f"不可作为水约束的成本引用，"
        f"扣掉罚金后的 {(float(delta.sum()) - slack['total']) / band['base']:+.2%} 也只是下界，"
        f"模型并未守住 K 的指标。", 108),
        fontsize=5.0, color="#555555", ha="left", va="bottom", linespacing=1.45)

    save_fig(fig, "ind_fig3_water_coupling", "main")
    report(off, on, delta, band, slack, lead)


def report(off: pd.DataFrame, on: pd.DataFrame, delta: pd.Series, band: dict[str, float],
           slack: dict[str, float], lead: int | None) -> None:
    print(f"  {'年份':<6}{'空冷 关/开 (GW)':<22}{'煤电耗水 关/开':<20}"
          f"{'工业取水 关/开':<20}{'工业减排 关/开 (Mt)'}")
    for year in off.index:
        o, n = off.loc[year], on.loc[year]
        print(f"  {year:<6}{o['air_gw']:7.1f} /{n['air_gw']:7.1f}      "
              f"{o['coal_consumption']:6.2f} /{n['coal_consumption']:6.2f}      "
              f"{o['ind_withdrawal']:6.2f} /{n['ind_withdrawal']:6.2f}      "
              f"{o['ind_reduction']:7.1f} /{n['ind_reduction']:7.1f}")
    total = float(delta.sum())
    print(f"  工业动作提前 {lead} 年" if lead else "  两臂工业动作起始年相同")
    print(f"  目标函数差 {total / 1e9:+,.1f} 十亿元 = {band['point']:+.3%}  "
          f"可证区间 [{band['lo']:+.3%}, {band['hi']:+.3%}]")
    print(f"  其中罚金 {slack['total'] / 1e9:,.1f}（流域 {slack['basin'] / 1e9:,.1f} + "
          f"其他 {slack['other'] / 1e9:,.1f}），扣掉后 "
          f"{(total - slack['total']) / band['base']:+.3%}")
    print(f"  1% 容差地板 ±{band['floor_cny'] / 1e9:,.1f} 十亿元；带内类别："
          + ", ".join(CATEGORY_ZH[k] for k, v in delta.items()
                      if abs(v) < band["floor_cny"] and abs(v) > 1e6))


if __name__ == "__main__":
    main()
