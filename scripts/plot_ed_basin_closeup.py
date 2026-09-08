# -*- coding: utf-8 -*-
"""附录图：两条水规则，各自最紧的那个流域 —— 而它们不是同一个流域。

WHY TWO BASINS. v9.1 splits the water rule into two constraints on two different water
bases, and the split immediately produces a result that a single-basin closeup cannot show:
**the tightest basin is not the same basin under the two rules.**

    左列  海河，生态流量档     节点耗水 <= 枯水期径流 x 0.20      （耗水口径，一条消耗性规则）
    右列  西北内陆河，总量指标档 流域取水 <= 用水总量控制指标 - 非电既有取水（取水口径，一条配额规则）

Measured on the same fleet, the same year and the same 20-member slice, full-capture demand
reaches 103% of the Hai's environmental-flow allowance and 1078% of the Northwest's residual
allocation -- while the Hai sits at 52% of ITS allocation and the Northwest at 19% of ITS
environmental-flow allowance. A depletion rule and an allocation rule are different
institutions, metered on different quantities, and a basin can be comfortable under one while
breaching the other. Through v9 the two were multiplied into a single factor and this figure
could not exist.

THE TWO BREACHES ALSO HAVE DIFFERENT SHAPES. Where the environmental-flow rule is exceeded it
is capture that does it: as built no basin is near the line. Where the allocation rule is
exceeded the breach PRE-DATES capture -- the standing Northwest fleet already claims 533% of
that basin's residual allocation, and capture doubles an overshoot that is there without it.

WHY THESE TWO, AND WHY NOT THE HAI. The basin is chosen by RESPONSE -- where the fleet
actually moves when that rung is switched on -- because two of the three panels in each column
show a response. At 2030 the dry-cooling conversion each rung buys is:

    rung 1  BASE -> envonly    黄河 +22.6 GW   西北内陆河 +25.5 GW   其余流域 0
    rung 2  envonly -> oq      西北内陆河 +49.2 GW   黄河 -6.2   海河 -0.7   其余 0

so the Yellow is the basin whose behaviour rung 1 drives, and the Northwest is the basin whose
behaviour rung 2 drives. (The Northwest responds to both; the Yellow responds to rung 1 and
partly hands the conversion back under rung 2, which is itself worth seeing.)

THE HAI IS THE TIGHTEST BASIN ON RUNG 1 AND IS STILL NOT DRAWN, which is a result rather than
an omission. Its full-capture consumption is 103% of its environmental-flow allowance -- the
only basin over 100% on that rung -- but "full capture" is a COUNTERFACTUAL: at 2030 the
emission target does not bind, the model captures nothing anywhere, and the Hai's solved
consumption is 8.4 against an allowance of 15.8. A basin can be the tightest on paper and
never move. Fig 2(b) reports the tightness ranking; this figure reports the response.

Note also that the ledger here is a BASIN AGGREGATE while the environmental-flow rule is a
NODE constraint. The Yellow's aggregate never approaches its allowance (42% even at full
capture) and its fleet still converts 22.6 GW, because individual nodes inside it saturate
while the basin total looks comfortable. Reading a node constraint off a basin aggregate is
the error Fig 3(c) exists to expose, and panel (b) here is not evidence about slackness.

THE NORTHWEST'S DENOMINATOR IS NOT A CLEAN NUMBER, and panel (e) says so on its face. Its
2025 metered withdrawal (729.0) already exceeds its own 2030 用水总量控制指标 (641.2), so the
raw residual is NEGATIVE (-85.5). `write_basin_caps` scales every user's claim pro rata
(x0.8795) so the basin fits its cap, which leaves 2.1 -- and dividing by a near-zero residual
is what produces four digits. The breach is a real statement about that basin's red line; the
exact multiple is not a robust number.

面板：
    a / d  站点地图：哪些厂址在约束下改造了冷却系统
    b / e  水量台账：本档配额、现状需求、全量捕集需求、求解后实际用量
    c / f  厂址明细：让出水量的是哪些厂址（前后哑铃图）

两列的台账**口径不同、量纲相同但不可相减**：左列全部是耗水，右列全部是取水。

厂址归属流域走 `ed_plant_data.plant_basin()`（最近可达供水节点），而图 2、图 3 走
`plot_style.assign_basin()`（最近流域多边形）。两套规则都是有意的、各自文档化的选择——
节点规则对应"这台机组从哪个流域取水"，多边形规则覆盖全部 350 个 hub——在共同覆盖的厂址上
一致率 93.5%，所以同一流域同一步的合计量在本图与图 2/3 之间会相差数个百分点
（黄河 rung 1：本图 20 GW，plot_style.hub_frame 口径 22.6 GW）。**两处的数不要互相引用。**

配额切片写死为 (YEAR=2030, 全部 20 个成员)，与图 2(b) 相同。图 1(c) 用的是它自己的
2060/SSP3-7.0 英雄年，海河配额在那里是 16.84 而不是 15.76 —— 同一个流域的两个不同切片，
都对，但不能互相引用。见 plot_fig1_water_footprint.basin_drying 的说明。
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

import ed_plant_data as EP
from plot_style import (  # noqa: E402
    MM,
    save_fig,
    PROV_ZH,
    BASIN_NAMES_ZH,
    RESULTS_DIR,
    ARMS,
    treat_of,
    BASE_SCENARIO,
)

ENVONLY = ARMS[0]                 # WA_cwatm_126_dry_oq_envonly
QUOTA = treat_of(ENVONLY)         # WA_cwatm_126_dry_oq
YEAR = 2030
EXTRACTABLE = 0.20                # 生态流量：Richter et al. 2012, doi:10.1002/rra.1511

ACT_C = "#CC3311"       # 在约束下改造：强调红（CLAUDE.md §3.3 把该色分给空冷改造）
QUIET = "#9AA5B1"       # 未改造 / 改造前
SUPPLY_C = "#4477AA"    # 配额与供水节点
DEMAND_C = "#882255"    # 全量捕集需求

# 每列一个流域、一条规则、**一个属于这条规则自己的对比**。
#
# EACH COLUMN DRAWS THE STEP ITS OWN RUNG INTRODUCES. Drawing envonly->oq in both columns was
# wrong and looked it: the Hai converts NOTHING across that step (its allocation rung sits at
# 52% and never binds), so the left column showed an empty map and a flat dumbbell chart --
# not because the environmental-flow rule does nothing in the Hai, but because that step is
# not the environmental-flow rule's step. The Hai's rung is introduced by BASE -> envonly.
#
# basis 决定台账用耗水还是取水——两者绝不混用。
COLUMNS = [
    {"code": "D", "rung": "env", "basis": "耗水",
     "control": BASE_SCENARIO, "treat": ENVONLY,
     "ctrl_label": "不考虑水（对照）", "treat_label": "仅生态流量",
     "rule": "生态流量档：节点耗水 ≤ 枯水期径流 × 0.20"},
    {"code": "K", "rung": "quota", "basis": "取水",
     "control": ENVONLY, "treat": QUOTA,
     "ctrl_label": "仅生态流量（对照）", "treat_label": "＋用水总量指标",
     "rule": "用水总量指标档：流域取水 ≤ 用水总量控制指标 减去非电既有取水"},
]


# ── 读数 ─────────────────────────────────────────────────────────────────────
def basin_frame(spec: dict) -> pd.DataFrame:
    """该流域内的厂址，附本列自己那一步的控制组/处理组配对列。"""
    pb = EP.plant_basin()
    members = set(pb.loc[pb["basin_code"] == spec["code"], "plant_id"])
    pr = EP.pair(spec["control"], spec["treat"], YEAR)
    return pr[pr["plant_id"].isin(members)].copy()


def env_allowance(code: str) -> float:
    """该流域枯水期生态流量配额，1e8 m3/yr。

    走图 1 的 basin_drying，并且显式指定切片 (YEAR, 全部 20 个成员)。本图与图 2(b) 用的是
    同一个切片，图 1(c) 用的是它自己的 2060/SSP3-7.0 英雄年——那是两个不同的数（海河
    15.76 与 16.84），谁也没错，错的是让两张图在不点明切片的情况下各报一个。
    """
    import plot_fig1_water_footprint as f1

    return float(f1.basin_drying(YEAR, None).loc[code, "dry_season_1e8"]) * EXTRACTABLE


def quota_allowance(code: str) -> tuple[float, float]:
    """该流域残余配额（1e8 m3/yr），返回 (按比例折减后, 折减前原值)。

    两个都返回是有意的：西北内陆河折减前是负数，只画折减后的那个会让读者以为
    这是一个正常的小配额，而不是"这个流域实测取水已经超了自己的指标"。
    """
    caps = pd.read_csv(EP.INPUTS / "water_basin_caps.csv")
    row = caps[(caps["basin_code"] == code) & (caps["planning_year"] == YEAR)].iloc[0]
    return float(row["residual_1e8_m3"]), float(row["residual_uncapped_1e8_m3"])


def counterfactual_demand(code: str) -> tuple[float, float, float, float]:
    """现状与全量捕集的需求，耗水与取水各一对，1e8 m3/yr。

    DELEGATES TO FIG 1 rather than re-deriving: 图 1、图 2 与本图引用的是同一套强度与
    同一次直流冷却标定，任何一处自己重推都会在三张图之间产生互相矛盾的数字。
    """
    from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
    import plot_fig1_water_footprint as f1
    from plot_style import assign_basin

    assumptions = OptimizationAssumptions()
    scenario = OptimizationScenario(experiment_id="ed_closeup", description="ed_closeup")
    sites = f1.site_table(assumptions, scenario)
    sites = sites[assign_basin(sites).values == code]
    return (float(sites["base_demand_1e8"].sum()), float(sites["ccs_demand_1e8"].sum()),
            float(sites["base_withdrawal_1e8"].sum()), float(sites["ccs_withdrawal_1e8"].sum()))


def realised_withdrawal(scenario: str, code: str) -> float:
    """求解器自己记录的该流域取水量，1e8 m3/yr。

    直接读 resource_use.csv 的 water_basin_quota 行，而不是拿机组强度重算：那一行正是
    约束左端本身，重算出来的量与被约束的量不是同一个数就说不清"贴着上限"这件事。
    """
    ru = pd.read_csv(RESULTS_DIR / scenario / "resource_use.csv")
    row = ru[(ru["resource_type"] == "water_basin_quota") & (ru["region"] == code)
             & (ru["year"] == YEAR)]
    return float(row["used"].iloc[0]) / 1e8 if len(row) else float("nan")


# ── 面板 ─────────────────────────────────────────────────────────────────────
def panel_map(ax, h: pd.DataFrame, code: str) -> None:
    from plot_extended import _load_provinces
    import geopandas as gpd
    from shapely.geometry import Point

    provinces = _load_provinces()
    nodes = pd.read_csv(EP.INPUTS / "water_nodes.csv")
    nodes = nodes[nodes["basin_code"] == code]

    provinces.plot(ax=ax, color="#F7F8F9", lw=0, zorder=0)
    provinces.boundary.plot(ax=ax, lw=0.35, color="#C6CDD4", zorder=1)

    nd = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in
                                    zip(nodes["longitude"], nodes["latitude"])],
                          crs="EPSG:4326").to_crs(provinces.crs)
    ax.scatter(nd.geometry.x, nd.geometry.y, s=3.0, c=SUPPLY_C, lw=0, alpha=0.30, zorder=2)

    pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in
                                     zip(h["centroid_longitude"], h["centroid_latitude"])],
                           crs="EPSG:4326").to_crs(provinces.crs)
    conv = (h["d_converted_gw"] > 0.01).to_numpy()
    size = 6.0 + 52.0 * (h["capacity_mw"] / h["capacity_mw"].max()).to_numpy()
    ax.scatter(pts.geometry.x[~conv], pts.geometry.y[~conv], s=size[~conv], c="white",
               lw=0.7, edgecolor=QUIET, alpha=0.95, zorder=4)
    ax.scatter(pts.geometry.x[conv], pts.geometry.y[conv], s=size[conv], c=ACT_C,
               lw=0.5, edgecolor="white", alpha=0.92, zorder=5)

    pad = 1.4e5
    ax.set_xlim(nd.geometry.x.min() - pad, nd.geometry.x.max() + pad)
    ax.set_ylim(nd.geometry.y.min() - pad, nd.geometry.y.max() + pad)
    ax.set_axis_off()

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bar = 100_000 if (x1 - x0) < 1.6e6 else 300_000
    ax.plot([x0 + 0.06 * (x1 - x0), x0 + 0.06 * (x1 - x0) + bar],
            [y0 + 0.06 * (y1 - y0)] * 2, color="#333333", lw=1.1, zorder=20,
            solid_capstyle="butt")
    ax.text(x0 + 0.06 * (x1 - x0) + bar / 2, y0 + 0.06 * (y1 - y0) - 0.018 * (y1 - y0),
            f"{bar // 1000} km", ha="center", va="top", fontsize=5.2, color="#333333",
            zorder=20)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.4, mfc=ACT_C, mec="white", mew=0.4,
               label=f"在约束下改造（{int(conv.sum())} 个厂址）"),
        Line2D([], [], marker="o", ls="none", ms=4.4, mfc="white", mec=QUIET, mew=0.8,
               label=f"未改造（{int((~conv).sum())} 个）"),
        Line2D([], [], marker="o", ls="none", ms=2.2, mfc=SUPPLY_C, mec="none", alpha=0.5,
               label="供水节点")],
        fontsize=5.1, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02),
        handletextpad=0.6, labelspacing=0.34)

    # 两个数不同，短写法会把它们混成一个：响应厂址持有的装机是一回事，
    # 它们新增的冷凝设备改造量是另一回事。
    ax.set_title(f"{BASIN_NAMES_ZH.get(code, code)}：{len(h)} 个厂址，"
                 f"{h['unit_count'].sum():,.0f} 台机组，{h['capacity_mw'].sum() / 1e3:.0f} GW\n"
                 f"{int(conv.sum())} 个厂址（{h.loc[conv, 'capacity_mw'].sum() / 1e3:.0f} GW）"
                 f"参与，合计改造 {h['d_converted_gw'].sum():.0f} GW 冷凝设备",
                 fontsize=6.4, linespacing=1.3, x=0.52)


def panel_ledger(ax, spec: dict, h: pd.DataFrame) -> dict:
    """本档配额与针对它的需求，适应前后各一根。"""
    code, rung = spec["code"], spec["rung"]
    built_c, cap_c, built_w, cap_w = counterfactual_demand(code)

    if rung == "env":
        allowance = env_allowance(code)
        as_built, full_capture = built_c, cap_c
        realised = float(h["water_use_m3_t"].sum()) / 1e8
        before = float(h["water_use_m3_c"].sum()) / 1e8
        allow_label = "配额：枯水期径流 × 0.20\n（生态流量上限）"
        raw = None
    else:
        allowance, raw = quota_allowance(code)
        as_built, full_capture = built_w, cap_w
        realised = realised_withdrawal(spec["treat"], code)
        # NO "BEFORE" ON THIS RUNG, AND NOT BECAUSE IT WAS FORGOTTEN. The control run has
        # `apply_basin_cap=False`, so it carries no water_basin_quota row at all and the
        # solver never reports its basin withdrawal. Reconstructing it from plant_detail
        # would mean rebuilding the pathway-weighted, air-share-adjusted, once-through
        # calibrated intensity outside the solver -- the exact reconstruction that was
        # already wrong by 24% once in this study. The as-built counterfactual is the honest
        # comparator here, and it is already drawn.
        before = float("nan")
        allow_label = "配额：用水总量控制指标\n减去非电既有取水"

    rows = [(allow_label, allowance, SUPPLY_C),
            (f"需求：全部机组捕集（{spec['basis']}）", full_capture, DEMAND_C),
            (f"需求：机组按现状配置（{spec['basis']}）", as_built, QUIET),
            (f"求解后实际{spec['basis']}", realised, ACT_C)]
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, colour) in zip(y, rows):
        ax.barh(yi, v, height=0.58, color=colour, edgecolor="white", lw=0.4,
                alpha=0.55 if colour == SUPPLY_C else 0.92, zorder=3)
        ax.annotate(f"{v:.2f}", xy=(v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=5.6, fontweight="bold", color=colour)
    ax.axvline(allowance, color=SUPPLY_C, lw=1.0, ls=(0, (3, 2)), zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.4, linespacing=1.2)
    ax.set_xscale("log")
    lo = max(min(v for _, v, _ in rows if v > 0) * 0.45, 1e-3)
    ax.set_xlim(lo, max(v for _, v, _ in rows) * 3.2)
    ax.set_xlabel(f"{YEAR} 年{spec['basis']}量（$10^8$ m$^3$ yr$^{{-1}}$，对数轴）", fontsize=6.2)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    if raw is not None and raw < 0:
        ax.set_title(f"全量捕集需求为配额的 {full_capture / allowance:.0f} 倍；"
                     f"该配额本身已被按比例折减\n"
                     f"（折减前残余为 {raw:.1f}，即 2025 年实测取水已超其 {YEAR} 年指标）",
                     fontsize=6.2, linespacing=1.25)
    elif np.isfinite(before):
        ax.set_title(f"全量捕集需求为配额的 {full_capture / allowance:.2f} 倍，"
                     f"现状为 {as_built / allowance:.2f} 倍；\n"
                     f"改造把实际{spec['basis']}从 {before:.2f} 收到 {realised:.2f}",
                     fontsize=6.2, linespacing=1.25)
    else:
        ax.set_title(f"全量捕集需求为配额的 {full_capture / allowance:.2f} 倍，"
                     f"现状为 {as_built / allowance:.2f} 倍；\n"
                     f"改造把实际{spec['basis']}收到 {realised:.2f}（贴着上限）",
                     fontsize=6.2, linespacing=1.25)
    return {"code": code, "rung": rung, "allowance": allowance, "raw_residual": raw,
            "ctrl_label": spec["ctrl_label"], "treat_label": spec["treat_label"],
            "as_built": as_built, "full_capture": full_capture,
            "before": before, "realised": realised}


def panel_sites(ax, h: pd.DataFrame, spec: dict) -> None:
    """让出水量的是哪些厂址，前后哑铃图。"""
    d = h.copy()
    d["c"] = d["water_use_m3_c"] / 1e8
    d["t"] = d["water_use_m3_t"] / 1e8
    d = d.sort_values("c", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples()):
        ax.plot([r.t, r.c], [yi, yi], color=QUIET, lw=1.0, zorder=2, solid_capstyle="round")
        ax.plot([r.c], [yi], marker="o", ms=3.4, color=QUIET, mec="white", mew=0.5, zorder=4)
        ax.plot([r.t], [yi], marker="o", ms=3.4, color=ACT_C, mec="white", mew=0.5, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{PROV_ZH.get(r.province_name, r.province_name)}  {r.capacity_mw / 1e3:.1f} GW"
         for r in d.itertuples()], fontsize=5.2)
    ax.set_xlabel("厂址枯水期耗水量（$10^8$ m$^3$ yr$^{-1}$）", fontsize=6.2)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.4, mfc=QUIET, mec="white",
                              label=spec["ctrl_label"]),
                       Line2D([], [], marker="o", ls="none", ms=3.4, mfc=ACT_C, mec="white",
                              label=spec["treat_label"])],
              fontsize=5.1, frameon=False, loc="lower right", handletextpad=0.5,
              labelspacing=0.3)
    total_c = float(h["water_use_m3_c"].sum())
    cut = 100 * (1 - float(h["water_use_m3_t"].sum()) / total_c) if total_c > 0 else 0.0
    ax.set_title(f"流域耗水下降 {cut:.0f}%，让出水量的\n主要是最大的那些厂址",
                 fontsize=6.4, linespacing=1.25)


# ── 装配 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    fig = plt.figure(figsize=(183 * MM, 184 * MM))
    letters = iter("abcdef")
    ledgers = []
    for col, spec in enumerate(COLUMNS):
        h = basin_frame(spec)
        x0 = 0.030 + col * 0.505
        # 地图顶到 0.915：它的两行标题要落在列首规则文字（0.975）之下。之前 0.965 的顶
        # 让标题直接压在规则文字上，右列尤其糊成一片。
        ax_map = fig.add_axes([x0 - 0.020, 0.660, 0.430, 0.255])
        ax_led = fig.add_axes([x0 + 0.115, 0.400, 0.300, 0.185])
        ax_sit = fig.add_axes([x0 + 0.115, 0.128, 0.300, 0.205])

        panel_map(ax_map, h, spec["code"])
        ledgers.append(panel_ledger(ax_led, spec, h))
        panel_sites(ax_sit, h, spec)

        for ax, dx, dy in ((ax_map, 0.005, 1.02), (ax_led, -0.52, 1.28), (ax_sit, -0.52, 1.20)):
            ax.text(dx, dy, next(letters), transform=ax.transAxes, fontsize=8.0,
                    fontweight="bold", va="top", ha="left")
        fig.text(x0 + 0.19, 0.992, spec["rule"], ha="center", va="top", fontsize=6.6,
                 fontweight="bold", color="#333333")
        fig.text(x0 + 0.19, 0.972, f"对比：{spec['ctrl_label']} → {spec['treat_label']}",
                 ha="center", va="top", fontsize=5.8, color="#666666")

    fig.text(0.030, 0.005,
             "左右两列的台账口径不同：左列全部是耗水，右列全部是取水（《中国水资源公报》"
             "用水量口径，含直流冷却过流量）。两列的数值不可相减，也不可放在同一根轴上比较。\n"
             "面板 b 是流域合计，而生态流量约束逐节点施加：黄河合计从未接近配额，其机组仍然"
             "改造，是因为流域内部分节点已经贴到各自上限。不得据面板 b 判断该约束是否松弛。\n"
             "面板 b、e 中的“全部机组捕集”是反事实：2030 年排放目标尚不约束，各情景捕集量均为 0。\n"
             "厂址归属流域用“最近可达供水节点”，与图 2、图 3 的“最近流域多边形”不是同一套规则"
             "（二者在共同覆盖的厂址上一致率 93.5%），因此同一流域同一步的合计量在两处会相差数个百分点。",
             fontsize=5.2, color="#555555", va="bottom", linespacing=1.5)

    save_fig(fig, "ed_fig6_basin_closeup", subdir="extended")

    print("Extended Data - 两条水规则各自最紧的流域")
    for rec in ledgers:
        name = BASIN_NAMES_ZH.get(rec["code"], rec["code"])
        print(f"  {rec['code']} {name} ({rec['rung']}): 配额 {rec['allowance']:.2f}"
              + (f" (折减前 {rec['raw_residual']:.1f})" if rec["raw_residual"] is not None else "")
              + f"  现状 {rec['as_built']:.2f}  全量捕集 {rec['full_capture']:.2f}"
              + (f"  {rec['ctrl_label']} {rec['before']:.2f}"
                 if np.isfinite(rec["before"]) else f"  {rec['ctrl_label']} n/a")
              + f" -> {rec['treat_label']} {rec['realised']:.2f}")
        print(f"    全量捕集 / 配额 = {rec['full_capture'] / rec['allowance']:.2f}；"
              f"现状 / 配额 = {rec['as_built'] / rec['allowance']:.2f}")


if __name__ == "__main__":
    main()
