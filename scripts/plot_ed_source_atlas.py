"""Extended Data: 排放源图谱 —— 煤电与五类高耗能工业点源，以及它们争的同一份流域水量。

WHY THIS FIGURE EXISTS. Every map in this repository up to now draws the coal fleet and nothing
else: 13 plotting scripts call the basemap helpers and not one of them reads
`inputs/industry_sources.csv`. That is a real omission rather than a stylistic one, because the
water constraint this paper is built on is enforced at BASIN level against a residual that the
coal fleet and the industrial point sources have to share -- `write_basin_caps` carves the
modelled industry's current withdrawal back OUT of the reservation precisely so that the two
compete inside one number. A figure series in which industry never appears cannot show that.

WHAT IS DRAWN, AND WHAT IS NOT. Five sectors, because five is what the model scopes in
(`constants_industry.INDUSTRY_SECTORS`). 炼化 and 现代煤化工 (which is where 乙二醇 and 烯烃 sit)
are in `SECTORS_OUT_OF_SCOPE` and are therefore absent by decision, not by data gap; the caption
says so rather than leaving a reader to infer a missing layer.

MARKER AREA IS CO2, FOR BOTH SIDES, ON ONE SCALE. Sizing coal by GW and industry by Mt would put
two units on one visual channel and make the comparison meaningless. Baseline emissions are the
one quantity both carry, and `baseline_emissions_mt` is verified scenario-independent (max
per-hub difference between BASE and `WA_cwatm_126_dry_oq` is exactly 0.0), so this figure depends
on NO solved scenario and cannot mix input versions -- the failure mode CLAUDE.md 二.6 exists for.

NO CLIPPING ON THE SIZE SCALE. A reference value that keeps the industrial cloud legible would
flatten 62 of 350 coal hubs at vref = 25 Mt. The un-clipped scale instead says something true:
individual coal hubs are much larger emitters (median 9.4 Mt, max 114.8) than individual
industrial plants (median 0.83, max 20.7), and industry's 3,269 Mt arrives as 2,552 small points
rather than as a few large ones. That asymmetry is the reason the two behave differently under a
per-basin cap, so it should be visible rather than normalised away.

STYLE follows 昊天 (Haotian Tang) 氢管网论文 EST, `GIS_layer/plot.ipynb` Figure 6B: pentagon
markers, Dark2 palette, alpha 0.55, category legend at upper right with `frameon=False` and
`markerscale`, and a South China Sea inset that redraws every business layer. Coal keeps
`#636363` from this repository's own pathway table so the fleet reads the same colour here as in
every other figure. Steel's two routes share a green family, following the CLAUDE.md 三.3 rule
that one class stays in one ramp.

Panel (b) is not in Haotian's figure and is the reason this is our figure rather than a copy of
his: it puts the two source populations against the constraint that binds them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_style import (
    apply_style,
    add_scs_inset,
    cjk_fill,
    assign_basin,
    BASIN_NAMES_ZH,
    BASIN_ORDER,
    draw_china_basemap,
    hub_frame,
    mainland_extent,
    MM,
    panel_label,
    RESULTS_DIR,
    ROOT,
    save_fig,
    to_map_xy,
)

INPUTS = ROOT / "inputs"
# RESULTS_DIR，不要自己拼 `ROOT / "_v91tree" / "results"`：脚本在求解树里跑时 ROOT 就是
# _v91tree 本身，硬拼会变成 _v91tree/_v91tree/results。

# 煤电沿用本仓库通路色表（CLAUDE.md 三.3），工业用昊天 Fig 6B 的 Dark2 系。
# 钢铁两条路线共用绿色族 —— 同一大类留在同一条 ramp 上。
COAL_COLOUR = "#636363"
SECTOR_STYLE: dict[str, tuple[str, str]] = {
    "steel_bf_bof": ("#1B9E77", "钢铁（高炉-转炉）"),
    "steel_eaf":    ("#66A61E", "钢铁（电炉）"),
    "cement":       ("#D95F02", "水泥"),
    "ammonia":      ("#7570B3", "合成氨"),
    "methanol":     ("#E7298A", "甲醇"),
}
SECTOR_ORDER = ["steel_bf_bof", "steel_eaf", "cement", "ammonia", "methanol"]

MARKER_INDUSTRY = "p"        # 昊天 Fig 6B 的五边形
MARKER_COAL = "o"
ALPHA = 0.55
AREA_MIN, AREA_MAX = 1.2, 42.0
SIZE_TICKS = [1.0, 10.0, 50.0, 100.0]      # Mt CO2/yr，图例分档

C_COAL_BAR = "#636363"
C_IND_BAR = "#1B9E77"
C_LIMIT = "#CC3311"


def _area(values: np.ndarray, vmax: float) -> np.ndarray:
    """Marker AREA proportional to emissions, with a floor so small sources stay visible.

    Area rather than radius: encoding magnitude on the radius understates large sources by the
    square root, which is the most common way a point map misleads.
    """
    return AREA_MIN + (AREA_MAX - AREA_MIN) * np.clip(values / vmax, 0.0, 1.0)


def coal_sources() -> pd.DataFrame:
    """350 厂址级 hub：坐标、装机、基准排放。不依赖任何求解结果。"""
    plants = pd.read_csv(INPUTS / "plants.csv")
    emissions = (hub_frame("BASE", 2030, results_dir=RESULTS_DIR)
                 .set_index("plant_id")["baseline_emissions_mt"])
    plants = plants.set_index("plant_id")
    plants["co2_mt"] = emissions.reindex(plants.index)
    plants["cap_gw"] = plants["total_capacity_mw"] / 1000.0
    return plants.reset_index()


def industry_sources() -> pd.DataFrame:
    """2 552 个在模型范围内的工业点源。"""
    frame = pd.read_csv(INPUTS / "industry_sources.csv")
    return frame[frame["sector"].isin(SECTOR_ORDER)].copy()


def basin_competition() -> pd.DataFrame:
    """逐流域：煤电现状取水、工业现状取水，对被执行的余量。

    余量取 `inputs/water_basin_caps.csv` 的 `residual_1e8_m3`，即约束右端项
    `指标 - 预留x折减 + 已建模工业x折减`，**不是**诊断脚本里那个中间量 `指标 - 预留`
    （两者全国差 161 亿，西北诸河差一个符号）。这一点很要紧：余量里本来就把已建模工业
    的取水加了回去，所以能与它相比的分子是"煤电 + 已建模工业"，不是煤电一家。
    """
    from coal_retrofit.paths import ProjectPaths
    from diagnose_official_water_budget import fleet_water_by_basin

    paths = ProjectPaths(ROOT)
    fleet, _factor = fleet_water_by_basin(paths)
    coal = fleet["withdrawal_calibrated"]

    industry = industry_sources()
    industry["basin"] = assign_basin(industry, "longitude", "latitude")
    ind = industry.groupby("basin")["water_m3_per_year"].sum() / 1e8

    caps = pd.read_csv(INPUTS / "water_basin_caps.csv")
    caps = caps[caps["planning_year"] == caps["planning_year"].min()]
    residual = caps.set_index("basin_code")["residual_1e8_m3"].astype(float)

    frame = pd.DataFrame(index=BASIN_ORDER)
    frame["coal"] = coal.reindex(frame.index).fillna(0.0)
    frame["industry"] = ind.reindex(frame.index).fillna(0.0)
    frame["residual"] = residual.reindex(frame.index)
    frame["coal_pct"] = frame["coal"] / frame["residual"] * 100.0
    frame["ind_pct"] = frame["industry"] / frame["residual"] * 100.0
    frame["joint_pct"] = frame["coal_pct"] + frame["ind_pct"]
    return frame


def _draw_points(ax, coal: pd.DataFrame, industry: pd.DataFrame, vmax: float,
                 scale: float = 1.0) -> None:
    """业务图层。主图与南海小图各调一次，保证两处同色标、同尺寸律。"""
    for sector in SECTOR_ORDER:
        part = industry[industry["sector"] == sector]
        if part.empty:
            continue
        x, y = to_map_xy(part["longitude"].to_numpy(), part["latitude"].to_numpy())
        ax.scatter(x, y, s=_area(part["co2_mt_per_year"].to_numpy(), vmax) * scale,
                   c=SECTOR_STYLE[sector][0], marker=MARKER_INDUSTRY, alpha=ALPHA,
                   linewidths=0.0, zorder=3)
    x, y = to_map_xy(coal["centroid_longitude"].to_numpy(),
                     coal["centroid_latitude"].to_numpy())
    ax.scatter(x, y, s=_area(coal["co2_mt"].to_numpy(), vmax) * scale, c=COAL_COLOUR,
               marker=MARKER_COAL, alpha=ALPHA, edgecolors="white",
               linewidths=0.30 * scale, zorder=4)


def panel_a(ax, coal: pd.DataFrame, industry: pd.DataFrame) -> float:
    vmax = float(max(coal["co2_mt"].max(), industry["co2_mt_per_year"].max()))
    draw_china_basemap(ax, facecolor="#F7F8F9")
    _draw_points(ax, coal, industry, vmax)
    mainland_extent(ax)
    ax.set_axis_off()
    # 小图必须重画全部业务图层（CLAUDE.md 四.3）：九段线南端到 3.85N，主图 ylim 从 17N 起
    # 就把它裁掉了，小图不是装饰。点在小图里按 0.55 缩，否则会盖住岛礁轮廓。
    add_scs_inset(ax.get_figure(), ax,
                  draw=lambda a: _draw_points(a, coal, industry, vmax, scale=0.55))

    handles = [Line2D([], [], marker=MARKER_COAL, color="none", markerfacecolor=COAL_COLOUR,
                      markeredgewidth=0, markersize=4.2, alpha=ALPHA,
                      label=f"煤电厂址（{len(coal)}）")]
    handles += [Line2D([], [], marker=MARKER_INDUSTRY, color="none",
                       markerfacecolor=SECTOR_STYLE[s][0], markeredgewidth=0, markersize=4.2,
                       alpha=ALPHA,
                       label=f"{SECTOR_STYLE[s][1]}（{int((industry['sector'] == s).sum())}）")
                for s in SECTOR_ORDER]
    # 左下（青藏那一角）而不是左上：左上是新疆，机组密集，图例会把它整片盖掉。
    # 75% 白底而不是 frameon=False：昊天原图的无框图例在地图**外面**，我们没有外侧空间，
    # 无框会让"（271）"这类文字被省界穿过。可读性优先于照搬。
    first = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.235),
                      fontsize=5.6, frameon=True, facecolor="white", framealpha=0.75,
                      edgecolor="none", handletextpad=0.35,
                      labelspacing=0.34, borderaxespad=0.0)
    ax.add_artist(first)

    # 尺寸图例另建：不能直接把数据点当图例句柄（CLAUDE.md 四.4）。
    size_handles = [Line2D([], [], marker=MARKER_COAL, color="none", markerfacecolor="#B0B0B0",
                           markeredgewidth=0, alpha=0.85,
                           markersize=float(np.sqrt(_area(np.array([v]), vmax)[0])),
                           label=f"{v:g}")
                    for v in SIZE_TICKS]
    ax.legend(handles=size_handles, loc="lower left", bbox_to_anchor=(0.020, 0.012),
              fontsize=5.6, frameon=True, facecolor="white", framealpha=0.75,
              edgecolor="none", handletextpad=0.60, labelspacing=0.62,
              borderaxespad=0.0, alignment="left", title_fontsize=5.8,
              title=r"点面积 $\propto$ 排放量" "\n" r"（Mt CO$_2$ yr$^{-1}$）")
    return vmax


def panel_b(ax, frame: pd.DataFrame) -> None:
    """逐流域：煤电与工业各占被执行余量的多少，对数轴，100% 为红线。"""
    ys = np.arange(len(frame))[::-1]
    h = 0.34
    ax.barh(ys + h / 2, frame["coal_pct"], height=h, color=C_COAL_BAR, label="煤电（现状取水）",
            zorder=3)
    ax.barh(ys - h / 2, frame["ind_pct"], height=h, color=C_IND_BAR,
            label="五类工业点源（现状取水）", zorder=3)
    ax.scatter(frame["joint_pct"], ys, s=9, marker="D", facecolor="white",
               edgecolor="#333333", linewidths=0.6, zorder=5, label="两者合计")

    ax.axvline(100.0, color=C_LIMIT, lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax.text(100.0, len(frame) - 0.35, "用水总量指标余量 = 100%", color=C_LIMIT, fontsize=5.4,
            ha="right", va="bottom", rotation=0)

    ax.set_xscale("log")
    ax.set_xlim(0.4, 1600)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1%", "10%", "100%", "1000%"], fontsize=5.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c} {BASIN_NAMES_ZH[c]}" for c in frame.index], fontsize=5.8)
    ax.set_xlabel("占被执行的流域余量（对数轴）", fontsize=6.2, labelpad=2.0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="x", color="#E4E7EA", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.4, loc="lower right", frameon=False, handlelength=1.0,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)

    k = frame.loc["K"]
    y_k = float(ys[list(frame.index).index("K")])
    # 放 K 行正上方、贴轴右缘。贴在工业条右端会撞上同一行 646% 处的『两者合计』菱形；
    # 放行下方会压到长江那一行的条。E 行与 K 行之间、x > 100% 的这块实测是空的。
    ax.text(1520.0, y_k + 0.58,
            f"工业一家就占 {k['ind_pct']:.0f}%\n（{k['industry']:.1f} 对 {k['residual']:.1f} 亿 m$^3$）",
            fontsize=5.2, color="#12805F", ha="right", va="center", linespacing=1.35)

    # 西南诸河的煤电取水是 0.0，对数轴上画不出条，空着会被读成缺数据。
    if float(frame.loc["J", "coal"]) == 0.0:
        y_j = float(ys[list(frame.index).index("J")])
        ax.text(0.52, y_j + h / 2, "无煤电机组", fontsize=5.0, color="#8A9199",
                ha="left", va="center", style="italic")


def report(coal: pd.DataFrame, industry: pd.DataFrame, frame: pd.DataFrame) -> None:
    print(f"  煤电 {len(coal)} 厂址  {coal['cap_gw'].sum():.0f} GW  "
          f"{coal['co2_mt'].sum():.0f} Mt CO2/yr")
    print(f"  工业 {len(industry)} 点源  {industry['co2_mt_per_year'].sum():.0f} Mt CO2/yr  "
          f"（其中过程排放 {industry['process_co2_mt_per_year'].sum():.0f} Mt）")
    print(f"  取水：煤电 {frame['coal'].sum():.1f} 亿 vs 工业 {frame['industry'].sum():.1f} 亿 "
          f"m3/yr，比 {frame['coal'].sum() / frame['industry'].sum():.1f}:1")
    over = frame[frame["joint_pct"] > 100.0]
    print(f"  合计越过余量的流域：{list(over.index) or '无'}")
    for code, row in frame.iterrows():
        print(f"    {code} {BASIN_NAMES_ZH[code]:<7} 煤电 {row['coal_pct']:7.1f}%  "
              f"工业 {row['ind_pct']:6.1f}%  合计 {row['joint_pct']:7.1f}%  "
              f"（余量 {row['residual']:.1f} 亿 m3）")


def main() -> None:
    apply_style()
    coal, industry = coal_sources(), industry_sources()
    frame = basin_competition()

    fig = plt.figure(figsize=(183 * MM, 104 * MM))
    ax_map = fig.add_axes([0.005, 0.045, 0.60, 0.93])
    ax_bar = fig.add_axes([0.700, 0.170, 0.285, 0.735])

    panel_a(ax_map, coal, industry)
    panel_b(ax_bar, frame)
    panel_label(ax_map, "a", x=0.02, y=0.99)
    panel_label(ax_bar, "b", x=-0.30, y=1.05)

    # 用 cjk_fill 折行：一行摊开是 189 mm，越过 183 mm 的双栏宽守卫。中英混排时
    # 直接用 textwrap 会把全角字当一列，折出来的行仍然超宽。
    fig.text(0.005, 0.004, cjk_fill(
        "五类为模型范围内行业；炼化与现代煤化工（含乙二醇、烯烃）在 "
        "constants_industry.SECTORS_OUT_OF_SCOPE 内，按设计不画。"
        "(b) 的分母是约束右端项 residual_1e8_m3，其中已把这些工业点源的现状取水加回，"
        "故分子取煤电与工业之和。", 112),
        fontsize=5.0, color="#555555", ha="left", va="bottom", linespacing=1.45)

    report(coal, industry, frame)
    save_fig(fig, "ed_fig15_source_atlas", subdir="extended")


if __name__ == "__main__":
    main()
