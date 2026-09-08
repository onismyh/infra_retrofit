"""Extended Data: the accounting basis decides which cooling technology looks thirsty.

WHY THIS MATTERS ENOUGH FOR ITS OWN FIGURE, AND MORE SO SINCE v9.1. The model now constrains on
TWO of these three bases at once: the environmental-flow rule acts on CONSUMPTION and the
用水总量控制指标 acts on WITHDRAWAL, which is what the 公报 meters. The regulatory QUOTA
(qushui ding'e) is priced but constrains nothing. So this is no longer a note about bookkeeping
with one live basis and two diagnostics -- picking the wrong basis for the wrong rule changes
the answer, and the three bases do not merely differ in magnitude, they RANK THE COOLING
TECHNOLOGIES IN DIFFERENT ORDERS.

The 定额 excludes the once-through condenser flow; the 公报 includes it (直流火(核)电 453.8亿 is
a sub-column of 工业用水). Constraining the allocation rule on the 定额 would understate the
fleet's claim on it roughly five-fold. That is the single most consequential basis choice in the
study, and panel (a) is the evidence for it.

Capacity-weighted over the 350 sites, in m3 MWh-1. Sites are grouped by their CAPACITY-dominant
cooling technology, not by the unit-count mode the model's own `dominant_cooling_technology`
column uses -- that mode lets many small units outvote the big ones, and it mislabels 37 sites
carrying 193.2 GW:

    basis          once-through   recirculating   air (dry)     spread
    withdrawal           34.34           7.00        2.23        15.4x
    quota                 0.72           2.27        0.85         3.2x
    consumption           0.44           1.49        0.43         3.5x

    (under the unit-count mode: 38.95 / 9.28 / 1.83 and 0.57 / 2.03 / 0.76 -- the inversion
     holds under both, and is STRONGER on capacity: the once-through / wet-tower withdrawal
     ratio goes 4.20x -> 4.91x while the quota ratio stays below one, 0.28 -> 0.32)

Once-through is the thirstiest technology in the country under withdrawal and the LEAST thirsty
under the quota, because the quota excludes the condenser flow a once-through plant returns to
the river a few degrees warmer. A study that reports "water use" without naming its basis has not
said which of these two opposite statements it means.

Panel (c) answers the question the first two provoke: are the constrained basins short
because their plants are thirsty, or because the water is not there? Capacity-weighted they
consume 1.09 m3 MWh-1 against 0.77 for the rest -- a factor of 1.4, not a factor of ten -- and
the Yellow, the largest of the four, is the LEAST intense of them because 56% of its capacity is
already dry-cooled. The shortage is of supply, and the north has adapted where it has had to.

Note on the Chinese term: the repo font is SimHei and carries CJK fine (CLAUDE.md 3.1); what it
lacks is U+2212 and the superscript digits, so units go through mathtext. The old warning here
predates the SimHei migration.
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

import ed_plant_data as EP
from plot_style import BASIN_NAMES_ZH, MM, save_fig, safe_log_axis

TECH_COLOUR = {"once-through": "#4477AA", "recirculating": "#88CCEE", "Air": "#DDAA33"}
# 键是数据里的取值，不能动；只改显示名。
TECH_LABEL = {"once-through": "直流冷却", "recirculating": "循环冷却（湿冷塔）",
              "Air": "空冷（干冷）"}
BASES = [("withdrawal_intensity_m3_per_mwh", "取水量"),
         ("quota_intensity_m3_per_mwh", "取水定额"),
         ("consumption_intensity_m3_per_mwh", "耗水量")]
CONSTRAINED_C = "#CC3311"
QUIET = "#9AA5B1"


def _weighted(frame: pd.DataFrame, col: str) -> float:
    w = frame["total_capacity_mw"]
    return float((frame[col] * w).sum() / w.sum())


def panel_ranks(ax, fleet: pd.DataFrame) -> None:
    """The rank inversion, drawn as a slope chart on a log axis."""
    x = np.arange(len(BASES))
    # Per-technology label offsets: the three lines converge at the quota node, where the
    # once-through (0.72) and air (0.85) labels landed on top of each other.
    # Per node, not per technology: once-through leaves the first node on a steep descent, so a
    # single downward offset dropped its 34.34 label straight onto its own line.
    offset = {"once-through": [(0, 8), (0, -13), (0, -13)],
              "recirculating": [(0, 8), (0, 9), (0, 9)],
              "Air": [(0, 8), (0, 10), (0, 10)]}
    align = {"once-through": ["center", "center", "center"],
             "recirculating": ["center", "center", "center"],
             "Air": ["center", "center", "center"]}
    for tech, grp in fleet.groupby("cooling_by_capacity"):
        if tech not in TECH_COLOUR:
            continue
        y = [_weighted(grp, col) for col, _ in BASES]
        ax.plot(x, y, color=TECH_COLOUR[tech], lw=1.8, marker="o", ms=4.2,
                mec="white", mew=0.6, zorder=4, label=TECH_LABEL[tech])
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=offset[tech][i],
                        textcoords="offset points", fontsize=5.2, ha=align[tech][i],
                        color=TECH_COLOUR[tech], fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in BASES], fontsize=6.0)
    ax.set_xlim(-0.35, len(BASES) - 0.60)
    ax.set_ylim(0.17, 95)
    ax.set_ylabel("容量加权水强度\n（m$^3$ MWh$^{-1}$，对数轴）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="y", lw=0.3, alpha=0.30, which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=5.4, frameon=False, loc="lower left", handlelength=1.4,
              handletextpad=0.5, labelspacing=0.28)
    ax.set_title("按取水量算，直流冷却最耗水；按取水权配额算，\n"
                 "它又最省——两条线交叉",
                 fontsize=6.8, linespacing=1.25)


def panel_scatter(ax, fleet: pd.DataFrame) -> None:
    """Every site on the two bases that matter: what it takes, and what it consumes."""
    for tech, grp in fleet.groupby("cooling_by_capacity"):
        if tech not in TECH_COLOUR:
            continue
        ax.scatter(grp["withdrawal_intensity_m3_per_mwh"],
                   grp["consumption_intensity_m3_per_mwh"],
                   s=1.5 + 26.0 * grp["capacity_gw"] / fleet["capacity_gw"].max(),
                   c=TECH_COLOUR[tech], lw=0.2, edgecolor="white", alpha=0.85, zorder=3)
    lim = (0.06, 130)
    ax.plot(lim, lim, color="#999999", lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.annotate("1:1 线——取走的水\n全部被消耗", xy=(2.0, 2.0), xytext=(0.10, 5.0),
                fontsize=5.2, color="#777777", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999"))
    # The scatter groups by the CAPACITY-dominant cooling technology. The model's own
    # `dominant_cooling_technology` is a unit-COUNT mode, so a hub of many small units outvotes
    # the big ones: 37 sites carrying 193.2 GW change label when the vote is weighted by
    # capacity, and this cloud goes from 71 sites / 349.9 GW to 94 sites / 486.2 GW. Neither
    # equals the fleet-wide once-through COLUMN of 465.3 GW, because a once-through-dominated
    # site can still hold wet-tower units.
    ax.annotate("94 个以直流冷却为主的厂址、\n486.2 GW，位于远右侧", xy=(55, 0.5),
                xytext=(3.0, 0.10), fontsize=5.2, color=TECH_COLOUR["once-through"],
                linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=TECH_COLOUR["once-through"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    safe_log_axis(ax, "both")   # lim 低至 0.06
    ax.set_xlim(*lim)
    ax.set_ylim(0.06, 12)
    ax.set_xlabel("取水量（m$^3$ MWh$^{-1}$）", fontsize=6.4)
    ax.set_ylabel("耗水量（m$^3$ MWh$^{-1}$）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(lw=0.3, alpha=0.30, which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("与 1:1 线的距离就是全部论点：\n"
                 "直流冷却取走的水几乎全部又还了回去",
                 fontsize=6.8, linespacing=1.25)


def panel_adapted(ax, fleet: pd.DataFrame) -> pd.DataFrame:
    """Is the shortage about thirsty plants, or about absent water?

    Drawn as a ranked dot plot rather than a bubble chart. An earlier version encoded basin
    capacity as marker area, which at 291 GW produced markers of roughly 880 pt2 -- they
    swallowed their own labels and overlapped one another. Capacity is text here; position
    carries the two quantities that matter.
    """
    g = (fleet.groupby(["basin_code", "basin_name"])
         .apply(lambda x: pd.Series({
             "gw": x["total_capacity_mw"].sum() / 1e3,
             "cons": (x["consumption_intensity_m3_per_mwh"] * x["total_capacity_mw"]).sum()
                     / x["total_capacity_mw"].sum(),
             "air": (x["share_air"] * x["total_capacity_mw"]).sum()
                    / x["total_capacity_mw"].sum()}), include_groups=False)
         .reset_index())
    g = g[g["gw"] > 1.0].sort_values("cons").reset_index(drop=True)
    g["constrained"] = g["basin_code"].isin(EP.CONSTRAINED_BASINS)

    y = np.arange(len(g))
    for yi, r in zip(y, g.itertuples()):
        colour = CONSTRAINED_C if r.constrained else QUIET
        ax.plot([0, r.cons], [yi, yi], color=colour, lw=0.9, alpha=0.55, zorder=2)
        ax.plot([r.cons], [yi], marker="o", ms=4.6, color=colour, mec="white", mew=0.6,
                zorder=4)
        ax.annotate(f"{r.gw:.0f} GW    空冷占 {100 * r.air:.0f}%", xy=(r.cons, yi),
                    xytext=(9, 0), textcoords="offset points", va="center", fontsize=5.3,
                    color="#444444")

    con, unc = g[g["constrained"]], g[~g["constrained"]]
    # The two means are 0.32 m3/MWh apart, which at this scale is close enough that two centred
    # labels at the same height overprinted each other. Stagger them and hang each off its line.
    for sub, colour, name, ha, dy in ((con, CONSTRAINED_C, "超出配额", "left", 0.55),
                                      (unc, QUIET, "配额之内", "right", -0.05)):
        w = (sub["cons"] * sub["gw"]).sum() / sub["gw"].sum()
        ax.axvline(w, color=colour, lw=0.9, ls=(0, (3, 2)), zorder=3)
        pad = 0.012 * ax.get_xlim()[1] * (1 if ha == "left" else -1)
        ax.annotate(f"{name} {w:.2f}", xy=(w + pad, len(g) - 0.30 + dy), fontsize=5.2,
                    color=colour, ha=ha, va="bottom", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([BASIN_NAMES_ZH.get(r.basin_code, r.basin_name) for r in g.itertuples()], fontsize=5.8)
    for tick, flag in zip(ax.get_yticklabels(), g["constrained"]):
        if flag:
            tick.set_color(CONSTRAINED_C)
            tick.set_fontweight("bold")
    ax.set_xlim(0, g["cons"].max() * 2.30)
    ax.set_ylim(-0.7, len(g) + 0.75)
    ax.set_xlabel("容量加权耗水强度（m$^3$ MWh$^{-1}$）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("四个缺水流域的单位电量耗水只高出 1.4 倍；其中规模最大的黄河流域\n"
                 "强度反而最低，因为它已有 56% 采用空冷",
                 fontsize=6.8, linespacing=1.25)
    return g


def main() -> None:
    fleet = EP.fleet()
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    ax_rank = fig.add_axes([0.085, 0.615, 0.370, 0.305])
    ax_sc = fig.add_axes([0.600, 0.615, 0.375, 0.305])
    ax_ad = fig.add_axes([0.130, 0.085, 0.610, 0.330])

    panel_ranks(ax_rank, fleet)
    panel_scatter(ax_sc, fleet)
    g = panel_adapted(ax_ad, fleet)

    for ax, lab, dx in ((ax_rank, "a", -0.130), (ax_sc, "b", -0.130), (ax_ad, "c", -0.190)):
        ax.text(dx, 1.24, lab, transform=ax.transAxes, fontsize=8.0,
                fontweight="bold", va="top", ha="left")

    save_fig(fig, "ed_fig3_water_basis", subdir="extended")

    print("  capacity-weighted intensity by technology (m3/MWh):")
    for tech, grp in fleet.groupby("cooling_by_capacity"):
        if tech not in TECH_COLOUR:
            continue
        vals = "  ".join(f"{lab}={_weighted(grp, col):7.3f}" for col, lab in BASES)
        print(f"    {tech:>14} ({grp['capacity_gw'].sum():6.1f} GW)  {vals}")
    con, unc = g[g["constrained"]], g[~g["constrained"]]
    wc = (con["cons"] * con["gw"]).sum() / con["gw"].sum()
    wu = (unc["cons"] * unc["gw"]).sum() / unc["gw"].sum()
    print(f"  constrained {wc:.3f} vs unconstrained {wu:.3f} m3/MWh  ({wc / wu:.2f}x)")


if __name__ == "__main__":
    main()
