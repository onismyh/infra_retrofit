"""Extended Data: the fleet converts to dry cooling for the transition, then reverts to wet.

WHAT THIS FIGURE CORRECTS. The paper's headline physical series -- dry-cooling conversion peaking
near 380 GW around 2040 and falling to about 40 GW by 2060 -- is an OPERATING share, not a stock.
Nothing in the model outputs records the stock: `air_installed` is never written. Reconstructing it
as each site's running-maximum operating fraction, applied to that year's surviving capacity so
retirement cannot masquerade as idle plant, gives a very different picture of the same run:

    year   converted stock   operating dry   installed but running WET
    2030        354.7 GW         354.7 GW            0.0 GW    ( 0 %)
    2040        384.1 GW         379.8 GW            4.3 GW    ( 1 %)
    2050        225.4 GW         178.1 GW           47.3 GW    (21 %)
    2060        140.6 GW          43.6 GW           97.0 GW    (69 %)

So the decline is two different things added together. Of the 336 GW fall in operating dry cooling
between 2040 and 2060, 244 GW is converted capacity RETIRING and 93 GW is capacity REVERTING to wet
operation once water stops binding at its node. 125 of the 350 sites end below their own peak.

This matters for how the result is described. "Water forces a wave of dry-cooling retrofit" is
right; "and the fleet stays dry" is not. The retrofit is bought for the 2030-2050 window when the
constraint binds and the capital is committed, and a large part of it is then idled rather than
used -- which is also why the capex assumption matters less to the cost than one would expect.
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

from plot_style import MM, save_fig

TREAT, CONTROL = "WA_cwatm_126_dry_oq", "WA_cwatm_126_dry_oq_envonly"
YEARS = (2030, 2040, 2050, 2060)
STOCK_C = "#4477AA"
OP_C = "#CC3311"
QUIET = "#9AA5B1"


def series(scenario: str) -> pd.DataFrame:
    """Converted dry-cooling stock and operation, GW, net of capacity built dry.

    Only CONVERTED capacity can revert: a unit built dry has no wet condenser to fall back to.
    So the quantity tracked is `air_cooled_share - already_air_share`, floored at zero, and the
    stock is its running maximum per site.
    """
    d = pd.read_csv(Path("results") / scenario / "plant_detail.csv").sort_values(
        ["plant_id", "year"])
    d["cap_gw"] = d["capacity_mw"] / 1e3
    d["alive"] = 1.0 - d["share_retire"].fillna(0.0)
    # `air_cooled_share` is the dry-operating fraction OF THE STILL-WET capacity, not of the
    # whole site: `plant_matrices._air_cooling_matrices` sets still_wet = 1 - already_air_share and applies the
    # air share to it. The decisive test is the water arithmetic, not a row count: on the
    # 26 plant-years where the two readings differ, the multiplicative form reproduces the
    # solver's own `water_use_m3` to 1e-16 while the subtractive form is wrong by 24-97%.
    # (An earlier note here cited '184 plant-years with air < already'; 166 of those are
    #  simply air == 0 and prove nothing. Only 18 are genuinely 0 < air < already.)
    d["conv_op"] = (d["air_cooled_share"].fillna(0.0)
                    * (1.0 - d["already_air_share"].fillna(0.0)))
    d["conv_stock"] = d.groupby("plant_id")["conv_op"].cummax()
    rows = []
    for y in YEARS:
        s = d[d["year"] == y]
        op = float((s["cap_gw"] * s["alive"] * s["conv_op"]).sum())
        st = float((s["cap_gw"] * s["alive"] * s["conv_stock"]).sum())
        rows.append({"year": y, "operated": op, "stock": st, "wet": st - op,
                     "pct_wet": 100 * (st - op) / st if st > 0 else 0.0})
    return pd.DataFrame(rows).set_index("year")


def panel_wedge(ax, t: pd.DataFrame, c: pd.DataFrame) -> None:
    x = np.array(YEARS)
    ax.fill_between(x, t["operated"], t["stock"], color=OP_C, alpha=0.16, lw=0, zorder=2,
                    label="已改造但仍按湿冷运行")
    ax.plot(x, t["stock"], color=STOCK_C, lw=1.8, marker="o", ms=4.0, mec="white", mew=0.6,
            zorder=5, label="已改造存量（重建）")
    ax.plot(x, t["operated"], color=OP_C, lw=1.8, marker="s", ms=4.0, mec="white", mew=0.6,
            zorder=5, label="实际按空冷运行")
    ax.plot(x, c["operated"], color=QUIET, lw=1.2, ls=(0, (3, 2)), marker="^", ms=3.4,
            mec="white", mew=0.5, zorder=4, label="按空冷运行，无预留（对照）")

    for y in (2050, 2060):
        # 两个标注都放在蓝线之上、两点之间的空档里：2050 的在点右上，2060 的在点左上。
        # 放在两线之间的粉色带里会压住下行的红线。
        ax.annotate(f"{t.loc[y, 'wet']:.0f} GW\n（占存量 {t.loc[y, 'pct_wet']:.0f}%）",
                    xy=(y, 0.5 * (t.loc[y, "operated"] + t.loc[y, "stock"])),
                    xytext=(-30, 34) if y == 2060 else (34, 26), textcoords="offset points",
                    fontsize=5.4, color=OP_C, ha="center", linespacing=1.15, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in YEARS], fontsize=5.8)
    ax.set_ylabel("已改造空冷容量（GW）", fontsize=6.4)
    # Peaks reach 384 GW; at ylim 430 the upper-right legend sat on the curves.
    ax.set_ylim(0, 530)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=5.3, frameon=False, loc="upper right", handlelength=1.6,
              handletextpad=0.5, labelspacing=0.3)
    ax.set_title("已发表的改造曲线是“运行量”序列。\n"
                 "其背后的存量下降要慢得多。",
                 fontsize=6.8, linespacing=1.25)


def panel_decompose(ax, t: pd.DataFrame) -> None:
    """Split the 2040-2060 fall in operating dry cooling into retirement and reversion."""
    peak, end = t.loc[2040], t.loc[2060]
    total = peak["operated"] - end["operated"]
    retire = peak["stock"] - end["stock"]
    revert = total - retire
    vals = [peak["operated"], -retire, -revert, end["operated"]]
    labels = ["2040 年\n运行量", "退役", "回退为\n湿冷", "2060 年\n运行量"]
    colours = ["#4477AA", "#6B7683", OP_C, "#4477AA"]
    run = 0.0
    for i, (v, colour) in enumerate(zip(vals, colours)):
        if i in (0, 3):
            ax.bar(i, v, width=0.6, color=colour, edgecolor="white", lw=0.4, zorder=3)
            run = v
            ax.annotate(f"{v:.0f}", xy=(i, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=5.6, fontweight="bold", color=colour)
        else:
            ax.bar(i, v, bottom=run, width=0.6, color=colour, edgecolor="white", lw=0.4,
                   zorder=3)
            # Placed OUTSIDE the bar in its own colour. White text inside assumed the bar was
            # tall enough to contain two lines, which it is not for the reversion step.
            ax.annotate(f"{v:+.0f} GW\n({100 * abs(v) / total:.0f}%)",
                        xy=(i, run + v / 2.0), xytext=(0, 0), textcoords="offset points",
                        ha="center", va="center", fontsize=5.3, fontweight="bold",
                        # Always outside-in on a white plate: the white variant
                        # overflowed the bar, clipping the minus sign and the "W",
                        # so a -244 GW decrement read as "244 GW".
                        color=colour,
                        bbox=dict(facecolor="white", lw=0, alpha=0.82, pad=0.8))
            run += v
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=5.5, linespacing=1.15)
    ax.set_ylabel("运行中的空冷容量（GW）", fontsize=6.4)
    ax.set_ylim(0, 430)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="y", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"下降量中只有 {100 * retire / total:.0f}% 来自退役，\n"
                 f"其余是设备被闲置。", fontsize=6.8, linespacing=1.25)


def panel_sites(ax) -> None:
    """How many sites end below their own peak, and by how much."""
    d = pd.read_csv(Path("results") / TREAT / "plant_detail.csv").sort_values(
        ["plant_id", "year"])
    d["conv_op"] = (d["air_cooled_share"].fillna(0.0)
                    * (1.0 - d["already_air_share"].fillna(0.0)))
    d["cap_gw"] = d["capacity_mw"] / 1e3
    piv = d.pivot_table(index="plant_id", columns="year", values="conv_op")
    cap = d.groupby("plant_id")["cap_gw"].first()
    peak = piv.max(axis=1)
    drop = (peak - piv[2060]).clip(lower=0.0)
    active = peak > 0.01
    gw_lost = (drop * cap)[active]
    reverted = active & (drop > 0.02)

    # WEIGHT BY SURVIVING CAPACITY. The first version weighted by each site's full capacity,
    # so of the 586.5 GW it drew, 421.1 GW (71.8%) had already RETIRED by 2060 and 58 of the
    # 123 sites were >=90% retired. Panel (b) already accounts for retirement separately, so
    # that panel was drawing the retirement term a second time under a reversion title.
    alive60 = 1.0 - (d[d["year"] == 2060].set_index("plant_id")["share_retire"].fillna(0.0))
    live_cap = (cap * alive60).reindex(cap.index).fillna(0.0)
    sel = active & (drop > 0.02)
    bins = np.linspace(0, 1, 21)
    ax.hist(drop[sel], bins=bins, weights=live_cap[sel],
            color=OP_C, alpha=0.85, edgecolor="white", lw=0.3, zorder=3)
    ax.set_xlabel("相对该厂址自身峰值改造\n份额的回落，至 2060 年", fontsize=6.4,
                  linespacing=1.2)
    ax.set_ylabel("存续厂址容量（GW）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="y", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.annotate(f"曾经改造过的 {int(active.sum())} 个厂址中，\n"
                f"有 {int(sel.sum())} 个最终低于自身峰值，\n按存续到 2060 年的容量加权\n"
                f"（{live_cap[sel].sum():.0f} GW）",
                xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top",
                fontsize=5.4, color="#444444", linespacing=1.25)
    ax.set_title("回退不是个别异常点", fontsize=6.8, linespacing=1.25)


def main() -> None:
    t, c = series(TREAT), series(CONTROL)
    fig = plt.figure(figsize=(183 * MM, 88 * MM))
    ax1 = fig.add_axes([0.065, 0.235, 0.265, 0.545])
    ax2 = fig.add_axes([0.420, 0.235, 0.225, 0.545])
    ax3 = fig.add_axes([0.740, 0.235, 0.240, 0.545])

    panel_wedge(ax1, t, c)
    panel_decompose(ax2, t)
    panel_sites(ax3)

    for ax, lab in ((ax1, "a"), (ax2, "b"), (ax3, "c")):
        ax.text(-0.155, 1.30, lab, transform=ax.transAxes, fontsize=8.0,
                fontweight="bold", va="top", ha="left")

    save_fig(fig, "ed_fig5_reversion", subdir="extended")

    print("  treatment:"); print(t.round(1).to_string())
    print("  control:");   print(c.round(1).to_string())
    peak, end = t.loc[2040], t.loc[2060]
    total = peak["operated"] - end["operated"]
    retire = peak["stock"] - end["stock"]
    print(f"  2040->2060 fall {total:.1f} GW = retirement {retire:.1f} "
          f"({100 * retire / total:.0f}%) + reversion {total - retire:.1f} "
          f"({100 * (total - retire) / total:.0f}%)")


if __name__ == "__main__":
    main()
