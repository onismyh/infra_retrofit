"""Extended Data: which plants actually carry the water response, and why those.

WHY THIS FIGURE IS NEW. The paper's central physical result is that making the non-power water
reservation bind pushes the fleet to convert condensers from wet to dry cooling -- 127 GW to
355 GW at 2030, a difference of +229 GW. Until now that number appeared only as a national
total, in a scenario-level bar. A national total cannot answer the question a referee asks
immediately: WHICH plants, and is it a broad shift or a handful of sites?

Measured at site level it is emphatically the latter, and the concentration is the result:

  * 74 of 350 sites respond at all. The other 276 convert exactly as much as they would with
    no reservation.
  * The ten largest responders carry 38% of the national response, twenty-five carry 71%,
    fifty carry 93%.
  * 97% of the response lands in the four basins already over their dry-season allowance
    (Hai 88 GW, Yellow 59 GW, Northwest Interior 40 GW, Huai 34 GW).
  * The discriminator is COOLING TECHNOLOGY, not size, age or storage access. Responding sites
    have a median recirculating (wet tower) share of 1.00 against 0.48 for the rest, while
    median age differs by one year and median site capacity by 1.4 GW. Once-through plants do
    not respond at all -- they are unconstrained in this model -- and already-dry plants have
    nothing left to convert.

So the adaptation is not a fleet-wide behaviour. It is available to wet-tower capacity sitting
in a stressed basin, and that is a well-defined 364 GW subset of a 1,416 GW fleet.
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
from plot_style import BASIN_NAMES_ZH, MM, save_fig

CONTROL, TREATMENT, YEAR = "WA_cwatm_126_dry_oq_envonly", "WA_cwatm_126_dry_oq", 2030
_NET_GW = 0.0
BASIN_COLOUR = {"C": "#CC3311", "D": "#EE7733", "E": "#CCBB44", "K": "#AA3377"}
QUIET = "#B7BEC6"


def panel_map(ax, pr: pd.DataFrame) -> None:
    from plot_extended import _load_provinces, _reproj
    from plot_style import draw_china_basemap, add_scs_inset, mainland_extent

    provinces, _country = draw_china_basemap(ax, facecolor="#F7F8F9")

    pts = _reproj(pr.dropna(subset=["centroid_longitude", "centroid_latitude"]),
                  "centroid_longitude", "centroid_latitude").copy()
    resp = pts[pts["d_converted_gw"] > 0.01]
    rest = pts[pts["d_converted_gw"] <= 0.01]

    ax.scatter(rest.geometry.x, rest.geometry.y, s=1.6, c=QUIET, lw=0, alpha=0.8, zorder=2)
    scale = 60.0 / resp["d_converted_gw"].max()
    for code, colour in BASIN_COLOUR.items():
        sub = resp[resp["basin_code"] == code]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=1.5 + scale * sub["d_converted_gw"],
                   c=colour, lw=0.25, edgecolor="white", alpha=0.92, zorder=4)
    other = resp[~resp["basin_code"].isin(BASIN_COLOUR)]
    ax.scatter(other.geometry.x, other.geometry.y,
               s=1.5 + scale * other["d_converted_gw"], c="#6B7683", lw=0.25,
               edgecolor="white", alpha=0.9, zorder=3)

    x0, y0, x1, y1 = mainland_extent(ax)
    ax.set_axis_off()
    add_scs_inset(ax.get_figure(), ax)

    sx, sy = x0 + 0.06 * (x1 - x0), y0 + 0.10 * (y1 - y0)
    ax.plot([sx, sx + 500_000], [sy, sy], color="#333333", lw=1.1, zorder=20,
            solid_capstyle="butt")
    ax.text(sx + 250_000, sy - 0.018 * (y1 - y0), "500 km", ha="center", va="top",
            fontsize=5.2, color="#333333", zorder=20)

    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=np.sqrt(1.5 + scale * v),
                              mfc="#6B7683", mec="white", mew=0.3, label=f"+{v:g} GW")
                       for v in (2, 5, 10)]
                      + [Line2D([], [], marker="o", ls="none", ms=np.sqrt(1.6), mfc=QUIET,
                                mec="none", label="无响应")],
              title="2030 年新增\n空冷改造", fontsize=5.2, title_fontsize=5.4,
              frameon=False, loc="lower left", bbox_to_anchor=(0.005, 0.16),
              labelspacing=0.7, handletextpad=0.7, borderpad=0.15)

    resp_n, tot = len(resp), pr["d_converted_gw"].clip(lower=0).sum()
    share = 100 * pr.loc[pr["constrained"].fillna(False), "d_converted_gw"].clip(lower=0).sum() / tot
    ax.set_title(f"{len(pr)} 个厂址中只有 {resp_n} 个有响应，且新增的 +{tot:.0f} GW 中\n"
                 f"有 {share:.0f}% 落在四个超配额流域",
                 fontsize=7.0, linespacing=1.3)


def panel_concentration(ax, pr: pd.DataFrame) -> None:
    global _NET_GW
    _NET_GW = float(pr["d_converted_gw"].sum())
    """How concentrated the response is, drawn as a cumulative share against site rank."""
    d = np.sort(pr["d_converted_gw"].clip(lower=0).to_numpy())[::-1]
    cum = 100 * np.cumsum(d) / d.sum()
    rank = np.arange(1, len(d) + 1)
    ax.plot(rank, cum, color="#CC3311", lw=1.4, zorder=4)
    ax.fill_between(rank, 0, cum, color="#CC3311", alpha=0.10, lw=0, zorder=2)
    for k, colour in ((10, "#333333"), (25, "#333333"), (50, "#333333")):
        ax.plot([k, k], [0, cum[k - 1]], lw=0.6, ls=(0, (2, 2)), color=colour, zorder=3)
        ax.plot([0, k], [cum[k - 1]] * 2, lw=0.6, ls=(0, (2, 2)), color=colour, zorder=3)
        ax.annotate(f"{cum[k - 1]:.0f}%", xy=(k, cum[k - 1]), xytext=(k + 8, cum[k - 1] - 7),
                    fontsize=5.4, color="#333333")
    n_resp = int((d > 0.01).sum())
    ax.axvline(n_resp, color="#666666", lw=0.8)
    ax.annotate(f"全部 {n_resp} 个有响应厂址", xy=(n_resp, 55), xytext=(n_resp + 12, 45),
                fontsize=5.4, color="#666666",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666"))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 103)
    ax.set_xlabel("厂址，按自身响应量排序", fontsize=6.4)
    # NET, not the positive-clipped sum. clip(lower=0) totals 228.79 and drops one site at
    # -1.14 GW; the net treatment-minus-control conversion is 227.65 GW, which is the number
    # the ledger carries and the docstring subtracts (354.71 - 127.06).
    ax.set_ylabel(f"占全国 +{_NET_GW:.0f} GW 的\n累计比例（%）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # 数就在同一根轴上（上面的 cum[k-1] 注记），标题不能写另一个。
    ax.set_title(f"10 个厂址承担了全国\n响应量的 {cum[9]:.0f}%",
                 fontsize=6.8, linespacing=1.25)


def panel_discriminator(ax, pr: pd.DataFrame) -> None:
    """What separates a responder from a non-responder: cooling technology, not size or age."""
    resp = pr["d_converted_gw"] > 0.01
    groups = [("有响应\n(n = %d)" % resp.sum(), pr[resp], "#CC3311"),
              ("无响应\n(n = %d)" % (~resp).sum(), pr[~resp], QUIET)]
    # ONE METRIC. A second violin for once-through share sat at 0.00 in BOTH groups, so it
    # showed no contrast while taking half the panel and colliding with panel (b) title.
    metrics = [("share_recirculating", "湿冷塔（循环冷却）\n占厂址容量的份额", 1.0)]
    width = 0.42
    for mi, (col, label, _) in enumerate(metrics):
        for gi, (name, sub, colour) in enumerate(groups):
            v = sub[col].to_numpy(dtype=float)
            x = mi + (gi - 0.5) * width * 1.35
            parts = ax.violinplot([v], positions=[x], widths=width, showextrema=False,
                                  showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(colour)
                body.set_alpha(0.55)
                body.set_edgecolor("white")
                body.set_linewidth(0.4)
            ax.plot([x - width * 0.42, x + width * 0.42], [np.median(v)] * 2,
                    color=colour, lw=1.6, zorder=5, solid_capstyle="butt")
            ax.annotate(f"{np.median(v):.2f}", xy=(x, np.median(v)),
                        xytext=(x, np.median(v) + 0.07), fontsize=5.2, color=colour,
                        ha="center", fontweight="bold")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=5.8, linespacing=1.15)
    ax.set_ylim(-0.08, 1.16)
    ax.set_ylabel("占厂址容量的比例", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="y", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=g[2], lw=3, alpha=0.6, label=g[0].replace("\n", " "))
                       for g in groups],
              fontsize=5.4, frameon=False, loc="lower center", ncol=1,
              bbox_to_anchor=(0.5, -0.40), handlelength=1.1, handletextpad=0.5,
              columnspacing=1.0, labelspacing=0.25)
    ax.set_title("可被改造的是\n湿冷塔机组", fontsize=6.8, linespacing=1.25)


def main() -> None:
    pr = EP.pair(CONTROL, TREATMENT, YEAR)
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    ax_map = fig.add_axes([0.010, 0.050, 0.500, 0.885])
    ax_con = fig.add_axes([0.590, 0.560, 0.225, 0.300])
    ax_dis = fig.add_axes([0.890, 0.560, 0.095, 0.300])
    ax_bas = fig.add_axes([0.590, 0.090, 0.395, 0.330])

    panel_map(ax_map, pr)
    panel_concentration(ax_con, pr)
    panel_discriminator(ax_dis, pr)
    panel_basin(ax_bas, pr)

    for ax, lab, x, y in ((ax_map, "a", 0.005, 0.985), (ax_con, "b", -0.26, 1.32),
                          (ax_dis, "c", -0.66, 1.32), (ax_bas, "d", -0.14, 1.12)):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=8.0, fontweight="bold",
                va="top", ha="left")

    save_fig(fig, "ed_fig2_who_converts", subdir="extended")

    d = pr["d_converted_gw"].clip(lower=0)
    order = np.sort(d.to_numpy())[::-1]
    cum = np.cumsum(order) / order.sum()
    print(f"  national response {d.sum():+.1f} GW over {int((d > 0.01).sum())} of {len(pr)} sites")
    for k in (10, 25, 50):
        print(f"    top {k:3d} sites: {100 * cum[k - 1]:5.1f}%")
    print(f"  constrained-basin share {100 * d[pr['constrained'].fillna(False)].sum() / d.sum():.1f}%")


def panel_basin(ax, pr: pd.DataFrame) -> None:
    """Where the response lands, and what it is worth as a share of each basin's own fleet."""
    g = (pr.assign(resp=pr["d_converted_gw"].clip(lower=0))
         .groupby(["basin_code", "basin_name"])
         .agg(resp=("resp", "sum"), cap=("capacity_mw", "sum")).reset_index())
    g = g[g["resp"] > 0.05].sort_values("resp")
    g["cap_gw"] = g["cap"] / 1e3
    y = np.arange(len(g))
    colours = [BASIN_COLOUR.get(c, "#6B7683") for c in g["basin_code"]]
    ax.barh(y, g["resp"], height=0.62, color=colours, edgecolor="white", lw=0.3, zorder=3)
    for yi, r in zip(y, g.itertuples()):
        ax.annotate(f"{r.resp:.0f} GW（占该流域机组的 {100 * r.resp / r.cap_gw:.0f}%）",
                    xy=(r.resp, yi), xytext=(r.resp + 1.5, yi), fontsize=5.4,
                    va="center", color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([BASIN_NAMES_ZH.get(r.basin_code, r.basin_name) for r in g.itertuples()], fontsize=5.8)
    for tick, code in zip(ax.get_yticklabels(), g["basin_code"]):
        if code in BASIN_COLOUR:
            tick.set_color(BASIN_COLOUR[code])
            tick.set_fontweight("bold")
    ax.set_xlim(0, g["resp"].max() * 1.62)
    ax.set_xlabel("2030 年新增空冷改造（GW）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    # 领头流域与它的份额都从图里的同一张表读，不写死：v9 时是海河，v9.1 换成官方指标口径后
    # 响应移到黄河与西北内陆河，写死的流域名会直接与它下面的条形矛盾。
    _lead = g.sort_values("resp", ascending=False).iloc[0]
    _share = 100.0 * float(_lead["resp"]) / float(g["resp"].sum())
    _name = BASIN_NAMES_ZH.get(_lead["basin_code"], _lead["basin_name"])
    _top2 = g.sort_values("resp", ascending=False).head(2)
    _share2 = 100.0 * float(_top2["resp"].sum()) / float(g["resp"].sum())
    if _share >= 55.0:
        _title = f"仅{_name}一个流域就吸收了 {_share:.0f}%"
    else:
        _names = "、".join(BASIN_NAMES_ZH.get(r.basin_code, r.basin_name)
                          for r in _top2.itertuples())
        _title = f"{_names}两个流域吸收了 {_share2:.0f}%"
    ax.set_title(_title, fontsize=6.8, linespacing=1.25)


if __name__ == "__main__":
    main()
