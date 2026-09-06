"""Extended Data: the whole argument, in the one basin where it is sharpest.

WHY THE HAI. The paper's strongest and most fragile claim lives in a single level-1 basin. Across
the 20-member hydrology ensemble the critical reservation share s* for full capture in the Hai is
NEGATIVE in 13 members at 2030 -- the basin breaches its environmental-flow allowance even if power
is given the entire extractable share. In every other basin, and for today's fleet everywhere, the
overshoot is an allocation outcome. So the Hai is where capture stops tightening an allocation
constraint and starts creating a scarcity constraint.

At site level the ledger closes, in 1e8 m3 yr-1 of dry-season water, at 2030:

    Hai dry-season budget (median of 20 members)          78.8
      x 0.20 extractable (Richter et al. 2012)            15.76
      x (1 - 0.85) non-power reservation                   2.36   <- what power is actually offered
    demand, fleet as built                                 7.89   <- 3.3x the offer
    demand, capture on every unit                         16.08   <- above the 15.76 ceiling, by 2%
    realised demand after the fleet adapts                 2.52   <- within 7% of the offer

The adaptation that closes a 3.3x gap is concrete and local: 21 of the basin's 32 sites convert,
88.3 GW of condensers move from wet to dry, and basin water use falls 68%. That is 58% of the
Hai's entire 153.1 GW fleet, spread over 411 generating units.

The figure is drawn at site level because the basin aggregate hides which plants carry it, and
because "the Hai breaches" is a statement about 32 identifiable places rather than about a polygon.
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
from plot_style import MM, save_fig, PROV_ZH

CONTROL, TREAT, YEAR = "WA_cwatm_126_dry", "WA_cwatm_126_dry_wd085", 2030
EXTRACTABLE, RESERVATION = 0.20, 0.85
HAI_C = "#CC3311"
QUIET = "#9AA5B1"
SUPPLY_C = "#4477AA"


def hai_frame() -> pd.DataFrame:
    pb = EP.plant_basin()
    hai = set(pb.loc[pb["basin_code"] == "C", "plant_id"])
    pr = EP.pair(CONTROL, TREAT, YEAR)
    return pr[pr["plant_id"].isin(hai)].copy()


def basin_budget() -> float:
    """Median dry-season budget over the 20-member ensemble, 1e8 m3/yr."""
    nodes = pd.read_csv(EP.INPUTS / "water_nodes.csv")
    nodes = nodes[nodes["basin_code"] == "C"]
    av = pd.read_csv(EP.INPUTS / "water_availability.csv")
    av = av[(av["planning_year"] == YEAR) & (av["water_node_id"].isin(nodes["water_node_id"]))]
    return float(av.groupby("scenario_id")["dry_season_water_m3_per_year"].sum().median()) / 1e8


def panel_map(ax, h: pd.DataFrame) -> None:
    from plot_extended import _load_provinces
    import geopandas as gpd
    from shapely.geometry import Point

    provinces = _load_provinces()
    nodes = pd.read_csv(EP.INPUTS / "water_nodes.csv")
    nodes = nodes[nodes["basin_code"] == "C"]

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
    ax.scatter(pts.geometry.x[conv], pts.geometry.y[conv], s=size[conv], c=HAI_C,
               lw=0.5, edgecolor="white", alpha=0.92, zorder=5)

    pad = 1.4e5
    ax.set_xlim(nd.geometry.x.min() - pad, nd.geometry.x.max() + pad)
    ax.set_ylim(nd.geometry.y.min() - pad, nd.geometry.y.max() + pad)
    ax.set_axis_off()

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.plot([x0 + 0.06 * (x1 - x0), x0 + 0.06 * (x1 - x0) + 100_000],
            [y0 + 0.06 * (y1 - y0)] * 2, color="#333333", lw=1.1, zorder=20,
            solid_capstyle="butt")
    ax.text(x0 + 0.06 * (x1 - x0) + 50_000, y0 + 0.06 * (y1 - y0) - 0.018 * (y1 - y0),
            "100 km", ha="center", va="top", fontsize=5.2, color="#333333", zorder=20)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.4, mfc=HAI_C, mec="white", mew=0.4,
               label=f"在约束下改造（{int(conv.sum())} 个厂址）"),
        Line2D([], [], marker="o", ls="none", ms=4.4, mfc="white", mec=QUIET, mew=0.8,
               label=f"未改造（{int((~conv).sum())} 个）"),
        Line2D([], [], marker="o", ls="none", ms=2.2, mfc=SUPPLY_C, mec="none", alpha=0.5,
               label="供水节点")],
        fontsize=5.3, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02),
        handletextpad=0.6, labelspacing=0.38)

    # State BOTH numbers, because they are different and the short form conflated them: the
    # responding sites hold 108 GW of capacity, but the condenser conversion they add is 88 GW.
    ax.set_title(f"海河流域：{len(h)} 个厂址，{h['unit_count'].sum():,.0f} 台机组，"
                 f"{h['capacity_mw'].sum() / 1e3:.0f} GW\n"
                 f"{int(conv.sum())} 个厂址、共 "
                 f"{h.loc[conv, 'capacity_mw'].sum() / 1e3:.0f} GW 参与改造，"
                 f"合计改造 {h['d_converted_gw'].sum():.0f} GW 冷凝设备",
                 fontsize=6.9, linespacing=1.3, x=0.52)


def panel_ledger(ax, h: pd.DataFrame, budget: float) -> None:
    """Supply offered against demand, before and after the fleet adapts."""
    offer_ceiling = budget * EXTRACTABLE
    offer = offer_ceiling * (1 - RESERVATION)
    as_built = float(h["water_use_m3_c"].sum()) / 1e8
    realised = float(h["water_use_m3_t"].sum()) / 1e8
    # READ IT, AND PUT IT ON THIS PANEL'S FLEET. This was hardcoded as 17.03, which is the
    # value in the s* ensemble -- but that record is computed over a 162.18 GW Hai fleet
    # (solver flow weights), while the four measured bars here are the 32 nearest-node sites
    # holding 153.13 GW. Mixing the two overstated the margin over the ceiling: rescaled to
    # this panel's fleet the full-capture demand is 16.08, so it clears the 15.76 ceiling by
    # 2.0% rather than 8.1%. The claim survives; the comfort does not.
    import json as _json
    _rows = _json.loads((EP.RESULTS / "figures" / "v6" / "v6_s_star_ensemble.json")
                        .read_text(encoding="utf-8"))
    _rec = next(r for r in _rows if r["year"] == YEAR and r["basin"] == "C"
                and r["state"] == "full_capture")
    panel_fleet_gw = float(h["capacity_mw"].sum()) / 1e3
    full_capture = float(_rec["demand_1e8"]) * panel_fleet_gw / float(_rec["gw"])

    rows = [("枯水期水量 x 0.20\n（生态流量上限）", offer_ceiling, SUPPLY_C),
            ("扣除 0.85 非电力预留后\n供给电力的配额", offer, SUPPLY_C),
            ("需求：全部机组捕集", full_capture, "#882255"),
            ("需求：机组按现状配置", as_built, QUIET),
            ("实际取水：88 GW 完成改造后", realised, HAI_C)]
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, colour) in zip(y, rows):
        ax.barh(yi, v, height=0.58, color=colour, edgecolor="white", lw=0.4,
                alpha=0.55 if colour == SUPPLY_C else 0.92, zorder=3)
        ax.annotate(f"{v:.2f}", xy=(v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=5.6, fontweight="bold", color=colour)
    ax.axvline(offer, color=SUPPLY_C, lw=1.0, ls=(0, (3, 2)), zorder=6)
    ax.axvline(offer_ceiling, color=SUPPLY_C, lw=0.8, ls=(0, (1, 2)), zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6, linespacing=1.2)
    ax.set_xlim(0, max(full_capture, offer_ceiling) * 1.16)
    ax.set_xlabel("枯水期水量，2030 年（$10^8$ m$^3$ yr$^{-1}$）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    # 两行、6.4 pt：三行的版本会顶到面板 c 的标号与标题。
    ax.set_title(f"全量捕集需求（{full_capture:.1f}）仅超出上限（{offer_ceiling:.1f}）"
                 f"{100 * (full_capture / offer_ceiling - 1):.0f}%，\n"
                 f"适应性改造把 {as_built / offer:.1f} 倍的缺口收敛到 {realised / offer:.2f} 倍",
                 fontsize=6.4, linespacing=1.25)


def panel_sites(ax, h: pd.DataFrame) -> None:
    """Which sites give up the water, as a before/after dumbbell."""
    d = h.copy()
    d["c"] = d["water_use_m3_c"] / 1e8
    d["t"] = d["water_use_m3_t"] / 1e8
    d = d.sort_values("c", ascending=False).head(14).iloc[::-1]
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples()):
        ax.plot([r.t, r.c], [yi, yi], color=QUIET, lw=1.0, zorder=2, solid_capstyle="round")
        ax.plot([r.c], [yi], marker="o", ms=3.6, color=QUIET, mec="white", mew=0.5, zorder=4)
        ax.plot([r.t], [yi], marker="o", ms=3.6, color=HAI_C, mec="white", mew=0.5, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{PROV_ZH.get(r.province_name, r.province_name)}  {r.capacity_mw / 1e3:.1f} GW"
                        for r in d.itertuples()], fontsize=5.2)
    ax.set_xlabel("厂址枯水期用水量（$10^8$ m$^3$ yr$^{-1}$）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.6, mfc=QUIET, mec="white",
                              label="无预留（对照）"),
                       Line2D([], [], marker="o", ls="none", ms=3.6, mfc=HAI_C, mec="white",
                              label="预留约束起作用")],
              fontsize=5.3, frameon=False, loc="lower right", handletextpad=0.5,
              labelspacing=0.3)
    cut = 100 * (1 - float(h["water_use_m3_t"].sum()) / float(h["water_use_m3_c"].sum()))
    ax.set_title(f"流域用水量下降 {cut:.0f}%，且让出水量的\n主要是最大的那些厂址",
                 fontsize=6.8, linespacing=1.25)


def main() -> None:
    h = hai_frame()
    budget = basin_budget()
    fig = plt.figure(figsize=(183 * MM, 92 * MM))
    # 右列从 0.815+0.170=0.985 收到 0.800+0.160=0.960：面板 c 的数值注记画在轴外，
    # bbox_inches="tight" 会把画布撑到把它们包进来，整幅因此定格在 184.7 mm，
    # 越过 183 mm 双栏宽 1.7 mm。收窄右列比缩字号便宜，5 pt 的下限不能再动。
    ax_map = fig.add_axes([0.005, 0.055, 0.330, 0.815])
    ax_led = fig.add_axes([0.450, 0.190, 0.245, 0.610])
    ax_sit = fig.add_axes([0.800, 0.190, 0.160, 0.610])

    panel_map(ax_map, h)
    panel_ledger(ax_led, h, budget)
    panel_sites(ax_sit, h)

    # c 的标号原在 -1.00，落在 b 的标题正上方；-0.42 正好在 c 的省名列上方、b 的标题右端之外。
    for ax, lab, dx, dy in ((ax_map, "a", 0.005, 1.00), (ax_led, "b", -0.60, 1.135),
                            (ax_sit, "c", -0.42, 1.135)):
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=8.0, fontweight="bold",
                va="top", ha="left")

    save_fig(fig, "ed_fig6_hai_closeup", subdir="extended")

    conv = h["d_converted_gw"] > 0.01
    print(f"  Hai: {len(h)} sites, {h['unit_count'].sum():.0f} units, "
          f"{h['capacity_mw'].sum() / 1e3:.1f} GW")
    print(f"  converts: {int(conv.sum())} sites, +{h['d_converted_gw'].sum():.1f} GW")
    print(f"  budget {budget:.1f} -> ceiling {budget * EXTRACTABLE:.2f} -> offered "
          f"{budget * EXTRACTABLE * (1 - RESERVATION):.2f} (1e8 m3)")
    print(f"  demand as built {h['water_use_m3_c'].sum() / 1e8:.2f} -> realised "
          f"{h['water_use_m3_t'].sum() / 1e8:.2f}  "
          f"({100 * (1 - h['water_use_m3_t'].sum() / h['water_use_m3_c'].sum()):.0f}% cut)")


if __name__ == "__main__":
    main()
