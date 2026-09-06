"""Extended Data: the bioenergy that pays for the water adaptation is grown where the water is short.

WHY THIS FIGURE EXISTS. The paper's mechanism is a chain: making the water reservation bind pushes
the fleet to convert condensers to dry cooling; dry cooling costs about 1.5 points of thermal
efficiency; a non-negative-emissions constraint then forces biomass co-firing to cancel the
resulting emissions rise. Biomass is therefore the PAYMENT for the water adaptation.

And biomass carries no feedstock water in this model (`biomass_water_multiplier = 1.00`): co-firing
draws the host plant's cooling water and nothing else, with no cultivation, harvest or processing
demand. That is defensible condenser accounting and indefensible to leave unexamined once the
mechanism runs through it, because the size of the exposure depends entirely on WHERE the fuel is
grown. If it comes from somewhere else, the loop closes. If it comes from the same stressed basins,
the adaptation creates a new water demand in exactly the place it was trying to relieve.

Measured from the solved origin-destination flows, it is the second:

    year   biomass drawn   sourced inside the burning basin   flow-weighted haul
    2030        0.65 EJ                 100.0 %                      29 km
    2040       23.77 EJ                  90.9 %                      65 km
    2050       26.18 EJ                  88.9 %                      76 km
    2060       25.61 EJ                  88.8 %                      74 km

Hai River plants draw 2.84 EJ at 2050 and take 80.7 % of it from inside the Hai. The supply radius
binds at 200 km, so this is structural rather than incidental: the model cannot haul biomass out of
a wet basin into a dry one even if it wanted to.

The national resource utilisation that goes with it -- 2.2 % at 2030, then 79.1 %, 87.1 %, 85.2 % --
is drawn on panel (b) because a mechanism running at 87 % of its resource ceiling is one increment
of feedstock water away from not working at all.
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import ed_plant_data as EP
from plot_style import MM, save_fig, BASIN_NAMES_ZH

SCENARIO = "WA_cwatm_126_dry_wd085"
MAP_YEAR = 2050
YEARS = (2030, 2040, 2050, 2060)
CONSTRAINED_C = "#CC3311"
QUIET = "#9AA5B1"
BIO = "#117733"



from pathlib import Path as _SP
SLIDE_DIR = _SP(__file__).resolve().parent.parent / "results" / "figures" / "slides"
SLIDE_DIR.mkdir(parents=True, exist_ok=True)
BASIN_SHORT = {"A": "松辽", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
               "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北内陆"}


def save_fig(fig, name, subdir=""):  # noqa: F811  幻灯版：直接落到 slides/
    p = SLIDE_DIR / f"slide_{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"[slide] -> {p}")

def flows() -> pd.DataFrame:
    """Origin-destination biomass flows, with a basin on each end.

    Source nodes are assigned to a basin by nearest level-1 water node -- the same rule used for
    plants, so "same basin" means the same thing at both ends of a flow.
    """
    from scipy.spatial import cKDTree

    nodes = pd.read_csv(EP.INPUTS / "water_nodes.csv")
    tree = cKDTree(nodes[["longitude", "latitude"]].to_numpy())
    f = pd.read_csv(EP.RESULTS / SCENARIO / "biomass_flows.csv")

    src = f.drop_duplicates("biomass_node_id")[
        ["biomass_node_id", "node_longitude", "node_latitude"]].copy()
    _, idx = tree.query(src[["node_longitude", "node_latitude"]].to_numpy())
    src["src_basin"] = nodes["basin_code"].to_numpy()[idx]

    pb = EP.plant_basin().set_index("plant_id")
    f = f.merge(src[["biomass_node_id", "src_basin"]], on="biomass_node_id", how="left")
    f["dst_basin"] = f["plant_id"].map(pb["basin_code"])
    f["same_basin"] = f["src_basin"] == f["dst_basin"]
    f["dst_constrained"] = f["dst_basin"].isin(EP.CONSTRAINED_BASINS)
    return f


def utilisation() -> pd.DataFrame:
    r = pd.read_csv(EP.RESULTS / SCENARIO / "resource_use.csv")
    r = r[r["resource_type"].astype(str).str.contains("bio", case=False, na=False)]
    g = r.groupby("year").agg(used=("used", "sum"), avail=("available", "sum"))
    g["util"] = 100 * g["used"] / g["avail"]
    return g


def panel_map(ax, f: pd.DataFrame) -> None:
    from plot_extended import _load_provinces
    from plot_style import draw_china_basemap, add_scs_inset, mainland_extent
    import geopandas as gpd
    from shapely.geometry import Point

    provinces, _country = draw_china_basemap(ax, facecolor="#F7F8F9")

    d = f[f["year"] == MAP_YEAR].copy()
    # Reproject both endpoints once, together, so the segments cannot drift apart.
    pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(d["node_longitude"], d["node_latitude"])]
                 + [Point(x, y) for x, y in zip(d["centroid_longitude"], d["centroid_latitude"])],
        crs="EPSG:4326").to_crs(provinces.crs)
    n = len(d)
    sx = pts.geometry.x.to_numpy()[:n]
    sy = pts.geometry.y.to_numpy()[:n]
    px = pts.geometry.x.to_numpy()[n:]
    py = pts.geometry.y.to_numpy()[n:]

    # 26,111 hairlines would be a smear. Draw the flows that carry the mass: the largest 3,000
    # here are 68% of the year's energy, and the omission is stated on the panel.
    order = np.argsort(d["flow_gj"].to_numpy())[::-1]
    keep = order[:3000]
    share = 100 * d["flow_gj"].to_numpy()[keep].sum() / d["flow_gj"].sum()
    segs = [[(sx[i], sy[i]), (px[i], py[i])] for i in keep]
    w = d["flow_gj"].to_numpy()[keep]
    lw = 0.12 + 0.9 * w / w.max()
    colour = np.where(d["dst_constrained"].to_numpy()[keep], CONSTRAINED_C, QUIET)
    ax.add_collection(LineCollection(segs, colors=colour, linewidths=lw, alpha=0.42, zorder=3))

    burn = d.groupby("plant_id").agg(gj=("flow_gj", "sum"), x=("centroid_longitude", "first"),
                                     y=("centroid_latitude", "first"),
                                     con=("dst_constrained", "first")).reset_index()
    bp = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(burn["x"], burn["y"])],
                          crs="EPSG:4326").to_crs(provinces.crs)
    ax.scatter(bp.geometry.x, bp.geometry.y,
               s=1.2 + 34.0 * burn["gj"] / burn["gj"].max(),
               c=np.where(burn["con"], CONSTRAINED_C, "#6B7683"),
               lw=0.25, edgecolor="white", alpha=0.95, zorder=5)

    x0, y0, x1, y1 = mainland_extent(ax)
    ax.set_axis_off()
    add_scs_inset(ax.get_figure(), ax)

    sxb, syb = x0 + 0.06 * (x1 - x0), y0 + 0.10 * (y1 - y0)
    ax.plot([sxb, sxb + 500_000], [syb, syb], color="#333333", lw=1.1, zorder=20,
            solid_capstyle="butt")
    ax.text(sxb + 250_000, syb - 0.018 * (y1 - y0), "500 km", ha="center", va="top",
            fontsize=5.2, color="#333333", zorder=20)

    ax.legend(handles=[Line2D([], [], color=CONSTRAINED_C, lw=1.4,
                              label="在超配额流域燃烧"),
                       Line2D([], [], color=QUIET, lw=1.4, label="在其他流域燃烧")],
              fontsize=5.3, frameon=False, loc="lower left", bbox_to_anchor=(0.005, 0.15),
              handlelength=1.6, handletextpad=0.6, labelspacing=0.35)

    self_pct = 100 * d.loc[d["same_basin"], "flow_gj"].sum() / d["flow_gj"].sum()
    haul = np.average(d["distance_km"], weights=d["flow_gj"])
    print(f"[stat] total={d['flow_gj'].sum() / 1e9:.2f} EJ self_pct={self_pct:.1f} haul={haul:.1f} km nkeep={len(keep)} share={share:.1f}")
    ax.set_title(f"{MAP_YEAR} 年生物质由产地到电厂的流向\n"
                 f"（图示最大的 {len(keep):,} 条流，占能量的 {share:.0f}%）",
                 fontsize=6.8, linespacing=1.3, x=0.54)


def panel_growth(ax, f: pd.DataFrame, util: pd.DataFrame) -> None:
    """How fast the draw grows, against the ceiling it is growing into."""
    ej = [f.loc[f["year"] == y, "flow_gj"].sum() / 1e9 for y in YEARS]
    x = np.arange(len(YEARS))
    ax.bar(x, ej, width=0.56, color=BIO, alpha=0.85, edgecolor="white", lw=0.4, zorder=3)
    for xi, v in zip(x, ej):
        ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=5.4, color=BIO, fontweight="bold",
                    zorder=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=5.8)
    ax.set_ylabel("生物质燃用量（EJ yr$^{-1}$）", fontsize=6.4, color=BIO)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.set_ylim(0, max(ej) * 1.30)
    for side in ("top",):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    u = [float(util.loc[y, "util"]) if y in util.index else np.nan for y in YEARS]
    ax2.plot(x, u, color="#AA3377", lw=1.4, marker="D", ms=3.6, mec="white", mew=0.5, zorder=6)
    for xi, v in zip(x, u):
        # Offsets alternate: at 2030 the bar is 0.6 EJ and the marker sits near the axis, so a
        # downward label collided with the tick; later years have room below the marker.
        ax2.annotate(f"{v:.0f}%", xy=(xi, v), xytext=(14 if xi == 0 else 0, 6 if xi == 0 else -12),
                     textcoords="offset points", ha="center", fontsize=5.4, color="#AA3377",
                     fontweight="bold")
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("占全国资源量的比例（%）", fontsize=6.4, color="#AA3377")
    ax2.tick_params(labelsize=5.6, length=1.8, colors="#AA3377")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#AA3377")
    ax.set_title("生物质燃用量及其占全国资源量的比例",
                 fontsize=6.8, linespacing=1.25)


def panel_self(ax, f: pd.DataFrame) -> pd.DataFrame:
    """Self-sourcing by destination basin: is the fuel local, and how local?"""
    d = f[f["year"] == MAP_YEAR]
    g = (d.groupby("dst_basin")
         .apply(lambda x: pd.Series({
             "ej": x["flow_gj"].sum() / 1e9,
             "self": 100 * x.loc[x["same_basin"], "flow_gj"].sum() / x["flow_gj"].sum()}),
             include_groups=False)
         .reset_index())
    g = g[g["ej"] > 0.05].sort_values("ej")
    g["constrained"] = g["dst_basin"].isin(EP.CONSTRAINED_BASINS)
    y = np.arange(len(g))
    colours = [CONSTRAINED_C if c else QUIET for c in g["constrained"]]
    ax.barh(y, g["self"], height=0.62, color=colours, edgecolor="white", lw=0.3, zorder=3)
    for yi, r in zip(y, g.itertuples()):
        ax.annotate(f"{r.self:.0f}%   ({r.ej:.1f} EJ)", xy=(r.self, yi), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=5.3, color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels([BASIN_NAMES_ZH.get(b, b) for b in g["dst_basin"]], fontsize=5.8)
    for tick, c in zip(ax.get_yticklabels(), g["constrained"]):
        if c:
            tick.set_color(CONSTRAINED_C)
            tick.set_fontweight("bold")
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"流域内自产生物质占其燃用量的比例（{MAP_YEAR} 年，%）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("各流域生物质的自给比例",
                 fontsize=6.8, linespacing=1.25)
    return g


def main() -> None:
    f = flows()
    util = utilisation()
    fig = plt.figure(figsize=(183 * MM, 124 * MM))
    ax_map = fig.add_axes([0.010, 0.050, 0.505, 0.880])
    ax_gro = fig.add_axes([0.605, 0.575, 0.310, 0.335])
    ax_self = fig.add_axes([0.640, 0.085, 0.300, 0.340])

    panel_map(ax_map, f)
    panel_growth(ax_gro, f, util)
    g = panel_self(ax_self, f)

    for ax, lab, dx, dy in ((ax_map, "a", 0.005, 0.985), (ax_gro, "b", -0.235, 1.24),
                            (ax_self, "c", -0.320, 1.22)):
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=8.0, fontweight="bold",
                va="top", ha="left")

    save_fig(fig, "ed_fig4_biomass_sourcing", subdir="extended")

    for y in YEARS:
        d = f[f["year"] == y]
        print(f"  {y}: {d['flow_gj'].sum() / 1e9:6.2f} EJ  self {100 * d.loc[d['same_basin'], 'flow_gj'].sum() / d['flow_gj'].sum():5.1f}%"
              f"  haul {np.average(d['distance_km'], weights=d['flow_gj']):5.1f} km"
              f"  utilisation {float(util.loc[y, 'util']):5.1f}%")


if __name__ == "__main__":
    main()
