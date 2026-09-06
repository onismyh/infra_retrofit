"""Extended Data: the coal fleet at unit resolution, and where the water constraint reaches it.

WHAT THIS REPLACES, AND WHY. The v6 appendix showed the fleet as a provincial choropleth of mean
design retirement year (`plot_extended.ed_fig5_fleet_age_map`). That figure had two defects and
one structural problem.

  * It FABRICATED DATA. Provinces with no coal fleet were filled with a literal 2040
    (`plot_extended.py:861`, `.fillna(2040)`) and then drawn on the same 2035-2060 colour scale
    as measured provinces. 2040 renders orange, so Tibet and Taiwan -- which hold no capacity in
    this model at all; only 28 provinces do -- carried the most conspicuous colour on the map.
  * A choropleth encodes by AREA, and area is close to inversely related to where Chinese coal
    capacity is. Xinjiang and Inner Mongolia dominate the visual field; Shandong and Jiangsu,
    which hold 123 and 112 GW, are almost invisible.
  * Province is the wrong unit for this paper. The constraint is a BASIN constraint, provinces
    straddle basin divides, and a provincial mean averages across the exact boundary the study
    is about.

This figure is drawn on the 350 sites and 3,623 units themselves. Capacity is encoded as marker
area, so the eye weights the fleet the way the model does, and the four basins whose dry-season
allowance the standing fleet already exceeds are the only ones given colour.
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
from matplotlib.patches import Patch

import ed_plant_data as EP
from plot_style import BASIN_NAMES_ZH, MM, save_fig

# One colour per constrained basin, colour-blind safe (Tol bright). Everything unconstrained is
# grey: the figure's job is to show where the constraint lands, not to name nine basins.
BASIN_COLOUR = {"C": "#CC3311", "D": "#EE7733", "E": "#CCBB44", "K": "#AA3377"}
UNCONSTRAINED = "#9AA5B1"
COOLING_COLOUR = {"once_through": "#4477AA", "recirculating": "#88CCEE", "air": "#DDAA33"}
COOLING_LABEL = {"once_through": "直流冷却", "recirculating": "循环冷却",
                 "air": "空冷（干冷）"}



from pathlib import Path as _SP
SLIDE_DIR = _SP(__file__).resolve().parent.parent / "results" / "figures" / "slides"
SLIDE_DIR.mkdir(parents=True, exist_ok=True)
BASIN_SHORT = {"A": "松辽", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
               "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北内陆"}


def save_fig(fig, name, subdir=""):  # noqa: F811  幻灯版：直接落到 slides/
    p = SLIDE_DIR / f"slide_{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"[slide] -> {p}")

def _sized(cap_gw: np.ndarray) -> np.ndarray:
    """Marker AREA proportional to capacity, so a 4 GW site reads as 4x a 1 GW site.

    Encoding capacity on the radius instead -- the default temptation -- understates large sites
    by the square root and is the most common way a plant map misleads.
    """
    return 2.2 + 26.0 * cap_gw / np.nanmax(cap_gw)


def panel_map(ax, fleet: pd.DataFrame) -> None:
    import geopandas as gpd
    from plot_extended import _load_provinces, _reproj, _add_map_elements
    from plot_style import draw_china_basemap, add_scs_inset, mainland_extent

    # 省界 + 国界（国界图层含九段线，见 plot_style.load_country 的断言）
    provinces, _country = draw_china_basemap(ax, facecolor="#F7F8F9")

    pts = _reproj(fleet.dropna(subset=["centroid_longitude", "centroid_latitude"]),
                  "centroid_longitude", "centroid_latitude")
    pts = pts.copy()
    pts["_size"] = _sized(pts["capacity_gw"].to_numpy(dtype=float))

    other = pts[~pts["constrained"].fillna(False)]
    ax.scatter(other.geometry.x, other.geometry.y, s=other["_size"], c=UNCONSTRAINED,
               lw=0.15, edgecolor="white", alpha=0.75, zorder=3)
    for code, colour in BASIN_COLOUR.items():
        sub = pts[pts["basin_code"] == code]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=sub["_size"], c=colour,
                   lw=0.2, edgecolor="white", alpha=0.92, zorder=4)

    # FRAME ON THE DATA, NOT ON A FIXED LON/LAT BOX. `plot_extended._map_bounds` spans
    # lon 80-150, which is far wider than China and leaves the mainland floating small in a
    # field of empty ocean. Framing on the province geometry roughly doubles the drawn map.
    # provinces.total_bounds 南端到 6.32N（含南海要素），会把大陆压扁；裁到 17N
    x0, y0, x1, y1 = mainland_extent(ax)
    ax.set_axis_off()
    # 南海小图：九段线南端到 3.85N，被上面的 ylim 裁掉，所以小图是必需的而非装饰。
    # 这张图的散点全在陆上，小图只需底图。
    add_scs_inset(ax.get_figure(), ax)

    # Own scale bar and north arrow. The shared helper places both at fixed axes fractions
    # sized for a full-page single-panel map, where they collided with this panel's legend.
    sx = x0 + 0.06 * (x1 - x0)
    sy = y0 + 0.13 * (y1 - y0)
    ax.plot([sx, sx + 500_000], [sy, sy], color="#333333", lw=1.1, zorder=20,
            solid_capstyle="butt")
    ax.text(sx + 250_000, sy - 0.018 * (y1 - y0), "500 km", ha="center", va="top",
            fontsize=5.2, color="#333333", zorder=20)
    # Grouped with the scale bar rather than pinned to the top-right corner: at the corner it
    # fell outside the drawn landmass and read as furniture belonging to the neighbouring panel.
    ax.annotate("N", xy=(0.052, 0.300), xycoords="axes fraction", fontsize=6.2,
                fontweight="bold", ha="center", va="bottom", color="#333333")
    ax.annotate("", xy=(0.052, 0.296), xycoords="axes fraction",
                xytext=(0.052, 0.235), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))

    cap = fleet["capacity_gw"]
    ax.legend(handles=[Line2D([], [], marker="o", ls="none",
                              ms=np.sqrt(2.2 + 26.0 * v / cap.max()),
                              mfc="#5A6570", mec="white", mew=0.3, label=f"{v:g} GW")
                       for v in (1, 5, 10)],
              title="厂址容量", fontsize=5.2, title_fontsize=5.4, frameon=False,
              loc="lower left", bbox_to_anchor=(0.005, 0.17), labelspacing=0.7,
              handletextpad=0.7, borderpad=0.15)

    con = fleet[fleet["constrained"].fillna(False)]
    ax.set_title(f"{len(fleet)} 个厂址，{fleet['unit_count'].sum():,.0f} 台机组，"
                 f"{cap.sum():,.0f} GW\n"
                 f"其中 {con['capacity_gw'].sum():,.0f} GW（{100 * con['capacity_gw'].sum() / cap.sum():.0f}%）"
                 f"位于四个已超配额的流域",
                 fontsize=7.0, linespacing=1.3)


def panel_cooling(ax, fleet: pd.DataFrame) -> None:
    """Capacity by basin, split by the cooling technology actually installed."""
    order = (fleet.groupby(["basin_code", "basin_name"])["capacity_gw"].sum()
             .sort_values().reset_index())
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for tech in ("once_through", "recirculating", "air"):
        vals = np.array([fleet.loc[fleet["basin_code"] == b,
                                   f"capacity_mw_{tech}"].sum() / 1e3
                         for b in order["basin_code"]])
        ax.barh(y, vals, left=left, height=0.66, color=COOLING_COLOUR[tech],
                edgecolor="white", lw=0.3, zorder=3,
                label=COOLING_LABEL[tech] if len(left) else None)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([BASIN_NAMES_ZH.get(r.basin_code, r.basin_name) for r in order.itertuples()],
                       fontsize=5.8)
    for tick, code in zip(ax.get_yticklabels(), order["basin_code"]):
        if code in BASIN_COLOUR:
            tick.set_color(BASIN_COLOUR[code])
            tick.set_fontweight("bold")
    ax.set_xlabel("装机容量（GW）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=5.4, frameon=False, loc="lower right", handlelength=1.1,
              handletextpad=0.5, labelspacing=0.25, borderpad=0.2)
    air = fleet["capacity_mw_air"].sum() / 1e3
    ax.set_title(f"已有 {air:.0f} GW 采用空冷，且集中\n"
                 f"在缺水的那几个流域",
                 fontsize=6.8, linespacing=1.25)


def panel_age(ax, fleet: pd.DataFrame) -> None:
    """Capacity by commission year — the fleet is young, which is why retirement is expensive."""
    bins = np.arange(1994, 2030, 2)
    # STACKED, NOT OVERLAID. Two translucent histograms on one axis do not read as a total and
    # the lower one is hidden behind the upper; the first draft of this panel showed the grey
    # series only where the red happened to be short. Stacking makes both the split and the
    # fleet-wide total readable from the same bars.
    con = fleet[fleet["constrained"].fillna(False)]
    oth = fleet[~fleet["constrained"].fillna(False)]
    ax.hist([con["mean_commission_year"], oth["mean_commission_year"]], bins=bins,
            weights=[con["capacity_gw"], oth["capacity_gw"]], stacked=True,
            color=["#CC3311", UNCONSTRAINED], lw=0.3, edgecolor="white", zorder=3,
            label=["四个受约束流域", "全国其余地区"])
    med = np.average(fleet["mean_commission_year"], weights=fleet["capacity_gw"])
    ax.axvline(med, color="#333333", lw=0.9, ls=(0, (3, 2)), zorder=5)
    ax.annotate(f"容量加权平均\n投产年 {med:.0f}",
                xy=(med, ax.get_ylim()[1] * 0.92), xytext=(med - 15, ax.get_ylim()[1] * 0.72),
                fontsize=5.4, color="#333333", linespacing=1.2,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#333333"))
    ax.set_xlabel("厂址内机组的平均投产年份", fontsize=6.4)
    ax.set_ylabel("容量（GW）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="y", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.1,
              handletextpad=0.5, labelspacing=0.25)
    ax.set_title("厂址内机组平均投产年份的分布\n"
                 "（容量加权，按是否位于超配额流域区分）", fontsize=6.8, linespacing=1.25)


def main() -> None:
    fleet = EP.fleet()
    fig = plt.figure(figsize=(112 * MM, 64 * MM))
    ax_map = fig.add_axes([0.010, 0.040, 0.560, 0.900])
    ax_age = fig.add_axes([0.660, 0.170, 0.325, 0.680])
    panel_map(ax_map, fleet)
    panel_age(ax_age, fleet)
    for ax, lab, x, y in ((ax_map, "a", 0.005, 0.985), (ax_age, "b", -0.200, 1.12)):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=8.0, fontweight="bold",
                va="top", ha="left")
    save_fig(fig, "fleet_atlas", subdir="extended")
if __name__ == "__main__":
    main()
