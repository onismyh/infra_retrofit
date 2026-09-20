"""Fig 1 — the water price of abating a tonne of CO2, and where that water has to come from.

Three things this figure has to establish before the rest of the paper can be read:

  * three bases, not one. Consumption is what the basin loses and the ONLY basis the
    availability constraint acts on; the Chinese abstraction quota (取水定额) is what is
    metered and charged; withdrawal is everything diverted including the condenser flow
    returned downstream, and is diagnostic only, never constrained.
  * the hero metric is m3 per tonne CO2 abated, on the INCREMENTAL basis: the extra water
    a retrofit draws, divided by the extra abatement it buys.
  * the basin, not the province, is the unit -- the Yellow River crosses nine provinces.

  (a) intensity by cooling system on all three bases, with and without capture (log axis)
  (b) HERO -- incremental m3 per tonne abated, by pathway and cooling system
  (c) spatial coincidence: 2060 ensemble-median drying against the CCS water price

Usage:  python scripts/plot_fig1_water_footprint.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plot_style import (  # noqa: E402
    apply_style,
    load_country,
    load_map_provinces,
    COUNTRY_GBCODES,
    add_scs_inset,
    cjk_fill,
    safe_log_axis,
    save_fig,
    panel_label,
    panel_label_inside,
    ROOT,
    DOUBLE_COL,
    BASIN_NAMES_ZH,
    BASIN_ORDER,
    assign_basin,
    load_basins,
)
from coal_retrofit.constants import (  # noqa: E402
    WATER_INTENSITY_BY_TECH_M3_PER_MWH,
    CHINA_WATER_QUOTA_M3_PER_MWH,
    COMBUSTION_CLASS_MAP,
    WATER_EXTRACTABLE_FRACTION,
    quota_capacity_band,
)

def apply_fig1_thesis_style() -> None:
    """Apply the notebook's SimHei/ticks aesthetic at journal-figure scale."""
    apply_style()
    plt.rcParams.update({
        "font.family": ["SimHei"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


apply_fig1_thesis_style()

ANNOTATION_SIZE = 6.3
DETAIL_SIZE = 6.0
LEGEND_SIZE = 6.4

INPUTS = ROOT / "inputs"
TARGET_CRS = "EPSG:2380"  # matches plot_style.assign_basin and plot_spatial.py
HERO_YEAR = 2060  # the year the emission target binds, so the year CCS is actually built
# v9.1: there is no reservation SHARE any more. The non-power claim is not assumed, it is
# read off 国办发〔2013〕2号 附件1 + 2025年中国水资源公报 表9 in inputs/water_basin_caps.csv,
# and it lives on the WITHDRAWAL basis, not the consumption basis the environmental-flow
# rule uses. The two rungs are therefore two different numbers on two different bases, and
# this figure reports both rather than one aliased product (see docs/官方指标口径水预算.md).
BASIN_CAPS_CSV = "water_basin_caps.csv"
COOLINGS = ("once-through", "recirculating", "air")
COOLING_LABELS = ("直流冷却", "循环冷却", "空冷（干冷）")

# Colours held locally, not added to plot_style: three sibling figure scripts are editing
# that module concurrently.
BASIS_COLORS = {"withdrawal": "#999999", "quota": "#DDAA33", "consumption": "#004488"}
BASIS_LABELS = {
    "withdrawal": "取水量（诊断口径）",
    "quota": "中国取水定额（计价口径）",
    "consumption": "耗水量（约束口径）",
}
# Panel B needs four clearly separated categorical hues because all pathways share each
# cooling-system group. The palette is deliberately bright on white while preserving enough
# contrast for the thin connectors and the CCS value labels.
PATHWAY_PLOT_C = {
    "ccs": "#2F80ED",
    "beccs": "#27AE60",
    "biomass": "#F2994A",
    "ammonia": "#BB6BD9",
}
PATHWAY_L = {
    "ccs": "CCS 改造",
    "beccs": "BECCS（掺烧 50%）",
    "biomass": "生物质掺烧（50%）",
    "ammonia": "掺氨（20%）",
}
PATHWAY_SEQ = ("ccs", "beccs", "biomass", "ammonia")
BIOMASS_BLEND = 0.50
AMMONIA_BLEND = 0.20


# ── fleet-weighted intensities ───────────────────────────────────────────────
def cooling_intensities() -> pd.DataFrame:
    """Capacity-weighted intensity per cooling system, on all three bases.

    Weighted over the 3 623 units in `inputs/plants_unit.csv` by their own steam cycle, so
    the bars carry the fleet's real combustion mix rather than one representative row. The
    quota is looked up per unit by (cooling, size band) with `quota_capacity_band`.
    """
    units = pd.read_csv(INPUTS / "plants_unit.csv")
    combustion = (
        units["combustion"].astype(str).str.lower().str.replace("/ccs", "", regex=False)
        .map(COMBUSTION_CLASS_MAP)
    )
    cooling = units["cooling_technology"].astype(str).str.lower()
    rows = []
    for cool in COOLINGS:
        mask = (cooling == cool) & combustion.notna()
        capacity = units.loc[mask, "capacity_mw"].astype(float).to_numpy()
        classes = combustion[mask].to_numpy()
        total = capacity.sum()
        entry = {"cooling": cool, "capacity_gw": total / 1000.0}
        for key in ("withdrawal", "consumption", "withdrawal_ccs", "consumption_ccs"):
            entry[key] = float(
                sum(w * WATER_INTENSITY_BY_TECH_M3_PER_MWH[(c, cool)][key]
                    for w, c in zip(capacity, classes)) / total
            )
        bands = [quota_capacity_band(c) for c in capacity]
        entry["quota"] = float(
            sum(w * CHINA_WATER_QUOTA_M3_PER_MWH[(cool, b)] for w, b in zip(capacity, bands)) / total
        )
        # The quota has no published with-capture value. The capture increment is a real
        # abstraction and is metered, so it is added at the consumption increment -- the same
        # approximation `data_prep.py:156-163` makes when it prices the tariff.
        entry["quota_ccs"] = entry["quota"] + max(0.0, entry["consumption_ccs"] - entry["consumption"])
        rows.append(entry)
    return pd.DataFrame(rows).set_index("cooling")


def abatement_t_per_mwh(assumptions, scenario) -> dict[str, float]:
    """Net abatement per MWh by pathway, reproducing the solver's own accounting.

    Mirrors `plot_style.residual_emissions_mt` / `optimization/constraints.py`: the CCS
    energy penalty burns extra coal whose uncaptured share is vented, and biomass co-firing
    carries its own efficiency penalty. Ignoring both would overstate CCS abatement by 3%
    and make the hero metric look better than the model believes it is.
    """
    ef = float(assumptions.coal_emission_factor_t_per_mwh)
    heat_rate = float(assumptions.heat_rate_gj_per_mwh)
    ef_per_gj = ef / heat_rate
    eta = float(scenario.capture_rate)
    penalty = float(assumptions.ccs_energy_penalty_ratio(HERO_YEAR))
    vented_ccs = penalty * heat_rate * ef_per_gj * (1.0 - eta)
    vented_bio_coeff = (
        assumptions.biomass_efficiency_penalty_per_ratio / assumptions.coal_plant_base_efficiency
    ) * heat_rate * ef_per_gj
    return {
        "ccs": ef * eta - vented_ccs,
        "beccs": ef - (ef * (1.0 - eta - BIOMASS_BLEND) + vented_ccs
                       + vented_bio_coeff * BIOMASS_BLEND * (1.0 - eta)),
        "biomass": ef - (ef * (1.0 - BIOMASS_BLEND) + vented_bio_coeff * BIOMASS_BLEND),
        "ammonia": ef * AMMONIA_BLEND,
    }


def hero_table(intensities: pd.DataFrame, assumptions, scenario) -> pd.DataFrame:
    """Incremental and total m3 per tonne abated, per (cooling system, pathway).

    Pathway water intensity follows `data_prep.py:742-749` exactly: capture pathways take
    the table's own with-capture consumption, biomass scales the base by
    `biomass_water_multiplier`, ammonia by `ammonia_water_multiplier`.
    """
    abatement = abatement_t_per_mwh(assumptions, scenario)
    multiplier = {
        "ccs": None, "beccs": None,
        "biomass": float(assumptions.biomass_water_multiplier),
        "ammonia": float(assumptions.ammonia_water_multiplier),
    }
    rows = []
    for cool in COOLINGS:
        base = float(intensities.loc[cool, "consumption"])
        capture = float(intensities.loc[cool, "consumption_ccs"])
        for pathway in PATHWAY_SEQ:
            retrofit = capture if multiplier[pathway] is None else base * multiplier[pathway]
            rows.append({
                "cooling": cool, "pathway": pathway,
                "base": base, "retrofit": retrofit, "increment": retrofit - base,
                "abatement": abatement[pathway],
                "incremental_m3_per_t": (retrofit - base) / abatement[pathway],
                "total_m3_per_t": retrofit / abatement[pathway],
            })
    return pd.DataFrame(rows)


# ── panel (a) ────────────────────────────────────────────────────────────────
def panel_a(ax, intensities: pd.DataFrame) -> None:
    """All three bases, by cooling system, unabated and with capture.

    Log y-axis rather than a broken axis: the point IS the three-order span, and a break
    would hide the one thing worth seeing. The bases sit side by side within each cooling
    group so the quota inversion -- once-through metered lowest while it diverts the most --
    is a direct visual comparison rather than a claim in the caption.
    """
    x = np.arange(len(COOLINGS))
    bases = ("withdrawal", "quota", "consumption")
    width = 0.26
    for offset, basis in zip((-width, 0.0, width), bases):
        base_v = intensities[basis].to_numpy()
        ccs_v = intensities[f"{basis}_ccs"].to_numpy()
        ax.bar(x + offset, base_v, width * 0.92, color=BASIS_COLORS[basis],
               label=BASIS_LABELS[basis], zorder=3)
        # Capture increment stacked on top, hatched: same basis, extra volume.
        ax.bar(x + offset, ccs_v - base_v, width * 0.92, bottom=base_v,
               color=BASIS_COLORS[basis], alpha=0.45, hatch="////",
               edgecolor="white", linewidth=0.3, zorder=3)
        for xi, (base_value, total_value) in enumerate(zip(base_v, ccs_v)):
            base_label = f"{base_value:.0f}" if base_value >= 10 else f"{base_value:.2f}"
            total_label = f"{total_value:.0f}" if total_value >= 10 else f"{total_value:.2f}"
            xpos = xi + offset
            ax.text(
                xpos, base_value / 1.45, base_label,
                ha="center", va="center", rotation=0, fontsize=5.8,
                color="white", zorder=5,
            )
            ax.text(
                xpos, total_value * 1.08, total_label,
                ha="center", va="bottom", rotation=0, fontsize=5.8,
                color="#000000", zorder=5,
            )

    ax.set_yscale("log")
    safe_log_axis(ax, "y")   # ylim 下探到 0.05 -> 10^-1/10^-2 的负号
    ax.set_ylim(0.05, 3000)
    ax.set_xticks(x)
    ax.set_xticklabels(COOLING_LABELS)
    ax.set_ylabel("水强度（m$^3$ MWh$^{-1}$）")
    ax.set_title("")

    handles = [Patch(facecolor=BASIS_COLORS[b], label=BASIS_LABELS[b]) for b in bases]
    handles.append(Patch(facecolor="#999999", alpha=0.45, hatch="////", edgecolor="white",
                         label="捕集增量"))
    ax.legend(handles=handles, frameon=False, fontsize=LEGEND_SIZE, loc="upper right",
              ncol=1, handlelength=1.1, labelspacing=0.2, borderpad=0.1)


# ── panel (b) ────────────────────────────────────────────────────────────────
def panel_b(ax, hero: pd.DataFrame) -> None:
    """HERO: incremental m3 per tonne abated. Log axis, two orders of magnitude.

    Incremental, not total: a water regulator asks what the EXTRA tonne of abatement costs
    in water, so the draw of a plant that would have run anyway is not charged to the
    retrofit. Total basis is shown as an open marker on the same row.

    NO BARS ON THE LOG AXIS. This panel used to draw bars rising from an invisible floor at
    0.004 m3/tCO2, a value with no physical meaning chosen only because a log axis cannot
    reach zero. A bar encodes quantity by LENGTH, and on a log axis length is
    log(value) - log(floor): the recirculating-CCS bar was 2.7x the length of the air-CCS
    bar while the values differ by 10x, and moving the invisible floor to 0.04 would have
    redrawn that ratio as 1.7x without changing a single datum. Both bases are now drawn as
    markers joined by a thin connector -- position is the datum, the connector is a guide
    with no quantitative reading, and the arbitrary origin is gone.
    """
    x = np.arange(len(COOLINGS))
    width = 0.19
    floor = 0.004  # log-axis floor: exact zeros cannot be drawn, so they are marked, not drawn
    for index, pathway in enumerate(PATHWAY_SEQ):
        sub = hero[hero["pathway"] == pathway].set_index("cooling").reindex(list(COOLINGS))
        offset = (index - (len(PATHWAY_SEQ) - 1) / 2) * width
        values = sub["incremental_m3_per_t"].to_numpy()
        totals = sub["total_m3_per_t"].to_numpy()
        colour = PATHWAY_PLOT_C[pathway]
        for xi, value, total in zip(x, values, totals):
            xpos = xi + offset
            if value > floor:
                ax.plot([xpos, xpos], [value, total], lw=0.7, color=colour, alpha=0.55,
                        zorder=3, solid_capstyle="round")
                ax.plot([xpos], [value], marker="o", ms=3.0, color=colour, zorder=5)
            else:
                # Exact zero -- AND THAT IS A MODEL BOUNDARY, NOT A PROPERTY OF BIOMASS.
                # `biomass_water_multiplier = 1.00` (scenario.py:127) makes co-firing draw
                # exactly the host plant's cooling water and nothing else: no feedstock
                # cultivation, no harvest, no processing. That is defensible for the CONDENSER
                # accounting this figure is about, and indefensible to leave unlabelled once
                # the paper's mechanism is that the basin constraint is discharged BY BUYING
                # BIOMASS -- 1.77 PJ of it per GW of wet-to-dry conversion, R2 = 0.9997. A zero
                # the model cannot falsify must be drawn as a zero the model cannot falsify.
                # symbol on the axis floor and labelled "0" -- never drawn as a short bar,
                # which would read as a small positive quantity.
                ax.plot([xpos], [floor], marker="_", ms=4.5, mew=1.0, color=colour, zorder=5)
                ax.plot([xpos, xpos], [floor * 1.6, total], lw=0.7, color=colour, alpha=0.35,
                        zorder=3, ls=(0, (1, 1)))
            ax.plot([xpos], [total], marker="o", ms=3.4, mfc="none", mec=colour,
                    mew=0.7, zorder=5)
            if pathway == "ccs" and value > floor:
                ax.text(
                    xpos + 0.035, value, f"{value:.2f}",
                    ha="left", va="center", fontsize=ANNOTATION_SIZE,
                    fontweight="bold", color=colour, zorder=6,
                )
        ax.plot([], [], marker="o", ms=3.0, ls="none", color=colour, label=PATHWAY_L[pathway])

    ax.set_yscale("log")
    safe_log_axis(ax, "y")   # floor 可低于 1
    ax.set_ylim(floor * 0.7, 400)
    ax.set_xticks(x)
    ax.set_xticklabels(COOLING_LABELS)
    ax.set_ylabel("每吨减排的耗水\n（m$^3$ tCO$_2^{-1}$）")
    ax.set_title("每吨 CO$_2$ 的水代价")

    pathway_legend = ax.legend(
        frameon=False, fontsize=LEGEND_SIZE, loc="upper left", ncol=2,
        handlelength=1.0, columnspacing=0.7, labelspacing=0.2, borderpad=0.1,
    )
    ax.add_artist(pathway_legend)
    metric_handles = [
        Line2D([], [], marker="o", ms=4.0, ls="none", color="#555555",
               markerfacecolor="#555555", label="增量水代价"),
        Line2D([], [], marker="o", ms=4.0, ls="none", color="#555555",
               markerfacecolor="none", markeredgewidth=0.8, label="总水代价"),
    ]
    ax.legend(
        handles=metric_handles, frameon=False, fontsize=LEGEND_SIZE,
        loc="upper right", bbox_to_anchor=(0.995, 0.84),
        handlelength=1.0, labelspacing=0.25, borderpad=0.1,
    )


# ── panel (c) ────────────────────────────────────────────────────────────────
def basin_drying(year: int = HERO_YEAR, ssp: str | None = "ssp370") -> pd.DataFrame:
    """Ensemble drying signal and dry-season allowance per level-1 basin.

    20 members = 2 hydrology models x 5 GCMs x 2 SSPs, so 10 hydrology-GCM combinations
    each give one paired SSP3-7.0 / SSP1-2.6 ratio. The median of those 10 is the signal;
    the min and max are the spread, which for the Hai is enormous and reported as such.

    THE SLICE IS A PARAMETER, NOT A CONSTANT, and every caller has to name it. The drying
    SIGNAL is always the paired SSP3-7.0 / SSP1-2.6 ratio -- that is what a drying signal is
    -- but the dry-season ALLOWANCE depends on which members you stand on:

        year=2060, ssp="ssp370"   Fig 1's hero year, the year the emission target binds and
                                  the year CCS is actually built. Hai 16.84.
        year=2030, ssp=None       Fig 2(b) and the basin closeup: the standing fleet is still
                                  nearly intact, so the constraint is tightest, and the median
                                  is taken over all 20 members rather than one SSP. Hai 15.76.

    Those are different numbers for the same basin and neither is wrong. What would be wrong
    is two figures quoting them as though they were the same, which is what an implicit slice
    allowed.
    """
    frame = pd.read_csv(INPUTS / "water_availability.csv")
    frame = frame[frame["planning_year"].astype(int) == int(year)]
    paired = (
        frame.groupby(["basin_code", "hydrology_model", "gcm", "ssp"])["available_water_m3_per_year"]
        .sum().unstack("ssp")
    )
    paired["pct"] = (paired["ssp370"] / paired["ssp126"] - 1.0) * 100.0
    out = paired.groupby("basin_code")["pct"].agg(["median", "min", "max", "count"])
    out.columns = ["pct_median", "pct_min", "pct_max", "n_members"]
    out["n_drying"] = paired.groupby("basin_code")["pct"].apply(lambda s: int((s < 0).sum()))
    hot = frame if ssp is None else frame[frame["ssp"] == ssp]
    for column, label in (("available_water_m3_per_year", "annual"),
                          ("dry_season_water_m3_per_year", "dry_season")):
        # 分组键里带上 ssp：ssp=None 时不加它就会把两个 SSP 的水量加起来，
        # 得到的是"两倍的水"，而不是 20 个成员的中位数。
        keys = ["basin_code", "hydrology_model", "gcm"] + (["ssp"] if ssp is None else [])
        out[f"{label}_1e8"] = (
            hot.groupby(keys)[column].sum().groupby("basin_code").median() / 1e8
        )
    return out


def site_table(assumptions, scenario) -> pd.DataFrame:
    """The 350 sites with basin, capacity, and the CCS water price of each."""
    plants = pd.read_csv(INPUTS / "plants.csv").reset_index(drop=True)
    plants["basin"] = assign_basin(plants).values
    abatement = abatement_t_per_mwh(assumptions, scenario)["ccs"]
    hours = plants["province_mode"].astype(str).map(lambda p: assumptions.province_cf(p) * 8760.0)
    plants["generation_mwh"] = plants["total_capacity_mw"].astype(float) * hours
    plants["capacity_gw"] = plants["total_capacity_mw"].astype(float) / 1000.0
    increment = (
        plants["consumption_ccs_intensity_m3_per_mwh"].astype(float)
        - plants["consumption_intensity_m3_per_mwh"].astype(float)
    )
    plants["ccs_m3_per_t"] = increment / abatement
    plants["ccs_demand_1e8"] = (
        plants["generation_mwh"] * plants["consumption_ccs_intensity_m3_per_mwh"] / 1e8
    )
    # The withdrawal counterpart, on the basis the 用水总量控制指标 is actually metered on.
    # It goes through the SAME once-through calibration the solver uses, so the numerator
    # here and the numerator inside the basin constraint are the same quantity. The raw
    # `withdrawal_ccs_intensity_m3_per_mwh` column would undercount the fleet by the
    # calibration factor and make every basin look comfortably inside its allocation.
    from coal_retrofit.builders.water_quota import calibrated_withdrawal_intensities

    base_intensity, capture_intensity, _, _, _ = calibrated_withdrawal_intensities(
        plants, plants["generation_mwh"]
    )
    plants["ccs_withdrawal_1e8"] = plants["generation_mwh"] * capture_intensity / 1e8
    # As-built counterparts of both bases. Without them the stress columns answer "how big
    # would the claim be if everything were retrofitted" but not "is the basin already over",
    # and those are different questions with different answers.
    plants["base_demand_1e8"] = (
        plants["generation_mwh"] * plants["consumption_intensity_m3_per_mwh"] / 1e8
    )
    plants["base_withdrawal_1e8"] = plants["generation_mwh"] * base_intensity / 1e8
    return plants


def panel_c(ax, sites: pd.DataFrame, drying: pd.DataFrame) -> pd.DataFrame:
    """Map: basin drying underneath, per-site CCS water price on top."""
    import geopandas as gpd

    basins = load_basins()
    provinces = load_map_provinces()          # 统一底图，见 plot_style.load_map_provinces
    provinces.boundary.plot(ax=ax, linewidth=0.15, edgecolor="black", zorder=1)
    # 国界图层：boundary.shp 的 GBCODE 61010 是国界/海岸线，26100 是九段线。
    # 省界图层里没有九段线，所以此前这张全国图从来没有画出过它。
    _country = load_country()
    _country[_country["GBCODE"].isin(COUNTRY_GBCODES)].plot(
        ax=ax, facecolor="none", edgecolor="black", linewidth=0.55, zorder=1.5)

    basins = basins.merge(drying, left_on="code", right_index=True, how="left")
    limit = float(np.nanmax(np.abs(basins["pct_median"])))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    basins.plot(ax=ax, column="pct_median", cmap="BrBG", norm=norm, zorder=2,
                edgecolor="#666666", linewidth=0.4, missing_kwds={"color": "#F2F2F2"})

    points = gpd.GeoDataFrame(
        sites.copy(),
        geometry=gpd.points_from_xy(sites["centroid_longitude"], sites["centroid_latitude"]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    price_norm = Normalize(vmin=0.0, vmax=float(sites["ccs_m3_per_t"].max()))
    scatter = ax.scatter(
        points.geometry.x, points.geometry.y,
        s=3.0 + points["capacity_gw"] * 4.0,
        c=points["ccs_m3_per_t"], cmap="plasma_r", norm=price_norm,
        edgecolors="black", linewidths=0.25, alpha=0.9, zorder=4,
    )
    # Reserve the left margin for the South China Sea inset and the right for colour bars.
    xmin, ymin, xmax, ymax = basins.total_bounds
    span_x, span_y = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.24 * span_x, xmax + 0.30 * span_x)
    ax.set_ylim(ymin - 0.04 * span_y, ymax + 0.04 * span_y)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # Neutral title: the panel maps spatial distributions rather than claiming correlation.
    ax.set_title(f"流域径流变化与煤电 CCS 水代价（{HERO_YEAR} 年）", pad=2)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap="BrBG")
    bar2 = ax.figure.colorbar(mappable, cax=ax.inset_axes([0.885, 0.56, 0.014, 0.34]))
    bar2.set_label("径流情景差异（%）\nSSP3-7.0 相对 SSP1-2.6", fontsize=LEGEND_SIZE)
    bar2.ax.tick_params(labelsize=DETAIL_SIZE, length=2.5, width=0.6)
    bar2.outline.set_linewidth(0.4)

    bar = ax.figure.colorbar(scatter, cax=ax.inset_axes([0.885, 0.11, 0.014, 0.34]))
    bar.set_label("CCS 水代价\n（m$^3$ tCO$_2^{-1}$）", fontsize=LEGEND_SIZE)
    bar.ax.tick_params(labelsize=DETAIL_SIZE, length=2.5, width=0.6)
    bar.outline.set_linewidth(0.4)

    # Label only the four water-constrained basins that carry the paper's mechanism.
    for code in ("C", "D", "E", "K"):
        row = basins[basins["code"] == code]
        if row.empty or not np.isfinite(row["pct_median"].iloc[0]):
            continue
        point = row.geometry.iloc[0].representative_point()
        ax.text(point.x, point.y, f"{BASIN_NAMES_ZH[code]}\n{row['pct_median'].iloc[0]:+.1f}%",
                fontsize=DETAIL_SIZE, ha="center", va="center", color="#222222", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.62))

    # Hai has the widest ensemble spread, so its median alone would be misleading.
    hai = basins[basins["code"] == "C"]
    if not hai.empty and np.isfinite(hai["pct_median"].iloc[0]):
        point = hai.geometry.iloc[0].representative_point()
        ax.annotate(
            f"海河流域不确定性最大\n"
            f"{hai['pct_min'].iloc[0]:+.0f}% 至 {hai['pct_max'].iloc[0]:+.0f}%",
            xy=(point.x, point.y), xycoords="data",
            xytext=(0.735, 0.985), textcoords="axes fraction",
            fontsize=DETAIL_SIZE, color="#7A3000", ha="left", va="top", zorder=6,
            arrowprops=dict(arrowstyle="->", lw=0.6, color="#7A3000",
                            connectionstyle="arc3,rad=0.2"),
        )

    size_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#BBBBBB",
               markeredgecolor="black", markeredgewidth=0.25,
               markersize=np.sqrt(3.0 + gw * 4.0), label=f"{gw:.0f} GW")
        for gw in (2.0, 6.0, 12.0)
    ]
    ax.legend(handles=size_handles, frameon=False, fontsize=DETAIL_SIZE,
              loc="upper left", bbox_to_anchor=(0.075, 0.985),
              labelspacing=0.6, handletextpad=0.6, borderpad=0.2,
              title="厂址容量", title_fontsize=LEGEND_SIZE)

    # 南海小图复用主图的数据图层与色标，并放在大陆左侧的独立空白区。
    def draw_scs_data(inset_ax) -> None:
        basins.plot(
            ax=inset_ax, column="pct_median", cmap="BrBG", norm=norm, zorder=2,
            edgecolor="#666666", linewidth=0.25,
            missing_kwds={"color": "#F2F2F2"},
        )
        inset_ax.scatter(
            points.geometry.x, points.geometry.y,
            s=3.0 + points["capacity_gw"] * 4.0,
            c=points["ccs_m3_per_t"], cmap="plasma_r", norm=price_norm,
            edgecolors="black", linewidths=0.20, alpha=0.9, zorder=4,
        )

    scs_ax = add_scs_inset(
        ax.figure, ax, draw=draw_scs_data,
        axes_rect=[0.010, 0.055, 0.135, 0.30],
    )
    for spine in scs_ax.spines.values():
        spine.set_visible(True)
    return basin_summary(sites, drying)


def basin_summary(sites: pd.DataFrame, drying: pd.DataFrame) -> pd.DataFrame:
    """Per-basin capacity, CCS water price, drying signal, and dry-season stress."""
    grouped = sites.groupby("basin")
    frame = grouped.agg(
        capacity_gw=("capacity_gw", "sum"),
        n_sites=("plant_id", "size"),
        ccs_demand_1e8=("ccs_demand_1e8", "sum"),
        ccs_withdrawal_1e8=("ccs_withdrawal_1e8", "sum"),
        base_demand_1e8=("base_demand_1e8", "sum"),
        base_withdrawal_1e8=("base_withdrawal_1e8", "sum"),
    )
    frame["ccs_m3_per_t"] = grouped.apply(
        lambda g: float(np.average(g["ccs_m3_per_t"], weights=g["capacity_gw"])),
        include_groups=False,
    )
    frame["already_air_share"] = grouped.apply(
        lambda g: float(np.average(g["already_air_share"], weights=g["capacity_gw"])),
        include_groups=False,
    )
    frame = frame.join(drying, how="left")
    # Two rungs, two bases -- reported side by side and never combined into one number.
    #
    # (1) ENVIRONMENTAL FLOW, on CONSUMPTION. What `*_oq_envonly` enforces per node:
    #        dry-season flow x WATER_EXTRACTABLE_FRACTION
    #     Richter et al. (2012) River Res. Applic. 28(8):1312-1321 -- protecting 80% of daily
    #     flows maintains ecological integrity, so 20% is offered to consumptive users. There
    #     is NO second factor here: v9's `(1 - existing_withdrawal_share)` was an ASSUMED
    #     allocation rule, and multiplying it in aliased the depletion standard with the
    #     allocation rule so that no result could say which one was binding.
    #
    # (2) ALLOCATION, on WITHDRAWAL. What `*_oq` adds per basin:
    #        用水总量控制指标 - 非电既有取水
    #     read from inputs/water_basin_caps.csv. The 公报 meters WITHDRAWAL (用水量, including
    #     the once-through pass-through), so this rung must be compared against the fleet's
    #     calibrated withdrawal, not against its consumption. Comparing the consumption
    #     numerator to this denominator would understate the fleet's claim about five-fold.
    frame["env_allowance_1e8"] = frame["dry_season_1e8"] * WATER_EXTRACTABLE_FRACTION
    frame["env_stress_pct"] = frame["ccs_demand_1e8"] / frame["env_allowance_1e8"] * 100.0
    frame["env_stress_base_pct"] = frame["base_demand_1e8"] / frame["env_allowance_1e8"] * 100.0

    caps = pd.read_csv(INPUTS / BASIN_CAPS_CSV)
    caps = caps[caps["planning_year"] == caps["planning_year"].min()]
    residual = caps.set_index("basin_code")["residual_1e8_m3"].astype(float)
    frame["quota_residual_1e8"] = residual.reindex(frame.index)
    frame["quota_stress_pct"] = (
        frame["ccs_withdrawal_1e8"] / frame["quota_residual_1e8"] * 100.0
    )
    frame["quota_stress_base_pct"] = (
        frame["base_withdrawal_1e8"] / frame["quota_residual_1e8"] * 100.0
    )
    return frame.reindex([b for b in BASIN_ORDER if b in frame.index])


# ── assembly ─────────────────────────────────────────────────────────────────
def main() -> None:
    from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario

    assumptions = OptimizationAssumptions()
    scenario = OptimizationScenario(experiment_id="fig1", description="fig1")

    intensities = cooling_intensities()
    hero = hero_table(intensities, assumptions, scenario)
    sites = site_table(assumptions, scenario)
    drying = basin_drying()

    fig = plt.figure(figsize=(DOUBLE_COL[0], 6.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.80, 1.0], hspace=0.24, wspace=0.26,
                            left=0.075, right=0.975, top=0.965, bottom=0.055)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    # `aspect="equal"` shrinks the map's box to the data ratio; anchoring north spends the
    # slack below the map instead of splitting it above and below.
    ax_c.set_anchor("N")

    panel_a(ax_a, intensities)
    panel_b(ax_b, hero)
    summary = panel_c(ax_c, sites, drying)

    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    panel_label_inside(ax_c, "c", x=-0.005, y=1.02)

    # Keep only source and scope information that is required to interpret the plotted values.
    fig.text(
        0.045, 0.004,
        cjk_fill(
            "注：水强度来自 Wang et al. (2023)，取水定额来自水利部《十八项工业用水定额》；"
            "径流集合含 20 个成员，可取用比例为 20%（Richter et al., 2012）；"
            "生物质原料种植用水未纳入。",
            width=150),
        ha="left", va="bottom", fontsize=DETAIL_SIZE, color="#555555", linespacing=1.35,
    )

    save_fig(fig, "fig1_water_footprint")
    _report(intensities, hero, sites, summary, assumptions, scenario)


def _report(intensities, hero, sites, summary, assumptions, scenario) -> None:
    """Print every number the figure plots, so the caption can be written from this log."""
    print("\npanel a - capacity-weighted intensity by cooling system (m3/MWh)")
    columns = ["capacity_gw", "withdrawal", "withdrawal_ccs", "quota", "quota_ccs",
               "consumption", "consumption_ccs"]
    print(intensities[columns].round(3).to_string())
    once, recirc = intensities.loc["once-through"], intensities.loc["recirculating"]
    print(f"  quota inversion: once-through {once['quota']:.3f} < air {intensities.loc['air','quota']:.3f}"
          f" < recirculating {recirc['quota']:.3f}, while once-through withdrawal is"
          f" {once['withdrawal'] / recirc['withdrawal']:.0f}x recirculating")

    print(f"\npanel b - abatement basis ({HERO_YEAR}, capture {scenario.capture_rate:.0%},"
          f" CCS energy penalty {assumptions.ccs_energy_penalty_ratio(HERO_YEAR):.4f})")
    abatement = abatement_t_per_mwh(assumptions, scenario)
    print("  " + ", ".join(f"{k} {v:.4f} tCO2/MWh" for k, v in abatement.items()))
    print(hero[["cooling", "pathway", "base", "retrofit", "increment",
                "incremental_m3_per_t", "total_m3_per_t"]].round(4).to_string(index=False))
    nonzero = hero[hero["incremental_m3_per_t"] > 0]
    print(f"  incremental spread: {nonzero['incremental_m3_per_t'].min():.4f} to"
          f" {nonzero['incremental_m3_per_t'].max():.4f} m3/tCO2"
          f" = {nonzero['incremental_m3_per_t'].max() / nonzero['incremental_m3_per_t'].min():.0f}x")

    print("\npanel c - per-basin summary")
    columns = ["n_sites", "capacity_gw", "already_air_share", "ccs_m3_per_t", "pct_median",
               "pct_min", "pct_max", "n_drying", "dry_season_1e8", "env_allowance_1e8",
               "ccs_demand_1e8", "env_stress_pct", "env_stress_base_pct",
               "quota_residual_1e8", "ccs_withdrawal_1e8", "quota_stress_pct",
               "quota_stress_base_pct"]
    print(summary[columns].round(2).to_string())
    print(f"  fleet capacity-weighted CCS water price:"
          f" {np.average(sites['ccs_m3_per_t'], weights=sites['capacity_gw']):.3f} m3/tCO2")
    over_env = summary[summary["env_stress_pct"] > 100]
    print(f"  rung 1 (env. flow, consumption): over the extractable dry-season flow:"
          f" {list(over_env.index)}  ({over_env['capacity_gw'].sum():.0f} GW)")
    over_quota = summary[summary["quota_stress_pct"] > 100]
    print(f"  rung 2 (用水总量控制指标, withdrawal): over the basin residual:"
          f" {list(over_quota.index)}  ({over_quota['capacity_gw'].sum():.0f} GW)")
    dry_and_thirsty = summary[
        (summary["pct_median"] < summary["pct_median"].median())
        & (summary["ccs_m3_per_t"] > np.average(sites["ccs_m3_per_t"], weights=sites["capacity_gw"]))
    ]
    print(f"  below-median SSP gap AND above-mean CCS water price:"
          f" {list(dry_and_thirsty.index)} ({dry_and_thirsty['capacity_gw'].sum():.0f} GW)")


if __name__ == "__main__":
    main()
