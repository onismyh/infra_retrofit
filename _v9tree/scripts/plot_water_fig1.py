"""Fig 1 for the Nature Water framing: the water footprint of coal decarbonisation.

The protagonist is the CAPTURE FLEET — every pathway that carries a capture island, i.e.
CCS retrofit and BECCS together. Both draw the same 1.82x cooling-water multiplier, and in
the solved scenarios they are what the CO2 network is built for, so they are treated as one
water-intensive class rather than split.

  (a) water intensity by cooling technology and pathway, m3/MWh
  (b) HERO: incremental water per tonne of CO2 abated, m3/tCO2
  (c) where that水 has to come from: provincial capture-fleet water demand against
      dry-season supply, on the map

Usage:
    python scripts/plot_water_fig1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plot_style import apply_style, save_fig, panel_label, ROOT, CN_TO_EN, DOUBLE_COL
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.spatial import load_provinces

apply_style()

INPUTS = ROOT / "inputs"
EXTRACTABLE = 0.20
YEAR = 2050
MEMBER = "cwatm|gfdl-esm4|ssp370"

CAPTURE_COLOR = "#CC3311"
OTHER_COLOR = "#88AABB"
SAVING_COLOR = "#117733"


def _pathway_water_table(year: int = 2060) -> pd.DataFrame:
    """Water and abatement per MWh for each cooling technology and pathway."""
    a = OptimizationAssumptions()
    s = OptimizationScenario(experiment_id="fig1", description="fig1")
    emission = a.coal_emission_factor_t_per_mwh
    eta = s.capture_rate
    penalty_co2 = a.ccs_energy_penalty_ratio(year) * a.heat_rate_gj_per_mwh * (emission / a.heat_rate_gj_per_mwh)

    coolings = [
        ("Once-through", a.cooling_once_through_water_intensity_m3_per_mwh),
        ("Recirculating", a.cooling_recirculating_water_intensity_m3_per_mwh),
        ("Air-cooled", a.cooling_air_water_intensity_m3_per_mwh),
    ]
    # (label, water multiplier, residual emissions per MWh, is part of the capture fleet)
    pathways = [
        ("CCS retrofit", a.ccs_water_multiplier, (emission + penalty_co2) * (1 - eta), True),
        ("BECCS (50% blend)", a.beccs_water_multiplier,
         (emission + penalty_co2) * (1 - eta) - emission * 0.50, True),
        ("Biomass (50% blend)", a.biomass_water_multiplier, emission * 0.50, False),
        ("Ammonia (20% blend)", a.ammonia_water_multiplier, emission * 0.80, False),
        ("Retirement", 0.0, 0.0, False),
    ]

    rows = []
    for cooling, base_water in coolings:
        for name, multiplier, residual, is_capture in pathways:
            water = base_water * multiplier
            abated = emission - residual
            rows.append({
                "cooling": cooling, "pathway": name,
                "water_m3_per_mwh": water,
                "delta_water_m3_per_mwh": water - base_water,
                "abated_t_per_mwh": abated,
                "water_per_tco2": water / abated if abated > 0 else np.nan,
                "delta_water_per_tco2": (water - base_water) / abated if abated > 0 else np.nan,
                "is_capture": is_capture,
            })
    return pd.DataFrame(rows)


def _capture_demand_by_province() -> pd.Series:
    """Water the capture fleet would need if every plant took capture, m3/yr by province."""
    a = OptimizationAssumptions()
    plants = pd.read_csv(INPUTS / "plants.csv")
    cf = plants["province_mode"].map(lambda p: a.province_cf(str(p)))
    generation = plants["total_capacity_mw"].astype(float) * cf * 8760.0
    demand = (
        generation
        * plants["weighted_water_intensity_m3_per_mwh"].astype(float)
        * a.ccs_water_multiplier
        * 1.15
    )
    return demand.groupby(plants["province_mode"]).sum()


def _dry_supply_by_province(member: str, year: int) -> pd.Series:
    availability = pd.read_csv(INPUTS / "water_availability.csv")
    nodes = pd.read_csv(INPUTS / "water_nodes.csv")
    node_province = dict(zip(nodes["water_node_id"], nodes["province_name"]))
    frame = availability[
        (availability["planning_year"].astype(int) == year)
        & (availability["scenario_id"].astype(str) == member)
    ].copy()
    frame["province"] = frame["water_node_id"].map(node_province).map(CN_TO_EN)
    return frame.groupby("province")["dry_season_water_m3_per_year"].sum() * EXTRACTABLE


def main_fig1() -> None:
    table = _pathway_water_table()

    fig = plt.figure(figsize=(DOUBLE_COL[0], 6.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.5], hspace=0.30, wspace=0.30)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    coolings = ["Once-through", "Recirculating", "Air-cooled"]
    pathways = table["pathway"].unique().tolist()
    width = 0.15
    x = np.arange(len(coolings))

    # (a) absolute water intensity
    for i, pathway in enumerate(pathways):
        sub = table[table["pathway"] == pathway].set_index("cooling")
        values = [sub.loc[c, "water_m3_per_mwh"] for c in coolings]
        is_capture = bool(sub["is_capture"].iloc[0])
        ax_a.bar(x + (i - 2) * width, values, width,
                 color=CAPTURE_COLOR if is_capture else OTHER_COLOR,
                 edgecolor="white", linewidth=0.4,
                 hatch="///" if pathway.startswith("BECCS") else None,
                 label=pathway)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(coolings, fontsize=7)
    ax_a.set_ylabel(r"Water consumption (m$^3$ MWh$^{-1}$)")
    ax_a.legend(fontsize=5.5, frameon=False, ncol=1, loc="upper right")
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.set_title("Per unit of electricity", fontsize=8, pad=12)
    panel_label(ax_a, "a")

    # (b) HERO — incremental water per tonne abated
    for i, pathway in enumerate(pathways):
        sub = table[table["pathway"] == pathway].set_index("cooling")
        values = [sub.loc[c, "delta_water_per_tco2"] for c in coolings]
        is_capture = bool(sub["is_capture"].iloc[0])
        colour = CAPTURE_COLOR if is_capture else (SAVING_COLOR if pathway == "Retirement" else OTHER_COLOR)
        ax_b.bar(x + (i - 2) * width, values, width, color=colour,
                 edgecolor="white", linewidth=0.4,
                 hatch="///" if pathway.startswith("BECCS") else None)
    ax_b.axhline(0, color="#444444", lw=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(coolings, fontsize=7)
    ax_b.set_ylabel("Incremental water per tonne abated\n" + r"(m$^3$ per tCO$_2$)")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.set_title(r"Per tonne of CO$_2$ abated", fontsize=8, pad=12)
    recirc = table[(table["cooling"] == "Recirculating")].set_index("pathway")
    ax_b.annotate(f"CCS {recirc.loc['CCS retrofit', 'delta_water_per_tco2']:.2f}",
                  xy=(1 - 2 * width, recirc.loc["CCS retrofit", "delta_water_per_tco2"]),
                  xytext=(0, 4), textcoords="offset points", fontsize=6, ha="center", color=CAPTURE_COLOR)
    ax_b.annotate(f"retire {recirc.loc['Retirement', 'delta_water_per_tco2']:.2f}",
                  xy=(1 + 2 * width, recirc.loc["Retirement", "delta_water_per_tco2"]),
                  xytext=(0, -10), textcoords="offset points", fontsize=6, ha="center", color=SAVING_COLOR)
    panel_label(ax_b, "b")

    # (c) provincial stress map: capture-fleet demand as % of dry-season supply
    demand = _capture_demand_by_province()
    supply = _dry_supply_by_province(MEMBER, YEAR)
    stress = (demand / supply.reindex(demand.index) * 100).replace([np.inf, -np.inf], np.nan)

    provinces = load_provinces(ROOT / "data" / "ChinaMap" / "provinces.shp").to_crs("EPSG:2380")
    name_col = "province_name" if "province_name" in provinces.columns else provinces.columns[0]
    provinces["en"] = provinces[name_col].map(CN_TO_EN)
    provinces["stress"] = provinces["en"].map(stress)

    norm = TwoSlopeNorm(vmin=0, vcenter=100, vmax=500)
    provinces.plot(ax=ax_c, column="stress", cmap="RdYlBu_r", norm=norm,
                   edgecolor="white", linewidth=0.3, missing_kwds={"color": "#EEEEEE"})
    ax_c.set_axis_off()
    ax_c.set_title(
        f"Capture-fleet water demand as % of dry-season supply · {YEAR} · {MEMBER.split('|')[0]} SSP3-7.0",
        fontsize=8,
    )
    scalar = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    cbar = fig.colorbar(scalar, ax=ax_c, fraction=0.020, pad=0.01, shrink=0.75, extend="max")
    cbar.set_label("% of dry-season usable water", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.axhline(100, color="black", lw=0.8)

    over = stress[stress > 100].sort_values(ascending=False)
    caption = "Above 100% of dry-season supply:  " + " · ".join(
        f"{p} {v:.0f}%" for p, v in over.items())
    ax_c.text(0.5, -0.04, caption, transform=ax_c.transAxes, fontsize=6.2,
              va="top", ha="center", color="#772222", wrap=True)
    panel_label(ax_c, "c", x=-0.02, y=1.0)

    fig.tight_layout()
    save_fig(fig, "water_fig1_capture_footprint")

    print("\nIncremental water per tonne abated (m³/tCO₂), recirculating:")
    print(recirc[["delta_water_m3_per_mwh", "abated_t_per_mwh", "delta_water_per_tco2"]].round(3).to_string())
    print(f"\nProvinces above 100% of dry-season supply: {int((stress > 100).sum())} of {int(stress.notna().sum())}")


if __name__ == "__main__":
    main_fig1()
