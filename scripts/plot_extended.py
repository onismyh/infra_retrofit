"""Extended Data figures (ED 1-12) for coal retrofit paper.

Generates 12 extended data figures with consistent styling from plot_style:
  ed_fig1_biomass_feasibility   — 3-panel: supply/demand + distance + saturation
  ed_fig2_emissions_trajectory  — Emission trajectory lines
  ed_fig3_sensitivity           — 4-panel: tornado + robustness + ammonia + retire
  ed_fig4_provincial_choropleth — CCS+BECCS / retirement choropleth (EPSG:2380)
  ed_fig5_fleet_age_map         — Avg retirement year map (EPSG:2380)
  ed_fig6_provincial_bars       — 2-panel provincial stacked bars by region
  ed_fig7_distance_vs_ccs       — Distance to storage vs CCS share scatter
  ed_fig8_storage_utilization   — 2-panel: utilization bars + capacity scatter
  ed_fig9_stranded_assets       — 3-panel: retirement wave + early years + value
  ed_fig10_pathway_restrictions — Key scenario pathway allocation comparison
  ed_fig11_ccs_only_cost        — CCS-only vs BASE cost structure
  ed_fig12_plant_cost_map       — Plant-level cost map (EPSG:2380)

Usage:
    python scripts/plot_extended.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (
    apply_style, save_fig, panel_label, panel_label_inside,
    PATHWAY_COLORS, PATHWAY_LABELS, PATHWAY_ORDER, pathway_legend,
    clean_shares, format_pct, annotate_value,
    SINGLE_COL, DOUBLE_COL, HALF_PAGE, FULL_PAGE, WIDE, TALL,
    ROOT, RESULTS_DIR, FIGURES_DIR, BASE_DIR,
    REGIONS, CN_TO_EN, EN_TO_CN,
    SCENARIO_COLORS, DIVERGING,
    MAP_SINGLE, MAP_DOUBLE,
    residual_emissions_mt,
)

apply_style()

YEARS = [2030, 2040, 2050, 2060]

# The scenarios these figures draw. Named explicitly rather than globbed, so a stray result
# file cannot silently join a panel and so a missing solve is reported instead of dropped.
# The `SA_biomass_cost_150` entry is intentionally present: the earlier ledger carried it and
# panel ed_fig3(b) reads every SA_* key it finds.
SCENARIO_LEDGER = (
    "BASE", "BASE_neg", "BASE_zero",
    "RQ3_ccs_only", "RQ3_no_ammonia", "RQ3_no_biomass", "RQ3_no_ccs", "RQ3_retire_only",
    "SA_ammonia_cost_50", "SA_ammonia_cost_70",
    "SA_biomass_cost_150", "SA_biomass_cost_200",
    "SA_carbon_high", "SA_carbon_low",
    "SA_ccs_capex_high", "SA_ccs_capex_low",
    "SA_discount_3pct", "SA_discount_8pct",
    "SA_injectivity_half",
    "SA_pipe_full", "SA_pipe_mid",
    "SA_retire_cost_1000", "SA_retire_cost_500",
    # The water scenario the PAPER is about. ed_fig2 used to label `WA_grid_200km` -- a
    # superseded 200 km grid-supply variant with no reservation and no dry-season basis --
    # as "Water constraint", which is not the treatment any main figure uses.
    "WA_cwatm_126_dry", "WA_cwatm_126_dry_wd085",
    "WA_grid_200km",
)

# The year the biomass feasibility panels are drawn at. 2030 is useless for this purpose:
# national biomass utilisation is 0.59% there and 86.8% at 2050, where twelve provinces are
# at exactly 100% and the supply curve is the binding constraint.
BIOMASS_PANEL_YEAR = 2050


# ── Data loaders ─────────────────────────────────────────────────────────────

def _slack_share(result: dict) -> float:
    """Largest single-year share of the objective taken by the big-M slack penalty."""
    worst = 0.0
    for payload in result.get("years", {}).values():
        if not isinstance(payload, dict):
            continue
        breakdown = payload.get("cost_breakdown", {}) or {}
        slack = float(breakdown.get("slack_penalty", 0.0) or 0.0)
        total = float(payload.get("objective_cny", 0.0) or 0.0)
        if total <= 0:
            total = sum(float(v) for v in breakdown.values()
                        if isinstance(v, (int, float)))
        if total > 0:
            worst = max(worst, slack / total)
    return worst


def _load_json() -> dict:
    """Read the LIVE per-scenario results, the same files the main figures read.

    THIS REPLACES A FROZEN, POST-PROCESSED SNAPSHOT, and the replacement is the point.
    Every Extended Data panel used to be drawn from `results/experiment_results_clean.json`,
    a file written 2026-08-09 whose own `_metadata` recorded that it

      * rounded pathway shares below 0.1% to zero and renormalised the rest to 1.0,
      * ZEROED any slack penalty worth less than 5% of the period cost, calling it a
        "solver artifact",
      * recomputed both the per-year and the global objective as sums of those cleaned
        components.

    Three consequences, all measured rather than asserted:

      1. It disagreed with the solver. BASE came to 14.0034e12 there against 13.9688e12 in
         `results/BASE.json` -- 0.247% apart, on a paper whose headline effect is 5.8%.
      2. Zeroing small slack is the one operation the main figures explicitly refuse. Fig 3(a)
         draws the unserved-water penalty as its own hatched bar because it is scarcity
         priced, not money spent; an appendix that silently deletes the same quantity below a
         threshold is on a different cost definition from the main text, and the two cannot
         be quoted side by side.
      3. Rounding shares below 0.1% to zero erases real answers. BASE 2030 biomass is
         0.2744% of generation in the live result and exactly 0.0000% in the cleaned one.

    The `infeasible` flag was the one thing the cleaner added that the raw files lack, so it
    is recomputed here from the same rule it used -- slack >= 5% of a period objective -- and
    nothing else about the numbers is touched.
    """
    live: dict = {}
    missing: list[str] = []
    for name in sorted(SCENARIO_LEDGER):
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["infeasible"] = _slack_share(payload) >= 0.05
        live[name] = payload

    print(f"  [source] live per-scenario JSON from {RESULTS_DIR.name}/  "
          f"({len(live)} of {len(SCENARIO_LEDGER)} scenarios)")
    if missing:
        print(f"  [source] not solved, panels using them will skip: {', '.join(missing)}")
    flagged = [n for n, r in live.items() if r["infeasible"]]
    if flagged:
        print(f"  [source] slack >= 5% of a period objective (drawn as infeasible): "
              f"{', '.join(sorted(flagged))}")
    stale = RESULTS_DIR / "experiment_results_clean.json"
    if stale.exists():
        import datetime as _dt
        stamp = _dt.datetime.fromtimestamp(stale.stat().st_mtime).strftime("%Y-%m-%d")
        print(f"  [source] NOT reading {stale.name} ({stamp}); it zeroes slack below 5% and "
              f"recomputes objectives, so it is on a different cost definition from the "
              f"main figures")
    return live


def _base(data: dict) -> dict:
    return data.get("BASE")


def _shares(result: dict, year: int) -> dict:
    return result.get("years", {}).get(str(year), {}).get("pathway_shares", {})


def _cost_bd(result: dict, year: int) -> dict:
    return result.get("years", {}).get(str(year), {}).get("cost_breakdown", {})


# ── Map helpers ──────────────────────────────────────────────────────────────

TARGET_CRS = "EPSG:2380"
_MAP_CACHE: dict = {}


def _load_provinces():
    if "prov" in _MAP_CACHE:
        return _MAP_CACHE["prov"]
    import geopandas as gpd
    from plot_style import load_map_provinces
    prov = load_map_provinces().to_crs(TARGET_CRS)     # 统一底图（2023 版 GeoJSON）
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    prov["province_en"] = prov["NAME"].map(cn_to_en)
    _MAP_CACHE["prov"] = prov
    return prov


def _load_country():
    """国界 + 九段线。原来是 provinces.dissolve()，那样画不出九段线。"""
    if "country" in _MAP_CACHE:
        return _MAP_CACHE["country"]
    from plot_style import load_country
    _MAP_CACHE["country"] = load_country().to_crs(TARGET_CRS)
    return _MAP_CACHE["country"]


def _map_bounds():
    import geopandas as gpd
    from shapely.geometry import Point
    if "bounds" not in _MAP_CACHE:
        _MAP_CACHE["bounds"] = gpd.GeoDataFrame(
            geometry=[Point(80, 15), Point(150, 50)], crs="EPSG:4326"
        ).to_crs(TARGET_CRS)
        _MAP_CACHE["scs"] = gpd.GeoDataFrame(
            geometry=[Point(106.5, 2.8), Point(123, 24.5)], crs="EPSG:4326"
        ).to_crs(TARGET_CRS)
    return _MAP_CACHE["bounds"], _MAP_CACHE["scs"]


def _basemap(ax, facecolor="#f5f5f5"):
    prov = _load_provinces()
    country = _load_country()
    bounds, _ = _map_bounds()
    prov.plot(ax=ax, facecolor=facecolor, edgecolor="black",
              linewidth=0.2, zorder=1)
    country.plot(ax=ax, facecolor="none", edgecolor="black",
                 linewidth=0.75, zorder=2)
    ax.set_xlim(bounds.geometry.iloc[0].x, bounds.geometry.iloc[1].x)
    ax.set_ylim(bounds.geometry.iloc[0].y, bounds.geometry.iloc[1].y)
    ax.set_axis_off()


def _choropleth(ax, merged, column, cmap, vmin=0, vmax=100,
                legend_kwds=None):
    country = _load_country()
    merged.plot(column=column, ax=ax, cmap=cmap, edgecolor="black",
                linewidth=0.2, legend=True, vmin=vmin, vmax=vmax,
                legend_kwds=legend_kwds or {})
    country.plot(ax=ax, facecolor="none", edgecolor="black",
                 linewidth=0.75, zorder=10)
    bounds, _ = _map_bounds()
    ax.set_xlim(bounds.geometry.iloc[0].x, bounds.geometry.iloc[1].x)
    ax.set_ylim(bounds.geometry.iloc[0].y, bounds.geometry.iloc[1].y)
    ax.set_axis_off()


def _add_map_elements(ax):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([x0, x0 + 500_000], [y0, y0], "k-", linewidth=1.5, zorder=20)
    ax.text(x0 + 250_000, y0 - (ylim[1] - ylim[0]) * 0.02, "500 km",
            ha="center", va="top", fontsize=6, zorder=20)
    ax.annotate("N", xy=(0.95, 0.95), xycoords="axes fraction",
                fontsize=9, fontweight="bold", ha="center")
    ax.annotate("", xy=(0.95, 0.93), xycoords="axes fraction",
                xytext=(0.95, 0.87), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))


def _scs_inset(fig, ax_main):
    prov = _load_provinces()
    country = _load_country()
    _, scs = _map_bounds()
    pos = ax_main.get_position()
    inset = fig.add_axes([pos.x1 - 0.12, pos.y0 + 0.01, 0.11, 0.16])
    prov.plot(ax=inset, facecolor="#f5f5f5", edgecolor="black", linewidth=0.2)
    country.plot(ax=inset, facecolor="none", edgecolor="black", linewidth=0.75)
    inset.set_xlim(scs.geometry.iloc[0].x, scs.geometry.iloc[1].x)
    inset.set_ylim(scs.geometry.iloc[0].y, scs.geometry.iloc[1].y)
    inset.set_xticks([])
    inset.set_yticks([])


def _reproj(df, lon_col, lat_col):
    import geopandas as gpd
    valid = df.dropna(subset=[lon_col, lat_col])
    gdf = gpd.GeoDataFrame(
        valid, geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    gdf["px"] = gdf.geometry.x
    gdf["py"] = gdf.geometry.y
    return gdf


# ── ED Fig 1: Biomass Feasibility ───────────────────────────────────────────

def ed_fig1_biomass_feasibility() -> None:
    """3-panel: (a) province supply/demand, (b) transport distance, (c) node saturation."""
    resource = pd.read_csv(BASE_DIR / "resource_use.csv")
    bio_flows = pd.read_csv(BASE_DIR / "biomass_flows.csv")
    # RESTRICT TO PROVINCES THAT HOST COAL PLANTS IN THIS STUDY.
    # The biomass layer covers 33 provinces; the fleet occupies 28. The five extras --
    # Beijing, Hong Kong, Shanghai, Taiwan, Tibet -- carry biomass nodes but no plant that
    # could burn from them, so their utilisation is decided entirely by cross-province links
    # and they were taking slots in a "top 12" panel about fleet feasibility. Every province
    # that does host a plant also has a biomass node, so nothing the fleet can use is lost.
    coal_provinces = set(
        pd.read_csv(BASE_DIR / "plant_detail.csv")["province_name"].unique()
    )

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.0, 3.5))

    # (a) Province biomass supply vs demand.
    # YEAR. This was drawn at 2030, where national biomass utilisation is 0.59% -- so every
    # "Used" bar was invisible and every annotation read "0%". A feasibility panel drawn at
    # the one year in which the resource is untouched cannot test feasibility. The binding
    # year is what matters: utilisation runs 0.6% / 78.5% / 86.8% / 84.5% across
    # 2030/2040/2050/2060, and at 2050 twelve provinces sit at EXACTLY 100%.
    # SELECTION. `sort_values("utilization").tail(12)` was a no-op at 2030, because 29 of the
    # 33 provinces tie at exactly 0 and a stable sort then returns the alphabetical tail --
    # which is why Taiwan and Tibet, neither of which hosts a coal plant in this study, took
    # two of the twelve slots. Sorting by used volume as a tie-break makes the selection real.
    bio_2030 = resource[
        (resource["resource_type"] == "biomass") & (resource["year"] == BIOMASS_PANEL_YEAR)
    ].copy()
    # Province names may be Chinese in resource_use.csv
    bio_2030["prov_en"] = bio_2030["province_name"].map(CN_TO_EN).fillna(
        bio_2030["province_name"])
    bio_2030 = bio_2030[bio_2030["prov_en"].isin(coal_provinces)]
    prov_bio = bio_2030.groupby("prov_en")[["used", "available"]].sum()
    prov_bio["utilization"] = np.where(
        prov_bio["available"] > 0,
        prov_bio["used"] / prov_bio["available"] * 100, 0)
    prov_bio = prov_bio.sort_values(["utilization", "used"], ascending=True).tail(12)

    y_a = np.arange(len(prov_bio))
    # Nested, not overplotted at equal height. Where utilisation is 100% the two bars are
    # identical and the dark one hid the light one completely, so the "Available" key
    # matched no visible ink. Drawing Used at 55% height keeps both readable and makes a
    # fully-consumed province look like what it is: a filled envelope.
    ax_a.barh(y_a, prov_bio["available"] / 1e9, height=0.62,
              color="#DDEEDD", edgecolor="#88BB88", linewidth=0.4,
              label="Available")
    ax_a.barh(y_a, prov_bio["used"] / 1e9, height=0.62 * 0.55,
              color="#228833", edgecolor="none",
              label="Used")
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels(prov_bio.index, fontsize=6)
    # resource_use.csv carries GJ/yr (its own `unit` column), so /1e9 is EJ. The axis said
    # PJ, which is 1 000x out: Shandong reads 2.03 and that is 2.03 EJ, not 2.03 PJ.
    ax_a.set_xlabel(f"Biomass (EJ yr$^{{-1}}$), {BIOMASS_PANEL_YEAR}")
    ax_a.legend(fontsize=6)
    # Annotate utilization %
    for i, (idx, row) in enumerate(prov_bio.iterrows()):
        ax_a.text(row["available"] / 1e9 + 0.2, i,
                  f"{row['utilization']:.0f}%", fontsize=5, va="center")
    panel_label(ax_a, "a")

    # (b) Transport distance distribution.
    _dist_slot = [0]
    # Years matched to panels (a) and (c). The old pair was 2030 vs 2040, and the 2030 arm is
    # drawn from the 0.59% of the resource that year uses -- a handful of flows, which is why
    # it rendered as a comb of isolated spikes next to a smooth 2040 distribution. Comparing
    # the first year in which biomass is used at scale with the year the resource saturates
    # is the comparison that carries information.
    for yr, color, ls, label in [
        (2040, "#228833", "-", "2040"),
        (BIOMASS_PANEL_YEAR, "#CCBB44", "--", str(BIOMASS_PANEL_YEAR)),
    ]:
        bf = bio_flows[bio_flows["year"] == yr]
        if len(bf) == 0:
            continue
        weights = bf["flow_gj"].values
        distances = bf["distance_km"].values
        mask = weights > 0
        if mask.sum() == 0:
            continue
        ax_b.hist(distances[mask], bins=30, weights=weights[mask],
                  density=True, alpha=0.5, color=color, label=label,
                  edgecolor="white", linewidth=0.3)
        wmean = np.average(distances[mask], weights=weights[mask])
        ax_b.axvline(wmean, color=color, linewidth=1, linestyle=ls)
        # Axes-fraction placement, one slot per series. Data coordinates keyed to
        # `get_ylim()` moved between the two hist() calls, so both labels landed at the same
        # height and overprinted each other.
        ax_b.text(0.97, 0.88 - 0.09 * _dist_slot[0], f"Mean {label}: {wmean:.0f} km",
                  transform=ax_b.transAxes, fontsize=5.5, color=color,
                  ha="right", va="top")
        _dist_slot[0] += 1

    ax_b.set_xlabel("Transport distance (km)")
    ax_b.set_ylabel("Density")
    ax_b.legend(fontsize=6)
    panel_label(ax_b, "b")

    # (c) Node saturation: % of nodes at 100% utilization per province (2030)
    bio_2030_all = resource[
        (resource["resource_type"] == "biomass") & (resource["year"] == BIOMASS_PANEL_YEAR)
    ].copy()
    bio_2030_all["prov_en"] = bio_2030_all["province_name"].map(
        CN_TO_EN).fillna(bio_2030_all["province_name"])
    bio_2030_all = bio_2030_all[bio_2030_all["prov_en"].isin(coal_provinces)]
    bio_2030_all["saturated"] = bio_2030_all["utilization"] >= 0.99

    prov_sat = bio_2030_all.groupby("prov_en").agg(
        total_nodes=("saturated", "count"),
        sat_nodes=("saturated", "sum"),
    ).reset_index()
    prov_sat["sat_pct"] = prov_sat["sat_nodes"] / prov_sat["total_nodes"] * 100
    prov_sat = prov_sat[prov_sat["sat_pct"] > 0].sort_values(
        "sat_pct", ascending=True).tail(12)

    if len(prov_sat) > 0:
        y_c = np.arange(len(prov_sat))
        ax_c.barh(y_c, prov_sat["sat_pct"], color="#CC3311", height=0.6,
                  edgecolor="white", linewidth=0.3)
        ax_c.set_yticks(y_c)
        ax_c.set_yticklabels(prov_sat["prov_en"], fontsize=6)
        ax_c.set_xlabel(f"Saturated nodes (%), {BIOMASS_PANEL_YEAR}")
        for i, v in enumerate(prov_sat["sat_pct"]):
            ax_c.text(v + 0.5, i, f"{v:.0f}%", fontsize=5, va="center")
    else:
        # Turn the axes OFF in the empty state. Leaving it live drew a default 0-1 box with
        # ticks and four spines around the words "No saturated nodes", which reads as a
        # rendering failure rather than as a result.
        ax_c.text(0.5, 0.5, "No saturated nodes", transform=ax_c.transAxes,
                  ha="center", va="center", fontsize=8)
        ax_c.axis("off")
    panel_label(ax_c, "c")

    fig.tight_layout()
    save_fig(fig, "ed_fig1_biomass_feasibility", subdir="extended")


# ── ED Fig 2: Emissions Trajectory ──────────────────────────────────────────

BASE_YEAR = 2025
EMISSION_YEARS = [BASE_YEAR, *YEARS]
CAPTURE_RATE = 0.90
BIOMASS_BLEND_LEVELS = (0.10, 0.25, 0.50, 0.75, 1.00)
AMMONIA_BLEND_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50)


def _blend_level_to_ratio(level: object, levels: tuple[float, ...]) -> float:
    """Convert solver blend-level index to its physical blend ratio."""
    if pd.isna(level):
        return 0.0
    value = float(level)
    if value <= 0.0:
        return 0.0
    level_idx = int(round(value))
    if abs(value - level_idx) <= 1e-6 and 1 <= level_idx <= len(levels):
        return float(levels[level_idx - 1])
    return value


def _scenario_plant_detail(scenario_key: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / scenario_key / "plant_detail.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found for {scenario_key}")
        return None
    return pd.read_csv(path)


def _year_residual_emissions_mt(plant: pd.DataFrame, year: int) -> float:
    # Delegate to the model-consistent accounting in plot_style (retrofit CF
    # boost, rebuilt-plant efficiency, chosen blend levels, penalty emissions).
    return residual_emissions_mt(plant[plant["year"] == year], year)


def _baseline_emissions_mt(plant: pd.DataFrame) -> float:
    first_model_year = int(plant["year"].min())
    base_year = plant[plant["year"] == first_model_year]
    return float(base_year["baseline_emissions_mt"].astype(float).sum())


def _emissions_trajectory_mt(
    scenario_key: str,
    base_year_emissions_mt: float | None = None,
) -> list[float]:
    plant = _scenario_plant_detail(scenario_key)
    if plant is None:
        return [float("nan") for _ in EMISSION_YEARS]

    if base_year_emissions_mt is None:
        base_year_emissions_mt = _baseline_emissions_mt(plant)
    emissions = [base_year_emissions_mt]
    emissions.extend(_year_residual_emissions_mt(plant, year) for year in YEARS)
    return emissions


def ed_fig2_emissions_trajectory(data: dict) -> None:
    """Emission trajectory for core target, pathway, and water scenarios."""
    scenarios = [
        ("BASE", "BASE", "#666666", "-", "o", 1.7),
        ("Net zero", "BASE_zero", "#999999", "--", "s", 1.1),
        ("Net negative", "BASE_neg", "#CCBB44", ":", "v", 1.1),
        ("No ammonia", "RQ3_no_ammonia", "#EE6677", (0, (1, 1)), "X", 1.2),
        ("No CCS", "RQ3_no_ccs", "#CC3311", "--", "^", 1.4),
        ("No biomass", "RQ3_no_biomass", "#228833", "-.", "D", 1.4),
        ("Retire only", "RQ3_retire_only", "#555555", (0, (5, 1)), "*", 1.1),
        ("CCS/BECCS only", "RQ3_ccs_only", "#4477AA", (0, (3, 1)), "<", 1.1),
        # THE WATER LINES ARE THE PAPER'S OWN WATER SCENARIOS NOW.
        # This slot used to hold `WA_grid_200km` under the label "Water constraint". That run
        # is a superseded 200 km grid-supply variant: it has no non-power reservation, is on
        # the annual rather than the dry-season basis, and is not the treatment any main
        # figure uses. Labelling it "Water constraint" in the appendix put a different water
        # treatment beside the main text under the same name.
        # Both sides of the headline contrast are drawn instead, so the appendix shows the
        # same comparison Fig 3 and Fig 5 do: accounting for basin water at all, and then
        # reserving the non-power share so it binds.
        ("Water accounted", "WA_cwatm_126_dry", "#33BBEE", (0, (3, 1, 1, 1)), "P", 1.2),
        ("Water binds (0.85 reserved)", "WA_cwatm_126_dry_wd085", "#0077BB", "-", "P", 1.6),
    ]

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    plotted_values: list[float] = []
    base_plant = _scenario_plant_detail("BASE")
    base_year_emissions = (
        _baseline_emissions_mt(base_plant) if base_plant is not None else None
    )

    annotated_y: list[float] = []
    for label, key, color, ls, marker, lw in scenarios:
        sc = data.get(key)
        if sc is None:
            continue
        # "INFEASIBLE" WAS DOING TOO MUCH WORK AS A SINGLE BINARY.
        # Three quite different things were collapsed into one dotted line and one italic
        # "(infeasible)" tag:
        #   * a run the solver could not solve to optimality at all (status != optimal, or a
        #     zero-filled degenerate solve like RQ3_retire_only);
        #   * RQ3_ccs_only, which IS solved but buys 91.4% of its objective as big-M penalty
        #     for demand it cannot serve -- numerically meaningless as a cost;
        #   * WA_cwatm_126_dry_wd085, the paper's own headline water run, which leaves 1.6%
        #     of 2030 water demand unserved (7.99% of that year's objective). That is a
        #     REPORTED PHYSICAL RESULT discussed at length in the main text, not a failure.
        # Flattening the third into the first two would have the appendix disown the main
        # finding. The line style now grades on how much of the objective is penalty, and
        # the annotation states the number instead of a verdict.
        slack = _slack_share(sc)
        statuses = {
            str(yr_data.get("status", ""))
            for yr_data in sc.get("years", {}).values()
            if isinstance(yr_data, dict)
        }
        unsolved = bool(statuses) and statuses != {"optimal"}
        dominated = slack >= 0.25          # penalty is most of the objective: not a cost
        emissions = _emissions_trajectory_mt(key, base_year_emissions)
        plotted_values.extend(value for value in emissions if np.isfinite(value))
        style = ":" if (unsolved or dominated) else ls
        ax.plot(EMISSION_YEARS, emissions, color=color, linestyle=style,
                linewidth=lw, marker=marker, markersize=4, label=label)
        if unsolved:
            note = "not solved to optimality"
        elif dominated:
            note = f"penalty is {slack:.0%} of a period objective -- not a cost"
        elif slack > 0.005:
            note = f"peak {slack:.1%} of a period objective is unserved-water penalty"
        else:
            note = ""
        if note:
            # NUDGE, DO NOT DROP. The previous rule silently skipped any annotation landing
            # within 150 Mt of an earlier one, which is how the single most important caveat
            # in this panel -- that CCS-only spends 91% of its objective on penalty -- ended
            # up unlabelled, because "Retire only" happened to end 100 Mt away.
            y_text = float(emissions[-1])
            while any(abs(y_text - y_prev) < 210 for y_prev in annotated_y):
                y_text -= 210
            ax.annotate(note, xy=(YEARS[-1], emissions[-1]),
                        xytext=(YEARS[-1] + 1.2, y_text),
                        fontsize=5.2, color=color, fontstyle="italic", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=color, alpha=0.6)
                        if abs(y_text - emissions[-1]) > 1 else None)
            annotated_y.append(y_text)

    ax.axhline(0, color="black", linewidth=0.3, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual emissions (MtCO$_2$/yr)")
    ax.set_xticks(EMISSION_YEARS)
    ax.legend(fontsize=6.5, ncol=2)
    if plotted_values:
        y_min = min(plotted_values)
        y_max = max(plotted_values)
        ax.set_ylim(min(-500, y_min * 1.1), y_max * 1.08)

    fig.tight_layout()
    save_fig(fig, "ed_fig2_emissions_trajectory", subdir="extended")


# ── ED Fig 3: Sensitivity (4-panel) ─────────────────────────────────────────

def ed_fig3_sensitivity(data: dict) -> None:
    """4-panel: (a) tornado, (b) biomass robustness, (c) ammonia, (d) retire."""
    base_cost = _base(data)["global_objective_cny"] / 1e9

    # THE DISCOUNT RATE DOES NOT BELONG IN THIS TORNADO, and it used to dominate it.
    # `global_objective_cny` is a discounted NPV, so changing the discount rate moves it by
    # arithmetic whether or not a single decision changes. That bar spanned -31% to +93% of
    # BASE -- five times the next largest -- and by sorting on range it pushed every genuine
    # sensitivity to the bottom of the chart and compressed them against a meaningless scale.
    # It is reported separately below the panel instead of ranked alongside real ones.
    #
    # One-sided entries are marked. "Biomass cost" and "Storage injectivity" have no low
    # variant, so their low end IS BASE; a tornado compares RANGES, and a one-sided range is
    # not comparable with a two-sided one unless the reader is told which is which.
    sensitivity_pairs = [
        ("CCS CAPEX", "SA_ccs_capex_low", "SA_ccs_capex_high"),
        ("Replacement cost", "SA_retire_cost_500", "SA_retire_cost_1000"),
        ("Biomass cost *", None, "SA_biomass_cost_200"),
        ("Ammonia cost", "SA_ammonia_cost_50", "SA_ammonia_cost_70"),
        ("Pipeline corridor", "SA_pipe_mid", "SA_pipe_full"),
        ("Storage injectivity *", None, "SA_injectivity_half"),
    ]
    excluded_npv = [("Discount rate", "SA_discount_3pct", "SA_discount_8pct")]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) Tornado
    labels_t, lows, highs = [], [], []
    for label, low_key, high_key in sensitivity_pairs:
        low_c = (data[low_key]["global_objective_cny"] / 1e9
                 if low_key and low_key in data else base_cost)
        high_c = (data[high_key]["global_objective_cny"] / 1e9
                  if high_key and high_key in data else base_cost)
        low_d = (low_c - base_cost) / base_cost * 100
        high_d = (high_c - base_cost) / base_cost * 100
        labels_t.append(label)
        lows.append(min(low_d, high_d))
        highs.append(max(low_d, high_d))

    ranges = [h - l for l, h in zip(lows, highs)]
    order = np.argsort(ranges)
    labels_t = [labels_t[i] for i in order]
    lows = [lows[i] for i in order]
    highs = [highs[i] for i in order]

    y_t = np.arange(len(labels_t))
    ax_a.barh(y_t, [h - max(l, 0) for l, h in zip(lows, highs)],
              left=[max(l, 0) for l in lows],
              color=DIVERGING["pos"], alpha=0.7, height=0.5,
              label="Cost increase")
    ax_a.barh(y_t, [min(0, l) for l in lows],
              color=DIVERGING["neg"], alpha=0.7, height=0.5,
              label="Cost decrease")
    ax_a.set_yticks(y_t)
    ax_a.set_yticklabels(labels_t)
    ax_a.set_xlabel("$\\Delta$Cost vs BASE (%)")
    ax_a.axvline(0, color="black", linewidth=0.5)
    ax_a.legend(fontsize=6, loc="lower right")
    notes = ["* one-sided: low end is BASE (no low variant solved)"]
    for label, low_key, high_key in excluded_npv:
        if low_key in data and high_key in data:
            lo = (data[low_key]["global_objective_cny"] / 1e9 - base_cost) / base_cost * 100
            hi = (data[high_key]["global_objective_cny"] / 1e9 - base_cost) / base_cost * 100
            notes.append(f"{label} excluded: {min(lo, hi):+.0f}% to {max(lo, hi):+.0f}% "
                         f"-- the objective is an NPV, so this is discounting"
                         f" arithmetic, not a change in the answer")
    ax_a.text(0.02, -0.30, "\n".join(notes), transform=ax_a.transAxes,
              fontsize=5.0, color="#666666", va="top", ha="left", linespacing=1.3)
    panel_label(ax_a, "a", x=0.02, y=0.98)

    # (b) Biomass share robustness.
    # THIS PANEL WAS DRAWING NOTHING AT ALL, in three compounding ways:
    #   * the year was 2030, where BASE biomass share is 0.0000% and 13 of the 15 SA runs are
    #     also exactly 0 (only SA_carbon_high 5.02% and SA_discount_3pct 0.22% are nonzero);
    #   * `set_xlim(70, 100)` then put every one of those values off the left edge, so not a
    #     single bar could be rendered -- the panel was blank by construction;
    #   * the BASE reference was the literal `axvline(85)`. BASE 2030 biomass is 0%, not 85%.
    # The colour thresholds (>80 green, >70 yellow) were cut for that same phantom 85% scale.
    # Fixed by reading a year where the pathway exists, computing the BASE line from the
    # data, and letting the limits follow the values.
    all_sa = sorted([k for k in data if k.startswith("SA_")])
    bio_shares, sa_labels = [], []
    for sa in all_sa:
        s = _shares(data[sa], BIOMASS_PANEL_YEAR)
        bio_shares.append(s.get("biomass", 0) * 100)
        sa_labels.append(sa.replace("SA_", "").replace("_", " "))
    base_bio = _shares(_base(data), BIOMASS_PANEL_YEAR).get("biomass", 0) * 100

    if sa_labels:
        y_b = np.arange(len(sa_labels))
        span = max(max(bio_shares + [base_bio]) - min(bio_shares + [base_bio]), 1e-9)
        colors_b = ["#228833" if abs(b - base_bio) < 0.1 * span
                    else "#CCBB44" if abs(b - base_bio) < 0.3 * span
                    else "#CC3311" for b in bio_shares]
        ax_b.barh(y_b, bio_shares, color=colors_b, height=0.6,
                  edgecolor="white", linewidth=0.3)
        ax_b.axvline(base_bio, color="black", linewidth=0.5, linestyle="--")
        ax_b.text(base_bio, len(sa_labels) - 0.4, f" BASE {base_bio:.1f}%",
                  fontsize=6, va="top")
        ax_b.set_yticks(y_b)
        ax_b.set_yticklabels(sa_labels, fontsize=5.5)
        ax_b.set_xlabel(f"Biomass share {BIOMASS_PANEL_YEAR} (%)")
        ax_b.set_xlim(0, max(bio_shares + [base_bio]) * 1.25 + 1e-9)
    else:
        ax_b.text(0.5, 0.5, "No sensitivity scenarios available",
                  transform=ax_b.transAxes, ha="center", va="center", fontsize=8)
    panel_label(ax_b, "b", x=0.02, y=0.98)

    # (c) Capture wedge robustness.
    # THIS QUADRANT USED TO HOLD ONE SENTENCE. Ammonia co-firing is identically zero in every
    # scenario, so the panel fell through to its empty-state branch and spent a quarter of an
    # Extended Data figure printing "Ammonia never exceeds 0.5%" -- on a live 0-1 axes with
    # ticks, which read as a rendering failure. The bounded zero is worth ONE LINE, not a
    # quadrant, and it is now stated as such underneath.
    #
    # What goes here instead is the result that quadrant was hiding: the capture wedge does
    # not move. CCS + BECCS at 2050 is 14.2-15.0% of generation in FIFTEEN of the sixteen
    # runs -- across carbon prices spanning a factor of several, CCS capex high and low,
    # biomass cost, ammonia cost, discount rate 3-8%, pipeline corridor and replacement cost.
    # The single exception is halving storage injectivity, which cuts it to 8.7%. The wedge is
    # set by geology, not by economics, and that is a stronger statement than any of the
    # individual sensitivities in panel (a).
    ccs_rows = []
    for sa in all_sa:
        s = _shares(data[sa], 2050)
        ccs_rows.append((sa.replace("SA_", "").replace("_", " "),
                         s.get("ccs", 0) * 100, s.get("beccs", 0) * 100))
    base_s = _shares(_base(data), 2050)
    base_capture = (base_s.get("ccs", 0) + base_s.get("beccs", 0)) * 100

    if ccs_rows:
        ccs_rows.sort(key=lambda r: r[1] + r[2])
        c_labels = [r[0] for r in ccs_rows]
        c_ccs = np.array([r[1] for r in ccs_rows])
        c_beccs = np.array([r[2] for r in ccs_rows])
        y_c = np.arange(len(c_labels))
        ax_c.barh(y_c, c_ccs, height=0.6, color="#4477AA",
                  edgecolor="white", linewidth=0.3, label="CCS")
        ax_c.barh(y_c, c_beccs, left=c_ccs, height=0.6, color="#CCBB44",
                  edgecolor="white", linewidth=0.3, label="BECCS")
        ax_c.axvline(base_capture, color="black", linewidth=0.5, linestyle="--")
        ax_c.text(base_capture, len(c_labels) - 0.4, f" BASE {base_capture:.1f}%",
                  fontsize=6, va="top")
        ax_c.set_yticks(y_c)
        ax_c.set_yticklabels(c_labels, fontsize=5.5)
        ax_c.set_xlabel("CCS + BECCS share 2050 (%)")
        # Reserve a strip to the right of the longest bar for the legend; at the data limits
        # it sat on top of the "carbon low" row.
        ax_c.set_xlim(0, float((c_ccs + c_beccs).max()) * 1.28)
        ax_c.legend(fontsize=5.5, loc="lower right", frameon=False, handlelength=1.0,
                    labelspacing=0.25)
        totals = c_ccs + c_beccs
        others = np.sort(totals)[1:]
        ax_c.text(0.02, -0.22,
                  f"All but one run lie in {others.min():.1f}-{others.max():.1f}%; the "
                  f"outlier is halved storage\ninjectivity at {totals.min():.1f}%. Ammonia "
                  f"co-firing is identically zero in every run.",
                  transform=ax_c.transAxes, fontsize=5.0, color="#666666",
                  va="top", ha="left", linespacing=1.3)
    else:
        ax_c.text(0.5, 0.5, "No sensitivity scenarios available",
                  transform=ax_c.transAxes, ha="center", va="center", fontsize=8)
        ax_c.axis("off")
    panel_label(ax_c, "c", x=0.02, y=0.98)

    # (d) Retirement share 2050
    ret_data = []
    for sa in all_sa:
        s = _shares(data[sa], 2050)
        ret_data.append((sa.replace("SA_", "").replace("_", " "),
                         s.get("retire", 0) * 100))
    if ret_data:
        ret_data.sort(key=lambda x: x[1])
        ret_labels, ret_vals = zip(*ret_data)
        y_d = np.arange(len(ret_labels))
        # BASE line computed, not hardcoded. It was `axvline(74)`; BASE 2050 retirement is
        # 41.6%. Being 32 points out INVERTED the panel: every scenario appeared to retire
        # far less than BASE, when in fact they bracket it (36.5-57.4% around 41.6%).
        base_ret = _shares(_base(data), 2050).get("retire", 0) * 100
        colors_d = ["#999999" if r > base_ret else "#CC3311" for r in ret_vals]
        ax_d.barh(y_d, ret_vals, color=colors_d, height=0.6,
                  edgecolor="white", linewidth=0.3)
        ax_d.axvline(base_ret, color="black", linewidth=0.5, linestyle="--")
        ax_d.text(base_ret, len(ret_labels) - 0.4, f" BASE {base_ret:.1f}%",
                  fontsize=6, va="top")
        ax_d.set_yticks(y_d)
        ax_d.set_yticklabels(ret_labels, fontsize=5.5)
        ax_d.set_xlabel("Retirement share 2050 (%)")
    else:
        ax_d.text(0.5, 0.5, "No sensitivity scenarios available",
                  transform=ax_d.transAxes, ha="center", va="center", fontsize=8)
    panel_label(ax_d, "d", x=0.02, y=0.98)

    fig.tight_layout()
    save_fig(fig, "ed_fig3_sensitivity", subdir="extended")


# ── ED Fig 4: Provincial Choropleth ─────────────────────────────────────────

def ed_fig4_provincial_choropleth() -> None:
    """CCS+BECCS and retirement share maps (2050, EPSG:2380)."""
    try:
        import geopandas as gpd  # noqa: F811
    except ImportError:
        print("  [skip] geopandas not available — skipping ed_fig4")
        return

    provinces = _load_provinces()
    plant = pd.read_csv(BASE_DIR / "plant_detail.csv")
    yr50 = plant[plant["year"] == 2050]

    pw_names = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]
    share_cols = [f"share_{pw}" for pw in pw_names]

    rows = []
    for prov, grp in yr50.groupby("province_name"):
        cap = grp["capacity_mw"].values
        total = cap.sum()
        if total == 0:
            continue
        shares = {}
        for col, pw in zip(share_cols, pw_names):
            values = grp[col].values if col in grp else np.zeros(len(grp))
            shares[pw] = float((values * cap).sum() / total)
        rows.append({"province_en": prov, **shares})
    dom = pd.DataFrame(rows)

    legend_kwds = {"shrink": 0.6, "orientation": "horizontal", "pad": 0.03}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5))

    # (a) CCS+BECCS
    dom["ccs_beccs"] = dom["ccs"].fillna(0) + dom["beccs"].fillna(0)
    merged = provinces.merge(dom[["province_en", "ccs_beccs"]],
                             on="province_en", how="left")
    merged["ccs_beccs"] = merged["ccs_beccs"].fillna(0) * 100
    _choropleth(axes[0], merged, "ccs_beccs", "Blues",
                legend_kwds={**legend_kwds, "label": "CCS+BECCS share (%)"})
    _add_map_elements(axes[0])
    panel_label_inside(axes[0], "a")

    # (b) Retirement
    merged2 = provinces.merge(dom[["province_en", "retire"]],
                              on="province_en", how="left")
    merged2["retire"] = merged2["retire"].fillna(1.0) * 100
    _choropleth(axes[1], merged2, "retire", "Greys",
                legend_kwds={**legend_kwds, "label": "Retirement share (%)"})
    _add_map_elements(axes[1])
    panel_label_inside(axes[1], "b")

    fig.tight_layout()
    _scs_inset(fig, axes[0])
    _scs_inset(fig, axes[1])
    save_fig(fig, "ed_fig4_provincial_choropleth", subdir="extended")


# ── ED Fig 5: Fleet Age Map ────────────────────────────────────────────────

def ed_fig5_fleet_age_map() -> None:
    """Average retirement year by province (EPSG:2380)."""
    try:
        import geopandas as gpd  # noqa: F811
    except ImportError:
        print("  [skip] geopandas not available — skipping ed_fig5")
        return

    provinces = _load_provinces()
    plant = pd.read_csv(BASE_DIR / "plant_detail.csv")
    yr30 = plant[plant["year"] == 2030]

    prov_age = []
    for prov, grp in yr30.groupby("province_name"):
        cap = grp["capacity_mw"].values
        total = cap.sum()
        if total == 0:
            continue
        avg_ret = float((grp["retirement_year"].values * cap).sum() / total)
        prov_age.append({"province_en": prov, "avg_retirement_year": avg_ret})

    df_age = pd.DataFrame(prov_age)
    legend_kwds = {"shrink": 0.6, "orientation": "horizontal", "pad": 0.03,
                   "label": "Avg. design retirement year"}

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    merged = provinces.merge(df_age, on="province_en", how="left")
    merged["avg_retirement_year"] = merged["avg_retirement_year"].fillna(2040)
    _choropleth(ax, merged, "avg_retirement_year", "RdYlBu",
                vmin=2035, vmax=2060, legend_kwds=legend_kwds)
    _add_map_elements(ax)

    fig.tight_layout()
    _scs_inset(fig, ax)
    save_fig(fig, "ed_fig5_fleet_age_map", subdir="extended")


# ── ED Fig 6: Provincial Bars ───────────────────────────────────────────────

def ed_fig6_provincial_bars() -> None:
    """2-panel provincial stacked bars by region."""
    df = pd.read_csv(BASE_DIR / "province_pathways.csv")

    # Pivot: (year, province) → pathway shares
    pivoted = df.pivot_table(
        index=["year", "province_name"], columns="pathway",
        values="generation_share", fill_value=0,
    ).reset_index()

    # Province capacity, used ONLY to order provinces within each region.
    # MWh / h = MW, and MW / 1e3 = GW. The divisor was 1e6, so this variable held TW under a
    # name that says GW: Inner Mongolia came out at 0.132 when the correct figure is 132 GW.
    # The error never reached the canvas because the value is only used as a sort key, but a
    # 1 000x-wrong quantity named `cap_gw` is one refactor away from being plotted.
    gen_2030 = df[df["year"] == 2030].groupby("province_name")[
        "province_generation_mwh"].first()
    cap_gw = gen_2030 / (0.55 * 8760) / 1e3

    # Region grouping
    prov_to_region = {}
    for region, provs in REGIONS.items():
        for p in provs:
            prov_to_region[p] = region

    region_order = ["North", "Northeast", "East", "South-Central",
                    "Southwest", "Northwest"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 5.5))

    for ax, year_pair, panel_letter in [
        (ax_a, (2030, 2040), "a"),
        (ax_b, (2050, 2060), "b"),
    ]:
        yr1, yr2 = year_pair
        d1 = pivoted[pivoted["year"] == yr1].set_index("province_name")
        d2 = pivoted[pivoted["year"] == yr2].set_index("province_name")
        all_provs = sorted(
            set(d1.index) | set(d2.index),
            key=lambda p: (region_order.index(prov_to_region.get(p, "Northwest"))
                           if prov_to_region.get(p) in region_order else 99,
                           -cap_gw.get(p, 0)),
        )

        y_pos = np.arange(len(all_provs))
        bar_h = 0.35
        gap = 0.05
        # CENTRE THE PAIR ON ITS TICK. The offsets were `y_pos - gap` and `y_pos + bar_h`,
        # i.e. -0.05 and +0.35, so the two-bar group spanned -0.225 to +0.525 about the tick
        # -- displaced downwards by 0.15 of a row. Every province label therefore sat low
        # inside its own first bar and close to the second, which is what made the pairing
        # ambiguous. A symmetric +/- (bar_h + gap) / 2 puts the label between the two bars.
        half = (bar_h + gap) / 2.0

        for pw in PATHWAY_ORDER:
            vals1 = [d1.at[p, pw] * 100 if p in d1.index and pw in d1.columns
                     else 0 for p in all_provs]
            vals2 = [d2.at[p, pw] * 100 if p in d2.index and pw in d2.columns
                     else 0 for p in all_provs]

            left1 = np.zeros(len(all_provs))
            left2 = np.zeros(len(all_provs))
            for prev_pw in PATHWAY_ORDER:
                if prev_pw == pw:
                    break
                left1 += np.array([
                    d1.at[p, prev_pw] * 100
                    if p in d1.index and prev_pw in d1.columns else 0
                    for p in all_provs
                ])
                left2 += np.array([
                    d2.at[p, prev_pw] * 100
                    if p in d2.index and prev_pw in d2.columns else 0
                    for p in all_provs
                ])

            ax.barh(y_pos - half, vals1, height=bar_h, left=left1,
                    color=PATHWAY_COLORS[pw], edgecolor="white",
                    linewidth=0.2)
            ax.barh(y_pos + half, vals2, height=bar_h, left=left2,
                    color=PATHWAY_COLORS[pw], edgecolor="white",
                    linewidth=0.2, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_provs, fontsize=5.5)
        ax.set_xlabel("Share (%)")
        ax.set_xlim(0, 105)
        ax.invert_yaxis()

        # Region separators
        prev_region = None
        for i, p in enumerate(all_provs):
            r = prov_to_region.get(p, "")
            if r != prev_region and prev_region is not None:
                ax.axhline(i - 0.5, color="#cccccc", linewidth=0.5)
            prev_region = r

        # Name the pairing, and say when a uniform row is a RESULT rather than a failure.
        # Every province is 100% unabated at 2030, so panel (a)'s first bar is a solid grey
        # band in all 28 rows; without a note that reads as a rendering fault.
        note = f"Top: {yr1} | Bottom: {yr2}"
        uniform = [
            yr for yr in (yr1, yr2)
            if yr in set(pivoted["year"])
            and float(pivoted[pivoted["year"] == yr]["unabated"].min()) > 0.995
        ]
        if uniform:
            note += (f"\n{', '.join(str(y) for y in uniform)}: every province is "
                     f"100% unabated, so that row is uniform by result")
        ax.text(0.98, 0.02, note,
                transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom",
                style="italic", linespacing=1.3)
        panel_label(ax, panel_letter, x=-0.12, y=1.03)

    pathway_legend(ax_a, ncol=5, loc="upper center", bbox=(1.0, 1.10))
    fig.tight_layout()
    save_fig(fig, "ed_fig6_provincial_bars", subdir="extended")


# ── ED Fig 7: Distance vs CCS ──────────────────────────────────────────────

def ed_fig7_distance_vs_ccs() -> None:
    """Scatter: distance to storage vs CCS+BECCS share (2050).

    THE PANEL'S PREMISE IS NOT SUPPORTED BY ITS OWN DATA, and it now says so.
    Across the 89 plants with a finite storage distance (379 GW), the correlation between
    distance and captured share is r = +0.08 unweighted and r = -0.04 capacity-weighted, and
    the quartile means run 32 / 34 / 47 / 39% -- not monotone in either direction. 47 of the
    89 sit at exactly 0% and 25 at exactly 100%, so the cloud is bimodal rather than graded.
    Distance to storage is not what decides who captures in this model.

    That is worth showing -- it is the reason Fig 4's pipeline core is a no-regret set rather
    than a distance-ranked queue -- but only if the panel states the null instead of leaving
    a reader to infer a trend from a scatter that has none.
    """
    plant = pd.read_csv(BASE_DIR / "plant_detail.csv")
    yr50 = plant[plant["year"] == 2050].copy()
    yr50["ccs_beccs"] = yr50["share_ccs"] + yr50["share_beccs"]
    yr50 = yr50.dropna(subset=["min_distance_to_storage_km"])
    yr50 = yr50[yr50["min_distance_to_storage_km"] > 0]

    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    for pw in ["retire", "ccs", "biomass", "beccs"]:
        sub = yr50[yr50["dominant_pathway"] == pw]
        if len(sub) == 0:
            continue
        ax.scatter(sub["min_distance_to_storage_km"],
                   sub["ccs_beccs"] * 100,
                   s=sub["capacity_mw"] / 80,
                   c=PATHWAY_COLORS[pw], alpha=0.6,
                   edgecolors="black", linewidth=0.2,
                   label=PATHWAY_LABELS[pw])

    distance = yr50["min_distance_to_storage_km"].to_numpy(dtype=float)
    share = yr50["ccs_beccs"].to_numpy(dtype=float) * 100.0
    weights = yr50["capacity_mw"].to_numpy(dtype=float)
    r_plain = float(np.corrcoef(distance, share)[0, 1])
    mean_d = np.average(distance, weights=weights)
    mean_s = np.average(share, weights=weights)
    cov = np.average((distance - mean_d) * (share - mean_s), weights=weights)
    r_weighted = float(cov / np.sqrt(np.average((distance - mean_d) ** 2, weights=weights)
                                     * np.average((share - mean_s) ** 2, weights=weights)))

    # Quartile means drawn as a step, so the absence of a trend is visible and not merely
    # asserted in a caption.
    edges = np.quantile(distance, [0.0, 0.25, 0.5, 0.75, 1.0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (distance >= lo) & (distance <= hi)
        if sel.sum():
            ax.plot([lo, hi], [share[sel].mean()] * 2, color="#333333", lw=1.4,
                    solid_capstyle="butt", zorder=6)
    ax.plot([], [], color="#333333", lw=1.4, label="Quartile mean")

    ax.set_xlabel("Distance to nearest storage (km)")
    ax.set_ylabel("CCS + BECCS share of the plant, 2050 (%)")
    ax.set_ylim(-6, 118)
    ax.set_xlim(-15, float(distance.max()) * 1.06)
    ax.set_title("Distance to storage does not decide who captures", fontsize=8)
    # Legend ABOVE the data, not inside it. At "best" matplotlib placed the four key bubbles
    # on top of real plants near (400 km, 0-10%), so four legend markers were sitting in the
    # data region and being read as observations.
    ax.legend(fontsize=6.5, markerscale=1.2, loc="upper center", ncol=5,
              frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.1,
              handletextpad=0.4)
    # Marker area is capacity and was never keyed. Three reference bubbles, drawn clear of
    # the cloud in the lower right.
    for i, gw in enumerate((1.0, 4.0, 8.0)):
        ax.scatter([float(distance.max()) * (0.80 + 0.06 * i)], [88],
                   s=gw * 1000.0 / 80.0, facecolors="none", edgecolors="#555555",
                   linewidth=0.5, zorder=6)
        ax.text(float(distance.max()) * (0.80 + 0.06 * i), 78, f"{gw:.0f}",
                ha="center", va="top", fontsize=5.4, color="#555555")
    ax.text(float(distance.max()) * 0.86, 72, "plant capacity (GW)",
            ha="center", va="top", fontsize=5.4, color="#555555")

    # Mid-left, not the bottom edge: the 0% row is the densest part of the cloud (47 of 89
    # plants) and the note was printed straight through it.
    ax.text(0.015, 0.30,
            f"n = {len(yr50)} plants, {weights.sum() / 1e3:.0f} GW.  "
            f"r = {r_plain:+.2f} unweighted, {r_weighted:+.2f} capacity-weighted; "
            f"quartile means {', '.join(f'{share[(distance >= lo) & (distance <= hi)].mean():.0f}%' for lo, hi in zip(edges[:-1], edges[1:]))}.\n"
            f"{int((share < 1).sum())} plants sit at 0% and {int((share > 99).sum())} at 100%, "
            f"so the distribution is bimodal rather than graded in distance.",
            transform=ax.transAxes, fontsize=5.4, color="#555555",
            va="top", ha="left", linespacing=1.35)

    fig.tight_layout()
    save_fig(fig, "ed_fig7_distance_vs_ccs", subdir="extended")


# ── ED Fig 8: Storage Utilization ───────────────────────────────────────────

def ed_fig8_storage_utilization() -> None:
    """2-panel: (a) injectivity utilization bars, (b) capacity vs injection."""
    storage = pd.read_csv(BASE_DIR / "storage_utilization.csv")
    yr50 = storage[storage["year"] == 2050].copy()
    yr50["util_pct"] = yr50["injectivity_utilization"] * 100

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # (a) Sorted utilization bars
    yr50_sorted = yr50.sort_values("util_pct", ascending=True)
    yr50_active = yr50_sorted[yr50_sorted["util_pct"] > 0]
    if len(yr50_active) > 0:
        y_a = np.arange(len(yr50_active))
        colors_u = ["#CC3311" if u >= 99 else "#CCBB44" if u >= 80
                     else "#4477AA" for u in yr50_active["util_pct"]]
        ax_a.barh(y_a, yr50_active["util_pct"], color=colors_u, height=0.6,
                  edgecolor="white", linewidth=0.3)
        hub_labels = yr50_active["storage_hub_id"].astype(str)
        if "province" in yr50_active.columns:
            hub_labels = yr50_active["province"].fillna("") + " " + hub_labels
        ax_a.set_yticks(y_a)
        ax_a.set_yticklabels(hub_labels, fontsize=5)
        ax_a.set_xlabel("Injectivity utilization (%)")
        ax_a.axvline(100, color="black", linewidth=0.5, linestyle=":")

        n_sat = (yr50_active["util_pct"] >= 99).sum()
        ax_a.text(0.98, 0.02, f"{n_sat} hubs saturated",
                  transform=ax_a.transAxes, fontsize=6, ha="right",
                  va="bottom", fontweight="bold", color="#CC3311")
    panel_label(ax_a, "a")

    # (b) Capacity vs injection rate scatter
    if len(yr50) > 0:
        ax_b.scatter(yr50["available_capacity_mt"] / 1000,
                     yr50["storage_use_mtpa"],
                     s=40, c=yr50["util_pct"], cmap="RdYlBu_r",
                     vmin=0, vmax=100,
                     edgecolors="black", linewidth=0.3, alpha=0.8)
        ax_b.set_xlabel("Available capacity (Gt)")
        ax_b.set_ylabel("Injection rate (MtCO$_2$/yr)")
        ax_b.set_xscale("log")
        cb = plt.colorbar(ax_b.collections[0], ax=ax_b, shrink=0.7,
                          pad=0.02)
        cb.set_label("Utilization (%)", fontsize=7)
    panel_label(ax_b, "b")

    fig.tight_layout()
    save_fig(fig, "ed_fig8_storage_utilization", subdir="extended")


# ── ED Fig 9: Stranded Assets ──────────────────────────────────────────────

def ed_fig9_stranded_assets() -> None:
    """3-panel: (a) retirement wave, (b) early retirement years, (c) provincial value."""
    plant = pd.read_csv(BASE_DIR / "plant_detail.csv")

    # 6.8 in rather than 7.0: the assumption note under panel (c) is wider than its axes,
    # and bbox_inches='tight' grows the canvas to include it -- the saved file came out at
    # 186.6 mm, over the 183 mm limit, even though the nominal width was inside it.
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(6.8, 3.5))

    # (a) Retirement wave.
    # `share_retire` IS A STOCK, NOT A FLOW. It is the fraction of a plant retired AS OF that
    # snapshot year, and the model constrains it to be monotone (verified: 0 of 350 plants
    # ever decrease). The previous version took np.cumsum of it and labelled the result
    # "Cumulative", which added the 2050 stock to the 2060 stock and counted every plant
    # retired before 2050 four times over. It printed 1 344 GW cumulative against a national
    # fleet of 1 416 GW -- i.e. "almost the entire fleet shuts" -- when the true 2060 retired
    # stock is 770 GW. The overstatement was 1.75x. It also drew the same stock twice, once
    # as the "Period" bar and once as the "Cumulative" line, under two different names.
    stock_gw = []
    for yr in YEARS:
        yr_data = plant[plant["year"] == yr]
        stock_gw.append((yr_data["share_retire"] * yr_data["capacity_mw"]).sum() / 1000)
    cumulative = np.asarray(stock_gw, dtype=float)
    period_gw = np.diff(cumulative, prepend=0.0)   # the genuine per-period increment
    ax_a.bar(YEARS, period_gw, width=8, color="#999999",
             edgecolor="white", linewidth=0.3, label="Added in period")
    ax_a.plot(YEARS, cumulative, "ko-", markersize=4, linewidth=1.0,
              label="Retired stock")
    for i, (y, c) in enumerate(zip(YEARS, cumulative)):
        ax_a.text(y, c + 5, f"{c:.0f}", fontsize=6, ha="center",
                  fontweight="bold")
    # Pin the ticks to the snapshot years. The default locator dropped 2030 and 2050,
    # leaving a four-point series labelled at two of its points.
    ax_a.set_xticks(YEARS)
    ax_a.set_xticklabels([str(y) for y in YEARS], fontsize=6)
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Retired capacity (GW)")
    ax_a.legend(fontsize=6)
    panel_label(ax_a, "a")

    # (b) Early retirement years distribution (weighted by capacity)
    # Use 2050 data: compare retirement_year vs actual operation end
    yr50 = plant[plant["year"] == 2050].copy()
    retired = yr50[yr50["share_retire"] > 0.5].copy()
    if "retirement_year" in retired.columns:
        retired["years_early"] = retired["retirement_year"] - 2050
        retired["years_early"] = retired["years_early"].clip(lower=0)
        # capacity_mw x share_retire is MW; the axis says GW, so convert here. Previously
        # the histogram ran to 375 000 on an axis labelled GW, which is 250x world capacity.
        weights = retired["capacity_mw"] * retired["share_retire"] / 1000.0

        if len(retired) > 0 and weights.sum() > 0:
            ax_b.hist(retired["years_early"], bins=15, weights=weights,
                      color="#CC3311", alpha=0.7, edgecolor="white",
                      linewidth=0.3)
            wmean = np.average(retired["years_early"],
                               weights=weights.values)
            ax_b.axvline(wmean, color="black", linewidth=1, linestyle="--")
            ax_b.text(wmean + 0.5, ax_b.get_ylim()[1] * 0.85,
                      f"Mean: {wmean:.0f} yr", fontsize=6,
                      fontweight="bold")

    ax_b.set_xlabel("Years before design retirement")
    ax_b.set_ylabel("Capacity (GW)")
    panel_label(ax_b, "b")

    # (c) Provincial stranded value
    # Stranded value = years_early × capacity × 3500 CNY/kW / 20yr lifetime
    if "retirement_year" in plant.columns:
        all_stranded = []
        for yr in [2030, 2040, 2050]:
            yr_data = plant[plant["year"] == yr].copy()
            ret = yr_data[yr_data["share_retire"] > 0.5].copy()
            ret["years_early"] = (ret["retirement_year"] - yr).clip(lower=0)
            ret["stranded_b"] = (ret["years_early"] * ret["capacity_mw"]
                                 * ret["share_retire"] * 3500 / 20) / 1e6
            all_stranded.append(ret[["province_name", "stranded_b"]])

        stranded_df = pd.concat(all_stranded, ignore_index=True)
        prov_stranded = stranded_df.groupby("province_name")["stranded_b"].sum()
        # Drop provinces with essentially nothing stranded. `tail(10)` took the top ten by
        # value even when the tenth was 0.0, so Xinjiang appeared as a labelled row with a
        # zero-length bar -- indistinguishable from missing data.
        prov_stranded = prov_stranded[prov_stranded > 0.05].sort_values(
            ascending=True).tail(10)

        y_c = np.arange(len(prov_stranded))
        ax_c.barh(y_c, prov_stranded.values, color="#CC3311", height=0.6,
                  edgecolor="white", linewidth=0.3)
        ax_c.set_yticks(y_c)
        ax_c.set_yticklabels(prov_stranded.index, fontsize=6)
        ax_c.set_xlabel("Stranded value (B CNY)")
        for i, v in enumerate(prov_stranded.values):
            ax_c.text(v + 0.5, i, f"{v:.0f}" if v >= 1 else f"{v:.1f}",
                      fontsize=5, va="center")
        # Both assumptions behind this quantity, stated on the panel: neither is derived from
        # the model. The 0.5 threshold also makes the measure discontinuous -- a plant at
        # share_retire = 0.51 contributes 51% of its capacity and one at 0.49 contributes
        # nothing.
        ax_c.text(0.02, -0.30,
                  "Assumed 3 500 CNY/kW replacement value over a 20-year" + "\n" +
                  " residual life; counts plants with share_retire > 0.5 only.",
                  transform=ax_c.transAxes, fontsize=5.0, color="#666666",
                  va="top", ha="left", linespacing=1.3)

    panel_label(ax_c, "c")

    fig.tight_layout()
    save_fig(fig, "ed_fig9_stranded_assets", subdir="extended")


# ── ED Fig 10: Pathway Restrictions ─────────────────────────────────────────

def ed_fig10_pathway_restrictions(data: dict) -> None:
    """Key scenario pathway allocation: 3-year comparison."""
    # SCENARIO LIST. "Water constraint" was `WA_grid_200km`, a superseded 200 km grid-supply
    # variant with no reservation and on the annual rather than the dry-season basis -- not
    # the treatment any main figure uses. Replaced by the two sides of the headline contrast,
    # so this panel shows the same comparison Fig 3 and Fig 5 do.
    scenarios = [
        ("BASE", "BASE", "BASE"),
        ("Net zero", "BASE_zero", None),
        ("Net negative", "BASE_neg", None),
        ("No ammonia", "RQ3_no_ammonia", None),
        ("No biomass", "RQ3_no_biomass", None),
        ("No CCS", "RQ3_no_ccs", None),
        ("Retire only", "RQ3_retire_only", None),
        ("CCS/BECCS only", "RQ3_ccs_only", None),
        ("Water accounted", "WA_cwatm_126_dry", None),
        ("Water binds", "WA_cwatm_126_dry_wd085", None),
    ]
    scenarios = [
        item for item in scenarios
        if item[1] in data or (item[2] is not None and item[2] in data)
    ]
    compare_years = [2030, 2040, 2050]

    # 8.6 in = 218 mm; bbox_inches='tight' brought it to 213 mm, over the 183 mm limit.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.8), sharey=True)

    for j, yr in enumerate(compare_years):
        ax = axes[j]
        x = np.arange(len(scenarios))
        width = 0.72
        bottoms = np.zeros(len(scenarios))
        # Collect first, label after: the in-bar numerals used to be drawn inside the pathway
        # loop with a bare `if v > 8` test, so with ten scenarios at 5.5 pt across a 60 mm
        # axis every column printed its own numerals into its neighbours -- the 2040 panel
        # read "27272727%" and the 2030 panel "00000000". A segment now has to be both tall
        # enough to hold text AND wide enough that the text fits inside its own column.
        segments: list[tuple[int, float, float, str]] = []

        for pw in PATHWAY_ORDER:
            vals = []
            for _, key1, key2 in scenarios:
                sc = data.get(key1) or (data.get(key2) if key2 else None)
                if sc is None:
                    vals.append(0)
                    continue
                s = clean_shares(_shares(sc, yr))
                vals.append(s.get(pw, 0) * 100)
            ax.bar(x, vals, width, bottom=bottoms,
                   color=PATHWAY_COLORS[pw], edgecolor="white",
                   linewidth=0.3)
            # LABEL THE REFERENCE AND THE DIFFERENCES, NOT ALL SIXTY SEGMENTS.
            # Ten columns share ~55 mm at print size, so a numeral in every segment produced
            # runs of identical adjacent values that read as one string: "100100100100" in
            # 2030, "27272727" in 2040, "41414141" in 2050. The shape comparison is what this
            # panel is for; the numbers that earn ink are BASE (the reference every other
            # column is read against) and any column that actually departs from it.
            base_v = vals[0] if vals else 0.0
            for i, v in enumerate(vals):
                if v < 12.0:
                    continue
                if i == 0 or abs(v - base_v) >= 3.0:
                    segments.append((i, bottoms[i] + v / 2.0, v,
                                     "white" if v > 25 else "black"))
            bottoms += np.array(vals)

        for i, y_mid, v, colour in segments:
            ax.text(x[i], y_mid, f"{v:.0f}", ha="center", va="center",
                    fontsize=5.0, fontweight="bold", color=colour)

        # A degenerate solve is not a fleet composition. RQ3_retire_only is zero-filled and
        # drew as an empty column indistinguishable from "everything retired".
        for i, (_, key1, key2) in enumerate(scenarios):
            sc = data.get(key1) or (data.get(key2) if key2 else None)
            if sc is None:
                continue
            statuses = {str(p.get("status", "")) for p in sc.get("years", {}).values()
                        if isinstance(p, dict)}
            if bottoms[i] < 1.0 or (statuses and statuses != {"optimal"}):
                ax.text(x[i], 50, "not\nsolved", ha="center", va="center",
                        fontsize=5.0, color="#AA3333", fontstyle="italic",
                        linespacing=1.1, rotation=90)

        ax.set_xticks(x)
        # Vertical, not 40 degrees. Ten labels rotated at 40 degrees across a ~55 mm panel
        # overlapped each other into an unreadable band; at 90 degrees each label occupies
        # only its own column's width.
        ax.set_xticklabels([s[0] for s in scenarios], fontsize=5.0,
                           rotation=90, ha="center", va="top")
        ax.set_title(str(yr), fontsize=9, fontweight="bold")
        ax.set_ylim(0, 105)
        # -0.85 rather than -0.7: the BASE column's numeral is centred on x = 0 and was
        # being clipped by the y-axis spine.
        ax.set_xlim(-0.85, len(scenarios) - 0.3)
        if j == 0:
            # The model allocates GENERATION, not capacity: `pathway_shares` is
            # generation-weighted (solver.py builds it from annual_generation_mwh), so the
            # axis said "Capacity share" for a quantity that is nothing of the kind.
            ax.set_ylabel("Generation share (%)")
        panel_label(ax, chr(ord("a") + j),
                    x=-0.12 if j == 0 else -0.05, y=1.05)

    pathway_legend(axes[1], ncol=6, loc="upper center", bbox=(0.5, 1.16))
    fig.tight_layout()
    save_fig(fig, "ed_fig10_pathway_restrictions", subdir="extended")


# ── ED Fig 11: CCS-Only Cost ───────────────────────────────────────────────

def ed_fig11_ccs_only_cost(data: dict) -> None:
    """CCS-only vs BASE cost structure (2050)."""
    cost_keys = [
        ("Baseline net", "baseline_net_cost"),
        ("Carbon cost", "carbon_cost"),
        ("Energy penalty", "energy_penalty_cost"),
        ("CCS O&M", "ccs_om_cost"),
        ("Incremental O&M", "incremental_om"),
        ("Biomass fuel", "biomass_cost"),
        ("Transport OPEX", "transport_opex"),
        ("Storage cost", "storage_cost"),
        ("CCS CAPEX", "ccs_retrofit_capex"),
        ("Pipeline CAPEX", "pipe_capex"),
        # THE COMPONENT THAT MAKES THIS RUN INFEASIBLE HAS TO BE ON THE CHART.
        # RQ3_ccs_only carries 252.8e12 CNY of big-M penalty on demand it could not serve --
        # 91.4% of its whole objective, and 45.8% of its 2050 objective alone. The ten
        # components above sum to 64.8% of the 2050 objective, so the figure previously drew
        # a "cost structure" that omitted the single largest term in it and left the reader
        # to read the remainder as if it were the cost of a CCS-only strategy. It is not a
        # cost; it is the price the model was charged for not meeting demand.
        ("Unserved-demand penalty", "slack_penalty"),
        # Negative since 2026-09-22 (end-of-horizon salvage on capex); absent in older runs.
        ("Salvage credit", "salvage_credit"),
    ]

    base_sc = _base(data)
    ccs_sc = data.get("RQ3_ccs_only")
    if ccs_sc is None:
        print("  [skip] RQ3_ccs_only not found — skipping ed_fig11")
        return

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    y = np.arange(len(cost_keys))
    width = 0.35

    for offset, (sc, label, color) in enumerate([
        (base_sc, "BASE", "#666666"),
        (ccs_sc, "CCS-only", "#4477AA"),
    ]):
        cbd = _cost_bd(sc, 2050)
        vals = [cbd.get(key, 0) / 1e9 for _, key in cost_keys]
        ax.barh(y + (offset - 0.5) * width, vals, height=width,
                color=color, edgecolor="white", linewidth=0.3,
                label=label, alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([k[0] for k in cost_keys], fontsize=6.5)
    ax.set_xlabel("Cost (B CNY)")
    ax.axvline(0, color="black", linewidth=0.4)
    ax.legend(fontsize=7)

    # Mark infeasible
    if ccs_sc.get("infeasible", False):
        ax.text(0.98, 0.98, "CCS-only: INFEASIBLE", transform=ax.transAxes,
                fontsize=8, fontweight="bold", color="#CC3311",
                ha="right", va="top")

    fig.tight_layout()
    save_fig(fig, "ed_fig11_ccs_only_cost", subdir="extended")


# ── ED Fig 12: Plant Cost Map ──────────────────────────────────────────────

def ed_fig12_plant_cost_map() -> None:
    """Plant-level cost map (2050, EPSG:2380)."""
    try:
        import geopandas as gpd  # noqa: F811
    except ImportError:
        print("  [skip] geopandas not available — skipping ed_fig12")
        return

    cost = pd.read_csv(BASE_DIR / "plant_cost.csv")
    detail = pd.read_csv(BASE_DIR / "plant_detail.csv")

    yr50_cost = cost[cost["year"] == 2050]
    yr50_detail = detail[detail["year"] == 2050][
        ["plant_id", "centroid_longitude", "centroid_latitude", "capacity_mw"]
    ]
    yr50 = yr50_cost.merge(yr50_detail, on="plant_id", how="left",
                           suffixes=("", "_d")).dropna(
        subset=["centroid_longitude"])
    if "capacity_mw" not in yr50.columns and "capacity_mw_d" in yr50.columns:
        yr50["capacity_mw"] = yr50["capacity_mw_d"]
    yr50["total_b"] = yr50["total_plant_cost_cny"] / 1e9

    yr50_proj = _reproj(yr50, "centroid_longitude", "centroid_latitude")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _basemap(ax)

    sc = ax.scatter(
        yr50_proj["px"], yr50_proj["py"],
        s=yr50_proj["capacity_mw"] / 80,
        c=yr50_proj["total_b"], cmap="RdYlBu_r", alpha=0.7,
        edgecolors="black", linewidth=0.3, vmin=-5, vmax=20, zorder=5,
    )
    plt.colorbar(sc, ax=ax, label="Net cost (B CNY)", shrink=0.6, pad=0.02,
                 orientation="horizontal")
    _add_map_elements(ax)

    fig.tight_layout()
    _scs_inset(fig, ax)
    save_fig(fig, "ed_fig12_plant_cost_map", subdir="extended")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading experiment results...")
    data = _load_json()

    print("Generating Extended Data figures...")
    ed_fig1_biomass_feasibility()
    ed_fig2_emissions_trajectory(data)
    ed_fig3_sensitivity(data)

    # Map figures (need geopandas)
    ed_fig4_provincial_choropleth()
    ed_fig5_fleet_age_map()

    ed_fig6_provincial_bars()
    ed_fig7_distance_vs_ccs()
    ed_fig8_storage_utilization()
    ed_fig9_stranded_assets()
    ed_fig10_pathway_restrictions(data)
    ed_fig11_ccs_only_cost(data)
    ed_fig12_plant_cost_map()

    print(f"\nAll ED figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
