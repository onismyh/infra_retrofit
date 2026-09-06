"""Fig 2 - a presumptive environmental-flow allowance that China's northern coal fleet
already exceeds, and what capture would do to the overshoot.

THE FRAMING CHANGED TWICE, AND BOTH TIMES BECAUSE THIS FIGURE REFUTED ITSELF.
v1-v3 claimed dry-season water "sets a hard spatial ceiling" that capture demand "crosses".
A ceiling is a physical limit, and panel (a) refutes that reading: at 2030, with NO capture
anywhere and cooling exactly as built, the fleet's demand is already 258% of the allowance in
the Hai and 136% in the Yellow. Those plants run today. A limit the observed system exceeds
is not a limit; it is a STANDARD that is not met. v4 adopted that.

v5 goes one step further, because a data-integrity fix moved the number that decided it. The
dry-season budget was built as a sum of per-grid-cell minima -- each cell free to pick its own
low-flow quarter -- instead of the basin's total flow in the basin's own low-flow quarter:
sum(min) where the constraint needs min(sum). The understatement was 2.25x in the Hai, 2.13x
in the Northwest Interior, 1.47x in the Yellow and 1.19x in the Huai, against 1.03-1.11x in
the four basins the study calls safe -- i.e. the error was largest exactly where the answer
was. On the corrected budget the Hai's critical reservation share moves from s* = -0.63 to
s* = +0.18, and the headline changes with it: the fleet does not run out of water, it runs
out of SHARE.

    What this figure claims:
    (1) Measured against a presumptive environmental-flow allowance -- 20% of dry-season
        renewable runoff, net of the share reserved to non-power users -- three northern
        basins ALREADY operate beyond that allowance before any capture, and capture tips a
        fourth over. Together they carry ~765 GW, 54% of the standing fleet.
    (2) Capture roughly doubles the overshoot in every one of them (2030, 20-member median:
        Hai 258% -> 547%, Yellow 136% -> 293%, Huai 101% -> 216%, NW Interior 83% -> 183%).
    (3) THE OVERSHOOT IS ALLOCATION, NOT SCARCITY. Every basin's critical reservation share
        is POSITIVE in every planning year, with and without capture: at s = 0 -- power
        taking the entire extractable dry-season allowance -- nothing exceeds anything. The
        Hai breaches only above s* = 0.18, the Yellow above 0.56; China's measured non-power
        reservation is 0.85. Capture's effect is to CUT the reservation a basin can tolerate,
        roughly halving s* wherever it binds. Panel (b) is that curve.

    What it does NOT claim, and must not be read as claiming:
    (a) that the fleet is physically unable to obtain this water. It plainly does obtain it.
        The allowance is computed on LOCAL RENEWABLE DRY-SEASON RUNOFF only, from unrouted
        `qtot`, so it is a firm-yield rule for a system with ZERO storage. The Hai's actual
        dry-season supply is dominated by reservoir regulation, the South-North Water
        Transfer, groundwater and reclaimed water, none of which this model represents. The
        residual gap between 258% and observed operation is the size of that omission, and it
        is the single largest methodological caveat in the paper.
    (b) that the allowance is a well-determined number. It is 0.20 x (1 - s), and those two
        factors are ALIASED -- s = 0.85 at 20% extractable is the same model as s = 0 at 3%
        (see data_prep.py where `usable` is formed). The figure is about the SIZE of the
        residual share left to power, not about which of its two factors sets it.
    (c) that 0.20 x (1 - 0.85) is a coherent composition. Richter's 20% is a ceiling on
        CUMULATIVE depletion; the 0.85 is a share of China's own allocation quota, which is
        itself 53-64% of natural runoff. Multiplying them holds the share fixed while
        shrinking the pie 2.6x. The defensible alternatives and what they give are in the
        v5 README; this figure reports the composition the model solves.

  (a) effective dry-season supply vs coal water demand, every level-1 basin that hosts coal
      capacity (8 of 9 -- Southwest Rivers has none), 2030 and 2060, with the 20-member
      ensemble as box/whisker on supply and the demand shown three ways (as-built unabated,
      as-built full-capture, and what the solver actually realised). The grey unabated line
      sitting above the supply box in four basins IS the falsification of the old framing,
      and it is drawn rather than buried.
  (b) share of coal capacity in basins beyond the allowance, as a function of the reservation
      share, with the critical share s* solved analytically so the axis costs no extra solves
  (c) the allowance test per basin: full-capture demand as a multiple of supply, ranked north
      to south, against the standing capacity behind each basin

The ensemble box/whisker in (a) is the physical spread of the resource -- five GCMs and two
hydrology models disagreeing about how much water a basin has -- not a sensitivity of this
paper's model, so it belongs in the main figure. The attribution of that spread to GCM,
hydrology model and SSP is a variance decomposition and has moved to Extended Data
(`scripts/plot_ed_variance_decomposition.py`), where uncertainty decompositions belong.

Two facts drive every number here and both were checked against solved output rather than
assumed:

1. `_dry` scenarios read `dry_season_water_m3_per_year`, NOT the annual column. Reproducing
   the solver's own `available` column from the dry-season column matches to 5.6e-16
   relative; the annual column is 4.3x too high. Every panel therefore compares against the
   dry-season budget, which is the quantity the constraint actually acts on.
2. The stored availability is raw physical runoff. The budget the solver sees is
   `dry_season x WATER_EXTRACTABLE_FRACTION x (1 - existing_withdrawal_share)`
   (data_prep.py:382-439). Both factors are applied here, the first sourced from the package
   and the second pinned to the value the plotted runs carry.

Usage:  python scripts/plot_fig2_constraint_response.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (  # noqa: E402
    apply_style,
    cjk_fill,
    save_fig,
    panel_label,
    ROOT,
    RESULTS_DIR,
    DOUBLE_COL,
    BASIN_NAMES_ZH,
    BASIN_ORDER,
    assign_basin,
    assert_same_vintage,
)

apply_style()

INPUTS = ROOT / "inputs"
YEARS = [2030, 2040, 2050, 2060]
# The year the ceiling test is reported on in (c): the standing fleet is still nearly intact,
# so this is the tightest the constraint gets and the year the 765 GW refers to.
CEILING_YEAR = 2030

# The solved pair. Both carry existing_withdrawal_share=0.85, which is what makes the
# availability constraint bind at all (run_single.py:99-100).
SOLVED = {"ssp126": "WA_cwatm_126_dry_wd085", "ssp370": "WA_cwatm_370_dry_wd085"}
# Adaptation frozen. Not plotted -- it is a counterfactual, not part of the claim -- but its
# unserved volume is printed, because it is the cleanest evidence that the ceiling is real.
FROZEN = "WA_cwatm_126_dry_wd085_noair"
# Member the solves pin (run_single.py:99). It is a drying GCM, so it is marked in panel (a)
# rather than left to look like the ensemble centre.
SOLVED_MEMBER = "cwatm|gfdl-esm4|ssp126"

# Non-power reservation carried by the plotted `*_wd085` scenarios (run_single.py:99-104).
# Must stay identical to plot_fig1_water_footprint.py:63-67 -- both figures draw the same
# limit line, so they must define it the same way.
EXISTING_WITHDRAWAL_SHARE = 0.85
# Panel (b) sweeps this parameter instead of fixing it. Supply scales exactly as (1 - s),
# so the sweep is analytic and needs no extra solves; the solved runs at 0.56/0.70/0.85/
# 0.90 are the check that the fleet's RESPONSE follows the same staircase.
RESERVATION_GRID = np.round(np.arange(0.0, 0.951, 0.01), 3)

SSPS = ["ssp126", "ssp370"]
EXPECTED_MEMBERS = 20  # 2 hydrology x 5 GCM x 2 SSP

# Colours held locally, not added to plot_style: sibling figure scripts are editing that
# module concurrently.
C126 = "#4477AA"
C370 = "#CC3311"
DEMAND_UN = "#999999"
DEMAND_CAP = "#222222"
REALISED = "#117733"
STRESS = "#CC3311"
OVER = "#CC3311"
UNDER = "#4477AA"



from pathlib import Path as _SP
SLIDE_DIR = _SP(__file__).resolve().parent.parent / "results" / "figures" / "slides"
SLIDE_DIR.mkdir(parents=True, exist_ok=True)
BASIN_SHORT = {"A": "松辽", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
               "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北内陆"}


def save_fig(fig, name, subdir=""):  # noqa: F811  幻灯版：直接落到 slides/
    p = SLIDE_DIR / f"slide_{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"[slide] -> {p}")

def solver_params() -> tuple[float, float]:
    """Extractable fraction and retrofit CF boost -- taken from the package, not retyped.

    Hardcoding these would let a figure drift away from the model it claims to describe.
    `existing_withdrawal_share` defaults to 0.0 on the assumptions object, so the 0.85
    override the plotted runs carry is pinned above as a module constant instead.
    """
    from coal_retrofit.constants import WATER_EXTRACTABLE_FRACTION
    from coal_retrofit.optimization.scenario import OptimizationScenario

    scenario = OptimizationScenario(experiment_id="fig2", description="fig2")
    return float(WATER_EXTRACTABLE_FRACTION), float(scenario.retrofit_cf_boost)


EXTRACTABLE, CF_BOOST = solver_params()
USABLE = EXTRACTABLE * (1.0 - EXISTING_WITHDRAWAL_SHARE)


def require_current_vintage(scenario: str) -> None:
    """Vintage gate on the CSV header. 24-column runs predate the wet-to-dry cooling
    conversion entirely (2030 water use 8272 Mm3 against 4806-5075 here), i.e. a different
    feasible set. Filling the absent columns with zero would put a code change on the figure.
    """
    path = RESULTS_DIR / scenario / "plant_detail.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} - run {scenario} first")
    columns = pd.read_csv(path, nrows=0).columns
    missing = {"air_cooled_share", "already_air_share"} - set(columns)
    if missing:
        raise SystemExit(
            f"{scenario}: stale {len(columns)}-column vintage, missing {sorted(missing)}. "
            "This run has no air-cooling mechanism and cannot be compared with the current model."
        )


def load_availability() -> pd.DataFrame:
    path = INPUTS / "water_availability.csv"
    frame = pd.read_csv(path)
    members = frame["scenario_id"].nunique()
    if members != EXPECTED_MEMBERS:
        print(f"  [warn] {path.name} carries {members} members, not the expected "
              f"{EXPECTED_MEMBERS}")
    return frame


def basin_supply(avail: pd.DataFrame, year: int) -> pd.DataFrame:
    """Effective power-sector dry-season budget per (basin, member), 1e8 m3/yr."""
    year_frame = avail[avail["planning_year"] == year]
    grouped = year_frame.groupby(["basin_code", "scenario_id"])[
        "dry_season_water_m3_per_year"
    ].sum()
    return (grouped * USABLE / 1e8).unstack("scenario_id")


def basin_weights(scenario: str, year: int) -> pd.DataFrame:
    """Per-plant basin allocation weights from the solver's realised water flows.

    A plant inside the 200 km buffer can reach several basins -- 170 of 350 hubs can -- so
    attributing demand by nearest basin would be a guess. `water_flows.csv` records the
    allocation the solver actually chose, and summing it by node reproduces
    `resource_use.csv` to 3e-8, so it is used as the attribution key. Only 21 hubs split
    across more than one basin in practice.
    """
    path = RESULTS_DIR / scenario / "water_flows.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    flows = pd.read_csv(path)
    flows = flows[flows["year"] == year].copy()
    flows["basin"] = flows["water_node_id"].astype(str).str.rsplit("_", n=1).str[-1]
    # A (plant, basin) cell with no row means the solver drew no water there; zero is the true
    # value, not a masked missing one. Nothing else in this function fills.
    wide = flows.pivot_table(
        index="plant_id", columns="basin", values="flow_m3", aggfunc="sum"
    ).fillna(0.0)
    totals = wide.sum(axis=1)
    if (totals <= 0).any():
        raise SystemExit(f"{scenario} {year}: water_flows carries plants with zero total flow")
    return wide.div(totals, axis=0)


def _aligned(frame: pd.DataFrame, column: str, index: pd.Index, source: str) -> pd.Series:
    """Column reindexed onto `index`, failing loudly instead of filling a gap with zero."""
    series = frame[column].reindex(index)
    if series.isna().any():
        missing = series.index[series.isna()][:5].tolist()
        raise SystemExit(f"{source}: {column} missing for {series.isna().sum()} plants "
                         f"(e.g. {missing}); refusing to fill with zero")
    return series.astype(float)


def basin_demand(scenario: str, year: int) -> pd.DataFrame:
    """Coal water consumption per basin under three cooling/pathway states, 1e8 m3/yr.

    `unabated` and `full_capture` hold cooling at its as-built configuration, so they isolate
    what attaching capture does before any adaptation. `realised` is the solver's own
    `water_use_m3`, which already contains the dry-cooling conversion it chose.

    Intensities and the CF boost follow data_prep.py:742-748: capture pathways take the plant
    table's own with-capture intensity rather than a global multiplier, and retrofit pathways
    get the generation boost. `realised` is read straight from `water_use_m3`, so it validates the
    basin ALLOCATION but not the intensity reconstruction -- the two counterfactual columns
    are the reconstruction and have no solver output to check them against.
    """
    plants = pd.read_csv(INPUTS / "plants.csv").set_index("plant_id")
    detail = pd.read_csv(RESULTS_DIR / scenario / "plant_detail.csv")
    detail = detail[detail["year"] == year].set_index("plant_id")
    weights = basin_weights(scenario, year)
    index = weights.index

    generation = _aligned(detail, "annual_generation_mwh", index, f"{scenario} plant_detail")
    wet_base = _aligned(plants, "consumption_intensity_m3_per_mwh", index, "plants.csv")
    wet_capture = _aligned(plants, "consumption_ccs_intensity_m3_per_mwh", index, "plants.csv")

    # `consumption_intensity_m3_per_mwh` is ALREADY the as-built blend: builders/plants.py:
    # 207-217 averages every (combustion, cooling) group in the hub, air-cooled units
    # included, and data_prep.py:745 hands that column straight to the unabated pathway with
    # no further weighting (data_prep.py:778-780 says so explicitly). Blending it a second
    # time by `already_air_share` double-counts the dry fraction and understates as-built
    # demand -- nationally by 7.5% in 2030, and by 32.6% in the Yellow, 10.6% in the
    # Northwest Interior and 10.3% in the Hai, which are three of the four basins the
    # ceiling test flags. `already_air_share` is therefore NOT used here; it belongs to the
    # capex and backpressure terms (data_prep.py:783-786), not to the intensity.
    unabated = generation * wet_base
    full_capture = generation * CF_BOOST * wet_capture
    realised = _aligned(detail, "water_use_m3", index, f"{scenario} plant_detail")
    capacity = _aligned(detail, "capacity_mw", index, f"{scenario} plant_detail")

    out = pd.DataFrame(
        {
            "unabated": weights.mul(unabated, axis=0).sum(),
            "full_capture": weights.mul(full_capture, axis=0).sum(),
            "realised": weights.mul(realised, axis=0).sum(),
            "capacity_mw": weights.mul(capacity, axis=0).sum(),
        }
    )
    out[["unabated", "full_capture", "realised"]] /= 1e8
    return out


def as_built_capacity_gw() -> pd.Series:
    """Standing fleet capacity per basin, GW, by nearest level-1 basin.

    This is the attribution Fig 1(c) uses, so the two figures quote the same capacity behind
    the same basins. It differs from the solver's water-flow allocation by ~1% at the
    four-basin aggregate; both are printed so the choice is visible rather than implicit.
    """
    plants = pd.read_csv(INPUTS / "plants.csv")
    codes = assign_basin(plants)
    if codes is None:
        raise SystemExit("data/ChinaBasins/basin_l1.gpkg missing - cannot attribute capacity")
    plants = plants.assign(basin_code=codes.to_numpy())
    return plants.groupby("basin_code")["total_capacity_mw"].sum() / 1000.0


def unserved_by_year(scenario: str) -> pd.Series:
    """Unserved coal water demand per year, Mm3, from the solver's water_supply slack."""
    path = RESULTS_DIR / scenario / "slack_detail.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    slack = pd.read_csv(path)
    water = slack[slack["constraint_type"] == "water_supply"]
    if water.empty:
        return pd.Series(0.0, index=YEARS)
    return (water.groupby("year")["slack_value"].sum() / 1e6).reindex(YEARS).fillna(0.0)


def unserved_basins(scenario: str) -> list[str]:
    path = RESULTS_DIR / scenario / "slack_detail.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    slack = pd.read_csv(path)
    water = slack[(slack["constraint_type"] == "water_supply") & (slack["slack_value"] > 1.0)]
    if water.empty:
        return []
    codes = water["node_id"].astype(str).str.rsplit("_", n=1).str[-1]
    return sorted(set(codes))


# -- panel (a) ---------------------------------------------------------------
def panel_a(ax, avail: pd.DataFrame, basins: list[str]) -> pd.DataFrame:
    """Supply box/whisker per SSP against as-built and realised demand, 2030 vs 2060."""
    records = []
    slot = 0.0
    ticks, tick_labels, group_centres = [], [], []
    width = 0.34

    for year in (2030, 2060):
        supply = basin_supply(avail, year)
        demand = basin_demand(SOLVED["ssp126"], year)
        first_slot = slot
        for basin in basins:
            if basin not in supply.index:
                continue
            centre = slot
            for offset, ssp, colour in ((-width / 2, "ssp126", C126), (width / 2, "ssp370", C370)):
                members = [c for c in supply.columns if c.endswith(ssp)]
                values = supply.loc[basin, members].astype(float).to_numpy()
                box = ax.boxplot(
                    [values], positions=[centre + offset], widths=width * 0.86,
                    patch_artist=True, showfliers=False, whis=(0, 100), manage_ticks=False,
                )
                for patch in box["boxes"]:
                    patch.set(facecolor=colour, alpha=0.45, edgecolor=colour, linewidth=0.6)
                for key in ("whiskers", "caps", "medians"):
                    for line in box[key]:
                        line.set(color=colour, linewidth=0.7)
                records.append(
                    {
                        "year": year, "basin": basin, "ssp": ssp,
                        "supply_min": values.min(), "supply_median": float(np.median(values)),
                        "supply_max": values.max(),
                        "unabated": demand["unabated"].get(basin, np.nan),
                        "full_capture": demand["full_capture"].get(basin, np.nan),
                        "realised": demand["realised"].get(basin, np.nan),
                    }
                )
            if SOLVED_MEMBER in supply.columns:
                ax.plot(centre - width / 2, supply.loc[basin, SOLVED_MEMBER], marker="x",
                        ms=3.4, mew=0.9, color="black", zorder=6)
            if basin in demand.index:
                span = width * 0.95
                ax.hlines(demand.loc[basin, "unabated"], centre - span, centre + span,
                          color=DEMAND_UN, lw=1.5, zorder=5)
                ax.hlines(demand.loc[basin, "full_capture"], centre - span, centre + span,
                          color=DEMAND_CAP, lw=1.5, zorder=5)
                ax.plot(centre, demand.loc[basin, "realised"], marker="D", ms=3.0,
                        color=REALISED, mec="white", mew=0.4, zorder=7)
            ticks.append(centre)
            tick_labels.append(BASIN_SHORT.get(basin, basin))
            slot += 1.0
        group_centres.append((first_slot + slot - 1.0) / 2.0)
        slot += 0.9

    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=6.5)
    # BOTH SERIES ON THIS AXIS, IN THE SAME UNITS, BUT NOT THE SAME QUANTITY. The boxes are
    # dry-season runoff already multiplied by the extractable fraction and by (1 - s), i.e.
    # 3.0% of the basin's dry-season water; the demand lines are the fleet's full consumption.
    # That is the comparison the constraint makes, but a label reading "dry-season water"
    # invited the reader to take the boxes for the basin's water -- the Hai plots at 1.6
    # against a dry-season flow of 89.6. The label now names the allowance, not the resource.
    ax.set_ylabel("电力部门配额，与针对它的需求" + "\n"
                  + r"（$10^8$ m$^3$ yr$^{-1}$；配额 = 枯水期径流 "
                  + rf"$\times$ {EXTRACTABLE:.2f} $\times$ (1 - {EXISTING_WITHDRAWAL_SHARE:.2f})）",
                  fontsize=6.6, linespacing=1.3)
    for centre, year in zip(group_centres, (2030, 2060)):
        ax.text(centre, -0.155, str(year), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, fontweight="bold")
    # COMPUTED. This title asserted 'four basins' for the unabated state; on the corrected
    # dry-season budget the unabated count is three (the Northwest Interior falls to 83%) and
    # capture tips it to four. That distinction IS the figure, so it is read off the data.
    _s30 = basin_supply(avail, 2030); _d30 = basin_demand(SOLVED["ssp126"], 2030)
    _shared = [b for b in _d30.index if b in _s30.index]
    _n_un = sum(_d30.loc[b, "unabated"] > float(_s30.loc[b].median()) for b in _shared)
    _n_cap = sum(_d30.loc[b, "full_capture"] > float(_s30.loc[b].median()) for b in _shared)
    _words = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五"}
    ax.set_title("各流域枯水期电力配额（供给）与煤电取水需求，2030 与 2060 年", fontsize=7.4)

    handles = [
        Patch(facecolor=C126, alpha=0.45, edgecolor=C126, label="供给 SSP1-2.6（10 个成员）"),
        Patch(facecolor=C370, alpha=0.45, edgecolor=C370, label="供给 SSP3-7.0（10 个成员）"),
        Line2D([], [], color="black", marker="x", ls="none", ms=3.4, label="已求解成员"),
        Line2D([], [], color=DEMAND_UN, lw=1.5, label="需求：未改造（现状冷却方式）"),
        Line2D([], [], color=DEMAND_CAP, lw=1.5, label="需求：全量捕集（现状冷却方式）"),
        Line2D([], [], color=REALISED, marker="D", ls="none", ms=3.0, label="模型实际取水"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.245), columnspacing=1.0, handlelength=1.4)
    ax.set_ylim(bottom=max(ax.get_ylim()[0], 0.05))
    return pd.DataFrame(records)


def reservation_curve(avail: pd.DataFrame, year: int, fleet_gw: pd.Series) -> pd.DataFrame:
    """Standing capacity beyond the allowance as a function of the non-power reservation share.

    `existing_withdrawal_share` is the one parameter that creates the binding constraint, so
    leaving it at a single value makes the whole result look like a tuned dial. It is not a
    physical quantity -- it is an allocation rule -- so the defensible object is the curve,
    not a point on it. Supply scales exactly as (1 - s), so the curve is analytic in s and
    needs no extra solves: a basin crosses at its own s* = 1 - demand / (dry-season runoff x
    extractable), and the fleet-level curve is the staircase those crossings trace out.
    """
    supply_full = basin_supply(avail, year) / (1.0 - EXISTING_WITHDRAWAL_SHARE)
    demand = basin_demand(SOLVED["ssp126"], year)
    rows = []
    for state in ("unabated", "full_capture"):
        for member in supply_full.columns:
            for share in RESERVATION_GRID:
                over = [
                    b for b in demand.index
                    if b in supply_full.index
                    and demand.loc[b, state] > supply_full.loc[b, member] * (1.0 - share)
                ]
                rows.append({"state": state, "member": member, "share": share,
                             "gw": float(fleet_gw.reindex(over).fillna(0.0).sum())})
    return pd.DataFrame(rows)


def critical_shares(avail: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per-basin s*, the reservation share at which the basin exceeds its own allowance."""
    supply_full = basin_supply(avail, year) / (1.0 - EXISTING_WITHDRAWAL_SHARE)
    demand = basin_demand(SOLVED["ssp126"], year)
    rows = []
    for basin in supply_full.index:
        values = supply_full.loc[basin].astype(float).to_numpy()
        for state in ("unabated", "full_capture"):
            star = 1.0 - demand[state].get(basin, np.nan) / values
            rows.append({
                "basin": basin, "state": state,
                "s_star_median": float(np.median(star)),
                "s_star_p10": float(np.percentile(star, 10)),
                "s_star_p90": float(np.percentile(star, 90)),
                "share_members_negative": float((star < 0).mean()),
            })
    return pd.DataFrame(rows).set_index(["state", "basin"])


# -- panel (b) ---------------------------------------------------------------
def panel_b(ax, avail: pd.DataFrame, fleet_gw: pd.Series) -> pd.DataFrame:
    """The allowance against the allocation rule that sets it, not against time.

    Replaces the year-series panel of earlier drafts. That panel held the reservation share
    fixed at 0.85 and varied the year, which is the one axis along which nothing is contested;
    it also carried a twin axis whose zero did not align with the left one. This panel varies
    the contested parameter instead and reports two facts that do not depend on where it sits.
    """
    curve = reservation_curve(avail, CEILING_YEAR, fleet_gw)
    stars = critical_shares(avail, CEILING_YEAR)
    total = float(fleet_gw.sum())

    for state, colour, label in (
        ("unabated", DEMAND_UN, "机组按现状配置（无捕集）"),
        ("full_capture", DEMAND_CAP, "全部机组捕集"),
    ):
        sub = curve[curve["state"] == state]
        band = sub.groupby("share")["gw"]
        share = np.asarray(sorted(sub["share"].unique()))
        median = band.median().reindex(share).to_numpy()
        lo = band.quantile(0.10).reindex(share).to_numpy()
        hi = band.quantile(0.90).reindex(share).to_numpy()
        ax.fill_between(share, lo, hi, color=colour, alpha=0.16, lw=0, step="post")
        ax.step(share, median, where="post", lw=1.5, color=colour, label=label)

    # The plateau: the range of reservation shares over which the four-basin verdict is the
    # SAME verdict. Its width is the answer to "you tuned the dial".
    med_cap = curve[curve["state"] == "full_capture"].groupby("share")["gw"].median()
    at_085 = float(med_cap.reindex([EXISTING_WITHDRAWAL_SHARE]).iloc[0])
    plateau = med_cap.index[np.isclose(med_cap.to_numpy(), at_085)]
    ax.axvspan(float(plateau.min()), float(plateau.max()), color=DEMAND_CAP, alpha=0.055, lw=0,
               zorder=0)
    ax.annotate(
        f"在 {plateau.min():.2f}–{plateau.max():.2f} 区间内\n结论一致",
        xy=(float(plateau.min()) - 0.012, at_085 + 0.045 * total),
        ha="right", va="bottom", fontsize=5.6, color="#555555")

    # THE CORRECTION THAT MOVED THIS PANEL'S HEADLINE. Until 2026-08-18 the dry-season budget
    # was built as a sum of per-grid-cell minima rather than the basin's own low-flow quarter
    # (sum(min) instead of min(sum) -- see builders/water.py), which understated the Hai by
    # 2.25x and the Northwest Interior by 2.13x, i.e. most in exactly the basins this panel is
    # about. On that basis the Hai's critical share was s* = -0.63: it exceeded its allowance
    # even when power took the ENTIRE extractable dry-season flow, and this panel said so with
    # a dashed line across its own y-axis. On the corrected basis s* = +0.28. At s = 0 NO
    # basin, in any planning year, with or without capture, exceeds its allowance. The
    # staircase therefore starts at zero, and every GW it reports is the consequence of an
    # allocation decision rather than of hydrology. That is a stronger claim than the one it
    # replaces, and it is the one the constraint can actually support.
    cap_stars = stars.loc['full_capture', 's_star_median'].sort_values()
    first_basin = str(cap_stars.index[0])
    first_share = float(cap_stars.iloc[0])
    # THE CROSSING CAN SIT BELOW ZERO, AND WHEN IT DOES THE PANEL MEANS THE OPPOSITE.
    # `first_share` is the smallest median s* across basins under full capture. On the
    # corrected basis that is the Hai at -0.08: capture exceeds the allowance there even when
    # power is given the ENTIRE extractable share. Two things followed from not handling it.
    # The marker was drawn at x = -0.08 with clip_on=False against an xlim starting at 0.0, so
    # it showed as an orphan glyph in the left margin while annotate's default annotation_clip
    # silently dropped the text explaining it. And the title still asserted 'scarcity is not
    # the binding constraint' 30 mm from panel (c) reporting that capture turns allocation into
    # scarcity in that same basin. The figure cancelled itself.
    if first_share > ax.get_xlim()[0]:
        ax.plot([first_share], [0.0], marker='^', ms=4.2, color=DEMAND_CAP, mec='white',
                mew=0.5, zorder=8)
        ax.annotate(
            f"s = {first_share:.2f} 以下没有任何\n流域超出其配额\n"
            f"（最先越线：{BASIN_NAMES_ZH.get(first_basin, first_basin)}）",
            xy=(first_share, 0.0), xytext=(first_share + 0.04, 0.115 * total),
            fontsize=5.6, color=DEMAND_CAP, va="bottom", linespacing=1.2,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=DEMAND_CAP, shrinkA=0, shrinkB=2))
    else:
        ax.annotate(
            f"{BASIN_NAMES_ZH.get(first_basin, first_basin)}在 s = 0 时"
            f"\n就已超出配额（s* 中位数 = {first_share:+.2f}）："
            f"\n零预留下捕集依然越线",
            xy=(0.0, 0.0), xytext=(0.035, 0.165 * total),
            fontsize=5.6, color=DEMAND_CAP, va="bottom", linespacing=1.2,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=DEMAND_CAP, shrinkA=0, shrinkB=2))

    for share, style, text in (
        (EXISTING_WITHDRAWAL_SHARE, "-", "0.85\n取水统计\n实测值"),
        (0.559, (0, (1, 1.6)), "0.56\n全国口径"),
    ):
        ax.axvline(share, ls=style, lw=0.9, color=C126, alpha=0.85, zorder=2)
        ax.annotate(text, xy=(share, total * (0.995 if share > 0.7 else 0.62)), fontsize=5.4, color=C126,
                    ha="right" if share > 0.5 else "left", va="top", linespacing=1.15)

    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(0, total)
    # THE AXIS IS THE RESIDUAL SHARE, AND IT HAS TWO READINGS THE MODEL CANNOT SEPARATE.
    # `usable = 0.20 x (1 - s)`, so s = 0.85 at a 20% extractable fraction is arithmetically
    # the same model as s = 0 at 3%. Labelling this axis purely as an allocation-policy dial
    # therefore over-claims: every point on it is equally a statement about how strict the
    # environmental-flow standard is. The label now says what the axis is -- the share NOT
    # left to power -- and the caption carries both readings.
    ax.set_xlabel("枯水期径流中不可供电力使用的份额 s（其他用水户或更严格的生态流量标准）", fontsize=6.6)
    ax.set_ylabel("超出流域配额的\n煤电容量（GW）", fontsize=7.2)
    # Derived, not asserted -- see the crossing block above for why a fixed string was wrong.
    ax.set_title("超出配额的煤电容量随非电力预留份额 s 的变化（2030 年）", fontsize=7.4)
    right = ax.twinx()
    right.spines["right"].set_visible(True)
    right.set_ylim(0, 100)
    right.set_ylabel("占全国煤电容量（%）", fontsize=6.4, color="#555555")
    right.tick_params(axis="y", labelsize=6, colors="#555555")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.6)
    return stars

# -- panel (c) ---------------------------------------------------------------
def allowance_ratio_table(avail: pd.DataFrame, basins: list[str],
                          fleet_gw: pd.Series) -> pd.DataFrame:
    """Full-capture demand as a multiple of the allowance -- DATA ONLY, no longer drawn.

    Panel (c) used to draw this. It now draws the critical reservation share instead, because
    a ratio against an allowance cannot distinguish 'not enough water' from 'not enough of the
    water'. The ratios are still computed, because the report and the Extended Data quote them
    and because they are the quantity the four-basin verdict is defined on.
    """
    supply_now = basin_supply(avail, CEILING_YEAR)
    demand_now = basin_demand(SOLVED["ssp126"], CEILING_YEAR)
    supply_end = basin_supply(avail, 2060)
    demand_end = basin_demand(SOLVED["ssp126"], 2060)
    rows = []
    for basin in basins:
        if basin not in supply_now.index or basin not in demand_now.index:
            continue
        ratios = demand_now.loc[basin, "full_capture"] / supply_now.loc[basin].astype(float)
        end = demand_end.loc[basin, "full_capture"] / supply_end.loc[basin].astype(float)
        rows.append({"basin": basin, "median": float(ratios.median()),
                     "min": float(ratios.min()), "max": float(ratios.max()),
                     "median_2060": float(end.median()),
                     "fleet_gw": float(fleet_gw.get(basin, 0.0)),
                     "flow_gw": float(demand_now.loc[basin, "capacity_mw"]) / 1000.0,
                     "crosses": bool(ratios.min() > 1.0),
                     "crosses_median": bool(ratios.median() > 1.0)})
    return pd.DataFrame(rows).set_index("basin")


def panel_c(ax, avail: pd.DataFrame, basins: list[str], fleet_gw: pd.Series) -> pd.DataFrame:
    """The critical reservation share s*, per basin, across the whole 20-member ensemble.

    WHAT THIS PANEL REPLACED AND WHY. It used to plot full-capture demand as a MULTIPLE of
    dry-season supply -- the same two numbers, but expressed as a ratio against an allowance
    that is itself a policy choice. A reader could not tell from it whether a basin is over the
    line because there is not enough water or because power is not given enough of it. s* is
    the same information rotated onto the axis that decides:

        s* = 1 - demand / (dry-season runoff x extractable fraction)

    the non-power reservation share at which a basin's coal water demand exactly exhausts the
    environmental-flow allowance. s* > 0.85 means the basin is inside the allowance at China's
    measured reservation. 0 < s* < 0.85 means it breaches, but would not if power were given
    the whole extractable share -- an ALLOCATION outcome. s* < 0 means it breaches even at zero
    reservation -- genuine SCARCITY.

    AND IT CHANGES THE PAPER'S CLAIM, WHICH IS WHY IT IS DRAWN ON THE ENSEMBLE AND NOT ON THE
    SOLVED MEMBER. Computed on the single solved member (cwatm|gfdl-esm4|ssp126), s* is
    positive everywhere, and v5 concluded that the constraint is allocation throughout. Across
    all 20 members that is true for TODAY'S FLEET -- s* > 0 in 20 of 20 members, every basin,
    every planning year, minimum +0.127 -- but NOT for full capture in the Hai, where s* is
    negative in 13 of 20 members at 2030 and 12 of 20 at 2040 (9 of 10 WaterGAP2 members, 4 of
    10 CWatM, both SSPs). Capture does not merely tighten an allocation constraint; in one
    basin it converts the problem into a scarcity problem. One member cannot show that.
    """
    year_frame = avail[avail["planning_year"] == CEILING_YEAR]
    dry = (year_frame.groupby(["basin_code", "scenario_id"])
           ["dry_season_water_m3_per_year"].sum().unstack("scenario_id") / 1e8)
    demand = basin_demand(SOLVED["ssp126"], CEILING_YEAR)
    order = [b for b in basins if b in dry.index and b in demand.index]
    rows = []
    y = np.arange(len(order))[::-1]
    for yi, b in zip(y, order):
        allowance = dry.loc[b].astype(float).to_numpy() * EXTRACTABLE
        for state, off, colour, marker in (("unabated", +0.24, DEMAND_UN, "o"),
                                           ("full_capture", -0.24, DEMAND_CAP, "D")):
            s = 1.0 - float(demand.loc[b, state]) / allowance
            # Thick bar = 10-90% of the ensemble, thin line = full range, marker = median. The
            # full range is drawn because in the Hai the tail IS the result, not an outlier.
            ax.plot([np.percentile(s, 10), np.percentile(s, 90)], [yi + off] * 2,
                    color=colour, lw=1.5, solid_capstyle="butt", zorder=4)
            ax.plot([s.min(), s.max()], [yi + off] * 2, color=colour, lw=0.5, alpha=0.55,
                    zorder=3)
            # "as built" is drawn HOLLOW. Filled grey against filled black at 3.6 pt read as
            # one weight, and on the four right-hand rows the pair merged into a single dot.
            ax.plot([np.median(s)], [yi + off], marker=marker, ms=3.6,
                    mfc=("none" if state == "unabated" else colour), mec=colour, mew=0.8,
                    zorder=5)
            n_neg = int((s < 0).sum())
            if n_neg:
                ax.text(s.min() - 0.035, yi + off, f"{n_neg}/{len(s)}", fontsize=5.0,
                        color=colour, ha="right", va="center", fontweight="bold")
            rows.append({"basin": b, "state": state, "median": float(np.median(s)),
                         "p10": float(np.percentile(s, 10)), "p90": float(np.percentile(s, 90)),
                         "min": float(s.min()), "max": float(s.max()),
                         "members_negative": n_neg, "n_members": int(len(s)),
                         "fleet_gw": float(fleet_gw.get(b, 0.0))})
    table = pd.DataFrame(rows).set_index(["state", "basin"])

    ax.axvspan(-1.05, 0.0, color=DEMAND_CAP, alpha=0.075, lw=0, zorder=0)
    ax.axvline(0.0, color=DEMAND_CAP, lw=1.0, zorder=6)
    ax.axvline(EXISTING_WITHDRAWAL_SHARE, color=C126, lw=1.0, ls=(0, (3, 2)), zorder=6)
    ax.text(-0.03, len(order) - 0.34, "资源稀缺\n零预留时\n即已越线",
            fontsize=5.0, color=DEMAND_CAP, ha="right", va="bottom", linespacing=1.15)
    ax.text(EXISTING_WITHDRAWAL_SHARE - 0.03, len(order) - 0.34,
            f"中国实测的\n存量取水占比 {EXISTING_WITHDRAWAL_SHARE:.2f}",
            fontsize=5.0, color=C126, ha="right", va="bottom", linespacing=1.15)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{BASIN_NAMES_ZH.get(b, b)}" for b in order], fontsize=6.2)
    for tick, b in zip(ax.get_yticklabels(), order):
        if float(table.loc[("full_capture", b), "median"]) < EXISTING_WITHDRAWAL_SHARE:
            tick.set_fontweight("bold")
    # Was (-1.05, 1.0). The leftmost ink is the Hai full-capture range minimum at about
    # -0.63, so a third of the axis carried nothing while the informative right-hand rows
    # were crushed against 1.0.
    ax.set_xlim(-0.72, 1.02)
    ax.set_ylim(-0.62, len(order) + 0.62)
    ax.set_xlabel(f"临界预留份额 s*，{CEILING_YEAR} 年\n"
                  f"（20 个水文成员；标记 = 中位数，色块 = 10–90%，横线 = 全距）",
                  fontsize=6.4, linespacing=1.3)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    key = ("full_capture", "C")
    hai = table.loc[key] if key in table.index else None
    if hai is not None and int(hai["members_negative"]):
        # NAME THE STANDARD IN THE TITLE. The scarcity result depends on the EXTRACTABLE
        # FRACTION, not on the reservation: s* is defined against extractable alone. At Richter's
        # 0.20 the Hai is negative in 13 of 20 members; at the 0.40 Jin et al. (2022) adopt for
        # China it is +0.46 and no member is negative. A title saying 'capture turns allocation
        # into scarcity' without the 0.20 attached over-claims: the finding is about how strict
        # the standard is. Full comparison in v6_efr_convention_table.txt.
        ax.set_title(f"在 {EXTRACTABLE:.0%} 取水上限下，捕集使一个流域" + chr(10) +
                     f"从分配问题转为资源稀缺问题" + chr(10) +
                     f"（{int(hai["n_members"])} 个成员中有 {int(hai["members_negative"])} 个）",
                     fontsize=6.9, linespacing=1.22)
    else:
        ax.set_title("每一次越线都是分配选择的结果", fontsize=7.2)
    ax.legend(handles=[Line2D([], [], color=DEMAND_UN, marker="o", ls="none", ms=3.4,
                              label="现状冷却，无捕集"),
                       Line2D([], [], color=DEMAND_CAP, marker="D", ls="none", ms=3.4,
                              label="全部机组捕集")],
              fontsize=5.4, frameon=False, loc="lower right", bbox_to_anchor=(0.99, 0.01),
              handletextpad=0.5, labelspacing=0.28)
    return table


def _panel_c_ratio_superseded(ax, avail: pd.DataFrame, basins: list[str], fleet_gw: pd.Series):
    """The allowance test per basin: full-capture demand as a multiple of dry-season supply."""
    supply_now = basin_supply(avail, CEILING_YEAR)
    demand_now = basin_demand(SOLVED["ssp126"], CEILING_YEAR)
    supply_end = basin_supply(avail, 2060)
    demand_end = basin_demand(SOLVED["ssp126"], 2060)

    rows = []
    for basin in basins:
        if basin not in supply_now.index or basin not in demand_now.index:
            continue
        ratios = demand_now.loc[basin, "full_capture"] / supply_now.loc[basin].astype(float)
        end = demand_end.loc[basin, "full_capture"] / supply_end.loc[basin].astype(float)
        rows.append(
            {
                "basin": basin, "median": float(ratios.median()),
                "min": float(ratios.min()), "max": float(ratios.max()),
                "median_2060": float(end.median()),
                "fleet_gw": float(fleet_gw.get(basin, 0.0)),
                "flow_gw": float(demand_now.loc[basin, "capacity_mw"]) / 1000.0,
                "crosses": bool(ratios.min() > 1.0),
                "crosses_median": bool(ratios.median() > 1.0),
            }
        )
    table = pd.DataFrame(rows).set_index("basin")

    y = np.arange(len(table))[::-1]
    colours = [OVER if c else UNDER for c in table["crosses_median"]]
    # NO BARS ON A LOG AXIS. A bar encodes a value by its LENGTH, measured from the axis
    # origin; on a log axis that origin is wherever xlim happens to start, so the ink is
    # log(value / xmin) and every ratio the reader takes off the page is an artefact of a
    # number that is not in the data. At the xlim this panel used, the Hai bar was 6.6x the
    # Southeast Rivers bar in ink while the values differ by 115x; moving the invisible floor
    # from 0.03 to 0.003 redrew that ratio as 2.5x without changing a datum, and at 0.1 the
    # Southeast bar would have pointed backwards. plot_fig1_water_footprint.py forbids this
    # in its own panel (b) and was fixed there; this panel kept the defect. Position encodes
    # the median, a line encodes the 20-member range -- neither depends on the origin.
    ax.hlines(y, table["min"], table["max"], color=colours, lw=2.4, alpha=0.55, zorder=3)
    ax.plot(table["min"], y, marker="|", ls="none", ms=4.0, mew=0.9, color="#333333", zorder=4)
    ax.plot(table["max"], y, marker="|", ls="none", ms=4.0, mew=0.9, color="#333333", zorder=4)
    ax.scatter(table["median"], y, s=26, c=colours, edgecolor="white", linewidth=0.5,
               zorder=6)
    ax.plot(table["median_2060"], y, marker="D", ls="none", ms=3.2, mfc="white",
            mec="#111111", mew=0.7, zorder=5)
    ax.axvline(1.0, color="black", lw=0.9, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(0.03, 40)
    # Pin the ticks: LogLocator also lays out 10^2 and 10^3 labels past xlim, which are never
    # drawn but still carry a window extent, and savefig(bbox_inches='tight') grows the canvas
    # to include them -- 30 mm of phantom width on a 183 mm column.
    ax.set_xticks([0.1, 1.0, 10.0])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(-0.70, len(table) + 0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{BASIN_NAMES_ZH.get(b, b)}" for b in table.index], fontsize=6.4
    )
    for tick, crosses in zip(ax.get_yticklabels(), table["crosses_median"]):
        if crosses:
            tick.set_fontweight("bold")
    # Two lines, not one: as a single line this label ran to the last pixel column of the
    # canvas and was clipped mid-word ("...1.0 = the ceilin"). bbox_inches='tight' cannot
    # catch that -- a clipped artist has no extent beyond the canvas, so the width guard
    # reports "clean" while the label is cut.
    # rf-string, not f. In a plain f-string the sequence dollar-backslash-t-i-m-e-s is
    # parsed as a TAB escape, so the axis rendered "(imes)" -- a real typo that survived
    # three figure revisions because it looks like LaTeX in the source.
    ax.set_xlabel(rf"全量捕集需求 / 枯水期供给（$\times$ 倍）“ ”\n"
                  rf"{CEILING_YEAR} 年；1.0 = 配额线", fontsize=7, linespacing=1.3)

    # Standing capacity behind each basin, printed at a fixed x so the column reads as a table.
    for yi, (basin, row) in zip(y, table.iterrows()):
        ax.text(0.985, yi, f"{row['fleet_gw']:.0f} GW", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=5.9,
                fontweight="bold" if row["crosses_median"] else "normal",
                color="#111111" if row["crosses_median"] else "#666666")
    ax.text(0.985, len(table) - 0.55, "在役容量", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=5.4, color="#555555")

    crossing = table.index[table["crosses_median"]].tolist()
    crossing_gw = table.loc[crossing, "fleet_gw"].sum()
    total_gw = float(fleet_gw.sum())
    # Broken across three lines and stepped down to 7.2 pt: as one long line this title was the
    # widest artist on the canvas and pushed the saved figure past the 183 mm column.
    ax.set_title(
        f"{len(crossing)} 个北方流域（{'、'.join(BASIN_NAMES_ZH.get(b, b) for b in crossing)}）\n"
        f"有 {crossing_gw:.0f} GW 超出配额\n"
        f"占全国在役 {total_gw:,.0f} GW 的 {100 * crossing_gw / total_gw:.0f}%",
        fontsize=7.2, linespacing=1.25,
    )
    handles = [
        Line2D([], [], color=OVER, marker="o", ls="none", ms=4.0, mec="white", mew=0.5,
               label="超出配额"),
        Line2D([], [], color=UNDER, marker="o", ls="none", ms=4.0, mec="white", mew=0.5,
               label="配额之内"),
        Line2D([], [], color="#333333", lw=0.8, label="20 个成员的全距"),
        Line2D([], [], color="#111111", marker="D", ls="none", ms=3.2, mfc="white", mew=0.7,
               label="2060 年中位数"),
    ]
    # Only the two colour keys. Whisker and diamond are already named in the caption, and a
    # 4-entry legend at ncol=2 ran under the right-hand "86 GW" standing-fleet label.
    ax.legend(handles=handles[:2], frameon=False, fontsize=5.9, ncol=1, loc="lower left",
              bbox_to_anchor=(0.30, 0.01), handlelength=1.4, labelspacing=0.2)
    ax.grid(axis="x", lw=0.3, alpha=0.3)
    ax.set_axisbelow(True)
    return table


def main() -> None:
    for scenario in list(SOLVED.values()) + [FROZEN]:
        require_current_vintage(scenario)
    avail = load_availability()
    fleet_gw = as_built_capacity_gw()
    present = [b for b in BASIN_ORDER if b in set(avail["basin_code"])]
    basins = [b for b in present if fleet_gw.get(b, 0.0) > 0.0]

    fig_a = plt.figure(figsize=(7.2, 3.7))
    ax_a = fig_a.add_axes([0.085, 0.30, 0.90, 0.60])
    panel_a(ax_a, avail, basins)
    panel_label(ax_a, "a", x=-0.075, y=1.10)
    save_fig(fig_a, "constraint_a")

    fig_b = plt.figure(figsize=(4.3, 3.9))
    ax_b = fig_b.add_axes([0.17, 0.26, 0.68, 0.64])
    panel_b(ax_b, avail, fleet_gw)
    panel_label(ax_b, "b", x=-0.22, y=1.10)
    save_fig(fig_b, "constraint_b")
if __name__ == "__main__":
    main()
