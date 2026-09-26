"""Fig 2 - two water institutions, on two water bases, and which of them the coal fleet
actually runs into.

v9.1 CHANGED WHAT THIS FIGURE IS ABOUT, because it changed what the model enforces.

Through v9 the water rule was a single node constraint whose right-hand side was
`dry-season runoff x 0.20 x (1 - 0.85)`. Those two factors are an environmental-flow
STANDARD and an allocation RULE, and multiplying them ALIASED the two: 0.85 at a 20%
extractable fraction is arithmetically the same model as 0 at 3%, so no result the study
could produce was able to say which of the two a response belonged to. The figure's old
panels (b) and (c) swept that reservation share and solved for a critical share s*, which
made the aliasing the subject rather than removing it.

v9.1 splits them into two constraints on two bases, and the split is what this figure now
reports:

  rung 1  ENVIRONMENTAL FLOW, on CONSUMPTION.  node <= dry-season runoff x 0.20
          Richter et al. (2012) River Res. Applic. 28(8):1312-1321 -- protecting 80% of
          daily flows maintains ecological integrity. A depletion rule. Enforced by the
          `*_oq_envonly` runs.
  rung 2  ALLOCATION, on WITHDRAWAL.  basin <= 用水总量控制指标 - 非电既有取水
          国办发〔2013〕2号 附件1 (the 最严格水资源管理制度 caps, which sum exactly to the
          national 6350/6700/7000 亿 m3) net of what the 2025 水资源公报 表9 records other
          users already taking. Added by the `*_oq` runs.

The bases are not interchangeable and are never combined. The 公报 meters 用水量, which
INCLUDES the once-through condenser pass-through (直流火(核)电 453.8亿 m3 is a sub-column of
工业用水); the 取水定额 standard excludes it, and the fleet's consumption is smaller again by
another order of magnitude. Comparing a consumption numerator against an allocation
denominator understates the fleet's claim roughly five-fold, which is why rung 2 is drawn
against the CALIBRATED withdrawal the solver itself constrains.

    What this figure claims:
    (1) The two rungs do not agree about which basin is tightest. That disagreement is a
        result, not a nuisance: a depletion rule and an allocation rule are different
        institutions and a basin can sit comfortably inside one while breaching the other.
    (2) On the allocation rung the breach is not created by capture. It is there in the
        standing fleet, and capture roughly doubles it -- the same shape the old figure
        claimed, but now on the basis China's own red-line policy is written against, and in
        a different and much smaller set of basins.
    (3) The fleet's response to each rung is priced separately (panel c). BASE -> envonly is
        the environmental-flow standard's price; envonly -> oq is the allocation rule's.
        Neither is quoted unless it clears its own degeneracy floor.

    What it does NOT claim, and must not be read as claiming:
    (a) that the fleet is physically unable to obtain this water. Rung 1 is computed on LOCAL
        RENEWABLE DRY-SEASON RUNOFF from unrouted `qtot`, i.e. a firm-yield rule for a system
        with ZERO storage. Northern basins are supplied by reservoir regulation, the
        South-North Water Transfer, groundwater and reclaimed water, none of which the
        hydrology represents. Rung 2 does not have this problem -- the official caps are
        written against delivered supply, transfers and groundwater included -- and that is
        the main reason the study moved onto it.
    (b) that the Northwest's four-digit utilisation is a robust multiple. Its 2025 metered
        withdrawal already exceeds its own 2030 cap, so the raw residual is negative and the
        plotted denominator is the pro-rata-scaled one. The breach is real; the multiple is
        an artefact of dividing by a near-zero residual, and the panel says so on its face.
    (c) that rung 2's per-basin caps are published numbers. 附件1 allocates by PROVINCE. The
        province-to-basin split here is demand-weighted on the ISIMIP grid; area weighting
        flips the sign of the Northwest residual and is reported as a sensitivity. See
        docs/官方指标口径水预算.md.

  (a) rung 1 in volume: effective dry-season allowance vs coal water demand, every level-1
      basin that hosts coal capacity (8 of 9 -- Southwest Rivers has none), 2030 and 2060,
      with the 20-member ensemble as box/whisker on supply and demand shown three ways
      (as-built unabated, as-built full-capture, and what the solver actually realised).
  (b) both rungs as utilisations, per basin, on a log axis: bar = all units retrofitted,
      tick = the fleet as built, reference line at 100% of that rung's own allowance.
  (c) the three-rung response BASE -> 仅生态流量 -> ＋用水总量指标, for dry-cooling
      conversion, capture and early retirement, each normalised to BASE and each carrying
      the seed family's degeneracy floor.

The ensemble box/whisker in (a) is the physical spread of the resource -- five GCMs and two
hydrology models disagreeing about how much water a basin has -- not a sensitivity of this
paper's model, so it belongs in the main figure. The attribution of that spread to GCM,
hydrology model and SSP has moved to Extended Data
(`scripts/plot_ed_variance_decomposition.py`), where uncertainty decompositions belong.

Two facts drive every number in panel (a) and both were checked against solved output rather
than assumed:

1. `_dry` scenarios read `dry_season_water_m3_per_year`, NOT the annual column. Reproducing
   the solver's own `available` column from the dry-season column matches to 5.6e-16
   relative; the annual column is 4.3x too high.
2. The stored availability is raw physical runoff; the budget the solver sees is
   `dry_season x WATER_EXTRACTABLE_FRACTION` (`water_access._water_available_by_node`, the
   official-quota budget). One factor now, not two.

Panel (b) does not re-derive its numbers: it calls Fig 1's `site_table`/`basin_summary`, so
the two main figures cannot quote different stresses for the same basin.

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
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
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
    national,
    degeneracy_floor,
    seeds_of,
    treat_of,
    BASE_SCENARIO,
    ARMS,
    SEED_ARM,
)

apply_style()

INPUTS = ROOT / "inputs"
YEARS = [2030, 2040, 2050, 2060]
# The year the ceiling test is reported on in (c): the standing fleet is still nearly intact,
# so this is the tightest the constraint gets and the year the 765 GW refers to.
CEILING_YEAR = 2030

# The solved pair: v9.1 official-quota solves (`_v91tree/scripts/run_single.py:383,385`), so
# there is no reservation share (see below).
SOLVED = {"ssp126": "WA_cwatm_126_dry_oq", "ssp370": "WA_cwatm_370_dry_oq"}
# Adaptation frozen. Not plotted -- it is a counterfactual, not part of the claim -- but its
# unserved volume is printed, because it is the cleanest evidence that the ceiling is real.
FROZEN = "WA_cwatm_126_dry_oq_noair"
# Member the solves pin (`_v91tree/scripts/run_single.py:383`). It is a drying GCM, so it is marked in panel (a)
# rather than left to look like the ensemble centre.
SOLVED_MEMBER = "cwatm|gfdl-esm4|ssp126"

# v9.1: there is no reservation SHARE. The non-power claim is not assumed here, it is read
# off 国办发〔2013〕2号 附件1 + 2025年中国水资源公报 表9 into inputs/water_basin_caps.csv, and
# it lives on the WITHDRAWAL basis rather than the consumption basis the environmental-flow
# rule uses. The two rungs are two numbers on two bases and are never multiplied together.
# Panel (b) reports both; the derivation is Fig 1's and is imported, not repeated.

SSPS = ["ssp126", "ssp370"]
EXPECTED_MEMBERS = 20  # 2 hydrology x 5 GCM x 2 SSP

# Colours held locally, not added to plot_style: sibling figure scripts are editing that
# module concurrently.
# 两档水规则的配色（CLAUDE.md §3.3）：耗水深蓝 / 取水浅蓝，同族分深浅正好对应
# "同一条水、两种计量口径"，不是两件无关的事；上限线用强调红。
C_BASE_RUNG = "#969696"    # 不考虑水：中性灰
C_ENV_RUNG = "#08519C"     # 生态流量档（耗水口径）：Blues 深端
C_QUOTA_RUNG = "#6BAED6"   # 用水总量指标档（取水口径）：Blues 浅端
C_LIMIT = "#CC3311"        # 配额上限线
C126 = "#4477AA"
C370 = "#CC3311"
DEMAND_UN = "#999999"
DEMAND_CAP = "#222222"
REALISED = "#117733"
STRESS = "#CC3311"
OVER = "#CC3311"
UNDER = "#4477AA"


def solver_params() -> tuple[float, float]:
    """Extractable fraction and retrofit CF boost -- taken from the package, not retyped.

    Hardcoding these would let a figure drift away from the model it claims to describe.
    """
    from coal_retrofit.constants import WATER_EXTRACTABLE_FRACTION
    from coal_retrofit.optimization.scenario import OptimizationScenario

    scenario = OptimizationScenario(experiment_id="fig2", description="fig2")
    return float(WATER_EXTRACTABLE_FRACTION), float(scenario.retrofit_cf_boost)


EXTRACTABLE, CF_BOOST = solver_params()
# The node budget is the environmental-flow rule ALONE (see above): no second factor.
USABLE = EXTRACTABLE


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

    Intensities and the CF boost follow `plant_matrices._water_intensity_matrices` and
    `_plant_operating_matrices`: capture pathways take the plant
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
    # included, and `plant_matrices._water_intensity_matrices` hands that column straight to
    # the unabated pathway with no further weighting (the `_air_cooling_matrices` docstring
    # says so explicitly). Blending it a second time by `already_air_share` double-counts the
    # dry fraction and understates as-built
    # demand -- nationally by 7.5% in 2030, and by 32.6% in the Yellow, 10.6% in the
    # Northwest Interior and 10.3% in the Hai, which are three of the four basins the
    # ceiling test flags. `already_air_share` is therefore NOT used here; it belongs to the
    # capex and backpressure terms (`plant_matrices._air_cooling_matrices`), not to the intensity.
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
            tick_labels.append(basin)
            slot += 1.0
        group_centres.append((first_slot + slot - 1.0) / 2.0)
        slot += 0.9

    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=6.5)
    # BOTH SERIES ON THIS AXIS, IN THE SAME UNITS, BUT NOT THE SAME QUANTITY. The boxes are
    # dry-season runoff already multiplied by the extractable fraction, i.e. 20% of the
    # basin's dry-season water; the demand lines are the fleet's full consumption.
    # That is the comparison the constraint makes, but a label reading "dry-season water"
    # invited the reader to take the boxes for the basin's water -- the Hai plots at 1.6
    # against a dry-season flow of 89.6. The label now names the allowance, not the resource.
    ax.set_ylabel("生态流量配额，与针对它的耗水需求" + "\n"
                  + r"（$10^8$ m$^3$ yr$^{-1}$；配额 = 枯水期径流 "
                  + rf"$\times$ {EXTRACTABLE:.2f}）",
                  fontsize=6.6, linespacing=1.3)
    for centre, year in zip(group_centres, (2030, 2060)):
        ax.text(centre, -0.155, str(year), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, fontweight="bold")
    # COMPUTED, SENTENCE INCLUDED. The v9 version slotted two counts into a fixed sentence
    # ("N 个流域已超出；捕集把第 M 个也推过线"). On v9.1's numbers the slots come out 0 and 1
    # and the sentence stops being one. Both the counts AND which sentence they license are
    # read off the data, so the title cannot assert a shape the numbers do not have.
    _s30 = basin_supply(avail, 2030); _d30 = basin_demand(SOLVED["ssp126"], 2030)
    _shared = [b for b in _d30.index if b in _s30.index]
    _un = [b for b in _shared if _d30.loc[b, "unabated"] > float(_s30.loc[b].median())]
    _cap = [b for b in _shared if _d30.loc[b, "full_capture"] > float(_s30.loc[b].median())]

    def _zh(codes) -> str:
        return "、".join(BASIN_NAMES_ZH.get(b, b) for b in codes)

    _added = [b for b in _cap if b not in _un]
    if not _un and not _cap:
        _title = ("现状机组与全量捕集的耗水需求都在生态流量配额之内" + chr(10)
                  + f"（{CEILING_YEAR} 年，集合中位数；最紧的流域见面板 b）")
    elif not _un and _added:
        _title = (f"现状机组的耗水需求全部在生态流量配额之内；" + chr(10)
                  + f"全量捕集把 {_zh(_added)} 推过配额线（{CEILING_YEAR} 年，集合中位数）")
    elif _un and _added:
        _title = (f"{_zh(_un)} 的现状需求已超出生态流量配额；" + chr(10)
                  + f"全量捕集再把 {_zh(_added)} 推过线（{CEILING_YEAR} 年，集合中位数）")
    else:
        _title = (f"{_zh(_un)} 的现状需求已超出生态流量配额；" + chr(10)
                  + f"全量捕集不改变越限流域的集合（{CEILING_YEAR} 年，集合中位数）")
    ax.set_title(_title, linespacing=1.25)

    handles = [
        Patch(facecolor=C126, alpha=0.45, edgecolor=C126, label="供给 SSP1-2.6（10 个成员）"),
        Patch(facecolor=C370, alpha=0.45, edgecolor=C370, label="供给 SSP3-7.0（10 个成员）"),
        Line2D([], [], color="black", marker="x", ls="none", ms=3.4, label="已求解成员"),
        Line2D([], [], color=DEMAND_UN, lw=1.5, label="需求：未改造（现状冷却方式）"),
        Line2D([], [], color=DEMAND_CAP, lw=1.5, label="需求：全量捕集（现状冷却方式）"),
        Line2D([], [], color=REALISED, marker="D", ls="none", ms=3.0, label="模型实际耗水"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.245), columnspacing=1.0, handlelength=1.4)
    ax.set_ylim(bottom=max(ax.get_ylim()[0], 0.05))
    return pd.DataFrame(records)


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


def two_rung_utilisation() -> pd.DataFrame:
    """Per-basin utilisation on both v9.1 water rungs, computed from inputs alone.

    DELEGATES TO FIG 1 rather than re-deriving. Both main figures quote the same basin
    stress numbers, and two independent derivations of one quantity is the first
    inconsistency a reviewer finds. Fig 1 owns the derivation; this reads it.

    Columns used here, all in %:
        env_stress_pct        full-capture CONSUMPTION / (dry-season runoff x 0.20)
        env_stress_base_pct   as-built     CONSUMPTION / same
        quota_stress_pct      full-capture calibrated WITHDRAWAL / basin residual allocation
        quota_stress_base_pct as-built     calibrated WITHDRAWAL / same
    """
    from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
    import plot_fig1_water_footprint as f1

    assumptions = OptimizationAssumptions()
    scenario = OptimizationScenario(experiment_id="fig2", description="fig2")
    # CEILING_YEAR and the full 20-member ensemble, named explicitly. Fig 1(c) stands on its
    # own 2060 / SSP3-7.0 hero year, which is a different and equally valid slice; the ED
    # basin closeup stands on this one. See plot_fig1_water_footprint.basin_drying.
    drying = f1.basin_drying(CEILING_YEAR, None)
    return f1.basin_summary(f1.site_table(assumptions, scenario), drying)


def panel_b(ax, util: pd.DataFrame) -> pd.DataFrame:
    """Both water rungs, per basin, as a share of their own allowance.

    TWO RUNGS, TWO BASES, ONE AXIS -- and that is only legitimate because both bars are
    RATIOS against their own denominator. The volumes behind them are not comparable: the
    environmental-flow rule acts on CONSUMPTION against 20% of dry-season runoff, the
    用水总量控制指标 acts on WITHDRAWAL against what the basin's allocation leaves after
    non-power users. Putting the two volumes on one axis would be exactly the aliasing this
    revision exists to undo; putting the two utilisations on one axis is the comparison.

    Log x: the values span 0.4% to 1078%, three decades. On a linear axis every basin but one
    collapses onto the spine.

    The open marker on each bar is the AS-BUILT claim, the bar is the claim if every unit took
    capture. The gap between them is what capture costs the basin; whether the marker is
    already past 100% is a different and more important fact, and the panel has to show both
    or it answers only one of the two questions.
    """
    rows = [b for b in BASIN_ORDER if b in util.index]
    y = np.arange(len(rows))
    h = 0.36
    series = [
        ("env", "生态流量档（耗水／枯水期径流 20%）", C_ENV_RUNG, +h / 2),
        ("quota", "用水总量指标档（取水／流域残余配额）", C_QUOTA_RUNG, -h / 2),
    ]
    left = 0.25
    for key, label, colour, off in series:
        full = util.loc[rows, f"{key}_stress_pct"].astype(float).to_numpy()
        built = util.loc[rows, f"{key}_stress_base_pct"].astype(float).to_numpy()
        ax.barh(y + off, np.maximum(full - left, 1e-9), height=h, left=left,
                color=colour, edgecolor="white", lw=0.3, zorder=3, label=label)
        # 白描边 + 深芯的双层竖线：条带颜色深浅不一，单色标记在其中一档上会看不见。
        ax.scatter(built, y + off, s=16, marker="|", color="white", linewidths=1.2, zorder=5)
        ax.scatter(built, y + off, s=16, marker="|", color="#222222", linewidths=0.6, zorder=6)
        # 千分位逗号在 SimHei 下占一个全角位，"1,078" 会被顶到画幅外；直接不加分隔符。
        for basin, yi, v in zip(rows, y + off, full):
            note = "\n残余配额近零" if (key == "quota" and basin == "K") else ""
            # 白色描边：长江 86、珠江 74、海河 96 三条的标签正好落在 100% 虚线上，
            # 不加 halo 就是红虚线穿过数字。移动标签会破坏"标签紧跟条端"的读法。
            ax.text(v * 1.12, yi, f"{v:.0f}{note}", va="center", ha="left",
                    fontsize=5.2, color="#333333", zorder=7, linespacing=1.35,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

    ax.axvline(100.0, color=C_LIMIT, lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax.text(100.0, -0.60, "配额上限", fontsize=5.4, color=C_LIMIT, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(left, 6000.0)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1 000"])
    ax.set_ylim(len(rows) - 0.35, -0.75)
    ax.set_yticks(y)
    ax.set_yticklabels([BASIN_NAMES_ZH.get(b, b) for b in rows], fontsize=6.0)
    ax.set_xlabel("占本档配额的比例（%，对数轴）", fontsize=6.2, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=5.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=5.1, loc="lower right", frameon=False, handlelength=1.4,
              handletextpad=0.5, borderaxespad=0.2, labelspacing=0.30)

    # THE NORTHWEST'S DENOMINATOR IS NOT A CLEAN NUMBER, AND THE PANEL SAYS SO -- on the bar
    # label itself, above. Its 2025 metered withdrawal (729.0) already exceeds its own 2030
    # 用水总量控制指标 (641.2), so the raw residual is NEGATIVE (-85.5). write_basin_caps
    # scales every user's claim pro rata (x0.8795) to fit the cap, which leaves 2.1 -- and
    # dividing by a near-zero residual is what produces four digits. The overshoot is real;
    # its exact multiple is not. The full sentence is in the figure footnote: an arrow drawn
    # across the panel to carry it collided with the Yangtze row and its own value label.
    return util.loc[rows, ["env_stress_base_pct", "env_stress_pct",
                           "quota_stress_base_pct", "quota_stress_pct"]]


LADDER_STATS = [
    ("空冷改造容量", "空冷改造\n容量", 2040, "GW"),
    ("捕集量", "CO$_2$\n捕集量", 2060, "Mt"),
    ("退役容量", "提前退役\n容量", 2040, "GW"),
]


def ladder_table():
    """The three-rung response BASE -> envonly -> oq, each step against its own floor.

    THE TWO STEPS PRICE TWO DIFFERENT INSTITUTIONS, which is the whole reason the water budget
    moved onto the official basis. BASE->envonly is the environmental-flow standard (a
    depletion rule, on consumption); envonly->oq is the 用水总量控制指标 (an allocation rule,
    on withdrawal). Under v9 the two were multiplied into one factor and no result could say
    which of them a response belonged to.

    Every step is reported against its own degeneracy floor (CLAUDE.md 二.4). A step that does
    not clear the floor is 未分辨 -- which is neither zero nor an effect. The floor is taken as
    the LARGER of the two arms' seed families: a floor measured on one run cannot judge a
    difference of two, and the looser of the pair is the one that has to be cleared.
    """
    needed = [BASE_SCENARIO] + ARMS + [treat_of(a) for a in ARMS]
    missing = [n for n in needed if not (RESULTS_DIR / n / "plant_detail.csv").exists()]
    if missing:
        print(f"  [panel c] not yet solved, skipped: {', '.join(missing)}")
        return None

    seed_names = seeds_of(SEED_ARM) + seeds_of(treat_of(SEED_ARM))
    have_seeds = all((RESULTS_DIR / n / "plant_detail.csv").exists() for n in seed_names)
    if not have_seeds:
        print("  [panel c] seed replicates incomplete -- steps drawn WITHOUT a degeneracy "
              "floor, and no step may be called an effect until the floor exists")

    rows = []
    for key, _label, year, unit in LADDER_STATS:
        base = national(BASE_SCENARIO, year)[key]
        ctrl = [national(a, year)[key] for a in ARMS]
        treat = [national(treat_of(a), year)[key] for a in ARMS]
        floor = float("nan")
        if have_seeds:
            floor = max(
                degeneracy_floor([national(n, year)[key] for n in seeds_of(SEED_ARM)]),
                degeneracy_floor([national(n, year)[key] for n in seeds_of(treat_of(SEED_ARM))]),
            )
        rows.append({"stat": key, "year": year, "unit": unit, "base": float(base),
                     "ctrl": float(np.median(ctrl)), "treat": float(np.median(treat)),
                     "ctrl_lo": float(min(ctrl)), "ctrl_hi": float(max(ctrl)),
                     "treat_lo": float(min(treat)), "treat_hi": float(max(treat)),
                     "floor": float(floor)})
    frame = pd.DataFrame(rows).set_index("stat")
    frame["step_env"] = frame["ctrl"] - frame["base"]
    frame["step_quota"] = frame["treat"] - frame["ctrl"]
    # NaN floor means "not measured yet", which is not the same statement as "measured and not
    # cleared". Kept as NaN so both the panel and the report can tell them apart.
    frame["env_resolved"] = np.where(frame["floor"].isna(), np.nan,
                                     frame["step_env"].abs() > frame["floor"])
    frame["quota_resolved"] = np.where(frame["floor"].isna(), np.nan,
                                       frame["step_quota"].abs() > frame["floor"])
    # A quantity that is zero in BASE cannot be expressed as a multiple of BASE. Capture is 0
    # everywhere at 2030 and retirement may be 0 at 2040; normalising against it yields inf,
    # which matplotlib draws as nothing at all and no one notices.
    frame["normalisable"] = frame["base"].abs() > 1e-9
    return frame


def panel_c(ax, table):
    """Each quantity's response to the two rungs, normalised so one axis can carry all three.

    Normalised to BASE = 1 because the three quantities are GW, Mt and GW-of-retirement and
    share no unit. The absolute values are printed, never read off this panel.

    The grey band is the degeneracy floor under the same normalisation: a point whose band
    reaches its predecessor is a step the solver's own search-path spread can produce on its
    own, and it is labelled 未分辨 rather than drawn as a result.
    """
    if table is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "面板 c 需 BASE 与两档水情景全部求解完成后绘制",
                ha="center", va="center", fontsize=6.0, color="#888888",
                transform=ax.transAxes)
        return

    drawable = table.index[table["normalisable"]]
    dropped = [s for s in table.index if s not in drawable]
    if dropped:
        print(f"  [panel c] BASE = 0，无法归一化，未画：{'、'.join(dropped)}（绝对值见下表）")
    stats = [s for s, *_ in LADDER_STATS if s in drawable]
    labels = [lab for s, lab, *_ in LADDER_STATS if s in drawable]
    if not stats:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "面板 c：三个量在 BASE 下均为 0，无法以 BASE 归一化",
                ha="center", va="center", fontsize=6.0, color="#888888",
                transform=ax.transAxes)
        return
    xs = np.arange(len(stats))
    rung_x = (-0.26, 0.0, 0.26)
    colours = (C_BASE_RUNG, C_ENV_RUNG, C_QUOTA_RUNG)
    names = ("不考虑水", "仅生态流量", "＋用水总量指标")

    for i, stat in enumerate(stats):
        r = table.loc[stat]
        base = float(r["base"])
        vals = [1.0, float(r["ctrl"]) / base, float(r["treat"]) / base]
        ax.plot([xs[i] + dx for dx in rung_x], vals, color="#AAAAAA", lw=0.7, zorder=2)
        if np.isfinite(r["floor"]):
            f = float(r["floor"]) / base
            for dx, v in zip(rung_x[1:], vals[1:]):
                ax.add_patch(plt.Rectangle((xs[i] + dx - 0.085, v - f / 2), 0.17, f,
                                           facecolor="#CCCCCC", edgecolor="none",
                                           alpha=0.55, zorder=3))
        for j, (dx, v, colour) in enumerate(zip(rung_x, vals, colours)):
            ax.scatter(xs[i] + dx, v, s=22, color=colour, edgecolor="white", lw=0.4,
                       zorder=5, label=names[j] if i == 0 else None)
        for j, key in enumerate(("env_resolved", "quota_resolved")):
            if r[key] is None or (isinstance(r[key], float) and np.isnan(r[key])):
                continue          # 地板还没测出来，不是"未分辨"
            if not bool(r[key]):
                ax.text(xs[i] + rung_x[j + 1], vals[j + 1], " 未分辨", fontsize=4.8,
                        color="#B02418", va="bottom", ha="center", zorder=6)

    ax.axhline(1.0, color="#BBBBBB", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_xlim(-0.55, len(stats) - 0.45)
    ax.set_ylabel("相对不考虑水的倍数（对数轴）", fontsize=6.2, labelpad=2.0)
    # 对数轴，且范围显式给：线性轴上 4.6 倍的那一点会被自动范围切掉，而 0.99 与 1.07 两条
    # 又会一起压在 1.0 那根线上——三个量共用一根轴的前提就是每十倍等距。
    _all = [v for s in stats for v in
            (1.0, float(table.loc[s, "ctrl"]) / float(table.loc[s, "base"]),
             float(table.loc[s, "treat"]) / float(table.loc[s, "base"]))]
    ax.set_yscale("log")
    _lo, _hi = min(_all) / 1.18, max(_all) * 1.30
    ax.set_ylim(_lo, _hi)
    # 显式刻度：这个范围内唯一的十进位刻度就是 1，默认只会给出一个标签加一排无标注的次刻度。
    _cand = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    ax.set_yticks([v for v in _cand if _lo <= v <= _hi])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _pos: f"{v:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="y", labelsize=5.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # 左下：空冷那一列从 1.0 一路冲到顶，左上正好被它的处理组点占住。
    ax.legend(fontsize=5.1, loc="lower left", frameon=False, handlelength=0.9,
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.28)


def main() -> None:
    print("Fig 2 - two water institutions on two water bases, and which one the fleet runs into")
    for scenario in list(SOLVED.values()) + [FROZEN]:
        require_current_vintage(scenario)
    print(f"  vintage gate passed (26 columns): {', '.join(list(SOLVED.values()) + [FROZEN])}")

    avail = load_availability()
    fleet_gw = as_built_capacity_gw()
    present = [b for b in BASIN_ORDER if b in set(avail["basin_code"])]
    # Basins that host coal capacity. J (Southwest Rivers) has none, so there is nothing there
    # to attach capture to and no ceiling to test; it is reported in stdout, not plotted.
    basins = [b for b in present if fleet_gw.get(b, 0.0) > 0.0]
    dropped = [b for b in present if b not in basins]
    if dropped:
        print(f"  basins with no coal capacity, reported but not plotted: "
              f"{', '.join(f'{b} {BASIN_NAMES_ZH.get(b, b)}' for b in dropped)}")
    print(f"  rung 1 supply = dry season x {EXTRACTABLE:.2f} extractable "
          f"= {USABLE:.4f} of raw dry-season runoff (consumption basis)")
    print("  rung 2 supply = 用水总量控制指标 - 非电既有取水, from inputs/water_basin_caps.csv "
          "(withdrawal basis)")

    # Was DOUBLE_COL[0] * 1.45 = 265 mm. Nature Water's double column is 183 mm, and a figure
    # submitted at 265 mm is scaled by 0.69 in production, dropping every 5.3 pt label to 3.7 pt
    # -- below the 5 pt floor. Sized to the column instead, and the type kept as-is.
    fig = plt.figure(figsize=(DOUBLE_COL[0], 8.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.60,
                             left=0.075, right=0.985, top=0.955, bottom=0.135)
    ax_a = fig.add_subplot(outer[0])
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.50)
    ax_b = fig.add_subplot(bottom[0])
    ax_c = fig.add_subplot(bottom[1])

    table_a = panel_a(ax_a, avail, basins)
    util = two_rung_utilisation()
    table_b = panel_b(ax_b, util)
    ceiling = allowance_ratio_table(avail, basins, fleet_gw)
    table_c = ladder_table()
    panel_c(ax_c, table_c)

    for ax, letter, xoff, yoff in (
        (ax_a, "a", -0.052, 1.11), (ax_b, "b", -0.170, 1.16), (ax_c, "c", -0.245, 1.16)
    ):
        panel_label(ax, letter, x=xoff, y=yoff)

    crossing = ceiling.index[ceiling["crosses_median"]].tolist()
    # Wrapped, not one long line: `savefig(bbox_inches="tight")` grows the canvas to whatever
    # the widest artist needs, so an unwrapped footnote silently doubles the figure width.
    note = (
        f"水规则分两档，落在两个不同的水量口径上，二者不相乘也不合并："
        f"生态流量档为节点耗水 ≤ 枯水期径流 x {EXTRACTABLE:.2f}（Richter et al. 2012, "
        f"doi:10.1002/rra.1511）；用水总量指标档为流域取水 ≤ 用水总量控制指标"
        f"（国办发〔2013〕2号 附件1）扣除非电既有取水（2025年中国水资源公报 表9）。"
        f"面板 a 为生态流量档，与图 1 口径一致；面板 b 两档并列，条为全部机组加装捕集后的用量，"
        f"竖线标记为现状机组的用量。集合为 2 个水文模型 x 5 个 GCM x 2 个 SSP = "
        f"{EXPECTED_MEMBERS} 个成员，箱线表示水资源本身的物理离散度，不是本模型的敏感性；"
        f"归因见附录图（plot_ed_variance_decomposition.py）。"
        f"面板 a 中的需求把冷却方式固定在现状配置，因此是“适应之前”的配额检验，"
        f"绿色菱形是模型完成空冷改造后的实际取水。"
        f"面板 c 的每一步都配简并度地板，未越过地板者标为“未分辨”，不作为效应。"
        f"图中为 {len(present)} 个一级流域中承载煤电容量的 {len(basins)} 个。"
        f"生态流量档超出配额的流域：{'、'.join(BASIN_NAMES_ZH.get(b, b) for b in crossing) or '无'}。"
    )
    fig.text(0.075, 0.004, cjk_fill(note, width=175), fontsize=5.3, color="#555555",
             va="bottom", linespacing=1.45)
    save_fig(fig, "fig2_constraint_response")

    # ---- every plotted number ------------------------------------------------
    print("\npanel a -- effective supply vs demand, 10^8 m3/yr")
    print(table_a.set_index(["year", "basin", "ssp"]).round(2).to_string())

    print("\n  demand/supply ratio against the ensemble median (%)")
    for year in (2030, 2060):
        sub = table_a[(table_a["year"] == year) & (table_a["ssp"] == "ssp126")]
        for _, row in sub.iterrows():
            print(f"    {year} {row['basin']} {BASIN_NAMES_ZH.get(row['basin'], ''):<18} "
                  f"unabated {row['unabated'] / row['supply_median'] * 100:6.0f}%   "
                  f"full-capture {row['full_capture'] / row['supply_median'] * 100:6.0f}%   "
                  f"realised {row['realised'] / row['supply_median'] * 100:6.0f}%")

    print("\npanel b -- both rungs, per basin, as a share of their own allowance (%)")
    print(table_b.round(1).to_string())
    _over_env = table_b.index[table_b["env_stress_pct"] > 100.0].tolist()
    _over_quota = table_b.index[table_b["quota_stress_pct"] > 100.0].tolist()
    _built_env = table_b.index[table_b["env_stress_base_pct"] > 100.0].tolist()
    _built_quota = table_b.index[table_b["quota_stress_base_pct"] > 100.0].tolist()
    print(f"    rung 1 环境流量 (consumption): over at full capture "
          f"{', '.join(_over_env) or 'none'}; already over as built "
          f"{', '.join(_built_env) or 'none'}")
    print(f"    rung 2 用水总量控制指标 (withdrawal): over at full capture "
          f"{', '.join(_over_quota) or 'none'}; already over as built "
          f"{', '.join(_built_quota) or 'none'}")
    print(f"    capacity behind the rung-2 breach: "
          f"{sum(fleet_gw.get(b, 0.0) for b in _over_quota):.0f} GW of {fleet_gw.sum():.0f} GW")

    # WHICH RUNG BINDS IS THE RESULT OF THE RECODE, so it is computed and printed rather than
    # asserted. Under v9 the two rules were one product and this question had no answer.
    print()
    print("  WHICH INSTITUTION IS THE BINDING ONE?")
    print("    The two rungs disagree about which basin is tightest, and that disagreement is")
    print("    itself a finding: the environmental-flow standard is a DEPLETION rule measured")
    print("    on consumption, the 用水总量控制指标 is an ALLOCATION rule measured on the")
    print("    withdrawal the 公报 actually meters, and a basin can be comfortable on one")
    print("    while breaching the other.")
    _tight_env = table_b["env_stress_pct"].idxmax()
    _tight_quota = table_b["quota_stress_pct"].idxmax()
    print(f"    tightest on rung 1: {_tight_env} {BASIN_NAMES_ZH.get(_tight_env, '')} "
          f"{table_b.loc[_tight_env, 'env_stress_pct']:.0f}%")
    print(f"    tightest on rung 2: {_tight_quota} {BASIN_NAMES_ZH.get(_tight_quota, '')} "
          f"{table_b.loc[_tight_quota, 'quota_stress_pct']:.0f}%")
    if "K" in table_b.index and _tight_quota == "K":
        print("    CAVEAT ON THE NORTHWEST: its 2025 metered withdrawal already exceeds its own")
        print("    2030 用水总量控制指标, so the raw residual is NEGATIVE and the published")
        print("    figure is the pro-rata-scaled one (inputs/water_basin_caps.csv carries both,")
        print("    residual_uncapped_1e8_m3 and residual_1e8_m3). The overshoot is a real")
        print("    statement about that basin's own 红线; its exact multiple is not a robust")
        print("    number, because the denominator is near zero by construction.")

    print("\n  unserved coal water demand, solved runs (Mm3)")
    for ssp, scenario in SOLVED.items():
        series = unserved_by_year(scenario)
        print(f"    {ssp:<7} " + "  ".join(f"{int(y)}: {v:8.2f}" for y, v in series.items())
              + f"   basins {','.join(unserved_basins(scenario)) or 'none'}")
    frozen = unserved_by_year(FROZEN)
    print(f"    {'noair':<7} " + "  ".join(f"{int(y)}: {v:8.2f}" for y, v in frozen.items())
          + f"   basins {','.join(unserved_basins(FROZEN)) or 'none'}")
    # MEASURED, NOT ASSUMED. v9's noair run leaned 18.5% of its objective on the big-M and
    # that is what this line used to describe. Under v9.1 it leans on nothing: with the
    # retrofit forbidden the fleet still meets both water rules, by retiring capacity and
    # capturing less. At 2040 retirement goes 4.6 -> 102.6 GW and capture 802 -> 704 Mt.
    print("    (noair is not plotted. It carries NO unserved water either: with the retrofit")
    print("     forbidden the model still complies, by retiring capacity and capturing less")
    print("     rather than by paying the big-M. Fig 3 and Fig 5 price that. Do not describe")
    print("     it as a ceiling the fleet cannot meet.)")

    def _verdict(flag) -> str:
        if flag is None or (isinstance(flag, float) and np.isnan(flag)):
            return "地板未测"
        return "分辨得出" if bool(flag) else "未分辨"

    print("\npanel c -- the three-rung response, BASE -> 仅生态流量 -> ＋用水总量指标")
    if table_c is None:
        print("    not drawn: the required scenarios are not all solved yet")
    else:
        print(table_c.round(2).to_string())
        for _stat, _r in table_c.iterrows():
            _u = _r["unit"]
            print(f"    {_stat} ({int(_r['year'])}, {_u}): "
                  f"{_r['base']:.1f} -> {_r['ctrl']:.1f} -> {_r['treat']:.1f}   "
                  f"step1 {_r['step_env']:+.1f} ({_verdict(_r['env_resolved'])}), "
                  f"step2 {_r['step_quota']:+.1f} ({_verdict(_r['quota_resolved'])}), "
                  f"floor {_r['floor']:.1f}")
        print("    A step below its floor is 未分辨 -- neither zero nor an effect. CLAUDE.md 二.4")
        print("    also records that on this model only the objective and the air-cooling")
        print("    conversion have ever cleared their floors; treat any other resolved step as")
        print("    a claim that needs checking, not as a default.")

    print(f"\nrung-1 allowance test, {CEILING_YEAR} "
          f"(full-capture consumption / dry-season environmental-flow allowance)")
    print(ceiling.round(2).to_string())
    crossing_gw = ceiling.loc[crossing, "fleet_gw"].sum()
    flow_gw = ceiling.loc[crossing, "flow_gw"].sum()
    print(f"\n  basins above the ceiling on the ensemble median: {', '.join(crossing)}")
    print(f"  standing capacity behind them (nearest basin, plants.csv):  {crossing_gw:.1f} GW "
          f"of {fleet_gw.sum():.1f} GW ({100 * crossing_gw / fleet_gw.sum():.1f}%)")
    print(f"  same, by the solver's own water-flow allocation at {CEILING_YEAR}: {flow_gw:.1f} GW "
          f"({100 * (flow_gw - crossing_gw) / crossing_gw:+.1f}% vs the nearest-basin figure)")
    unanimous = ceiling.index[ceiling["crosses"]].tolist()
    print(f"  basins above the ceiling in ALL {EXPECTED_MEMBERS} members: "
          f"{', '.join(unanimous) or 'none'} "
          f"({ceiling.loc[unanimous, 'fleet_gw'].sum():.1f} GW)")
    print("  per-basin capacity, nearest basin (GW):")
    print("    " + "  ".join(f"{b} {fleet_gw.get(b, 0.0):.0f}" for b in present))
    for basin in dropped:
        supply = basin_supply(avail, CEILING_YEAR).loc[basin].astype(float)
        demand = basin_demand(SOLVED["ssp126"], CEILING_YEAR)
        ratio = demand["full_capture"].get(basin, 0.0) / supply
        print(f"    not plotted: {basin} {BASIN_NAMES_ZH.get(basin, basin)} carries "
              f"{fleet_gw.get(basin, 0.0):.1f} GW of coal; ceiling ratio "
              f"{ratio.median():.3f} [{ratio.min():.3f}-{ratio.max():.3f}]")


if __name__ == "__main__":
    main()
