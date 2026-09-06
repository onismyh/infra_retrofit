"""Fig 4 - the no-regret pipeline core: what to build now, whatever water does.

The claim this figure carries:

    A no-regret pipeline core -- the edges every water scenario selects, carrying about
    four-fifths of 2060 transport work -- is robust infrastructure to build now, independent
    of how water evolves.

Every quantity in that sentence is computed at plot time from `results/` and `inputs/` and
printed to stdout; none is written into this file.

WHAT THIS FIGURE DELIBERATELY DOES NOT CLAIM
--------------------------------------------
An earlier version of this figure claimed that water redraws the CO2 network. That claim was
refuted by its own evidence and is not reasserted anywhere here, in any panel, title or note.
The control is computed in panel (c) rather than assumed:

  * two runs differing ONLY by the bias-correction switch -- same hydrology model, same SSP,
    objectives a few thousandths of a percent apart -- still disagree about several percent of
    their transport work. That is the DEGENERACY FLOOR: the re-routing the solver produces for
    no physical reason at all, because many corridor combinations cost almost exactly the same.
  * more binding vs less binding lands INSIDE that floor's range, on two different overlap
    metrics (work on shared edges, and work in the intersection over work in the union). The
    second metric exists so the null does not rest on one definition of overlap.

So the network difference attributable to water is not distinguishable from solver
arbitrariness. Only two summary statistics separate cleanly; panel (c) shows those two, and the
right-hand column names the ones that do not and gives their overlapping ranges. Which
statistics separate is TESTED, not assumed, so a change in the runs cannot leave this figure
asserting a difference that is no longer there. An honest null is publishable; a false positive
is not.

  (a) the union network across the usable water scenarios, every edge coloured by how many of
      them select it, the unanimous core drawn on top. Selection frequency only -- there is no
      two-map "before/after" contrast to imply a treatment effect.
  (b) how much 2060 transport work rides on edges selected in at least k of n scenarios, one
      line per scenario. The k = n point is the no-regret core.
  (c) the two statistics that genuinely separate binding from non-binding runs, and the
      degeneracy-floor control that every other statistic fails against.

Usage:  python scripts/plot_fig4_network_reconfiguration.py
"""

from __future__ import annotations

import sys
import textwrap
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (  # noqa: E402
    apply_style,
    load_country,
    load_map_provinces,
    COUNTRY_GBCODES,
    add_scs_inset,
    mainland_extent,
    to_map_xy,
    cjk_fill,
    save_fig,
    panel_label,
    RESULTS_DIR,
    DOUBLE_COL,
    load_basins,
    require_valid_scenarios,
    scenario_validity,
    same_model_runs,
    assert_same_vintage,
)

def apply_fig4_thesis_style() -> None:
    """Match the SimHei, tick, and spine system established for Fig. 1."""
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


apply_fig4_thesis_style()

ANNOTATION_SIZE = 6.3
DETAIL_SIZE = 6.0
LEGEND_SIZE = 6.4

ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = ROOT / "inputs"
YEARS = [2030, 2040, 2050, 2060]
MAP_YEAR = 2060

# ── Scenario admission ───────────────────────────────────────────────────────
# Only 26-column runs, checked against the CSV header by `require_current_vintage`. The
# 24-column vintage (WA_cwatm_370_annual, WA_wgap_*_annual, WA_wgap_370_dry, all
# RQ3_*/SA_*/WP_*, WA_grid_200km) has no air-cooling mechanism at all; its 2030 water use is
# 8272.3 Mm3 against 4806-5075 in the current code, i.e. a different feasible set. Pooling the
# two vintages would read a code change as a water signal.
ENSEMBLE = [
    "WA_cwatm_126_dry",
    "WA_wgap_126_dry",
    "WA_wgap_126_dry_nobias",
    "WA_cwatm_370_dry",
    "WA_cwatm_370_dry_nobias",
    "WA_cwatm_126_dry_wd085",
    "WA_cwatm_370_dry_wd085",
]
# No water constraint at all. Never folded into the frequency count; used only to ask whether
# the core survives dropping the water constraint entirely.
REFERENCE = "BASE"
# Water binds in BOTH the s=0 and the s=0.85 runs; the difference is intensity, not
# presence. At 2030 the s=0 run already has 33 of 373 water nodes at >=99.9% of budget
# carrying 20.1% of fleet demand, against 113 nodes and 46.2% at s=0.85. The old comment
# here ("water genuinely binds only in the wd085 pair") read a 3.5% basin AGGREGATE as
# slack, which is the error Fig 3(c) exists to expose. The treatment is therefore a
# step UP in binding intensity, not a switch from off to on.
BINDING = ["WA_cwatm_126_dry_wd085", "WA_cwatm_370_dry_wd085"]
NON_BINDING = [s for s in ENSEMBLE if s not in BINDING]

# `scenario_validity` rejects a run leaning >1% of its objective on slack. The wd085 pair sits
# at ~3.45%, so it fails that gate, yet it is the only pair in which water actually binds.
# Admitted explicitly, never silently: the slack share and the unserved volume are printed on
# the figure and in the report. What stays excluded is anything whose slack makes the solution
# non-physical:
#   RQ3_retire_only  objective is NaN (infeasible_or_unbounded)
#   RQ3_ccs_only     91% of the objective is slack penalty
#   *_wd085_noair    18.5% slack, 14.3% of 2030 water demand unserved -> its +51% objective is
#                    a penalty artefact, not an economic cost, so it is not plotted
SLACK_EXCEPTION = {"WA_cwatm_126_dry_wd085": 0.05, "WA_cwatm_370_dry_wd085": 0.05}

# Pairs differing only by the bias-correction switch: same hydrology, same SSP, objectives
# within 0.007%. Anything these two disagree about is solver arbitrariness by construction.
FLOOR_PAIRS = [("WA_wgap_126_dry", "WA_wgap_126_dry_nobias"),
               ("WA_cwatm_370_dry", "WA_cwatm_370_dry_nobias")]
# Pairs differing only by whether water binds.
TREATMENT_PAIRS = [("WA_cwatm_126_dry", "WA_cwatm_126_dry_wd085"),
                   ("WA_cwatm_370_dry", "WA_cwatm_370_dry_wd085")]

# Colours held locally, not added to plot_style: sibling figure scripts are editing that
# module concurrently.
# 配色统一到 .claude/CLAUDE.md §3.3。原来这里自成一套，有两处跨图撞义：
# 橙色 #EE7733 在项目色表里是"掺氨"的族色，却被拿来标海上封存汇；
# 蓝色同时承担频率色带、陆上汇点、"无水约束也被选中"三种含义，读者建立不起稳定映射。
# 现在管网一律 Blues 单色系（深=核心，浅=非核心），对照/零假设一律中性灰，处理用强调红。
CORE_COLOUR = "#08306B"      # Blues 最深端：全情景一致的核心管段
NONCORE_COLOUR = "#9ECAE1"   # Blues 浅端：并集里其余管段（图例与面板 b 用）
# 选中频率的离散色阶：ColorBrewer Blues 中高段，7 档一一对应 1/7 … 7/7。
# 不从 #F7FBFF 起步：流域底色是 #FAFAFA，近白的两档在底图上看不见。
# 浅端从 #B3D3EA 起，不是 #C6DBEF：陆地填充是 #F1F3F5，更浅的蓝在上面立不住。
# 深端保留 #08306B，与国界的 #3D4348 拉开黑度，两者不抢。
FREQ_STEPS = ["#B3D3EA", "#8FC0DE", "#69A7D0", "#458CBF", "#2A6FA8", "#12548D", "#08306B"]
FREQ_CMAP = ListedColormap(FREQ_STEPS)
# 面板 c 的比值色阶：连续量用连续编码，灰 → 强调红。
RATIO_CMAP = LinearSegmentedColormap.from_list(
    "ratio", ["#D9D9D9", "#BDBDBD", "#F4A582", "#D6604D", "#CC3311"])
SINK_EDGE = "#123F6B"        # 封存汇：白心 + 深蓝描边，与实心的管段分通道
ONSHORE_COLOUR = "white"     # 陆上封存汇（圆）
OFFSHORE_COLOUR = "white"    # 海上封存汇（方）
NULL_COLOUR = "#969696"      # 种子零假设 / 偏差订正对照，中性灰
TREAT_COLOUR = "#CC3311"     # 水约束处理，强调红
B_LINE_COLOUR = "#2F80ED"    # 与图 1 一致的亮蓝色：面板 b 水情景中位数
B_FILL_COLOUR = "#DCEBFA"    # 水情景范围，保持轻量而可见
# 兼容旧引用
NB_COLOUR = NULL_COLOUR
BD_COLOUR = TREAT_COLOUR

# Every summary statistic is TESTED for separation; only the ones that pass are plotted.
STAT_SPECS = {
    "total_length_km": ("管道总长度\n（1000 km）", 1e-3, ".2f"),
    "haul_km": ("流量加权运距\n（km/t）", 1.0, ".1f"),
    "active_sinks": ("在用封存汇数\n（共 35 个）", 1.0, ".1f"),
    "offshore_share": ("海上注入量\n占比（%）", 1.0, ".2f"),
    "pipe_capex_bn": ("管道资本支出\n（十亿元，累计）", 1.0, ".1f"),
    "injected_mtpa": ("CO$_2$ 注入量\n（Mt yr$^{-1}$）", 1.0, ".1f"),
}


def read(scenario: str, name: str) -> pd.DataFrame:
    path = RESULTS_DIR / scenario / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} - run {scenario} first")
    return pd.read_csv(path)


def objective_1e12(scenario: str) -> float:
    """Global objective in 1e12 CNY, read from the run's own JSON. Never retyped."""
    import json

    path = RESULTS_DIR / f"{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} - run {scenario} first")
    return float(json.loads(path.read_text(encoding="utf-8"))["global_objective_cny"]) / 1e12


def require_current_vintage(scenario: str) -> bool:
    """Vintage gate on the CSV header, not on a name pattern.

    Pre-2026-08-10 runs write 24 columns and have no wet-to-dry conversion mechanism. Filling
    the two absent columns with zero would put a code change on the figure, so a stale run is
    dropped rather than repaired.
    """
    path = RESULTS_DIR / scenario / "plant_detail.csv"
    if not path.exists():
        print(f"  [skip] {scenario}: not solved yet")
        return False
    columns = pd.read_csv(path, nrows=0).columns
    missing = {"air_cooled_share", "already_air_share"} - set(columns)
    if missing:
        print(f"  [skip] {scenario}: stale {len(columns)}-column vintage, missing "
              f"{sorted(missing)} - air-blind, different feasible set")
        return False
    return True


def admit(names: list[str]) -> list[str]:
    """Current-vintage, physically meaningful runs, plus the binding pair under its exception."""
    current = [n for n in names if require_current_vintage(n)]
    keep = require_valid_scenarios(current, warn=False)
    for name in current:
        if name in keep or name not in SLACK_EXCEPTION:
            continue
        verdict = scenario_validity(name)
        share = verdict["slack_share"]
        if verdict["statuses"] == ["optimal"] and share <= SLACK_EXCEPTION[name]:
            keep.append(name)
            print(f"  [admit] {name}: slack {share:.2%} of objective, above the 1% default gate "
                  "but near-feasible; the only runs in which water binds")
    for name in current:
        if name not in keep:
            print(f"  [skip] {name}: {scenario_validity(name)['reason']}")
    return [n for n in names if n in keep]


# ── shared geometry / sink helpers ───────────────────────────────────────────
def candidate_edges() -> pd.DataFrame:
    """The candidate corridor layer: digitised geometry and length for every candidate edge."""
    return pd.read_csv(INPUTS_DIR / "pipeline_candidate_edges.csv")


def edge_geometry(cand: pd.DataFrame) -> pd.Series:
    """Real pipeline centrelines, keyed by edge_id, as (lon, lat) arrays.

    An edge is drawn along its actual digitised route rather than as a straight line between
    endpoints. All active edges are present in that layer, so nothing is dropped.
    """
    from shapely import wkt

    out = {}
    for edge_id, geom in zip(cand["edge_id"].astype(str), cand["geometry_wkt"]):
        if not isinstance(geom, str):
            continue
        try:
            coords = np.asarray(wkt.loads(geom).coords)
        except Exception:  # noqa: BLE001 - a malformed row loses one edge, not the figure
            continue
        # 在这里一次性投到 MAP_CRS：下游的 LineCollection 直接吃米制坐标，
        # 不必在绘图函数里再散落一遍投影逻辑。
        x, y = to_map_xy(coords[:, 0], coords[:, 1])
        out[edge_id] = np.column_stack([x, y])
    return pd.Series(out)


def offshore_hub_ids() -> set[str]:
    hubs = pd.read_csv(INPUTS_DIR / "storage_hubs.csv")
    flag = hubs["offshore"].astype(str).str.lower().isin({"1", "true", "yes"})
    return set(hubs.loc[flag, "storage_hub_id"].astype(str))


def hub_coords() -> pd.DataFrame:
    hubs = pd.read_csv(INPUTS_DIR / "storage_hubs.csv")
    return hubs.set_index(hubs["storage_hub_id"].astype(str))[["longitude", "latitude"]]


OFFSHORE = offshore_hub_ids()
HUBS = hub_coords()


def active_edges(scenario: str, year: int = MAP_YEAR) -> pd.DataFrame:
    edges = read(scenario, "network_edges")
    return edges[(edges["year"] == year) & (edges["edge_active"].astype(float) > 0)].copy()


def selection_count(scenarios: list[str], year: int = MAP_YEAR) -> pd.Series:
    """How many of the usable scenarios select each edge. n is small and printed on the figure.

    This is the honest stand-in for climate-member selection frequency: the 20-member ensemble
    has not been solved, so the quantity is defined over the runs that exist.
    """
    counts: Counter = Counter()
    for scenario in scenarios:
        for edge_id in active_edges(scenario, year)["edge_id"].astype(str):
            counts[edge_id] += 1
    return pd.Series(counts, dtype=float).sort_index()


def union_capacity(scenarios: list[str], year: int = MAP_YEAR) -> pd.Series:
    """Median built capacity per edge across the scenarios that select it, Mtpa.

    The map is a union across runs, so an edge has no single capacity. The median over the
    runs that select it is used for line width, and nothing else depends on it.
    """
    frames = []
    for scenario in scenarios:
        edges = active_edges(scenario, year)
        frames.append(edges.set_index(edges["edge_id"].astype(str))["total_capacity_mtpa"])
    return pd.concat(frames, axis=1).median(axis=1)


def mean_injection(scenarios: list[str], year: int = MAP_YEAR) -> pd.Series:
    """Mean injection per storage hub across the scenarios, Mtpa (absent = not used = 0)."""
    frames = []
    for scenario in scenarios:
        store = read(scenario, "storage_utilization")
        store = store[(store["year"] == year) & (store["storage_use_mtpa"] > 1e-6)]
        frames.append(store.groupby(store["storage_hub_id"].astype(str))["storage_use_mtpa"].sum())
    # A hub absent from a run was genuinely not used in it, so zero is the true value here.
    return pd.concat(frames, axis=1).fillna(0.0).mean(axis=1)


def network_stats(scenario: str, year: int = MAP_YEAR) -> dict:
    """Network summary for one scenario/year. Every field traces to a CSV column."""
    edges = active_edges(scenario, year)
    flows = read(scenario, "co2_flow_direction")
    flows = flows[(flows["year"] == year) & (flows["net_flow_mtpa"].abs() > 1e-6)]
    store = read(scenario, "storage_utilization")
    store = store[(store["year"] == year) & (store["storage_use_mtpa"] > 1e-6)]
    costs = read(scenario, "cost_breakdown")

    work = (flows["net_flow_mtpa"].abs() * flows["length_km"]).sum()
    tonnes = flows["net_flow_mtpa"].abs().sum()
    hub_ids = store["storage_hub_id"].astype(str)
    injected = store["storage_use_mtpa"].to_numpy(float)
    offshore_inj = injected[hub_ids.isin(OFFSHORE).to_numpy()].sum()

    return {
        "total_length_km": edges["length_km"].sum(),
        "new_length_km": edges.loc[edges["new_capacity_mtpa"] > 1e-6, "length_km"].sum(),
        # Flow-weighted mean haul: the distance a tonne of CO2 actually travels. Robust to the
        # many near-zero-flow edges an LP leaves active, unlike a mean over edges.
        "haul_km": work / tonnes if tonnes > 0 else np.nan,
        "active_sinks": float(hub_ids.nunique()),
        "offshore_sinks": float(hub_ids[hub_ids.isin(OFFSHORE)].nunique()),
        "offshore_share": 100.0 * offshore_inj / injected.sum() if injected.sum() > 0 else 0.0,
        "injected_mtpa": injected.sum(),
        # Whole-horizon pipe capex: an edge built in 2040 still serves 2060.
        "pipe_capex_bn": costs.loc[costs["category"] == "pipe_capex", "cost_cny"].sum() / 1e9,
        "transport_opex_bn":
            costs.loc[costs["category"] == "transport_opex", "cost_cny"].sum() / 1e9,
        "n_edges": float(len(edges)),
        "transport_work": work,
    }


def edge_work(scenario: str, year: int = MAP_YEAR) -> pd.Series:
    """Transport work (Mt x km) per edge in one run/year, zero-flow edges dropped.

    Work-weighting, not edge counting: an LP leaves many near-zero-flow edges active and a raw
    edge-overlap count is dominated by them, which would make any two runs look different for
    no material reason.
    """
    flows = read(scenario, "co2_flow_direction")
    flows = flows[(flows["year"] == year) & (flows["net_flow_mtpa"].abs() > 1e-6)]
    work = flows["net_flow_mtpa"].abs() * flows["length_km"]
    return work.groupby(flows["edge_id"].astype(str)).sum()


def shared_transport_work(a: str, b: str, year: int = MAP_YEAR) -> float:
    """Percent of transport work on edges both runs use, symmetrised over the two directions.

    Symmetric on purpose: the one-directional form differs by up to 3 pp depending on which run
    is the reference, which is enough to flip a marginal verdict.
    """
    wa, wb = edge_work(a, year), edge_work(b, year)
    if wa.sum() <= 0 or wb.sum() <= 0:
        return np.nan
    forward = wb[wb.index.isin(wa.index)].sum() / wb.sum()
    backward = wa[wa.index.isin(wb.index)].sum() / wa.sum()
    return 100.0 * 0.5 * (forward + backward)


def intersection_over_union_work(a: str, b: str, year: int = MAP_YEAR) -> float:
    """Second, independent overlap metric: work in the intersection over work in the union.

    Stricter than the shared-edge measure because it also penalises using the same edge at a
    different intensity. Reported alongside it because a null should not hinge on one
    definition of overlap.
    """
    wa, wb = edge_work(a, year), edge_work(b, year)
    both = sorted(set(wa.index) & set(wb.index))
    inter = float(np.minimum(wa.reindex(both), wb.reindex(both)).sum())
    union = float(wa.sum() + wb.sum() - inter)
    return 100.0 * inter / union if union > 0 else np.nan


def unserved_share(scenario: str, year: int) -> float:
    """Unserved water as a share of total (served + unserved) demand, in percent."""
    use = read(scenario, "resource_use")
    served = use[(use["resource_type"] == "water") & (use["year"] == year)]["used"].sum()
    slack = read(scenario, "slack_detail")
    unserved = slack[slack["year"] == year]["slack_value"].sum()
    total = served + unserved
    return 100.0 * unserved / total if total > 0 else 0.0


def unserved_volume_mm3(scenario: str) -> pd.Series:
    slack = read(scenario, "slack_detail")
    water = slack[slack["constraint_type"] == "water_supply"]
    if water.empty:
        return pd.Series(0.0, index=YEARS)
    return (water.groupby("year")["slack_value"].sum() / 1e6).reindex(YEARS).fillna(0.0)


def capture_trajectory(scenario: str) -> pd.Series:
    plants = read(scenario, "plant_detail")
    return plants.groupby("year")["captured_mt"].sum().reindex(YEARS)


def work_share_by_threshold(scenario: str, counts: pd.Series, n: int,
                            year: int = MAP_YEAR) -> pd.Series:
    """Share of this run's transport work carried by edges selected in at least k of n runs."""
    flows = read(scenario, "co2_flow_direction")
    flows = flows[(flows["year"] == year) & (flows["net_flow_mtpa"].abs() > 1e-6)]
    work = flows["net_flow_mtpa"].abs() * flows["length_km"]
    ids = flows["edge_id"].astype(str)
    total = work.sum()
    out = {}
    for k in range(n, 0, -1):
        selected = set(counts.index[counts.round() >= k])
        out[k] = 100.0 * work[ids.isin(selected)].sum() / total
    return pd.Series(out)


# ── panel (a): the union network, coloured by selection frequency ─────────────
MAP_LON = (72.0, 136.0)
MAP_LAT = (17.0, 55.0)
MAP_ASPECT = 1.18
MAP_WH_RATIO = (MAP_LON[1] - MAP_LON[0]) / ((MAP_LAT[1] - MAP_LAT[0]) * MAP_ASPECT)


def panel_a(fig, ax, cax, legend_anchor, counts: pd.Series, capacity: pd.Series,
            geoms: pd.Series, injection: pd.Series, n: int, core: list[str],
            core_len: float, union_len: float, core_work: float) -> None:
    """无悔核心地图 —— 只分核心 / 非核心两级。

    原来用 7 级频率色带（1/7 … 7/7）。在 183 mm 双栏里这张地图宽不到 90 mm，
    7 档蓝色实际只能读出"深/浅"两级；色带的精度是浪费的，还占掉一条色标的高度。
    图要说的也只有两件事：哪些管段每个水情景都选，哪些不是。所以直接画两级。
    """
    basins = load_basins()
    if basins is None:
        raise SystemExit("data/ChinaBasins/basin_l1.gpkg missing")
    # 陆地作为底色块，不画流域界 —— 本图讲管网，流域线在这里是纯噪点。
    basins.plot(ax=ax, facecolor="#F1F3F5", edgecolor="none", zorder=0)
    # 省界只给一点地理骨架，细到几乎看不见即可。
    load_map_provinces().boundary.plot(ax=ax, linewidth=0.13, edgecolor="#DFE3E7", zorder=1)
    _country = load_country()
    _country[_country["GBCODE"].isin(COUNTRY_GBCODES)].plot(
        ax=ax, facecolor="none", edgecolor="#3D4348", linewidth=0.45, zorder=1.5)

    core_set = set(core)
    # 分档边界放在半整数上（(k-0.5)/n），一档对应一个可达到的计数。
    # 用 np.linspace(0, 1, n+1) 会把边界压在 1.0 上，而 BoundaryNorm 把等于末边界的值
    # 归进最后一档 —— 6/7 和 7/7 会拿到同一个 RGB，图面上的"核心"就凭空多出十几条。
    norm = BoundaryNorm((np.arange(1, n + 2) - 0.5) / n, FREQ_CMAP.N)
    # 非核心在下、核心在上：结论是关于核心的，不能被压在底下。
    for is_core in (False, True):
        segments, colours, widths = [], [], []
        for edge_id, count in counts.items():
            if (edge_id in core_set) != is_core:
                continue
            coords = geoms.get(edge_id)
            if coords is None:
                continue
            segments.append(coords)
            colours.append(FREQ_CMAP(norm(count / n)))
            cap = max(float(capacity.get(edge_id, 0.0)), 0.0)
            # 线宽只编码容量，颜色只编码频率：两个通道各管一件事。
            # 用容量的线性比例而不是 sqrt：容量本身近似两档（229 条 20 Mt、13 条 40 Mt），
            # sqrt 会把这两档压得更近，等于白丢一个通道。
            frac = min(cap / 40.0, 1.0)
            widths.append((0.60 + 1.20 * frac) if is_core else (0.28 + 0.42 * frac))
        if not segments:
            continue
        ax.add_collection(LineCollection(
            segments, colors=colours, linewidths=widths, capstyle="round",
            alpha=1.0 if is_core else 0.85, zorder=4 if is_core else 3))

    biggest = max(float(injection.max()), 1e-9)

    def _sink_layer(a) -> None:
        for hub_id, inj in injection.items():
            if hub_id not in HUBS.index or inj <= 1e-6:
                continue
            lon, lat = HUBS.loc[hub_id, ["longitude", "latitude"]]
            hx, hy = to_map_xy([float(lon)], [float(lat)])
            is_off = hub_id in OFFSHORE
            # 空心：白填充 + 深色描边。汇点原来用 #3182BD 实心，正好落在色阶第 4 档上，
            # 点和线在同一个通道里，读者分不开哪个是汇、哪个是管段。
            a.scatter(hx, hy, s=4 + 22 * np.sqrt(inj / biggest),
                      marker="s" if is_off else "o",
                      facecolor="white", edgecolor=SINK_EDGE,
                      linewidth=0.55 if is_off else 0.45,
                      zorder=6 if is_off else 5.5)

    _sink_layer(ax)

    ax.set_title(f"跨水情景稳定的 CO$_2$ 管网核心（{MAP_YEAR} 年）", pad=5.0)
    ax.text(
        0.018, 0.030,
        f"{len(core)} 条核心管段｜承担 {core_work:.0f}% 输送量\n"
        f"长度 {core_len:,.0f} km（并集的 {100 * core_len / union_len:.0f}%）",
        transform=ax.transAxes, fontsize=ANNOTATION_SIZE, color="#000000",
        fontweight="bold", va="bottom", linespacing=1.45,
        bbox=dict(boxstyle="square,pad=0.28", facecolor="white", edgecolor="none", alpha=0.82),
        zorder=8,
    )

    mainland_extent(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    scs_ax = add_scs_inset(
        ax.get_figure(), ax, draw=_sink_layer,
        # 右下角的海面上：既是它自己的地理位置，也不遮挡珠三角一带的管段与封存汇。
        axes_rect=[0.790, 0.015, 0.155, 0.215],
    )
    # 全局样式隐藏上、右轴线；南海小图必须单独恢复完整边框。
    for spine in scs_ax.spines.values():
        spine.set_visible(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FREQ_CMAP), cax=cax,
                       orientation="horizontal", ticks=[(k - 0.0) / n for k in range(1, n + 1)],
                       spacing="proportional")
    cax.set_xticklabels([f"{k}/{n}" for k in range(1, n + 1)], fontsize=DETAIL_SIZE)
    cax.tick_params(length=2.0, width=0.6, pad=1.5)
    bar.outline.set_linewidth(0.5)
    cax.set_title("管段入选频次", fontsize=DETAIL_SIZE, pad=3.0)

    handles = [
        # 图例线宽 = 图面上的最大线宽（40 Mt yr^-1），不另设一个更粗的示意值。
        Line2D([], [], color=CORE_COLOUR, lw=1.8,
               label=f"核心管段（{n}/{n}）"),
        Line2D([], [], color=CORE_COLOUR, lw=1.2,
               label="管容量：细 20、粗 40 Mt yr$^{-1}$"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", mew=0.45,
               markeredgecolor=SINK_EDGE, markersize=3.6, label="陆上封存汇"),
        Line2D([], [], marker="s", color="none", markerfacecolor="white", mew=0.55,
               markeredgecolor=SINK_EDGE, markersize=3.4, label="海上封存汇"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=legend_anchor,
               fontsize=LEGEND_SIZE, frameon=False, handlelength=1.6, handletextpad=0.5,
               labelspacing=0.34, ncol=2, columnspacing=1.2)


def panel_b(ax, curves: pd.DataFrame, counts: pd.Series, n: int, null=None) -> None:
    """输送量占比对选中阈值 —— 用包络带，不是十条线。

    原来一条情景画一条线（7 条），加中位数、加种子零假设，一个 60 mm 宽的面板上十条线。
    最关键的一组对比是"水扰动的离散度 vs 仅换种子的离散度"，却只靠线的粗细和红色深浅区分，
    两者同属红族，实际分不开。改成两条包络带：谁更宽一眼可见，而这正是本面板的全部论点。
    """
    ks = list(range(n, 0, -1))
    block = curves[ks]
    lo, hi, median = block.min(), block.max(), block.median()

    ax.fill_between(ks, lo, hi, color=B_FILL_COLOUR, alpha=0.95, lw=0, zorder=2,
                    label=f"水情景范围（n={n}）")
    ax.plot(ks, median, color=B_LINE_COLOUR, lw=1.65, zorder=5, label="水情景中位数")
    ax.scatter([n], [median[n]], s=28, facecolor=B_LINE_COLOUR, edgecolor="white",
               linewidth=0.5, zorder=6)

    if null is not None and null.get("median_curve") is not None:
        nk = null["k"]
        xs = [1 + (k - 1) * (n - 1) / max(nk - 1, 1) for k in range(nk, 0, -1)]
        nlo, nhi = null.get("lo_curve"), null.get("hi_curve")
        if nlo is not None and nhi is not None:
            # 浅灰实心 + 虚线上下沿，不用 hatch：斜线填充在 5 pt 尺度上印刷会糊成灰块，
            # 而且和面板 a 的管段线在视觉上抢"细线"这个通道。
            ax.fill_between(xs, [nlo[k] for k in range(nk, 0, -1)],
                            [nhi[k] for k in range(nk, 0, -1)],
                            facecolor="#E8E8E8", alpha=0.90, lw=0, zorder=3,
                            label=f"随机种子范围（n={nk}）")
        nc = null["median_curve"]
        ax.plot(xs, [nc[k] for k in range(nk, 0, -1)], color="#7A7A7A", lw=1.1,
                ls=(0, (3.2, 1.8)), zorder=4, label="随机种子中位数")

    ax.text(n - 0.28, median[n] - 5.0,
            f"{int(counts.round().ge(n).sum())} 条｜{median[n]:.0f}%",
            fontsize=ANNOTATION_SIZE, color="#000000", ha="left", va="top",
            fontweight="bold")

    ax.set_xlim(n + 0.5, 0.5)
    ax.set_ylim(min(60.0, float(block.to_numpy().min()) - 4), 103.0)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k}/{n}" for k in ks])
    ax.set_xlabel("管段至少入选的情景数", labelpad=3)
    ax.set_ylabel(f"{MAP_YEAR} 年输送量占比（%）", labelpad=3)
    ax.grid(axis="y", lw=0.45, alpha=0.22, color="#9AA0A6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#5A5A5A")
    ax.set_title("核心管段承担的输送量", pad=5.0)
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, frameon=False, handlelength=1.6,
              labelspacing=0.30, borderpad=0.2)


# 面板 c 的短标签。横轴是无量纲比值，单位不进标签。
SHORT_LABELS = {
    "total_length_km": "管道总长度",
    "haul_km": "流量加权运距",
    "active_sinks": "在用封存汇数",
    "offshore_share": "海上注入量占比",
    "pipe_capex_bn": "管道资本支出",
    "injected_mtpa": "CO$_2$ 注入量",
    "共用管段上的\n输送量": "共用管段输送量",
    "交集输送量 /\n并集输送量": "交集 / 并集输送量",
}


def panel_tests(ax, tests: dict, control: pd.DataFrame) -> list[dict]:
    """检验汇总：每项统计量对它自己的简并度地板的比值。

    合并了原来的 (c)(d) 两个面板。原 (c) 画"核心里有多少不需要水约束"（管段条数），
    原 (d) 画两项重合度指标对地板的位置，两个面板口径不同、共占半幅图，而且 (d) 的标题
    替那些被移出画布的统计量下了结论。

    现在一个无量纲轴列全 8 项：横轴是"两组极差之间的间隙 / 该统计量的简并度地板"。
    越过 1.0 才算可分辨。这样"哪些能说、哪些不能说"一次看完，也和 fig5 面板 c 同构。
    """
    rows = []
    for key, result in tests.items():
        floor = float(result["floor"])
        ratio = float(result["range_gap"]) / floor if floor > 0 else np.nan
        # 只取统计量的名字，丢掉单位：本面板的横轴是无量纲比值，单位在这里不承载信息，
        # 却会把 y 标签撑到吃掉左侧版面。
        rows.append({"label": SHORT_LABELS.get(key, STAT_SPECS[key][0].split("\n")[0]),
                     "ratio": ratio, "sep": bool(result["separates"])})

    # 两项重合度指标：对照 = 偏差订正对，处理 = 约束更紧/更松对。地板取对照的极差。
    for metric in control["metric"].unique():
        sub = control[control["metric"] == metric]
        c = sub[sub["kind"] == "control"]["overlap"].to_numpy(float)
        t = sub[sub["kind"] == "treatment"]["overlap"].to_numpy(float)
        if not len(c) or not len(t):
            continue
        floor = float(c.max() - c.min())
        if t.min() > c.max():
            gap = float(t.min() - c.max())
        elif c.min() > t.max():
            gap = float(c.min() - t.max())
        else:
            gap = 0.0
        ratio = gap / floor if floor > 0 else np.nan
        rows.append({"label": SHORT_LABELS.get(metric, metric.replace("\n", "")),
                     "ratio": ratio, "sep": ratio > 1.0})

    rows = sorted(rows, key=lambda r: (0.0 if np.isnan(r["ratio"]) else r["ratio"]))
    y = np.arange(len(rows))
    vals = [0.0 if np.isnan(r["ratio"]) else r["ratio"] for r in rows]
    # 比值是连续量，用连续编码：灰 → 强调红。原来按"越过/未越过"两色填充，
    # 把 3.39x 和 1.46x 画成一样深，也把 0.19x 和 0.00x 画成一样浅 —— 丢掉了强度信息。
    vmax = max(max(vals), 1.6)
    colours = [RATIO_CMAP(min(v / vmax, 1.0)) for v in vals]
    # 零值给一段最小可见长度：0.00x 是"两组极差完全重叠"这一实测结果，
    # 画成空白会让读者以为这一项没有算。
    # 零值行给一段固定长度的柱头（满量程的 1.2%）。0.00x 是"两组极差完全重叠"
    # 这一实测结果，画成一片空白会让读者以为这几项根本没算；柱旁已标 "0.00x"，
    # 不会被读成非零。
    stub = 0.012 * max(max(vals), 1.6)
    ax.barh(y, [max(v, stub) for v in vals], height=0.56, color=colours,
            edgecolor="white", lw=0.3, zorder=3)
    ax.axvline(1.0, color="#3F3F3F", lw=0.9, ls=(0, (3, 2)), zorder=4)
    top = max(max(vals) * 1.35, 1.6)
    ax.text(1.0, len(rows) - 0.30, "1.0× 分辨阈值", fontsize=DETAIL_SIZE, color="#000000",
            ha="center", va="bottom")
    for yi, r, v in zip(y, rows, vals):
        ax.text(v + top * 0.015, yi, "—" if np.isnan(r["ratio"]) else f"{v:.2f}×",
                va="center", ha="left", fontsize=ANNOTATION_SIZE,
                color="#000000" if r["sep"] else "#666666",
                fontweight="bold" if r["sep"] else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlim(0, top)
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.set_xlabel("水情景差异相对于求解不确定性的倍数", labelpad=3)
    # 左侧 spine 已隐藏，刻度线却还在，每个标签右边多出一根小横杠。
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.45, alpha=0.20, color="#9AA0A6")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#5A5A5A")
    ax.set_title("网络指标对水约束的响应", pad=5.0)
    ax.legend(handles=[Patch(facecolor=RATIO_CMAP(0.95), label=">1：可分辨"),
                       Patch(facecolor=RATIO_CMAP(0.05), label="≤1：未分辨")],
              loc="lower right", fontsize=LEGEND_SIZE, frameon=False, handlelength=1.1,
              handletextpad=0.5, labelspacing=0.28, borderpad=0.2)
    return rows


# ── panel (c): what separates, and the floor that says the rest does not ─────
def separation(stats: dict[str, dict], key: str) -> dict:
    """Does `key` separate binding from non-binding by MORE than solver degeneracy?

    Two tests, and a statistic must pass both:

      1. the binding and non-binding ranges do not overlap;
      2. the gap between those ranges exceeds the bias-correction control -- the spread the solver
         produces between FLOOR_PAIRS, which differ only by a bias-correction switch and are
         therefore physically identical by construction.

    Test 2 is the one that matters and was missing. Disjoint ranges alone are a false positive
    whenever n is small: with five non-binding and two binding runs, a statistic can separate
    cleanly and still move less than re-solving the same physical problem moves it. Flow-weighted
    haul is exactly that case -- on the current snapshot the ranges are disjoint by 0.44 km/t
    against a 4.82 km/t floor (the numbers were 2.33 and 3.63 when this note was written; both
    are recomputed at run time and printed below, so neither is quoted from here) -- so it
    was reported as a real effect until the floor was applied here. This figure exists to
    avoid attributing degeneracy to water; the test has to carry that burden, not the eye.
    """
    nb = np.array([stats[s][key] for s in NON_BINDING if s in stats], float)
    bd = np.array([stats[s][key] for s in BINDING if s in stats], float)
    disjoint = bool(bd.max() < nb.min() or bd.min() > nb.max())
    if bd.min() > nb.max():
        range_gap = float(bd.min() - nb.max())
    elif nb.min() > bd.max():
        range_gap = float(nb.min() - bd.max())
    else:
        range_gap = 0.0
    pairs = [(a, b) for a, b in FLOOR_PAIRS if a in stats and b in stats]
    floor = max((abs(stats[a][key] - stats[b][key]) for a, b in pairs), default=0.0)
    nb_med, bd_med = float(np.median(nb)), float(np.median(bd))
    return {
        "nb": nb, "bd": bd,
        "disjoint": disjoint,
        "range_gap": range_gap,
        "floor": float(floor),
        # The verdict the figure is allowed to draw.
        "separates": bool(disjoint and range_gap > floor),
        "nb_median": nb_med, "bd_median": bd_med,
        "delta": bd_med - nb_med,
        "pct": (bd_med - nb_med) / nb_med * 100 if nb_med else np.nan,
    }


def panel_c(axes, stats: dict[str, dict], separating: list[str], control: pd.DataFrame) -> None:
    """The separating statistics, then the degeneracy-floor control.

    `axes` must hold len(separating) + 1 entries.
    """
    if len(axes) != len(separating) + 1:
        raise ValueError(f"panel_c needs {len(separating) + 1} axes, got {len(axes)}")

    for ax, key in zip(axes, separating):
        label, scale, fmt = STAT_SPECS[key]
        result = separation(stats, key)
        nb, bd = result["nb"] * scale, result["bd"] * scale
        for x, vals, colour in ((0, nb, NB_COLOUR), (1, bd, BD_COLOUR)):
            ax.scatter(np.full(len(vals), x) + np.linspace(-0.07, 0.07, len(vals)), vals,
                       s=13, facecolor=colour, edgecolor="white", linewidth=0.3, zorder=3)
            ax.hlines(np.median(vals), x - 0.21, x + 0.21, color=colour, lw=1.3, zorder=4)
        # The gap between the two ranges, drawn: this is what "separates" means.
        if bd.max() < nb.min():
            ax.axhspan(bd.max(), nb.min(), color="#DDDDDD", alpha=0.55, lw=0, zorder=1)
        else:
            ax.axhspan(nb.max(), bd.min(), color="#DDDDDD", alpha=0.55, lw=0, zorder=1)

        lo, hi = min(nb.min(), bd.min()), max(nb.max(), bd.max())
        pad = max((hi - lo) * 0.40, abs(hi) * 0.010, 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(label, fontsize=5.8, pad=19.0)
        ax.text(0.5, 1.055, f"$\\Delta$ {result['delta'] * scale:+{fmt}} "
                f"({result['pct']:+.1f}%)", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.2, color="#333333")
        # Quote the margin against the floor, not a bare "disjoint": the floor is what makes the
        # claim admissible, so the reader should see both numbers.
        ax.text(0.5, 1.006,
                f"gap {result['range_gap'] * scale:{fmt}} > floor {result['floor'] * scale:{fmt}}",
                transform=ax.transAxes, ha="center",
                va="bottom", fontsize=5.0, color="#117733")
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"水约束\n不起作用（n={len(nb)}）", f"水约束\n起作用（n={len(bd)}）"],
                           fontsize=5)
        ax.tick_params(axis="y", labelsize=5, length=1.6, pad=1)
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.grid(axis="y", lw=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # The control. Each pair is drawn individually rather than as a mean, because the null is
    # that the two GROUPS are indistinguishable: the treatment range has to be seen sitting
    # inside the control range, which a pair of means would hide.
    ax = axes[-1]
    metrics = list(control["metric"].unique())
    for x, metric in enumerate(metrics):
        sub = control[control["metric"] == metric]
        for kind, colour, marker, offset in (("control", "#333333", "s", -0.16),
                                             ("treatment", BD_COLOUR, "o", 0.16)):
            values = sub.loc[sub["kind"] == kind, "overlap"].to_numpy(float)
            ax.scatter(np.full(len(values), x + offset), values, s=15, marker=marker,
                       facecolor=colour, edgecolor="white", linewidth=0.3, zorder=4)
            ax.hlines(values.min(), x + offset - 0.10, x + offset + 0.10, color=colour, lw=0.6,
                      zorder=3)
            ax.hlines(values.max(), x + offset - 0.10, x + offset + 0.10, color=colour, lw=0.6,
                      zorder=3)
            ax.vlines(x + offset, values.min(), values.max(), color=colour, lw=0.6, zorder=3)
        control_vals = sub.loc[sub["kind"] == "control", "overlap"].to_numpy(float)
        ax.axhspan(control_vals.min(), control_vals.max(), xmin=(x + 0.06) / len(metrics),
                   xmax=(x + 0.94) / len(metrics), color="#CCCCCC", alpha=0.45, lw=0, zorder=1)

    lo = float(control["overlap"].min())
    ax.set_ylim(lo - 5.0, 101.5)
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=5)
    # A PANEL SHOULD STATE ITS CLAIM, not name its axes. The previous title described the
    # quantity; the point is the verdict, which the reader otherwise had to reconstruct from
    # a grey band and two marker colours.
    # THE TITLE IS DERIVED, BECAUSE A HARDCODED ONE HERE STATED THE OPPOSITE OF THE PANEL.
    # This panel used to read "Rerouting stays inside the control band: no network change is
    # attributable to water". The function that reports this same frame prints, from the same
    # numbers, "ABOVE the floor on both metrics - attributable to water": on both overlap
    # statistics the treatment range sits ENTIRELY BELOW the seed-null range (shared edges
    # 91.97-95.37 vs 97.19-98.64, delta -4.25 pp; intersection 69.95-81.52 vs 87.02-87.21,
    # delta -11.38 pp). The null holds for the network's SIZE -- total length, haul, sinks,
    # offshore share, capex, injected tonnage are all inside the floor -- not for its SHAPE.
    # So the honest claim is the more interesting one: water changes WHICH corridors carry the
    # CO2 without changing HOW MUCH pipeline gets built.
    _sep, _deltas = True, []
    for _metric, _block in control.groupby("metric", sort=False):
        _c = _block.loc[_block["kind"] == "control", "overlap"].to_numpy(float)
        _t = _block.loc[_block["kind"] == "treatment", "overlap"].to_numpy(float)
        _sep &= bool(_t.max() < _c.min())
        _deltas.append(_t.mean() - _c.mean())
    if _sep:
        ax.set_title(f"水约束改写了管廊的选用格局"
                     f"{chr(10)}（超出种子零假设 {_deltas[0]:+.1f} 与 {_deltas[1]:+.1f} pp），"
                     f"{chr(10)}但没有改变管道建设总量",
                     fontsize=6.2, pad=19.0)
    else:
        ax.set_title("重新选线仍落在对照带内：" + chr(10) +
                     "没有任何网络变化可归因于水约束",
                     fontsize=6.2, pad=19.0)
    deltas = []
    for metric in metrics:
        sub = control[control["metric"] == metric]
        treat = sub.loc[sub["kind"] == "treatment", "overlap"].mean()
        ctrl = sub.loc[sub["kind"] == "control", "overlap"].mean()
        deltas.append(treat - ctrl)
    ax.text(0.5, 1.012, "$\\Delta$ " + " / ".join(f"{d:+.1f}" for d in deltas) + " pp（相对地板）",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=5.2, color="#333333")
    # The red verdict line is gone: the panel title now states it, and carrying both put the
    # same conclusion on the canvas twice, in two type sizes and two colours.
    ax.tick_params(axis="y", labelsize=5, length=1.6, pad=1)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([], [], color="#333333", marker="s", ls="none", ms=3.4,
               label="对照：仅切换偏差订正\n（灰带 = 其变幅）"),
        Line2D([], [], color=BD_COLOUR, marker="o", ls="none", ms=3.4,
               label="处理：约束更紧\n相对约束更松"),
    ]
    # Below the axes, not inside them. The 'work in the intersection' group spans 66-82% and
    # a lower-left legend printed through it; the two-line labels also collided with the
    # x tick labels. Outside is cleaner and costs no information.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.155),
              ncol=2, fontsize=5.1, frameon=False, handlelength=1.0, columnspacing=1.6,
              labelspacing=0.30, borderpad=0.1, handletextpad=0.5)


# -- panel (c): the attribution the caption used to make in one subordinate clause --------
def panel_base_attribution(ax, counts, n: int, base_ids: set, lengths) -> dict:
    """For each selection-frequency bin, how much of it also appears with NO water constraint.

    THIS IS WHAT THE FIGURE'S CONCLUSION RESTS ON, AND IT USED TO BE ONE CLAUSE AT LINE 6 OF
    A FOURTEEN-LINE CAPTION: '93 of the 95 are also selected with no water constraint at all'.
    Read with the seed null alongside it, that clause IS the argument -- the shared skeleton
    is what China's CO2 geography gives you, and water neither builds it nor moves it. A
    sentence buried in caption prose cannot carry a conclusion; a panel can.

    Bars are edge counts per bin, split into the part BASE also selects and the part it does
    not. The length-weighted split is returned for the report, because a count treats a 12 km
    spur and a 700 km trunk alike and a reader is entitled to ask whether the water-specific
    edges are simply the short ones.
    """
    ks = list(range(n, 0, -1))
    tot, shared, shared_len_pct = [], [], []
    for k in ks:
        ids = [e for e, c in counts.items() if int(round(c)) == k]
        inb = [e for e in ids if e in base_ids]
        tot.append(len(ids)); shared.append(len(inb))
        L_all = float(lengths.reindex(ids).fillna(0.0).sum())
        L_in = float(lengths.reindex(inb).fillna(0.0).sum())
        shared_len_pct.append(100.0 * L_in / L_all if L_all > 0 else float("nan"))
    x = np.arange(len(ks))
    ax.bar(x, tot, 0.68, color="#E2E2E2", zorder=2,
           label="仅在有水约束时被选中")
    ax.bar(x, shared, 0.68, color=CORE_COLOUR, zorder=3,
           label="无水约束时同样被选中")
    top = max(tot) if tot else 1
    for xi, (t, s) in enumerate(zip(tot, shared)):
        if t == 0:
            continue
        ax.text(xi, t + top * 0.035, f"{100 * s / t:.0f}%", ha="center", va="bottom",
                fontsize=5.2, color="#1A1A1A" if xi == 0 else "#666666",
                fontweight="bold" if xi == 0 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}/{n}" for k in ks], fontsize=5.4)
    ax.set_xlabel("管段恰好被这么多水情景选中",
                  fontsize=6.0, labelpad=1.8)
    ax.set_ylabel("管道走廊", fontsize=6.0, labelpad=2)
    ax.tick_params(labelsize=5.4, length=1.8, pad=1.5)
    ax.set_ylim(0, top * 1.30)
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    core_share = 100.0 * shared[0] / tot[0] if tot[0] else float("nan")
    ax.set_title(f"一致核心中有 {core_share:.0f}% 不需要" + chr(10) +
                 f"水约束就会被选中", fontsize=6.6, pad=3.5)
    # Upper RIGHT: the 7/7 bar is the tallest and sits hard against the left edge, so an
    # upper-left legend prints straight through this panel's own headline number.
    ax.legend(fontsize=5.1, frameon=False, loc="upper right", handlelength=1.0,
              labelspacing=0.22, borderpad=0.1, handletextpad=0.45)
    return {"bins": ks, "total": tot, "in_base": shared,
            "len_pct_in_base": shared_len_pct, "core_share_pct": core_share}


SEED_NULL_RUNS = ["WA_cwatm_126_dry_wd085", "WA_cwatm_126_dry_wd085_seed2",
                  "WA_cwatm_126_dry_wd085_seed3", "WA_cwatm_126_dry_wd085_seed4"]


def seed_null(counts_fn, work_fn):
    """Run this figure's own core statistic on replicates of ONE scenario.

    THE STATISTIC HAD NO NULL, AND THE NULL BEATS THE EFFECT. `WA_cwatm_126_dry_wd085_seed{2,3,4}`
    are true replicates: identical model fingerprint 0xbe7b31c2, identical variable and
    constraint counts, thread count pinned, only the Gurobi seed differing. Every run in this
    study terminates at node_count = 1, so each solution is a root heuristic and the spread is
    pure search-path degeneracy. Running selection_count and work_share_by_threshold on those
    four replicates gives a core that is LARGER and carries MORE of the 2060 work than the
    seven water scenarios do:

        4 seed replicates of ONE scenario   core/union 106/167   core carries 86.0% of work
        7 water scenarios                   core/union  95/185   core carries 78.4% of work

    A statistic that cannot separate a seed change from a hydrological ensemble is not
    measuring agreement across water futures. This does NOT mean nothing happens: per-plant
    2060 capture reallocates by 13.5-29.1% in L1 distance across the water runs, so the
    problems genuinely differ. It means the CORE is the wrong summary of that difference, and
    the figure now reports the null next to the effect rather than omitting it.
    """
    # Two gates, not one: the 26-column plant_detail vintage AND model identity. A null built
    # from replicates of two different models measures the version difference.
    available = same_model_runs([n for n in SEED_NULL_RUNS if require_current_vintage(n)])
    if len(available) < 3:
        return None
    k = len(available)
    try:
        counts = counts_fn(available)
        core = sorted(counts.index[counts.round() >= k])
        curves = pd.DataFrame({s: work_fn(s, counts, k) for s in available}).T
    except (FileNotFoundError, ValueError, KeyError):
        return None
    ks = list(range(k, 0, -1))
    have = all(c in curves.columns for c in ks)
    median = curves[ks].median() if have else None
    # 包络而不是只有中位数：面板 b 要比的是"种子扰动的离散度"与"水扰动的离散度"，
    # 只画中位数就把离散度这个被比较的量本身丢掉了。
    lo = curves[ks].min() if have else None
    hi = curves[ks].max() if have else None
    return {"k": k, "core": len(core), "union": len(counts),
            "work": float(curves[k].mean()) if k in curves.columns else float("nan"),
            "median_curve": median, "lo_curve": lo, "hi_curve": hi}


def main() -> None:
    print("Fig 4 - the no-regret pipeline core")
    print("\nScenario admission (26-column vintage only, gated on the CSV header):")
    scenarios = admit(ENSEMBLE)
    n = len(scenarios)
    if n < 2:
        raise SystemExit("need at least two usable scenarios")
    print(f"\n  ensemble n = {n}: {', '.join(scenarios)}")
    if not require_current_vintage(REFERENCE):
        raise SystemExit(f"{REFERENCE} is not usable as the no-water reference")
    print(f"  reference, excluded from the frequency count: {REFERENCE}")
    # REFUSE TO DIFFERENCE ACROSS WATER-INPUT VINTAGES. See
    # plot_style.assert_same_vintage: the dry-season correction changed every
    # value and no column name, so nothing else in this repository could see it.
    assert_same_vintage(scenarios + [REFERENCE], "plot_fig4_network_reconfiguration")

    cand = candidate_edges()
    geoms = edge_geometry(cand)
    lengths = cand.set_index(cand["edge_id"].astype(str))["length_km"]
    counts = selection_count(scenarios)
    capacity = union_capacity(scenarios)
    injection = mean_injection(scenarios)
    core = sorted(counts.index[counts.round() >= n])
    union_len = float(lengths.reindex(counts.index).sum())
    core_len = float(lengths.reindex(core).sum())
    stats = {s: network_stats(s) for s in scenarios}
    curves = pd.DataFrame({s: work_share_by_threshold(s, counts, n) for s in scenarios}).T
    seed_stats = seed_null(selection_count, work_share_by_threshold)
    core_work = float(curves[n].mean()) if n in curves.columns else float("nan")
    print()
    print("  THE CORE STATISTIC AGAINST ITS OWN NULL")
    print(f"    {n} water scenarios          core/union {len(core)}/{len(counts)}  "
          f"core carries {core_work:.1f}% of {MAP_YEAR} work")
    if seed_stats is None:
        print("    seed replicates            NOT AVAILABLE -- the core is UNCONTROLLED")
    else:
        print(f"    {seed_stats['k']} seed replicates of ONE   core/union {seed_stats['core']}/{seed_stats['union']}  "
              f"core carries {seed_stats['work']:.1f}% of {MAP_YEAR} work")
        if seed_stats['work'] >= core_work:
            print("    -> THE NULL BEATS THE EFFECT. Re-solving a bit-identical model with a")
            print("       different seed produces a MORE unanimous core carrying MORE of the")
            print("       work than seven distinct water futures do, so this statistic cannot")
            print("       discriminate and must not be presented as a water result. What does")
            print("       survive is the per-plant reallocation in panel (c); the core belongs")
            print("       in Extended Data as a solver-degeneracy diagnostic.")

    # The control that decides whether any map difference means anything.
    usable_floor_pairs = [(a, b) for a, b in FLOOR_PAIRS if a in scenarios and b in scenarios]
    usable_treatment_pairs = [(a, b) for a, b in TREATMENT_PAIRS
                              if a in scenarios and b in scenarios]
    if not usable_floor_pairs or not usable_treatment_pairs:
        raise SystemExit("the degeneracy-floor control needs both a bias-variant pair and a "
                         "binding/non-binding pair; one is missing")
    METRICS = {
        "共用管段上的\n输送量": shared_transport_work,
        "交集输送量 /\n并集输送量": intersection_over_union_work,
    }
    control_rows = []
    for metric, function in METRICS.items():
        for kind, pairs in (("control", usable_floor_pairs),
                            ("treatment", usable_treatment_pairs)):
            for a, b in pairs:
                control_rows.append(
                    {
                        "metric": metric, "kind": kind, "a": a, "b": b,
                        "overlap": float(function(a, b)),
                        "objective_a": objective_1e12(a), "objective_b": objective_1e12(b),
                    }
                )
    control = pd.DataFrame(control_rows)
    control["objective_gap_pct"] = (
        (control["objective_b"] - control["objective_a"]).abs() / control["objective_a"] * 100.0
    )

    primary = list(METRICS)[0]
    treatment = control[(control["metric"] == primary)
                        & (control["kind"] == "treatment")]["overlap"].to_numpy(float)

    # The tightest control pair, quoted on the figure: same hydrology model, same SSP,
    # objectives closest together, so anything they disagree about is arbitrary by construction.
    tight = control[(control["metric"] == primary) & (control["kind"] == "control")]
    tight = tight.loc[tight["objective_gap_pct"].idxmin()]

    tests = {key: separation(stats, key) for key in STAT_SPECS}
    # A statistic is only allowed onto the figure if it beats the control band, not merely
    # if its ranges happen not to overlap. See separation() for why disjointness alone is a
    # false positive at this n.
    separating = [key for key, result in tests.items() if result["separates"]]
    nulls = [key for key in STAT_SPECS if key not in separating]
    print(f"\n  statistics that separate BEYOND the bias-correction control: "
          f"{', '.join(separating) or 'none'}")
    for key, result in tests.items():
        if result["disjoint"] and not result["separates"]:
            print(f"  !! {key}: ranges disjoint by {result['range_gap']:.2f} but the floor is "
                  f"{result['floor']:.2f} -- NOT attributable to water")
    print(f"  statistics that do NOT separate: {', '.join(nulls) or 'none'}")

    # ---- the two text blocks, written before the layout --------------------
    # Their wrapped line counts drive the right-hand column's geometry, so they are built first
    # and measured rather than assumed: the numbers in them change with the runs, and a fixed
    # y-offset would either collide with panel (c) or leave a band of blank paper.
    core_vals = curves[n].to_numpy(float)
    base_core = len(set(core) & set(active_edges(REFERENCE)["edge_id"].astype(str)))
    claim = cjk_fill(
        f"一副共用的管网骨架，以及它不能证明什么。任一情景选中的 {len(counts)} 条管段"
        f"（候选走廊共 {len(cand)} 条）中，有 {len(core)} 条在全部 n = {n} 个已求解水情景中都被选中："
        f"{core_len:,.0f} km，占并集长度的 {100 * core_len / union_len:.0f}%，"
        f"承担 {MAP_YEAR} 年输送量的 {np.median(core_vals):.0f}%"
        f"（中位数；区间 {core_vals.min():.0f}–{core_vals.max():.0f}%）。"
        f"这 {len(core)} 条里有 {base_core} 条在完全没有水约束时（BASE）同样被选中。"
        f"与面板 b 的种子零假设一起读，要点就在这里：这副骨架来自中国 CO$_2$ 源汇的地理格局，"
        f"而不是水约束造成的。它仍然是可施工的产出——现在就值得投入的走廊——"
        f"但它不是一个水资源结论。", width=55)
    null = cjk_fill(
        f"水约束没有做到什么。仅在偏差订正开关上不同的两次求解，目标函数相差 "
        f"{tight['objective_gap_pct']:.3f}%（{tight['objective_a']:.3f} 对 "
        f"{tight['objective_b']:.3f}e12 元），输送量的重合度却只有 {tight['overlap']:.1f}%，"
        f"因为大量走廊组合的成本几乎完全相同。"
        f"约束更紧与更松之间给出 {treatment.min():.1f}–{treatment.max():.1f}%，"
        f"在两项重合度指标上都落在该对照区间之内，因此本图不主张水约束重画了管网。"
        f"注意：该对照是对偏差订正的敏感性，不是零假设——在起约束作用的流域中，"
        f"这个开关会把可用水量最多放大 2.43 倍（wgap 组；cwatm 组为 1.26 倍）。"
        f"没能越过它并不等于效应不存在——管网这一结论是“未分辨”，不是“被否定”。", width=55)
    # The nulls belong beside their own evidence in the right column, not buried in a footnote:
    # naming them is the point, because presenting them as differences would be the false
    # positive this figure exists to avoid.
    null_lines = ["以下指标无法区分“起约束”与“不起约束”，", "因此两者都不标注："]
    for key in nulls:
        result = tests[key]
        label = STAT_SPECS[key][0].replace("\n", " ")
        null_lines.append(cjk_fill(
            f"- {label}: {result['nb_median']:.1f} -> {result['bd_median']:.1f} "
            f"({result['pct']:+.1f}%), but binding [{result['bd'].min():.1f}, "
            f"{result['bd'].max():.1f}] sits inside non-binding [{result['nb'].min():.1f}, "
            f"{result['nb'].max():.1f}].", width=57, subsequent_indent="  "))
    nulls_text = "\n".join(null_lines)

    # ---- layout ------------------------------------------------------------
    # 左列一张地图占满高，右列两个检验面板上下叠。
    #
    # 上一版是"两行四格"：地图在左上，三个检验面板分列右上、左下、右下。四个面板里三个的
    # 标题是否定句，正面结论（无悔核心）被压在四分之一的版面上，读者看完记住的是"水什么也
    # 没干"。图自述的主张是"这批走廊现在就该建"，那句话必须占主导版面。
    #
    # 用显式 add_axes 而不是 gridspec：地图有固定数据纵横比，gridspec 会在单元格里缩它，
    # 又把空白挤回来。
    fig_w, fig_h = DOUBLE_COL[0], 5.55
    fig = plt.figure(figsize=(fig_w, fig_h))

    # 图注压缩为一行后，把节省的纵向空间还给右侧两个定量面板。
    TOP, BOT, LEFT, RIGHT = 0.930, 0.115, 0.045, 0.982
    col_gap = 0.075
    map_w = 0.520
    # 地图轴的高度按数据的真实宽高比给，而不是"左列剩多少占多少"。
    # mainland_extent 在 EPSG:2380 下是 5.164e6 x 4.387e6 m，宽高比 1.177（实测）；
    # aspect="equal" 只能充满其中一维，轴框比它更高时上下就各留一条空白带。
    MAP_WH = 1.177
    map_h = (map_w * fig_w / MAP_WH) / fig_h
    ax_a = fig.add_axes([LEFT, TOP - map_h - 0.030, map_w, map_h])
    cbar_y = TOP - map_h - 0.030 - 0.062
    cax = fig.add_axes([LEFT + 0.030, cbar_y, map_w * 0.46, 0.011])
    legend_anchor = (LEFT + 0.010, cbar_y - 0.052)

    r_left = LEFT + map_w + col_gap
    r_w = RIGHT - r_left
    row_gap = 0.100
    r_h = (TOP - BOT - row_gap) / 2.0
    ax_b = fig.add_axes([r_left, BOT + r_h + row_gap, r_w, r_h - 0.055])
    ax_c = fig.add_axes([r_left, BOT, r_w, r_h - 0.030])

    base_ids = set(active_edges(REFERENCE)["edge_id"].astype(str))
    attribution = {"base_core": base_core, "core": len(core)}

    panel_a(fig, ax_a, cax, legend_anchor, counts, capacity, geoms, injection, n, core,
            core_len, union_len, float(np.median(core_vals)))
    panel_b(ax_b, curves, counts, n, seed_stats)
    test_rows = panel_tests(ax_c, tests, control)

    panel_label(ax_a, "a", x=-0.020, y=1.030)
    panel_label(ax_b, "b", x=-0.150, y=1.115)
    panel_label(ax_c, "c", x=-0.150, y=1.100)

    print("\n  检验汇总（面板 c 画的就是这张表）:")
    for r in test_rows:
        ratio = "—" if r["ratio"] != r["ratio"] else f"{r['ratio']:.2f}x"
        print(f"    {r['label']:<28} {ratio:>7}  {'可分辨' if r['sep'] else '未分辨'}")

    print("\n  statistics that do NOT separate, with their overlapping ranges "
          "(caption/ED material, no longer drawn on the canvas):")
    for line in nulls_text.split("\n")[2:]:
        print(f"    {line}")

    binding_here = [s for s in BINDING if s in scenarios]
    # ONE YEAR, NOT TWO. `unserved_volume_mm3(...).max()` takes the maximum over ALL years --
    # both peaks are at 2040 (70.2 and 59.0 Mm3) -- while `unserved_share` is 2030 only, so the
    # caption read a 2040 volume against a 2030 percentage. 2030's own volumes are 53.3-58.4.
    UNSERVED_YEAR = 2030
    unserved_pct = [unserved_share(s, UNSERVED_YEAR) for s in binding_here]
    unserved_vol = [float(unserved_volume_mm3(s).reindex([UNSERVED_YEAR]).iloc[0])
                    for s in binding_here]
    unserved_peak = [float(unserved_volume_mm3(s).max()) for s in binding_here]
    unserved_peak_year = [int(unserved_volume_mm3(s).idxmax()) for s in binding_here]
    slack_share = [scenario_validity(s)["slack_share"] for s in binding_here]
    traj = pd.DataFrame({s: capture_trajectory(s) for s in scenarios})
    spread = float(traj.loc[MAP_YEAR].max() - traj.loc[MAP_YEAR].min())

    note = (f"注：核心管段指在全部 {n} 个水情景中均被选中的走廊；"
            "b 图阴影表示情景范围，c 图超过 1.0× 表示水情景差异大于求解不确定性。")
    fig.text(0.052, 0.025, note, fontsize=DETAIL_SIZE, color="#4A4A4A",
             va="bottom", ha="left")

    save_fig(fig, "fig4_network_reconfiguration")
    report(scenarios, n, cand, counts, core, core_len, union_len, curves, stats, tests,
           separating, nulls, control, traj, base_core)


def report(scenarios, n, cand, counts, core, core_len, union_len, curves, stats, tests,
           separating, nulls, control, traj, base_core) -> None:
    """Print every number that reaches the figure, so the caption can be checked."""
    print("\n" + "=" * 78)
    print("panel (a) - the union network at 2060, coloured by selection frequency")
    print("=" * 78)
    print(f"  candidate corridors in inputs/pipeline_candidate_edges.csv: {len(cand)}")
    print(f"  edges selected by at least one scenario (the union): {len(counts)}")
    print(f"  edges selected by all n = {n} scenarios (the no-regret core): {len(core)}")
    print(f"  core length {core_len:,.0f} km of the {union_len:,.0f} km union "
          f"({100 * core_len / union_len:.1f}%)")
    print(f"  core edges also selected in {REFERENCE} (no water constraint): "
          f"{base_core}/{len(core)}")
    hist = Counter(counts.round().astype(int))
    print("\n  selection frequency histogram:")
    for k in sorted(hist, reverse=True):
        print(f"    {k}/{n}: {hist[k]:>3} edges")

    print("\n" + "=" * 78)
    print("panel (b) - share of 2060 transport work by selection-frequency threshold (%)")
    print("=" * 78)
    print(curves[list(range(n, 0, -1))].round(1).to_string())
    print("\n  median by threshold, with the edge count at or above it:")
    for k in range(n, 0, -1):
        print(f"    >= {k}/{n}: {int(counts.round().ge(k).sum()):>3} edges  "
              f"median {curves[k].median():5.1f}%  "
              f"[{curves[k].min():.1f}-{curves[k].max():.1f}]")
    core_vals = curves[n].to_numpy(float)
    print(f"\n  NO-REGRET CORE: {len(core)} edges, {np.mean(core_vals):.1f}% of 2060 transport "
          f"work (median {np.median(core_vals):.1f}%, range {core_vals.min():.1f}-"
          f"{core_vals.max():.1f}%)")

    print("\n" + "=" * 78)
    print("panel (c) - separation test on every network statistic")
    print("=" * 78)
    frame = pd.DataFrame(stats).T
    frame["binding"] = [s in BINDING for s in frame.index]
    cols = ["total_length_km", "haul_km", "active_sinks", "offshore_sinks", "offshore_share",
            "pipe_capex_bn", "injected_mtpa", "n_edges"]
    print(frame[cols + ["binding"]].round(1).to_string())
    print("\n  binding vs non-binding, median and range, vs the bias-correction control:")
    for key, result in tests.items():
        label = STAT_SPECS[key][0].replace("\n", " ")
        if result["separates"]:
            verdict = f"SEPARATES (gap {result['range_gap']:.2f} > floor {result['floor']:.2f})"
        elif result["disjoint"]:
            verdict = (f"INSIDE THE FLOOR (gap {result['range_gap']:.2f} <= "
                       f"floor {result['floor']:.2f}) -> not attributable to water")
        else:
            verdict = "does NOT separate (ranges overlap)"
        print(f"    {label:<38} {result['nb_median']:>9.1f} -> {result['bd_median']:>9.1f}  "
              f"({result['pct']:+6.1f}%)  "
              f"nb [{result['nb'].min():.1f}, {result['nb'].max():.1f}]  "
              f"bd [{result['bd'].min():.1f}, {result['bd'].max():.1f}]  {verdict}")
    print(f"\n  plotted as differences: {', '.join(separating)}")
    print(f"  reported as nulls, not plotted as differences: {', '.join(nulls)}")

    print("\n  degeneracy-floor control, pair by pair (objectives in 1e12 CNY):")
    show = control.copy()
    show["metric"] = show["metric"].str.replace("\n", " ")
    print(show[["metric", "kind", "a", "b", "objective_a", "objective_b", "objective_gap_pct",
                "overlap"]].round(3).to_string(index=False))

    print("\n  group ranges, per overlap metric:")
    separated = True
    for metric, block in control.groupby("metric", sort=False):
        ctrl = block.loc[block["kind"] == "control", "overlap"].to_numpy(float)
        treat = block.loc[block["kind"] == "treatment", "overlap"].to_numpy(float)
        # "Beyond the floor" means the treatment range sits entirely BELOW the control range,
        # i.e. water re-routes more than solver arbitrariness does. Overlapping ranges are a null.
        beyond = bool(treat.max() < ctrl.min())
        separated &= beyond
        print(f"    {metric.replace(chr(10), ' '):<24} "
              f"control [{ctrl.min():.2f}, {ctrl.max():.2f}] mean {ctrl.mean():.2f}   "
              f"treatment [{treat.min():.2f}, {treat.max():.2f}] mean {treat.mean():.2f}   "
              f"delta {treat.mean() - ctrl.mean():+.2f} pp   "
              f"{'beyond the floor' if beyond else 'RANGES OVERLAP -> null'}")
    verdict = ("ABOVE the floor on both metrics - attributable to water" if separated else
               "INSIDE the floor - NOT attributable to water, on either metric")
    print(f"    verdict on 'water redraws the network': {verdict}")

    print("\n" + "=" * 78)
    print("context printed, not plotted")
    print("=" * 78)
    print("  captured CO2 (Mt/yr):")
    print(traj.round(1).to_string())
    spread = float(traj.loc[MAP_YEAR].max() - traj.loc[MAP_YEAR].min())
    print(f"    {MAP_YEAR} spread across n = {len(scenarios)}: {spread:.1f} Mt "
          f"({100 * spread / traj.loc[MAP_YEAR].median():.1f}% of median);  "
          f"{REFERENCE} {capture_trajectory(REFERENCE).loc[MAP_YEAR]:.1f} Mt")
    print("\n  unserved water in the binding runs (admitted, not hidden):")
    for scenario in BINDING:
        if scenario not in scenarios:
            continue
        volume = unserved_volume_mm3(scenario)
        parts = [f"{year} {unserved_share(scenario, year):.2f}% "
                 f"({volume[year]:.1f} Mm3)" for year in YEARS]
        print(f"    {scenario:<28} slack {scenario_validity(scenario)['slack_share']:.2%} of "
              f"objective;  " + "  ".join(parts))


if __name__ == "__main__":
    main()
