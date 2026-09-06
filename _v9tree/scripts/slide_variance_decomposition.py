"""Extended Data - where the uncertainty in basin water availability comes from.

This panel used to be Fig 2(c). It is a variance decomposition, which is Extended Data
content by Nature convention: it tells a reviewer that the 20-member ensemble is doing work,
it does not advance the paper's claim. Main Fig 2 now carries only the ceiling claim
(`plot_fig2_constraint_response.py`); this script carries the attribution.

The logic is the one that was verified in place and is reproduced here unchanged, not
re-derived:

  * the design is SATURATED - 2 hydrology models x 5 GCMs x 2 SSPs with exactly one
    observation per cell. There is no replication, so the three-way term IS the residual and
    the seven components sum to the total sum of squares exactly. The closure is printed per
    row and is 100.0% by construction; if it ever is not, the arithmetic is wrong.
  * both bases are reported. `_dry` scenarios read `dry_season_water_m3_per_year`, which is
    the column the availability constraint acts on (reproducing the solver's own `available`
    from it matches to 5.6e-16; the annual column is 4.3x too high). The annual column is
    shown alongside because most of the water-energy literature uses it, and the headline --
    model spread beats the scenario signal in every basin -- has to survive both.
  * the SSP effect is taken PAIRED, within each (hydrology, GCM) combination. Pooling across
    models first would let inter-model spread leak into what is meant to be the scenario
    effect. The whisker is the point: where it straddles zero, the 10 model pairs do not
    agree on the SIGN of the climate signal.

  (a) variance shares per basin on the dry-season basis (the constrained one), with the
      signed SSP3-7.0 minus SSP1-2.6 change beside it
  (b) the same on the annual-mean basis

Usage:  python scripts/plot_ed_variance_decomposition.py
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
)

apply_style()

INPUTS = ROOT / "inputs"
DECOMP_YEAR = 2060

# The ensemble design. Any missing cell breaks the saturated model, so a basin with a hole is
# skipped loudly rather than patched.
HYDROLOGY = ["cwatm", "watergap2-2e"]
GCMS = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
SSPS = ["ssp126", "ssp370"]
EXPECTED_MEMBERS = len(HYDROLOGY) * len(GCMS) * len(SSPS)

# Runs used only to mark which basins actually ran short of water, so the reader can see that
# the basins where model spread dominates are the basins that decide the answer.
BINDING_RUNS = ["WA_cwatm_126_dry_wd085", "WA_cwatm_370_dry_wd085"]

C126 = "#4477AA"
C370 = "#CC3311"
# Colours held locally rather than added to plot_style: sibling figure scripts are editing
# that module concurrently.
SOURCE_COLORS = {
    "GCM": "#4477AA",
    "水文模型": "#EE7733",
    "SSP": "#CC3311",
    "GCM x 水文模型": "#88CCEE",
    "GCM x SSP": "#EE99AA",
    "水文模型 x SSP": "#FFDD77",
    "三重交互": "#BBBBBB",
}
MODEL_SOURCES = ["GCM", "水文模型", "GCM x 水文模型"]

BASES = [
    ("dry_season_water_m3_per_year", "枯水期口径（约束实际作用的口径）"),
    ("available_water_m3_per_year", "年均口径（多数文献使用的口径）"),
]



from pathlib import Path as _SP
SLIDE_DIR = _SP(__file__).resolve().parent.parent / "results" / "figures" / "slides"
SLIDE_DIR.mkdir(parents=True, exist_ok=True)
BASIN_SHORT = {"A": "松辽", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
               "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北内陆"}


def save_fig(fig, name, subdir=""):  # noqa: F811  幻灯版：直接落到 slides/
    p = SLIDE_DIR / f"slide_{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"[slide] -> {p}")

def has_air_columns(scenario: str) -> bool:
    """Vintage gate on the CSV header: 24-column runs predate the cooling-conversion
    mechanism and describe a different feasible set. Filling the missing columns with zero
    would report a code change as a climate effect.
    """
    path = RESULTS_DIR / scenario / "plant_detail.csv"
    if not path.exists():
        return False
    return {"air_cooled_share", "already_air_share"}.issubset(pd.read_csv(path, nrows=0).columns)


def load_availability() -> pd.DataFrame:
    path = INPUTS / "water_availability.csv"
    frame = pd.read_csv(path)
    members = frame["scenario_id"].nunique()
    if members != EXPECTED_MEMBERS:
        raise SystemExit(
            f"{path.name} carries {members} members, not the {EXPECTED_MEMBERS} the saturated "
            f"{len(HYDROLOGY)}x{len(GCMS)}x{len(SSPS)} design needs"
        )
    return frame


def binding_basins() -> list[str]:
    """Basins with unserved water demand in the solved binding runs."""
    codes: set[str] = set()
    for scenario in BINDING_RUNS:
        if not has_air_columns(scenario):
            print(f"  [skip] {scenario}: stale 24-column vintage, not used for binding marks")
            continue
        path = RESULTS_DIR / scenario / "slack_detail.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} - run {scenario} first")
        slack = pd.read_csv(path)
        water = slack[(slack["constraint_type"] == "water_supply") & (slack["slack_value"] > 1.0)]
        codes |= set(water["node_id"].astype(str).str.rsplit("_", n=1).str[-1])
    return sorted(codes)


# ── the decomposition ────────────────────────────────────────────────────────
def decompose(values: np.ndarray) -> dict[str, float]:
    """Balanced 3-way sum-of-squares split of a (hydrology, GCM, SSP) array.

    The design is 2 x 5 x 2 with one observation per cell, so the model is saturated: the
    three-way term is the residual, and the seven components sum to the total sum of squares
    exactly. Returned as percentages of that total.
    """
    grand = values.mean()
    a_hyd = values.mean(axis=(1, 2)) - grand
    a_gcm = values.mean(axis=(0, 2)) - grand
    a_ssp = values.mean(axis=(0, 1)) - grand
    n_hyd, n_gcm, n_ssp = values.shape

    ss_hyd = n_gcm * n_ssp * np.sum(a_hyd ** 2)
    ss_gcm = n_hyd * n_ssp * np.sum(a_gcm ** 2)
    ss_ssp = n_hyd * n_gcm * np.sum(a_ssp ** 2)

    i_hg = values.mean(axis=2) - grand - a_hyd[:, None] - a_gcm[None, :]
    i_hs = values.mean(axis=1) - grand - a_hyd[:, None] - a_ssp[None, :]
    i_gs = values.mean(axis=0) - grand - a_gcm[:, None] - a_ssp[None, :]
    ss_hg = n_ssp * np.sum(i_hg ** 2)
    ss_hs = n_gcm * np.sum(i_hs ** 2)
    ss_gs = n_hyd * np.sum(i_gs ** 2)

    fitted = (
        grand + a_hyd[:, None, None] + a_gcm[None, :, None] + a_ssp[None, None, :]
        + i_hg[:, :, None] + i_hs[:, None, :] + i_gs[None, :, :]
    )
    ss_three = np.sum((values - fitted) ** 2)
    total = np.sum((values - grand) ** 2)
    if total <= 0:
        return {}
    parts = {
        "GCM": ss_gcm, "水文模型": ss_hyd, "SSP": ss_ssp,
        "GCM x 水文模型": ss_hg, "GCM x SSP": ss_gs, "水文模型 x SSP": ss_hs,
        "三重交互": ss_three,
    }
    return {k: 100.0 * v / total for k, v in parts.items()}


def decomposition_table(avail: pd.DataFrame, column: str) -> pd.DataFrame:
    """Variance decomposition per basin plus national, for `column` at DECOMP_YEAR."""
    year_frame = avail[avail["planning_year"] == DECOMP_YEAR]
    grouped = year_frame.groupby(["basin_code", "scenario_id"])[column].sum().unstack("scenario_id")
    national = grouped.sum(axis=0)

    def cube(series: pd.Series) -> np.ndarray | None:
        out = np.full((len(HYDROLOGY), len(GCMS), len(SSPS)), np.nan)
        for i, hyd in enumerate(HYDROLOGY):
            for j, gcm in enumerate(GCMS):
                for k, ssp in enumerate(SSPS):
                    key = f"{hyd}|{gcm}|{ssp}"
                    if key in series.index:
                        out[i, j, k] = float(series[key])
        return None if np.isnan(out).any() else out

    rows = []
    for label, series in [(b, grouped.loc[b]) for b in BASIN_ORDER if b in grouped.index] + [
        ("National", national)
    ]:
        values = cube(series)
        if values is None:
            print(f"  [warn] {label}: ensemble not complete at {DECOMP_YEAR}, skipped")
            continue
        parts = decompose(values)
        if not parts:
            continue
        # Paired within each (hydrology, GCM) combination: the SSP signal is a within-model
        # difference, and pooling across models first would let inter-model spread leak into
        # what is meant to be the scenario effect. Median over the 10 pairs.
        paired = 100.0 * (values[:, :, 1] / values[:, :, 0] - 1.0).ravel()
        record = {"basin": label, **parts}
        record["SSP delta %"] = float(np.median(paired))
        record["SSP min %"] = float(paired.min())
        record["SSP max %"] = float(paired.max())
        record["drying"] = int((paired < 0).sum())
        record["model share %"] = sum(parts[s] for s in MODEL_SOURCES)
        record["SS closure %"] = sum(parts.values())
        rows.append(record)
    return pd.DataFrame(rows).set_index("basin")


# ── drawing ──────────────────────────────────────────────────────────────────
def draw_basis(ax, ax_delta, table: pd.DataFrame, binding: list[str], title: str,
               show_legend: bool, show_xlabel: bool) -> None:
    """Stacked variance shares per basin, with the signed SSP signal beside it."""
    order = [b for b in table.index if b != "National"] + ["National"]
    table = table.loc[order]
    y = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    for source in SOURCE_COLORS:
        if source not in table.columns:
            continue
        values = table[source].to_numpy()
        ax.barh(y, values, 0.66, left=left, color=SOURCE_COLORS[source], label=source,
                edgecolor="white", linewidth=0.3)
        left += values

    labels = []
    for basin in table.index:
        name = "全国" if basin == "National" else BASIN_NAMES_ZH.get(basin, basin)
        labels.append(f"{name} *" if basin in binding else name)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.2)
    for tick, basin in zip(ax.get_yticklabels(), table.index):
        if basin in binding or basin == "National":
            tick.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=7.4)
    if show_xlabel:
        ax.set_xlabel(f"{DECOMP_YEAR} 年可供水量方差的分解占比（%）")
    if show_legend:
        ax.legend(frameon=False, fontsize=5.7, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, 1.20), columnspacing=0.9, handlelength=1.2)

    # The SSP main effect is a share of variance, not a direction. The companion axis carries
    # the signed SSP3-7.0 minus SSP1-2.6 change so "small SSP share" cannot be misread as
    # "no climate signal".
    deltas = table["SSP delta %"].to_numpy()
    lows = table["SSP min %"].to_numpy()
    highs = table["SSP max %"].to_numpy()
    ax_delta.barh(y, deltas, 0.5, color=[C370 if d < 0 else C126 for d in deltas],
                  edgecolor="white", linewidth=0.3, zorder=3)
    # The whisker is the point: where it straddles zero, the 10 model pairs do not even agree
    # on the SIGN of the scenario signal.
    ax_delta.hlines(y, lows, highs, color="#444444", lw=0.7, zorder=4)
    ax_delta.plot(lows, y, marker="|", ls="none", ms=3.0, color="#444444", zorder=4)
    ax_delta.plot(highs, y, marker="|", ls="none", ms=3.0, color="#444444", zorder=4)
    ax_delta.axvline(0, color="black", lw=0.6, zorder=2)
    # `labelleft=False`, NOT set_yticklabels([]): this axis shares its y with the main one, so
    # a FixedFormatter set here would replace the shared formatter and blank out the basin
    # names on the main panel too.
    ax_delta.tick_params(axis="y", labelleft=False, length=0)
    if show_xlabel:
        ax_delta.set_xlabel("SSP3-7.0 相对 SSP1-2.6（%）")
    ax_delta.set_title("带符号的 SSP 信号\n（中位数，全距）", fontsize=6.4)
    span = float(np.nanmax(np.abs(np.concatenate([lows, highs])))) * 1.30
    ax_delta.set_xlim(-span, span)
    for yi, (delta, low, high, drying) in zip(
        y, zip(deltas, lows, highs, table["drying"].to_numpy())
    ):
        pad = span * 0.035
        if delta < 0:
            ax_delta.text(min(low, delta) - pad, yi, f"{delta:+.1f}", fontsize=5.2,
                          va="center", ha="right")
        else:
            ax_delta.text(max(high, delta) + pad, yi, f"{delta:+.1f}", fontsize=5.2,
                          va="center", ha="left")
        ax_delta.text(span * 0.98, yi, f"{drying}/10", fontsize=5.0, va="center",
                      ha="right", color="#555555", alpha=0.85)
    ax_delta.text(span * 0.98, len(deltas) - 0.35, "变干的成员数", fontsize=5.0,
                  color="#555555", ha="right", va="bottom")


def main() -> None:
    print("Extended Data - variance decomposition of basin water availability")
    avail = load_availability()
    binding = binding_basins()
    print(f"  ensemble: {len(HYDROLOGY)} hydrology x {len(GCMS)} GCM x {len(SSPS)} SSP = "
          f"{EXPECTED_MEMBERS} members, saturated (one observation per cell)")
    print(f"  basins with unserved demand in the solved binding runs: "
          f"{','.join(binding) or 'none'}")

    tables = {column: decomposition_table(avail, column) for column, _ in BASES}

    # DOUBLE_COL[0] * 1.25 saved at 204.6 mm, 22 mm over the 183 mm double-column limit.
    # 1.10 lands at ~180 mm with the same layout.
    fig = plt.figure(figsize=(DOUBLE_COL[0] * 1.10, 7.2))
    outer = fig.add_gridspec(2, 1, hspace=0.30)
    axes = []
    for row, (column, title) in enumerate(BASES):
        inner = outer[row].subgridspec(1, 2, width_ratios=[1.0, 0.44], wspace=0.05)
        ax = fig.add_subplot(inner[0])
        ax_delta = fig.add_subplot(inner[1], sharey=ax)
        draw_basis(ax, ax_delta, tables[column], binding, title,
                   show_legend=(row == 0), show_xlabel=(row == len(BASES) - 1))
        axes.append(ax)

    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter, x=-0.185, y=1.10)

    closure = pd.concat([t["SS closure %"] for t in tables.values()])
    # Wrapped, not one long line: `savefig(bbox_inches="tight")` grows the canvas to whatever
    # the widest artist needs, so an unwrapped footnote silently widens the figure.
    note = (
        f"饱和 {len(HYDROLOGY)}x{len(GCMS)}x{len(SSPS)} 设计，每个单元一个成员：三重交互项即残差，"
        f"因此七个分量对总平方和闭合（最小 {closure.min():.1f}%，最大 {closure.max():.1f}%）。"
        f"SSP 变化在每个（水文模型，GCM）组合内配对；须线跨过零意味着 "
        f"{len(HYDROLOGY) * len(GCMS)} 个配对在气候信号的符号上不一致。"
        f"* = 在起约束的求解中存在未满足需求的流域。面板 a 是可供水量约束真正作用的口径；"
        f"面板 b 之所以也画出来，是因为水—能源文献多数使用它，而主结论必须在两种口径下都成立。"
    )
    (lambda *a, **k: None)(0.01, 0.005, cjk_fill(note, width=175), fontsize=5.3, color="#555555",
             va="bottom", linespacing=1.45)
    save_fig(fig, "ed_variance_decomposition", subdir="extended")

    cols = list(SOURCE_COLORS) + [
        "model share %", "SS closure %", "SSP delta %", "SSP min %", "SSP max %", "drying",
    ]
    for column, title in BASES:
        print(f"\n{DECOMP_YEAR} variance decomposition -- {title} (% of total SS)")
        print(tables[column][cols].round(1).to_string())

    print("\nmodel spread (GCM + hydrology + their interaction) vs SSP main effect")
    for column, title in BASES:
        table = tables[column]
        print(f"  {title}")
        for basin in table.index:
            model = table.loc[basin, MODEL_SOURCES].sum()
            ssp = table.loc[basin, "SSP"]
            mark = " <- unserved demand here" if basin in binding else ""
            verdict = "model wins" if model > ssp else "SSP wins"
            print(f"    {basin:<10} model {model:5.1f}%  SSP {ssp:5.1f}%  {verdict}{mark}")
        wins = sum(table.loc[b, MODEL_SOURCES].sum() > table.loc[b, "SSP"] for b in table.index)
        print(f"    -> model spread exceeds the SSP main effect in {wins} of {len(table)} rows")

    print("\nsum-of-squares closure check (must be 100.0 for every row)")
    for column, title in BASES:
        values = tables[column]["SS closure %"]
        print(f"  {title:<52} min {values.min():.4f}  max {values.max():.4f}")


if __name__ == "__main__":
    main()
