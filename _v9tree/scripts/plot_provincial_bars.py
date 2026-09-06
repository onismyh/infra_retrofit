"""Provincial technology choice bar charts (ED Fig 6).

Two-panel stacked horizontal bar chart:
  Panel A: Near-term Transition (2030 + 2040)
  Panel B: CCS Entry & Retirement (2050 + 2060)

Usage:
    python scripts/plot_provincial_bars.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "SimHei",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PATHWAYS = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]
COLORS = {
    "unabated": "#BBBBBB",
    "biomass": "#228833",
    "ccs": "#4477AA",
    "beccs": "#CCBB44",
    "ammonia": "#EE6677",
    "retire": "#999999",
}
LABELS = {
    "unabated": "Unabated operation",
    "biomass": "Biomass co-firing",
    "ccs": "CCS",
    "beccs": "BECCS",
    "ammonia": "Ammonia co-firing",
    "retire": "Retirement",
}

REGIONS = OrderedDict([
    ("North China", ["Inner Mongolia", "Shanxi", "Shandong", "Hebei", "Henan"]),
    ("Northeast", ["Heilongjiang", "Jilin", "Liaoning"]),
    ("East China", ["Jiangsu", "Zhejiang", "Anhui", "Jiangxi", "Fujian", "Shanghai"]),
    ("Central", ["Hubei", "Hunan"]),
    ("South", ["Guangdong", "Guangxi", "Hainan", "Yunnan", "Guizhou", "Sichuan", "Chongqing"]),
    ("Northwest", ["Shaanxi", "Xinjiang", "Gansu", "Ningxia", "Qinghai", "Tianjin"]),
])


def _load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(int)
    return df


def _pivot_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to (year, province) x pathway shares."""
    piv = df.pivot_table(
        index=["year", "province_name"],
        columns="pathway",
        values="generation_share",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    # Compute GW capacity
    gen_by_prov = df.groupby(["year", "province_name"])["province_generation_mwh"].first().reset_index()
    gen_by_prov["capacity_gw"] = gen_by_prov["province_generation_mwh"] / 5500 / 1e3
    piv = piv.merge(gen_by_prov[["year", "province_name", "capacity_gw"]], on=["year", "province_name"], how="left")
    for pw in PATHWAYS:
        if pw not in piv.columns:
            piv[pw] = 0.0
    return piv


def _sort_provinces_in_region(
    piv: pd.DataFrame, provinces: list[str], sort_year: int, sort_col: str, ascending: bool = False
) -> list[str]:
    """Sort provinces within a region by a specific column/year."""
    sub = piv[piv["year"] == sort_year].set_index("province_name")
    present = [p for p in provinces if p in sub.index]
    missing = [p for p in provinces if p not in sub.index]
    if sort_col in sub.columns:
        present.sort(key=lambda p: sub.loc[p, sort_col] if p in sub.index else 0, reverse=not ascending)
    return present + missing


def _draw_panel(
    ax: plt.Axes,
    piv: pd.DataFrame,
    years: tuple[int, int],
    sort_year: int,
    sort_col: str,
    title: str,
):
    """Draw one panel of stacked horizontal bars."""
    y1, y2 = years
    bar_height = 0.35
    y_positions = []
    y_labels = []
    region_boundaries = []
    gw_annotations = []

    current_y = 0
    for region_name, provinces in REGIONS.items():
        sorted_provs = _sort_provinces_in_region(piv, provinces, sort_year, sort_col)
        region_start = current_y

        for prov in sorted_provs:
            # Two bars per province: year1 on top, year2 on bottom
            y_top = current_y
            y_bot = current_y - bar_height - 0.05

            for yr, y_pos in [(y1, y_top), (y2, y_bot)]:
                row = piv[(piv["year"] == yr) & (piv["province_name"] == prov)]
                if row.empty:
                    continue
                row = row.iloc[0]
                left = 0.0
                for pw in PATHWAYS:
                    val = float(row.get(pw, 0.0))
                    if val > 0.001:
                        ax.barh(y_pos, val, height=bar_height, left=left,
                                color=COLORS[pw], edgecolor="white", linewidth=0.3)
                        left += val

            # Label and GW annotation (use year1 data)
            row1 = piv[(piv["year"] == y1) & (piv["province_name"] == prov)]
            gw = row1.iloc[0]["capacity_gw"] if not row1.empty else 0
            mid_y = current_y - bar_height / 2 - 0.025
            y_labels.append(prov)
            y_positions.append(mid_y)
            gw_annotations.append(gw)

            current_y -= (2 * bar_height + 0.25)

        region_boundaries.append((region_name, region_start, current_y + 0.1))
        current_y -= 0.5  # Gap between regions

    # Set y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Generation share")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Year labels in small text on the right
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f"{gw:.0f} GW" for gw in gw_annotations], fontsize=6.5, color="#666666")
    ax2.tick_params(length=0)

    # Region separators and labels
    for region_name, y_start, y_end in region_boundaries:
        sep_y = y_end - 0.15
        ax.axhline(y=sep_y, color="#cccccc", linewidth=0.5, linestyle="-")
        ax.text(-0.18, (y_start + y_end) / 2, region_name,
                transform=ax.get_yaxis_transform(),
                fontsize=8, fontweight="bold", ha="right", va="center",
                color="#444444")

    # Year indicator
    ax.text(0.98, 0.02, f"Top bar: {y1}  |  Bottom bar: {y2}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="#888888", style="italic")

    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    csv_path = ROOT / "results" / "BASE" / "province_pathways.csv"
    if not csv_path.exists():
        print(f"Data not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = _load_data(csv_path)
    piv = _pivot_shares(df)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(14, 26))
    fig.subplots_adjust(hspace=0.15, left=0.22, right=0.88)

    # Panel A: Near-term (2030 + 2040), sorted by capacity
    _draw_panel(ax_a, piv, years=(2030, 2040), sort_year=2030,
                sort_col="capacity_gw", title="A  Near-term Transition (2030 + 2040)")

    # Panel B: CCS entry (2050 + 2060), sorted by CCS share
    _draw_panel(ax_b, piv, years=(2050, 2060), sort_year=2050,
                sort_col="ccs", title="B  CCS Entry & Final Retirement (2050 + 2060)")

    # Legend at top
    handles = [mpatches.Patch(facecolor=COLORS[pw], edgecolor="white", label=LABELS[pw])
               for pw in PATHWAYS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.55, 0.995))

    # Save
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = out_dir / f"ed_fig6_provincial_bars.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
