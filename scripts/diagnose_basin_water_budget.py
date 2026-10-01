"""逐流域核算节点余量：径流 x 0.20 − 生活耗水 − 灌溉耗水。诊断脚本，只读 inputs/，不写。

径流取 `water_availability.csv`（偏差订正后、按流域预算分到节点，节点合计即流域合计），耗水取
`water_basin_use.csv`（`scripts/build_water_use.py`）。余量为负的流域，求解时节点可用量取归到该节点的
煤电不改造同年耗水（存量不增，`optimization/water_access._water_available_by_node`）；这里只列余量本身。

用法：python scripts/diagnose_basin_water_budget.py [--member cwatm|gfdl-esm4|ssp126] [--year 2050] [--season annual|dry]
"""
from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT

from coal_retrofit.constants import WATER_EXTRACTABLE_FRACTION

NAMES = {"A": "东北诸河", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
         "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北诸河"}


def basin_budget(member: str, year: int, season: str) -> pd.DataFrame:
    """逐流域的径流、可提取量、生活与灌溉耗水与余量，10^8 m3/yr。"""
    dry = season == "dry"
    availability = pd.read_csv(ROOT / "inputs" / "water_availability.csv")
    use = pd.read_csv(ROOT / "inputs" / "water_basin_use.csv")
    runoff = availability[(availability["scenario_id"] == member) & (availability["planning_year"] == year)]
    use = use[(use["scenario_id"] == member) & (use["planning_year"] == year)].set_index("basin_code")
    if runoff.empty or use.empty:
        raise SystemExit(f"{member} {year} 不在 water_availability.csv 或 water_basin_use.csv 里")
    prefix = "dry_season_" if dry else ""
    frame = pd.DataFrame({
        "径流": runoff.groupby("basin_code")["dry_season_water_m3_per_year" if dry else "available_water_m3_per_year"].sum(),
        "生活": use[f"{prefix}domestic_m3_per_year"],
        "灌溉": use[f"{prefix}irrigation_m3_per_year"],
    }).dropna() / 1e8
    frame["x0.20"] = frame["径流"] * WATER_EXTRACTABLE_FRACTION
    frame["余量"] = frame["x0.20"] - frame["生活"] - frame["灌溉"]
    frame["占可提取"] = (frame["生活"] + frame["灌溉"]) / frame["x0.20"]
    frame.index = [f"{code} {NAMES.get(code, '')}" for code in frame.index]
    return frame[["径流", "x0.20", "生活", "灌溉", "余量", "占可提取"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--member", default="cwatm|gfdl-esm4|ssp126")
    parser.add_argument("--year", type=int, default=2050)
    parser.add_argument("--season", choices=("annual", "dry"), default="annual")
    args = parser.parse_args()
    frame = basin_budget(args.member, args.year, args.season)
    print(f"{args.member}  {args.year}  {args.season}  单位 10^8 m3/yr（枯水期为年化值）\n")
    shown = frame.round(0)
    shown["占可提取"] = (frame["占可提取"] * 100).round(0).astype(int).astype(str) + "%"
    print(shown.to_string())
    print(f"\n各流域余量的代数和 {frame['余量'].sum():.0f}（模型不跨流域抵扣，余量为负的节点取煤电存量）；"
          f"为负的流域：{[i for i, v in frame['余量'].items() if v < 0] or '无'}")


if __name__ == "__main__":
    main()
