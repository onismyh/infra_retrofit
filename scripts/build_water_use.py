"""写 `_indtree/inputs/water_basin_use.csv`：各气候成员、各规划年、各一级流域的生活与灌溉耗水（全年与枯水期）。

求解时节点可用量从 径流 x 0.20 里扣掉这两项（`optimization/water_access._water_available_by_node`）。
成员取 `inputs/water_scenarios.csv`；要读 `data/water/` 下各成员的 qtot 与耗水文件（`scripts/download_isimip_water_use.py`）。
枯水期窗口按各成员的 qtot 重选，与 `water_availability.csv` 的枯水期列同一段代码，所以那张表不用重建。
口径见 `builders/water_use.py` 的模块说明。打印一个成员在 2050 年的逐流域合计（10^8 m3/yr）供核对。
"""
from __future__ import annotations

import logging
import sys

from _bootstrap import ROOT

from coal_retrofit.builders.water_use import write_water_use
from coal_retrofit.paths import ProjectPaths

HEAD_MEMBER = "cwatm|gfdl-esm4|ssp126"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = write_water_use(ProjectPaths(ROOT))
    print(f"写出 {len(frame)} 行：{frame['scenario_id'].nunique()} 个成员 x {frame['planning_year'].nunique()} 个规划年"
          f" x {frame['basin_code'].nunique()} 个流域")
    head = frame[(frame["scenario_id"] == HEAD_MEMBER) & (frame["planning_year"] == 2050)].set_index("basin_code")
    columns = ["domestic_m3_per_year", "irrigation_m3_per_year",
               "dry_season_domestic_m3_per_year", "dry_season_irrigation_m3_per_year"]
    print(f"\n{HEAD_MEMBER} 2050（10^8 m3/yr；枯水期起始月见末列）")
    print((head[columns] / 1e8).round(1).join(head["dry_season_start_month"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
