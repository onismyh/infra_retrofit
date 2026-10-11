"""写生物质与氨两张供给表（`_indtree/inputs/biomass_supply_curve.csv`、`ammonia_supply_curve.csv`）。

生物质读 Wang 2023 情景 4 栅格 `bioenergy_s4_AF_max.tif`（`data/` 下任意位置），氨表读 `data/H2/` 的风光栅格；
两张表一起写，缺任一栅格就报错、两张都不写。
"""

from __future__ import annotations

from _bootstrap import ROOT
from coal_retrofit.builders.supply import write_supply_inputs
from coal_retrofit.paths import ProjectPaths


def main() -> None:
    paths = ProjectPaths(ROOT)
    biomass_supply, ammonia_supply = write_supply_inputs(paths)
    print(f"Wrote {len(biomass_supply)} biomass rows.")
    print(f"Wrote {len(ammonia_supply)} ammonia rows.")


if __name__ == "__main__":
    main()
