"""Write `inputs/water_basin_caps.csv` -- the institutional half of the water constraint.

The runoff-derived node budget (`water_availability.csv`) is the PHYSICAL half: how much water
the environmental-flow rule leaves in the channel. This file is the INSTITUTIONAL half: how much
the 用水总量控制指标 lets anyone withdraw in each basin, net of the users the model does not
decide. Splitting them is the point of the switch -- `WATER_EXTRACTABLE_FRACTION` and
`existing_withdrawal_share` used to be aliased into one product, and no experiment could say
which of the two was binding.
"""
from __future__ import annotations

import logging
import sys

from _bootstrap import ROOT

from coal_retrofit.builders.water_quota import write_basin_caps
from coal_retrofit.paths import ProjectPaths


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = ProjectPaths(ROOT)
    frame = write_basin_caps(paths)
    headline = frame[frame["planning_year"] == frame["planning_year"].min()]
    print(headline[["basin_code", "basin_name", "cap_1e8_m3", "actual_2025_1e8_m3",
                    "reserved_1e8_m3", "modelled_industry_1e8_m3", "compliance_ratio",
                    "residual_uncapped_1e8_m3", "residual_1e8_m3"]].to_string(index=False))
    over = headline.loc[headline["compliance_ratio"] < 1.0, "basin_code"].tolist()
    print(f"\n现状用水已超其分摊指标的流域: {over or '无'}"
          "  （预留按 compliance_ratio 等比压缩；原始负余量见 residual_uncapped_1e8_m3）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
