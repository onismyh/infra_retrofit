from __future__ import annotations

import argparse

from _bootstrap import ROOT
from coal_retrofit.builders.storage import write_storage_hubs
from coal_retrofit.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inputs/storage_hubs.csv")
    parser.add_argument(
        "--offshore-basin-dilation-px", type=int, default=None,
        help="Merge offshore bodies per basin and storage type; basins are the connected "
             "components of the aquifer raster dilated by this many 5 km pixels. 6 (30 km) "
             "reproduces the source assessment's 7 offshore basins. Default: no merge.",
    )
    args = parser.parse_args()
    paths = ProjectPaths(ROOT)
    storage = write_storage_hubs(paths, offshore_basin_dilation_px=args.offshore_basin_dilation_px)
    offshore = int(storage["offshore"].sum())
    print(f"Wrote {len(storage)} storage hub rows ({offshore} offshore).")


if __name__ == "__main__":
    main()
