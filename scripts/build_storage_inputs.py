from __future__ import annotations

import argparse

from _bootstrap import ROOT
from coal_retrofit.builders.storage import write_storage_hubs
from coal_retrofit.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inputs/storage_hubs.csv")
    parser.add_argument(
        "--offshore-basin-dilation-px", type=int, default=6,
        help="Merge offshore bodies per basin and storage type; basins are the connected "
             "components of the aquifer raster dilated by this many 5 km pixels. The default 6 "
             "(30 km) reproduces the source assessment's 7 offshore basins and the checked-in "
             "storage_hubs.csv (89 hubs); 0 keeps every offshore body separate.",
    )
    args = parser.parse_args()
    if args.offshore_basin_dilation_px < 0:
        parser.error("--offshore-basin-dilation-px must be >= 0")
    paths = ProjectPaths(ROOT)
    dilation = args.offshore_basin_dilation_px or None
    storage = write_storage_hubs(paths, offshore_basin_dilation_px=dilation)
    offshore = int(storage["offshore"].sum())
    print(f"Wrote {len(storage)} storage hub rows ({offshore} offshore).")


if __name__ == "__main__":
    main()
