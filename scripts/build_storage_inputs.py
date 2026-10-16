from __future__ import annotations

import argparse

from _bootstrap import ROOT
from coal_retrofit.builders.storage import write_storage_hubs
from coal_retrofit.builders.storage_clusters import CLUSTER_LINKAGES
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.sink_inputs import SINK_VARIANTS, cluster_variant


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build inputs/storage_hubs.csv, or with --cluster-km the sink variant table "
                    "inputs/storage_hubs_<km>km_<linkage>.csv (coal_retrofit.sink_inputs; then build its network "
                    "with build_network_inputs.py --sinks <km>km_<linkage>)"
    )
    method = parser.add_mutually_exclusive_group()
    method.add_argument(
        "--offshore-basin-dilation-px", type=int, default=6,
        help="Merge offshore bodies per basin and storage type; basins are the connected "
             "components of the aquifer raster dilated by this many 5 km pixels. The default 6 "
             "(30 km) reproduces the source assessment's 7 offshore basins and the checked-in "
             "storage_hubs.csv (89 hubs); 0 keeps every offshore body separate.",
    )
    method.add_argument(
        "--cluster-km", type=float, default=None,
        help="Cluster the raster cells by distance instead of by connected component: same storage type, onshore "
             "and offshore apart, cut at this many km; each cluster sums its cells' storage and injectivity. "
             "Replaces the connected components and the offshore basin merge and writes the sink variant table "
             "storage_hubs_<km>km_<linkage>.csv, leaving storage_hubs.csv alone (2026-10-16).",
    )
    parser.add_argument(
        "--linkage", choices=CLUSTER_LINKAGES, default=None,
        help="With --cluster-km (default complete): complete = no two 25 km blocks in a cluster farther apart than the "
             "threshold; average = clusters merge while their mean block-to-block distance stays within it.",
    )
    args = parser.parse_args()
    if args.offshore_basin_dilation_px < 0:
        parser.error("--offshore-basin-dilation-px must be >= 0")
    if args.cluster_km is None:
        if args.linkage is not None:
            parser.error("--linkage only goes with --cluster-km")
    elif not args.cluster_km > 0:
        parser.error("--cluster-km must be > 0")
    linkage = args.linkage or "complete"
    if args.cluster_km is not None and cluster_variant(args.cluster_km, linkage) not in SINK_VARIANTS:
        parser.error(f"sink variant {cluster_variant(args.cluster_km, linkage)!r} is not in "
                     f"coal_retrofit.sink_inputs.SINK_VARIANTS {SINK_VARIANTS}; add it there first")
    storage, path = write_storage_hubs(
        ProjectPaths(ROOT), offshore_basin_dilation_px=args.offshore_basin_dilation_px,
        cluster_km=args.cluster_km, cluster_linkage=linkage,
    )
    offshore = int(storage["offshore"].sum())
    print(f"Wrote {len(storage)} storage hub rows ({offshore} offshore) to {path}.")


if __name__ == "__main__":
    main()
