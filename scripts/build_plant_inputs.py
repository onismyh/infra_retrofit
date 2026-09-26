"""Build the two plant tables.

`--hubs` re-runs the site clustering and rewrites `inputs/plants.csv`. It is off by default
because that file is an input to every solved scenario: regenerating it changes the
`digest_plants` provenance field even when the numbers are identical, and two runs that differ
in it are not comparable (see `solver_provenance._input_digest`). Pass it only when a column has actually
been added or a value corrected, and re-solve everything downstream.
"""
from __future__ import annotations

import argparse

from _bootstrap import ROOT
from coal_retrofit.builders.plants import write_plants, write_plants_unit
from coal_retrofit.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hubs", action="store_true",
                        help="also re-cluster and rewrite inputs/plants.csv")
    args = parser.parse_args()
    paths = ProjectPaths(ROOT)
    output = write_plants_unit(paths)
    print(f"Wrote {len(output)} rows to {paths.inputs_dir / 'plants_unit.csv'}")
    if args.hubs:
        write_plants(paths)
        print(f"Wrote {paths.inputs_dir / 'plants.csv'}")


if __name__ == "__main__":
    main()
