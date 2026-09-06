from __future__ import annotations

from _bootstrap import ROOT
from coal_retrofit.builders.plants import write_plants_unit
from coal_retrofit.paths import ProjectPaths


def main() -> None:
    paths = ProjectPaths(ROOT)
    output = write_plants_unit(paths)
    print(f"Wrote {len(output)} rows to {paths.inputs_dir / 'plants_unit.csv'}")


if __name__ == "__main__":
    main()
