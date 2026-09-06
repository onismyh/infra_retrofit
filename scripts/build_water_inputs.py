from __future__ import annotations

from _bootstrap import ROOT
from coal_retrofit.builders.water import write_water_inputs
from coal_retrofit.paths import ProjectPaths


def main() -> None:
    paths = ProjectPaths(ROOT)
    scenarios, base, nodes, availability = write_water_inputs(paths)
    print(f"Wrote {len(scenarios)} water scenario rows.")
    print(f"Wrote {len(base)} water baseline rows.")
    print(f"Wrote {len(nodes)} water node rows.")
    print(f"Wrote {len(availability)} water availability rows.")


if __name__ == "__main__":
    main()
