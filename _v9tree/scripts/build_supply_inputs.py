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
