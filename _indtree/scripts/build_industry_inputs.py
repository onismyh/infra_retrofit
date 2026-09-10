from __future__ import annotations

import logging

from _bootstrap import ROOT
from coal_retrofit.builders.industry import write_industry_inputs
from coal_retrofit.paths import ProjectPaths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = ProjectPaths(ROOT)
    sources, hubs = write_industry_inputs(paths)
    print(f"Wrote {len(sources)} rows to {paths.inputs_dir / 'industry_sources.csv'}")
    print(f"Wrote {len(hubs)} rows to {paths.inputs_dir / 'industry_hubs.csv'}")
    summary = (
        sources.groupby("sector_zh")
        .agg(
            厂点数=("source_id", "size"),
            hub数=("hub_id", "nunique"),
            产量Mt=("production_kt_per_year", lambda s: s.sum() / 1000.0),
            排放Mt=("co2_mt_per_year", "sum"),
            需氢Mt=("h2_demand_kt_per_year", lambda s: s.sum() / 1000.0),
            取水亿m3=("water_m3_per_year", lambda s: s.sum() / 1e8),
            投产年实测占比=("commission_year_observed", "mean"),
        )
        .round(2)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
