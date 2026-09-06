from __future__ import annotations

import argparse

from _bootstrap import ROOT
from coal_retrofit.builders.network import load_corridor_layer, write_network_inputs
from coal_retrofit.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-corridors", action="store_true",
                        help="Build terminal-only network (no existing oil/gas corridors)")
    args = parser.parse_args()

    paths = ProjectPaths(ROOT)
    use_corridors = not args.no_corridors

    if use_corridors:
        gas_layer, _, gas_fields = load_corridor_layer(paths, "gas_pipelines.shp", "gas")
        oil_layer, _, oil_fields = load_corridor_layer(paths, "oil_pipeline.shp", "oil")
        print(f"gas_pipelines.shp: CRS={gas_layer.crs}, geometry={gas_layer.geometry.geom_type.unique().tolist()}, fields={gas_fields}")
        print(f"oil_pipeline.shp: CRS={oil_layer.crs}, geometry={oil_layer.geometry.geom_type.unique().tolist()}, fields={oil_fields}")
    else:
        print("Building terminal-only network (no existing corridors)")

    nodes, edges = write_network_inputs(paths, use_corridors=use_corridors)
    print(f"Wrote {len(nodes)} nodes to {paths.inputs_dir / 'pipeline_nodes.csv'}")
    print(f"Wrote {len(edges)} edges to {paths.inputs_dir / 'pipeline_candidate_edges.csv'}")


if __name__ == "__main__":
    main()
