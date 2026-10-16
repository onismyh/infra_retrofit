from __future__ import annotations

import argparse

import pandas as pd
from _bootstrap import ROOT
from coal_retrofit.artifacts import write_csv
from coal_retrofit.builders.network import write_network_inputs
from coal_retrofit.builders.network_corridors import load_corridor_layer
from coal_retrofit.builders.network_offshore import offshore_length_km
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.sink_inputs import SINK_VARIANTS, sink_input_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-corridors", action="store_true",
                        help="Build terminal-only network (no existing oil/gas corridors)")
    parser.add_argument("--offshore-only", action="store_true",
                        help="不重建管网，只给现有 pipeline_candidate_edges.csv 补算 offshore_length_km 列")
    parser.add_argument("--sinks", choices=[variant for variant in SINK_VARIANTS if variant], default="",
                        help="汇变体（coal_retrofit.sink_inputs）：读 storage_hubs_<变体>.csv，写同后缀的两张管网表；"
                             "缺省读写不带后缀的表")
    args = parser.parse_args()

    paths = ProjectPaths(ROOT)
    if args.offshore_only:
        edges_path = sink_input_path(paths.inputs_dir, "pipeline_candidate_edges", args.sinks)
        nodes = pd.read_csv(sink_input_path(paths.inputs_dir, "pipeline_nodes", args.sinks))
        edges = pd.read_csv(edges_path)
        edges["offshore_length_km"] = offshore_length_km(paths, nodes, edges)
        write_csv(edges, edges_path)
        print(f"{edges_path}: {int((edges['offshore_length_km'] > 0).sum())} of {len(edges)} edges have "
              f"offshore segments, {edges['offshore_length_km'].sum():.1f} km")
        return
    use_corridors = not args.no_corridors

    if use_corridors:
        gas_layer, _, gas_fields = load_corridor_layer(paths, "gas_pipelines.shp", "gas")
        oil_layer, _, oil_fields = load_corridor_layer(paths, "oil_pipeline.shp", "oil")
        print(f"gas_pipelines.shp: CRS={gas_layer.crs}, geometry={gas_layer.geometry.geom_type.unique().tolist()}, fields={gas_fields}")
        print(f"oil_pipeline.shp: CRS={oil_layer.crs}, geometry={oil_layer.geometry.geom_type.unique().tolist()}, fields={oil_fields}")
    else:
        print("Building terminal-only network (no existing corridors)")

    nodes, edges = write_network_inputs(paths, use_corridors=use_corridors, sink_variant=args.sinks)
    print(f"Wrote {len(nodes)} nodes to {sink_input_path(paths.inputs_dir, 'pipeline_nodes', args.sinks)}")
    print(f"Wrote {len(edges)} edges to {sink_input_path(paths.inputs_dir, 'pipeline_candidate_edges', args.sinks)}")


if __name__ == "__main__":
    main()
