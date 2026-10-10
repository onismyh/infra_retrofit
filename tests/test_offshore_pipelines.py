"""海上管道按每条边的海上段计价（2026-10-10 起）。

`builders.network_offshore.offshore_length_km` 量出边的几何落在陆地（省界与 Natural Earth 陆地）之外的份额，折到计价长度上；求解时每条边的
capex 与按流量计的运维都按 `length_km + (倍率 − 1) x offshore_length_km` 计（`year_matrices._edge_matrices`）。
用一块 1° x 1° 的玩具陆地验算，不需要 Gurobi。
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from coal_retrofit.builders.network_offshore import NATURAL_EARTH_LAND, offshore_length_km
from coal_retrofit.constants import NETWORK_DETOUR_FACTOR
from coal_retrofit.optimization.data_prep import prepare_inputs
from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario
from coal_retrofit.optimization.year_matrices import _edge_matrices
from coal_retrofit.paths import ProjectPaths
from coal_retrofit.run_controls import initial_state
from coal_retrofit.spatial import geodesic_length_km
from toy_inputs import _write_targets, _write_toy_inputs


def _write_land(root) -> ProjectPaths:
    """省界只有一块：东经 110–111°、北纬 30–31°；Natural Earth 陆地另有一块"邻国"：东经 110–111°、北纬 31–32°。"""
    folder = root / "data" / "ChinaMap"
    folder.mkdir(parents=True)
    gpd.GeoDataFrame({"OBJECTID": [1], "NAME": ["Land"]}, geometry=[box(110.0, 30.0, 111.0, 31.0)], crs="EPSG:4326").to_file(
        folder / "provinces.shp"
    )
    neighbour = root / "data" / NATURAL_EARTH_LAND
    neighbour.parent.mkdir(parents=True)
    gpd.GeoDataFrame({"featurecla": ["Land"]}, geometry=[box(110.0, 31.0, 111.0, 32.0)], crs="EPSG:4326").to_file(
        neighbour, driver="GPKG", layer="land"
    )
    return ProjectPaths(root)


def test_offshore_length_counts_only_the_part_of_an_edge_at_sea(tmp_path) -> None:
    """支线（无几何，取两端连线）一半在海上、三角剖分边全在陆上、走廊折线（有几何）全在海上；
    穿过邻国陆地（省界之外、Natural Earth 陆地之内）的边不算海上段。"""
    paths = _write_land(tmp_path)
    nodes = pd.DataFrame({"node_id": ["A", "B", "C", "D"], "lon": [110.5, 111.5, 110.8, 110.5], "lat": [30.5, 30.5, 30.8, 31.5]})
    branch_km = geodesic_length_km([(110.5, 30.5), (111.5, 30.5)]) * NETWORK_DETOUR_FACTOR
    edges = pd.DataFrame({
        "edge_id": ["E1", "E2", "E3", "E4"],
        "from_node_id": ["A", "A", "B", "A"],
        "to_node_id": ["B", "C", "B", "D"],
        "length_km": [branch_km, 40.0, 120.0, 120.0],
        "geometry_wkt": [np.nan, "LINESTRING (110.5 30.5, 110.8 30.8)", "LINESTRING (111.5 30.5, 112 30.6, 112.5 30.5)",
                         np.nan],
    })
    offshore = offshore_length_km(paths, nodes, edges)
    assert offshore[0] == pytest.approx(0.5 * branch_km, rel=0.01)
    assert offshore[1] == 0.0
    assert offshore[2] == pytest.approx(120.0)
    assert offshore[3] == 0.0


def test_edge_capex_prices_only_the_offshore_segment(tmp_path) -> None:
    """100 km 的边有 40 km 在海上：倍率 1.5 时按 100 + 0.5 x 40 = 120 km 计，倍率 1 时按 100 km 计。"""
    paths = _write_toy_inputs(tmp_path, retirement_year=9999)
    edges_path = paths.inputs_dir / "pipeline_candidate_edges.csv"
    pd.read_csv(edges_path).assign(offshore_length_km=40.0).to_csv(edges_path, index=False)
    _write_targets(paths, {2030: 1.0})
    scenario = OptimizationScenario(experiment_id="T", description="toy", sector_target_source="toy")
    for multiplier, priced_km in ((1.5, 120.0), (1.0, 100.0)):
        assumptions = OptimizationAssumptions(offshore_transport_multiplier=multiplier, route_opex_cny_per_t_km=0.15)
        prepared = prepare_inputs(paths, scenario, assumptions)
        matrices = _edge_matrices(prepared, assumptions, initial_state(prepared))
        tier_capex = np.asarray(assumptions.pipe_capex_cny_per_km_by_tier, dtype=float)
        np.testing.assert_allclose(matrices["edge_tier_capex"][0], priced_km * tier_capex)
        assert matrices["edge_route_opex_coeff"][0] == pytest.approx(priced_km * 0.15 * 1e6)
