"""候选边的海上段长度：边的几何落在陆地省界之外的那部分，按大地线量，再按比例折到边的计价长度上。

求解时只有海上段乘 `OptimizationAssumptions.offshore_transport_multiplier`（`year_matrices._edge_matrices`）。
陆地取 `data/ChinaMap/provinces.shp` 各省的并，再并上 Natural Earth 1:10m 陆地（`NATURAL_EARTH_LAND`）：省界之外
还有邻国陆地，穿过邻国的边（如云南经老挝、越南北部到北部湾，内蒙古经蒙古国东部）不算海上段。
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely import get_parts
from shapely.geometry import LineString
from shapely.prepared import prep

from ..paths import ProjectPaths
from ..spatial import geodesic_length_km, load_provinces
from .network_repair import _build_edge_geometry

logger = logging.getLogger(__name__)

# Natural Earth 1:10m 陆地 v5.1.2 裁到 60–150°E、0–60°N（`data/NaturalEarth/README.md`），相对 `data_dir`。
NATURAL_EARTH_LAND = "NaturalEarth/ne_10m_land_60e150e_0n60n.gpkg"
# 两点直线边沿大圆加密的间距（km）：在经纬度平面上与海岸线求交时，线要沿大圆走，而不是经纬度上的直线。
_DENSIFY_KM = 5.0
_GEOD = Geod(ellps="WGS84")


def _along_great_circle(line: LineString) -> LineString:
    """两点直线边换成沿大圆、每 `_DENSIFY_KM` 一个点的折线；本来就是折线的（既有干线走廊）原样返回。"""
    if len(line.coords) != 2:
        return line
    (lon1, lat1), (lon2, lat2) = line.coords
    n = int(geodesic_length_km(line.coords) // _DENSIFY_KM)
    if n < 1:
        return line
    return LineString([(lon1, lat1), *_GEOD.npts(lon1, lat1, lon2, lat2, n), (lon2, lat2)])


def _sea_share(line: LineString, land, land_prepared) -> float:
    """边的几何落在陆地之外的大地线长度份额。"""
    if land_prepared.contains(line):
        return 0.0
    total = geodesic_length_km(line.coords)
    if total <= 0.0:
        return 0.0
    sea_km = sum(
        geodesic_length_km(part.coords) for part in get_parts(line.difference(land)) if part.geom_type == "LineString"
    )
    return min(1.0, sea_km / total)


def offshore_length_km(paths: ProjectPaths, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    """每条边的海上段长度（km）= 计价长度 `length_km` x 几何落在陆地之外的份额。

    几何取 `geometry_wkt`，没有的（支线）取两端节点的连线；两点直线先沿大圆加密再与陆地求交。
    """
    provinces = load_provinces(paths.data_dir / "ChinaMap" / "provinces.shp").to_crs("EPSG:4326").geometry.union_all()
    land = provinces.union(gpd.read_file(paths.data_dir / NATURAL_EARTH_LAND).to_crs("EPSG:4326").geometry.union_all())
    land_prepared = prep(land)
    coords = {str(n): (float(x), float(y)) for n, x, y in zip(nodes["node_id"], nodes["lon"], nodes["lat"])}
    shares = []
    for _, row in edges.iterrows():
        geom = _build_edge_geometry(row, coords)
        if geom is None:
            raise ValueError(f"{row['edge_id']}：没有几何，端点也不全在节点表里，量不了海上段")
        shares.append(_sea_share(_along_great_circle(geom), land, land_prepared))
    out = pd.Series(np.round(edges["length_km"].astype(float).to_numpy() * np.asarray(shares), 3), index=edges.index)
    logger.info("offshore segments: %d of %d edges, %.0f km", int((out > 0).sum()), len(out), float(out.sum()))
    return out
