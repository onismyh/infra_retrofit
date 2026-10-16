"""中国底图（CLAUDE.md §4.1–4.4）：EPSG:2380 坐标、省界与国界、九段线、南海小图、图例不压国土的检查与管段折线。

地图（图 1、2、6）都用这里的封装，不各自拼图层与定位点；底图与南海小图的画法见 CLAUDE.md §4.3。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT
from plot_style import LAND, MUTED

MAP_CRS = "EPSG:2380"        # Xian 1980 / 3-degree Gauss-Kruger CM 105E，只用于显示，不在其下算长度与面积
SCS_LONLAT = ((106.5, 2.8), (123.0, 24.5))   # 南海小图的西南角、东北角
MAP_DIR = "ChinaMapTHT"
PROV_FILE = "中华人民共和国.json"
COUNTRY_FILE = "china_country_proj.shp"
DASH_ADCODE = "100000_JD"    # GeoJSON 里单独成要素的九段线
ISLAND_MIN_AREA_KM2 = 1000.0  # 主图国界只留大陆、台湾、海南三块（次大的岛只有 458 km2）
_MAP_CACHE: dict = {}
MAP_LEGEND: dict[str, Any] = {"frameon": True, "facecolor": "white", "framealpha": 0.85, "edgecolor": "none",
                              "borderpad": 0.4}   # 地图上的图例：白底半透明，压在省界上的字才看得清


def to_map_xy(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    """经纬度 -> EPSG:2380 米制坐标。散点、折线的坐标都走这里。"""
    from pyproj import Transformer

    if "tf" not in _MAP_CACHE:
        _MAP_CACHE["tf"] = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
    x, y = _MAP_CACHE["tf"].transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def map_layer(name: str):
    """底图图层（EPSG:2380，读一次后缓存）：

    - "provinces"：`中华人民共和国.json`（2023 版，含台湾与港澳）的省界，已剔除 adcode = 100000_JD 的九段线要素；
    - "dash"：同一文件里 adcode = 100000_JD 的九段线，主图和小图都要画；
    - "country"：`china_country_proj.shp` 国界，单要素、1 260 个部件，南到 3.83°N，只在南海小图里整层画；
    - "country_main"：国界中面积 ≥ 1 000 km² 的部件，只有大陆、台湾、海南三块，主图用。面积在 shp 原生的
      Albers 等积投影下算（§4.1：不在 EPSG:2380 下算面积）。
    """
    import geopandas as gpd

    if name not in _MAP_CACHE:
        base = REPO_ROOT / "data" / MAP_DIR   # 仓库根 data/ 入库；_indtree/data 只是本机的目录链接
        if name == "country":
            country = gpd.read_file(base / COUNTRY_FILE).to_crs("EPSG:4326")
            # 断言不是防御性编程：换底图时九段线会静默消失，缺九段线的中国地图在国内是发表阻断项。
            if float(country.total_bounds[1]) > 5.0:
                raise RuntimeError(f"底图缺九段线：{base / COUNTRY_FILE} 的南界只到 "
                                   f"{country.total_bounds[1]:.2f}N")
            _MAP_CACHE[name] = country.to_crs(MAP_CRS)
        elif name == "country_main":
            native = gpd.read_file(base / COUNTRY_FILE)
            parts = [g for g in native.geometry.iloc[0].geoms if g.area / 1e6 >= ISLAND_MIN_AREA_KM2]
            _MAP_CACHE[name] = gpd.GeoDataFrame(geometry=parts, crs=native.crs).to_crs(MAP_CRS)
        else:
            layer = gpd.read_file(base / PROV_FILE).to_crs(MAP_CRS)
            is_dash = layer["adcode"].astype(str) == DASH_ADCODE
            if not is_dash.any():
                raise RuntimeError(f"底图缺九段线要素 {DASH_ADCODE}：{base / PROV_FILE}")
            _MAP_CACHE["provinces"] = layer[~is_dash].reset_index(drop=True)
            _MAP_CACHE["dash"] = layer[is_dash]
    return _MAP_CACHE[name]


def draw_china_basemap(ax, *, facecolor: str = LAND, islands: bool = False,
                       province_lw: float = 0.2, country_lw: float = 0.75) -> None:
    """省界 + 国界 + 九段线（线宽与 zorder 见 CLAUDE.md §4.4）。

    主图只画大陆、台湾、海南三块国界：国界层另有 1 257 个小岛部件（中位 1.0 km2，等积投影下算），在主图尺度上
    每个都画不满一个像素，叠起来是东南海岸一圈黑毛刺。南海小图里 islands=True，岛礁在那个尺度上才是内容。
    """
    prov = map_layer("provinces")
    if facecolor != "none":
        prov.plot(ax=ax, facecolor=facecolor, edgecolor="none", zorder=0)
    prov.boundary.plot(ax=ax, color="black", linewidth=province_lw, zorder=1)
    country = map_layer("country" if islands else "country_main")
    country.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=country_lw, zorder=1.5)
    map_layer("dash").plot(ax=ax, facecolor="none", edgecolor="black", linewidth=country_lw, zorder=1.6)


def mainland_extent(ax, *, south_lat: float = 17.5, pad: float = 0.02) -> None:
    """主图范围：省界四至，南边裁到 17.5°N（南边不留白）。

    省界层含南海要素（海南省要素含南沙），四至南到 3.8°N，直接用会把大陆压到画面上半部；
    九段线主体与南海岛礁交给南海小图，这正是小图必需的原因（§4.2）。17.5°N 在中央经线上量：海南（最南 18.15°N）完整保留、离底边 77 km；
    西沙（最北 17.12°N）与 16°N 附近的两段九段线离底边 30 km 以上，主图底边不会露出碎片。
    """
    x0, _, x1, y1 = map_layer("provinces").total_bounds
    y0 = float(to_map_xy([105.0], [south_lat])[1][0])
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0, y1 + dy)
    ax.set_aspect("equal")
    ax.set_axis_off()


def add_scs_inset(ax, draw=None, *, height: float = 0.26):
    """南海小图：与主图同底图、同业务图层、同尺寸律，只换范围（§4.3）。在 mainland_extent 之后调用。

    小图放在主图右下角、台湾以东的海面上，高为主图的 *height*，宽按小图范围的纵横比。省界四至的东界（抚远）
    离台湾只有 490 km，放不下小图，所以先把主图东界往东扩到"小图这一带陆地的最东点 + 间距 + 小图宽"，
    小图不压台湾与沿海。*draw(inset_ax)* 重画业务图层。
    """
    from shapely.geometry import box

    (lon0, lat0), (lon1, lat1) = SCS_LONLAT
    xs, ys = to_map_xy([lon0, lon1], [lat0, lat1])
    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    margin = 0.01 * (xb - xa)
    inset_h = height * (yb - ya)
    inset_w = inset_h * (xs[1] - xs[0]) / (ys[1] - ys[0])
    band = box(xa, ya, xb, ya + margin + inset_h)
    land_east = max(g.intersection(band).bounds[2] for g in map_layer("country_main").geometry
                    if g.intersects(band))
    xb = max(xb, land_east + 3 * margin + inset_w)
    ax.set_xlim(xa, xb)
    inset = ax.inset_axes((xb - margin - inset_w, ya + margin, inset_w, inset_h), transform=ax.transData)
    draw_china_basemap(inset, islands=True, province_lw=0.15, country_lw=0.5)
    if draw is not None:
        draw(inset)
    inset.set_xlim(xs[0], xs[1])
    inset.set_ylim(ys[0], ys[1])
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    # geopandas 1.2 起对投影 CRS 的图层自动写轴标签（"Easting [metre]" / "Northing [metre]"）；主图 `mainland_extent`
    # 关了整个坐标轴看不到，小图只关刻度、保留方框，标签会落在小图左侧与下方（§4.5 无坐标轴文字）。放在底图与
    # 业务图层都画完之后清，回调里再画 geopandas 图层会重设。
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.set_facecolor("white")
    for side in inset.spines.values():
        side.set_visible(True)     # apply_style 关了上、右轴线，小图要完整方框
        side.set_linewidth(0.4)
        side.set_edgecolor(MUTED)
    return inset


def check_off_land(ax, *artists) -> None:
    """地图上的图例不许压到国土（大陆、台湾、海南）：按画出来的外框查，压到就报错，不出图。"""
    from shapely.geometry import box

    ax.apply_aspect()          # 等比例的轴画之前才定下外框，先定下来，数据坐标换算才与出图一致
    renderer = ax.figure.canvas.get_renderer()
    to_data = ax.transData.inverted()
    for artist in artists:
        (x0, y0), (x1, y1) = to_data.transform(artist.get_window_extent(renderer).get_points())
        if any(g.intersects(box(x0, y0, x1, y1)) for g in map_layer("country_main").geometry):
            raise RuntimeError("图例压到了国土：挪到国界以外的空白处（如西藏以南）或缩小，不出图")


def edge_lines(edges: pd.DataFrame, nodes: pd.DataFrame) -> list[np.ndarray]:
    """管段的 EPSG:2380 折线：有 WKT 的用 WKT，否则连两端节点；端点不在节点表里就报错（不静默少画）。

    *edges* 要有 edge_id、from_node_id、to_node_id，可选 geometry_wkt；*nodes* 要有 node_id、lon、lat。
    """
    from shapely import wkt

    lon = dict(zip(nodes["node_id"].astype(str), nodes["lon"].astype(float)))
    lat = dict(zip(nodes["node_id"].astype(str), nodes["lat"].astype(float)))
    has_wkt = "geometry_wkt" in edges.columns
    out = []
    for edge in edges.itertuples(index=False):
        geom = getattr(edge, "geometry_wkt", None) if has_wkt else None
        if isinstance(geom, str) and geom.startswith("LINESTRING"):
            coords = np.asarray(wkt.loads(geom).coords)
            x, y = to_map_xy(coords[:, 0], coords[:, 1])
        else:
            a, b = str(edge.from_node_id), str(edge.to_node_id)
            if a not in lon or b not in lon:
                raise ValueError(f"管段 {edge.edge_id} 的端点不在节点表里，图上会少画")
            x, y = to_map_xy([lon[a], lon[b]], [lat[a], lat[b]])
        out.append(np.column_stack([x, y]))
    return out
