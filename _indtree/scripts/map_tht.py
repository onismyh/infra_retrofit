# -*- coding: utf-8 -*-
"""统一底图：按唐昊天 GIS_layer/plot.ipynb 的画法重写，全仓库地图共用这一套。

与旧的 plot_style.draw_china_basemap 的区别（旧版仍保留，不动已投稿的图）：
  省界  data/ChinaMapTHT/中华人民共和国.json —— GeoJSON，35 个要素，2023 年版，
        取代 2018 年的 ChinaMap/provinces.shp；线色改黑，lw 0.2。
  国界  data/ChinaMapTHT/china_country_proj.shp —— 含九段线，黑色 lw 0.75。
  范围  主图固定 (80E,15N)-(150E,50N)，南海小图 (106.5E,2.8N)-(123E,24.5N)，
        与 plot.ipynb 的 bound 四点完全一致，而不是按要素四至自适应。
  投影  EPSG:2380，与 plot_style.MAP_CRS 相同，散点仍可用 plot_style.to_map_xy。
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "data" / "ChinaMapTHT"
PROV_JSON = MAP_DIR / "中华人民共和国.json"
DASH_ADCODE = "100000_JD"
COUNTRY_SHP = MAP_DIR / "china_country_proj.shp"
CRS = "EPSG:2380"

BOUND_MAIN = (80.0, 15.0, 150.0, 50.0)          # lon0, lat0, lon1, lat1
BOUND_SCS = (106.5, 2.8, 123.0, 24.5)
PROV_LW, COUNTRY_LW, LINE_C = 0.2, 0.75, "black"

_CACHE: dict = {}


def layers():
    """(省界, 国界) 两个 GeoDataFrame，已投到 EPSG:2380；只读一次。"""
    if "layers" not in _CACHE:
        import geopandas as gpd
        prov = gpd.read_file(PROV_JSON).to_crs(CRS)
        country = gpd.read_file(COUNTRY_SHP).to_crs(CRS)
        _CACHE["layers"] = (prov, country)
    return _CACHE["layers"]


def xy(lon, lat):
    """经纬度 -> EPSG:2380 米制坐标。"""
    import numpy as np
    from pyproj import Transformer
    if "tf" not in _CACHE:
        _CACHE["tf"] = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    return _CACHE["tf"].transform(np.asarray(lon, float), np.asarray(lat, float))


def _box(bound):
    x0, y0 = xy([bound[0], bound[2]], [bound[1], bound[3]])
    return (x0[0], x0[1]), (y0[0], y0[1])


ISLAND_MIN_AREA_KM2 = 1000.0    # 主图保留的最小部件面积；次大的岛只有 490 km2


def dash():
    """九段线：GeoJSON 里 adcode = 100000_JD 的那个要素，单独成层，任何尺度都要画。"""
    if "dash" not in _CACHE:
        import geopandas as gpd
        g = gpd.read_file(PROV_JSON).to_crs(CRS)
        d = g[g["adcode"].astype(str) == DASH_ADCODE]
        if d.empty:
            raise RuntimeError("底图缺九段线要素 %s：%s" % (DASH_ADCODE, PROV_JSON))
        _CACHE["dash"] = d
    return _CACHE["dash"]


def _main_parts():
    if "main_parts" not in _CACHE:
        import geopandas as gpd
        c = layers()[1]
        parts = [g for g in c.geometry.iloc[0].geoms if g.area / 1e6 >= ISLAND_MIN_AREA_KM2]
        _CACHE["main_parts"] = gpd.GeoDataFrame(geometry=parts, crs=c.crs)
    return _CACHE["main_parts"]


def draw(ax, province_lw: float = PROV_LW, country_lw: float = COUNTRY_LW,
         facecolor: str = "none", edgecolor: str = LINE_C, zorder: float = 1.0,
         province_ec: str | None = None, country_ec: str | None = None,
         islands: bool = False):
    """省界 + 国界（含九段线）。facecolor 传颜色时给省面上底色。

    province_ec / country_ec 可以分别改线色；不传就都用 edgecolor（昊天的画法是全黑）。
    """
    prov, country = layers()
    if facecolor != "none":
        prov.plot(ax=ax, facecolor=facecolor, edgecolor="none", zorder=zorder - 0.5)
    prov.plot(ax=ax, facecolor="none", edgecolor=province_ec or edgecolor,
              linewidth=province_lw, zorder=zorder)
    # 主图只画大陆 / 台湾 / 海南：国界层另有 1 257 个小岛部件，中位 1.1 km2，在主图尺度上
    # 每个都画不满一个像素，叠起来就是东南海岸一圈黑毛刺。小图（islands=True）才是它们的舞台。
    (country if islands else _main_parts()).plot(
        ax=ax, facecolor="none", edgecolor=country_ec or edgecolor,
        linewidth=country_lw, zorder=zorder + 0.1)
    dash().plot(ax=ax, facecolor="none", edgecolor=country_ec or edgecolor,
                linewidth=country_lw, zorder=zorder + 0.15)


def bound_tight(south_lat: float = 17.0, pad: float = 0.02):
    """省界四至、南边裁到 south_lat 的紧凑画框（EPSG:2380 的 xlim/ylim）。

    昊天的固定画框 (80E,15N)-(150E,50N) 是给 8x8 单幅图用的，右侧那块空白正好放图例
    和南海小图；多面板或右侧另有栏目的版式用它会把中国压得很小，这时改用本函数。
    """
    prov, _ = layers()
    x0, y0, x1, y1 = prov.total_bounds
    _, ys = xy([105.0], [south_lat])
    y0 = max(y0, ys[0])
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def set_extent(ax, bound=BOUND_MAIN):
    """bound 可以是 (lon0, lat0, lon1, lat1)，也可以是 bound_tight() 返回的 ((x0,x1),(y0,y1))。"""
    if len(bound) == 2:
        (x0, x1), (y0, y1) = bound
    else:
        (x0, x1), (y0, y1) = _box(bound)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()


def scs_inset(ax, business=None, rect=(0.755, 0.012, 0.235, 0.30),
              province_lw: float = 0.12, country_lw: float = 0.45, facecolor: str = "none",
              bg: str = "white", province_ec: str | None = None, country_ec: str | None = None):
    """南海小图：同底图、同业务图层，只换 xlim/ylim。business(inset_ax) 是回调。

    bg 给小图一个不透明底：主图的业务图层会画到小图所在的位置，透明底会让两层叠在一起。
    """
    ins = ax.inset_axes(rect, zorder=8)
    ins.set_facecolor(bg)
    ins.patch.set_alpha(1.0)
    draw(ins, province_lw=province_lw, country_lw=country_lw, facecolor=facecolor,
         province_ec=province_ec, country_ec=country_ec, islands=True)
    if business is not None:
        business(ins)
    set_extent(ins, BOUND_SCS)
    ins.set_xticks([]); ins.set_yticks([]); ins.set_title("")
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor("#8A8079")
    ins.set_frame_on(True)
    return ins
