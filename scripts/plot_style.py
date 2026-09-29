"""全仓库唯一一套绘图样式与工具（CLAUDE.md §三、§四）。

出图脚本 `scripts/plot_fig*.py` 都从这里取：字体与版式（`apply_style`）、配色（§3.3）、中英对照（`labels`）、
中国底图（EPSG:2380、九段线、南海小图）、读求解结果（`read_result`）、存图（`save_fig`：字号、图宽、缺字三道检查）。
各图脚本不自设字体、不自带色表。

    from plot_style import apply_style, figure_cli, langs, save_fig
    args = figure_cli(__doc__).parse_args()
    for lang in langs(args.lang):
        apply_style(lang)
        save_fig(draw(data, lang), "fig3_power_pathways", lang)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from _bootstrap import REPO_ROOT, ROOT  # noqa: E402  数据树 _indtree/
from coal_retrofit.constants_water_quota import BASIN_NAMES_ZH  # noqa: E402

INPUTS_DIR = ROOT / "inputs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
# 结果图缺省画的主情景（scenarios/st.toml）。只画单个情景：ST_ 系的差值只能按可证区间报告（CLAUDE.md 二.2）。
MAIN_SCENARIO = "ST_WA_cwatm_126_dry_oq"

# ── 版式（CLAUDE.md §3.2）────────────────────────────────────────────────────────
MM = 1 / 25.4                # mm -> inch
FULL_WIDTH_MM = 183.0        # 双栏宽，save_fig 的图宽上限
MIN_FONT_PT = 5.0            # 排版社缩放后仍可读的下限，save_fig 检查

# ── 字体：中文 SimHei、英文 Arial，各自单字体、不配 fallback（CLAUDE.md §3.1）──────────
LANGS = ("zh", "en")
FONTS = {"zh": "SimHei", "en": "Arial"}
# 面板标号是单个拉丁字母，两种语言都用 Arial 加粗：SimHei 只有一个字重，bold 会被静默忽略。
_LABEL_FONT = {"fontname": "Arial", "fontweight": "bold"}


def apply_style(lang: str = "zh") -> None:
    """字体、字号、线宽与存图参数。每张图画之前调用一次。

    SimHei 没有 U+2212 与上标数字：负号靠 axes.unicode_minus = False，上下标与单位一律走 mathtext
    （r"Mt CO$_2$ yr$^{-1}$"），不要打字面的上标。英文版同样写法，两版只差字体与文字。
    """
    plt.rcParams.update({
        "font.family": FONTS[lang],
        "axes.unicode_minus": False,
        "mathtext.default": "regular",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.titleweight": "normal",
        "axes.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.labelpad": 3,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.4,
        "hatch.linewidth": 0.4,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


# ── 配色（CLAUDE.md §3.3）─────────────────────────────────────────────────────────
INK = "#333333"              # 文字与描边
MUTED = "#8C8C8C"            # 次要文字、参考线
LAND = "#F7F8F9"             # 底图省份填充
# 煤电改造路径：同族同色系，深端 = 带 CCS，浅端 = 不带 CCS。
PATHWAY_ORDER = ("unabated", "ccs", "biomass", "beccs", "ammonia", "retire")
PATHWAY_COLORS = {
    "unabated": "#969696", "ccs": "#636363",       # Greys
    "biomass": "#74C476", "beccs": "#31A354",      # Greens
    "ammonia": "#FDAE6B",                          # Oranges 浅端
    "retire": "#D8DCE0",                           # 中性灰
}
# 工业减排路线：与煤电同一逻辑（灰 = 化石，深 = 带 CCS），氢路线取 §3.3 的 Purples 浅端。
ROUTE_ORDER = ("unabated", "ccs", "h2")
ROUTE_COLORS = {"unabated": "#969696", "ccs": "#636363", "h2": "#9E9AC8"}
# 部门（NPG 色，同一大类同一族）：钢铁红、水泥绿、化工棕；煤电用灰。避开蓝色，蓝色留给水与封存。
SECTOR_ORDER = ("coal", "steel_bf_bof", "steel_eaf", "cement", "ammonia", "methanol")
SECTOR_COLORS = {
    "coal": "#636363",
    "steel_bf_bof": "#E64B35", "steel_eaf": "#F39B7F",
    "cement": "#00A087",
    "ammonia": "#7E6148", "methanol": "#B09C85",
}
# 部门碳目标的四个组（constants_industry.SECTOR_TARGET_GROUP 与 POWER_TARGET_GROUP）。
GROUP_ORDER = ("power", "steel", "cement", "chemicals")
GROUP_COLORS = {"power": "#636363", "steel": "#E64B35", "cement": "#00A087", "chemicals": "#7E6148"}
SINK_COLORS = {"dsa": "#3182BD", "eor": "#9ECAE1"}   # 深部咸水层、驱油封存
PIPE_COLOR = "#3182BD"                               # CO2 管网与 DSA 同色（§3.3 Blues 深端）
AIR_COLOR = "#CC3311"                                # 空冷改造
# 标记：煤电圆、工业菱形、封存汇方块，各图一致。
MARKERS = {"coal": "o", "industry": "D", "sink": "s"}

# ── 中英对照 ─────────────────────────────────────────────────────────────────────
# 各图共用的名词放这里，各图专有的文字放在各脚本顶部的 TEXT 里。流域中文名取模型自己的常量。
_BASIN_EN = {
    "A": "Northeast Rivers", "C": "Hai River", "D": "Yellow River", "E": "Huai River",
    "F": "Yangtze River", "G": "Southeast Rivers", "H": "Pearl River",
    "J": "Southwest Rivers", "K": "Northwest Rivers",
}
BASIN_ORDER = ("A", "C", "D", "E", "K", "F", "H", "G", "J")   # 由北向南，读起来与地图一致
LABELS: dict[str, dict[str, dict[str, str]]] = {
    "zh": {
        "pathway": {"unabated": "未改造", "ccs": "CCS 改造", "biomass": "生物质掺烧",
                    "beccs": "BECCS", "ammonia": "掺氨", "retire": "退役"},
        "route": {"unabated": "未改造", "ccs": "CCS", "h2": "氢路线"},
        "sector": {"coal": "煤电", "steel_bf_bof": "长流程钢铁", "steel_eaf": "电炉钢铁",
                   "cement": "水泥", "ammonia": "合成氨", "methanol": "甲醇"},
        "group": {"power": "电力", "steel": "钢铁", "cement": "水泥", "chemicals": "化工"},
        "sink": {"dsa": "深部咸水层", "eor": "驱油封存"},
        "basin": dict(BASIN_NAMES_ZH),
    },
    "en": {
        "pathway": {"unabated": "Unabated", "ccs": "CCS retrofit", "biomass": "Biomass co-firing",
                    "beccs": "BECCS", "ammonia": "Ammonia co-firing", "retire": "Retired"},
        "route": {"unabated": "Unabated", "ccs": "CCS", "h2": "Hydrogen route"},
        "sector": {"coal": "Coal power", "steel_bf_bof": "BF-BOF steel", "steel_eaf": "EAF steel",
                   "cement": "Cement", "ammonia": "Ammonia", "methanol": "Methanol"},
        "group": {"power": "Power", "steel": "Steel", "cement": "Cement", "chemicals": "Chemicals"},
        "sink": {"dsa": "Deep saline aquifer", "eor": "EOR"},
        "basin": _BASIN_EN,
    },
}
MT_CO2_YR = r"Mt CO$_2$ yr$^{-1}$"


def labels(lang: str) -> dict[str, dict[str, str]]:
    """共用名词的对照表：labels(lang)["pathway"]["ccs"]。"""
    return LABELS[lang]


# ── 命令行与读结果 ───────────────────────────────────────────────────────────────
def figure_cli(doc: str | None, *, scenario: bool = True) -> argparse.ArgumentParser:
    """出图脚本的统一参数：--lang zh|en|both（缺省两版都出），结果图另有 --scenario。"""
    parser = argparse.ArgumentParser(description=(doc or "").strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=("zh", "en", "both"), default="both",
                        help="出中文版、英文版或两版（缺省 both；英文版文件名加 _en）")
    if scenario:
        parser.add_argument("--scenario", default=MAIN_SCENARIO,
                            help=f"读 _indtree/results/<情景>/ 的结果表（缺省 {MAIN_SCENARIO}）")
    return parser


def langs(choice: str) -> tuple[str, ...]:
    return LANGS if choice == "both" else (choice,)


def result_dir(scenario: str) -> Path:
    return RESULTS_DIR / scenario


def read_result(scenario: str, table: str) -> pd.DataFrame:
    """读 `_indtree/results/<情景>/<表>.csv`；没有就报错并提示先求解。"""
    path = result_dir(scenario) / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在：先求解 python -m coal_retrofit run {scenario}")
    return pd.read_csv(path)


def read_result_json(scenario: str) -> dict:
    """读 `_indtree/results/<情景>.json`，并打印求解时的提交号，便于核对结果来自哪一版代码。"""
    path = RESULTS_DIR / f"{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在：先求解 python -m coal_retrofit run {scenario}")
    with path.open(encoding="utf-8") as f:
        result = json.load(f)
    code = (result.get("resolved") or {}).get("code") or {}
    print(f"  {scenario}：求解提交 {code.get('commit') or '未记录'}"
          f"{'（有未提交的改动）' if code.get('dirty') else ''}")
    return result


def read_input(name: str) -> pd.DataFrame:
    """读 `_indtree/inputs/<name>.csv`（列名去掉 BOM）。"""
    frame = pd.read_csv(INPUTS_DIR / f"{name}.csv", encoding="utf-8-sig")
    return frame


def check_close(what: str, got, expected, *, atol: float = 1e-3, rtol: float = 1e-6) -> None:
    """出图前的自检：图上要画的量与模型记的量对不上就报错，不出图。"""
    got_arr = np.asarray(got, dtype=float)
    exp_arr = np.asarray(expected, dtype=float)
    if got_arr.shape != exp_arr.shape or not np.allclose(got_arr, exp_arr, rtol=rtol, atol=atol):
        diff = np.nanmax(np.abs(got_arr - exp_arr)) if got_arr.shape == exp_arr.shape else float("nan")
        raise ValueError(f"自检不通过：{what}（最大偏差 {diff:.6g}），不出图")


# ── 中国底图（CLAUDE.md §4.1–4.4）────────────────────────────────────────────────
MAP_CRS = "EPSG:2380"        # Xian 1980 / 3-degree Gauss-Kruger CM 105E，只用于显示，不在其下算长度与面积
SCS_LONLAT = ((106.5, 2.8), (123.0, 24.5))   # 南海小图的西南角、东北角
MAP_DIR = "ChinaMapTHT"
PROV_FILE = "中华人民共和国.json"
COUNTRY_FILE = "china_country_proj.shp"
DASH_ADCODE = "100000_JD"    # GeoJSON 里单独成要素的九段线
ISLAND_MIN_AREA_KM2 = 1000.0  # 主图国界只留大陆、台湾、海南三块（次大的岛只有 490 km2）
_MAP_CACHE: dict = {}


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
    - "country_main"：国界中面积 ≥ 1 000 km² 的部件，只有大陆、台湾、海南三块，主图用。
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
            c = map_layer("country")
            parts = [g for g in c.geometry.iloc[0].geoms if g.area / 1e6 >= ISLAND_MIN_AREA_KM2]
            _MAP_CACHE[name] = gpd.GeoDataFrame(geometry=parts, crs=c.crs)
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

    主图只画大陆、台湾、海南三块国界：国界层另有 1 257 个小岛部件（中位 1.1 km2），在主图尺度上
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

    省界层含南海要素，四至南到 6.3°N，直接用会把大陆压到画面上半部；九段线主体与南海岛礁交给南海小图，
    这正是小图必需的原因（§4.2）。17.5°N 在中央经线上量：海南（最南 18.15°N）完整保留、离底边 77 km；
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
    inset.set_facecolor("white")
    for side in inset.spines.values():
        side.set_visible(True)     # apply_style 关了上、右轴线，小图要完整方框
        side.set_linewidth(0.4)
        side.set_edgecolor(MUTED)
    return inset


def check_off_land(ax, *artists) -> None:
    """地图上的图例不许压到国土（大陆、台湾、海南）：按画出来的外框查，压到就报错，不出图。"""
    from shapely.geometry import box

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


# ── 常用画法 ─────────────────────────────────────────────────────────────────────
def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    """面板标号：加粗小写字母，不加括号，放在面板左上角外侧（§3.2）。"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            color="black", **_LABEL_FONT)


def figure_label(fig, letter: str, x: float, y: float) -> None:
    """面板标号放在图幅坐标上：地图这类四周没有轴外空白的面板用它。"""
    fig.text(x, y, letter, fontsize=8, va="top", ha="left", color="black", **_LABEL_FONT)


def stacked_bars(ax, x, table: pd.DataFrame, colors: dict[str, str], *, width: float = 0.62) -> None:
    """按列堆叠的柱：正值向上、负值向下各自累加，负值不会把上面的段拉下来。

    *table* 的行对应 *x*，列按堆叠顺序（自下而上）。
    """
    pos = np.zeros(len(table))
    neg = np.zeros(len(table))
    for key in table.columns:
        values = table[key].to_numpy(dtype=float)
        bottom = np.where(values >= 0, pos, neg)
        ax.bar(x, values, width=width, bottom=bottom, color=colors[key],
               edgecolor="white", linewidth=0.4, zorder=3)
        pos = pos + np.clip(values, 0, None)
        neg = neg + np.clip(values, None, 0)
    if (neg < 0).any():
        ax.axhline(0.0, color=INK, linewidth=0.5, zorder=4)


def fmt_number(value: float) -> str:
    """柱顶、格子里的数字：绝对值 ≥ 10 取整，否则一位小数，不到 0.05 记 0。"""
    if abs(value) < 0.05:
        return "0"
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}"


MAP_LEGEND: dict[str, Any] = {"frameon": True, "facecolor": "white", "framealpha": 0.85, "edgecolor": "none",
                              "borderpad": 0.4}   # 地图上的图例：白底半透明，压在省界上的字才看得清


def legend_patches(keys, colors: dict[str, str], names: dict[str, str]) -> list[Patch]:
    return [Patch(facecolor=colors[k], edgecolor="none", label=names[k]) for k in keys]


def area_scale(values, vmax: float, *, smin: float = 1.5, smax: float = 42.0) -> np.ndarray:
    """散点面积（pt²）与数值成正比，加一个下限让小点看得见；各图的尺寸图例用同一函数。"""
    v = np.clip(np.asarray(values, dtype=float) / max(float(vmax), 1e-12), 0.0, 1.0)
    return smin + (smax - smin) * v


def size_legend(values, vmax: float, marker: str, color: str, fmt: str = "{:g}",
                smax: float = 42.0) -> list[Line2D]:
    """尺寸图例：按代表值另建句柄，不直接拿数据点当图例（§4.4）；*smax* 与画点时一致。"""
    sizes = area_scale(values, vmax, smax=smax)
    return [Line2D([], [], marker=marker, linestyle="none", markersize=float(np.sqrt(s)),
                   markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.3,
                   label=fmt.format(v)) for v, s in zip(values, sizes)]


# ── 存图 ─────────────────────────────────────────────────────────────────────────
def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _check_font_sizes(fig, stem: str) -> None:
    small = sorted({(round(float(t.get_fontsize()), 2), t.get_text()[:30])
                    for t in fig.findobj(Text)
                    if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_FONT_PT})
    if small:
        raise RuntimeError(f"{stem}：有文字小于 {MIN_FONT_PT:g} pt，排版社缩放后不可读：{small[:5]}")


def _check_width(fig, stem: str) -> float:
    """图宽检查：bbox_inches="tight" 会把画到轴外的东西都撑进画布，超过 183 mm 就报错并列出最宽的文字。"""
    renderer = fig.canvas.get_renderer()
    width_mm = fig.get_tightbbox(renderer).width * 25.4
    if width_mm > FULL_WIDTH_MM:
        widest = sorted(((t.get_window_extent(renderer).width / fig.dpi * 25.4, t.get_text()[:40])
                         for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()),
                        reverse=True)[:3]
        raise RuntimeError(f"{stem}：图宽 {width_mm:.1f} mm 超过 {FULL_WIDTH_MM:g} mm；最宽的文字 {widest}")
    return width_mm


def save_fig(fig, name: str, lang: str = "zh", out_dir: Path | None = None) -> list[Path]:
    """存 PDF + PNG（300 dpi），英文版文件名加 `_en`；缺省写到 `_indtree/results/figures/`。

    先查字号与图宽，再写临时文件并收集缺字警告：缺字（SimHei / Arial 没有的字符会画成方框）就删掉临时文件、
    报错，已有的同名图不动；都通过才换上正式文件名。
    """
    out_dir = FIGURES_DIR if out_dir is None else Path(out_dir)
    stem = name if lang == "zh" else f"{name}_en"
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_font_sizes(fig, stem)
    width_mm = _check_width(fig, stem)
    pad = 0.02 if width_mm + 2 * 0.02 * 25.4 <= FULL_WIDTH_MM else 0.0
    tmp = {ext: out_dir / f".{stem}.tmp{ext}" for ext in (".pdf", ".png")}
    written = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            for ext, path in tmp.items():
                fig.savefig(path, format=ext[1:], dpi=300, bbox_inches="tight", pad_inches=pad)
        glyphs = sorted({str(w.message) for w in caught
                         if "missing from" in str(w.message) and "font" in str(w.message)})
        if glyphs:
            raise RuntimeError(
                f"{stem}：字体缺字，图里会出现方框，已拒绝出图：\n  " + "\n  ".join(glyphs[:8])
                + "\n上下标与单位走 mathtext（r\"$10^8$ m$^3$\"），负号靠 axes.unicode_minus=False；"
                  "英文版出现这条通常是有中文没翻译。")
        for ext, path in tmp.items():
            final = out_dir / f"{stem}{ext}"
            os.replace(path, final)
            written.append(final)
    finally:
        for path in tmp.values():
            path.unlink(missing_ok=True)
        plt.close(fig)
    print(f"  [ok] {_relative(out_dir / stem)}.pdf / .png（{width_mm:.0f} mm 宽）")
    return written
