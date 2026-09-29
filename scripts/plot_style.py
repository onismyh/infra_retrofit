"""Shared publication style for all coal retrofit figures.

Import this module at the top of any plotting script to ensure consistent:
- Font sizes, families, and weights
- Color palettes (thesis.ipynb colours, CLAUDE.md §3.3)
- Panel labels a, b, c (bold, no parentheses)
- Legend style
- Axis formatting
- Save function with dual PDF+PNG output

Usage:
    from plot_style import *
    apply_style()
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    ...
    panel_label(ax, "a")
    save_fig(fig, "main_fig1_pathway_allocation")
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
from _bootstrap import ROOT  # 数据树 _indtree/
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MAIN_FIGURES_DIR = FIGURES_DIR / "main"
EXTENDED_FIGURES_DIR = FIGURES_DIR / "extended"
PATENT_FIGURES_DIR = FIGURES_DIR / "patent"

# ── Figure sizes (Nature Energy: single=89mm, double=183mm) ─────────────────
MM = 1 / 25.4  # mm to inches
DOUBLE_COL = (183 * MM, 89 * MM * 1.4)         # ~(7.2, 4.9) for double-column

# ── Color palette (CLAUDE.md §3.3) ──────────────────────────────────────────
# 同族同色系，深端 = 带 CCS，浅端 = 不带 CCS。蒸馏自 thesis.ipynb 的配色逻辑
# （Coal w/ CCS #636363 vs w/o #969696；Biomass w/ CCS #31A354 vs w/o #74C476），
# 见 .claude/CLAUDE.md §3.3。取代原先的 Tol bright 系。
PATHWAY_COLORS = {
    "unabated": "#969696",   # Greys 浅端
    "ccs":      "#636363",   # Greys 深端
    "biomass":  "#74C476",   # Greens 浅端
    "beccs":    "#31A354",   # Greens 深端
    "ammonia":  "#FDAE6B",   # Oranges 浅端
    "ammonia_ccs": "#FD8D3C",
    "hydrogen": "#9E9AC8",   # Purples 浅端
    "hydrogen_ccs": "#756BB1",
    "retire":   "#D8DCE0",   # 中性灰
}
PATHWAY_LABELS = {
    "unabated": "未改造燃煤",
    "retire":  "提前退役",
    "ccs":     "CCS 改造",
    "biomass": "生物质掺烧",
    "beccs":   "BECCS",
    "ammonia": "掺氨",
    "hydrogen": "掺氢",
}
PATHWAY_ORDER = ["unabated", "biomass", "ccs", "beccs", "ammonia", "retire"]

# ── 字体：模块级生效，不依赖 apply_style() ──────────────────────────────────
# 九个 ED 脚本只 import MM/save_fig，从不调用 apply_style()。英文时代这没问题，
# 默认 DejaVu Sans 排英文本来就对；改中文之后它是致命的 —— DejaVu 没有一个汉字，
# 整张图的中文全部渲染成空心方框（实测 ed_fig8 一次出 537 条缺字警告）。
# 所以字体设置必须在 import 时就落地，且只动字体，不动版式：
# 没调 apply_style() 的脚本拿到正确字形，其余 rcParams 一概不变，排版零回归。
FONT_RC = {
    "font.family": "SimHei",     # 与 thesis.ipynb 一致，单字体，不配 fallback 栈
    "axes.unicode_minus": False,  # SimHei 无 U+2212，不关负号必成方框
    "mathtext.default": "regular",
}
plt.rcParams.update(FONT_RC)


# ── Global style application ────────────────────────────────────────────────
def apply_style():
    """Apply publication-quality matplotlib style. Call once at script start."""
    plt.rcParams.update({
        # Font
        # 字体：与 thesis.ipynb 完全一致，全局唯一
        #   plt.rcParams['font.family']  = ["SimHei"]
        #   plt.rcParams['axes.unicode_minus'] = False
        # SimHei 缺 U+2212 与上标 ³/²，所以 unicode_minus 必须关，
        # 单位与幂次一律走 mathtext（r"Mt CO$_2$ yr$^{-1}$"），不要打字面上标。
        **FONT_RC,           # 单一来源，见上方 FONT_RC
        "font.size": 8,
        # Axes
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.labelpad": 4,
        # Ticks
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Legend
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": False,
        # Save
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white",
    })


# ── 中国底图：投影、九段线、南海小图（见 .claude/CLAUDE.md §4.1-4.3）────────────
MAP_CRS = "EPSG:2380"          # Xian 1980 / 3-degree Gauss-Kruger CM 105E，单位 m
# [主图西南, 主图东北, 南海小图西南, 南海小图东北]，经纬度
BOUND_LONLAT = [(80.0, 15.0), (150.0, 50.0), (106.5, 2.8), (123.0, 24.5)]

_MAP_CACHE: dict = {}


# 2026-09-12 起全仓库底图统一为唐昊天 GIS_layer/plot.ipynb 那一套（见 scripts/map_tht.py）：
# 省界换 2023 版 GeoJSON，国界与九段线换 china_country_proj.shp。旧的 ChinaMap/*.shp 不再用于
# 出图，但 builders 仍在读（海岸线距离、西藏剔除、省份归属），不能删；出图要改回去只需把
# 下面两个函数的路径换回来。
THT_DIR = "ChinaMapTHT"
THT_PROV = "中华人民共和国.json"
THT_COUNTRY = "china_country_proj.shp"
DASH_ADCODE = "100000_JD"      # GeoJSON 里单独成要素的九段线
PROV_NAME_FIX = {"新疆维吾尔自治区": "新疆维吾尔族自治区"}   # 对齐旧 shp 的写法，保住下游按名连接


def load_country(root=None):
    """国界 + 九段线，投影到 MAP_CRS。

    换成 `data/ChinaMapTHT/china_country_proj.shp`：单要素、1 260 个部件，南到 3.83N，
    九段线与南海岛礁都在这一个多边形里，不再需要按 GBCODE 挑图层。

    断言不是防御性编程：换底图时九段线会静默消失，而缺九段线的中国地图在国内是发表阻断项。
    为兼容既有调用，补一列 GBCODE = 61010，`country[country["GBCODE"].isin(...)]` 仍然可用。
    """
    if "country" in _MAP_CACHE:
        return _MAP_CACHE["country"]
    import geopandas as gpd
    from pathlib import Path

    root = Path(root) if root is not None else ROOT
    path = root / "data" / THT_DIR / THT_COUNTRY
    country = gpd.read_file(path).to_crs("EPSG:4326")
    if float(country.total_bounds[1]) > 5.0:
        raise RuntimeError(f"底图缺九段线：{path} 的南界只到 {country.total_bounds[1]:.2f}N")
    country = country.to_crs(MAP_CRS)
    country["GBCODE"] = COUNTRY_GBCODES[0]
    _MAP_CACHE["country"] = country
    return _MAP_CACHE["country"]


def load_map_provinces(root=None):
    """省界，投影到 MAP_CRS。

    换成 `data/ChinaMapTHT/中华人民共和国.json`（2023 版，含台湾与港澳）。GeoJSON 里把
    九段线单独放成 adcode = 100000_JD 的要素，这里剔掉：九段线由 load_country() 负责，
    留在省界层里会被当成一个省去填色。补 NAME 列（并把新疆的写法对齐旧 shp），
    下游按省名连接的代码不用改。
    """
    if "prov" in _MAP_CACHE:
        return _MAP_CACHE["prov"]
    import geopandas as gpd
    from pathlib import Path

    root = Path(root) if root is not None else ROOT
    prov = gpd.read_file(root / "data" / THT_DIR / THT_PROV).to_crs(MAP_CRS)
    prov = prov[prov["adcode"].astype(str) != DASH_ADCODE].copy()
    prov["NAME"] = prov["name"].astype(str).replace(PROV_NAME_FIX)
    _MAP_CACHE["prov"] = prov.reset_index(drop=True)
    return _MAP_CACHE["prov"]


def map_bounds(root=None):
    """BOUND_LONLAT 的四个定位点，投影到 MAP_CRS。

    定位点必须和图层走同一次投影，否则 set_xlim 对不上 —— 这是把经纬度常数直接当
    xlim 用时最常见的错误。
    """
    if "bounds" in _MAP_CACHE:
        return _MAP_CACHE["bounds"]
    import geopandas as gpd
    from shapely.geometry import Point

    _MAP_CACHE["bounds"] = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in BOUND_LONLAT], crs="EPSG:4326"
    ).to_crs(MAP_CRS).geometry
    return _MAP_CACHE["bounds"]


def to_map_xy(lon, lat, root=None):
    """经纬度数组 -> MAP_CRS 的米制坐标。散点、折线的坐标都要走这里。"""
    import numpy as np
    from pyproj import Transformer

    if "tf" not in _MAP_CACHE:
        _MAP_CACHE["tf"] = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
    x, y = _MAP_CACHE["tf"].transform(np.asarray(lon, dtype=float),
                                      np.asarray(lat, dtype=float))
    return x, y


# boundary.shp 的 GBCODE 语义（要素数与经纬度范围实测，见 scratchpad/fix_basemap_layers.py）
COUNTRY_GBCODES = (61010, 26100)   # 国界+海岸线，九段线
PROV_EDGE = "black"                # 昊天的画法：省界黑色细线，不再用浅灰 #C6CDD4


ISLAND_MIN_AREA_KM2 = 1000.0       # 主图上只保留大陆 / 台湾 / 海南三块（次大的岛只有 490 km2）


def load_dash_line(root=None):
    """九段线，单独一层。GeoJSON 里就是 adcode = 100000_JD 的那个要素（10 个部件）。"""
    if "dash" in _MAP_CACHE:
        return _MAP_CACHE["dash"]
    import geopandas as gpd
    from pathlib import Path

    root = Path(root) if root is not None else ROOT
    g = gpd.read_file(root / "data" / THT_DIR / THT_PROV).to_crs(MAP_CRS)
    dash = g[g["adcode"].astype(str) == DASH_ADCODE]
    if dash.empty:
        raise RuntimeError(f"底图缺九段线要素 {DASH_ADCODE}：{root / 'data' / THT_DIR / THT_PROV}")
    _MAP_CACHE["dash"] = dash
    return dash


def country_main(root=None):
    """国界层里面积大于 ISLAND_MIN_AREA_KM2 的部件，用于主图。"""
    if "country_main" in _MAP_CACHE:
        return _MAP_CACHE["country_main"]
    import geopandas as gpd

    c = load_country(root)
    parts = [g for g in c.geometry.iloc[0].geoms if g.area / 1e6 >= ISLAND_MIN_AREA_KM2]
    _MAP_CACHE["country_main"] = gpd.GeoDataFrame(geometry=parts, crs=c.crs)
    return _MAP_CACHE["country_main"]


def draw_china_basemap(ax, root=None, province_lw: float = 0.20,
                       country_lw: float = 0.75, facecolor: str = "none",
                       islands: bool = False):
    """省界 + 国界（含九段线）。线宽与 zorder 见 CLAUDE.md §4.4。

    默认不画岛礁。boundary.shp 里 26010 + 26080 共 1 035 条岛屿轮廓，在 183 mm 幅面上
    每条都画不满一个像素，叠起来就是南海一片黑斑加东部海岸毛刺 —— 那不是信息，是噪点。
    南海小图里把 islands=True 打开：在那个尺度上岛礁才是内容。
    """
    prov = load_map_provinces(root)
    country = load_country(root)
    if facecolor != "none":
        prov.plot(ax=ax, facecolor=facecolor, edgecolor="none", zorder=0)
    prov.boundary.plot(ax=ax, edgecolor=PROV_EDGE, linewidth=province_lw, zorder=1)
    if islands:
        # 小图尺度上岛礁是内容：整层 1 260 个部件全画。
        country.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=country_lw, zorder=1.5)
    else:
        # 主图尺度上不是。1 257 个小部件（中位 1.1 km2）每个都画不满一个像素，叠起来
        # 就是东南海岸一圈黑毛刺 —— 只留大陆、台湾、海南三块。
        country_main(root).plot(ax=ax, facecolor="none", edgecolor="black",
                                linewidth=country_lw, zorder=1.5)
    # 九段线单独一层，两种情形都必须画。
    load_dash_line(root).plot(ax=ax, facecolor="none", edgecolor="black",
                              linewidth=country_lw, zorder=1.6)
    return prov, country


def mainland_extent(ax, root=None, pad: float = 0.02, south_lat: float = 17.0):
    """主图范围：省界的四至，但南边裁到 *south_lat*。

    为什么要裁：`provinces.shp` 的 34 个要素本身就延伸到 6.32N —— 南海诸岛包含在里面。
    直接用 `provinces.total_bounds` 会得到一个 6.3-53.6N 的画框，纵向多出近 280 km 的
    空白。英文版看不出来，因为那片区域什么都不画；一旦补上九段线，大陆就被压扁到
    画面上半部，九段线拖在下面，比例很难看。

    17N 的取法：海南岛最南 18.15N，所以 17N 完整保留海南，同时把九段线主体裁出去 ——
    九段线由 add_scs_inset() 的南海小图承担，这正是 CLAUDE.md §4.3 要小图的原因。
    """
    prov = load_map_provinces(root)
    x0, _, x1, y1 = prov.total_bounds
    y0 = to_map_xy([105.0], [south_lat], root)[1][0]
    mx, my = pad * (x1 - x0), pad * (y1 - y0)
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(y0 - my, y1 + my)
    ax.set_aspect("equal")
    return x0, y0, x1, y1


def add_scs_inset(fig, ax_main, draw=None, root=None,
                  rect=None, width: float = 0.105, height: float = 0.150,
                  axes_rect=None):
    """南海小图。

    为什么必需而不是装饰：九段线南端到 3.85N，主图 ylim 从 15N 起，直接被裁掉。
    小图与主图同底图、同业务图层、同色标，只换 xlim/ylim（CLAUDE.md §4.3）。

    *draw* 是重画业务图层的回调，签名 draw(inset_ax)；只画底图时传 None。
    """
    b = map_bounds(root)
    if axes_rect is not None:
        # 多面板时按轴坐标定位更稳：图幅坐标要自己算面板位置，改一次版式就得重算一次。
        inset = ax_main.inset_axes(axes_rect)
    else:
        pos = ax_main.get_position()
        if rect is None:
            rect = [pos.x1 - width - 0.004, pos.y0 + 0.004, width, height]
        inset = fig.add_axes(rect)
    # islands=True：这个尺度上南海岛礁是内容，不是噪点
    draw_china_basemap(inset, root, province_lw=0.15, country_lw=0.55,
                       facecolor="#F7F8F9", islands=True)
    if draw is not None:
        draw(inset)
    inset.set_xlim(b[2].x, b[3].x)
    inset.set_ylim(b[2].y, b[3].y)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("")
    inset.set_xlabel("")
    inset.set_ylabel("")
    for side in inset.spines.values():
        side.set_linewidth(0.4)
        side.set_edgecolor("#888888")
    return inset


# ── 中英混排折行 ─────────────────────────────────────────────────────────────
import unicodedata  # noqa: E402


def _cols(ch: str) -> int:
    """一个字符占几个显示列：东亚全角算 2，其余算 1。"""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def cjk_fill(text: str, width: int, subsequent_indent: str = "") -> str:
    """按显示列数折行，取代 textwrap.fill。

    为什么不能用 textwrap.fill：它按 len() 计数，即按字符数。一个汉字的字面宽度是
    拉丁字母的两倍，所以同样 width=210 的图注，英文排出来 181 mm，中文排出来 272 mm ——
    图直接从 183 mm 撑到 280 mm。这不是排版偏好问题，是图会被排版社拒收。

    另一个 textwrap 处理不了的点：中文没有空格。textwrap 以空白为唯一断点，
    一整段没有空格的中文会被当成一个超长单词，根本不折。这里对 CJK 逐字断行，
    对拉丁文仍然只在空格处断，避免把英文单词和数字劈开。

    width 沿用原来的列数，不必换算 —— 因为这里数的就是列，不是字符。
    """
    out, line, cols = [], "", 0
    token, tok_cols = "", 0

    def flush_token():
        nonlocal line, cols, token, tok_cols
        if not token:
            return
        if cols + tok_cols > width and line:
            out.append(line.rstrip())
            line, cols = subsequent_indent, len(subsequent_indent)
        line += token
        cols += tok_cols
        token, tok_cols = "", 0

    for ch in text:
        if ch == "\n":
            flush_token()
            out.append(line.rstrip())
            line, cols = "", 0
            continue
        w = _cols(ch)
        # CJK 与空白可以在任意位置断开；拉丁字母/数字必须凑成完整 token
        if w == 2 or ch.isspace():
            flush_token()
            if cols + w > width and line and not ch.isspace():
                out.append(line.rstrip())
                line, cols = subsequent_indent, len(subsequent_indent)
            if ch.isspace() and not line:
                continue          # 不要以空格开头一行
            line += ch
            cols += w
        else:
            token += ch
            tok_cols += w
    flush_token()
    if line.strip():
        out.append(line.rstrip())
    return "\n".join(out)


# ── Log-axis ticks under a CJK body font ─────────────────────────────────────
from matplotlib.ticker import LogFormatterSciNotation  # noqa: E402


class CJKLogFormatter(LogFormatterSciNotation):
    r"""Log tick labels that survive a body font with no U+2212.

    matplotlib writes a decade as ``$\mathdefault{10^{-2}}$``. ``\mathdefault`` resolves to the
    BODY font -- SimHei -- and mathtext performs no per-glyph fallback, so the exponent's minus
    sign (U+2212, which SimHei does not carry) renders as an empty box. No rcParam fixes this:
    mathtext.default, a font.family fallback list and axes.unicode_minus were all measured and
    all still produce the box.

    Switching that one label to ``\mathrm`` takes the minus from the math fontset (DejaVu Sans),
    which has it. Labels WITHOUT a minus are returned untouched, so every axis whose decades are
    all >= 1 keeps byte-identical ticks and nothing that already worked changes appearance.
    """

    def __call__(self, x, pos=None):
        label = super().__call__(x, pos)
        if "-" in label and r"\mathdefault" in label:
            return label.replace(r"\mathdefault", r"\mathrm")
        return label


def safe_log_axis(ax, axis: str = "y") -> None:
    """Install :class:`CJKLogFormatter` on a log axis. Call right after ``set_?scale("log")``."""
    for name in (("x", "y") if axis == "both" else (axis,)):
        target = getattr(ax, f"{name}axis")
        target.set_major_formatter(CJKLogFormatter())


# ── Helper functions ─────────────────────────────────────────────────────────

# 面板标号：加粗小写字母，**不加括号**（CLAUDE.md §3.2）。
#
# 字体在这里显式指定 Arial，是 §3.1"单字体 SimHei、不配 fallback"的唯一例外，
# 理由是可验证的：SimHei 只有一个字重，fontweight="bold" 落到 SimHei 上会被静默
# 忽略，标号根本粗不起来；而面板标号永远只是一个拉丁字母，不含中文，
# 指定 Arial 不会引入任何缺字风险。其余所有 artist 仍走 SimHei。
_LABEL_FONT = {"fontname": "Arial", "fontweight": "bold"}


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.08,
                fontsize: float = 10):
    """Add a bold panel label (a, b, c — no parentheses) outside the axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fontsize, va="top", ha="left", **_LABEL_FONT)


def panel_label_inside(ax, letter: str, x: float = 0.03, y: float = 0.97,
                       fontsize: float = 10):
    """Add panel label inside the plot area (for maps/heatmaps)."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fontsize, va="top", ha="left", **_LABEL_FONT,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))


def save_fig(fig, name: str, subdir: str = ""):
    """Save figure as PDF + PNG with publication naming.
    
    Args:
        fig: matplotlib figure object
        name: figure filename (without extension)
        subdir: subdirectory under figures/ ("main", "extended", "patent", or "" for root)
    """
    if subdir == "main":
        out_dir = MAIN_FIGURES_DIR
    elif subdir == "extended":
        out_dir = EXTENDED_FIGURES_DIR
    elif subdir == "patent":
        out_dir = PATENT_FIGURES_DIR
    else:
        out_dir = FIGURES_DIR
    
    out_dir.mkdir(parents=True, exist_ok=True)
    # WIDTH GUARD, SELF-CORRECTING, WITH A CULPRIT WHEN IT CANNOT CORRECT. bbox_inches='tight'
    # does not clip -- it GROWS the canvas to include any artist drawn outside the axes -- and
    # `pad_inches` is added on top of that. A figure laid out at exactly the 183 mm double
    # column therefore SAVES at 184 mm purely because of the 0.02 in pad on each side, and the
    # publisher's downscale then puts every 5 pt label under the 5 pt floor. Drop the pad first;
    # only if the drawing itself is too wide is there a real problem, and then name the artist.
    _tight = fig.get_tightbbox(fig.canvas.get_renderer())
    _mm = _tight.width * 25.4
    if _mm > 183.0:
        _worst = []
        for _ax in fig.axes:
            for _art in [_ax.title, _ax.xaxis.label, _ax.yaxis.label, *_ax.texts,
                         *([_ax.get_legend()] if _ax.get_legend() else [])]:
                try:
                    _b = _art.get_window_extent(fig.canvas.get_renderer())
                except Exception:
                    continue
                _over = max(_tight.x1 * fig.dpi - _b.x1, _b.x0 - _tight.x0 * fig.dpi)
                _worst.append((_b.width / fig.dpi * 25.4, str(getattr(_art, 'get_text',
                               lambda: type(_art).__name__)())[:56].replace(chr(10), ' / ')))
        for _art in fig.texts:
            try:
                _b = _art.get_window_extent(fig.canvas.get_renderer())
                _worst.append((_b.width / fig.dpi * 25.4,
                               'fig.text: ' + _art.get_text()[:48].replace(chr(10), ' / ')))
            except Exception:
                pass
        _worst.sort(reverse=True)
        print(f"  [WIDTH] {name}: {_mm:.1f} mm > 183 mm. Widest artists:")
        for _w, _t in _worst[:4]:
            print(f"           {_w:6.1f} mm  {_t}")
    _pad = 0.02 if _tight.width * 25.4 + 1.02 <= 183.0 else 0.0
    # MISSING-GLYPH GUARD (CLAUDE.md 4.6.1). The font stack is SimHei alone, deliberately, and
    # SimHei has no U+2212 and no superscript two/three. A character it lacks does not fail
    # loudly -- matplotlib draws a hollow box and emits a UserWarning -- and every renderer in
    # this repo runs under `-W ignore`, so the box would ship. Catch it here, where the file is
    # written, and name the character instead of leaving someone to find the box in a proof.
    #
    # Filtered on the glyph warnings SPECIFICALLY rather than escalating UserWarning wholesale:
    # savefig also raises unrelated UserWarnings (tight-layout, FixedFormatter), and turning
    # those into hard failures would lose whole figures over cosmetics. The rule is "no boxes
    # in the PDF", not "no warnings".
    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always", UserWarning)
        fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight", pad_inches=_pad)
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=_pad)
    _glyphs = sorted({str(w.message) for w in _caught
                      if "missing from" in str(w.message) and "font" in str(w.message)})
    if _glyphs:
        # DELETE WHAT WAS JUST WRITTEN. savefig has already run by the time the warning is in
        # hand, so a box-containing PDF is on disk; render_version.tree_figures() picks up every
        # PDF/PNG written during the run (every one it finds under --skip-plots) regardless of the
        # script's exit code, so leaving them would ship the very file this guard exists to stop.
        for _ext in (".pdf", ".png"):
            (out_dir / f"{name}{_ext}").unlink(missing_ok=True)
        raise RuntimeError(
            f"{name}: 字体缺字，图里会出现方框，已拒绝出图。\n  "
            + "\n  ".join(_glyphs[:8])
            + "\n改写法绕开：上下标与单位走 mathtext（r\"$10^8$ m$^3$\"、"
              "r\"Mt CO$_2$ yr$^{-1}$\"），负号靠 axes.unicode_minus=False 或改写成"
              "\"减去\"。不要为一个字符加 fallback 字体栈——那会让同一个符号在有无 SimHei 的"
              "机器上落到不同字形。")
    plt.close(fig)
    print(f"  [ok] {name} -> {subdir or 'root'}/")


# ── Model-consistent emission accounting ───────────────────────────────────────
# Residual emissions must reproduce the solver's accounting exactly:
# per-pathway generation (retrofit CF boost), rebuilt-plant efficiency ratio,
# chosen blend levels, and efficiency-penalty fuel emissions. Using stylized
# per-pathway factors (e.g. ccs -> 0.10) contradicts the model and is forbidden
# in figure scripts. All parameters below are the scenario defaults shared by
# every scenario in scenarios/st.toml (none of them override these).
def _plot_accounting_params():
    from coal_retrofit.optimization.scenario import OptimizationAssumptions, OptimizationScenario

    return OptimizationAssumptions(), OptimizationScenario(experiment_id="plot", description="plot")


def residual_emissions_mt(plant_detail_year: "pd.DataFrame", year: int) -> float:
    """Model-consistent residual emissions (Mt) for one scenario-year.

    plant_detail_year: rows of results/<scenario>/plant_detail.csv for `year`. The table must
    carry the `*_blend_ratio` columns (written since PR #7, 2026-09-26); without them this raises
    ValueError instead of guessing the blend ratios from the level index. An empty table still
    returns NaN.
    Mirrors the residual expression in optimization/constraints.py.
    """
    d = plant_detail_year
    if d.empty:
        return float("nan")
    _ratio_cols = ("biomass_blend_ratio", "beccs_blend_ratio", "ammonia_blend_ratio")
    missing = [c for c in _ratio_cols if c not in d.columns]
    if missing:
        raise ValueError(
            f"plant_detail 缺少 {', '.join(missing)}：残余排放只按有效掺烧比例（*_blend_ratio）算，"
            "档位下标分不出独热档位与连续 hub（下标恰为整数也可能是几档的混合）。"
            "PR #7（2026-09-26）之前落盘的 ST_ 结果重解后再画（README §0）；"
            "v9 / v9.1 的独热结果只能在历史里画，当前版本不再维护复现步骤（README §0）。"
            "IND_ 系旧结果也没有这几列，残余排放不要去 cf073be 算：那里的同名函数把非整数下标原样当比例"
            "（CLAUDE.md §1.5 说的在 cf073be 的副本里重画，只指 IND_ 的图；见 README §0）。"
        )
    assumptions, scenario = _plot_accounting_params()

    eta = float(scenario.capture_rate)
    boost = float(scenario.retrofit_cf_boost)
    eff_ratio = np.where(
        year >= d["retirement_year"].to_numpy(dtype=float),
        assumptions.coal_plant_base_efficiency / max(float(scenario.rebuild_efficiency), 1e-9),
        1.0,
    )
    e_op = d["baseline_emissions_mt"].to_numpy(dtype=float) * eff_ratio
    e_rt = e_op * boost
    gen = d["annual_generation_mwh"].to_numpy(dtype=float)
    hr_eff = assumptions.heat_rate_gj_per_mwh * eff_ratio
    ef_t_per_gj = assumptions.coal_emission_factor_t_per_mwh / assumptions.heat_rate_gj_per_mwh
    eta_coal = assumptions.coal_plant_base_efficiency
    # Extra-fuel ratio is year-dependent; the penalty CO2 is captured at `capture_rate`
    # like the rest of the flue gas, so only the uncaptured share is vented.
    pen_ratio = assumptions.ccs_energy_penalty_ratio(int(year))
    pen_ccs_mt_per_mwh = pen_ratio * hr_eff * ef_t_per_gj * (1.0 - eta) / 1e6
    pen_bio_coeff = (assumptions.biomass_efficiency_penalty_per_ratio / eta_coal) * hr_eff * ef_t_per_gj / 1e6

    # Blend ratios: the effective ratio per pathway, sum_l beta_l * z_l / share -- the quantity the
    # constraints use (biomass and BECCS differ). The level index sum_l l * select is no substitute:
    # under continuous hubs it is a weighted index, and even an integer value may be a mix of levels
    # (half level 1, half level 3 reads as level 2) that no column tells apart from a one-hot run.
    # Tables without the ratio columns were refused above.
    beta_bio = d["biomass_blend_ratio"].to_numpy(dtype=float)
    beta_beccs = d["beccs_blend_ratio"].to_numpy(dtype=float)
    beta_a = d["ammonia_blend_ratio"].to_numpy(dtype=float)

    s_un = d["share_unabated"].to_numpy(dtype=float)
    s_ccs = d["share_ccs"].to_numpy(dtype=float)
    s_bio = d["share_biomass"].to_numpy(dtype=float)
    s_beccs = d["share_beccs"].to_numpy(dtype=float)
    s_amm = d["share_ammonia"].to_numpy(dtype=float)

    # DRY-COOLING BACKPRESSURE. `constraints.py` adds
    #     air_penalty_emissions_matrix[p,k] * air_share[p,k]
    # to every plant's residual: converting a condenser from wet to dry costs
    # `air_retrofit_efficiency_penalty_pp` of thermal efficiency, and that coal is burnt and
    # vented like any other. This function omitted the term, which INVERTED THE SIGN of the
    # study's own water-carbon coupling: at 2030 it reported BASE 5378.9 / accounted 5367.8 /
    # reserved 5336.8 Mt -- a 42 Mt "saving" from converting 411 GW to dry cooling. Restoring
    # the term gives 5392.19 Mt for ALL THREE runs, against a baseline of 5392.20. Emissions
    # do not fall at all. What actually happens is that the non-negative-reduction constraint
    # (the v9 model's emission target, 0.0 at 2030; since replaced by `model_year._add_sector_targets`)
    # binds: dry cooling pushes emissions above baseline,
    # and the model buys just enough biomass co-firing to push them back -- 0.27% of generation
    # at 89.5 GW converted, 1.15% at 411.1 GW. The omitted term was counting that compensating
    # biomass as a saving while hiding the emissions it compensates for.
    #
    # `air_share` is a per-(plant, pathway) variable; `plant_detail.csv` reports only the
    # plant aggregate, so the reconstruction below assumes it is uniform across a plant's
    # active pathways. That assumption is exact at 2030 (identity above holds to 0.01 Mt) and
    # is the tightest available from the written outputs.
    # THE 24-COLUMN VINTAGE HAS NO COOLING COLUMNS, AND FOR IT THE TERM IS ZERO BY
    # CONSTRUCTION -- that build has no wet-to-dry conversion variable at all, so no unit can
    # incur a backpressure penalty. Falling back to zero is therefore correct rather than
    # merely convenient, but it is done explicitly: a silent 0.0 would let a 24-column run be
    # differenced against a 26-column one with no sign that the two describe different feasible
    # sets -- exactly the cross-version differencing CLAUDE.md 二.6 forbids.
    _air_cols = ("already_air_share", "air_cooled_share")
    if all(c in d.columns for c in _air_cols):
        still_wet = 1.0 - d["already_air_share"].to_numpy(dtype=float)
        air = d["air_cooled_share"].to_numpy(dtype=float)
    else:
        still_wet = np.zeros(len(d), dtype=float)
        air = np.zeros(len(d), dtype=float)
    s_retire = d["share_retire"].to_numpy(dtype=float)
    pen_air_ratio = (
        float(assumptions.air_retrofit_efficiency_penalty_pp)
        / max(1e-9, float(assumptions.coal_plant_base_efficiency))
    )
    air_penalty_mt = (
        gen * assumptions.coal_emission_factor_t_per_mwh / 1e6
        * pen_air_ratio * eff_ratio * still_wet * air * (1.0 - s_retire)
    )

    residual = (
        e_op * s_un
        + e_rt * (1.0 - eta) * s_ccs
        + e_rt * (1.0 - beta_bio) * s_bio
        + e_rt * (1.0 - eta - beta_beccs) * s_beccs
        + e_rt * (1.0 - beta_a) * s_amm
        + pen_ccs_mt_per_mwh * gen * boost * (s_ccs + s_beccs)
        + pen_bio_coeff * gen * boost * (beta_bio * s_bio + (1.0 - eta) * beta_beccs * s_beccs)
        + air_penalty_mt
    )
    return float(residual.sum())


# ── Level-1 water-resource regions ────────────────────────────────────────────
# Water is a basin quantity: the Yellow River crosses nine provinces and the Hai five, so a
# provincial cut of a water result mixes together catchments that cannot share water. The
# availability budget is built per basin (see `builders.water`), and every water figure
# should aggregate on the same unit the constraint acts on.

# North to south, so bar charts and small multiples read like the map.
BASIN_NAMES_ZH = {
    "A": "松花江与辽河", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
    "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北内陆河",
}
BASIN_ORDER = ["A", "C", "D", "E", "K", "F", "H", "G", "J"]


def load_basins(root=None):
    """Level-1 basin polygons in EPSG:2380, or None when the layer is absent."""
    import geopandas as gpd
    from pathlib import Path

    root = Path(root) if root is not None else ROOT
    path = root / "data" / "ChinaBasins" / "basin_l1.gpkg"
    if not path.exists():
        return None
    return gpd.read_file(path).to_crs("EPSG:2380")


def assign_basin(frame, lon_col: str = "centroid_longitude", lat_col: str = "centroid_latitude",
                 root=None):
    """Level-1 basin code for each row of a table carrying coordinates.

    Nearest polygon, not containment: coastal hub centroids fall a few kilometres offshore
    and would otherwise be dropped. The largest reassignment distance across the 350 hubs is
    11.6 km, so nothing is being pulled across a divide.
    """
    import geopandas as gpd

    basins = load_basins(root)
    if basins is None:
        return None
    points = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame[lon_col], frame[lat_col]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2380")
    joined = gpd.sjoin_nearest(points, basins[["code", "geometry"]], how="left")
    joined = joined[~joined.index.duplicated()]
    return joined["code"].astype(str)
