"""全仓库唯一一套绘图样式（CLAUDE.md §三）：字体与版式、配色、中英对照、常用画法与存图。

出图脚本 `scripts/plot_fig*.py` 都从这里取字体与版式（`apply_style`）、配色（§3.3）、中英对照（`labels`）与存图
（`save_fig`：字体、字号、图宽、缺字四道检查）；中国底图（EPSG:2380、九段线、南海小图）在 `plot_map`，命令行与读求解结果
在 `plot_io`。各图脚本不自设字体、不自带色表。

    from plot_io import figure_cli, langs
    from plot_style import apply_style, save_fig
    args = figure_cli(__doc__).parse_args()
    for lang in langs(args.lang):
        apply_style(lang)
        save_fig(draw(data, lang), "fig3_power_pathways", lang)
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

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

# 图写到 `_indtree/results/figures/`，不入库（结果图读的结果目录见 `plot_io.SOLVED_DIR`）。
FIGURES_DIR = ROOT / "results" / "figures"

# ── 版式（CLAUDE.md §3.2）────────────────────────────────────────────────────────
MM = 1 / 25.4                # mm -> inch
FULL_WIDTH_MM = 183.0        # 双栏宽，save_fig 的图宽上限
MIN_FONT_PT = 5.0            # 排版社缩放后仍可读的下限，save_fig 检查
# mathtext 的上下标按 0.7 缩放（matplotlib `_mathtext.SHRINK_FACTOR`，没有 rcParam）：6 pt 图例条目里 CO$_2$ 的下标只有
# 4.2 pt。带上下标的文字（单位、幂次）只放正文字号及以上的位置——坐标轴标签、图例标题、注记；图例条目与刻度不行。
MIN_SCRIPT_HOST_PT = 7.0

# ── 字体：中文 SimHei、英文 Arial，各自单字体、不配 fallback（CLAUDE.md §3.1）──────────
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
        "legend.title_fontsize": 7,     # 图例标题写单位（含 mathtext 上下标），不低于 MIN_SCRIPT_HOST_PT
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
# CO2 管网与封存用 Blues（§3.3），由深到浅：图 2 的四类候选边按序取，图 6 在役而当年无流量的管道取浅端。
BLUES = ("#08519C", "#3182BD", "#6BAED6", "#9ECAE1")
SINK_COLORS = {"dsa": "#3182BD", "eor": "#9ECAE1"}   # 深部咸水层、驱油封存
SINK_EDGE = "#08519C"                                # 封存汇方块的描边（§4.4）
PIPE_COLOR = "#3182BD"                               # CO2 管网与 DSA 同色（§3.3 Blues 深端）
IDLE_PIPE_COLOR = "#9ECAE1"                          # 在役而当年无流量的管道（图 6）
AIR_COLOR = "#CC3311"                                # 空冷改造
OVER_COLOR = "#A50F15"                               # 超上限、超指标（Reds 深端，与空冷的强调红分开）
NEUTRAL = "#B0B0B0"                                  # 尺寸图例的点：只示大小，不指类别
BASELINE_COLOR = "#D9D9D9"                           # 冻结技术排放（图 5）
# 成本构成（图 10）：煤电 Greys、燃料与资源 Greens、工业 Reds、管网与封存 Blues，同族同色。只在含义相同处与别的色表同值
# （煤电改造 = 部门色的煤电、管网 = PIPE_COLOR、松弛惩罚 = 超上限色）；其余取同族里别处没用的一档，免得与路径色、部门色
# 撞值（2026-10-13 前煤电运行 = 未改造路径色、资源 = 生物质路径色、工业 = 长流程钢与电炉钢的部门色、封存 = EOR 汇色、
# 碳成本 = 合成氨部门色、残值 = 退役路径色）。碳成本与残值不属于哪一族，各取一个别处没用的色。
COST_ORDER = ("coal_operating", "coal_capex", "resources", "industry_operating", "industry_capex", "network",
              "storage", "carbon", "slack", "salvage")
COST_COLORS = {
    "coal_operating": "#BDBDBD", "coal_capex": "#636363",          # Greys 浅、深
    "resources": "#A1D99B",                                        # Greens 浅
    "industry_operating": "#FCBBA1", "industry_capex": "#FB6A4A",  # Reds 浅、中
    "network": PIPE_COLOR, "storage": "#C6DBEF",                   # Blues 深、最浅
    "carbon": "#E6AB02", "slack": OVER_COLOR, "salvage": "#E5D8BD",
}
# 标记：煤电圆、工业菱形、封存汇方块，各图一致。
MARKERS = {"coal": "o", "industry": "D", "sink": "s"}
# 散点的 s 对圆点是外接正方形的面积（圆面积 π/4·s），对方块、菱形就是图形面积：乘 π/4 才与同值的圆点等面积。
MARKER_AREA = {"o": 1.0, "D": np.pi / 4, "s": np.pi / 4}

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


# ── 常用画法 ─────────────────────────────────────────────────────────────────────
def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    """面板标号：加粗小写字母，不加括号，放在面板左上角外侧（§3.2）。"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            color="black", **_LABEL_FONT)


def figure_label(fig, letter: str, x: float, y: float) -> None:
    """面板标号放在图幅坐标上：地图这类四周没有轴外空白的面板用它。"""
    fig.text(x, y, letter, fontsize=8, va="top", ha="left", color="black", **_LABEL_FONT)


def stacked_bars(ax, x, table: pd.DataFrame, colors: dict[str, str], *, width: float = 0.62,
                 hatches: dict[str, str] | None = None) -> None:
    """按列堆叠的柱：正值向上、负值向下各自累加，负值不会把上面的段拉下来。

    *table* 的行对应 *x*，列按堆叠顺序（自下而上）；*hatches* 给某些列加深色斜线（如海上汇，图 9）：斜线另画一层
    无填充、无描边的柱，线色取 INK（斜线色跟描边色走，白色斜线在浅色柱上看不清）。
    """
    pos = np.zeros(len(table))
    neg = np.zeros(len(table))
    for key in table.columns:
        values = table[key].to_numpy(dtype=float)
        bottom = np.where(values >= 0, pos, neg)
        ax.bar(x, values, width=width, bottom=bottom, color=colors[key], edgecolor="white", linewidth=0.4, zorder=3)
        hatch = (hatches or {}).get(key)
        if hatch:
            ax.bar(x, values, width=width, bottom=bottom, facecolor="none", edgecolor=INK, linewidth=0.0,
                   hatch=hatch, zorder=3.1)
        pos = pos + np.clip(values, 0, None)
        neg = neg + np.clip(values, None, 0)
    if (neg < 0).any():
        ax.axhline(0.0, color=INK, linewidth=0.5, zorder=4)


def fmt_number(value: float) -> str:
    """柱顶、格子里的数字：绝对值 ≥ 10 取整，否则一位小数，不到 0.05 记 0。"""
    if abs(value) < 0.05:
        return "0"
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}"


def legend_patches(keys, colors: dict[str, str], names: dict[str, str],
                   hatches: dict[str, str] | None = None) -> list[Patch]:
    return [Patch(facecolor=colors[k], edgecolor=INK if (hatches or {}).get(k) else "none", linewidth=0.0,
                  hatch=(hatches or {}).get(k), label=names[k]) for k in keys]


def area_scale(values, vmax: float, *, smin: float = 1.5, smax: float = 42.0, marker: str = "o") -> np.ndarray:
    """散点的 s（pt²）= 下限 *smin* + 与数值成正比的一项：下限让小点看得见，所以面积随数值线性增大、不成正比。
    按 `MARKER_AREA` 换算形状，同值的圆点、菱形、方块面积相同。各图的尺寸图例用同一函数。"""
    v = np.clip(np.asarray(values, dtype=float) / max(float(vmax), 1e-12), 0.0, 1.0)
    return (smin + (smax - smin) * v) * MARKER_AREA[marker]


def size_legend(values, vmax: float, marker: str, color: str, fmt: str = "{:g}",
                smax: float = 42.0) -> list[Line2D]:
    """尺寸图例：按代表值另建句柄，不直接拿数据点当图例（§4.4）；*smax* 与画点时一致。"""
    sizes = area_scale(values, vmax, smax=smax, marker=marker)
    return [Line2D([], [], marker=marker, linestyle="none", markersize=float(np.sqrt(s)),
                   markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.3,
                   label=fmt.format(v)) for v, s in zip(values, sizes)]


# ── 存图 ─────────────────────────────────────────────────────────────────────────
def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _check_fonts(fig, stem: str) -> None:
    """单字体规则（§3.1）：图里文字用的字体本机没有时，matplotlib 会静默换成 DejaVu Sans 照常出图，这里报错。"""
    from matplotlib import font_manager

    families = {family for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
                for family in t.get_fontproperties().get_family()}
    missing = []
    for family in sorted(families):
        try:
            font_manager.findfont(font_manager.FontProperties(family=[family]), fallback_to_default=False)
        except ValueError:
            missing.append(family)
    if missing:
        raise RuntimeError(f"{stem}：本机没有字体 {missing}，matplotlib 会静默换成别的字体照常出图，已拒绝出图"
                           "（中文 SimHei、英文与面板标号 Arial）")


def _has_script(text: str) -> bool:
    """mathtext 段（成对的未转义 `$...$` 之间）里有上下标；`$` 不成对时 matplotlib 不按 mathtext 处理。"""
    parts = re.split(r"(?<!\\)\$", text)
    if len(parts) < 3 or len(parts) % 2 == 0:
        return False
    return any("_" in part or "^" in part for part in parts[1::2])


def _check_font_sizes(fig, stem: str) -> None:
    """字号检查：所有文字不小于 MIN_FONT_PT；带上下标的文字不小于 MIN_SCRIPT_HOST_PT（上下标按 0.7 缩放）。"""
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    small = sorted({(round(float(t.get_fontsize()), 2), t.get_text()[:30]) for t in texts if t.get_fontsize() < MIN_FONT_PT})
    if small:
        raise RuntimeError(f"{stem}：有文字小于 {MIN_FONT_PT:g} pt，排版社缩放后不可读：{small[:5]}")
    scripted = sorted({(round(float(t.get_fontsize()), 2), t.get_text()[:30])
                       for t in texts if t.get_fontsize() < MIN_SCRIPT_HOST_PT and _has_script(t.get_text())})
    if scripted:
        raise RuntimeError(f"{stem}：带上下标的文字小于 {MIN_SCRIPT_HOST_PT:g} pt（上下标按 0.7 缩放后不到 "
                           f"{MIN_SCRIPT_HOST_PT * 0.7:.1f} pt，不可读）："
                           f"{scripted[:5]}")


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

    先查字体（本机没有就报错，不静默换字体）、字号与图宽，再写临时文件并收集缺字警告：缺字（SimHei / Arial
    没有的字符会画成方框）就删掉临时文件、报错，已有的同名图不动；都通过才换上正式文件名。
    """
    out_dir = FIGURES_DIR if out_dir is None else Path(out_dir)
    stem = name if lang == "zh" else f"{name}_en"
    _check_fonts(fig, stem)
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
