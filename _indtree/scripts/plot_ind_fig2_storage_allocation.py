# -*- coding: utf-8 -*-
"""ind_fig2 — 封存空间归谁：煤电占满全国注入能力，工业只能转氢。

ind_fig1 给的是"谁减了多少"。这张图回答紧接着的那个问题：**工业为什么一吨都不捕集。**
求解结果里工业捕集 0.0 Mt，读起来像是模型没把工业接进管网；实际接进去了（390 条支线
全部建成、约束全部在位），是**没有空间可分**。三个面板把这件事从三个独立角度证到底：

    a  2060 年的管网、封存汇与两类源      —— 汇被谁占着（空间格局）
    b  89 个汇的注入能力利用率排序        —— 80 个正好压在 100%
    c  全国注入能力 vs 两侧的需求          —— 工业若上 CCS 需要 2.3 倍的全国能力

面板 c 是本图的落点，也是唯一不依赖求解的一格：工业基线 3 269 Mt、捕集率 0.90，
要 2 942 Mt/yr 的注入能力，而全国只有 1 283 Mt/yr。**即使把煤电整个拿走，工业也装不下。**
所以"工业不上 CCS"是结构性结论，不是求解噪声，也不是这一版管网的缺陷。

数据源固定 `_indtree`（v9 管网，89 个汇全部可达）。仓库根 `results/IND_*` 用 v7 管网，
够不着 38% 的封存能力，其注入分布不可入图（见 `_indtree/README.md`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from plot_style import (  # noqa: E402
    MM,
    RESULTS_DIR,
    ROOT,
    add_scs_inset,
    apply_style,
    cjk_fill,
    draw_china_basemap,
    mainland_extent,
    panel_label,
    save_fig,
    to_map_xy,
)
from coal_retrofit.constants_industry import SECTOR_LABELS_ZH  # noqa: E402

INPUTS = ROOT / "inputs"
RUN = "IND_WA_cwatm_126_dry_oq_t95"
YEAR = 2060
CAPTURE_RATE = 0.90            # scenario.OptimizationScenario.capture_rate 的默认值

# --- 配色（CLAUDE.md §3.3）---------------------------------------------------------------
PIPE_C = "#3182BD"             # CO2 管网：Blues 深端
SINK_FULL = "#CC3311"          # 注入能力压满的汇：强调红
SINK_ROOM = "#123F6B"          # 尚有余量的汇：深蓝描边
COAL_ON = "#636363"            # 有捕集的煤电厂址（Greys 深端 = 带 CCS）
COAL_OFF = "#B9C0C6"           # 无捕集的煤电厂址。比 Greys 浅端再淡一档即可，
                               # 但不能淡到 #D5DADE：底图是 #F4F6F7，实测那一档在
                               # 300 dpi 下几乎看不见，209 个点会被读成"没有画"。
ROUTE_C = {"unabated": "#969696", "ccs": "#636363", "h2": "#9E9AC8"}
ROUTE_ZH = {"unabated": "不改造", "ccs": "CCS 捕集", "h2": "绿氢替代"}
DEMAND_C = "#00A087"           # 工业侧需求：与 ind_fig1 的工业色一致
GRID = "#E4E7EA"

EXPECTED_SINKS = 89            # v9 管网的汇数，用来把求解树认出来
FLOW_REF = 40.0                # Mt/yr，管段线宽标度（上限即单管容量）
SINK_REF = 200.0               # Mt/yr，汇点面积标度（= 单个汇的注入能力上限）
SINK_AREA_MIN, SINK_AREA_MAX = 2.0, 34.0   # pt^2，汇点面积区间
ACTIVE_MTPA = 0.01             # Mt/yr，低于此视为未启用
ACTIVE_FRAC = 0.001            # 无量纲，利用率低于此视为未启用（与上一条不是同一个量）


# =========================================================================================
# 读数
# =========================================================================================
def _read_input(name: str) -> pd.DataFrame:
    """读一个输入 CSV。这些文件带 UTF-8 BOM，不指定 utf-8-sig 首列名会变成 '﻿xxx'。"""
    return pd.read_csv(INPUTS / name, encoding="utf-8-sig")


def assert_v9_tree(run: str) -> int:
    """确认脚本正跑在 v9 求解树上，而不是仓库根的 v7 结果上。

    两棵树里都有同名目录 `IND_WA_cwatm_126_dry_oq_t95`，`RESULTS_DIR` 只跟着脚本自己的
    位置走（`plot_style.py`：`__file__.parent.parent / "results"`）。在仓库根跑会画出
    35 汇的 v7 结果，而图注里"89 个汇全部可达"是算出来的、"v9 管网"却是写死的字符串——
    图看上去完全正常。这一条就是防那个。
    """
    sinks_here = pd.read_csv(RESULTS_DIR / run / "storage_utilization.csv")[
        "storage_hub_id"].nunique()
    if sinks_here != EXPECTED_SINKS:
        raise RuntimeError(
            f"{RESULTS_DIR} 不是 v9 求解树：{run} 只有 {sinks_here} 个汇"
            f"（应为 {EXPECTED_SINKS}）。请在 _indtree/ 下运行本脚本。")
    return int(sinks_here)


def node_xy() -> dict[str, tuple[float, float]]:
    """666 个基础节点的 EPSG:2380 坐标，键为 node_id。"""
    nodes = _read_input("pipeline_nodes.csv")
    x, y = to_map_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    return dict(zip(nodes["node_id"].astype(str), zip(x, y)))


def edge_geometry() -> dict[str, np.ndarray]:
    """候选边的数字化中心线（EPSG:2380）。运行时新增的支线不在这个文件里。

    走真实路由而不是端点直线：管长本来就按 haversine x 1.136 的绕行系数算，
    画成直线会与长度口径对不上（CLAUDE.md §1.3）。够不到几何的边退回端点直线，
    并在终端报出条数——静默退回会让"这版管网没有走廊"这种问题看不见。
    """
    from shapely import wkt
    from shapely.errors import GEOSException, ShapelyError

    cand = _read_input("pipeline_candidate_edges.csv")
    out: dict[str, np.ndarray] = {}
    for eid, geom in zip(cand["edge_id"].astype(str), cand["geometry_wkt"]):
        if not isinstance(geom, str):
            continue
        try:
            coords = np.asarray(wkt.loads(geom).coords)
        except (GEOSException, ShapelyError, ValueError):
            # 一行坏几何只丢一条边（会退回端点直线并被计数），不该拖垮整张图。
            # 不用裸 Exception：那会把 MemoryError 一起吞掉。
            continue
        x, y = to_map_xy(coords[:, 0], coords[:, 1])
        out[eid] = np.column_stack([x, y])
    return out


def active_edges(run: str, year: int) -> pd.DataFrame:
    edges = pd.read_csv(RESULTS_DIR / run / "network_edges.csv")
    edges = edges[(edges["year"] == year) & (edges["edge_active"].astype(bool))]
    return edges[edges["edge_flow_mtpa"] > ACTIVE_MTPA].copy()


def sinks(run: str, year: int) -> pd.DataFrame:
    """封存汇：坐标 + 本年注入量与注入能力利用率。

    这里不能 `fillna`：`injectivity_mtpa` 的合计是面板 c "2.3 倍"那句话的**分母**，
    而 `Series.sum()` 会跳过 NaN——少一个汇就等于把它的能力当成 0，分母变小、倍数变大，
    图仍然画得出来。所以对齐之后先断言全都对上了，再往下走。
    """
    hubs = _read_input("storage_hubs.csv")
    use = pd.read_csv(RESULTS_DIR / run / "storage_utilization.csv")
    use = use[use["year"] == year]
    use = use.set_index(use["storage_hub_id"].astype(str))
    hubs = hubs.set_index(hubs["storage_hub_id"].astype(str))
    aligned = use.reindex(hubs.index)
    missing = list(aligned.index[aligned["injectivity_mtpa"].isna()])
    if missing:
        raise RuntimeError(
            f"{run} 的 storage_utilization.csv 在 {year} 年缺 {len(missing)} 个汇"
            f"（{missing[:5]}…），全国注入能力会被少算。")
    hubs["use_mtpa"] = aligned["storage_use_mtpa"]
    hubs["util"] = aligned["injectivity_utilization"]
    hubs["injectivity_mtpa"] = aligned["injectivity_mtpa"]
    return hubs.reset_index(drop=True)


def coal_hubs(run: str, year: int) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_DIR / run / "plant_detail.csv")
    return frame[frame["year"] == year].copy()


def industry_hubs(run: str, year: int) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_DIR / run / "industry_detail.csv")
    frame = frame[frame["year"] == year].copy()
    shares = frame[["share_unabated", "share_ccs", "share_h2"]].to_numpy()
    frame["route"] = np.array(["unabated", "ccs", "h2"])[shares.argmax(axis=1)]
    return frame


# =========================================================================================
# 面板 a：地图
# =========================================================================================
def _sink_area(values: np.ndarray) -> np.ndarray:
    """汇点的**面积**（pt²）正比注入量，图注里也是这么写的。

    曾经写成 `s = A + B*sqrt(v/REF)`。`scatter(s=)` 收的是面积，所以那样写等于
    面积正比 sqrt(v)、半径正比 v^0.25 —— 大汇被开四次方压扁，正是
    `plot_ed_source_atlas._area` 的注释点名的那种误导。这里保持线性。
    """
    return SINK_AREA_MIN + (SINK_AREA_MAX - SINK_AREA_MIN) * np.clip(
        np.asarray(values, dtype=float) / SINK_REF, 0.0, 1.0)


def _legend_ms(area_pt2: float) -> float:
    """把面积（pt²，`scatter` 的 s）换成 `Line2D` 的 markersize（直径，pt）。

    两者不是同一个量纲。直接把 s 填进 markersize，图例里的点会比图上的大好几倍，
    而且相对大小关系也不对。
    """
    return float(np.sqrt(max(area_pt2, 0.0)))


def _business(ax: plt.Axes, edges: pd.DataFrame, geom: dict[str, np.ndarray],
              xy: dict[str, tuple[float, float]], sink: pd.DataFrame, coal: pd.DataFrame,
              ind: pd.DataFrame, scale: float = 1.0) -> None:
    """业务图层。主图与南海小图各调一次，保证两处同色标、同尺寸律（CLAUDE.md §4.3）。"""
    segs, widths = [], []
    for row in edges.itertuples():
        line = geom.get(str(row.edge_id))
        if line is None:
            a, b = xy.get(str(row.from_node_id)), xy.get(str(row.to_node_id))
            if a is None or b is None:
                continue
            line = np.array([a, b])
        segs.append(line)
        widths.append((0.28 + 1.30 * np.sqrt(min(row.edge_flow_mtpa / FLOW_REF, 1.0))) * scale)
    if segs:
        ax.add_collection(LineCollection(segs, colors=PIPE_C, linewidths=widths,
                                         capstyle="round", alpha=0.80, zorder=3))

    # 煤电两级：不捕集的在下、更淡。350 个厂址里多数不捕集，画上去才看得出管网只连了一部分。
    on = coal[coal["captured_mt"] > 0.01]
    off = coal[coal["captured_mt"] <= 0.01]
    for frame, colour, size, z in ((off, COAL_OFF, 1.0, 1.6), (on, COAL_ON, 2.2, 4.0)):
        if frame.empty:
            continue
        px, py = to_map_xy(frame["centroid_longitude"].to_numpy(),
                           frame["centroid_latitude"].to_numpy())
        ax.scatter(px, py, s=size * scale, color=colour, linewidth=0, zorder=z)

    # 工业：五边形（昊天 Fig 6B 的记号），按占优通路着色。面积不编码排放——本图问的是
    # "走哪条路"，不是"多大"；排放量的图谱在 ed_source_atlas 里。
    for route in ("unabated", "h2", "ccs"):
        part = ind[ind["route"] == route]
        if part.empty:
            continue
        px, py = to_map_xy(part["longitude"].to_numpy(), part["latitude"].to_numpy())
        ax.scatter(px, py, s=3.4 * scale, color=ROUTE_C[route], marker="p",
                   alpha=0.75, linewidth=0, zorder=3.6)

    # 汇：白心，面积正比注入量；压满注入能力的换红描边。
    used = sink[sink["use_mtpa"] > ACTIVE_MTPA]
    if not used.empty:
        hx, hy = to_map_xy(used["longitude"].to_numpy(), used["latitude"].to_numpy())
        size = _sink_area(used["use_mtpa"].to_numpy()) * scale
        full = used["util"].to_numpy() >= 0.99
        off_shore = used["offshore"].to_numpy().astype(bool)
        for mask, colour in ((full, SINK_FULL), (~full, SINK_ROOM)):
            for marker, shape in (("o", ~off_shore), ("s", off_shore)):
                sel = mask & shape
                if not sel.any():
                    continue
                ax.scatter(hx[sel], hy[sel], s=size[sel], marker=marker, facecolor="white",
                           edgecolor=colour, linewidth=0.45 * scale, zorder=5)


def panel_a(fig: plt.Figure, ax: plt.Axes, edges: pd.DataFrame,
            geom: dict[str, np.ndarray], xy: dict[str, tuple[float, float]],
            sink: pd.DataFrame, coal: pd.DataFrame, ind: pd.DataFrame) -> None:
    draw_china_basemap(ax, province_lw=0.12, country_lw=0.42, facecolor="#F4F6F7")
    _business(ax, edges, geom, xy, sink, coal, ind)
    mainland_extent(ax)
    ax.set_axis_off()
    add_scs_inset(fig, ax, draw=lambda a: _business(a, edges, geom, xy, sink, coal, ind,
                                                    scale=0.55))

    n_full = int((sink["util"] >= 0.99).sum())
    # 图例点的大小一律由 `_legend_ms` 从图上实际用的面积换算，不再手填。手填的那一版
    # 把面积（pt²）当直径（pt）写，图例里每个点都比图上大 2-3 倍，相对大小也不对。
    sink_ms = _legend_ms(_sink_area(np.array([SINK_REF / 2.0]))[0])
    handles = [
        Line2D([], [], color=PIPE_C, lw=1.1, label=f"CO$_2$ 管网（{len(edges)} 段有流量）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=SINK_FULL, markeredgewidth=0.7, markersize=sink_ms,
               label=f"封存汇：注入能力已满（{n_full}）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=SINK_ROOM, markeredgewidth=0.7, markersize=sink_ms,
               label=f"封存汇：尚有余量（{int((sink['use_mtpa'] > ACTIVE_MTPA).sum()) - n_full}）"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COAL_ON,
               markeredgewidth=0, markersize=_legend_ms(2.2),
               label=f"煤电厂址：有捕集（{int((coal['captured_mt'] > 0.01).sum())}）"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COAL_OFF,
               markeredgewidth=0, markersize=_legend_ms(1.0),
               label=f"煤电厂址：无捕集（{int((coal['captured_mt'] <= 0.01).sum())}）"),
    ]
    # 遍历全部三条通路，不是只列出预期会出现的两条：某个厂址若占优通路是 CCS，
    # 地图上会画出一个没有图例的紫灰点，而"工业不捕集"恰恰是本图的结论，
    # 漏画那一条等于把反例藏起来。
    handles += [Line2D([], [], marker="p", color="none", markerfacecolor=ROUTE_C[r],
                       markeredgewidth=0, markersize=_legend_ms(3.4), alpha=0.85,
                       label=f"工业点源：{ROUTE_ZH[r]}（{int((ind['route'] == r).sum())}）")
                for r in ("unabated", "ccs", "h2") if (ind["route"] == r).any()]
    # 左下（青藏那一角）：左上是新疆，源点密集，图例会整片压掉。75% 白底而不是无框，
    # 否则省界会从"（350）"这类文字里穿过去。
    first = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.012),
                      fontsize=5.4, frameon=True, facecolor="white", framealpha=0.78,
                      edgecolor="none", handletextpad=0.42, labelspacing=0.34,
                      borderaxespad=0.0)
    ax.add_artist(first)

    # 线宽与点面积各自的分档钥匙。没有它，读者看得见粗细差别却无法把它换算回 Mt/yr；
    # 图例句柄必须另建（CLAUDE.md §4.4），不能拿数据 artist 当句柄。
    flow_keys = [2.0, 10.0, 40.0]
    width_handles = [Line2D([], [], color=PIPE_C,
                            lw=0.28 + 1.30 * np.sqrt(min(v / FLOW_REF, 1.0)),
                            label=f"{v:g}") for v in flow_keys]
    sink_keys = [20.0, 100.0, 200.0]
    width_handles += [Line2D([], [], marker="o", color="none", markerfacecolor="white",
                             markeredgecolor=SINK_FULL, markeredgewidth=0.5,
                             markersize=_legend_ms(_sink_area(np.array([v]))[0]),
                             label=f"{v:g}") for v in sink_keys]
    # 放左上（新疆以北的图外空白），不能挨着主图例：两者并排时这一格的标题会横穿
    # 主图例的"封存汇：尚有余量（1）"那一行。
    ax.legend(handles=width_handles, loc="upper left", bbox_to_anchor=(0.005, 0.995),
              fontsize=5.2, frameon=True, facecolor="white", framealpha=0.78,
              edgecolor="none", handletextpad=0.45, labelspacing=0.42, borderaxespad=0.0,
              ncol=2, columnspacing=0.9, title_fontsize=5.4,
              title="管段流量 / 汇注入量\n" r"（Mt CO$_2$ yr$^{-1}$）")


# =========================================================================================
# 面板 b：89 个汇的注入能力利用率
# =========================================================================================
def panel_b(ax: plt.Axes, sink: pd.DataFrame) -> None:
    order = sink.sort_values("util", ascending=False).reset_index(drop=True)
    xs = np.arange(len(order))
    colours = np.where(order["util"] >= 0.99, SINK_FULL,
                       np.where(order["util"] > ACTIVE_FRAC, SINK_ROOM, "#C8CDD2"))
    ax.bar(xs, order["util"], width=0.86, color=colours, linewidth=0, zorder=3)
    ax.axhline(1.0, color="#333333", lw=0.5, ls=(0, (3, 2)), zorder=4)

    n_full = int((order["util"] >= 0.99).sum())
    n_idle = int((order["util"] <= ACTIVE_FRAC).sum())
    ax.text(1.0, 1.055, f"{n_full} 个汇正好压在注入能力上限", fontsize=5.4,
            color=SINK_FULL, ha="left", va="bottom")
    # 未启用的汇利用率为 0，柱高为零、画不出来。标签必须自己说明这一段不是缺数据。
    # 用一段贴地的浅灰底衬把这一区间标出来，再把文字放在它正上方：之前那根从柱顶拉到
    # 轴底的斜引线在图上读起来像一条下降的曲线，比它要解释的东西更容易误导。
    if n_idle:
        # 灰底衬标出这一段，计数写进 x 轴标题。写成轴内文字的话，无论居中还是右对齐，
        # 都会顶到图幅右缘被切掉一个字——这一格右边就是整幅的边界，没有让的余地。
        first_idle = len(order) - n_idle
        ax.axvspan(first_idle - 0.5, len(order) - 0.5, ymin=0.0, ymax=0.055,
                   color="#C8CDD2", zorder=2)

    ax.set_xlim(-1.0, len(order))
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50%", "100%"], fontsize=5.8)
    ax.set_xticks([0, len(order) - 1])
    ax.set_xticklabels(["1", f"{len(order)}"], fontsize=5.8)
    ax.set_xlabel(f"{len(order)} 个封存汇（按利用率排序，末 {n_idle} 个未启用，灰底）",
                  fontsize=6.0, labelpad=1.5)
    ax.set_ylabel(f"{YEAR} 年注入能力利用率", fontsize=6.0, labelpad=2.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


# =========================================================================================
# 面板 c：全国注入能力 vs 两侧需求
# =========================================================================================
def panel_c(ax: plt.Axes, sink: pd.DataFrame, ind: pd.DataFrame,
            coal_captured: float) -> float:
    """把"工业不上 CCS"从求解结果变成一条算术：需求 2.3 倍于全国能力。"""
    capacity = float(sink["injectivity_mtpa"].sum())
    ind_need = float(ind["baseline_co2_mt"].sum()) * CAPTURE_RATE

    rows = [
        ("全国注入能力", capacity, "#8A9199"),
        (f"{YEAR} 年煤电已注入", coal_captured, COAL_ON),
        ("工业若全部上 CCS\n所需注入能力", ind_need, DEMAND_C),
    ]
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, [r[1] for r in rows], height=0.56, color=[r[2] for r in rows],
            linewidth=0, zorder=3)
    for y, (_label, value, _c) in zip(ys, rows):
        ax.text(value + capacity * 0.025, y, f"{value:,.0f}", fontsize=5.8,
                va="center", ha="left", color="#333333")

    ax.axvline(capacity, color=SINK_FULL, lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax.text(ind_need * 0.995, ys[-1] + 0.52,
            f"= 全国能力的 {ind_need / capacity:.1f} 倍", fontsize=5.4, color=DEMAND_C,
            ha="right", va="bottom")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.8, linespacing=1.25)
    ax.set_xlim(0, ind_need * 1.16)
    ax.set_xlabel(r"Mt CO$_2$ yr$^{-1}$", fontsize=6.0, labelpad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=5.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    return ind_need / capacity


# =========================================================================================
def _degenerate_sector(ind: pd.DataFrame) -> tuple[str, int, int]:
    """占优通路份额里最像 LP 简并的那个行业：多少个厂址共用同一个 CO2 强度。

    图注里曾经写死"49/60"。写死的数字会在换情景、换输入后静默变成错的，
    而这一句恰恰是在告诉读者"别读这一层空间信息"——它自己更不能过期。
    """
    best = ("", 0, 0)
    for sector, group in ind.groupby("sector"):
        production = group["production_kt_per_year"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            intensity = np.where(production > 0,
                                 group["baseline_co2_mt"].to_numpy() / (production / 1000.0),
                                 np.nan)
        counts = pd.Series(np.round(intensity, 4)).value_counts()
        if len(counts) and int(counts.iloc[0]) > best[1]:
            # 只在这个行业内部确实出现了通路分歧时才值得提：全行业同一条路的话，
            # 强度相同不会造成任何"位置看起来有信息"的错觉。
            if group["share_h2"].nunique() > 1:
                best = (str(sector), int(counts.iloc[0]), int(len(group)))
    return best


def main() -> None:
    apply_style()
    n_sinks = assert_v9_tree(RUN)
    edges = active_edges(RUN, YEAR)
    geom, xy = edge_geometry(), node_xy()
    sink = sinks(RUN, YEAR)
    coal = coal_hubs(RUN, YEAR)
    ind = industry_hubs(RUN, YEAR)
    coal_captured = float(sink["use_mtpa"].sum())      # 工业捕集为 0，注入全部来自煤电

    missing = sum(1 for e in edges["edge_id"].astype(str) if e not in geom)
    if missing:
        print(f"  [note] {missing}/{len(edges)} 条有流量的边没有数字化几何，按端点直线画")

    fig = plt.figure(figsize=(183 * MM, 108 * MM))
    # 底部 0.105 是留给图注的三行；地图轴若压到 0.05，图注会直接盖在地图图例上。
    ax_map = fig.add_axes([0.004, 0.105, 0.592, 0.885])
    ax_b = fig.add_axes([0.680, 0.635, 0.300, 0.318])
    ax_c = fig.add_axes([0.680, 0.190, 0.300, 0.258])

    panel_a(fig, ax_map, edges, geom, xy, sink, coal, ind)
    panel_b(ax_b, sink)
    ratio = panel_c(ax_c, sink, ind, coal_captured)

    panel_label(ax_map, "a", x=0.015, y=0.995)
    panel_label(ax_b, "b", x=-0.20, y=1.14)
    panel_label(ax_c, "c", x=-0.20, y=1.14)

    n_full = int((sink["util"] >= 0.99).sum())
    deg_sector, deg_n, deg_total = _degenerate_sector(ind)
    deg_note = (f"{SECTOR_LABELS_ZH.get(deg_sector, deg_sector)}的厂址级分布是 LP 边际简并"
                f"（{deg_n}/{deg_total} 个厂址的 CO$_2$ 强度完全相同），不承载空间信息。"
                if deg_n else "")
    fig.text(0.004, 0.004, cjk_fill(
        f"情景 {RUN}（v9.2 管网，2026-09-12 重建，{n_sinks} 个汇对每个源都可达），{YEAR} 年横截面。"
        f"注入合计 {coal_captured:,.1f}，"
        f"全国注入能力 {sink['injectivity_mtpa'].sum():,.1f} Mt CO$_2$ yr$^{{-1}}$"
        f"（{coal_captured / sink['injectivity_mtpa'].sum():.1%}），其中 {n_full} 个汇正好压在各自的"
        f"注入能力上限，工业捕集为 0。(c) 第三条按捕集率 {CAPTURE_RATE:.0%} 计，是结构性结论："
        f"即使把煤电全部让出，工业的 CCS 需求仍是全国注入能力的 {ratio:.1f} 倍。"
        f"(a) 管段线宽正比 $\\sqrt{{流量}}$，汇点面积正比注入量，工业点不编码排放量。"
        f"钢铁长流程与合成氨全部转氢、水泥与电炉钢全部不改造，是结构性的；" + deg_note,
        106),
        fontsize=5.0, color="#555555", ha="left", va="bottom", linespacing=1.45)

    save_fig(fig, "ind_fig2_storage_allocation", "main")
    report(edges, sink, coal, ind, coal_captured, ratio)


def report(edges: pd.DataFrame, sink: pd.DataFrame, coal: pd.DataFrame, ind: pd.DataFrame,
           coal_captured: float, ratio: float) -> None:
    cap = float(sink["injectivity_mtpa"].sum())
    print(f"  管网：{len(edges)} 段有流量，合计输送 {edges['edge_flow_mtpa'].sum():,.1f} Mt/yr")
    print(f"  封存：{int((sink['use_mtpa'] > ACTIVE_MTPA).sum())}/{len(sink)} 个汇在用，"
          f"注入 {coal_captured:,.1f} / 能力 {cap:,.1f} Mt/yr = {coal_captured / cap:.1%}")
    print(f"        其中 {int((sink['util'] >= 0.99).sum())} 个压满注入能力")
    print(f"  煤电：{int((coal['captured_mt'] > 0.01).sum())}/{len(coal)} 个厂址有捕集，"
          f"捕集 {coal['captured_mt'].sum():,.1f} Mt/yr")
    print(f"  工业：{len(ind)} 个厂址，捕集 {ind['captured_mt'].sum():.1f} Mt/yr，"
          f"通路 " + "  ".join(f"{ROUTE_ZH[r]}={int((ind['route'] == r).sum())}"
                               for r in ("unabated", "ccs", "h2")))
    print(f"  若工业全部上 CCS：需要 {float(ind['baseline_co2_mt'].sum()) * CAPTURE_RATE:,.0f} Mt/yr"
          f" = 全国注入能力的 {ratio:.2f} 倍")


if __name__ == "__main__":
    main()
