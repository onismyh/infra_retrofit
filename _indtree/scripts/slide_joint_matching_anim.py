# -*- coding: utf-8 -*-
"""幻灯片用：煤电 + 工业联合求解的源汇匹配跨期演变（2030/2040/2050/2060）。

数据源固定为本树（v9 管网 + 工业点源联合三角剖分）的部门碳目标算例：
    ST_WA_cwatm_126_dry_oq  考虑水约束（主情景）
    ST_BASE                 不考虑水约束（对照）
工业活动量与部门目标出自 China TIMES V2.0（inputs/sector_targets_times_cn60.csv、
industry_output_index_times_cn60.csv）。行业分组沿用 TIMES 的口径：钢铁 / 水泥 / 化工。

视觉分工（改版要点）：管网一律黑色、粗细编码流量；参与改造的源用彩色实心点、按行业
或技术着色；既不改造也不接管网的源用淡灰半透明点，只留背景密度不抢主线。
图例不画在帧里，由幻灯片统一给（两张图共用）。输出 results/figures/slides/anim_joint/。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import map_tht as M                                        # noqa: E402
from plot_style import (                                   # noqa: E402
    PATHWAY_COLORS, apply_style, to_map_xy,
)

TREE = HERE.parent
INP, RES = TREE / "inputs", TREE / "results"
OUT = TREE.parent / "results" / "figures" / "slides" / "anim_joint"
YEARS = (2030, 2040, 2050, 2060)
MAIN, CTRL = "#5C3A6E", "#8A6520"
CASES = (("ST_WA_cwatm_126_dry_oq", "water", "考虑水约束 · 主情景", MAIN),
         ("ST_BASE", "nowater", "不考虑水约束 · 对照", CTRL))
LAND, PROV, BORDER = "#F7F3EC", "#E4DCD1", "#3F3730"
PIPE, SINK_E, IDLE = "#121212", "#121212", "#CCC7C0"
INK, MUTE, HAIR = "#2B2723", "#8A8079", "#DCD4CA"
FIG_W, FIG_H, DPI = 5.80, 3.19, 200
PX = 0.720
ACTIVE, FLOW_REF, SINK_REF = 0.01, 40.0, 60.0
# 煤电：主导改造技术。(键, 颜色, 记号, 尺寸系数, 透明度, 图例名, 是否实心)
# 参与改造的三种用更艳的同色系（PATHWAY_COLORS 的提亮版，形状仍是圆 / 三角 / 方），
# 不参与的两种退到极淡的灰，让主角是黑色管网和彩色亮点。
BIO, BEC, CCS = "#4FD06B", "#12A54F", "#4A4A4A"
COAL = [("unabated", IDLE, "o", 0.26, 0.16, "未改造燃煤", False),
        ("retire", IDLE, "o", 0.28, 0.20, "提前退役", True),
        ("biomass", BIO, "o", 0.58, 0.98, "生物质掺烧", True),
        ("beccs", BEC, "^", 0.62, 1.00, "BECCS", True),
        ("ccs", CCS, "s", 0.56, 1.00, "CCS 改造", True)]
# 工业：按 TIMES V2.0 的部门口径分组
IND_GROUP = {"cement": "水泥", "steel_bf_bof": "钢铁", "steel_eaf": "钢铁",
             "ammonia": "化工", "methanol": "化工"}
IND_COLOR = {"水泥": "#8B2FD0", "钢铁": "#D9911F", "化工": "#F5308F"}
CCS_ON = 0.05                                   # 份额超过此值才算选了该技术
# 工业技术：形状编码（颜色仍编码行业）。(键, 记号, 透明度, 图例名, 是否参与改造)
IND_TECH = [("ccs", "D", 1.00, "配 CCS", True), ("h2", "v", 1.00, "氢替代", True),
            ("none", "o", 0.16, "未改造", False)]


def geometry():
    """edge_id -> 地图坐标折线。备选网络的每条边都带 WKT，端点直线只作兜底。"""
    from shapely import wkt
    cand = pd.read_csv(INP / "pipeline_candidate_edges.csv")
    cand.columns = [c.lstrip("\ufeff") for c in cand.columns]
    geo = {}
    for eid, g in zip(cand["edge_id"].astype(str), cand["geometry_wkt"]):
        if not isinstance(g, str):
            continue
        try:
            c = np.asarray(wkt.loads(g).coords)
        except Exception:                        # noqa: BLE001
            continue
        x, y = to_map_xy(c[:, 0], c[:, 1])
        geo[eid] = np.column_stack([x, y])
    nd = pd.read_csv(INP / "pipeline_nodes.csv")
    nd.columns = [c.lstrip("\ufeff") for c in nd.columns]
    pos = {str(a): (float(b), float(c)) for a, b, c in zip(nd["node_id"], nd["lon"], nd["lat"])}
    return geo, pos


GEO, POS = geometry()
HUBS = pd.read_csv(INP / "storage_hubs.csv").set_index("storage_hub_id")


def read(case, name):
    return pd.read_csv(RES / case / name)


def segs_for(case_edges):
    """活动边 -> [(折线, 流量)]，缺几何的按端点直线补。"""
    out = []
    for eid, f, a, b in zip(case_edges["edge_id"].astype(str), case_edges["edge_flow_mtpa"],
                            case_edges["from_node_id"].astype(str),
                            case_edges["to_node_id"].astype(str)):
        g = GEO.get(eid)
        if g is None:
            pa, pb = POS.get(a), POS.get(b)
            if pa is None or pb is None:
                continue
            x, y = to_map_xy(np.array([pa[0], pb[0]]), np.array([pa[1], pb[1]]))
            g = np.column_stack([x, y])
        out.append((g, float(f)))
    return out


def draw_layers(a, edges, sinks, coal, ind, k=1.0, points=True):
    """图层顺序：淡灰的"不参与"点 → 黑色管网 → 彩色改造点 → 封存汇。"""
    IDLE_KEYS = {"unabated", "retire"}           # 既不改造、也不接管网
    idle_coal = [c for c in COAL if c[0] in IDLE_KEYS]
    live_coal = [c for c in COAL if c[0] not in IDLE_KEYS]

    def coal_points(spec, zbase):
        for key, colour, marker, rel, alpha, _, filled in spec:
            s = coal[coal["dominant_pathway"] == key]
            if s.empty:
                continue
            px, py = to_map_xy(s["centroid_longitude"].to_numpy(), s["centroid_latitude"].to_numpy())
            a.scatter(px, py, s=k * rel * (0.9 + 1.45 * s["capacity_mw"].to_numpy() / 1000.0),
                      marker=marker, facecolor=colour if filled else "none", edgecolor=colour,
                      linewidth=k * (0.10 if filled else 0.22), alpha=alpha, zorder=zbase + rel * 0.3)

    def ind_points(live, zbase):
        for tech, marker, alpha, _, is_live in IND_TECH:
            if is_live != live:
                continue
            for grp, colour in IND_COLOR.items():
                s = ind[(ind["group"] == grp) & (ind["tech"] == tech)]
                if s.empty:
                    continue
                px, py = to_map_xy(s["longitude"].to_numpy(), s["latitude"].to_numpy())
                co2 = np.sqrt(np.clip(s["baseline_co2_mt"].to_numpy(), 0.0, None))
                if live:
                    a.scatter(px, py, s=k * (0.6 + 1.05 * co2), marker=marker, facecolor=colour,
                              edgecolor=colour, linewidth=k * 0.08, alpha=alpha, zorder=zbase)
                else:
                    a.scatter(px, py, s=k * (0.45 + 0.50 * co2), marker=marker, facecolor=IDLE,
                              edgecolor="none", alpha=alpha, zorder=zbase)

    if points:
        coal_points(idle_coal, 1.6)
        ind_points(False, 1.8)
    if edges:
        ss = [g for g, _ in edges]
        # 管段流量上限就是 FLOW_REF（管径分档决定），所以直接按比例给线宽：
        # 中位 0.34 pt、九分位 1.28 pt、满管 2.06 pt，粗细一眼能读出输量。
        ww = [k * (0.11 + 1.95 * min(f / FLOW_REF, 1.0)) for _, f in edges]
        a.add_collection(LineCollection(ss, colors=PIPE, linewidths=ww, capstyle="round",
                                        alpha=0.93, zorder=3.0))
    if points:
        coal_points(live_coal, 3.6)
        ind_points(True, 3.9)
    if sinks.empty:
        return
    hx, hy = to_map_xy(sinks["longitude"].to_numpy(), sinks["latitude"].to_numpy())
    off = sinks["offshore"].to_numpy().astype(bool)
    # 汇是终点不是主角：缩到刚好能看出"注入量在长大"，描边黑、与管网同色系。
    size = k * (2.2 + 20.0 * np.sqrt(sinks["storage_use_mtpa"].to_numpy() / SINK_REF))
    for m, sel in (("o", ~off), ("H", off)):
        if sel.any():
            a.scatter(hx[sel], hy[sel], s=size[sel], marker=m, facecolor="#FFFFFF",
                      edgecolor=SINK_E, linewidth=k * 0.38, alpha=0.96, zorder=5)


def year_slice(case, year):
    n = read(case, "network_edges.csv")
    e = n[(n.year == year) & (n.edge_flow_mtpa > ACTIVE)]
    p = read(case, "plant_detail.csv")
    coal = p[p.year == year]
    d = read(case, "industry_detail.csv")
    ind = d[d.year == year].copy()
    ind["group"] = ind["sector"].map(IND_GROUP)
    c, h = ind["share_ccs"].to_numpy(), ind["share_h2"].to_numpy()
    ind["tech"] = np.where((c > CCS_ON) & (c >= h), "ccs",
                           np.where(h > CCS_ON, "h2", "none"))
    s = read(case, "storage_utilization.csv")
    s = s[(s.year == year) & (s.storage_use_mtpa > ACTIVE)].copy()
    s = s.join(HUBS[["longitude", "latitude", "offshore"]], on="storage_hub_id")
    return segs_for(e), s, coal, ind, float(e["length_km"].sum())


def progress(fig, year, accent):
    x0, x1 = 0.045, 0.640
    fig.add_artist(Line2D([x0, x1], [0.040, 0.040], color=HAIR, lw=1.3,
                          transform=fig.transFigure))
    for i, y in enumerate(YEARS):
        x = x0 + i * (x1 - x0) / (len(YEARS) - 1.0)
        on = y == year
        fig.add_artist(Line2D([x], [0.040], marker="o", ms=8.5 if on else 4.6, ls="none",
                              markerfacecolor=accent if on else "#FFFFFF",
                              markeredgecolor=accent if on else HAIR, mew=1.3,
                              transform=fig.transFigure))
        fig.text(x, 0.0, "%d" % y, fontsize=8.2, ha="center", va="bottom",
                 color=accent if on else MUTE, fontweight="bold" if on else "normal")


def frame(case, year, label, accent):
    edges, sinks, coal, ind, km = year_slice(case, year)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.075, 0.700, 0.905])

    def business(a, k=1.0, points=True):
        draw_layers(a, edges, sinks, coal, ind, k, points)

    M.draw(ax, province_lw=0.26, country_lw=0.72, facecolor=LAND,
           province_ec=PROV, country_ec=BORDER)
    business(ax)
    M.set_extent(ax, M.bound_tight())
    for art in list(ax.collections) + list(ax.lines) + list(ax.patches):
        art.set_clip_on(True)
        art.set_clip_box(ax.bbox)
    scs = M.scs_inset(ax, business=lambda a: business(a, 0.5, points=False),
                      rect=[0.012, 0.02, 0.150, 0.265], province_lw=0.18, country_lw=0.5,
                      facecolor=LAND, bg=LAND, province_ec=PROV, country_ec=BORDER)
    for s in scs.spines.values():
        s.set_visible(True)
        s.set_edgecolor(HAIR)

    fig.text(PX, 0.928, "%d" % year, fontsize=28, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(PX + 0.172, 0.916, "年", fontsize=12, color=MUTE, ha="left", va="center")
    fig.add_artist(FancyBboxPatch((PX, 0.800), 0.272, 0.056,
                                  boxstyle="round,pad=0.004,rounding_size=0.013",
                                  facecolor=accent, edgecolor="none", transform=fig.transFigure))
    fig.text(PX + 0.136, 0.828, label, fontsize=8.6, color="white", ha="center", va="center",
             fontweight="bold")
    fig.add_artist(Line2D([PX, 0.995], [0.762, 0.762], color=HAIR, lw=0.9,
                          transform=fig.transFigure))
    nosink = sinks.empty
    rows = ([("煤电捕集", "尚未出现"), ("工业捕集", "尚未出现"), ("在用封存汇", "—"),
             ("管网", "尚未建设")] if nosink else
            [("煤电捕集", "%.0f Mt/年" % coal["captured_mt"].sum()),
             ("工业捕集", "%.0f Mt/年" % ind["captured_mt"].sum()),
             ("在用封存汇", "%d 个" % len(sinks)),
             ("管网", "%.1f 万 km" % (km / 1e4))])
    for i, (k, v) in enumerate(rows):
        y = 0.700 - i * 0.076
        fig.text(PX, y, k, fontsize=9.2, color=MUTE, ha="left", va="center")
        fig.text(0.995, y, v, fontsize=11.0, color=INK, ha="right", va="center", fontweight="bold")
    progress(fig, year, accent)
    return fig


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for case, key, label, accent in CASES:
        paths = []
        for year in YEARS:
            fig = frame(case, year, label, accent)
            p = OUT / ("joint_%s_%d.png" % (key, year))
            fig.savefig(p, dpi=DPI, facecolor="white")
            plt.close(fig)
            paths.append(p)
            print("  frame", p.name)
        ims = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=220) for p in paths]
        gif = OUT / ("joint_%s.gif" % key)
        ims[0].save(gif, save_all=True, append_images=ims[1:],
                    duration=[1150, 1150, 1150, 1550], loop=0, disposal=2, optimize=True)
        print("GIF", gif.name, "%.2f MB" % (gif.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
