# -*- coding: utf-8 -*-
"""幻灯片 p51：电力 + 工业的多资源供需匹配（生物质 / 绿氢 / 水 / CO2），2050 年横截面。

替换原来的三联图（只有煤电，只有 CO2 / 生物质 / 水）。这一版把工业点源一起放进来，
并补上绿氢这条链，于是四个面板正好回答"哪条资源链服务谁"：
  a 生物质 -> 只进煤电（工业没有掺烧路径）
  b 绿氢   -> 只进工业（煤电掺氨在所有情景下都是 0）
  c 水     -> 两个部门都要，工业要得更多
  d CO2    -> 两个部门共用一张管网和同一批封存汇
底图统一按唐昊天 GIS_layer/plot.ipynb 的画法，走 scripts/map_tht.py。
输出 results/figures/slides/multiflow_joint.png / .pdf。
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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import map_tht as M                                        # noqa: E402
from plot_style import apply_style, load_basins            # noqa: E402

TREE = HERE.parent
RES = TREE / "results"
OUT = TREE.parent / "results" / "figures" / "slides"
CASE = "ST_WA_cwatm_126_dry_oq"
YEAR = 2050

INK, MUTE, IDLE = "#2B2723", "#8A8079", "#D5D0C9"
BIO, H2, WAT, PIPE = "#3FBF63", "#D9911F", "#2E6FB7", "#121212"
COAL = "#12A54F"
SECTOR_C = {"cement": "#8B2FD0", "steel_bf_bof": "#D9911F", "steel_eaf": "#D9911F",
            "ammonia": "#F5308F", "methanol": "#F5308F"}
SECTOR_ZH = {"cement": "水泥", "steel_bf_bof": "钢铁", "steel_eaf": "钢铁",
             "ammonia": "化工", "methanol": "化工"}
BASIN_ZH = {"A": "东北诸河", "C": "海河", "D": "黄河", "E": "淮河", "F": "长江",
            "G": "东南诸河", "H": "珠江", "J": "西南诸河", "K": "西北诸河"}


def read(name: str) -> pd.DataFrame:
    d = pd.read_csv(RES / CASE / name)
    return d[d["year"] == YEAR].copy() if "year" in d.columns else d


_K = 1.0        # 小图里把所有点径、线宽按同一比例收小，见 frame()


def spokes(ax, lon0, lat0, lon1, lat1, weight, colour, ref, lo=0.08, hi=0.85,
           alpha=0.45, z=2.0):
    """供给节点 -> 需求点的辐条；线宽 ∝ sqrt(量)，与昊天的 sqrt(flow)/scale 同族。"""
    x0, y0 = M.xy(lon0, lat0)
    x1, y1 = M.xy(lon1, lat1)
    segs = np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y1])], axis=1)
    w = _K * (lo + (hi - lo) * np.sqrt(np.clip(np.asarray(weight, float) / ref, 0, 1)))
    ax.add_collection(LineCollection(segs, colors=colour, linewidths=w, alpha=alpha,
                                     capstyle="round", zorder=z))


def pts(ax, lon, lat, size, colour, marker="o", ec="none", lw=0.0, alpha=1.0, z=3.0):
    x, y = M.xy(lon, lat)
    ax.scatter(x, y, s=np.asarray(size, float) * _K * _K, c=colour, marker=marker,
               edgecolors=ec, linewidths=lw * _K,
               alpha=alpha, zorder=z)


def bar(ax, cmap, vmin, vmax, label, norm=None, ticks=None,
        rect=(0.018, 0.055, 0.235, 0.021)):
    """面板内的小色标，昊天那套离散色带的做法：贴在左下角，不占版面。"""
    import matplotlib as mpl
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((rect[0] - 0.018, rect[1] - 0.048), rect[2] + 0.05, rect[3] + 0.115,
                           transform=ax.transAxes, facecolor="white", edgecolor="none",
                           alpha=0.88, zorder=8.5))
    cax = ax.inset_axes(rect, zorder=9)
    if norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal",
                                   ticks=ticks)
    cb.outline.set_linewidth(0.35)
    cb.outline.set_edgecolor(MUTE)
    cax.tick_params(labelsize=6.4, length=1.4, width=0.35, pad=1.2, colors=INK)
    if ticks is None:
        cax.set_xticks([vmin, vmax])
        cax.set_xticklabels(["0", "%.0f" % vmax])
    cax.set_title(label, fontsize=6.8, pad=1.8, color=INK)
    return cb


def frame(ax, title: str, inset_business=None):
    M.draw(ax, province_lw=0.16, country_lw=0.6)
    M.set_extent(ax)
    def small(a):
        global _K
        _K = 0.45
        try:
            inset_business(a)
        finally:
            _K = 1.0

    M.scs_inset(ax, business=small if inset_business is not None else None,
                rect=(0.775, 0.010, 0.215, 0.30))
    ax.set_title(title, fontsize=9.6, pad=3.0, color=INK)


def panel_bio(ax, pl, ind):
    """a 生物质 -> 煤电。工业不烧生物质，画成淡灰底衬。"""
    bf = read("biomass_flows.csv")
    use = pl[pl["biomass_use_gj"] > 0]

    def biz(a):
        spokes(a, bf.node_longitude, bf.node_latitude, bf.centroid_longitude,
               bf.centroid_latitude, bf.flow_gj, BIO, bf.flow_gj.quantile(0.97), lo=0.12, hi=1.0, alpha=0.5, z=2)
        pts(a, ind.longitude, ind.latitude, 1.8, IDLE, marker="D", alpha=0.6, z=2.4)
        pts(a, use.centroid_longitude, use.centroid_latitude,
            2 + 20 * np.sqrt(use.biomass_use_gj / use.biomass_use_gj.max()),
            COAL, ec="white", lw=0.2, z=3.2)
    biz(ax)
    frame(ax, "a 生物质 → 煤电　%.1f EJ（顶到全国上限）" % (bf.flow_gj.sum() / 1e9),
          inset_business=biz)
    return dict(nodes=bf.biomass_node_id.nunique(), plants=len(use),
                ej=bf.flow_gj.sum() / 1e9, km=bf.distance_km.median())


def panel_h2(ax, pl, ind):
    """b 绿氢 -> 工业。煤电掺氨恒为 0，煤电画成淡灰底衬。底图按省给绿氨供给潜力上色。"""
    sup = pd.read_csv(TREE / "inputs" / "ammonia_supply_curve.csv")
    sup.columns = [c.lstrip("﻿") for c in sup.columns]
    sup = sup[sup["year"] == YEAR]
    by_prov = sup.groupby("province_name")["nh3_supply_kg_per_year"].sum() / 1e9
    prov, _ = M.layers()
    key = {str(n)[:2]: n for n in prov["name"]}
    vals = {}
    for p, v in by_prov.items():
        k = key.get(str(p)[:2])
        if k is not None:
            vals[k] = vals.get(k, 0.0) + float(v)
    pv = prov.assign(sup=prov["name"].map(vals).fillna(0.0))
    cmap = LinearSegmentedColormap.from_list("amber", ["#FFFFFF", "#FDF0DC", "#F7DDB4", "#EFC078"])
    vmax = float(np.nanpercentile(pv.sup[pv.sup > 0], 88)) if (pv.sup > 0).any() else 1.0
    use = ind[ind["h2_kg"] > 0]

    def biz(a):
        pv.plot(ax=a, column="sup", cmap=cmap, vmin=0.0, vmax=vmax, edgecolor="none",
                alpha=0.9, zorder=0.5)
        pts(a, pl.centroid_longitude, pl.centroid_latitude, 1.8, IDLE, alpha=0.6, z=2.0)
        for zh, colour in (("水泥", SECTOR_C["cement"]), ("钢铁", SECTOR_C["steel_bf_bof"]),
                           ("化工", SECTOR_C["ammonia"])):
            g = use[use.sector.map(SECTOR_ZH) == zh]
            if len(g):
                pts(a, g.longitude, g.latitude,
                    3 + 30 * np.sqrt(g.h2_kg / use.h2_kg.max()), colour,
                    marker="D", ec="white", lw=0.22, z=3.2)
    biz(ax)
    frame(ax, "b 绿氢 → 工业　%.1f Mt H$_2$（煤电掺氨 0）" % (use.h2_kg.sum() / 1e9),
          inset_business=biz)
    bar(ax, cmap, 0.0, vmax, "分省绿氨供给潜力（Mt NH$_3$/yr）")
    top = use.groupby(use.sector.map(SECTOR_ZH))["h2_kg"].sum().sort_values(ascending=False)
    return dict(hubs=len(use), mt=use.h2_kg.sum() / 1e9,
                top=top.index[0], top_mt=top.iloc[0] / 1e9)


def panel_water(ax, pl, ind):
    """c 水 -> 煤电 + 工业。流域底色 = 流域取水 / 配额余量（模型自己的 water_basin_quota）。"""
    wf = read("water_flows.csv")
    wn = pd.read_csv(TREE / "inputs" / "water_nodes.csv")
    wn.columns = [c.lstrip("﻿") for c in wn.columns]
    wf = wf.merge(wn[["water_node_id", "longitude", "latitude"]], on="water_node_id", how="left")
    wf = wf.dropna(subset=["longitude", "latitude"])
    ru = pd.read_csv(RES / CASE / "resource_use.csv")
    q = ru[(ru.year == YEAR) & (ru.resource_type == "water_basin_quota")]
    util = {str(k): float(v) for k, v in zip(q.region, q.utilization)}
    basins = load_basins()
    cmap = ListedColormap(["#EAF2F8", "#C9DDF0", "#F6D9A8", "#E88C5A"])
    norm = BoundaryNorm([0.0, 0.25, 0.5, 0.75, 1.0], ncolors=cmap.N, clip=True)
    iu = ind[ind["water_m3"] > 0]

    def biz(a):
        if basins is not None:
            bs = basins.assign(u=[util.get(str(c), np.nan) for c in basins["code"]])
            bs[bs.u.notna()].plot(ax=a, column="u", cmap=cmap, norm=norm, edgecolor="#FFFFFF",
                                  linewidth=0.25, alpha=0.85, zorder=0.5)
        spokes(a, wf.longitude, wf.latitude, wf.centroid_longitude, wf.centroid_latitude,
               wf.flow_m3, WAT, wf.flow_m3.quantile(0.92), lo=0.22, hi=1.5, alpha=0.75, z=2.2)
        pts(a, pl.centroid_longitude, pl.centroid_latitude,
            1.6 + 15 * np.sqrt(pl.water_use_m3 / max(pl.water_use_m3.max(), 1.0)),
            COAL, ec="white", lw=0.18, alpha=0.9, z=3.0)
        pts(a, iu.longitude, iu.latitude,
            1.6 + 15 * np.sqrt(iu.water_m3 / iu.water_m3.max()),
            "#8B2FD0", marker="D", ec="white", lw=0.18, alpha=0.9, z=3.1)
    biz(ax)
    frame(ax, "c 水 → 煤电 + 工业　煤电 %.1f｜工业 %.1f 亿 m$^3$"
          % (pl.water_use_m3.sum() / 1e8, ind.water_m3.sum() / 1e8), inset_business=biz)
    cb = bar(ax, cmap, 0.0, 1.0, "流域取水 / 配额余量", norm=norm, ticks=[0, 0.5, 1.0])
    cb.ax.set_xticklabels(["0", "0.5", "1.0"])
    tight = q.sort_values("utilization", ascending=False).iloc[0]
    return dict(coal=pl.water_use_m3.sum() / 1e8, ind=ind.water_m3.sum() / 1e8,
                basin=BASIN_ZH.get(str(tight.region), str(tight.region)),
                util=float(tight.utilization))


def panel_co2(ax, pl, ind):
    """d CO2 -> 管网 -> 封存汇。两个部门共用同一张网。"""
    ne = read("network_edges.csv")
    ne = ne[ne.edge_flow_mtpa > 1e-6]
    nodes = pd.read_csv(TREE / "inputs" / "pipeline_nodes.csv")
    nodes.columns = [c.lstrip("\ufeff") for c in nodes.columns]
    xy_ = nodes.set_index("node_id")[["lon", "lat"]]
    ne = ne.join(xy_.rename(columns={"lon": "x0", "lat": "y0"}), on="from_node_id")
    ne = ne.join(xy_.rename(columns={"lon": "x1", "lat": "y1"}), on="to_node_id").dropna(
        subset=["x0", "y0", "x1", "y1"])
    st = read("storage_utilization.csv")
    st = st[st.storage_use_mtpa > 1e-6]
    sh = pd.read_csv(TREE / "inputs" / "storage_hubs.csv")
    sh.columns = [c.lstrip("\ufeff") for c in sh.columns]
    lonc = "longitude" if "longitude" in sh.columns else "lon"
    latc = "latitude" if "latitude" in sh.columns else "lat"
    st = st.merge(sh[["storage_hub_id", lonc, latc]], on="storage_hub_id", how="left")
    cp, ci = pl[pl.captured_mt > 0], ind[ind.captured_mt > 0]

    def biz(a):
        spokes(a, ne.x0, ne.y0, ne.x1, ne.y1, ne.edge_flow_mtpa, PIPE, 40.0,
               lo=0.12, hi=1.7, alpha=0.9, z=2.2)
        pts(a, cp.centroid_longitude, cp.centroid_latitude,
            3 + 30 * np.sqrt(cp.captured_mt / max(cp.captured_mt.max(), 1e-9)),
            COAL, ec="white", lw=0.22, z=3.0)
        pts(a, ci.longitude, ci.latitude,
            3 + 30 * np.sqrt(ci.captured_mt / ci.captured_mt.max()),
            "#8B2FD0", marker="D", ec="white", lw=0.22, z=3.1)
        pts(a, st[lonc], st[latc], 6 + 60 * np.sqrt(st.storage_use_mtpa / st.storage_use_mtpa.max()),
            "none", marker="o", ec=PIPE, lw=0.6, z=3.6)
    biz(ax)
    frame(ax, "d CO$_2$ → 管网 → 封存汇　%.1f 万 km，注入 %.0f Mt"
          % (ne.length_km.sum() / 1e4, st.storage_use_mtpa.sum()), inset_business=biz)
    return dict(edges=len(ne), km=ne.length_km.sum() / 1e4, sinks=len(st),
                inj=st.storage_use_mtpa.sum(), coal=cp.captured_mt.sum(), ind=ci.captured_mt.sum())


def legend(fig):
    h = [Line2D([0], [0], color=BIO, lw=1.6, label="生物质供给辐条"),
         Line2D([0], [0], marker="s", color="none", markerfacecolor=H2, markersize=5,
                alpha=0.5, label="绿氢供给潜力"),
         Line2D([0], [0], color=WAT, lw=1.6, label="取水链路"),
         Line2D([0], [0], color=PIPE, lw=1.8, label="CO$_2$ 管段（线宽 ∝ 流量）"),
         Line2D([0], [0], marker="o", color="none", markerfacecolor=COAL, markersize=6,
                label="煤电机组"),
         Line2D([0], [0], marker="D", color="none", markerfacecolor="#8B2FD0", markersize=5.5,
                label="工业点源（b 面板按行业着色）"),
         Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                markeredgecolor=PIPE, markersize=6.5, label="在用封存汇"),
         Line2D([0], [0], marker="o", color="none", markerfacecolor=IDLE, markersize=5.5,
                markeredgecolor="none", label="不参与该资源链（淡灰）")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8.4,
               handlelength=1.5, columnspacing=1.4, handletextpad=0.5,
               bbox_to_anchor=(0.5, -0.004))


def main() -> None:
    apply_style()
    plt.rcParams["axes.titlesize"] = 9.6
    pl = read("plant_detail.csv")
    ind = read("industry_detail.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.10, 5.72))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.962, bottom=0.090,
                        wspace=0.015, hspace=0.13)
    s = {}
    s["bio"] = panel_bio(axes[0, 0], pl, ind)
    s["h2"] = panel_h2(axes[0, 1], pl, ind)
    s["wat"] = panel_water(axes[1, 0], pl, ind)
    s["co2"] = panel_co2(axes[1, 1], pl, ind)
    legend(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("multiflow_joint." + ext), dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("%s %d 年（%s）" % ("multiflow_joint", YEAR, CASE))
    print("  a 生物质：%d 个供给节点 -> %d 台煤电，%.2f EJ，运距中位 %.0f km"
          % (s["bio"]["nodes"], s["bio"]["plants"], s["bio"]["ej"], s["bio"]["km"]))
    print("  b 绿氢  ：%d 个工业厂址用 %.1f Mt H2，其中%s %.1f Mt；煤电掺氨 0"
          % (s["h2"]["hubs"], s["h2"]["mt"], s["h2"]["top"], s["h2"]["top_mt"]))
    print("  c 水    ：煤电 %.1f 亿 m3，工业 %.1f 亿 m3；%s 配额利用率 %.2f"
          % (s["wat"]["coal"], s["wat"]["ind"], s["wat"]["basin"], s["wat"]["util"]))
    print("  d CO2   ：%d 段有流量、%.1f 万 km，注入 %.0f Mt 到 %d 个汇；煤电 %.0f + 工业 %.0f Mt"
          % (s["co2"]["edges"], s["co2"]["km"], s["co2"]["inj"], s["co2"]["sinks"],
             s["co2"]["coal"], s["co2"]["ind"]))


if __name__ == "__main__":
    main()
