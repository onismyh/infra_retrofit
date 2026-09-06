# -*- coding: utf-8 -*-
"""ED14（替换版）：多级源汇匹配 —— 燃料、水、CO₂ 三条链在同一批机组上怎么接。

替换原来的"六张年份 × 口径地图"。那一版只画了 CO₂ 管网，回答不了三个问题：
生物质（和氨）从哪来、水从哪个流域取、哪些机组什么都没匹配上。本图一行三张图分别
画三条链，第二行用同一年的匹配台账和四个年份的链路量把"匹配到 / 没匹配到"讲清楚。

面板
  a  CO₂：机组 → 管网 → 封存汇。管段线宽 ∝ 流量并标出流向；汇的面积 ∝ 注入量；
     未启用的汇画成小空心灰点。机组点面积 ∝ 装机，填色 = 主导路径，深色描边 = 已接入管网。
  b  燃料：生物质节点 → 机组（绿色辐条，线宽 ∝ 供给量）；若有氨供给则为橙色辐条。
  c  水：水源节点 → 机组（蓝色辐条，线宽 ∝ 取水量）；底图按流域填色 = 电力取水 / 配额。
     机组填色 = 湿冷取水 / 已转空冷 / 退役。
  d  同年匹配台账：三条链各自"匹配到多少装机、多少个 hub"，不考虑水 vs s = 0.85 并列。
  e–g 四个年份的链路量：CO₂（捕集、陆上注入、海上注入）、生物质（PJ）、取水（亿 m³）。

地图年份取 2050：捕集已成规模而机组尚未大批退役。两种口径的地图只画 s = 0.85；
不考虑水的差异全部进 d–g 的数字里，因为 ED13 已证明两种口径的空间格局肉眼不可分。
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_style import (  # noqa: E402
    AIR_COOLING_C, DOUBLE_COL, PATHWAY_COLORS, PATHWAY_LABELS, add_scs_inset, apply_style,
    draw_china_basemap, load_basins, mainland_extent, panel_label, save_fig, to_map_xy,
)
from plot_ed_water_on_off import BASE, RES, hub_frame  # noqa: E402
from plot_ed_water_abatement import HUBS, INPUTS, injection, network  # noqa: E402
from plot_ed_source_sink_matching import (  # noqa: E402
    ACTIVE, EXCL_C, FLOW_REF, GEOM, PIPE_C, SINK_EDGE, SINK_REF, TREAT,
)
from coal_retrofit.constants import WATER_EXTRACTABLE_FRACTION  # noqa: E402

MAP_YEAR = 2050
YEARS = (2030, 2040, 2050, 2060)
CASES = ((BASE, "不考虑水", "#969696", (0, (2.2, 1.4))), (TREAT, "考虑水（s = 0.85）", "#CC3311", "-"))
WATER_SCEN_ID = "cwatm|gfdl-esm4|ssp126"
EXISTING_SHARE = 0.85
USABLE = float(WATER_EXTRACTABLE_FRACTION) * (1.0 - EXISTING_SHARE)

BIO_C = "#31A354"          # 生物质辐条：Greens 深端
NH3_C = "#FD8D3C"          # 氨辐条：Oranges 深端
WATER_C = "#08519C"        # 取水辐条：Blues 深端
WET_C, AIR_C, RET_C = "#6BAED6", AIR_COOLING_C, PATHWAY_COLORS["retire"]
ON_C, OFF_C = "#3182BD", "#9ECAE1"   # 陆上 / 海上注入
SINK_IDLE_C = "#B9C0C7"
ARROW_C = "#08306B"
CAPTION_C = "#4A4A4A"

# 流域填色：电力取水 / 配额。<1 走 Blues，>1 走警示色；分档显式给出（CLAUDE.md §3.3）。
RATIO_BOUNDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0]
RATIO_CMAP = ListedColormap(["#F7FBFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#FDAE6B", "#CC3311"])
RATIO_NORM = BoundaryNorm(RATIO_BOUNDS, ncolors=RATIO_CMAP.N, clip=True)


# =========================================================================================
# 读数
# =========================================================================================
def flows_table(scen: str, name: str, year: int) -> pd.DataFrame:
    path = RES / scen / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    if d.empty or "year" not in d.columns:
        return pd.DataFrame()
    return d[d["year"] == year].copy()


def water_nodes() -> pd.DataFrame:
    n = pd.read_csv(INPUTS / "water_nodes.csv")
    n.columns = [c.lstrip("﻿") for c in n.columns]
    return n.set_index(n["water_node_id"].astype(str))


WATER_NODES = water_nodes()


def basin_ratio(scen: str, year: int) -> pd.Series:
    """流域尺度 电力实际取水 / 配额。取水按 water_flows 的节点归属流域（与图 2 一致）。"""
    avail = pd.read_csv(INPUTS / "water_availability.csv")
    avail.columns = [c.lstrip("﻿") for c in avail.columns]
    a = avail[(avail["planning_year"] == year) & (avail["scenario_id"] == WATER_SCEN_ID)]
    supply = a.groupby("basin_code")["dry_season_water_m3_per_year"].sum() * USABLE
    w = flows_table(scen, "water_flows", year)
    if w.empty:
        return pd.Series(dtype=float)
    w["basin"] = w["water_node_id"].astype(str).str.rsplit("_", n=1).str[-1]
    demand = w.groupby("basin")["flow_m3"].sum()
    return (demand / supply.reindex(demand.index)).dropna()


def hub_table(scen: str, year: int) -> pd.DataFrame:
    """机组表 + 三条链的匹配状态。"""
    p = hub_frame(scen, year).set_index("plant_id")
    p["co2_matched"] = p["captured_mt"] > 0.01
    p["bio_matched"] = p["biomass_use_gj"] > 1.0
    p["nh3_matched"] = p["ammonia_use_kg"] > 1.0
    eff_air = p["already_air_share"] + (1.0 - p["already_air_share"]) * p["air_cooled_share"]
    p["water_class"] = np.where(p["share_retire"] > 0.5, "retire",
                                np.where(eff_air >= 0.5, "air", "wet"))
    return p


def flow_direction(scen: str, year: int) -> dict:
    """edge_id -> +1 表示净流向与候选边几何方向一致，-1 表示相反。"""
    d = flows_table(scen, "co2_flow_direction", year)
    if d.empty:
        return {}
    cand = pd.read_csv(INPUTS / "pipeline_candidate_edges.csv")
    cand.columns = [c.lstrip("﻿") for c in cand.columns]
    start = dict(zip(cand["edge_id"].astype(str), cand["from_node_id"].astype(str)))
    out = {}
    for eid, src, net in zip(d["edge_id"].astype(str), d["source_node"].astype(str), d["net_flow_mtpa"]):
        if abs(float(net)) < ACTIVE:
            continue
        out[eid] = 1 if start.get(eid) == src else -1
    return out


# =========================================================================================
# 地图公共件
# =========================================================================================
def hub_xy(p: pd.DataFrame):
    return to_map_xy(p["centroid_longitude"].to_numpy(), p["centroid_latitude"].to_numpy())


def hub_size(p: pd.DataFrame) -> np.ndarray:
    return 1.2 + 1.5 * p["cap_gw"].to_numpy()


def draw_hubs(ax, p: pd.DataFrame, face, matched: np.ndarray, z: float = 3.0) -> None:
    """机组点：面积 ∝ 装机；匹配到的深描边、不透明，没匹配到的淡出。"""
    x, y = hub_xy(p)
    s = hub_size(p)
    face = np.asarray(face, dtype=object)
    for sel, alpha, edge, lw in ((~matched, 0.45, "none", 0.0), (matched, 0.95, "#1F2933", 0.25)):
        if sel.any():
            ax.scatter(x[sel], y[sel], s=s[sel], c=list(face[sel]), alpha=alpha,
                       edgecolor=edge, linewidth=lw, zorder=z)


def draw_spokes(ax, src_lon, src_lat, dst_lon, dst_lat, weight, colour, ref, z=2.5,
                lw_min=0.18, lw_max=1.3, alpha=0.75):
    x0, y0 = to_map_xy(np.asarray(src_lon, float), np.asarray(src_lat, float))
    x1, y1 = to_map_xy(np.asarray(dst_lon, float), np.asarray(dst_lat, float))
    segs = [((a, b), (c, d)) for a, b, c, d in zip(x0, y0, x1, y1)]
    w = lw_min + lw_max * np.sqrt(np.clip(np.asarray(weight, float) / ref, 0.0, 1.0))
    ax.add_collection(LineCollection(segs, colors=colour, linewidths=w, alpha=alpha,
                                     capstyle="round", zorder=z))


def finish_map(fig, ax, business, title: str, inset_rect, facecolor="#F1F3F5",
               basins_fill=None) -> None:
    draw_china_basemap(ax, province_lw=0.12, country_lw=0.42, facecolor=facecolor)
    if basins_fill is not None:
        basins_fill(ax)
    business(ax)
    mainland_extent(ax)
    ax.set_axis_off()
    ax.set_title(title, fontsize=6.6, pad=2.0)

    def inset(a):
        if basins_fill is not None:
            basins_fill(a)
        business(a)

    scs = add_scs_inset(fig, ax, draw=inset, axes_rect=inset_rect)
    for spine in scs.spines.values():
        spine.set_visible(True)


# =========================================================================================
# 面板 a：CO₂ 链
# =========================================================================================
def panel_co2(fig, ax, scen: str, year: int, inset_rect) -> dict:
    p = hub_table(scen, year)
    flows = network(scen, year)
    flows = flows.groupby(flows["edge_id"].astype(str))["edge_flow_mtpa"].sum()
    flows = flows[flows > ACTIVE]
    direction = flow_direction(scen, year)
    inj = injection(scen, year)
    used = inj[inj > ACTIVE]
    idle = HUBS.index.difference(used.index)

    def business(a):
        segs, widths = [], []
        arrows = []
        for eid, f in flows.items():
            g = GEOM.get(eid)
            if g is None:
                continue
            segs.append(g)
            widths.append(0.35 + 1.55 * np.sqrt(min(f / FLOW_REF, 1.0)))
            if f >= 5.0 and len(g) >= 3:
                m = len(g) // 2
                d = g[min(m + 1, len(g) - 1)] - g[max(m - 1, 0)]
                d = d * direction.get(eid, 1)
                ang = np.degrees(np.arctan2(d[1], d[0])) - 90.0
                arrows.append((g[m][0], g[m][1], ang))
        if segs:
            a.add_collection(LineCollection(segs, colors=PIPE_C, linewidths=widths,
                                            capstyle="round", alpha=0.9, zorder=3.5))
        for x, y, ang in arrows:
            a.plot([x], [y], marker=(3, 0, ang), ms=3.8, color="white", mec=ARROW_C,
                   mew=0.35, ls="none", zorder=3.7)
        face = [PATHWAY_COLORS.get(k, "#969696") for k in p["dominant_pathway"]]
        draw_hubs(a, p, face, p["co2_matched"].to_numpy(), z=3.0)
        # 汇：在用的白心深蓝描边、面积 ∝ 注入量；未启用的小空心灰点，读者能数出"没匹配到的汇"。
        for keys, colour, lw, base, scale in ((list(idle), SINK_IDLE_C, 0.35, 2.6, 0.0),
                                              (list(used.index), SINK_EDGE, 0.45, 2.0, 26.0)):
            if not keys:
                continue
            hx, hy = to_map_xy(np.array([float(HUBS["longitude"][k]) for k in keys]),
                               np.array([float(HUBS["latitude"][k]) for k in keys]))
            off = np.array([bool(HUBS["offshore"].get(k, False)) for k in keys])
            vals = used.reindex(keys).fillna(0.0).to_numpy()
            size = base + scale * np.sqrt(vals / SINK_REF)
            for m, sel in (("o", ~off), ("s", off)):
                if sel.any():
                    a.scatter(hx[sel], hy[sel], s=size[sel], marker=m, facecolor="white",
                              edgecolor=colour, linewidth=lw, zorder=5)

    finish_map(fig, ax, business, f"CO$_2$：机组 → 管网 → 封存汇（{year} 年）", inset_rect)
    offshore_used = np.array([bool(HUBS["offshore"].get(k, False)) for k in used.index])
    return {"n_hub_matched": int(p["co2_matched"].sum()), "gw_matched": float(p.loc[p["co2_matched"], "cap_gw"].sum()),
            "gw_total": float(p["cap_gw"].sum()), "n_edge": len(flows), "flow": float(flows.sum()),
            "n_sink_used": len(used), "n_sink_total": len(HUBS), "inj": float(used.sum()),
            "inj_off": float(used[offshore_used].sum()), "captured": float(p["captured_mt"].sum())}


# =========================================================================================
# 面板 b：燃料链
# =========================================================================================
def panel_fuel(fig, ax, scen: str, year: int, inset_rect) -> dict:
    p = hub_table(scen, year)
    bio = flows_table(scen, "biomass_flows", year)
    bio = bio[bio["flow_gj"] > 1.0] if not bio.empty else bio
    nh3 = flows_table(scen, "ammonia_flows", year)
    nh3 = nh3[nh3["flow_kg"] > 1.0] if not nh3.empty else nh3
    if not nh3.empty:
        # ammonia_flows 不带坐标：从供给曲线补节点坐标，从机组表补机组坐标。
        curve = pd.read_csv(INPUTS / "ammonia_supply_curve.csv")
        curve.columns = [c.lstrip("﻿") for c in curve.columns]
        node_xy = curve.drop_duplicates("ammonia_node_id").set_index("ammonia_node_id")
        nh3["node_longitude"] = nh3["ammonia_node_id"].map(node_xy["longitude"])
        nh3["node_latitude"] = nh3["ammonia_node_id"].map(node_xy["latitude"])
        nh3["centroid_longitude"] = nh3["plant_id"].map(p["centroid_longitude"])
        nh3["centroid_latitude"] = nh3["plant_id"].map(p["centroid_latitude"])
        nh3 = nh3.dropna(subset=["node_longitude", "centroid_longitude"])
    bio_ref = float(bio["flow_gj"].quantile(0.95)) if not bio.empty else 1.0
    nh3_ref = float(nh3["flow_kg"].quantile(0.95)) if not nh3.empty else 1.0

    def business(a):
        # 每个 hub 平均从 ~40 个节点取料、运距 ≤ 200 km，近万条辐条只能画成半透明的"集料星"：
        # 星的半径就是集料半径，星的密度就是供给密度。单条辐条不必可辨。
        if not bio.empty:
            draw_spokes(a, bio["node_longitude"], bio["node_latitude"],
                        bio["centroid_longitude"], bio["centroid_latitude"],
                        bio["flow_gj"], BIO_C, bio_ref, z=2.5, lw_min=0.08, lw_max=0.55, alpha=0.32)
        if not nh3.empty:
            draw_spokes(a, nh3["node_longitude"], nh3["node_latitude"],
                        nh3["centroid_longitude"], nh3["centroid_latitude"],
                        nh3["flow_kg"], NH3_C, nh3_ref, z=2.6)
        face = [PATHWAY_COLORS.get(k, "#969696") for k in p["dominant_pathway"]]
        draw_hubs(a, p, face, (p["bio_matched"] | p["nh3_matched"]).to_numpy(), z=3.0)

    finish_map(fig, ax, business, f"燃料：生物质{'、氨' if not nh3.empty else ''} → 机组（{year} 年）", inset_rect)
    return {"n_hub_bio": int(p["bio_matched"].sum()), "gw_bio": float(p.loc[p["bio_matched"], "cap_gw"].sum()),
            "n_hub_nh3": int(p["nh3_matched"].sum()), "gw_nh3": float(p.loc[p["nh3_matched"], "cap_gw"].sum()),
            "n_bio_node": int(bio["biomass_node_id"].nunique()) if not bio.empty else 0,
            "bio_pj": float(bio["flow_gj"].sum()) / 1e6 if not bio.empty else 0.0,
            "bio_km": float((bio["flow_gj"] * bio["distance_km"]).sum() / bio["flow_gj"].sum()) if not bio.empty else float("nan"),
            "nh3_kt": float(nh3["flow_kg"].sum()) / 1e6 if not nh3.empty else 0.0,
            "gw_total": float(p["cap_gw"].sum())}


# =========================================================================================
# 面板 c：水链
# =========================================================================================
def panel_water(fig, ax, scen: str, year: int, inset_rect) -> dict:
    p = hub_table(scen, year)
    w = flows_table(scen, "water_flows", year)
    w = w[w["flow_m3"] > 1.0] if not w.empty else w
    if not w.empty:
        w["node_longitude"] = w["water_node_id"].astype(str).map(WATER_NODES["longitude"])
        w["node_latitude"] = w["water_node_id"].astype(str).map(WATER_NODES["latitude"])
        w = w.dropna(subset=["node_longitude"])
    ratio = basin_ratio(scen, year)
    basins = load_basins()
    w_ref = float(w["flow_m3"].quantile(0.95)) if not w.empty else 1.0

    def fill(a):
        if basins is None or ratio.empty:
            return
        b = basins.copy()
        b["ratio"] = b["code"].map(ratio)
        b.plot(ax=a, column="ratio", cmap=RATIO_CMAP, norm=RATIO_NORM, edgecolor="none",
               missing_kwds={"color": "#F1F3F5"}, zorder=0.5, alpha=0.85)

    def business(a):
        if not w.empty:
            draw_spokes(a, w["node_longitude"], w["node_latitude"],
                        w["centroid_longitude"], w["centroid_latitude"],
                        w["flow_m3"], WATER_C, w_ref, z=2.5, lw_max=1.1)
        face = [{"wet": WET_C, "air": AIR_C, "retire": RET_C}[k] for k in p["water_class"]]
        draw_hubs(a, p, face, (p["water_class"] == "wet").to_numpy(), z=3.0)

    finish_map(fig, ax, business, f"水：水源节点 → 机组，按流域（{year} 年）", inset_rect,
               facecolor="#FFFFFF", basins_fill=fill)
    cls = p.groupby("water_class")["cap_gw"].sum()
    n_cls = p["water_class"].value_counts()
    return {"gw_wet": float(cls.get("wet", 0.0)), "gw_air": float(cls.get("air", 0.0)),
            "gw_retire": float(cls.get("retire", 0.0)), "n_wet": int(n_cls.get("wet", 0)),
            "n_air": int(n_cls.get("air", 0)), "n_retire": int(n_cls.get("retire", 0)),
            "withdraw_e8": float(p["wat_e8"].sum()), "n_water_node": int(w["water_node_id"].nunique()) if not w.empty else 0,
            "ratio": ratio, "gw_total": float(p["cap_gw"].sum())}


# =========================================================================================
# 面板 d：同年匹配台账
# =========================================================================================
def ledger(year: int) -> pd.DataFrame:
    rows = []
    for scen, label, _, _ in CASES:
        p = hub_table(scen, year)
        tot = float(p["cap_gw"].sum())
        rows.append({"case": label, "chain": "CO$_2$ 管网", "matched": float(p.loc[p["co2_matched"], "cap_gw"].sum()),
                     "total": tot, "n": int(p["co2_matched"].sum()), "n_total": len(p)})
        fuel = p["bio_matched"] | p["nh3_matched"]
        rows.append({"case": label, "chain": "生物质 / 氨供给", "matched": float(p.loc[fuel, "cap_gw"].sum()),
                     "total": tot, "n": int(fuel.sum()), "n_total": len(p)})
        wet = p["water_class"] == "wet"
        air = p["water_class"] == "air"
        rows.append({"case": label, "chain": "湿冷取水", "matched": float(p.loc[wet, "cap_gw"].sum()),
                     "total": tot, "n": int(wet.sum()), "n_total": len(p),
                     "air": float(p.loc[air, "cap_gw"].sum()), "n_air": int(air.sum())})
    return pd.DataFrame(rows)


def panel_ledger(ax, year: int) -> pd.DataFrame:
    t = ledger(year)
    chains = ["CO$_2$ 管网", "生物质 / 氨供给", "湿冷取水"]
    dark = {"CO$_2$ 管网": PIPE_C, "生物质 / 氨供给": BIO_C, "湿冷取水": WET_C}
    light = "#E4E7EA"
    h = 0.30
    yticks, ylabels = [], []
    for i, chain in enumerate(chains):
        base_y = i * 1.25
        ax.text(0, base_y - 0.50, chain, fontsize=5.8, ha="left", va="center", color="#222222")
        for j, (_, label, _, _) in enumerate(CASES):
            r = t[(t["chain"] == chain) & (t["case"] == label)].iloc[0]
            y = base_y + (-0.17 if j == 0 else 0.17)
            ax.barh(y, r["total"], h, color=light, zorder=1)
            ax.barh(y, r["matched"], h, color=dark[chain], zorder=2, alpha=0.95 if j else 0.55)
            if chain == "湿冷取水":
                ax.barh(y, r["air"], h, left=r["matched"], color=AIR_C, zorder=2,
                        alpha=0.95 if j else 0.55)
                txt = f"{r['matched']:.0f} GW 取水（{r['n']} 个 hub）＋{r['air']:.0f} GW 空冷"
            else:
                txt = f"{r['matched']:.0f} GW，{r['n']} / {r['n_total']} 个 hub"
            ax.text(r["total"] + 8, y, txt, va="center", ha="left", fontsize=5.0, color="#333333")
            yticks.append(y)
            ylabels.append(label)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5.0, color="#555555")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(len(chains) * 1.25 - 0.55, -0.85)
    ax.set_xlim(0, float(t["total"].max()) * 1.85)
    ax.set_xlabel(f"{year} 年在役装机（GW）；浅灰 = 未匹配到该链", fontsize=5.8)
    ax.tick_params(axis="x", labelsize=5.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("同年匹配台账：每条链接住了多少机组", fontsize=6.4, pad=3)
    return t


# =========================================================================================
# 面板 e–g：链路量随年份
# =========================================================================================
def chain_series() -> pd.DataFrame:
    rows = []
    for scen, label, _, _ in CASES:
        for y in YEARS:
            p = hub_table(scen, y)
            inj = injection(scen, y)
            used = inj[inj > ACTIVE]
            off = np.array([bool(HUBS["offshore"].get(k, False)) for k in used.index])
            bio = flows_table(scen, "biomass_flows", y)
            rows.append({"case": label, "year": y,
                         "captured": float(p["captured_mt"].sum()),
                         "inj_on": float(used[~off].sum()) if len(used) else 0.0,
                         "inj_off": float(used[off].sum()) if len(used) else 0.0,
                         "bio_pj": float(bio["flow_gj"].sum()) / 1e6 if not bio.empty else 0.0,
                         "withdraw_e8": float(p["wat_e8"].sum()),
                         "air_gw": float(p["air_gw"].sum()),
                         "n_sink": int(len(used))})
    return pd.DataFrame(rows)


def panel_lines(ax, series: pd.DataFrame, cols: list[tuple[str, str, str]], ylabel: str, title: str) -> None:
    for scen, label, _, ls in CASES:
        s = series[series["case"] == label].set_index("year")
        for col, colour, name in cols:
            ax.plot(s.index, s[col], ls=ls, color=colour, lw=1.1, marker="o", ms=2.2,
                    label=f"{name}" if ls == "-" else None, zorder=3)
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=5.4)
    ax.tick_params(axis="y", labelsize=5.4)
    ax.set_ylabel(ylabel, fontsize=5.8)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=6.4, pad=3)
    ax.grid(axis="y", lw=0.25, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if len(cols) > 1:
        ax.legend(fontsize=5.0, frameon=False, loc="center right", handlelength=1.4,
                  bbox_to_anchor=(1.0, 0.42))


# =========================================================================================
# 组图
# =========================================================================================
def build_figure():
    apply_style()
    fig_w, fig_h = DOUBLE_COL[0], 7.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    left, right = 0.030, 0.985
    gap = 0.018
    cell_w = (right - left - 2 * gap) / 3.0
    map_h = 0.300
    map_bottom = 0.665
    inset = [0.795, 0.020, 0.150, 0.215]
    panels = (panel_co2, panel_fuel, panel_water)
    records = {}
    axes_map = []
    for i, fn in enumerate(panels):
        ax = fig.add_axes([left + i * (cell_w + gap), map_bottom, cell_w, map_h])
        records[fn.__name__] = fn(fig, ax, TREAT, MAP_YEAR, inset)
        panel_label(ax, "abc"[i], x=0.005, y=1.03, fontsize=8.5)
        axes_map.append(ax)

    # 地图图例，三张图各一条，放在地图下缘。
    lg_y = map_bottom + 0.012
    handles_a = [
        Line2D([], [], color=PIPE_C, lw=1.7, label=f"管段（线宽 ∝ 流量，满标度 {FLOW_REF:.0f} Mt yr$^{{-1}}$；三角 = 流向）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=SINK_EDGE,
               mew=0.45, markersize=3.4, label="在用封存汇（面积 ∝ 注入量；□ 海上）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=SINK_IDLE_C,
               mew=0.35, markersize=2.4, label="未启用的封存汇"),
        Line2D([], [], marker="o", color="none", markerfacecolor=PATHWAY_COLORS["ccs"],
               markeredgecolor="#1F2933", mew=0.3, markersize=3.2, label="机组（面积 ∝ 装机；描边 = 已接入）"),
    ]
    fig.legend(handles=handles_a, loc="upper left", bbox_to_anchor=(left, lg_y), fontsize=4.9,
               frameon=False, handlelength=1.5, handletextpad=0.45, labelspacing=0.28, ncol=1)
    pathway_handles = [Patch(facecolor=PATHWAY_COLORS[k], label=PATHWAY_LABELS[k])
                       for k in ("unabated", "biomass", "ccs", "beccs", "retire")]
    fig.legend(handles=pathway_handles, loc="upper left", bbox_to_anchor=(left + cell_w + gap, lg_y),
               fontsize=4.9, frameon=False, handlelength=1.1, handletextpad=0.4, labelspacing=0.28,
               ncol=2, columnspacing=0.9, title="a、b 机组填色 = 主导路径", title_fontsize=4.9)
    handles_b = [
        Line2D([], [], color=BIO_C, lw=1.3, label="生物质供给辐条（线宽 ∝ 供给量）"),
    ]
    if records["panel_fuel"]["nh3_kt"] > 0:
        handles_b.append(Line2D([], [], color=NH3_C, lw=1.3, label="氨供给辐条"))
    fig.legend(handles=handles_b, loc="upper left", bbox_to_anchor=(left + cell_w + gap, lg_y - 0.055),
               fontsize=4.9, frameon=False, handlelength=1.5, handletextpad=0.45, labelspacing=0.28)
    handles_c = [
        Line2D([], [], color=WATER_C, lw=1.1, label="取水辐条（线宽 ∝ 取水量）"),
        Patch(facecolor=WET_C, label="湿冷取水机组"),
        Patch(facecolor=AIR_C, label="已转空冷（含既有）"),
        Patch(facecolor=RET_C, label="退役"),
    ]
    fig.legend(handles=handles_c, loc="upper left", bbox_to_anchor=(left + 2 * (cell_w + gap), lg_y),
               fontsize=4.9, frameon=False, handlelength=1.3, handletextpad=0.45, labelspacing=0.28, ncol=2,
               columnspacing=0.9)
    # 流域填色色标
    cax = fig.add_axes([left + 2 * (cell_w + gap) + 0.012, lg_y - 0.062, cell_w - 0.03, 0.010])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=RATIO_CMAP, norm=RATIO_NORM, orientation="horizontal",
                                          spacing="uniform", ticks=RATIO_BOUNDS[:-1] + [RATIO_BOUNDS[-1]])
    cb.ax.set_xticklabels([f"{b:g}" for b in RATIO_BOUNDS[:-1]] + [">1.5"], fontsize=4.6)
    cb.ax.tick_params(length=1.5, pad=1)
    cb.set_label("流域电力取水 / 配额（配额 = 枯水期径流 × 0.20 × (1 - 0.85)）", fontsize=4.8, labelpad=1.5)
    cb.outline.set_linewidth(0.3)

    # 第二行
    row_top = 0.565
    row_h = 0.335
    ax_d = fig.add_axes([0.095, row_top - row_h, 0.345, row_h])
    ledger_table = panel_ledger(ax_d, MAP_YEAR)
    panel_label(ax_d, "d", x=-0.22, y=1.06, fontsize=8.5)

    series = chain_series()
    small_w = 0.122
    x_e = 0.510
    ax_e = fig.add_axes([x_e, row_top - row_h, small_w, row_h])
    ax_f = fig.add_axes([x_e + small_w + 0.050, row_top - row_h, small_w, row_h])
    ax_g = fig.add_axes([x_e + 2 * (small_w + 0.050), row_top - row_h, small_w, row_h])
    panel_lines(ax_e, series, [("captured", "black", "捕集"), ("inj_on", ON_C, "陆上注入"),
                                ("inj_off", OFF_C, "海上注入")], "Mt CO$_2$ yr$^{-1}$", "CO$_2$ 链")
    panel_lines(ax_f, series, [("bio_pj", BIO_C, "生物质")], "PJ yr$^{-1}$", "生物质供给")
    panel_lines(ax_g, series, [("withdraw_e8", WATER_C, "取水")], r"取水量（$10^8$ m$^3$）", "电力取水")
    for ax, letter in ((ax_e, "e"), (ax_f, "f"), (ax_g, "g")):
        panel_label(ax, letter, x=-0.30, y=1.06, fontsize=8.5)
    case_handles = [Line2D([], [], color="#333333", ls=ls, lw=1.1, label=label) for _, label, _, ls in CASES]
    fig.legend(handles=case_handles, loc="lower center", bbox_to_anchor=(0.74, row_top - row_h - 0.065),
               fontsize=5.2, frameon=False, ncol=2, handlelength=2.2, columnspacing=1.6)

    import textwrap
    text = "\n".join(textwrap.wrap(caption(records, ledger_table, series), width=118))
    fig.text(left, 0.010, text, fontsize=5.4, color=CAPTION_C, ha="left", va="bottom", linespacing=1.35)
    return fig, records, ledger_table, series


def caption(rec: dict, t: pd.DataFrame, s: pd.DataFrame) -> str:
    a, b, c = rec["panel_co2"], rec["panel_fuel"], rec["panel_water"]
    st = s[s["case"] == CASES[1][1]].set_index("year")
    sb = s[s["case"] == CASES[0][1]].set_index("year")
    nh3 = f"，氨供给 {b['nh3_kt']:.0f} kt" if b["nh3_kt"] > 0 else "，本批求解中氨供给为 0"
    over = [k for k, v in c["ratio"].items() if v > 1.0]
    over_txt = "无流域超出配额" if not over else f"超出配额的流域：{'、'.join(over)}"
    return (
        f"注：三张地图均为考虑水（s = 0.85）口径 {MAP_YEAR} 年。a：{a['n_hub_matched']} 个 hub（{a['gw_matched']:.0f} GW，"
        f"占在役 {a['gw_total']:.0f} GW 的 {100 * a['gw_matched'] / a['gw_total']:.0f}%）接入管网，"
        f"{a['n_edge']} 条管段输送 {a['flow']:.0f} Mt，{a['n_sink_used']} / {a['n_sink_total']} 个汇在用，"
        f"海上注入 {a['inj_off']:.0f} Mt（{100 * a['inj_off'] / max(a['inj'], 1e-9):.0f}%）。"
        f"b：{b['n_hub_bio']} 个 hub 从 {b['n_bio_node']} 个生物质节点取得 {b['bio_pj']:.0f} PJ，"
        f"供给量加权运距 {b['bio_km']:.0f} km{nh3}。"
        f"c：湿冷取水 {c['gw_wet']:.0f} GW / 已转空冷 {c['gw_air']:.0f} GW / 退役 {c['gw_retire']:.0f} GW，"
        f"取水 {c['withdraw_e8']:.1f} 亿 m$^3$，{over_txt}。"
        f"e–g：2030→2060 捕集 {st.loc[2030, 'captured']:.0f}→{st.loc[2060, 'captured']:.0f} Mt，"
        f"不考虑水为 {sb.loc[2030, 'captured']:.0f}→{sb.loc[2060, 'captured']:.0f} Mt；"
        f"取水 {st.loc[2030, 'withdraw_e8']:.1f}→{st.loc[2060, 'withdraw_e8']:.1f} 亿 m$^3$，"
        f"不考虑水为 {sb.loc[2030, 'withdraw_e8']:.1f}→{sb.loc[2060, 'withdraw_e8']:.1f}。"
        f"两种口径的地图格局在 ED13 中已证明肉眼不可分，故只画一种；差异见 d–g。"
    )


def main() -> None:
    fig, records, t, s = build_figure()
    save_fig(fig, "ed_fig14_multihop_matching", subdir="extended")
    print(f"ED14 - multihop matching, map year {MAP_YEAR}")
    for k, v in records.items():
        print(f"  {k}: " + ", ".join(f"{kk}={vv:.1f}" if isinstance(vv, float) else f"{kk}={vv}"
                                    for kk, vv in v.items() if kk != "ratio"))
    print("  basin ratio (treat):", {k: round(float(v), 2) for k, v in records["panel_water"]["ratio"].items()})
    print(t.to_string())
    print(s.to_string())


if __name__ == "__main__":
    main()
