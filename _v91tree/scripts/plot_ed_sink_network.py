"""Extended Data: how the CO2 sinks are defined and how sources reach them.

WHY THIS FIGURE EXISTS. Everything downstream of "which plant can afford capture" runs through
two choices that were never drawn: what counts as a storage sink, and which pipeline routes the
optimiser is allowed to consider. Both were wrong in ways that pushed the same direction --
against CCS -- and both are now rebuilt from the source data and from method-literature rules.
This figure is the evidence for that rebuild, panel by panel.

(a) THE SINKS. The 5 km assessment rasters resolve into 103 contiguous geological storage bodies
    (33 deep saline, 70 oilfield). The model previously merged any two whose centres fell within
    200 km, which collapsed them to 35 hubs, relabelled 7 232 Mt of EOR capacity as deep saline
    -- deleting its 12 CNY/t revenue credit, because the merged hub takes the storage type of
    whichever component is larger -- and pushed the capacity-weighted haul from a plant to its
    nearest sink from 200 km to 234 km. 200 km corresponds to no physical scale: the CO2 plume
    from 1 Mt/yr injected for 30 years has a radius of 2-3 km.

(b) THE MERGE, PRICED. Sweeping the merge threshold shows what it buys and what it costs. Total
    storage and buildable injection rate are invariant to it -- injectivity is derived from
    candidate-site density (one project per 50 x 50 km at 2 Mt/yr, the NETL commercial-scale
    definition of >=50 Mt over 20-30 years) and pixel counts are additive under merging -- so the
    only thing the threshold changes is how far the CO2 has to travel.

(c) THE DETOUR FACTOR, MEASURED. Straight-line candidate edges need a terrain factor. The
    pipeline literature borrows 1.2-1.4 from international routes; this model has something
    better in its own corridor layer -- 62 segments of China's built oil and gas trunk network
    with real polyline geometry. They run 24 338 km along 21 418 km of straight line: a
    length-weighted factor of 1.136. Using it also removes an internal inconsistency, because
    corridor edges already carried routed length while triangulation edges carried the geodesic,
    which made new build look cheaper than corridor reuse by construction.

(d) CONNECTIVITY. The candidate network was filtered to be planar. Pipelines are not planar, and
    the filter severed terminals: 7 plants holding 20.4 GW could reach no sink at all, one of
    them 24.5 km from one, and 471 GW was routed at more than 1.5x its straight-line distance --
    P0077 sits 12.0 km from a sink and was routed 617.7 km. SimCCS treats source and sink
    connectivity as an invariant that survives however heavily the cost surface discourages
    crossings; the rebuild reproduces that by filtering first and then restoring the shortest
    dropped edges until nothing is stranded (3 edges, out of 651 removed), and by giving every
    plant direct arcs to its 5 nearest sinks at the 2.8x premium the assumptions already carried.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_style import MM, save_fig

ROOT = Path(__file__).resolve().parent.parent
# 已发布输入永远在主仓库 inputs/；重建输入在仓库内的隔离求解树 _v*tree/inputs/。
# 本脚本既可从主仓库运行（对比 _v9tree），也可从求解树的 scripts 副本运行（对比上级主仓库）。
# 以前把重建输入指向会话 scratchpad，那个目录 2026-08-30 被清空，脚本随之失效——不要再那样写。
_MAIN = ROOT.parent if ROOT.name.startswith("_v") else ROOT
OLD_INPUTS = _MAIN / "inputs"
NEW_INPUTS = ROOT / "inputs" if ROOT != _MAIN else _MAIN / "_v9tree" / "inputs"

DSA_C = "#4477AA"
EOR_C = "#CC6677"
OLD_C = "#B0B7BE"
FIX_C = "#117733"
BAD_C = "#CC3311"
DETOUR = 1.136


def _read(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / name, encoding="utf-8-sig")


def haversine_km(lon1, lat1, lon2, lat2):
    a = [np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2)]
    h = np.sin((a[3] - a[1]) / 2) ** 2 + np.cos(a[1]) * np.cos(a[3]) * np.sin((a[2] - a[0]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def detour_table(root: Path) -> pd.DataFrame:
    """Geodesic and network distance from each plant to its nearest sink."""
    nodes, edges = _read(root, "pipeline_nodes.csv"), _read(root, "pipeline_candidate_edges.csv")
    plants = _read(root, "plants.csv")
    gw = plants.set_index("plant_id")["total_capacity_mw"].div(1e3)
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.from_node_id), str(row.to_node_id), w=float(row.length_km))
    sinks = nodes[nodes["node_type"] == "storage_hub"]
    sink_ids = set(sinks["node_id"].astype(str))
    slon, slat = sinks["lon"].to_numpy(float), sinks["lat"].to_numpy(float)

    rows = []
    for row in nodes[nodes["node_type"] == "plant"].itertuples(index=False):
        node_id = str(row.node_id)
        net = np.nan
        if node_id in graph:
            lengths = nx.single_source_dijkstra_path_length(graph, node_id, weight="w")
            reach = [lengths[s] for s in sink_ids if s in lengths]
            if reach:
                net = min(reach)
        rows.append({"plant_id": row.plant_id, "gw": float(gw[row.plant_id]),
                     "geo": float(haversine_km(row.lon, row.lat, slon, slat).min()), "net": net})
    frame = pd.DataFrame(rows)
    frame["ratio"] = frame["net"] / frame["geo"]
    return frame


def panel_sinks(ax, old: pd.DataFrame, new: pd.DataFrame) -> None:
    import geopandas as gpd
    from plot_extended import _load_provinces, _reproj
    from plot_style import draw_china_basemap, add_scs_inset, mainland_extent

    provinces, _country = draw_china_basemap(ax, facecolor="#F7F8F9")

    def sized(frame: pd.DataFrame) -> np.ndarray:
        inj = (frame["pixel_count"] / 100.0 * 2.0).clip(upper=200.0).to_numpy(float)
        return 2.0 + 34.0 * np.sqrt(inj / max(inj.max(), 1e-9))

    old_pts = _reproj(old, "longitude", "latitude")
    new_pts = _reproj(new, "longitude", "latitude").assign(_s=sized(new))

    def _sink_layers(a) -> None:
        """汇图层。抽成函数是因为南海小图要用同样的样式和色标重画一遍（CLAUDE.md §4.3）。"""
        a.scatter(old_pts.geometry.x, old_pts.geometry.y, s=sized(old) * 2.4,
                  facecolor="none", edgecolor=OLD_C, lw=0.7, zorder=3)
        for kind, colour, marker in (("dsa", DSA_C, "o"), ("eor", EOR_C, "^")):
            sub = new_pts[new_pts["storage_type"] == kind]
            offshore = sub["offshore"].astype(bool)
            a.scatter(sub.geometry.x[~offshore], sub.geometry.y[~offshore],
                      s=sub["_s"][~offshore], c=colour, marker=marker, lw=0.25,
                      edgecolor="white", alpha=0.92, zorder=5)
            a.scatter(sub.geometry.x[offshore], sub.geometry.y[offshore], s=sub["_s"][offshore],
                      facecolor="white", marker=marker, lw=0.7, edgecolor=colour, zorder=5)

    _sink_layers(ax)

    x0, y0, x1, y1 = mainland_extent(ax)
    ax.set_axis_off()
    # 这张图有海上封存汇，南海范围内可能落点，所以小图要重画汇图层而不只是底图。
    add_scs_inset(ax.get_figure(), ax, draw=_sink_layers)
    sx, sy = x0 + 0.06 * (x1 - x0), y0 + 0.10 * (y1 - y0)
    ax.plot([sx, sx + 500_000], [sy, sy], color="#333333", lw=1.0, zorder=20, solid_capstyle="butt")
    ax.text(sx + 250_000, sy - 0.018 * (y1 - y0), "500 km", ha="center", va="top",
            fontsize=5.0, color="#333333", zorder=20)

    handles = [
        Line2D([], [], marker="o", ls="", mfc=DSA_C, mec="white", ms=4.2,
               label=f"深部咸水层（{int((new['storage_type'] == 'dsa').sum())} 个）"),
        Line2D([], [], marker="^", ls="", mfc=EOR_C, mec="white", ms=4.2,
               label=f"油田 / EOR（{int((new['storage_type'] == 'eor').sum())} 个）"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=DSA_C, mew=0.8, ms=4.2,
               label=f"海上（{int(new['offshore'].astype(bool).sum())} 个）"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=OLD_C, mew=0.8, ms=6.0,
               label=f"200 km 合并汇（{len(old)} 个）"),
    ]
    # 图例放左下、比例尺之上：右下被南海小图占住，中下压在华南海岸线和海南的汇上，
    # 只有青藏区域没有任何汇，是这张图唯一的空白。
    ax.legend(handles=handles, fontsize=5.0, frameon=False, loc="lower left",
              bbox_to_anchor=(0.02, 0.12), handlelength=1.0, handletextpad=0.4,
              labelspacing=0.30, borderpad=0.2)
    # The map axes are taller than the drawn landmass, so a default title floats at the top of
    # the page above the panel letter. y pins it to the axes box instead.
    n_off = int(new["offshore"].astype(bool).sum())
    ax.set_title(f"{len(new)} 个汇，而不是 {len(old)} 个按距离合并的汇\n"
                 f"（陆上 {len(new) - n_off} 个封存体、海上 {n_off} 个盆地级汇）；\n"
                 "标记面积代表可建设的注入能力",
                 fontsize=6.8, linespacing=1.25, pad=2.0, y=0.965)


def panel_merge_cost(ax, sweep: pd.DataFrame) -> None:
    ax.plot(sweep["threshold_km"], sweep["haul_km"], color=DSA_C, lw=1.5, marker="o", ms=4.0,
            mec="white", mew=0.6, zorder=4)
    for row in sweep.itertuples():
        ax.annotate(f"{int(row.hubs)}", xy=(row.threshold_km, row.haul_km), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=5.0, color="#555555")
    for x, colour, label in ((200.0, OLD_C, "已发表版本"), (0.0, FIX_C, "重建版本")):
        row = sweep[sweep["threshold_km"] == x].iloc[0]
        ax.plot([x], [row["haul_km"]], marker="o", ms=7.0, mfc="none", mec=colour, mew=1.4,
                zorder=6)
        # 阈值 0 的点贴着 x 轴且紧邻 50 km 点的计数标签：下方压刻度、右上压 "63"，
        # 只能引线到更高处。
        if x > 0:
            ax.annotate(label, xy=(x, row["haul_km"]), xytext=(6, -11), textcoords="offset points",
                        fontsize=5.4, color=colour, fontweight="bold")
        else:
            ax.annotate(label, xy=(x, row["haul_km"]), xytext=(26, 30), textcoords="offset points",
                        fontsize=5.4, color=colour, fontweight="bold", ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=colour, shrinkA=0, shrinkB=4))
    ax.set_xlabel("汇的合并阈值（km）", fontsize=6.4)
    ax.set_ylabel("到最近汇的容量加权距离\n（km，直线）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("合并汇能减少节点数，代价是运距变长；它完全\n"
                 "不改变总封存量，也不改变可建设注入能力",
                 fontsize=6.8, linespacing=1.25)


def panel_detour_factor(ax, corridors: pd.DataFrame) -> None:
    ax.axhspan(1.2, 1.4, color="#DDAA33", alpha=0.16, lw=0, zorder=1)
    ax.annotate("文献中管道绕行系数的\n取值范围，1.2–1.4", xy=(30, 1.30),
                fontsize=5.0, color="#8A6D1F", linespacing=1.15, va="center")
    ax.scatter(corridors["chord"], corridors["tort"],
               s=1.5 + 16.0 * corridors["chord"] / corridors["chord"].max(),
               c=DSA_C, lw=0.2, edgecolor="white", alpha=0.85, zorder=4)
    ax.axhline(DETOUR, color=FIX_C, lw=1.2, zorder=5)
    ax.annotate(f"长度加权 {DETOUR:.3f}\n（本研究采用）", xy=(corridors["chord"].max(), DETOUR),
                xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=5.4,
                color=FIX_C, fontweight="bold", linespacing=1.15)
    ax.axhline(1.0, color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=3)
    # Parked in the empty corner above the literature band; at the dashed line itself it sat on
    # top of the cloud of segments it describes.
    ax.annotate("已发表版本：所有直线候选边\n一律取 1.000",
                xy=(corridors["chord"].min() * 1.9, 1.005),
                xytext=(corridors["chord"].min() * 1.05, 1.44),
                ha="left", fontsize=5.0, color="#777777", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#AAAAAA"))
    ax.set_xscale("log")
    ax.set_xlabel("管段直线长度（km，对数轴）", fontsize=6.4)
    ax.set_ylabel("实际路由长度 / 直线长度", fontsize=6.4)
    ax.set_ylim(0.97, 1.62)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"绕行系数是实测的，不是借来的：中国已建油气干线的\n"
                 f"{len(corridors)} 条管段，合计 21 418 km",
                 fontsize=6.8, linespacing=1.25)


def panel_connectivity(ax, old: pd.DataFrame, new: pd.DataFrame) -> None:
    grid = np.linspace(1.0, 6.0, 400)

    def curve(frame: pd.DataFrame) -> np.ndarray:
        ratio = frame["ratio"].to_numpy(float)
        gw = frame["gw"].to_numpy(float)
        return np.array([gw[np.nan_to_num(ratio, nan=np.inf) > g].sum() for g in grid])

    ax.plot(grid, curve(old), color=BAD_C, lw=1.6, zorder=4, label="已发表版本")
    ax.plot(grid, curve(new), color=FIX_C, lw=1.6, zorder=5, label="重建版本")
    ax.axvline(1.5, color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=2)

    # Split the two failures rather than reporting their sum: 471 GW is routed the long way,
    # a further 20.4 GW has no route at all. Adding them gives 492 and hides the second.
    routed_bad = float(old.loc[old["ratio"] > 1.5, "gw"].sum())
    old_bad = float(old.loc[np.nan_to_num(old["ratio"], nan=np.inf) > 1.5, "gw"].sum())
    ax.annotate(f"有 {routed_bad:.0f} GW 的路由长度\n超过其直线距离的 1.5 倍",
                xy=(1.5, old_bad), xytext=(1.80, 790),
                fontsize=5.2, color=BAD_C, linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=BAD_C))
    orphan_gw = float(old.loc[old["net"].isna(), "gw"].sum())
    n_orphan = int(old["net"].isna().sum())
    tail = float(old.loc[np.nan_to_num(old["ratio"], nan=np.inf) > 5.2, "gw"].sum())
    ax.annotate(f"曲线永远到不了零：{n_orphan} 个厂址、{orphan_gw:.1f} GW\n"
                f"与任何汇都没有通路，其中一个距最近的汇仅 24.5 km",
                xy=(5.15, tail), xytext=(2.35, 118),
                fontsize=5.2, color=BAD_C, linespacing=1.15, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=BAD_C))
    ax.annotate("重建后：所有路由都落在 1.136 的绕行系数上，\n"
                "因此曲线在该点之后为空；且它从全部机组起算，\n"
                "因为已不存在任何“裸大地线”候选边",
                xy=(1.16, 900), xytext=(1.98, 1120), fontsize=5.2, color=FIX_C,
                linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=FIX_C))

    ax.set_xlim(1.0, 5.2)
    ax.set_ylim(0, max(curve(old).max() * 1.20, 1.0))
    ax.set_xlabel("网络路由长度 / 到最近汇的直线距离", fontsize=6.4)
    ax.set_ylabel("绕行至少达到该倍数的\n容量（GW）", fontsize=6.4)
    ax.tick_params(labelsize=5.6, length=1.8)
    ax.grid(lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Bottom-left: the only region both curves leave empty. At the top right it sat under the
    # green callout.
    ax.legend(fontsize=5.4, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.06),
              handlelength=1.4, handletextpad=0.5)
    ax.set_title("管道并不具有平面性。强行施加平面性使 7 个厂址\n"
                 "失去通路，并让三分之一的机组绕了远路",
                 fontsize=6.8, linespacing=1.25)


def merge_sweep() -> pd.DataFrame:
    """Capacity-weighted haul to the nearest sink as the merge threshold varies."""
    sys.path.insert(0, str(ROOT / "src"))
    from coal_retrofit.builders.storage import _cluster_storage_nodes

    raw = _read(NEW_INPUTS, "storage_hubs.csv")
    plants = _read(OLD_INPUTS, "plants.csv")
    plon = plants["centroid_longitude"].to_numpy(float)
    plat = plants["centroid_latitude"].to_numpy(float)
    gw = plants["total_capacity_mw"].to_numpy(float) / 1e3

    rows = []
    for threshold in (0.0, 50.0, 100.0, 200.0, 300.0, 400.0):
        hubs = raw if threshold == 0 else _cluster_storage_nodes(raw.copy(), max_distance_km=threshold)
        hlon, hlat = hubs["longitude"].to_numpy(float), hubs["latitude"].to_numpy(float)
        geo = np.array([haversine_km(a, b, hlon, hlat).min() for a, b in zip(plon, plat)])
        rows.append({"threshold_km": threshold, "hubs": len(hubs),
                     "haul_km": float((geo * gw).sum() / gw.sum())})
    return pd.DataFrame(rows)


def corridor_tortuosity() -> pd.DataFrame:
    from shapely import wkt

    edges = _read(OLD_INPUTS, "pipeline_candidate_edges.csv")
    main = edges[edges["edge_class"] == "existing_main_corridor"]
    rows = []
    for row in main.itertuples(index=False):
        coords = list(wkt.loads(row.geometry_wkt).coords)
        chord = float(haversine_km(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]))
        if chord > 5:
            rows.append({"chord": chord, "routed": float(row.length_km),
                         "tort": float(row.length_km) / chord})
    return pd.DataFrame(rows)


def main() -> None:
    if not NEW_INPUTS.exists():
        raise SystemExit(f"rebuilt inputs not found at {NEW_INPUTS}")
    old_sinks, new_sinks = _read(OLD_INPUTS, "storage_hubs.csv"), _read(NEW_INPUTS, "storage_hubs.csv")
    old_route, new_route = detour_table(OLD_INPUTS), detour_table(NEW_INPUTS)
    sweep, corridors = merge_sweep(), corridor_tortuosity()

    fig = plt.figure(figsize=(183 * MM, 156 * MM))
    ax_map = fig.add_axes([0.030, 0.520, 0.455, 0.400])
    ax_sweep = fig.add_axes([0.590, 0.560, 0.375, 0.335])
    ax_tort = fig.add_axes([0.085, 0.075, 0.370, 0.320])
    ax_conn = fig.add_axes([0.590, 0.075, 0.375, 0.320])

    panel_sinks(ax_map, old_sinks, new_sinks)
    panel_merge_cost(ax_sweep, sweep)
    panel_detour_factor(ax_tort, corridors)
    panel_connectivity(ax_conn, old_route, new_route)

    for ax, lab, dx, dy in ((ax_map, "a", -0.060, 1.115), (ax_sweep, "b", -0.135, 1.235),
                            (ax_tort, "c", -0.145, 1.235), (ax_conn, "d", -0.135, 1.235)):
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=8.0, fontweight="bold",
                va="top", ha="left")

    save_fig(fig, "ed_fig10_sink_network", subdir="extended")

    print(f"  sinks {len(old_sinks)} -> {len(new_sinks)}  "
          f"(dsa {int((new_sinks['storage_type'] == 'dsa').sum())}, "
          f"eor {int((new_sinks['storage_type'] == 'eor').sum())})")
    print(f"  storage {old_sinks['storage_all_mt'].sum() / 1e3:.1f} -> "
          f"{new_sinks['storage_all_mt'].sum() / 1e3:.1f} Gt")
    for tag, frame in (("as published", old_route), ("rebuilt", new_route)):
        bad = float(frame.loc[np.nan_to_num(frame["ratio"], nan=np.inf) > 1.5, "gw"].sum())
        print(f"  {tag:14s} orphans {int(frame['net'].isna().sum())} "
              f"({frame.loc[frame['net'].isna(), 'gw'].sum():.1f} GW), "
              f">1.5x detour {bad:.1f} GW, max ratio {frame['ratio'].max():.2f}")
    print(f"  corridor detour factor: length-weighted "
          f"{corridors['routed'].sum() / corridors['chord'].sum():.3f} over {len(corridors)} segments")


if __name__ == "__main__":
    main()
