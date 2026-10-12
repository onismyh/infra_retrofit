"""图 8  工业减排量与捕集量（求解结果）

图含义：各工业部门（面板 a 起各一个）在各规划年按路线的减排量（Mt CO2/yr，柱按路线堆叠：CCS、氢路线），黑色短横线
  是当年的 CO2 捕集量（进管网、去封存的量）。各部门纵轴各自取范围。
读图注意：减排量 = 该路线的减排系数 × 份额，逐 hub 相加（results_sources._build_source_route_table）；CCS 的减排量比
  捕集量小，差额是燃煤锅炉供再生蒸汽放空的 CO2（industry_matrices，余热供汽的份额不排）；氢路线不捕集。合成氨、甲醇的 CCS 只捕集纯流股，所以图 4 的 CCS 产量份额与这里的
  捕集量不成比例。部门排放与碳目标的对照见图 5。
数据：results/solved/<情景>/sources.csv、source_routes.csv（source_type 为 industry 的行），results/solved/<情景>.json。
自检：结果里有工业 hub，逐路线表的 hub 都在 sources.csv 里；逐 hub 各路线的减排与捕集之和 = sources.csv 该 hub 的
  reduction_co2_mtpa、captured_co2_mtpa；逐年合计 = result.json 的
  industry.reduction_mt、industry.captured_mt；未改造路线没有减排（图上不画它）；路线都在 ROUTE_ORDER 里，部门都在
  SECTOR_ORDER 里。
输出：_indtree/results/figures/fig8_industry_abatement{,_en}.{pdf,png}
用法：python scripts/plot_fig8_industry_abatement.py [--scenario 情景] [--lang zh|en|both]
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_io import check_close, figure_cli, langs, read_result, read_result_json
from plot_style import (
    FULL_WIDTH_MM, MM, MT_CO2_YR, ROUTE_COLORS, ROUTE_ORDER, SECTOR_ORDER,
    apply_style, labels, legend_patches, panel_label, save_fig, stacked_bars,
)

NAME = "fig8_industry_abatement"
ABATING = tuple(r for r in ROUTE_ORDER if r != "unabated")   # 有减排的路线：CCS、氢路线
TEXT = {
    "zh": {"y": f"减排量与捕集量（{MT_CO2_YR}）", "captured": "捕集量"},
    "en": {"y": f"Abatement and capture ({MT_CO2_YR})", "captured": "Captured"},
}


def load(scenario: str) -> dict:
    sources, routes = read_result(scenario, "sources"), read_result(scenario, "source_routes")
    hubs = sources[sources["source_type"] == "industry"].astype({"source_id": str})
    routes = routes[routes["source_type"] == "industry"].astype({"source_id": str})
    return {"hubs": hubs, "routes": routes, "result": read_result_json(scenario)}


def by_sector(data: dict) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    """部门 -> {"reduction": 年 × 路线的减排量, "captured": 年 -> 捕集量}（Mt/yr）；部门按 SECTOR_ORDER。"""
    routes = _with_sector(data)
    out: dict[str, dict[str, Any]] = {}
    for sector in [s for s in SECTOR_ORDER if s in set(routes["sector"])]:
        part = routes[routes["sector"] == sector]
        reduction = part.pivot_table(index="year", columns="route", values="reduction_co2_mtpa", aggfunc="sum")
        out[sector] = {"reduction": reduction.reindex(columns=list(ABATING)).fillna(0.0),
                       "captured": part.groupby("year")["captured_co2_mtpa"].sum()}
    return out


def _with_sector(data: dict) -> pd.DataFrame:
    """逐路线表并上 hub 的部门（sources.csv）。"""
    return data["routes"].merge(data["hubs"][["year", "source_id", "sector"]], on=["year", "source_id"],
                                how="left", validate="many_to_one")


def check(data: dict) -> None:
    hubs, routes, years = data["hubs"], data["routes"], data["result"]["years"]
    if hubs.empty:
        raise ValueError("结果里没有工业 hub，图 8 没有可画的，不出图")
    if _with_sector(data)["sector"].isna().any():
        raise ValueError("自检不通过：source_routes.csv 有 sources.csv 里没有的工业 hub，图上会漏画，不出图")
    unknown = (set(routes["route"]) - set(ROUTE_ORDER)) | (set(hubs["sector"]) - set(SECTOR_ORDER[1:]))
    if unknown:
        raise ValueError(f"自检不通过：路线或部门 {sorted(unknown)} 不在 ROUTE_ORDER / SECTOR_ORDER 里，不出图")
    unabated = routes.loc[routes["route"] == "unabated", "reduction_co2_mtpa"]
    check_close("未改造路线的减排应为 0（图上不画这一路线）", unabated, np.zeros(len(unabated)), atol=1e-9)
    total = hubs.set_index(["year", "source_id"])
    split = routes.groupby(["year", "source_id"])[["reduction_co2_mtpa", "captured_co2_mtpa"]].sum()
    split = split.reindex(total.index)
    for column in ("reduction_co2_mtpa", "captured_co2_mtpa"):
        check_close(f"逐 hub 各路线的 {column} 之和应等于 sources.csv 的值", split[column], total[column], atol=1e-6)
    by_year = hubs.groupby("year")[["reduction_co2_mtpa", "captured_co2_mtpa"]].sum()
    check_close("逐年工业减排合计应等于 result.json 的 industry.reduction_mt", by_year["reduction_co2_mtpa"],
                [years[str(y)]["industry"]["reduction_mt"] for y in by_year.index], atol=1e-6)
    check_close("逐年工业捕集合计应等于 result.json 的 industry.captured_mt", by_year["captured_co2_mtpa"],
                [years[str(y)]["industry"]["captured_mt"] for y in by_year.index], atol=1e-6)


def draw(data: dict, lang: str):
    lab, text = labels(lang), TEXT[lang]
    sectors = by_sector(data)
    fig, axes = plt.subplots(1, len(sectors), figsize=(FULL_WIDTH_MM * MM, 60 * MM), squeeze=False,
                             gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.8)
    for i, (ax, (sector, parts)) in enumerate(zip(axes[0], sectors.items())):
        reduction, captured = parts["reduction"], parts["captured"].reindex(parts["reduction"].index)
        x = np.arange(len(reduction))
        stacked_bars(ax, x, reduction, ROUTE_COLORS, width=0.6)
        ax.hlines(captured.to_numpy(), x - 0.36, x + 0.36, colors="black", linewidth=1.0, zorder=4)
        top = max(float(reduction.clip(lower=0).sum(axis=1).max()), float(captured.max()), 1e-9)
        bottom = min(0.0, float(reduction.clip(upper=0).sum(axis=1).min()) * 1.1)
        ax.set_ylim(bottom, top * 1.12)
        ax.set_xticks(x, [str(y) for y in reduction.index])
        ax.tick_params(axis="x", length=0)
        ax.set_title(lab["sector"][sector], pad=4)
        panel_label(ax, "abcdefg"[i])
        if i == 0:
            ax.set_ylabel(text["y"])
    # 路线色块加一条黑线（捕集量）；Figure.legend 的类型存根只认 Artist 列表，所以标成 Any
    handles: list[Any] = legend_patches(ABATING, ROUTE_COLORS, lab["route"])
    handles.append(Line2D([], [], color="black", linewidth=1.0, label=text["captured"]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
    return fig


def main(argv: list[str] | None = None) -> None:
    args = figure_cli(__doc__).parse_args(argv)
    data = load(args.scenario)
    check(data)
    for lang in langs(args.lang):
        apply_style(lang)
        save_fig(draw(data, lang), NAME, lang)


if __name__ == "__main__":
    main()
