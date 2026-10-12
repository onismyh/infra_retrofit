"""图 9  CO2 封存按汇型与陆海汇总（求解结果）

图含义：
  a  各规划年的年注入量（Mt CO2/yr），按汇型（深部咸水层、驱油封存）与陆上 / 海上堆叠，海上加深色斜线；黑色短横线是
     全国合计的年注入上限（部署上限，result.json 的 storage_national_injection_mtpa；不设上限的情景不画），柱顶是合计。
  b  到各规划区间末的累计封存量（Gt CO2），同样堆叠，柱顶是合计；横轴是区间的起始规划年（2060 一栏含 2060 年代表的
     整个末区间的注入）。
读图注意：年注入量是该规划年代表整个区间的年量；累计量 = 各汇封存容量 − 区间末剩余容量（sinks.csv 的
  cumulative_injected_mt），即逐区间注入量 × 区间年数之和。陆上 / 海上取 storage_hubs.csv 的海上标记（sinks.csv 的 offshore）。
数据：results/solved/<情景>/sinks.csv、sources.csv，results/solved/<情景>.json。
自检：每年 封存注入 = 煤电捕集 + 工业捕集（管网节点守恒，同图 6）；全国合计不超过部署上限；汇型只有 dsa / eor，海上标记
  只有 True / False。
输出：_indtree/results/figures/fig9_storage{,_en}.{pdf,png}
用法：python scripts/plot_fig9_storage.py [--scenario 情景] [--lang zh|en|both]
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_io import check_close, figure_cli, langs, read_result, read_result_json
from plot_style import (
    FULL_WIDTH_MM, INK, MM, MT_CO2_YR, SINK_COLORS,
    apply_style, fmt_number, labels, legend_patches, panel_label, save_fig, stacked_bars,
)

NAME = "fig9_storage"
# 堆叠顺序（自下而上）：汇型 × 陆海；海上加深色斜线。
CLASSES = tuple((kind, offshore) for kind in SINK_COLORS for offshore in (False, True))
HATCH = "//////"
TEXT = {
    "zh": {"a": f"年注入量（{MT_CO2_YR}）", "b": "区间末累计封存量（Gt CO$_2$）", "cap": "全国年注入上限",
           "onshore": "{sink}，陆上", "offshore": "{sink}，海上"},
    "en": {"a": f"Annual injection ({MT_CO2_YR})", "b": "Stored by period end (Gt CO$_2$)", "cap": "National injection cap",
           "onshore": "{sink}, onshore", "offshore": "{sink}, offshore"},
}


def load(scenario: str) -> dict:
    sources = read_result(scenario, "sources")
    return {"sinks": read_result(scenario, "sinks"),
            "captured": sources.groupby("year")["captured_co2_mtpa"].sum(),
            "result": read_result_json(scenario)}


def key(kind: str, offshore: bool) -> str:
    return f"{kind}_{'offshore' if offshore else 'onshore'}"


def summary(sinks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """年 × 汇型陆海类的年注入量（Mt/yr）与区间末累计封存量（Gt），列按 CLASSES 的顺序。"""
    classes = sinks["storage_type"].astype(str) + "_" + np.where(sinks["offshore"].astype(bool), "offshore", "onshore")
    columns = [key(kind, offshore) for kind, offshore in CLASSES]
    out = {}
    for name, column, scale in (("annual", "injected_mtpa", 1.0), ("cumulative", "cumulative_injected_mt", 1e-3)):
        table = sinks.assign(cls=classes).pivot_table(index="year", columns="cls", values=column, aggfunc="sum")
        out[name] = table.reindex(columns=columns).fillna(0.0) * scale
    return out


def national_cap(result: dict, years) -> np.ndarray:
    """逐年的全国年注入上限（Mt/yr）；不设上限（null）记 nan。"""
    caps = [result["years"][str(y)].get("storage_national_injection_mtpa") for y in years]
    return np.array([np.nan if c is None else float(c) for c in caps])


def check(data: dict) -> None:
    sinks = data["sinks"]
    if not sinks["storage_type"].isin(list(SINK_COLORS)).all():
        raise ValueError("自检不通过：封存汇类型只应是 dsa / eor（CLAUDE.md §1.3），不出图")
    if not sinks["offshore"].isin([True, False]).all():
        raise ValueError("自检不通过：sinks.csv 的 offshore 只应是 True / False，陆海分不出来，不出图")
    injected = sinks.groupby("year")["injected_mtpa"].sum()
    check_close("每年 封存注入 应等于 煤电捕集 + 工业捕集（管网节点守恒）", injected,
                data["captured"].reindex(injected.index), atol=1e-3)
    caps = national_cap(data["result"], injected.index)
    over = injected.to_numpy() - caps
    if (np.nan_to_num(over, nan=-np.inf) > 1e-3).any():
        raise ValueError("自检不通过：全国合计年注入量超过部署上限（model_year 的 national_injection_limit 是硬约束），不出图")


def draw(data: dict, lang: str):
    lab, text = labels(lang), TEXT[lang]
    tables = summary(data["sinks"])
    colors = {key(kind, offshore): SINK_COLORS[kind] for kind, offshore in CLASSES}
    hatches = {key(kind, True): HATCH for kind in SINK_COLORS}
    names = {key(kind, offshore): text["offshore" if offshore else "onshore"].format(sink=lab["sink"][kind])
             for kind, offshore in CLASSES}
    caps = national_cap(data["result"], tables["annual"].index)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_MM * MM, 70 * MM), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.76)
    for ax, (name, table), ylabel, letter in zip(axes, tables.items(), (text["a"], text["b"]), "ab"):
        years = table.index.to_numpy()
        x = np.arange(len(years))
        stacked_bars(ax, x, table, colors, hatches=hatches)
        totals = table.sum(axis=1).to_numpy()
        top = max(float(totals.max()), 1e-9)
        if name == "annual" and np.isfinite(caps).any():
            shown = np.isfinite(caps)
            ax.hlines(caps[shown], x[shown] - 0.36, x[shown] + 0.36, colors="black", linewidth=1.0, zorder=4)
            top = max(top, float(np.nanmax(caps)))
        for xi, total in zip(x, totals):
            ax.annotate(fmt_number(total), (xi, total), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, color=INK)
        ax.set_ylim(0, top * 1.12)
        ax.set_xticks(x, [str(y) for y in years])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    # 色块加一条黑线（上限）；Figure.legend 的类型存根只认 Artist 列表，所以标成 Any
    handles: list[Any] = legend_patches([key(kind, offshore) for kind, offshore in CLASSES], colors, names,
                                        hatches=hatches)
    if np.isfinite(caps).any():
        handles.append(Line2D([], [], color="black", linewidth=1.0, label=text["cap"]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
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
