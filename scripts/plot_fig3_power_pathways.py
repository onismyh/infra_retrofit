"""图 3  煤电改造路径（求解结果）

图含义：
  a  各规划年煤电装机按改造路径的构成（GW）：装机 × 该路径份额，逐厂相加。份额是模型的决策变量，
     每个厂址 hub 各路径份额之和为 1，所以柱高各年相同；退役包括到设计寿命的退役与提前退役。
  b  各规划年煤电减排量按路径拆分（Mt CO2/yr），柱顶数字是合计。合计即模型的逐厂减排量之和，
     部门目标里电力的残余排放 = 冻结技术排放 − 这个合计（图 5）；退役一栏是退掉的份额原本会排的量。
读图注意：b 取结果表 pathway_shares.csv 的 abatement_mt：各路径 = 基线排放 × 份额 − 该路径的残余排放，按约束逐项
  拆分（含改造后利用小时提升与惩罚燃料），逐厂相加即模型的逐厂减排量。改造路径按提升后的利用小时排放，低比例掺烧
  可能比基线多排，这时该路径的减排为负，画在零线以下；"未改造"一栏是到期原址重建后效率提高少排的量减去空冷背压
  多排的量，通常很小。
数据：results/solved/<情景>/plant_detail.csv、pathway_shares.csv、sanity_checks.csv，results/solved/<情景>.json。
自检：结果表是按约束逐项拆分的（sanity_checks.csv 有 pathway_split_closure 行且通过；此前落盘的结果没有这一行，
  b 会画成旧口径，报错）；每个厂每年的路径份额之和为 1；逐厂各路径减排之和 = 该厂 reduction_mt（差额是份额合计为 1
  的可行性容差乘基线排放）；逐年合计 = result.json 的 coal_reduction_mt。
输出：_indtree/results/figures/fig3_power_pathways{,_en}.{pdf,png}
用法：python scripts/plot_fig3_power_pathways.py [--scenario 情景] [--lang zh|en|both]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    INK, MM, MT_CO2_YR, PATHWAY_COLORS, PATHWAY_ORDER,
    apply_style, check_close, figure_cli, fmt_number, labels, langs, legend_patches, panel_label, read_result,
    read_result_json, save_fig, stacked_bars,
)

NAME = "fig3_power_pathways"
TEXT = {
    "zh": {"a": "煤电装机（GW）", "b": f"减排量（{MT_CO2_YR}）"},
    "en": {"a": "Coal capacity (GW)", "b": f"Abatement ({MT_CO2_YR})"},
}


def load(scenario: str) -> dict:
    detail = read_result(scenario, "plant_detail").astype({"plant_id": str})
    pathways = read_result(scenario, "pathway_shares").astype({"plant_id": str})
    return {"detail": detail, "pathways": pathways, "sanity": read_result(scenario, "sanity_checks"),
            "result": read_result_json(scenario)}


def capacity_by_pathway(detail: pd.DataFrame) -> pd.DataFrame:
    """年 × 路径的装机（GW）：装机 × 份额，逐厂相加。"""
    gw = pd.DataFrame({k: detail["capacity_mw"] * detail[f"share_{k}"] / 1000.0 for k in PATHWAY_ORDER})
    return gw.groupby(detail["year"]).sum()


def abatement_by_pathway(pathways: pd.DataFrame) -> pd.DataFrame:
    """年 × 路径的减排量（Mt/yr），取 pathway_shares.csv 的 abatement_mt。"""
    table = pathways.pivot_table(index="year", columns="pathway", values="abatement_mt", aggfunc="sum")
    return table.reindex(columns=list(PATHWAY_ORDER)).fillna(0.0)


def check(data: dict) -> None:
    detail, pathways, result = data["detail"], data["pathways"], data["result"]
    closure = data["sanity"][data["sanity"]["check_name"] == "pathway_split_closure"]
    if closure.empty:
        raise ValueError("自检不通过：sanity_checks.csv 没有 pathway_split_closure 行，结果早于逐路径按约束拆分"
                         "（此前各路径的值与符号可能不对），重解后再画")
    if (closure["status"] != "pass").any():
        raise ValueError("自检不通过：逐路径的减排与捕集加起来对不上求解器的值（sanity_checks.csv 的 "
                         "pathway_split_closure），不出图")
    unknown = set(pathways["pathway"]) - set(PATHWAY_ORDER)
    if unknown:
        raise ValueError(f"自检不通过：路径 {sorted(unknown)} 不在 PATHWAY_ORDER 里，不出图")
    share_sum = detail[[f"share_{k}" for k in PATHWAY_ORDER]].sum(axis=1)
    check_close("每个厂每年的路径份额之和应为 1（模型约束 share_sum）", share_sum, np.ones(len(detail)), atol=1e-4)
    total = detail.set_index(["year", "plant_id"])["reduction_mt"]
    split = pathways.groupby(["year", "plant_id"])["abatement_mt"].sum().reindex(total.index)
    check_close("逐厂各路径减排之和应等于该厂 reduction_mt", split, total, atol=1e-3)
    years = sorted(detail["year"].unique())
    check_close("逐年煤电减排合计应等于 result.json 的 coal_reduction_mt",
                detail.groupby("year")["reduction_mt"].sum().reindex(years),
                [result["years"][str(y)]["coal_reduction_mt"] for y in years], atol=1e-6)


def draw(data: dict, lang: str):
    lab, text = labels(lang), TEXT[lang]
    capacity = capacity_by_pathway(data["detail"])
    abatement = abatement_by_pathway(data["pathways"])
    years = capacity.index.to_numpy()
    x = np.arange(len(years))
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 66 * MM), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.83)
    for ax, table, ylabel, letter in ((axes[0], capacity, text["a"], "a"), (axes[1], abatement, text["b"], "b")):
        stacked_bars(ax, x, table, PATHWAY_COLORS)
        ax.set_xticks(x, [str(y) for y in years])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    axes[0].set_ylim(0, capacity.sum(axis=1).max() * 1.05)
    top = abatement.clip(lower=0).sum(axis=1).to_numpy()
    axes[1].set_ylim(min(0.0, abatement.clip(upper=0).sum(axis=1).min() * 1.1), max(top.max(), 1e-9) * 1.12)
    for xi, y_top, total in zip(x, top, abatement.sum(axis=1).to_numpy()):
        axes[1].annotate(fmt_number(total), (xi, y_top), xytext=(0, 1.5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=6, color=INK)
    fig.legend(handles=legend_patches(PATHWAY_ORDER, PATHWAY_COLORS, lab["pathway"]), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(PATHWAY_ORDER), frameon=False)
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
