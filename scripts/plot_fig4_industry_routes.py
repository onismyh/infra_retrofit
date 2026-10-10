"""图 4  工业减排路线（求解结果）

图含义：各工业部门在各规划年的路线构成（%）：未改造、CCS、氢路线，按当年产量加权到部门。CCS 与氢路线只计改造的
  点源（结果表的 `abatable_production_share`；混合原料的甲醇 hub 里份额为 0 的焦炉煤气制、天然气制点源不改造，计入
  未改造，2026-10-02 起；此前的结果没有这一列，按 1 取）。氢路线在长流程钢铁
  是氢直接还原，在合成氨、甲醇是绿氢替代或补充原料氢；电炉钢与水泥没有氢路线（constants_industry.SECTOR_HAS_H2_ROUTE）。
读图注意：份额由模型的决策变量（每个 hub 各路线份额之和为 1）按改造的产量加权，当年产量 = 基年产量 × 产量指数；份额不是减排量，
  各部门排放与碳目标的对照见图 5。
数据：results/solved/<情景>/sources.csv（source_type 为 industry 的行；产量即 activity，kt/yr）。
自检：每个 hub 每年的路线份额之和为 1；部门都在 SECTOR_ORDER 里；产量为正；改造点源的产量比例在 [0, 1] 内（有这一列时）。
输出：_indtree/results/figures/fig4_industry_routes{,_en}.{pdf,png}
用法：python scripts/plot_fig4_industry_routes.py [--scenario 情景] [--lang zh|en|both]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    MM, ROUTE_COLORS, ROUTE_ORDER, SECTOR_ORDER,
    apply_style, check_close, figure_cli, labels, langs, legend_patches, read_result, save_fig, stacked_bars,
)

NAME = "fig4_industry_routes"
TEXT = {"zh": {"y": "产量份额（%）"}, "en": {"y": "Share of output (%)"}}


def load(scenario: str) -> pd.DataFrame:
    sources = read_result(scenario, "sources")
    return sources[sources["source_type"] == "industry"]


def route_shares(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """部门 -> 年 × 路线的产量加权份额（%）：路线份额乘改造的产量，不改造的点源的产量计入未改造。"""
    out = {}
    for sector in [s for s in SECTOR_ORDER if s in set(detail["sector"])]:
        part = detail[detail["sector"] == sector]
        rebuilt = part["activity"] * part.get("abatable_production_share", 1.0)
        weighted = pd.DataFrame({r: rebuilt * part[f"share_{r}"] for r in ROUTE_ORDER})
        weighted["unabated"] += part["activity"] - rebuilt
        sums = weighted.groupby(part["year"]).sum()
        out[sector] = sums.div(part.groupby("year")["activity"].sum(), axis=0) * 100.0
    return out


def check(detail: pd.DataFrame) -> None:
    unknown = set(detail["sector"]) - set(SECTOR_ORDER)
    if unknown:
        raise ValueError(f"自检不通过：部门 {sorted(unknown)} 不在 SECTOR_ORDER 里，不出图")
    if not (detail["activity"] > 0).all():
        raise ValueError("自检不通过：有 hub 的产量不为正，按产量加权会失真，不出图")
    if "abatable_production_share" in detail and not detail["abatable_production_share"].between(0.0, 1.0).all():
        raise ValueError("自检不通过：改造点源的产量比例缺失或不在 [0, 1] 内，按它加权会失真，不出图")
    share_sum = detail[[f"share_{r}" for r in ROUTE_ORDER]].sum(axis=1)
    check_close("每个 hub 每年的路线份额之和应为 1", share_sum, np.ones(len(detail)), atol=1e-4)


def draw(detail: pd.DataFrame, lang: str):
    lab, text = labels(lang), TEXT[lang]
    shares = route_shares(detail)
    fig, axes = plt.subplots(1, len(shares), figsize=(183 * MM, 58 * MM), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.8)
    for i, (ax, (sector, table)) in enumerate(zip(axes[0], shares.items())):
        x = np.arange(len(table))
        stacked_bars(ax, x, table[list(ROUTE_ORDER)], ROUTE_COLORS, width=0.66)
        ax.set_xticks(x, [str(y) for y in table.index])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(lab["sector"][sector], pad=4)
        if i == 0:
            ax.set_ylabel(text["y"])
        else:
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
    fig.legend(handles=legend_patches(ROUTE_ORDER, ROUTE_COLORS, lab["route"]), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(ROUTE_ORDER), frameon=False)
    return fig


def main(argv: list[str] | None = None) -> None:
    args = figure_cli(__doc__).parse_args(argv)
    detail = load(args.scenario)
    check(detail)
    for lang in langs(args.lang):
        apply_style(lang)
        save_fig(draw(detail, lang), NAME, lang)


if __name__ == "__main__":
    main()
