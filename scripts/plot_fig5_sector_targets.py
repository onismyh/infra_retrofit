"""图 5  部门排放与碳目标（求解结果）

图含义：四个目标组（电力、钢铁、水泥、化工）各规划年的排放。灰柱是冻结技术排放（当年利用小时或产量下不做任何
  减排时的排放），彩色柱是残余排放，黑色短横线是部门上限；残余排放高出上限的部分用红字标出。
  上限 = 碳目标轨迹给的比例 × 该组自身 2030 年的冻结技术排放（model_year._add_sector_targets）。
读图注意：电力的残余排放 = 冻结技术排放 − 逐厂减排量之和；工业按 hub 的 baseline − reduction 汇总到目标组
  （constants_industry.SECTOR_TARGET_GROUP）。超过上限的部分由带罚项的目标缺口变量承担，即该组目标没达到。
  结果里只记了上限的比例，2030 年的基线要从 2030 年的结果里取，所以 2030 必须是规划年，否则脚本报错。
数据：_indtree/results/<情景>/plant_detail.csv、industry_detail.csv，_indtree/results/<情景>.json。
自检：各组逐年的残余与冻结技术排放 = result.json 记的值；残余 − 上限 ≤ 目标缺口（模型约束）。
输出：_indtree/results/figures/fig5_sector_targets{,_en}.{pdf,png}
用法：python scripts/plot_fig5_sector_targets.py [--scenario 情景] [--lang zh|en|both]
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from coal_retrofit.constants_industry import POWER_TARGET_GROUP
from plot_style import (
    AIR_COLOR, GROUP_COLORS, GROUP_ORDER, MM, MT_CO2_YR,
    apply_style, check_close, figure_cli, fmt_number, labels, langs, read_result, read_result_json, save_fig,
)

NAME = "fig5_sector_targets"
BASE_YEAR = 2030             # 部门上限按各组 2030 年冻结技术排放的比例给
BASELINE_GREY = "#D9D9D9"
TEXT = {
    "zh": {"baseline": "冻结技术排放", "residual": "残余排放", "cap": "部门上限", "over": "超 {v}",
           "y": f"排放（{MT_CO2_YR}）"},
    "en": {"baseline": "Frozen-technology emissions", "residual": "Residual emissions", "cap": "Sector cap",
           "over": "+{v}", "y": f"Emissions ({MT_CO2_YR})"},
}


def load(scenario: str) -> dict:
    return {"plants": read_result(scenario, "plant_detail"), "industry": read_result(scenario, "industry_detail"),
            "result": read_result_json(scenario)}


def sector_table(data: dict) -> pd.DataFrame:
    """每组每年一行：冻结技术排放、残余排放、上限、目标缺口（Mt/yr）。"""
    plants, industry, years = data["plants"], data["industry"], data["result"]["years"]
    power = plants.groupby("year")[["baseline_emissions_mt", "reduction_mt"]].sum()
    frames = [pd.DataFrame({"group": POWER_TARGET_GROUP, "year": power.index,
                            "baseline": power["baseline_emissions_mt"],
                            "residual": power["baseline_emissions_mt"] - power["reduction_mt"]})]
    if len(industry):
        ind = industry.groupby(["target_group", "year"])[["baseline_co2_mt", "residual_mt"]].sum().reset_index()
        frames.append(ind.rename(columns={"target_group": "group", "baseline_co2_mt": "baseline",
                                          "residual_mt": "residual"}))
    table = pd.concat(frames, ignore_index=True)
    if BASE_YEAR not in set(table["year"]):
        raise ValueError(f"结果里没有 {BASE_YEAR} 年：部门上限 = 比例 × {BASE_YEAR} 年冻结技术排放，"
                         "缺这一年算不出上限，不出图")
    base = table[table["year"] == BASE_YEAR].set_index("group")["baseline"]
    table["cap"] = [years[str(y)]["sector_cap_fraction"][g] * base[g] for g, y in zip(table["group"], table["year"])]
    table["shortfall"] = [years[str(y)]["target_shortfall_by_group_mt"][g]
                          for g, y in zip(table["group"], table["year"])]
    return table


def check(data: dict, table: pd.DataFrame) -> None:
    unknown = set(table["group"]) - set(GROUP_ORDER)
    if unknown:
        raise ValueError(f"自检不通过：目标组 {sorted(unknown)} 不在 GROUP_ORDER 里，不出图")
    years = data["result"]["years"]
    power = table[table["group"] == POWER_TARGET_GROUP]
    check_close("电力残余排放应等于 result.json 的 coal_residual_mt", power["residual"],
                [years[str(y)]["coal_residual_mt"] for y in power["year"]], atol=1e-6)
    check_close("电力冻结技术排放应等于 result.json 的 coal_baseline_mt", power["baseline"],
                [years[str(y)]["coal_baseline_mt"] for y in power["year"]], atol=1e-6)
    industry = table[table["group"] != POWER_TARGET_GROUP]
    check_close("工业各组残余排放应等于 result.json 的 industry.residual_by_group_mt", industry["residual"],
                [years[str(y)]["industry"]["residual_by_group_mt"][g]
                 for g, y in zip(industry["group"], industry["year"])], atol=1e-6)
    industry_base = industry.groupby("year")["baseline"].sum()
    check_close("工业冻结技术排放合计应等于 result.json 的 industry.baseline_mt", industry_base,
                [years[str(y)]["industry"]["baseline_mt"] for y in industry_base.index], atol=1e-6)
    excess = table["residual"] - table["cap"] - table["shortfall"] - (1e-4 + 1e-7 * table["cap"].abs())
    if (excess > 0).any():
        worst = table.iloc[int(np.argmax(excess.to_numpy()))]
        raise ValueError(f"自检不通过：{worst['group']} {worst['year']} 年残余 − 上限 大于目标缺口，上限算错了"
                         "（模型约束 residual − shortfall ≤ cap），不出图")


def draw(table: pd.DataFrame, lang: str):
    lab, text = labels(lang), TEXT[lang]
    groups = [g for g in GROUP_ORDER if g in set(table["group"])]
    fig, axes = plt.subplots(1, len(groups), figsize=(183 * MM, 60 * MM), squeeze=False,
                             gridspec_kw={"wspace": 0.35})
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.8)
    for i, (ax, group) in enumerate(zip(axes[0], groups)):
        part = table[table["group"] == group].sort_values("year")
        x = np.arange(len(part))
        ax.bar(x, part["baseline"], width=0.72, color=BASELINE_GREY, edgecolor="none", zorder=2)
        ax.bar(x, part["residual"], width=0.42, color=GROUP_COLORS[group], edgecolor="none", zorder=3)
        ax.hlines(part["cap"], x - 0.36, x + 0.36, colors="black", linewidth=1.0, zorder=4)
        top = float(np.maximum(part["baseline"], part["cap"]).max())
        for xi, residual, over in zip(x, part["residual"], (part["residual"] - part["cap"]).to_numpy()):
            if over > 1e-3:           # 标在残余排放柱顶，超上限的量
                ax.annotate(text["over"].format(v=fmt_number(over)), (xi, residual), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=5.5, color=AIR_COLOR)
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(x, [str(y) for y in part["year"]])
        ax.tick_params(axis="x", length=0)
        ax.set_title(lab["group"][group], pad=4)
        if i == 0:
            ax.set_ylabel(text["y"])
    residual_handle = tuple(Patch(facecolor=GROUP_COLORS[g], edgecolor="none") for g in groups)
    # 含元组句柄（HandlerTuple 画成多色色块）；Figure.legend 的类型存根只认单个 Artist，所以标成 Any
    handles: list[Any] = [Patch(facecolor=BASELINE_GREY, edgecolor="none"), residual_handle,
                          Line2D([], [], color="black", linewidth=1.0)]
    fig.legend(handles, [text["baseline"], text["residual"], text["cap"]], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.4,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    return fig


def main(argv: list[str] | None = None) -> None:
    args = figure_cli(__doc__).parse_args(argv)
    data = load(args.scenario)
    table = sector_table(data)
    check(data, table)
    for lang in langs(args.lang):
        apply_style(lang)
        save_fig(draw(table, lang), NAME, lang)


if __name__ == "__main__":
    main()
