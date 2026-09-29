"""图 7  流域取水指标与空冷改造（求解结果）

图含义：
  a  各流域各规划年的取水指标利用率 = 煤电与工业的取水 ÷ 流域余量（官方用水总量控制指标扣掉非电用水，
     model_resources.add_basin_withdrawal_cap）。红格（> 100%）表示模型动用了带罚项的松弛、该流域指标没守住；
     "无余量"是余量 ≤ 0 而仍有取水的流域。本情景没启用流域上限时 a 只写一行说明。
  b  各规划年湿冷改空冷的容量（GW）= 装机 × 空冷份额（air_cooled_share）× 仍湿冷的比例（1 − already_air_share），
     逐厂相加；已是空冷的部分不重复计（plant_matrices._air_cooling_matrices）。
读图注意：a 是取水口径的制度约束，与节点耗水（resource_use.csv 的 water 行）不是一个量，不能相加。b 的空冷份额是
  各路径之和，其中退役路径上的一份在已装存量以内对用水与成本都没有作用、模型可以任取，后期年份的数字可能含已退役的改造。
数据：_indtree/results/<情景>/resource_use.csv、plant_detail.csv。
自检：利用率 = 取水 ÷ 余量（余量为正时，与结果表的 utilization 列一致）；流域代码都在 BASIN_ORDER 里；
  空冷份额与已空冷比例都在 [0, 1]。
输出：_indtree/results/figures/fig7_water{,_en}.{pdf,png}
用法：python scripts/plot_fig7_water.py [--scenario 情景] [--lang zh|en|both]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from plot_style import (
    AIR_COLOR, BASIN_ORDER, INK, MM, MUTED,
    apply_style, check_close, figure_cli, figure_label, fmt_number, labels, langs, read_result, save_fig,
)

NAME = "fig7_water"
OVER = 1.0 + 1e-4            # 利用率高于它算超指标（约束取紧时求解器给出的是 1 ± 容差）
# 利用率分档（上界）与颜色：四档蓝色由浅到深，超指标与无余量用红色（CLAUDE.md §3.3 取水 Blues、强调红）。
CLASS_EDGES = (0.5, 0.8, 0.95, OVER)
CLASS_COLORS = ("#DEEBF7", "#9ECAE1", "#4292C6", "#08519C", AIR_COLOR)
TEXT = {
    "zh": {"classes": ("≤ 50%", "50–80%", "80–95%", "95–100%", "> 100% 或无余量"), "none": "无余量",
           "off": "本情景没有启用流域取水上限", "b": "湿冷改空冷的容量（GW）", "a_title": "流域取水指标利用率"},
    "en": {"classes": ("≤ 50%", "50–80%", "80–95%", "95–100%", "> 100% or no quota"), "none": "no quota",
           "off": "Basin withdrawal caps are off in this scenario", "b": "Wet-to-air converted capacity (GW)",
           "a_title": "Utilisation of basin withdrawal quotas"},
}


def load(scenario: str) -> dict[str, pd.DataFrame]:
    resource = read_result(scenario, "resource_use")
    return {"basins": resource[resource["resource_type"] == "water_basin_quota"].astype({"region": str}),
            "plants": read_result(scenario, "plant_detail")}


def utilization(basins: pd.DataFrame) -> pd.DataFrame:
    """流域 × 年的利用率；余量 ≤ 0 而仍有取水记为 inf（无余量），两者都为零记 0。"""
    used, available = basins["used"].to_numpy(float), basins["available"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(available > 0, used / available, np.where(used > 1e-6, np.inf, 0.0))
    table = pd.DataFrame({"basin": basins["region"], "year": basins["year"], "ratio": ratio})
    table = table.pivot(index="basin", columns="year", values="ratio")
    return table.reindex([b for b in BASIN_ORDER if b in table.index])


def converted_gw(plants: pd.DataFrame) -> pd.Series:
    """年 -> 湿冷改空冷的容量（GW）。"""
    gw = plants["capacity_mw"] * plants["air_cooled_share"] * (1.0 - plants["already_air_share"]) / 1000.0
    return gw.groupby(plants["year"]).sum()


def check(data: dict[str, pd.DataFrame]) -> None:
    basins, plants = data["basins"], data["plants"]
    unknown = set(basins["region"]) - set(BASIN_ORDER)
    if unknown:
        raise ValueError(f"自检不通过：流域代码 {sorted(unknown)} 不在 BASIN_ORDER 里，不出图")
    positive = basins[basins["available"] > 0]
    check_close("利用率应等于 取水 ÷ 余量", positive["used"] / positive["available"], positive["utilization"],
                atol=1e-9)
    for col in ("air_cooled_share", "already_air_share"):
        if not plants[col].between(-1e-9, 1 + 1e-9).all():
            raise ValueError(f"自检不通过：plant_detail.{col} 超出 [0, 1]，不出图")


def draw(data: dict[str, pd.DataFrame], lang: str):
    lab, text = labels(lang), TEXT[lang]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(183 * MM, 72 * MM), gridspec_kw={"width_ratios": (1.45, 1.0),
                                                                                "wspace": 0.42})
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.9)
    table = utilization(data["basins"])
    if table.empty:
        ax.text(0.5, 0.5, text["off"], transform=ax.transAxes, ha="center", va="center", color=MUTED)
        ax.set_axis_off()
    else:
        classes = np.digitize(np.nan_to_num(table.to_numpy(float), nan=0.0), CLASS_EDGES, right=True)
        ax.imshow(classes, cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=len(CLASS_COLORS) - 0.5,
                  aspect="auto", interpolation="none")
        for (i, j), value in np.ndenumerate(table.to_numpy(float)):
            label = "—" if np.isnan(value) else text["none"] if np.isinf(value) else f"{value * 100:.0f}%"
            ax.text(j, i, label, ha="center", va="center", fontsize=6,
                    color="white" if classes[i, j] >= 2 else INK)
        ax.set_xticks(range(table.shape[1]), [str(y) for y in table.columns])
        ax.set_yticks(range(table.shape[0]), [lab["basin"][b] for b in table.index])
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(text["a_title"], pad=4)
        ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=name)
                           for c, name in zip(CLASS_COLORS, text["classes"])],
                  loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, handlelength=1.0)
    figure_label(fig, "a", 0.005, 0.99)

    gw = converted_gw(data["plants"])
    x = np.arange(len(gw))
    bx.bar(x, gw.to_numpy(), width=0.55, color=AIR_COLOR, edgecolor="none", zorder=3)
    for xi, value in zip(x, gw.to_numpy()):
        bx.annotate(fmt_number(value), (xi, value), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color=INK)
    bx.set_xticks(x, [str(y) for y in gw.index])
    bx.tick_params(axis="x", length=0)
    bx.set_ylim(0, max(float(gw.max()), 1e-9) * 1.15)
    bx.set_ylabel(text["b"])
    figure_label(fig, "b", bx.get_position().x0 - 0.075, 0.99)
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
