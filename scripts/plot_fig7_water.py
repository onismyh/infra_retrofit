"""图 7  流域取水指标与空冷改造（求解结果）

图含义：
  a  各流域各规划年的取水指标利用率 = 煤电与工业的取水 ÷ 流域余量。流域余量 = 用水总量控制指标 −（生活 + 农业
     + 生态 + 非电工业）+ 已建模工业的现状取水（builders/water_quota.write_basin_caps），约束见
     model_resources.add_basin_withdrawal_cap。深红格（> 100%）表示模型动用了带罚项的松弛、该流域指标没守住；
     "无余量"是余量 ≤ 0 而仍有取水的流域。本情景没启用流域上限时 a 只写一行说明。
  b  各规划年当年以空冷运行的改造容量（GW）= 装机 × 空冷运行份额（air_operating_share）× 仍湿冷的比例
     （1 − already_air_share），逐厂相加；已是空冷的部分不重复计（plant_matrices._air_cooling_matrices）。
读图注意：a 是取水口径的制度约束，与节点耗水（resource_use.csv 的 water 行）不是一个量，不能相加。b 是当年在运行的
  空冷容量，不是累计改造量：退役路径上的空冷份额不计入；已建成的空冷存量（plant_detail.csv 的 air_installed_share，
  只增不减，含此后退役的容量；已全空冷的 hub 除外：两列不保证为零、也不保证只增不减，乘 1 − already_air_share 后为零）
  这里不画。
数据：_indtree/results/<情景>/resource_use.csv、slack_detail.csv、plant_detail.csv。
自检：各流域超出余量的取水 = slack_detail.csv 记的该流域松弛（没超的流域没有松弛），两边各算各的；流域代码都在
  BASIN_ORDER 里；空冷运行份额与已空冷比例都在 [0, 1]。
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
    AIR_COLOR, BASIN_ORDER, INK, MM, MUTED, OVER_COLOR,
    apply_style, check_close, figure_cli, figure_label, fmt_number, labels, langs, read_result, save_fig,
)

NAME = "fig7_water"
OVER = 1.0 + 1e-4            # 利用率高于它算超指标（约束取紧时求解器给出的是 1 ± 容差）
# 利用率分档（上界）与颜色：四档蓝色由浅到深，超指标与无余量用深红（CLAUDE.md §3.3 取水 Blues；强调红留给 b 的空冷）。
CLASS_EDGES = (0.5, 0.8, 0.95, OVER)
CLASS_COLORS = ("#DEEBF7", "#9ECAE1", "#4292C6", "#08519C", OVER_COLOR)
# 中文版不用 ≤、–、—：SimHei 有没有这几个字没核实过（缺字时 save_fig 拒绝出图），≤ 走 mathtext，区间与缺值用连字符。
TEXT = {
    "zh": {"classes": ("$\\leq$ 50%", "50-80%", "80-95%", "95-100%", "> 100% 或无余量"), "none": "无余量", "nan": "-",
           "off": "本情景没有启用流域取水上限", "b": "当年以空冷运行的改造容量（GW）", "a_title": "流域取水指标利用率"},
    "en": {"classes": ("≤ 50%", "50–80%", "80–95%", "95–100%", "> 100% or no quota"), "none": "no quota", "nan": "—",
           "off": "Basin withdrawal caps are off in this scenario", "b": "Retrofitted capacity running air-cooled (GW)",
           "a_title": "Utilisation of basin withdrawal quotas"},
}


def load(scenario: str) -> dict[str, pd.DataFrame]:
    resource = read_result(scenario, "resource_use")
    slack = read_result(scenario, "slack_detail")      # 没有松弛时也带列名（只有表头）
    return {"basins": resource[resource["resource_type"] == "water_basin_quota"].astype({"region": str}),
            "slack": slack[slack["constraint_type"] == "water_basin_quota"],
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
    """年 -> 当年以空冷运行的改造容量（GW）。"""
    gw = plants["capacity_mw"] * plants["air_operating_share"] * (1.0 - plants["already_air_share"]) / 1000.0
    return gw.groupby(plants["year"]).sum()


def cell_label(value: float, text: dict) -> str:
    """a 的格子里写的字。超了指标但四舍五入是 100% 的写成 >100%，免得与守住的格子看不出分别。"""
    if np.isnan(value):
        return text["nan"]
    if np.isinf(value):
        return text["none"]
    if OVER < value < 1.005:
        return ">100%"
    return f"{value * 100:.0f}%"


def check(data: dict[str, pd.DataFrame]) -> None:
    basins, plants, slack = data["basins"], data["plants"], data["slack"]
    if "air_operating_share" not in plants.columns:
        raise ValueError("自检不通过：plant_detail.csv 没有 air_operating_share 列（2026-09-30 之前落盘，那时的"
                         " air_cooled_share 含退役路径上可任取的份额），重解后再画")
    unknown = set(basins["region"]) - set(BASIN_ORDER)
    if unknown:
        raise ValueError(f"自检不通过：流域代码 {sorted(unknown)} 不在 BASIN_ORDER 里，不出图")
    # 模型约束 取水 ≤ 余量 + 松弛，松弛带罚项取到最小：超出余量的取水就是该流域的松弛（slack_detail 只记正值）。
    # 容差 1 000 m³：求解按百万 m³ 计、开了数值缩放，可行性误差换回 m³ 不止 1。
    booked = {(int(y), str(code)): float(v)
              for y, code, v in zip(slack["year"], slack["node_id"], slack["slack_value"])}
    check_close("超出流域余量的取水应等于 slack_detail.csv 的流域松弛",
                (basins["used"] - basins["available"]).clip(lower=0.0),
                [booked.get((int(y), code), 0.0) for y, code in zip(basins["year"], basins["region"])], atol=1e3)
    for col in ("air_operating_share", "already_air_share"):
        if not plants[col].between(-1e-6, 1 + 1e-6).all():
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
            ax.text(j, i, cell_label(value, text), ha="center", va="center", fontsize=6,
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
