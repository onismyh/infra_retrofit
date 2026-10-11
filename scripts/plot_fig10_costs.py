"""图 10  成本构成（求解结果）

图含义：目标函数的各成本类别归成十类（颜色同族：煤电灰、煤电的生物质氨水绿、工业红、管网与封存蓝；a、b 的颜色与 c 的
  类名对应）。
  a  各规划年的年度费用（$10^9$ 元/年，不折现）：运行、燃料与资源、运输、封存、碳成本、松弛惩罚，按类堆叠；某类合计为负
     （如煤电运行费相对不改造省下的多于多花的，省煤抵扣并在这一类里）画在零线以下。黑色短横线是各年净值，数字是净值，
     统一写在各栏最高的柱顶之上。
  b  各规划年的一次性投资（$10^9$ 元，不折现）：煤电改造、重建与搁浅资产，工业改造，管道。
  c  整个规划期的折现值（$10^9$ 元）按类合计，期末残值抵扣为负；各类相加即目标函数值。
读图注意：煤电按增量口径计（2026-10-10 起，与工业同口径）：运行费是相对"全部维持不改造运行"的差，可以为负；参照本身的
  净运行成本记在 result.json 的 coal_operating_reference_cny，不进目标、不在图上。年度费用乘区间年金与折现权重、一次性投资只乘
  折现才进目标（model_costs.add_year_costs），所以 a、b 的不折现值不能直接相加成 c。
数据：results/solved/<情景>/system.csv（逐年逐类别的求解器合计）、checks.csv，results/solved/<情景>.json。
自检：system.csv 的成本类别都归得了类；checks.csv 的 cost_attribution_closure 每年都通过（逐实体成本与求解器合计对得上）；
  折现值合计 = result.json 的 global_objective_cny。
输出：_indtree/results/figures/fig10_costs{,_en}.{pdf,png}
用法：python scripts/plot_fig10_costs.py [--scenario 情景] [--lang zh|en|both]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    COST_COLORS, COST_ORDER, FULL_WIDTH_MM, INK, MM,
    apply_style, check_close, figure_cli, fmt_number, langs, panel_label, read_result, read_result_json, save_fig,
    stacked_bars,
)

NAME = "fig10_costs"
SCALE = 1e9                  # 元 -> 10^9 元
# 目标函数的成本类别（model_costs 的 cost_exprs 键、salvage 的 salvage_credit）归成 COST_ORDER 的十类。
GROUPS = {
    "coal_operating": ("coal_operating_delta", "coal_savings_credit", "energy_penalty_cost", "ccs_om_cost",
                       "incremental_om"),
    "coal_capex": ("ccs_retrofit_capex", "blend_upgrade_capex", "air_retrofit_capex", "rebuild_capex", "stranded_capex"),
    "resources": ("biomass_cost", "ammonia_cost", "water_cost"),
    "industry_operating": ("industry_cost",),
    "industry_capex": ("industry_capex",),
    "network": ("transport_opex", "pipe_capex"),
    "storage": ("storage_cost",),
    "carbon": ("carbon_cost",),
    "slack": ("slack_penalty",),
    "salvage": ("salvage_credit",),
}
CATEGORY_GROUP = {category: group for group, categories in GROUPS.items() for category in categories}
TEXT = {
    "zh": {"a": "年度费用（$10^9$ 元/年）", "b": "一次性投资（$10^9$ 元）", "c": "折现值合计（$10^9$ 元）",
           "total": "合计 {v}",
           "groups": {"coal_operating": "煤电运行（相对不改造）", "coal_capex": "煤电改造、重建与搁浅",
                      "resources": "煤电购生物质、氨与水", "industry_operating": "工业运行（含买氢）",
                      "industry_capex": "工业改造投资", "network": "管网运维与投资", "storage": "封存",
                      "carbon": "碳成本", "slack": "松弛惩罚", "salvage": "期末残值抵扣"}},
    "en": {"a": "Annual cost (10$^9$ CNY yr$^{-1}$)", "b": "Investment (10$^9$ CNY)",
           "c": "Present value (10$^9$ CNY)", "total": "Total {v}",
           "groups": {"coal_operating": "Coal operation (vs. no retrofit)", "coal_capex": "Coal retrofit, rebuild, stranded",
                      "resources": "Coal biomass, ammonia, water", "industry_operating": "Industry operation (incl. hydrogen)",
                      "industry_capex": "Industry retrofit capex", "network": "Pipelines (O&M, capex)",
                      "storage": "Storage", "carbon": "Carbon cost", "slack": "Slack penalties",
                      "salvage": "Salvage credit"}},
}


def load(scenario: str) -> dict:
    return {"system": read_result(scenario, "system"), "checks": read_result(scenario, "checks"),
            "result": read_result_json(scenario)}


def grouped(system: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """a、b：年 × 类的不折现值（年度费用、一次性投资，10^9 元）；c：类 -> 规划期折现值合计（10^9 元）。"""
    frame = system.assign(group=system["category"].map(CATEGORY_GROUP))
    out: dict[str, pd.DataFrame | pd.Series] = {}
    for kind in ("annual", "one_off"):
        part = frame[frame["kind"] == kind]
        table = part.pivot_table(index="year", columns="group", values="cost_cny", aggfunc="sum") / SCALE
        table = table.reindex(index=sorted(frame["year"].unique()))
        out[kind] = table.reindex(columns=[g for g in COST_ORDER if g in table.columns]).fillna(0.0)
    out["present"] = frame.groupby("group")["cost_discounted_cny"].sum().reindex(COST_ORDER).fillna(0.0) / SCALE
    return out


def check(data: dict) -> None:
    system, checks = data["system"], data["checks"]
    unknown = set(system["category"]) - set(CATEGORY_GROUP)
    if unknown:
        raise ValueError(f"自检不通过：成本类别 {sorted(unknown)} 没有归类（GROUPS），图上会漏掉，不出图")
    closure = checks[checks["check_name"] == "cost_attribution_closure"]
    if closure.empty or (closure["status"] != "pass").any():
        raise ValueError("自检不通过：checks.csv 的 cost_attribution_closure 缺失或没通过，逐实体成本与求解器合计"
                         "对不上，不出图")
    check_close("折现值合计应等于 result.json 的 global_objective_cny", system["cost_discounted_cny"].sum(),
                data["result"]["global_objective_cny"], atol=1.0, rtol=1e-6)


def draw(data: dict, lang: str):
    text = TEXT[lang]
    tables = grouped(data["system"])
    fig = plt.figure(figsize=(FULL_WIDTH_MM * MM, 70 * MM))
    axes = [fig.add_axes(box) for box in ((0.07, 0.12, 0.22, 0.8), (0.37, 0.12, 0.16, 0.8), (0.79, 0.12, 0.2, 0.8))]
    for ax, kind, letter in ((axes[0], "annual", "a"), (axes[1], "one_off", "b")):
        table = tables[kind]
        x = np.arange(len(table))
        stacked_bars(ax, x, table, COST_COLORS, width=0.6)
        net = table.sum(axis=1).to_numpy()
        top = max(float(table.clip(lower=0).sum(axis=1).max()), 1e-9)
        bottom = min(0.0, float(table.clip(upper=0).sum(axis=1).min()))
        ax.hlines(net, x - 0.3, x + 0.3, colors="black", linewidth=0.8, zorder=5)
        for xi, value in zip(x, net):
            ax.annotate(fmt_number(value), (xi, top), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, color=INK)
        ax.set_ylim(bottom * 1.1, top * 1.15)
        ax.set_xticks(x, [str(y) for y in table.index])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(text[letter])
        panel_label(ax, letter)
    present = tables["present"]
    cx = axes[2]
    y = np.arange(len(present))[::-1]
    cx.barh(y, present.to_numpy(), height=0.62, color=[COST_COLORS[g] for g in present.index],
            edgecolor="none", zorder=3)
    cx.axvline(0.0, color=INK, linewidth=0.5, zorder=4)
    span = float(present.abs().max()) or 1.0
    for yi, value in zip(y, present.to_numpy()):
        cx.text(value + (0.03 if value >= 0 else -0.03) * span, yi, fmt_number(value), va="center",
                ha="left" if value >= 0 else "right", fontsize=5.5, color=INK)
    cx.set_yticks(y, [text["groups"][g] for g in present.index])
    cx.tick_params(axis="y", length=0)
    cx.spines["left"].set_visible(False)
    # 负值的数字写在柱端左侧，要留出写一个四位数的地方，否则压到类名
    left = 0.45 if float(present.min()) < 0 else 0.0
    cx.set_xlim(min(0.0, float(present.min())) - left * span, max(0.0, float(present.max())) + 0.3 * span)
    cx.set_xlabel(text["c"])
    cx.set_title(text["total"].format(v=fmt_number(float(present.sum()))), pad=4, loc="right")
    panel_label(cx, "c", x=-0.95)
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
