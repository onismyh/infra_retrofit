# -*- coding: utf-8 -*-
"""ind_ed1 — 联合目标定在什么水平才会动员工业。

`IND_*` 一共四个运行，两个目标水平 × 水约束开关：

    f = 0.5914   把 2060 年的**绝对**减排量固定在煤电单独建模时的 5 122 Mt
    f = 0.95     让联合目标真正咬住（8 228 Mt）

这张图存在的理由是一个容易被误读的事实：**在 0.5914 上，联合目标根本不是约束。**
煤电一家在经济最优处就减到 6 000 Mt 量级（BECCS 的减排因子是 1.05，可以减过基线），
已经越过 0.5914 × 8 661 = 5 122 Mt。所以那一对运行里工业不动，**不是因为工业没有减排空间，
而是因为没人要求它减**。把"工业不减排"写成结论，前提是先说清目标水平。

三个面板：

    a  2060 年联合减排量 vs 当年目标要求，四个运行  —— 目标咬没咬住
    b  2060 年工业减排量与绿氢份额，四个运行        —— 工业被动员的程度
    c  目标份额数轴：经济最优点在哪，两个水平在哪  —— 阈值本身

面板 c 的阈值是从**无水、目标不咬**的那个运行读出来的：那时煤电的减排量就是它的经济最优，
阈值 = 该减排量 / 联合基线。阈值以下目标是惰性的，以上才开始花钱。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from plot_style import (  # noqa: E402
    MM,
    RESULTS_DIR,
    apply_style,
    cjk_fill,
    panel_label,
    save_fig,
)

YEAR = 2060
# 顺序即面板里的顺序：先按目标水平分组，组内先无水后有水。
# 第三项标记这一支是否关掉了水约束——阈值只能从**无水**且目标不咬的那一支读，
# 有水那一支的减排量被流域上限扭过，不是经济最优。
RUNS: tuple[tuple[str, str, bool], ...] = (
    ("IND_BASE", "f = 0.5914\n水约束关", True),
    ("IND_WA_cwatm_126_dry_oq", "f = 0.5914\n水约束开", False),
    ("IND_BASE_t95", "f = 0.95\n水约束关", True),
    ("IND_WA_cwatm_126_dry_oq_t95", "f = 0.95\n水约束开", False),
)

COAL_C = "#636363"
IND_C = "#00A087"
TARGET_C = "#CC3311"
H2_C = "#9E9AC8"
IDLE_C = "#C8CDD2"
GRID = "#E4E7EA"


def _require_registered(runs) -> None:
    """IND_ 系已下线：明确停下并说明怎么重画旧图，而不是在读登记表时抛 KeyError。"""
    from run_single import EXPERIMENTS

    missing = [run for run in runs if run not in EXPERIMENTS]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} 已不在情景登记表：IND_ 系（单一联合目标）于 ba967c1 删除，只剩 ST_。"
            "这些结果是 2026-09-22 之前的旧成本口径，新代码既不能重解，也不该用新口径的成本函数去标注。"
            "重画旧图请在 cf073be 的工作副本里、_indtree/ 下运行本脚本；ST_ 版需另行设计。"
        )


def target_fraction(run: str) -> float:
    """该运行的末年联合目标份额，直接读 `scripts/run_single.py` 的登记表。"""
    from run_single import EXPERIMENTS

    scenario_kw, _ = EXPERIMENTS[run]
    return float(scenario_kw.get("emission_target_fraction", (0.0, 0.0, 0.0, 0.95))[-1])


def load(run: str, water_off: bool) -> dict | None:
    """一个运行在 YEAR 年的联合分配。未求解返回 None，让图自己说明缺哪一格。"""
    meta_path = RESULTS_DIR / f"{run}.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    block = meta["years"][str(YEAR)]
    ind = block.get("industry") or {}
    detail = pd.read_csv(RESULTS_DIR / run / "industry_detail.csv")
    detail = detail[detail["year"] == YEAR]
    plants = pd.read_csv(RESULTS_DIR / run / "plant_detail.csv")
    coal_baseline = float(plants[plants["year"] == plants["year"].min()]
                          ["baseline_emissions_mt"].sum())
    ind_baseline = float(ind.get("baseline_mt", detail["baseline_co2_mt"].sum()))
    joint_baseline = coal_baseline + ind_baseline
    fraction = target_fraction(run)
    weight = detail["baseline_co2_mt"]
    return {
        "run": run,
        "coal": float(block["coal_reduction_mt"]),
        "industry": float(ind.get("reduction_mt", 0.0)),
        "shortfall": float(block["target_shortfall_mt"]),
        "required": fraction * joint_baseline,
        "fraction": fraction,
        "joint_baseline": joint_baseline,
        "h2_share": float((detail["share_h2"] * weight).sum() / max(float(weight.sum()), 1e-9)),
        "water_off": water_off,
        "objective": float(meta["global_objective_cny"]),
        "gap": float(meta["solver_quality"].get("mip_gap", float("nan"))),
    }


# =========================================================================================
def panel_a(ax, rows: list[dict | None], labels: list[str]) -> None:
    xs = np.arange(len(rows), dtype=float)
    coal = np.array([r["coal"] if r else 0.0 for r in rows])
    ind = np.array([r["industry"] if r else 0.0 for r in rows])
    req = np.array([r["required"] if r else np.nan for r in rows])

    ax.bar(xs, coal, width=0.52, color=COAL_C, label="煤电机队", zorder=3)
    ax.bar(xs, ind, width=0.52, bottom=coal, color=IND_C, label="工业点源", zorder=3)
    for xi, rv in zip(xs, req):
        if np.isnan(rv):
            continue
        ax.plot([xi - 0.34, xi + 0.34], [rv, rv], lw=1.5, color=TARGET_C, zorder=5,
                solid_capstyle="butt")
    ax.plot([], [], lw=1.5, color=TARGET_C, label=f"{YEAR} 年目标要求")

    for xi, r in zip(xs, rows):
        if r is None:
            ax.text(xi, 200.0, "未求解", fontsize=5.6, color="#8A9199", ha="center",
                    va="bottom", rotation=90)
            continue
        slack = r["coal"] + r["industry"] - r["required"]
        # 富余量比"缺口"更能说明问题：缺口恒为 0（模型不许缺），富余量才区分
        # "目标恰好咬住"与"目标根本没起作用"。
        ax.text(xi, r["coal"] + r["industry"] + 130.0,
                ("恰好咬住" if abs(slack) < 5.0 else f"富余 {slack:,.0f}"),
                fontsize=5.4, color=("#333333" if abs(slack) < 5.0 else TARGET_C),
                ha="center", va="bottom")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5.6, linespacing=1.35)
    ax.set_ylabel(r"2060 年减排量（Mt CO$_2$ yr$^{-1}$）", fontsize=6.2, labelpad=2.0)
    ax.set_ylim(0, max(float(np.nanmax(coal + ind)), float(np.nanmax(req))) * 1.20)
    _clean(ax)
    ax.legend(fontsize=5.4, loc="upper left", frameon=False, handlelength=1.0,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)


def panel_b(ax, rows: list[dict | None], labels: list[str]) -> None:
    xs = np.arange(len(rows), dtype=float)
    ind = np.array([r["industry"] if r else 0.0 for r in rows])
    h2 = np.array([r["h2_share"] if r else 0.0 for r in rows])

    ax.bar(xs, ind, width=0.52, color=IND_C, zorder=3)
    ax.set_ylabel(r"工业减排量（Mt CO$_2$ yr$^{-1}$）", fontsize=6.2, labelpad=2.0)
    ax.set_ylim(0, max(float(ind.max()), 1.0) * 1.28)
    # 柱值靠左、菱形与百分比靠右。两者都居中时，0.5914 有水那一格的 53 Mt 会正好被
    # 1.8% 的菱形盖住——两个量在这一格里恰好落到同一高度，而菱形画在 twinx 上，永远压在上面。
    for xi, v in zip(xs, ind):
        ax.text(xi - 0.30, v + float(ind.max()) * 0.02, f"{v:,.0f}", fontsize=5.4,
                color=IND_C, ha="right", va="bottom")

    # 绿氢份额画在第二根轴上：它是比例，与 Mt 不同量纲，共用一根轴会骗人。
    ax2 = ax.twinx()
    ax2.plot(xs + 0.16, h2 * 100.0, ls="none", marker="D", ms=3.4, color=H2_C, zorder=6)
    for xi, v in zip(xs, h2):
        ax2.text(xi + 0.30, v * 100.0, f"{v:.0%}", fontsize=5.4, color=H2_C,
                 ha="left", va="center")
    ax2.set_ylabel("绿氢份额（按基线排放加权）", fontsize=6.2, labelpad=3.0, color=H2_C)
    ax2.set_ylim(-4, 108)
    ax2.set_yticks([0, 50, 100])
    ax2.set_yticklabels(["0", "50%", "100%"], fontsize=5.8, color=H2_C)
    ax2.tick_params(axis="y", colors=H2_C, labelsize=5.8)
    for side in ("top", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(H2_C)
    ax2.spines["right"].set_linewidth(0.6)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5.6, linespacing=1.35)
    _clean(ax)
    ax.legend(handles=[Patch(facecolor=IND_C, label="工业减排量（左轴）"),
                       Line2D([], [], ls="none", marker="D", ms=3.4, color=H2_C,
                              label="绿氢份额（右轴）")],
              fontsize=5.4, loc="upper left", frameon=False, handlelength=1.0,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)


def panel_c(ax, rows: list[dict | None]) -> float | None:
    """目标份额数轴：经济最优点在哪，两个已跑水平在哪。"""
    solved = [r for r in rows if r is not None]
    if not solved:
        ax.text(0.5, 0.5, "四个运行都未求解", transform=ax.transAxes, ha="center", va="center",
                fontsize=6.0, color="#8A9199")
        ax.set_axis_off()
        return None
    joint = solved[0]["joint_baseline"]
    # 阈值只能从**无水**且目标不咬的那一支读：那时的减排量就是经济最优，目标在它以下是惰性的。
    # 有水那一支的减排量被流域上限改写过，拿它当经济最优会把阈值算偏。
    inert = [r for r in solved
             if r["water_off"] and r["coal"] + r["industry"] - r["required"] > 5.0]
    threshold = (max(r["coal"] + r["industry"] for r in inert) / joint) if inert else None

    ax.axhline(0.0, color="#5A5A5A", lw=0.8, zorder=3)
    if threshold is not None:
        ax.axvspan(0.0, threshold, color=IDLE_C, alpha=0.55, zorder=1)
        ax.axvline(threshold, color=TARGET_C, lw=0.9, ls=(0, (3, 2)), zorder=4)
        ax.text(threshold, 0.62, f"经济最优 {threshold:.3f}\n（目标在此以下是惰性的）",
                fontsize=5.4, color=TARGET_C, ha="center", va="bottom", linespacing=1.35)
        ax.text(threshold / 2.0, -0.62, "目标不起作用\n工业只为水而动",
                fontsize=5.4, color="#6B7177", ha="center", va="top", linespacing=1.35)
        ax.text((threshold + 1.0) / 2.0, -0.62, "目标开始花钱\n工业被动员",
                fontsize=5.4, color="#333333", ha="center", va="top", linespacing=1.35)

    for fraction in sorted({r["fraction"] for r in solved}):
        ax.plot([fraction], [0.0], marker="o", ms=5.0, mfc="white", mec="#333333", mew=0.9,
                zorder=6)
        ax.text(fraction, 0.16, f"f = {fraction:g}", fontsize=5.6, color="#333333",
                ha="center", va="bottom")

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_xlabel(f"末年联合减排目标份额 f（分母 = 煤电 + 工业基线 {joint:,.0f} Mt）",
                  fontsize=6.2, labelpad=1.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    return threshold


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.tick_params(axis="x", length=0)


# =========================================================================================
def main() -> None:
    apply_style()
    _require_registered([run for run, _, _ in RUNS])
    rows = [load(run, water_off) for run, _, water_off in RUNS]
    labels = [label for _, label, _ in RUNS]
    missing = [run for (run, _, _), r in zip(RUNS, rows) if r is None]

    # 95 mm 而不是 112：112 mm 时面板 c 下方会空出 28 mm，图注够不到、面板也用不上，
    # 排版社缩放后只是一块浪费的版面。
    fig = plt.figure(figsize=(183 * MM, 95 * MM))
    # 右缘停在 0.895 而不是 0.94：面板 b 的第二根轴把刻度与轴标题画在轴外，
    # 顶到 0.94 时整幅会被 bbox_inches="tight" 撑到 188.5 mm，越过双栏守卫。
    ax_a = fig.add_axes([0.075, 0.500, 0.375, 0.450])
    ax_b = fig.add_axes([0.585, 0.500, 0.310, 0.450])
    # 数轴面板画的是一条线，但线上下各有两行注记（阈值在上、两个区间的说明在下），
    # 压得太扁时下面那两行会跨到 x 轴脊上。
    ax_c = fig.add_axes([0.075, 0.200, 0.820, 0.150])

    panel_a(ax_a, rows, labels)
    panel_b(ax_b, rows, labels)
    threshold = panel_c(ax_c, rows)
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        panel_label(ax, letter, x=-0.13, y=1.08)

    tail = (f"阈值 {threshold:.3f} 由无水且目标不咬的运行读出：那时的减排量就是经济最优。"
            if threshold is not None else
            "阈值需要一个无水且目标不咬的运行才能读出，本批尚缺。")
    miss = f"缺 {len(missing)} 个未求解运行（{', '.join(missing)}）。" if missing else ""
    fig.text(0.004, 0.004, cjk_fill(
        f"四个 `IND_*` 运行，v9 管网，{YEAR} 年横截面。"
        f"f = 0.5914 把绝对减排量固定在煤电单独建模时的 5 122 Mt；f = 0.95 让目标真正咬住。"
        f"{tail}"
        f"面板 a 的红杠是当年目标要求，柱高是达成量，两者之差即富余；缺口在四个运行里恒为 0。"
        f"{miss}"
        # 注意：不要在图上写 U+26A0（⚠）。SimHei 没有这个字形，save_fig 的缺字守卫会拒绝出图。
        f"注意：不同 f 之间的目标函数不可相减——目标本身变了，那是两个不同的问题。", 108),
        fontsize=5.0, color="#555555", ha="left", va="bottom", linespacing=1.45)

    save_fig(fig, "ind_ed1_target_level", "extended")
    report(rows, threshold)


def report(rows: list[dict | None], threshold: float | None) -> None:
    for r in rows:
        if r is None:
            continue
        print(f"  {r['run']:32s} f={r['fraction']:.4f}  煤电 {r['coal']:7.1f}  "
              f"工业 {r['industry']:7.1f}  要求 {r['required']:7.1f}  "
              f"富余 {r['coal'] + r['industry'] - r['required']:+8.1f}  "
              f"H2 {r['h2_share']:5.1%}  obj {r['objective']:.6e}  gap {r['gap']:.4f}")
    if threshold is not None:
        print(f"  目标开始咬住的阈值 f* = {threshold:.4f}")


if __name__ == "__main__":
    main()
