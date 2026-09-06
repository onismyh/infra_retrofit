"""Generate schematic figures for patent application (zhuanli.md).

This script creates 4 schematic diagrams described in the patent disclosure:
  Fig 1 — System architecture / workflow
  Fig 2 — Candidate CO2 pipeline network generation flowchart
  Fig 3 — Multi-period joint optimization constraint structure
  Fig 4 — Inter-period state transition diagram

Output: results/figures/patent_fig{1,2,3,4}.png + .pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "SimHei",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Color palette (colorblind-safe)
C_INPUT = "#E8F4FD"      # light blue — raw data / input
C_PHASE = "#FFF4E6"      # light orange — phase / process
C_MODEL = "#E8F5E9"      # light green — model / optimization
C_OUTPUT = "#FCE4EC"     # light pink — output / result
C_ARROW = "#333333"      # dark grey — arrows
C_TEXT = "#212121"       # near black — text
C_BORDER = "#555555"     # border


def _save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png")
    fig.savefig(OUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  [ok] {name}")


def _box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor=C_BORDER, linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=C_TEXT, fontweight=weight, zorder=3)
    return box


def _arrow(ax, x1, y1, x2, y2, text=""):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.5,
                                connectionstyle="arc3,rad=0"),
                zorder=1)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.08, text, ha="center", va="bottom",
                fontsize=8, color=C_TEXT, style="italic")


def _dashed_arrow(ax, x1, y1, x2, y2):
    """Draw a dashed feedback arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.2,
                                linestyle="--", connectionstyle="arc3,rad=0.2"),
                zorder=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — System Architecture / Workflow
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(5, 6.6, "图1  系统整体架构图", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_TEXT)

    # ── Layer 1: Raw Data ──
    ax.text(1.5, 6.0, "原始数据", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_TEXT)
    raw_items = [
        (1.5, 5.4, "煤电机组\n(GEM数据库)"),
        (1.5, 4.7, "生物质潜力图层\n(GJ/m2)"),
        (1.5, 4.0, "绿氢成本图层\n(USD/kg H2)"),
        (1.5, 3.3, "水资源数据\n(冷却方式/水量)"),
        (1.5, 2.6, "油气管道矢量\n(SHP格式)"),
    ]
    for x, y, txt in raw_items:
        _box(ax, x, y, 2.2, 0.55, txt, C_INPUT, fontsize=8)

    # ── Layer 2: Phase A Preprocessing ──
    ax.text(5, 6.0, "Phase A 预处理", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_TEXT)
    phase_items = [
        (5, 5.4, "机组聚类\n→ 改造枢纽"),
        (5, 4.7, "封存点聚类\n→ 储存枢纽"),
        (5, 4.0, "候选管网生成\n(走廊+三角剖分)"),
        (5, 3.3, "供给曲线构建\n(生物质/绿氨/水)"),
        (5, 2.6, "数据标准化\n(单位/坐标/年份)"),
    ]
    for x, y, txt in phase_items:
        _box(ax, x, y, 2.4, 0.55, txt, C_PHASE, fontsize=8)

    # ── Layer 3: Optimization Model ──
    ax.text(8.5, 6.0, "多期联合优化模型", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_TEXT)
    model_items = [
        (8.5, 5.4, "决策变量\n(份额/流量/建设)"),
        (8.5, 4.7, "目标函数\n(跨期贴现总成本)"),
        (8.5, 4.0, "路径耦合约束\n(5条路径竞争)"),
        (8.5, 3.3, "网络流约束\n(CO2管输+封存)"),
        (8.5, 2.6, "资源竞争约束\n(生物质/绿氨/水)"),
    ]
    for x, y, txt in model_items:
        _box(ax, x, y, 2.4, 0.55, txt, C_MODEL, fontsize=8)

    # ── Layer 4: Output (bottom) ──
    ax.text(5, 1.8, "标准化输出", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_TEXT)
    out_items = [
        (2.0, 1.2, "改造路径方案\n(机组级份额)"),
        (4.0, 1.2, "管网建设方案\n(边流量/扩容)"),
        (6.0, 1.2, "资源配置方案\n(燃料/水/CO2)"),
        (8.0, 1.2, "成本分解报告\n(Markdown/CSV)"),
    ]
    for x, y, txt in out_items:
        _box(ax, x, y, 1.8, 0.55, txt, C_OUTPUT, fontsize=8)

    # ── Arrows: Raw → Phase A ──
    for y in [5.4, 4.7, 4.0, 3.3, 2.6]:
        _arrow(ax, 2.6, y, 3.8, y)

    # ── Arrows: Phase A → Model ──
    for y in [5.4, 4.7, 4.0, 3.3, 2.6]:
        _arrow(ax, 6.2, y, 7.3, y)

    # ── Arrows: Model → Output ──
    for y_target in [1.2, 1.2, 1.2, 1.2]:
        pass  # handled below
    _arrow(ax, 8.5, 2.3, 8.0, 1.5, "结果提取")
    _arrow(ax, 7.5, 2.3, 6.0, 1.5)
    _arrow(ax, 6.5, 2.3, 4.0, 1.5)
    _arrow(ax, 5.5, 2.3, 2.0, 1.5)

    # ── Feedback arrow ──
    _dashed_arrow(ax, 5, 0.9, 5, 2.0)
    ax.text(5.3, 1.45, "迭代调参", ha="left", va="center",
            fontsize=8, color=C_TEXT, style="italic")

    _save(fig, "patent_fig1_system_architecture")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Candidate CO2 Pipeline Network Generation Flowchart
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_network_generation():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5, 7.6, "图2  候选CO2管网生成流程图", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_TEXT)

    # Step 1
    _box(ax, 5, 6.8, 4.5, 0.6, "Step 1: 读取既有油气管道矢量数据\n(gas_pipelines.shp + oil_pipeline.shp)", C_INPUT, fontsize=9, bold=True)
    _arrow(ax, 5, 6.4, 5, 5.9)

    # Step 2
    _box(ax, 5, 5.5, 4.5, 0.6, "Step 2: 提取管线端点/交点 → 网络节点\n标记为 existing_main_corridor", C_PHASE, fontsize=9, bold=True)
    _arrow(ax, 5, 5.1, 5, 4.6)

    # Step 3
    _box(ax, 5, 4.2, 4.5, 0.6, "Step 3: Delaunay三角剖分\n(煤电枢纽 + 封存枢纽)", C_PHASE, fontsize=9, bold=True)
    _arrow(ax, 5, 3.8, 5, 3.3)

    # Step 4 — two branches
    _box(ax, 5, 2.9, 4.5, 0.6, "Step 4: 剪枝筛选", C_PHASE, fontsize=9, bold=True)
    _arrow(ax, 5, 2.5, 3.0, 1.9)
    _arrow(ax, 5, 2.5, 7.0, 1.9)

    # Branch left — keep
    _box(ax, 3.0, 1.6, 2.8, 0.6, "保留条件:\n绕行系数 ≤ 1.5 且 边长 ≤ 500km", "#C8E6C9", fontsize=8)
    _arrow(ax, 3.0, 1.2, 3.0, 0.7)

    # Branch right — prune
    _box(ax, 7.0, 1.6, 2.8, 0.6, "剔除条件:\n绕行系数 > 1.5 或 边长 > 500km", "#FFCDD2", fontsize=8)
    _arrow(ax, 7.0, 1.2, 7.0, 0.7)
    ax.text(7.0, 0.4, "丢弃", ha="center", va="center", fontsize=8, color="#B71C1C", style="italic")

    # Step 5
    _box(ax, 3.0, 0.3, 2.8, 0.5, "Step 5: 短接支线生成\n(hub-to-corridor)", C_PHASE, fontsize=8, bold=True)
    _arrow(ax, 3.0, 0.0, 3.0, -0.4)

    # Step 6
    _box(ax, 3.0, -0.8, 2.8, 0.5, "Step 6: 属性标注\n(length/capex/edge_class)", C_MODEL, fontsize=8, bold=True)

    # Legend for edge classes
    legend_items = [
        (8.5, 0.8, "既有主干", "#4477AA"),
        (8.5, 0.3, "三角新建", "#228833"),
        (8.5, -0.2, "短接支线", "#CCBB44"),
        (8.5, -0.7, "点对点备选", "#EE6677"),
    ]
    for x, y, label, color in legend_items:
        ax.add_patch(plt.Rectangle((x-0.3, y-0.1), 0.4, 0.2, facecolor=color, edgecolor=C_BORDER))
        ax.text(x+0.2, y, label, ha="left", va="center", fontsize=8, color=C_TEXT)

    _save(fig, "patent_fig2_network_generation")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Multi-period Joint Optimization Constraint Structure
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_constraint_structure():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5.5, 7.2, "图3  多期联合优化模型约束结构图", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_TEXT)

    # ── Central objective ──
    _box(ax, 5.5, 6.3, 4.0, 0.6,
         "目标函数: min sum_t delta_t x (C_retrofit + C_fuel + C_capture\n+ C_CO2transport + C_storage + C_water + C_retirement)",
         "#FFF9C4", fontsize=9, bold=True)

    # ── Constraint families (arranged in a semi-circle below) ──
    constraints = [
        (1.5, 4.8, "份额归一化\nΣ_w share = 1"),
        (3.5, 4.8, "过期重建\n退役 or 重建"),
        (5.5, 4.8, "掺烧等级离散\nselect + McCormick"),
        (7.5, 4.8, "减排量计算\n5条路径显式"),
        (9.5, 4.8, "碳排放目标\nΣReduction ≥ target"),
        (1.5, 3.2, "网络流守恒\nB×(fwd-bwd)=outflow"),
        (3.5, 3.2, "管道运力\nflow ≤ capacity"),
        (5.5, 3.2, "建设联动\nnew_cap ≤ build×max"),
        (7.5, 3.2, "封存注入\ninjectivity + capacity"),
        (9.5, 3.2, "资源供给\nbiomass/ammonia/water"),
    ]
    for x, y, txt in constraints:
        _box(ax, x, y, 1.7, 0.7, txt, C_PHASE, fontsize=8)
        # Arrow from constraint to objective
        ax.annotate("", xy=(5.5, 6.0), xytext=(x, y+0.4),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8,
                                    connectionstyle=f"arc3,rad={0.1 if x < 5.5 else -0.1}"),
                    zorder=1)

    # ── Cross-period constraints (bottom) ──
    ax.text(5.5, 2.3, "跨期约束", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_TEXT)
    cross_items = [
        (2.5, 1.6, "掺烧等级单调性\nCDF: 不可回退"),
        (5.5, 1.6, "退役单调性\nshare_retire↑"),
        (8.5, 1.6, "管道建设不可逆\nbuild_edge↑"),
    ]
    for x, y, txt in cross_items:
        _box(ax, x, y, 2.0, 0.6, txt, C_OUTPUT, fontsize=8)
        ax.annotate("", xy=(5.5, 2.0), xytext=(x, y+0.35),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
                    zorder=1)

    # ── Decision variables (left sidebar) ──
    ax.text(0.6, 5.5, "决策变量", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT, rotation=90)
    var_items = [
        (0.6, 4.8, "share"),
        (0.6, 4.3, "rebuild"),
        (0.6, 3.8, "co2_flow"),
        (0.6, 3.3, "build_edge"),
        (0.6, 2.8, "new_cap"),
        (0.6, 2.3, "biomass_flow"),
        (0.6, 1.8, "ammonia_flow"),
        (0.6, 1.3, "water_flow"),
        (0.6, 0.8, "storage_use"),
    ]
    for x, y, txt in var_items:
        ax.text(x, y, txt, ha="center", va="center", fontsize=8, color=C_TEXT)

    _save(fig, "patent_fig3_constraint_structure")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Inter-period State Transition Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_state_transition():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5, 5.7, "图4  跨期状态传递示意图", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_TEXT)

    # ── Three time periods ──
    periods = [
        (2.0, "2040", "t=1"),
        (5.0, "2050", "t=2"),
        (8.0, "2060", "t=3"),
    ]
    for x, label, sub in periods:
        ax.text(x, 5.1, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=C_TEXT)
        ax.text(x, 4.8, sub, ha="center", va="center",
                fontsize=9, color="#666666")

    # ── State boxes for each period ──
    states = [
        # (x, y, text, color)
        (2.0, 3.8, "管道存量\nstock(t=1) = 0", C_PHASE),
        (2.0, 2.6, "封存剩余\nS(t=1) = S_total", C_PHASE),
        (2.0, 1.4, "建设决策\nbuild(t=1)", C_PHASE),

        (5.0, 3.8, "管道存量\nstock(t=2) = stock1 + new2", C_MODEL),
        (5.0, 2.6, "封存剩余\nS(t=2) = S1 - use1*dt", C_MODEL),
        (5.0, 1.4, "建设决策\nbuild(t=2) >= build1", C_MODEL),

        (8.0, 3.8, "管道存量\nstock(t=3) = stock2 + new3", C_OUTPUT),
        (8.0, 2.6, "封存剩余\nS(t=3) = S2 - use2*dt", C_OUTPUT),
        (8.0, 1.4, "建设决策\nbuild(t=3) >= build2", C_OUTPUT),
    ]
    for x, y, txt, color in states:
        _box(ax, x, y, 2.2, 0.7, txt, color, fontsize=8)

    # ── Horizontal arrows (time progression) ──
    _arrow(ax, 3.2, 3.8, 3.8, 3.8)
    _arrow(ax, 6.2, 3.8, 6.8, 3.8)
    ax.text(3.5, 4.05, "+new_cap", ha="center", va="bottom", fontsize=8, color=C_TEXT)
    ax.text(6.5, 4.05, "+new_cap", ha="center", va="bottom", fontsize=8, color=C_TEXT)

    _arrow(ax, 3.2, 2.6, 3.8, 2.6)
    _arrow(ax, 6.2, 2.6, 6.8, 2.6)
    ax.text(3.5, 2.35, "-use×10yr", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(6.5, 2.35, "-use×10yr", ha="center", va="top", fontsize=8, color=C_TEXT)

    _arrow(ax, 3.2, 1.4, 3.8, 1.4)
    _arrow(ax, 6.2, 1.4, 6.8, 1.4)
    ax.text(3.5, 1.15, "不可逆", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(6.5, 1.15, "不可逆", ha="center", va="top", fontsize=8, color=C_TEXT)

    # ── Vertical dashed arrows (decision → state update) ──
    for x in [2.0, 5.0, 8.0]:
        ax.plot([x, x], [3.4, 3.0], "k--", lw=1.0, alpha=0.5)
        ax.plot([x, x], [2.2, 1.8], "k--", lw=1.0, alpha=0.5)

    # ── Bottom annotation ──
    ax.text(5, 0.5, "注: 前期决策通过状态变量 (stock, S, build) 影响后期可行域，\n"
                     "实现跨期统筹优化而非逐年孤立求解。",
            ha="center", va="center", fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="none"))

    _save(fig, "patent_fig4_state_transition")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating patent figures...")
    fig1_system_architecture()
    fig2_network_generation()
    fig3_constraint_structure()
    fig4_state_transition()
    print(f"\nAll figures saved to: {OUT_DIR}")
