"""Generate schematic figures for patent application (zhuanli.md) — REVISED VERSION.

This script creates 4 schematic diagrams that meet patent requirements:
  Fig 1 — System architecture with solver and feedback loop
  Fig 2 — Candidate CO2 pipeline network generation (3-layer strategy)
  Fig 3 — Multi-period joint optimization constraint structure (grouped by module)
  Fig 4 — Inter-period state transition (all state variables + generic periods)

Output: results/figures/patent_fig{1,2,3,4}_v2.png + .pdf
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
OUT_DIR = ROOT / "results" / "figures" / "patent"
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
C_SOLVER = "#F3E5F5"     # light purple — solver
C_OUTPUT = "#FCE4EC"     # light pink — output / result
C_ARROW = "#333333"      # dark grey — arrows
C_TEXT = "#212121"       # near black — text
C_BORDER = "#555555"     # border
C_OPTIONAL = "#FFF9C4"   # light yellow — optional constraints


def _save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png")
    fig.savefig(OUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  [ok] {name}")


def _box(ax, x, y, w, h, text, color, fontsize=9, bold=False, linestyle="-"):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor=C_BORDER, linewidth=1.2,
        linestyle=linestyle, zorder=2,
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


def _dashed_arrow(ax, x1, y1, x2, y2, text=""):
    """Draw a dashed feedback arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.2,
                                linestyle="--", connectionstyle="arc3,rad=0.2"),
                zorder=1)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, text, ha="left", va="center",
                fontsize=8, color=C_TEXT, style="italic")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — System Architecture / Workflow (REVISED)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(5.5, 7.2, "图1  系统整体架构图", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_TEXT)

    # ── Layer 1: Raw Data (left) ──
    ax.text(1.8, 6.6, "原始数据层", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)
    raw_items = [
        (1.8, 6.0, "煤电机组\n(GEM数据库)"),
        (1.8, 5.3, "机组参数\n(容量/年龄/排放)"),
        (1.8, 4.6, "生物质潜力图层\n(GJ/m2)"),
        (1.8, 3.9, "绿氢成本图层\n(USD/kg H2)"),
        (1.8, 3.2, "水资源数据\n(冷却方式/水量)"),
        (1.8, 2.5, "油气管道矢量\n(SHP格式)"),
    ]
    for x, y, txt in raw_items:
        _box(ax, x, y, 2.4, 0.55, txt, C_INPUT, fontsize=8)

    # ── Layer 2: Phase A Preprocessing (center-left) ──
    ax.text(5.0, 6.6, "Phase A 预处理层", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)
    phase_items = [
        (5.0, 6.0, "机组聚类\n→ 改造枢纽"),
        (5.0, 5.3, "封存点聚类\n→ 储存枢纽"),
        (5.0, 4.6, "候选管网生成\n(走廊+三角剖分+短接)"),
        (5.0, 3.9, "供给曲线构建\n(生物质/绿氨/水)"),
        (5.0, 3.2, "数据标准化\n(单位/坐标/年份)"),
        (5.0, 2.5, "关联矩阵构建\n(节点-边-关联)"),
    ]
    for x, y, txt in phase_items:
        _box(ax, x, y, 2.6, 0.55, txt, C_PHASE, fontsize=8)

    # ── Layer 3: Optimization Model (center-right) ──
    ax.text(8.2, 6.6, "多期联合优化模型层", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)
    model_items = [
        (8.2, 6.0, "决策变量定义\n(份额/流量/建设)"),
        (8.2, 5.3, "目标函数构建\n(跨期贴现总成本)"),
        (8.2, 4.6, "路径耦合约束\n(5条路径竞争)"),
        (8.2, 3.9, "网络流约束\n(CO2管输+封存)"),
        (8.2, 3.2, "资源竞争约束\n(生物质/绿氨/水)"),
        (8.2, 2.5, "跨期传递约束\n(单调性/存量累积)"),
    ]
    for x, y, txt in model_items:
        _box(ax, x, y, 2.6, 0.55, txt, C_MODEL, fontsize=8)

    # ── Layer 4: Solver (right) ──
    ax.text(10.2, 4.5, "求解器", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT, rotation=90)
    _box(ax, 10.2, 5.5, 1.0, 1.8, "Gurobi\nMIP\n求解", C_SOLVER, fontsize=9, bold=True)

    # ── Layer 5: Output (bottom) ──
    ax.text(5.5, 1.7, "标准化输出层", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)
    out_items = [
        (2.2, 1.1, "改造路径方案\n(机组级份额)"),
        (4.2, 1.1, "管网建设方案\n(边流量/扩容)"),
        (6.2, 1.1, "资源配置方案\n(燃料/水/CO2)"),
        (8.2, 1.1, "成本分解报告\n(Markdown/CSV)"),
        (10.0, 1.1, "松弛细节表\n(约束紧张度)"),
    ]
    for x, y, txt in out_items:
        _box(ax, x, y, 1.8, 0.5, txt, C_OUTPUT, fontsize=8)

    # ── Arrows: Raw → Phase A ──
    for y in [6.0, 5.3, 4.6, 3.9, 3.2, 2.5]:
        _arrow(ax, 3.1, y, 3.7, y)

    # ── Arrows: Phase A → Model ──
    for y in [6.0, 5.3, 4.6, 3.9, 3.2, 2.5]:
        _arrow(ax, 6.3, y, 6.9, y)

    # ── Arrows: Model → Solver ──
    _arrow(ax, 9.5, 4.5, 9.7, 4.5)

    # ── Arrows: Solver → Output ──
    _arrow(ax, 10.2, 4.0, 10.2, 1.4, "结果提取")
    _arrow(ax, 10.0, 2.8, 8.2, 1.4)
    _arrow(ax, 9.0, 2.8, 6.2, 1.4)
    _arrow(ax, 8.0, 2.8, 4.2, 1.4)
    _arrow(ax, 7.0, 2.8, 2.2, 1.4)

    # ── Feedback loop: Output → Phase A (dashed) ──
    _dashed_arrow(ax, 5.5, 0.8, 5.0, 2.0, "迭代调参")
    ax.text(5.5, 0.3, "注: 虚线表示反馈闭环，输出结果用于调整预处理参数\n"
                       "     实现数据-模型-结果的迭代优化。",
            ha="center", va="center", fontsize=8, color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5", edgecolor="none"))

    # ── Legend ──
    legend_items = [
        (0.3, 0.8, "原始数据", C_INPUT),
        (0.3, 0.5, "预处理", C_PHASE),
        (0.3, 0.2, "优化模型", C_MODEL),
        (1.8, 0.8, "求解器", C_SOLVER),
        (1.8, 0.5, "输出结果", C_OUTPUT),
    ]
    for x, y, label, color in legend_items:
        ax.add_patch(plt.Rectangle((x, y-0.08), 0.25, 0.16, facecolor=color, edgecolor=C_BORDER))
        ax.text(x+0.3, y, label, ha="left", va="center", fontsize=7, color=C_TEXT)

    _save(fig, "patent_fig1_system_architecture_v2")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Candidate CO2 Pipeline Network Generation (REVISED)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_network_generation():
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5, 8.2, "图2  候选CO2管网生成流程图", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_TEXT)

    # ── Layer 1: Existing corridor reuse ──
    ax.text(1.2, 7.6, "第一层\n既有走廊复用", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#BBDEFB", edgecolor="#1976D2"))

    _box(ax, 5, 7.3, 4.5, 0.5, "Step 1: 读取既有油气管道矢量数据\n(gas_pipelines.shp + oil_pipeline.shp)", C_INPUT, fontsize=9, bold=True)
    _arrow(ax, 5, 6.95, 5, 6.55)
    _box(ax, 5, 6.3, 4.5, 0.5, "Step 2: 提取管线端点/交点 → 网络节点\n标记为 existing_main_corridor", C_PHASE, fontsize=9, bold=True)
    _arrow(ax, 5, 6.05, 5, 5.65)

    # ── Layer 2: Triangulation supplement ──
    ax.text(1.2, 5.3, "第二层\n三角剖分补充", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#C8E6C9", edgecolor="#388E3C"))

    _box(ax, 5, 5.0, 4.5, 0.5, "Step 3: Delaunay三角剖分\n(煤电枢纽 + 封存枢纽)", C_PHASE, fontsize=9, bold=True)
    _arrow(ax, 5, 4.75, 5, 4.35)
    _box(ax, 5, 3.9, 4.5, 0.5, "Step 4: 剪枝筛选\n(绕行系数 + 边长上限)", C_PHASE, fontsize=9, bold=True)

    # Branch
    _arrow(ax, 5, 3.65, 3.0, 3.0)
    _arrow(ax, 5, 3.65, 7.0, 3.0)

    # Branch left — keep
    _box(ax, 3.0, 2.5, 2.8, 0.5, "保留:\n绕行系数≤1.5 且 边长≤500km", "#C8E6C9", fontsize=8)
    _arrow(ax, 3.0, 2.25, 3.0, 1.85)

    # Branch right — prune
    _box(ax, 7.0, 2.5, 2.8, 0.5, "剔除:\n绕行系数>1.5 或 边长>500km", "#FFCDD2", fontsize=8)
    _arrow(ax, 7.0, 2.25, 7.0, 1.85)
    ax.text(7.0, 1.65, "丢弃", ha="center", va="center", fontsize=8, color="#B71C1C", style="italic")

    # ── Layer 3: Short branch & backup ──
    ax.text(1.2, 1.5, "第三层\n短接支线保底", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF9C4", edgecolor="#FBC02D"))

    _box(ax, 3.0, 1.4, 2.8, 0.5, "Step 5: 短接支线生成\n(hub-to-corridor)", C_PHASE, fontsize=8, bold=True)
    _arrow(ax, 3.0, 1.15, 3.0, 0.75)
    _box(ax, 3.0, 0.4, 2.8, 0.5, "Step 6: 属性标注\n(length/capex/edge_class)", C_MODEL, fontsize=8, bold=True)

    # ── Formula box ──
    ax.text(7.0, 1.0, "绕行系数 = 三角边长度 / 沿走廊最短路径长度",
            ha="center", va="center", fontsize=8, color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5", edgecolor=C_BORDER))

    # ── Edge class legend (right side) ──
    ax.text(8.8, 5.5, "边类别图例", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_TEXT)
    legend_items = [
        (8.8, 5.0, "既有主干走廊", "#4477AA", "capex×0.4"),
        (8.8, 4.5, "三角新建干线", "#228833", "capex×1.0"),
        (8.8, 4.0, "短接支线", "#CCBB44", "capex×1.35"),
        (8.8, 3.5, "点对点备选", "#EE6677", "capex×2.8"),
    ]
    for x, y, label, color, cost in legend_items:
        ax.add_patch(plt.Rectangle((x-0.4, y-0.1), 0.3, 0.2, facecolor=color, edgecolor=C_BORDER))
        ax.text(x+0.0, y, f"{label}  ({cost})", ha="left", va="center", fontsize=8, color=C_TEXT)

    # ── Bottom note ──
    ax.text(5, -0.2, "注: 三层策略确保候选边集既充分利用既有基础设施，又保证空间覆盖完整性，\n"
                      "     同时通过剪枝控制问题规模，使优化模型可解。",
            ha="center", va="center", fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="none"))

    _save(fig, "patent_fig2_network_generation_v2")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Constraint Structure (REVISED)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_constraint_structure():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6, 7.7, "图3  多期联合优化模型约束结构图", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_TEXT)

    # ── Central objective ──
    _box(ax, 6, 6.9, 5.0, 0.6,
         "目标函数: 最小化 跨期贴现总成本\n= 改造成本 + 燃料成本 + 捕集成本 + CO2运输 + 封存 + 水资源 + 退役成本",
         "#FFF9C4", fontsize=9, bold=True)

    # ── Module labels ──
    ax.text(2.0, 6.0, "模块四\n路径耦合", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFCCBC", edgecolor="none"))
    ax.text(6.0, 6.0, "模块五\n网络流与基础设施", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#C5CAE9", edgecolor="none"))
    ax.text(10.0, 6.0, "模块六\n资源供给", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#B2DFDB", edgecolor="none"))

    # ── Module 4 constraints (left) ──
    m4_items = [
        (1.0, 5.1, "(1) 份额归一化\nΣ share = 1"),
        (2.0, 5.1, "(2) 过期重建\n退役 or 重建"),
        (3.0, 5.1, "(3) 掺烧离散\nMcCormick包络"),
        (1.5, 4.2, "(4) 减排计算\n5路径显式"),
        (2.5, 4.2, "(5) 排放目标\nΣReduction≥target"),
    ]
    for x, y, txt in m4_items:
        _box(ax, x, y, 1.8, 0.6, txt, C_PHASE, fontsize=8)
        ax.annotate("", xy=(6, 6.6), xytext=(x, y+0.35),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8,
                                    connectionstyle="arc3,rad=0.15"), zorder=1)

    # ── Module 5 constraints (center) ──
    m5_items = [
        (5.0, 5.1, "(6) 节点守恒\nB×(fwd-bwd)=outflow"),
        (6.0, 5.1, "(7) 管道运力\nflow ≤ capacity"),
        (7.0, 5.1, "(8) 建设联动\nnew_cap ≤ build×max"),
        (5.5, 4.2, "(9) 封存注入\ninjectivity + capacity"),
        (6.5, 4.2, "(10) 运力累积\n寿命期约束"),
    ]
    for x, y, txt in m5_items:
        _box(ax, x, y, 1.8, 0.6, txt, C_PHASE, fontsize=8)
        ax.annotate("", xy=(6, 6.6), xytext=(x, y+0.35),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8), zorder=1)

    # ── Module 6 constraints (right) ──
    m6_items = [
        (9.0, 5.1, "(11) 生物质供给\nnode_out ≤ supply"),
        (10.0, 5.1, "(12) 绿氨供给\nmin_cost包络"),
        (11.0, 5.1, "(13) 水资源\ncooling×multiplier"),
        (10.0, 4.2, "(14) 枢纽平衡\ninflow = demand"),
    ]
    for x, y, txt in m6_items:
        _box(ax, x, y, 1.8, 0.6, txt, C_PHASE, fontsize=8)
        ax.annotate("", xy=(6, 6.6), xytext=(x, y+0.35),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8,
                                    connectionstyle="arc3,rad=-0.15"), zorder=1)

    # ── Optional constraints (dashed boxes) ──
    ax.text(6, 3.4, "可选约束（政策分析场景）", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#666666")
    opt_items = [
        (3.5, 2.8, "路径强制激活\n最低发电份额", C_OPTIONAL),
        (6.0, 2.8, "路径禁用\n稳健性分析", C_OPTIONAL),
        (8.5, 2.8, "松弛惩罚\n保证可行性", C_OPTIONAL),
    ]
    for x, y, txt, color in opt_items:
        _box(ax, x, y, 2.0, 0.5, txt, color, fontsize=8, linestyle="--")
        ax.annotate("", xy=(6, 6.3), xytext=(x, y+0.3),
                    arrowprops=dict(arrowstyle="->", color="#BBBBBB", lw=0.6,
                                    linestyle="--"), zorder=1)

    # ── Cross-period constraints (bottom) ──
    ax.text(6, 2.0, "模块七：跨期单调性与状态传递约束", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)
    cross_items = [
        (2.5, 1.3, "掺烧等级单调性\nCDF: 不可回退"),
        (4.5, 1.3, "退役单调性\nshare_retire递增"),
        (6.5, 1.3, "建设不可逆\nbuild_edge递增"),
        (8.5, 1.3, "CCS增量CAPEX\nmax(0, Δshare)×unit_cost"),
        (10.5, 1.3, "掺烧升级CAPEX\n等级跃迁成本"),
    ]
    for x, y, txt in cross_items:
        _box(ax, x, y, 1.8, 0.55, txt, C_OUTPUT, fontsize=8)
        ax.annotate("", xy=(6, 1.7), xytext=(x, y+0.3),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8), zorder=1)

    # ── Decision variables (left sidebar) ──
    ax.text(0.4, 5.0, "决策变量", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT, rotation=90)
    var_items = [
        (0.4, 4.3, "share"),
        (0.4, 3.9, "rebuild"),
        (0.4, 3.5, "co2_flow"),
        (0.4, 3.1, "build_edge"),
        (0.4, 2.7, "new_cap"),
        (0.4, 2.3, "biomass_flow"),
        (0.4, 1.9, "ammonia_flow"),
        (0.4, 1.5, "water_flow"),
        (0.4, 1.1, "storage_use"),
    ]
    for x, y, txt in var_items:
        ax.text(x, y, txt, ha="center", va="center", fontsize=8, color=C_TEXT)

    # ── Legend ──
    ax.text(0.4, 0.4, "图例:", ha="left", va="center", fontsize=8, fontweight="bold", color=C_TEXT)
    ax.add_patch(plt.Rectangle((0.8, 0.32), 0.3, 0.16, facecolor=C_PHASE, edgecolor=C_BORDER))
    ax.text(1.2, 0.4, "必选约束", ha="left", va="center", fontsize=8, color=C_TEXT)
    ax.add_patch(plt.Rectangle((2.2, 0.32), 0.3, 0.16, facecolor=C_OPTIONAL, edgecolor=C_BORDER, linestyle="--"))
    ax.text(2.6, 0.4, "可选约束", ha="left", va="center", fontsize=8, color=C_TEXT)

    _save(fig, "patent_fig3_constraint_structure_v2")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Inter-period State Transition (REVISED)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_state_transition():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5.5, 6.7, "图4  跨期状态传递示意图", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_TEXT)

    # ── Three generic time periods ──
    periods = [
        (2.0, "前期", "t=1"),
        (5.5, "中期", "t=2"),
        (9.0, "后期", "t=3"),
    ]
    for x, label, sub in periods:
        ax.text(x, 6.2, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=C_TEXT)
        ax.text(x, 5.9, sub, ha="center", va="center",
                fontsize=9, color="#666666")

    # ── State variable rows (5 rows) ──
    row_labels = [
        (5.3, "管道存量 stock"),
        (4.3, "封存剩余 S"),
        (3.3, "建设决策 build"),
        (2.3, "掺烧等级 level"),
        (1.3, "CCS改造状态 retrofit"),
    ]
    for y, label in row_labels:
        ax.text(0.3, y, label, ha="left", va="center", fontsize=8, color=C_TEXT)

    # ── State boxes for each period ──
    states = [
        # Row 1: stock
        (2.0, 5.3, "stock(1) = 0", C_PHASE),
        (5.5, 5.3, "stock(2) = stock1 + new2", C_MODEL),
        (9.0, 5.3, "stock(3) = stock2 + new3", C_OUTPUT),
        # Row 2: storage remaining
        (2.0, 4.3, "S(1) = S_total", C_PHASE),
        (5.5, 4.3, "S(2) = S1 - use1*dt", C_MODEL),
        (9.0, 4.3, "S(3) = S2 - use2*dt", C_OUTPUT),
        # Row 3: build decision
        (2.0, 3.3, "build(1)", C_PHASE),
        (5.5, 3.3, "build(2) >= build1", C_MODEL),
        (9.0, 3.3, "build(3) >= build2", C_OUTPUT),
        # Row 4: blend level
        (2.0, 2.3, "level(1)", C_PHASE),
        (5.5, 2.3, "level(2) >= level1", C_MODEL),
        (9.0, 2.3, "level(3) >= level2", C_OUTPUT),
        # Row 5: CCS retrofit
        (2.0, 1.3, "retrofit(1) = 0", C_PHASE),
        (5.5, 1.3, "Δshare = max(0,\nshare2-share1)", C_MODEL),
        (9.0, 1.3, "Δshare = max(0,\nshare3-share2)", C_OUTPUT),
    ]
    for x, y, txt, color in states:
        _box(ax, x, y, 2.4, 0.5, txt, color, fontsize=8)

    # ── Horizontal arrows (time progression) ──
    for y in [5.3, 4.3, 3.3, 2.3, 1.3]:
        _arrow(ax, 3.3, y, 4.3, y)
        _arrow(ax, 6.8, y, 7.8, y)

    # ── Transfer labels ──
    ax.text(3.8, 5.5, "+new_cap", ha="center", va="bottom", fontsize=8, color=C_TEXT)
    ax.text(7.3, 5.5, "+new_cap", ha="center", va="bottom", fontsize=8, color=C_TEXT)
    ax.text(3.8, 4.1, "-use*dt", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(7.3, 4.1, "-use*dt", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(3.8, 3.1, "不可逆", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(7.3, 3.1, "不可逆", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(3.8, 2.1, "不可回退", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(7.3, 2.1, "不可回退", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(3.8, 1.1, "增量成本", ha="center", va="top", fontsize=8, color=C_TEXT)
    ax.text(7.3, 1.1, "增量成本", ha="center", va="top", fontsize=8, color=C_TEXT)

    # ── Vertical dashed arrows (decision → state update) ──
    for x in [2.0, 5.5, 9.0]:
        for y_pair in [(5.1, 4.5), (4.1, 3.5), (3.1, 2.5), (2.1, 1.5)]:
            ax.plot([x, x], y_pair, "k--", lw=0.8, alpha=0.4)

    # ── Bottom annotation ──
    ax.text(5.5, 0.4, "注: 前期决策通过5类状态变量(stock, S, build, level, retrofit)影响后期可行域，\n"
                       "     实现跨期统筹优化。其中CCS改造状态采用增量成本计算: CAPEX = Δshare × unit_cost。",
            ha="center", va="center", fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="none"))

    _save(fig, "patent_fig4_state_transition_v2")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating patent figures (v2)...")
    fig1_system_architecture()
    fig2_network_generation()
    fig3_constraint_structure()
    fig4_state_transition()
    print(f"\nAll figures saved to: {OUT_DIR}")
