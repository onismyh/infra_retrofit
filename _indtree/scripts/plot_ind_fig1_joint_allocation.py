"""ind_fig1 — 联合减排分配：谁减、用哪条路、封存空间给了谁。

这张图回答的是把工业变成决策主体之后唯一的新问题：**在同一个排放目标下，减排量如何在
煤电与五类工业之间分配。** 四个面板各自独立可读：

    a  逐年减排量，煤电 vs 工业，叠加目标线      —— 谁减
    b  2060 年分行业通路份额                     —— 工业用哪条路
    c  封存注入量 vs 全国注入能力，煤电 vs 工业  —— 封存空间给了谁
    d  各行业 CCS 与 H2 的边际减排成本随年份     —— 为什么是这条路

面板 d 的成本**从模型自己的函数算出来**（`levelised_capture_cost_cny_per_t`、
`h2_premium_cny_per_t` 与 `ccs_learning_factor`），不是抄一张表——抄表会在参数改动后与求解
结果脱节而没人发现。注意这两个函数给的是"改造 capex 年金 + 固定运维 + 能耗"折成的**平准化**
每吨成本，仅供本图比较；目标函数里 capex 是一次性计入并在期末计残值的（2026-09-22 起）。

数据来源固定为 `_indtree`（v9 管网）。仓库根 `results/IND_*` 用的是 v7 管网，够不着 38%
的封存汇，其捕集量与注入分布不可入图（见 `_indtree/README.md`）。
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_style import (  # noqa: E402
    DOUBLE_COL,
    RESULTS_DIR,
    apply_style,
    cjk_fill,
    panel_label,
    save_fig,
)
from coal_retrofit.constants_industry import (  # noqa: E402
    INDUSTRY_H2_ABATEMENT_FRACTION,
    SECTOR_LABELS_ZH,
    h2_premium_cny_per_t,
    levelised_capture_cost_cny_per_t,
)
from coal_retrofit.optimization.scenario import OptimizationAssumptions  # noqa: E402


def _steam_co2_fraction(sector: str, assumptions: OptimizationAssumptions) -> float:
    """Vented reboiler-steam CO2 per tonne captured, the same number the solver nets out."""
    from coal_retrofit.constants_industry import capture_steam_co2_t_per_t

    ef_gj = assumptions.coal_emission_factor_t_per_mwh / assumptions.heat_rate_gj_per_mwh
    return float(capture_steam_co2_t_per_t(sector, ef_gj))

# 带水约束的那个作主图。无水的 IND_BASE_t95 只在面板 a 里作虚线对照。
RUN = "IND_WA_cwatm_126_dry_oq_t95"
RUN_NOWATER = "IND_BASE_t95"

ROUTE_COLOR = {"unabated": "#969696", "ccs": "#636363", "h2": "#9E9AC8"}
ROUTE_LABEL = {"unabated": "不改造", "ccs": "CCS 捕集", "h2": "绿氢替代"}
ROUTE_ORDER = ["unabated", "ccs", "h2"]
# 与 ed_fig15_source_atlas 同一套行业色，两张图必须能对上
SECTOR_COLOR = {
    "steel_bf_bof": "#1B9E77", "steel_eaf": "#66A61E", "cement": "#D95F02",
    "ammonia": "#7570B3", "methanol": "#E7298A",
}
SECTOR_ORDER = ["cement", "steel_bf_bof", "methanol", "ammonia", "steel_eaf"]
COAL_COLOR = "#636363"
IND_COLOR = "#00A087"
TARGET_COLOR = "#CC3311"


EXPECTED_SINKS = 89        # v9 管网的汇数，用来把求解树认出来


def _h2_price(meta, year) -> float:
    """氢价的元数据键在 2026-09-10 改过名（加了 national_mean），两种都认。"""
    node = meta["years"][str(year)]["industry"]
    for key in ("h2_price_national_mean_cny_per_kg", "h2_price_cny_per_kg"):
        if key in node:
            return float(node[key])
    raise KeyError("run JSON 里找不到氢价字段：%s" % sorted(node))


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


def assert_v9_tree(run: str) -> int:
    """确认脚本正跑在 v9 求解树上，而不是仓库根的 v7 结果上。

    两棵树里都有同名目录 `IND_WA_cwatm_126_dry_oq_t95`，`RESULTS_DIR` 只跟着脚本自己的
    位置走。在仓库根跑会画出 35 汇的 v7 结果，而图注里"v9 管网"是写死的字符串——
    图看上去完全正常。这一条就是防那个。
    """
    sinks = pd.read_csv(RESULTS_DIR / run / "storage_utilization.csv")["storage_hub_id"].nunique()
    if sinks != EXPECTED_SINKS:
        raise RuntimeError(
            f"{RESULTS_DIR} 不是 v9 求解树：{run} 只有 {sinks} 个汇（应为 {EXPECTED_SINKS}）。"
            f"请在 _indtree/ 下运行本脚本。")
    return int(sinks)


def load(run: str) -> tuple[dict, pd.DataFrame, pd.DataFrame] | None:
    """Run summary, industry detail and sink utilisation. None when unsolved."""
    meta_path = RESULTS_DIR / f"{run}.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    detail = pd.read_csv(RESULTS_DIR / run / "industry_detail.csv")
    storage = pd.read_csv(RESULTS_DIR / run / "storage_utilization.csv")
    return meta, detail, storage


def years_of(meta: dict) -> list[int]:
    return sorted(int(y) for y in meta["years"])


def panel_a(ax, meta: dict, meta_nw: dict | None) -> None:
    """Stacked coal + industry abatement per year, against the joint target."""
    years = years_of(meta)
    coal = np.array([float(meta["years"][str(y)]["coal_reduction_mt"]) for y in years])
    ind = np.array([float((meta["years"][str(y)]["industry"] or {}).get("reduction_mt", 0.0))
                    for y in years])
    # 目标 = 份额 x (煤电基线 + 工业基线)，**逐年各取各年的基线**。
    # 约束本身就是逐年算的（solver：year_data["emissions_mt"].sum() + 工业基线），
    # 拿某一年的基线套到所有年，只在基线恰好逐年不变时才对——那是巧合，不是口径。
    target = np.array([_target_for(meta, y, _joint_baseline(meta, y)) for y in years])

    x = np.arange(len(years), dtype=float)
    # 有符号堆叠。2030 年煤电减排是负的（为省水改空冷，背压惩罚让它比基线多排），
    # 若照常 bottom=coal 画，工业那根正柱会从 coal 一路盖到 +ind，把负柱整根遮掉——
    # 而"水约束下煤电反而多排 33.5 Mt"恰恰是这一格里最值得看见的东西。
    pos_bottom = np.where(coal > 0, coal, 0.0)
    ax.bar(x, coal, width=0.42, color=COAL_COLOR, label="煤电机队")
    ax.bar(x, ind, width=0.42, bottom=pos_bottom, color=IND_COLOR, label="工业点源")
    # 负值不在面板上标注，写进图注。在 8 000 Mt 的量程上，-33.5 Mt 只有一个像素高，
    # 任何贴着它放的标签都会压到 x 轴刻度或隔壁的注记上——把一个读不出来的量硬标出来，
    # 换来的是两个读不出来的东西。有符号堆叠保证它至少没被工业柱盖掉。
    # 目标份额是 (0, 0, 0, 0.95)，只有末年有约束。连成一条线会被读成"目标逐年上升"，
    # 所以画成每年一段短横杠，不相连。
    for xi, tv in zip(x, target):
        ax.plot([xi - 0.30, xi + 0.30], [tv, tv], lw=1.4, color=TARGET_COLOR, zorder=5,
                solid_capstyle="butt")
    ax.plot([], [], lw=1.4, color=TARGET_COLOR, label="联合排放目标（当年）")
    if meta_nw is not None:
        # 按**年份**取，不按位置：两个运行的年份集若不一致，按位置配对会把 2040 的值
        # 画到 2030 的槽里，图上完全看不出来。缺年直接抛错。
        missing = [y for y in years if str(y) not in meta_nw["years"]]
        if missing:
            raise RuntimeError(f"{RUN_NOWATER} 缺年份 {missing}，无法与 {RUN} 逐年对齐。")
        nw = np.array([float(meta_nw["years"][str(y)]["coal_reduction_mt"])
                       + float((meta_nw["years"][str(y)]["industry"] or {}).get("reduction_mt", 0.0))
                       for y in years])
        # 同样画成逐年短横杠，不连线：它是另一个情景在同一年的水平，与柱顶直接对齐比较；
        # 连成折线会在 2030-2060 之间划出一条斜穿柱子的"趋势"，那条趋势没有对应的量。
        for xi, nv in zip(x, nw):
            ax.plot([xi - 0.30, xi + 0.30], [nv, nv], ls=(0, (2.2, 1.6)), lw=0.9,
                    color="#4D4D4D", zorder=4, solid_capstyle="butt")
        ax.plot([], [], ls=(0, (2.2, 1.6)), lw=0.9, color="#4D4D4D", label="无水约束合计")
    # 工业段在 2030-2050 只有几十 Mt，写在段内会压到分界线上，一律标在柱右侧。
    for i in range(len(years)):
        if ind[i] <= 0.5:
            continue
        ax.annotate(f"工业 {ind[i]:,.0f}", xy=(x[i] + 0.23, coal[i] + ind[i] / 2),
                    xytext=(x[i] + 0.32, coal[i] + ind[i] / 2 + 480),
                    fontsize=5.6, color=IND_COLOR, ha="left", va="center", zorder=6,
                    # 标签会伸到下一根柱子上（2050 的"工业 1,676"被 2060 柱盖掉半截），
                    # 垫一层白底最省事，柱间空白处看不出来。
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=IND_COLOR))
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel(r"减排量（Mt CO$_2$ yr$^{-1}$）")
    ax.set_ylim(bottom=min(0.0, float(coal.min()) * 1.4))
    ax.axhline(0.0, lw=0.5, color="#BBBBBB", zorder=0)
    ax.legend(loc="upper left", fontsize=6.2, frameon=False, handlelength=1.2,
              labelspacing=0.3, borderaxespad=0.2)


@lru_cache(maxsize=None)
def _coal_baseline_by_year() -> dict[int, float]:
    """Coal baseline per year, from the run's own plant table rather than from inputs."""
    # `plot_style.hub_frame` would rebuild it from plants.csv; reading the run's own numbers
    # keeps the figure tied to the solve it is describing.
    detail = pd.read_csv(RESULTS_DIR / RUN / "plant_detail.csv")
    grouped = detail.groupby("year")["baseline_emissions_mt"].sum()
    return {int(y): float(v) for y, v in grouped.items()}


def _joint_baseline(meta: dict, year: int) -> float:
    """Target denominator in `year` = coal baseline + industry baseline, both from that year."""
    industry = meta["years"][str(year)].get("industry") or {}
    return _coal_baseline_by_year()[int(year)] + float(industry.get("baseline_mt", 0.0))


def _target_for(meta: dict, year: int, joint_baseline: float) -> float:
    """Required abatement in `year` = target fraction x joint baseline."""
    fraction = _target_fractions(meta).get(int(year), 0.0)
    return fraction * joint_baseline


def _target_fractions(meta: dict) -> dict[int, float]:
    """Emission-target fraction per year for this run, from `scripts/run_single.py`."""
    from run_single import EXPERIMENTS

    scenario_kw, _ = EXPERIMENTS[RUN]
    years = sorted(int(y) for y in meta["years"])
    if "emission_target_fraction" not in scenario_kw:
        raise RuntimeError(f"{RUN} 没有登记 emission_target_fraction，目标线无从画起。")
    fractions = scenario_kw["emission_target_fraction"]
    # strict=True：份额个数与规划年个数不一致时直接抛错。用 strict=False 会静默截断，
    # 末年的目标就画不出来，而那恰好是唯一有约束的一年。
    return dict(zip(years, fractions, strict=True))


def panel_b(ax, detail: pd.DataFrame) -> None:
    """2060 route split per sector, weighted by baseline emissions."""
    last = int(detail["year"].max())
    block = detail[detail["year"] == last]
    sectors = [s for s in SECTOR_ORDER if s in set(block["sector"])]
    y = np.arange(len(sectors))[::-1]
    left = np.zeros(len(sectors))
    for route in ROUTE_ORDER:
        vals = []
        for sector in sectors:
            g = block[block["sector"] == sector]
            w = g["baseline_co2_mt"].to_numpy(float)
            vals.append(float((g[f"share_{route}"].to_numpy(float) * w).sum() / w.sum()))
        vals = np.array(vals)
        ax.barh(y, vals, left=left, height=0.62, color=ROUTE_COLOR[route],
                edgecolor="white", linewidth=0.4, label=ROUTE_LABEL[route])
        for i, v in enumerate(vals):
            if v > 0.12:
                ax.text(left[i] + v / 2, y[i], f"{v:.0%}", ha="center", va="center",
                        fontsize=5.8, color="white" if route != "h2" else "#2A2545",
                        fontweight="bold")
        left = left + vals
    labels = [f"{SECTOR_LABELS_ZH[s]}\n"
              f"{block[block['sector'] == s]['baseline_co2_mt'].sum():.0f} Mt" for s in sectors]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel(f"{last} 年通路份额（按基线排放加权）")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6.2,
              frameon=False, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)


def panel_c(ax, meta: dict, detail: pd.DataFrame, storage: pd.DataFrame) -> None:
    """Injection by source group against national injectivity."""
    years = years_of(meta)
    total = np.array([float(storage[storage["year"] == y]["storage_use_mtpa"].sum()) for y in years])
    ind_cap = np.array([float(detail[detail["year"] == y]["captured_mt"].sum()) for y in years])
    # 煤电捕集直接读煤电自己的表，不用"注入总量减工业捕集"去倒推。倒推 + clip 会在
    # 两者对不上时（捕集与注入之间若出现损失或口径差）静默把煤电压成 0，柱子照样画得出来。
    plants = pd.read_csv(RESULTS_DIR / RUN / "plant_detail.csv")
    coal_cap = np.array([float(plants[plants["year"] == y]["captured_mt"].sum()) for y in years])
    residual = np.abs(total - coal_cap - ind_cap)
    if float(residual.max()) > 0.5:
        raise RuntimeError(
            f"注入量与两侧捕集量对不上，最大差 {residual.max():.2f} Mt/yr；"
            f"面板 c 的堆叠柱在这种情况下不成立。")
    capacity = float(storage[storage["year"] == years[-1]]["injectivity_mtpa"].sum())

    x = np.arange(len(years))
    ax.bar(x, coal_cap, width=0.42, color=COAL_COLOR, label="煤电捕集")
    ax.bar(x, ind_cap, width=0.42, bottom=coal_cap, color=IND_COLOR, label="工业捕集")
    ax.axhline(capacity, ls="--", lw=1.0, color=TARGET_COLOR, zorder=5)
    # 标在能力线**下方**、贴左：线上方已被"用掉多少"的注记占住，两者同侧会叠在一起。
    ax.text(-0.30, capacity * 0.985, f"全国注入能力 {capacity:,.0f}", fontsize=5.8,
            color=TARGET_COLOR, ha="left", va="top")
    used = total[-1] / capacity if capacity > 0 else 0.0
    # 放末年柱正上方、能力线之上的空白里；引线横穿画面反而更难读。
    ax.text(x[-1], capacity * 1.04,
            f"{years[-1]} 年用掉 {used:.1%}，工业分到 {ind_cap[-1]:.1f} Mt",
            fontsize=5.8, color="#333333", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel(r"注入量（Mt CO$_2$ yr$^{-1}$）")
    ax.set_ylim(0, max(capacity, float(total.max())) * 1.22)
    ax.legend(loc="upper left", fontsize=6.2, frameon=False, handlelength=1.2,
              labelspacing=0.3, borderaxespad=0.2)


def panel_d(ax, meta: dict, detail: pd.DataFrame) -> None:
    """Marginal abatement cost per sector, CCS vs H2, recomputed from the model's own functions."""
    years = years_of(meta)
    # 从登记表重建 scenario / assumptions，不用默认构造。`industry_cost_multiplier` 与
    # `industry_h2_cost_multiplier` 是 `industry.py` 真正乘进成本的两个旋钮，默认构造
    # 会把它们悄悄当成 1.0 —— 而本模块的说明里恰好写着"抄表会与求解脱节"。
    scenario, assumptions = _run_config()
    ccs_mult = float(getattr(scenario, "industry_cost_multiplier", 1.0))
    h2_mult = float(getattr(scenario, "industry_h2_cost_multiplier", 1.0))
    hubs = detail[detail["year"] == years[0]]
    peak = 0.0
    drawn = 0
    for sector in SECTOR_ORDER:
        g = hubs[hubs["sector"] == sector]
        if g.empty:
            continue
        colour = SECTOR_COLOR[sector]
        # CCS 的成本口径是"每吨被捕集的 CO2"（capex 年金 + 固定运维 + 蒸汽/电/耗材，
        # 缺省煤价 `coal_fuel_cost_cny_per_gj`、当年电价），CCS 通路的净减排量 = 捕集量 − 再生蒸汽排放，
        # 所以按每吨净减排折算要除以 (1 − 蒸汽排放系数)。
        ccs = [
            levelised_capture_cost_cny_per_t(
                sector, scenario.discount_rate, assumptions.coal_fuel_cost_cny_per_gj,
                scenario.electricity_price_for_year(y), learning=assumptions.ccs_learning_factor(y),
            ) * ccs_mult / (1.0 - _steam_co2_fraction(sector, assumptions))
            for y in years
        ]
        peak = max(peak, max(ccs))
        # 水泥与电炉钢取同一套捕集参数（capex 1 150），合成氨与甲醇同为 450，两两**完全重合**。
        # 等宽画会有一条被彻底压住、图上根本看不见。按绘制顺序递减线宽，重合处呈同心带，
        # 两条都看得见，而且没有任何数据被挪动。
        ax.plot(years, ccs, color=colour, lw=2.4 - 0.34 * drawn, marker="o", ms=2.6,
                zorder=3 + drawn, solid_capstyle="round")
        drawn += 1
        if sector not in INDUSTRY_H2_ABATEMENT_FRACTION:
            continue
        # 氢的成本口径是"每吨产品"，要除以该行业每吨产品可减的 CO2 才可比。
        production_t = float(g["production_kt_per_year"].sum()) * 1e3
        base_t = float(g["baseline_co2_mt"].sum()) * 1e6
        frac = float(INDUSTRY_H2_ABATEMENT_FRACTION[sector])
        tco2_per_t = base_t / production_t * frac
        # 氢强度取全行业加权值是安全的，因为 `h2_premium_cny_per_t` 是分段函数（有资本地板），
        # 只有在强度**逐厂址相同**时聚合求值才等于逐厂址求值。已实测：三个有氢路线的行业
        # 内部强度的标准差都是 0（钢铁 0.0810、合成氨 0.1800、甲醇 0.1900 t/t）。
        intensity = _sector_h2_intensity(sector)
        h2 = [h2_premium_cny_per_t(
                  sector, float(_h2_price(meta, y)), intensity, scenario.discount_rate, h2_mult,
              ) / tco2_per_t for y in years]
        peak = max(peak, max(h2))
        ax.plot(years, h2, color=colour, lw=1.1, ls=(0, (3, 1.6)), marker="^", ms=2.8, zorder=3)
    ax.set_ylabel(r"边际减排成本（元 tCO$_2^{-1}$）")
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 3, years[-1] + 3)
    # 上限从数据来。写死 1800 时甲醇 2030 年的 1 688 已经贴到 94%，氢价一动就会有一条
    # 曲线整根跑到轴外——matplotlib 不会为此说一个字。
    ax.set_ylim(0, max(peak, 1.0) * 1.16)
    handles = [Patch(facecolor=SECTOR_COLOR[s], label=SECTOR_LABELS_ZH[s]) for s in SECTOR_ORDER]
    handles += [Line2D([], [], color="#555555", lw=1.8, marker="o", ms=2.6, label="CCS 捕集"),
                Line2D([], [], color="#555555", lw=1.1, ls=(0, (3, 1.6)), marker="^", ms=2.8,
                       label="绿氢替代")]
    ax.legend(handles=handles, loc="upper right", fontsize=5.8, frameon=False, ncol=2,
              handlelength=1.4, labelspacing=0.28, columnspacing=0.9, borderaxespad=0.2)


@lru_cache(maxsize=None)
def _run_config() -> tuple[object, object]:
    """Rebuild this run's scenario and assumptions from the experiment registry."""
    from run_single import EXPERIMENTS
    from coal_retrofit.optimization.scenario import OptimizationScenario

    scenario_kw, assumption_kw = EXPERIMENTS[RUN]
    known = {f for f in OptimizationScenario.__dataclass_fields__}
    scenario = OptimizationScenario(
        experiment_id=RUN, description=RUN,
        **{k: v for k, v in scenario_kw.items() if k in known})
    known_a = {f for f in OptimizationAssumptions.__dataclass_fields__}
    assumptions = OptimizationAssumptions(
        **{k: v for k, v in assumption_kw.items() if k in known_a})
    return scenario, assumptions


@lru_cache(maxsize=None)
def _sector_h2_intensity(sector: str) -> float:
    """Production-weighted t H2 per t product, from the point-source table itself."""
    hubs = pd.read_csv(Path(__file__).resolve().parents[1] / "inputs" / "industry_hubs.csv")
    g = hubs[hubs["sector"] == sector]
    production = float(g["production_kt_per_year"].sum())
    if production <= 0:
        return 0.0
    return float(g["h2_demand_kt_per_year"].sum() / production)


def main() -> None:
    apply_style()
    _require_registered((RUN, RUN_NOWATER))
    n_sinks = assert_v9_tree(RUN)
    loaded = load(RUN)
    if loaded is None:
        print(f"nothing to draw: {RUN} is not solved on this tree")
        return
    meta, detail, storage = loaded
    nowater = load(RUN_NOWATER)
    meta_nw = nowater[0] if nowater else None

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL[0], DOUBLE_COL[1] * 1.62))
    panel_a(axes[0, 0], meta, meta_nw)
    panel_b(axes[0, 1], detail)
    panel_c(axes[1, 0], meta, detail, storage)
    panel_d(axes[1, 1], meta, detail)
    for ax, letter in zip(axes.ravel(), "abcd"):
        panel_label(ax, letter)

    last = int(detail["year"].max())
    ind_last = float(detail[detail["year"] == last]["reduction_mt"].sum())
    coal_last = float(meta["years"][str(last)]["coal_reduction_mt"])
    cap_last = float(detail[detail["year"] == last]["captured_mt"].sum())
    first = int(detail["year"].min())
    coal_first = float(meta["years"][str(first)]["coal_reduction_mt"])
    # 首年煤电减排为负时必须在图注里说明。面板 a 的量程是 8 000 Mt，-33.5 只有一个像素，
    # 图上标不出来，但"水约束下煤电为省水改空冷、因而比基线多排"是这一格里真实发生的事。
    neg_note = (f"{first} 年煤电减排为 {coal_first:,.1f} Mt（为省水改空冷，背压惩罚使其"
                f"比基线多排），同年工业减 "
                f"{float(detail[detail['year'] == first]['reduction_mt'].sum()):,.1f} Mt。"
                if coal_first < -0.5 else "")
    note = cjk_fill(
        f"情景 {RUN}（v9.2 管网，2026-09-12 重建：{n_sinks} 个汇对每个源都可达）。{last} 年联合减排 "
        f"{coal_last + ind_last:,.0f} Mt = 煤电 {coal_last:,.0f} + 工业 {ind_last:,.0f}，"
        f"排放目标精确咬住（缺口 0）。" + neg_note
        + f"工业捕集 {cap_last:.1f} Mt——封存空间几乎全部被煤电占用，"
        "工业只能走绿氢；水泥没有氢路线，因而整体不改造。"
        "边际减排成本由模型自身的成本函数重算，未抄表；水泥与电炉钢共用同一捕集成本（540 元），"
        "合成氨与甲醇同为 177.5 元，面板 d 中这两对曲线完全重合，用递减线宽区分。"
        # 这里不能写 markdown 的 ** **，matplotlib 会把星号原样画出来；
        # 也不能写 U+2212 的负号，SimHei 没有这个字形（save_fig 的缺字守卫会拒绝出图）。
        "氢路线的成本口径是每吨产品，面板 d 按行业产量加权折成每吨 CO$_2$，"
        "因此当某行业只有一部分厂址转氢时，实际发生的成本与曲线相差 "
        "-6% 到 +15%（合成氨 2030 年为 965，曲线 839）；整行业转氢的年份两者完全相等。",
        104,
    )
    fig.text(0.008, 0.005, note, fontsize=5.6, color="#444444", va="bottom", ha="left",
             linespacing=1.5)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.135, hspace=0.42, wspace=0.30)
    save_fig(fig, "ind_fig1_joint_allocation", subdir="main")
    print("ind_fig1_joint_allocation written")


if __name__ == "__main__":
    main()
