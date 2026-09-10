"""Extended Data 7: the sensitivity tornado, read against solver noise.

Three things this script refuses to do, each because it produced a wrong figure once:

1. It does not hardcode the degeneracy floor. The old value (0.3277 %) was measured on a seed
   family of a different input version; the floor is now computed from the seed replicates that
   sit next to the SA runs in `results/`: floor = 1.96 * sqrt(2) * range / d2(k), in per cent of the
   family's mean objective (CLAUDE.md §二 4). If no family is solved, no floor is drawn and every
   bar is labelled as untested -- never as "resolved".
2. It does not difference every SA run against `BASE`. Half of the registry's SA entries carry the
   CWatM SSP1-2.6 dry-season water constraint (they perturb the cooling-retrofit and retirement
   parameters that only matter under water), so their baseline is `WA_cwatm_126_dry`; the other
   half have no water and belong to `BASE`. Differencing a water-constrained run against BASE would
   put the +0.5 % water effect inside the bar. The baseline is read from the registry.
3. It does not present a point difference alone. Each bar carries the certified interval
   lo = (LB_t - INC_c)/INC_c, hi = (INC_t - LB_c)/LB_c (CLAUDE.md §二 3).

The tornado is drawn as before: one main axis with the full range, one zoom axis for the small
bars, and the floor shaded on both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_style import MM, RESULTS_DIR, save_fig

RESULTS = RESULTS_DIR
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534}
BASE_DRY = "BASE"
WATER_HEAD = "WA_cwatm_126_dry_oq_envonly"          # baseline of the water-constrained SA entries
WATER_SEEDS = (2, 3, 4)                  # its seed family -> the floor
TREAT_HEAD = "WA_cwatm_126_dry_oq"    # second family, printed for reference only
TREAT_SEEDS = (2, 3, 4, 5, 6)

# Grouped so each parameter shows both arms around zero, which is what makes a tornado readable.
GROUPS = [
    ("贴现率", [("SA_discount_3pct", "3%"), ("SA_discount_8pct", "8%")]),
    ("碳价", [("SA_carbon_low", "低"), ("SA_carbon_high", "高")]),
    ("退役成本", [("SA_retire_cost_250", "250"), ("SA_retire_cost_500", "500"),
                  ("SA_retire_cost_650", "650"), ("SA_retire_cost_1000", "1000")]),
    ("生物质成本", [("SA_biomass_cost_150", "×1.5"), ("SA_biomass_cost_200", "×2.0")]),
    ("CO$_2$ 注入能力", [("SA_injectivity_half", "减半")]),
    ("CCS 资本支出", [("SA_ccs_capex_low", "低"), ("SA_ccs_capex_high", "高")]),
    ("退役上限", [("SA_retire_cap_010", "0.10"), ("SA_retire_cap_025", "0.25")]),
    ("空冷改造资本支出", [("SA_air_capex_low", "低"), ("SA_air_capex_high", "高")]),
    ("空冷效率惩罚", [("SA_air_penalty_high", "高")]),
    ("管道走廊", [("SA_pipe_mid", "中"), ("SA_pipe_full", "全")]),
    ("氨成本", [("SA_ammonia_cost_50", "×0.5"), ("SA_ammonia_cost_70", "×0.7")]),
]

RESOLVED_C = "#3C5488"
NOISE_C = "#BBBBBB"
UNTESTED_C = "#8491B4"
FLOOR_C = "#CC3311"


def _registry() -> dict:
    """The scenario registry, imported from run_single so the baseline is never guessed."""
    from run_single import EXPERIMENTS
    return EXPERIMENTS


def baseline_of(name: str, registry: dict) -> str:
    scenario_kw, _ = registry[name]
    return WATER_HEAD if scenario_kw.get("water_mode") == "grid_supply" else BASE_DRY


def quality(name: str) -> tuple[float, float] | None:
    """(incumbent, bound) in CNY, or None if the run is not solved."""
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    sq = doc.get("solver_quality", {})
    inc = doc.get("global_objective_cny")
    lb = sq.get("objective_bound_cny", inc)
    return float(inc), float(lb)


def seed_floor(head: str, seeds: tuple[int, ...]) -> tuple[float | None, int]:
    """Degeneracy floor in % of the family mean, and the family size k (head run included)."""
    objs = []
    for name in [head] + [f"{head}_seed{s}" for s in seeds]:
        q = quality(name)
        if q is not None:
            objs.append(q[0])
    k = len(objs)
    if k < 2:
        return None, k
    rng = max(objs) - min(objs)
    return 100.0 * 1.96 * np.sqrt(2.0) * rng / D2[min(k, 6)] / float(np.mean(objs)), k


def main() -> None:
    registry = _registry()
    floor, k_floor = seed_floor(WATER_HEAD, WATER_SEEDS)
    floor_treat, k_treat = seed_floor(TREAT_HEAD, TREAT_SEEDS)
    if floor is None:
        print(f"  [warn] no seed family for {WATER_HEAD}: floor not measurable, bars drawn as untested")

    rows = []
    for label, members in GROUPS:
        for name, arm in members:
            q = quality(name)
            if q is None:
                print(f"  [skip] {name} not solved")
                continue
            base_name = baseline_of(name, registry)
            qb = quality(base_name)
            if qb is None:
                print(f"  [skip] {name}: baseline {base_name} not solved")
                continue
            inc_t, lb_t = q
            inc_c, lb_c = qb
            rows.append({
                "group": label, "arm": arm, "name": name, "base": base_name,
                "delta": 100.0 * (inc_t - inc_c) / inc_c,
                "lo": 100.0 * (lb_t - inc_c) / inc_c,
                "hi": 100.0 * (inc_t - lb_c) / lb_c,
            })
    if not rows:
        print("  nothing to draw: no SA_* run is solved on this tree")
        return

    magnitude: dict[str, float] = {}
    for r in rows:
        magnitude[r["group"]] = max(magnitude.get(r["group"], 0.0), abs(r["delta"]))
    ordered_groups = sorted(magnitude, key=magnitude.get)

    order, y, ticks = [], [], []
    pos = 0.0
    for label in ordered_groups:
        block = [r for r in rows if r["group"] == label]
        block.sort(key=lambda r: r["delta"])
        centre = pos + (len(block) - 1) / 2.0
        for r in block:
            order.append(r)
            y.append(pos)
            pos += 1.0
        ticks.append((centre, label))
        pos += 0.7

    def verdict(r: dict) -> str:
        if floor is None:
            return "untested"
        return "resolved" if abs(r["delta"]) > floor else "noise"

    colour_of = {"resolved": RESOLVED_C, "noise": NOISE_C, "untested": UNTESTED_C}

    fig = plt.figure(figsize=(183 * MM, 108 * MM))
    ax = fig.add_axes([0.235, 0.115, 0.520, 0.780])
    axz = fig.add_axes([0.815, 0.115, 0.170, 0.780])

    full = max(4.0, 1.15 * max(max(abs(r["lo"]), abs(r["hi"])) for r in order))
    ZOOM = max(0.62, 1.6 * (floor or 0.3))
    for axis in (ax, axz):
        if floor is not None:
            axis.axvspan(-floor, floor, color=FLOOR_C, alpha=0.10, lw=0, zorder=1)
        axis.axvline(0, color="#333333", lw=0.8, zorder=4)

    for r, yi in zip(order, y):
        v = verdict(r)
        colour = colour_of[v]
        hatch = "///" if v == "noise" else None
        ax.barh(yi, r["delta"], height=0.72, color=colour, edgecolor="white", lw=0.35,
                zorder=3, hatch=hatch)
        ax.plot([r["lo"], r["hi"]], [yi, yi], color="#333333", lw=0.55, zorder=5,
                solid_capstyle="butt")
        if abs(r["delta"]) <= ZOOM:
            axz.barh(yi, r["delta"], height=0.72, color=colour, edgecolor="white",
                     lw=0.35, zorder=3, hatch=hatch)
            axz.plot([max(r["lo"], -ZOOM), min(r["hi"], ZOOM)], [yi, yi], color="#333333",
                     lw=0.55, zorder=5, solid_capstyle="butt")
        else:
            edge = ZOOM * 0.93 * (1 if r["delta"] > 0 else -1)
            axz.plot([edge], [yi], marker=">" if r["delta"] > 0 else "<", ms=3.0,
                     color=colour, mec="white", mew=0.3, zorder=5)
        anchor = r["hi"] if r["delta"] >= 0 else r["lo"]
        ax.annotate(f"{r['delta']:+.2f}", xy=(anchor, yi),
                    xytext=(3 if r["delta"] >= 0 else -3, 0), textcoords="offset points",
                    va="center", ha="left" if r["delta"] >= 0 else "right",
                    fontsize=5.2, color="#333333" if v == "resolved" else colour)
        axz.annotate(r["arm"], xy=(0, yi), xytext=(0, 0), textcoords="offset points",
                     va="center", ha="center", fontsize=5.2, color="#555555",
                     bbox=dict(facecolor="white", lw=0, alpha=0.85, pad=0.6), zorder=6)

    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks], fontsize=6.2)
    ax.set_ylim(-1.0, pos - 0.2)
    ax.set_xlim(-0.55 * full, 1.05 * full)
    ax.set_xlabel("相对各自基准的系统成本变化（占基准目标函数的 %）；细线为可证区间", fontsize=6.4)
    ax.tick_params(labelsize=5.8, length=1.8)
    ax.grid(axis="x", lw=0.3, alpha=0.30)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    axz.set_xlim(-ZOOM, ZOOM)
    axz.set_xticks([-round(ZOOM * 0.8, 1), 0.0, round(ZOOM * 0.8, 1)])
    axz.set_ylim(-1.0, pos - 0.2)
    axz.set_yticks([])
    axz.set_xlabel("放大视图（%）", fontsize=6.2)
    axz.tick_params(labelsize=5.4, length=1.8)
    axz.grid(axis="x", lw=0.3, alpha=0.30)
    axz.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axz.spines[side].set_visible(False)
    if floor is not None:
        axz.annotate(f"地板 ±{floor:.2f}%", xy=(0.0, -0.90), fontsize=5.2, color=FLOOR_C,
                     ha="center", va="center", fontweight="bold")
    axz.annotate("超出量程", xy=(ZOOM * 0.93, -0.90), fontsize=5.0, color="#666666",
                 ha="center", va="center")

    n_res = sum(1 for r in order if verdict(r) == "resolved")
    n_noise = sum(1 for r in order if verdict(r) == "noise")
    handles = []
    if floor is not None:
        handles += [Patch(facecolor=RESOLVED_C, label=f"可分辨（{len(order)} 项中的 {n_res} 项）"),
                    Patch(facecolor=NOISE_C, hatch="///", label=f"落在求解器噪声内（{n_noise} 项）"),
                    Patch(facecolor=FLOOR_C, alpha=0.10,
                          label=f"简并度地板（{WATER_HEAD} 种子族，k = {k_floor}）")]
    else:
        handles += [Patch(facecolor=UNTESTED_C, label="未对地板检验（种子族未求解）")]
    ax.legend(handles=handles, fontsize=5.4, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 0.01), handlelength=1.3, handletextpad=0.6, labelspacing=0.35)

    n_water = sum(1 for r in order if r["base"] == WATER_HEAD)
    if floor is not None:
        head = (f"{len(order)} 个情景在同一输入版本上求解；其中 {n_noise} 个对目标函数的影响小于求解器自身的\n"
                f"可复现精度（斜纹），不构成敏感性。{n_water} 个带水约束的情景以 {WATER_HEAD} 为基准，其余以 BASE 为基准。")
    else:
        head = (f"{len(order)} 个情景在同一输入版本上求解；种子族尚未求解，地板不可测，\n"
                f"任何一根柱都不能读成“可分辨”。{n_water} 个带水约束的情景以 {WATER_HEAD} 为基准，其余以 BASE 为基准。")
    fig.text(0.235, 0.975, head, fontsize=6.8, linespacing=1.3, va="top")
    fig.text(0.010, 0.975, "a", fontsize=8.0, fontweight="bold", va="top", family="Arial")

    save_fig(fig, "ed_fig7_sensitivity", subdir="extended")

    print(f"  floor from {WATER_HEAD} family: "
          + (f"{floor:.4f}% (k = {k_floor})" if floor is not None else "not measurable"))
    print(f"  floor from {TREAT_HEAD} family: "
          + (f"{floor_treat:.4f}% (k = {k_treat})" if floor_treat is not None else "not measurable"))
    for r in sorted(order, key=lambda r: -abs(r["delta"])):
        flag = {"resolved": "", "noise": "   <- inside noise", "untested": "   <- untested"}[verdict(r)]
        print(f"  {r['name']:26s} vs {r['base']:18s} {r['delta']:+8.3f}%  [{r['lo']:+.3f}, {r['hi']:+.3f}]{flag}")
    if floor is not None:
        print(f"\n  {n_res} of {len(order)} clear the {floor:.3f}% floor")


if __name__ == "__main__":
    main()
