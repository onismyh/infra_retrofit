"""Check that runs which are compared against each other are actually comparable.

WHY THIS EXISTS. Every contrast published in v3 differenced two BUILDS, not two scenarios.
The controls were solved 2026-08-09/10; the treatments on 08-17, and between the two batches
`data_prep.py`, `builders/water.py`, `inputs/water_nodes.csv` and
`inputs/water_supply_links.csv` all changed. The evidence sat in the Gurobi logs the whole
time and nothing read it: the run used as the treatment carried 632 446 columns and model
fingerprint 0xb8630838, while the three "seed replicates" supposed to be bit-identical to it
carried 632 442 and 0xbe7b31c2 -- four water-supply-link variables that existed in one model
and not the other. The 0.249% reported as this model's solver degeneracy was that difference.

A second, independent failure rode along: Gurobi is deterministic only for a fixed
(model, parameters, THREAD COUNT), and `OptimizationScenario.solver_threads` defaults to 0 = auto,
so no run before this campaign pinned it. Observed thread counts across
runs that were differenced against each other: 32, 14, 12, 9, 8.

WHAT THIS CHECKS. `solver_provenance._run_provenance` now stamps every result JSON with the model
fingerprint, its dimensions, the thread parameter and the seed. This script reads them back
and answers two questions no figure could previously ask:

  1. seed replicates -- MUST agree on fingerprint. Disagreement means they are replicates of
     different models, and any floor measured from them is a version artefact, not degeneracy.
  2. any contrast -- MAY differ in fingerprint when the scenario legitimately changes the
     model (a different availability file, a different supply multiplier), but the thread
     count must still be pinned and equal, or the comparison inherits solver nondeterminism
     on top of the physics.

Exit status is 1 if any hard rule is violated, so this can gate a figure build.

2026-09-27 起 result.json 带 `resolved` 段（全部参数、求解树、运行选项、`COAL_RETROFIT_*` 环境变量），每组对照
另列两边的参数差与环境变量差；LP 松弛或热启动两边不一致记 failure。记了 failure 的对照不再给相减的判断。
`--pair A B` 只核这两次求解，不跑下面写死的 v9 seed 族与对照表；缺一边、任一边没有 `resolved` 段（2026-09-27
之前落盘，参数与环境变量都核不了）、两边都是 LP 松弛、任一边没有可用的解（目标函数不是有限值），也记 failure。
本脚本读自己所在树的 `results/`：`scripts/` 这份读仓库根 `results/`，`_indtree/scripts/` 这份读
`_indtree/results/`（`ST_` 系的结果在这里）；`--results` 可换目录（只配 `--pair`）。

2026-09-28 起（求解流程进情景定义）又加了几条：
- 可证区间按 CLAUDE.md 二.3，下界用 result.json 记的 ObjBound（`objective_bound_cny`）；没有这个键（旧结果）或记的
  是空值时按 gap 反推。对照（c，`--pair` 的第一个结果）的目标函数或下界不是正数时只给绝对区间（元）。
- 热启动看两处：情景 `warm_start = "lp_relax"`（运行器自动两步）或手工设 COAL_RETROFIT_START_SOL，两边一个热启动
  一个没有记 failure。碳价（`carbon_price_cny_per_t_by_year`）两边不同记 failure：目标函数含的碳价支出不同。
- `--pair` 另比两边都有的输入摘要（`digest_*`），不同就记 failure（CLAUDE.md 二.6）；两边读的不是同一个文件的
  （`resolved.input_files` 不同）与只有一边有的只列出。任一边没有 `resolved.code`（这之前落盘，跨 PR #11 连续 hub
  掺烧等式的模型改动分不出来）记 failure；提交号不同、求解时有未提交的改动、记不了提交号，只告警。

    python scripts/check_run_provenance.py
    python scripts/check_run_provenance.py --strict   # also fail on unpinned threads
    python _indtree/scripts/check_run_provenance.py --pair ST_BASE ST_WA_cwatm_126_dry_oq
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  （把仓库根的 src/ 放进 sys.path）

from coal_retrofit.scenarios import diff_resolved

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Families whose members must be bit-identical models. Anything differing only by the Gurobi
# seed belongs here; a mismatch inside a family is a hard failure.
SEED_FAMILIES = {
    "WA_cwatm_126_dry_wd085": ["WA_cwatm_126_dry_wd085",
                               "WA_cwatm_126_dry_wd085_seed2",
                               "WA_cwatm_126_dry_wd085_seed3",
                               "WA_cwatm_126_dry_wd085_seed4",
                               "WA_cwatm_126_dry_wd085_seed5",
                               "WA_cwatm_126_dry_wd085_seed6"],
    "WA_cwatm_126_dry": ["WA_cwatm_126_dry",
                         "WA_cwatm_126_dry_seed2",
                         "WA_cwatm_126_dry_seed3",
                         "WA_cwatm_126_dry_seed4"],
}

# Contrasts the figures actually draw. These MAY differ in fingerprint -- the scenario changes
# the model on purpose -- but they must share a pinned thread count.
# Non-water scenarios that ALSO build the water network and were left on the pre-08-17
# build by stage 1. BASE is the one that matters most: it appears in a main-figure contrast
# and is the BASE_DIR every Extended Data figure reads.
LEGACY_BUILD = (
    "BASE", "BASE_zero", "BASE_neg", "WA_grid_200km",
    "RQ3_ccs_only", "RQ3_no_ammonia", "RQ3_no_biomass", "RQ3_no_ccs", "RQ3_retire_only",
)

CONTRASTS = [
    ("Fig 3 accounted", "BASE", "WA_cwatm_126_dry"),
    ("Fig 3 reserved", "WA_cwatm_126_dry", "WA_cwatm_126_dry_wd085"),
    ("Fig 3 climate", "WA_cwatm_126_dry_wd085", "WA_cwatm_370_dry_wd085"),
    ("Fig 5 frozen", "WA_cwatm_126_dry_wd085", "WA_cwatm_126_dry_wd085_noair_capfree"),
    ("Fig 5 bind wgap126", "WA_wgap_126_dry", "WA_wgap_126_dry_wd085"),
    ("Fig 5 bind wgap370", "WA_wgap_370_dry", "WA_wgap_370_dry_wd085"),
    ("capfree pair", "WA_cwatm_126_dry_capfree", "WA_cwatm_126_dry_wd085_capfree"),
    ("biomass realism", "WA_cwatm_126_dry_bio015", "WA_cwatm_126_dry_wd085_bio015"),
]

FIELDS = ("fingerprint", "num_vars", "num_constrs", "num_nonzeros",
          "threads_param", "threads_pinned", "seed", "mip_focus")

D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847}

# `resolved.env` 记下求解时所有 COAL_RETROFIT_* 环境变量。运行器的三个兼容开关只看设没设（按非空判断；热启动的
# .sol 路径按情景不同）；WRITE_SOL 只决定写不写 .sol，不比；其余（MIPFOCUS、GUROBI_SEED 等）按原值列差，
# 它们实际生效的值记在情景字段 `mip_focus`、`solver_seed` 里，MIPFocus 不同另由 solver_quality 的 mip_focus 判 failure。
ENV_SWITCHES = ("COAL_RETROFIT_LP_RELAX", "COAL_RETROFIT_START_SOL", "COAL_RETROFIT_LOG_INCUMBENTS")
ENV_SKIPPED = ("COAL_RETROFIT_WRITE_SOL",)
# 两边不一致就不能相减。热启动的两种来源（情景字段与 START_SOL）合起来由 `_warm_started` 判。
ENV_BLOCKING = {
    "COAL_RETROFIT_LP_RELAX": "一边解的是 LP 松弛（整数变量改成了连续变量）",
}
# 这些参数两边不同，目标函数的口径就不同，目标函数不能相减。
OBJECTIVE_BASIS = {
    "scenario.carbon_price_cny_per_t_by_year": "目标函数含的碳价支出不同，只能比路径结构",
}


def load(name, results=RESULTS):
    path = results / f"{name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload.get("solver_quality", {})
    out = {key: quality.get(key) for key in FIELDS}
    out["objective"] = payload.get("global_objective_cny")
    out["mip_gap"] = quality.get("mip_gap")
    out["bound"] = quality.get("objective_bound_cny")
    out["digests"] = {key: value for key, value in quality.items() if key.startswith("digest_")}
    out["resolved"] = payload.get("resolved")
    return out


def _lp_relaxed(row):
    return bool(((row["resolved"] or {}).get("env") or {}).get("COAL_RETROFIT_LP_RELAX"))


def _warm_started(resolved):
    """热启动了没有：情景 `warm_start = "lp_relax"`（运行器自动两步），或手工设了 COAL_RETROFIT_START_SOL。
    两者是同一套流程（实现说明 §9.7），只看有没有。"""
    scenario = resolved.get("scenario") or {}
    env = resolved.get("env") or {}
    return scenario.get("warm_start", "none") != "none" or bool(env.get("COAL_RETROFIT_START_SOL"))


def _env_view(resolved):
    env = resolved.get("env") or {}
    view = {key: ("已设" if env.get(key) else "未设") for key in ENV_SWITCHES}
    view.update({key: value for key, value in sorted(env.items())
                 if key not in ENV_SWITCHES and key not in ENV_SKIPPED})
    return view


def describe_params(label, names, a, b, failures, warnings):
    """两次求解的参数差与环境变量差（`resolved` 段）。LP 松弛或热启动两边不一致、碳价不同（`OBJECTIVE_BASIS`）
    记进 *failures*；提交号不同、有未提交的改动、记不了提交号记进 *warnings*（`resolved.code`，旧结果没有这一项）。

    任一边没有 `resolved`（2026-09-27 之前落盘）就说明比不了。
    """
    if a.get("resolved") is None or b.get("resolved") is None:
        return ["    -> 参数差：至少一边没有 resolved 段（2026-09-27 之前落盘），比不了"]
    diffs = diff_resolved(a["resolved"], b["resolved"])
    lines = [f"    -> 参数差 {len(diffs)} 项："] if diffs else ["    -> 参数完全相同"]
    lines += [f"         {key}: {va!r} -> {vb!r}" for key, va, vb in diffs]
    for key, _, _ in diffs:
        if key in OBJECTIVE_BASIS:
            failures.append(f"{label}: {key} 两边不同 -- {OBJECTIVE_BASIS[key]}，目标函数不能相减")
    ea, eb = _env_view(a["resolved"]), _env_view(b["resolved"])
    env_diffs = [(key, ea.get(key), eb.get(key)) for key in [*ea, *(k for k in eb if k not in ea)]
                 if ea.get(key) != eb.get(key)]
    lines.append(f"    -> 环境变量差 {len(env_diffs)} 项：" if env_diffs else "    -> 求解相关的环境变量相同")
    lines += [f"         {key}: {va!r} -> {vb!r}" for key, va, vb in env_diffs]
    for key, va, vb in env_diffs:
        if key in ENV_BLOCKING:
            failures.append(f"{label}: {key} 两边不同（{va} vs {vb}）-- {ENV_BLOCKING[key]}，这两次求解不能相减")
    warm = [_warm_started(a["resolved"]), _warm_started(b["resolved"])]
    if warm[0] != warm[1]:
        which = "、".join(name for name, flag in zip(names, warm) if flag)
        failures.append(f"{label}: 只有 {which} 热启动了（实现说明 §9.7：同一族必须同法），这两次求解不能相减")
    codes = [a["resolved"].get("code"), b["resolved"].get("code")]
    if all(code is not None for code in codes):
        commits = [code.get("commit") for code in codes]
        lines.append(f"    -> 提交号：{commits[0]} -> {commits[1]}")
        if None in commits:
            warnings.append(f"{label}: 至少一边求解时记不了提交号（没有 git），两边是不是同一份代码核不了")
        elif commits[0] != commits[1]:
            warnings.append(f"{label}: 两边的提交号不同（{commits[0][:9]} vs {commits[1][:9]}），"
                            f"核对这两个提交之间没有模型改动（CLAUDE.md 二.7）")
        warnings += [f"{label}: {name} 求解时有未提交的改动，提交号不能完全代表所用的代码"
                     for name, code in zip(names, codes) if code.get("dirty")]
    return lines


def describe(name, row):
    if row is None:
        return f"    {name:44s} NOT SOLVED"
    if row["fingerprint"] is None:
        return (f"    {name:44s} NO PROVENANCE -- solved before solver_provenance._run_provenance "
                f"existed; re-solve to make it checkable")
    return (f"    {name:44s} fp {row['fingerprint']}  vars {row['num_vars']}  "
            f"nz {row['num_nonzeros']}  threads {row['threads_param']}  seed {row['seed']}")


def check_seed_families(failures, warnings):
    print("=" * 96)
    print("seed families -- members MUST share a model fingerprint")
    print("=" * 96)
    for family, members in SEED_FAMILIES.items():
        rows = {name: load(name) for name in members}
        present = {n: r for n, r in rows.items() if r is not None}
        print()
        print(f"  {family}   ({len(present)}/{len(members)} solved)")
        for name in members:
            print(describe(name, rows[name]))
        prints = {r["fingerprint"] for r in present.values() if r["fingerprint"] is not None}
        if len(prints) > 1:
            failures.append(f"{family}: seed replicates span {len(prints)} model fingerprints "
                            f"{sorted(prints)} -- any floor from them is a version artefact")
        threads = {r["threads_param"] for r in present.values()
                   if r["threads_param"] is not None}
        if len(threads) > 1:
            failures.append(f"{family}: seed replicates ran on {sorted(threads)} threads -- "
                            f"Gurobi is deterministic only at a fixed thread count, so the "
                            f"spread mixes thread nondeterminism into the seed effect")
        seeds = [r["seed"] for r in present.values() if r["seed"] is not None]
        if len(seeds) != len(set(seeds)):
            warnings.append(f"{family}: duplicate seeds {sorted(seeds)} -- a repeated seed "
                            f"measures nothing, Gurobi reproduces it exactly")
        if len(present) >= 2:
            objectives = [r["objective"] for r in present.values()]
            spread = 100.0 * (max(objectives) - min(objectives)) / min(objectives)
            d2 = D2.get(len(objectives), 3.078)
            sigma = spread / d2
            print(f"    -> raw objective range {spread:.4f}% over n = {len(objectives)}; "
                  f"sigma ~ {sigma:.4f}% (range / d2); 95% envelope on a DIFFERENCE of two "
                  f"such runs = {1.96 * (2 ** 0.5) * sigma:.4f}%")


def lower_bound(row):
    """下界：result.json 记的 ObjBound（`objective_bound_cny`）。没有这个键（旧结果）或值为空时按 gap 反推：Gurobi 的
    gap = |ObjBound − ObjVal| / |ObjVal|，最小化时 ObjBound ≤ ObjVal，所以 LB = INC − gap·|INC|。"""
    if row.get("bound") is not None:
        return float(row["bound"])
    objective = float(row["objective"])
    return objective - float(row["mip_gap"] or 0.0) * abs(objective)


def certified_interval(a, b):
    """CLAUDE.md 二.3 的可证区间，a 是对照（c）、b 是处理（t）：lo = (LB_t − INC_c)/INC_c，hi = (INC_t − LB_c)/LB_c。

    返回 (点估计, lo, hi, 单位)。INC_c 或 LB_c 不是正数时（toy 的目标函数就是负的）相对区间的正负与大小都不对，
    只给绝对区间 [LB_t − INC_c, INC_t − LB_c]，单位是元。
    """
    inc_c, inc_t = float(a["objective"]), float(b["objective"])
    lb_c, lb_t = lower_bound(a), lower_bound(b)
    if inc_c > 0 and lb_c > 0:
        return 100.0 * (inc_t - inc_c) / inc_c, 100.0 * (lb_t - inc_c) / inc_c, 100.0 * (inc_t - lb_c) / lb_c, "%"
    return inc_t - inc_c, lb_t - inc_c, inc_t - lb_c, "元"


def describe_digests(label, names, a, b, failures):
    """`--pair`：两边都有的输入摘要（`digest_*`）逐项比，不同就记 failure（CLAUDE.md 二.6：不同输入版本的结果不得
    相减）。两边读的不是同一个文件（`resolved.input_files` 不同：部门目标、产量指数的文件随情景的来源换）、
    只有一边有（如无水约束时没有水的文件）的只列出。"""
    da, db = a["digests"], b["digests"]
    files = [(row["resolved"] or {}).get("input_files") or {} for row in (a, b)]
    lines, same = [], 0
    for key in [*da, *(k for k in db if k not in da)]:
        logical = key.removeprefix("digest_")
        fa, fb = files[0].get(logical), files[1].get(logical)
        if key not in da or key not in db:
            lines.append(f"         {key}: 只有 {names[0] if key in da else names[1]} 读了，不比")
        elif da[key] == db[key]:
            same += 1
        elif fa is not None and fb is not None and fa != fb:
            lines.append(f"         {key}: 两边读的文件不同（{fa} vs {fb}），不比")
        else:
            lines.append(f"         {key}: {da[key]} -> {db[key]}")
            failures.append(f"{label}: 输入 {logical} 两边不同（{da[key]} vs {db[key]}）-- CLAUDE.md 二.6："
                            f"不同输入版本的结果不得相减")
    if not da and not db:
        return ["    -> 两边都没有输入摘要，输入版本核不了"]
    return [f"    -> 输入摘要 {same} 项相同" + (f"，另有 {len(lines)} 项：" if lines else ""), *lines]


def check_contrasts(failures, warnings, strict, contrasts=CONTRASTS, results=RESULTS, require_both=False):
    """*require_both*（`--pair`）时缺一边、任一边没有 `resolved` 或 `resolved.code`、两边都是 LP 松弛、任一边没有可用的解、
    输入摘要不同，都记 failure；写死的对照表照旧：缺一边只记 warning，没有 `resolved` 只注明比不了。"""
    print()
    print("=" * 96)
    print("published contrasts -- fingerprints may differ, the thread pin may not")
    print("=" * 96)
    for label, base, variant in contrasts:
        a, b = load(base, results), load(variant, results)
        print()
        print(f"  {label}")
        print(describe(base, a))
        print(describe(variant, b))
        if a is None or b is None:
            if require_both:
                missing = "、".join(name for name, row in ((base, a), (variant, b)) if row is None)
                failures.append(f"{label}: {results} 下没有 {missing} 的结果")
            else:
                warnings.append(f"{label}: one side not solved yet")
            continue
        before = len(failures)
        for line in describe_params(label, (base, variant), a, b, failures, warnings):
            print(line)
        unchecked = [name for name, row in ((base, a), (variant, b)) if row["resolved"] is None]
        if require_both and unchecked:
            failures.append(f"{label}: {'、'.join(unchecked)} 没有 resolved 段（2026-09-27 之前落盘），"
                            f"参数与环境变量都核不了，重解后再比")
        if require_both:
            for line in describe_digests(label, (base, variant), a, b, failures):
                print(line)
            uncoded = [name for name, row in ((base, a), (variant, b))
                       if row["resolved"] is not None and "code" not in row["resolved"]]
            if uncoded:
                failures.append(f"{label}: {'、'.join(uncoded)} 的 resolved 段没有 code（加这一项之前落盘）："
                                f"是不是在 PR #11（连续 hub 掺烧等式，模型改动）之后求解的核不了，重解后再比")
            # 只有一边是 LP 松弛的，上面的环境变量差已记 failure；两边都是时环境变量相同，要另记。
            if _lp_relaxed(a) and _lp_relaxed(b):
                failures.append(f"{label}: 两边都是 LP 松弛的解（整数变量改成了连续变量），"
                                f"只是下界，不能拿来相减")
            for name, row in ((base, a), (variant, b)):
                objective = row["objective"]
                if not isinstance(objective, (int, float)) or not math.isfinite(objective):
                    failures.append(f"{label}: {name} 没有可用的解（目标函数是 {objective!r}：求解状态不可接受，"
                                    f"结果表是补零的），不能拿来相减")
        if a["fingerprint"] is None or b["fingerprint"] is None:
            warnings.append(f"{label}: at least one side has no provenance, so comparability "
                            f"cannot be checked")
        # Older stamps read MIPFocus from an environment variable and wrote 0 when it was unset,
        # although the model ran at the `_new_gurobi_model` default of 1; current stamps read it
        # back from the model and write 1. A 0-vs-1 failure across the two stamp versions is
        # intended: re-solve the older side rather than difference across them.
        fa, fb = a.get("mip_focus"), b.get("mip_focus")
        if fa is not None and fb is not None and fa != fb:
            failures.append(
                f"{label}: MIPFocus differs ({fa} vs {fb}) -- Gurobi is deterministic "
                f"only for a fixed (model, params, threads), and MIPFocus is a param, "
                f"so this contrast mixes two search strategies")
        ta, tb = a["threads_param"], b["threads_param"]
        if ta is not None and tb is not None:
            if ta != tb:
                failures.append(f"{label}: thread counts differ ({ta} vs {tb})")
            elif ta == 0:
                message = (f"{label}: both sides left Threads at 0 (auto), so neither is "
                           f"reproducible")
                (failures if strict else warnings).append(message)
        if a["objective"] and b["objective"]:
            point, lo, hi, unit = certified_interval(a, b)
            verdict = "SIGN RESOLVED" if lo > 0 or hi < 0 else "inside solver bounds"
            if len(failures) > before:
                verdict = "不可相减（见 FAILURES）"
            if unit == "%":
                print(f"    -> effect {point:+.3f}%  certified [{lo:+.3f}, {hi:+.3f}]%  -> {verdict}")
            else:
                print(f"    -> effect {point:+.4e} 元  certified [{lo:+.4e}, {hi:+.4e}] 元"
                      f"（对照的目标函数或下界不是正数，不给相对区间）  -> {verdict}")


def check_legacy(warnings):
    """Report scenarios still carrying no provenance stamp, i.e. still on an older build."""
    print()
    print("=" * 96)
    print("scenarios that build the water network but predate the provenance stamp")
    print("=" * 96)
    stale = []
    for name in LEGACY_BUILD:
        row = load(name)
        if row is None:
            print(f"    {name:44s} NOT SOLVED")
            continue
        print(describe(name, row))
        if row["fingerprint"] is None:
            stale.append(name)
    if stale:
        warnings.append(f"{len(stale)} scenario(s) still on a pre-provenance build "
                        f"({', '.join(stale)}); every Extended Data figure reads BASE, and "
                        f"Fig 3's first contrast differences BASE against a re-solved run")


def main():
    parser = argparse.ArgumentParser(description="Check run comparability")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when a compared run left Threads at 0 (auto)")
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"),
                        help="只核这两次求解（结果名），不跑写死的 v9 seed 族与对照表；缺一边、没有 resolved 段或"
                             " resolved.code、一边 LP 松弛或热启动而另一边不是、两边都是 LP 松弛、没有可用的解、碳价不同、"
                             "同一个输入文件的摘要不同、线程数或 MIPFocus 不同，都记 failure；提交号不同、记不了提交号、有未提交的改动只告警")
    parser.add_argument("--results", type=Path, default=None,
                        help="--pair 读哪个结果目录（缺省：本脚本所在树的 results/）")
    args = parser.parse_args()
    if args.results is not None and not args.pair:
        parser.error("--results 只配 --pair 用")
    failures, warnings = [], []
    if args.pair:
        results = args.results if args.results is not None else RESULTS
        print(f"结果目录：{results}")
        check_contrasts(failures, warnings, args.strict, contrasts=[("pair", *args.pair)],
                        results=results, require_both=True)
    else:
        check_seed_families(failures, warnings)
        check_contrasts(failures, warnings, args.strict)
        check_legacy(warnings)
    print()
    print("=" * 96)
    if warnings:
        print(f"WARNINGS ({len(warnings)})")
        for item in warnings:
            print(f"  - {item}")
    if failures:
        print(f"FAILURES ({len(failures)})")
        for item in failures:
            print(f"  ! {item}")
        return 1
    print("no hard failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
