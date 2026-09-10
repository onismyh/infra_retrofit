"""Read the joint coal + industry runs and report what the coupling actually did.

Two things this answers and nothing else can:

1. **Who abates.** Under one joint target the solver allocates between coal power and five
   industrial sectors. The split, and how it moves when the basin water cap is switched on,
   is the whole point of making industry endogenous.
2. **What the water cap costs.** `IND_BASE` -> `IND_WA_*_oq` is a legitimate difference: same
   model, same target, one knob. (A difference against any NON-`IND_` run is not legitimate --
   objective, target and source population all changed at once.)

Usage:
    python scripts/summarize_industry_runs.py [BASE_RUN] [CAP_RUN]

Defaults to IND_BASE and IND_WA_cwatm_126_dry_oq under `results/`.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Windows 控制台默认 GBK，报告里有 U+26A0 和中文行业名，不重设会在打印松弛告警时
# 抛 UnicodeEncodeError —— 恰好把最该看见的那几行吃掉。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SECTOR_ZH = {
    "steel_bf_bof": "钢铁(高炉-转炉)",
    "steel_eaf": "钢铁(电炉)",
    "cement": "水泥",
    "ammonia": "合成氨",
    "methanol": "甲醇",
}


def _load(name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one run's JSON summary and the CSVs this report needs.

    Raises:
        FileNotFoundError: The run has not been solved.
    """
    meta_path = RESULTS / f"{name}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; solve it with scripts/run_single.py {name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    run_dir = RESULTS / name
    industry = pd.read_csv(run_dir / "industry_detail.csv")
    resource = pd.read_csv(run_dir / "resource_use.csv")
    slack = pd.read_csv(run_dir / "slack_detail.csv")
    edges = pd.read_csv(run_dir / "network_edges.csv")
    storage = pd.read_csv(run_dir / "storage_utilization.csv")
    return meta, industry, resource, slack, edges, storage


def report(base_name: str, cap_name: str) -> None:
    """Print the cross-sector allocation and what the basin cap changed."""
    runs: dict[str, tuple] = {}
    for name in (base_name, cap_name):
        try:
            runs[name] = _load(name)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
    if not runs:
        print("no solved joint runs found")
        return

    for name, (meta, industry, resource, slack, edges, storage) in runs.items():
        print("=" * 78)
        print(f"{name}   目标 {meta['global_objective_cny']:.6e} CNY   "
              f"gap {meta['solver_quality'].get('mip_gap', float('nan')):.4f}   "
              f"{meta['elapsed_seconds']:.0f}s")
        print("=" * 78)
        for year in sorted(int(y) for y in meta["years"]):
            block = meta["years"][str(year)]
            ind = block.get("industry")
            shortfall = float(block["target_shortfall_mt"])
            coal = float(block.get("coal_reduction_mt", float("nan")))
            line = f"  {year}  煤电减排 {coal:7.1f} Mt  缺口 {shortfall:8.2f} Mt"
            if ind:
                line += (f" | 工业 减排 {ind['reduction_mt']:7.1f} Mt  捕集 {ind['captured_mt']:7.1f} Mt"
                         f"  取水 {ind['water_m3'] / 1e8:6.2f} 亿m3"
                         f"  成本 {ind['cost_cny'] / 1e12:5.3f} 万亿"
                         f"  H2 {ind['h2_price_cny_per_kg']:.1f} 元/kg")
            print(line)
            if ind:
                routes = ind["share_by_route"]
                print("        通路(按排放加权): " + "  ".join(
                    f"{k}={v:.1%}" for k, v in routes.items()))

        if len(industry):
            print("\n  分行业（末年，按排放加权的通路份额）:")
            last = int(industry["year"].max())
            block = industry[industry["year"] == last]
            for sector, group in block.groupby("sector"):
                weight = group["baseline_co2_mt"]
                total = float(weight.sum())
                if total <= 0:
                    continue
                shares = {r: float((group[f"share_{r}"] * weight).sum() / total)
                          for r in ("unabated", "ccs", "h2")}
                print(f"    {SECTOR_ZH.get(sector, sector):16s} 基线 {total:7.1f} Mt  "
                      f"不改造 {shares['unabated']:5.1%}  CCS {shares['ccs']:5.1%}  H2 {shares['h2']:5.1%}"
                      f"  捕集 {group['captured_mt'].sum():6.1f} Mt")

        basin = resource[resource["resource_type"] == "water_basin_quota"]
        if len(basin):
            print("\n  流域取水额度利用率:")
            for year, group in basin.groupby("year"):
                over = group[group["utilization"] > 1.0]
                worst = ", ".join(f"{r.region}={r.utilization:.2f}" for r in over.itertuples())
                print(f"    {year}: 越限 {len(over)}/{len(group)} 个流域"
                      + (f"  [{worst}]" if worst else ""))
        # Industrial CO2 actually moving through the shared network. `runtime_industry_branch`
        # is each hub's own spur, so the flow on it IS that hub's injection into the pipeline
        # system -- the direct evidence that source-sink matching was exercised on the
        # industrial side rather than merely wired up.
        if "edge_class" in edges.columns:
            spur = edges[edges["edge_class"] == "runtime_industry_branch"]
            flow_col = next((c for c in ("flow_mtpa", "edge_flow_mtpa", "flow_mt_per_year")
                             if c in edges.columns), None)
            if len(spur) and flow_col:
                print("")
                print("  工业支线上的 CO2 流量（源汇匹配是否真被起用）:")
                for year, group in spur.groupby("year"):
                    active = group[group[flow_col] > 1e-6]
                    print(f"    {year}: {len(active):3d}/{len(group)} 条支线有流量，"
                          f"合计 {group[flow_col].sum():7.1f} Mt/yr")
        used_col = next((c for c in ("storage_use_mtpa", "injected_mtpa", "use_mtpa")
                         if c in storage.columns), None)
        if len(storage) and used_col:
            print("")
            print("  封存汇利用（全部源合计）:")
            for year, group in storage.groupby("year"):
                active = group[group[used_col] > 1e-6]
                print(f"    {year}: {len(active):2d}/{len(group)} 个汇在用，"
                      f"合计注入 {group[used_col].sum():7.1f} Mt/yr")

        # Every slack family the model can pay, not just the ones expected to bind. A report
        # that greps only for the constraint it predicts would stay silent through the one it
        # did not -- and silence reads exactly like compliance. Names must match
        # `results._build_slack_detail_table` verbatim; `edge_capacity` was written here as
        # `pipeline_capacity` at first, which would have hidden every pipeline breach.
        for kind, label in (("storage_injectivity", "注入能力"),
                            ("storage_capacity", "封存容量"),
                            ("edge_capacity", "管道容量"),
                            ("biomass_supply", "生物质供给"),
                            ("ammonia_supply", "氨供给"),
                            ("water_supply", "逐节点水（生态流量）"),
                            ("emission_target", "排放目标缺口")):
            rows = slack[slack["constraint_type"] == kind]
            if len(rows):
                print("")
                print(f"  ⚠ {label}松弛（模型付了 big-M 而没有守住约束）:")
                for year, group in rows.groupby("year"):
                    unit = str(group["unit"].iloc[0])
                    print(f"    {year}: {len(group)} 处，合计 {group['slack_value'].sum():.3g} {unit}")

        breach = slack[slack["constraint_type"] == "water_basin_quota"]
        if len(breach):
            print("\n  ⚠ 流域额度松弛（模型付了 big-M 而没有守住指标）:")
            for row in breach.itertuples():
                print(f"    {row.year} {row.node_id}: {row.slack_value / 1e8:.3f} 亿 m3")
        else:
            print("\n  流域额度松弛: 无")
        print()

    if len(runs) == 2:
        base_meta = runs[base_name][0]
        cap_meta = runs[cap_name][0]
        d = cap_meta["global_objective_cny"] - base_meta["global_objective_cny"]
        print("=" * 78)
        print(f"流域总量指标的代价（{base_name} -> {cap_name}，同一模型、同一目标、一个旋钮）:")
        print(f"  目标函数 {base_meta['global_objective_cny']:.6e} -> "
              f"{cap_meta['global_objective_cny']:.6e}   {d / base_meta['global_objective_cny']:+.3%}")
        base_ind = runs[base_name][1]
        cap_ind = runs[cap_name][1]
        if len(base_ind) and len(cap_ind):
            last = int(base_ind["year"].max())
            b = base_ind[base_ind["year"] == last]["captured_mt"].sum()
            c = cap_ind[cap_ind["year"] == last]["captured_mt"].sum()
            print(f"  {last} 年工业捕集 {b:.1f} -> {c:.1f} Mt   {c - b:+.1f}")
            bw = base_ind[base_ind["year"] == last]["water_m3"].sum() / 1e8
            cw = cap_ind[cap_ind["year"] == last]["water_m3"].sum() / 1e8
            print(f"  {last} 年工业取水 {bw:.2f} -> {cw:.2f} 亿 m3   {cw - bw:+.2f}")
        print()
        print("  ⚠ 不得与任何非 IND_ 运行相减：目标函数、约束与源群同时变了。")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base_name = sys.argv[1] if len(sys.argv) > 1 else "IND_BASE"
    cap_name = sys.argv[2] if len(sys.argv) > 2 else "IND_WA_cwatm_126_dry_oq"
    report(base_name, cap_name)


if __name__ == "__main__":
    main()
