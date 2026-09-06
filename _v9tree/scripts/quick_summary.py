"""Quick summary of all scenario results.

Usage:
    python scripts/quick_summary.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

SCENARIOS = [
    "BASE", "BASE_zero", "BASE_neg",
    "RQ3_no_ammonia", "RQ3_no_biomass", "RQ3_no_ccs",
    "RQ3_retire_only", "RQ3_ccs_only",
    "WA_grid_200km",
]


def main() -> None:
    print(f"{'Scenario':<20} {'Obj (T CNY)':>12} {'Slack (B)':>10} {'Status':>10}")
    print("-" * 56)

    for name in SCENARIOS:
        json_path = RESULTS_DIR / f"{name}.json"
        if not json_path.exists():
            print(f"{name:<20} {'MISSING':>12}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        obj = data.get("global_objective_cny", 0) / 1e12
        elapsed = data.get("elapsed_seconds", 0)

        # Sum slack across all years
        total_slack = 0.0
        for yr_str, yr_data in data.get("years", {}).items():
            cb = yr_data.get("cost_breakdown", {})
            total_slack += cb.get("slack_penalty", 0)
        slack_b = total_slack / 1e9

        # Check status
        statuses = set()
        for yr_data in data.get("years", {}).values():
            statuses.add(yr_data.get("status", "?"))
        status = ",".join(sorted(statuses))

        print(f"{name:<20} {obj:>12.3f} {slack_b:>10.1f} {status:>10}")

    # Detailed pathway shares
    print("\n" + "=" * 80)
    print("Pathway shares by scenario and year:")
    print("=" * 80)

    for name in SCENARIOS:
        json_path = RESULTS_DIR / f"{name}.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n{name}:")
        for yr_str in ["2030", "2040", "2050", "2060"]:
            yr_data = data.get("years", {}).get(yr_str)
            if yr_data is None:
                continue
            shares = yr_data.get("pathway_shares", {})
            parts = " ".join(f"{pw}={v:.1%}" for pw, v in sorted(shares.items()) if v > 0.005)
            slack = yr_data.get("cost_breakdown", {}).get("slack_penalty", 0) / 1e9
            slack_str = f" [slack={slack:.1f}B]" if abs(slack) > 0.1 else ""
            print(f"  {yr_str}: {parts}{slack_str}")


if __name__ == "__main__":
    main()
