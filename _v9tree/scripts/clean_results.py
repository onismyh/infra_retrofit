"""Clean numerical noise from experiment results.

Addresses solver artifacts:
  - Solver slack penalties masking true cost
  - Sub-0.1% pathway shares that are numerical noise
  - Degenerate / infeasible scenarios flagged

Reads:  results/{scenario}.json (individual scenario files)
Writes: results/experiment_results_clean.json (consolidated + cleaned)

Usage:
    python scripts/clean_results.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CLEAN_PATH = RESULTS_DIR / "experiment_results_clean.json"

# All scenario names to consolidate
SCENARIO_NAMES = [
    "BASE", "BASE_zero", "BASE_neg",
    "RQ3_no_ammonia", "RQ3_no_biomass", "RQ3_no_ccs",
    "RQ3_retire_only", "RQ3_ccs_only",
    "WA_grid_200km",
    # Sensitivity Analysis
    "SA_discount_3pct", "SA_discount_8pct",
    "SA_ccs_capex_low", "SA_ccs_capex_high",
    "SA_biomass_cost_150", "SA_biomass_cost_200",
    "SA_ammonia_cost_50", "SA_ammonia_cost_70",
    "SA_retire_cost_500", "SA_retire_cost_1000",
    "SA_pipe_mid", "SA_pipe_full",
    "SA_injectivity_half",
    "SA_carbon_low", "SA_carbon_high",
]
CORE_SCENARIO_NAMES = [
    "BASE", "BASE_zero", "BASE_neg",
    "RQ3_no_ammonia", "RQ3_no_biomass", "RQ3_no_ccs",
]

SHARE_THRESHOLD = 0.001  # 0.1 %

# Scenarios excluded entirely: degenerate / infeasible solver outputs
REMOVE_SCENARIOS: list[str] = []

SLACK_INFEASIBILITY_THRESHOLD = 0.05  # 5% of period objective

CLEANING_DATE = "2026-04-02"

METADATA: dict[str, Any] = {
    "cleaned": True,
    "cleaning_date": CLEANING_DATE,
    "removed_scenarios": [s for s in REMOVE_SCENARIOS if s != "experiment_results"],
    "cleaning_rules": [
        "Shares < 0.1% rounded to zero and remaining shares renormalised to 1.0",
        "Minor slack_penalty (<5% of period cost) zeroed as solver artifact",
        "Large slack_penalty (>=5% of period cost) RETAINED as infeasibility indicator",
        "Scenarios with retained slack flagged as 'infeasible' in metadata",
        "objective_cny per year recalculated as sum of cost_breakdown values",
        "global_objective_cny recalculated as sum of cleaned per-year objectives",
    ],
}


# ── Share cleaning ────────────────────────────────────────────────────────────

def _clean_shares(shares: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Zero shares below threshold and renormalise.

    Returns:
        cleaned: dict with zeroed/renormalised shares
        changed: list of pathway names that were zeroed
    """
    changed: list[str] = []
    cleaned: dict[str, float] = {}
    for pathway, share in shares.items():
        if share < SHARE_THRESHOLD:
            if share != 0.0:
                changed.append(pathway)
            cleaned[pathway] = 0.0
        else:
            cleaned[pathway] = share

    total = sum(cleaned.values())
    if total > 0:
        cleaned = {k: v / total for k, v in cleaned.items()}
    return cleaned, changed


# ── Cost cleaning ─────────────────────────────────────────────────────────────

def _clean_costs(
    year_data: dict[str, Any],
) -> tuple[dict[str, Any], float, float, bool]:
    """Zero minor slack_penalty and recompute objective_cny from cost breakdown.

    Large slack (>=5% of objective) is retained as it indicates infeasibility.

    Returns:
        cleaned_year_data: modified copy
        old_obj: original objective_cny
        new_obj: recalculated objective_cny
        infeasible: True if large slack was retained
    """
    old_obj: float = year_data["objective_cny"]
    breakdown: dict[str, float] = dict(year_data["cost_breakdown"])
    slack = breakdown.get("slack_penalty", 0.0)
    infeasible = False

    if abs(old_obj) > 0 and abs(slack) / abs(old_obj) >= SLACK_INFEASIBILITY_THRESHOLD:
        # Large slack: retain (infeasibility indicator)
        infeasible = True
    else:
        # Minor slack: zero it
        breakdown["slack_penalty"] = 0.0

    new_obj = sum(breakdown.values())

    cleaned = dict(year_data)
    cleaned["cost_breakdown"] = breakdown
    cleaned["objective_cny"] = new_obj
    return cleaned, old_obj, new_obj, infeasible


# ── Per-scenario cleaning ─────────────────────────────────────────────────────

def clean_scenario(name: str, scenario: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Clean one scenario dict in-place (returns new dict).

    Returns:
        cleaned_scenario: cleaned copy
        log_lines: human-readable summary of changes
    """
    log_lines: list[str] = []
    cleaned_scenario = dict(scenario)
    cleaned_years: dict[str, Any] = {}
    new_global_obj = 0.0
    has_infeasible = False

    for year_str, year_data in scenario.get("years", {}).items():
        # --- shares ---
        raw_shares: dict[str, float] = year_data.get("pathway_shares", {})
        cleaned_shares, zeroed = _clean_shares(raw_shares)
        if zeroed:
            for pathway in zeroed:
                log_lines.append(
                    f"  {name} {year_str}: zeroed {pathway} share "
                    f"({raw_shares[pathway]:.6f} -> 0)"
                )

        # --- costs ---
        year_with_shares = dict(year_data)
        year_with_shares["pathway_shares"] = cleaned_shares
        cleaned_year, old_obj, new_obj, infeasible = _clean_costs(year_with_shares)
        if infeasible:
            has_infeasible = True

        slack_was = year_data["cost_breakdown"].get("slack_penalty", 0.0)
        if abs(slack_was) > 1.0:
            if infeasible:
                log_lines.append(
                    f"  {name} {year_str}: RETAINED slack_penalty "
                    f"{slack_was/1e9:.3f}B CNY (infeasibility indicator)"
                )
            else:
                log_lines.append(
                    f"  {name} {year_str}: removed slack_penalty "
                    f"{slack_was/1e9:.3f}B CNY; "
                    f"obj {old_obj/1e9:.3f}B -> {new_obj/1e9:.3f}B CNY"
                )

        cleaned_years[year_str] = cleaned_year
        new_global_obj += new_obj

    cleaned_scenario["years"] = cleaned_years
    old_global = scenario.get("global_objective_cny", float("nan"))
    cleaned_scenario["global_objective_cny"] = new_global_obj
    cleaned_scenario["infeasible"] = has_infeasible
    if has_infeasible:
        log_lines.append(f"  {name}: FLAGGED as infeasible (large slack retained)")
    if abs(new_global_obj - old_global) > 1e6:
        log_lines.append(
            f"  {name}: global_objective_cny "
            f"{old_global/1e12:.4f}T -> {new_global_obj/1e12:.4f}T CNY"
        )
    return cleaned_scenario, log_lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate and clean scenario JSON outputs.")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only consolidate the six rerun main scenarios.",
    )
    args = parser.parse_args()

    # 1. Consolidate individual scenario JSON files
    scenario_names = CORE_SCENARIO_NAMES if args.core_only else SCENARIO_NAMES
    raw: dict[str, Any] = {}
    for name in scenario_names:
        json_path = RESULTS_DIR / f"{name}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Convert integer year keys to strings for consistency
            if "years" in data:
                data["years"] = {str(k): v for k, v in data["years"].items()}
            raw[name] = data
            print(f"  loaded {name}")
        else:
            print(f"  MISSING {name}")

    print(f"\nScenarios loaded: {list(raw.keys())}")

    # 1. Remove degenerate scenarios
    removed: list[str] = []
    for key in REMOVE_SCENARIOS:
        if key in raw:
            del raw[key]
            removed.append(key)
    if removed:
        print(f"\nRemoved scenarios: {removed}")

    # 2. Clean remaining scenarios
    cleaned_data: dict[str, Any] = {}
    all_log_lines: list[str] = []

    for name, scenario in raw.items():
        if not isinstance(scenario, dict) or "years" not in scenario:
            # Pass through non-scenario top-level keys untouched (e.g. _metadata)
            cleaned_data[name] = scenario
            continue
        cleaned_scenario, log_lines = clean_scenario(name, scenario)
        cleaned_data[name] = cleaned_scenario
        all_log_lines.extend(log_lines)

    # 3. Add metadata
    cleaned_data["_metadata"] = METADATA

    # 4. Write output
    print(f"\nCleaning summary:")
    if all_log_lines:
        for line in all_log_lines:
            print(line)
    else:
        print("  No numerical changes detected.")

    print(f"\nWriting {CLEAN_PATH}")
    with open(CLEAN_PATH, "w", encoding="utf-8") as fh:
        json.dump(cleaned_data, fh, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()
