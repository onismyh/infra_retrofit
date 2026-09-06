"""Compile sensitivity analysis results into a summary table.

Usage:
    python scripts/compile_sensitivity.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = {
    # name: (description, parameter_changed, value)
    "BASE_tight": ("Baseline (MIPGap=1%)", "—", "default"),
    "SA_discount_3pct": ("Discount rate 3%", "discount_rate", "0.03"),
    "SA_discount_8pct": ("Discount rate 8%", "discount_rate", "0.08"),
    "SA_pipe_mid": ("Pipeline corridor ×0.6", "corridor_capex_multiplier", "0.6"),
    "SA_pipe_full": ("Pipeline corridor ×1.0", "corridor_capex_multiplier", "1.0"),
    "SA_ccs_capex_low": ("CCS CAPEX 900 CNY/kW", "ccs_retrofit_capex", "900"),
    "SA_ccs_capex_high": ("CCS CAPEX 1800 CNY/kW", "ccs_retrofit_capex", "1800"),
    "SA_biomass_cost_150": ("Biomass cost ×1.5", "biomass_cost_multiplier", "1.5"),
    "SA_biomass_cost_200": ("Biomass cost ×2.0", "biomass_cost_multiplier", "2.0"),
    "SA_ammonia_cost_70": ("Ammonia cost ×0.7", "ammonia_cost_multiplier", "0.7"),
    "SA_ammonia_cost_50": ("Ammonia cost ×0.5", "ammonia_cost_multiplier", "0.5"),
    "SA_retire_cost_500": ("Retire cost 500 CNY/MWh", "retire_cost_cny_per_mwh", "500"),
    "SA_retire_cost_1000": ("Retire cost 1000 CNY/MWh", "retire_cost_cny_per_mwh", "1000"),
    "SA_injectivity_half": ("Storage injectivity ×0.5", "injectivity_multiplier", "0.5"),
}


def load_result(name: str) -> dict | None:
    path = ROOT / "results" / f"{name}.json"
    if not path.exists():
        return None
    return json.load(open(path))


def extract_summary(data: dict) -> dict:
    obj = data["global_objective_cny"]
    shares_2030 = data["years"]["2030"]["pathway_shares"]
    shares_2050 = data["years"]["2050"]["pathway_shares"]
    total_slack = sum(
        y["cost_breakdown"].get("slack_penalty", 0)
        for y in data["years"].values()
    )
    return {
        "obj_b": obj / 1e9,
        "biomass_2030": shares_2030.get("biomass", 0),
        "retire_2030": shares_2030.get("retire", 0),
        "ccs_2050": shares_2050.get("ccs", 0),
        "beccs_2050": shares_2050.get("beccs", 0),
        "retire_2050": shares_2050.get("retire", 0),
        "ammonia_any": max(
            max(y["pathway_shares"].get("ammonia", 0) for y in data["years"].values()),
            0,
        ),
        "slack_b": total_slack / 1e9,
        "status": "optimal" if not any(
            y["status"] != "optimal" for y in data["years"].values()
        ) else "WARNING",
    }


def main():
    base_data = load_result("BASE_tight")
    if base_data is None:
        # Fall back to original BASE
        base_data = load_result("BASE")
    base_obj = base_data["global_objective_cny"] if base_data else 0

    lines = []
    lines.append("# Sensitivity Analysis Summary")
    lines.append("")
    lines.append("| Scenario | Description | Total Cost (B CNY) | Δ vs BASE | Bio 2030 | CCS 2050 | Retire 2050 | NH3 max | Slack | Status |")
    lines.append("|----------|------------|-------------------|-----------|----------|----------|------------|---------|-------|--------|")

    for name, (desc, param, val) in SCENARIOS.items():
        data = load_result(name)
        if data is None:
            lines.append(f"| {name} | {desc} | — | — | — | — | — | — | — | NOT RUN |")
            continue
        s = extract_summary(data)
        delta = ((s["obj_b"] * 1e9 - base_obj) / base_obj * 100) if base_obj > 0 else 0
        delta_str = f"{delta:+.1f}%" if name != "BASE_tight" else "—"
        lines.append(
            f"| {name} | {desc} | {s['obj_b']:.0f} | {delta_str} | "
            f"{s['biomass_2030']:.0%} | {s['ccs_2050']:.0%} | {s['retire_2050']:.0%} | "
            f"{s['ammonia_any']:.2%} | {s['slack_b']:.0f}B | {s['status']} |"
        )

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")

    # Ammonia robustness check
    for name in ["SA_ammonia_cost_50", "SA_ammonia_cost_70"]:
        data = load_result(name)
        if data:
            s = extract_summary(data)
            lines.append(f"- **{name}**: Ammonia max share = {s['ammonia_any']:.2%} → {'still dominated' if s['ammonia_any'] < 0.01 else 'enters solution'}")

    # Biomass robustness check
    for name in ["SA_biomass_cost_150", "SA_biomass_cost_200"]:
        data = load_result(name)
        if data:
            s = extract_summary(data)
            lines.append(f"- **{name}**: Biomass 2030 = {s['biomass_2030']:.0%}, CCS 2050 = {s['ccs_2050']:.0%}")

    # Retirement cost check
    for name in ["SA_retire_cost_500", "SA_retire_cost_1000"]:
        data = load_result(name)
        if data:
            s = extract_summary(data)
            lines.append(f"- **{name}**: Retire 2050 = {s['retire_2050']:.0%} (vs BASE ~74%)")

    text = "\n".join(lines)
    out_path = ROOT / "results" / "sensitivity_summary.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
