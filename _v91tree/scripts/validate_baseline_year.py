"""Baseline-year validation: does the model reproduce the fleet it claims to represent?

Two blocks, deliberately separated, because they carry different evidential weight:

  IDENTITY     the reference IS the model's own input or assumption. A ratio of 1.00 here
               says the pipeline arithmetic is intact, nothing more, so no ratio is printed.
               Reporting these as agreement would be circular.
  INDEPENDENT  the reference comes from a document outside this model. Only these rows
               constitute validation, and only these get a ratio.

Where no independent reference could be pinned to a primary document the row says so and
prints no number. An unvalidated quantity is reported as unvalidated, not given a plausible
figure to compare against.

Reads results only; nothing is re-solved.

Usage:  python scripts/validate_baseline_year.py [scenario]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# What each identity row actually proves, stated so it cannot be mistaken for validation.
IDENTITY: list[tuple[str, str, str]] = [
    ("Installed coal capacity 2025", "GW",
     "GEM Global Coal Plant Tracker Jul 2025 is the model input; this checks unit-to-hub aggregation"),
    ("Coal generation 2030", "TWh",
     "generation = capacity x assumed 4 643 utilisation hours; checks the assumption was applied, "
     "not that the hours are right"),
    ("Coal CO2 emissions 2030", "Mt/yr",
     "emissions = generation x 0.82 t/MWh fleet factor; the factor is an input"),
    ("Freshwater withdrawal 2030", "10^8 m3/yr",
     "withdrawal = generation x fleet-weighted 15.55 m3/MWh from the intensity table; the "
     "intensity is an input"),
]

# (label, reference, unit, source) -- reference is a point value or a (low, high) range.
INDEPENDENT: list[tuple[str, float | tuple[float, float], str, str]] = [
    ("Coal CO2 emissions 2030", (5000.0, 5300.0), "Mt/yr",
     "IEA, China coal-fired electricity and heat CO2, 2023 (model year is 2030, so the model "
     "should sit at or above this)"),
    ("Freshwater withdrawal 2030", 490.0, "10^8 m3/yr",
     "China Water Resources Bulletin 2023, thermal-power once-through withdrawal"),
    ("CO2 stored 2060", (1000.0, 1800.0), "Mt/yr",
     "ACCA21 / CAEP CCUS technology roadmap, 2060 storage requirement"),
]

# Quantities the model reports for which no comparable published national total was found.
UNVALIDATED: list[tuple[str, str, str]] = [
    ("Freshwater consumption 2030", "10^8 m3/yr",
     "Chinese statistics report thermal-power WITHDRAWAL, not consumption; no national "
     "consumptive total on this basis was located, so this quantity is unvalidated"),
]


def load(scenario: str) -> dict:
    path = RESULTS / f"{scenario}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found - run the scenario first")
    return json.loads(path.read_text(encoding="utf-8"))


def modelled(scenario: str) -> dict[str, float]:
    detail = pd.read_csv(RESULTS / scenario / "plant_detail.csv")
    storage = pd.read_csv(RESULTS / scenario / "storage_utilization.csv")
    plants = pd.read_csv(ROOT / "inputs" / "plants.csv")

    first = detail[detail["year"] == detail["year"].min()]
    out = {
        "Installed coal capacity 2025": float(plants["total_capacity_mw"].sum()) / 1000.0,
        "Coal generation 2030": float(first["annual_generation_mwh"].sum()) / 1e6,
        "Coal CO2 emissions 2030": float(first["baseline_emissions_mt"].sum()),
        "Freshwater consumption 2030": float(first["water_use_m3"].sum()) / 1e8,
        "CO2 stored 2060": float(
            storage.loc[storage["year"] == storage["year"].max(), "storage_use_mtpa"].sum()
        ),
    }
    if "withdrawal_intensity_m3_per_mwh" in plants.columns:
        merged = first.merge(
            plants.set_index("plant_id")[["withdrawal_intensity_m3_per_mwh"]],
            left_on="plant_id", right_index=True, how="left",
        )
        out["Freshwater withdrawal 2030"] = float(
            (merged["annual_generation_mwh"] * merged["withdrawal_intensity_m3_per_mwh"]).sum()
        ) / 1e8
    return out


def format_reference(reference: float | tuple[float, float]) -> str:
    if isinstance(reference, tuple):
        return f"{reference[0]:.0f}-{reference[1]:.0f}"
    return f"{reference:.1f}"


def verdict(model_value: float, reference: float | tuple[float, float]) -> tuple[str, str]:
    if isinstance(reference, tuple):
        low, high = reference
        if low <= model_value <= high:
            return "in range", ""
        ratio = model_value / (low if model_value < low else high)
        return f"{ratio:.2f}x", "below" if model_value < low else "above"
    return f"{model_value / reference:.2f}x", ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    scenario = sys.argv[1] if len(sys.argv) > 1 else "BASE"
    load(scenario)  # fails fast if the scenario has not been run
    values = modelled(scenario)

    width = 34
    print(f"Baseline validation against {scenario}")

    print(f"\n{'=' * 92}\n  INDEPENDENT -- the only rows that validate anything\n{'=' * 92}")
    print(f"{'quantity':<{width}}{'model':>11}{'reference':>14}{'unit':>13}{'verdict':>16}")
    for label, reference, unit, source in INDEPENDENT:
        model_value = values.get(label)
        if model_value is None:
            print(f"{label:<{width}}{'n/a':>11}{format_reference(reference):>14}{unit:>13}{'':>16}")
        else:
            mark, side = verdict(model_value, reference)
            tail = f"{mark} ({side})" if side else mark
            print(f"{label:<{width}}{model_value:>11.1f}{format_reference(reference):>14}"
                  f"{unit:>13}{tail:>16}")
        print(f"{'':<{width}}  source: {source}")

    print(f"\n{'=' * 92}\n  IDENTITY -- checks the pipeline, not the model. No ratio is meaningful here.\n{'=' * 92}")
    print(f"{'quantity':<{width}}{'model':>11}{'unit':>13}  what the number rests on")
    for label, unit, explanation in IDENTITY:
        model_value = values.get(label)
        shown = f"{model_value:.1f}" if model_value is not None else "n/a"
        print(f"{label:<{width}}{shown:>11}{unit:>13}  {explanation}")

    print(f"\n{'=' * 92}\n  UNVALIDATED\n{'=' * 92}")
    for label, unit, explanation in UNVALIDATED:
        model_value = values.get(label)
        shown = f"{model_value:.1f}" if model_value is not None else "n/a"
        print(f"{label:<{width}}{shown:>11}{unit:>13}  {explanation}")

    print(
        "\nThe withdrawal row is the substantive disagreement: the model's freshwater "
        "once-through fleet draws about twice the bulletin's thermal-power figure. Intensity "
        "factors, GEM cooling labels and the seawater split have each been excluded as the "
        "cause; the residual points at the utilisation hours assumed for inland once-through "
        "units (the bulletin implies ~2 900 h against the model's 4 459). Withdrawal is a "
        "reported quantity here, not a constrained one, so this does not propagate into the "
        "optimisation -- but it is not resolved either."
    )


if __name__ == "__main__":
    main()
