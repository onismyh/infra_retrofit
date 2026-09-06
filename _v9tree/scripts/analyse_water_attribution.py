"""Numeric backing for Fig 2 and Fig 3: attribute the water result to its sources.

Three tables, all read from results on disk. This is the table form of what Fig 2 and Fig 3
show graphically; the figures are the argument, this is the audit trail.

  A  ADAPTATION   what the fleet does under scarcity when dry cooling is allowed vs frozen
  B  ATTRIBUTION  one factor at a time from the main case, against the solver's own tolerance
  D  DIAGNOSTIC   is "no water-carbon trade-off" physics, or an artefact of the hard 2060 target?

Table C (robustness of the retirement cap and retirement cost, neither of which has a source)
needs the SA_* runs, which are deferred. It is not silently omitted -- see the note printed
at the end.

Earlier versions of table B reported "hydrology model" as the second-largest source. That was
an artefact: WaterGAP2-2e ran 2.43x wetter than the official baseline in the Hai basin, so its
constraint never bound. With basin bias correction in the input build, whatever remains here
is genuine model disagreement about the RATE of change.

Usage:  python scripts/analyse_water_attribution.py
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
YEARS = ["2030", "2040", "2050", "2060"]

ANCHOR = "WA_cwatm_126_dry"

# Each factor differs from the anchor in exactly one thing.
FACTORS: list[tuple[str, str]] = [
    ("constraint  (none vs dry season)", "BASE"),
    ("adaptation  (cooling frozen)", "WA_cwatm_126_dry_noair"),
    ("accounting  (annual mean)", "WA_cwatm_126_annual"),
    ("hydrology   (WaterGAP2)", "WA_wgap_126_dry"),
    ("climate     (SSP3-7.0)", "WA_cwatm_370_dry"),
]

DEFERRED = [
    "SA_retire_cap_010", "SA_retire_cap_025", "SA_retire_cost_250", "SA_retire_cost_650",
    "SA_air_capex_low", "SA_air_capex_high", "SA_air_penalty_high",
]


def read(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summary(name: str) -> dict[str, float] | None:
    data = read(name)
    if data is None:
        return None
    years = data["years"]
    row: dict[str, float] = {"objective_Ttn": data["global_objective_cny"] / 1e12}
    for year in YEARS:
        shares = years[year]["pathway_shares"]
        row[f"retire_{year}"] = float(shares.get("retire", 0.0))
        row[f"capture_{year}"] = float(shares.get("ccs", 0.0)) + float(shares.get("beccs", 0.0))
    row["shortfall_2060_mt"] = float(years["2060"].get("target_shortfall_mt", 0.0))
    row["mip_gap_2060"] = float(years["2060"]["solver_quality"].get("mip_gap") or 0.0)
    breakdown = years["2060"]["cost_breakdown"]
    row["air_capex_2060_Ttn"] = float(breakdown.get("air_retrofit_capex", 0.0)) / 1e12
    row["water_cost_2060_Ttn"] = float(breakdown.get("water_cost", 0.0)) / 1e12
    detail_path = RESULTS / name / "plant_detail.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        last = detail[detail["year"] == detail["year"].max()]
        weight = last["capacity_mw"]
        if "air_cooled_share" in last.columns and weight.sum() > 0:
            # air_cooled_share is conversion progress over the hub's remaining wet units, not
            # a share of capacity, so converting it to GW needs the still-wet fraction.
            converted_mw = weight * (1.0 - last["already_air_share"]) * last["air_cooled_share"]
            operating = weight * (1.0 - last["share_retire"])
            dry_mw = converted_mw + operating * last["already_air_share"]
            row["air_converted_gw"] = float(converted_mw.sum()) / 1000.0
            row["dry_share_of_operating"] = float(dry_mw.sum() / operating.sum()) if operating.sum() > 0 else 0.0
        row["captured_2060_mt"] = float(last["captured_mt"].sum())
        row["water_use_2060_1e8m3"] = float(last["water_use_m3"].sum()) / 1e8
    return row


def show(title: str, names: list[str], columns: list[str]) -> pd.DataFrame | None:
    rows = {name: summary(name) for name in names}
    missing = [name for name, value in rows.items() if value is None]
    present = {name: value for name, value in rows.items() if value is not None}
    print(f"\n{'=' * 104}\n  {title}\n{'=' * 104}")
    if missing:
        print(f"  not yet run: {', '.join(missing)}")
    if not present:
        return None
    frame = pd.DataFrame(present).T
    available = [column for column in columns if column in frame.columns]
    print(frame[available].round(4).to_string())
    return frame


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    core = ["objective_Ttn", "retire_2030", "retire_2060", "capture_2060",
            "captured_2060_mt", "air_converted_gw", "dry_share_of_operating",
            "air_capex_2060_Ttn", "water_use_2060_1e8m3"]

    show(
        "A  ADAPTATION -- the 2x2: (water constraint off/on) x (cooling adaptive/frozen)",
        ["BASE", ANCHOR, "BASE_noair", "WA_cwatm_126_dry_noair"],
        core + ["shortfall_2060_mt"],
    )

    frame = show(
        "B  ATTRIBUTION -- one factor at a time from the main case",
        [ANCHOR] + [name for _, name in FACTORS],
        core,
    )
    if frame is not None and ANCHOR in frame.index:
        anchor_cost = float(frame.loc[ANCHOR, "objective_Ttn"])
        tolerance = float(frame["mip_gap_2060"].max()) * 100 if "mip_gap_2060" in frame else 0.0
        print(f"\n  effect on system cost, relative to {ANCHOR} ({anchor_cost:.3f} Ttn)")
        effects = []
        for label, name in FACTORS:
            if name not in frame.index:
                continue
            delta = (float(frame.loc[name, "objective_Ttn"]) - anchor_cost) / anchor_cost * 100
            effects.append((label, delta))
        for label, delta in sorted(effects, key=lambda item: -abs(item[1])):
            mark = "  <- inside solver tolerance, not resolvable" if abs(delta) <= tolerance else ""
            print(f"    {label:<36} {delta:+7.2f} %{mark}")
        print(f"\n    solver tolerance is +/-{tolerance:.2f}% of the objective"
              f" = {tolerance / 100 * anchor_cost:.3f} Ttn")

    show(
        "D  DIAGNOSTIC -- is 'no water-carbon trade-off' physics or the hard 2060 target?",
        [ANCHOR, "DIAG_no_target"],
        core + ["shortfall_2060_mt"],
    )

    print(f"\n{'=' * 104}\n  C  ROBUSTNESS -- DEFERRED, not omitted\n{'=' * 104}")
    print("  The retirement cap and the retirement cost have no literature source and are the "
          "two parameters\n  the 'scarcity -> retirement' chain rests on. Their sensitivity runs "
          f"({len(DEFERRED)} scenarios) are\n  defined in run_single.py but deliberately not run "
          "yet: the framework comes first.\n  Until they are run, no claim about robustness to "
          "those two parameters can be made.")


if __name__ == "__main__":
    main()
