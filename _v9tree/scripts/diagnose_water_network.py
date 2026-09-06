"""Diagnostic: does the water constraint reshape the CO2 pipeline network?

Reads existing scenario outputs only (no re-solve). Answers three questions that
together decide whether "pipeline infrastructure" can stay in the paper's headline:

  Q1 SCALE     how much CO2 enters the network with vs without the water constraint
  Q2 SPACE     how the captured tonnage redistributes across provinces
  Q3 TOPOLOGY  how the active edge set, mileage, haul distance and sinks change

Usage:  python scripts/diagnose_water_network.py [reference] [comparison ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
INPUTS_DIR = Path(__file__).resolve().parents[1] / "inputs"
YEARS = ["2030", "2040", "2050", "2060"]


def _offshore_hub_ids() -> set[str]:
    hubs = pd.read_csv(INPUTS_DIR / "storage_hubs.csv")
    if "offshore" not in hubs.columns:
        return set()
    flag = hubs["offshore"].astype(str).str.lower().isin({"1", "true", "yes"})
    return set(hubs.loc[flag, "storage_hub_id"].astype(str))


OFFSHORE_HUBS = _offshore_hub_ids()


def _read(scenario: str, name: str) -> pd.DataFrame:
    path = RESULTS_DIR / scenario / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def network_metrics(scenario: str) -> pd.DataFrame:
    """Per-year network summary for one scenario."""
    edges = _read(scenario, "network_edges")
    flows = _read(scenario, "co2_flow_direction")
    storage = _read(scenario, "storage_utilization")
    plants = _read(scenario, "plant_detail")

    rows: List[Dict[str, float]] = []
    for year in YEARS:
        yr = int(year)
        e = edges[(edges["year"] == yr) & (edges["edge_active"].astype(float) > 0)]
        f = flows[flows["year"] == yr]
        s = storage[(storage["year"] == yr) & (storage["storage_use_mtpa"].astype(float) > 1e-6)]
        p = plants[plants["year"] == yr]

        flow = f["net_flow_mtpa"].abs()
        length = f["length_km"]
        haul = float((flow * length).sum() / flow.sum()) if flow.sum() > 0 else 0.0

        offshore_use = float(s.loc[s["storage_hub_id"].astype(str).isin(OFFSHORE_HUBS), "storage_use_mtpa"].sum())
        total_use = float(s["storage_use_mtpa"].sum())

        rows.append(
            {
                "year": yr,
                "captured_mt": float(p["captured_mt"].sum()),
                "piped_mt": float(flow.sum()),
                "active_edges": int(len(e)),
                "mileage_km": float(e["length_km"].sum()),
                "new_capacity_mtpa": float(e["new_capacity_mtpa"].sum()),
                "haul_km": haul,
                "active_sinks": int(len(s)),
                "injected_mt": total_use,
                "offshore_share": offshore_use / total_use if total_use > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("year")


def active_edge_set(scenario: str, year: int) -> set[str]:
    edges = _read(scenario, "network_edges")
    e = edges[(edges["year"] == year) & (edges["edge_active"].astype(float) > 0)]
    return set(e["edge_id"].astype(str))


def province_capture(scenario: str, year: int) -> pd.Series:
    p = _read(scenario, "province_pathways")
    p = p[(p["year"] == year) & (p["captured_mt"] > 0)]
    return p.groupby("province_name")["captured_mt"].sum()


def js_divergence(a: pd.Series, b: pd.Series) -> float:
    """Jensen-Shannon divergence (base 2) between two capture distributions."""
    idx = a.index.union(b.index)
    p = a.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    q = b.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compare(reference: str, scenario: str, year: int = 2060) -> None:
    ref_m, sc_m = network_metrics(reference), network_metrics(scenario)

    print(f"\n{'=' * 78}")
    print(f"  {scenario}  vs  {reference}")
    print(f"{'=' * 78}")

    print("\n[Q1+Q3] network metrics by year   (ref -> scenario, delta%)")
    cols = ["captured_mt", "piped_mt", "active_edges", "mileage_km", "haul_km", "active_sinks", "offshore_share"]
    header = f"  {'year':<6}" + "".join(f"{c:>22}" for c in cols)
    print(header)
    for yr in [int(y) for y in YEARS]:
        line = f"  {yr:<6}"
        for c in cols:
            a, b = ref_m.loc[yr, c], sc_m.loc[yr, c]
            pct = (b - a) / a * 100 if a not in (0, 0.0) else float("nan")
            line += f"{a:>8.1f}->{b:>7.1f}{pct:>+6.0f}%"
        print(line)

    ref_e, sc_e = active_edge_set(reference, year), active_edge_set(scenario, year)
    shared, only_ref, only_sc = ref_e & sc_e, ref_e - sc_e, sc_e - ref_e
    union = len(ref_e | sc_e)
    print(f"\n[Q3] active edge set, {year}")
    print(f"  reference {len(ref_e)}  scenario {len(sc_e)}  shared {len(shared)}")
    print(f"  only in reference {len(only_ref)}   only in scenario {len(only_sc)}")
    print(f"  Jaccard similarity {len(shared) / union:.3f}" if union else "  (empty)")

    ref_p, sc_p = province_capture(reference, year), province_capture(scenario, year)
    print(f"\n[Q2] provincial capture, {year}   total {ref_p.sum():.0f} -> {sc_p.sum():.0f} Mt")
    print(f"  Jensen-Shannon divergence {js_divergence(ref_p, sc_p):.4f} bits  (0 = identical shape)")
    idx = ref_p.index.union(sc_p.index)
    delta = (sc_p.reindex(idx).fillna(0.0) - ref_p.reindex(idx).fillna(0.0)).sort_values()
    movers = pd.concat([delta.head(5), delta.tail(5)]).drop_duplicates()
    for prov, d in movers.items():
        a = ref_p.get(prov, 0.0)
        b = sc_p.get(prov, 0.0)
        print(f"    {prov:<16} {a:>8.1f} -> {b:>8.1f} Mt   {d:>+8.1f}")


def main() -> None:
    args = sys.argv[1:]
    reference = args[0] if args else "BASE"
    comparisons = args[1:] if len(args) > 1 else [
        "WA_cwatm_126_dry",
        "WA_cwatm_370_dry",
        "WD_cwatm_126_dry",
        "WD_cwatm_370_dry",
    ]
    print(f"offshore storage hubs: {len(OFFSHORE_HUBS)}")
    for scenario in comparisons:
        compare(reference, scenario)


if __name__ == "__main__":
    main()
