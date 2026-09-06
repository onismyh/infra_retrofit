"""Regenerate every headline number the manuscript quotes, from the live results.

WHY THIS EXISTS. Across v1-v4 the same failure recurred: a number computed correctly by a
figure script, copied into prose, and then left behind when the underlying run changed. The
v4 review found the storyline quoting bias factors that belonged to a different ensemble
member, a conversion multiple of x3.45 against a run that gives x2.47, and a capture loss
quoted as both 150 and 122 Mt in the same document. None of those were computation errors --
they were transcription that outlived its source.

This script is the answer. It recomputes each quoted quantity from `results/` and writes both
a machine-readable ledger and a markdown table with provenance, so prose can be checked
against it mechanically instead of by eye. Run it after any re-solve; diff the JSON to see
exactly which claims moved.

    python scripts/build_numbers_ledger.py
    python scripts/build_numbers_ledger.py --compare results/numbers_ledger_prev.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import input_vintage, same_model_runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "numbers_ledger.json"
OUT_MD = RESULTS / "numbers_ledger.md"

CONTROL = "WA_cwatm_126_dry"
TREAT = "WA_cwatm_126_dry_wd085"
BASE = "BASE"
FROZEN_CAPFREE = "WA_cwatm_126_dry_wd085_noair_capfree"
NEAR, END = 2030, 2060
PEAK = 2040
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847}


def load(name):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def detail(name, year):
    path = RESULTS / name / "plant_detail.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame[frame["year"] == year]


def converted_gw(frame):
    """Wet capacity converted to dry cooling, GW.

    `air_cooled_share` is progress over the plant's REMAINING WET capacity, so the already-dry
    fraction has to be netted out or the 248 GW of fleet that was built dry gets counted as an
    outcome of this model.
    """
    return float((frame["capacity_mw"] * (1.0 - frame["already_air_share"])
                  * frame["air_cooled_share"]).sum()) / 1e3


def entry(value, unit, source, note=""):
    return {"value": value, "unit": unit, "source": source, "note": note}


def cost_entries(ledger):
    a, b = load(CONTROL), load(TREAT)
    if a is None or b is None:
        return
    qa, qb = a["solver_quality"], b["solver_quality"]
    oa = a["global_objective_cny"] / 1e12
    ob = b["global_objective_cny"] / 1e12
    ga, gb = qa.get("mip_gap"), qb.get("mip_gap")
    if ga is None or gb is None:
        # `or 0.0` here used to collapse a missing gap to zero, which does not widen the
        # interval -- it makes it look CERTIFIED AND TIGHT on a run that reported no gap at
        # all. RQ3_retire_only already carries mip_gap: null.
        raise SystemExit(f"{CONTROL}/{TREAT}: missing mip_gap; cannot certify an interval")
    ga, gb = float(ga), float(gb)
    lb_a, lb_b = oa * (1 - ga), ob * (1 - gb)
    point = 100.0 * (ob - oa) / oa
    # The quantity is (OPT_t - OPT_c) / OPT_c, and OPT_c appears in BOTH numerator and
    # denominator. Over the certified box OPT_c in [LB_c, INC_c] the ratio is decreasing in
    # OPT_c, so its maximum is at OPT_c = LB_c -- and the upper endpoint must divide by LB_c,
    # not by INC_c. Dividing by INC_c understated it (+2.170 where the truth is +2.188).
    lo = 100.0 * (lb_b - oa) / oa
    hi = 100.0 * (ob - lb_a) / lb_a
    ledger["reservation_effect_pct"] = entry(
        round(point, 3), "% of the control objective",
        f"{CONTROL}.json vs {TREAT}.json",
        "the headline single-factor contrast: reserving the non-power share of the allowance")
    ledger["reservation_effect_certified_pct"] = entry(
        [round(lo, 3), round(hi, 3)], "% of the control objective",
        "solver bounds of both runs",
        "the tolerance on a DIFFERENCE of two incumbents is the SUM of the two runs' "
        "gap*objective, not the larger of the two; quote this interval, not the point")
    ledger["reservation_sign_resolved"] = entry(
        bool(lo > 0 or hi < 0), "bool", "derived",
        "true when the certified interval excludes zero")
    for tag, name, quality in (("control", CONTROL, qa), ("treatment", TREAT, qb)):
        ledger[f"{tag}_fingerprint"] = entry(
            quality.get("fingerprint"), "gurobi model hash", f"{name}.json",
            "two runs meant to differ only by seed must agree here; a mismatch means they "
            "are replicates of different models")
        ledger[f"{tag}_threads"] = entry(
            quality.get("threads_param"), "count", f"{name}.json",
            "Gurobi is deterministic only at a fixed thread count; 0 means auto, i.e. not "
            "reproducible")


def physical_entries(ledger):
    near_c, near_t = detail(CONTROL, NEAR), detail(TREAT, NEAR)
    end_c, end_t = detail(CONTROL, END), detail(TREAT, END)
    if near_c is None or near_t is None:
        return
    cc, ct = converted_gw(near_c), converted_gw(near_t)
    ledger["conversion_control_gw"] = entry(
        round(cc, 1), "GW", f"{CONTROL}/plant_detail.csv {NEAR}",
        "net of the already-dry fleet")
    ledger["conversion_treatment_gw"] = entry(
        round(ct, 1), "GW", f"{TREAT}/plant_detail.csv {NEAR}", "")
    ledger["conversion_multiple"] = entry(
        round(ct / cc, 2) if cc else None, "x", "derived",
        "the storyline has quoted x3.45 and the 20.6/71.1 GW pair for this; both were stale")
    if end_c is not None and end_t is not None:
        for tag, frame in (("control", end_c), ("treatment", end_t)):
            ledger[f"capture_{tag}_mt"] = entry(
                round(float(frame["captured_mt"].sum()), 1), "Mt CO2/yr",
                f"plant_detail.csv {END}", "")
            ledger[f"retirement_{tag}_gw_{END}"] = entry(
                round(float((frame["capacity_mw"] * frame["share_retire"]).sum()) / 1e3, 1),
                "GW", f"plant_detail.csv {END}",
                "share_retire is a STOCK and monotone; never cumsum it across years")
    ledger["fleet_gw"] = entry(
        round(float(near_c["capacity_mw"].sum()) / 1e3, 1), "GW",
        f"{CONTROL}/plant_detail.csv {NEAR}", "standing fleet in the model")


def seed_entries(ledger):
    """Solver-degeneracy floor, as a 95% envelope rather than a raw range.

    THE FAMILY IS GATED, NOT GLOBBED, AND THIS IS NOT A REFINEMENT. Until v6 this function
    collected replicates by filename alone. `WA_cwatm_126_dry_wd085_seed5` had been solved on
    the superseded per-cell-minimum water basis and was never re-queued -- it carries
    fingerprint 0xbe7b31c2 against the corrected family's 0xbf2f6de, and an objective 4.5%
    above them. Pooling it raised the reported floor from 0.375% to 4.50%, i.e. ABOVE the
    1.34% headline effect, which would have told a referee the main result is indistinguishable
    from solver noise. It is not; the effect clears the true floor 3.6x.

    A degeneracy floor is only meaningful within one model. Replicates must therefore agree on
    fingerprint, dimensions, thread pin and MIPFocus (`same_model_runs`) AND on the input
    vintage (`input_vintage`), because the dry-season correction changed every water value
    without changing any of the former. Runs that fail either gate are dropped and named.
    """
    for label, stem in (("treatment", TREAT), ("control", CONTROL)):
        candidates = [f"{stem}{s}" for s in [""] + [f"_seed{i}" for i in range(2, 11)]]
        candidates = [n for n in candidates if load(n) is not None]
        kept = same_model_runs(candidates, quiet=True)
        vintages = {n: input_vintage(n) for n in kept}
        ref = vintages.get(stem)
        kept = [n for n in kept if vintages[n] == ref]
        dropped = [n for n in candidates if n not in kept]
        if dropped:
            print(f"  [seed:{label}] dropped {len(dropped)} replicate(s) off the reference "
                  f"model/vintage: {', '.join(dropped)}")
        objectives = [load(n)["global_objective_cny"] for n in kept]
        names = list(kept)
        if len(objectives) < 2:
            print(f"  [seed:{label}] only {len(objectives)} replicate(s) survive gating; "
                  f"no degeneracy floor reported")
            continue
        rng = 100.0 * (max(objectives) - min(objectives)) / min(objectives)
        sigma = rng / D2.get(len(objectives), 3.078)
        ledger[f"seed_range_{label}_pct"] = entry(
            round(rng, 4), "% of objective", ", ".join(names),
            "observed range over the gated seed family (same fingerprint, dims, thread pin, MIPFocus and input vintage); ungated globbing pooled a superseded-basis run and reported 4.50%")
        ledger[f"seed_sigma_{label}_pct"] = entry(
            round(sigma, 4), "% of objective", "derived",
            f"range / d2({len(objectives)}); a range is d2 sigma in expectation, not an "
            f"envelope, and a 2-sample range is only 1.128 sigma")
        ledger[f"seed_envelope95_{label}_pct"] = entry(
            round(1.96 * (2 ** 0.5) * sigma, 4), "% of objective", "derived",
            "95% bound on a DIFFERENCE of two such runs; the floor is applied to a "
            "difference but measured within one family, hence sqrt(2)")


def basin_entries(ledger):
    """Node-level binding, which the basin aggregate does not show."""
    path = RESULTS / TREAT / "resource_use.csv"
    if not path.exists():
        return
    usage = pd.read_csv(path)
    usage = usage[(usage["resource_type"] == "water") & (usage["year"] == NEAR)].copy()
    usage["basin"] = usage["region"].astype(str).str.rsplit("_", n=1).str[-1]
    rows = {}
    for basin, group in usage.groupby("basin"):
        used = float(group["used"].sum())
        saturated = group[group["utilization"] >= 0.999]
        rows[basin] = {
            "basin_aggregate_pct": round(100.0 * used / float(group["available"].sum()), 1),
            "median_node_pct": round(100.0 * float(group["utilization"].median()), 1),
            "demand_at_limit_pct": round(100.0 * float(saturated["used"].sum()) / used, 1)
            if used else 0.0,
            "nodes_at_limit": int(len(saturated)),
            "nodes": int(len(group)),
        }
    ledger["basin_node_binding"] = entry(
        rows, "percent / count", f"{TREAT}/resource_use.csv {NEAR}",
        "demand-weighted, not node-counted: counting nodes weights a node serving 4 GW the "
        "same as one serving 40 MW and reverses the sign of the divergence in most basins")


def write_markdown(ledger, path):
    lines = ["# Numbers ledger", "",
             "Regenerated by `scripts/build_numbers_ledger.py` from the live results in",
             "`results/`. Every number the manuscript quotes should appear here; if a number",
             "in the prose is not in this table, it has no current source.", "",
             "| key | value | unit | source | note |",
             "|---|---|---|---|---|"]
    for key, item in ledger.items():
        value = item["value"]
        if isinstance(value, dict):
            value = f"({len(value)} basins, see JSON)"
        elif isinstance(value, list):
            value = f"[{value[0]}, {value[1]}]"
        note = item["note"].replace("|", "/")
        lines.append(f"| `{key}` | {value} | {item['unit']} | `{item['source']}` | {note} |")
    lines.append("")
    if "basin_node_binding" in ledger:
        lines += ["## Node-level binding by basin", "",
                  "| basin | basin aggregate % | median node % | demand at limit % | nodes at limit |",
                  "|---|---|---|---|---|"]
        for basin, row in sorted(ledger["basin_node_binding"]["value"].items()):
            lines.append(f"| {basin} | {row['basin_aggregate_pct']} | "
                         f"{row['median_node_pct']} | {row['demand_at_limit_pct']} | "
                         f"{row['nodes_at_limit']}/{row['nodes']} |")
        lines.append("")
    path.write_text(chr(10).join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Rebuild the manuscript numbers ledger")
    parser.add_argument("--compare", help="a previous ledger JSON to diff against")
    args = parser.parse_args()

    ledger = {}
    cost_entries(ledger)
    physical_entries(ledger)
    seed_entries(ledger)
    basin_entries(ledger)

    if not ledger:
        print("no results found; solve something first")
        return 1

    OUT_JSON.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(ledger, OUT_MD)
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)} "
          f"({len(ledger)} entries)")

    for key, item in ledger.items():
        value = item["value"]
        if isinstance(value, dict):
            continue
        print(f"  {key:42s} {value}  {item['unit']}")

    if args.compare:
        previous = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        moved = []
        for key, item in ledger.items():
            if key in previous and previous[key]["value"] != item["value"]:
                moved.append((key, previous[key]["value"], item["value"]))
        print()
        if moved:
            print(f"CHANGED SINCE {args.compare} ({len(moved)}):")
            for key, old, new in moved:
                print(f"  {key:42s} {old}  ->  {new}")
            print("  Every claim in the prose that quotes one of these is now stale.")
        else:
            print(f"no ledger entry changed since {args.compare}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
