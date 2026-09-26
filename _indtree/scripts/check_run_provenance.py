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

    python scripts/check_run_provenance.py
    python scripts/check_run_provenance.py --strict   # also fail on unpinned threads
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def load(name):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload.get("solver_quality", {})
    out = {key: quality.get(key) for key in FIELDS}
    out["objective"] = payload.get("global_objective_cny")
    out["mip_gap"] = quality.get("mip_gap")
    return out


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


def check_contrasts(failures, warnings, strict):
    print()
    print("=" * 96)
    print("published contrasts -- fingerprints may differ, the thread pin may not")
    print("=" * 96)
    for label, base, variant in CONTRASTS:
        a, b = load(base), load(variant)
        print()
        print(f"  {label}")
        print(describe(base, a))
        print(describe(variant, b))
        if a is None or b is None:
            warnings.append(f"{label}: one side not solved yet")
            continue
        if a["fingerprint"] is None or b["fingerprint"] is None:
            warnings.append(f"{label}: at least one side has no provenance, so comparability "
                            f"cannot be checked")
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
            point = 100.0 * (b["objective"] - a["objective"]) / a["objective"]
            ga = float(a["mip_gap"] or 0.0)
            gb = float(b["mip_gap"] or 0.0)
            lo = 100.0 * ((b["objective"] * (1 - gb)) - a["objective"]) / a["objective"]
            hi = 100.0 * (b["objective"] - a["objective"] * (1 - ga)) / a["objective"]
            verdict = "SIGN RESOLVED" if lo > 0 or hi < 0 else "inside solver bounds"
            print(f"    -> effect {point:+.3f}%  certified [{lo:+.3f}, {hi:+.3f}]%  "
                  f"-> {verdict}")


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
    args = parser.parse_args()
    failures, warnings = [], []
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
