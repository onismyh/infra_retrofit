#!/usr/bin/env bash
# T1: the three withdrawal-constraint scenarios, run sequentially so Gurobi gets the whole box.
# Sequential rather than parallel because each solve is memory-heavy and the joint 2030-2060
# formulation already uses every core.
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== T1 wd085 runs start $(date) ==="
for s in WA_cwatm_126_dry_wd085 WA_cwatm_370_dry_wd085 WA_cwatm_126_dry_wd085_noair; do
  echo "--- $s ---"
  python -u scripts/run_single.py "$s" || echo "FAILED: $s"
done
echo "=== T1 wd085 runs done $(date) ==="
