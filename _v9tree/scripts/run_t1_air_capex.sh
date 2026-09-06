#!/usr/bin/env bash
# Air-retrofit capex sensitivity, queued behind run_t1_wd085.sh.
# Waits for that batch's completion marker rather than running concurrently: the joint
# 2030-2060 solve already saturates the box, and two Gurobi processes would just thrash.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=results/t1_wd085_run.log
echo "=== air-capex sweep waiting for the wd085 batch $(date) ==="
while ! grep -q "runs done" "$LOG" 2>/dev/null; do sleep 60; done
echo "=== air-capex sweep start $(date) ==="
for s in WA_cwatm_126_dry_wd085_air1000 WA_cwatm_126_dry_wd085_air1370; do
  echo "--- $s ---"
  python -u scripts/run_single.py "$s" || echo "FAILED: $s"
done
echo "=== air-capex sweep done $(date) ==="
