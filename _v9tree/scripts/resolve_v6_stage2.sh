#!/usr/bin/env bash
# v6 stage 2: the bias-correction controls, and the remaining figure-consumed runs.
#
# `plot_style.assert_same_vintage`, added after three main figures were found to be differencing
# corrected against superseded runs, reports these as the last stale inputs:
#   Fig 4: WA_cwatm_370_dry_nobias, WA_wgap_126_dry_nobias
#   Fig 5: the same two, used as the bias-correction band that stands in for a null whenever the
#          seed family is empty
# Both are the CONTROL side of the admission gate that decides what Fig 4 is allowed to plot, so
# leaving them on the old dry-season right-hand side means the gate is calibrated on one basis
# and applied on another.
set -u
QUEUE="
WA_cwatm_370_dry_nobias
WA_wgap_126_dry_nobias
WA_cwatm_126_dry_nobias
WA_wgap_370_dry_nobias
"
LOG=results/resolve_v6.log
DRIVER=results/resolve_v6_driver.log
while ! grep -q "V6 QUEUE COMPLETE" "$DRIVER" 2>/dev/null; do
  if grep -q "V6 QUEUE ABORTED" "$DRIVER" 2>/dev/null; then
    echo "=== V6 STAGE2 NOT STARTED: stage 1 aborted $(date -Iseconds) ===" >> "$DRIVER"; exit 1
  fi
  sleep 120
done
for s in $QUEUE; do
  unset COAL_RETROFIT_GUROBI_SEED || true
  if ! python scripts/run_single.py --list 2>/dev/null | grep -qx "$s"; then
    echo "=== SKIP $s (not registered) $(date -Iseconds) ===" >> "$DRIVER"; continue
  fi
  echo "=== START $s $(date -Iseconds) threads=8 ===" >> "$DRIVER"
  python scripts/run_single.py "$s" --threads 8 >> "$LOG" 2>&1
  echo "=== END   $s rc=$? $(date -Iseconds) ===" >> "$DRIVER"
done
echo "=== V6 STAGE2 COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
