#!/usr/bin/env bash
# The two scenario families that were registered with a rationale and never solved.
#
# LANE A -- biomass supply realism. The paper's mechanism is that water adaptation is PAID FOR in
# bioenergy, and it draws 87% of the national resource by 2050. The registry's own comment says
# the direction is indeterminate a priori, because cutting supply removes a water-free wedge
# (co-firing at 1.00x) and a water-hungry one (BECCS at 1.82x) together, and the replacement
# splits between CCS retrofit (1.82x) and retirement (zero). 0.15 -> 4.5 EJ is the residue-only
# realistic case; 0.42 -> 12.6 EJ is a generous upper bound. Both arms plus both controls.
#
# LANE B -- the CONTROL seed replicates, registered but never run, which is why the ledger reports
# a degeneracy floor for the treatment and none for the control. Solving them gives a control-side
# floor, tests the equal-variance assumption behind the sqrt(2) in the current envelope, and lets
# the headline be read as a paired difference. Plus the one run that completes the air 2x2.
set -u
LANE="${1:-A}"
if [ "$LANE" = "A" ]; then
  QUEUE="WA_cwatm_126_dry_bio015 WA_cwatm_126_dry_wd085_bio015
         WA_cwatm_126_dry_bio042 WA_cwatm_126_dry_wd085_bio042"
else
  QUEUE="WA_cwatm_126_dry_seed2 WA_cwatm_126_dry_seed3 WA_cwatm_126_dry_seed4 WA_cwatm_370_dry_noair"
fi
LOG="results/resolve_gaps_${LANE}.log"
DRIVER="results/resolve_gaps_${LANE}_driver.log"
for s in $QUEUE; do
  case "$s" in
    *_seed2) export COAL_RETROFIT_GUROBI_SEED=2 ;;
    *_seed3) export COAL_RETROFIT_GUROBI_SEED=3 ;;
    *_seed4) export COAL_RETROFIT_GUROBI_SEED=4 ;;
    *)       unset COAL_RETROFIT_GUROBI_SEED || true ;;
  esac
  echo "=== START $s $(date -Iseconds) seed=${COAL_RETROFIT_GUROBI_SEED:-none} threads=8 ===" >> "$DRIVER"
  python scripts/run_single.py "$s" --threads 8 >> "$LOG" 2>&1
  rc=$?
  echo "=== END   $s rc=$rc $(date -Iseconds) ===" >> "$DRIVER"
  [ "$rc" -ne 0 ] && { echo "!!! $s FAILED rc=$rc -- aborting lane $LANE" >> "$DRIVER"; exit 1; }
done
echo "=== GAPS LANE${LANE} COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
