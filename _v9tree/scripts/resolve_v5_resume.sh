#!/usr/bin/env bash
# Resume of the v5 corrected-basis campaign.
#
# WHY A RESUME EXISTS. `WA_cwatm_126_dry` and `WA_cwatm_126_dry_wd085` solved cleanly on the
# corrected dry-season basis (13.95520 and 14.14216 e12 CNY, gaps 0.830% and 0.691%, both
# thread-pinned at 8). The remaining thirteen then failed in under two seconds each with
#     SyntaxError: from __future__ imports must occur at the beginning of the file
# because an `import os` added to solver.py mid-campaign landed on line 1, above the
# `from __future__` line. The import has been moved; nothing else in the model changed, and
# the two completed runs are unaffected because they were launched before the edit.
set -u
QUEUE="
WA_wgap_126_dry
WA_wgap_126_dry_wd085
WA_wgap_370_dry
WA_wgap_370_dry_wd085
BASE
WA_cwatm_126_dry_wd085_seed2
WA_cwatm_126_dry_wd085_seed3
WA_cwatm_126_dry_wd085_seed4
WA_cwatm_126_dry_capfree
WA_cwatm_126_dry_wd085_capfree
WA_cwatm_126_dry_wd085_noair_capfree
WA_cwatm_370_dry
WA_cwatm_370_dry_wd085
"
LOG=results/resolve_v5.log
DRIVER=results/resolve_v5_driver.log
while pgrep -f "run_single.py" > /dev/null 2>&1; do sleep 30; done
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
  if [ "$rc" -ne 0 ]; then
    echo "!!! $s FAILED rc=$rc -- aborting so a broken build cannot silently empty the queue" >> "$DRIVER"
    echo "=== V5 RESUME ABORTED $(date -Iseconds) ===" >> "$DRIVER"
    exit 1
  fi
done
echo "=== V5 RESUME COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
