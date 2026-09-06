#!/usr/bin/env bash
# Re-solve the one seed replicate the v5/v6 queues missed.
#
# WHY. `WA_cwatm_126_dry_wd085_seed5` was solved on the superseded per-cell-minimum water
# basis and never re-queued, because the seed family in the resume queue was written as
# seed2/3/4 only. It carries fingerprint 0xbe7b31c2 against the corrected family's 0xbf2f6de
# and an objective 4.5% above them. `build_numbers_ledger.seed_entries` pools the family by
# filename alone, so that single stale run raised the reported degeneracy floor from 0.375%
# to 4.50% -- above the 1.34% headline, i.e. it made the paper's main result look
# unresolvable against solver noise. The gate is now closed in the ledger as well; this run
# restores the replicate so the floor is measured at n=5 (d2 2.326) instead of n=4 (2.059).
set -u
LOG=results/resolve_seed5.log
DRIVER=results/resolve_seed5_driver.log
while pgrep -f "run_single.py" > /dev/null 2>&1; do sleep 30; done
export COAL_RETROFIT_GUROBI_SEED=5
echo "=== START WA_cwatm_126_dry_wd085_seed5 $(date -Iseconds) seed=5 threads=8 ===" >> "$DRIVER"
python scripts/run_single.py WA_cwatm_126_dry_wd085_seed5 --threads 8 >> "$LOG" 2>&1
rc=$?
echo "=== END WA_cwatm_126_dry_wd085_seed5 rc=$rc $(date -Iseconds) ===" >> "$DRIVER"
[ "$rc" -ne 0 ] && { echo "!!! FAILED rc=$rc -- aborting" >> "$DRIVER"; exit 1; }
echo "=== SEED5 COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
