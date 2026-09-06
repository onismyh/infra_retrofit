#!/usr/bin/env bash
# Lane 2 of the v6 provenance close-out. Runs CONCURRENTLY with the seed5 lane: two solves at
# 8 pinned threads each on a 32-core / 137 GB host, ~42 GB peak per solve. Thread count stays
# pinned at 8, so determinism is unaffected -- contention costs wall clock, not reproducibility.
#
# WHY THESE FOUR. After the dry-season correction every run in the campaign is on the corrected
# inputs, but only runs solved after the input-digest patch carry proof of it. `input_vintage`
# now verifies a digest-less run by matching its solved water right-hand side, to the bit,
# against a digested run of the same hydrology member. These four have no digested counterpart
# to match -- `WA_wgap_126_dry_nobias` removes the bias correction and so legitimately holds
# different water -- and therefore resolve to `pre-digest:current-unverified`, an mtime guess.
# Re-solving them replaces the last inference in the provenance chain with a stamped digest.
set -u
QUEUE="WA_wgap_126_dry WA_wgap_126_dry_wd085 WA_wgap_370_dry BASE"
LOG=results/resolve_v6_lane2.log
DRIVER=results/resolve_v6_lane2_driver.log
for s in $QUEUE; do
  unset COAL_RETROFIT_GUROBI_SEED || true
  echo "=== START $s $(date -Iseconds) threads=8 ===" >> "$DRIVER"
  python scripts/run_single.py "$s" --threads 8 >> "$LOG" 2>&1
  rc=$?
  echo "=== END   $s rc=$rc $(date -Iseconds) ===" >> "$DRIVER"
  if [ "$rc" -ne 0 ]; then
    echo "!!! $s FAILED rc=$rc -- aborting" >> "$DRIVER"; exit 1
  fi
done
echo "=== V6 LANE2 COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
