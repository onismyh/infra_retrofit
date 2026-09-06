#!/usr/bin/env bash
# The last uncertified run in the six-member cross-replication.
#
# WHY. `WA_wgap_370_dry` was re-solved in lane 2 and now carries digest 9c613b0ddf4b, but its
# treatment arm `WA_wgap_370_dry_wd085` was not in that queue -- it had been solved during the
# v5 resume, before the digest patch, so it resolves to `pre-digest:current-unverified` and has
# no digested same-member counterpart to verify against. Its water is almost certainly correct
# (it was solved after the 06:55 input rebuild); the point is that the pair it is differenced
# in is the only one of six that cannot be shown to be same-vintage.
set -u
LOG=results/resolve_v6_lane4.log
DRIVER=results/resolve_v6_lane4_driver.log
echo "=== START WA_wgap_370_dry_wd085 $(date -Iseconds) threads=8 ===" >> "$DRIVER"
python scripts/run_single.py WA_wgap_370_dry_wd085 --threads 8 >> "$LOG" 2>&1
rc=$?
echo "=== END WA_wgap_370_dry_wd085 rc=$rc $(date -Iseconds) ===" >> "$DRIVER"
[ "$rc" -ne 0 ] && { echo "!!! FAILED rc=$rc" >> "$DRIVER"; exit 1; }
echo "=== V6 LANE4 COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
