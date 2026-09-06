#!/usr/bin/env bash
# Lane A, resumed at MIPGap = 2%.
#
# WHY 2% AND NOT 1%. Cutting biomass supply removes the water-free wedge (co-firing) and the
# water-hungry one (BECCS) together, so the replacement has to split between CCS retrofit and
# retirement and the integer part gets hard: the bio015 CONTROL took 10.03 h to reach 1.009%,
# and its treatment was still at 5.63% after an hour. The registry entries now carry
# mip_gap 0.02 for the three runs below.
#
# WHAT THAT COSTS. The certified interval on each pair is roughly twice as wide. These runs were
# registered to test the DIRECTION of the biomass mechanism, and after the source-sink rebuild
# that mechanism is withdrawn anyway -- rounds 1 and 2 disagree in SIGN on the biomass response
# while agreeing on the objective to 0.002 pp (源汇聚类与划分_审查.md §9.4). A direction test
# does not need a 1% interval.
#
# WHAT IS NOT RE-RUN. WA_cwatm_126_dry_bio015 is already solved at 1.009% and is left alone, so
# the bio015 pair is asymmetric: control certified at 1.009%, treatment at 2%. The interval
# formula uses each run's own gap, so it stays valid -- just wider on the treatment side. Note
# that when reading it.
set -u
QUEUE="WA_cwatm_126_dry_wd085_bio015 WA_cwatm_126_dry_bio042 WA_cwatm_126_dry_wd085_bio042"
LOG=results/resolve_gaps_A2.log
DRIVER=results/resolve_gaps_A2_driver.log
unset COAL_RETROFIT_GUROBI_SEED || true
for s in $QUEUE; do
  echo "=== START $s $(date -Iseconds) threads=8 mip_gap=0.02 ===" >> "$DRIVER"
  python scripts/run_single.py "$s" --threads 8 >> "$LOG" 2>&1
  rc=$?
  echo "=== END   $s rc=$rc $(date -Iseconds) ===" >> "$DRIVER"
  [ "$rc" -ne 0 ] && { echo "!!! $s FAILED rc=$rc -- aborting" >> "$DRIVER"; exit 1; }
done
echo "=== GAPS LANE A2 COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
