#!/usr/bin/env bash
# Lane 3: GCM diversity and a solved bracket around the Hai sign change.
#
# WHY. The paper's ensemble statement rests on 20 hydrology members, but every member SOLVED in
# the MILP is gfdl-esm4 -- four of them (cwatm/watergap2 x ssp126/ssp370), as control/treatment
# pairs. So the solved spread is a hydrology-model and SSP spread with one GCM inside it, and a
# referee will say so. These two members add a second and third GCM AND are chosen by where
# they sit on the statistic the paper turns on: s* for full capture in the Hai at 2030.
#
#   solved today:  wgap|gfdl|ssp370  -0.627   wgap|gfdl|ssp126 -0.433
#                  cwatm|gfdl|ssp370 +0.014   cwatm|gfdl|ssp126 +0.050
#   added here:    cwatm|ipsl|ssp126 -0.029   (immediately below the sign change)
#                  cwatm|mpi|ssp370  +0.293   (the wettest Hai in the ensemble)
#
# That gives six solved members straddling s* = 0 with three GCMs, which converts "the scarcity
# result is analytic on 20 members but solved on one" into "solved on six, spanning the sign".
# Waits for lane 2 so at most two solves share the 32-core host at 8 pinned threads each.
set -u
QUEUE="WA_cwatm_ipsl126_dry WA_cwatm_ipsl126_dry_wd085 WA_cwatm_mpi370_dry WA_cwatm_mpi370_dry_wd085"
LOG=results/resolve_v6_lane3.log
DRIVER=results/resolve_v6_lane3_driver.log
until grep -q "V6 LANE2 COMPLETE" results/resolve_v6_lane2_driver.log 2>/dev/null; do
  grep -q "FAILED" results/resolve_v6_lane2_driver.log 2>/dev/null && {
    echo "=== lane2 failed; lane3 not starting $(date -Iseconds) ===" >> "$DRIVER"; exit 1; }
  sleep 60
done
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
echo "=== V6 LANE3 COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
