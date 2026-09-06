#!/usr/bin/env bash
# v5 re-solve campaign, on the CORRECTED dry-season basis.
#
# WHY EVERYTHING IS RE-SOLVED. `builders/water.py` selected each basin's low-flow quarter
# PER GRID CELL and then summed those independent minima over the basin -- sum(min) where the
# constraint needs min(sum). The understatement was 2.25x in the Hai, 2.13x in the Northwest
# Interior, 1.47x in the Yellow and 1.19x in the Huai, i.e. largest in exactly the four basins
# the study calls over-limit, and 1.03-1.11x in the four it calls safe. Every water-constrained
# result published before 2026-08-18 07:00 was solved against a right-hand side that was too
# small, by a factor that varied systematically with the answer.
#
# `inputs/water_nodes.csv` and `inputs/water_supply_links.csv` are byte-identical after the
# rebuild and annual availability is unchanged to 0 m3, so the model STRUCTURE is the same --
# only the water constraint's RHS moves. Fingerprints should therefore be comparable to the
# superseded set, which is itself a check on the rebuild.
#
# Ordering is by what the figures need first: the three binding contrasts, then the no-water
# reference, then the degeneracy floor, then the cap and frozen-pathway diagnostics.
set -u
QUEUE="
WA_cwatm_126_dry
WA_cwatm_126_dry_wd085
WA_wgap_126_dry
WA_wgap_126_dry_wd085
WA_wgap_370_dry
WA_wgap_370_dry_wd085
BASE
WA_cwatm_370_dry
WA_cwatm_370_dry_wd085
WA_cwatm_126_dry_wd085_seed2
WA_cwatm_126_dry_wd085_seed3
WA_cwatm_126_dry_wd085_seed4
WA_cwatm_126_dry_capfree
WA_cwatm_126_dry_wd085_capfree
WA_cwatm_126_dry_wd085_noair_capfree
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
  echo "=== END   $s rc=$? $(date -Iseconds) ===" >> "$DRIVER"
done
echo "=== V5 QUEUE COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
