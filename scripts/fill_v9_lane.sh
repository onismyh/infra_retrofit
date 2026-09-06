#!/usr/bin/env bash
# One lane of the v9 fill campaign (89-sink inputs, solved inside _v9tree/).
#
# Usage: bash scripts/fill_v9_lane.sh <lane-letter> <scenario> [<scenario> ...]
# Run from the tree root (cd _v9tree && bash scripts/fill_v9_lane.sh a ...). Two lanes at most
# (CLAUDE.md §二 5: `_biomass_access_matrices` allocates 7.39 GiB per solve, three lanes OOM).
#
# Rules baked in:
#   * threads pinned at 8 (CLAUDE.md §二 1);
#   * MIPGap left at the registry value (1 %) -- never pass --mip-gap here, the seed families
#     are degeneracy probes and stop being one the moment tolerance is loosened (§二 2);
#   * `*_seedN` names export COAL_RETROFIT_GUROBI_SEED=N, everything else runs with seed 0;
#   * a scenario whose results/<name>.json already exists is skipped, so a lane can be
#     restarted after an interruption without re-solving;
#   * a failed solve is logged and the lane moves on -- one bad run must not stall two days
#     of queue; the driver log (results/campaign.log) records every rc.
set -u
cd "$(dirname "$0")/.." || exit 1
LANE="$1"; shift
DRIVER=results/campaign.log
echo "=== fill lane $LANE start $(date) queue: $* ===" >> "$DRIVER"
for s in "$@"; do
  if [ -f "results/$s.json" ]; then
    echo "--- $s skip (results/$s.json exists) $(date) ---" >> "$DRIVER"
    continue
  fi
  unset COAL_RETROFIT_GUROBI_SEED
  case "$s" in
    *_seed[0-9]) export COAL_RETROFIT_GUROBI_SEED="${s##*_seed}" ;;
  esac
  echo "--- $s start $(date) lane=$LANE seed=${COAL_RETROFIT_GUROBI_SEED:-0} threads=8 ---" >> "$DRIVER"
  python scripts/run_single.py "$s" --threads 8 > "results/$s.solve.log" 2>&1
  rc=$?
  echo "--- $s done rc=$rc $(date) ---" >> "$DRIVER"
  if [ "$rc" -ne 0 ]; then
    echo "!!! $s FAILED rc=$rc (lane $LANE continues) $(date)" >> "$DRIVER"
  fi
done
echo "=== fill lane $LANE complete $(date) ===" >> "$DRIVER"
