#!/usr/bin/env bash
# v9.1 solve campaign: the minimal scenario closure for the main + extended figure set.
#
# Two lanes, never more. `_biomass_access_matrices` allocates a 13,949 x 71,148 float64 dense
# matrix (7.39 GiB) per solve; a third lane OOMs (CLAUDE.md 二.5).
# Eight threads each, fixed. Gurobi is deterministic only for a fixed (model, params, THREADS),
# so two runs that will be differenced must share it (CLAUDE.md 二.1).
# MIPGap comes from the scenario registry (0.01 throughout, seeds included) and is never
# overridden here -- a loosened gap on a seed family would be read as degeneracy (二.2).
#
# Resumable: a scenario whose <name>.json already exists is skipped, so the campaign can be
# killed and restarted without losing work. Progress goes to campaign.log.
set -u
cd "$(dirname "$0")" || exit 1

THREADS=8
LOG=campaign.log

# Ordered so the earliest completions unlock the most figures:
#   BASE + control + treatment      -> fig3, ed_hai_closeup, ed_province_transition,
#                                      ed_reversion, ed_who_converts, ed_biomass_sourcing
#   + noair + the other two arms    -> fig2, ed_water_on_off, ed_variance_decomposition
#   + seeds                         -> fig4, fig5, ed_source_sink_matching, ed_water_abatement
#   + capex / capfree / nobias      -> the remaining panels of fig3, fig4, fig5
LANE_A=(
  BASE
  WA_cwatm_126_dry_oq
  WA_cwatm_126_dry_oq_noair
  WA_cwatm_370_dry_oq
  WA_wgap_126_dry_oq
  WA_cwatm_126_dry_oq_seed2
  WA_cwatm_126_dry_oq_seed3
  WA_cwatm_126_dry_oq_seed4
  WA_cwatm_126_dry_oq_air1000
  WA_cwatm_126_dry_oq_air1370
)
LANE_B=(
  WA_cwatm_126_dry_oq_envonly
  WA_cwatm_370_dry_oq_envonly
  WA_wgap_126_dry_oq_envonly
  WA_cwatm_126_dry_oq_envonly_seed2
  WA_cwatm_126_dry_oq_envonly_seed3
  WA_cwatm_126_dry_oq_envonly_seed4
  WA_cwatm_126_dry_oq_noair_capfree
  WA_cwatm_370_dry_oq_envonly_nobias
  WA_wgap_126_dry_oq_envonly_nobias
)

run_lane() {
  local lane=$1; shift
  for name in "$@"; do
    if [ -f "results/${name}.json" ]; then
      echo "$(date -Is) lane$lane skip $name (already solved)" >> "$LOG"
      continue
    fi
    # The seed is read from the environment, not the scenario table: the registry entries for
    # the *_seedN replicates are deliberately IDENTICAL to their base run, so that the model,
    # its parameters and its feasible set are bit-identical and the only thing that differs is
    # the solver's search path. That is what makes the family a degeneracy measurement.
    local seed=""
    case "$name" in *_seed[0-9]*) seed="${name##*_seed}" ;; esac
    echo "$(date -Is) lane$lane start $name seed=${seed:-0}" >> "$LOG"
    COAL_RETROFIT_GUROBI_SEED="$seed" PYTHONIOENCODING=utf-8 \
      python -W ignore -u scripts/run_single.py "$name" --threads "$THREADS" \
      > "results/${name}.solve.log" 2>&1
    echo "$(date -Is) lane$lane done $name rc=$?" >> "$LOG"
  done
  echo "$(date -Is) lane$lane FINISHED" >> "$LOG"
}

mkdir -p results
echo "$(date -Is) campaign start: ${#LANE_A[@]} + ${#LANE_B[@]} scenarios, 2 lanes x $THREADS threads" >> "$LOG"
run_lane A "${LANE_A[@]}" &
run_lane B "${LANE_B[@]}" &
wait
echo "$(date -Is) campaign FINISHED" >> "$LOG"
