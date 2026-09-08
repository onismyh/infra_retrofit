#!/usr/bin/env bash
# Scenarios added AFTER run_campaign.sh was already running, so they could not join a lane.
#
# WA_cwatm_126_dry_oq_capfree -- the one-factor comparator for Fig 5's frozen-cooling
# contrast. Without it the pair is (BINDS, *_noair_capfree), which moves the dry-cooling
# switch AND the retirement cap at the same time; plot_fig5_pathway_succession.py detects
# that and prints a warning, but a warned-about confound is still a confound.
#
# Same conventions as run_campaign.sh: 8 threads, gap from the registry, resumable.
# Single lane -- run it AFTER run_campaign.sh has finished, never alongside it, or three
# solves are live at once and `_biomass_access_matrices` OOMs (CLAUDE.md 二.5).
set -u
cd "$(dirname "$0")" || exit 1

THREADS=8
LOG=campaign.log
EXTRA=(
  WA_cwatm_126_dry_oq_capfree
)

mkdir -p results
for name in "${EXTRA[@]}"; do
  if [ -f "results/${name}.json" ]; then
    echo "$(date -Is) extra skip $name (already solved)" >> "$LOG"
    continue
  fi
  echo "$(date -Is) extra start $name" >> "$LOG"
  PYTHONIOENCODING=utf-8 python -W ignore -u scripts/run_single.py "$name" --threads "$THREADS" \
    > "results/${name}.solve.log" 2>&1
  echo "$(date -Is) extra done $name rc=$?" >> "$LOG"
done
echo "$(date -Is) extra FINISHED" >> "$LOG"
