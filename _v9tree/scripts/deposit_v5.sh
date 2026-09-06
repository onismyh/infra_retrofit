#!/usr/bin/env bash
# Re-render every figure from the CORRECTED-basis results and deposit into results/figures/v5.
#
# Run directly to deposit whatever is currently on disk, or with --wait to block until the
# v5 re-solve campaign writes its completion marker and then deposit. The two are the same
# code path so a mid-campaign deposit and the final one cannot drift.
set -u
export PYTHONIOENCODING=utf-8
export PYTHONPATH=src:scripts
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEST=results/figures/v5
LOG=results/deposit_v5.log

if [ "${1:-}" = "--wait" ]; then
  echo "waiting for V5 RESUME COMPLETE ..." >> "$LOG"
  while ! grep -q "V5 RESUME COMPLETE" results/resolve_v5_driver.log 2>/dev/null; do sleep 120; done
  echo "=== campaign complete $(date -Iseconds), rendering ===" >> "$LOG"
fi

mkdir -p "$DEST/figures" "$DEST/extended"
{
  echo "=== DEPOSIT $(date -Iseconds) ==="
  python scripts/check_run_provenance.py || true
  for f in plot_fig1_water_footprint plot_fig2_constraint_response plot_fig3_mechanism            plot_fig4_network_reconfiguration plot_fig5_pathway_succession; do
    echo "--- $f ---"
    python "scripts/$f.py" || echo "FAILED $f"
  done
  echo "--- extended ---"
  python scripts/plot_extended.py || echo "FAILED extended"
  echo "--- numbers ledger ---"
  python scripts/build_numbers_ledger.py || echo "FAILED ledger"
} >> "$LOG" 2>&1

cp -f results/figures/fig1_water_footprint.*        "$DEST/figures/" 2>/dev/null
cp -f results/figures/fig2_constraint_response.*    "$DEST/figures/" 2>/dev/null
cp -f results/figures/fig3_mechanism.*              "$DEST/figures/" 2>/dev/null
cp -f results/figures/fig4_network_reconfiguration.* "$DEST/figures/" 2>/dev/null
cp -f results/figures/fig5_pathway_succession.*     "$DEST/figures/" 2>/dev/null
cp -f results/figures/extended/*                    "$DEST/extended/" 2>/dev/null
cp -f results/numbers_ledger.json results/numbers_ledger.md "$DEST/" 2>/dev/null
cp -f results/v5_corrected_water_ledger.txt results/v5_corrected_water_ledger.json "$DEST/" 2>/dev/null

echo "=== DEPOSITED $(date -Iseconds) ===" >> "$LOG"
ls -1 "$DEST/figures" | wc -l >> "$LOG"
