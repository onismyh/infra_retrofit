#!/usr/bin/env bash
# Re-render every figure and deposit into results/figures/v6.
# --wait blocks until the v6 stage-2 campaign marker appears, so a deposit cannot be made on a
# mixed-vintage result set. Run without --wait to deposit whatever is on disk now.
set -u
export PYTHONIOENCODING=utf-8
export PYTHONPATH=src:scripts
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEST=results/figures/v6
LOG=results/deposit_v6.log

if [ "${1:-}" = "--wait" ]; then
  echo "waiting for V6 STAGE2 COMPLETE ..." >> "$LOG"
  while ! grep -q "V6 STAGE2 COMPLETE" results/resolve_v6_driver.log 2>/dev/null; do
    if grep -qE "V6 (QUEUE|STAGE2) ABORTED" results/resolve_v6_driver.log 2>/dev/null; then
      echo "=== campaign aborted, depositing anyway $(date -Iseconds) ===" >> "$LOG"; break
    fi
    sleep 180
  done
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

for f in fig1_water_footprint fig2_constraint_response fig3_mechanism          fig4_network_reconfiguration fig5_pathway_succession; do
  cp -f "results/figures/$f.pdf" "results/figures/$f.png" "$DEST/figures/" 2>/dev/null
done
cp -f results/figures/extended/* "$DEST/extended/" 2>/dev/null
cp -f results/numbers_ledger.json results/numbers_ledger.md "$DEST/" 2>/dev/null
cp -f results/v6_s_star_ensemble.txt results/v6_s_star_ensemble.json "$DEST/" 2>/dev/null
cp -f results/v6_corrected_basis_summary.txt "$DEST/" 2>/dev/null
cp -f results/figures/v5/existing_withdrawal_share_sourcing.md "$DEST/" 2>/dev/null
# round-2 artifacts
cp -f results/v6_allocation_curve.txt results/v6_allocation_curve.csv "$DEST/" 2>/dev/null
cp -f results/v6_round2_findings.md results/v6_round2_review_adjudication.md "$DEST/" 2>/dev/null
# the Chinese final storyline lives in the deposit directory itself; keep it out of the wipe
cp -f results/v6_efr_convention_table.txt "$DEST/" 2>/dev/null

echo "=== DEPOSITED $(date -Iseconds); $(ls -1 "$DEST/figures" | wc -l) figure files ===" >> "$LOG"
grep -c "VINTAGE" "$LOG" >> "$LOG" 2>/dev/null || true
