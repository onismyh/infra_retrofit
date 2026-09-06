#!/usr/bin/env bash
# Re-render every figure and deposit into results/figures/v7.
#
# WHAT IS NEW IN v7 versus v6.
#   * The Extended Data set is now the UNIT-LEVEL set. `plot_extended.py`, which drew almost
#     everything at province resolution off the BASE scenario and still carries known unfixed
#     defects (ed_fig2/7/9/11/12), is deposited to legacy/ and marked superseded rather than
#     deleted, because five of its panels have no replacement yet.
#   * ED3 groups sites by their CAPACITY-dominant cooling technology instead of the unit-count
#     mode the model carries. The rank inversion survives and strengthens (once-through / wet
#     tower withdrawal ratio 4.20x -> 4.91x); the cloud goes from 71 sites / 349.9 GW to 94
#     sites / 486.2 GW.
#   * ED8 and ED9 book each site to the province holding most of its CAPACITY. Six sites and
#     54.0 GW move, including the 27.9 GW hub standing on Shanghai's coordinates that the
#     unit-count mode booked entirely to Jiangsu.
#   * NEW ED10 documents the source-sink rebuild: 103 geological storage bodies instead of 35
#     distance-merged hubs, the detour factor measured on China's own trunk network, and the
#     connectivity repair that ends 7 stranded plants and 471 GW of routing artefact.
#
# BOTH EXTENDED SETS WRITE ed_fig<N> INTO results/figures/extended/, so they collide. The legacy
# set is rendered first and copied out before the new set overwrites it.
set -u
export PYTHONIOENCODING=utf-8
export PYTHONPATH=src:scripts
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEST=results/figures/v7
LOG=results/deposit_v7.log

mkdir -p "$DEST/figures" "$DEST/extended" "$DEST/legacy" "$DEST/docs"
: > "$LOG"

{
  echo "=== DEPOSIT v7 $(date -Iseconds) ==="
  python scripts/check_run_provenance.py || true

  echo "--- main figures ---"
  for f in plot_fig1_water_footprint plot_fig2_constraint_response plot_fig3_mechanism \
           plot_fig4_network_reconfiguration plot_fig5_pathway_succession; do
    echo "  [$f]"
    python "scripts/$f.py" || echo "  FAILED $f"
  done

  echo "--- legacy extended (superseded; kept for the five panels with no replacement) ---"
  python scripts/plot_extended.py || echo "  FAILED plot_extended"
} >> "$LOG" 2>&1

cp -f results/figures/extended/* "$DEST/legacy/" 2>/dev/null
rm -f results/figures/extended/ed_fig*.pdf results/figures/extended/ed_fig*.png 2>/dev/null

{
  echo "--- unit-level extended data ---"
  for f in plot_ed_fleet_atlas plot_ed_who_converts plot_ed_water_basis \
           plot_ed_biomass_sourcing plot_ed_reversion plot_ed_hai_closeup \
           plot_ed_sensitivity plot_ed_province_transition plot_ed_sink_network \
           plot_ed_variance_decomposition; do
    echo "  [$f]"
    python "scripts/$f.py" || echo "  FAILED $f"
  done
  echo "--- numbers ledger ---"
  python scripts/build_numbers_ledger.py || echo "  FAILED ledger"
} >> "$LOG" 2>&1

for f in fig1_water_footprint fig2_constraint_response fig3_mechanism \
         fig4_network_reconfiguration fig5_pathway_succession; do
  cp -f "results/figures/$f.pdf" "results/figures/$f.png" "$DEST/figures/" 2>/dev/null
done
cp -f results/figures/extended/* "$DEST/extended/" 2>/dev/null

cp -f results/numbers_ledger.json results/numbers_ledger.md "$DEST/" 2>/dev/null
cp -f results/v6_s_star_ensemble.txt results/v6_s_star_ensemble.json "$DEST/docs/" 2>/dev/null
cp -f results/v6_corrected_basis_summary.txt results/v6_efr_convention_table.txt "$DEST/docs/" 2>/dev/null
cp -f results/v6_allocation_curve.txt results/v6_allocation_curve.csv "$DEST/docs/" 2>/dev/null
for d in nature_water_storyline_FINAL_zh.md v6_round2_findings.md v6_round2_review_adjudication.md \
         附录图_审查与重绘.md 退役与机组年份_审查.md 源汇聚类与划分_审查.md 源汇方法学_重建.md \
         existing_withdrawal_share_sourcing.md; do
  cp -f "results/figures/v6/$d" "$DEST/docs/" 2>/dev/null
done

{
  echo "=== DEPOSITED $(date -Iseconds) ==="
  echo "main    $(ls -1 "$DEST/figures" 2>/dev/null | wc -l) files"
  echo "ED      $(ls -1 "$DEST/extended" 2>/dev/null | wc -l) files"
  echo "legacy  $(ls -1 "$DEST/legacy" 2>/dev/null | wc -l) files"
  echo "docs    $(ls -1 "$DEST/docs" 2>/dev/null | wc -l) files"
} >> "$LOG"
tail -6 "$LOG"
