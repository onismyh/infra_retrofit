#!/usr/bin/env bash
# v6 extension of the corrected-basis campaign.
#
# WHY THESE RUNS. Three independent audits converged on the same two gaps.
#
# 1. THE FIGURES ARE MIXED-VINTAGE. `WA_cwatm_126_dry_wd085_noair` feeds four figures and was
#    never in the v5 queue, so Figs 2, 3 and 5 currently difference a corrected run against a
#    superseded one. The proof is a dominance violation: the retirement-cap RELAXATION
#    (`_wd085_capfree`) reports a dual bound of 14.55719e12 above its own restriction's feasible
#    primal of 14.14216e12, which is impossible on one model. `solver._input_digest` now stamps
#    the water-input hashes so this cannot recur silently, and `plot_style.assert_same_vintage`
#    refuses to draw across the boundary.
#
# 2. THE ALLOCATION CLAIM HAS NEVER BEEN MEASURED. The paper's headline is now about the
#    reservation share s, and the response to s is strongly convex -- on the superseded basis
#    s = 0 -> 0.559 -> 0.70 -> 0.85 -> 0.90 gave +0.00 / +0.57 / +1.04 / +5.83 / +9.56%, i.e. a
#    marginal cost per unit of s running 1.0 -> 3.3 -> 32 -> 75. A two-point contrast reports a
#    chord across that hockey stick and cannot locate the knee, which IS the allocation question.
#    wd056/wd070/wd090 are registered and were solved on the old basis, and no plot script reads
#    them. On the corrected basis the knee has moved and its new location is unmeasured.
#
# 3. The air-retrofit capex is an unsourced placeholder (300 CNY/kW) against a literature anchor
#    of ~1370. The VOLUME response is flat across that 4.6x swing while the COST moves more than
#    the entire headline water effect. Both sensitivities are on the superseded basis.
#
# Ordered so the figure-blocking runs land first.
set -u
QUEUE="
WA_cwatm_126_dry_wd085_noair
WA_cwatm_126_dry_noair
WA_cwatm_126_dry_wd070
WA_cwatm_126_dry_wd090
WA_cwatm_126_dry_wd056
WA_cwatm_126_dry_wd085_air1370
WA_cwatm_126_dry_wd085_air1000
"
LOG=results/resolve_v6.log
DRIVER=results/resolve_v6_driver.log
# Wait for the v5 resume campaign to finish before starting.
while ! grep -q "V5 RESUME COMPLETE" results/resolve_v5_driver.log 2>/dev/null; do
  if grep -q "V5 RESUME ABORTED" results/resolve_v5_driver.log 2>/dev/null; then
    echo "=== V6 NOT STARTED: v5 resume aborted $(date -Iseconds) ===" >> "$DRIVER"; exit 1
  fi
  sleep 120
done
for s in $QUEUE; do
  unset COAL_RETROFIT_GUROBI_SEED || true
  echo "=== START $s $(date -Iseconds) threads=8 ===" >> "$DRIVER"
  python scripts/run_single.py "$s" --threads 8 >> "$LOG" 2>&1
  rc=$?
  echo "=== END   $s rc=$rc $(date -Iseconds) ===" >> "$DRIVER"
  if [ "$rc" -ne 0 ]; then
    echo "!!! $s FAILED rc=$rc -- aborting" >> "$DRIVER"
    echo "=== V6 QUEUE ABORTED $(date -Iseconds) ===" >> "$DRIVER"; exit 1
  fi
done
echo "=== V6 QUEUE COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
