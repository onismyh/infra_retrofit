#!/usr/bin/env bash
# Re-solve the sensitivity family on the CURRENT build.
#
# WHY. ED Fig 3's tornado differences each SA run against BASE. BASE was re-solved during this
# campaign and carries 18 cost categories including `air_retrofit_capex`; every SA run dates
# from 2026-08-09 and carries 17, because the wet-to-dry retrofit did not exist in that build.
# The tornado is therefore a difference of two different MODELS, not a sensitivity, and the
# offset is large enough to reverse conclusions: SA_ammonia_cost_50 plots at +0.548% against the
# current BASE and at -0.109% against its own contemporaneous BASE. Two of six published bars
# have the wrong SIGN. Nothing in `plot_extended.py` could catch this -- it holds no vintage
# guard, and `input_vintage` compares only the water right-hand side, which did not change here.
#
# Lane argument: 1 or 2. Two lanes at 8 pinned threads each on the 32-core host; the thread pin
# is what determinism depends on, and it is unchanged, so concurrency costs wall clock only.
set -u
LANE="${1:-1}"
if [ "$LANE" = "1" ]; then
  QUEUE="SA_air_capex_low SA_air_capex_high SA_air_penalty_high SA_retire_cap_010 SA_retire_cap_025
         SA_retire_cost_250 SA_retire_cost_650 SA_discount_3pct SA_discount_8pct SA_ccs_capex_low
         SA_ccs_capex_high"
else
  QUEUE="SA_biomass_cost_150 SA_biomass_cost_200 SA_ammonia_cost_50 SA_ammonia_cost_70
         SA_retire_cost_500 SA_retire_cost_1000 SA_pipe_mid SA_pipe_full SA_injectivity_half
         SA_carbon_low SA_carbon_high"
fi
LOG="results/resolve_sa_lane${LANE}.log"
DRIVER="results/resolve_sa_lane${LANE}_driver.log"
for s in $QUEUE; do
  unset COAL_RETROFIT_GUROBI_SEED || true
  echo "=== START $s $(date -Iseconds) threads=8 ===" >> "$DRIVER"
  python scripts/run_single.py "$s" --threads 8 >> "$LOG" 2>&1
  rc=$?
  echo "=== END   $s rc=$rc $(date -Iseconds) ===" >> "$DRIVER"
  if [ "$rc" -ne 0 ]; then
    echo "!!! $s FAILED rc=$rc -- aborting lane $LANE" >> "$DRIVER"; exit 1
  fi
done
echo "=== SA LANE${LANE} COMPLETE $(date -Iseconds) ===" >> "$DRIVER"
