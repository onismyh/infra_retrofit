#!/usr/bin/env bash
# Try every rendered figure script once and record pass/fail with the first error line.
#
# Not a substitute for scripts/render_version.py -- this is the PROBE that says which figures
# the currently-solved scenario set can already produce, so the version README can carry a
# real per-figure status table instead of a guess. Safe to run while the solve campaign is
# going: it touches no results, only reads them.
set -u
cd "$(dirname "$0")" || exit 1

OUT=render_probe.log
: > "$OUT"
SKIP="plot_fig3_attribution.py plot_ed_source_sink_matching_years.py"

for f in scripts/plot_fig*.py scripts/plot_ed_*.py; do
  base=$(basename "$f")
  case " $SKIP " in *" $base "*) echo "SKIP $base" >> "$OUT"; continue ;; esac
  log="results/render_${base%.py}.log"
  if PYTHONIOENCODING=utf-8 timeout 1800 python -W ignore "$f" > "$log" 2>&1; then
    echo "PASS $base" >> "$OUT"
  else
    # 最后一行非空的输出通常就是异常本身；情景缺失的脚本会在这里给出缺哪个
    err=$(grep -E "Error|error|Traceback|not yet solved|FileNotFound|skip" "$log" | tail -1)
    echo "FAIL $base  ${err:0:150}" >> "$OUT"
  fi
done
echo "PROBE FINISHED $(date -Is)" >> "$OUT"
