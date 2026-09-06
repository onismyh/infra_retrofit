"""Generate the five main figures, in order, from committed inputs and results.

These five are the paper's argument. Everything else under scripts/ is either an extended-data
figure or a superseded draft kept for its logic -- in particular plot_water_fig1.py,
plot_water_fig3.py, plot_water_nature.py and plot_water_cost_tradeoff.py predate the
three-basis water accounting and the basin budget, and their output must not be used.

  Fig 1  the water price of abatement, and where that water has to come from
  Fig 2  basin dry-season water as a hard spatial ceiling on capture
  Fig 3  institution vs climate as the binding mechanism, and the dry-cooling adaptation channel
  Fig 4  the no-regret pipeline core: what to build now, whatever water does

The Fig 3 slot dispatched to plot_fig3_attribution.py until 2026-08-17, which had been demoted
to Extended Data -- `make_main_figures.py 3` regenerated the wrong figure. It now dispatches to
plot_fig3_mechanism.py; plot_fig3_attribution.py is run on its own when the ED panel is wanted.

Note on Fig 4: it does NOT claim that water redraws the CO2 network. That claim was refuted by
its own evidence -- two runs differing only by a bias-correction switch, objectives 0.006%
apart, disagree about more of the network than binding-vs-non-binding does -- so the figure
states the degeneracy floor instead and carries the no-regret core, which survives.

Uncertainty decompositions are Extended Data, not main figures. The 3-way GCM / hydrology / SSP
variance decomposition that used to be Fig 2(c) now lives in
`scripts/plot_ed_variance_decomposition.py`, which writes to results/figures/extended/.

Each figure script fails loudly rather than degrading to zeros when its inputs are stale, so
a clean run of this script is itself the check that results and figures are in step.

All five are sized to Nature Water's 183 mm double column. They are NOT free to grow: save_fig
uses bbox_inches='tight', which silently ENLARGES the saved canvas to fit any artist drawn past
the figure edge, and a figure submitted at 238 mm is scaled by 0.77 in production, dropping a
5 pt annotation to 4 pt. `scripts/_check_figure_widths.py [fig1 ...]` reports the saved size and
names the offending artists; keep every figure at or under 183 mm rather than letting the type
shrink.

Usage:  python scripts/make_main_figures.py [1 2 3 4 5]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

FIGURES: dict[str, tuple[str, str]] = {
    "1": ("plot_fig1_water_footprint.py", "water price of abatement"),
    "2": ("plot_fig2_constraint_response.py", "basin water ceiling on capture"),
    "3": ("plot_fig3_mechanism.py", "institution vs climate, and the adaptation channel"),
    "4": ("plot_fig4_network_reconfiguration.py", "no-regret pipeline core"),
    "5": ("plot_fig5_pathway_succession.py", "pathway succession and the adaptation channel"),
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    wanted = sys.argv[1:] or list(FIGURES)
    unknown = [key for key in wanted if key not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figures: {unknown}; choose from {list(FIGURES)}")

    failures: list[str] = []
    for key in wanted:
        script, description = FIGURES[key]
        print(f"\n{'=' * 78}\n  Fig {key} -- {description}\n{'=' * 78}", flush=True)
        started = time.time()
        completed = subprocess.run(
            [sys.executable, str(SCRIPTS / script)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        print(completed.stdout, end="")
        if completed.returncode != 0:
            print(completed.stderr, end="", file=sys.stderr)
            failures.append(f"Fig {key} ({script})")
            print(f"  FAILED after {time.time() - started:.0f}s", flush=True)
        else:
            print(f"  done in {time.time() - started:.0f}s", flush=True)

    if failures:
        print(f"\n{len(failures)} of {len(wanted)} figures failed: {', '.join(failures)}")
        return 1
    print(f"\nall {len(wanted)} figures regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
