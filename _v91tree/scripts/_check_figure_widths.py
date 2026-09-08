"""Throwaway: find the artists that make each main figure save wider than it asks for.

`plot_style.save_fig` saves with bbox_inches='tight', which GROWS the saved canvas to include
anything drawn outside the figure rectangle. A figure that requests 183 mm and saves at 278 mm
has not been widened harmlessly: the publisher scales it back to the column, so every font is
divided by 1.52 and a 5 pt annotation prints at 3.3 pt, under Nature's 5 pt floor.

This locates the offenders instead of guessing: intercept each figure before it is saved,
compare the tight bbox with the requested size, and list the artists sticking out furthest.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MODULES = {
    "fig1": "plot_fig1_water_footprint",
    "fig2": "plot_fig2_constraint_response",
    "fig3": "plot_fig3_mechanism",
    "fig4": "plot_fig4_network_reconfiguration",
    "fig5": "plot_fig5_pathway_succession",
}
MM = 25.4


def describe(artist) -> str:
    text = getattr(artist, "get_text", None)
    if callable(text):
        raw = (text() or "").replace("\n", " ")[:58]
        if raw.strip():
            return f"{type(artist).__name__}('{raw}')"
    return type(artist).__name__


def diagnose(fig, label: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    req_w, req_h = fig.get_size_inches()
    tight = fig.get_tightbbox(renderer)
    print(f"\n=== {label} ===")
    print(f"  requested {req_w * MM:6.1f} x {req_h * MM:6.1f} mm")
    print(f"  saved     {tight.width * MM:6.1f} x {tight.height * MM:6.1f} mm"
          f"   (x {tight.x0 * MM:+.1f} .. {tight.x1 * MM:+.1f})")
    over_w = tight.width * MM - req_w * MM
    if over_w < 2.0:
        print("  width is clean")
        return
    print(f"  OVERFLOW {over_w:+.1f} mm wide -> print scale {req_w / tight.width:.3f}, "
          f"a 5.0 pt label prints at {5.0 * req_w / tight.width:.1f} pt")

    offenders = []
    for artist in fig.findobj():
        if artist is fig or not artist.get_visible():
            continue
        try:
            box = artist.get_window_extent(renderer)
        except Exception:
            continue
        if box.width <= 0 or box.height <= 0:
            continue
        left = -box.x0 / dpi
        right = (box.x1 / dpi) - req_w
        worst = max(left, right)
        if worst > 0.02:
            side = "right" if right >= left else "left"
            offenders.append((worst * MM, side, describe(artist)))

    offenders.sort(reverse=True)
    seen = set()
    shown = 0
    for amount, side, what in offenders:
        if what in seen:
            continue
        seen.add(what)
        print(f"    {amount:6.1f} mm past {side:5s}  {what}")
        shown += 1
        if shown >= 8:
            break


def main() -> None:
    wanted = sys.argv[1:] or list(MODULES)
    for label, modname in ((k, v) for k, v in MODULES.items() if k in wanted):
        module = importlib.import_module(modname)
        captured: dict[str, object] = {}

        def grab(fig, name, subdir="", _c=captured):
            _c["fig"] = fig

        module.save_fig = grab
        try:
            module.main()
        except Exception as exc:  # noqa: BLE001 - a broken figure must not stop the audit
            print(f"\n=== {label} ===\n  FAILED to build: {type(exc).__name__}: {exc}")
            continue
        fig = captured.get("fig")
        if fig is None:
            print(f"\n=== {label} ===\n  built but never called save_fig")
            continue
        diagnose(fig, label)
        plt.close(fig)


if __name__ == "__main__":
    main()
