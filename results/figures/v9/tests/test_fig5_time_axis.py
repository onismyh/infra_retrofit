from __future__ import annotations

import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parents[1]
REPO = Path(r"D:\000. Paper work\煤电改造claude")
SCRIPT = HERE / "plot_fig5_pathway_succession_work.py"


def load_fig5():
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    module = types.ModuleType("fig5_under_test")
    original = REPO / "scripts" / "plot_fig5_pathway_succession.py"
    module.__file__ = str(original)
    exec(compile(SCRIPT.read_text(encoding="utf-8"), str(original), "exec"),
         module.__dict__)
    return module


def test_pathway_succession_top_row_axis_starts_at_2025():
    fig5 = load_fig5()
    frame = pd.DataFrame(
        {
            "year": [2030, 2040, 2050, 2060],
            "unabated": [100.0, 80.0, 40.0, 0.0],
            "biomass": [0.0, 10.0, 20.0, 20.0],
            "ccs": [0.0, 0.0, 10.0, 10.0],
            "beccs": [0.0, 5.0, 15.0, 20.0],
            "ammonia": [0.0, 0.0, 0.0, 0.0],
            "retire": [0.0, 5.0, 15.0, 50.0],
        }
    )
    for key in fig5.STACK_ORDER:
        frame[f"{key}_gw"] = frame[key] * 12.6
    trajectories = {name: frame.copy() for name, _ in fig5.PANEL_A_RUNS}

    fig, axes = plt.subplots(1, 3)
    fig5.panel_a(axes, trajectories)

    for ax in axes:
        assert ax.get_xlim() == pytest.approx((2025.0, 2060.0))
        assert np.array_equal(ax.get_xticks(), [2025, 2030, 2040, 2050, 2060])
    plt.close(fig)


def test_pathway_succession_uses_capacity_and_adds_2025_baseline():
    """Removing the observed 2025 baseline must break the plotted pathway contract."""
    fig5 = load_fig5()
    frame = pd.DataFrame(
        {
            "year": [2030, 2040, 2050, 2060],
            "unabated": [90.0, 60.0, 30.0, 0.0],
            "biomass": [5.0, 15.0, 20.0, 20.0],
            "ccs": [0.0, 5.0, 10.0, 10.0],
            "beccs": [0.0, 10.0, 20.0, 20.0],
            "ammonia": [0.0, 0.0, 0.0, 0.0],
            "retire": [5.0, 10.0, 20.0, 50.0],
        }
    )
    for key in fig5.STACK_ORDER:
        frame[f"{key}_gw"] = frame[key] * 12.6

    plotted = fig5.with_2025_capacity_baseline(frame)

    assert plotted["year"].tolist() == [2025, 2030, 2040, 2050, 2060]
    assert plotted.loc[0, fig5.CAPACITY_STACK_ORDER].tolist() == [
        1260.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    trajectories = {name: frame.copy() for name, _ in fig5.PANEL_A_RUNS}
    fig, axes = plt.subplots(1, 3)
    fig5.panel_a(axes, trajectories)

    assert sum("1260 GW" in text.get_text() for text in axes[0].texts) == 1
    assert axes[0].get_ylabel() == "煤电装机容量（GW）"
    endpoint_label = next(text for text in axes[0].texts
                          if text.get_text() == "630 GW")
    assert endpoint_label.get_ha() == "right"
    assert all(
        min(path.vertices[:, 0].min() for collection in ax.collections
            for path in collection.get_paths()) == pytest.approx(2025.0)
        for ax in axes
    )
    plt.close(fig)


def test_trajectory_aggregates_pathways_by_installed_capacity(tmp_path, monkeypatch):
    """Generation weights must not replace capacity weights in panel a's GW data."""
    fig5 = load_fig5()
    rows = []
    for year in fig5.YEARS:
        rows.extend(
            [
                {
                    "year": year,
                    "capacity_mw": 100.0,
                    "annual_generation_mwh": 900.0,
                    "share_unabated": 1.0,
                    "share_biomass": 0.0,
                    "share_ccs": 0.0,
                    "share_beccs": 0.0,
                    "share_ammonia": 0.0,
                    "share_retire": 0.0,
                    "already_air_share": 0.0,
                    "air_cooled_share": 0.0,
                    "captured_mt": 0.0,
                    "water_use_m3": 0.0,
                },
                {
                    "year": year,
                    "capacity_mw": 300.0,
                    "annual_generation_mwh": 100.0,
                    "share_unabated": 0.0,
                    "share_biomass": 1.0,
                    "share_ccs": 0.0,
                    "share_beccs": 0.0,
                    "share_ammonia": 0.0,
                    "share_retire": 0.0,
                    "already_air_share": 0.0,
                    "air_cooled_share": 0.0,
                    "captured_mt": 0.0,
                    "water_use_m3": 0.0,
                },
            ]
        )
    detail = tmp_path / "plant_detail.csv"
    pd.DataFrame(rows).to_csv(detail, index=False)
    monkeypatch.setattr(fig5, "_require_current_vintage", lambda _: detail)
    monkeypatch.setattr(fig5, "_admit", lambda _: {})

    frame = fig5.trajectory("toy")

    assert frame.loc[0, "unabated_gw"] == pytest.approx(0.1)
    assert frame.loc[0, "biomass_gw"] == pytest.approx(0.3)
    assert frame.loc[0, fig5.CAPACITY_STACK_ORDER].sum() == pytest.approx(0.4)
