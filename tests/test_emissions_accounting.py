from __future__ import annotations

import math

from coal_retrofit.optimization.emissions import (
    captured_fraction,
    reduction_fraction,
    residual_fraction,
)
from coal_retrofit.optimization.scenario import OptimizationAssumptions, PATHWAYS


def test_beccs_capture_and_net_reduction_are_separate() -> None:
    capture_rate = 0.9
    biomass_blend = 0.25

    assert math.isclose(captured_fraction("beccs", capture_rate, biomass_blend), 0.9)
    assert math.isclose(reduction_fraction("beccs", capture_rate, biomass_blend), 1.15)
    assert math.isclose(residual_fraction("beccs", capture_rate, biomass_blend), -0.15)


def test_unabated_pathway_has_no_reduction_or_capture() -> None:
    assert "unabated" in PATHWAYS
    assert captured_fraction("unabated", 0.9, 0.5) == 0.0
    assert reduction_fraction("unabated", 0.9, 0.5) == 0.0
    assert residual_fraction("unabated", 0.9, 0.5) == 1.0


def test_fuel_substitution_pathways_use_their_own_blend_ratios() -> None:
    assert math.isclose(reduction_fraction("biomass", 0.9, biomass_blend=0.25), 0.25)
    assert math.isclose(
        reduction_fraction("ammonia", 0.9, biomass_blend=0.25, ammonia_blend=0.2),
        0.2,
    )


def test_pathway_and_retirement_cost_defaults_are_publication_ready() -> None:
    assumptions = OptimizationAssumptions()

    assert PATHWAYS[0] == "unabated"
    assert assumptions.fixed_cost_cny_per_mwh("unabated") == 0.0
    assert assumptions.fixed_cost_cny_per_mwh("retire") > 0.0
