from __future__ import annotations

import math

from coal_retrofit.optimization.emissions import reduction_fraction
from coal_retrofit.optimization.scenario import OptimizationAssumptions, PATHWAYS


def test_beccs_net_reduction_adds_capture_and_blend() -> None:
    # 物理捕集只有 eta = 0.9，净减排还含生物质替代与被捕集的生物源部分，合计 eta + beta。
    assert math.isclose(reduction_fraction("beccs", 0.9, 0.25), 1.15)


def test_unabated_reduces_nothing_and_retire_reduces_all() -> None:
    assert "unabated" in PATHWAYS
    assert reduction_fraction("unabated", 0.9, 0.5) == 0.0
    assert reduction_fraction("retire", 0.9, 0.5) == 1.0


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
