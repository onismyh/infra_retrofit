from __future__ import annotations

from coal_retrofit.optimization.scenario import OptimizationAssumptions, PATHWAYS


def test_pathway_and_retirement_cost_defaults_are_publication_ready() -> None:
    assumptions = OptimizationAssumptions()

    assert PATHWAYS[0] == "unabated"
    assert assumptions.fixed_cost_cny_per_mwh("unabated") == 0.0
    assert assumptions.fixed_cost_cny_per_mwh("retire") > 0.0
