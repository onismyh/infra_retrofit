from __future__ import annotations

import pytest

from coal_retrofit.experiments.registry import ExperimentDefinition
from coal_retrofit.experiments.scenario import ScenarioParameters, ScenarioRunContext
from coal_retrofit.optimization.model import _map_context_to_scenario


def _context(tmp_path, parameters: dict) -> ScenarioRunContext:
    definition = ExperimentDefinition(
        experiment_id="EXP-T", title="t", stage="s", description="d", scenarios=()
    )
    scenario = ScenarioParameters(
        experiment_id="EXP-T", scenario_id="sc", label="label", parameters=parameters
    )
    return ScenarioRunContext(
        experiment=definition, scenario=scenario, run_dir=tmp_path, scenario_dir=tmp_path
    )


def test_known_fields_and_aliases_are_mapped(tmp_path) -> None:
    mapped = _map_context_to_scenario(
        _context(
            tmp_path,
            {
                "year": 2050,
                "mip_gap": 0.02,
                "pathway_disable": ["ammonia"],
                "emission_target_fraction": [0.5],
                "forced_pathways": ["CCS "],
                "min_path_share": 0.1,
            },
        )
    )
    assert mapped.planning_years == (2050,)
    assert mapped.mip_gap == 0.02
    assert mapped.pathway_disable == ("ammonia",)
    assert mapped.emission_target_fraction == (0.5,)
    assert mapped.forced_pathways == ("ccs",)
    assert mapped.min_forced_path_share == 0.1


def test_unknown_parameter_raises_instead_of_silent_drop(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown scenario parameter"):
        _map_context_to_scenario(_context(tmp_path, {"mip_gapp": 0.02}))


def test_managed_field_passed_directly_raises(tmp_path) -> None:
    # planning_years must be set via the 'year' alias, not passed as a raw field.
    with pytest.raises(ValueError, match="Unknown scenario parameter"):
        _map_context_to_scenario(_context(tmp_path, {"planning_years": [2050]}))


def test_force_all_pathways_takes_precedence(tmp_path) -> None:
    mapped = _map_context_to_scenario(
        _context(tmp_path, {"force_all_pathways": True, "forced_pathways": ["ccs"]})
    )
    assert set(mapped.forced_pathways) == {
        "unabated", "retire", "ccs", "biomass", "beccs", "ammonia"
    }
