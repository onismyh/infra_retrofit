from __future__ import annotations

from .registry import ExperimentDefinition, ExperimentRegistry
from .scenario import ScenarioParameters


def _case(
    experiment_id: str,
    scenario_id: str,
    label: str,
    *,
    parameters: dict | None = None,
    notes: str = "",
    tags: tuple[str, ...] = (),
) -> ScenarioParameters:
    return ScenarioParameters(
        experiment_id=experiment_id,
        scenario_id=scenario_id,
        label=label,
        parameters=dict(parameters or {}),
        notes=notes,
        tags=tags,
    )


def _build_base() -> ExperimentDefinition:
    scenarios = (
        _case(
            "BASE",
            "all-pathways",
            "All pathways open (unabated, retire, CCS, biomass, BECCS, ammonia)",
            notes="Base scenario with all retrofit pathways enabled and default parameters.",
            tags=("baseline",),
        ),
    )
    return ExperimentDefinition(
        experiment_id="BASE",
        title="Base scenario — all pathways open",
        stage="baseline",
        description="Multi-period joint optimization with all six pathways enabled under default cost and constraint assumptions.",
        scenarios=scenarios,
    )


def build_registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    registry.register(_build_base())
    return registry


def get_available_experiment_ids() -> tuple[str, ...]:
    return build_registry().experiment_ids()
