from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .registry import ExperimentDefinition


@dataclass(frozen=True)
class ScenarioParameters:
    experiment_id: str
    scenario_id: str
    label: str
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "label": self.label,
            "notes": self.notes,
            "tags": list(self.tags),
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ScenarioRunContext:
    experiment: "ExperimentDefinition"
    scenario: ScenarioParameters
    run_dir: Path
    scenario_dir: Path
