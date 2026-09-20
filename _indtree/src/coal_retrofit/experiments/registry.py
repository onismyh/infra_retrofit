from __future__ import annotations

from dataclasses import dataclass, field

from .scenario import ScenarioParameters


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    title: str
    stage: str
    description: str
    scenarios: tuple[ScenarioParameters, ...]
    notes: str = ""

    def to_record(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "stage": self.stage,
            "description": self.description,
            "notes": self.notes,
            "scenario_count": len(self.scenarios),
        }


@dataclass
class ExperimentRegistry:
    _definitions: dict[str, ExperimentDefinition] = field(default_factory=dict)

    def register(self, definition: ExperimentDefinition) -> None:
        if definition.experiment_id in self._definitions:
            raise ValueError(f"Duplicate experiment id: {definition.experiment_id}")
        self._definitions[definition.experiment_id] = definition

    def get(self, experiment_id: str) -> ExperimentDefinition | None:
        return self._definitions.get(experiment_id)

    def require(self, experiment_id: str) -> ExperimentDefinition:
        definition = self.get(experiment_id)
        if definition is None:
            available = ", ".join(self.experiment_ids())
            raise KeyError(f"Unknown experiment id {experiment_id!r}. Available: {available}")
        return definition

    def items(self) -> tuple[tuple[str, ExperimentDefinition], ...]:
        return tuple(sorted(self._definitions.items(), key=lambda item: item[0]))

    def experiment_ids(self) -> tuple[str, ...]:
        return tuple(experiment_id for experiment_id, _ in self.items())
