from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json

import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv_records(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_value(path: Path, value: Any) -> tuple[Path, ...]:
    if isinstance(value, pd.DataFrame):
        target = path if path.suffix else path.with_suffix(".csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        value.to_csv(target, index=False, encoding="utf-8-sig")
        return (target,)
    if is_dataclass(value):
        return _write_value(path.with_suffix(".json"), asdict(value))
    if isinstance(value, Mapping):
        target = path if path.suffix else path.with_suffix(".json")
        write_json(target, value)
        return (target,)
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, Mapping) for item in value):
        target = path if path.suffix else path.with_suffix(".csv")
        fieldnames = sorted({field for row in value for field in row.keys()})
        write_csv_records(target, value, fieldnames)
        return (target,)
    if isinstance(value, (str, int, float, bool)) or value is None:
        target = path if path.suffix else path.with_suffix(".txt")
        write_text(target, "" if value is None else str(value))
        return (target,)
    return _write_value(path.with_suffix(".txt"), str(value))


@dataclass(frozen=True)
class ScenarioRunRecord:
    experiment_id: str
    scenario_id: str
    label: str
    status: str
    scenario_dir: Path
    parameters: dict[str, Any]
    output_files: tuple[Path, ...] = ()
    error: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "label": self.label,
            "status": self.status,
            "scenario_dir": str(self.scenario_dir),
            "parameters_json": json.dumps(self.parameters, ensure_ascii=False, sort_keys=True, default=_json_default),
            "output_files": ";".join(str(path) for path in self.output_files),
            "error": self.error,
        }


@dataclass(frozen=True)
class ExperimentRunResult:
    experiment_id: str
    run_dir: Path
    manifest_path: Path
    scenarios: tuple[ScenarioRunRecord, ...]
    model_function: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "model_function": self.model_function,
            "scenario_count": len(self.scenarios),
        }


@dataclass(frozen=True)
class ResultStore:
    output_dir: Path

    def prepare_run_dir(self, experiment_id: str) -> Path:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.output_dir / experiment_id / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def prepare_scenario_dir(self, run_dir: Path, scenario_id: str) -> Path:
        scenario_dir = run_dir / "scenarios" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        return scenario_dir

    def write_manifest(
        self,
        run_dir: Path,
        definition: "ExperimentDefinition",
        model_function: str,
        scenarios: Sequence[ScenarioRunRecord],
    ) -> Path:
        manifest_path = run_dir / "experiment.json"
        payload = {
            "experiment": definition.to_record(),
            "model_function": model_function,
            "scenarios": [record.to_record() for record in scenarios],
        }
        write_json(manifest_path, payload)
        write_csv_records(
            run_dir / "scenarios.csv",
            [record.to_record() for record in scenarios],
            fieldnames=[
                "experiment_id",
                "scenario_id",
                "label",
                "status",
                "scenario_dir",
                "parameters_json",
                "output_files",
                "error",
            ],
        )
        return manifest_path

    def write_scenario_context(self, scenario_dir: Path, context: "ScenarioRunContext") -> Path:
        context_path = scenario_dir / "context.json"
        payload = {
            "experiment_id": context.experiment.experiment_id,
            "experiment_title": context.experiment.title,
            "stage": context.experiment.stage,
            "scenario": context.scenario.to_record(),
            "run_dir": str(context.run_dir),
            "scenario_dir": str(context.scenario_dir),
        }
        write_json(context_path, payload)
        return context_path

    def write_model_result(self, scenario_dir: Path, result: Any) -> tuple[Path, ...]:
        if isinstance(result, Mapping) and "artifacts" in result and isinstance(result["artifacts"], Mapping):
            payload = dict(result)
            artifacts = dict(payload.pop("artifacts"))
            output_files = list(_write_value(scenario_dir / "result", payload))
            artifacts_dir = scenario_dir / "artifacts"
            for artifact_name, artifact_value in artifacts.items():
                output_files.extend(_write_value(artifacts_dir / artifact_name, artifact_value))
            return tuple(output_files)
        return _write_value(scenario_dir / "result", result)
