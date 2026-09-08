from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

from .catalog import build_registry
from .registry import ExperimentRegistry
from .results import ExperimentRunResult, ResultStore, ScenarioRunRecord
from .scenario import ScenarioRunContext
from ..paths import ProjectPaths


ModelFunction = Callable[[ScenarioRunContext], Any]
BundleRunner = Callable[[ProjectPaths, str, Path], ExperimentRunResult]

DEFAULT_BUNDLE_RUNNER_SPEC = "coal_retrofit.optimization:run_experiment_bundle"


def load_model_function(spec: str) -> ModelFunction:
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ValueError("Model function must be in the form 'module:function'")
    module = import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{spec!r} does not resolve to a callable")
    return function


def _model_function_name(model_function: ModelFunction | None) -> str:
    if model_function is None:
        return ""
    module_name = getattr(model_function, "__module__", "")
    function_name = getattr(model_function, "__name__", model_function.__class__.__name__)
    return f"{module_name}:{function_name}" if module_name else function_name


def load_bundle_runner(spec: str = DEFAULT_BUNDLE_RUNNER_SPEC) -> BundleRunner:
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ValueError("Bundle runner must be in the form 'module:function'")
    module = import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{spec!r} does not resolve to a callable")
    return function


def _local_bundle_runner(
    paths: ProjectPaths,
    experiment_id: str,
    output_dir: Path,
    *,
    registry: ExperimentRegistry | None = None,
) -> ExperimentRunResult:
    _ = paths
    return run_experiment(
        experiment_id,
        output_dir,
        model_function=None,
        registry=registry,
    )


def run_experiment(
    experiment_id: str,
    output_dir: Path,
    *,
    model_function: ModelFunction | None = None,
    registry: ExperimentRegistry | None = None,
) -> ExperimentRunResult:
    registry = registry or build_registry()
    definition = registry.require(experiment_id)
    store = ResultStore(Path(output_dir))
    run_dir = store.prepare_run_dir(definition.experiment_id)
    scenario_records: list[ScenarioRunRecord] = []
    model_function_name = _model_function_name(model_function)

    for scenario in definition.scenarios:
        scenario_dir = store.prepare_scenario_dir(run_dir, scenario.scenario_id)
        context = ScenarioRunContext(
            experiment=definition,
            scenario=scenario,
            run_dir=run_dir,
            scenario_dir=scenario_dir,
        )
        store.write_scenario_context(scenario_dir, context)

        if model_function is None:
            result_payload: Any = {
                "status": "no_model_function",
                "experiment_id": definition.experiment_id,
                "scenario_id": scenario.scenario_id,
                "label": scenario.label,
                "parameters": scenario.parameters,
            }
        else:
            result_payload = model_function(context)

        try:
            output_files = store.write_model_result(scenario_dir, result_payload)
            scenario_records.append(
                ScenarioRunRecord(
                    experiment_id=definition.experiment_id,
                    scenario_id=scenario.scenario_id,
                    label=scenario.label,
                    status="ok",
                    scenario_dir=scenario_dir,
                    parameters=dict(scenario.parameters),
                    output_files=output_files,
                )
            )
        except Exception as exc:  # noqa: BLE001
            error_path = scenario_dir / "error.txt"
            error_path.write_text(str(exc), encoding="utf-8")
            scenario_records.append(
                ScenarioRunRecord(
                    experiment_id=definition.experiment_id,
                    scenario_id=scenario.scenario_id,
                    label=scenario.label,
                    status="error",
                    scenario_dir=scenario_dir,
                    parameters=dict(scenario.parameters),
                    output_files=(error_path,),
                    error=str(exc),
                )
            )
            manifest_path = store.write_manifest(run_dir, definition, model_function_name, scenario_records)
            raise

    manifest_path = store.write_manifest(run_dir, definition, model_function_name, scenario_records)
    return ExperimentRunResult(
        experiment_id=definition.experiment_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        scenarios=tuple(scenario_records),
        model_function=model_function_name,
    )


def run_experiment_bundle(
    paths: ProjectPaths,
    experiment_id: str,
    output_dir: Path,
    *,
    bundle_runner: BundleRunner | None = None,
    registry: ExperimentRegistry | None = None,
) -> ExperimentRunResult:
    runner = bundle_runner
    if runner is None:
        try:
            runner = load_bundle_runner()
        except (ModuleNotFoundError, AttributeError, ValueError, TypeError):
            runner = lambda resolved_paths, resolved_experiment_id, resolved_output_dir: _local_bundle_runner(
                resolved_paths,
                resolved_experiment_id,
                resolved_output_dir,
                registry=registry,
            )
    return runner(paths, experiment_id, Path(output_dir))
