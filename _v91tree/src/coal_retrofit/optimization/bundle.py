from __future__ import annotations

from pathlib import Path

from ..experiments.runner import run_experiment
from ..paths import ProjectPaths
from .model import run_context_model


def run_experiment_bundle(paths: ProjectPaths, experiment_id: str, output_dir: Path):
    def _model(context):
        return run_context_model(paths, context)

    return run_experiment(experiment_id, Path(output_dir), model_function=_model)
