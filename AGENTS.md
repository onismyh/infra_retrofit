# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11+ `src`-layout project. Reusable code lives in `src/coal_retrofit/`: `builders/` prepares model inputs, `optimization/` contains the Gurobi model, `experiments/` defines and runs scenarios, and `reporting/` summarizes results. Keep `scripts/` as thin command-line orchestration. Tests belong in `tests/`. Treat `data/` as raw source material, `inputs/` as normalized model inputs, and `results/` or `outputs/` as generated artifacts. Research notes and reference material live in `plan/`, root-level Markdown files, and `reference/`.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
python -m pip install -e . pytest
python -m pytest
python scripts/validate_inputs.py
python scripts/run_phase_a.py
python scripts/run_experiment.py --experiment-id EXP-B1 --output-dir results/baseline
python scripts/summarize_results.py results/baseline/EXP-B1/<run_id> -o results/reviews/summary.md
```

The first command installs the package in editable mode with the test runner. Phase A regenerates standardized inputs from raw data. Experiment runs require a working Gurobi installation and license.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Add explicit type hints to functions and prefer immutable `@dataclass(frozen=True)` value objects. Keep reusable logic in the package rather than scripts, avoid mutable defaults and bare `except`, and use logging instead of debug `print()` calls. No repository-wide formatter or linter is currently configured.

## Testing Guidelines

Use pytest. Name files `test_<topic>.py` and tests `test_<behavior>()`. Add focused regression tests for changes to emissions accounting, scenario assumptions, constraints, or result schemas. Run `python -m pytest` before every pull request. No coverage threshold is configured, so prioritize meaningful boundary and numerical-invariant checks.

## Commit & Pull Request Guidelines

Root Git history is unavailable in this checkout; use Conventional Commits consistently, for example `fix(optimization): correct BECCS residual emissions`. Keep commits focused. Pull requests should explain the research or engineering motivation, changed assumptions or datasets, validation commands, and any result-schema impact. Link relevant issues and include before/after figures for visualization changes.

## Security & Data Hygiene

Never commit Gurobi licenses, API keys, credentials, or local environment files. Avoid silently replacing raw datasets; document provenance and regeneration steps. Inspect staged files carefully because the current `.gitignore` is minimal.
