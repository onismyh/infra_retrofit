# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11+ `src`-layout project. Reusable code lives in `src/coal_retrofit/`: `builders/` prepares model inputs, `optimization/` contains the Gurobi model and its result tables, and `runner.py` solves a registered scenario and writes its results. Scenarios are registered in `scenarios/*.toml` (parsed by `scenarios.py`); each names its solve tree, e.g. `tree = "_indtree"` reads `_indtree/inputs/` and writes `_indtree/results/`. The entry point is `python -m coal_retrofit` (`list`, `show`, `diff`, `run`); `scripts/run_single.py` and its copy in `_indtree/scripts/` are thin wrappers kept for the old commands. Keep `scripts/` as thin command-line orchestration. Tests belong in `tests/`. Treat `data/` as raw source material, `inputs/` as normalized model inputs, and `results/` or `outputs/` as generated artifacts. Research notes and reference material live in `plan/`, root-level Markdown files, and `reference/`.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
python -m pip install -e . pytest
python -m pytest
python -m coal_retrofit list
```

The first command installs the package in editable mode with the test runner; `python -m coal_retrofit` needs this install, because it reads the registry from `scenarios/` in the checkout the package is installed from. The `scripts/build_*.py` scripts regenerate standardized inputs from raw data. Solves require a working Gurobi installation and license; `ST_` scenarios are solved in `_indtree/` with 8 threads and the LP-relaxation warm start described in `_indtree/README.md` (see `.claude/CLAUDE.md` §二), which `run` performs itself for scenarios with `warm_start = "lp_relax"` (all `ST_` scenarios). `run` refuses to overwrite an existing result unless `--force` is given, and exits with 3 when there is no usable solution. Before differencing the results of two scenarios, `python -m coal_retrofit diff A B` lists every parameter in which they differ; `python _indtree/scripts/check_run_provenance.py --pair A B` does the same for two saved results; it fails if either result is missing, was saved before 2026-09-27 (no `resolved` section) or before 2026-09-28 (no `resolved.code` commit record) or has no usable solution (NaN objective), if one side was an LP relaxation or warm-started and the other was not, if both sides are LP relaxations, or if the carbon prices or the digests of the same input file differ; differing commits or uncommitted changes only warn. Each copy of that script reads its own tree's `results/` (the root copy reads the root `results/`); with `--pair`, `--results DIR` points it elsewhere.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Add explicit type hints to functions and prefer immutable `@dataclass(frozen=True)` value objects. Keep reusable logic in the package rather than scripts, avoid mutable defaults and bare `except`, and use logging instead of debug `print()` calls. No repository-wide formatter or linter is currently configured.

## Testing Guidelines

Use pytest. Name files `test_<topic>.py` and tests `test_<behavior>()`. Add focused regression tests for changes to emissions accounting, scenario assumptions, constraints, or result schemas. Run `python -m pytest` before every pull request. No coverage threshold is configured, so prioritize meaningful boundary and numerical-invariant checks.

## Commit & Pull Request Guidelines

Use Conventional Commits consistently, for example `fix(optimization): correct BECCS residual emissions`. Keep commits focused. Pull requests should explain the research or engineering motivation, changed assumptions or datasets, validation commands, and any result-schema impact. Link relevant issues and include before/after figures for visualization changes.

## Security & Data Hygiene

Never commit Gurobi licenses, API keys, credentials, or local environment files. Avoid silently replacing raw datasets; document provenance and regeneration steps. Inspect staged files carefully because the current `.gitignore` is minimal.
