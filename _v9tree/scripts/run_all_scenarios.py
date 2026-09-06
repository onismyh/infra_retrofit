"""Run scenario batches through the canonical run_single.py entrypoint.

Examples:
    python scripts/run_all_scenarios.py --dry-run
    python scripts/run_all_scenarios.py --group all --threads 8 --time-limit 36000 --clean
    python scripts/run_all_scenarios.py --group core --clean --clean-core-only
    python scripts/run_all_scenarios.py --scenarios BASE WA_grid_200km --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from _bootstrap import ROOT
from run_single import EXPERIMENTS


RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"

CORE_SCENARIOS = (
    "BASE",
    "BASE_zero",
    "BASE_neg",
)
PATHWAY_SCENARIOS = (
    "RQ3_no_ammonia",
    "RQ3_no_biomass",
    "RQ3_no_ccs",
    "RQ3_retire_only",
    "RQ3_ccs_only",
)
WATER_SCENARIOS = ("WA_grid_200km",)
SENSITIVITY_SCENARIOS = tuple(name for name in EXPERIMENTS if name.startswith("SA_"))


@dataclass(frozen=True)
class ScenarioRunRecord:
    name: str
    returncode: int
    started_at: str
    finished_at: str
    elapsed_seconds: float
    status: str
    objective_cny: float | None
    mip_gap: float | None
    log_file: str
    note: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _scenario_names(group: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        names = explicit
    elif group == "core":
        names = list(CORE_SCENARIOS)
    elif group == "pathway":
        names = list(PATHWAY_SCENARIOS)
    elif group == "water":
        names = list(WATER_SCENARIOS)
    elif group == "sensitivity":
        names = list(SENSITIVITY_SCENARIOS)
    elif group == "main":
        names = list(CORE_SCENARIOS + PATHWAY_SCENARIOS + WATER_SCENARIOS)
    else:
        names = list(EXPERIMENTS)

    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        known = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown scenario(s): {unknown}. Known scenarios: {known}")
    return names


def _archive_existing(name: str, archive_dir: Path) -> None:
    json_path = RESULTS_DIR / f"{name}.json"
    scenario_dir = RESULTS_DIR / name
    archive_dir.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        shutil.copy2(json_path, archive_dir / f"{name}.json")
    if scenario_dir.exists():
        target = archive_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(scenario_dir, target)


def _read_result(name: str) -> tuple[str, float | None, float | None]:
    json_path = RESULTS_DIR / f"{name}.json"
    if not json_path.exists():
        return "missing_json", None, None

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid_json", None, None

    statuses = {
        str(year_data.get("status", "unknown"))
        for year_data in data.get("years", {}).values()
        if isinstance(year_data, dict)
    }
    status = ",".join(sorted(statuses)) if statuses else str(data.get("status", "unknown"))
    objective = data.get("global_objective_cny")
    quality = data.get("solver_quality", {})
    mip_gap = quality.get("mip_gap") if isinstance(quality, dict) else None
    return status, _as_float_or_none(objective), _as_float_or_none(mip_gap)


def _as_float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _run_command(cmd: list[str], log_file: Path, dry_run: bool) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("  [DRY-RUN] " + " ".join(cmd))
        return 0

    with log_file.open("w", encoding="utf-8") as fh:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            fh.write(line)
        return process.wait()


def _write_summary(records: Iterable[ScenarioRunRecord], summary_dir: Path) -> None:
    rows = [record.__dict__ for record in records]
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "summary.json"
    csv_path = summary_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ScenarioRunRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def _run_clean(clean_core_only: bool, dry_run: bool) -> int:
    cmd = [sys.executable, str(SCRIPTS_DIR / "clean_results.py")]
    if clean_core_only:
        cmd.append("--core-only")
    log_file = RESULTS_DIR / "run_logs" / f"clean_results_{_timestamp()}.log"
    print("\n=== Cleaning consolidated results ===")
    return _run_command(cmd, log_file, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run coal retrofit scenario batches.")
    parser.add_argument(
        "--group",
        choices=("all", "main", "core", "pathway", "water", "sensitivity"),
        default="all",
        help="Scenario group to run. Default: all scenarios in run_single.EXPERIMENTS.",
    )
    parser.add_argument("--scenarios", nargs="+", help="Explicit scenario names. Overrides --group.")
    parser.add_argument("--threads", type=int, default=0, help="Gurobi threads per scenario. Default: run_single default.")
    parser.add_argument("--time-limit", type=int, default=36000, help="Gurobi time limit per scenario in seconds.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenarios with an existing results/<name>.json.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop the batch after the first failed scenario.")
    parser.add_argument("--no-archive", action="store_true", help="Do not copy existing JSON/CSV outputs before rerunning.")
    parser.add_argument("--clean", action="store_true", help="Run scripts/clean_results.py after the batch.")
    parser.add_argument("--clean-core-only", action="store_true", help="Pass --core-only to clean_results.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Gurobi.")
    args = parser.parse_args()

    names = _scenario_names(args.group, args.scenarios)
    run_id = _timestamp()
    batch_dir = RESULTS_DIR / "run_logs" / f"batch_{run_id}"
    archive_dir = RESULTS_DIR / "_archives" / f"batch_{run_id}"

    print(f"Scenario batch: {args.group}")
    print(f"Run ID: {run_id}")
    print(f"Total scenarios: {len(names)}")
    print("Scenarios: " + ", ".join(names))
    print(f"Logs: {batch_dir}")
    if not args.no_archive:
        print(f"Archive: {archive_dir}")

    records: list[ScenarioRunRecord] = []
    for index, name in enumerate(names, 1):
        result_json = RESULTS_DIR / f"{name}.json"
        if args.skip_existing and result_json.exists():
            status, objective, mip_gap = _read_result(name)
            records.append(
                ScenarioRunRecord(
                    name=name,
                    returncode=0,
                    started_at="",
                    finished_at="",
                    elapsed_seconds=0.0,
                    status=status,
                    objective_cny=objective,
                    mip_gap=mip_gap,
                    log_file="",
                    note="skipped_existing",
                )
            )
            print(f"\n[{index}/{len(names)}] {name}: skipped existing result")
            continue

        if not args.no_archive and not args.dry_run:
            _archive_existing(name, archive_dir)

        cmd = [sys.executable, str(SCRIPTS_DIR / "run_single.py"), name]
        if args.threads > 0:
            cmd.extend(["--threads", str(args.threads)])
        if args.time_limit != 36000:
            cmd.extend(["--time-limit", str(args.time_limit)])

        print(f"\n{'=' * 72}")
        print(f"[{index}/{len(names)}] {name}: START")
        print(f"{'=' * 72}")

        started_at = datetime.now().isoformat(timespec="seconds")
        start = time.time()
        log_file = batch_dir / f"{index:02d}_{name}.log"
        returncode = _run_command(cmd, log_file, args.dry_run)
        elapsed = time.time() - start
        finished_at = datetime.now().isoformat(timespec="seconds")
        status, objective, mip_gap = _read_result(name) if returncode == 0 and not args.dry_run else ("dry_run", None, None)
        note = "ok" if returncode == 0 else "failed"

        records.append(
            ScenarioRunRecord(
                name=name,
                returncode=returncode,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_seconds=round(elapsed, 1),
                status=status,
                objective_cny=objective,
                mip_gap=mip_gap,
                log_file=str(log_file),
                note=note,
            )
        )
        print(f"[{index}/{len(names)}] {name}: rc={returncode} status={status} time={elapsed:.0f}s")

        if returncode != 0 and args.stop_on_failure:
            print("Stopping after first failure because --stop-on-failure was set.")
            break

    _write_summary(records, batch_dir)

    if args.clean:
        clean_rc = _run_clean(args.clean_core_only, args.dry_run)
        print(f"clean_results.py return code: {clean_rc}")

    ok = sum(1 for record in records if record.returncode == 0)
    failed = [record for record in records if record.returncode != 0]
    print(f"\nBatch complete: {ok}/{len(records)} successful, {len(failed)} failed.")
    if failed:
        for record in failed:
            print(f"  FAILED {record.name}: see {record.log_file}")
    print(f"Summary written to {batch_dir}")


if __name__ == "__main__":
    main()
