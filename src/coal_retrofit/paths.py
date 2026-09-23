from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def find_data_file(self, pattern: str) -> Path:
        matches = sorted(self.data_dir.rglob(pattern))
        if not matches:
            raise FileNotFoundError(f"Could not find {pattern} under {self.data_dir}")
        return matches[0]

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def ensure_inputs_dir(self) -> None:
        self.inputs_dir.mkdir(parents=True, exist_ok=True)


def discover_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).absolute()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate
    raise FileNotFoundError("Could not discover project root containing data/ and scripts/")
