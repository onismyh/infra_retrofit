from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Sequence

import pandas as pd


SANITY_NAME_HINTS = ("sanity", "review", "audit", "check", "diagnostic", "verification")
SENSITIVITY_NAME_HINTS = ("sensitivity", "matrix", "tornado", "spider", "assumption", "scenario")
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml", ".ini"}
TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx"}
PASS_WORDS = ("pass", "passed", "ok", "okay", "true", "complete", "completed", "verified", "通过")
WARN_WORDS = ("warn", "warning", "issue", "needs", "todo", "fixme", "失败", "警告", "异常")
FAIL_WORDS = ("fail", "failed", "error", "infeasible", "invalid", "missing", "reject", "失败", "错误")


@dataclass(frozen=True)
class FileEntry:
    path: Path
    size_bytes: int
    suffix: str
    modified_at: datetime


@dataclass(frozen=True)
class SanityFinding:
    source: str
    status: str
    detail: str
    severity: str = "info"


@dataclass(frozen=True)
class SensitivityTable:
    source: str
    shape: tuple[int, int]
    summary: str
    preview: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ResultsSummary:
    root: Path
    files: list[FileEntry]
    directories: list[Path]
    extension_counts: dict[str, int]
    total_size_bytes: int
    newest_files: list[FileEntry]
    largest_files: list[FileEntry]
    sanity_findings: list[SanityFinding]
    sensitivity_tables: list[SensitivityTable]
    notes: list[str]


def _normalize_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "[no suffix]"


def _file_entry(path: Path) -> FileEntry:
    stat = path.stat()
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone()
    return FileEntry(
        path=path,
        size_bytes=stat.st_size,
        suffix=_normalize_suffix(path),
        modified_at=modified_at,
    )


def _discover_experiment_tags(root: Path) -> list[str]:
    tags: list[str] = []
    pattern = re.compile(r"EXP-[A-Z]\d+", re.IGNORECASE)
    for part in [root.name, *[parent.name for parent in root.parents]]:
        for match in pattern.findall(part):
            tag = match.upper()
            if tag not in tags:
                tags.append(tag)
    return tags


def _iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _iter_directories(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_dir())


def _load_text(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="ignore")[:limit]
    except OSError:
        return ""


def _load_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t")
        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path)
    except Exception:
        return None
    return None


def _stringify(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not headers:
        return ""
    escaped_headers = [str(header).replace("|", "\\|") for header in headers]
    rendered_rows = [[_stringify(cell).replace("|", "\\|") for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rendered_rows:
        padded = list(row)[: len(headers)]
        if len(padded) < len(headers):
            padded.extend([""] * (len(headers) - len(padded)))
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _is_text_like(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def _find_sanity_from_markdown(path: Path, text: str) -> list[SanityFinding]:
    findings: list[SanityFinding] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if not any(word in lower for word in (*PASS_WORDS, *WARN_WORDS, *FAIL_WORDS)):
            continue
        status = "info"
        severity = "info"
        if any(word in lower for word in FAIL_WORDS):
            status = "fail"
            severity = "warning"
        elif any(word in lower for word in WARN_WORDS):
            status = "warn"
            severity = "warning"
        elif any(word in lower for word in PASS_WORDS):
            status = "pass"
        findings.append(
            SanityFinding(
                source=path.name,
                status=status,
                detail=line[:220],
                severity=severity,
            )
        )
        if len(findings) >= 12:
            break
    return findings


def _find_sanity_from_table(path: Path, frame: pd.DataFrame) -> list[SanityFinding]:
    findings: list[SanityFinding] = []
    lower_cols = {str(col).lower(): col for col in frame.columns}
    status_col = next((lower_cols[key] for key in lower_cols if key in {"status", "result", "outcome", "check_status"}), None)
    passed_col = next((lower_cols[key] for key in lower_cols if key in {"passed", "pass", "ok", "valid"}), None)
    issue_col = next((lower_cols[key] for key in lower_cols if key in {"issue", "message", "detail", "note", "description"}), None)

    if status_col is not None:
        series = frame[status_col].astype(str).str.lower()
        counts = series.value_counts(dropna=False).to_dict()
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        findings.append(
            SanityFinding(
                source=path.name,
                status="summary",
                detail=f"Status counts: {summary}",
            )
        )
        failing = frame[series.str.contains("fail|error|warn|issue", regex=True, na=False)]
        for _, row in failing.head(5).iterrows():
            detail = issue_col and row.get(issue_col)
            if pd.isna(detail) or detail == "":
                detail = f"{status_col}={row.get(status_col)}"
            findings.append(
                SanityFinding(
                    source=path.name,
                    status=str(row.get(status_col)),
                    detail=_stringify(detail),
                    severity="warning",
                )
            )
        return findings

    if passed_col is not None:
        series = frame[passed_col]
        values = series.astype(str).str.lower()
        passed = values.isin({"true", "1", "yes", "y", "pass", "passed", "ok"})
        failures = int((~passed).sum())
        findings.append(
            SanityFinding(
                source=path.name,
                status="summary",
                detail=f"{passed.sum()} passed, {failures} not clearly passed",
                severity="warning" if failures else "info",
            )
        )
        if issue_col is not None:
            for _, row in frame.loc[~passed].head(5).iterrows():
                detail = row.get(issue_col)
                findings.append(
                    SanityFinding(
                        source=path.name,
                        status="not_passed",
                        detail=_stringify(detail) or f"{passed_col}={row.get(passed_col)}",
                        severity="warning",
                    )
                )
        return findings

    if issue_col is not None:
        non_empty = frame[frame[issue_col].notna() & (frame[issue_col].astype(str).str.strip() != "")]
        if not non_empty.empty:
            findings.append(
                SanityFinding(
                    source=path.name,
                    status="issues",
                    detail=f"{len(non_empty)} rows carry non-empty {issue_col} values",
                    severity="warning",
                )
            )
            for _, row in non_empty.head(5).iterrows():
                findings.append(
                    SanityFinding(
                        source=path.name,
                        status="issue",
                        detail=_stringify(row.get(issue_col)),
                        severity="warning",
                    )
                )
    return findings


def _summarize_table(path: Path, frame: pd.DataFrame) -> SensitivityTable | None:
    if frame.empty:
        return None

    lowered_name = path.stem.lower()
    lowered_cols = [str(col).lower() for col in frame.columns]
    has_hint = any(hint in lowered_name for hint in SENSITIVITY_NAME_HINTS) or any(
        any(hint in col for hint in SENSITIVITY_NAME_HINTS) for col in lowered_cols
    )
    if not has_hint and len(frame) > 20 and len(frame.columns) > 8:
        return None

    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    summary_parts = [f"shape={frame.shape[0]}x{frame.shape[1]}"]
    if numeric_cols:
        parts = []
        for col in numeric_cols[:4]:
            series = pd.to_numeric(frame[col], errors="coerce")
            parts.append(
                f"{col}: min={series.min():.6g}, max={series.max():.6g}"
                if series.notna().any()
                else f"{col}: non-numeric"
            )
        summary_parts.append("; ".join(parts))
    else:
        summary_parts.append("no numeric columns detected")

    preview_df = frame.head(8).copy()
    preview_rows = [
        {str(col): _stringify(row[col]) for col in preview_df.columns}
        for _, row in preview_df.iterrows()
    ]
    return SensitivityTable(
        source=path.name,
        shape=frame.shape,
        summary=" | ".join(summary_parts),
        preview=preview_rows,
    )


def _looks_like_sensitivity_table(path: Path, frame: pd.DataFrame) -> bool:
    lowered_name = path.stem.lower()
    if any(hint in lowered_name for hint in SENSITIVITY_NAME_HINTS):
        return True
    lowered_cols = [str(col).lower() for col in frame.columns]
    if any(any(hint in col for hint in SENSITIVITY_NAME_HINTS) for col in lowered_cols):
        return True
    if frame.shape[0] <= 30 and frame.shape[1] <= 12:
        return any(pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns)
    return False


def collect_results_summary(results_path: Path) -> ResultsSummary:
    root = results_path.expanduser().resolve()
    if root.is_file():
        root = root.parent
    if not root.exists():
        raise FileNotFoundError(f"Results path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {root}")

    files = [_file_entry(path) for path in _iter_files(root)]
    directories = _iter_directories(root)
    total_size_bytes = sum(file.size_bytes for file in files)

    extension_counts: dict[str, int] = {}
    for file in files:
        extension_counts[file.suffix] = extension_counts.get(file.suffix, 0) + 1

    newest_files = sorted(files, key=lambda entry: entry.modified_at, reverse=True)[:8]
    largest_files = sorted(files, key=lambda entry: entry.size_bytes, reverse=True)[:8]

    sanity_findings: list[SanityFinding] = []
    sensitivity_tables: list[SensitivityTable] = []

    for path in (entry.path for entry in files):
        lower_name = path.name.lower()
        lower_parent = path.parent.name.lower()
        matches_sanity = any(hint in lower_name or hint in lower_parent for hint in SANITY_NAME_HINTS)

        if _is_text_like(path) and matches_sanity:
            text = _load_text(path)
            sanity_findings.extend(_find_sanity_from_markdown(path, text))

        if path.suffix.lower() in TABLE_SUFFIXES:
            frame = _load_table(path)
            if frame is not None:
                if matches_sanity:
                    sanity_findings.extend(_find_sanity_from_table(path, frame))
                if _looks_like_sensitivity_table(path, frame):
                    table_summary = _summarize_table(path, frame)
                    if table_summary is not None:
                        sensitivity_tables.append(table_summary)

        if _is_text_like(path) and not matches_sanity:
            text = _load_text(path)
            if any(word in text.lower() for word in ("failed", "warning", "issue", "sanity", "check", "error")):
                sanity_findings.extend(_find_sanity_from_markdown(path, text))

    notes: list[str] = []
    experiment_tags = _discover_experiment_tags(root)
    if experiment_tags:
        notes.append(f"Detected experiment tags: {', '.join(experiment_tags)}")
    if not sanity_findings:
        notes.append("No dedicated sanity-check findings were detected in the directory scan.")
    if not sensitivity_tables:
        notes.append("No dedicated sensitivity matrix files were detected in the directory scan.")

    return ResultsSummary(
        root=root,
        files=files,
        directories=directories,
        extension_counts=dict(sorted(extension_counts.items(), key=lambda item: item[0])),
        total_size_bytes=total_size_bytes,
        newest_files=newest_files,
        largest_files=largest_files,
        sanity_findings=sanity_findings,
        sensitivity_tables=sensitivity_tables,
        notes=notes,
    )


def render_results_markdown(summary: ResultsSummary) -> str:
    lines: list[str] = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append(f"Source directory: `{summary.root.as_posix()}`")
    lines.append("")

    overview_rows = [
        ("Files", len(summary.files)),
        ("Directories", len(summary.directories)),
        ("Total size", _format_size(summary.total_size_bytes)),
        ("Newest file", summary.newest_files[0].path.name if summary.newest_files else "n/a"),
        ("Largest file", summary.largest_files[0].path.name if summary.largest_files else "n/a"),
    ]
    lines.append("## Overview")
    lines.append("")
    lines.append(_markdown_table(["Metric", "Value"], overview_rows))
    lines.append("")

    if summary.extension_counts:
        lines.append("## File Types")
        lines.append("")
        rows = [(suffix, count) for suffix, count in summary.extension_counts.items()]
        lines.append(_markdown_table(["Suffix", "Count"], rows))
        lines.append("")

    if summary.largest_files:
        lines.append("## Largest Files")
        lines.append("")
        rows = [
            (
                file.path.relative_to(summary.root).as_posix(),
                _format_size(file.size_bytes),
                file.modified_at.strftime("%Y-%m-%d %H:%M"),
            )
            for file in summary.largest_files
        ]
        lines.append(_markdown_table(["File", "Size", "Modified"], rows))
        lines.append("")

    if summary.sanity_findings:
        lines.append("## Sanity Checks")
        lines.append("")
        rows = [(finding.source, finding.status, finding.severity, finding.detail) for finding in summary.sanity_findings[:20]]
        lines.append(_markdown_table(["Source", "Status", "Severity", "Detail"], rows))
        if len(summary.sanity_findings) > 20:
            lines.append("")
            lines.append(f"_Additional findings omitted: {len(summary.sanity_findings) - 20}_")
        lines.append("")
    else:
        lines.append("## Sanity Checks")
        lines.append("")
        lines.append("No dedicated sanity-check findings were detected.")
        lines.append("")

    if summary.sensitivity_tables:
        lines.append("## Assumption Sensitivity")
        lines.append("")
        for table in summary.sensitivity_tables:
            lines.append(f"### {table.source}")
            lines.append("")
            lines.append(table.summary)
            lines.append("")
            if table.preview:
                preview_headers = list(table.preview[0].keys())
                preview_rows = [[row.get(header, "") for header in preview_headers] for row in table.preview]
                lines.append(_markdown_table(preview_headers, preview_rows))
                lines.append("")
    else:
        lines.append("## Assumption Sensitivity")
        lines.append("")
        lines.append("No dedicated sensitivity matrix files were detected.")
        lines.append("")

    if summary.newest_files:
        lines.append("## Recent Files")
        lines.append("")
        rows = [
            (
                file.path.relative_to(summary.root).as_posix(),
                file.modified_at.strftime("%Y-%m-%d %H:%M"),
                _format_size(file.size_bytes),
            )
            for file in summary.newest_files
        ]
        lines.append(_markdown_table(["File", "Modified", "Size"], rows))
        lines.append("")

    if summary.notes:
        lines.append("## Notes")
        lines.append("")
        for note in summary.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def summarize_results_directory(results_path: Path | str) -> str:
    summary = collect_results_summary(Path(results_path))
    return render_results_markdown(summary)


def write_results_markdown(results_path: Path | str, output_path: Path | str) -> Path:
    markdown = summarize_results_directory(results_path)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return destination
