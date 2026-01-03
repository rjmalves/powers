"""Utilities for executing collectors and persisting profiling runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .schemas import CollectorResult, GitInfo, HistoryEntry, ProfilingRun

RUNS_DIRNAME = "runs"
HISTORY_FILENAME = "history.json"


def generate_run_id(git_info: GitInfo) -> str:
    """Generate a deterministic run identifier using timestamp and git short SHA."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = git_info.commit_short or "unknown"
    return f"{timestamp}-{suffix}"


def status_from_results(results: Dict[str, CollectorResult]) -> str:
    """Aggregate collector statuses into a run status."""
    if not results:
        return "failed"
    successes = sum(1 for result in results.values() if result.success)
    if successes == len(results):
        return "success"
    if successes == 0:
        return "failed"
    return "partial"


def ensure_output_dirs(base_dir: Path, run_id: str) -> Path:
    """Create base output directory and run subdirectory."""
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / RUNS_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run(run: ProfilingRun, output_dir: Path) -> Path:
    """Persist a run to disk."""
    run_dir = output_dir / RUNS_DIRNAME / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / "run.json"
    run_path.write_text(run.to_json())
    return run_path


def _history_path(output_dir: Path) -> Path:
    return output_dir / HISTORY_FILENAME


def load_history(output_dir: Path) -> List[HistoryEntry]:
    """Load history entries from disk."""
    path = _history_path(output_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [HistoryEntry.from_dict(entry) for entry in payload]


def write_history(output_dir: Path, entries: Iterable[HistoryEntry]) -> Path:
    """Write history entries back to disk."""
    path = _history_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [entry.to_dict() for entry in entries]
    path.write_text(json.dumps(serialized, indent=2))
    return path


def append_history(output_dir: Path, entry: HistoryEntry) -> Path:
    """Append an entry to history, keeping most recent first."""
    entries = load_history(output_dir)
    entries.insert(0, entry)
    return write_history(output_dir, entries)


def latest_history_entry(output_dir: Path) -> Optional[HistoryEntry]:
    """Return the newest history entry if available."""
    entries = load_history(output_dir)
    return entries[0] if entries else None


def find_run_path(output_dir: Path, run_id: str) -> Path:
    """Resolve run.json path for a given run id."""
    candidate = output_dir / RUNS_DIRNAME / run_id / "run.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Run {run_id} not found under {output_dir}")
    return candidate


def load_run(path: Path) -> ProfilingRun:
    """Load a ProfilingRun from disk."""
    return ProfilingRun.from_json(path.read_text())


def history_entry_from_run(run: ProfilingRun, run_path: Path) -> HistoryEntry:
    """Create a history entry based on run metadata."""
    return HistoryEntry(
        run_id=run.run_id,
        timestamp=run.timestamp,
        git_commit=run.git_info.commit_sha,
        git_branch=run.git_info.branch,
        collectors_run=run.collectors_run,
        status=run.status,
        path=str(run_path),
    )

