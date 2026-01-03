"""Git repository state detection."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from ..schemas.git import GitInfo


def _run_git(args: list[str], cwd: Optional[Path]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return ""
    return ""


def detect_git_info(repo_path: Optional[Path] = None) -> GitInfo:
    """Detect git repository state."""
    cwd = repo_path if repo_path is not None else Path(".")

    commit_sha = _run_git(["rev-parse", "HEAD"], cwd) or "unknown"
    commit_short = _run_git(["rev-parse", "--short", "HEAD"], cwd) or "unknown"

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if branch == "HEAD":
        branch = "detached"
    if not branch:
        branch = "unknown"

    is_dirty = bool(_run_git(["status", "--porcelain"], cwd))
    commit_date = _run_git(["show", "-s", "--format=%cI", "HEAD"], cwd)
    commit_message = _run_git(["show", "-s", "--format=%s", "HEAD"], cwd)

    tags_output = _run_git(["tag", "--points-at", "HEAD"], cwd)
    tags = [t for t in tags_output.split("\n") if t] if tags_output else []

    return GitInfo(
        commit_sha=commit_sha,
        commit_short=commit_short,
        branch=branch,
        is_dirty=is_dirty,
        commit_date=commit_date,
        commit_message=commit_message,
        tags=tags,
    )

