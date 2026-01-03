"""History index entry schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .base import SerializableMixin


@dataclass
class HistoryEntry(SerializableMixin):
    """Entry in profiling history index."""

    run_id: str
    timestamp: str
    git_commit: str
    git_branch: str
    collectors_run: List[str] = field(default_factory=list)
    status: str = ""
    path: str = ""

