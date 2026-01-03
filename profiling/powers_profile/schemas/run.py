"""Profiling run schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import SerializableMixin
from .git import GitInfo
from .results import CollectorResult
from .system import SystemInfo


@dataclass
class ProfilingRun(SerializableMixin):
    """Complete profiling session."""

    run_id: str
    timestamp: str
    system_info: SystemInfo
    git_info: GitInfo
    config: Dict[str, Any]
    binary_path: str
    binary_args: List[str]
    collectors_run: List[str]
    results: Dict[str, CollectorResult] = field(default_factory=dict)
    total_duration_seconds: float = 0.0
    status: str = "success"

