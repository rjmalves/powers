"""Comparison schemas for profiling runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .base import SerializableMixin
from .git import GitInfo


@dataclass
class MetricDelta(SerializableMixin):
    """Change in a single metric."""

    metric_name: str
    baseline_value: float
    target_value: float
    absolute_delta: float
    percent_delta: float
    improved: bool
    significant: bool


@dataclass
class Comparison(SerializableMixin):
    """Comparison between two profiling runs."""

    baseline_run_id: str
    target_run_id: str
    baseline_git: GitInfo
    target_git: GitInfo
    timestamp: str
    deltas: Dict[str, List[MetricDelta]] = field(default_factory=dict)
    summary: str = ""
    regressions: List[MetricDelta] = field(default_factory=list)
    improvements: List[MetricDelta] = field(default_factory=list)

