"""Collector and analysis result schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import SerializableMixin


@dataclass
class CollectorResult(SerializableMixin):
    """Result from a single collector."""

    collector_name: str
    success: bool
    duration_seconds: float
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_files: List[str] = field(default_factory=list)

