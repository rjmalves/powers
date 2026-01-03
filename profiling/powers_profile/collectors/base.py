"""Collector base interfaces and registry helpers."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Iterable, List

from ..config import ProfilingConfig
from ..schemas import CollectorResult


class Collector(abc.ABC):
    """Abstract collector interface."""

    name: str

    @abc.abstractmethod
    def collect(
        self,
        *,
        binary: Path,
        args: List[str],
        config: ProfilingConfig,
        run_dir: Path,
    ) -> CollectorResult:
        """Run the collector and return structured results."""


Registry = Dict[str, Collector]


def resolve_collectors(requested: Iterable[str], registry: Registry) -> List[str]:
    """
    Resolve requested collector names to concrete entries in the registry.

    "all" selects every registered collector. Unknown names are ignored;
    callers should handle missing collectors explicitly.
    """
    requested_list = list(requested)
    if not requested_list:
        return []
    if "all" in requested_list:
        return list(registry.keys())
    return [name for name in requested_list if name in registry]

