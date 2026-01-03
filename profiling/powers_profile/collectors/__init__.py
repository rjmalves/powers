"""Collector implementations for profiling domains."""

from .base import Collector, Registry, resolve_collectors
from .timing import TimingCollector

REGISTRY: Registry = {
    TimingCollector.name: TimingCollector(),
}

__all__ = ["Collector", "Registry", "resolve_collectors", "TimingCollector", "REGISTRY"]
