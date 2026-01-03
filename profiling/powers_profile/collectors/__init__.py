"""Collector implementations for profiling domains."""

from .base import Collector, Registry, resolve_collectors
from .cpu import CPUCollector
from .timing import TimingCollector

REGISTRY: Registry = {
    TimingCollector.name: TimingCollector(),
    CPUCollector.name: CPUCollector(),
}

__all__ = [
    "Collector",
    "Registry",
    "resolve_collectors",
    "TimingCollector",
    "CPUCollector",
    "REGISTRY",
]
