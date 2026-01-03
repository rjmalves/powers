"""Collector implementations for profiling domains."""

from .base import Collector, Registry, resolve_collectors
from .cachegrind import CachegrindCollector
from .cpu import CPUCollector
from .dhat import DhatCollector
from .massif import MassifCollector
from .memory import MemoryCollector
from .rss import RssCollector
from .timing import TimingCollector

REGISTRY: Registry = {
    TimingCollector.name: TimingCollector(),
    CPUCollector.name: CPUCollector(),
    MemoryCollector.name: MemoryCollector(),
    DhatCollector.name: DhatCollector(),
    MassifCollector.name: MassifCollector(),
    CachegrindCollector.name: CachegrindCollector(),
    RssCollector.name: RssCollector(),
}

__all__ = [
    "Collector",
    "Registry",
    "resolve_collectors",
    "TimingCollector",
    "CPUCollector",
    "MemoryCollector",
    "DhatCollector",
    "MassifCollector",
    "CachegrindCollector",
    "RssCollector",
    "REGISTRY",
]
