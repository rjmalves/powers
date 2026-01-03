"""System information schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import SerializableMixin


@dataclass
class SystemInfo(SerializableMixin):
    """System hardware and software information."""

    hostname: str
    os_name: str
    os_version: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_freq_mhz: Optional[float]
    ram_total_gb: float
    rust_version: str
    powers_version: str
    timestamp: str

