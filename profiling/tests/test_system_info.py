"""System info detection tests."""

from powers_profile.utils import detect_system_info
from powers_profile.schemas import SystemInfo


def test_detect_system_info_returns_values() -> None:
    info = detect_system_info()
    assert isinstance(info, SystemInfo)
    assert info.hostname
    assert info.cpu_cores_logical >= 1
    assert info.ram_total_gb >= 0.0
