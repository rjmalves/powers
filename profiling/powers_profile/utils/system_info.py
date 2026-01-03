"""System information detection."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..schemas.system import SystemInfo


def _read_first_match(path: Path, prefix: str) -> Optional[str]:
    try:
        with path.open("r") as handle:
            for line in handle:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return None
    return None


def _get_cpu_model() -> str:
    value = _read_first_match(Path("/proc/cpuinfo"), "model name")
    return value or "unknown"


def _get_cpu_freq() -> Optional[float]:
    value = _read_first_match(Path("/proc/cpuinfo"), "cpu MHz")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _get_physical_cores() -> int:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        physical_cores = set()
        with cpuinfo.open("r") as handle:
            physical_id = None
            core_id = None
            for line in handle:
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                if physical_id is not None and core_id is not None:
                    physical_cores.add((physical_id, core_id))
                    physical_id = None
                    core_id = None
        if physical_cores:
            return len(physical_cores)
    except FileNotFoundError:
        pass
    return os.cpu_count() or 1


def _get_ram_total_gb() -> float:
    try:
        with Path("/proc/meminfo").open("r") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return round(kb / (1024 * 1024), 2)
    except (FileNotFoundError, ValueError):
        return 0.0
    return 0.0


def _run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return ""
    return ""


def _get_rust_version() -> str:
    return _run_command(["rustc", "--version"]) or "unknown"


def _get_powers_version(binary_path: Optional[Path] = None) -> str:
    if binary_path and binary_path.exists():
        version = _run_command([str(binary_path), "--version"])
        if version:
            return version

    cargo_toml = Path(__file__).resolve().parents[2] / "Cargo.toml"
    if cargo_toml.exists():
        try:
            import toml

            parsed = toml.load(cargo_toml)
            package = parsed.get("package", {})
            version = package.get("version")
            if version:
                return str(version)
        except Exception:
            return "unknown"
    return "unknown"


def _get_os_version() -> str:
    proc_version = Path("/proc/version")
    if proc_version.exists():
        try:
            return proc_version.read_text().strip()
        except Exception:
            pass
    return platform.release()


def detect_system_info(binary_path: Optional[Path] = None) -> SystemInfo:
    """Detect system information for profiling context."""
    return SystemInfo(
        hostname=_safe_hostname(),
        os_name=platform.system() or "unknown",
        os_version=_get_os_version() or "unknown",
        cpu_model=_get_cpu_model(),
        cpu_cores_physical=_get_physical_cores(),
        cpu_cores_logical=os.cpu_count() or 1,
        cpu_freq_mhz=_get_cpu_freq(),
        ram_total_gb=_get_ram_total_gb(),
        rust_version=_get_rust_version(),
        powers_version=_get_powers_version(binary_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _safe_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"

