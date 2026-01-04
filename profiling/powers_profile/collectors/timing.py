"""Timing collector that executes the target binary and parses timing markers."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import ProfilingConfig
from ..schemas import CollectorResult
from .base import Collector

TIMING_PATTERNS: Tuple[re.Pattern[str], ...] = (
    # Pattern for POWE.RS format: "Training time: 00:00:19.488" or "Total running time: 00:00:20.099"
    re.compile(
        r"(?P<name>[\w\s]+?)\s+time:\s*(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\.(?P<millis>\d{3})",
        re.IGNORECASE,
    ),
    # Pattern for [TIMING] markers: "[TIMING] phase_name: 123.45 ms"
    re.compile(
        r"\[TIMING\]\s*(?P<name>[\w\-/\.]+)\s*[:=]\s*(?P<value>[\d\.]+)\s*(?P<unit>ms|s|sec|seconds|millis)?",
        re.IGNORECASE,
    ),
    # Pattern for _time_ms suffix: "phase_name_time_ms = 123.45"
    re.compile(
        r"(?P<name>[\w\-/\.]+)_time_ms\s*=\s*(?P<value>[\d\.]+)",
        re.IGNORECASE,
    ),
    # Pattern for TIMING keyword: "TIMING phase_name = 123.45 s"
    re.compile(
        r"TIMING\s+(?P<name>[\w\-/\.]+)\s*=\s*(?P<value>[\d\.]+)\s*(?P<unit>ms|s|sec|seconds|millis)?",
        re.IGNORECASE,
    ),
)


def _to_seconds(value: str, unit: str | None) -> float:
    val = float(value)
    if unit is None:
        return val
    unit_lower = unit.lower()
    if unit_lower in {"ms", "millis"}:
        return val / 1000.0
    return val


def _extract_unit(match: re.Match[str]) -> str | None:
    unit = match.groupdict().get("unit")
    if unit:
        return unit
    text = match.group(0).lower()
    if "time_ms" in text or "ms" in text.split("=")[0]:
        return "ms"
    return None


def parse_timings(output: str) -> Dict[str, float]:
    """Extract timing values (seconds) from program output.
    
    Supports multiple formats:
    - HH:MM:SS.mmm format: "Training time: 00:00:19.488"
    - [TIMING] markers: "[TIMING] phase = 123.45 s"
    - Suffix format: "phase_time_ms = 123.45"
    - TIMING keyword: "TIMING phase = 123.45 s"
    """
    metrics: Dict[str, float] = {}
    for line in output.splitlines():
        for pattern in TIMING_PATTERNS:
            match = pattern.search(line)
            if match:
                groups = match.groupdict()
                name = groups.get("name", "").strip()
                
                # Handle HH:MM:SS.mmm format
                if "hours" in groups:
                    hours = int(groups["hours"])
                    minutes = int(groups["minutes"])
                    seconds = int(groups["seconds"])
                    millis = int(groups["millis"])
                    total_seconds = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
                    # Normalize name: "Total running" -> "total_running"
                    normalized_name = name.lower().replace(" ", "_")
                    metrics[normalized_name] = total_seconds
                else:
                    # Handle numeric value with optional unit
                    value = groups.get("value")
                    unit = _extract_unit(match)
                    metrics[name] = _to_seconds(value, unit)
                
                break  # Stop after first match for this line
    return metrics


class TimingCollector(Collector):
    """Collector that runs the binary and parses timing markers from stdout."""

    name = "timing"

    def collect(
        self,
        *,
        binary: Path,
        args: List[str],
        config: ProfilingConfig,
        run_dir: Path,
    ) -> CollectorResult:
        start = time.perf_counter()
        collector_dir = run_dir / self.name
        collector_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = collector_dir / "stdout.log"
        stderr_path = collector_dir / "stderr.log"

        errors: List[str] = []
        warnings: List[str] = []

        if not binary.exists():
            errors.append(f"Binary not found: {binary}")
            duration = time.perf_counter() - start
            return CollectorResult(
                collector_name=self.name,
                success=False,
                duration_seconds=duration,
                data={},
                errors=errors,
                warnings=warnings,
                raw_files=[str(stdout_path), str(stderr_path)],
            )

        try:
            completed = subprocess.run(
                [str(binary), *args],
                capture_output=True,
                text=True,
                check=False,
                cwd=config.repo_root,
            )
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Failed to execute binary: {exc}")
            duration = time.perf_counter() - start
            return CollectorResult(
                collector_name=self.name,
                success=False,
                duration_seconds=duration,
                data={},
                errors=errors,
                warnings=warnings,
                raw_files=[str(stdout_path), str(stderr_path)],
            )

        duration = time.perf_counter() - start
        timings = parse_timings(completed.stdout)
        if not timings:
            warnings.append("No timing markers detected in stdout.")

        success = completed.returncode == 0
        if not success:
            errors.append(f"Binary exited with code {completed.returncode}")

        data = {
            "timings_seconds": timings,
            "exit_code": completed.returncode,
        }
        if completed.stderr:
            data["stderr_preview"] = completed.stderr.splitlines()[:5]

        return CollectorResult(
            collector_name=self.name,
            success=success and not errors,
            duration_seconds=duration,
            data=data,
            errors=errors,
            warnings=warnings,
            raw_files=[str(stdout_path), str(stderr_path)],
        )
