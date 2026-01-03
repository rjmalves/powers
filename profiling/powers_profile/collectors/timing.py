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
    re.compile(
        r"\[TIMING\]\s*(?P<name>[\w\-/\.]+)\s*[:=]\s*(?P<value>[\d\.]+)\s*(?P<unit>ms|s|sec|seconds|millis)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<name>[\w\-/\.]+)_time_ms\s*=\s*(?P<value>[\d\.]+)",
        re.IGNORECASE,
    ),
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
    """Extract timing values (seconds) from program output."""
    metrics: Dict[str, float] = {}
    for line in output.splitlines():
        for pattern in TIMING_PATTERNS:
            match = pattern.search(line)
            if match:
                name = match.group("name")
                value = match.group("value")
                unit = _extract_unit(match)
                metrics[name] = _to_seconds(value, unit)
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
                cwd=binary.parent,
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
