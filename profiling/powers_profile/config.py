"""TOML configuration loading with layered overrides."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "profiling" / "config" / "default.toml"
USER_CONFIG_PATH = Path.home() / ".config" / "powers-profile" / "config.toml"

DEFAULTS: Dict[str, Any] = {
    "general": {
        "output_dir": "profiling_results",
        "binary": "target/release/powers",
        "default_example": "examples/05-large-scale-brazilian",
    },
    "collectors": {
        "default": ["timing", "cpu", "memory"],
        "available": ["timing", "cpu", "memory", "parallel", "io"],
    },
    "cpu": {
        "perf_frequency": 99,
        "perf_events": ["cycles", "instructions", "cache-misses"],
        "flamegraph_width": 1200,
        "flamegraph_colors": "hot",
    },
    "memory": {
        "dhat_enabled": True,
        "massif_enabled": True,
        "massif_time_unit": "ms",
        "cachegrind_enabled": False,
        "rss_interval_ms": 500,
    },
    "parallel": {
        "thread_counts": [1, 2, 4, 8, 16, 32],
        "warmup_iterations": 1,
    },
    "timing": {"parse_stdout": True, "log_level": "debug"},
    "io": {"enabled": False},
    "tools": {
        "perf": None,
        "valgrind": None,
        "flamegraph": None,
    },
    "thresholds": {
        "regression_percent": 5.0,
        "improvement_percent": 5.0,
        "rss_growth_mb": 10.0,
    },
}


@dataclass
class ProfilingConfig:
    """Resolved profiling configuration."""

    output_dir: Path
    binary: Path
    default_example: Path
    default_collectors: List[str]
    available_collectors: List[str]
    perf_frequency: int
    perf_events: List[str]
    flamegraph_width: int
    flamegraph_colors: str
    dhat_enabled: bool
    massif_enabled: bool
    massif_time_unit: str
    cachegrind_enabled: bool
    rss_interval_ms: int
    thread_counts: List[int]
    warmup_iterations: int
    regression_percent: float
    improvement_percent: float
    rss_growth_mb: float
    perf_path: Optional[Path] = None
    valgrind_path: Optional[Path] = None
    flamegraph_path: Optional[Path] = None
    source: Optional[Path] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    repo_root: Path = REPO_ROOT  # Working directory for running binaries


def _deep_merge(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _dict_to_config(
    data: Dict[str, Any], source: Optional[Path]
) -> ProfilingConfig:
    general = data.get("general", {})
    collectors = data.get("collectors", {})
    cpu = data.get("cpu", {})
    memory = data.get("memory", {})
    parallel = data.get("parallel", {})
    timing = data.get("timing", {})
    thresholds = data.get("thresholds", {})
    tools = data.get("tools", {})

    output_dir = _resolve_path(
        REPO_ROOT, general.get("output_dir", "profiling_results")
    )
    binary = _resolve_path(
        REPO_ROOT, general.get("binary", "target/release/powers")
    )
    default_example = _resolve_path(
        REPO_ROOT,
        general.get("default_example", "examples/05-large-scale-brazilian"),
    )

    return ProfilingConfig(
        output_dir=output_dir,
        binary=binary,
        default_example=default_example,
        default_collectors=list(collectors.get("default", [])),
        available_collectors=list(collectors.get("available", [])),
        perf_frequency=int(cpu.get("perf_frequency", 99)),
        perf_events=list(cpu.get("perf_events", [])),
        flamegraph_width=int(cpu.get("flamegraph_width", 1200)),
        flamegraph_colors=str(cpu.get("flamegraph_colors", "hot")),
        dhat_enabled=bool(memory.get("dhat_enabled", True)),
        massif_enabled=bool(memory.get("massif_enabled", True)),
        massif_time_unit=str(memory.get("massif_time_unit", "ms")),
        cachegrind_enabled=bool(memory.get("cachegrind_enabled", False)),
        rss_interval_ms=int(memory.get("rss_interval_ms", 500)),
        thread_counts=[int(t) for t in parallel.get("thread_counts", [])],
        warmup_iterations=int(parallel.get("warmup_iterations", 1)),
        regression_percent=float(thresholds.get("regression_percent", 5.0)),
        improvement_percent=float(thresholds.get("improvement_percent", 5.0)),
        rss_growth_mb=float(thresholds.get("rss_growth_mb", 10.0)),
        perf_path=Path(tools["perf"]).expanduser()
        if tools.get("perf")
        else None,
        valgrind_path=Path(tools["valgrind"]).expanduser()
        if tools.get("valgrind")
        else None,
        flamegraph_path=Path(tools["flamegraph"]).expanduser()
        if tools.get("flamegraph")
        else None,
        source=source,
        raw=data,
    )


def load_config(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> ProfilingConfig:
    """Load configuration from defaults, files, then CLI overrides."""
    effective = copy.deepcopy(DEFAULTS)

    search_paths = [
        config_path if config_path else None,
        DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None,
        USER_CONFIG_PATH if USER_CONFIG_PATH.exists() else None,
    ]

    source_used: Optional[Path] = None
    for path in search_paths:
        if path and path.exists():
            loaded = toml.load(path)
            effective = _deep_merge(effective, loaded)
            source_used = path
            break

    if cli_overrides:
        effective = _deep_merge(effective, cli_overrides)

    return _dict_to_config(effective, source_used)
