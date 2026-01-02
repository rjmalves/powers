# T-006: Create Configuration Loading

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Foundation](./00-sprint-overview.md)
> **Dependencies**: T-001
> **Blocks**: T-008

---

## Context

### Background

The profiling framework needs configurable defaults for output directories, default collectors, benchmark definitions, and tool paths. This ticket implements TOML-based configuration loading with sensible defaults.

### Current State

A placeholder `config/default.toml` was created in T-001.

---

## Specification

### Configuration Schema

```toml
# config/default.toml

[general]
# Default output directory (relative to repo root)
output_dir = "profiling_results"
# Default binary to profile
binary = "target/release/powers"
# Default example to run
default_example = "examples/05-large-scale-brazilian"

[collectors]
# Default collectors to run
default = ["timing", "cpu", "memory"]
# Available collectors
available = ["timing", "cpu", "memory", "parallel", "io"]

[cpu]
# perf record options
perf_frequency = 99
perf_events = ["cycles", "instructions", "cache-misses"]
# FlameGraph options
flamegraph_width = 1200
flamegraph_colors = "hot"

[memory]
# DHAT options
dhat_enabled = true
# Massif options
massif_enabled = true
massif_time_unit = "ms"
# Cachegrind options
cachegrind_enabled = false  # Slow, disabled by default
# RSS monitoring
rss_interval_ms = 500

[parallel]
# Thread counts for scaling analysis
thread_counts = [1, 2, 4, 8, 16, 32]
# Warmup iterations before measurement
warmup_iterations = 1

[timing]
# Extract timing from program output
parse_stdout = true
# Timing log level to capture
log_level = "debug"

[io]
# I/O profiling (future)
enabled = false

[tools]
# External tool paths (auto-detected if not specified)
# perf = "/usr/bin/perf"
# valgrind = "/usr/bin/valgrind"
# flamegraph = "~/.local/FlameGraph"

[thresholds]
# Performance regression thresholds
regression_percent = 5.0
improvement_percent = 5.0
# Memory thresholds
rss_growth_mb = 10.0
```

### Configuration Hierarchy

1. Built-in defaults (hardcoded)
2. `profiling/config/default.toml` (repo config)
3. `~/.config/powers-profile/config.toml` (user config)
4. CLI arguments (highest priority)

### Config Class

```python
@dataclass
class ProfilingConfig:
    """Profiling configuration."""
    output_dir: Path
    binary: Path
    default_example: Path
    default_collectors: List[str]
    available_collectors: List[str]
    
    # CPU config
    perf_frequency: int
    perf_events: List[str]
    flamegraph_width: int
    flamegraph_colors: str
    
    # Memory config
    dhat_enabled: bool
    massif_enabled: bool
    massif_time_unit: str
    cachegrind_enabled: bool
    rss_interval_ms: int
    
    # Parallel config
    thread_counts: List[int]
    warmup_iterations: int
    
    # Thresholds
    regression_percent: float
    improvement_percent: float
    rss_growth_mb: float
    
    # Tool paths (Optional)
    perf_path: Optional[Path]
    valgrind_path: Optional[Path]
    flamegraph_path: Optional[Path]
```

### Loading Function

```python
def load_config(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> ProfilingConfig:
    """Load configuration from files and CLI overrides."""
    pass
```

---

## Acceptance Criteria

- [ ] Default config file created with all options
- [ ] Config loads from default location
- [ ] Config loads from custom path via `--config`
- [ ] CLI arguments override config file
- [ ] Missing config file uses built-in defaults
- [ ] Invalid TOML produces clear error message
- [ ] All paths resolved relative to repo root

---

## Implementation Guide

### Suggested Approach

1. Create `config.py` with default values as constants
2. Implement TOML parsing with `toml` library
3. Implement config merging (defaults → file → CLI)
4. Create `ProfilingConfig` dataclass
5. Add unit tests

### Key Files to Create/Modify

- `profiling/powers_profile/config.py`
- `profiling/config/default.toml`

### Patterns to Follow

```python
# config.py
import toml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Built-in defaults
DEFAULTS = {
    "general": {
        "output_dir": "profiling_results",
        "binary": "target/release/powers",
        "default_example": "examples/05-large-scale-brazilian",
    },
    "collectors": {
        "default": ["timing", "cpu", "memory"],
        "available": ["timing", "cpu", "memory", "parallel", "io"],
    },
    # ... etc
}

def load_config(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> ProfilingConfig:
    """Load and merge configuration."""
    config = DEFAULTS.copy()
    
    # Load from file
    paths_to_try = [
        config_path,
        Path("profiling/config/default.toml"),
        Path.home() / ".config/powers-profile/config.toml",
    ]
    
    for path in paths_to_try:
        if path and path.exists():
            file_config = toml.load(path)
            config = _deep_merge(config, file_config)
            break
    
    # Apply CLI overrides
    if cli_overrides:
        config = _deep_merge(config, cli_overrides)
    
    return _config_to_dataclass(config)

def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

### Pitfalls to Avoid

- ⚠️ Resolve paths relative to repo root, not CWD
- ⚠️ Handle missing nested keys gracefully
- ⚠️ TOML doesn't support `None` - use sentinel values
- ⚠️ Validate thread_counts are positive integers

---

## Testing Requirements

### Unit Tests

```python
# tests/test_config.py

def test_load_default_config():
    config = load_config()
    assert config.output_dir == Path("profiling_results")
    assert "timing" in config.default_collectors

def test_cli_overrides():
    config = load_config(cli_overrides={
        "general": {"output_dir": "/tmp/test"}
    })
    assert config.output_dir == Path("/tmp/test")

def test_missing_config_uses_defaults():
    config = load_config(config_path=Path("/nonexistent/path.toml"))
    assert config is not None

def test_invalid_toml_raises_error():
    # Create temp file with invalid TOML
    # Expect clear error message
    pass
```

---

## Documentation Requirements

- [ ] Document all config options in default.toml with comments
- [ ] Document config hierarchy in README

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Standard config loading pattern with TOML

---

## Definition of Done

- [ ] Config file created with all options
- [ ] Loading and merging works
- [ ] CLI overrides work
- [ ] Tests pass
- [ ] Code reviewed
