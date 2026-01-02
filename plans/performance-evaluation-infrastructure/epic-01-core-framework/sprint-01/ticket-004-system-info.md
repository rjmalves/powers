# T-004: Implement System Info Detection

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Foundation](./00-sprint-overview.md)
> **Dependencies**: T-001, T-003
> **Blocks**: T-008

---

## Context

### Background

Every profiling run must capture the system configuration to enable meaningful comparisons. A run on a 4-core laptop cannot be directly compared to a run on a 192-core server without knowing the hardware context.

### Current State

No system detection exists. The `SystemInfo` schema was defined in T-003.

---

## Specification

### Detection Requirements

| Field | Source | Fallback |
|-------|--------|----------|
| hostname | `socket.gethostname()` | "unknown" |
| os_name | `platform.system()` | "unknown" |
| os_version | `/proc/version` or `platform.release()` | "unknown" |
| cpu_model | `/proc/cpuinfo` model name | "unknown" |
| cpu_cores_physical | `psutil` or `/proc/cpuinfo` | logical count |
| cpu_cores_logical | `os.cpu_count()` | 1 |
| cpu_freq_mhz | `/proc/cpuinfo` or `/sys/` | None |
| ram_total_gb | `/proc/meminfo` | 0.0 |
| rust_version | `rustc --version` | "unknown" |
| powers_version | Parse Cargo.toml or binary | "unknown" |
| timestamp | Current UTC time | (required) |

### Implementation Notes

**Linux-specific parsing:**
```python
def get_cpu_model() -> str:
    """Extract CPU model from /proc/cpuinfo."""
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":")[1].strip()
    return "unknown"

def get_ram_total_gb() -> float:
    """Extract total RAM from /proc/meminfo."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                kb = int(line.split()[1])
                return round(kb / (1024 * 1024), 2)
    return 0.0
```

**Rust version detection:**
```python
def get_rust_version() -> str:
    """Get rustc version."""
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # "rustc 1.89.0 (29483883e 2025-08-04)"
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
```

**POWE.RS version detection:**
```python
def get_powers_version(binary_path: Optional[Path] = None) -> str:
    """Get POWE.RS version from binary or Cargo.toml."""
    # Try binary --version
    if binary_path and binary_path.exists():
        try:
            result = subprocess.run(
                [str(binary_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
    
    # Fall back to Cargo.toml
    cargo_toml = Path("Cargo.toml")
    if cargo_toml.exists():
        with open(cargo_toml) as f:
            for line in f:
                if line.startswith("version"):
                    return line.split('"')[1]
    
    return "unknown"
```

### Output

Function `detect_system_info() -> SystemInfo` that populates all fields.

---

## Acceptance Criteria

- [ ] All fields populated on Linux
- [ ] Graceful fallbacks for missing data
- [ ] No exceptions thrown for missing files
- [ ] Works on WSL2
- [ ] Rust version correctly detected
- [ ] CPU core counts accurate
- [ ] RAM total accurate (within 1%)

---

## Implementation Guide

### Suggested Approach

1. Create `utils/system_info.py`
2. Implement individual detection functions
3. Create main `detect_system_info()` function
4. Handle all exceptions gracefully
5. Add unit tests with mocking

### Key Files to Create

- `profiling/powers_profile/utils/system_info.py`

### Patterns to Follow

```python
# utils/system_info.py
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..schemas.system import SystemInfo

def detect_system_info(binary_path: Optional[Path] = None) -> SystemInfo:
    """Detect system information for profiling context."""
    return SystemInfo(
        hostname=_get_hostname(),
        os_name=platform.system(),
        os_version=_get_os_version(),
        cpu_model=_get_cpu_model(),
        cpu_cores_physical=_get_physical_cores(),
        cpu_cores_logical=os.cpu_count() or 1,
        cpu_freq_mhz=_get_cpu_freq(),
        ram_total_gb=_get_ram_total(),
        rust_version=_get_rust_version(),
        powers_version=_get_powers_version(binary_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

def _get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"

# ... etc
```

### Pitfalls to Avoid

- ⚠️ Don't assume `/proc/` exists (though we're Linux-only)
- ⚠️ Handle subprocess timeouts
- ⚠️ Don't let any exception propagate - always fallback
- ⚠️ CPU frequency may not be available in VMs/containers

---

## Testing Requirements

### Unit Tests

```python
# tests/test_system_info.py
from unittest.mock import patch, mock_open

def test_detect_system_info_returns_system_info():
    info = detect_system_info()
    assert isinstance(info, SystemInfo)
    assert info.hostname != ""
    assert info.cpu_cores_logical >= 1

def test_cpu_model_detection():
    mock_cpuinfo = "model name\t: Intel Core i7\n"
    with patch("builtins.open", mock_open(read_data=mock_cpuinfo)):
        model = _get_cpu_model()
        assert "Intel" in model

def test_fallback_on_missing_proc():
    with patch("builtins.open", side_effect=FileNotFoundError):
        model = _get_cpu_model()
        assert model == "unknown"
```

---

## Documentation Requirements

- [ ] Docstrings on all functions
- [ ] Document which fields may be "unknown"

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Standard system introspection, well-known patterns

---

## Definition of Done

- [ ] All detection functions implemented
- [ ] Fallbacks work correctly
- [ ] Tests pass
- [ ] Works on WSL2
- [ ] Code reviewed
