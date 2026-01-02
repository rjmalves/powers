# T-003: Define Core Data Schemas

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Foundation](./00-sprint-overview.md)
> **Dependencies**: T-001
> **Blocks**: T-007, T-008, T-009, T-010

---

## Context

### Background

All profiling data must follow well-defined schemas to enable machine-readable output, version comparison, and programmatic analysis. This ticket defines the core data structures using Python dataclasses with JSON serialization support.

### Current State

No schema definitions exist. Previous scripts used ad-hoc data formats.

---

## Specification

### Schema Definitions

#### 1. SystemInfo

```python
@dataclass
class SystemInfo:
    """System hardware and software information."""
    hostname: str
    os_name: str           # e.g., "Linux"
    os_version: str        # e.g., "6.6.87.2-microsoft-standard-WSL2"
    cpu_model: str         # e.g., "Intel Core Ultra 7 165U"
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_freq_mhz: Optional[float]
    ram_total_gb: float
    rust_version: str
    powers_version: str
    timestamp: str         # ISO 8601 format
```

#### 2. GitInfo

```python
@dataclass
class GitInfo:
    """Git repository state."""
    commit_sha: str        # Full SHA
    commit_short: str      # 7-char short SHA
    branch: str
    is_dirty: bool         # Uncommitted changes
    commit_date: str       # ISO 8601
    commit_message: str    # First line only
    tags: List[str]        # Tags pointing to this commit
```

#### 3. CollectorResult

```python
@dataclass
class CollectorResult:
    """Result from a single collector."""
    collector_name: str    # e.g., "cpu", "memory", "timing"
    success: bool
    duration_seconds: float
    data: Dict[str, Any]   # Collector-specific data
    errors: List[str]      # Any errors encountered
    warnings: List[str]    # Any warnings
    raw_files: List[str]   # Paths to raw output files
```

#### 4. ProfilingRun

```python
@dataclass
class ProfilingRun:
    """Complete profiling session."""
    run_id: str            # UUID or timestamp-based
    timestamp: str         # ISO 8601
    system_info: SystemInfo
    git_info: GitInfo
    config: Dict[str, Any] # Profiling configuration used
    binary_path: str
    binary_args: List[str]
    collectors_run: List[str]
    results: Dict[str, CollectorResult]  # Keyed by collector name
    total_duration_seconds: float
    status: str            # "success", "partial", "failed"
```

#### 5. Comparison

```python
@dataclass
class MetricDelta:
    """Change in a single metric."""
    metric_name: str
    baseline_value: float
    target_value: float
    absolute_delta: float
    percent_delta: float
    improved: bool         # True if change is positive
    significant: bool      # True if > threshold

@dataclass
class Comparison:
    """Comparison between two profiling runs."""
    baseline_run_id: str
    target_run_id: str
    baseline_git: GitInfo
    target_git: GitInfo
    timestamp: str
    deltas: Dict[str, List[MetricDelta]]  # Keyed by category
    summary: str           # Human-readable summary
    regressions: List[MetricDelta]
    improvements: List[MetricDelta]
```

#### 6. HistoryEntry

```python
@dataclass
class HistoryEntry:
    """Entry in profiling history index."""
    run_id: str
    timestamp: str
    git_commit: str
    git_branch: str
    collectors_run: List[str]
    status: str
    path: str              # Relative path to run directory
```

### JSON Serialization

All schemas must support:
- `to_dict()` → Dict conversion
- `to_json()` → JSON string (pretty-printed)
- `from_dict(d)` → Class method for deserialization
- `from_json(s)` → Class method for JSON parsing

### Schema Versioning

Include a schema version in serialized output:
```json
{
  "_schema_version": "1.0",
  "_schema_type": "ProfilingRun",
  "run_id": "...",
  ...
}
```

---

## Acceptance Criteria

- [ ] All 6 schema classes implemented
- [ ] JSON serialization/deserialization works round-trip
- [ ] Schema version included in JSON output
- [ ] Optional fields have sensible defaults
- [ ] Type hints complete and mypy-compatible
- [ ] Docstrings on all classes and fields
- [ ] Unit tests for serialization round-trip

---

## Implementation Guide

### Suggested Approach

1. Create `schemas/base.py` with serialization mixin
2. Create `schemas/system.py` with SystemInfo
3. Create `schemas/git.py` with GitInfo
4. Create `schemas/results.py` with CollectorResult
5. Create `schemas/run.py` with ProfilingRun
6. Create `schemas/comparison.py` with Comparison, MetricDelta
7. Create `schemas/history.py` with HistoryEntry
8. Export all from `schemas/__init__.py`

### Key Files to Create

- `profiling/powers_profile/schemas/base.py`
- `profiling/powers_profile/schemas/system.py`
- `profiling/powers_profile/schemas/git.py`
- `profiling/powers_profile/schemas/results.py`
- `profiling/powers_profile/schemas/run.py`
- `profiling/powers_profile/schemas/comparison.py`
- `profiling/powers_profile/schemas/history.py`

### Patterns to Follow

```python
# schemas/base.py
from dataclasses import dataclass, asdict, fields
import json
from typing import TypeVar, Type
from datetime import datetime

T = TypeVar('T')
SCHEMA_VERSION = "1.0"

class SerializableMixin:
    """Mixin for JSON serialization support."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary with schema metadata."""
        d = asdict(self)
        d["_schema_version"] = SCHEMA_VERSION
        d["_schema_type"] = self.__class__.__name__
        return d
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls: Type[T], d: dict) -> T:
        """Create instance from dictionary."""
        # Remove metadata fields
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        return cls(**d)
    
    @classmethod
    def from_json(cls: Type[T], s: str) -> T:
        """Create instance from JSON string."""
        return cls.from_dict(json.loads(s))

# Usage:
@dataclass
class SystemInfo(SerializableMixin):
    hostname: str
    os_name: str
    # ...
```

### Pitfalls to Avoid

- ⚠️ Nested dataclasses need special handling in `from_dict`
- ⚠️ Use `default=str` in `json.dumps` for datetime/Path
- ⚠️ Don't use mutable default arguments (use `field(default_factory=list)`)

---

## Testing Requirements

### Unit Tests

```python
# tests/test_schemas.py
def test_system_info_round_trip():
    info = SystemInfo(hostname="test", ...)
    json_str = info.to_json()
    restored = SystemInfo.from_json(json_str)
    assert info == restored

def test_schema_version_included():
    info = SystemInfo(hostname="test", ...)
    d = info.to_dict()
    assert d["_schema_version"] == "1.0"
    assert d["_schema_type"] == "SystemInfo"

def test_profiling_run_nested_serialization():
    run = ProfilingRun(
        system_info=SystemInfo(...),
        git_info=GitInfo(...),
        ...
    )
    json_str = run.to_json()
    restored = ProfilingRun.from_json(json_str)
    assert run.system_info == restored.system_info
```

---

## Documentation Requirements

- [ ] Docstrings on all schema classes
- [ ] Example JSON output in comments
- [ ] Document schema versioning approach

---

## Effort Estimate

**Points**: 5
**Confidence**: High
**Rationale**: Multiple dataclasses with serialization, but straightforward pattern

---

## Definition of Done

- [ ] All 6 schema classes implemented
- [ ] Serialization round-trip works
- [ ] Tests pass
- [ ] Type hints complete
- [ ] Code reviewed
