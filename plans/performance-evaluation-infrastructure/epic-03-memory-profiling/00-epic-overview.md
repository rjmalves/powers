# Epic 3: Memory Profiling Suite

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: ✅ Complete

---

## Summary

This epic implements comprehensive memory profiling using valgrind tools (DHAT, Massif, Cachegrind) and RSS monitoring. It enables developers to analyze heap allocation patterns, memory growth over time, cache efficiency, and physical memory footprint.

---

## Scope

### Included

1. **DHAT Integration**
   - Heap allocation profiling
   - Block count and size analysis
   - Allocation hotspot identification
   - JSON output parsing

2. **Massif Integration**
   - Heap usage over time
   - Peak memory identification
   - Snapshot analysis
   - ms_print output parsing

3. **Cachegrind Integration**
   - Cache miss analysis (L1, L2, LL)
   - Branch prediction analysis
   - Instruction count
   - cg_annotate parsing

4. **RSS Monitoring**
   - Physical memory tracking during execution
   - Per-iteration memory deltas
   - Memory stability detection
   - Replaces old `monitor_rss.py`

5. **Memory Collector**
   - Unified JSON output for all memory metrics
   - Configurable tool selection

### Excluded

- Custom allocator profiling
- Memory leak detection (valgrind memcheck)
- NUMA-aware analysis

---

## Dependencies

- **Requires**: Epic 1 (Core Framework)
- **Enables**: Epic 5 (Visualization)

---

## Progress Updates (2026-01-03)

**SPRINT 1: 100% COMPLETE** ✅ (24/24 story points)

### All Implementation Completed

#### Memory Collectors
- ✅ **T-020: DHAT Collector** - Full valgrind DHAT integration with JSON parsing
  - Extracts heap allocation metrics and hotspots
  - Parses stack traces from frame table
  - Handles missing valgrind gracefully
  
- ✅ **T-021: Massif Collector** - Heap usage over time profiling
  - Parses massif.out snapshots
  - Identifies peak memory usage
  - Detects memory growth patterns
  
- ✅ **T-022: Cachegrind Collector** - Cache efficiency analysis
  - Extracts L1/LL cache miss rates
  - Instruction count and cache simulation
  - Configurable enable/disable
  
- ✅ **T-023: RSS Monitor** - Physical memory tracking
  - Background thread polling /proc/{pid}/status
  - Timestamped samples with summary stats
  - Growth detection heuristics
  
- ✅ **T-024: Unified Memory Collector** - Orchestration layer
  - Runs all memory tools in sequence
  - Aggregates results into memory_data.json
  - Partial failure handling (succeeds if any tool succeeds)

#### Memory Comparison Analysis  
- ✅ **T-025: Memory Comparison Analyzer** - Compare baseline vs target
  - `MetricDelta` class for computing deltas with percent changes
  - `MemoryComparison` class tracking all memory metrics
  - Threshold-based regression/improvement detection
  - Markdown formatter with colored indicators (🟢/🔴)
  - JSON export for programmatic access
  - Integrated into `powers-profile compare` command
  - Handles missing metrics and zero baselines gracefully

### CLI Improvements (2026-01-03)
- ✅ **Default Example Handling** - Automatically uses config.default_example when no args provided
- ✅ **Binary Validation** - Checks binary exists with helpful error message
- ✅ **Working Directory Fix** - All collectors now run from repo_root for proper path resolution
- ✅ **Comma-Separated Collectors** - Support `-c timing,rss` syntax
- ✅ **Memory Comparison** - Integrated into compare command with rich output

### Test Coverage
- **23 comprehensive unit tests** across 4 test files
- Mocked valgrind execution (no external dependencies for tests)
- Error handling validation
- Summary statistics verification
- Live testing completed: memory comparison working end-to-end

### Documentation
- ✅ Comprehensive README with usage examples
- ✅ Configuration guide
- ✅ Troubleshooting section
- ✅ All ticket documentation updated

---

## Acceptance Criteria
- [x] `powers-profile run --collectors memory` runs all memory tools
- [x] DHAT data extracted and parsed to JSON
- [x] Massif snapshots captured and analyzed
- [x] Cachegrind metrics extracted (when enabled)
- [x] RSS tracked with per-iteration breakdown
- [x] All data in machine-readable JSON format
- [x] Works with valgrind 3.18+

---

## Technical Approach

### External Tools Required

| Tool | Purpose | Installation |
|------|---------|--------------|
| `valgrind` | DHAT, Massif, Cachegrind | `apt install valgrind` |

### Tool Commands

```bash
# DHAT
valgrind --tool=dhat --dhat-out-file=dhat.out ./binary args

# Massif
valgrind --tool=massif --massif-out-file=massif.out --time-unit=ms ./binary args

# Cachegrind
valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out ./binary args
```

### Data Flow

```
powers-profile run --collectors memory
        │
        ▼
┌─────────────────────────────────────────────┐
│                 PARALLEL                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │
│  │  DHAT   │  │ Massif  │  │ RSS Monitor │  │
│  └────┬────┘  └────┬────┘  └──────┬──────┘  │
│       │            │              │          │
│       ▼            ▼              ▼          │
│  dhat.json   massif.out     rss_data.json   │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐
│ Memory Collector│ → memory_data.json
└─────────────────┘
```

---

## Sprints

### [Sprint 1: Memory Tools](./sprint-01/00-sprint-overview.md)

| ID | Title | Points | Status |
|----|-------|--------|--------|
| T-020 | Implement DHAT collector | 5 | ✅ |
| T-021 | Implement Massif collector | 5 | ✅ |
| T-022 | Implement Cachegrind collector | 3 | ✅ |
| T-023 | Implement RSS monitor | 5 | ✅ |
| T-024 | Create unified memory collector | 3 | ✅ |
| T-025 | Add memory comparison analysis | 3 | ✅ |

**Sprint Points**: 24
**Completed**: 24/24 (100%) ✅

---

## Estimated Effort

- **Duration**: 1 sprint (2 weeks)
- **Story Points**: 24
- **Risk Level**: Medium (valgrind overhead, parsing complexity)

---

## Definition of Done

- [x] All tickets complete (6/6 done) ✅
- [x] All 4 memory tools integrated (DHAT, Massif, Cachegrind, RSS)
- [x] JSON output for all metrics
- [x] Comparison between runs works (T-025 complete)
- [x] Documentation complete
