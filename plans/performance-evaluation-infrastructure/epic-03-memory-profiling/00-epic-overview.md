# Epic 3: Memory Profiling Suite

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: ⬜ Not Started

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

## Acceptance Criteria

- [ ] `powers-profile run --collectors memory` runs all memory tools
- [ ] DHAT data extracted and parsed to JSON
- [ ] Massif snapshots captured and analyzed
- [ ] Cachegrind metrics extracted (when enabled)
- [ ] RSS tracked with per-iteration breakdown
- [ ] All data in machine-readable JSON format
- [ ] Works with valgrind 3.18+

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
| T-020 | Implement DHAT collector | 5 | ⬜ |
| T-021 | Implement Massif collector | 5 | ⬜ |
| T-022 | Implement Cachegrind collector | 3 | ⬜ |
| T-023 | Implement RSS monitor | 5 | ⬜ |
| T-024 | Create unified memory collector | 3 | ⬜ |
| T-025 | Add memory comparison analysis | 3 | ⬜ |

**Sprint Points**: 24

---

## Estimated Effort

- **Duration**: 1 sprint (2 weeks)
- **Story Points**: 24
- **Risk Level**: Medium (valgrind overhead, parsing complexity)

---

## Definition of Done

- [ ] All tickets complete
- [ ] All 4 memory tools integrated
- [ ] JSON output for all metrics
- [ ] Comparison between runs works
- [ ] Documentation complete
