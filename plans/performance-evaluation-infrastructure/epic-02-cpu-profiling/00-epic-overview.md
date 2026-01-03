# Epic 2: CPU & Execution Profiling

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: 🟢 Completed

---

## Summary

This epic implements CPU profiling capabilities using Linux perf and FlameGraph visualization. It enables developers to identify CPU hotspots, analyze call stacks, and generate interactive flame graphs for performance investigation.

---

## Scope

### Included

1. **perf Integration**
   - `perf record` wrapper for CPU sampling
   - `perf report` output parsing
   - CPU cycle, instruction, and cache miss collection
   - Hotspot identification and ranking

2. **FlameGraph Generation**
   - Integration with Brendan Gregg's FlameGraph scripts
   - SVG generation from perf data
   - Customizable colors and dimensions
   - Folded stack generation

3. **Differential FlameGraphs**
   - Compare two runs to show hot path changes
   - Highlight regressions vs improvements

4. **CPU Collector**
   - Structured JSON output of CPU metrics
   - Integration with core framework

### Excluded

- eBPF-based profiling
- Off-CPU analysis
- Custom perf events (hardware PMU)

---

## Dependencies

- **Requires**: Epic 1 (Core Framework)
- **Enables**: Epic 5 (Visualization)

---

## Acceptance Criteria

- [x] `powers-profile run --collectors cpu` generates perf data
- [x] FlameGraph SVG automatically generated
- [x] Top 20 hotspots listed in JSON output
- [x] `powers-profile compare` shows differential flamegraph
- [x] Works on WSL2 (with perf limitations documented)
- [x] Works on bare-metal Linux

---

## Technical Approach

### External Tools Required

| Tool | Purpose | Installation |
|------|---------|--------------|
| `perf` | CPU sampling | `apt install linux-tools-$(uname -r)` |
| `FlameGraph` | SVG generation | Clone from GitHub |

### Data Flow

```
powers-profile run --collectors cpu
        │
        ▼
┌─────────────────┐
│ perf record     │ → perf.data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ perf script     │ → stacks.txt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ stackcollapse   │ → folded.txt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ flamegraph.pl   │ → flamegraph.svg
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parse & Extract │ → cpu_data.json
└─────────────────┘
```

---

## Sprints

### [Sprint 1: CPU Profiling](./sprint-01/00-sprint-overview.md)

| ID | Title | Points | Status |
|----|-------|--------|--------|
| T-014 | Implement perf record wrapper | 5 | ✅ |
| T-015 | Implement FlameGraph integration | 5 | ✅ |
| T-016 | Implement CPU collector | 5 | ✅ |
| T-017 | Parse perf report for hotspots | 3 | ✅ |
| T-018 | Implement differential flamegraph | 3 | ✅ |
| T-019 | Document perf setup and limitations | 2 | ✅ |

**Sprint Points**: 23

---

## Estimated Effort

- **Duration**: 1 sprint (2 weeks)
- **Story Points**: 23
- **Risk Level**: Medium (perf permissions, WSL2 limitations)

---

## Definition of Done

- [x] All tickets complete
- [x] FlameGraph SVGs generated successfully
- [x] CPU hotspots in JSON output
- [x] Differential comparison works
- [x] Documentation complete
- [x] Works on both WSL2 and bare-metal
