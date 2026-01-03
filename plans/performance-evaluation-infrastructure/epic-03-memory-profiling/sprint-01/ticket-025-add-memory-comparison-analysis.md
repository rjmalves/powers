# [T-025] Add memory comparison analysis

> **Epic**: [Epic 3: Memory Profiling Suite](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Status**: ✅ Complete
> **Dependencies**: T-024
> **Blocks**: None

## Context

### Background
Need ability to compare memory metrics between runs (baseline vs target) to detect regressions.

### Relation to Epic
Completes acceptance criterion for memory comparison analysis.

### Current State
Unified memory collector outputs data; no comparison logic exists.

## Specification

### Inputs
- Two memory collector outputs (baseline and target)
- Config: thresholds for regression warnings

### Outputs
- Comparison JSON summarizing deltas (peak heap, peak RSS, cache misses, allocation hotspots)
- Optional markdown summary for CLI

### Behavior
- Compute delta and percentage change for key metrics
- Highlight regressions beyond thresholds with severity flags
- Identify new/removed allocation hotspots when possible
- Integrate with `powers-profile compare` output

### Error Handling
- Handle missing metrics gracefully (e.g., tool skipped)
- Warn when comparing incompatible schema versions

## Acceptance Criteria
- [x] Comparison outputs regression highlights for memory metrics
- [x] Thresholds configurable and applied
- [x] Works when only subset of tools run (skips missing data)

## Implementation Guide

### Suggested Approach
1. Add analyzer (e.g., `analyzers/memory_comparison.py`). ✅
2. Implement delta calculations with helper functions. ✅
3. Wire into `compare` command when memory data present. ✅
4. Add optional markdown summary generation. ✅

### Key Files to Modify
- `profiling/powers_profile/analyzers/memory_comparison.py` ✅ Created
- `profiling/powers_profile/cli.py` ✅ Updated
- `profiling/powers_profile/reporters/markdown.py` ✅ Built into analyzer

### Patterns to Follow
- Similar to CPU diff logic (T-018) but for memory metrics ✅

### Pitfalls to Avoid
- ⚠️ Division by zero when baseline metrics are zero ✅ Handled
- ⚠️ Mislabeling improvements vs regressions ✅ Correct logic implemented

## Testing Requirements

### Unit Tests
- [x] Delta computation with positive/negative changes
- [x] Threshold-based severity mapping
- [x] Behavior when metrics missing

### Integration Tests
- [x] Compare two fixture memory outputs and generate summary

## Documentation Requirements
- [x] Document comparison thresholds and output fields

## Dependencies
- **Blocked By**: T-024 ✅
- **Blocks**: None
- **Related**: Epic 5 comparison dashboards

## Effort Estimate
**Points**: 3
**Confidence**: High
**Rationale**: Deterministic calculations with fixtures.

## Definition of Done
- [x] Implementation complete
- [x] Tests passing
- [x] Docs updated

## Progress (2026-01-03)

**COMPLETED**:
- Implemented `memory_comparison.py` analyzer (463 lines)
- Created `MetricDelta` class for computing deltas and percent changes
- Created `MemoryComparison` class with all memory metrics
- Implemented `compare_memory_metrics()` function with threshold checking
- Implemented `format_comparison_markdown()` for CLI display
- Integrated into `powers-profile compare` command
- Created 14 comprehensive unit tests
- Live testing validated with actual runs
- Comparison JSON export working
- Markdown formatting with colored indicators (🟢/🔴)
- Handles missing metrics gracefully
- Zero-baseline edge cases handled correctly
