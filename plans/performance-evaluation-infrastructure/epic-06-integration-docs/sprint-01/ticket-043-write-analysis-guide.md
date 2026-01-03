# [T-043] Write Analysis Guide

> **Epic**: [Epic 6: Integration & Documentation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-041
> **Blocks**: T-047

## Context

### Background
Developers need guidance on interpreting profiling results across domains and acting on findings.

### Relation to Epic
Provides actionable analysis steps and interpretation guidance.

### Current State
No analysis guide exists.

## Specification

### Inputs
- Example outputs from collectors and dashboards
- Best practices for CPU/memory/parallel analysis

### Outputs
- `docs/profiling/ANALYSIS_GUIDE.md` covering interpretation and next steps

### Behavior
- Describe how to read timing breakdowns, CPU hotspots, memory peaks, scaling efficiency
- Provide remediation checklists for common findings
- Include screenshots/snippets from dashboard and CLI summaries

### Error Handling
- N/A (docs), but include caveats for noisy data

## Acceptance Criteria
- [ ] ANALYSIS_GUIDE.md includes per-domain interpretation sections
- [ ] Includes remediation checklists and example visuals
- [ ] Links to Tools Reference and Quick Start

## Implementation Guide

### Suggested Approach
1. Outline by domain; include “What good looks like” vs “Red flags”.
2. Add examples from actual outputs or fixtures.
3. Provide action steps (optimize hotspot, reduce allocations, adjust threads).

### Key Files to Modify
- `docs/profiling/ANALYSIS_GUIDE.md`

### Patterns to Follow
- Use headings and bullet lists for clarity

### Pitfalls to Avoid
- ⚠️ Overly verbose theory; focus on actionable guidance

## Testing Requirements

### Documentation Checks
- [ ] Ensure examples align with actual outputs

## Documentation Requirements
- [ ] ANALYSIS_GUIDE.md completed and linked

## Dependencies
- **Blocked By**: T-041
- **Blocks**: T-047
- **Related**: T-032, T-019

## Effort Estimate
**Points**: 3
**Confidence**: Medium
**Rationale**: Documentation with examples.

## Definition of Done
- [ ] Guide merged
- [ ] Links added
