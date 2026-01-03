# [T-033] Implement dashboard base template

> **Epic**: [Epic 5: Visualization & Reporting Dashboard](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: Epics 1-4 data schemas
> **Blocks**: T-034, T-035, T-036, T-037, T-038, T-040

## Context

### Background
Need a reusable Plotly dashboard shell that accepts profiling data and renders core layout.

### Relation to Epic
Foundation for all visualization components and comparison views.

### Current State
No dashboard code exists.

## Specification

### Inputs
- Profiling run JSON (full run or aggregated)
- Config: offline/online plotlyjs, theme, output path

### Outputs
- `dashboard.html` base with placeholders/containers for all sections
- Shared CSS/JS assets (inline or embedded)

### Behavior
- Build layout with header, summary cards, tabs (Timing, CPU, Memory, Parallel, Comparison)
- Include placeholder components to be filled by subsequent tickets
- Support offline embedding of plotly.js (configurable)
- Provide function to render dashboard given data object

### Error Handling
- Validate required data sections exist; show user-friendly message if missing
- Fail when output path unwritable

## Acceptance Criteria
- [ ] Base dashboard renders with stub data without errors
- [ ] Tabs and layout structure present
- [ ] Offline mode supported
- [ ] Exported as single self-contained HTML file

## Implementation Guide

### Suggested Approach
1. Use Plotly + Jinja2/simple string templates to build HTML.
2. Define reusable components/helpers for cards and charts.
3. Add option for CDN vs inline plotly.js.

### Key Files to Modify
- `profiling/powers_profile/reporters/dashboard.py`
- `profiling/powers_profile/config/default.toml`

### Patterns to Follow
- Self-contained HTML similar to existing repo docs if any

### Pitfalls to Avoid
- ⚠️ Loading external assets without offline fallback
- ⚠️ Hardcoding dimensions; use responsive layout

## Testing Requirements

### Unit Tests
- [ ] Base render produces HTML containing all tabs
- [ ] Offline mode includes plotly.js inline

### Integration Tests
- [ ] Render with sample data and open file to ensure no JS errors (headless check)

## Documentation Requirements
- [ ] Document dashboard generation command and options

## Dependencies
- **Blocked By**: Epics 1-4 data schemas
- **Blocks**: T-034, T-035, T-036, T-037, T-038, T-040
- **Related**: T-039

## Effort Estimate
**Points**: 5
**Confidence**: Medium
**Rationale**: Layout and templating setup.

## Definition of Done
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Docs updated
