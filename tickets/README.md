# Implementation Tickets: Explicit Load/Inflow Lag Separation

This directory contains detailed implementation tickets for the explicit load/inflow lag separation epic, which addresses a critical bug in cut generation for systems with mixed AR models.

## Quick Links

- **Epic Overview:** [EPIC_EXPLICIT_LAG_SEPARATION.md](EPIC_EXPLICIT_LAG_SEPARATION.md)
- **Sprint Planning:** [SPRINT_PLAN.md](SPRINT_PLAN.md)
- **Architecture Analysis:** [../docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md](../docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md)
- **Bug Report:** [../BUG_FIX_PAR_LOWER_BOUND.md](../BUG_FIX_PAR_LOWER_BOUND.md)

## Problem Summary

The current implementation uses a unified `Vec<Vec<usize>>` to store lag variables for both loads and inflows, erasing type information. This forces code to use fragile heuristics to distinguish entity types, leading to:

- **Critical Bug:** Cut generation incorrectly matches coefficients to variables when loads and inflows have similar AR orders
- **Invalid Lower Bounds:** Lower bound exceeds simulation results (violating SDDP correctness)
- **Fragile Code:** Changes to AR models can break existing functionality
- **Performance Issues:** Unnecessary filtering and type checking

## Solution Approach

Introduce explicit, type-safe structures:
- `LoadLagVariables` indexed by `bus_id`
- `InflowLagVariables` indexed by `hydro_id`
- `LoadLagConstraints` indexed by `bus_id`
- `InflowLagConstraints` indexed by `hydro_id`

This follows the successful pattern from the `Realization` refactoring and embodies the principle: **Make illegal states unrepresentable**.

## Implementation Tickets

### Sprint 1: Foundation (Week 1-2)

| Ticket | Title | Effort | Priority | Status |
|--------|-------|--------|----------|--------|
| [001](TICKET-001-design-lag-structures.md) | Design and Implement Lag Variable Data Structures | 3 SP | P0 | Planned |
| [002](TICKET-002-parallel-variable-creation.md) | Add Parallel Lag Variable Creation in Subproblem | 5 SP | P0 | Planned |
| [003](TICKET-003-validation-framework.md) | Implement Validation Framework for Migration | 3 SP | P1 | Planned |

**Sprint 1 Goal:** Establish new architecture with backward compatibility and validation

### Sprint 2: Critical Bug Fixes (Week 3-4)

| Ticket | Title | Effort | Priority | Status |
|--------|-------|--------|----------|--------|
| [004](TICKET-004-fix-cut-generation-bug.md) | Fix Critical Bug in add_cut_constraint_to_model | 5 SP | P0 | Planned |
| [005](TICKET-005-migrate-dual-extraction.md) | Migrate Dual Extraction to Use Explicit Structures | 3 SP | P1 | Planned |

**Sprint 2 Goal:** Fix the critical bug and validate with Example 07

### Sprint 3: Complete Migration (Week 5-6)

| Ticket | Title | Effort | Priority | Status |
|--------|-------|--------|----------|--------|
| [006](TICKET-006-update-state-extraction.md) | Update State Lag Extraction Methods | 3 SP | P1 | Planned |
| [007](TICKET-007-migrate-constraint-fixing.md) | Migrate Lag Constraint Fixing Logic | 2 SP | P2 | Planned |
| [008](TICKET-008-integration-tests.md) | Add Comprehensive Integration Tests | 5 SP | P1 | Planned |

**Sprint 3 Goal:** Complete migration and validate with comprehensive tests

### Sprint 4: Validation & Cleanup (Week 7-8)

| Ticket | Title | Effort | Priority | Status |
|--------|-------|--------|----------|--------|
| [009](TICKET-009-performance-benchmarking.md) | Performance Benchmarking and Optimization | 3 SP | P2 | Planned |
| [010](TICKET-010-cleanup-documentation.md) | Remove Deprecated Code and Update Documentation | 3 SP | P1 | Planned |

**Sprint 4 Goal:** Validate performance and finalize refactoring

## Ticket Structure

Each ticket follows a consistent structure:

- **Context:** Why this work is needed
- **Acceptance Criteria:** Specific, measurable outcomes
- **Tasks:**
  - Implementation (with code hints)
  - Testing (unit, integration, regression, performance)
  - Documentation (inline, module, architecture, examples)
- **Technical Notes:** Implementation details, edge cases, performance considerations
- **Dependencies:** Blockers and related tickets
- **Definition of Done:** Checklist for completion

## Story Point Scale

- **1 SP:** Half-day task, straightforward implementation
- **2 SP:** 1-day task, moderate complexity
- **3 SP:** 2-day task, significant work or multiple components
- **5 SP:** 3-day task, complex with multiple integration points
- **8 SP:** 4+ days, consider breaking down further

## Priority Levels

- **P0:** Blocker - must complete before dependent work
- **P1:** High - critical for epic success
- **P2:** Medium - important but not blocking
- **P3:** Low - nice-to-have, can defer if needed

## Dependencies

```
TICKET-001 (Foundation)
    ↓
TICKET-002 (Parallel Creation)
    ↓
TICKET-003 (Validation)
    ↓
    ├─→ TICKET-004 (Cut Bug Fix) ─┐
    └─→ TICKET-005 (Dual Extract) ─┤
                                    ↓
    ┌─→ TICKET-006 (State Extract) ─┤
    └─→ TICKET-007 (Constraint Fix) ┤
                                    ↓
           TICKET-008 (Integration Tests)
                    ↓
    ┌─→ TICKET-009 (Benchmarks) ────┤
    └─→ TICKET-010 (Cleanup) ───────┘
```

## Success Metrics

### Technical
- ✅ 100% test pass rate
- ✅ Zero invalid bounds in any scenario
- ✅ 50% faster cut generation
- ✅ 60% faster dual extraction
- ✅ < 5% overhead in full SDDP
- ✅ No memory overhead

### Quality
- ✅ Zero clippy warnings
- ✅ All public APIs documented
- ✅ Comprehensive CHANGELOG
- ✅ Code review approved

### Project
- ✅ All tickets completed
- ✅ 8-week timeline met
- ✅ No critical regressions

## Getting Started

### For Implementers

1. Read the [Architecture Analysis](../docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md)
2. Review the [Sprint Plan](SPRINT_PLAN.md)
3. Start with [TICKET-001](TICKET-001-design-lag-structures.md)
4. Follow the dependency chain
5. Run validation framework after each ticket

### For Reviewers

1. Understand the problem from [BUG_FIX_PAR_LOWER_BOUND.md](../BUG_FIX_PAR_LOWER_BOUND.md)
2. Review architecture rationale in analysis document
3. Check each ticket's acceptance criteria are met
4. Verify tests are comprehensive
5. Ensure documentation is clear

### For Testers

1. Enable `migration_validation` feature for Sprints 1-3
2. Run full test suite after each ticket
3. Run Example 07 with multiple seeds
4. Monitor performance benchmarks
5. Validate all edge cases covered

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Bug fix doesn't resolve invalid bounds | Low | High | Comprehensive tests in TICKET-008 |
| Performance regression | Very Low | Medium | Benchmarks in TICKET-009 |
| Migration introduces new bugs | Low | High | Validation framework in TICKET-003 |
| Timeline overrun | Medium | Medium | 2-week buffer in 8-week estimate |

## Communication

### Daily
- Standup updates on ticket progress
- Blocker identification and escalation

### Weekly
- Sprint progress report
- Metrics dashboard update

### Sprint Boundaries
- Sprint review: demo functionality
- Sprint retro: lessons learned
- Sprint planning: refine upcoming tickets

## Post-Epic

After completion:
1. ✅ Present architecture changes to team
2. ✅ Update onboarding documentation
3. ✅ Monitor production for issues
4. ✅ Document lessons learned
5. ✅ Plan follow-up improvements

---

## Questions?

- **Architecture:** See [ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md](../docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md)
- **Bug Details:** See [BUG_FIX_PAR_LOWER_BOUND.md](../BUG_FIX_PAR_LOWER_BOUND.md)
- **Sprint Planning:** See [SPRINT_PLAN.md](SPRINT_PLAN.md)
- **Specific Ticket:** See individual ticket files

---

**Epic Status:** Planned  
**Start Date:** TBD  
**Target Completion:** TBD + 8 weeks  
**Total Effort:** 29 story points  
**Confidence:** High (85%)
