# Epic: Explicit Load/Inflow Lag Variable Separation

## Overview

**Priority:** HIGH  
**Status:** Planned  
**Estimated Duration:** 4 weeks  
**Confidence:** High

## Business Value

This epic addresses a critical bug in cut generation caused by implicit entity type handling in lag variables and constraints. The current unified approach uses heuristics to distinguish between load and inflow lag variables, leading to incorrect cut coefficients when loads and inflows have similar AR orders.

## Problem Statement

Current implementation stores all lag variables (loads + inflows) in a unified `Vec<Vec<usize>>` structure, erasing type information. This forces the code to use fragile heuristics to determine which variables correspond to which entity type, resulting in:

1. **Critical Bug:** `add_cut_constraint_to_model` incorrectly matches cut coefficients to lag variables
2. **Invalid Lower Bounds:** Lower bound exceeds simulation results in Example 07
3. **Fragile Code:** Future changes to AR models can break existing functionality
4. **Performance Issues:** O(n_entities) filtering instead of O(n_hydros) direct access

## Solution Approach

Separate lag variables and constraints into explicit type-safe structures:
- `LoadLagVariables` indexed by bus_id
- `InflowLagVariables` indexed by hydro_id
- `LoadLagConstraints` indexed by bus_id
- `InflowLagConstraints` indexed by hydro_id

This follows the successful pattern used in the `Realization` refactoring.

## Success Metrics

1. ✅ Cut generation correctly handles mixed AR orders (loads + inflows)
2. ✅ Lower bound ≤ simulation results in all test cases
3. ✅ No heuristic-based entity type detection in codebase
4. ✅ Performance improvement: ~2x faster cut generation
5. ✅ All existing tests pass
6. ✅ Code review feedback cites improved clarity

## Architecture Principles

1. **Make illegal states unrepresentable** - Type system prevents load/inflow confusion
2. **Align code with domain model** - Mathematical distinction reflected in types
3. **Explicit over implicit** - Clear intent over clever abstractions
4. **Fail fast** - Compile-time type errors vs runtime bugs

## Implementation Phases

### Phase 1: Foundation (Sprint 1)
- Design and implement new data structures
- Parallel implementation alongside existing code
- Validation framework to ensure consistency

### Phase 2: Migration (Sprint 2-3)
- Update critical bug fixes first
- Migrate performance-critical paths
- Update remaining consumers

### Phase 3: Cleanup (Sprint 4)
- Remove deprecated code
- Comprehensive testing
- Documentation updates

## Related Documents

- Architecture Analysis: `docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md`
- Bug Report: `BUG_FIX_PAR_LOWER_BOUND.md`

## Dependencies

None - self-contained refactoring

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing functionality | Medium | High | Parallel implementation with validation |
| Migration taking longer than estimated | Low | Medium | Phased approach allows incremental progress |
| Performance regression | Very Low | High | Benchmarks at each phase |
| Test coverage gaps | Low | Medium | Comprehensive test suite with edge cases |

## Tickets

- [TICKET-001] Design and implement lag variable data structures
- [TICKET-002] Add parallel lag variable creation in subproblem
- [TICKET-003] Implement validation framework for migration
- [TICKET-004] Fix critical bug in add_cut_constraint_to_model
- [TICKET-005] Migrate dual extraction to use explicit structures
- [TICKET-006] Update state lag extraction methods
- [TICKET-007] Migrate lag constraint fixing logic
- [TICKET-008] Add comprehensive integration tests
- [TICKET-009] Performance benchmarking and optimization
- [TICKET-010] Remove deprecated code and update documentation
