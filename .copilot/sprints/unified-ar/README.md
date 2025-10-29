# Unified AR Model Refactoring Sprint

This directory contains the complete sprint plan for refactoring the AR (AutoRegressive) model representation in POWE.RS.

## Overview

The goal of this epic is to **simplify the AR model representation** by eliminating conditional logic and using a unified, constraint-based formulation for all inflow models (both independent and autoregressive).

**Key Insight:** AR(0) = Independent, allowing a single code path with variable coefficients.

## Quick Links

- **[SPRINT_STATUS.md](SPRINT_STATUS.md)** - Current progress, dependencies, and quality gates
- **[UNIFIED_AR_ROADMAP.md](../../UNIFIED_AR_ROADMAP.md)** - Detailed architectural analysis and design

## Sprint Structure

### Sprint 1: Foundation (8 days)

Build UnifiedInflowModel infrastructure.

- [TICKET-001](TICKET-001-create-unified-inflow-model-struct.md) - Create struct (2d)
- [TICKET-002](TICKET-002-implement-constraint-generation.md) - LP constraints (3d)
- [TICKET-003](TICKET-003-implement-lag-buffer-management.md) - Lag buffer (3d)

### Sprint 2: Subproblem Refactor (10 days)

Integrate unified model and eliminate conditionals.

- [TICKET-004](TICKET-004-refactor-variables-struct.md) - Variables struct (2d)
- [TICKET-005](TICKET-005-refactor-constraints-struct.md) - Constraints struct (1d)
- [TICKET-006](TICKET-006-update-realization-struct.md) - Realization struct (1d)
- [TICKET-007](TICKET-007-integrate-unified-model-into-subproblem.md) - Subproblem integration (2d)
- [TICKET-008](TICKET-008-simplify-realize-uncertainties.md) - Simplify realize_uncertainties (3d) **[CRITICAL]**
- [TICKET-009](TICKET-009-implement-update-from-trajectory.md) - Update from trajectory (2d)

### Sprint 3: Cleanup (5 days)

Remove obsolete code and update tests.

- [TICKET-010](TICKET-010-refactor-state-trait-interface.md) - Refactor State trait interface (1d)
- [TICKET-011](TICKET-011-remove-observation-space-par.md) - Remove observation PAR (2d)
- [TICKET-012](TICKET-012-update-integration-tests.md) - Update tests/examples (2d)

### Sprint 4: Optimization (4 days)

Performance optimizations.

- [TICKET-013](TICKET-013-precompute-seasonal-cache.md) - Seasonal cache (2d)
- [TICKET-014](TICKET-014-optimize-batch-rhs-updates.md) - Batch RHS updates (2d)

## Expected Benefits

- **15-30% performance improvement** in SDDP algorithm
- **Simplified codebase** with zero conditionals in hot paths
- **Better maintainability** through clear separation of concerns
- **Improved correctness** with explicit space transformations
- **Flexibility** - Users can choose StorageState (simpler) or StorageAndInflowState (richer cuts)

## Getting Started

1. Read [SPRINT_STATUS.md](SPRINT_STATUS.md) for current progress
2. Read the architectural design in [UNIFIED_AR_ROADMAP.md](../../UNIFIED_AR_ROADMAP.md)
3. Pick the next "Not Started" ticket from Sprint 1
4. Follow the ticket's implementation checklist
5. Update SPRINT_STATUS.md with progress

## Ticket Format

Each ticket contains:

- **Context** - Why this work is needed
- **Acceptance Criteria** - Specific, measurable outcomes
- **Tasks** - Implementation, testing, and documentation checklists
- **Technical Notes** - Design patterns, edge cases, performance considerations
- **Dependencies** - Blocked by / Blocks relationships
- **Validation Checklist** - Definition of done

## Critical Path

**Minimum completion time: 22 days** (with parallelization)

Key bottleneck: **TICKET-008** (realize_uncertainties) - This is where the major complexity reduction happens.

## Contact

For questions about this sprint:

- Architecture decisions: See [UNIFIED_AR_ROADMAP.md](../../UNIFIED_AR_ROADMAP.md)
- Implementation details: See individual ticket files
- Progress tracking: Update [SPRINT_STATUS.md](SPRINT_STATUS.md)
