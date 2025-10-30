# Code Cleanup Sprint Plan - Complete Implementation Tickets

## Overview

This directory contains complete implementation tickets for the code cleanup plan documented in `CLEANING_PLAN.md`. The work is organized into 2 focused sprints (Priority 1-3) with Priority 4 deferred to future feature work.

**Created**: October 30, 2025  
**Based on**: CLEANING_PLAN.md comprehensive code analysis  
**Total Effort**: 8.5 story points (approximately 3-4 days)

## Sprint Structure

### Sprint 1: Critical TODOs & Dead Code (Priority 1-2)
**Duration**: 2 weeks  
**Effort**: 7 story points  
**Focus**: Resolve all TODOs, remove dead code, improve code quality

| Ticket | Title | Story Points | Risk |
|--------|-------|--------------|------|
| CLEANUP-001 | Remove Unimplemented BaseNoiseMethod Variants | 0.5 | Medium |
| CLEANUP-002 | Verify and Resolve Stochastic Process TODOs | 1.0 | HIGH |
| CLEANUP-003 | Resolve Input Validation TODOs | 1.5 | Medium |
| CLEANUP-004 | Document Low-Priority TODOs in Issue Tracker | 0.5 | Low |
| CLEANUP-005 | Audit and Remove HighsBasisStatus Enum | 0.5 | Low |
| CLEANUP-006 | Verify SDDP TerminationReason Variants Are Reachable | 0.5 | Low-Med |
| CLEANUP-007 | Audit Remaining #[allow(dead_code)] Attributes | 1.5 | Low |
| CLEANUP-008 | Remove Redundant "What" Comments | 1.0 | Very Low |
| CLEANUP-009 | Move Performance Comments to Module Documentation | 1.0 | Very Low |
| CLEANUP-010 | Create Sprint 1 Summary and Validation Report | 0.5 | N/A |

**Key Outcomes**:
- ✅ Zero TODOs remaining (except issue references)
- ✅ All dead code removed or documented
- ✅ Comments cleaned up
- ✅ Performance documentation organized

### Sprint 2: Documentation Reorganization (Priority 3-4)
**Duration**: 2 weeks  
**Effort**: 3.75 story points  
**Focus**: Improve documentation organization and readability

| Ticket | Title | Story Points | Risk |
|--------|-------|--------------|------|
| CLEANUP-011 | Consolidate Mathematical Derivation Comments in Tests | 1.5 | Very Low |
| CLEANUP-012 | Extract lognormal3.rs Tutorial to Documentation | 1.0 | Low |
| CLEANUP-013 | Extract solver.rs Comparison to Architecture Docs | 0.5 | Low |
| CLEANUP-014 | Condense SDDP Module-Level Documentation | 0.25 | Very Low |

**Key Outcomes**:
- ✅ Test documentation organized with derivations in doc comments
- ✅ Tutorial-level docs moved to docs/reference/
- ✅ Architecture documentation created in docs/architecture/
- ✅ Module docs concise and focused

### Priority 5: Large Module Refactoring (Deferred)

**Intentionally deferred to future feature work** as recommended in CLEANING_PLAN.md:
- Split mod.rs (5221 LOC) into forward_pass.rs, backward_pass.rs, simulation.rs
- Split subproblem.rs (3358 LOC) into subproblem/model.rs and solver.rs
- Extract unified_noise_spec.rs conversion logic
- Split state.rs into state/storage.rs and state/inflow.rs
- Extract builder.rs validation logic

**Rationale**: Large-scale refactoring should be done in context of actual feature development, not as standalone cleanup.

## Ticket Details

Each ticket follows the sprint planner methodology and includes:

### Standard Ticket Structure
- **Context**: Why this work is needed
- **Acceptance Criteria**: Specific, measurable outcomes
- **Tasks**: Broken down into:
  - Implementation steps
  - Testing requirements (unit, integration, performance)
  - Documentation updates
- **Technical Notes**: Implementation guidance and considerations
- **Dependencies**: Blocked by, blocks, related tickets
- **Estimated Effort**: Story points with confidence level

### Quality Requirements (All Tickets)
Every ticket must pass:
- [ ] `cargo fmt -- --check` (zero warnings)
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` (zero warnings)
- [ ] `cargo test --workspace` (all tests pass)
- [ ] `cargo build --workspace --release` (clean build)
- [ ] Documentation builds correctly

## How to Use These Tickets

### For Developers

1. **Read the ticket thoroughly** - Understand context and acceptance criteria
2. **Create feature branch** - Branch from main/ar-model
3. **Follow the task checklist** - Work through tasks systematically
4. **Run pre-checks frequently** - Catch issues early
5. **Update CHANGELOG.md** - Document user-facing changes
6. **Request review** - Ensure quality before merge

### For Project Managers

1. **Assign tickets based on priority** - Sprint 1 before Sprint 2
2. **Monitor dependencies** - Respect "blocked by" relationships
3. **Track velocity** - Use story points to measure progress
4. **Review acceptance criteria** - Verify completion before closing
5. **Collect metrics** - Document actual effort vs estimates

### For Reviewers

1. **Check acceptance criteria** - All must be met
2. **Verify pre-checks pass** - Run validation commands
3. **Review code quality** - No shortcuts taken
4. **Check documentation** - CHANGELOG.md and docs updated
5. **Verify no regressions** - Tests and benchmarks pass

## Success Metrics

### Sprint 1 Success Criteria
- [ ] All 10 tickets completed
- [ ] Zero TODOs remaining in codebase
- [ ] Zero `#[allow(dead_code)]` without documentation
- [ ] Zero build warnings
- [ ] All tests passing
- [ ] No performance regressions

### Sprint 2 Success Criteria
- [ ] All 4 tickets completed
- [ ] Documentation structure established
- [ ] Module docs concise (10-15 lines)
- [ ] docs/reference/ and docs/architecture/ created
- [ ] Test derivations in doc comments
- [ ] All documentation builds correctly

### Overall Success Criteria
- [ ] Codebase quality improved measurably
- [ ] No loss of valuable information
- [ ] Code more maintainable and readable
- [ ] Documentation well-organized
- [ ] Zero functional regressions
- [ ] Team velocity and process documented

## Project Context

This cleanup work supports the **POWE.RS** project, a high-performance Rust implementation of the SDDP algorithm. Key characteristics:

- **Performance-Critical**: Changes must not degrade algorithmic performance
- **Numerical Correctness**: Careful testing for correctness and stability required
- **Zero Warnings Policy**: Build must pass with `-D warnings` (strictly enforced)
- **Extensive Testing**: 34% of codebase is tests (maintain high coverage)
- **Production-Ready**: Already in excellent condition (9.5/10 quality score)

## Repository Structure

```
.copilot/
  sprints/
    cleanup/
      README.md                      # This file
      sprint-1/
        CLEANUP-001-*.md            # Sprint 1 tickets (10 tickets)
        ...
      sprint-2/
        README.md                   # Sprint 2 overview
        CLEANUP-011-*.md            # Sprint 2 tickets (4 tickets)
        ...
```

## References

- **CLEANING_PLAN.md**: Comprehensive analysis that generated these tickets
- **.copilot/agents/sprint-planner.md**: Methodology used to create tickets
- **docs/reference/INPUT-SPECIFICATION.md**: Input format documentation
- **CHANGELOG.md**: User-facing change log
- **CONTRIBUTING.md**: Contribution guidelines

## Timeline

**Recommended Schedule**:

```
Week 1-2: Sprint 1 (Critical TODOs & Dead Code)
  - High-priority TODO resolution (CLEANUP-001 through CLEANUP-004)
  - Dead code removal (CLEANUP-005 through CLEANUP-007)
  - Comment cleanup (CLEANUP-008, CLEANUP-009)
  - Sprint validation (CLEANUP-010)

Week 3-4: Sprint 2 (Documentation Reorganization)
  - Test documentation (CLEANUP-011)
  - Extract tutorials (CLEANUP-012, CLEANUP-013)
  - Condense module docs (CLEANUP-014)
  - Final validation and review

Week 5: Buffer and code review
  - Address review feedback
  - Final integration testing
  - Documentation review
  - Merge to main
```

**Note**: Timelines assume single developer working part-time. Adjust based on team size and availability.

## Questions or Issues?

- **Missing context?** Check CLEANING_PLAN.md for detailed analysis
- **Unclear acceptance criteria?** Refer to sprint-planner.md methodology
- **Need help?** Consult with maintainers before making assumptions
- **Found new issues?** Create new tickets following the template

## Next Steps

1. **Review Sprint 1 tickets** - Understand scope and priorities
2. **Create feature branch** - Start with Sprint 1
3. **Begin with CLEANUP-001** - Quick win to establish workflow
4. **Work systematically** - Complete tickets in dependency order
5. **Track progress** - Update ticket status as you work
6. **Validate frequently** - Run pre-checks after each ticket
7. **Document learnings** - Update CLEANUP-010 with insights
8. **Proceed to Sprint 2** - After Sprint 1 is complete and merged

---

**Prepared by**: GitHub Copilot (Sprint Planner Agent)  
**Date**: October 30, 2025  
**Version**: 1.0  
**Status**: Ready for implementation
