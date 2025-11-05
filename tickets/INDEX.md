# Ticket Index

Quick navigation for all epic documents and tickets.

## 📋 Planning Documents

| Document | Description | Size |
|----------|-------------|------|
| [README.md](README.md) | Main entry point with overview and navigation | 7.9 KB |
| [EPIC_EXPLICIT_LAG_SEPARATION.md](EPIC_EXPLICIT_LAG_SEPARATION.md) | Epic definition and success criteria | 3.8 KB |
| [SPRINT_PLAN.md](SPRINT_PLAN.md) | Detailed 4-sprint breakdown and timeline | 11 KB |
| [SUMMARY.md](SUMMARY.md) | Executive summary of all tickets created | 7.8 KB |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Developer quick reference for implementation | 8.7 KB |
| [ticket-tree.txt](ticket-tree.txt) | Visual dependency tree (ASCII art) | 4.5 KB |

## 🎯 Sprint 1: Foundation (Week 1-2)

**Goal:** Establish new architecture with validation framework

| Ticket | Title | Priority | Effort | File |
|--------|-------|----------|--------|------|
| 001 | Design and Implement Lag Variable Data Structures | P0 | 3 SP | [TICKET-001](TICKET-001-design-lag-structures.md) |
| 002 | Add Parallel Lag Variable Creation in Subproblem | P0 | 5 SP | [TICKET-002](TICKET-002-parallel-variable-creation.md) |
| 003 | Implement Validation Framework for Migration | P1 | 3 SP | [TICKET-003](TICKET-003-validation-framework.md) |

**Sprint Total:** 11 story points

## 🐛 Sprint 2: Critical Bug Fixes (Week 3-4)

**Goal:** Fix the critical bug and migrate high-priority consumers

| Ticket | Title | Priority | Effort | File |
|--------|-------|----------|--------|------|
| 004 | Fix Critical Bug in add_cut_constraint_to_model | P0 | 5 SP | [TICKET-004](TICKET-004-fix-cut-generation-bug.md) |
| 005 | Migrate Dual Extraction to Use Explicit Structures | P1 | 3 SP | [TICKET-005](TICKET-005-migrate-dual-extraction.md) |

**Sprint Total:** 8 story points

## 🔄 Sprint 3: Complete Migration (Week 5-6)

**Goal:** Migrate remaining consumers and validate with comprehensive tests

| Ticket | Title | Priority | Effort | File |
|--------|-------|----------|--------|------|
| 006 | Update State Lag Extraction Methods | P1 | 3 SP | [TICKET-006](TICKET-006-update-state-extraction.md) |
| 007 | Migrate Lag Constraint Fixing Logic | P2 | 2 SP | [TICKET-007](TICKET-007-migrate-constraint-fixing.md) |
| 008 | Add Comprehensive Integration Tests | P1 | 5 SP | [TICKET-008](TICKET-008-integration-tests.md) |

**Sprint Total:** 10 story points

## ✅ Sprint 4: Validation & Cleanup (Week 7-8)

**Goal:** Validate performance improvements and finalize refactoring

| Ticket | Title | Priority | Effort | File |
|--------|-------|----------|--------|------|
| 009 | Performance Benchmarking and Optimization | P2 | 3 SP | [TICKET-009](TICKET-009-performance-benchmarking.md) |
| 010 | Remove Deprecated Code and Update Documentation | P1 | 3 SP | [TICKET-010](TICKET-010-cleanup-documentation.md) |

**Sprint Total:** 6 story points

## 📊 Statistics

- **Total Tickets:** 10
- **Total Story Points:** 29
- **Estimated Duration:** 8 weeks
- **Planning Documents:** 6
- **Total Documentation:** ~122 KB

## 🔗 External References

- **Architecture Analysis:** `../docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md`
- **Bug Report:** `../BUG_FIX_PAR_LOWER_BOUND.md`
- **Source Code:** `../src/subproblem.rs`, `../src/state.rs`

## 📖 Reading Order

### For Project Managers
1. [SUMMARY.md](SUMMARY.md) - Executive overview
2. [EPIC_EXPLICIT_LAG_SEPARATION.md](EPIC_EXPLICIT_LAG_SEPARATION.md) - Business value
3. [SPRINT_PLAN.md](SPRINT_PLAN.md) - Detailed timeline

### For Developers
1. [README.md](README.md) - Overview and context
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Code patterns
3. Individual tickets in dependency order (001 → 002 → ...)

### For Reviewers
1. [EPIC_EXPLICIT_LAG_SEPARATION.md](EPIC_EXPLICIT_LAG_SEPARATION.md) - Goals
2. Individual tickets for detailed acceptance criteria
3. [SPRINT_PLAN.md](SPRINT_PLAN.md) - Success metrics

### For QA/Testers
1. [README.md](README.md) - Testing strategy
2. [TICKET-008](TICKET-008-integration-tests.md) - Comprehensive test requirements
3. [TICKET-003](TICKET-003-validation-framework.md) - Validation approach

## 🎯 Critical Path

The minimal set of tickets that must complete on time:

```
TICKET-001 → TICKET-002 → TICKET-004 → TICKET-008
```

If any critical path ticket is blocked, escalate immediately.

## 🔍 Quick Search

### By Priority
- **P0 (Critical):** 001, 002, 004
- **P1 (High):** 003, 005, 006, 008, 010
- **P2 (Medium):** 007, 009

### By Type
- **Infrastructure:** 001, 002, 003
- **Bug Fix:** 004
- **Migration:** 005, 006, 007
- **Testing:** 008
- **Performance:** 009
- **Cleanup:** 010

### By Sprint
- **Sprint 1:** 001, 002, 003
- **Sprint 2:** 004, 005
- **Sprint 3:** 006, 007, 008
- **Sprint 4:** 009, 010

## 📝 Ticket Template

Each ticket includes:
- **Context:** Why this work is needed
- **Acceptance Criteria:** Specific, measurable outcomes
- **Tasks:** Implementation, Testing, Documentation
- **Technical Notes:** Code examples and patterns
- **Dependencies:** Blockers and related work
- **Definition of Done:** Completion checklist

## 🚀 Getting Started

```bash
# Navigate to tickets directory
cd tickets/

# Read the overview
cat README.md

# View dependency tree
cat ticket-tree.txt

# Start with first ticket
cat TICKET-001-design-lag-structures.md
```

## 📧 Contacts

For questions about:
- **Project Management:** See [SPRINT_PLAN.md](SPRINT_PLAN.md)
- **Technical Details:** See individual tickets
- **Architecture:** See `../docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md`
- **Implementation Help:** See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

**Last Updated:** 2025-01-05  
**Status:** Ready for Review  
**Next Review:** Sprint 1 Planning Meeting
