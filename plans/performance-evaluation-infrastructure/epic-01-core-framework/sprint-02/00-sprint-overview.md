# Sprint 2: First Collector

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Duration**: 1.5 weeks
> **Status**: 🟢 Completed

---

## Goals

- Implement the Collector base class and interface
- Create the first working collector (timing)
- Implement JSON export and run storage
- Deliver working `powers-profile run` command

---

## Tickets

| ID | Title | Points | Status | Dependencies |
|----|-------|--------|--------|--------------|
| T-007 | Implement Collector base class | 3 | ✅ | T-001, T-003 |
| T-008 | Implement timing collector | 5 | ✅ | T-007, T-004, T-005, T-006 |
| T-009 | Implement JSON reporter | 3 | ✅ | T-003, T-007 |
| T-010 | Implement run storage and history | 3 | ✅ | T-003, T-005 |
| T-011 | Implement summary command | 2 | ✅ | T-009, T-010 |
| T-012 | Add unit tests for core modules | 5 | ✅ | T-007, T-008, T-009 |
| T-013 | Create quick start README | 2 | ✅ | All above |

**Total Points**: 23

---

## Dependencies

- **From Previous Sprint**: T-001 through T-006 complete
- **To Next Sprint (Epic 2)**: Collector base class enables CPU collector

---

## Risks

- Timing extraction from POWE.RS output may need format adjustments
  - *Mitigation*: Use flexible regex patterns, document expected format

---

## Definition of Done

- [x] All tickets complete
- [x] `powers-profile run --collectors timing` works end-to-end
- [x] JSON output saved to `profiling_results/runs/`
- [x] `powers-profile summary` shows last run
- [x] `powers-profile history` lists past runs
- [x] All tests pass
- [x] README documents usage
