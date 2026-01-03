# Sprint 1: Memory Tools

> **Epic**: [Epic 3: Memory Profiling Suite](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## Goals

- Integrate valgrind DHAT, Massif, and Cachegrind
- Replace legacy RSS monitor with structured collector
- Unify memory collector output in JSON
- Provide comparison analysis across runs

---

## Tickets

| ID | Title | Points | Status | Dependencies |
|----|-------|--------|--------|--------------|
| T-020 | Implement DHAT collector | 5 | ⬜ | T-007, T-003 |
| T-021 | Implement Massif collector | 5 | ⬜ | T-007, T-003 |
| T-022 | Implement Cachegrind collector | 3 | ⬜ | T-007, T-003 |
| T-023 | Implement RSS monitor | 5 | ⬜ | T-007 |
| T-024 | Create unified memory collector | 3 | ⬜ | T-020, T-021, T-022, T-023 |
| T-025 | Add memory comparison analysis | 3 | ⬜ | T-024 |

**Total Points**: 24

---

## Dependencies

- **From Previous Epics**: Core framework collectors/reporters from Epic 1
- **To Next Epics**: Visualization dashboards (Epic 5)

---

## Risks

- Valgrind overhead makes runs slow
  - *Mitigation*: Support selective tools and sampling subsets
- Parsing valgrind outputs may vary by version
  - *Mitigation*: Use fixtures from target version (3.18+)

---

## Definition of Done

- [ ] All tickets complete
- [ ] `powers-profile run --collectors memory` runs selected tools
- [ ] JSON output includes DHAT, Massif, Cachegrind, RSS
- [ ] Comparison analysis works between runs
- [ ] Docs updated for valgrind usage
