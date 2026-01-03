# Sprint 1: Scalability Analysis

> **Epic**: [Epic 4: Parallelism & Scalability Analysis](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## Goals

- Automate scaling runs across thread counts
- Compute speedup, efficiency, and Amdahl estimates
- Detect contention indicators using perf
- Output parallel collector JSON and CLI summaries
- Prepare for high core counts (up to 192 threads)

---

## Tickets

| ID | Title | Points | Status | Dependencies |
|----|-------|--------|--------|--------------|
| T-026 | Implement scaling test runner | 5 | ⬜ | T-007, T-010 |
| T-027 | Implement speedup/efficiency calculation | 3 | ⬜ | T-026 |
| T-028 | Implement Amdahl estimation | 3 | ⬜ | T-027 |
| T-029 | Implement contention detection | 5 | ⬜ | T-026, T-014 |
| T-030 | Create parallel collector | 3 | ⬜ | T-026, T-027, T-028, T-029 |
| T-031 | Add scaling CLI summary | 2 | ⬜ | T-030 |
| T-032 | Document multi-socket preparation | 2 | ⬜ | All above |

**Total Points**: 23

---

## Dependencies

- **From Previous Epics**: CPU collector (perf) and storage/history from Epics 1-2
- **To Next Epics**: Visualization (scaling charts), documentation

---

## Risks

- Scaling variability due to system noise
  - *Mitigation*: Multiple runs and median aggregation
- perf contention detection may require elevated permissions
  - *Mitigation*: Document requirements, provide opt-out

---

## Definition of Done

- [ ] All tickets complete
- [ ] `powers-profile scaling --threads ...` produces scaling JSON
- [ ] Efficiency and Amdahl metrics computed
- [ ] Contention metrics captured when available
- [ ] Docs updated for high-core systems
