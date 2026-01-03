# Sprint 1: CPU Profiling

> **Epic**: [Epic 2: CPU & Execution Profiling](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: 🟢 Completed

---

## Goals

- Wrap `perf record/script/report` with safe defaults
- Generate FlameGraph SVGs automatically from perf data
- Produce structured CPU collector output (JSON)
- Enable differential flamegraphs for comparisons
- Document perf setup and limitations (WSL2 vs bare-metal)

---

## Tickets

| ID | Title | Points | Status | Dependencies |
|----|-------|--------|--------|--------------|
| T-014 | Implement perf record wrapper | 5 | ✅ | T-007 |
| T-015 | Implement FlameGraph integration | 5 | ✅ | T-014 |
| T-016 | Implement CPU collector | 5 | ✅ | T-014, T-015, T-007 |
| T-017 | Parse perf report for hotspots | 3 | ✅ | T-014 |
| T-018 | Implement differential flamegraph | 3 | ✅ | T-015, T-016 |
| T-019 | Document perf setup and limitations | 2 | ✅ | All above |

**Total Points**: 23

---

## Dependencies

- **From Previous Sprint**: Epic 1 Sprint 2 (collector base, storage, reporter)
- **To Next Epics**: Enables Visualization dashboards (Epic 5)

---

## Risks

- perf permissions/capabilities may block sampling on some systems
  - *Mitigation*: Document setup, fallback to sudo, warn users
- FlameGraph scripts may be missing
  - *Mitigation*: Add install checks and docs

---

## Definition of Done

- [x] All tickets complete
- [x] `powers-profile run --collectors cpu` produces perf data + flamegraph
- [x] Hotspot JSON generated
- [x] Differential flamegraph supported in `compare`
- [x] Documentation updated for perf setup
