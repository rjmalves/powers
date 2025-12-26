# Sprint 1: Arc-Wrapped Cut Implementation

## Goals

- Change `BendersCutPool` to store `Arc<BendersCut>`
- Update all cut creation and usage sites
- Eliminate ~80 MB of transient allocation from cloning

## Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| 001 | Update BendersCutPool to use Arc<BendersCut> | 2 | None |
| 002 | Update cut consumers to work with Arc | 3 | 001 |

## Definition of Done

- [ ] All tickets complete
- [ ] `cargo test` passes
- [ ] Examples 01 and 07 produce correct results
- [ ] No `.clone()` on BendersCut in hot paths
