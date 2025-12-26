# Sprint 1: Solver Buffer Reuse Implementation

## Goals

- Add buffer-into methods to `Solution` and `Basis` structs
- Update `realize_and_solve()` to reuse preallocated buffers
- Eliminate ~4.9 GB of allocation churn during training

## Status

**Status**: ✅ Complete (2025-12-26)

## Tickets

| ID | Title | Points | Dependencies | Status |
|----|-------|--------|--------------|--------|
| 001 | Add Solution::with_capacity and get_solution_into | 2 | None | ✅ Complete |
| 002 | Add Basis::with_size and get_basis_into | 2 | None | ✅ Complete |
| 003 | Update realize_and_solve to use buffer-into pattern | 1 | 001, 002 | ✅ Complete |

## Definition of Done

- [x] All tickets complete
- [x] `cargo test` passes
- [x] Examples 01 and 07 produce correct results (tests pass)
- [ ] Memory profiling shows no per-solve allocations (to be validated)
- [x] No performance regression
