# Sprint 1: Output & Cleanup

> **Epic**: [Epic 5: Output & Cleanup](../00-epic-overview.md)  
> **Duration**: 1 week  
> **Total Points**: 10

## Goals

- Update CSV training output to use new timing types
- Update Parquet training output to use new timing types
- Remove old timing types from `timing/metrics.rs`
- Clean up dead code and unused imports
- Update documentation
- Final verification that refactor is complete

## Tickets

| ID | Title | Points | Dependencies | Assignable |
|----|-------|--------|--------------|------------|
| T-023 | [Update CSV training output](./ticket-023-update-csv-output.md) | 2 | Epic 4 | Yes |
| T-024 | [Update Parquet training output](./ticket-024-update-parquet-output.md) | 2 | Epic 4 | Yes |
| T-025 | [Remove old timing/metrics.rs types](./ticket-025-remove-old-metrics.md) | 2 | T-023, T-024 | Yes |
| T-026 | [Dead code cleanup](./ticket-026-dead-code-cleanup.md) | 1 | T-025 | Yes |
| T-027 | [Update documentation](./ticket-027-update-documentation.md) | 2 | T-026 | Yes |
| T-028 | [Final verification](./ticket-028-final-verification.md) | 1 | T-027 | Yes |

## Dependencies

- **From Epic 4**: `IterationResult` uses `IterationTimingOutput`
- **From Epic 1**: All new timing types in `src/timing/`

## Key Files

Files to modify:
- `src/output/csv/training.rs` - CSV training output writer
- `src/output/parquet/writer.rs` - Parquet training output
- `src/output/parquet/schemas.rs` - Parquet schema definitions
- `src/timing/metrics.rs` - Old timing types to remove
- `src/timing/mod.rs` - Clean up exports
- `CHANGELOG.md` - Document changes

Files to read before starting:
- `src/timing/output.rs` - New output types
- `src/sddp/mod.rs` - `IterationResult` struct
- `plans/timing-refactor/00-master-plan.md` - Output schema mapping

## Output Field Changes

### CSV Fields

| Old Field | New Field | Status |
|-----------|-----------|--------|
| `forward_saa_sampling_ms` | Same | Keep |
| `forward_model_preprocessing_ms` | Same | Keep |
| `forward_solver_ms` | Same | Keep |
| `forward_model_postprocessing_ms` | Same | Keep |
| `forward_postprocessing_ms` | Same | Keep |
| `forward_total_ms` | Same | Keep |
| `backward_preprocessing_ms` | - | **Remove** |
| `backward_model_preprocessing_ms` | Same | Keep |
| `backward_solver_ms` | Same | Keep |
| `backward_model_postprocessing_ms` | Same | Keep |
| `backward_cut_selection_ms` | Same | Keep |
| `backward_fcf_state_update_ms` | `backward_problem_update_ms` | **Combined** |
| `backward_cut_cloning_ms` | - | **Merged** |
| `backward_handler_application_ms` | - | **Merged** |
| `backward_total_ms` | Same | Keep |

### New Optional Fields

| New Field | Source | Purpose |
|-----------|--------|---------|
| `model_allocation_ms` | `timing.model_allocation` | Per-iteration model creation |
| `model_cleanup_ms` | `timing.model_cleanup` | Per-iteration model cleanup |
| `forward_parallel_wall_ms` | `timing.forward.parallel_wall` | External parallel view |
| `forward_parallel_overhead_ms` | `timing.forward.parallel_overhead` | Scheduling overhead |
| `forward_solver_max_ms` | `timing.forward.solver_max` | Load balance metric |

## Risks

- **Schema breaking change**: Removed fields may break downstream tools
- **Parquet schema mismatch**: Arrow schema must match new types

## Definition of Done

- [ ] All 6 tickets complete
- [ ] CSV output uses new timing types
- [ ] Parquet output uses new timing types
- [ ] `timing/metrics.rs` old types removed
- [ ] No unused imports or dead code
- [ ] `cargo clippy --all-targets` clean
- [ ] All tests passing
- [ ] CHANGELOG.md updated
- [ ] Documentation updated
