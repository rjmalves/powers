# [T-024] Update Parquet training output

> **Epic**: [Epic 5: Output & Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: Epic 4 complete  
> **Blocks**: [T-025](./ticket-025-remove-old-metrics.md)

## Files to Read Before Starting

- `src/output/parquet/writer.rs` - Parquet training writer
- `src/output/parquet/schemas.rs` - Arrow schema definitions
- `src/timing/output.rs` - New timing output types
- `src/sddp/mod.rs` - Updated `IterationResult` struct

## Context

### Background

The Parquet writer outputs training iteration data with timing columns. After Epic 4, the timing types changed, requiring updates to both the schema definitions and the data writing code.

### Current State

The Parquet writer reads from old timing struct:
```rust
result.forward_timing.saa_sampling_time
result.backward_timing.backward_preprocessing_time
```

### Target State

```rust
result.timing.forward.saa_sampling
// backward_preprocessing removed
result.timing.backward.problem_update
```

## Specification

### Schema Changes

The Arrow schema in `schemas.rs` needs updates:

| Old Column | New Column | Status |
|------------|------------|--------|
| `forward_saa_sampling_ms` | Same | Keep |
| `backward_preprocessing_ms` | - | **Remove** |
| `backward_fcf_state_update_ms` | `backward_problem_update_ms` | **Rename** |
| `backward_cut_cloning_ms` | - | **Remove** |
| `backward_handler_application_ms` | - | **Remove** |

### Data Writing Changes

Update all field accesses similar to T-023.

## Acceptance Criteria

- [ ] Arrow schema updated in `schemas.rs`
- [ ] Parquet writer compiles with new timing types
- [ ] Removed columns no longer in schema
- [ ] `cargo test -p powers-rs parquet` passes
- [ ] Output Parquet has correct schema
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Update Arrow schema

In `src/output/parquet/schemas.rs`, update the training schema:

```rust
// Update field definitions
// Remove: backward_preprocessing_ms
// Remove: backward_cut_cloning_ms
// Remove: backward_handler_application_ms
// Rename: backward_fcf_state_update_ms -> backward_problem_update_ms
```

### Step 2: Update writer

In `src/output/parquet/writer.rs`, update data collection:

```rust
// OLD
.with_column(
    "backward_preprocessing_ms",
    results.iter().map(|r| r.backward_timing.backward_preprocessing_time.as_millis() as i64)
)

// NEW - removed column

// OLD
.with_column(
    "backward_fcf_state_update_ms",
    results.iter().map(|r| r.backward_timing.fcf_state_update_time.as_millis() as i64)
)

// NEW
.with_column(
    "backward_problem_update_ms",
    results.iter().map(|r| r.timing.backward.problem_update.as_millis() as i64)
)
```

### Step 3: Update all field references

Apply same changes as T-023 for all timing field accesses.

### Pitfalls to Avoid

- ⚠️ Arrow schema must match data columns exactly
- ⚠️ Column order may matter for some readers
- ⚠️ Test with actual Parquet reader after changes

## Testing Requirements

### Unit Tests

- Test schema creation
- Test data writing with mock results

### Integration Tests

```bash
cargo test -p powers-rs parquet
cargo test -p powers-rs output
```

### Validation Tests

- Run training with Parquet output enabled
- Read output with pyarrow or similar to verify schema
- Verify data types and values

## Documentation Requirements

- [ ] Update schema documentation in `schemas.rs`
- [ ] Document column changes

## Effort Estimate

**Points**: 2  
**Confidence**: Medium  
**Rationale**: Schema changes require careful verification
