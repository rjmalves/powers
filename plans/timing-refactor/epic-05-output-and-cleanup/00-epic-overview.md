# Epic 5: Output & Cleanup

> **Duration**: 1 week  
> **Depends on**: Epic 4  
> **Enables**: None (final epic)

## Summary

Update output writers to use the new timing types, clean up any remaining legacy code, update documentation, and verify the refactor is complete.

## Scope

### Included

- Update `output/csv/training.rs` to use new timing output types
- Update `output/parquet/writer.rs` to use new timing output types
- Update `output/parquet/schemas.rs` if needed
- Remove dead code (unused timing imports, etc.)
- Update documentation in timing module
- Update CHANGELOG.md
- Clean up old timing types from `timing/metrics.rs`
- Final verification that all legacy timing code is removed

### Excluded

- N/A (final epic)

## Dependencies

- **Requires**: Epic 4 complete
- **Enables**: Feature complete

## Acceptance Criteria

- [ ] CSV output uses new `ForwardTimingOutput`, `BackwardTimingOutput`
- [ ] Parquet output uses new timing types
- [ ] Output field names match specification (some removed per plan)
- [ ] No unused imports or dead code
- [ ] `cargo clippy` clean
- [ ] `timing/metrics.rs` cleaned up (old types removed)
- [ ] CHANGELOG.md updated
- [ ] All tests passing
- [ ] No performance regression (manual verification)

## Technical Approach

1. Update CSV writer to read from new timing output structs
2. Update Parquet writer similarly
3. Remove old timing types from `timing/metrics.rs`
4. Run `cargo clippy` to find dead code
5. Update documentation
6. Final grep verification

## Files to Modify

| File | Changes |
|------|---------|
| `src/output/csv/training.rs` | Update field access |
| `src/output/parquet/writer.rs` | Update field access |
| `src/output/parquet/schemas.rs` | Update if needed |
| `src/timing/metrics.rs` | Remove old types |
| `src/timing/mod.rs` | Clean up exports |
| `CHANGELOG.md` | Document changes |

## Sprint Breakdown

### Sprint 1: Output & Cleanup

| Ticket | Title | Points |
|--------|-------|--------|
| T-023 | Update CSV training output | 2 |
| T-024 | Update Parquet training output | 2 |
| T-025 | Remove old timing/metrics.rs types | 2 |
| T-026 | Dead code cleanup | 1 |
| T-027 | Update documentation | 2 |
| T-028 | Final verification | 1 |

**Total**: 10 points (~1 week)

---

## Final Verification Checklist

Run these commands to verify the refactor is complete:

```bash
# No Instant::now in sddp/mod.rs
grep -n "Instant::now" src/sddp/mod.rs
# Expected: no output

# No elapsed() in sddp/mod.rs
grep -n "\.elapsed()" src/sddp/mod.rs
# Expected: no output

# No legacy timing structs
grep -n "ForwardPassTiming\|BackwardPassTiming\|ForwardPassTimingAccumulator" src/sddp/mod.rs
# Expected: no output

# All timing types in timing module
grep -rn "struct.*Timing" src/ | grep -v "src/timing" | grep -v test
# Expected: only RealizeUncertaintiesTiming in subproblem.rs

# Clippy clean
cargo clippy --all-targets
# Expected: no warnings

# All tests pass
cargo test
# Expected: all pass
```
