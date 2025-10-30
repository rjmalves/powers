# CLEANUP-001: Remove Unimplemented BaseNoiseMethod Variants

## Context

The `BaseNoiseMethod` enum in `src/base_noise.rs` contains three unimplemented variants (`KMeans`, `QuasiMonteCarlo`, `LatinHypercube`) that are marked with `#[allow(dead_code)]` and TODO comments. These variants add unused API surface without providing value and should be removed to simplify the codebase.

**Current State** (lines 30, 36, 40):
```rust
#[allow(dead_code)]
KMeans { clusters: usize },  // TODO
QuasiMonteCarlo,             // TODO
LatinHypercube,              // TODO
```

**Risk Level**: Medium (API change but likely no external users)

## Acceptance Criteria

- [ ] `BaseNoiseMethod` enum simplified to only `Standard` variant
- [ ] All `#[allow(dead_code)]` attributes removed from the enum
- [ ] All TODO comments removed
- [ ] No compiler warnings introduced
- [ ] All existing tests pass
- [ ] CHANGELOG.md updated with breaking change note (if enum is public)

## Tasks

### Implementation
- [ ] Remove `KMeans`, `QuasiMonteCarlo`, and `LatinHypercube` variants from `BaseNoiseMethod` enum
- [ ] Remove associated `#[allow(dead_code)]` attributes
- [ ] Check if enum has any match statements that need updating
- [ ] Verify no serialization/deserialization code depends on removed variants

### Testing
- [ ] Run `cargo test --workspace` to ensure no breakage
- [ ] Run `cargo build --workspace --release` to verify clean build
- [ ] Check if any examples reference the removed variants

### Documentation
- [ ] Update doc comments on `BaseNoiseMethod` if they reference removed variants
- [ ] Add CHANGELOG.md entry: "**Breaking**: Removed unimplemented BaseNoiseMethod variants (KMeans, QuasiMonteCarlo, LatinHypercube)"
- [ ] Check if INPUT-SPECIFICATION.md references these variants

## Technical Notes

**Location**: `src/base_noise.rs`

**Expected Result**:
```rust
pub enum BaseNoiseMethod {
    Standard,
}
```

**Verification**: After removal, search codebase for references to removed variants:
```bash
rg "KMeans|QuasiMonteCarlo|LatinHypercube" src/
```

## Dependencies

- Blocked by: None
- Blocks: None
- Related: CLEANUP-002 (validates overall dead code removal strategy)

## Estimated Effort

**0.5 story points** (2-3 hours, confidence: high)

Simple removal with validation. Most time spent on verification and documentation.
