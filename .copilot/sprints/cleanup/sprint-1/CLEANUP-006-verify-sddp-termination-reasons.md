# CLEANUP-006: Verify SDDP TerminationReason Variants Are Reachable

## Context

The `TerminationReason` enum in `src/sddp/mod.rs` has two variants marked with `#[allow(dead_code)]`: `MaxIterations` (line 193) and `ConvergedGap` (line 197). These appear to be part of the public API but may not be constructed by all code paths in the `train()` function.

**Current State**:
```rust
pub enum TerminationReason {
    #[allow(dead_code)]
    MaxIterations,
    ConvergedStability { iterations_stable: usize },
    #[allow(dead_code)]
    ConvergedGap,
    // ... other variants
}
```

**Risk Level**: LOW-MEDIUM (API completeness)

## Acceptance Criteria

- [ ] All code paths in `train()` examined to identify which variants are returned
- [ ] Verification that `MaxIterations` and `ConvergedGap` are reachable OR confirmation they're unreachable
- [ ] If unreachable: Variants removed and CHANGELOG.md updated with breaking change
- [ ] If reachable: `#[allow(dead_code)]` attributes removed
- [ ] If partially implemented: Implementation completed or future ticket created
- [ ] Test coverage added for any newly confirmed reachable paths
- [ ] No compiler warnings after changes

## Tasks

### Investigation
- [ ] Locate `TerminationReason` enum definition in `src/sddp/mod.rs`
- [ ] Find all places where `TerminationReason` is constructed
- [ ] Trace `train()` function to identify all return paths:
  ```bash
  # Search for TerminationReason construction
  rg "TerminationReason::" src/sddp/
  ```
- [ ] Check configuration options to see if max iterations termination is configurable
- [ ] Check if gap-based convergence is implemented anywhere
- [ ] Review tests to see which termination reasons are tested:
  ```bash
  rg "TerminationReason" tests/
  ```

### Analysis
- [ ] **MaxIterations Variant**:
  - [ ] Check if `max_iterations` config parameter exists
  - [ ] Check if algorithm can terminate due to reaching max iterations
  - [ ] Verify if iteration limit is enforced in training loop
- [ ] **ConvergedGap Variant**:
  - [ ] Check if gap-based convergence criteria exists
  - [ ] Check if relative or absolute gap is computed
  - [ ] Verify if gap convergence check exists in training loop
- [ ] Document findings: which variants are reachable under what conditions

### Option A: Variants Are Reachable
- [ ] Remove `#[allow(dead_code)]` attributes
- [ ] Add test cases that trigger each termination reason:
  ```rust
  #[test]
  fn test_max_iterations_termination() { ... }
  
  #[test]
  fn test_gap_convergence_termination() { ... }
  ```
- [ ] Update documentation to explain when each termination reason occurs
- [ ] Verify all match statements handle these variants

### Option B: Variants Are Unreachable (Never Constructed)
- [ ] Remove `MaxIterations` variant from enum
- [ ] Remove `ConvergedGap` variant from enum
- [ ] Update all match statements that handle `TerminationReason`
- [ ] Check for any pattern matching that would break
- [ ] Update doc comments to reflect actual termination conditions
- [ ] Add CHANGELOG.md entry: "**Breaking**: Removed unreachable TerminationReason variants"

### Option C: Variants Are Partially Implemented
- [ ] Determine what's needed to complete the implementation
- [ ] Create follow-up ticket for implementation if work is substantial
- [ ] Decide: complete now OR document as future enhancement
- [ ] If deferring: Remove variants and create proper issue for future work
- [ ] If implementing: Complete the termination logic and add tests

### Testing
- [ ] Run existing tests: `cargo test test_sddp`
- [ ] If keeping variants: Add test coverage for each reachable variant
- [ ] If removing variants: Verify no tests reference removed variants
- [ ] Run full test suite: `cargo test --workspace`
- [ ] Run clippy: `cargo clippy --all-targets -- -D warnings`

### Documentation
- [ ] Update `TerminationReason` doc comments to clarify when each variant is returned
- [ ] Update CHANGELOG.md if this is a breaking change
- [ ] Update any examples that check termination reason
- [ ] If removing: Note in docs what termination conditions are actually supported

## Technical Notes

**Location**: `src/sddp/mod.rs:193, 197`

**Investigation Strategy**:
1. Find the main training loop in `train()` function
2. Identify all `return` statements or final expression that creates `TerminationReason`
3. Check configuration structures for max_iterations and gap_tolerance parameters
4. Check if convergence detection includes gap calculation

**Common Termination Conditions in SDDP**:
- **Stability**: Upper bound stops changing (implemented as `ConvergedStability`)
- **Max Iterations**: Hard limit on training iterations (possibly implemented)
- **Gap Convergence**: Upper-lower bound gap below threshold (possibly not implemented)
- **Time Limit**: Wall-clock time exceeded (check if this variant exists)
- **User Interrupt**: External signal to stop (check if handled)

**API Design Consideration**:
If these variants are truly unreachable, keeping them in the enum is misleading to users who might expect to be able to configure max-iteration-based termination or gap-based convergence.

**Related Code**:
- Training loop logic
- Convergence detection code
- Configuration structures (`Config` in `src/input.rs`)
- Tests that create `TerminationReason` values

## Dependencies

- Blocked by: None
- Blocks: None
- Related: CLEANUP-005 (both deal with dead code verification in core modules)

## Estimated Effort

**0.5 story points** (2-4 hours, confidence: medium-high)

Investigation is straightforward. Resolution depends on findings:
- If reachable: 1 hour to remove attributes and add tests
- If unreachable: 2 hours to remove variants and update matches
- If partially implemented: Could spawn separate implementation ticket
