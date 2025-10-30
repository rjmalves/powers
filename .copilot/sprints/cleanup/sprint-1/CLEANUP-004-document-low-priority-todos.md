# CLEANUP-004: Document Low-Priority TODOs in Issue Tracker

## Context

There are 7 low-priority TODOs scattered across the codebase representing future enhancements that are not critical for current functionality. These should be removed from inline code and tracked properly in the issue tracker or a FUTURE_WORK.md file.

**Locations**:
1. `src/sddp/mod.rs:2120` - Graph type extensions (Markovian, cyclic)
2. `src/seasonal_params.rs:491` - Better stationarity check using eigenvalues
3. `src/noise_model_cache.rs:287` - Multi-season initial conditions
4. `src/sddp/builder.rs:682` - Extract from PAR config
5. `src/output.rs:377` - Transformation for CSV output

**Risk Level**: LOW (nice-to-have improvements)

## Acceptance Criteria

- [ ] All 7 low-priority TODOs documented in issue tracker or FUTURE_WORK.md
- [ ] Each TODO has clear description, context, and potential value
- [ ] Inline TODO comments removed from source code
- [ ] Where appropriate, replace with issue references: `// Future enhancement: see #XXX`
- [ ] FUTURE_WORK.md created with proper structure (if chosen over issues)
- [ ] No loss of context or intent from original TODOs

## Tasks

### Investigation
- [ ] Read context around each TODO to understand full scope
- [ ] Determine if any TODOs are duplicates or related
- [ ] Assess estimated effort and value for each enhancement
- [ ] Check git history to understand when/why TODO was added
- [ ] Determine priority order if implemented

### Option A: Create GitHub Issues
- [ ] Create issue for graph type extensions (Markovian/cyclic support)
- [ ] Create issue for eigenvalue-based stationarity check
- [ ] Create issue for multi-season initial conditions
- [ ] Create issue for PAR config extraction
- [ ] Create issue for CSV output transformation
- [ ] Label issues appropriately (enhancement, low-priority, future-work)
- [ ] Link related issues together

### Option B: Create FUTURE_WORK.md
- [ ] Create `FUTURE_WORK.md` in repository root
- [ ] Structure document by category (Algorithm, Input/Output, Configuration, etc.)
- [ ] Document each enhancement with:
  - Title and brief description
  - Current limitation or behavior
  - Proposed enhancement
  - Potential value/use cases
  - Estimated effort (T-shirt size)
  - Related code locations
- [ ] Add link to FUTURE_WORK.md in README.md

### Code Cleanup
- [ ] Remove TODO from `src/sddp/mod.rs:2120`
  - Replace with: `// Future enhancement: see FUTURE_WORK.md (Markovian/Cyclic Graphs)`
- [ ] Remove TODO from `src/seasonal_params.rs:491`
  - Replace with: `// Future enhancement: see FUTURE_WORK.md (Eigenvalue Stationarity)`
- [ ] Remove TODO from `src/noise_model_cache.rs:287`
  - Replace with: `// Future enhancement: see FUTURE_WORK.md (Multi-Season Initial Conditions)`
- [ ] Remove TODO from `src/sddp/builder.rs:682`
  - Replace with: `// Future enhancement: see FUTURE_WORK.md (PAR Config Extraction)`
- [ ] Remove TODO from `src/output.rs:377`
  - Replace with: `// Future enhancement: see FUTURE_WORK.md (CSV Transformation)`

### Testing
- [ ] Run `cargo test --workspace` to ensure no breakage
- [ ] Verify no other TODOs remain: `rg "TODO" src/ --type rust`
- [ ] Check that context is preserved in new documentation

### Documentation
- [ ] Ensure FUTURE_WORK.md or issues contain sufficient detail for future implementation
- [ ] Update CONTRIBUTING.md to mention FUTURE_WORK.md if created
- [ ] Add note in README.md about where to find planned enhancements

## Technical Notes

### TODO Details

**1. Graph Type Extensions** (`src/sddp/mod.rs:2120`)
```rust
// TODO - for the path graph case, this is enough. But for markovian graphs
// and cyclic graphs (infinite horizon) this might not be enough.
```
Context: Relates to trajectory extraction in simulation. Current approach works for path graphs but may need revision for more complex graph types.

**2. Stationarity Check** (`src/seasonal_params.rs:491`)
```rust
// TODO: For production, consider using nalgebra to compute eigenvalues
```
Context: Current stationarity check may not be as robust as eigenvalue-based approach. Would require adding nalgebra dependency.

**3. Multi-Season Initial Conditions** (`src/noise_model_cache.rs:287`)
```rust
// TODO: Consider multi-season initial conditions
```
Context: Current implementation may assume single-season initial conditions. Enhancement would allow more flexible initialization.

**4. PAR Config Extraction** (`src/sddp/builder.rs:682`)
```rust
// TODO: Extract from PAR config when available
```
Context: Builder pattern could be enhanced to extract configuration from PAR model directly.

**5. CSV Transformation** (`src/output.rs:377`)
```rust
// TODO: Add transformation to observations for CSV output.
```
Context: CSV output could include transformed/derived values, not just raw observations.

### FUTURE_WORK.md Template (if chosen)

```markdown
# Future Enhancements for POWE.RS

This document tracks planned enhancements that are not currently prioritized but would add value to the project.

## Algorithm Enhancements

### Markovian and Cyclic Graph Support
...

## Input/Output Enhancements

### CSV Output Transformations
...

## Configuration Enhancements

### PAR Config Extraction
...

## Numerical Methods

### Eigenvalue-Based Stationarity Check
...
```

## Dependencies

- Blocked by: None
- Blocks: None
- Related: CLEANUP-002, CLEANUP-003 (completes TODO resolution across codebase)

## Estimated Effort

**0.5 story points** (2-3 hours, confidence: high)

Straightforward documentation work. Most time spent understanding context and writing clear descriptions.
