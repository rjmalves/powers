# [T-026] Silence Legacy Logging in lib.rs

> **Epic**: [Epic 02.5: UI Fixes](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-030](./ticket-030-fix-minor-issues.md)

## Context

### Background

The main entry point `src/lib.rs` contains legacy `log::info!` calls that output `[INFO]` prefixed messages. These now conflict with the new display system, creating a chaotic mixed output. The new display renderers already handle the greeting, configuration display, and training output, so the legacy calls need to be removed or replaced.

### Current State

```rust
// Lines 99-111 in src/lib.rs
::log::info!("");
::log::info!("POWE.RS - Power Optimization for the World of Energy - in pure RuSt");
::log::info!("--------------------------------------------------------------------");
::log::info!("");
::log::info!("Reading input files from '{}'", path_str);
```

These produce:
```
[INFO] 
[INFO] POWE.RS - Power Optimization for the World of Energy - in pure RuSt
[INFO] --------------------------------------------------------------------
[INFO] 
[INFO] Reading input files from 'examples/05-large-scale-brazilian'
```

The new display system renders:
```
╭───────────────────────────────────────────────────────────────────────────────────╮
│ POWE.RS - Power Optimization for the World of Energy                              │
│ Training: 3 iterations × 4 forward passes | Cut selection: enabled               │
╰───────────────────────────────────────────────────────────────────────────────────╯
```

## Specification

### Changes Required

1. **Remove greeting banner** (lines 99-105): Already handled by `render_header()`
2. **Remove "Reading input files" message** (line 111): Not needed with visual header
3. **Keep or adapt simulation skip message** (lines 133-138): Needs display integration
4. **Keep or adapt output path message** (lines 163-164): Consider keeping for debugging
5. **Replace final timing message** (lines 188-195): Display system should handle this

### Decision Points

For non-training messages (simulation skip, output path, total time), we have two options:

**Option A**: Keep as `log::info!` (simpler, provides backward compatibility)
- Pro: Less code changes, log filtering still works
- Con: Inconsistent with new display system

**Option B**: Route through display system (full unification)
- Pro: Consistent appearance, profile-aware
- Con: More implementation work, display system needs these methods

**Recommendation**: Use **Option A** for this ticket (minimal fix), defer full unification to Epic 03.

### Behavior

- **Training phase**: No `[INFO]` prefixed output; all output from display system
- **Non-training phases**: May still use `log::info!` for now (simulation skip, output writing)
- **Final timing**: Remove or make conditional on log level (duplicates display summary)

## Acceptance Criteria

- [x] Running `powers examples/05-large-scale-brazilian` shows NO `[INFO]` prefix before the display header box
- [x] Training output is clean with no legacy logging mixed in
- [x] The greeting banner ("POWE.RS - Power...") appears exactly ONCE via display system
- [x] "Reading input files" message is removed (path visible in display or not needed)
- [x] Total running time still appears (via display summary or kept as log)
- [x] All existing tests pass

## Implementation Guide

### Suggested Approach

1. Open `src/lib.rs`
2. Comment out or remove lines 99-111 (greeting and "Reading input files")
3. Optionally keep lines 163-164 (output path) and 188-195 (total time) for debugging
4. Test with `cargo run --release -- examples/05-large-scale-brazilian`
5. Verify clean output with each profile

### Key Files to Modify

- `src/lib.rs`: Lines 99-111, possibly 188-195

### Pitfalls to Avoid

- ⚠️ Don't remove error logging - only remove info-level output
- ⚠️ Keep simulation skip message for now (will be handled in T-027)
- ⚠️ Don't break the output writing functionality

## Testing Requirements

### Manual Tests

- [x] `powers examples/05-large-scale-brazilian` produces clean header
- [x] `powers examples/05-large-scale-brazilian --profile standard` works
- [x] `powers examples/05-large-scale-brazilian --profile minimal` works
- [x] `powers examples/05-large-scale-brazilian --profile automation` works

### Automated Tests

- [x] All existing 749 tests pass
- [x] No new clippy warnings

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward removal of code, minimal risk

## Definition of Done

- [x] Legacy greeting removed
- [x] "Reading input files" removed
- [x] Clean training output (no `[INFO]` before display header)
- [x] All tests passing
- [x] Manual verification with all profiles

## Status: ✅ COMPLETE

**Implementation Summary:**
- Removed greeting banner (lines 99-105 in src/lib.rs)
- Removed "Reading input files" message (line 111)
- Removed total running time message (lines 188-195) - display summary handles this
- Removed unused `std::time::Instant` import
- All 749 tests passing
- All 4 profiles tested and working correctly
- No new clippy warnings in lib.rs
