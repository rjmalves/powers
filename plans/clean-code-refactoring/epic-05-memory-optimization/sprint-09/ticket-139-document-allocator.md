# [T-139] Document Allocator Configuration

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: T-136
> **Blocks**: None

---

## Context

### Background

Users need to understand the allocator configuration options and when they might want to use alternatives. This documentation ensures users can make informed decisions about memory management.

### Current State

- Allocator features exist but are undocumented
- main.rs has comments but no user-facing docs
- README.md doesn't mention memory configuration

## Specification

### Documentation to Create/Update

1. **README.md**: Add "Memory Management" section
2. **CHANGELOG.md**: Document the change
3. **Cargo.toml**: Improve feature documentation
4. **docs/MEMORY_CONFIGURATION.md**: Detailed guide

### Content Requirements

#### README.md Section

```markdown
## Memory Management

Powers uses [mimalloc/jemalloc] by default for improved memory management.
This allocator returns freed memory to the OS more aggressively than the
system allocator, preventing unbounded RSS growth during long training runs.

### Configuration Options

| Feature Flag | Allocator | Use Case |
|--------------|-----------|----------|
| (default) | mimalloc | Recommended for most users |
| `--features jemalloc` | jemalloc | Alternative with similar behavior |
| `--no-default-features` | System | Debugging, compatibility |

### Example

```bash
# Default (recommended)
cargo build --release

# Use system allocator
cargo build --release --no-default-features
```
```

#### docs/MEMORY_CONFIGURATION.md

- Why alternative allocators are needed
- How glibc malloc behaves vs alternatives
- RSS growth problem and solution
- Configuration options with examples
- Troubleshooting guide

## Acceptance Criteria

- [ ] README.md has "Memory Management" section
- [ ] CHANGELOG.md documents the change
- [ ] Cargo.toml features are well-documented
- [ ] Detailed docs/MEMORY_CONFIGURATION.md created
- [ ] All examples are tested and working

## Implementation Guide

### Suggested Approach

1. Draft README.md section
2. Write detailed MEMORY_CONFIGURATION.md
3. Update Cargo.toml with feature descriptions
4. Add CHANGELOG.md entry
5. Test all documented commands

### Key Files to Modify

- `README.md`: Add memory section
- `CHANGELOG.md`: Document change
- `Cargo.toml`: Improve feature comments
- `docs/MEMORY_CONFIGURATION.md` (new): Detailed guide

### CHANGELOG Entry

```markdown
## [Unreleased]

### Changed

- **Default allocator changed to mimalloc/jemalloc** for improved memory management.
  This prevents unbounded RSS growth during long training runs. Use
  `--no-default-features` to opt-out and use the system allocator.

### Added

- `system-allocator` feature flag to explicitly use system (glibc) allocator
- Memory configuration documentation in `docs/MEMORY_CONFIGURATION.md`
```

### docs/MEMORY_CONFIGURATION.md Template

```markdown
# Memory Configuration

## Overview

Powers uses an optimized memory allocator by default to prevent
unbounded RSS (Resident Set Size) growth during training.

## The Problem

The SDDP algorithm creates and destroys solver models each iteration.
With the standard glibc allocator, freed memory is not returned to
the operating system, causing RSS to grow monotonically even though
actual memory usage is stable.

## The Solution

Powers uses [mimalloc/jemalloc] by default. These allocators return
freed memory to the OS more aggressively, resulting in stable RSS.

## Configuration Options

### Default (Recommended)

```bash
cargo build --release
```

Uses [mimalloc/jemalloc], recommended for:
- Long training runs (100+ iterations)
- Memory-constrained environments
- Production deployments

### System Allocator

```bash
cargo build --release --no-default-features
```

Uses system allocator (glibc on Linux), useful for:
- Debugging memory issues
- Compatibility with memory profilers (valgrind, etc.)
- Environments where custom allocators cause issues

### Alternative Allocators

```bash
# Use jemalloc instead of mimalloc
cargo build --release --no-default-features --features jemalloc
```

## Troubleshooting

### High Memory Usage

If you observe high memory usage:
1. Ensure you're using release builds (`--release`)
2. Check that default allocator is enabled
3. Monitor RSS over iterations (should stabilize after warmup)

### Compatibility Issues

If you experience crashes or issues with the default allocator:
1. Try the system allocator: `--no-default-features`
2. Report the issue with platform details

## Technical Details

For in-depth analysis of the memory optimization work, see:
- `docs/SPRINT_08_RSS_ANALYSIS.md` - Problem analysis
- `docs/ALLOCATOR_COMPARISON.md` - Allocator comparison
```

### Pitfalls to Avoid

- ⚠️ Test all documented commands before committing
- ⚠️ Keep README section concise, link to detailed docs
- ⚠️ Ensure feature flag syntax is correct

## Testing Requirements

### Documentation Tests

- [ ] All bash commands in docs work
- [ ] Feature flags are correctly spelled
- [ ] Links to other docs are valid

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Documentation writing, well-defined content

## Definition of Done

- [ ] README.md updated
- [ ] CHANGELOG.md updated
- [ ] docs/MEMORY_CONFIGURATION.md created
- [ ] Cargo.toml comments improved
- [ ] All documented commands tested
