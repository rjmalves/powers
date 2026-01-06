# T-001: Add crossterm dependency and display module structure

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-002, T-003, T-004, T-005, T-006, T-007

## Files to Read Before Starting

- `Cargo.toml` - Current dependencies
- `src/lib.rs` - Module structure
- `src/logging/mod.rs` - Existing logging module pattern to follow

## Context

### Background

The terminal UI overhaul requires the `crossterm` crate for cross-platform terminal styling and detection. This ticket sets up the dependency and creates the initial module structure that all subsequent display work builds upon.

### Current State

No display module exists. Terminal output is handled entirely through the `logging` module with basic ANSI color codes in `src/logging/formatters/terminal.rs`.

## Specification

### Create Module Structure

```
src/display/
├── mod.rs              # Module root with public exports
├── config.rs           # (stub) Will contain DisplayProfile, DisplayConfig
├── context.rs          # (stub) Will contain DisplayContext, CostStatistics  
├── renderer.rs         # (stub) Will contain DisplayRenderer trait
├── terminal.rs         # (stub) Will contain terminal detection
└── renderers/
    ├── mod.rs          # Renderer implementations module
    ├── advanced.rs     # (stub) AdvancedRenderer
    ├── standard.rs     # (stub) StandardRenderer
    ├── minimal.rs      # (stub) MinimalRenderer
    └── automation.rs   # (stub) AutomationRenderer
```

### Add Dependency

Add to `Cargo.toml`:
```toml
crossterm = "0.27"
```

### Module Stubs

Each stub file should contain:
- Module documentation comment explaining purpose
- Placeholder types or functions with `todo!()` or `unimplemented!()`
- Proper visibility (`pub` for public API)

## Acceptance Criteria

- [ ] `crossterm = "0.27"` added to `[dependencies]` in `Cargo.toml`
- [ ] `src/display/mod.rs` exists and exports submodules
- [ ] `src/lib.rs` includes `pub mod display;`
- [ ] All stub files created with documentation comments
- [ ] `cargo build` succeeds
- [ ] `cargo doc --open` shows display module documentation

## Implementation Guide

### Step 1: Add dependency

Edit `Cargo.toml`:
```toml
[dependencies]
# ... existing deps ...
crossterm = "0.27"  # Terminal styling, colors, detection
```

### Step 2: Create directory structure

```bash
mkdir -p src/display/renderers
```

### Step 3: Create mod.rs

```rust
//! Display system for POWE.RS terminal output.
//!
//! This module provides a rich, profile-based display system for training
//! and simulation progress. Supports multiple output profiles from detailed
//! advanced views to minimal progress bars and machine-readable JSON.
//!
//! # Profiles
//!
//! - `Advanced`: Full metrics with colors, statistics, and trend indicators
//! - `Standard`: Key metrics with simplified layout
//! - `Minimal`: Progress bar with final summary only
//! - `Automation`: JSON lines for machine parsing
//!
//! # Example
//!
//! ```ignore
//! use powers_rs::display::{DisplayConfig, DisplayProfile};
//! 
//! let config = DisplayConfig::default();
//! assert_eq!(config.profile, DisplayProfile::Advanced);
//! ```

pub mod config;
pub mod context;
pub mod renderer;
pub mod renderers;
pub mod terminal;

// Re-exports will be added as types are implemented
```

### Step 4: Create stub files

Each file should have appropriate doc comments. Example for `config.rs`:

```rust
//! Display configuration types.
//!
//! Defines the display profile selection and configuration options
//! that control how output is rendered.

// TODO: Implement DisplayProfile enum (T-002)
// TODO: Implement DisplayConfig struct (T-002)
```

### Step 5: Update lib.rs

Add after other module declarations:
```rust
pub mod display;
```

## Testing Requirements

### Build Tests

- [ ] `cargo build` succeeds
- [ ] `cargo build --release` succeeds
- [ ] No warnings from new module stubs

### Documentation Tests

- [ ] `cargo doc` generates documentation for display module
- [ ] Module-level docs appear correctly

## Documentation Requirements

- [ ] Module-level documentation in `mod.rs` with overview
- [ ] Each stub file has doc comment explaining its future purpose
- [ ] No `#![allow(missing_docs)]` - enforce documentation from start

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Straightforward file creation and dependency addition. No logic to implement.

## Definition of Done

- [ ] Dependency added
- [ ] Module structure created
- [ ] All stubs have documentation
- [ ] Builds successfully
- [ ] PR reviewed and merged
