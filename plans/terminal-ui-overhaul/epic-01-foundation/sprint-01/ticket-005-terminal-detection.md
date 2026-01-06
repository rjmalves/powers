# T-005: Implement terminal capability detection

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [T-001](./ticket-001-add-crossterm-dependency.md)
> **Blocks**: [T-007](./ticket-007-automation-renderer.md)

## Files to Read Before Starting

- `src/display/mod.rs` - Module structure
- `src/logging/logger.rs` - Lines 160-164 where `atty::is()` is currently used
- `crossterm` documentation for terminal detection

## Context

### Background

The display system needs to detect terminal capabilities to automatically disable colors and special formatting when output is piped to a file or running in CI/CD. This ticket creates the terminal detection utilities.

### Current State

The codebase uses `atty::is(atty::Stream::Stdout)` for basic TTY detection in the logging module. We'll consolidate and extend this with `crossterm`.

## Specification

### Terminal Module

```rust
//! Terminal capability detection utilities.
//!
//! Provides functions to detect terminal features like color support
//! and interactivity for automatic output adaptation.

use crossterm::tty::IsTty;
use std::io::{stdout, IsTerminal};

/// Terminal capability information.
#[derive(Debug, Clone)]
pub struct TerminalCapabilities {
    /// Whether stdout is connected to an interactive terminal.
    pub is_interactive: bool,
    
    /// Whether the terminal supports ANSI colors.
    pub supports_color: bool,
    
    /// Whether the terminal supports Unicode (UTF-8).
    pub supports_unicode: bool,
    
    /// Terminal width in columns (if detectable).
    pub width: Option<u16>,
    
    /// Terminal height in rows (if detectable).
    pub height: Option<u16>,
}

impl TerminalCapabilities {
    /// Detect terminal capabilities.
    ///
    /// Checks stdout for interactivity, color support, and dimensions.
    /// Uses conservative defaults when detection fails.
    pub fn detect() -> Self {
        let is_interactive = stdout().is_terminal();
        
        // Color support: interactive terminal and not explicitly disabled
        let supports_color = is_interactive && !is_color_disabled_by_env();
        
        // Unicode: assume supported on modern terminals
        // Could check LANG/LC_ALL but this is conservative enough
        let supports_unicode = is_interactive;
        
        // Terminal dimensions
        let (width, height) = crossterm::terminal::size()
            .map(|(w, h)| (Some(w), Some(h)))
            .unwrap_or((None, None));
        
        Self {
            is_interactive,
            supports_color,
            supports_unicode,
            width,
            height,
        }
    }
    
    /// Create capabilities for non-interactive (pipe/file) output.
    pub fn non_interactive() -> Self {
        Self {
            is_interactive: false,
            supports_color: false,
            supports_unicode: false,
            width: None,
            height: None,
        }
    }
    
    /// Create capabilities forcing color output.
    pub fn force_color() -> Self {
        let mut caps = Self::detect();
        caps.supports_color = true;
        caps
    }
    
    /// Apply ColorMode override.
    pub fn with_color_mode(mut self, mode: super::config::ColorMode) -> Self {
        match mode {
            super::config::ColorMode::Auto => { /* keep detected */ }
            super::config::ColorMode::Always => {
                self.supports_color = true;
            }
            super::config::ColorMode::Never => {
                self.supports_color = false;
            }
        }
        self
    }
}

/// Check if color is disabled by environment variables.
///
/// Respects common conventions:
/// - `NO_COLOR` environment variable (https://no-color.org/)
/// - `TERM=dumb`
fn is_color_disabled_by_env() -> bool {
    // NO_COLOR standard: presence disables color (value doesn't matter)
    if std::env::var("NO_COLOR").is_ok() {
        return true;
    }
    
    // TERM=dumb is a conventional signal for minimal terminal
    if let Ok(term) = std::env::var("TERM") {
        if term == "dumb" {
            return true;
        }
    }
    
    false
}

/// Check if output is being piped (quick check without full detection).
#[inline]
pub fn is_piped() -> bool {
    !stdout().is_terminal()
}

/// Get terminal width, or default if not detectable.
pub fn terminal_width_or(default: u16) -> u16 {
    crossterm::terminal::size()
        .map(|(w, _)| w)
        .unwrap_or(default)
}
```

### Default Export

Add to `src/display/mod.rs`:
```rust
pub use terminal::{TerminalCapabilities, is_piped, terminal_width_or};
```

## Acceptance Criteria

- [ ] `TerminalCapabilities::detect()` correctly identifies interactive terminal
- [ ] `TerminalCapabilities::detect()` returns `is_interactive = false` when piped
- [ ] `NO_COLOR` environment variable disables colors
- [ ] `TERM=dumb` disables colors
- [ ] `ColorMode::Always` overrides detection to enable colors
- [ ] `ColorMode::Never` overrides detection to disable colors
- [ ] Terminal width/height detection works or returns None gracefully
- [ ] `is_piped()` helper works correctly
- [ ] Unit tests (may need integration tests for actual terminal detection)

## Implementation Guide

### Step 1: Create terminal.rs

Implement `TerminalCapabilities` struct and detection logic.

### Step 2: Implement environment checks

Check `NO_COLOR` and `TERM=dumb`.

### Step 3: Add ColorMode integration

Implement `with_color_mode()` method.

### Step 4: Add helper functions

`is_piped()`, `terminal_width_or()`.

### Step 5: Testing strategy

Terminal detection is tricky to unit test because it depends on runtime environment. Use:
- Mock tests for environment variable handling
- Integration tests that check behavior when run in different modes
- Manual testing with pipes: `cargo run | cat`

## Pitfalls to Avoid

- ⚠️ Don't use `atty` crate anymore - `crossterm` and std `IsTerminal` are sufficient
- ⚠️ `NO_COLOR` presence disables color regardless of its value (even empty string)
- ⚠️ Terminal size detection can fail in some environments - always handle `None`
- ⚠️ `is_terminal()` is for stdout specifically; stderr might differ

## Testing Requirements

### Unit Tests

- [ ] Test `is_color_disabled_by_env()` with NO_COLOR set
- [ ] Test `is_color_disabled_by_env()` with TERM=dumb
- [ ] Test `is_color_disabled_by_env()` returns false normally
- [ ] Test `TerminalCapabilities::non_interactive()` values
- [ ] Test `with_color_mode(ColorMode::Always)` enables color
- [ ] Test `with_color_mode(ColorMode::Never)` disables color

### Integration Tests (manual or CI)

- [ ] Run with `| cat` and verify no ANSI codes
- [ ] Run interactively and verify colors present
- [ ] Set `NO_COLOR=1` and verify no colors

### Environment Test Helpers

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    // Note: Environment tests should be run with --test-threads=1
    // to avoid race conditions between tests modifying env vars.
    
    #[test]
    fn test_no_color_env() {
        // Save original
        let original = env::var("NO_COLOR").ok();
        
        // Set NO_COLOR
        env::set_var("NO_COLOR", "1");
        assert!(is_color_disabled_by_env());
        
        // Restore
        match original {
            Some(v) => env::set_var("NO_COLOR", v),
            None => env::remove_var("NO_COLOR"),
        }
    }
}
```

## Documentation Requirements

- [ ] Module-level docs explaining detection strategy
- [ ] Doc comments on `TerminalCapabilities` fields
- [ ] Doc comments explaining `NO_COLOR` standard
- [ ] Usage examples in module docs

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Terminal detection has edge cases. Core logic is simple but testing is tricky.

## Definition of Done

- [ ] All types and functions implemented
- [ ] Environment variable handling tested
- [ ] ColorMode override tested
- [ ] Types exported from `src/display/mod.rs`
- [ ] Manual testing with pipes confirms no ANSI codes
- [ ] PR reviewed and merged
