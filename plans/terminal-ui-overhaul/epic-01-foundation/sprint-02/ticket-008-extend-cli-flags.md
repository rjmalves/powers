# T-008: Extend CLI with display flags

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: Sprint 1 complete (DisplayProfile type exists)
> **Blocks**: [T-009](./ticket-009-extend-config-schema.md)

## Files to Read Before Starting

- `src/cli.rs` - Current CLI definition
- `src/display/config.rs` - DisplayProfile, ColorMode types (from T-002)
- `src/lib.rs` - `run()` function where CLI args are processed

## Context

### Background

The display system needs CLI control for profile selection and color behavior. This follows the existing pattern for `--log-level` and `--log-format` flags.

### Current State

CLI has:
- `--log-level` - Override log level
- `--log-format` - Override log format

We add:
- `--profile` - Select display profile
- `--no-color` - Disable colors
- `--quiet` - Shortcut for minimal profile

## Specification

### New CLI Arguments

```rust
/// POWE.RS - High-performance SDDP for hydrothermal dispatch
#[derive(Parser, Debug)]
#[command(...)]
pub struct Cli {
    // ... existing fields ...
    
    /// Display profile: advanced, standard, minimal, automation
    ///
    /// Controls output verbosity and format:
    /// - advanced: Full metrics with colors and statistics
    /// - standard: Key metrics with simplified layout
    /// - minimal: Progress bar and final summary only
    /// - automation: JSON lines for machine parsing
    #[arg(long, global = true, value_name = "PROFILE")]
    pub profile: Option<String>,
    
    /// Disable colored output
    ///
    /// Forces plain text output even in interactive terminals.
    /// Equivalent to setting NO_COLOR environment variable.
    #[arg(long, global = true)]
    pub no_color: bool,
    
    /// Minimal output mode
    ///
    /// Shortcut for --profile minimal. Shows only progress bar
    /// and final summary.
    #[arg(long, short = 'q', global = true)]
    pub quiet: bool,
}
```

### Processing Logic

In `src/lib.rs` `run()` function:

```rust
// Determine display profile
let display_profile = if cli.quiet {
    DisplayProfile::Minimal
} else if let Some(profile_str) = cli.profile {
    profile_str.parse().map_err(|e| -> Box<dyn Error> { e.into() })?
} else {
    // Will be overridden by config if set
    DisplayProfile::Advanced
};

// Determine color mode
let color_mode = if cli.no_color {
    ColorMode::Never
} else {
    ColorMode::Auto
};
```

### Precedence Rules

1. `--quiet` takes precedence over `--profile` (error if both specified)
2. CLI `--profile` overrides config file `display.profile`
3. CLI `--no-color` overrides config file `display.color`
4. If nothing specified, use defaults (Advanced, Auto color)

## Acceptance Criteria

- [ ] `--profile` flag accepts: advanced, standard, minimal, automation
- [ ] `--profile` with invalid value produces clear error message
- [ ] `--no-color` flag parsed as boolean
- [ ] `--quiet` / `-q` flag parsed as boolean
- [ ] `--quiet` and `--profile` together produce error (clap conflict)
- [ ] Flags work with all subcommands (global = true)
- [ ] Help text explains each option clearly

## Implementation Guide

### Step 1: Add fields to Cli struct

Add the three new arguments with proper attributes.

### Step 2: Add clap conflicts

```rust
#[arg(long, short = 'q', global = true, conflicts_with = "profile")]
pub quiet: bool,
```

### Step 3: Update lib.rs

Add profile/color processing after log config overrides.

### Step 4: Test CLI parsing

## Pitfalls to Avoid

- ⚠️ Use `global = true` so flags work with `powers run` subcommand
- ⚠️ Use `conflicts_with` for `--quiet` and `--profile` mutual exclusion
- ⚠️ Parse profile string with proper error handling

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_cli_profile_flag() {
    let cli = Cli::parse_from(["powers", "--profile", "automation", "run", "examples/04-cascade"]);
    assert_eq!(cli.profile, Some("automation".to_string()));
}

#[test]
fn test_cli_quiet_flag() {
    let cli = Cli::parse_from(["powers", "-q", "run", "examples/04-cascade"]);
    assert!(cli.quiet);
}

#[test]
fn test_cli_no_color_flag() {
    let cli = Cli::parse_from(["powers", "--no-color", "run", "examples/04-cascade"]);
    assert!(cli.no_color);
}

#[test]
fn test_cli_quiet_profile_conflict() {
    let result = Cli::try_parse_from(["powers", "-q", "--profile", "advanced", "run", "examples/04-cascade"]);
    assert!(result.is_err());
}
```

### Integration Tests

- [ ] `cargo run -- --profile automation examples/04-cascade` produces JSON
- [ ] `cargo run -- --help` shows new flags with descriptions

## Documentation Requirements

- [ ] Help text for each flag
- [ ] Update README.md if it documents CLI flags
- [ ] Doc comments on Cli struct fields

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Follows existing pattern for log flags. clap makes this straightforward.

## Definition of Done

- [ ] All flags added to Cli struct
- [ ] Conflict between --quiet and --profile configured
- [ ] CLI tests passing
- [ ] `--help` shows new flags
- [ ] PR reviewed and merged
