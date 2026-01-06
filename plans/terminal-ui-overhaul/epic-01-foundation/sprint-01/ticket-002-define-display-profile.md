# T-002: Define DisplayProfile and DisplayConfig types

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [T-001](./ticket-001-add-crossterm-dependency.md)
> **Blocks**: [T-004](./ticket-004-define-display-context.md), [T-006](./ticket-006-define-display-renderer.md)

## Files to Read Before Starting

- `src/display/mod.rs` - Module structure (from T-001)
- `src/logging/config.rs` - Pattern for config enums with serde
- `src/input.rs` - How config is parsed from JSON

## Context

### Background

The display system supports multiple output profiles tailored to different use cases. This ticket defines the core configuration types that control profile selection and display behavior.

### Current State

T-001 created stub files. This ticket implements the actual types in `src/display/config.rs`.

## Specification

### DisplayProfile Enum

```rust
/// Output display profile controlling verbosity and format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayProfile {
    /// Full metrics with colors, real-time statistics, and trend indicators.
    /// Shows forward cost distributions, first-stage branching stats, timing breakdown.
    #[default]
    Advanced,
    
    /// Key metrics with simplified layout.
    /// Shows bounds, gap, timing without detailed statistics.
    Standard,
    
    /// Minimal output with progress bar and final summary only.
    /// Suitable for long runs where per-iteration detail is not needed.
    Minimal,
    
    /// Machine-readable JSON lines format.
    /// No ANSI codes, structured for parsing by external tools.
    Automation,
}
```

### ColorMode Enum

```rust
/// Color output mode for terminal display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// Detect terminal capabilities automatically.
    /// Disables color for non-interactive terminals and pipes.
    #[default]
    Auto,
    
    /// Always use colors (even in pipes).
    Always,
    
    /// Never use colors.
    Never,
}
```

### DisplayConfig Struct

```rust
/// Configuration for the display system.
#[derive(Debug, Clone)]
pub struct DisplayConfig {
    /// Selected display profile.
    pub profile: DisplayProfile,
    
    /// Color output mode.
    pub color: ColorMode,
    
    /// Target convergence gap (optional).
    /// When set, enables progress visualization toward this target.
    pub target_gap: Option<f64>,
    
    /// Whether terminal is interactive (computed at runtime).
    /// Set by terminal detection, not user configuration.
    pub is_interactive: bool,
    
    /// Whether color output is actually enabled (computed at runtime).
    /// Combines `color` mode with terminal detection.
    pub color_enabled: bool,
}
```

### Serde Implementation

Both enums should be serializable/deserializable:

```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DisplayProfile { ... }

#[derive(Serialize, Deserialize)]  
#[serde(rename_all = "lowercase")]
pub enum ColorMode { ... }
```

### FromStr Implementation

For CLI parsing:

```rust
impl FromStr for DisplayProfile {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "advanced" => Ok(Self::Advanced),
            "standard" => Ok(Self::Standard),
            "minimal" => Ok(Self::Minimal),
            "automation" | "json" => Ok(Self::Automation),
            _ => Err(format!("Invalid display profile: '{}'. Valid options: advanced, standard, minimal, automation", s)),
        }
    }
}
```

### Default Implementation

```rust
impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            profile: DisplayProfile::default(),
            color: ColorMode::default(),
            target_gap: None,
            is_interactive: true,  // Conservative default
            color_enabled: true,   // Will be computed at init
        }
    }
}
```

## Acceptance Criteria

- [ ] `DisplayProfile` enum with 4 variants implemented
- [ ] `ColorMode` enum with 3 variants implemented
- [ ] `DisplayConfig` struct with all fields implemented
- [ ] Serde derive for JSON serialization/deserialization
- [ ] `FromStr` for CLI argument parsing
- [ ] `Default` implementations
- [ ] Unit tests for parsing and serialization
- [ ] Doc comments on all public types

## Implementation Guide

### Step 1: Implement enums

Start with `DisplayProfile` and `ColorMode` with derives.

### Step 2: Implement DisplayConfig

Add the struct with all fields and Default impl.

### Step 3: Add FromStr

Implement string parsing for CLI integration.

### Step 4: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_profile_from_str() {
        assert_eq!("advanced".parse::<DisplayProfile>().unwrap(), DisplayProfile::Advanced);
        assert_eq!("MINIMAL".parse::<DisplayProfile>().unwrap(), DisplayProfile::Minimal);
        assert_eq!("json".parse::<DisplayProfile>().unwrap(), DisplayProfile::Automation);
        assert!("invalid".parse::<DisplayProfile>().is_err());
    }
    
    #[test]
    fn test_profile_serde() {
        let json = r#""advanced""#;
        let profile: DisplayProfile = serde_json::from_str(json).unwrap();
        assert_eq!(profile, DisplayProfile::Advanced);
    }
    
    #[test]
    fn test_config_default() {
        let config = DisplayConfig::default();
        assert_eq!(config.profile, DisplayProfile::Advanced);
        assert_eq!(config.color, ColorMode::Auto);
        assert!(config.target_gap.is_none());
    }
}
```

### Key Files to Modify

- `src/display/config.rs` - Main implementation
- `src/display/mod.rs` - Add re-exports

## Pitfalls to Avoid

- ⚠️ Don't forget `#[serde(rename_all = "lowercase")]` for consistent JSON keys
- ⚠️ `is_interactive` and `color_enabled` are runtime-computed, not deserialized
- ⚠️ Accept "json" as alias for "automation" in FromStr for convenience

## Testing Requirements

### Unit Tests

- [ ] Test `DisplayProfile::from_str` with all valid values
- [ ] Test `DisplayProfile::from_str` with invalid value returns Err
- [ ] Test case-insensitive parsing
- [ ] Test serde roundtrip for `DisplayProfile`
- [ ] Test serde roundtrip for `ColorMode`
- [ ] Test `DisplayConfig::default()` values

## Documentation Requirements

- [ ] Doc comments on `DisplayProfile` explaining each variant's purpose
- [ ] Doc comments on `ColorMode` explaining behavior
- [ ] Doc comments on `DisplayConfig` fields
- [ ] Module-level docs in `config.rs`

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward type definitions following existing patterns in codebase.

## Definition of Done

- [ ] All types implemented with documentation
- [ ] All tests passing
- [ ] Types exported from `src/display/mod.rs`
- [ ] PR reviewed and merged
