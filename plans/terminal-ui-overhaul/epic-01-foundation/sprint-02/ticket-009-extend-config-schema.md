# T-009: Extend config schema for display section

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: [T-008](./ticket-008-extend-cli-flags.md)
> **Blocks**: [T-012](./ticket-012-integrate-training-loop.md)

## Files to Read Before Starting

- `src/input.rs` - Config file parsing, especially `ConfigInput` struct
- `src/display/config.rs` - DisplayProfile, ColorMode, DisplayConfig types
- `schemas/config.schema.json` - JSON schema for config file (if exists)
- `examples/04-cascade/config.json` - Example config file

## Context

### Background

The display configuration needs to be settable via `config.json` in addition to CLI flags. This follows the existing pattern for logging configuration.

### Current State

`config.json` has a `logging` section:
```json
{
  "logging": {
    "level": "info",
    "format": "terminal",
    "outputs": [{"type": "terminal"}]
  }
}
```

We add a parallel `display` section.

## Specification

### Config Schema Extension

```json
{
  "logging": { ... },
  "display": {
    "profile": "advanced",
    "color": "auto",
    "target_gap": null
  }
}
```

### Rust Types

```rust
// In src/input.rs or src/display/config.rs

/// Display configuration from config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplayConfigInput {
    /// Display profile selection.
    pub profile: DisplayProfile,
    
    /// Color output mode.
    pub color: ColorMode,
    
    /// Target convergence gap (optional).
    /// When set, enables progress visualization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_gap: Option<f64>,
}

impl Default for DisplayConfigInput {
    fn default() -> Self {
        Self {
            profile: DisplayProfile::Advanced,
            color: ColorMode::Auto,
            target_gap: None,
        }
    }
}
```

### Integration with ConfigInput

```rust
// In src/input.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfigInput {
    // ... existing fields ...
    
    /// Logging configuration.
    pub logging: LoggingConfig,
    
    /// Display configuration.
    #[serde(default)]
    pub display: DisplayConfigInput,
}
```

### Precedence in lib.rs

```rust
// After loading config
let mut display_config = DisplayConfig::from(config.display);

// Apply CLI overrides
if cli.quiet {
    display_config.profile = DisplayProfile::Minimal;
} else if let Some(ref profile_str) = cli.profile {
    display_config.profile = profile_str.parse()?;
}

if cli.no_color {
    display_config.color = ColorMode::Never;
}

// Apply terminal detection
let caps = TerminalCapabilities::detect();
display_config = display_config.with_terminal_caps(&caps);
```

## Acceptance Criteria

- [ ] `DisplayConfigInput` type created with serde derives
- [ ] `ConfigInput` has `display` field with `#[serde(default)]`
- [ ] Empty `display` section uses defaults
- [ ] Missing `display` section uses defaults
- [ ] All three fields (`profile`, `color`, `target_gap`) parseable
- [ ] `target_gap: null` parses as `None`
- [ ] CLI flags override config file settings
- [ ] Existing config files without `display` section continue to work

## Implementation Guide

### Step 1: Add DisplayConfigInput

Create the serde-compatible input type.

### Step 2: Add to ConfigInput

Add `display` field with default.

### Step 3: Implement From trait

```rust
impl From<DisplayConfigInput> for DisplayConfig {
    fn from(input: DisplayConfigInput) -> Self {
        Self {
            profile: input.profile,
            color: input.color,
            target_gap: input.target_gap,
            // Runtime fields set later
            is_interactive: true,
            color_enabled: true,
        }
    }
}
```

### Step 4: Update lib.rs run()

Add config loading and CLI override logic.

### Step 5: Test backward compatibility

Ensure existing configs work.

## Pitfalls to Avoid

- ⚠️ Use `#[serde(default)]` on both the field and the struct
- ⚠️ Handle case where config file has no `display` key at all
- ⚠️ `target_gap: null` in JSON should become `None`, not error
- ⚠️ Don't require updating all example config files immediately

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_display_config_defaults() {
    let json = r#"{}"#;
    let config: DisplayConfigInput = serde_json::from_str(json).unwrap();
    assert_eq!(config.profile, DisplayProfile::Advanced);
    assert_eq!(config.color, ColorMode::Auto);
    assert!(config.target_gap.is_none());
}

#[test]
fn test_display_config_full() {
    let json = r#"{"profile": "minimal", "color": "never", "target_gap": 5.0}"#;
    let config: DisplayConfigInput = serde_json::from_str(json).unwrap();
    assert_eq!(config.profile, DisplayProfile::Minimal);
    assert_eq!(config.color, ColorMode::Never);
    assert_eq!(config.target_gap, Some(5.0));
}

#[test]
fn test_display_config_null_target() {
    let json = r#"{"target_gap": null}"#;
    let config: DisplayConfigInput = serde_json::from_str(json).unwrap();
    assert!(config.target_gap.is_none());
}

#[test]
fn test_config_without_display_section() {
    let json = r#"{"logging": {"level": "info"}}"#;
    let config: ConfigInput = serde_json::from_str(json).unwrap();
    assert_eq!(config.display.profile, DisplayProfile::Advanced);
}
```

### Integration Tests

- [ ] Load example config files successfully
- [ ] Verify precedence: CLI > config > defaults

## Documentation Requirements

- [ ] Doc comments on DisplayConfigInput
- [ ] Update config file documentation if it exists
- [ ] Add example in a sample config

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Follows existing logging config pattern exactly.

## Definition of Done

- [ ] Types added with serde derives
- [ ] ConfigInput extended
- [ ] Default handling works
- [ ] Backward compatibility verified
- [ ] Tests passing
- [ ] PR reviewed and merged
