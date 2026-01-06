# T-025: Polish and visual consistency review

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-022 (AdvancedRenderer summary), T-023 (StandardRenderer)
> **Blocks**: Epic 3 (Simulation & Polish)

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - AdvancedRenderer
- `src/display/renderers/standard.rs` - StandardRenderer
- `src/display/renderers/minimal.rs` - MinimalRenderer
- `src/display/renderers/automation.rs` - AutomationRenderer
- Master plan - Sample outputs for all profiles

## Context

### Background

With all renderers implemented, this ticket performs a visual consistency review and polish pass. The goal is to ensure all profiles look polished, consistent where appropriate, and distinct where intentional.

### Current State

All four renderers implemented individually. Need to verify they work well together and look professional.

## Specification

### Consistency Checklist

#### Color Usage

- [ ] Same semantic colors used across all renderers
- [ ] Good/bad/caution colors consistent
- [ ] Colors readable on both light and dark terminals
- [ ] No ANSI codes in Automation profile

#### Number Formatting

- [ ] Same precision for same types of values
- [ ] Scientific notation threshold consistent
- [ ] Duration formatting consistent
- [ ] Percentage formatting consistent

#### Spacing and Alignment

- [ ] Labels consistently sized in summaries
- [ ] Numbers right-aligned in all tables
- [ ] Text left-aligned in all tables
- [ ] Consistent padding in box borders

#### Status Icons

- [ ] Same icons for same states (✓, ⋯, ↓, ↑, etc.)
- [ ] Icons colored consistently
- [ ] Icons positioned consistently

### Visual Testing Tasks

1. **Side-by-Side Comparison**: Run all profiles with same data, verify intentional differences
2. **Width Testing**: Test at 80, 100, 120 column widths
3. **Color Testing**: Test with and without colors enabled
4. **Unicode Testing**: Verify graceful handling if Unicode not supported

### Code Quality Tasks

1. **Extract Common Code**: Move shared rendering logic to shared helpers
2. **Consistent Error Messages**: Same formatting for warnings/errors
3. **Documentation**: Ensure all render methods are documented
4. **Dead Code**: Remove any unused helper functions

### Polish Items

#### AdvancedRenderer

- [ ] Box borders align perfectly
- [ ] Column widths balance well
- [ ] Continuation rows readable
- [ ] No overflow at standard terminal widths

#### StandardRenderer

- [ ] Noticeably simpler than Advanced
- [ ] Same key information visible
- [ ] Clean, professional appearance

#### MinimalRenderer

- [ ] Progress bar updates smoothly
- [ ] Summary is concise but complete
- [ ] Works well with carriage return

#### AutomationRenderer

- [ ] Valid JSON output (test with `jq`)
- [ ] All fields present
- [ ] Consistent field naming (snake_case)

## Acceptance Criteria

- [ ] All renderers visually reviewed and polished
- [ ] No visual artifacts or misalignment
- [ ] Consistent formatting across profiles
- [ ] Works at 80-column minimum width
- [ ] No regression in existing tests
- [ ] Manual testing documented with screenshots or terminal captures

## Implementation Guide

### Step 1: Create test harness

Create a simple test script to run all profiles:

```rust
// tests/display_visual_test.rs
#[test]
#[ignore] // Run manually with --ignored
fn visual_test_all_profiles() {
    let ctx = create_test_context();
    let config = DisplayConfig::default();
    
    for profile in [
        DisplayProfile::Advanced,
        DisplayProfile::Standard,
        DisplayProfile::Minimal,
        DisplayProfile::Automation,
    ] {
        println!("\n=== {} Profile ===\n", profile);
        let renderer = create_renderer(profile);
        println!("{}", renderer.render_header(&config));
        println!("{}", renderer.render_iteration(&ctx));
        // ... more iterations
        println!("{}", renderer.render_training_summary(&result));
    }
}
```

### Step 2: Review and fix consistency issues

Go through each renderer and fix inconsistencies:

```rust
// Example: Ensure consistent label width
const LABEL_WIDTH: usize = 14;

fn format_metric_line(label: &str, value: &str) -> String {
    format!("  {:<width$}{}", label, value, width = LABEL_WIDTH)
}
```

### Step 3: Extract shared helpers

```rust
// src/display/renderers/common.rs

/// Format a summary metric line with consistent label width
pub fn format_summary_line(label: &str, value: impl std::fmt::Display) -> String {
    format!("  {:14}{}", format!("{}:", label), value)
}

/// Get status icon for convergence
pub fn convergence_icon(converged: bool, color_config: &ColorConfig) -> String {
    if converged {
        colorize("✓", SemanticColor::Good, color_config)
    } else {
        colorize("⋯", SemanticColor::Caution, color_config)
    }
}
```

### Step 4: Update renderers to use shared code

Replace duplicated code with calls to common helpers.

### Step 5: Test at various widths

```rust
#[test]
fn test_narrow_terminal() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 80);
    let output = renderer.render_iteration(&ctx);
    
    // Check no line exceeds 80 chars
    for line in output.lines() {
        let visible_len = strip_ansi(line).chars().count();
        assert!(visible_len <= 80, "Line too long: {}", line);
    }
}
```

### Step 6: Validate JSON output

```rust
#[test]
fn test_automation_valid_json() {
    let renderer = AutomationRenderer::new();
    let output = renderer.render_iteration(&ctx);
    
    // Each line should be valid JSON
    for line in output.lines() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .expect(&format!("Invalid JSON: {}", line));
        
        // Verify expected fields
        assert!(parsed.get("iteration").is_some());
        assert!(parsed.get("lower_bound").is_some());
    }
}
```

### Patterns to Follow

- Use constants for shared values
- Create helpers for repeated patterns
- Test edge cases explicitly

### Pitfalls to Avoid

- ⚠️ Don't over-DRY - some duplication is OK for clarity
- ⚠️ Don't break existing tests while polishing
- ⚠️ Remember to test colors disabled mode

## Testing Requirements

### Automated Tests

- [ ] Test all renderers at 80 column width
- [ ] Test all renderers with colors disabled
- [ ] Validate AutomationRenderer JSON output
- [ ] Test empty/edge cases for all renderers

### Manual Testing

- [ ] Run visual test in real terminal
- [ ] Capture sample outputs for documentation
- [ ] Test on Linux (primary)
- [ ] Test on macOS if available

### Regression Testing

- [ ] All existing tests still pass
- [ ] No new warnings from clippy

## Documentation Requirements

- [ ] Update module docs with sample outputs
- [ ] Add screenshots or terminal captures to docs
- [ ] Document any behavior differences between profiles
- [ ] Add troubleshooting section for display issues

## Deliverables

1. **Code changes**: Fixes and consistency improvements
2. **Sample outputs**: Captured terminal output for each profile
3. **Test coverage**: New tests for edge cases found
4. **Documentation**: Updated docs with examples

## Effort Estimate

**Points**: 2
**Confidence**: Medium
**Rationale**: Scope depends on issues found. Mostly cleanup and verification.

## Definition of Done

- [ ] All consistency issues resolved
- [ ] All profiles visually polished
- [ ] Shared code extracted where beneficial
- [ ] All tests passing
- [ ] Manual testing completed
- [ ] Documentation updated
- [ ] Sample outputs captured
- [ ] PR reviewed and merged
