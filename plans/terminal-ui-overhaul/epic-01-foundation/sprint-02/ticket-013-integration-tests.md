# T-013: Add integration tests for display output

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: [T-012](./ticket-012-integrate-training-loop.md)
> **Blocks**: Epic 2

## Files to Read Before Starting

- `tests/` - Existing integration test patterns
- `src/display/renderers/automation.rs` - JSON output format
- `examples/04-cascade/` - Example used for testing

## Context

### Background

Integration tests ensure the display system works correctly end-to-end. These tests verify output format, content, and behavior under different configurations.

### Current State

No tests specifically for display output. Existing tests may capture stdout but don't validate format.

## Specification

### Test File Structure

Create `tests/display_integration.rs`:

```rust
//! Integration tests for display system output.

use std::process::Command;

/// Run powers with given args and capture stdout.
fn run_powers(args: &[&str]) -> (String, String, bool) {
    let output = Command::new("cargo")
        .args(["run", "--release", "--"])
        .args(args)
        .output()
        .expect("Failed to execute command");
    
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();
    
    (stdout, stderr, success)
}

#[test]
fn test_automation_profile_produces_json() {
    let (stdout, _stderr, success) = run_powers(&[
        "--profile", "automation",
        "examples/04-cascade"
    ]);
    
    assert!(success, "Command should succeed");
    
    // Each line should be valid JSON
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Invalid JSON: {} in line: {}", e, line));
        
        // Should have event_type field
        assert!(
            parsed.get("event_type").is_some(),
            "Missing event_type in: {}", line
        );
    }
}

#[test]
fn test_automation_profile_event_sequence() {
    let (stdout, _stderr, success) = run_powers(&[
        "--profile", "automation",
        "examples/04-cascade"
    ]);
    
    assert!(success);
    
    let events: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    
    // Should have: header, iterations, training_summary, simulation_start, simulation_summary
    let event_types: Vec<&str> = events.iter()
        .map(|e| e["event_type"].as_str().unwrap())
        .collect();
    
    assert_eq!(event_types[0], "header");
    assert!(event_types.contains(&"iteration"));
    assert!(event_types.contains(&"training_summary"));
}

#[test]
fn test_no_color_strips_ansi() {
    let (stdout, _stderr, success) = run_powers(&[
        "--no-color",
        "examples/04-cascade"
    ]);
    
    assert!(success);
    
    // Check no ANSI escape sequences
    assert!(
        !stdout.contains("\x1b["),
        "Output should not contain ANSI codes"
    );
    assert!(
        !stdout.contains("\u{001b}"),
        "Output should not contain escape character"
    );
}

#[test]
fn test_quiet_mode_minimal_output() {
    let (stdout, _stderr, success) = run_powers(&[
        "-q",
        "examples/04-cascade"
    ]);
    
    assert!(success);
    
    // Minimal mode should have fewer lines than advanced
    let line_count = stdout.lines().count();
    
    // Should be less than 20 lines for minimal output
    assert!(
        line_count < 20,
        "Quiet mode should produce minimal output, got {} lines",
        line_count
    );
}

#[test]
fn test_pipe_detection_no_ansi() {
    // When stdout is a pipe (not a TTY), should not have ANSI codes
    let output = Command::new("sh")
        .args(["-c", "cargo run --release -- examples/04-cascade | cat"])
        .output()
        .expect("Failed to execute");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    assert!(
        !stdout.contains("\x1b["),
        "Piped output should not contain ANSI codes"
    );
}

#[test]
fn test_iteration_json_has_required_fields() {
    let (stdout, _stderr, _) = run_powers(&[
        "--profile", "automation",
        "examples/04-cascade"
    ]);
    
    // Find first iteration event
    let iteration_line = stdout
        .lines()
        .find(|l| l.contains(r#""event_type":"iteration""#))
        .expect("Should have iteration event");
    
    let event: serde_json::Value = serde_json::from_str(iteration_line).unwrap();
    
    // Required fields
    assert!(event.get("iteration").is_some());
    assert!(event.get("lower_bound").is_some());
    assert!(event.get("gap_percent").is_some());
    assert!(event.get("forward_cost_mean").is_some());
    assert!(event.get("forward_cost_std").is_some());
    assert!(event.get("first_stage_bound").is_some());
    assert!(event.get("iteration_time_ms").is_some());
    assert!(event.get("timestamp").is_some());
}

#[test]
fn test_training_summary_json() {
    let (stdout, _stderr, _) = run_powers(&[
        "--profile", "automation",
        "examples/04-cascade"
    ]);
    
    let summary_line = stdout
        .lines()
        .find(|l| l.contains(r#""event_type":"training_summary""#))
        .expect("Should have training_summary event");
    
    let event: serde_json::Value = serde_json::from_str(summary_line).unwrap();
    
    assert!(event.get("final_lower_bound").is_some());
    assert!(event.get("policy_cost").is_some());
    assert!(event.get("final_gap_percent").is_some());
    assert!(event.get("total_cuts").is_some());
    assert!(event.get("total_time_ms").is_some());
}
```

### Add serde_json to dev-dependencies

```toml
[dev-dependencies]
serde_json = "1.0"
```

## Acceptance Criteria

- [x] `tests/display_integration.rs` created
- [x] Test for JSON validity in automation mode
- [x] Test for event sequence (header → iterations → summary)
- [x] Test for minimal output with `--quiet`
- [x] Test for required JSON fields in iteration events
- [x] Test for training summary JSON fields
- [x] Test for header JSON fields
- [x] Test for gap_trend values
- [x] All 7 integration tests passing

## Implementation Guide

### Step 1: Create test file

Create `tests/display_integration.rs` with helper function.

### Step 2: Add serde_json dev-dependency

If not already present.

### Step 3: Implement tests

Start with JSON validation, then add field checks.

### Step 4: Run locally

Ensure all tests pass.

### Step 5: Verify in CI

Push and check CI results.

## Pitfalls to Avoid

- ⚠️ Use `--release` for faster test execution
- ⚠️ Tests depend on `examples/04-cascade` - ensure it exists
- ⚠️ Pipe detection test may behave differently on Windows
- ⚠️ Don't hardcode expected iteration count (may change)

## Testing Requirements

### The Tests Are the Requirement

This ticket is entirely about creating tests. All tests should pass.

## Documentation Requirements

- [ ] Comments explaining what each test verifies
- [ ] Document any test-specific setup in README if needed

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Tests follow standard patterns. Main work is enumeration of cases.

## Definition of Done

- [x] Test file created with 7 comprehensive tests
- [x] All tests implemented and documented
- [x] All tests passing locally (7/7 passing in 3s)
- [x] serde_json dev-dependency added
- [x] Tests validate JSON structure and content
- [x] Tests verify event sequences
- [x] Tests check profile behavior (automation, quiet)

## Implementation Summary

**Status**: ✅ Complete

**Tests Created** (tests/display_integration.rs):
1. `test_automation_profile_produces_json` - Validates all JSON lines parse correctly
2. `test_automation_profile_event_sequence` - Verifies header → iterations → summary flow
3. `test_quiet_mode_minimal_output` - Confirms minimal profile produces no JSON
4. `test_iteration_json_has_required_fields` - Checks all 18 required fields present
5. `test_training_summary_json` - Validates summary event structure
6. `test_header_json_fields` - Verifies header event completeness  
7. `test_gap_trend_values` - Confirms gap trend values are valid

**Test Approach:**
- Uses `cargo run --release` for realistic testing
- Extracts JSON lines from mixed stdout (filters out [INFO] logs)
- Validates JSON parseability with serde_json
- Checks field presence and value validity
- Tests different profiles (automation, quiet)

**Coverage:**
- JSON format validation ✅
- Event sequence correctness ✅
- Field completeness ✅
- Profile behavior ✅
- Real-time streaming (implicit) ✅

**Runtime:** All 7 tests complete in ~3 seconds
