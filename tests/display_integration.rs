//! Integration tests for display system output.
//!
//! These tests verify the display system produces correct output formats
//! for different profiles and CLI flags.

use std::process::Command;

/// Run powers with given args and capture stdout/stderr.
///
/// Returns (stdout, stderr, success).
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

/// Extract JSON lines from combined stdout (filters out [INFO] lines).
fn extract_json_lines(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            // JSON lines start with { and don't have [INFO] prefix
            trimmed.starts_with('{') && !line.contains("[INFO]")
        })
        .map(|s| s.to_string())
        .collect()
}

#[test]
fn test_automation_profile_produces_json() {
    let (stdout, _stderr, success) =
        run_powers(&["--profile", "automation", "examples/04-cascade"]);

    assert!(success, "Command should succeed");

    let json_lines = extract_json_lines(&stdout);
    assert!(!json_lines.is_empty(), "Should have JSON output");

    // Each JSON line should be valid
    for line in &json_lines {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| {
                panic!("Invalid JSON: {} in line: {}", e, line)
            });

        // Should have event_type field
        assert!(
            parsed.get("event_type").is_some(),
            "Missing event_type in: {}",
            line
        );
    }
}

#[test]
fn test_automation_profile_event_sequence() {
    let (stdout, _stderr, success) =
        run_powers(&["--profile", "automation", "examples/04-cascade"]);

    assert!(success);

    let json_lines = extract_json_lines(&stdout);
    let events: Vec<serde_json::Value> = json_lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    assert!(!events.is_empty(), "Should have events");

    // Extract event types
    let event_types: Vec<&str> = events
        .iter()
        .map(|e| e["event_type"].as_str().unwrap())
        .collect();

    // Should have: header, iterations, training_summary
    assert_eq!(event_types[0], "header", "First event should be header");
    assert!(
        event_types.contains(&"iteration"),
        "Should have iteration events"
    );
    assert!(
        event_types.contains(&"training_summary"),
        "Should have training_summary"
    );

    // Verify at least one iteration event exists
    let iteration_count =
        event_types.iter().filter(|&&t| t == "iteration").count();
    assert!(
        iteration_count > 0,
        "Should have at least one iteration event"
    );
}

#[test]
fn test_quiet_mode_minimal_output() {
    let (stdout, _stderr, success) = run_powers(&["-q", "examples/04-cascade"]);

    assert!(success);

    // Quiet mode should have no JSON output
    let json_lines = extract_json_lines(&stdout);
    assert!(
        json_lines.is_empty(),
        "Quiet mode should not produce JSON events"
    );

    // Should still have some [INFO] lines but fewer than normal
    let info_lines: Vec<&str> =
        stdout.lines().filter(|l| l.contains("[INFO]")).collect();

    // Should be minimal (greeting, simulation, farewell)
    assert!(
        info_lines.len() < 20,
        "Quiet mode should have minimal log output, got {} lines",
        info_lines.len()
    );
}

#[test]
fn test_iteration_json_has_required_fields() {
    let (stdout, _stderr, _) =
        run_powers(&["--profile", "automation", "examples/04-cascade"]);

    let json_lines = extract_json_lines(&stdout);

    // Find first iteration event
    let iteration_line = json_lines
        .iter()
        .find(|l| l.contains(r#""event_type":"iteration""#))
        .expect("Should have iteration event");

    let event: serde_json::Value =
        serde_json::from_str(iteration_line).unwrap();

    // Required fields for iteration event
    assert!(event.get("iteration").is_some());
    assert!(event.get("total_iterations").is_some());
    assert!(event.get("lower_bound").is_some());
    assert!(event.get("gap_percent").is_some());
    assert!(event.get("gap_trend").is_some());
    assert!(event.get("forward_cost_mean").is_some());
    assert!(event.get("forward_cost_std").is_some());
    assert!(event.get("forward_cost_min").is_some());
    assert!(event.get("forward_cost_max").is_some());
    assert!(event.get("first_stage_bound").is_some());
    assert!(event.get("first_stage_mean").is_some());
    assert!(event.get("first_stage_std").is_some());
    assert!(event.get("cuts_added").is_some());
    assert!(event.get("cuts_removed").is_some());
    assert!(event.get("cuts_active").is_some());
    assert!(event.get("iteration_time_ms").is_some());
    assert!(event.get("solver_calls").is_some());
    assert!(event.get("timestamp").is_some());
}

#[test]
fn test_training_summary_json() {
    let (stdout, _stderr, _) =
        run_powers(&["--profile", "automation", "examples/04-cascade"]);

    let json_lines = extract_json_lines(&stdout);
    let summary_line = json_lines
        .iter()
        .find(|l| l.contains(r#""event_type":"training_summary""#))
        .expect("Should have training_summary event");

    let event: serde_json::Value = serde_json::from_str(summary_line).unwrap();

    // Required fields for training summary
    assert!(event.get("final_lower_bound").is_some());
    assert!(event.get("policy_cost").is_some());
    assert!(event.get("final_gap_percent").is_some());
    assert!(event.get("total_cuts").is_some());
    assert!(event.get("total_iterations").is_some());
    assert!(event.get("total_time_ms").is_some());
    assert!(event.get("timestamp").is_some());
}

#[test]
fn test_header_json_fields() {
    let (stdout, _stderr, _) =
        run_powers(&["--profile", "automation", "examples/04-cascade"]);

    let json_lines = extract_json_lines(&stdout);
    let header_line = json_lines
        .iter()
        .find(|l| l.contains(r#""event_type":"header""#))
        .expect("Should have header event");

    let event: serde_json::Value = serde_json::from_str(header_line).unwrap();

    // Required fields for header
    assert!(event.get("program").is_some());
    assert!(event.get("version").is_some());
    assert!(event.get("iterations").is_some());
    assert!(event.get("forward_passes").is_some());
    assert!(event.get("cut_selection").is_some());
    assert!(event.get("timestamp").is_some());
}

#[test]
fn test_gap_trend_values() {
    let (stdout, _stderr, _) =
        run_powers(&["--profile", "automation", "examples/04-cascade"]);

    let json_lines = extract_json_lines(&stdout);
    let iteration_events: Vec<serde_json::Value> = json_lines
        .iter()
        .filter(|l| l.contains(r#""event_type":"iteration""#))
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    assert!(!iteration_events.is_empty());

    // First iteration should have "unknown" trend
    let first_trend = iteration_events[0]["gap_trend"].as_str().unwrap();
    assert_eq!(first_trend, "unknown");

    // Subsequent iterations should have valid trend values
    for event in &iteration_events[1..] {
        let trend = event["gap_trend"].as_str().unwrap();
        assert!(
            ["improving", "stable", "degrading", "unknown"].contains(&trend),
            "Invalid gap_trend: {}",
            trend
        );
    }
}
