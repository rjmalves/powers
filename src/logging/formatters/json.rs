//! JSON formatter for machine-readable output

use crate::logging::context::LogContext;
use log::Record;
use serde::Serialize;

/// JSON Lines formatter for structured log output
#[derive(Default)]
pub struct JsonFormatter;

impl JsonFormatter {
    pub fn new() -> Self {
        Self
    }

    /// Format a log record as a JSON Lines entry
    pub fn format(&self, record: &Record, context: &LogContext) -> Vec<u8> {
        let entry = JsonLogEntry::from_record(record, context);

        let mut json = serde_json::to_vec(&entry).unwrap_or_default();
        json.push(b'\n');
        json
    }
}

/// Serializable log entry for JSON output
#[derive(Serialize)]
struct JsonLogEntry {
    timestamp: String,
    level: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    iteration: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lower_bound: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    simulation_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    forward_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backward_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_time_ms: Option<u64>,
}

impl JsonLogEntry {
    fn from_record(record: &Record, context: &LogContext) -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            level: record.level().to_string(),
            message: record.args().to_string(),
            iteration: context.iteration,
            lower_bound: context.lower_bound,
            simulation_cost: context.simulation_cost,
            forward_time_ms: context.forward_time.map(|d| d.as_millis() as u64),
            backward_time_ms: context
                .backward_time
                .map(|d| d.as_millis() as u64),
            total_time_ms: context.total_time.map(|d| d.as_millis() as u64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Level;
    use std::time::Duration;

    #[test]
    fn test_json_formatter_creates() {
        let _formatter = JsonFormatter::new();
    }

    #[test]
    fn test_format_basic_log() {
        let formatter = JsonFormatter::new();
        let context = LogContext::default();

        let record = log::Record::builder()
            .args(format_args!("Test message"))
            .level(Level::Info)
            .target("test")
            .build();

        let output = formatter.format(&record, &context);
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("\"level\":\"INFO\""));
        assert!(output_str.contains("\"message\":\"Test message\""));
        assert!(output_str.contains("\"timestamp\":"));
        assert!(output_str.ends_with('\n'));
    }

    #[test]
    fn test_format_with_context() {
        let formatter = JsonFormatter::new();
        let context = LogContext {
            iteration: Some(42),
            lower_bound: Some(1234.56),
            simulation_cost: Some(7890.12),
            forward_time: Some(Duration::from_millis(500)),
            backward_time: Some(Duration::from_millis(300)),
            total_time: Some(Duration::from_millis(800)),
        };

        let record = log::Record::builder()
            .args(format_args!("Iteration complete"))
            .level(Level::Info)
            .target("test")
            .build();

        let output = formatter.format(&record, &context);
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("\"iteration\":42"));
        assert!(output_str.contains("\"lower_bound\":1234.56"));
        assert!(output_str.contains("\"simulation_cost\":7890.12"));
        assert!(output_str.contains("\"forward_time_ms\":500"));
        assert!(output_str.contains("\"backward_time_ms\":300"));
        assert!(output_str.contains("\"total_time_ms\":800"));
    }

    #[test]
    fn test_json_parseable() {
        let formatter = JsonFormatter::new();
        let context = LogContext::default();

        let record = log::Record::builder()
            .args(format_args!("Test"))
            .level(Level::Debug)
            .target("test")
            .build();

        let output = formatter.format(&record, &context);
        let output_str = String::from_utf8(output).unwrap();

        // Should be valid JSON
        let _parsed: serde_json::Value =
            serde_json::from_str(output_str.trim()).unwrap();
    }
}
