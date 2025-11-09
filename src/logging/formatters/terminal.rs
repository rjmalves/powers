//! Terminal formatter for human-readable output

use crate::logging::context::LogContext;
use log::Record;
use std::io::Write;
use std::time::Duration;

/// Formatter for terminal output with colors and table formatting
pub struct TerminalFormatter {
    use_colors: bool,
}

impl TerminalFormatter {
    pub fn new(use_colors: bool) -> Self {
        Self { use_colors }
    }

    /// Format a log record to bytes for writing
    pub fn format(&self, record: &Record, context: &LogContext) -> Vec<u8> {
        // Check if this is a structured log with iteration context
        if let Some(iteration) = context.iteration {
            if let (Some(lower), Some(simul)) =
                (context.lower_bound, context.simulation_cost)
            {
                // Format training table row with log level prefix
                let row = self.format_table_row(
                    record.level(),
                    iteration,
                    lower,
                    simul,
                    context.forward_time.unwrap_or_default(),
                    context.backward_time.unwrap_or_default(),
                    context.total_time.unwrap_or_default(),
                );
                return row.into_bytes();
            }
        }

        // Default formatting for other log types
        let mut buffer = Vec::new();
        let level_str = if self.use_colors {
            self.colorize_level(record.level())
        } else {
            format!("{}", record.level())
        };

        writeln!(buffer, "[{}] {}", level_str, record.args()).unwrap();
        buffer
    }

    fn colorize_level(&self, level: log::Level) -> String {
        use log::Level;
        match level {
            Level::Error => format!("\x1b[31m{}\x1b[0m", level), // Red
            Level::Warn => format!("\x1b[33m{}\x1b[0m", level),  // Yellow
            Level::Info => format!("\x1b[32m{}\x1b[0m", level),  // Green
            Level::Debug => format!("\x1b[36m{}\x1b[0m", level), // Cyan
            Level::Trace => format!("\x1b[90m{}\x1b[0m", level), // Gray
        }
    }

    /// Format duration as HH:MM:SS.SSS
    fn format_duration(&self, duration: Duration) -> String {
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        let millis = duration.subsec_millis();

        format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
    }

    /// Format cost in scientific notation with appropriate precision
    fn format_cost(&self, cost: f64) -> String {
        format!("{:.6e}", cost)
    }

    /// Format a training table row with optional log level prefix
    #[allow(clippy::too_many_arguments)]
    fn format_table_row(
        &self,
        level: log::Level,
        iteration: usize,
        lower_bound: f64,
        simulation_cost: f64,
        forward_time: Duration,
        backward_time: Duration,
        total_time: Duration,
    ) -> String {
        // Format the level prefix (matches default formatting style)
        let level_str = if self.use_colors {
            self.colorize_level(level)
        } else {
            format!("{}", level)
        };

        // Format: [LEVEL] iter | lower | simul | fwd | bwd | total
        format!(
            "[{}] {: >4} | {: >14} | {: >14} | {: >12} | {: >12} | {: >12}\n",
            level_str,
            iteration,
            self.format_cost(lower_bound),
            self.format_cost(simulation_cost),
            self.format_duration(forward_time),
            self.format_duration(backward_time),
            self.format_duration(total_time)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_terminal_formatter_creates() {
        let formatter = TerminalFormatter::new(false);
        assert!(!formatter.use_colors);
    }

    #[test]
    fn test_format_duration() {
        let formatter = TerminalFormatter::new(false);

        let duration = Duration::from_millis(3661250);
        assert_eq!(formatter.format_duration(duration), "01:01:01.250");

        let duration = Duration::from_secs(60);
        assert_eq!(formatter.format_duration(duration), "00:01:00.000");

        let duration = Duration::from_secs(3600);
        assert_eq!(formatter.format_duration(duration), "01:00:00.000");
    }

    #[test]
    fn test_format_cost() {
        let formatter = TerminalFormatter::new(false);

        let cost = 123456.789;
        assert_eq!(formatter.format_cost(cost), "1.234568e5");

        let cost = 0.00123;
        assert_eq!(formatter.format_cost(cost), "1.230000e-3");
    }

    #[test]
    fn test_format_table_row() {
        let formatter = TerminalFormatter::new(false);

        let row = formatter.format_table_row(
            log::Level::Info,
            1,
            2499.394,
            2550.123,
            Duration::from_secs(10),
            Duration::from_secs(5),
            Duration::from_secs(15),
        );

        // Should include [INFO] prefix
        assert!(row.contains("[INFO]"));
        assert!(row.contains("   1"));
        assert!(row.contains("2.499394e3"));
        assert!(row.contains("2.550123e3"));
        assert!(row.contains("00:00:10.000"));
        assert!(row.contains("00:00:05.000"));
        assert!(row.contains("00:00:15.000"));
    }
}
