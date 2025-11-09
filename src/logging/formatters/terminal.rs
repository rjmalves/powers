//! Terminal formatter for human-readable output

use crate::logging::context::LogContext;
use log::Record;
use std::io::Write;

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
                // Call existing function from legacy log module
                crate::log::training_table_row(
                    iteration,
                    lower,
                    simul,
                    context.forward_time.unwrap_or_default(),
                    context.backward_time.unwrap_or_default(),
                    context.total_time.unwrap_or_default(),
                );
                return Vec::new(); // Already printed
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_formatter_creates() {
        let formatter = TerminalFormatter::new(false);
        assert!(!formatter.use_colors);
    }
}
