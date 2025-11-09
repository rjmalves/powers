//! Professional logging system for POWE.RS
//!
//! Provides structured logging with configurable outputs and formats.

pub mod config;
pub mod context;
pub mod formatters;
pub mod logger;

pub use config::{LogFormat, LogLevel, LogOutput, LoggingConfig};
pub use context::LogContext;
pub use logger::PowersLogger;

/// Initialize the logging system with the given configuration.
pub fn init(config: &LoggingConfig) -> Result<(), String> {
    logger::init_logger(config)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_exists() {
        // Smoke test to verify module compiles
    }
}
