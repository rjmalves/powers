//! Configuration types for the logging system

use serde::{Deserialize, Serialize};

/// Log level threshold
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
}

impl LogLevel {
    /// Convert to log crate's LevelFilter
    pub fn to_level_filter(self) -> log::LevelFilter {
        match self {
            Self::Error => log::LevelFilter::Error,
            Self::Warn => log::LevelFilter::Warn,
            Self::Info => log::LevelFilter::Info,
            Self::Debug => log::LevelFilter::Debug,
            Self::Trace => log::LevelFilter::Trace,
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "warn" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            _ => Err(format!("Invalid log level: {}", s)),
        }
    }
}

/// Output format for logs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Terminal,
    Json,
    Structured,
}

impl Default for LogFormat {
    fn default() -> Self {
        Self::Terminal
    }
}

impl std::str::FromStr for LogFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "terminal" => Ok(Self::Terminal),
            "json" => Ok(Self::Json),
            "structured" => Ok(Self::Structured),
            _ => Err(format!("Invalid log format: {}", s)),
        }
    }
}

/// Output destination for logs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LogOutput {
    Terminal,
    Silent,
    File { path: String },
}

impl Default for LogOutput {
    fn default() -> Self {
        Self::Terminal
    }
}

/// Main logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub format: LogFormat,
    #[serde(default = "default_show_timing_detail")]
    pub show_timing_detail: bool,
    pub outputs: Vec<LogOutput>,
}

fn default_show_timing_detail() -> bool {
    false
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::default(),
            format: LogFormat::default(),
            show_timing_detail: false,
            outputs: vec![LogOutput::default()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_serialization() {
        let json = r#"{"level":"debug","format":"json","show_timing_detail":true,"outputs":[{"type":"terminal"}]}"#;
        let config: LoggingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.level, LogLevel::Debug);
        assert_eq!(config.format, LogFormat::Json);
        assert!(config.show_timing_detail);
    }

    #[test]
    fn test_config_deserialization_with_defaults() {
        let json = r#"{}"#;
        let config: LoggingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.level, LogLevel::Info);
        assert_eq!(config.format, LogFormat::Terminal);
        assert!(!config.show_timing_detail);
    }

    #[test]
    fn test_loglevel_parse() {
        assert_eq!("debug".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert_eq!("DEBUG".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert!("invalid".parse::<LogLevel>().is_err());
    }

    #[test]
    fn test_logformat_parse() {
        assert_eq!("json".parse::<LogFormat>().unwrap(), LogFormat::Json);
        assert_eq!("JSON".parse::<LogFormat>().unwrap(), LogFormat::Json);
        assert!("invalid".parse::<LogFormat>().is_err());
    }

    #[test]
    fn test_loglevel_to_filter() {
        assert_eq!(LogLevel::Info.to_level_filter(), log::LevelFilter::Info);
    }
}
