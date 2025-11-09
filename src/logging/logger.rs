//! PowersLogger implementation

use crate::logging::config::{LogFormat, LogOutput, LoggingConfig};
use crate::logging::context::LogContext;
use crate::logging::formatters::TerminalFormatter;
use log::{Log, Metadata, Record};
use std::io::{self, Write};
use std::sync::Mutex;

/// Trait for log output sinks
trait Sink: Write + Send {}

/// Terminal output sink
struct TerminalSink;

impl Write for TerminalSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::stdout().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

impl Sink for TerminalSink {}

/// Silent sink that discards all output
struct SilentSink;

impl Write for SilentSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Sink for SilentSink {}

/// The main logger implementation
pub struct PowersLogger {
    level: log::LevelFilter,
    formatter: TerminalFormatter,
    sink: Mutex<Box<dyn Sink>>,
}

impl Log for PowersLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let context = LogContext::current();
        let formatted = self.formatter.format(record, &context);

        if !formatted.is_empty() {
            let mut sink = self.sink.lock().unwrap();
            let _ = sink.write_all(&formatted);
        }
    }

    fn flush(&self) {
        let mut sink = self.sink.lock().unwrap();
        let _ = sink.flush();
    }
}

/// Initialize the global logger with the given configuration
pub fn init_logger(config: &LoggingConfig) -> Result<(), String> {
    let level = config.level.to_level_filter();

    // Determine if we should use colors (only for terminal output)
    let use_colors = config
        .outputs
        .iter()
        .any(|o| matches!(o, LogOutput::Terminal))
        && atty::is(atty::Stream::Stdout);

    // Create formatter based on config
    let formatter = match config.format {
        LogFormat::Terminal | LogFormat::Structured => {
            TerminalFormatter::new(use_colors)
        }
        LogFormat::Json => {
            // For now, JSON formatter not implemented, fallback to terminal
            TerminalFormatter::new(false)
        }
    };

    // Create sink based on first output (for now, support single output)
    let sink: Box<dyn Sink> = match config.outputs.first() {
        Some(LogOutput::Terminal) => Box::new(TerminalSink),
        Some(LogOutput::Silent) => Box::new(SilentSink),
        Some(LogOutput::File { path: _ }) => {
            // File sink not yet implemented
            return Err("File output not yet implemented".to_string());
        }
        None => Box::new(TerminalSink), // Default to terminal
    };

    let logger = PowersLogger {
        level,
        formatter,
        sink: Mutex::new(sink),
    };

    log::set_logger(Box::leak(Box::new(logger)))
        .map_err(|e| format!("Failed to initialize logger: {}", e))?;

    log::set_max_level(level);

    Ok(())
}
