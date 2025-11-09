//! PowersLogger implementation

use crate::logging::config::{LogFormat, LogOutput, LoggingConfig};
use crate::logging::context::LogContext;
use crate::logging::formatters::{JsonFormatter, TerminalFormatter};
use log::{Log, Metadata, Record};
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
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

/// File output sink with buffering
struct FileSink {
    writer: BufWriter<File>,
}

impl FileSink {
    fn new(path: &str) -> io::Result<Self> {
        // Create parent directories if needed
        if let Some(parent) = Path::new(path).parent() {
            create_dir_all(parent)?;
        }

        let file = OpenOptions::new().create(true).append(true).open(path)?;

        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl Write for FileSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.writer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl Sink for FileSink {}

/// Multi-sink that writes to multiple outputs
struct MultiSink {
    sinks: Vec<Box<dyn Sink>>,
}

impl MultiSink {
    fn new(sinks: Vec<Box<dyn Sink>>) -> Self {
        Self { sinks }
    }
}

impl Write for MultiSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        for sink in &mut self.sinks {
            sink.write_all(buf)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        for sink in &mut self.sinks {
            sink.flush()?;
        }
        Ok(())
    }
}

impl Sink for MultiSink {}

/// Enum to hold different formatter types
enum Formatter {
    Terminal(TerminalFormatter),
    Json(JsonFormatter),
}

impl Formatter {
    fn format(&self, record: &Record, context: &LogContext) -> Vec<u8> {
        match self {
            Formatter::Terminal(f) => f.format(record, context),
            Formatter::Json(f) => f.format(record, context),
        }
    }
}

/// The main logger implementation
pub struct PowersLogger {
    level: log::LevelFilter,
    formatter: Formatter,
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
            // Flush immediately to ensure logs are written (important for file outputs)
            let _ = sink.flush();
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
            Formatter::Terminal(TerminalFormatter::new(use_colors))
        }
        LogFormat::Json => Formatter::Json(JsonFormatter::new()),
    };

    // Create sinks for all configured outputs
    let mut sinks: Vec<Box<dyn Sink>> = Vec::new();

    for output in &config.outputs {
        let sink: Box<dyn Sink> = match output {
            LogOutput::Terminal => Box::new(TerminalSink),
            LogOutput::Silent => Box::new(SilentSink),
            LogOutput::File { path } => {
                Box::new(FileSink::new(path).map_err(|e| {
                    format!("Failed to open log file '{}': {}", path, e)
                })?)
            }
        };
        sinks.push(sink);
    }

    // Use multi-sink if multiple outputs, otherwise use single sink
    let sink: Box<dyn Sink> = if sinks.is_empty() {
        Box::new(TerminalSink) // Default to terminal
    } else if sinks.len() == 1 {
        sinks.into_iter().next().unwrap()
    } else {
        Box::new(MultiSink::new(sinks))
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
