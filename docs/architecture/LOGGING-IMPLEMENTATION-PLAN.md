# Logging System Implementation Plan

**Project**: POWE.RS Logging Redesign  
**Version**: 1.0  
**Status**: Ready for Implementation  
**Estimated Effort**: 8-10 weeks (part-time)

---

## Quick Reference

**Goal**: Replace ad-hoc `println!` logging with professional structured logging system  
**Approach**: Incremental, backward-compatible migration using Rust `log` crate  
**Key Benefits**: Configurable verbosity, machine-readable output, zero-cost abstractions

**Documents**:
- Design: [`LOGGING-DESIGN.md`](./LOGGING-DESIGN.md)
- This document: Implementation roadmap and task breakdown

---

## Phase 0: Preparation (Week 1)

### Objectives
- Validate design with stakeholders
- Set up development environment
- Create baseline measurements

### Tasks

#### 0.1 Design Review
- [ ] Share `LOGGING-DESIGN.md` with maintainers
- [ ] Schedule design review meeting
- [ ] Address feedback and finalize configuration schema
- [ ] Get approval to proceed

**Deliverable**: Approved design document

#### 0.2 Baseline Measurements
```bash
# Capture current CLI output
powers examples/03-multistage > baseline-output.txt 2>&1

# Run performance benchmarks
cargo bench --bench sddp_benchmarks -- --save-baseline before-logging

# Measure binary size
cargo build --release
ls -lh target/release/powers
```

**Deliverable**: Baseline files for regression testing

#### 0.3 Development Branch
```bash
git checkout -b feature/structured-logging
```

**Deliverable**: Feature branch ready for work

---

## Phase 1: Infrastructure Setup (Weeks 2-3)

### Objectives
- Add `log` crate integration
- Create logging module structure
- Implement basic terminal formatter
- Achieve zero visual changes to CLI output

### Task Breakdown

#### 1.1 Add Dependencies
**File**: `Cargo.toml`

```toml
[dependencies]
log = "0.4"

[dev-dependencies]
env_logger = "0.11"  # For tests
```

**Testing**: `cargo build` succeeds

---

#### 1.2 Create Module Structure
```bash
mkdir -p src/logging/formatters
touch src/logging/mod.rs
touch src/logging/config.rs
touch src/logging/logger.rs
touch src/logging/context.rs
touch src/logging/formatters/mod.rs
touch src/logging/formatters/terminal.rs
```

**Files to create**:

**`src/logging/mod.rs`**:
```rust
//! Professional logging system for POWE.RS
//!
//! Provides structured logging with configurable outputs and formats.

pub mod config;
pub mod context;
pub mod formatters;
pub mod logger;

pub use config::{LoggingConfig, LogLevel, LogFormat, LogOutput};
pub use context::LogContext;
pub use logger::PowersLogger;

use log::LevelFilter;

/// Initialize the logging system with the given configuration.
pub fn init(config: &LoggingConfig) -> Result<(), String> {
    logger::init_logger(config)
}

/// Get default logging configuration (backward compatible).
pub fn default_config() -> LoggingConfig {
    LoggingConfig::default()
}
```

**Testing**: `cargo build` succeeds, module is accessible

---

#### 1.3 Implement Configuration Types
**File**: `src/logging/config.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn to_level_filter(&self) -> log::LevelFilter {
        match self {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Terminal,
    Json,
    Structured,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum LogOutput {
    Terminal,
    File { path: String },
    Silent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_level")]
    pub level: LogLevel,
    
    #[serde(default = "default_format")]
    pub format: LogFormat,
    
    #[serde(default = "default_show_timing")]
    pub show_timing_detail: bool,
    
    #[serde(default = "default_show_progress")]
    pub show_progress_bar: bool,
    
    #[serde(default = "default_outputs")]
    pub outputs: Vec<LogOutput>,
}

fn default_level() -> LogLevel { LogLevel::Info }
fn default_format() -> LogFormat { LogFormat::Terminal }
fn default_show_timing() -> bool { false }
fn default_show_progress() -> bool { true }
fn default_outputs() -> Vec<LogOutput> { vec![LogOutput::Terminal] }

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_level(),
            format: default_format(),
            show_timing_detail: default_show_timing(),
            show_progress_bar: default_show_progress(),
            outputs: default_outputs(),
        }
    }
}
```

**Testing**: Add unit test
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LoggingConfig::default();
        assert!(matches!(config.level, LogLevel::Info));
        assert!(matches!(config.format, LogFormat::Terminal));
        assert!(!config.show_timing_detail);
        assert!(config.show_progress_bar);
    }

    #[test]
    fn test_config_serialization() {
        let json = r#"{"level":"debug","format":"json"}"#;
        let config: LoggingConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(config.level, LogLevel::Debug));
        assert!(matches!(config.format, LogFormat::Json));
    }
}
```

---

#### 1.4 Implement Terminal Formatter
**File**: `src/logging/formatters/terminal.rs`

```rust
use crate::log as legacy_log; // Import existing log module
use crate::logging::context::LogContext;
use log::Record;
use std::fmt::Write;

pub struct TerminalFormatter {
    use_colors: bool,
}

impl TerminalFormatter {
    pub fn new(use_colors: bool) -> Self {
        Self { use_colors }
    }

    pub fn format(&self, record: &Record, context: &LogContext) -> Vec<u8> {
        // Phase 1: Simple pass-through to existing formatters
        // This ensures zero visual changes
        
        let mut buffer = String::new();
        
        // Check if this is a structured log with iteration context
        if let Some(iteration) = context.iteration {
            if let (Some(lower), Some(simul)) = (context.lower_bound, context.simulation_cost) {
                // Training table row
                legacy_log::training_table_row(
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
        
        // Default formatting
        write!(&mut buffer, "[{}] {}", record.level(), record.args()).unwrap();
        
        buffer.push('\n');
        buffer.into_bytes()
    }
}
```

**Note**: Phase 1 uses existing `src/log.rs` functions. Phase 2 will inline them.

---

#### 1.5 Implement Logger
**File**: `src/logging/logger.rs`

```rust
use crate::logging::config::{LoggingConfig, LogOutput};
use crate::logging::context::LogContext;
use crate::logging::formatters::terminal::TerminalFormatter;
use log::{Log, Metadata, Record};
use std::io::Write;
use std::sync::Mutex;

pub struct PowersLogger {
    level: log::LevelFilter,
    formatter: TerminalFormatter,
    sink: Mutex<Box<dyn Write + Send>>,
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

pub fn init_logger(config: &LoggingConfig) -> Result<(), String> {
    let level = config.level.to_level_filter();
    
    let use_colors = match &config.outputs.first() {
        Some(LogOutput::Terminal) => atty::is(atty::Stream::Stdout),
        _ => false,
    };
    
    let formatter = TerminalFormatter::new(use_colors);
    
    let sink: Box<dyn Write + Send> = match config.outputs.first() {
        Some(LogOutput::Terminal) => Box::new(std::io::stdout()),
        Some(LogOutput::File { path }) => {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("Failed to open log file: {}", e))?;
            Box::new(std::io::BufWriter::new(file))
        }
        Some(LogOutput::Silent) | None => Box::new(std::io::sink()),
    };
    
    let logger = PowersLogger {
        level,
        formatter,
        sink: Mutex::new(sink),
    };
    
    log::set_boxed_logger(Box::new(logger))
        .map(|()| log::set_max_level(level))
        .map_err(|e| format!("Failed to set logger: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::config::LogLevel;

    #[test]
    fn test_logger_level_filtering() {
        let config = LoggingConfig {
            level: LogLevel::Info,
            ..Default::default()
        };
        
        init_logger(&config).unwrap();
        
        // INFO and above should be enabled
        assert!(log::log_enabled!(log::Level::Info));
        assert!(log::log_enabled!(log::Level::Warn));
        assert!(log::log_enabled!(log::Level::Error));
        
        // DEBUG should be disabled
        assert!(!log::log_enabled!(log::Level::Debug));
    }
}
```

**Dependencies to add**: `atty = "0.2"` (detect terminal for colors)

---

#### 1.6 Implement Context (Minimal)
**File**: `src/logging/context.rs`

```rust
use std::cell::RefCell;
use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct LogContext {
    pub iteration: Option<usize>,
    pub lower_bound: Option<f64>,
    pub simulation_cost: Option<f64>,
    pub forward_time: Option<Duration>,
    pub backward_time: Option<Duration>,
    pub total_time: Option<Duration>,
}

thread_local! {
    static CONTEXT: RefCell<LogContext> = RefCell::new(LogContext::default());
}

impl LogContext {
    pub fn current() -> Self {
        CONTEXT.with(|ctx| ctx.borrow().clone())
    }
    
    pub fn set(context: LogContext) {
        CONTEXT.with(|ctx| *ctx.borrow_mut() = context);
    }
    
    pub fn clear() {
        CONTEXT.with(|ctx| *ctx.borrow_mut() = LogContext::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_isolation() {
        LogContext::set(LogContext {
            iteration: Some(42),
            ..Default::default()
        });
        
        let ctx = LogContext::current();
        assert_eq!(ctx.iteration, Some(42));
        
        LogContext::clear();
        assert_eq!(LogContext::current().iteration, None);
    }
}
```

---

#### 1.7 Update Config Struct
**File**: `src/input.rs`

```rust
use crate::logging::LoggingConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Existing fields...
    
    #[serde(default)]
    pub logging: LoggingConfig,
}
```

**Testing**: Load existing configs (should use defaults)

---

#### 1.8 Initialize Logger in Entry Point
**File**: `src/lib.rs`

```rust
pub fn run(input_path: &Path) -> Result<(), Box<dyn Error>> {
    // Load config first
    let config_path = input_path.join("config.json");
    let config = Config::from_file(&config_path)?;
    
    // Initialize logging
    crate::logging::init(&config.logging)
        .map_err(|e| -> Box<dyn Error> { e.into() })?;
    
    // Now use log macros instead of println
    log::info!("POWE.RS - Power Optimization for the World of Energy - in pure RuSt");
    log::info!("--------------------------------------------------------------------");
    
    // ... rest of existing code
}
```

---

#### 1.9 Testing Phase 1
```bash
# Build with new logging
cargo build

# Run existing examples (should look identical)
powers examples/01-deterministic > new-output.txt 2>&1

# Compare outputs
diff baseline-output.txt new-output.txt

# Expected: No differences (or only minor formatting)
```

**Acceptance Criteria**:
- ✅ Builds without errors
- ✅ CLI output visually identical
- ✅ All tests pass
- ✅ Existing configs work without `logging` field

**Deliverable**: Working logging infrastructure, zero visual changes

---

## Phase 2: Incremental Migration (Weeks 4-6)

### Objectives
- Replace all `println!`/`eprintln!` with `log` macros
- Maintain backward compatibility
- Add structured context to logs

### Task Breakdown

#### 2.1 Training Loop Migration
**File**: `src/sddp/mod.rs`

**Priority**: High (most visible)

**Changes**:
1. Replace `log::training_greeting()` with `info!` macro
2. Add iteration context to log calls
3. Replace `log::training_table_row()` with structured logs

**Before**:
```rust
log::training_greeting(num_iterations, num_forward_passes, enable_cut_selection);
log::training_table_divider();
for iter in 0..num_iterations {
    // ...
    log::training_table_row(iter + 1, lower_bound, simul, fwd_time, bwd_time, total);
}
log::training_table_divider();
```

**After**:
```rust
info!(
    num_iterations = num_iterations,
    num_forward_passes = num_forward_passes,
    cut_selection = enable_cut_selection;
    "Starting training"
);

for iter in 0..num_iterations {
    // Set context for this iteration
    LogContext::set(LogContext {
        iteration: Some(iter + 1),
        lower_bound: Some(lower_bound),
        simulation_cost: Some(simul),
        forward_time: Some(fwd_time),
        backward_time: Some(bwd_time),
        total_time: Some(total),
    });
    
    info!("Iteration complete");
    
    LogContext::clear();
}
```

**Testing**:
```bash
# Visual check
powers examples/03-multistage

# Should still show ASCII table
```

---

#### 2.2 Simulation Migration
**File**: `src/sddp/mod.rs`

**Changes**:
1. Replace `log::simulation_greeting()` with `info!` macro
2. Replace `log::simulation_stats()` with structured log

**Before**:
```rust
log::simulation_greeting(num_scenarios);
// ...
log::simulation_stats(mean, std);
```

**After**:
```rust
info!(num_scenarios = num_scenarios; "Starting simulation");
// ...
info!(
    mean_cost = mean,
    std_cost = std;
    "Simulation complete"
);
```

---

#### 2.3 Environment Variable Replacement
**File**: `src/sddp/mod.rs` (line 2127)

**Before**:
```rust
if std::env::var("POWERS_TIMING_DETAIL").is_ok() {
    log::training_iteration_timing(...);
}
```

**After**:
```rust
if config.logging.show_timing_detail {
    debug!(
        forward_saa_ms = saa_sampling_time.as_millis(),
        forward_model_prep_ms = forward_model_pre_time.as_millis(),
        forward_solver_ms = forward_solver_time.as_millis(),
        // ... all timing fields
        "Detailed timing breakdown"
    );
}
```

**Note**: Formatter will render this with box-drawing characters (reuse from `log.rs`)

---

#### 2.4 Error Handling Migration
**Files**: `src/main.rs`, `src/input.rs`, `src/subproblem.rs`

**Replace all `eprintln!` with `error!` or `warn!`**:

**Before** (`src/main.rs`):
```rust
if let Err(e) = result {
    eprintln!("Error: {}", e);
    process::exit(1);
}
```

**After**:
```rust
if let Err(e) = result {
    log::error!("Execution failed: {}", e);
    process::exit(1);
}
```

**Before** (`src/subproblem.rs`):
```rust
eprintln!("[ERROR] Solver infeasible! Let me check the constraint structure:");
```

**After**:
```rust
log::error!("Solver infeasible, diagnosing constraint structure");
```

---

#### 2.5 Debug Logging Migration
**File**: `src/subproblem.rs`

**Uncomment/enable debug logging with `debug!` macro**:

**Before** (commented out or always-on):
```rust
// eprintln!("Model exists: {}", subproblem.model.is_some());
```

**After**:
```rust
debug!(
    model_exists = subproblem.model.is_some(),
    "Subproblem state"
);
```

**Testing**: Run with `--log-level debug` to see output

---

#### 2.6 Integration Testing
```bash
# Test all examples
for example in examples/*/; do
    echo "Testing $example"
    powers "$example" > /dev/null 2>&1 || echo "FAILED: $example"
done

# Test with different log levels
powers examples/03-multistage --log-level warn
powers examples/03-multistage --log-level debug

# Test library usage (no output)
cargo test --lib -- --nocapture
```

**Acceptance Criteria**:
- ✅ All examples produce expected output
- ✅ `--log-level` flag works
- ✅ Library tests don't pollute output
- ✅ No `println!`/`eprintln!` remain in source

**Deliverable**: Fully migrated codebase using `log` macros

---

## Phase 3: Advanced Features (Weeks 7-8)

### Objectives
- Implement JSON formatter
- Add file output support
- Enhance formatters with better rendering

### Task Breakdown

#### 3.1 JSON Formatter
**File**: `src/logging/formatters/json.rs`

```rust
use crate::logging::context::LogContext;
use log::Record;
use serde::Serialize;
use chrono::Utc;

#[derive(Serialize)]
struct JsonLogEntry {
    timestamp: String,
    level: String,
    message: String,
    #[serde(flatten)]
    context: serde_json::Value,
}

pub struct JsonFormatter;

impl JsonFormatter {
    pub fn format(&self, record: &Record, context: &LogContext) -> Vec<u8> {
        let entry = JsonLogEntry {
            timestamp: Utc::now().to_rfc3339(),
            level: record.level().to_string(),
            message: record.args().to_string(),
            context: serde_json::to_value(context).unwrap_or_default(),
        };
        
        let mut json = serde_json::to_vec(&entry).unwrap();
        json.push(b'\n'); // JSON Lines format
        json
    }
}
```

**Testing**:
```bash
# Generate JSON logs
powers examples/03-multistage --log-format json > training.jsonl

# Validate JSON
cat training.jsonl | while read line; do echo "$line" | jq .; done
```

---

#### 3.2 File Sink Implementation
**File**: `src/logging/logger.rs`

**Add support for multiple outputs**:
```rust
pub struct MultiSink {
    sinks: Vec<Box<dyn Write + Send>>,
}

impl Write for MultiSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for sink in &mut self.sinks {
            sink.write_all(buf)?;
        }
        Ok(buf.len())
    }
    
    fn flush(&mut self) -> std::io::Result<()> {
        for sink in &mut self.sinks {
            sink.flush()?;
        }
        Ok(())
    }
}
```

**Config example**:
```json
{
  "logging": {
    "outputs": [
      {"type": "terminal", "level": "INFO"},
      {"type": "file", "path": "./logs/debug.log", "level": "DEBUG"}
    ]
  }
}
```

---

#### 3.3 CLI Flag Support
**File**: `src/cli.rs`

```rust
#[derive(Parser)]
pub struct Cli {
    /// Log level (error, warn, info, debug, trace)
    #[arg(long, global = true)]
    pub log_level: Option<String>,
    
    /// Log format (terminal, json)
    #[arg(long, global = true)]
    pub log_format: Option<String>,
    
    // ... existing fields
}
```

**File**: `src/main.rs`

```rust
// Override config with CLI args
if let Some(level) = cli.log_level {
    config.logging.level = level.parse()?;
}
if let Some(format) = cli.log_format {
    config.logging.format = format.parse()?;
}
```

---

#### 3.4 Documentation
**Create**: `docs/guides/LOGGING-GUIDE.md`

**Contents**:
- Configuration options
- Log levels and when to use them
- Examples (terminal, JSON, silent)
- Troubleshooting

---

#### 3.5 Testing Phase 3
```bash
# Test JSON output
powers examples/03-multistage --log-format json > test.jsonl
cat test.jsonl | jq -c . | wc -l  # Count log entries

# Test file output
powers examples/03-multistage --log-file ./test.log
test -f test.log && echo "Log file created"

# Test CLI overrides
powers examples/03-multistage --log-level debug --log-format terminal
```

**Acceptance Criteria**:
- ✅ JSON logs are valid JSON Lines
- ✅ File output works, no terminal output when configured
- ✅ CLI flags override config file
- ✅ Documentation complete

**Deliverable**: Full-featured logging system with docs

---

## Phase 4: Cleanup (Week 9)

### Objectives
- Remove deprecated code
- Update all documentation
- Final polish

### Task Breakdown

#### 4.1 Remove `src/log.rs`
```bash
git rm src/log.rs
```

**Update references**:
- `src/lib.rs`: Remove `mod log;`
- `src/sddp/mod.rs`: Remove `use crate::log;`

**Testing**: `cargo build` should succeed

---

#### 4.2 Remove Environment Variable Support
**Search and remove**:
```bash
grep -r "POWERS_TIMING_DETAIL" src/
# Remove all occurrences
```

---

#### 4.3 Update Documentation
**Files to update**:
- `README.md`: Add logging section
- `CHANGELOG.md`: Add v0.X.0 entry
- `docs/reference/INPUT-SPECIFICATION.md`: Document `logging` config

**README example**:
```markdown
### Logging Configuration

Control verbosity and output format:

```json
{
  "logging": {
    "level": "INFO",
    "format": "terminal",
    "show_timing_detail": false
  }
}
```

Or use CLI flags:
```bash
powers examples/03-multistage --log-level debug
```
```

---

#### 4.4 Update JSON Schema
**File**: `schemas/config.schema.json`

```json
{
  "properties": {
    "logging": {
      "type": "object",
      "properties": {
        "level": {
          "type": "string",
          "enum": ["error", "warn", "info", "debug", "trace"],
          "default": "info"
        },
        "format": {
          "type": "string",
          "enum": ["terminal", "json", "structured"],
          "default": "terminal"
        },
        "show_timing_detail": {
          "type": "boolean",
          "default": false
        },
        "show_progress_bar": {
          "type": "boolean",
          "default": true
        }
      }
    }
  }
}
```

---

#### 4.5 Final Testing
```bash
# Full test suite
cargo test --all-features

# Benchmarks (check for regressions)
cargo bench --bench sddp_benchmarks -- --baseline before-logging

# CLI visual check
powers examples/03-multistage
powers examples/05-large-scale-brazilian --log-level debug

# Library usage
cargo test --doc
```

**Acceptance Criteria**:
- ✅ No references to `src/log.rs`
- ✅ No environment variable checks
- ✅ All tests pass
- ✅ No performance regression (< 1%)
- ✅ Documentation complete
- ✅ Schema updated

**Deliverable**: Clean, production-ready logging system

---

## Phase 5: Release (Week 10)

### Objectives
- Prepare for merge
- Update version
- Announce changes

### Task Breakdown

#### 5.1 Version Bump
**File**: `Cargo.toml`

```toml
version = "0.3.0"  # Minor version bump (new feature)
```

---

#### 5.2 CHANGELOG Update
**File**: `CHANGELOG.md`

```markdown
## [0.3.0] - 2025-XX-XX

### Added
- Structured logging system with `log` crate
- Configurable log levels (ERROR, WARN, INFO, DEBUG, TRACE)
- JSON log format for machine-readable output
- File output support for logs
- CLI flags for log level override (`--log-level`, `--log-format`)

### Changed
- Replaced ad-hoc `println!` calls with structured logging
- Logging now configured via `config.json` (`logging` field)
- Environment variable `POWERS_TIMING_DETAIL` replaced with `show_timing_detail` config

### Removed
- `src/log.rs` module (functionality moved to `src/logging/`)

### Migration Guide
Old configs without `logging` field will use sensible defaults (INFO level, terminal output).
To enable timing detail, add to `config.json`:
```json
{
  "logging": {
    "show_timing_detail": true
  }
}
```
```

---

#### 5.3 Pull Request
```bash
# Rebase on main
git fetch origin main
git rebase origin/main

# Push feature branch
git push origin feature/structured-logging

# Create PR with template
```

**PR Description Template**:
```markdown
# Structured Logging System

## Summary
Replaces ad-hoc `println!` logging with professional structured logging using the `log` crate.

## Changes
- Configurable log levels (ERROR/WARN/INFO/DEBUG/TRACE)
- Multiple output formats (terminal, JSON)
- File output support
- CLI flag overrides
- Zero performance overhead when disabled

## Testing
- ✅ All existing tests pass
- ✅ CLI output visually identical (default config)
- ✅ Benchmarks show < 1% regression
- ✅ New integration tests for logging

## Documentation
- `docs/architecture/LOGGING-DESIGN.md` - Full design document
- `docs/guides/LOGGING-GUIDE.md` - User guide
- Updated README and CHANGELOG

## Migration
Backward compatible. Old configs work without changes.

## Reviewers
@maintainer1 @maintainer2
```

---

#### 5.4 Review and Merge
- Address review feedback
- Get approval from maintainers
- Merge to main
- Tag release

---

## Rollback Plan

If critical issues discovered after merge:

### Option 1: Hotfix
```bash
git revert <merge-commit>
# Create hotfix branch, fix issue, re-merge
```

### Option 2: Feature Flag
**Add to `Cargo.toml`**:
```toml
[features]
default = ["structured-logging"]
structured-logging = []
```

**In code**:
```rust
#[cfg(feature = "structured-logging")]
crate::logging::init(&config.logging)?;

#[cfg(not(feature = "structured-logging"))]
// Use old approach (keep src/log.rs in git history)
```

### Option 3: Config Flag
**Add to config**:
```json
{
  "logging": {
    "use_legacy": true
  }
}
```

---

## Success Metrics

### Functional
- ✅ CLI output unchanged (default config)
- ✅ All tests pass
- ✅ JSON logs parseable
- ✅ Debug mode shows timing detail

### Performance
- ✅ < 1% regression in training time
- ✅ < 10KB binary size increase
- ✅ Zero overhead for disabled log levels

### Quality
- ✅ 100% coverage for logging module
- ✅ Documentation complete
- ✅ No clippy warnings
- ✅ Passes `cargo fmt`

---

## Dependencies

### New Crates
- `log = "0.4"` - Logging facade (essential)
- `atty = "0.2"` - Terminal detection (for colors)
- `env_logger = "0.11"` - For tests (dev-dependency)

### Optional (Future)
- `indicatif = "0.17"` - Progress bars (Phase 3+)
- `tracing = "0.1"` - If we need spans/tracing (not v1.0)

---

## Risk Mitigation

### Risk: Output format changes
**Mitigation**: Extensive baseline testing, visual QA

### Risk: Performance regression
**Mitigation**: Benchmarks in CI, compile-time filtering

### Risk: Library users see unwanted output
**Mitigation**: Default to silent for library, only CLI enables

### Risk: Breaking existing workflows
**Mitigation**: Backward-compatible defaults, migration guide

---

## Team Responsibilities

**Developer 1** (Weeks 2-3):
- Phase 1: Infrastructure setup
- Unit tests

**Developer 2** (Weeks 4-6):
- Phase 2: Code migration
- Integration tests

**Developer 1** (Weeks 7-8):
- Phase 3: Advanced features
- Documentation

**Both** (Weeks 9-10):
- Phase 4-5: Cleanup and release
- Final QA

---

## Questions & Support

**Slack channel**: `#powers-logging-redesign`  
**Design doc**: `docs/architecture/LOGGING-DESIGN.md`  
**Progress tracker**: GitHub Project Board

**Office hours**: Wednesdays 2-3pm for Q&A

---

**End of Implementation Plan**
