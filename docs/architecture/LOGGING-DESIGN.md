# Logging Architecture Design for POWE.RS

**Status**: Design Proposal  
**Created**: 2025-11-09  
**Author**: Code Review Agent  
**Version**: 1.0

---

## Executive Summary

This document proposes a professional, structured logging system for POWE.RS to replace the current improvised approach. The design balances performance, flexibility, and maintainability while respecting the project's zero-allocation hot path requirements.

**Current State**: Ad-hoc `println!` calls, environment variable flags, ASCII art tables  
**Proposed State**: Structured logging with levels, contexts, and configurable outputs  
**Migration Path**: Incremental, backward-compatible, non-breaking

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Design Principles](#2-design-principles)
3. [Logging Architecture](#3-logging-architecture)
4. [Implementation Specification](#4-implementation-specification)
5. [Migration Plan](#5-migration-plan)
6. [Testing Strategy](#6-testing-strategy)
7. [Performance Considerations](#7-performance-considerations)
8. [Comparison with Similar Projects](#8-comparison-with-similar-projects)

---

## 1. Current State Analysis

### 1.1 Current Logging Patterns

**Location**: `src/log.rs` (209 lines), scattered `println!`/`eprintln!` across codebase

**Identified Patterns**:

1. **Training Progress Table** (ASCII art with separators)
   - Header: `iter | lower ($) | simul ($) | fwd | bwd | total`
   - Row-by-row iteration updates
   - Dividers: 88-character dash lines

2. **Detailed Timing Breakdown** (Environment-gated)
   - Triggered by: `POWERS_TIMING_DETAIL` environment variable
   - Uses Unicode box-drawing characters (`┌─┐`, `│`, `└─┘`)
   - Shows: SAA sampling, model prep, solver, cut selection times

3. **Statistics Reporting**
   - Simple `mean ± std` format
   - Scientific notation for costs (`{:.6e}`)
   - Gap as percentage with 4 decimals

4. **Debug Logging** (Commented/conditional)
   - Solver infeasibility diagnostics in `subproblem.rs`
   - Model structure dumps (num_cols, num_rows)
   - Currently using `eprintln!` (always to stderr)

5. **Error Reporting**
   - `eprintln!` for validation failures
   - Process exit on errors in `main.rs`

**Issues**:
- ❌ No log levels (INFO/WARN/ERROR mixed)
- ❌ No structured context (iteration, node, thread)
- ❌ Environment variables for feature flags (not config-driven)
- ❌ ASCII art not machine-parseable
- ❌ No ability to redirect/filter logs
- ❌ Debug code commented out or always-on
- ❌ No integration with Rust logging ecosystem

---

### 1.2 User Experience Requirements

From README and examples, users expect:

1. **Visual Progress Tracking**
   - Real-time iteration updates during training
   - Clear convergence indicators
   - Timing information (helpful for tuning)

2. **Statistical Summaries**
   - Confidence intervals (mean ± std)
   - Gap metrics for convergence assessment
   - Final policy quality metrics

3. **Optional Debugging**
   - Detailed timing breakdowns (currently via env var)
   - Solver state inspection (for debugging infeasibility)
   - Model structure diagnostics

4. **Machine-Readable Output**
   - CSV files for analysis (already implemented)
   - Structured data extraction from logs (missing)

---

## 2. Design Principles

### 2.1 Core Principles

1. **Performance First**
   - Zero allocation in hot paths (solver calls, forward/backward passes)
   - Logging decisions at compile time where possible
   - Async/buffered output for high-frequency events

2. **Flexibility Without Complexity**
   - Support CLI, library, and benchmark use cases
   - Config-driven behavior (not env vars)
   - Graceful degradation (missing config = sensible defaults)

3. **Backward Compatibility**
   - Maintain visual output format for CLI users
   - Support library users who don't want terminal output
   - Migration path for existing code

4. **Professional Standards**
   - Standard log levels (TRACE/DEBUG/INFO/WARN/ERROR)
   - Structured context (iteration, stage, thread)
   - Integration with Rust ecosystem (`log` or `tracing` facade)

5. **Maintainability**
   - Clear separation: logging vs. metrics vs. output files
   - Single responsibility: log module focuses on logging
   - No logging logic in hot paths

---

### 2.2 Anti-Goals

❌ **Not trying to**:
- Replace CSV output files (they serve different purpose)
- Add distributed tracing (overkill for single-machine workload)
- Support runtime log level changes (compile-time is sufficient)
- Create custom logging framework (use existing crates)

---

## 3. Logging Architecture

### 3.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (SDDP algorithm, CLI, library API, benchmarks)             │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Logging Facade Layer                      │
│  (log crate - provides macros: trace!, debug!, info!, ...)  │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                 POWE.RS Logger Implementation               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  PowersLogger (log::Log impl)                        │   │
│  │  - Formatting (terminal vs. structured)             │   │
│  │  - Filtering (log level, target modules)            │   │
│  │  - Context enrichment (iteration, thread, stage)    │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                     Output Sinks                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Terminal   │  │  JSON File   │  │    Silent    │       │
│  │   (pretty)   │  │ (structured) │  │ (benchmarks) │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**Why `log` crate?**
- ✅ Zero-cost abstraction (compile-time filtering)
- ✅ Standard in Rust ecosystem (used by 80%+ of crates)
- ✅ No runtime overhead when disabled
- ✅ Easy to swap implementations (env_logger, tracing, custom)
- ✅ Minimal dependencies (just the facade)

**Why not `tracing`?**
- ⚠️ More complex (spans, events, subscribers)
- ⚠️ Higher overhead (even with compile-time filtering)
- ⚠️ Overkill for this use case (no distributed systems)
- ✅ Could migrate later if needed (facades are compatible)

---

### 3.2 Log Levels

| Level   | Usage                                      | Examples                                    |
|---------|-------------------------------------------|---------------------------------------------|
| `ERROR` | Unrecoverable failures                     | Solver crashes, file I/O failures          |
| `WARN`  | Recoverable issues                         | Solver retries, numerical instability      |
| `INFO`  | High-level progress (default)              | Iteration updates, training completion     |
| `DEBUG` | Detailed diagnostics                       | Timing breakdowns, cut statistics          |
| `TRACE` | Hot path events (disabled in release)      | Subproblem solves, state updates           |

**Recommendation**: Default to `INFO` for CLI, `WARN` for library/benchmarks.

---

### 3.3 Logging Contexts

**Structured fields** to enrich log messages:

```rust
struct LogContext {
    // Training context
    iteration: Option<usize>,
    forward_pass: Option<usize>,
    
    // Spatial context
    stage: Option<usize>,
    node: Option<usize>,
    
    // Execution context
    thread_id: Option<usize>,
    phase: Option<Phase>,  // Forward, Backward, Simulation
    
    // Timing context
    elapsed_ms: Option<u64>,
}

enum Phase {
    Training,
    Simulation,
    Forward,
    Backward,
}
```

**Usage**:
```rust
info!(
    iteration = iter,
    lower_bound = lower,
    simulation_cost = simul;
    "Iteration complete"
);
```

**Output (terminal)**:
```
[INFO] Iteration 42 complete: lower=$1.23e6, simul=$1.25e6
```

**Output (JSON)**:
```json
{
  "level": "INFO",
  "timestamp": "2025-11-09T13:24:48Z",
  "iteration": 42,
  "lower_bound": 1230000.0,
  "simulation_cost": 1250000.0,
  "message": "Iteration complete"
}
```

---

### 3.4 Configuration Schema

**Add to `config.json`**:

```json
{
  "logging": {
    "level": "INFO",
    "format": "terminal",
    "show_timing_detail": false,
    "show_progress_bar": true,
    "outputs": [
      {
        "type": "terminal",
        "level": "INFO"
      }
    ]
  }
}
```

**Fields**:
- `level`: Global log level filter (ERROR/WARN/INFO/DEBUG/TRACE)
- `format`: Output format (`terminal`, `json`, `structured`)
- `show_timing_detail`: Replace `POWERS_TIMING_DETAIL` env var
- `show_progress_bar`: Enable ASCII progress table
- `outputs`: Array of sinks (terminal, file, both)

**Defaults (backward-compatible)**:
```rust
LoggingConfig {
    level: LogLevel::Info,
    format: LogFormat::Terminal,
    show_timing_detail: false,
    show_progress_bar: true,
    outputs: vec![LogOutput::Terminal],
}
```

---

## 4. Implementation Specification

### 4.1 Module Structure

```
src/
├── logging/
│   ├── mod.rs              # Public API, logger initialization
│   ├── config.rs           # LoggingConfig struct, deserialization
│   ├── logger.rs           # PowersLogger (log::Log implementation)
│   ├── formatters/
│   │   ├── mod.rs
│   │   ├── terminal.rs     # ANSI colors, ASCII art tables
│   │   ├── json.rs         # JSON Lines format
│   │   └── structured.rs   # Key-value pairs
│   ├── context.rs          # LogContext, thread-local storage
│   └── macros.rs           # Convenience macros (optional)
└── log.rs                  # DEPRECATED: Keep for backward compat (Phase 1)
```

**Migration**: `src/log.rs` → `src/logging/formatters/terminal.rs` (logic reuse)

---

### 4.2 Core Types

#### `PowersLogger`

```rust
use log::{Log, Metadata, Record};
use std::sync::Mutex;

pub struct PowersLogger {
    config: LoggingConfig,
    formatter: Box<dyn LogFormatter>,
    sink: Mutex<Box<dyn LogSink>>,
}

impl Log for PowersLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.config.level
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let context = LogContext::current(); // Thread-local
        let formatted = self.formatter.format(record, &context);
        
        let mut sink = self.sink.lock().unwrap();
        sink.write(&formatted);
    }

    fn flush(&self) {
        let mut sink = self.sink.lock().unwrap();
        sink.flush();
    }
}
```

#### `LogFormatter` Trait

```rust
pub trait LogFormatter: Send + Sync {
    fn format(&self, record: &Record, context: &LogContext) -> Vec<u8>;
}

pub struct TerminalFormatter {
    use_colors: bool,
    show_timestamp: bool,
}

impl LogFormatter for TerminalFormatter {
    fn format(&self, record: &Record, context: &LogContext) -> Vec<u8> {
        // Format with ANSI colors, preserve ASCII art tables
        // ...
    }
}

pub struct JsonFormatter;

impl LogFormatter for JsonFormatter {
    fn format(&self, record: &Record, context: &LogContext) -> Vec<u8> {
        // Serialize to JSON Lines format
        serde_json::to_vec(&LogEntry {
            level: record.level(),
            timestamp: Utc::now(),
            message: record.args().to_string(),
            context: context.clone(),
        }).unwrap()
    }
}
```

#### `LogSink` Trait

```rust
pub trait LogSink: Send {
    fn write(&mut self, data: &[u8]);
    fn flush(&mut self);
}

pub struct TerminalSink {
    handle: std::io::Stdout,
}

pub struct FileSink {
    file: std::io::BufWriter<std::fs::File>,
}

pub struct SilentSink; // For benchmarks
```

---

### 4.3 Context Management

**Thread-local storage** for zero-cost context passing:

```rust
use std::cell::RefCell;

thread_local! {
    static LOG_CONTEXT: RefCell<LogContext> = RefCell::new(LogContext::default());
}

impl LogContext {
    pub fn current() -> Self {
        LOG_CONTEXT.with(|ctx| ctx.borrow().clone())
    }
    
    pub fn with_iteration<F, R>(iteration: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        LOG_CONTEXT.with(|ctx| {
            let prev = ctx.borrow().iteration;
            ctx.borrow_mut().iteration = Some(iteration);
            let result = f();
            ctx.borrow_mut().iteration = prev;
            result
        })
    }
}

// Usage in training loop
LogContext::with_iteration(iter, || {
    info!("Starting forward pass");
    // All logs in this scope have iteration context
});
```

**Performance**: Zero allocation, single `RefCell` per thread.

---

### 4.4 Initialization API

```rust
// In src/logging/mod.rs
pub fn init(config: &LoggingConfig) -> Result<(), String> {
    let formatter: Box<dyn LogFormatter> = match config.format {
        LogFormat::Terminal => Box::new(TerminalFormatter::new(config)),
        LogFormat::Json => Box::new(JsonFormatter),
        LogFormat::Structured => Box::new(StructuredFormatter),
    };
    
    let sink: Box<dyn LogSink> = match config.outputs.first() {
        Some(LogOutput::Terminal) => Box::new(TerminalSink::new()),
        Some(LogOutput::File(path)) => Box::new(FileSink::new(path)?),
        Some(LogOutput::Silent) => Box::new(SilentSink),
        None => Box::new(SilentSink), // Default for library use
    };
    
    let logger = PowersLogger {
        config: config.clone(),
        formatter,
        sink: Mutex::new(sink),
    };
    
    log::set_boxed_logger(Box::new(logger))
        .map(|()| log::set_max_level(config.level.to_level_filter()))
        .map_err(|e| format!("Failed to initialize logger: {}", e))
}

// In src/lib.rs::run()
pub fn run(input_path: &Path) -> Result<(), Box<dyn Error>> {
    let config = Config::from_files(...)?;
    
    // Initialize logging
    crate::logging::init(&config.logging)
        .map_err(|e| -> Box<dyn Error> { e.into() })?;
    
    info!("Starting SDDP algorithm");
    // ...
}
```

---

### 4.5 Migration of Existing Logs

**Phase 1: Replace direct calls**

```diff
- println!("\nPOWE.RS - Power Optimization for the World of Energy - in pure RuSt");
+ info!("POWE.RS - Power Optimization for the World of Energy - in pure RuSt");

- eprintln!("Error: {}", e);
+ error!("Execution failed: {}", e);

- if std::env::var("POWERS_TIMING_DETAIL").is_ok() {
-     log::training_iteration_timing(...);
- }
+ if config.logging.show_timing_detail {
+     debug!(
+         forward_time = ?forward_timing,
+         backward_time = ?backward_timing;
+         "Iteration timing breakdown"
+     );
+ }
```

**Phase 2: Structured logging**

```rust
// Old (ad-hoc table formatting)
log::training_table_row(
    iteration,
    lower_bound,
    simulation_cost,
    forward_time,
    backward_time,
    total_time,
);

// New (structured logging, formatter handles display)
info!(
    iteration = iteration,
    lower_bound = lower_bound,
    simulation_cost = simulation_cost,
    forward_time_ms = forward_time.as_millis(),
    backward_time_ms = backward_time.as_millis(),
    total_time_ms = total_time.as_millis();
    "Iteration complete"
);

// TerminalFormatter renders this as the familiar ASCII table
// JsonFormatter renders as structured JSON
```

---

## 5. Migration Plan

### Phase 0: Preparation (No Code Changes)

**Duration**: 1 week  
**Goal**: Validate design with stakeholders

- [ ] Review this document with maintainers
- [ ] Gather feedback on logging requirements
- [ ] Finalize configuration schema
- [ ] Create example configurations

---

### Phase 1: Infrastructure Setup

**Duration**: 2 weeks  
**Goal**: Add logging infrastructure without breaking existing code

**Tasks**:
1. Add `log` dependency to `Cargo.toml`
2. Create `src/logging/` module structure
3. Implement `PowersLogger` with `TerminalFormatter` (reuse `log.rs` logic)
4. Implement `LoggingConfig` with sensible defaults
5. Update `Config` struct to include `logging` field (optional, with default)
6. Initialize logger in `lib.rs::run()` (with fallback if config missing)
7. Add tests for formatter output
8. Keep `src/log.rs` unchanged (backward compatibility)

**Testing**:
- Unit tests for formatters (text output matches current format)
- Integration test: CLI runs with new logger produce identical output
- Benchmark: Verify zero overhead when log level is disabled

**Deliverables**:
- `src/logging/` module with `log` facade integration
- Updated `config.json` schema (backward-compatible)
- CI passing with no visual changes

---

### Phase 2: Incremental Migration

**Duration**: 3 weeks  
**Goal**: Replace `println!`/`eprintln!` calls with `log` macros

**Priorities** (by volume and impact):

1. **Training loop** (highest visibility)
   - `src/sddp/mod.rs`: iteration updates, timing
   - Replace `log::training_*` calls with `info!` macros

2. **Simulation**
   - `src/sddp/mod.rs`: simulation statistics
   - Replace `log::simulation_*` calls with `info!` macros

3. **Greetings and farewells**
   - `src/lib.rs`: entry/exit messages
   - Replace with `info!` macros

4. **Error handling**
   - `src/main.rs`, `src/input.rs`: validation failures
   - Replace `eprintln!` with `error!` macros

5. **Debug logging**
   - `src/subproblem.rs`: solver diagnostics
   - Replace commented/conditional `eprintln!` with `debug!` macros

**Per-module checklist**:
- [ ] Replace print statements with log macros
- [ ] Add context (iteration, stage) where available
- [ ] Update tests if output is validated
- [ ] Manual QA: CLI output visually identical

**Testing**:
- Regression tests: CLI output matches baseline (with default config)
- Integration tests: Library usage produces no terminal output (when configured)
- Benchmark tests: Silent logging has zero overhead

---

### Phase 3: Advanced Features

**Duration**: 2 weeks  
**Goal**: Leverage structured logging benefits

**Tasks**:
1. Implement `JsonFormatter` for structured output
2. Add file sink support (`LogOutput::File`)
3. Implement progress bar (using `indicatif` crate)
4. Add timing detail formatting (replace ASCII art with debug logs)
5. Document logging configuration in user guide
6. Add examples: JSON logs, silent mode, debug mode

**Testing**:
- Test JSON output parsing (validate schema)
- Test file sink (log rotation, permissions)
- Visual QA: Progress bar doesn't interfere with logs

**Deliverables**:
- Multiple output formats (terminal, JSON, file)
- User documentation: `docs/guides/LOGGING-GUIDE.md`
- Example configurations

---

### Phase 4: Cleanup

**Duration**: 1 week  
**Goal**: Remove deprecated code

**Tasks**:
1. Remove `src/log.rs` (functionality moved to `src/logging/`)
2. Remove `POWERS_TIMING_DETAIL` environment variable support
3. Update all documentation references
4. Add deprecation notice in CHANGELOG

**Testing**:
- Full regression suite
- Update benchmark baselines if needed

---

### Rollback Plan

If issues arise during migration:

1. **Phase 1-2**: Keep `src/log.rs`, use feature flag to toggle new logger
   ```rust
   #[cfg(feature = "new-logging")]
   crate::logging::init(&config.logging)?;
   #[cfg(not(feature = "new-logging"))]
   // Use old approach
   ```

2. **Phase 3-4**: Revert specific features (JSON, file sinks)

3. **Emergency**: Revert entire PR, keep `log` facade calls but implement no-op logger

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Test formatter output**:
```rust
#[test]
fn test_terminal_formatter_iteration_line() {
    let formatter = TerminalFormatter::new(&default_config());
    let record = log::Record::builder()
        .level(log::Level::Info)
        .args(format_args!("Iteration complete"))
        .build();
    
    let context = LogContext {
        iteration: Some(42),
        lower_bound: Some(1.23e6),
        ..Default::default()
    };
    
    let output = String::from_utf8(formatter.format(&record, &context)).unwrap();
    
    assert!(output.contains("42"));
    assert!(output.contains("1.230000e6"));
}
```

**Test log level filtering**:
```rust
#[test]
fn test_debug_logs_filtered_at_info_level() {
    let config = LoggingConfig {
        level: LogLevel::Info,
        ..Default::default()
    };
    let logger = PowersLogger::new(config);
    
    // Debug log should be filtered
    let metadata = log::Metadata::builder()
        .level(log::Level::Debug)
        .build();
    
    assert!(!logger.enabled(&metadata));
}
```

---

### 6.2 Integration Tests

**Test CLI output** (regression test):
```rust
#[test]
fn test_cli_output_matches_baseline() {
    let output = Command::new("powers")
        .arg("examples/01-deterministic")
        .output()
        .unwrap();
    
    let stdout = String::from_utf8(output.stdout).unwrap();
    
    // Check for expected sections
    assert!(stdout.contains("POWE.RS - Power Optimization"));
    assert!(stdout.contains("# Training"));
    assert!(stdout.contains("Training time:"));
    
    // Verify table structure (not exact values, they may vary)
    assert!(stdout.contains("iter |"));
    assert!(stdout.contains("lower ($)"));
}
```

**Test library API** (no output):
```rust
#[test]
fn test_library_api_silent_by_default() {
    let mut sddp = SddpAlgorithm::from_files(...)?;
    
    // Capture stdout/stderr
    let output = gag::BufferRedirect::stdout().unwrap();
    
    sddp.train()?;
    
    let captured = output.into_string()?;
    
    // Library use should not print to terminal by default
    assert!(captured.is_empty() || captured.trim().is_empty());
}
```

---

### 6.3 Performance Tests

**Benchmark disabled logging** (should be zero-cost):
```rust
#[bench]
fn bench_disabled_debug_log(b: &mut Bencher) {
    // Initialize logger with INFO level (DEBUG is disabled)
    crate::logging::init(&LoggingConfig {
        level: LogLevel::Info,
        ..Default::default()
    })?;
    
    b.iter(|| {
        // This should be compiled out (zero cost)
        debug!("Hot path debug log");
    });
}
```

Expected: < 1ns per call (optimized away by compiler).

**Benchmark enabled logging**:
```rust
#[bench]
fn bench_enabled_info_log(b: &mut Bencher) {
    crate::logging::init(&LoggingConfig {
        level: LogLevel::Info,
        outputs: vec![LogOutput::Silent], // Avoid I/O overhead
        ..Default::default()
    })?;
    
    b.iter(|| {
        info!("Training iteration {}", 42);
    });
}
```

Expected: < 100ns per call (formatting + lock + no I/O).

---

### 6.4 Visual QA Checklist

**Before merging Phase 2** (manual testing):

- [ ] Run `powers examples/03-multistage` → output matches current format
- [ ] Run with `logging.level = "DEBUG"` → see detailed timing
- [ ] Run with `logging.show_progress_bar = false` → no table
- [ ] Run as library → no terminal output
- [ ] Run benchmark → silent operation
- [ ] Check ANSI colors in terminal (if `use_colors = true`)
- [ ] Check output in non-terminal (colors disabled)

---

## 7. Performance Considerations

### 7.1 Zero-Cost Abstractions

**Compile-time filtering**:
```rust
// At compile time, if max_level < DEBUG, this entire block is removed
debug!("Expensive computation: {}", expensive_fn());
```

**Key insight**: `log` crate uses macros that check log level at compile time. If `max_level` is set to `INFO`, all `debug!` and `trace!` calls are **completely removed** from the binary.

**Measurement**: Compare binary size with `max_level = INFO` vs. `max_level = TRACE`.

---

### 7.2 Hot Path Guidelines

**Rules for hot path code** (subproblem solve, state updates):

1. **Use `trace!` level** (disabled in release builds)
   ```rust
   trace!("Solving subproblem for node {}", node_id);
   ```

2. **Defer expensive computations**
   ```rust
   // BAD: Always computes, even if DEBUG is disabled
   debug!("State: {}", format_state_debug(&state));
   
   // GOOD: Only computes if DEBUG is enabled
   debug!("State: {:?}", state); // Use Debug trait
   ```

3. **Aggregate, don't log per-iteration**
   ```rust
   // BAD: Log every subproblem solve (100k+ logs)
   for node in nodes {
       debug!("Solving node {}", node.id);
       solve(node);
   }
   
   // GOOD: Log once per pass
   debug!("Solving {} nodes", nodes.len());
   for node in nodes {
       solve(node); // No logging
   }
   ```

4. **Use context, not parameters**
   ```rust
   // BAD: Passing extra parameters for logging
   fn solve_subproblem(node: &Node, iteration: usize, stage: usize) {
       info!("Solving node {} at iteration {}", node.id, iteration);
   }
   
   // GOOD: Context is thread-local (zero parameter overhead)
   fn solve_subproblem(node: &Node) {
       // Context is automatically enriched
       trace!("Solving node {}", node.id);
   }
   ```

---

### 7.3 Benchmarking

**Baseline measurement** (before migration):
```bash
cargo bench --bench sddp_benchmarks -- --save-baseline before-logging
```

**Post-migration measurement** (with logging at INFO level):
```bash
cargo bench --bench sddp_benchmarks -- --baseline before-logging
```

**Acceptance criteria**: < 1% regression in total training time.

---

## 8. Comparison with Similar Projects

### 8.1 Scientific Computing Projects

**Polars (DataFrame library)**:
- Uses: `env_logger` (simple, configurable via env vars)
- Levels: INFO for progress, DEBUG for query plans
- Insight: CSV parsing shows progress bar via `indicatif`

**nalgebra (linear algebra)**:
- Uses: No logging framework (library leaves it to users)
- Pattern: Return `Result<T, E>` with detailed error messages
- Insight: Performance-critical libraries avoid logging in hot paths

**HiGHS (LP solver - C++)**:
- Uses: Custom logging with callback functions
- Levels: None (0-3 verbosity integer)
- Output: Text-based iteration log
- Insight: Solver iteration logs are valuable for users

---

### 8.2 Optimization Frameworks

**OR-Tools (Google - C++/Python)**:
- Uses: Custom logging with `LOG(INFO)` macros
- Features: Progress callback, solution inspector
- Insight: Structured callbacks > print statements

**SDDP.jl (Julia)**:
- Uses: Julia's `@info`, `@warn`, `@error` macros
- Features: Progress meter, dashboard (web UI)
- Output: Training log with iteration table
- Insight: **Closest match to POWE.RS needs**

**SDDP.jl example output**:
```
Iteration    Simulation       Bound       Time (s)    Cuts  Std Dev
        1   2.499394e+03  2.499394e+03        0.0       1      0.0
        2   2.499394e+03  2.499394e+03        0.0       2      0.0
       ...
       10   2.499394e+03  2.499394e+03        0.1      10      0.0
```

**Key takeaway**: Training table is industry standard. POWE.RS should keep it.

---

### 8.3 Rust CLI Tools

**ripgrep (rg)**:
- Uses: Custom logger (no external crate)
- Pattern: `--quiet`, `--debug` flags control verbosity
- Insight: CLI args > config file for quick overrides

**cargo**:
- Uses: Custom logging with `shell::Shell` abstraction
- Levels: Quiet, Normal, Verbose
- Output: Progress bars, colored status lines
- Insight: Layered output (progress + logs) requires coordination

**Recommendation**: Support both config file AND CLI flags for log level.

---

### 8.4 Recommended Approach for POWE.RS

**Inspired by SDDP.jl + cargo**:

1. **Keep ASCII table** for training (familiar to optimization researchers)
2. **Add structured logging** underneath (for programmatic access)
3. **Use `log` crate** (standard, zero-cost)
4. **Support CLI overrides**: `powers run --log-level debug examples/03-multistage`
5. **Add progress bars** for long-running tasks (via `indicatif`)
6. **Provide JSON output** for CI/ML pipelines

**Differentiation**:
- SDDP.jl: Interactive (REPL, live dashboards)
- POWE.RS: Non-interactive (CLI, batch jobs, HPC)
- Focus: Machine-readable logs + visual progress

---

## 9. Examples

### 9.1 Terminal Output (Default)

```
$ powers examples/03-multistage

POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

[INFO] Reading input files from 'examples/03-multistage'
[INFO] Using 8 threads for training

# Training
- Iterations: 100
- Forward passes: 20
- Cut selection: true

----------------------------------------------------------------------------------------
iter |      lower ($) |      simul ($) |          fwd |          bwd |        total
----------------------------------------------------------------------------------------
   1 |     2.499394e3 |     2.499394e3 | 00:00:00.012 | 00:00:00.008 | 00:00:00.020
  10 |     2.499394e3 |     2.499394e3 | 00:00:00.010 | 00:00:00.007 | 00:00:00.017
  ...
 100 |     2.499394e3 |     2.499394e3 | 00:00:00.009 | 00:00:00.006 | 00:00:00.015
----------------------------------------------------------------------------------------

[INFO] Training time: 00:00:01.823
[INFO] Number of constructed cuts by node: 100

# Simulating
- Scenarios: 1000

[INFO] Expected cost ($): 2.499394e3 ± 1.234e2
[INFO] Simulation time: 00:00:00.456

[INFO] Total running time: 00:00:02.279
```

---

### 9.2 Debug Output (Detailed Timing)

```bash
$ powers examples/03-multistage --log-level debug
```

```
[DEBUG] Thread pool configured with 8 threads
[DEBUG] Loaded 12 hydros, 4 buses, 3 thermals, 2 lines
[DEBUG] Scenario tree: 12 nodes, 11 edges

[INFO] Starting training (iteration=1)
  ┌─ Forward Pass (00:00:00.012) ──────────────────────────────────┐
  │  SAA Sampling:  00:00:00.001  │  Model Prep:  00:00:00.003     │
  │  Solver:        00:00:00.007  │  Aggregation: 00:00:00.001     │
  └────────────────────────────────────────────────────────────────┘
  
  ┌─ Backward Pass (00:00:00.008) ─────────────────────────────────┐
  │  Model Prep:    00:00:00.002  │  Solver:      00:00:00.004     │
  │  Cut Select:    00:00:00.001  │  Model Update:00:00:00.001     │
  └────────────────────────────────────────────────────────────────┘
  
  Solver: 24 calls | Cuts: +12 new, -0 dominated, +0 returned, 12 active

[DEBUG] Lower bound: 2.499394e3, Gap: 0.00%
```

---

### 9.3 JSON Output (Machine-Readable)

```bash
$ powers examples/03-multistage --log-format json > training.jsonl
```

**File: `training.jsonl`**:
```json
{"timestamp":"2025-11-09T13:24:48Z","level":"INFO","message":"Reading input files from 'examples/03-multistage'"}
{"timestamp":"2025-11-09T13:24:48Z","level":"INFO","message":"Starting training","num_iterations":100,"num_forward_passes":20}
{"timestamp":"2025-11-09T13:24:48Z","level":"INFO","message":"Iteration complete","iteration":1,"lower_bound":2499.394,"simulation_cost":2499.394,"forward_time_ms":12,"backward_time_ms":8,"total_time_ms":20}
...
{"timestamp":"2025-11-09T13:24:50Z","level":"INFO","message":"Training complete","total_time_ms":1823,"num_cuts":100}
```

**Analysis**:
```python
import json
with open("training.jsonl") as f:
    logs = [json.loads(line) for line in f]

iterations = [log for log in logs if log.get("iteration")]
final_gap = iterations[-1]["lower_bound"] / iterations[-1]["simulation_cost"] - 1
print(f"Final gap: {final_gap:.2%}")
```

---

### 9.4 Silent Mode (Benchmarks)

```rust
// In benchmark code
let config = Config {
    logging: LoggingConfig {
        outputs: vec![LogOutput::Silent],
        ..Default::default()
    },
    ..Config::default()
};

let mut sddp = SddpAlgorithm::from_config(config)?;
sddp.train()?; // No terminal output
```

---

## 10. Open Questions

**For discussion with maintainers**:

1. **Log level defaults**:
   - CLI: INFO (current behavior)
   - Library: WARN (less noisy)
   - Benchmarks: Silent (no overhead)
   - **Decision needed**: Is this acceptable?

2. **Progress bars**:
   - Use `indicatif` for long-running tasks?
   - May conflict with log output (need layering)
   - **Decision needed**: Worth the complexity?

3. **File output**:
   - Should logs be written to file by default (in output dir)?
   - Or only when explicitly configured?
   - **Decision needed**: User expectation?

4. **Backward compatibility**:
   - Keep `src/log.rs` for 1 major version (deprecated)?
   - Or remove immediately after Phase 4?
   - **Decision needed**: Migration timeline?

5. **Structured context**:
   - Thread-local storage adds 1 pointer access overhead
   - Alternative: Pass `&LogContext` as parameter (explicit)
   - **Decision needed**: Performance vs. ergonomics?

---

## 11. Success Criteria

**Phase 1-2 (Core Migration)**:
- ✅ CLI output visually identical to current format
- ✅ No performance regression (< 1% in benchmarks)
- ✅ All tests pass with new logging system
- ✅ Zero terminal output for library users by default

**Phase 3 (Advanced Features)**:
- ✅ JSON logs parseable for analysis
- ✅ Debug mode shows detailed timing without env vars
- ✅ File output works with log rotation

**Phase 4 (Cleanup)**:
- ✅ No references to `src/log.rs` in codebase
- ✅ Documentation updated
- ✅ Examples demonstrate all logging modes

---

## 12. Future Enhancements

**Not in scope for v1.0, but possible later**:

1. **Distributed logging** (if multi-node support added)
   - OpenTelemetry integration
   - Tracing spans for distributed profiling

2. **Live dashboards** (like SDDP.jl)
   - Web UI showing training progress
   - Real-time convergence plots

3. **Log analysis tools**
   - CLI tool to parse JSON logs
   - Extract training statistics
   - Compare runs

4. **Adaptive logging**
   - Automatically increase verbosity on errors
   - Capture debug logs in circular buffer, dump on failure

5. **Performance profiling**
   - Integrate with `tracing` for flame graphs
   - Identify bottlenecks in training loop

---

## 13. References

### Documentation
- Rust `log` crate: https://docs.rs/log
- `env_logger`: https://docs.rs/env_logger
- `tracing`: https://docs.rs/tracing
- `indicatif`: https://docs.rs/indicatif

### Related Projects
- SDDP.jl: https://github.com/odow/SDDP.jl
- Polars: https://github.com/pola-rs/polars
- cargo: https://github.com/rust-lang/cargo

### Standards
- RFC 5424 (Syslog): Log levels and severity
- JSON Lines: http://jsonlines.org/

---

## Appendix A: Configuration Examples

### Example 1: Default (Terminal, INFO)

```json
{
  "num_iterations": 100,
  "num_forward_passes": 20,
  "logging": {
    "level": "INFO",
    "format": "terminal",
    "show_progress_bar": true
  }
}
```

### Example 2: Debug with Timing Detail

```json
{
  "logging": {
    "level": "DEBUG",
    "format": "terminal",
    "show_timing_detail": true,
    "show_progress_bar": true
  }
}
```

### Example 3: JSON Logs to File

```json
{
  "logging": {
    "level": "INFO",
    "format": "json",
    "outputs": [
      {"type": "terminal", "level": "WARN"},
      {"type": "file", "path": "./logs/training.jsonl", "level": "DEBUG"}
    ]
  }
}
```

### Example 4: Silent (Benchmarks)

```json
{
  "logging": {
    "outputs": [{"type": "silent"}]
  }
}
```

---

## Appendix B: Migration Checklist

**Per-file migration**:
- [ ] `src/sddp/mod.rs` (training loop)
- [ ] `src/lib.rs` (entry point)
- [ ] `src/main.rs` (error handling)
- [ ] `src/input.rs` (validation errors)
- [ ] `src/subproblem.rs` (solver diagnostics)
- [ ] `src/output/parquet/writer.rs` (Parquet messages)

**Global tasks**:
- [ ] Add `log` to `Cargo.toml`
- [ ] Create `src/logging/` module
- [ ] Update `Config` struct
- [ ] Update JSON schema
- [ ] Update documentation
- [ ] Update examples
- [ ] Update CI (check for log format)

---

## Appendix C: Logging Best Practices

**For contributors**:

1. **Use appropriate log levels**
   - ERROR: System cannot continue (file not found, solver crash)
   - WARN: Unexpected but recoverable (solver retry, numerical issue)
   - INFO: High-level progress (iteration, training complete)
   - DEBUG: Detailed diagnostics (timing, cut statistics)
   - TRACE: Hot path events (subproblem solve, usually disabled)

2. **Add context**
   ```rust
   // BAD: No context
   info!("Subproblem solved");
   
   // GOOD: Rich context
   info!(
       stage = stage_id,
       node = node_id,
       objective = obj_value;
       "Subproblem solved"
   );
   ```

3. **Defer expensive operations**
   ```rust
   // BAD: Always formats
   debug!("State: {}", format_complex_state(&state));
   
   // GOOD: Only formats if DEBUG enabled
   debug!("State: {:?}", state);
   ```

4. **Don't log in hot loops**
   ```rust
   // BAD: 100k logs
   for node in nodes {
       trace!("Processing node {}", node.id);
   }
   
   // GOOD: Aggregate
   trace!("Processing {} nodes", nodes.len());
   ```

5. **Use errors for error paths**
   ```rust
   // BAD: Log and return error
   error!("File not found: {}", path);
   return Err("File not found");
   
   // GOOD: Return error with context (caller logs if needed)
   return Err(format!("File not found: {}", path));
   ```

---

**End of Document**
