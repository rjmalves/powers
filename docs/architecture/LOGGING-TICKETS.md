# Logging System Implementation Tickets

**Epic**: Professional Structured Logging System for POWE.RS  
**Target Version**: 0.3.0  
**Timeline**: 10 weeks (5 sprints x 2 weeks)  
**Total Estimated Effort**: 80-100 hours

---

## Sprint Overview

| Sprint | Focus | Tickets | Estimated Days |
|--------|-------|---------|---------------|
| Sprint 0 | Preparation & Baseline | LOG-001 to LOG-003 | 3 days |
| Sprint 1 | Infrastructure Setup | LOG-004 to LOG-009 | 10 days |
| Sprint 2 | Code Migration (Training) | LOG-010 to LOG-014 | 10 days |
| Sprint 3 | Code Migration (Complete) | LOG-015 to LOG-020 | 10 days |
| Sprint 4 | Advanced Features | LOG-021 to LOG-026 | 10 days |
| Sprint 5 | Cleanup & Release | LOG-027 to LOG-032 | 7 days |

**Total**: 50 days (part-time over 10 weeks)

---

## Epic: Structured Logging System

### Goals
- Replace ad-hoc `println!` logging with professional structured logging
- Support multiple output formats (terminal, JSON, silent)
- Enable configurable verbosity via log levels
- Maintain backward compatibility
- Zero performance overhead for disabled log levels

### Success Criteria
- ✅ CLI output unchanged with default configuration
- ✅ All 189+ tests pass
- ✅ < 1% performance regression
- ✅ JSON logs parseable and documented
- ✅ 100% test coverage for logging module

---

## Sprint 0: Preparation & Baseline (Week 1)

**Goal**: Validate design and establish baseline measurements for regression testing

### LOG-001: Design Review and Approval

**Priority**: Critical  
**Estimated Effort**: 0.5 days (confidence: high)

#### Context
Before implementing the structured logging system, we need stakeholder approval and consensus on the architectural design. This ensures alignment on approach, configuration schema, and migration strategy.

#### Acceptance Criteria
- [ ] `LOGGING-DESIGN.md` reviewed by all maintainers
- [ ] Design review meeting conducted with notes captured
- [ ] All feedback addressed and design document updated
- [ ] Formal approval to proceed obtained
- [ ] Configuration schema finalized and agreed upon

#### Tasks

##### Communication
- [ ] Share `LOGGING-DESIGN.md` via email/Slack with all maintainers
- [ ] Share `LOGGING-IMPLEMENTATION-PLAN.md` for timeline review
- [ ] Schedule 60-minute design review meeting
- [ ] Prepare presentation slides highlighting key decisions

##### Review Meeting
- [ ] Present architecture overview (10 min)
- [ ] Discuss key design decisions (log crate choice, config schema) (15 min)
- [ ] Review migration strategy and backward compatibility (15 min)
- [ ] Discuss open questions (10 min)
- [ ] Capture action items and feedback (10 min)

##### Documentation
- [ ] Document meeting notes in `docs/meetings/logging-design-review.md`
- [ ] Update design document based on feedback
- [ ] Get final approval from tech lead
- [ ] Update CHANGELOG.md with planned changes

#### Technical Notes
- Focus on getting consensus on:
  - Configuration schema (`logging` field in `config.json`)
  - Log level defaults (INFO for CLI, WARN for library)
  - Migration timeline (10 weeks acceptable?)
  - Backward compatibility strategy
- Have answers ready for: "Why not `tracing`?", "Performance impact?", "Breaking changes?"

#### Dependencies
- Blocked by: None
- Blocks: All other tickets
- Related: None

---

### LOG-002: Capture Baseline Measurements

**Priority**: Critical  
**Estimated Effort**: 0.5 days (confidence: high)

#### Context
We need baseline measurements of current CLI output and performance to ensure the new logging system doesn't introduce regressions. These baselines will be used for comparison testing throughout the implementation.

#### Acceptance Criteria
- [ ] Current CLI output captured for all example configurations
- [ ] Performance benchmarks saved with unique baseline name
- [ ] Binary size recorded for release build
- [ ] Baseline files committed to `tests/baselines/before-logging/`
- [ ] Documentation of baseline capture process created

#### Tasks

##### Capture CLI Output
- [ ] Run `powers examples/01-deterministic` and save output
- [ ] Run `powers examples/03-multistage` and save output
- [ ] Run `powers examples/05-large-scale-brazilian` and save output
- [ ] Save both stdout and stderr for each example
- [ ] Create `tests/baselines/before-logging/` directory

##### Performance Benchmarks
- [ ] Run `cargo bench --bench sddp_benchmarks -- --save-baseline before-logging`
- [ ] Verify baseline saved successfully (`target/criterion/*/before-logging/`)
- [ ] Document baseline name and date in `BENCHMARK_RESULTS.md`
- [ ] Run benchmarks 3 times and verify consistency

##### Binary Size
- [ ] Run `cargo build --release`
- [ ] Record binary size: `ls -lh target/release/powers`
- [ ] Record output in `tests/baselines/before-logging/binary-size.txt`

##### Documentation
- [ ] Create `tests/baselines/README.md` explaining baseline usage
- [ ] Document how to compare outputs: `diff` commands
- [ ] Document how to compare benchmarks: `cargo bench -- --baseline`
- [ ] Commit all baseline files to git

#### Technical Notes
```bash
# CLI output capture
mkdir -p tests/baselines/before-logging
powers examples/01-deterministic > tests/baselines/before-logging/01-deterministic.txt 2>&1
powers examples/03-multistage > tests/baselines/before-logging/03-multistage.txt 2>&1

# Performance baseline
cargo bench --bench sddp_benchmarks -- --save-baseline before-logging

# Binary size
cargo build --release
ls -lh target/release/powers > tests/baselines/before-logging/binary-size.txt
```

#### Dependencies
- Blocked by: None
- Blocks: LOG-009 (Integration Testing Phase 1)
- Related: LOG-001 (Design Review)

---

### LOG-003: Create Feature Branch

**Priority**: Critical  
**Estimated Effort**: 0.1 days (confidence: high)

#### Context
Establish a dedicated feature branch for the logging redesign. This allows parallel development without affecting the main branch and provides a clear scope for the changes.

#### Acceptance Criteria
- [ ] Feature branch `feature/structured-logging` created
- [ ] Branch protection rules configured (if applicable)
- [ ] Branch pushed to remote repository
- [ ] Development environment verified working on new branch
- [ ] Initial commit with branch documentation created

#### Tasks

##### Branch Setup
- [ ] Ensure local main branch is up to date: `git pull origin main`
- [ ] Create feature branch: `git checkout -b feature/structured-logging`
- [ ] Push branch to remote: `git push -u origin feature/structured-logging`

##### Documentation
- [ ] Create `.github/BRANCH_README.md` documenting branch purpose
- [ ] Add branch strategy notes (rebase vs merge)
- [ ] Document how to sync with main during development
- [ ] Initial commit: "docs: Add structured logging branch documentation"

##### Environment Verification
- [ ] Verify `cargo build` succeeds
- [ ] Verify `cargo test` passes
- [ ] Verify `cargo clippy` has no warnings
- [ ] Verify benchmarks run: `cargo bench --bench sddp_benchmarks`

##### Branch Protection (if admin access)
- [ ] Require PR review before merging to main
- [ ] Require CI passing before merge
- [ ] Prevent force push to main

#### Technical Notes
```bash
git checkout main
git pull origin main
git checkout -b feature/structured-logging
git push -u origin feature/structured-logging

# Verify environment
cargo clean
cargo build
cargo test --lib
cargo clippy --all-targets --all-features
```

#### Dependencies
- Blocked by: LOG-001 (Design approved)
- Blocks: All Phase 1 tickets
- Related: None

---

## Sprint 1: Infrastructure Setup (Weeks 2-3)

**Goal**: Add logging infrastructure without changing any visual output

### LOG-004: Add Log Crate Dependencies

**Priority**: Critical  
**Estimated Effort**: 0.5 days (confidence: high)

#### Context
Add the `log` crate facade and supporting dependencies to enable structured logging. This is the foundation for the entire logging system.

#### Acceptance Criteria
- [ ] `log` crate added to `[dependencies]`
- [ ] `env_logger` added to `[dev-dependencies]` for tests
- [ ] `atty` crate added for terminal detection
- [ ] `cargo build` succeeds with new dependencies
- [ ] `cargo test` passes with no warnings
- [ ] Dependency versions documented with rationale

#### Tasks

##### Add Dependencies
- [ ] Add to `Cargo.toml` under `[dependencies]`:
  - `log = "0.4"` (logging facade)
  - `atty = "0.2"` (terminal detection for colors)
- [ ] Add to `[dev-dependencies]`:
  - `env_logger = "0.11"` (for test logging)

##### Verification
- [ ] Run `cargo build` and verify success
- [ ] Run `cargo test` and verify all tests pass
- [ ] Run `cargo clippy` and verify no new warnings
- [ ] Check `Cargo.lock` is updated
- [ ] Verify correct versions resolved

##### Documentation
- [ ] Update `README.md` dependencies section if present
- [ ] Add comment in `Cargo.toml` explaining each logging dependency
- [ ] Update `CHANGELOG.md`: "Added log, atty dependencies for structured logging"

##### Testing
- [ ] Verify `log` macros are available: create temporary test file
- [ ] Verify no conflicts with existing dependencies
- [ ] Check total dependency count (should be +3)

#### Technical Notes
```toml
# In Cargo.toml
[dependencies]
log = "0.4"  # Logging facade - zero-cost abstractions
atty = "0.2"  # Terminal detection for colored output

[dev-dependencies]
env_logger = "0.11"  # Simple logger for tests
```

**Why these versions:**
- `log 0.4`: Latest stable, widely used (80%+ of Rust ecosystem)
- `atty 0.2`: Lightweight terminal detection
- `env_logger 0.11`: Simple logger, commonly used in tests

#### Dependencies
- Blocked by: LOG-003 (Feature branch created)
- Blocks: LOG-005 (Create module structure)
- Related: None

---


### LOG-005: Create Logging Module Structure

**Priority**: Critical  
**Estimated Effort**: 0.5 days (confidence: high)

#### Context
Create the directory structure and skeleton files for the new logging module. This establishes the foundation for all logging components.

#### Acceptance Criteria
- [ ] `src/logging/` directory created with submodules
- [ ] All module files created with minimal boilerplate
- [ ] Module hierarchy compiles successfully
- [ ] Public API defined in `src/logging/mod.rs`
- [ ] No compiler warnings or errors

#### Tasks

##### Create Directory Structure
- [ ] Create `src/logging/` directory
- [ ] Create `src/logging/formatters/` subdirectory
- [ ] Verify directory structure matches design

##### Create Module Files
- [ ] Create `src/logging/mod.rs` with module declarations
- [ ] Create `src/logging/config.rs` with placeholder types
- [ ] Create `src/logging/logger.rs` with placeholder types
- [ ] Create `src/logging/context.rs` with placeholder types
- [ ] Create `src/logging/formatters/mod.rs` with module declarations
- [ ] Create `src/logging/formatters/terminal.rs` with placeholder type

##### Module Integration
- [ ] Add `pub mod logging;` to `src/lib.rs`
- [ ] Export public types from `src/logging/mod.rs`
- [ ] Verify `cargo build` succeeds

##### Documentation
- [ ] Add module-level doc comments to each file
- [ ] Document the purpose of each submodule
- [ ] Update `src/lib.rs` doc comment to mention logging module

##### Testing
- [ ] Create `src/logging/mod.rs` test module
- [ ] Add basic compilation test
- [ ] Verify `cargo test --lib` passes

#### Technical Notes

**File: `src/logging/mod.rs`**
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

/// Initialize the logging system with the given configuration.
pub fn init(config: &LoggingConfig) -> Result<(), String> {
    logger::init_logger(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exists() {
        // Smoke test to verify module compiles
    }
}
```

**Directory structure:**
```
src/logging/
├── mod.rs              # Public API
├── config.rs           # Configuration types
├── logger.rs           # Logger implementation
├── context.rs          # Thread-local context
└── formatters/
    ├── mod.rs          # Formatter exports
    └── terminal.rs     # Terminal formatter
```

#### Dependencies
- Blocked by: LOG-004 (Dependencies added)
- Blocks: LOG-006 (Implement configuration types)
- Related: None

---

### LOG-006: Implement Configuration Types

**Priority**: Critical  
**Estimated Effort**: 1 day (confidence: high)

#### Context
Implement the configuration types that define logging behavior. These types will be deserialized from `config.json` and control all logging system behavior.

#### Acceptance Criteria
- [ ] All configuration enums and structs defined
- [ ] Serde serialization/deserialization working
- [ ] Default values defined and tested
- [ ] Configuration types have comprehensive doc comments
- [ ] Unit tests cover all configuration options

#### Tasks

##### Implementation
- [ ] Implement `LogLevel` enum (Error, Warn, Info, Debug, Trace)
- [ ] Implement `LogFormat` enum (Terminal, Json, Structured)
- [ ] Implement `LogOutput` enum (Terminal, File, Silent)
- [ ] Implement `LoggingConfig` struct with all fields
- [ ] Implement `Default` trait for `LoggingConfig`
- [ ] Implement conversion methods (e.g., `to_level_filter()`)

##### Serde Integration
- [ ] Add `#[derive(Serialize, Deserialize)]` to all types
- [ ] Use `#[serde(rename_all = "snake_case")]` for enums
- [ ] Use `#[serde(default)]` for optional fields
- [ ] Test serialization round-trip

##### Documentation
- [ ] Add doc comments explaining each configuration option
- [ ] Add example configurations in doc comments
- [ ] Document the default behavior
- [ ] Document the relationship between options

##### Testing
- [ ] Unit test: Default configuration values
- [ ] Unit test: Serialization from JSON
- [ ] Unit test: Deserialization to JSON
- [ ] Unit test: Missing fields use defaults
- [ ] Unit test: Invalid values rejected
- [ ] Unit test: LogLevel conversion to LevelFilter

#### Technical Notes

See implementation plan section 1.3 for full code example. Key points:

- Use `#[serde(default = "function_name")]` for custom defaults
- Provide helper functions for each default value
- Make `LoggingConfig` easy to construct programmatically
- Ensure backward compatibility: missing `logging` field = defaults

**Example test:**
```rust
#[test]
fn test_config_serialization() {
    let json = r#"{"level":"debug","format":"json"}"#;
    let config: LoggingConfig = serde_json::from_str(json).unwrap();
    assert!(matches!(config.level, LogLevel::Debug));
    assert!(matches!(config.format, LogFormat::Json));
}
```

#### Dependencies
- Blocked by: LOG-005 (Module structure created)
- Blocks: LOG-007 (Implement terminal formatter)
- Related: None

---

### LOG-007: Implement Terminal Formatter

**Priority**: High  
**Estimated Effort**: 2 days (confidence: medium)

#### Context
Implement the terminal formatter that preserves the current ASCII table output while enabling future structured logging. In Phase 1, this acts as a pass-through to existing `log.rs` functions.

#### Acceptance Criteria
- [ ] `TerminalFormatter` struct implemented
- [ ] Formatter preserves existing ASCII table output
- [ ] ANSI colors work when terminal is detected
- [ ] No visual changes to CLI output
- [ ] Format method handles all log record types

#### Tasks

##### Implementation
- [ ] Create `TerminalFormatter` struct with configuration
- [ ] Implement `format()` method signature
- [ ] Import existing `crate::log` module functions
- [ ] Implement pass-through logic for training table rows
- [ ] Implement default formatting for other logs
- [ ] Add terminal detection using `atty` crate

##### ANSI Colors
- [ ] Implement color constants (if terminal detected)
- [ ] Add level-based coloring (ERROR=red, WARN=yellow, etc.)
- [ ] Ensure colors disabled for non-terminal output
- [ ] Test color output with `cargo run`

##### Integration with Existing Code
- [ ] Call `log::training_table_row()` for iteration logs
- [ ] Call `log::training_greeting()` for training start
- [ ] Call other `log::` functions as needed
- [ ] Ensure no duplication of output

##### Documentation
- [ ] Document formatter behavior
- [ ] Document when colors are enabled
- [ ] Add example of formatted output
- [ ] Document the pass-through strategy

##### Testing
- [ ] Unit test: Format level as string
- [ ] Unit test: Format with context
- [ ] Unit test: Format without context
- [ ] Unit test: Colors disabled for non-terminal
- [ ] Integration test: Verify output matches baseline

#### Technical Notes

**Phase 1 Strategy**: Use existing `log.rs` functions to ensure zero visual changes:

```rust
pub fn format(&self, record: &Record, context: &LogContext) -> Vec<u8> {
    // Check if this is a structured log with iteration context
    if let Some(iteration) = context.iteration {
        if let (Some(lower), Some(simul)) = (context.lower_bound, context.simulation_cost) {
            // Call existing function
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
    let mut buffer = String::new();
    write!(&mut buffer, "[{}] {}", record.level(), record.args()).unwrap();
    buffer.push('\n');
    buffer.into_bytes()
}
```

**Important**: This is a temporary implementation. Phase 2 will inline the formatting logic.

#### Dependencies
- Blocked by: LOG-006 (Config types implemented)
- Blocks: LOG-008 (Implement PowersLogger)
- Related: None

---


### LOG-008: Implement LogContext for Thread-Local Storage

**Priority**: High  
**Estimated Effort**: 1 day (confidence: high)

#### Context
Implement thread-local storage for logging context. This allows logs to be automatically enriched with iteration, stage, and other contextual information without passing parameters everywhere.

#### Acceptance Criteria
- [ ] `LogContext` struct defined with all fields
- [ ] Thread-local storage implemented correctly
- [ ] `current()`, `set()`, and `clear()` methods working
- [ ] Context is thread-isolated (no cross-thread pollution)
- [ ] Zero-cost abstraction (single RefCell per thread)

#### Tasks

##### Implementation
- [ ] Define `LogContext` struct with Optional fields
- [ ] Implement `thread_local!` storage
- [ ] Implement `current()` method to retrieve context
- [ ] Implement `set()` method to update context
- [ ] Implement `clear()` method to reset context
- [ ] Implement `Default` trait
- [ ] Implement `Clone` trait

##### Helper Methods
- [ ] Implement `with_iteration()` scoped context helper
- [ ] Consider adding helpers for other common contexts
- [ ] Ensure context is restored after scope

##### Documentation
- [ ] Document thread-local behavior
- [ ] Add usage examples in doc comments
- [ ] Document performance characteristics
- [ ] Warn about context leaking across async boundaries (if applicable)

##### Testing
- [ ] Unit test: Set and retrieve context
- [ ] Unit test: Clear context
- [ ] Unit test: Default context is empty
- [ ] Unit test: Context isolation between threads
- [ ] Unit test: Scoped context restoration
- [ ] Unit test: Multiple sequential context updates

#### Technical Notes

**Thread-local pattern:**
```rust
use std::cell::RefCell;

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
```

**Performance**: Single `RefCell::borrow()` per log call = ~1ns overhead.

#### Dependencies
- Blocked by: LOG-005 (Module structure created)
- Blocks: LOG-008 (PowersLogger implementation)
- Related: LOG-007 (Terminal formatter uses context)

---

### LOG-009: Implement PowersLogger and Initialize

**Priority**: Critical  
**Estimated Effort**: 2 days (confidence: medium)

#### Context
Implement the core logger that integrates with the `log` crate facade. This is the central component that routes log records to the appropriate formatter and output sink.

#### Acceptance Criteria
- [ ] `PowersLogger` implements `log::Log` trait
- [ ] Logger correctly filters by log level
- [ ] Logger uses terminal formatter
- [ ] Logger writes to appropriate sink (terminal/file/silent)
- [ ] `init_logger()` function sets global logger
- [ ] Logger can be initialized multiple times in tests

#### Tasks

##### PowersLogger Implementation
- [ ] Create `PowersLogger` struct with fields (level, formatter, sink)
- [ ] Implement `log::Log` trait's `enabled()` method
- [ ] Implement `log::Log` trait's `log()` method
- [ ] Implement `log::Log` trait's `flush()` method
- [ ] Add thread-safe sink access (Mutex)

##### Sink Implementations
- [ ] Implement `TerminalSink` (writes to stdout)
- [ ] Implement `SilentSink` (no-op)
- [ ] Add `FileSink` placeholder (to be implemented in Phase 3)
- [ ] Ensure sinks implement `Write` trait

##### Logger Initialization
- [ ] Implement `init_logger(config)` function
- [ ] Set global logger with `log::set_boxed_logger()`
- [ ] Set max log level with `log::set_max_level()`
- [ ] Handle initialization errors gracefully
- [ ] Support re-initialization in tests

##### Documentation
- [ ] Document logger lifecycle
- [ ] Document thread-safety guarantees
- [ ] Add usage examples
- [ ] Document how to test with custom logger

##### Testing
- [ ] Unit test: Logger enables correct levels
- [ ] Unit test: Logger filters disabled levels
- [ ] Unit test: Logger formats and writes
- [ ] Unit test: Terminal sink outputs to stdout
- [ ] Unit test: Silent sink produces no output
- [ ] Integration test: Initialize and use logger

#### Technical Notes

**Key Implementation:**
```rust
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
```

**Testing consideration**: Use `log::set_logger()` instead of `set_boxed_logger()` in tests to allow re-initialization.

#### Dependencies
- Blocked by: LOG-007 (Terminal formatter), LOG-008 (LogContext)
- Blocks: LOG-010 (Update Config struct)
- Related: None

---

### LOG-010: Update Config Struct with Logging Field

**Priority**: High  
**Estimated Effort**: 0.5 days (confidence: high)

#### Context
Add the `logging` field to the main `Config` struct so users can configure logging via `config.json`. Must be backward-compatible with existing configs that don't have this field.

#### Acceptance Criteria
- [ ] `logging` field added to `Config` struct
- [ ] Field is optional with serde default
- [ ] Existing configs without `logging` field work correctly
- [ ] New configs with `logging` field parse correctly
- [ ] Schema documentation updated

#### Tasks

##### Implementation
- [ ] Add `logging: LoggingConfig` field to `Config` struct in `src/input.rs`
- [ ] Add `#[serde(default)]` attribute to field
- [ ] Verify `Config` still derives `Serialize`, `Deserialize`
- [ ] Update `Config::from_file()` to handle new field

##### Testing
- [ ] Unit test: Parse config with logging field
- [ ] Unit test: Parse config without logging field (uses defaults)
- [ ] Unit test: Serialize config with logging field
- [ ] Integration test: Load existing example configs (no logging field)
- [ ] Integration test: Create new config with logging field

##### Documentation
- [ ] Update `Config` doc comments
- [ ] Add example config with logging field
- [ ] Update `docs/reference/INPUT-SPECIFICATION.md`
- [ ] Add to CHANGELOG.md

##### Schema Update
- [ ] Update `schemas/config.schema.json` with logging field
- [ ] Add JSON schema for `LoggingConfig` and sub-types
- [ ] Test schema validation with VS Code
- [ ] Verify auto-completion works

#### Technical Notes

**In `src/input.rs`:**
```rust
use crate::logging::LoggingConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub num_iterations: usize,
    pub num_forward_passes: usize,
    // ... existing fields ...
    
    #[serde(default)]
    pub logging: LoggingConfig,
}
```

**Backward compatibility**: The `#[serde(default)]` attribute ensures existing configs without the `logging` field will use `LoggingConfig::default()`.

**Example config.json:**
```json
{
  "num_iterations": 100,
  "logging": {
    "level": "INFO",
    "format": "terminal"
  }
}
```

#### Dependencies
- Blocked by: LOG-006 (Config types implemented)
- Blocks: LOG-011 (Initialize logger in entry point)
- Related: None

---

### LOG-011: Initialize Logger in Entry Point

**Priority**: Critical  
**Estimated Effort**: 1 day (confidence: high)

#### Context
Initialize the logging system in the application entry point (`src/lib.rs::run()`). This activates the logging system for CLI usage. Must handle initialization errors gracefully.

#### Acceptance Criteria
- [ ] Logger initialized before any log calls
- [ ] Initialization errors handled and reported
- [ ] CLI output unchanged (visual regression test)
- [ ] Logger only initialized once
- [ ] Library users can initialize independently

#### Tasks

##### Implementation
- [ ] Add logger initialization call in `src/lib.rs::run()`
- [ ] Call `crate::logging::init(&config.logging)?`
- [ ] Handle initialization error and convert to `Box<dyn Error>`
- [ ] Ensure initialization happens after config loading
- [ ] Ensure initialization happens before any log calls

##### Error Handling
- [ ] Test initialization failure scenarios
- [ ] Ensure clear error messages
- [ ] Verify error propagates to `main.rs`
- [ ] Test with invalid log level
- [ ] Test with invalid file path (if file sink configured)

##### Testing
- [ ] Unit test: Successful initialization
- [ ] Unit test: Initialization error handling
- [ ] Integration test: CLI runs without errors
- [ ] Integration test: Compare output to baseline
- [ ] Integration test: Logger works for library API

##### Visual Regression
- [ ] Run all example configs and compare to baseline
- [ ] Verify no visual differences in output
- [ ] Check for any extra log messages
- [ ] Verify table formatting unchanged

##### Documentation
- [ ] Add comment explaining initialization placement
- [ ] Update error handling documentation
- [ ] Add troubleshooting guide for init errors

#### Technical Notes

**In `src/lib.rs`:**
```rust
pub fn run(input_path: &Path) -> Result<(), Box<dyn Error>> {
    // Load config first
    let config = Config::from_files(
        input_path.join("config.json"),
        // ...
    )?;
    
    // Initialize logging
    crate::logging::init(&config.logging)
        .map_err(|e| -> Box<dyn Error> { e.into() })?;
    
    // Now logging is active
    log::info!("POWE.RS - Power Optimization for the World of Energy - in pure RuSt");
    
    // ... rest of function
}
```

**Critical**: Initialize **after** config is loaded but **before** any operations that might log.

#### Dependencies
- Blocked by: LOG-009 (PowersLogger implemented), LOG-010 (Config updated)
- Blocks: LOG-012 (Integration testing)
- Related: None

---

### LOG-012: Integration Testing and Visual Regression

**Priority**: Critical  
**Estimated Effort**: 1.5 days (confidence: medium)

#### Context
Verify that the new logging infrastructure produces identical output to the baseline. This is the gate for Phase 1 completion - no visual changes allowed.

#### Acceptance Criteria
- [ ] All example configs run without errors
- [ ] CLI output matches baseline (byte-for-byte or whitespace-normalized)
- [ ] No performance regression (< 1% difference)
- [ ] All existing tests pass
- [ ] No new clippy warnings

#### Tasks

##### Output Comparison
- [ ] Run `powers examples/01-deterministic` and capture output
- [ ] Compare to baseline using `diff`
- [ ] Run `powers examples/03-multistage` and capture output
- [ ] Compare to baseline using `diff`
- [ ] Run `powers examples/05-large-scale-brazilian` and capture output
- [ ] Compare to baseline using `diff`
- [ ] Document any intentional differences

##### Performance Regression
- [ ] Run `cargo bench --bench sddp_benchmarks`
- [ ] Compare to `before-logging` baseline
- [ ] Verify < 1% regression on all benchmarks
- [ ] Document any performance changes
- [ ] If regression > 1%, investigate and fix

##### Test Suite
- [ ] Run `cargo test --lib`
- [ ] Run `cargo test --all-targets`
- [ ] Verify all 189+ tests pass
- [ ] Check for any test output changes
- [ ] Verify no test failures

##### Code Quality
- [ ] Run `cargo clippy --all-targets --all-features`
- [ ] Fix any new warnings
- [ ] Run `cargo fmt --all -- --check`
- [ ] Verify formatting is correct

##### Documentation
- [ ] Create integration test report
- [ ] Document comparison methodology
- [ ] Document any known differences
- [ ] Update CHANGELOG.md with Phase 1 completion

#### Technical Notes

**Comparison script:**
```bash
# Run new implementation
powers examples/03-multistage > new-output.txt 2>&1

# Compare to baseline
diff tests/baselines/before-logging/03-multistage.txt new-output.txt

# If differences, analyze
# - Are they intentional (e.g., [INFO] prefix)?
# - Are they formatting-only (whitespace)?
# - Do they indicate a bug?
```

**Acceptance threshold**: Whitespace-normalized output should be identical, or all differences should be documented and approved.

#### Dependencies
- Blocked by: LOG-011 (Logger initialized), LOG-002 (Baselines captured)
- Blocks: None (Phase 1 complete!)
- Related: All Phase 1 tickets

---

## Sprint 1 Summary

**Total Effort**: 10 days  
**Tickets**: LOG-004 to LOG-012 (9 tickets)  
**Deliverable**: Working logging infrastructure with zero visual changes

**Completion Criteria**:
- ✅ `log` crate integrated
- ✅ Logging module structure in place
- ✅ Configuration types implemented
- ✅ Terminal formatter working (pass-through to old functions)
- ✅ PowersLogger initialized in entry point
- ✅ All tests pass
- ✅ CLI output matches baseline
- ✅ No performance regression

**Next Sprint**: Begin migrating `println!` calls to `log` macros in training loop.

---


## Sprint 2: Code Migration - Training Loop (Weeks 4-5)

**Goal**: Migrate training loop from `println!` to structured `log` macros

### LOG-013: Migrate Training Loop Greeting and Headers

**Priority**: High  
**Estimated Effort**: 1 day (confidence: high)

#### Context
Replace the training loop initialization messages (`training_greeting`, `training_table_header`) with structured log macros. This is the first visible migration of logging calls.

#### Acceptance Criteria
- [ ] `log::training_greeting()` replaced with `info!` macro
- [ ] `log::training_table_header()` replaced with structured output
- [ ] `log::training_table_divider()` replaced with structured output
- [ ] Visual output unchanged
- [ ] Context properly set for training phase

#### Tasks

##### Implementation
- [ ] In `src/sddp/mod.rs`, find `log::training_greeting()` call
- [ ] Replace with structured `info!` macro with fields
- [ ] Find `log::training_table_header()` call
- [ ] Replace with `info!` or keep function (decision needed)
- [ ] Find `log::training_table_divider()` calls
- [ ] Replace or keep function calls

##### Context Management
- [ ] Set training phase context at start
- [ ] Add iteration count to context
- [ ] Add forward passes count to context
- [ ] Ensure context cleared at end

##### Testing
- [ ] Unit test: Verify log messages emitted
- [ ] Integration test: Compare CLI output to baseline
- [ ] Verify table still renders correctly
- [ ] Check no duplicate messages

##### Documentation
- [ ] Update code comments
- [ ] Document structured fields used
- [ ] Add to CHANGELOG.md

#### Technical Notes

**Before:**
```rust
log::training_greeting(num_iterations, num_forward_passes, enable_cut_selection);
log::training_table_header();
log::training_table_divider();
```

**After:**
```rust
info!(
    num_iterations = num_iterations,
    num_forward_passes = num_forward_passes,
    cut_selection = enable_cut_selection;
    "Starting training"
);

// Table header - keep function for now or inline
log::training_table_header();  // Phase 1: keep function
log::training_table_divider();
```

**Decision Point**: Keep helper functions for now (Phase 1), inline in Phase 2.

#### Dependencies
- Blocked by: LOG-012 (Phase 1 complete)
- Blocks: LOG-014 (Migrate iteration rows)
- Related: None

---

### LOG-014: Migrate Training Iteration Rows

**Priority**: Critical  
**Estimated Effort**: 2 days (confidence: medium)

#### Context
Replace `log::training_table_row()` calls with structured logging using context. This is the core of the training output and must preserve visual formatting exactly.

#### Acceptance Criteria
- [ ] `log::training_table_row()` replaced with `info!` + context
- [ ] Iteration context set for each iteration
- [ ] Visual table output unchanged
- [ ] Context properly cleared after each iteration
- [ ] Performance overhead < 0.1% (minimal)

#### Tasks

##### Implementation
- [ ] In training loop, find `log::training_table_row()` calls
- [ ] Set `LogContext` with all iteration data
- [ ] Replace with `info!("Iteration complete")` or similar
- [ ] Ensure `TerminalFormatter` handles context correctly
- [ ] Clear context after each iteration

##### Context Fields
- [ ] `iteration`: current iteration number
- [ ] `lower_bound`: from iteration result
- [ ] `simulation_cost`: from iteration result
- [ ] `forward_time`: Duration
- [ ] `backward_time`: Duration
- [ ] `total_time`: Duration

##### Terminal Formatter Update
- [ ] Update `TerminalFormatter` to detect iteration context
- [ ] Call `legacy_log::training_table_row()` when context present
- [ ] Ensure no duplicate output

##### Testing
- [ ] Unit test: Context set and retrieved correctly
- [ ] Integration test: Table output matches baseline exactly
- [ ] Performance test: Measure context overhead
- [ ] Visual test: Run multiple examples and verify tables

##### Documentation
- [ ] Document the context pattern for iterations
- [ ] Add code comments explaining the approach
- [ ] Update CHANGELOG.md

#### Technical Notes

**Implementation pattern:**
```rust
for (index, iter_result) in iterations.iter().enumerate() {
    // Set context for this iteration
    LogContext::set(LogContext {
        iteration: Some(index + 1),
        lower_bound: Some(iter_result.lower_bound),
        simulation_cost: Some(simulation_cost),
        forward_time: Some(iter_result.forward_timing.total_time),
        backward_time: Some(iter_result.backward_timing.total_time),
        total_time: Some(iter_result.iteration_time),
    });
    
    // Structured log - formatter will render as table row
    info!("Iteration complete");
    
    // Clear context
    LogContext::clear();
}
```

**Formatter logic** (already implemented in LOG-007):
```rust
if let Some(iteration) = context.iteration {
    if context.lower_bound.is_some() {
        // Call legacy function to render table row
        legacy_log::training_table_row(...);
        return Vec::new();
    }
}
```

#### Dependencies
- Blocked by: LOG-013 (Training headers migrated)
- Blocks: LOG-015 (Migrate timing detail)
- Related: LOG-008 (LogContext)

---

### LOG-015: Migrate Timing Detail Logging

**Priority**: Medium  
**Estimated Effort**: 1.5 days (confidence: medium)

#### Context
Replace the environment variable check for `POWERS_TIMING_DETAIL` with config-driven behavior. Migrate `log::training_iteration_timing()` to structured debug logs.

#### Acceptance Criteria
- [ ] `std::env::var("POWERS_TIMING_DETAIL")` removed
- [ ] Replaced with `config.logging.show_timing_detail`
- [ ] `log::training_iteration_timing()` replaced with `debug!` macros
- [ ] Box-drawing output preserved
- [ ] Timing detail only shown when configured

#### Tasks

##### Remove Environment Variable
- [ ] In `src/sddp/mod.rs`, find env var check (line ~2127)
- [ ] Replace `std::env::var("POWERS_TIMING_DETAIL").is_ok()`
- [ ] With `config.logging.show_timing_detail`
- [ ] Pass config to training function if needed

##### Migrate Timing Function
- [ ] Replace `log::training_iteration_timing()` call
- [ ] With structured `debug!` macro with timing fields
- [ ] Formatter should render with box-drawing
- [ ] All timing fields included

##### Formatter Enhancement
- [ ] Update `TerminalFormatter` to detect timing detail context
- [ ] Render box-drawing format for timing logs
- [ ] Preserve exact visual format

##### Testing
- [ ] Unit test: Config flag controls timing output
- [ ] Integration test: Run with `show_timing_detail: true`
- [ ] Integration test: Run with `show_timing_detail: false`
- [ ] Visual test: Verify box-drawing characters correct
- [ ] Verify no output when flag is false

##### Documentation
- [ ] Update README.md: Remove env var, add config option
- [ ] Update INPUT-SPECIFICATION.md
- [ ] Add example config with timing detail enabled
- [ ] Update CHANGELOG.md: "BREAKING: POWERS_TIMING_DETAIL removed"

#### Technical Notes

**Before:**
```rust
if std::env::var("POWERS_TIMING_DETAIL").is_ok() {
    log::training_iteration_timing(...all_timing_fields...);
}
```

**After:**
```rust
if config.logging.show_timing_detail {
    debug!(
        forward_saa_ms = saa_sampling_time.as_millis(),
        forward_model_prep_ms = forward_model_pre_time.as_millis(),
        forward_solver_ms = forward_solver_time.as_millis(),
        backward_prep_ms = backward_pre_time.as_millis(),
        // ... all timing fields
        solver_calls = solver_calls,
        cuts_added = cuts_added;
        "Detailed timing breakdown"
    );
}
```

**Breaking Change**: Document that `POWERS_TIMING_DETAIL` env var is removed. Users must migrate to config.

#### Dependencies
- Blocked by: LOG-014 (Iteration rows migrated)
- Blocks: LOG-016 (Migrate simulation)
- Related: LOG-006 (Config types)

---

### LOG-016: Migrate Simulation Logging

**Priority**: High  
**Estimated Effort**: 1.5 days (confidence: high)

#### Context
Replace simulation-related `println!` calls with structured logging. This includes simulation greeting, statistics, and completion messages.

#### Acceptance Criteria
- [ ] `log::simulation_greeting()` replaced with `info!` macro
- [ ] `log::simulation_stats()` replaced with structured log
- [ ] `log::simulation_skipped()` replaced with structured log
- [ ] `log::final_simulation_stats()` replaced with structured log
- [ ] Visual output unchanged

#### Tasks

##### Implementation
- [ ] Find `log::simulation_greeting()` in `src/sddp/mod.rs`
- [ ] Replace with `info!(num_scenarios = ...; "Starting simulation")`
- [ ] Find `log::simulation_stats()` calls
- [ ] Replace with `info!(mean_cost = ..., std_cost = ...; "Simulation complete")`
- [ ] Find `log::simulation_skipped()` call
- [ ] Replace with `info!("Simulation skipped (not configured)")`
- [ ] Find `log::final_simulation_stats()` call
- [ ] Replace with structured log including gap

##### Context Management
- [ ] Set simulation phase context
- [ ] Add scenario count to context if relevant
- [ ] Clear context after simulation

##### Testing
- [ ] Unit test: Simulation logs emitted
- [ ] Integration test: With simulation configured
- [ ] Integration test: Without simulation (skipped)
- [ ] Visual test: Compare output to baseline

##### Documentation
- [ ] Update code comments
- [ ] Document structured fields
- [ ] Update CHANGELOG.md

#### Technical Notes

**Migration examples:**
```rust
// Before
log::simulation_greeting(num_scenarios);

// After
info!(num_scenarios = num_scenarios; "Starting simulation");

// Before
log::simulation_stats(mean, std);

// After
info!(
    mean_cost = mean,
    std_cost = std;
    "Simulation complete"
);
```

#### Dependencies
- Blocked by: LOG-015 (Timing detail migrated)
- Blocks: LOG-017 (Migrate greetings/farewells)
- Related: None

---

### LOG-017: Migrate Application Greetings and Farewells

**Priority**: Medium  
**Estimated Effort**: 1 day (confidence: high)

#### Context
Replace the application-level logging calls (greeting, input path, output path, farewell) with structured logging. These are in `src/lib.rs` and `src/log.rs`.

#### Acceptance Criteria
- [ ] `log::show_greeting()` replaced with `info!` macro
- [ ] `log::input_reading_line()` replaced with structured log
- [ ] `log::output_generation_line()` replaced with structured log
- [ ] `log::show_farewell()` replaced with structured log
- [ ] Visual output unchanged (including separator line)

#### Tasks

##### Implementation
- [ ] In `src/lib.rs::run()`, find `log::show_greeting()`
- [ ] Replace with `info!` calls (may need 2 calls for title + separator)
- [ ] Find `log::input_reading_line(input_path)`
- [ ] Replace with `info!(path = input_path; "Reading input files")`
- [ ] Find `log::output_generation_line(output_path)`
- [ ] Replace with `info!(path = output_path; "Writing outputs")`
- [ ] Find `log::show_farewell(duration)`
- [ ] Replace with `info!(duration_ms = ...; "Total running time")`

##### Formatting
- [ ] Ensure separator line still renders (ASCII dashes)
- [ ] Preserve exact message formatting
- [ ] Update `TerminalFormatter` if needed for special formatting

##### Testing
- [ ] Integration test: Compare full CLI output
- [ ] Verify greeting appears first
- [ ] Verify farewell appears last
- [ ] Check separator line renders

##### Documentation
- [ ] Update code comments
- [ ] Update CHANGELOG.md

#### Technical Notes

**Example migration:**
```rust
// Before
log::show_greeting();

// After
info!("POWE.RS - Power Optimization for the World of Energy - in pure RuSt");
info!("{}", "-".repeat(68));  // Or handle in formatter
```

**Consider**: Should the separator line be part of the formatter, or explicit in the log call?

#### Dependencies
- Blocked by: LOG-016 (Simulation migrated)
- Blocks: LOG-018 (Migrate error handling)
- Related: None

---


### LOG-018: Migrate Error Handling to error! Macro

**Priority**: High  
**Estimated Effort**: 1.5 days (confidence: high)

#### Context
Replace all `eprintln!` calls with `error!` macros for consistent error logging. This includes validation failures, solver errors, and I/O errors across multiple files.

#### Acceptance Criteria
- [ ] All `eprintln!` in `src/main.rs` replaced
- [ ] All `eprintln!` in `src/input.rs` replaced
- [ ] All `eprintln!` in `src/subproblem.rs` replaced
- [ ] Error messages preserve important context
- [ ] Visual output unchanged (stderr vs stdout OK)

#### Tasks

##### Survey Codebase
- [ ] Run `grep -rn "eprintln!" src/` to find all occurrences
- [ ] Categorize by file and context
- [ ] Identify which are errors vs warnings vs debug

##### Migrate main.rs
- [ ] Find `eprintln!("Error: {}", e)` in main
- [ ] Replace with `error!("Execution failed: {}", e)`
- [ ] Test error path manually

##### Migrate input.rs
- [ ] Find validation error `eprintln!` calls
- [ ] Replace with `error!("Input validation failed: {}", e)`
- [ ] Add structured fields if valuable (e.g., file path)

##### Migrate subproblem.rs
- [ ] Find solver infeasibility `eprintln!` calls
- [ ] Replace with `error!("Solver returned infeasible")`
- [ ] Consider if some should be `warn!` instead
- [ ] Remove or migrate debug `eprintln!` (lines 2894+)

##### Testing
- [ ] Test error path: invalid config file
- [ ] Test error path: missing input file
- [ ] Test error path: solver infeasibility (if possible)
- [ ] Verify error messages are clear

##### Documentation
- [ ] Update error handling documentation
- [ ] Add troubleshooting guide if needed
- [ ] Update CHANGELOG.md

#### Technical Notes

**Migration pattern:**
```rust
// Before
eprintln!("Error: {}", e);

// After
error!("Execution failed: {}", e);

// Before (with context)
eprintln!("[ERROR] Solver infeasible! Let me check the constraint structure:");

// After
error!(
    node_id = node.id,
    num_constraints = model.num_rows();
    "Solver returned infeasible, checking constraints"
);
```

**stdout vs stderr**: `error!` macro output goes to stdout by default (via our logger). This is a change from `eprintln!` but acceptable.

#### Dependencies
- Blocked by: LOG-017 (Greetings migrated)
- Blocks: LOG-019 (Migrate debug logging)
- Related: None

---

### LOG-019: Migrate Debug Logging to debug! Macro

**Priority**: Medium  
**Estimated Effort**: 1 day (confidence: medium)

#### Context
Enable debug logging that was previously commented out or always-on. Use `debug!` macros so these logs only appear when configured.

#### Acceptance Criteria
- [ ] Commented-out debug code enabled with `debug!` macros
- [ ] Always-on debug code gated by log level
- [ ] Debug logs only visible with `--log-level debug`
- [ ] No debug output in default INFO mode
- [ ] Performance impact zero when debug disabled

#### Tasks

##### Survey Debug Code
- [ ] Find commented-out `eprintln!` or `println!` (debug context)
- [ ] In `src/subproblem.rs`, find debug code (lines 2894-2897)
- [ ] Identify other debug-worthy information

##### Implement Debug Logging
- [ ] Uncomment and convert to `debug!` macros
- [ ] Add structured fields where valuable
- [ ] Use `debug!` for: model structure, solver state, intermediate values

##### Solver Diagnostics
- [ ] Add debug logging for solver calls
- [ ] Log model size (num_cols, num_rows)
- [ ] Log solver status
- [ ] Log solution quality metrics

##### Testing
- [ ] Run with `--log-level info` (default) → no debug output
- [ ] Run with `--log-level debug` → debug output appears
- [ ] Verify debug logs are informative
- [ ] Ensure no performance impact when disabled

##### Documentation
- [ ] Document available debug logging
- [ ] Add troubleshooting guide: when to use debug level
- [ ] Update CHANGELOG.md

#### Technical Notes

**Example debug logs:**
```rust
// Model structure
debug!(
    model_exists = model.is_some(),
    num_cols = model.as_ref().map(|m| m.num_cols()),
    num_rows = model.as_ref().map(|m| m.num_rows());
    "Subproblem model state"
);

// Solver call
debug!(
    node_id = node.id,
    stage = stage.id;
    "Solving subproblem"
);
```

**Performance**: With `--log-level info`, all `debug!` calls are compiled out (zero cost).

#### Dependencies
- Blocked by: LOG-018 (Error handling migrated)
- Blocks: LOG-020 (Verify no println! remains)
- Related: None

---

### LOG-020: Verify No println!/eprintln! Remains

**Priority**: Critical  
**Estimated Effort**: 0.5 days (confidence: high)

#### Context
Final verification that all direct print statements have been migrated to structured logging. This is the gate for Phase 2 completion.

#### Acceptance Criteria
- [ ] `grep -rn "println!" src/` returns no results (except comments/strings)
- [ ] `grep -rn "eprintln!" src/` returns no results (except comments/strings)
- [ ] All tests pass
- [ ] Visual output matches baseline or differences documented
- [ ] CHANGELOG.md updated

#### Tasks

##### Code Audit
- [ ] Run `grep -rn "println!" src/` and verify all hits are acceptable
- [ ] Run `grep -rn "eprintln!" src/` and verify all hits are acceptable
- [ ] Check for prints in test code (allowed)
- [ ] Check for prints in doc comments (allowed)

##### Integration Testing
- [ ] Run all example configs
- [ ] Compare output to baseline
- [ ] Document any visual differences
- [ ] Get approval for any changes

##### Test Suite
- [ ] Run `cargo test --all-targets`
- [ ] Verify all 189+ tests pass
- [ ] Check for any test failures
- [ ] Fix any issues

##### Code Quality
- [ ] Run `cargo clippy --all-targets --all-features`
- [ ] Fix any new warnings
- [ ] Run `cargo fmt --all -- --check`

##### Documentation
- [ ] Update CHANGELOG.md: "Migrated all logging to structured system"
- [ ] Create migration report document
- [ ] List any known visual differences
- [ ] Update sprint status

#### Technical Notes

**Acceptable prints:**
- In test code: `println!("Test output: {}", x)`
- In doc comments: `/// println!("Example");`
- In string literals: `let s = "println!";`

**Unacceptable prints:**
- Any production code logging with `println!` or `eprintln!`

**Verification commands:**
```bash
# Should return only acceptable results
grep -rn "println!" src/ --exclude-dir=target
grep -rn "eprintln!" src/ --exclude-dir=target

# Check specific files
rg 'println!' src/main.rs src/lib.rs src/sddp/mod.rs
```

#### Dependencies
- Blocked by: LOG-019 (Debug logging migrated)
- Blocks: None (Sprint 2 complete!)
- Related: All Sprint 2 tickets

---

## Sprint 2 Summary

**Total Effort**: 10 days  
**Tickets**: LOG-013 to LOG-020 (8 tickets)  
**Deliverable**: All training and simulation code migrated to structured logging

**Completion Criteria**:
- ✅ Training loop uses `log` macros
- ✅ Simulation uses `log` macros
- ✅ Error handling uses `error!` macro
- ✅ Debug logging uses `debug!` macro
- ✅ No `println!`/`eprintln!` in production code
- ✅ Environment variable removed
- ✅ Visual output acceptable (documented differences OK)

**Next Sprint**: Continue migration of remaining modules and add JSON formatter.

---

## Sprint 3: Code Migration Complete & JSON Formatter (Weeks 6-7)

**Goal**: Complete migration of all remaining code and add JSON output format

### LOG-021: Remove Dependency on src/log.rs

**Priority**: Critical  
**Estimated Effort**: 2 days (confidence: medium)

#### Context
Inline the formatting logic from `src/log.rs` into `TerminalFormatter` so we no longer depend on the legacy module. This is preparation for removing `src/log.rs` entirely.

#### Acceptance Criteria
- [ ] All `legacy_log::` calls removed from `TerminalFormatter`
- [ ] Formatting logic inlined into formatter
- [ ] Visual output unchanged
- [ ] No imports from `src/log` module

#### Tasks

##### Inline Table Formatting
- [ ] Copy `format_duration()` into `TerminalFormatter`
- [ ] Copy `format_cost()` into `TerminalFormatter`
- [ ] Copy `format_gap()` into `TerminalFormatter`
- [ ] Implement table row rendering logic
- [ ] Implement table header rendering logic
- [ ] Implement table divider rendering logic

##### Update TerminalFormatter
- [ ] Remove `use crate::log as legacy_log;`
- [ ] Implement `format_table_row()` method
- [ ] Implement `format_timing_detail()` method
- [ ] Handle all log record types

##### Testing
- [ ] Unit test: Format duration
- [ ] Unit test: Format cost
- [ ] Unit test: Format table row
- [ ] Integration test: Full training output
- [ ] Visual test: Compare to baseline

##### Documentation
- [ ] Document formatting functions
- [ ] Add examples of formatted output
- [ ] Update CHANGELOG.md

#### Technical Notes

**Inline helpers:**
```rust
impl TerminalFormatter {
    fn format_duration(&self, duration: Duration) -> String {
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        let millis = duration.subsec_millis();
        format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
    }
    
    fn format_cost(&self, cost: f64) -> String {
        format!("{:.6e}", cost)
    }
    
    fn format_table_row(&self, context: &LogContext) -> String {
        format!(
            "{0: >4} | {1: >14} | {2: >14} | {3: >12} | {4: >12} | {5: >12}",
            context.iteration.unwrap_or(0),
            self.format_cost(context.lower_bound.unwrap_or(0.0)),
            // ... etc
        )
    }
}
```

#### Dependencies
- Blocked by: LOG-020 (Migration complete)
- Blocks: LOG-027 (Remove src/log.rs)
- Related: LOG-007 (Terminal formatter)

---

### LOG-022: Implement JSON Formatter

**Priority**: High  
**Estimated Effort**: 2 days (confidence: high)

#### Context
Implement JSON Lines formatter for machine-readable logs. This enables programmatic analysis of training runs and integration with CI/ML pipelines.

#### Acceptance Criteria
- [ ] `JsonFormatter` struct implemented
- [ ] Outputs valid JSON Lines format (one JSON object per line)
- [ ] All context fields included in JSON
- [ ] ISO 8601 timestamps
- [ ] Can be parsed by `jq` and other tools

#### Tasks

##### Implementation
- [ ] Create `src/logging/formatters/json.rs`
- [ ] Define `JsonFormatter` struct
- [ ] Implement `format()` method
- [ ] Use `serde_json` for serialization
- [ ] Add timestamp to each log entry

##### LogEntry Structure
- [ ] Create serializable `JsonLogEntry` struct
- [ ] Include: timestamp, level, message, context fields
- [ ] Flatten context fields into top level
- [ ] Handle optional fields gracefully

##### Integration
- [ ] Add `JsonFormatter` to formatters module
- [ ] Update `PowersLogger` to support JSON format
- [ ] Update `init_logger()` to create JSON formatter
- [ ] Test format selection from config

##### Testing
- [ ] Unit test: JSON serialization
- [ ] Unit test: All fields present
- [ ] Integration test: Parse output with `jq`
- [ ] Integration test: Run full example, validate JSON
- [ ] Verify no extra output (single line per log)

##### Documentation
- [ ] Document JSON schema
- [ ] Add usage examples
- [ ] Show `jq` query examples
- [ ] Update CHANGELOG.md

#### Technical Notes

**Implementation:**
```rust
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
        json.push(b'\n');
        json
    }
}
```

**Example output:**
```json
{"timestamp":"2025-11-09T13:24:48Z","level":"INFO","iteration":1,"lower_bound":2499.394,"message":"Iteration complete"}
```

#### Dependencies
- Blocked by: LOG-020 (Migration complete)
- Blocks: LOG-023 (Add CLI flags)
- Related: LOG-006 (Config types support JSON format)

---


### LOG-023: Add CLI Flags for Log Level and Format

**Priority**: High  
**Estimated Effort**: 1.5 days (confidence: high)

#### Context
Add command-line flags to override logging configuration. This allows users to change log level and format without editing `config.json`.

#### Acceptance Criteria
- [ ] `--log-level` flag implemented
- [ ] `--log-format` flag implemented
- [ ] Flags override config.json values
- [ ] Invalid values handled gracefully
- [ ] Help text documents flags

#### Tasks

##### CLI Definition
- [ ] Add `log_level: Option<String>` to `Cli` struct in `src/cli.rs`
- [ ] Add `log_format: Option<String>` to `Cli` struct
- [ ] Add `#[arg(long, global = true)]` attributes
- [ ] Add help text for each flag

##### Parsing Logic
- [ ] Implement `str::parse()` for `LogLevel`
- [ ] Implement `str::parse()` for `LogFormat`
- [ ] Handle case-insensitive parsing
- [ ] Return clear error messages

##### Config Override
- [ ] In `src/main.rs` or `src/lib.rs`, override config with CLI values
- [ ] Apply overrides before logger initialization
- [ ] Log the effective configuration

##### Testing
- [ ] Unit test: Parse log level from string
- [ ] Unit test: Parse log format from string
- [ ] Integration test: `--log-level debug` works
- [ ] Integration test: `--log-format json` works
- [ ] Integration test: Invalid value shows error
- [ ] Test combined: `--log-level trace --log-format json`

##### Documentation
- [ ] Update `--help` output
- [ ] Update README.md with flag examples
- [ ] Add examples to user documentation
- [ ] Update CHANGELOG.md

#### Technical Notes

**CLI struct:**
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

**Config override:**
```rust
// In main or lib.rs
if let Some(level) = cli.log_level {
    config.logging.level = level.parse()
        .map_err(|_| format!("Invalid log level: {}", level))?;
}
if let Some(format) = cli.log_format {
    config.logging.format = format.parse()
        .map_err(|_| format!("Invalid log format: {}", format))?;
}
```

**Usage examples:**
```bash
powers examples/03-multistage --log-level debug
powers run --log-format json data/
powers examples/01-deterministic --log-level trace --log-format json
```

#### Dependencies
- Blocked by: LOG-022 (JSON formatter implemented)
- Blocks: LOG-024 (File sink)
- Related: LOG-006 (Config types)

---

### LOG-024: Implement File Sink for Log Output

**Priority**: Medium  
**Estimated Effort**: 1.5 days (confidence: medium)

#### Context
Add support for writing logs to a file in addition to or instead of terminal output. This enables persistent logging for production deployments.

#### Acceptance Criteria
- [ ] `FileSink` implemented with buffered writer
- [ ] Config supports file output path
- [ ] Multiple outputs supported (terminal + file)
- [ ] File created/appended correctly
- [ ] File errors handled gracefully

#### Tasks

##### Implementation
- [ ] Implement `FileSink` struct with `BufWriter<File>`
- [ ] Implement `Write` trait for `FileSink`
- [ ] Add file path to `LogOutput` enum (already done in LOG-006)
- [ ] Update `init_logger()` to create file sink

##### Multi-Sink Support
- [ ] Create `MultiSink` struct to wrap multiple sinks
- [ ] Implement `Write` trait for `MultiSink`
- [ ] Update logger to support multiple outputs
- [ ] Test terminal + file simultaneously

##### File Handling
- [ ] Create parent directories if needed
- [ ] Handle file open errors
- [ ] Use buffered writer for performance
- [ ] Flush on logger drop/exit

##### Testing
- [ ] Unit test: FileSink writes correctly
- [ ] Unit test: MultiSink broadcasts to all sinks
- [ ] Integration test: Log to file only
- [ ] Integration test: Log to terminal + file
- [ ] Test file permissions error
- [ ] Verify file content is valid

##### Documentation
- [ ] Document file output configuration
- [ ] Add example configs
- [ ] Warn about log file size growth
- [ ] Update CHANGELOG.md

#### Technical Notes

**MultiSink implementation:**
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

**Config example:**
```json
{
  "logging": {
    "outputs": [
      {"type": "terminal"},
      {"type": "file", "path": "./logs/training.log"}
    ]
  }
}
```

#### Dependencies
- Blocked by: LOG-023 (CLI flags implemented)
- Blocks: LOG-025 (User documentation)
- Related: LOG-006 (Config types), LOG-009 (PowersLogger)

---

### LOG-025: Create User Documentation

**Priority**: High  
**Estimated Effort**: 1.5 days (confidence: high)

#### Context
Create comprehensive user-facing documentation for the logging system. This is critical for adoption and reduces support burden.

#### Acceptance Criteria
- [ ] `docs/guides/LOGGING-GUIDE.md` created
- [ ] All configuration options documented
- [ ] Examples for common use cases provided
- [ ] Troubleshooting section included
- [ ] README.md updated with logging section

#### Tasks

##### Create LOGGING-GUIDE.md
- [ ] Overview of logging system
- [ ] Configuration options explained
- [ ] Log levels and when to use them
- [ ] Output formats comparison
- [ ] CLI flag usage
- [ ] Multiple examples

##### Configuration Examples
- [ ] Default configuration (minimal)
- [ ] Debug mode configuration
- [ ] JSON output configuration
- [ ] File output configuration
- [ ] Silent mode (benchmarks)

##### Usage Examples
- [ ] Basic CLI usage
- [ ] Analyzing JSON logs with `jq`
- [ ] Troubleshooting with debug level
- [ ] Production deployment setup

##### Troubleshooting Section
- [ ] No logs appearing → check initialization
- [ ] Too many logs → increase level threshold
- [ ] Performance issues → check hot path logging
- [ ] File permission errors → check paths

##### Update README.md
- [ ] Add logging configuration section
- [ ] Link to LOGGING-GUIDE.md
- [ ] Add quick example
- [ ] Update table of contents

##### Update INPUT-SPECIFICATION.md
- [ ] Document `logging` field in config.json
- [ ] Provide complete schema
- [ ] Link to LOGGING-GUIDE.md

##### Testing
- [ ] Review documentation for accuracy
- [ ] Test all code examples
- [ ] Verify links work
- [ ] Get peer review

#### Technical Notes

**LOGGING-GUIDE.md structure:**
1. Introduction
2. Quick Start
3. Configuration Reference
4. Log Levels
5. Output Formats
6. CLI Flags
7. Examples
8. Troubleshooting
9. Best Practices
10. FAQ

**README.md addition:**
```markdown
### Logging Configuration

Control verbosity and output format via `config.json`:

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
powers examples/03-multistage --log-level debug --log-format json
```

See [Logging Guide](docs/guides/LOGGING-GUIDE.md) for details.
```

#### Dependencies
- Blocked by: LOG-024 (File sink implemented)
- Blocks: LOG-026 (Final testing)
- Related: All previous tickets

---

### LOG-026: Final Testing and Performance Validation

**Priority**: Critical  
**Estimated Effort**: 2 days (confidence: medium)

#### Context
Comprehensive testing before moving to cleanup phase. Validate all functionality, performance, and edge cases. This is the gate for Phase 3 completion.

#### Acceptance Criteria
- [ ] All features tested (terminal, JSON, file, silent)
- [ ] Performance benchmarks meet targets (< 1% regression)
- [ ] All tests pass (unit + integration)
- [ ] No memory leaks
- [ ] Binary size increase acceptable (< 20KB)

#### Tasks

##### Feature Testing
- [ ] Test default configuration (terminal, INFO)
- [ ] Test JSON output → validate with `jq`
- [ ] Test file output → verify file created
- [ ] Test terminal + file simultaneously
- [ ] Test silent mode (no output)
- [ ] Test all log levels (ERROR to TRACE)
- [ ] Test CLI flag overrides
- [ ] Test show_timing_detail flag

##### Performance Testing
- [ ] Run `cargo bench --bench sddp_benchmarks`
- [ ] Compare to `before-logging` baseline
- [ ] Verify < 1% regression on all benchmarks
- [ ] If regression > 1%, profile and optimize
- [ ] Document any performance impacts

##### Binary Size
- [ ] Measure release binary size
- [ ] Compare to baseline
- [ ] Verify increase < 20KB (target: ~15KB)
- [ ] Document size change

##### Memory Testing
- [ ] Run with Valgrind (if available)
- [ ] Check for memory leaks
- [ ] Monitor RSS during long runs
- [ ] Verify no excessive allocations

##### Edge Cases
- [ ] Invalid log level in config
- [ ] Invalid log format in config
- [ ] File path doesn't exist
- [ ] File permissions denied
- [ ] Concurrent logging from multiple threads
- [ ] Log rotation (if implemented)

##### Test Suite
- [ ] Run `cargo test --all-targets --all-features`
- [ ] Verify all tests pass
- [ ] Check code coverage (should be 100% for logging module)
- [ ] Run clippy with no warnings

##### Documentation Testing
- [ ] Follow LOGGING-GUIDE.md examples
- [ ] Verify all code samples work
- [ ] Test configuration examples
- [ ] Verify links are valid

##### Create Test Report
- [ ] Document all test results
- [ ] List any known issues
- [ ] Performance comparison table
- [ ] Sign-off from QA/reviewer

#### Technical Notes

**Performance benchmark commands:**
```bash
# Before logging (baseline)
cargo bench --bench sddp_benchmarks -- --save-baseline before-logging

# After Phase 3
cargo bench --bench sddp_benchmarks -- --baseline before-logging

# Should show < 1% regression
```

**Acceptance criteria for regression:**
- Training time: < 1% slower
- Memory usage: < 5% increase
- Binary size: < 20KB increase

**Sign-off required before proceeding to cleanup phase.**

#### Dependencies
- Blocked by: LOG-025 (Documentation complete)
- Blocks: None (Phase 3 complete!)
- Related: All Phase 3 tickets

---

## Sprint 3 Summary

**Total Effort**: 10 days  
**Tickets**: LOG-021 to LOG-026 (6 tickets)  
**Deliverable**: Complete logging system with JSON formatter, CLI flags, and documentation

**Completion Criteria**:
- ✅ No dependency on `src/log.rs`
- ✅ JSON formatter implemented
- ✅ CLI flags working
- ✅ File output supported
- ✅ Comprehensive documentation
- ✅ All tests pass
- ✅ Performance validated

**Next Sprint**: Cleanup and prepare for release.

---

## Sprint 4-5: Cleanup & Release (Weeks 8-10)

**Goal**: Remove deprecated code, final polish, release preparation

### LOG-027: Remove src/log.rs Module

**Priority**: High  
**Estimated Effort**: 1 day (confidence: high)

#### Context
Remove the deprecated `src/log.rs` module now that all functionality has been migrated to `src/logging/`. This is a major cleanup step.

#### Acceptance Criteria
- [ ] `src/log.rs` file deleted
- [ ] All references to `mod log;` removed
- [ ] All imports from `crate::log` removed
- [ ] Code compiles without errors
- [ ] All tests pass

#### Tasks

##### Remove File
- [ ] Delete `src/log.rs`
- [ ] Remove `mod log;` from `src/lib.rs`
- [ ] Commit with clear message: "Remove deprecated log module"

##### Update References
- [ ] Search codebase for `use crate::log`
- [ ] Remove all such imports
- [ ] Search for `log::` calls (legacy module, not macro)
- [ ] Verify none remain

##### Testing
- [ ] Run `cargo build` and verify success
- [ ] Run `cargo test --all-targets`
- [ ] Run clippy and fix any warnings
- [ ] Visual test: CLI output unchanged

##### Documentation
- [ ] Update CHANGELOG.md: "Removed deprecated log module"
- [ ] Remove any references in documentation
- [ ] Update architecture docs if needed

#### Dependencies
- Blocked by: LOG-021 (Inlined formatting logic)
- Blocks: LOG-028 (Remove env var)
- Related: None

---

### LOG-028: Remove Environment Variable Support

**Priority**: Medium  
**Estimated Effort**: 0.5 days (confidence: high)

#### Context
Ensure all environment variable checks are removed. The `POWERS_TIMING_DETAIL` check was removed in LOG-015, but verify no others exist.

#### Acceptance Criteria
- [ ] No `std::env::var("POWERS_*")` calls remain
- [ ] Documentation updated to remove env var references
- [ ] Migration guide documents the change

#### Tasks

##### Code Audit
- [ ] Run `grep -rn "env::var" src/` and verify results
- [ ] Run `grep -rn "POWERS_" src/` and verify no env vars
- [ ] Check for any other environment variable usage

##### Documentation Update
- [ ] Update README.md: remove env var mentions
- [ ] Add migration note to CHANGELOG.md
- [ ] Create migration guide if not exists

##### Testing
- [ ] Verify application doesn't read any POWERS_* env vars
- [ ] Test that timing detail uses config only
- [ ] Document the breaking change

#### Dependencies
- Blocked by: LOG-027 (Module removed)
- Blocks: LOG-029 (Documentation update)
- Related: LOG-015 (Timing detail migrated)

---

### LOG-029: Update All Documentation

**Priority**: High  
**Estimated Effort**: 2 days (confidence: high)

#### Context
Final documentation sweep to ensure everything is updated for the new logging system. This includes README, CHANGELOG, INPUT-SPECIFICATION, and architecture docs.

#### Acceptance Criteria
- [ ] README.md fully updated
- [ ] CHANGELOG.md complete for v0.3.0
- [ ] INPUT-SPECIFICATION.md documents logging config
- [ ] Architecture docs updated
- [ ] All examples have comments

#### Tasks

##### README.md
- [ ] Add logging section (if not already done)
- [ ] Update quick start if needed
- [ ] Add link to LOGGING-GUIDE.md
- [ ] Update table of contents

##### CHANGELOG.md
- [ ] Add v0.3.0 section
- [ ] List all Added/Changed/Removed items
- [ ] Include migration guide
- [ ] Note breaking changes

##### INPUT-SPECIFICATION.md
- [ ] Document logging field
- [ ] Provide complete schema
- [ ] Add examples

##### Architecture Docs
- [ ] Add LOGGING-DESIGN.md reference to architecture index
- [ ] Update system architecture diagram if needed
- [ ] Document module structure

##### Examples
- [ ] Review all example configs
- [ ] Add logging field where appropriate
- [ ] Add README to examples dir if needed

##### Testing
- [ ] Review all docs for accuracy
- [ ] Check all links
- [ ] Verify all code samples work
- [ ] Get peer review

#### Dependencies
- Blocked by: LOG-028 (Env vars removed)
- Blocks: LOG-030 (JSON schema update)
- Related: LOG-025 (User documentation)

---

### LOG-030: Update JSON Schema

**Priority**: Medium  
**Estimated Effort**: 1 day (confidence: high)

#### Context
Update `schemas/config.schema.json` to include the new logging configuration. This enables IDE auto-completion and validation.

#### Acceptance Criteria
- [ ] Logging field added to config schema
- [ ] All logging sub-types defined
- [ ] Enum values documented
- [ ] Default values specified
- [ ] VS Code validation works

#### Tasks

##### Schema Definition
- [ ] Add `logging` property to config schema
- [ ] Define `LogLevel` enum
- [ ] Define `LogFormat` enum
- [ ] Define `LogOutput` object (with variants)
- [ ] Add descriptions for all fields

##### Default Values
- [ ] Specify default for level (INFO)
- [ ] Specify default for format (terminal)
- [ ] Specify defaults for all boolean flags
- [ ] Test defaults work in VS Code

##### Validation
- [ ] Test schema with valid configs
- [ ] Test schema with invalid configs
- [ ] Verify errors are helpful
- [ ] Test in VS Code for auto-complete

##### Documentation
- [ ] Add comments to schema
- [ ] Link to LOGGING-GUIDE.md in schema
- [ ] Update CHANGELOG.md

#### Technical Notes

**Schema example:**
```json
{
  "properties": {
    "logging": {
      "type": "object",
      "description": "Logging configuration. See docs/guides/LOGGING-GUIDE.md",
      "properties": {
        "level": {
          "type": "string",
          "enum": ["error", "warn", "info", "debug", "trace"],
          "default": "info",
          "description": "Log level threshold"
        },
        "format": {
          "type": "string",
          "enum": ["terminal", "json", "structured"],
          "default": "terminal",
          "description": "Output format"
        }
      }
    }
  }
}
```

#### Dependencies
- Blocked by: LOG-029 (Documentation complete)
- Blocks: LOG-031 (Final checks)
- Related: LOG-006 (Config types)

---

### LOG-031: Final Pre-Release Checks

**Priority**: Critical  
**Estimated Effort**: 1.5 days (confidence: medium)

#### Context
Final comprehensive check before release. Verify everything works, documentation is complete, and we're ready to merge.

#### Acceptance Criteria
- [ ] All tests pass (unit + integration + benchmarks)
- [ ] All documentation complete and accurate
- [ ] No TODO/FIXME comments remain
- [ ] Clippy clean
- [ ] Code formatted
- [ ] CHANGELOG.md finalized

#### Tasks

##### Code Quality
- [ ] Run `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Run `cargo fmt --all -- --check`
- [ ] Search for TODO/FIXME comments and address
- [ ] Review all new code for quality

##### Testing
- [ ] Run `cargo test --all-targets --all-features`
- [ ] Run all benchmarks and verify < 1% regression
- [ ] Visual test: all examples
- [ ] Test on clean checkout (no stale artifacts)

##### Documentation
- [ ] Review all docs for completeness
- [ ] Check all links work
- [ ] Verify examples are up-to-date
- [ ] Spell check

##### Version Bump
- [ ] Update version in `Cargo.toml` (0.2.0 → 0.3.0)
- [ ] Update version in documentation
- [ ] Finalize CHANGELOG.md with release date

##### PR Preparation
- [ ] Rebase on main if needed
- [ ] Squash/organize commits if needed
- [ ] Write PR description
- [ ] Create PR with all checks passing

#### Technical Notes

**PR description template:**
```markdown
# Structured Logging System (v0.3.0)

## Summary
Replaces ad-hoc `println!` logging with professional structured logging using the `log` crate.

## Changes
- Configurable log levels (ERROR/WARN/INFO/DEBUG/TRACE)
- Multiple output formats (terminal, JSON)
- File output support
- CLI flag overrides
- Zero performance overhead when disabled

## Testing
- ✅ All 189+ tests pass
- ✅ CLI output visually identical (default config)
- ✅ Benchmarks show 0.5% regression (acceptable)
- ✅ 100% test coverage for logging module

## Documentation
- `docs/architecture/LOGGING-DESIGN.md` - Complete design
- `docs/guides/LOGGING-GUIDE.md` - User guide
- Updated README, CHANGELOG, INPUT-SPECIFICATION

## Migration
Backward compatible. Old configs work without changes.
Breaking: `POWERS_TIMING_DETAIL` env var removed (use config).

Closes #XXX
```

#### Dependencies
- Blocked by: LOG-030 (Schema updated)
- Blocks: LOG-032 (Release and merge)
- Related: All previous tickets

---

### LOG-032: Release and Merge

**Priority**: Critical  
**Estimated Effort**: 0.5 days (confidence: high)

#### Context
Final release steps: merge PR, tag release, update documentation, and announce.

#### Acceptance Criteria
- [ ] PR reviewed and approved
- [ ] PR merged to main
- [ ] Release tagged (v0.3.0)
- [ ] Release notes published
- [ ] Announcement made

#### Tasks

##### PR Review
- [ ] Address all review comments
- [ ] Get approval from maintainers
- [ ] Verify CI passing
- [ ] Final check of diff

##### Merge
- [ ] Merge PR to main
- [ ] Verify main branch builds
- [ ] Verify main tests pass

##### Release
- [ ] Tag release: `git tag v0.3.0`
- [ ] Push tag: `git push origin v0.3.0`
- [ ] Create GitHub release with notes
- [ ] Attach binaries if applicable

##### Documentation
- [ ] Update GitHub wiki if exists
- [ ] Update project website if exists
- [ ] Update crates.io listing (if published)

##### Announcement
- [ ] Post to project Slack/Discord
- [ ] Tweet/social media if applicable
- [ ] Update project roadmap
- [ ] Close related issues

##### Cleanup
- [ ] Delete feature branch (if policy allows)
- [ ] Update project board
- [ ] Schedule retrospective
- [ ] Celebrate! 🎉

#### Dependencies
- Blocked by: LOG-031 (Final checks complete)
- Blocks: None (Done!)
- Related: All tickets

---

## Sprint 4-5 Summary

**Total Effort**: 7 days  
**Tickets**: LOG-027 to LOG-032 (6 tickets)  
**Deliverable**: Clean, production-ready logging system, v0.3.0 released

**Completion Criteria**:
- ✅ All deprecated code removed
- ✅ All documentation complete
- ✅ JSON schema updated
- ✅ All checks passing
- ✅ v0.3.0 released and merged

---

## Epic Summary

**Total Tickets**: 32  
**Total Estimated Effort**: 50 days (part-time over 10 weeks)  
**Sprints**: 5 (Preparation + 4 implementation sprints)

### Ticket Dependencies Graph

```
Sprint 0: LOG-001 → LOG-002, LOG-003
Sprint 1: LOG-003 → LOG-004 → LOG-005 → LOG-006 → LOG-007 → LOG-008 → LOG-009 → LOG-010 → LOG-011 → LOG-012
Sprint 2: LOG-012 → LOG-013 → LOG-014 → LOG-015 → LOG-016 → LOG-017 → LOG-018 → LOG-019 → LOG-020
Sprint 3: LOG-020 → LOG-021 → LOG-022 → LOG-023 → LOG-024 → LOG-025 → LOG-026
Sprint 4-5: LOG-026 → LOG-027 → LOG-028 → LOG-029 → LOG-030 → LOG-031 → LOG-032
```

### Final Deliverables

1. **Code**: Fully functional structured logging system
2. **Documentation**: 4 comprehensive docs (Design, Implementation Plan, User Guide, Quick Reference)
3. **Tests**: 100% coverage for logging module, all existing tests passing
4. **Performance**: < 1% regression on benchmarks
5. **Migration**: Backward-compatible, clear migration guide

**Epic Status**: ✅ Ready for Implementation

---

**End of Tickets Document**

Generated: 2025-11-09  
Version: 1.0  
Epic: Professional Structured Logging System for POWE.RS
