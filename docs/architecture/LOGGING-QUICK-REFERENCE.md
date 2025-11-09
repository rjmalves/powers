# Logging Quick Reference Card

**POWE.RS Structured Logging - Developer Cheat Sheet**

---

## Log Levels (When to Use)

| Level | Usage | Example |
|-------|-------|---------|
| `ERROR` | Unrecoverable failures | File I/O errors, solver crashes |
| `WARN` | Recoverable issues | Solver retries, numerical warnings |
| `INFO` | High-level progress | Iteration updates, training complete |
| `DEBUG` | Detailed diagnostics | Timing breakdowns, cut statistics |
| `TRACE` | Hot path events | Subproblem solves (disabled in release) |

**Default**: INFO for CLI, WARN for library

---

## Basic Usage

### Simple Messages
```rust
info!("Starting training");
warn!("Solver required retry");
error!("File not found: {}", path);
```

### Structured Logs
```rust
info!(
    iteration = iter,
    lower_bound = lower,
    simulation_cost = simul;
    "Iteration complete"
);
```

### Conditional Logging
```rust
// Only if DEBUG is enabled (zero-cost if not)
debug!("Detailed timing: {:?}", timings);

// Never log in hot loops (aggregate instead)
debug!("Processing {} nodes", nodes.len());
```

---

## Configuration

### Via `config.json`
```json
{
  "logging": {
    "level": "INFO",
    "format": "terminal",
    "show_timing_detail": false,
    "show_progress_bar": true,
    "outputs": [{"type": "terminal"}]
  }
}
```

### Via CLI Flags
```bash
# Override log level
powers examples/03-multistage --log-level debug

# Change output format
powers examples/03-multistage --log-format json

# Combine
powers run --log-level trace --log-format json data/
```

---

## Output Formats

### Terminal (Default)
```
[INFO] Starting training
[DEBUG] Thread pool: 8 threads
iter |      lower ($) |      simul ($) | ...
   1 |     2.499394e3 |     2.499394e3 | ...
```

### JSON (Machine-Readable)
```json
{"timestamp":"2025-11-09T13:24:48Z","level":"INFO","iteration":1,"lower_bound":2499.394}
```

### Silent (Benchmarks)
```json
{"logging": {"outputs": [{"type": "silent"}]}}
```

---

## Context Management

### Thread-Local Context
```rust
// Set context for scope
LogContext::with_iteration(42, || {
    info!("Processing");  // Automatically tagged with iteration=42
});

// Manual control
LogContext::set(LogContext {
    iteration: Some(42),
    stage: Some(5),
    ..Default::default()
});

info!("Node solved");  // Has iteration + stage context

LogContext::clear();  // Clean up
```

---

## Migration Guide

### Replace Print Statements
```rust
// OLD
println!("Training iteration {}", iter);

// NEW
info!(iteration = iter; "Training iteration");
```

### Replace Error Messages
```rust
// OLD
eprintln!("Error: {}", e);

// NEW
error!("Execution failed: {}", e);
```

### Replace Environment Variables
```rust
// OLD
if std::env::var("POWERS_TIMING_DETAIL").is_ok() {
    show_timing();
}

// NEW
if config.logging.show_timing_detail {
    debug!(timing = ?timing_data; "Detailed timing");
}
```

---

## Performance Rules

### ✅ DO
- Use `trace!` for hot path (compiled out in release)
- Defer expensive computations: `debug!("{:?}", value)`
- Aggregate counts: `debug!("Processed {} items", n)`
- Use context instead of parameters

### ❌ DON'T
- Don't log in tight loops
- Don't compute before checking level: `debug!("{}", expensive())`
- Don't pass log level as parameter
- Don't allocate in hot paths

---

## Testing

### Unit Tests
```rust
#[test]
fn test_with_logging() {
    // Initialize test logger
    let _ = env_logger::builder().is_test(true).try_init();
    
    info!("Test running");
    // Logs appear with `cargo test -- --nocapture`
}
```

### Integration Tests
```rust
// Capture logs
let output = std::process::Command::new("powers")
    .arg("examples/01-deterministic")
    .output()
    .unwrap();

let stdout = String::from_utf8(output.stdout).unwrap();
assert!(stdout.contains("[INFO] Starting training"));
```

---

## Common Patterns

### Training Loop
```rust
info!(
    num_iterations = config.num_iterations,
    num_forward_passes = config.num_forward_passes;
    "Starting training"
);

for iter in 0..num_iterations {
    LogContext::set(LogContext {
        iteration: Some(iter + 1),
        lower_bound: Some(lower),
        simulation_cost: Some(simul),
        ..Default::default()
    });
    
    info!("Iteration complete");
    
    if config.logging.show_timing_detail {
        debug!(
            forward_ms = forward_time.as_millis(),
            backward_ms = backward_time.as_millis();
            "Timing breakdown"
        );
    }
    
    LogContext::clear();
}
```

### Error Handling
```rust
match risky_operation() {
    Ok(result) => {
        debug!("Operation succeeded: {:?}", result);
        result
    }
    Err(e) => {
        error!(error = %e; "Operation failed");
        return Err(e);
    }
}
```

### Solver Diagnostics
```rust
if solver_status == Infeasible {
    error!(
        node_id = node.id,
        num_constraints = model.num_rows(),
        num_variables = model.num_cols();
        "Solver returned infeasible"
    );
    
    // Only compute expensive diagnostics if DEBUG enabled
    if log::log_enabled!(log::Level::Debug) {
        debug!("Constraint structure: {:?}", analyze_constraints(&model));
    }
}
```

---

## Formatter Cheat Sheet

### Terminal Formatter
- Pretty ANSI colors (if terminal)
- Preserves ASCII tables
- Timestamps optional
- Box-drawing for timing detail

### JSON Formatter
- JSON Lines format (one log per line)
- All fields flattened
- ISO 8601 timestamps
- Parseable with `jq`

---

## Troubleshooting

### No Logs Appearing
```rust
// Did you initialize?
crate::logging::init(&config.logging)?;

// Is level too restrictive?
// DEBUG logs won't show if level=INFO
```

### Too Many Logs
```rust
// Increase log level threshold
config.logging.level = LogLevel::Warn;  // Only WARN and ERROR
```

### Performance Regression
```bash
# Check if debug logs in hot path
rg "debug!" src/subproblem.rs

# Should use trace! instead (compiled out)
rg "trace!" src/subproblem.rs
```

---

## Examples

### Example 1: CLI with Debug
```bash
powers examples/03-multistage --log-level debug
```

### Example 2: JSON to File
```bash
powers examples/03-multistage --log-format json > training.jsonl
cat training.jsonl | jq 'select(.iteration) | {iteration, lower_bound}'
```

### Example 3: Silent Library
```rust
let config = Config {
    logging: LoggingConfig {
        outputs: vec![LogOutput::Silent],
        ..Default::default()
    },
    ..load_config("config.json")?
};

let mut sddp = SddpAlgorithm::from_config(config)?;
sddp.train()?;  // No terminal output
```

---

## References

- **Design Doc**: [`docs/architecture/LOGGING-DESIGN.md`](./LOGGING-DESIGN.md)
- **Implementation Plan**: [`docs/architecture/LOGGING-IMPLEMENTATION-PLAN.md`](./LOGGING-IMPLEMENTATION-PLAN.md)
- **User Guide**: [`docs/guides/LOGGING-GUIDE.md`](../guides/LOGGING-GUIDE.md) *(to be created)*
- **Rust `log` crate**: https://docs.rs/log

---

**Version**: 1.0  
**Last Updated**: 2025-11-09
