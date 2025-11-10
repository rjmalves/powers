# Logging System Guide

**POWE.RS Structured Logging** - Professional logging system for optimization runs

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Reference](#configuration-reference)
3. [Log Levels](#log-levels)
4. [Output Formats](#output-formats)
5. [CLI Flags](#cli-flags)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [FAQ](#faq)

## Quick Start

### Default Behavior

By default, POWE.RS logs to the terminal at INFO level:

```bash
powers examples/01-deterministic
```

Output:

```
POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

Reading input files from 'examples/01-deterministic'

# Training
- Iterations: 10
- Forward passes: 5
- Cut selection: false

iter |     lower ($) |     simul ($) |          fwd |          bwd |        total
----------------------------------------------------------------------------------------
   1 |   2.499394e3 |   2.550123e3 | 00:00:00.010 | 00:00:00.005 | 00:00:00.015
   2 |   2.485672e3 |   2.543890e3 | 00:00:00.009 | 00:00:00.004 | 00:00:00.013
...
```

### Using CLI Flags

Override configuration with command-line flags:

```bash
# Enable debug logging
powers examples/01-deterministic --log-level debug

# Use JSON output
powers examples/01-deterministic --log-format json

# Combine flags
powers run data/ --log-level trace --log-format json
```

### Configuration File

Add logging section to `config.json`:

```json
{
  "num_iterations": 100,
  "num_forward_passes": 5,
  "logging": {
    "level": "info",
    "format": "terminal",
    "outputs": [{ "type": "terminal" }]
  }
}
```

## Configuration Reference

### LoggingConfig

Main configuration structure:

| Field     | Type           | Default                 | Description                  |
| --------- | -------------- | ----------------------- | ---------------------------- |
| `level`   | LogLevel       | `"info"`                | Minimum log level to display |
| `format`  | LogFormat      | `"terminal"`            | Output format                |
| `outputs` | Vec<LogOutput> | `[{"type":"terminal"}]` | Output destinations          |

### LogLevel

Available log levels (increasing verbosity):

- **`error`**: Only critical errors that stop execution
- **`warn`**: Warnings about potential issues
- **`info`**: Standard operational messages (default)
- **`debug`**: Detailed information for troubleshooting
- **`trace`**: Very detailed information (rarely needed)

### LogFormat

Output format options:

- **`terminal`**: Human-readable ASCII tables with colors (default)
- **`json`**: Machine-readable JSON Lines format
- **`structured`**: Terminal format without colors

### LogOutput

Output destination options:

```json
// Terminal (stdout)
{"type": "terminal"}

// File
{"type": "file", "path": "./logs/training.log"}

// No output (for benchmarking)
{"type": "silent"}
```

Multiple outputs supported:

```json
"outputs": [
  {"type": "terminal"},
  {"type": "file", "path": "./logs/run.log"}
]
```

## Log Levels

### ERROR

Critical failures that stop execution:

```
[ERROR] Solver returned infeasible for node 5, stage 2
[ERROR] Failed to read config file: 'config.json' not found
```

**When to use**: Program cannot continue

### WARN

Issues that don't stop execution but need attention:

```
[WARN] Cut pool at 90% capacity, performance may degrade
[WARN] Convergence stalled for 5 iterations
```

**When to use**: Potential problems, degraded performance

### INFO (Default)

Standard operational messages:

```
[INFO] Reading input files from 'examples/03-multistage'
[INFO] Training complete in 00:05:32.120
[INFO] Final policy cost: 2.499394e3 ± 1.234567e2
```

**When to use**: Normal execution flow, important events

### DEBUG

Detailed information for troubleshooting:

```
[DEBUG] Iteration 42 timing: forward=12.3s, backward=8.5s, solver_calls=120
[DEBUG] Added 15 cuts, removed 3 dominated cuts
[DEBUG] State space size: 2500 points
```

**When to use**: Investigating slow performance, algorithm behavior

### TRACE

Very detailed execution trace:

```
[TRACE] Entering forward pass for scenario 5/10
[TRACE] Solving subproblem: stage=3, node=12
[TRACE] Cut evaluation at state [100.5, 200.3, 50.1]
```

**When to use**: Deep debugging, rarely needed

## Output Formats

### Terminal Format

Human-readable with ASCII tables and optional colors:

```
# Training
- Iterations: 100
- Forward passes: 5

iter |     lower ($) |     simul ($) |          fwd |          bwd |        total
----------------------------------------------------------------------------------------
   1 |   2.499394e3 |   2.550123e3 | 00:00:00.010 | 00:00:00.005 | 00:00:00.015
```

**Colors**: Enabled when outputting to terminal (disabled for pipes/files)

### JSON Format

Machine-readable JSON Lines (one JSON object per line):

```json
{"timestamp":"2025-11-09T16:24:26Z","level":"INFO","message":"Starting training"}
{"timestamp":"2025-11-09T16:24:27Z","level":"INFO","iteration":1,"lower_bound":2499.394,"simulation_cost":2550.123,"forward_time_ms":10,"backward_time_ms":5,"total_time_ms":15,"message":"Iteration complete"}
```

**Use cases**:

- CI/CD pipelines
- Automated analysis with `jq`
- Log aggregation systems
- Machine learning feature extraction

### Structured Format

Terminal format without ANSI colors (for pipes):

```bash
powers examples/01-deterministic --log-format structured > output.txt
```

## CLI Flags

### --log-level

Override configured log level:

```bash
# Show debug information
powers examples/03-multistage --log-level debug

# Only errors
powers examples/01-deterministic --log-level error

# Maximum verbosity
powers run data/ --log-level trace
```

**Values**: `error`, `warn`, `info`, `debug`, `trace` (case-insensitive)

### --log-format

Override configured output format:

```bash
# JSON output for CI
powers examples/05-large-scale-brazilian --log-format json

# No colors for piping
powers run data/ --log-format structured | tee output.log
```

**Values**: `terminal`, `json`, `structured` (case-insensitive)

### Combining Flags

Both flags work together and override `config.json`:

```bash
powers run data/ --log-level debug --log-format json > debug.jsonl
```

**Priority**: CLI flags > config.json > defaults

## Examples

### Example 1: Production Run with File Logging

`config.json`:

```json
{
  "logging": {
    "level": "info",
    "format": "terminal",
    "outputs": [
      { "type": "terminal" },
      { "type": "file", "path": "./logs/production.log" }
    ]
  }
}
```

Both terminal and file get the same formatted output.

### Example 2: Debug Slow Training

```bash
# Enable debug logging to see timing details
powers examples/05-large-scale-brazilian --log-level debug

# Look for slow iterations
# Output shows solver calls, cut statistics, timing breakdown
```

### Example 3: Analyze Results with jq

```bash
# Run with JSON output
powers examples/03-multistage --log-format json > results.jsonl

# Extract iteration statistics
jq 'select(.iteration != null) | {iter: .iteration, lb: .lower_bound, cost: .simulation_cost}' results.jsonl

# Calculate average iteration time
jq 'select(.total_time_ms != null) | .total_time_ms' results.jsonl | \
  awk '{sum+=$1; n++} END {print sum/n " ms"}'

# Find slowest iteration
jq 'select(.total_time_ms != null) | {iter: .iteration, time: .total_time_ms}' results.jsonl | \
  jq -s 'sort_by(.time) | reverse | .[0]'
```

### Example 4: Silent Mode for Benchmarking

```json
{
  "logging": {
    "level": "error",
    "outputs": [{ "type": "silent" }]
  }
}
```

No output except errors, minimal overhead.

### Example 5: Multiple Log Files

```json
{
  "logging": {
    "level": "debug",
    "format": "json",
    "outputs": [
      { "type": "terminal" },
      { "type": "file", "path": "./logs/full-debug.jsonl" },
      { "type": "file", "path": "./logs/backup.jsonl" }
    ]
  }
}
```

Terminal shows formatted output, files get JSON.

## Troubleshooting

### No Logs Appearing

**Problem**: Running POWE.RS but no output

**Solutions**:

1. Check log level isn't too restrictive: `--log-level info`
2. Verify output isn't `silent`: Check `config.json`
3. Ensure logger initialization succeeded (check for errors at startup)

### Too Many Logs

**Problem**: Debug logs overwhelming the output

**Solutions**:

```bash
# Increase level threshold
powers run data/ --log-level warn

# Or in config.json
{"logging": {"level": "warn"}}
```

### File Permission Errors

**Problem**: `Failed to open log file '/var/log/powers.log': Permission denied`

**Solutions**:

1. Use a writable directory: `./logs/output.log`
2. Create directory first: `mkdir -p logs`
3. Check file permissions

### Performance Impact

**Problem**: Logging slowing down training

**Diagnosis**:

```bash
# Run with silent mode to measure baseline
powers run data/ --log-level error

# Compare with normal logging
powers run data/
```

**Solutions**:

- Reduce log level: `info` instead of `debug`
- Use `silent` output for benchmarks

### JSON Parsing Errors

**Problem**: `jq` fails to parse log output

**Solution**:

```bash
# Verify each line is valid JSON
while IFS= read -r line; do
  echo "$line" | jq . > /dev/null || echo "Invalid: $line"
done < output.jsonl
```

Check for mixed formats (terminal + JSON).

## Best Practices

### For Development

```json
{
  "logging": {
    "level": "debug",
    "format": "terminal",
    "outputs": [{ "type": "terminal" }]
  }
}
```

- Use `debug` level to understand algorithm behavior
- Terminal format for readability
- Show timing details to find bottlenecks

### For Production

```json
{
  "logging": {
    "level": "info",
    "format": "json",
    "outputs": [
      { "type": "terminal" },
      { "type": "file", "path": "/var/log/powers/production.jsonl" }
    ]
  }
}
```

- Use `info` level (key events only)
- JSON format for automated analysis
- Both terminal (monitoring) and file (archives)

### For CI/CD

```bash
# In CI script
powers run test_case/ --log-format json | tee ci-run.jsonl

# Extract success/failure
if jq -e '.level == "ERROR"' ci-run.jsonl > /dev/null; then
  echo "Run failed"
  exit 1
fi
```

- JSON format for parsing
- Pipe to both stdout and file
- Parse for errors programmatically

### For Benchmarking

```json
{
  "logging": {
    "level": "error",
    "outputs": [{ "type": "silent" }]
  }
}
```

- Minimize logging overhead
- Only report critical errors
- Silent output for clean benchmarks

## FAQ

### Q: How do I see detailed timing information?

**A**: Use `--log-level debug`:

```json
{ "logging": { "level": "debug" } }
```

### Q: Can I change log level without editing config.json?

**A**: Yes, use CLI flags:

```bash
powers run data/ --log-level debug
```

### Q: How do I log to multiple files?

**A**: Add multiple file outputs:

```json
"outputs": [
  {"type": "file", "path": "./main.log"},
  {"type": "file", "path": "./backup.log"}
]
```

### Q: Does logging affect performance?

**A**: Minimal impact at INFO level. Debug/trace have overhead. Benchmark with `silent` output:

```bash
powers run data/ --log-level error
```

### Q: Can I use colors in files?

**A**: No, colors are automatically disabled for file output. Use `terminal` output type for colors.

### Q: What's the difference between `terminal` and `structured`?

**A**: Both produce ASCII tables, but `structured` never uses colors. Use `structured` when piping to files.

### Q: How do I parse JSON logs?

**A**: Each line is a complete JSON object:

```bash
# Extract all error messages
jq 'select(.level == "ERROR") | .message' output.jsonl

# Get iteration times
jq 'select(.iteration) | {iter: .iteration, time: .total_time_ms}' output.jsonl
```

### Q: Where should I put log files?

**A**: Recommended structure:

```
project/
├── data/           # Input files
├── output/         # SDDP results
└── logs/          # Log files
    ├── training.log
    └── debug.jsonl
```

Use relative paths in config: `./logs/training.log`

### Q: How do I reduce log file size?

**A**:

1. Use higher log level (warn/error only)
2. Disable timing details
3. Use log rotation (external tool)
4. Compress old logs: `gzip logs/*.log`

---

**Next Steps**:

- See [INPUT-SPECIFICATION.md](../reference/INPUT-SPECIFICATION.md) for full config schema
- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues
- Review [LOGGING-DESIGN.md](../architecture/LOGGING-DESIGN.md) for architecture details

**Version**: 0.3.0  
**Last Updated**: 2025-11-09
