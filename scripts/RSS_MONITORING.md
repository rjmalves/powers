# RSS Monitoring and Visualization Tools

External memory monitoring tools for analyzing RSS (Resident Set Size) behavior of HPC workloads.

## Quick Start

### Monitor and Plot Memory Usage

```bash
# Basic monitoring with plot
./scripts/monitor_rss.py -o rss_data.csv -p rss_plot.png -- \
    ./target/release/powers run examples/05-large-scale-brazilian --log-level debug

# Monitor with high-frequency sampling (100ms)
./scripts/monitor_rss.py -i 0.1 -o rss_data.csv -- \
    cargo run --release -- run examples/05-large-scale-brazilian

# Plot existing data
./scripts/monitor_rss.py --plot-only rss_data.csv -p my_plot.png
```

### Compare Allocators

```bash
# Test glibc
./scripts/monitor_rss.py -o glibc_rss.csv -p glibc.png -- \
    cargo run --release -- run examples/05-large-scale-brazilian --log-level debug

# Test mimalloc
./scripts/monitor_rss.py -o mimalloc_rss.csv -p mimalloc.png -- \
    cargo run --release --features mimalloc -- run examples/05-large-scale-brazilian --log-level debug

# Test jemalloc
./scripts/monitor_rss.py -o jemalloc_rss.csv -p jemalloc.png -- \
    cargo run --release --features jemalloc -- run examples/05-large-scale-brazilian --log-level debug
```

## Features

### External RSS Monitoring

- **Non-invasive**: Monitors via `/proc/{pid}/status` (Linux)
- **Configurable sampling**: 10ms to 10s intervals
- **Low overhead**: Minimal impact on monitored process

### Log Parsing

Automatically detects iteration boundaries from program logs:

```
[DEBUG] RSS at iteration 1 start: VmRSS:	  123456 kB
[DEBUG] RSS at iteration 1 end (after finalize): VmRSS:	  234567 kB
```

Also recognizes:
- `Iteration N complete`
- `Starting iteration N`

### Visualization

Creates dual-plot visualization:

1. **RSS Timeline**: Memory usage over time with iteration markers
   - Green triangles (▲): Iteration start
   - Red triangles (▼): Iteration end
   - Shaded area: RSS footprint

2. **Per-Iteration Delta**: Memory change per iteration
   - Green bars: Small delta (<5 MB)
   - Orange bars: Medium delta (5-10 MB)
   - Red bars: Large delta (>10 MB)

## Output Files

Running with `-o data.csv` generates:

| File | Content |
|------|---------|
| `data.csv` | RSS samples (timestamp, rss_kb, rss_mb, wall_time) |
| `data.iterations.json` | Iteration events (iteration, phase, timestamp, rss_kb) |
| `plot.png` (if `-p`) | Visualization |

## Example Output

### CSV Format

```csv
timestamp,rss_kb,rss_mb,wall_time
0.000,209624,204.6,1735747200.123
0.500,215432,210.4,1735747200.623
1.000,252424,246.5,1735747201.123
...
```

### JSON Format

```json
[
  {
    "iteration": 1,
    "phase": "start",
    "timestamp": 0.5,
    "rss_kb": 209624
  },
  {
    "iteration": 1,
    "phase": "end",
    "timestamp": 5.2,
    "rss_kb": 252424
  }
]
```

## Advanced Usage

### Custom Plotting

```python
from scripts.monitor_rss import RssMonitor, plot_rss_timeline

# Load data
samples, iterations = RssMonitor.load_data(Path('rss_data.csv'))

# Create plot
plot_rss_timeline(samples, iterations, Path('custom_plot.png'))
```

### Programmatic Monitoring

```python
from scripts.monitor_rss import RssMonitor

monitor = RssMonitor(interval=0.1)  # 100ms sampling
exit_code = monitor.run(['./my_program', 'arg1'], output_file=Path('data.csv'))

# Access samples
for sample in monitor.samples:
    print(f"t={sample.timestamp:.1f}s RSS={sample.rss_kb}KB")

# Access iterations
for event in monitor.iterations:
    print(f"Iteration {event.iteration} {event.phase} at t={event.timestamp:.1f}s")
```

## Requirements

- **Python 3.7+**
- **matplotlib** (for plotting): `pip install matplotlib`
- **Linux** (uses `/proc/{pid}/status`)

## Performance Notes

### Sampling Interval

| Interval | Use Case | Overhead |
|----------|----------|----------|
| 0.01s (10ms) | Detailed profiling | ~0.1% |
| 0.1s (100ms) | Standard monitoring | ~0.01% |
| 0.5s (500ms) | Long runs (default) | <0.001% |
| 1.0s (1s) | Production monitoring | Negligible |

### Memory Overhead

- Script: ~10-20 MB
- Per sample: ~40 bytes
- 1 hour @ 0.5s interval: ~300 KB data

## Integration with DHAT

RSS monitoring complements DHAT profiling:

| Tool | Measures | Use For |
|------|----------|---------|
| **DHAT** | Total allocations | Finding allocation hotspots |
| **RSS Monitor** | Physical memory | Memory footprint, leaks |

Run both for complete analysis:

```bash
# DHAT profiling
cargo build --release --features dhat-heap
./target/release/powers run examples/05-large-scale-brazilian

# RSS monitoring
./scripts/monitor_rss.py -o rss.csv -p rss.png -- \
    ./target/release/powers run examples/05-large-scale-brazilian --log-level debug
```

## Troubleshooting

### "Cannot open /proc/{pid}/status"

Ensure you're on Linux and have read permissions.

### "matplotlib not found"

```bash
pip install matplotlib
# or
pip install -r scripts/requirements.txt
```

### Iteration events not detected

Check log level:
```bash
./target/release/powers run examples/05-large-scale-brazilian --log-level debug
```

The RSS logging requires debug level.

### Plot shows no iteration markers

Verify that iteration events are in JSON file:
```bash
cat rss_data.iterations.json
```

## Examples

### Quick Test

```bash
# Monitor with data collection
./scripts/monitor_rss.py -o test_rss.csv -- \
    ./target/release/powers run examples/05-large-scale-brazilian --log-level debug

# Examine data
head test_rss.csv
cat test_rss.iterations.json | jq
```

### Allocator Comparison (Automated)

```bash
# Run full comparison suite
./scripts/compare_allocators.sh
```

This script:
1. Builds all allocator variants
2. Runs each with RSS monitoring
3. Generates individual and comparison plots
4. Creates summary report

### Manual Allocator Comparison

```bash
# Test each allocator
./scripts/monitor_rss.py -o glibc.csv -- \
    cargo run --release -- run examples/05-large-scale-brazilian --log-level debug

./scripts/monitor_rss.py -o mimalloc.csv -- \
    cargo run --release --features mimalloc -- run examples/05-large-scale-brazilian --log-level debug

./scripts/monitor_rss.py -o jemalloc.csv -- \
    cargo run --release --features jemalloc -- run examples/05-large-scale-brazilian --log-level debug

# Plot comparison (if matplotlib available)
./scripts/plot_rss.py glibc.csv -o comparison.png \
    --compare "mimalloc:mimalloc.csv" "jemalloc:jemalloc.csv"
```

### Data Analysis

```python
#!/usr/bin/env python3
import csv
import json

# Load RSS data
with open('rss_data.csv') as f:
    reader = csv.DictReader(f)
    samples = list(reader)

# Calculate stats
rss_values = [float(s['rss_mb']) for s in samples]
print(f"Min RSS: {min(rss_values):.1f} MB")
print(f"Max RSS: {max(rss_values):.1f} MB")
print(f"Avg RSS: {sum(rss_values)/len(rss_values):.1f} MB")

# Load iterations
with open('rss_data.iterations.json') as f:
    iterations = json.load(f)

# Per-iteration delta
for i, event in enumerate(iterations):
    if event['phase'] == 'start' and i+1 < len(iterations):
        next_event = iterations[i+1]
        if next_event['phase'] == 'end' and next_event['iteration'] == event['iteration']:
            delta = (next_event['rss_kb'] - event['rss_kb']) / 1024
            print(f"Iteration {event['iteration']}: {delta:+.1f} MB")
```

See `docs/ALLOCATOR_COMPARISON.md` for real-world results.

---

*Part of Epic 5: Memory Optimization*
*Sprint 9: RSS Stabilization*
