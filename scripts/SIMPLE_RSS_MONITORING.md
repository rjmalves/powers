# Simple RSS Monitoring Tools

Pure external RSS monitoring - no log parsing, comprehensive statistics.

## Quick Start

### Monitor RSS with Statistics

```bash
# Monitor and get automatic statistics
./scripts/monitor_rss_simple.py -o mydata -- \
    ./target/release/powers run examples/05-large-scale-brazilian

# Output:
# - mydata.csv (RSS samples with thread breakdown)
# - mydata.stats.json (comprehensive statistics)
# - Terminal output with formatted statistics
```

### Analyze Existing Data

```bash
# Re-analyze saved data
./scripts/monitor_rss_simple.py --analyze mydata.csv
```

### Plot Results

```bash
# Single plot
./scripts/plot_rss_simple.py mydata.csv -o plot.png

# Compare allocators
./scripts/plot_rss_simple.py glibc.csv -o comparison.png \
    --compare "mimalloc:mimalloc.csv" "jemalloc:jemalloc.csv"
```

## Features

### Pure External Monitoring

✅ **No log parsing** - Just monitors `/proc/{pid}/status`
✅ **Thread tracking** - Per-thread RSS breakdown
✅ **Automatic statistics** - Comprehensive analysis on completion
✅ **Simple output** - CSV + JSON

### Statistics Computed

**Basic RSS Metrics:**
- Min, Max, Mean, Median, StdDev
- P95, P99 percentiles
- Initial, Final, Delta
- Growth rate (MB/s)

**Stability Analysis:**
- Automatic warmup detection (10% of samples)
- Coefficient of variation
- Stable/Unstable classification

**Thread Analysis:**
- Thread count over time (min/max/mean)
- **Per-thread RSS** (min/max/mean/median)
  - Simple division: Total RSS / Thread count

## Output Format

### CSV (mydata.csv)

```csv
timestamp,rss_kb,rss_mb,thread_count,threads_detail
0.510,248744,242.9,5,28829:248872;28830:248872;28831:249000
1.011,438056,427.8,5,28829:438056;28830:438056;28831:438056
```

**Columns:**
- `timestamp`: Seconds since program start
- `rss_kb`: Total process RSS in KB
- `rss_mb`: Total process RSS in MB
- `thread_count`: Number of threads
- `threads_detail`: Per-thread RSS (tid:rss_kb;...)

### JSON Statistics (mydata.stats.json)

```json
{
  "duration_seconds": 43.81,
  "sample_count": 87,
  "rss_min": 248744,
  "rss_max": 782120,
  "rss_mean": 567085.2,
  "rss_median": 574848,
  "rss_stdev": 111983.1,
  "rss_p95": 736632,
  "rss_p99": 782120,
  "rss_initial": 248744,
  "rss_final": 460660,
  "rss_delta": 211916,
  "rss_growth_rate_mb_per_sec": 4.72,
  "thread_count_min": 5,
  "thread_count_max": 5,
  "thread_count_mean": 5.0,
  "rss_per_thread_min": 42872.0,
  "rss_per_thread_max": 158936.0,
  "rss_per_thread_mean": 111252.3,
  "rss_per_thread_median": 113793.6,
  "is_stable": false,
  "warmup_samples": 8,
  "stable_rss_mean": 579308.7,
  "stable_rss_stdev": 107342.5
}
```

### Terminal Statistics Output

```
======================================================================
RSS STATISTICS
======================================================================

Duration: 43.8s
Samples: 87 (interval: 0.5s)

--- RSS Memory Usage ---
Initial:     242.9 MB
Final:       449.9 MB
Delta:      +206.9 MB
Min:         242.9 MB
Max:         763.8 MB
Mean:        553.8 MB
Median:      561.4 MB
StdDev:      109.4 MB
P95:         719.4 MB
P99:         763.8 MB

--- Growth Rate ---
Rate: +4.724 MB/s

--- Thread Count ---
Min:  5
Max:  5
Mean: 5.0

--- Per-Thread RSS ---
Min:    41.9 MB/thread
Max:   155.2 MB/thread
Mean:  108.6 MB/thread
Median: 111.1 MB/thread

--- Stability Analysis ---
Warmup samples: 8
Stable mean:    565.7 MB
Stable stdev:   104.8 MB
Coeff. of var:  18.53%
Stable: ✗ NO
======================================================================
```

## Usage Examples

### Example 1: Quick Test

```bash
./scripts/monitor_rss_simple.py -o test -- \
    ./target/release/powers run examples/01-single-stage

# Check results
cat test.stats.json
```

### Example 2: High-Frequency Sampling

```bash
# 100ms sampling for detailed profiling
./scripts/monitor_rss_simple.py -i 0.1 -o detailed -- \
    ./target/release/powers run examples/05-large-scale-brazilian
```

### Example 3: Compare Allocators

```bash
# Test each allocator
./scripts/monitor_rss_simple.py -o glibc -- \
    cargo run --release -- run examples/05-large-scale-brazilian

./scripts/monitor_rss_simple.py -o mimalloc -- \
    cargo run --release --features mimalloc -- run examples/05-large-scale-brazilian

./scripts/monitor_rss_simple.py -o jemalloc -- \
    cargo run --release --features jemalloc -- run examples/05-large-scale-brazilian

# Compare statistics
for f in glibc mimalloc jemalloc; do
    echo "=== $f ==="
    python3 -c "import json; s=json.load(open('${f}.stats.json')); \
        print(f\"Final: {s['rss_final']/1024:.1f} MB\"); \
        print(f\"Stable: {'Yes' if s['is_stable'] else 'No'}\")"
    echo ""
done

# Plot comparison (if matplotlib available)
./scripts/plot_rss_simple.py glibc.csv -o comparison.png \
    --compare "mimalloc:mimalloc.csv" "jemalloc:jemalloc.csv"
```

### Example 4: Programmatic Analysis

```python
#!/usr/bin/env python3
import json

# Load statistics
with open('mydata.stats.json') as f:
    stats = json.load(f)

# Check if memory is stable
if stats['is_stable']:
    print(f"✓ Memory stable at {stats['stable_rss_mean']/1024:.1f} MB")
else:
    print(f"✗ Memory growing at {stats['rss_growth_rate_mb_per_sec']:.2f} MB/s")
    print(f"  CV: {stats['stable_rss_stdev']/stats['stable_rss_mean']*100:.1f}%")

# Check thread count
if stats['thread_count_max'] != stats['thread_count_min']:
    print(f"⚠ Thread count varies: {stats['thread_count_min']}-{stats['thread_count_max']}")
```

## Per-Thread Analysis

The monitor tracks individual thread RSS values. Note that on Linux, all threads report the same RSS (it's process-wide), but tracking thread count is useful:

```python
#!/usr/bin/env python3
import csv

# Analyze thread behavior
with open('mydata.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamp = float(row['timestamp'])
        threads = row['threads_detail'].split(';')
        
        if len(threads) != int(row['thread_count']):
            print(f"Thread count changed at t={timestamp:.1f}s: {len(threads)} threads")
```

## Interpreting Statistics

### Is Stable

- **Yes (CV < 5%)**: RSS variation is low after warmup → Good memory behavior
- **No (CV ≥ 5%)**: RSS varies significantly → Memory leak or allocator issue

### Growth Rate

- **~0 MB/s**: Stable memory usage
- **+0.1 to +1 MB/s**: Slow growth (may be acceptable for long runs)
- **>+5 MB/s**: Rapid growth (likely memory leak or poor allocator fit)

### Warmup Samples

- Default: 10% of total samples
- After warmup, we calculate "stable" statistics
- Allows for initial allocation spike without marking as unstable

## Performance

| Interval | CPU Overhead | Memory | Disk I/O |
|----------|--------------|--------|----------|
| 0.01s | ~0.1% | 10 MB | Negligible |
| 0.1s | ~0.01% | 10 MB | Negligible |
| 0.5s (default) | <0.001% | 10 MB | Negligible |
| 1.0s | <0.0001% | 10 MB | Negligible |

## Comparison with Old monitor_rss.py

| Feature | Old (monitor_rss.py) | New (monitor_rss_simple.py) |
|---------|----------------------|----------------------------|
| Log parsing | ✓ (iteration events) | ✗ (pure external) |
| RSS tracking | ✓ | ✓ |
| Thread tracking | ✗ | ✓ |
| Statistics | Basic | Comprehensive |
| Matplotlib dependency | Required for plot | Optional |
| Complexity | High | Low |
| Reliability | Depends on logs | Always works |

## Requirements

**Required:**
- Python 3.7+
- Linux (uses `/proc/{pid}/status`)

**Optional (for plotting):**
- matplotlib 3.5+: `pip install matplotlib`

## Troubleshooting

### "Cannot read /proc/{pid}/status"

Ensure you're on Linux with read permissions.

### No statistics output

Check that `-o` or `--analyze` flag is provided.

### Thread count always same

This is normal on Linux - RSS is process-wide, not per-thread.

## Integration with CI/CD

```yaml
# .github/workflows/memory-check.yml
- name: Monitor RSS
  run: |
    ./scripts/monitor_rss_simple.py -o rss_ci -- \
        cargo run --release -- run examples/05-large-scale-brazilian
    
    # Check if stable
    python3 << 'EOF'
    import json, sys
    with open('rss_ci.stats.json') as f:
        stats = json.load(f)
    
    if not stats['is_stable']:
        print(f"✗ RSS not stable (CV: {stats['stable_rss_stdev']/stats['stable_rss_mean']*100:.1f}%)")
        sys.exit(1)
    
    if stats['rss_final'] > 300 * 1024:  # 300 MB
        print(f"✗ Final RSS too high: {stats['rss_final']/1024:.1f} MB")
        sys.exit(1)
    
    print(f"✓ Memory OK: {stats['rss_final']/1024:.1f} MB, stable")
    EOF
```

---

**Simple, reliable, comprehensive RSS monitoring for HPC workloads.**
