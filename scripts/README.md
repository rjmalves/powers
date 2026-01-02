# RSS Monitoring Scripts

Simple, reliable external RSS monitoring tools for HPC workloads.

## Tools

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `monitor_rss_simple.py` | Monitor RSS + compute statistics | Python 3.7+ |
| `plot_rss_simple.py` | Plot RSS timeline | matplotlib |
| `compare_allocators.sh` | Compare allocator performance | bash, python3 |

## Quick Start

```bash
# Monitor RSS
./scripts/monitor_rss_simple.py -o mydata -- ./target/release/powers run examples/...

# Plot results (if matplotlib available)
./scripts/plot_rss_simple.py mydata.csv -o plot.png

# Analyze existing data
./scripts/monitor_rss_simple.py --analyze mydata.csv
```

## Features

✅ **Pure external monitoring** - No log parsing
✅ **Comprehensive statistics** - Min, Max, Mean, Median, StdDev, P95, P99
✅ **Per-thread analysis** - RSS per thread metrics
✅ **Stability detection** - Automatic warmup and CV% analysis
✅ **Thread tracking** - Thread count over time
✅ **Simple output** - CSV + JSON

## Output

**CSV**: RSS samples with thread breakdown
**JSON**: Complete statistics including per-thread metrics
**Terminal**: Formatted statistics report
**Plots**: RSS timeline + per-thread RSS (requires matplotlib)

## Statistics Computed

### RSS Metrics
- Basic: Min, Max, Mean, Median, StdDev, P95, P99
- Growth: Initial, Final, Delta, Rate (MB/s)
- Stability: CV%, Stable/Unstable classification

### Thread Metrics
- Count: Min, Max, Mean
- **Per-thread RSS: Min, Max, Mean, Median**

## Documentation

- **[SIMPLE_RSS_MONITORING.md](SIMPLE_RSS_MONITORING.md)** - Complete guide
- **[RSS_QUICK_REF.md](RSS_QUICK_REF.md)** - Quick reference
- **[PER_THREAD_RSS_GUIDE.md](PER_THREAD_RSS_GUIDE.md)** - Per-thread analysis guide

## Examples

### Basic Monitoring

```bash
./scripts/monitor_rss_simple.py -o test -- ./my_program
```

### High-Frequency Sampling

```bash
./scripts/monitor_rss_simple.py -i 0.1 -o detailed -- ./my_program
```

### Compare Allocators

```bash
# Automated
./scripts/compare_allocators.sh

# Manual
for alloc in glibc mimalloc jemalloc; do
    feature="${alloc:+--features $alloc}"
    ./scripts/monitor_rss_simple.py -o $alloc -- \
        cargo run --release $feature -- run examples/...
done
```

### Extract Per-Thread Stats

```python
import json

with open('mydata.stats.json') as f:
    s = json.load(f)

print(f"Threads: {s['thread_count_mean']:.0f}")
print(f"Per-thread mean: {s['rss_per_thread_mean']/1024:.1f} MB")
print(f"Per-thread max: {s['rss_per_thread_max']/1024:.1f} MB")
```

## Requirements

**Required:**
- Python 3.7+
- Linux (uses `/proc/{pid}/status`)

**Optional:**
- matplotlib 3.5+ (for plotting): `pip install matplotlib`

## Performance

| Interval | CPU | Memory | Disk |
|----------|-----|--------|------|
| 0.1s | 0.01% | 10 MB | Negligible |
| 0.5s (default) | <0.001% | 10 MB | Negligible |

## CI/CD Integration

```yaml
- name: Monitor RSS
  run: |
    ./scripts/monitor_rss_simple.py -o ci_rss -- cargo run --release
    
    python3 << 'EOF'
    import json, sys
    with open('ci_rss.stats.json') as f:
        s = json.load(f)
    
    # Check stability
    if not s['is_stable']:
        print(f"✗ Unstable (CV: {s['stable_rss_stdev']/s['stable_rss_mean']*100:.1f}%)")
        sys.exit(1)
    
    # Check final RSS
    if s['rss_final'] > 500 * 1024:  # 500 MB
        print(f"✗ Too much memory: {s['rss_final']/1024:.0f} MB")
        sys.exit(1)
    
    print(f"✓ OK: {s['rss_final']/1024:.0f} MB, stable")
    EOF
```

---

**Simple, reliable, comprehensive RSS monitoring for HPC workloads.**
