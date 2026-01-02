# RSS Monitoring - Quick Reference

## Monitor RSS

```bash
./scripts/monitor_rss_simple.py -o mydata -- ./target/release/powers run examples/...
```

**Output**: `mydata.csv` (samples), `mydata.stats.json` (statistics)

## Analyze Data

```bash
./scripts/monitor_rss_simple.py --analyze mydata.csv
```

## Plot Results

```bash
# Single plot (requires matplotlib)
./scripts/plot_rss_simple.py mydata.csv -o plot.png

# Compare allocators
./scripts/plot_rss_simple.py glibc.csv -o cmp.png --compare "mimalloc:mimalloc.csv"
```

## Compare Allocators

```bash
# Test each
for alloc in "" mimalloc jemalloc; do
    feature="${alloc:+--features $alloc}"
    ./scripts/monitor_rss_simple.py -o ${alloc:-glibc} -- \
        cargo run --release $feature -- run examples/05-large-scale-brazilian
done

# Show results
for f in glibc mimalloc jemalloc; do
    echo "=== $f ==="
    python3 -c "import json; s=json.load(open('${f}.stats.json')); \
        print(f\"Final: {s['rss_final']/1024:.1f} MB | \
Stable: {'Yes' if s['is_stable'] else 'No'}\")"
done
```

## Key Statistics

From `mydata.stats.json`:

```json
{
  "rss_final": 460660,        // Final RSS (KB)
  "rss_delta": 211916,         // Change from start (KB)
  "rss_growth_rate_mb_per_sec": 4.72,  // Growth rate
  "thread_count_mean": 5.0,    // Average threads
  "rss_per_thread_mean": 111252.3,  // Mean RSS per thread (KB)
  "is_stable": false,          // Stable after warmup?
  "stable_rss_stdev": 107342   // StdDev after warmup
}
```

## Interpreting Results

| Metric | Good | Bad |
|--------|------|-----|
| `is_stable` | `true` | `false` |
| Growth rate | ~0 MB/s | >5 MB/s |
| CV% (stdev/mean) | <5% | >10% |

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i` | Sample interval (seconds) | 0.5 |
| `-o` | Output file basename | None |
| `--analyze` | Analyze existing CSV | - |

---

**Full docs**: `scripts/SIMPLE_RSS_MONITORING.md`
