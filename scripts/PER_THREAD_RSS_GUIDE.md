# Per-Thread RSS Analysis Example

## Understanding Per-Thread RSS

The monitoring tools now compute **per-thread RSS** by dividing total process RSS by the number of threads:

```
RSS per thread = Total RSS / Thread count
```

### Why This Matters

For HPC workloads with constant thread count, per-thread RSS shows:
- Memory usage per worker thread
- Whether memory scales linearly with threads
- Per-thread memory efficiency

### Example Output

Running `examples/05-large-scale-brazilian` with 5 threads:

```
--- RSS Memory Usage ---
Initial:     209.3 MB
Final:       448.4 MB
Delta:      +239.1 MB
Mean:        543.2 MB
Max:         776.1 MB

--- Thread Count ---
Min:  5
Max:  5
Mean: 5.0

--- Per-Thread RSS ---
Min:    41.9 MB/thread  (at start)
Max:   155.2 MB/thread  (peak usage)
Mean:  108.6 MB/thread  (average)
Median: 111.1 MB/thread
```

### Interpretation

**Minimum per-thread** (41.9 MB):
- Initial memory footprint per thread
- Baseline thread overhead

**Maximum per-thread** (155.2 MB):
- Peak memory usage per thread
- Important for capacity planning

**Mean per-thread** (108.6 MB):
- Average memory per thread during execution
- Used for resource allocation estimates

### Visualization

The plot shows two subplots:

**Top: Total RSS Timeline**
```
800 MB ┤     ╭─╮
600 MB ┤   ╭─╯ ╰─╮
400 MB ┤  ╭╯     ╰╮
200 MB ┼──╯       ╰─
       └────────────> time
```

**Bottom: Per-Thread RSS**
```
160 MB ┤     ╭─╮
120 MB ┤   ╭─╯ ╰─╮  ← Mean: 108.6 MB/thread
 80 MB ┤  ╭╯     ╰╮
 40 MB ┼──╯       ╰─
       └────────────> time
       Thread count: 5
```

## Usage for Different Thread Counts

### Single Thread (thread_count = 1)

```bash
./scripts/monitor_rss_simple.py -o single -- \
    cargo run --release -- run examples/01-single-stage

# Per-thread RSS = Total RSS (same value)
```

### Multi-Thread (thread_count = N)

```bash
# Default (likely 4-8 threads based on CPU cores)
./scripts/monitor_rss_simple.py -o multi -- \
    cargo run --release -- run examples/05-large-scale-brazilian

# Per-thread RSS = Total RSS / N
```

## Comparing Thread Configurations

```bash
# Test different thread counts
for threads in 1 2 4 8; do
    ./scripts/monitor_rss_simple.py -o "threads_${threads}" -- \
        ./target/release/powers run examples/... --threads $threads
    
    # Extract per-thread stats
    python3 -c "
import json
with open('threads_${threads}.stats.json') as f:
    s = json.load(f)
    print(f'${threads} threads: {s[\"rss_per_thread_mean\"]/1024:.1f} MB/thread')
    "
done
```

Expected output:
```
1 threads: 450.0 MB/thread
2 threads: 240.0 MB/thread
4 threads: 130.0 MB/thread
8 threads: 70.0 MB/thread
```

## Memory Scaling Analysis

Per-thread RSS helps identify:

### Linear Scaling (Good)
```
Threads | Total RSS | Per-Thread RSS
   1    |  100 MB   |  100 MB
   2    |  200 MB   |  100 MB  ← Consistent
   4    |  400 MB   |  100 MB  ← Linear
   8    |  800 MB   |  100 MB  ← Ideal
```

### Sublinear Scaling (Common)
```
Threads | Total RSS | Per-Thread RSS
   1    |  100 MB   |  100 MB
   2    |  180 MB   |   90 MB  ← Shared overhead
   4    |  320 MB   |   80 MB  ← More sharing
   8    |  560 MB   |   70 MB  ← Economies of scale
```

### Superlinear Scaling (Problem)
```
Threads | Total RSS | Per-Thread RSS
   1    |  100 MB   |  100 MB
   2    |  240 MB   |  120 MB  ← Growing!
   4    |  560 MB   |  140 MB  ← Memory leak?
   8    | 1280 MB   |  160 MB  ← Critical
```

## Practical Examples

### Check if Memory is Thread-Local

```bash
# Run with different thread counts
./scripts/monitor_rss_simple.py -o t1 -- ./program --threads 1
./scripts/monitor_rss_simple.py -o t8 -- ./program --threads 8

# Compare per-thread mean
python3 << 'EOF'
import json

t1 = json.load(open('t1.stats.json'))
t8 = json.load(open('t8.stats.json'))

mean_t1 = t1['rss_per_thread_mean'] / 1024
mean_t8 = t8['rss_per_thread_mean'] / 1024

ratio = mean_t8 / mean_t1

if ratio < 0.5:
    print(f"✓ Good: Shared memory (8 threads use {ratio:.1%} per-thread vs 1 thread)")
elif ratio < 0.9:
    print(f"~ OK: Some sharing (8 threads use {ratio:.1%} per-thread vs 1 thread)")
else:
    print(f"✗ Bad: Thread-local memory (8 threads use {ratio:.1%} per-thread vs 1 thread)")
EOF
```

### Capacity Planning

```python
#!/usr/bin/env python3
import json

# Load statistics
with open('production.stats.json') as f:
    stats = json.load(f)

# Calculate resource needs
mean_per_thread_mb = stats['rss_per_thread_mean'] / 1024
p99_per_thread_mb = stats['rss_per_thread_max'] / 1024  # Use max as P99 proxy

# For production with 16 threads
threads = 16
estimated_mean = mean_per_thread_mb * threads
estimated_peak = p99_per_thread_mb * threads

print(f"Estimated resource needs for {threads} threads:")
print(f"  Mean RSS: {estimated_mean:.0f} MB")
print(f"  Peak RSS: {estimated_peak:.0f} MB")
print(f"  Recommended: {estimated_peak * 1.2:.0f} MB (20% safety margin)")
```

## Caveats

### Linux RSS Reporting

On Linux, `/proc/{pid}/status` reports the same RSS for all threads because RSS is process-wide. The tool still:
- Tracks thread count accurately
- Computes per-thread average (RSS / threads)
- Shows useful capacity planning metrics

### Shared Memory

Per-thread RSS doesn't account for:
- Shared libraries
- Shared memory segments
- Memory-mapped files

It's a **logical division** for capacity planning, not true per-thread physical memory.

---

**Use per-thread RSS for capacity planning and scaling analysis.**
