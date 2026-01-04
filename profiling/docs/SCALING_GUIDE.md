# Scaling Analysis for Multi-Socket Systems

This guide covers best practices for running parallel scaling analysis on high-core-count, multi-socket systems (e.g., AWS c7a.48xlarge with 192 cores).

## NUMA Considerations

### Understanding NUMA

Modern multi-socket systems use Non-Uniform Memory Access (NUMA) architecture where:
- Each CPU socket has local memory
- Accessing remote socket's memory is slower (higher latency, lower bandwidth)
- Thread placement affects performance significantly

### Checking NUMA Topology

```bash
# Install numactl
sudo apt install numactl

# View NUMA topology
numactl --hardware

# Example output:
# available: 2 nodes (0-1)
# node 0 cpus: 0 1 2 3 ... 95
# node 0 size: 384 GB
# node 1 cpus: 96 97 98 99 ... 191
# node 1 size: 384 GB
```

## Environment Variables

### Controlling Thread Count (Rayon)

```bash
# Set number of threads for Rayon thread pool
export RAYON_NUM_THREADS=16

# Run profiling
python -m powers_profile scaling --threads 16
```

The scaling command automatically sets `RAYON_NUM_THREADS` for each test iteration.

### Core Pinning

Pin threads to specific cores to avoid migration overhead:

```bash
# Pin to first 16 cores (NUMA node 0)
taskset -c 0-15 python -m powers_profile scaling --threads 1,2,4,8,16

# Pin to specific NUMA node
numactl --cpunodebind=0 --membind=0 python -m powers_profile scaling --threads 1,2,4,8,16

# Pin to both NUMA nodes (96 cores each on 2-socket system)
numactl --physcpubind=0-95,96-191 python -m powers_profile scaling --threads 1,2,4,8,16,32,64,96,128,192
```

### OpenMP Affinity (if using OpenMP instead of Rayon)

```bash
# Compact: pack threads on fewer cores
export GOMP_CPU_AFFINITY="0-15"

# Scatter: spread threads across sockets
export OMP_PROC_BIND=spread

# Places: define thread placement strategy
export OMP_PLACES=cores
```

## Recommended Thread Count Sets

### Development Machine (8-32 cores)
```bash
python -m powers_profile scaling --threads 1,2,4,8,16,24,32
```

### Production Server (96-192 cores)
```bash
# Full range with key inflection points
python -m powers_profile scaling --threads 1,2,4,8,16,32,48,64,96,128,192

# Focus on socket boundaries (2 sockets, 96 cores each)
python -m powers_profile scaling --threads 1,2,4,8,16,32,64,96,128,192

# Fine-grained around socket boundary
python -m powers_profile scaling --threads 80,88,96,104,112,120
```

**Rationale:**
- **1**: Baseline for speedup calculation
- **Powers of 2**: Common thread pool sizes, easy to reason about
- **Socket boundaries**: Detect NUMA effects (e.g., 96 threads on 2x96 system)
- **Hyperthreading**: Test physical cores vs. logical cores (e.g., 96 physical, 192 with HT)

## Best Practices

### 1. Isolate Benchmark from System Noise

```bash
# Disable CPU frequency scaling (requires root)
sudo cpupower frequency-set --governor performance

# Disable turbo boost (optional, for consistency)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Pin to specific NUMA node to avoid cross-socket traffic
numactl --cpunodebind=0 --membind=0 python -m powers_profile scaling
```

### 2. Increase Measurement Iterations

On large systems, timing variance can be higher:

```bash
# Use more iterations for statistical significance
python -m powers_profile scaling --iterations 5 --warmup 2
```

### 3. Monitor Contention

Enable contention detection to identify lock bottlenecks:

```bash
python -m powers_profile scaling --contention --threads 1,16,32,64,96,128,192
```

**Note:** Contention collection adds overhead but provides valuable insights at high thread counts.

### 4. Check for Efficiency Cliffs

Watch for sudden efficiency drops that indicate:
- **NUMA boundary crossed**: Efficiency drops when threads span multiple sockets
- **Hyperthreading saturation**: Diminishing returns beyond physical core count
- **Lock contention**: Synchronization overhead dominates at high thread counts

### 5. Timeout Considerations

High thread counts may take longer to initialize:

```bash
# Increase timeout for large thread counts
python -m powers_profile scaling --threads 128,192 --timeout 1200
```

## Interpreting Results on Multi-Socket Systems

### Expected Scaling Patterns

1. **Linear scaling (1-N cores on single socket)**
   - Efficiency: 90-100%
   - Example: 1→8 threads on 96-core socket

2. **NUMA slowdown (crossing socket boundary)**
   - Efficiency drop: 10-30%
   - Example: 64→128 threads on 2-socket system

3. **Hyperthreading benefit (N→2N logical cores)**
   - Speedup: 1.2-1.5x (not 2x)
   - Example: 96→192 threads with hyperthreading

4. **Contention plateau (beyond optimal thread count)**
   - Efficiency: <50%
   - Speedup stagnates or regresses

### Example: AWS c7a.48xlarge (2 sockets, 96 cores each, 192 logical)

```bash
python -m powers_profile scaling --threads 1,2,4,8,16,32,48,64,80,96,112,128,144,160,192
```

**Expected results:**
- **1-64 threads**: ~90% efficiency (within single socket)
- **96 threads**: ~80% efficiency (all physical cores on socket 0)
- **128 threads**: ~70% efficiency (spanning both sockets - NUMA penalty)
- **192 threads**: ~60% efficiency (hyperthreading saturation)

### Amdahl's Law Caveats

Amdahl estimation assumes:
- Uniform memory access (not true for NUMA)
- No lock contention (may appear at high thread counts)
- Infinite cores available (physical limits matter)

On multi-socket systems, use Amdahl estimate as **upper bound** for single-socket scaling.

## Troubleshooting

### Poor Scaling Beyond Socket Boundary

**Symptom:** Efficiency drops >30% when crossing socket boundary

**Solutions:**
1. Ensure NUMA-aware memory allocation in application code
2. Use `numactl --membind=all` to interleave memory across nodes
3. Profile with `perf c2c` to detect false sharing across sockets

### Contention Detection Fails

**Symptom:** `⚠️  perf not available or insufficient permissions`

**Solutions:**
```bash
# Enable perf without root (requires sysctl change)
sudo sysctl kernel.perf_event_paranoid=1

# Or run with sudo (not recommended for benchmarks)
sudo python -m powers_profile scaling --contention
```

### Inconsistent Timing

**Symptom:** High standard deviation in duration measurements

**Solutions:**
1. Pin to NUMA node: `numactl --cpunodebind=0`
2. Disable frequency scaling: `sudo cpupower frequency-set --governor performance`
3. Increase iterations: `--iterations 10`
4. Close background processes
5. Consider using `isolcpus` kernel parameter to reserve cores

## Example Workflows

### Basic Multi-Socket Scaling Test

```bash
# Test key thread counts with NUMA pinning
numactl --cpunodebind=0 --membind=0 \
  python -m powers_profile scaling --threads 1,2,4,8,16,32,64

numactl --cpunodebind=1 --membind=1 \
  python -m powers_profile scaling --threads 1,2,4,8,16,32,64

# Compare single-socket vs. cross-socket
python -m powers_profile scaling --threads 64,96,128
```

### Full Production Scaling Analysis

```bash
# Step 1: Set performance governor
sudo cpupower frequency-set --governor performance

# Step 2: Run comprehensive scaling test
python -m powers_profile scaling \
  --threads 1,2,4,8,16,32,48,64,80,96,112,128,144,160,176,192 \
  --iterations 5 \
  --warmup 2 \
  --contention

# Step 3: View results
python -m powers_profile summary latest

# Step 4: Extract key metrics
jq '.amdahl_estimate.serial_fraction' profiling_results/runs/latest/parallel/scaling_data.json
jq '.bottlenecks' profiling_results/runs/latest/parallel/scaling_data.json
```

### Debugging Scaling Issues

```bash
# Detailed contention analysis
python -m powers_profile scaling \
  --threads 64,96,128 \
  --contention \
  --iterations 10

# Check perf contention events
jq '.contention_analysis' profiling_results/runs/latest/parallel/scaling_data.json

# Look for high futex counts indicating lock contention
```

## References

- [NUMA Best Practices](https://queue.acm.org/detail.cfm?id=2513149)
- [Intel® 64 and IA-32 Performance Monitoring](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/overview-intel-vtune-profiler.html)
- [Linux perf Documentation](https://perf.wiki.kernel.org/index.php/Tutorial)
- [Rayon Thread Pool](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html)
