# Epic 4: Parallelism & Scalability Analysis

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: ⬜ Not Started

---

## Summary

This epic implements parallel scalability analysis to measure how POWE.RS performance scales with thread count. It enables developers to identify parallelization efficiency, contention points, and optimal thread configurations for different problem sizes.

---

## Scope

### Included

1. **Thread Scaling Analysis**
   - Automated runs at different thread counts (1, 2, 4, 8, 16, 32, ...)
   - Wall-clock time measurement
   - Speedup calculation (time_1 / time_n)
   - Efficiency calculation (speedup / n_threads)

2. **Scaling Metrics**
   - Strong scaling (fixed problem, vary threads)
   - Amdahl's law estimation
   - Parallel overhead detection

3. **Contention Analysis**
   - Lock contention via perf (futex events)
   - Thread synchronization overhead
   - Load imbalance detection

4. **Multi-Socket Preparation**
   - NUMA-aware benchmarking structure
   - Core pinning support
   - Preparation for c7a.48xlarge (192 cores)

5. **Parallel Collector**
   - JSON output of scaling data
   - Efficiency tables and curves

### Excluded

- GPU parallelism
- Distributed (MPI) profiling
- Dynamic load balancing optimization

---

## Dependencies

- **Requires**: Epic 1 (Core Framework), Epic 2 (CPU Profiling for perf)
- **Enables**: Epic 5 (Visualization - scaling charts)

---

## Acceptance Criteria

- [ ] `powers-profile scaling --threads 1,2,4,8,16` runs automated scaling test
- [ ] Speedup and efficiency calculated for each thread count
- [ ] Scaling data in JSON format
- [ ] Amdahl's law estimate derived from data
- [ ] Contention events captured (when perf available)
- [ ] Works up to 192 threads (for future use)
- [ ] CLI summary shows scaling table

---

## Technical Approach

### Scaling Test Workflow

```
powers-profile scaling --threads 1,2,4,8,16,32
        │
        ▼
┌─────────────────────────────────────────────────┐
│ For each thread_count in [1, 2, 4, 8, 16, 32]:  │
│                                                  │
│   1. Set RAYON_NUM_THREADS=thread_count         │
│   2. Run warmup iteration                        │
│   3. Run timed iterations (3-5x)                │
│   4. Record mean, std dev, min, max             │
│   5. Optionally collect perf data               │
│                                                  │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│ Analysis:                                        │
│   - Calculate speedup: T(1) / T(n)              │
│   - Calculate efficiency: speedup / n           │
│   - Estimate Amdahl serial fraction             │
│   - Identify scaling bottlenecks                │
└─────────────────────────────────────────────────┘
        │
        ▼
    scaling_data.json
```

### Key Metrics

| Metric | Formula | Ideal |
|--------|---------|-------|
| Speedup | T(1) / T(n) | n |
| Efficiency | Speedup / n | 1.0 (100%) |
| Serial Fraction (Amdahl) | (1/speedup - 1/n) / (1 - 1/n) | 0 |

### Environment Variables

```bash
# Control Rayon thread pool
export RAYON_NUM_THREADS=8

# (Future) NUMA control
export GOMP_CPU_AFFINITY="0-7"  # Pin to first 8 cores
```

---

## Sprints

### [Sprint 1: Scalability Analysis](./sprint-01/00-sprint-overview.md)

| ID | Title | Points | Status |
|----|-------|--------|--------|
| T-026 | Implement scaling test runner | 5 | ⬜ |
| T-027 | Implement speedup/efficiency calculation | 3 | ⬜ |
| T-028 | Implement Amdahl estimation | 3 | ⬜ |
| T-029 | Implement contention detection | 5 | ⬜ |
| T-030 | Create parallel collector | 3 | ⬜ |
| T-031 | Add scaling CLI summary | 2 | ⬜ |
| T-032 | Document multi-socket preparation | 2 | ⬜ |

**Sprint Points**: 23

---

## Estimated Effort

- **Duration**: 1 sprint (2 weeks)
- **Story Points**: 23
- **Risk Level**: Medium (requires multiple runs, timing variability)

---

## Definition of Done

- [ ] All tickets complete
- [ ] Scaling analysis works for 1-32 threads
- [ ] JSON output with speedup/efficiency
- [ ] CLI shows scaling table
- [ ] Amdahl estimation working
- [ ] Documentation for high-core-count systems
