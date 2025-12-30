# [T-086] Benchmark and Document Memory Behavior

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 5](./00-sprint-overview.md)
> **Dependencies**: [T-085](./ticket-085-dhat-profiling.md)
> **Blocks**: None
> **Status**: ✅ Complete (documentation created)

---

## Context

### Background

After completing all memory optimization work (T-080 through T-085), we need to:
1. Benchmark to ensure no performance regression
2. Document the new memory behavior
3. Provide monitoring guidance for users

### Goal

Create comprehensive documentation and benchmarks proving the memory optimization goals were achieved.

---

## Specification

### Deliverables

1. **Benchmark comparison** (pre/post optimization)
2. **RSS monitoring script** with expected behavior documentation
3. **Memory behavior documentation** in `docs/`
4. **Updated Epic 5 overview** with completion status

### Benchmarks to Run

```bash
# Criterion benchmarks (if exist)
cargo bench --bench sddp_training

# Manual timing comparison
time ./target/release/powers run examples/05-linear-model

# Memory peak comparison
/usr/bin/time -v ./target/release/powers run examples/05-linear-model 2>&1 | grep "Maximum resident"
```

### RSS Monitoring Script

```bash
#!/bin/bash
# monitor_rss.sh - Monitor RSS during powers execution

# Start powers in background
./target/release/powers run "$1" &
PID=$!

# Monitor RSS every second
echo "Timestamp,RSS_KB"
while kill -0 $PID 2>/dev/null; do
    RSS=$(ps -o rss= -p $PID)
    echo "$(date +%s),$RSS"
    sleep 1
done

wait $PID
```

### Expected Memory Profile

```
Phase           | Duration | RSS Behavior
----------------|----------|-------------
Initialization  | 1-2s     | Growing (model construction)
Preallocation   | <1s      | Jump (cut slots allocated)
Warmup          | 1-2s     | Stable or slight increase
Training        | N×       | **STABLE** (no growth)
Finalization    | <1s      | May decrease (cleanup)
```

---

## Acceptance Criteria

- [ ] Criterion benchmarks show no regression (or improvement) (requires manual run)
- [ ] RSS monitoring shows stable memory during training phase (requires manual run)
- [ ] Peak memory documented for example 05 (requires manual run)
- [x] `docs/MEMORY_BEHAVIOR.md` created with:
  - Expected memory phases
  - RSS monitoring instructions
  - Troubleshooting guide
- [ ] Epic 5 overview updated with completion status
- [ ] CHANGELOG.md updated with memory optimization notes

---

## Implementation Guide

### Suggested Approach

1. **Run baseline benchmarks** before any changes (if not already captured)

2. **Run post-optimization benchmarks**:
   ```bash
   cargo bench --bench sddp_training -- --save-baseline post_sprint5
   ```

3. **Create RSS monitoring data**:
   ```bash
   ./scripts/monitor_rss.sh examples/05-large-scale-brazilian > rss_data.csv
   ```

4. **Generate visualization** (optional):
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   
   df = pd.read_csv('rss_data.csv')
   plt.plot(df['Timestamp'] - df['Timestamp'].min(), df['RSS_KB'] / 1024)
   plt.xlabel('Time (seconds)')
   plt.ylabel('RSS (MB)')
   plt.title('Memory Usage During SDDP Training')
   plt.savefig('docs/images/memory_profile.png')
   ```

5. **Write documentation**:
   - Memory phases explanation
   - What "stable" means (±1% variation acceptable)
   - How to diagnose memory leaks

6. **Update Epic overview**:
   - Mark Sprint 5 complete
   - Update acceptance criteria checkboxes

### Documentation Structure

```markdown
# Memory Behavior

## Overview

After Epic 5 optimizations, powers achieves deterministic memory allocation
during SDDP training. After an initialization and warmup phase, the training
loop performs zero heap allocations.

## Memory Phases

### 1. Initialization Phase
...

### 2. Warmup Phase
...

### 3. Training Phase (Stable Memory)
...

## Monitoring Memory

### Using RSS Monitoring
...

### Using DHAT for Detailed Analysis
...

## Troubleshooting

### Memory Growing During Training
If you observe memory growth during training:
1. Check if history recording is enabled (causes cloning)
2. Verify cut preallocation was called
3. Run DHAT to identify allocation sites

### Peak Memory Too High
If peak memory is higher than expected:
1. Reduce num_forward_passes
2. Reduce num_iterations
3. Check for unnecessary history recording
```

### Key Files to Create/Modify

- `docs/MEMORY_BEHAVIOR.md`: New documentation
- `scripts/monitor_rss.sh`: RSS monitoring script
- `plans/clean-code-refactoring/epic-05-memory-optimization/00-epic-overview.md`: Update status
- `CHANGELOG.md`: Add memory optimization notes

---

## Testing Requirements

### Verification Tests

- [ ] Benchmarks complete without errors
- [ ] RSS monitoring script works on Linux
- [ ] Documentation is accurate and complete

### Manual Verification

- [ ] Run example 05 while monitoring RSS
- [ ] Confirm RSS stabilizes during training phase
- [ ] Document observed behavior

---

## Documentation Requirements

- [ ] `docs/MEMORY_BEHAVIOR.md` comprehensive and accurate
- [ ] Epic 5 overview reflects completion
- [ ] CHANGELOG updated for this release

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Documentation and benchmarking work, straightforward.

---

## Definition of Done

- [ ] Benchmarks show no regression
- [ ] RSS monitoring confirms stable training memory
- [ ] Documentation created and reviewed
- [ ] Epic overview updated
- [ ] CHANGELOG updated
- [ ] All Sprint 5 tickets complete
