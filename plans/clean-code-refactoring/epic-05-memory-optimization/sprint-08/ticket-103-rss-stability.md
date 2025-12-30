# [T-103] RSS Stability Verification During Training

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8: Validation and Documentation](./00-sprint-overview.md)
> **Dependencies**: Sprint 7 complete
> **Blocks**: T-105

---

## Context

### Background

One of Epic 5's success criteria is stable RSS (Resident Set Size) after the warmup phase. This ticket verifies that memory behavior by monitoring RSS during training.

### Expected Behavior

```
RSS Timeline:
1. Initialization: RSS grows as data structures are created
2. Warmup: RSS stabilizes as all buffers are preallocated
3. Training: RSS remains flat (no allocations in hot path)
4. Cleanup: RSS may decrease as results are written
```

## Specification

### Tasks

1. **Create RSS monitoring script**
2. **Run training with monitoring**
3. **Analyze RSS timeline**
4. **Verify stability criteria**

### Success Criteria

- RSS variance during training ≤ 2% of peak
- No upward trend during training iterations
- Clear plateau after warmup phase

## Acceptance Criteria

- [ ] RSS monitoring completed for example 05
- [ ] Timeline graph generated
- [ ] Stability criteria verified
- [ ] Results documented

## Implementation Guide

### Suggested Approach

1. **Create monitoring script**:
   ```bash
   #!/bin/bash
   # scripts/monitor_rss.sh
   
   OUTPUT_FILE="rss_timeline.csv"
   echo "timestamp_ms,rss_kb" > $OUTPUT_FILE
   
   # Start powers in background
   ./target/release/powers run examples/05-large-scale-brazilian &
   PID=$!
   
   START_TIME=$(date +%s%3N)
   
   while kill -0 $PID 2>/dev/null; do
       CURRENT_TIME=$(date +%s%3N)
       ELAPSED=$((CURRENT_TIME - START_TIME))
       RSS=$(ps -o rss= -p $PID 2>/dev/null || echo "0")
       echo "${ELAPSED},${RSS}" >> $OUTPUT_FILE
       sleep 0.5
   done
   
   echo "Monitoring complete. Data in $OUTPUT_FILE"
   ```

2. **Create analysis script**:
   ```python
   # scripts/analyze_rss.py
   
   import pandas as pd
   import matplotlib.pyplot as plt
   
   df = pd.read_csv('rss_timeline.csv')
   
   # Find warmup end (where RSS stabilizes)
   # Calculate training phase variance
   # Generate report
   ```

3. **Generate visualization**:
   ```python
   plt.figure(figsize=(12, 6))
   plt.plot(df['timestamp_ms'] / 1000, df['rss_kb'] / 1024, 'b-')
   plt.xlabel('Time (seconds)')
   plt.ylabel('RSS (MB)')
   plt.title('Memory Usage During SDDP Training')
   plt.axvline(x=warmup_end, color='r', linestyle='--', label='Warmup End')
   plt.legend()
   plt.savefig('docs/rss_timeline.png')
   ```

4. **Verify stability**:
   ```python
   training_rss = df[df['timestamp_ms'] > warmup_end_ms]['rss_kb']
   variance = training_rss.std() / training_rss.mean() * 100
   
   if variance <= 2.0:
       print(f"✅ RSS stable: {variance:.2f}% variance")
   else:
       print(f"❌ RSS unstable: {variance:.2f}% variance")
   ```

## Testing Requirements

- [ ] Monitoring script works on Linux
- [ ] Analysis produces valid results
- [ ] Visualization is clear

## Documentation Requirements

- [ ] Add RSS timeline graph to `docs/MEMORY_BEHAVIOR.md`
- [ ] Document stability criteria and results

## Effort Estimate

**Points**: 2
**Confidence**: High
