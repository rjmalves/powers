# [T-107] Create Memory Monitoring Guide for Users

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8: Validation and Documentation](./00-sprint-overview.md)
> **Dependencies**: T-105
> **Blocks**: None

---

## Context

### Background

Create user-facing documentation for monitoring and troubleshooting memory behavior in production deployments.

## Specification

### Content Requirements

1. **Quick Start** - How to monitor memory
2. **Expected Behavior** - What normal looks like
3. **Warning Signs** - When to investigate
4. **Troubleshooting** - Common issues and solutions
5. **Advanced** - DHAT profiling for deep analysis

### Target Audience

- Operations engineers deploying POWE.RS
- Developers integrating POWE.RS
- Users running large-scale studies

## Acceptance Criteria

- [ ] Guide created at `docs/MEMORY_MONITORING_GUIDE.md`
- [ ] Examples for common monitoring tools
- [ ] Troubleshooting section with solutions
- [ ] References to technical docs

## Implementation Guide

### Suggested Document Structure

```markdown
# Memory Monitoring Guide

## Quick Start

Monitor memory usage during POWE.RS training:

```bash
# Linux - Monitor RSS
./target/release/powers run study.json &
watch -n 1 'ps -o pid,rss,vsz,comm -p $(pgrep powers)'

# With memory limit
systemd-run --scope -p MemoryMax=4G ./target/release/powers run study.json
```

## Expected Memory Behavior

### Memory Phases

1. **Initialization (0-5 sec)**: Memory grows as model loads
2. **Warmup (5-10 sec)**: Pools preallocated, memory stabilizes
3. **Training**: Memory should be **flat** with ≤2% variance
4. **Cleanup**: Memory may spike briefly for output

### Typical Memory Usage

| Study Size | Expected Peak | Notes |
|------------|---------------|-------|
| Small (5 stages) | 200-400 MB | |
| Medium (30 stages) | 500 MB - 1 GB | |
| Large (60+ stages) | 1-4 GB | Depends on forward passes |

## Warning Signs

### 🔴 Memory Growing During Training

If RSS increases during training iterations:
- Pool may be undersized
- HiGHS warmup may have failed
- Bug in optimization code

**Action**: Run DHAT profiling to identify source

### 🟡 Higher Than Expected Peak

If peak memory exceeds expectations:
- Reduce `--forward-passes`
- Reduce `--iterations` for warmup
- Check for large scenario trees

### 🟢 Memory Spike at End

Normal - output files being written

## Troubleshooting

### Problem: Memory Grows Linearly

**Symptom**: RSS increases each iteration

**Cause**: Allocation in hot path not using preallocated buffers

**Solution**: 
1. Run DHAT profiling
2. Check allocation is in powers code, not HiGHS
3. Report as bug if in production path

### Problem: Out of Memory

**Symptom**: Process killed by OOM

**Cause**: Pool preallocation too large

**Solution**:
1. Reduce iteration count
2. Reduce forward pass count
3. Increase system memory

## Advanced: DHAT Profiling

For detailed allocation analysis:

```bash
# Build with debug symbols
cargo build --release

# Run with DHAT
valgrind --tool=dhat ./target/release/powers run study.json

# Analyze output
python3 scripts/analyze_dhat.py dhat.out
```

Key metrics:
- `tb`: Total bytes allocated
- `tbk`: Number of allocations
- `mb`: Peak bytes at any time

## Reference

- [Memory Behavior Technical Details](./MEMORY_BEHAVIOR.md)
- [Allocation Audit Report](./HOT_PATH_ALLOCATION_AUDIT.md)
```

## Testing Requirements

- [ ] Commands in guide work
- [ ] Links are valid

## Effort Estimate

**Points**: 2
**Confidence**: High
