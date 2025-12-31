# [T-120] RSS Verification Tests

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-117
> **Blocks**: T-122
> **Priority**: 3 (Validation)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `docs/HIGHS_RSS_MEMORY_INVESTIGATION.md` - RSS growth analysis
- T-117 implementation

---

## Context

### Background

The primary goal of this architecture is to reclaim HiGHS memory between iterations. This test verifies that RSS (Resident Set Size) decreases or stabilizes after `finalize_iteration()`.

---

## Specification

### Test Design

```rust
#[test]
fn test_rss_decreases_between_iterations() {
    let mut algorithm = create_test_algorithm();
    
    // Warm up
    algorithm.create_iteration_models(true).unwrap();
    algorithm.finalize_iteration(true);
    
    let rss_baseline = get_current_rss();
    
    // Run several iterations, checking RSS
    let mut rss_after_iteration = Vec::new();
    let mut rss_after_finalize = Vec::new();
    
    for _ in 0..10 {
        algorithm.create_iteration_models(true).unwrap();
        // ... forward/backward passes ...
        rss_after_iteration.push(get_current_rss());
        
        algorithm.finalize_iteration(true);
        rss_after_finalize.push(get_current_rss());
    }
    
    // Verify RSS doesn't grow monotonically
    // After finalize should be similar to baseline
    for (i, &rss) in rss_after_finalize.iter().enumerate() {
        let growth = (rss as f64 - rss_baseline as f64) / rss_baseline as f64;
        assert!(
            growth < 0.5,  // Allow 50% growth margin
            "Iteration {}: RSS grew by {:.1}% (expected stable)",
            i, growth * 100.0
        );
    }
    
    // RSS should decrease after finalize
    for (rss_during, rss_after) in rss_after_iteration.iter().zip(&rss_after_finalize) {
        assert!(
            rss_after <= rss_during,
            "RSS should decrease after finalize: {} -> {}",
            rss_during, rss_after
        );
    }
}

/// Get current process RSS in bytes.
fn get_current_rss() -> usize {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let status = fs::read_to_string("/proc/self/statm").unwrap();
        let pages: usize = status.split_whitespace().nth(1).unwrap().parse().unwrap();
        pages * 4096  // Page size typically 4KB
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        0  // Skip on non-Linux
    }
}
```

### Expected Behavior

```
Iteration 1: 
  - Create models: RSS ↑ (HiGHS allocates)
  - Finalize: RSS ↓ (HiGHS freed)

Iteration 2:
  - Create models: RSS ↑ (new HiGHS allocates)
  - Finalize: RSS ↓ (HiGHS freed)

Pattern: Sawtooth, with peaks at iteration and valleys at finalize
Overall: Stable between iterations (no monotonic growth)
```

---

## Acceptance Criteria

- [ ] Test measures RSS before/after operations
- [ ] RSS decreases after `finalize_iteration()`
- [ ] No monotonic RSS growth across iterations
- [ ] Test passes on Linux (skip on other platforms)
- [ ] Documents actual RSS behavior

---

## Testing Notes

### Platform Considerations

- Linux: Use `/proc/self/statm` for RSS
- macOS: Use `mach_task_basic_info`
- Windows: Use `GetProcessMemoryInfo`

For this ticket, Linux-only is acceptable; other platforms can skip.

### Measurement Accuracy

RSS may not immediately reflect freed memory due to:
- Allocator holding pages
- Memory mapped regions
- Lazy page reclamation

Use relaxed assertions (allow 50% margin).

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Platform-specific memory measurement

---

## Definition of Done

- [ ] RSS measurement implemented
- [ ] Test verifies decrease after finalize
- [ ] Test documents findings
- [ ] PR merged
