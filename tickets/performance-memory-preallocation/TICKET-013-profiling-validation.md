# TICKET-013: Profiling validation and final metrics collection

## Context

This ticket performs final profiling validation to confirm malloc overhead reduction and collect comprehensive performance metrics. It uses perf, massif, and flamegraphs to provide visual and quantitative evidence that the optimization goals were achieved. This is the final technical validation before documentation.

**Why this matters**: Benchmarks show execution time improvements, but profiling shows *why*—by verifying malloc overhead decreased, allocations were eliminated, and the hot paths are now allocation-free. This data completes the performance story and validates the implementation approach.

**Part of**: Performance Implementation Plan - Phase 4: Integration, Testing, and Validation

**Depends on**: TICKET-011, 012 (integration tests and benchmarks must pass)

## Acceptance Criteria

- [ ] Given profiling comparison, when malloc overhead is measured, then it's reduced from 5.28% to <2%
- [ ] Given allocation profiling, when hot paths are analyzed, then zero allocations are detected in loops
- [ ] Given flamegraph comparison, when visualized, then allocation overhead is visibly reduced
- [ ] Given memory profiling, when analyzed, then peak memory is stable and <2.5GB
- [ ] Given final metrics table, when compiled, then all target thresholds are met or exceeded
- [ ] All measurements are reproducible with documented methodology

## Tasks

### Implementation

- [ ] Create profiling script `scripts/profile_final_validation.sh`:
  - [ ] Run perf on baseline (main branch)
  - [ ] Run perf on optimized branch
  - [ ] Compare malloc/memset overhead
  - [ ] Generate side-by-side comparison
  - [ ] Save results with timestamps
- [ ] Create flamegraph generation script `scripts/generate_comparison_flamegraphs.sh`:
  - [ ] Generate flamegraph for baseline
  - [ ] Generate flamegraph for optimized
  - [ ] Create diff flamegraph
  - [ ] Highlight allocation reduction
- [ ] Create memory profiling script `scripts/profile_memory_final.sh`:
  - [ ] Run massif on optimized version
  - [ ] Analyze allocation patterns
  - [ ] Verify zero allocations in hot paths
  - [ ] Generate allocation timeline
- [ ] Create metrics collection script `scripts/collect_final_metrics.sh`:
  - [ ] Collect runtime measurements
  - [ ] Collect malloc overhead percentages
  - [ ] Collect memory usage statistics
  - [ ] Collect allocation counts
  - [ ] Export to JSON/CSV for analysis
- [ ] Create comprehensive comparison script `scripts/generate_performance_comparison.sh`:
  - [ ] Run all profiling tools
  - [ ] Generate all reports
  - [ ] Create summary comparison table
  - [ ] Generate visual comparisons

### Profiling - CPU and Allocation Overhead

- [ ] Profile baseline (main branch):
  - [ ] Checkout main branch
  - [ ] Build with debug info: `CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release`
  - [ ] Run: `perf record --call-graph dwarf -F 999 ./target/release/powers examples/05-large-scale-brazilian`
  - [ ] Generate report: `perf report --stdio > baseline_perf_report.txt`
  - [ ] Save to `profiling_results/final_validation_baseline/`
  - [ ] Extract malloc overhead: `grep -E "(malloc|_int_malloc|memset)" | awk '{sum+=$1} END {print sum}'`
  - [ ] Extract total runtime
- [ ] Profile optimized (feature branch):
  - [ ] Checkout feature/memory-optimization branch
  - [ ] Build with debug info
  - [ ] Run perf with same parameters
  - [ ] Generate report: `perf report --stdio > optimized_perf_report.txt`
  - [ ] Save to `profiling_results/final_validation_optimized/`
  - [ ] Extract malloc overhead
  - [ ] Extract total runtime
- [ ] Compare profiling results:
  - [ ] Calculate malloc overhead reduction
  - [ ] Calculate memset overhead reduction
  - [ ] Calculate total runtime improvement
  - [ ] Verify malloc overhead <2% (target achieved)
  - [ ] Verify runtime improvement >10% (minimum) or >15% (goal)
- [ ] Generate comparison report:
  - [ ] Side-by-side function time comparison
  - [ ] Overhead percentage comparison
  - [ ] Runtime improvement calculation
  - [ ] Statistical analysis of differences

### Profiling - Flamegraph Visualization

- [ ] Generate baseline flamegraph:
  - [ ] Convert perf data: `perf script > baseline.perf`
  - [ ] Collapse stacks: `stackcollapse-perf.pl baseline.perf > baseline.folded`
  - [ ] Generate SVG: `flamegraph.pl baseline.folded > baseline_flamegraph.svg`
- [ ] Generate optimized flamegraph:
  - [ ] Same process for optimized version
  - [ ] Save as `optimized_flamegraph.svg`
- [ ] Generate diff flamegraph:
  - [ ] Use differential flamegraph tool
  - [ ] Highlight areas with reduced overhead
  - [ ] Save as `diff_flamegraph.svg`
- [ ] Annotate flamegraphs:
  - [ ] Highlight malloc/memset regions (should be smaller in optimized)
  - [ ] Highlight backward/forward pass regions
  - [ ] Add annotations for key improvements

### Profiling - Memory and Allocation Analysis

- [ ] Memory profiling with massif:
  - [ ] Run: `valgrind --tool=massif --detailed-freq=1 --massif-out-file=optimized.massif ./target/release/powers examples/05-large-scale-brazilian`
  - [ ] Generate report: `ms_print optimized.massif > optimized_massif_report.txt`
  - [ ] Analyze peak memory usage
  - [ ] Analyze allocation patterns
  - [ ] Verify no memory leaks
- [ ] Allocation counting:
  - [ ] Use massif or custom allocator
  - [ ] Count allocations in hot paths
  - [ ] Compare with baseline
  - [ ] Verify zero allocations in loops (backward pass, forward pass)
- [ ] Memory timeline analysis:
  - [ ] Plot memory usage over time
  - [ ] Verify memory stabilizes
  - [ ] Check for any gradual growth
  - [ ] Document peak usage

### Metrics Collection

- [ ] Collect runtime metrics:
  - [ ] Total training time (baseline vs optimized)
  - [ ] Backward pass time (baseline vs optimized)
  - [ ] Forward pass time (baseline vs optimized)
  - [ ] Average iteration time
- [ ] Collect overhead metrics:
  - [ ] Malloc CPU overhead % (baseline vs optimized)
  - [ ] Memset CPU overhead % (baseline vs optimized)
  - [ ] Total allocation overhead % (baseline vs optimized)
- [ ] Collect allocation metrics:
  - [ ] Total allocation count (baseline vs optimized)
  - [ ] Allocations per iteration (baseline vs optimized)
  - [ ] Allocations in backward pass (baseline vs optimized)
  - [ ] Allocations in forward pass (baseline vs optimized)
- [ ] Collect memory metrics:
  - [ ] Peak memory usage (baseline vs optimized)
  - [ ] Average memory usage
  - [ ] Memory growth rate (should be zero)
- [ ] Compile final metrics table:
  - [ ] All metrics in structured format
  - [ ] Baseline vs optimized comparison
  - [ ] Improvement percentages
  - [ ] Target achievement status

### Documentation

- [ ] Create `PROFILING_VALIDATION_REPORT.md`:
  - [ ] Executive summary
  - [ ] Profiling methodology
  - [ ] CPU profiling results
  - [ ] Memory profiling results
  - [ ] Flamegraph analysis
  - [ ] Metrics comparison table
  - [ ] Conclusions
- [ ] Document profiling methodology:
  - [ ] Tools and versions used
  - [ ] Commands executed
  - [ ] System configuration
  - [ ] Reproduction instructions
- [ ] Create final metrics document `FINAL_PERFORMANCE_METRICS.md`:
  - [ ] Comprehensive metrics table
  - [ ] Visual comparison charts
  - [ ] Achievement vs target analysis
  - [ ] Key insights and learnings
- [ ] Update PERFORMANCE_REFACTORING_PLAN.md:
  - [ ] Mark all phases complete
  - [ ] Update all metrics tables with final numbers
  - [ ] Add validation complete status
  - [ ] Document lessons learned
- [ ] Add profiling evidence to documentation:
  - [ ] Include flamegraph comparisons
  - [ ] Include massif reports
  - [ ] Include metric summaries

## Technical Notes

### Profiling Commands Reference

**CPU profiling with perf**:
```bash
# Record with call graph
perf record --call-graph dwarf -F 999 -o perf.data ./target/release/powers examples/05-large-scale-brazilian

# Generate report
perf report --stdio -i perf.data > perf_report.txt

# Extract specific functions
perf report --stdio -i perf.data | grep -A10 "backward_pass"

# Calculate malloc overhead
grep -E "(_int_malloc|__libc_malloc|malloc[^_])" perf_report.txt | awk '{sum+=$1} END {print "malloc: " sum "%"}'
grep -E "(__memset|memset[^_])" perf_report.txt | awk '{sum+=$1} END {print "memset: " sum "%"}'
```

**Memory profiling with massif**:
```bash
# Run massif
valgrind --tool=massif \
  --detailed-freq=1 \
  --max-snapshots=1000 \
  --massif-out-file=massif.out \
  ./target/release/powers examples/05-large-scale-brazilian

# Generate report
ms_print massif.out > massif_report.txt

# Extract peak memory
grep "peak" massif_report.txt

# Find allocation hotspots
grep -A20 "snapshot=" massif_report.txt | grep -E "->.*%"
```

**Flamegraph generation**:
```bash
# Prerequisites: Install flamegraph tools
git clone https://github.com/brendangregg/FlameGraph
export PATH=$PATH:$PWD/FlameGraph

# Generate flamegraph
perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# Generate differential flamegraph
./FlameGraph/difffolded.pl baseline.folded optimized.folded | \
  flamegraph.pl --negate --title="Optimization Impact" > diff_flamegraph.svg
```

### Expected Profiling Results

Based on PERFORMANCE_IMPLEMENTATION_PLAN.md:

**Profiling Metrics**:
| Metric | Baseline | Target | Expected |
|--------|----------|--------|----------|
| Runtime | 34.0s | <29s | 28-30s |
| Malloc % | 5.28% | <2% | 1.5-1.8% |
| Memset % | 1.78% | <0.5% | 0.6-0.8% |
| Total alloc overhead | 7.06% | <2.5% | 2.0-2.5% |

**Allocation Counts**:
| Location | Baseline | Optimized | Reduction |
|----------|----------|-----------|-----------|
| Backward pass | ~60/iter | ~1/iter | ~98% |
| Forward pass | ~50/iter | ~10/iter | ~80% |
| Subproblem | ~30/solve | ~5/solve | ~83% |
| Overall | ~140/iter | ~15/iter | ~89% |

### Metrics Table Format

```markdown
# Final Performance Metrics

## Overall Performance

| Metric | Baseline | Optimized | Improvement | Target | Status |
|--------|----------|-----------|-------------|--------|--------|
| Training time | 34.0s | 28.8s | 15.3% | >10% | ✅ Exceeded |
| Malloc overhead | 5.28% | 1.7% | 67.8% | <2% | ✅ Achieved |
| Memory peak | 2.4GB | 2.5GB | +4% | <2.6GB | ✅ Within |

## Component Performance

| Component | Baseline | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Backward pass | 4.7s | 4.05s | 13.8% |
| Forward pass | 3.2s | 2.86s | 10.6% |
| Subproblem | 0.5s/solve | 0.48s/solve | 4.0% |

## Allocation Metrics

| Metric | Baseline | Optimized | Reduction |
|--------|----------|-----------|-----------|
| Allocs/iteration | 140 | 15 | 89% |
| Malloc CPU % | 3.34% | 0.9% | 73% |
| Memset CPU % | 1.78% | 0.7% | 61% |
```

### Flamegraph Analysis

**What to look for**:
1. **Malloc tower** should be significantly shorter in optimized version
2. **Backward pass** should show more solver time, less allocation time
3. **Forward pass** should show more computation, less vector growth
4. **Diff flamegraph** should show red (reduction) in allocation areas

**Annotations to add**:
- "Malloc overhead: 5.28% → 1.7%" in comparison
- "Backward pass allocation eliminated" pointing to relevant stacks
- "Forward pass buffer reuse" highlighting optimization

### Validation Thresholds

**Pass Criteria** (must achieve):
- ✅ Malloc overhead <2.5%
- ✅ Runtime improvement >10%
- ✅ No memory leaks (valgrind clean)
- ✅ No correctness regressions

**Target Criteria** (stretch goals):
- 🎯 Malloc overhead <2%
- 🎯 Runtime improvement >15%
- 🎯 Memory peak <2.5GB

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 4.3
- perf documentation: https://perf.wiki.kernel.org/
- massif documentation: https://valgrind.org/docs/manual/ms-manual.html
- flamegraph guide: http://www.brendangregg.com/flamegraphs.html

## Dependencies

- Blocked by: TICKET-011 (integration tests)
- Blocked by: TICKET-012 (benchmarks)
- Blocks: TICKET-014 (final documentation uses these metrics)

## Estimated Effort

**2 story points** (1 day)

**Confidence**: High

**Breakdown**:
- Running profiling tools: 0.25 day (automated scripts)
- Analysis: 0.25 day (interpreting results)
- Flamegraph generation: 0.25 day (visualization)
- Documentation: 0.25 day (reports and metrics)

## Validation Checklist

Before marking this ticket complete:

- [ ] Baseline profiling complete
- [ ] Optimized profiling complete
- [ ] Malloc overhead measured and <2%
- [ ] Runtime improvement measured and >10%
- [ ] Flamegraphs generated and analyzed
- [ ] Memory profiling complete (no leaks)
- [ ] Allocation counts verified (>80% reduction)
- [ ] Final metrics table compiled
- [ ] All target thresholds met
- [ ] PROFILING_VALIDATION_REPORT.md created
- [ ] FINAL_PERFORMANCE_METRICS.md created
- [ ] PERFORMANCE_REFACTORING_PLAN.md updated
- [ ] Results validated by team member

## Notes

**Visual Evidence Matters**: Flamegraphs provide compelling visual proof of optimization impact. Include them in documentation and presentations.

**Reproducibility**: Document system configuration carefully. Profiling results vary by hardware, so clear documentation enables others to reproduce results on their systems.

**Celebrate Achievement**: If profiling validates we exceeded targets (>15% improvement, <2% malloc), celebrate the data-driven success!

**Archive Results**: Save all profiling data with timestamps. These become historical baselines for future optimization work.

**Share Insights**: Profiling often reveals unexpected insights (e.g., "HiGHS dominates runtime" or "memset is surprisingly expensive"). Document these for future reference.
