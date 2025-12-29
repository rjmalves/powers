# [T-002] Capture Baseline Benchmarks

> **Epic**: [Epic 1: Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Infrastructure Setup](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None (but provides data for all future performance validation)

---

## ⚠️ CRITICAL: Performance Baseline

This ticket captures the **performance baseline** against which all future changes will be measured. A >5% regression from this baseline will block any PR during the refactoring.

Document the numbers carefully—they will be referenced throughout the project.

---

## Files to Read Before Starting

- `benches/` - Existing benchmark files
- `Cargo.toml` - Benchmark configuration
- `README.md` - Current performance claims
- `examples/05-large-scale-brazilian` - Large-scale example for realistic benchmarks
- `MEMORY_GROWTH_ANALYSIS.md` - Existing memory analysis documentation
- `REMAINING_ALLOCATIONS_ANALYSIS.md` - Allocation hotspot documentation

---

## Context

### Background

We need quantitative performance data before making any changes. This baseline will:
1. Detect performance regressions during refactoring
2. Validate our performance improvement claims at the end
3. Identify current bottlenecks for targeting

**Key benchmark**: The `sddp_e2e` benchmark is the most important benchmark for this refactoring. It uses example 05 (large-scale-brazilian) for time measurements.

### Current State

- Criterion benchmarks exist in `benches/`
- Performance characteristics are not formally documented
- Memory usage patterns are partially documented but need baseline capture
- **No memory-focused benchmarks exist** for detailed allocation analysis

---

## Specification

### Inputs

- Existing benchmarks in `benches/`
- Example cases of varying sizes
- Current release build

### Outputs

1. **Baseline document**: `docs/PERFORMANCE_BASELINE.md`
2. **Criterion baseline**: Saved criterion data for comparison
3. **Memory profile data**: Peak RSS and allocation patterns for key examples
4. **Memory analysis benchmarks**: New criterion benchmarks for memory profiling

### Metrics to Capture

#### Time Benchmarks

For the `sddp_e2e` benchmark (primary):
- **Mean execution time** with confidence interval
- **Iterations per second**
- **Per-phase breakdown** (if available): forward, backward, cut management

For a full training run (`05-large-scale-brazilian`):
- **Total training time**
- **Per-iteration time** (forward + backward)
- **Example 05 expected runtime**: ~2 minutes

#### Memory Benchmarks (NEW)

Create memory-focused benchmarks using example 05:

| Metric | Measurement Method |
|--------|-------------------|
| Peak RSS | `/usr/bin/time -v` |
| Allocation count per iteration | DHAT profiling |
| Allocation bytes per iteration | DHAT profiling |
| Memory growth pattern | RSS sampling during run |
| Hot path allocations | DHAT + flamegraph |

### Behavior

- Run each benchmark with criterion's default sample count
- Save baseline with `--save-baseline before-refactoring`
- Document results in markdown table format
- Include system information (CPU, RAM, OS) for reproducibility

---

## Acceptance Criteria

- [ ] `docs/PERFORMANCE_BASELINE.md` created with all metrics
- [ ] Criterion baseline saved as `before-refactoring`
- [ ] At least 3 benchmark runs show consistent results (within CI)
- [ ] System information documented
- [ ] Memory usage documented for example 05
- [ ] Instructions for reproducing benchmarks documented
- [ ] **Memory analysis benchmarks added** for example 05

### Correctness Verification

- [ ] No changes to any source code (except adding memory benchmarks)
- [ ] Only documentation and benchmark data added

---

## Implementation Guide

### Suggested Approach

1. **Run existing benchmarks**:
   ```bash
   cargo bench --bench sddp_e2e -- --save-baseline before-refactoring
   ```

2. **Capture detailed timing for large example** (expect ~2 minutes):
   ```bash
   # Time a full training run
   time ./target/release/powers run examples/05-large-scale-brazilian
   ```

3. **Measure memory usage**:
   ```bash
   # Using /usr/bin/time for peak RSS
   /usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | grep "Maximum resident set size"
   ```

4. **Profile allocations with DHAT** (if available):
   ```bash
   # Build with DHAT instrumentation
   RUSTFLAGS="-C target-cpu=native" cargo build --release --features dhat-heap
   
   # Run with DHAT
   ./target/release/powers run examples/05-large-scale-brazilian
   # Analyze dhat-heap.json output
   ```

5. **Create memory analysis benchmarks** (`benches/memory_analysis.rs`):
   ```rust
   //! Memory analysis benchmarks for tracking allocation patterns
   //! 
   //! These benchmarks focus on memory behavior rather than just speed.
   //! Use with DHAT or heaptrack for detailed allocation profiling.
   
   use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
   
   fn memory_benchmark(c: &mut Criterion) {
       let mut group = c.benchmark_group("memory_analysis");
       
       // Benchmark single iteration memory pattern
       group.bench_function("single_iteration_05", |b| {
           // Setup: load example 05
           // Benchmark: run single training iteration
           // This isolates per-iteration allocation patterns
       });
       
       // Benchmark forward pass allocations
       group.bench_function("forward_pass_05", |b| {
           // Isolate forward pass memory behavior
       });
       
       // Benchmark backward pass allocations  
       group.bench_function("backward_pass_05", |b| {
           // Isolate backward pass memory behavior
       });
       
       // Benchmark cut management allocations
       group.bench_function("cut_management_05", |b| {
           // Isolate cut creation/storage patterns
       });
       
       group.finish();
   }
   
   criterion_group!(benches, memory_benchmark);
   criterion_main!(benches);
   ```

6. **Create baseline document**:
   ```markdown
   # Performance Baseline
   
   Captured: YYYY-MM-DD
   Commit: <commit-hash>
   
   ## System Information
   - CPU: <model>
   - RAM: <size>
   - OS: <version>
   - Rust: <version>
   
   ## Time Benchmarks
   
   ### Criterion Benchmarks (sddp_e2e)
   
   | Benchmark | Mean | Std Dev | Throughput |
   |-----------|------|---------|------------|
   | sddp_e2e/05-large-scale | X.XX s | ±Y.YY ms | ZZZ iter/s |
   
   ### Full Training Run (05-large-scale-brazilian)
   
   | Metric | Value |
   |--------|-------|
   | Total time | ~2 min |
   | Iterations | N |
   | Time per iteration | X.XX s |
   
   ## Memory Benchmarks
   
   ### Peak Memory Usage
   
   | Example | Peak RSS | Notes |
   |---------|----------|-------|
   | 05-large-scale | XXX MB | Full training run |
   
   ### Allocation Analysis (DHAT)
   
   | Metric | Value |
   |--------|-------|
   | Total allocations | X,XXX,XXX |
   | Total bytes allocated | XXX MB |
   | Allocations per iteration | ~X,XXX |
   | Bytes per iteration | ~X MB |
   
   ### Hot Path Allocations
   
   | Location | Allocations | Bytes | Notes |
   |----------|-------------|-------|-------|
   | `Box<dyn State>` cloning | X,XXX | XX MB | Per cut creation |
   | `Vec<f64>` in solution extraction | X,XXX | XX MB | Per solve |
   | Realization allocation | X,XXX | XX MB | Per forward pass |
   
   ## Baseline Saved
   
   Criterion baseline saved as: `before-refactoring`
   
   To compare after changes:
   ```bash
   cargo bench -- --baseline before-refactoring
   ```
   ```

7. **Verify consistency**:
   ```bash
   # Run 3 times, check variance
   for i in {1..3}; do
     echo "=== Run $i ==="
     cargo bench --bench sddp_e2e 2>&1 | tee bench-run-$i.txt
   done
   ```

### Key Files to Create/Modify

- `docs/PERFORMANCE_BASELINE.md` - Main documentation
- `benches/memory_analysis.rs` - Memory-focused benchmarks (NEW)
- Criterion baseline data (in `target/criterion/`)

### Pitfalls to Avoid

- ⚠️ Don't run benchmarks on battery power or with other processes running
- ⚠️ Don't skip warm-up runs—criterion handles this, but manual tests need it
- ⚠️ Don't forget to document the exact commit hash
- ⚠️ Don't modify any algorithm source code in this ticket
- ⚠️ **Example 05 takes ~2 minutes**—don't assume it's fast
- ⚠️ Don't skip memory profiling—it's essential for Epic 5

---

## Testing Requirements

### Verification

- [ ] Benchmark results are consistent across 3 runs (within 5% variance)
- [ ] Baseline can be loaded for comparison: `cargo bench -- --baseline before-refactoring`
- [ ] Memory measurement is reproducible
- [ ] Memory benchmarks compile and run

---

## Documentation Requirements

- [ ] Create `docs/PERFORMANCE_BASELINE.md`
- [ ] Document benchmark reproduction steps
- [ ] Document system requirements for accurate comparison
- [ ] Document expected runtimes (especially 2 min for example 05)
- [ ] Document memory profiling methodology
- [ ] Add note to main README about baseline

---

## Effort Estimate

**Points**: 3 (increased from 2 due to memory benchmark requirements)
**Confidence**: High
**Rationale**: Running benchmarks and documenting results is straightforward; adding memory benchmarks adds some complexity

---

## Definition of Done

- [ ] Baseline document created with all metrics (time AND memory)
- [ ] Criterion baseline saved
- [ ] Memory analysis benchmarks created
- [ ] Results verified consistent across multiple runs
- [ ] Documentation complete
- [ ] No algorithm source code modified
