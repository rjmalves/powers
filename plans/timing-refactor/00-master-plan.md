# Master Plan: Timing Infrastructure Refactor

## Executive Summary

This refactor consolidates ~15 scattered timing structs into a unified, hierarchical timing system that cleanly separates measurement from business logic. The new architecture uses RAII-based `TimingGuard` throughout, eliminates manual `Instant::now()`/`.elapsed()` patterns, and provides explicit parallel overhead tracking without redistributing measured values.

## Goals & Non-Goals

### Goals

1. **Zero pollution**: Business logic in `sddp/mod.rs` training loop should have NO visible timing code
2. **Single source of truth**: One canonical set of timing types in `src/timing/`
3. **Preserve precision**: Never redistribute or approximate timing values
4. **Explicit parallel overhead**: Track wall-time vs CPU-time difference separately
5. **Clean naming**: Remove redundant `_time` suffixes, use consistent hierarchical naming
6. **Store raw data**: Keep all trajectory timings for statistical analysis (avg, max, percentiles)

### Non-Goals (Explicit Scope Exclusions)

- **Feature-gated compilation**: Not prioritized (always-on timing is acceptable)
- **Zero-cost when disabled**: Not required since timing is always enabled
- **Backward compatibility**: Old timing struct schemas can be replaced entirely
- **Per-stage detailed breakdown**: Deferred to future work (secondary goal)

---

## Architecture Overview

### Current State

**Problem**: 15+ timing structs scattered across the codebase with:

- Duplicate definitions (e.g., `BackwardPassTimingAccumulator` in both `sddp/mod.rs` and `algorithm/backward_pass.rs`)
- Manual `Instant::now()` / `.elapsed()` patterns polluting the training loop
- Rescaling logic that redistributes timing values (violates precision goal)
- Inconsistent naming (`_time` suffix, flat hierarchy)

**Current Timing Structs** (to be removed/consolidated):

| Location                     | Struct                          | Status                                  |
| ---------------------------- | ------------------------------- | --------------------------------------- |
| `sddp/mod.rs`                | `ForwardPassTiming`             | → Replace with `timing::ForwardTiming`  |
| `sddp/mod.rs`                | `BackwardPassTiming`            | → Replace with `timing::BackwardTiming` |
| `sddp/mod.rs`                | `ForwardPassTimingAccumulator`  | → Remove (merge into new system)        |
| `sddp/mod.rs`                | `BackwardPassTimingAccumulator` | → Remove (duplicate)                    |
| `sddp/mod.rs`                | `BackwardPhase1Timing`          | → Replace with `timing::Phase1Timing`   |
| `sddp/mod.rs`                | `BranchingsTiming`              | → Remove (internal detail)              |
| `sddp/mod.rs`                | `StepTiming` (private)          | → Remove (use guards)                   |
| `algorithm/context.rs`       | `TrajectoryTiming`              | → Move to `timing/`                     |
| `algorithm/context.rs`       | `BackwardStageTiming`           | → Remove (merge)                        |
| `algorithm/backward_pass.rs` | `BackwardPassTimingAccumulator` | → Keep, rename to `BackwardTiming`      |
| `algorithm/backward_pass.rs` | `BackwardPassTimingSnapshot`    | → Remove (use direct struct)            |
| `algorithm/processor.rs`     | `CutComputationTiming`          | → Replace with `timing::Phase1Timing`   |
| `algorithm/processor.rs`     | `FirstStageTiming`              | → Remove (internal)                     |
| `algorithm/forward_pass.rs`  | `StepTiming`                    | → Remove (use guards)                   |
| `subproblem.rs`              | `RealizeUncertaintiesTiming`    | → Keep (solver-level detail)            |

### Target State

```
src/timing/
├── mod.rs              # Public exports
├── guard.rs            # TimingGuard (RAII) - already exists
├── iteration.rs        # IterationTiming (top-level)
├── forward.rs          # ForwardTiming hierarchy
├── backward.rs         # BackwardTiming hierarchy
├── trajectory.rs       # TrajectoryTiming (per-trajectory)
├── output.rs           # Output conversion types (plain Duration)
└── aggregation.rs      # Statistical aggregation utilities
```

**Key Design Decision**: All timing flows through a single `IterationTiming` struct created at iteration start, passed down to sub-operations, and converted to output format at iteration end.

---

## Proposed Timing Schema

### Hierarchy Design Principles

1. **Hierarchical naming**: Use nested structs for phases (e.g., `forward.preprocessing.saa_sampling`)
2. **No `_time` suffix**: Field names are already in a timing context
3. **Explicit parallel sections**: Separate `wall` (external) vs `cpu` (internal sum) timing
4. **Raw trajectory storage**: Keep `Vec<TrajectoryTiming>` for per-trajectory analysis (internal only, not exposed in output)
5. **Two-tier structs**: Accumulators with `Cell<Duration>` during execution, plain `Duration` for output

### New Timing Structs (Accumulators - used during execution)

#### `IterationTiming` (Top-Level)

```rust
/// Complete timing for one SDDP training iteration.
///
/// This is the single entry point - created at iteration start,
/// passed to all sub-operations, converted to output at end.
pub struct IterationTiming {
    /// Time to create solver Models from cached Problems.
    pub model_allocation: Cell<Duration>,

    /// Forward pass timing.
    pub forward: ForwardTiming,

    /// Backward pass timing.
    pub backward: BackwardTiming,

    /// Time to cleanup solver Models at iteration end.
    pub model_cleanup: Cell<Duration>,

    /// Total iteration wall-clock time.
    pub total: Cell<Duration>,
}
```

#### `ForwardTiming` (Forward Pass)

```rust
/// Forward pass timing with hierarchical structure.
///
/// The forward pass has three phases:
/// 1. Preprocessing (sequential): SAA sampling
/// 2. Parallel execution: N trajectories solving in parallel
/// 3. Postprocessing (sequential): Result aggregation
pub struct ForwardTiming {
    /// Sequential preprocessing phase.
    pub preprocessing: ForwardPreprocessingTiming,

    /// Parallel trajectory execution.
    pub parallel: ForwardParallelTiming,

    /// Sequential postprocessing phase.
    pub postprocessing: ForwardPostprocessingTiming,

    /// Total forward pass time (wall clock).
    pub total: Cell<Duration>,
}

/// Forward pass preprocessing (sequential, before parallel section).
pub struct ForwardPreprocessingTiming {
    /// Time spent sampling scenarios from SAA tree.
    pub saa_sampling: Cell<Duration>,
}

/// Forward pass parallel section timing.
///
/// CRITICAL: We store BOTH wall time and individual trajectory times.
/// - `wall`: What we observe from outside the parallel section
/// - `trajectories`: Raw per-trajectory timing for statistical analysis (internal only)
/// - `cpu_total`: Sum of all trajectory CPU times (computed)
/// - `overhead`: wall - cpu_total (parallel scheduling overhead)
pub struct ForwardParallelTiming {
    /// Wall-clock time for entire parallel section.
    pub wall: Cell<Duration>,

    /// Individual trajectory timings (preallocated, length = num_forward_passes).
    /// Contains raw timing data for internal statistical analysis.
    /// NOT exposed in output - only aggregated stats are exported.
    pub trajectories: Vec<TrajectoryTiming>,

    // Computed fields (populated after parallel section completes):

    /// Sum of all trajectory CPU times.
    pub cpu_total: Cell<Duration>,

    /// Parallel overhead: wall - cpu_total.
    pub overhead: Cell<Duration>,

    // Aggregated statistics (populated after parallel section):

    /// Average model preprocessing time per trajectory.
    pub model_preprocessing_avg: Cell<Duration>,

    /// Average solver time per trajectory.
    pub solver_avg: Cell<Duration>,

    /// Average model postprocessing time per trajectory.
    pub model_postprocessing_avg: Cell<Duration>,

    /// Maximum solver time across trajectories (for load balance analysis).
    pub solver_max: Cell<Duration>,
}

/// Forward pass postprocessing (sequential, after parallel section).
pub struct ForwardPostprocessingTiming {
    /// Time spent capturing trajectory details when requested.
    pub detail_capturing: Cell<Duration>,
}

/// Per-trajectory timing collected during parallel forward pass.
///
/// Uses Cell<Duration> for interior mutability with TimingGuard.
pub struct TrajectoryTiming {
    /// Time preparing the subproblem model (state injection, cut updates).
    pub model_preprocessing: Cell<Duration>,

    /// Time in LP solver.
    pub solver: Cell<Duration>,

    /// Time extracting solution (primal/dual values, state update).
    pub model_postprocessing: Cell<Duration>,

    /// Number of solver calls in this trajectory.
    pub solver_calls: Cell<usize>,
}

impl TrajectoryTiming {
    /// Total CPU time for this trajectory.
    pub fn cpu_time(&self) -> Duration {
        self.model_preprocessing.get()
            + self.solver.get()
            + self.model_postprocessing.get()
    }
}
```

#### `BackwardTiming` (Backward Pass)

```rust
/// Backward pass timing with per-phase breakdown.
///
/// The backward pass iterates stages in reverse order:
/// - For each stage: Phase 1 (cut computation) → Phase 2 (cut selection) → Phase 3 (model updates)
pub struct BackwardTiming {
    /// Phase 1: Parallel cut computation (summed across stages).
    pub phase1: BackwardPhase1Timing,

    /// Phase 2: Sequential cut selection (summed across stages).
    pub phase2: BackwardPhase2Timing,

    /// Phase 3: Parallel problem update (summed across stages).
    pub phase3: BackwardPhase3Timing,

    /// Total backward pass time (wall clock).
    pub total: Cell<Duration>,

    /// Total solver calls across all stages.
    pub solver_calls: Cell<usize>,
}

/// Backward Phase 1: Parallel cut computation.
///
/// For each stage, we solve branchings in parallel across handlers.
pub struct BackwardPhase1Timing {
    /// Model preprocessing time (state injection, noise realization).
    pub model_preprocessing: Cell<Duration>,

    /// LP solver time.
    pub solver: Cell<Duration>,

    /// Model postprocessing time (cut coefficient extraction).
    pub model_postprocessing: Cell<Duration>,

    /// Cut aggregation and risk measure application.
    pub cut_computation: Cell<Duration>,
}

/// Backward Phase 2: Sequential cut selection.
pub struct BackwardPhase2Timing {
    /// Time spent in cut selection algorithm.
    pub cut_selection: Cell<Duration>,
}

/// Backward Phase 3: Parallel problem update.
pub struct BackwardPhase3Timing {
    /// Time applying cuts to handler models and problems.
    pub problem_update: Cell<Duration>,
}
```

#### `TrainingTiming` (Full Training Run)

```rust
/// Timing for the complete training run.
pub struct TrainingTiming {
    /// Preprocessing before iterations begin (graph construction, etc.).
    pub preprocessing: Cell<Duration>,

    /// Per-iteration timing (length = num_iterations).
    pub iterations: Vec<IterationTiming>,

    /// Postprocessing after iterations complete.
    pub postprocessing: Cell<Duration>,

    /// Total training time.
    pub total: Cell<Duration>,
}
```

---

## Output Schema Mapping

The new timing structs map to output fields as follows. Note that output structs use plain `Duration` (not `Cell<Duration>`).

### Forward Pass Output

| Output Field                      | New Source                                    |
| --------------------------------- | --------------------------------------------- |
| `forward_saa_sampling_ms`         | `forward.preprocessing.saa_sampling`          |
| `forward_model_preprocessing_ms`  | `forward.parallel.model_preprocessing_avg`    |
| `forward_solver_ms`               | `forward.parallel.solver_avg`                 |
| `forward_model_postprocessing_ms` | `forward.parallel.model_postprocessing_avg`   |
| `forward_postprocessing_ms`       | `forward.postprocessing.detail_capturing`     |
| `forward_total_ms`                | `forward.total`                               |

### Backward Pass Output

| Output Field                       | New Source                            |
| ---------------------------------- | ------------------------------------- |
| `backward_model_preprocessing_ms`  | `backward.phase1.model_preprocessing` |
| `backward_solver_ms`               | `backward.phase1.solver`              |
| `backward_model_postprocessing_ms` | `backward.phase1.model_postprocessing`|
| `backward_cut_selection_ms`        | `backward.phase2.cut_selection`       |
| `backward_problem_update_ms`       | `backward.phase3.problem_update`      |
| `backward_total_ms`                | `backward.total`                      |

**Removed Fields** (previously existed, now removed):
- `backward_preprocessing_ms` - Was misnamed, functionality absorbed into phase1
- `backward_fcf_state_update_ms` - Merged into `problem_update`
- `backward_cut_cloning_ms` - Merged into `problem_update`
- `backward_handler_application_ms` - Merged into `problem_update`

### New Output Fields (Optional Additions)

| New Field                      | Source                        | Purpose                           |
| ------------------------------ | ----------------------------- | --------------------------------- |
| `forward_parallel_wall_ms`     | `forward.parallel.wall`       | External view of parallel section |
| `forward_parallel_overhead_ms` | `forward.parallel.overhead`   | Scheduling overhead               |
| `forward_solver_max_ms`        | `forward.parallel.solver_max` | Load balance analysis             |
| `model_allocation_ms`          | `iteration.model_allocation`  | Per-iteration model creation      |
| `model_cleanup_ms`             | `iteration.model_cleanup`     | Per-iteration model cleanup       |

---

## Algorithm Hot Path Mapping

### Forward Pass Execution Flow

```
train() {
    for iteration in 0..num_iterations {
        let timing = IterationTiming::new(num_forward_passes);  // Preallocate

        // 1. Model allocation
        {
            let _guard = TimingGuard::new(&timing.model_allocation);
            for handler in handlers { handler.create_iteration_models(); }
        }

        // 2. Forward preprocessing
        {
            let _guard = TimingGuard::new(&timing.forward.preprocessing.saa_sampling);
            sample_scenarios();
        }

        // 3. Forward parallel section
        {
            let _guard = TimingGuard::new(&timing.forward.parallel.wall);
            handlers.par_iter_mut()
                .zip(timing.forward.parallel.trajectories.par_iter())
                .for_each(|(handler, traj_timing)| {
                    forward_trajectory(handler, traj_timing);
                });
        }
        timing.forward.parallel.compute_aggregates();  // avg, max, overhead

        // 4. Forward postprocessing
        {
            let _guard = TimingGuard::new(&timing.forward.postprocessing.detail_capturing);
            capture_trajectory_details_if_enabled();
        }
        timing.forward.compute_total();

        // 5. Backward pass
        {
            let _guard = TimingGuard::new(&timing.backward.total);
            backward_pass(&timing.backward);
        }

        // 6. Model cleanup
        {
            let _guard = TimingGuard::new(&timing.model_cleanup);
            for handler in handlers { handler.finalize_iteration(); }
        }

        timing.compute_total();
        iterations.push(timing.to_output());
    }
}
```

### Per-Trajectory Forward Pass

```
forward_trajectory(handler, timing: &TrajectoryTiming) {
    for stage in study_periods {
        // Model prep: state injection, cut retrieval
        {
            let _guard = TimingGuard::new(&timing.model_preprocessing);
            subproblem.prepare_from_trajectory(past_realizations);
        }

        // Solver: LP solve (timing comes from realize_and_solve internally)
        let solve_timing = subproblem.realize_and_solve(innovations, realization);
        timing.solver.set(timing.solver.get() + solve_timing.solver_time);
        timing.model_postprocessing.set(
            timing.model_postprocessing.get() + solve_timing.state_extraction_time
        );
        timing.solver_calls.set(timing.solver_calls.get() + 1);
    }
}
```

### Backward Pass Execution Flow

```
backward_pass(timing: &BackwardTiming) {
    for stage in study_periods.rev() {
        // Phase 1: Parallel cut computation
        {
            // Guards inside coordinator.compute_cuts_phase1()
            coordinator.compute_cuts_phase1(&timing.phase1);
        }

        // Phase 2: Sequential cut selection
        {
            let _guard = TimingGuard::new(&timing.phase2.cut_selection);
            cut_selector.select_cuts();
        }

        // Phase 3: Parallel problem update
        {
            let _guard = TimingGuard::new(&timing.phase3.problem_update);
            handlers.par_iter_mut().for_each(|h| h.apply_cuts_and_update());
        }
    }
}
```

---

## Data Flow

```
                    ┌─────────────────────────────────────┐
                    │         IterationTiming             │
                    │  (created at iteration start)       │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
   │ ForwardTiming │      │BackwardTiming │      │ model_alloc/  │
   │               │      │               │      │ model_cleanup │
   └───────┬───────┘      └───────┬───────┘      └───────────────┘
           │                      │
     ┌─────┴─────┐          ┌─────┴─────┐
     ▼           ▼          ▼           ▼
 ┌───────┐  ┌────────┐  ┌────────┐  ┌────────┐
 │preproc│  │parallel│  │ phase1 │  │phase2/3│
 └───────┘  └───┬────┘  └────────┘  └────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   ┌─────────┐    ┌─────────┐
   │Traj[0]  │... │Traj[N-1]│  (preallocated Vec)
   └─────────┘    └─────────┘
```

At iteration end:

1. `ForwardParallelTiming::compute_aggregates()` calculates avg/max/overhead
2. `IterationTiming::to_output()` converts to output format (plain Duration) for writers

---

## Phases & Milestones

| Phase | Epic                      | Duration | Milestone                                          |
| ----- | ------------------------- | -------- | -------------------------------------------------- |
| 1     | Core Timing Types         | 1 week   | New timing structs in `src/timing/` with tests     |
| 2     | Forward Pass Integration  | 1 week   | Forward pass uses new timing, legacy removed       |
| 3     | Backward Pass Integration | 1 week   | Backward pass uses new timing, legacy removed      |
| 4     | Training Loop Cleanup     | 1 week   | `sddp/mod.rs` training loop has zero timing code   |
| 5     | Output Adaptation         | 0.5 week | Output writers use new timing, CSV/Parquet updated |
| 6     | Cleanup & Documentation   | 0.5 week | Remove dead code, update docs                      |

**Total Estimated Duration**: 5 weeks

---

## Risk Analysis

| Risk                                       | Likelihood | Impact | Mitigation                                            |
| ------------------------------------------ | ---------- | ------ | ----------------------------------------------------- |
| Borrow checker conflicts with TimingGuard  | Medium     | High   | Already solved pattern in `algorithm/forward_pass.rs` |
| Performance regression from Cell access    | Low        | Medium | Cell<Duration> is zero-cost for Copy types            |
| Output format mismatch breaking downstream | Medium     | High   | Map new fields explicitly to old output schema        |
| Missed timing in parallel sections         | Medium     | Medium | Review all `par_iter` calls, ensure guards present    |
| Thread-safety issues with trajectory Vec   | Low        | High   | Use preallocated Vec with indexed access, no resize   |

---

## Success Metrics

- [ ] **Zero Instant::now()** in `sddp/mod.rs` (grep verification)
- [ ] **Single timing module**: All timing types in `src/timing/`
- [ ] **No timing redistribution**: Remove the "recalibrate" scaling code
- [ ] **Test coverage**: All new timing types have unit tests
- [ ] **Output compatibility**: CSV/Parquet output field names preserved where possible
- [ ] **Benchmark regression**: No measurable performance impact

---

## Clarifications Resolved

### Q1: Trajectory Timing Storage
**Decision**: Store all trajectory timings internally for analysis, but only expose aggregated stats in output (avg, max). No per-trajectory CSV columns.

### Q2: Backward "Preprocessing" Field
**Decision**: Remove `backward_preprocessing_ms` from output (was misnamed).

### Q3: Naming Convention
**Decision**: Use hierarchical struct nesting (e.g., `timing.forward.parallel.solver_avg`).

### Q4: Cell vs Direct Duration
**Decision**: Accumulators use `Cell<Duration>`, output structs use plain `Duration`. Add `to_output()` conversion methods.

---

## Next Steps

With master plan approved, proceeding to:

1. Create Epic breakdowns with detailed scope
2. Create Sprint plans with ticket sequencing
3. Generate atomic implementation tickets
