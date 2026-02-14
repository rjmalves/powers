---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §5.4.9 (Thread-Local Solver Infrastructure)"
  - "DATA_MODEL_SPECIFICATION.md §5.5 (5.5.1-5.5.5)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from DATA_MODEL_SPECIFICATION.md §5.4.9, §5.5.1-5.5.5"
---

# Solver Workspaces & LP Scaling

## Purpose

This spec defines the thread-local solver workspace infrastructure and the LP scaling specification. These components bridge the [solver abstraction layer](./solver-abstraction.md) with the HPC execution layer, ensuring each OpenMP thread has an exclusive, NUMA-local solver instance with pre-allocated buffers for the SDDP hot path. For the HiGHS reference implementation, see [HiGHS Implementation](./solver-highs-impl.md).

## 1. Thread-Local Solver Infrastructure

> **Design Rationale**: LP solvers (HiGHS, CLP, CPLEX) are **NOT thread-safe**. Each OpenMP thread requires its own solver instance. This section specifies the thread-local workspace pattern that: (1) eliminates contention via exclusive ownership, (2) enables warm-starting with persistent basis, (3) avoids hot-loop allocations with thread-local buffers, and (4) uses NUMA-aware allocation for memory bandwidth.
>
> **Key Insight**: Unlike a pool-based approach (borrow/return), we use **persistent ownership**. Each thread keeps its solver for the entire SDDP run, eliminating synchronization overhead and enabling optimal warm-starting.

### 1.1 Workspace Structure

```rust
/// Complete workspace for one OpenMP thread's LP solving
///
/// Lifetime: Created once at startup, persists for entire SDDP run.
/// Ownership: Exclusively owned by one thread (no sharing).
///
/// Memory Layout (for 1120-state problem):
///   - Solver instance: ~15 MB (HiGHS with pre-allocated cuts)
///   - RHS buffers: ~100 KB
///   - Solution storage: ~100 KB
///   - Basis storage: ~50 KB
///   - Total per thread: ~16 MB
///   - Total for 192 threads: ~3 GB
#[repr(C)]  // Predictable layout for NUMA allocation
pub struct ThreadSolverWorkspace {
    /// The solver instance (HiGHS, CLP, etc.)
    ///
    /// IMPORTANT: This solver instance is NEVER shared between threads.
    /// It persists for the entire SDDP run, enabling warm-starting.
    pub solver: ActiveSolver,

    /// Thread-local RHS buffer for stage problem updates
    ///
    /// During forward pass, we update RHS values (inflows, demands, state)
    /// without modifying the shared LpProblem. This buffer holds the
    /// thread-specific RHS values passed to solver.update_and_solve().
    pub rhs_buffer: Vec<f64>,

    /// Thread-local solution storage
    ///
    /// Avoids allocation when extracting solution from solver.
    pub primal_solution: Vec<f64>,
    pub dual_solution: Vec<f64>,
    pub reduced_costs: Vec<f64>,

    /// Cached basis from last solve (for warm-starting)
    ///
    /// When solving same stage with different RHS (common in forward pass),
    /// the basis is typically valid and warm-starting is very effective.
    pub cached_basis: Option<Basis>,

    /// Last stage solved (for basis validity tracking)
    ///
    /// If current stage != last_stage, basis may not be applicable.
    pub last_stage: Option<u32>,

    /// Statistics accumulated by this thread
    pub stats: ThreadSolverStats,

    /// NUMA node this workspace is allocated on
    pub numa_node: u32,

    /// Cache line padding to prevent false sharing between workspaces
    _padding: [u8; 64],
}

/// Statistics tracked per thread
#[derive(Debug, Default, Clone)]
pub struct ThreadSolverStats {
    pub solves: u64,
    pub warm_starts: u64,
    pub cold_starts: u64,
    pub retries: u64,
    pub simplex_iterations: u64,
    pub total_solve_time: f64,
    pub max_solve_time: f64,
}
```

### 1.2 NUMA-Aware Initialization

```rust
impl ThreadSolverWorkspace {
    /// Create workspace with NUMA-aware allocation
    ///
    /// MUST be called from the thread that will own this workspace
    /// (first-touch policy ensures memory is allocated on local NUMA node).
    pub fn new_numa_local(
        config: &SolverConfig,
        problem_dimensions: &ProblemDimensions,
        numa_node: u32,
    ) -> Self {
        // Pin to NUMA node during allocation
        let _guard = numa_bind_guard(numa_node);

        // Create solver (allocates internal working memory)
        let solver = ActiveSolver::new(config.clone());

        // Pre-allocate buffers with first-touch initialization
        let num_rows = problem_dimensions.num_constraints;
        let num_cols = problem_dimensions.num_variables;

        let mut rhs_buffer = vec![0.0; num_rows];
        let mut primal_solution = vec![0.0; num_cols];
        let mut dual_solution = vec![0.0; num_rows];
        let mut reduced_costs = vec![0.0; num_cols];

        // First-touch to ensure NUMA-local allocation
        for v in rhs_buffer.iter_mut() { *v = 0.0; }
        for v in primal_solution.iter_mut() { *v = 0.0; }
        for v in dual_solution.iter_mut() { *v = 0.0; }
        for v in reduced_costs.iter_mut() { *v = 0.0; }

        Self {
            solver,
            rhs_buffer,
            primal_solution,
            dual_solution,
            reduced_costs,
            cached_basis: None,
            last_stage: None,
            stats: ThreadSolverStats::default(),
            numa_node,
            _padding: [0u8; 64],
        }
    }
}
```

### 1.3 Stage Solve with Automatic Warm-Starting

```rust
impl ThreadSolverWorkspace {
    /// Solve stage LP with automatic warm-starting
    ///
    /// This is the main entry point for forward/backward pass solving.
    /// Handles basis caching and warm-start decisions internally.
    pub fn solve_stage(
        &mut self,
        stage: u32,
        problem: &LpProblem,
        rhs_updates: &[(usize, f64)],  // (row_index, value) pairs
    ) -> Result<StageSolution, SolverError> {
        let start = std::time::Instant::now();

        // Prepare RHS buffer with updates
        self.prepare_rhs(problem, rhs_updates);

        // Determine warm-start eligibility
        let use_warm_start = self.can_warm_start(stage);

        // Solve
        let result = if use_warm_start {
            self.stats.warm_starts += 1;
            let basis = self.cached_basis.as_ref().unwrap();
            self.solver.solve_with_basis(problem, basis)
        } else {
            self.stats.cold_starts += 1;
            self.solver.solve(problem)
        };

        // Update statistics
        let elapsed = start.elapsed().as_secs_f64();
        self.stats.solves += 1;
        self.stats.total_solve_time += elapsed;
        self.stats.max_solve_time = self.stats.max_solve_time.max(elapsed);

        // Handle result
        match result {
            Ok(solution) => {
                if let Some(basis) = solution.basis.clone() {
                    self.cached_basis = Some(basis);
                    self.last_stage = Some(stage);
                }
                self.extract_solution(&solution);
                self.stats.simplex_iterations += solution.simplex_iterations;
                Ok(StageSolution::from_lp_solution(solution))
            }
            Err(e) => {
                self.cached_basis = None;
                self.last_stage = None;
                self.stats.retries += 1;
                Err(e)
            }
        }
    }

    /// Check if warm-starting is likely beneficial
    fn can_warm_start(&self, stage: u32) -> bool {
        match (&self.cached_basis, self.last_stage) {
            (Some(_), Some(last)) => {
                // Same stage: basis is highly likely to be valid
                // Adjacent stage: basis may still be useful
                // Different stage: cold start is safer
                stage == last || stage.abs_diff(last) <= 1
            }
            _ => false,
        }
    }

    /// Prepare RHS buffer with scenario-specific updates
    fn prepare_rhs(&mut self, problem: &LpProblem, updates: &[(usize, f64)]) {
        self.rhs_buffer.copy_from_slice(&problem.row_rhs);
        for &(row, value) in updates {
            self.rhs_buffer[row] = value;
        }
    }

    /// Extract solution into thread-local storage
    fn extract_solution(&mut self, solution: &LpSolution) {
        self.primal_solution.copy_from_slice(&solution.primal);
        self.dual_solution.copy_from_slice(&solution.dual);
        if let Some(rc) = &solution.reduced_costs {
            self.reduced_costs.copy_from_slice(rc);
        }
    }

    /// Merge statistics from multiple workspaces (called after parallel region)
    pub fn merge_statistics(workspaces: &[ThreadSolverWorkspace]) -> ThreadSolverStats {
        let mut merged = ThreadSolverStats::default();
        for ws in workspaces {
            merged.solves += ws.stats.solves;
            merged.warm_starts += ws.stats.warm_starts;
            merged.cold_starts += ws.stats.cold_starts;
            merged.retries += ws.stats.retries;
            merged.simplex_iterations += ws.stats.simplex_iterations;
            merged.total_solve_time += ws.stats.total_solve_time;
            merged.max_solve_time = merged.max_solve_time.max(ws.stats.max_solve_time);
        }
        merged
    }
}
```

### 1.4 Workspace Manager

````rust
/// Manager for all thread-local workspaces in a rank
///
/// Creates and owns one workspace per OpenMP thread.
/// Provides access pattern that OpenMP threads use via thread ID.
pub struct WorkspaceManager {
    /// One workspace per thread, indexed by thread ID
    ///
    /// IMPORTANT: This Vec is never modified after initialization.
    /// Threads access their workspace by index: workspaces[omp_get_thread_num()]
    workspaces: Vec<ThreadSolverWorkspace>,

    /// Number of NUMA nodes on this system
    num_numa_nodes: u32,

    /// Threads per NUMA node
    threads_per_numa: u32,
}

impl WorkspaceManager {
    /// Initialize all workspaces with NUMA-aware allocation
    ///
    /// MUST be called from an OpenMP parallel region where each thread
    /// initializes its own workspace (first-touch policy).
    ///
    /// ```rust
    /// // Initialization pattern (pseudo-code for OpenMP)
    /// let mut workspaces = Vec::with_capacity(num_threads);
    ///
    /// #[omp_parallel]
    /// {
    ///     let tid = omp_get_thread_num();
    ///     let numa = tid / threads_per_numa;
    ///
    ///     let ws = ThreadSolverWorkspace::new_numa_local(&config, &dims, numa);
    ///
    ///     #[omp_critical]
    ///     workspaces.push((tid, ws));
    /// }
    ///
    /// // Sort by thread ID and extract workspaces
    /// workspaces.sort_by_key(|(tid, _)| *tid);
    /// let workspaces: Vec<_> = workspaces.into_iter().map(|(_, ws)| ws).collect();
    /// ```
    pub fn new(
        num_threads: usize,
        num_numa_nodes: u32,
        config: &SolverConfig,
        dims: &ProblemDimensions,
    ) -> Self {
        let threads_per_numa = (num_threads as u32 + num_numa_nodes - 1) / num_numa_nodes;

        let workspaces = (0..num_threads)
            .map(|tid| {
                let numa = (tid as u32) / threads_per_numa;
                ThreadSolverWorkspace::new_numa_local(config, dims, numa)
            })
            .collect();

        Self {
            workspaces,
            num_numa_nodes,
            threads_per_numa,
        }
    }

    /// Get workspace for current thread (called from OpenMP parallel region)
    #[inline]
    pub fn get(&self, thread_id: usize) -> &ThreadSolverWorkspace {
        &self.workspaces[thread_id]
    }

    /// Get mutable workspace for current thread
    #[inline]
    pub fn get_mut(&mut self, thread_id: usize) -> &mut ThreadSolverWorkspace {
        &mut self.workspaces[thread_id]
    }

    /// Get aggregated statistics across all threads
    pub fn aggregate_statistics(&self) -> ThreadSolverStats {
        ThreadSolverWorkspace::merge_statistics(&self.workspaces)
    }

    /// Number of workspaces (equals number of threads)
    pub fn len(&self) -> usize {
        self.workspaces.len()
    }
}
````

### 1.5 Thread Safety Invariants for LpProblem

> **Critical Design Rule**: The `LpProblem` struct is treated as **read-only during parallel forward pass**. All scenario-specific modifications (RHS values) go through thread-local `rhs_buffer` in `ThreadSolverWorkspace`.

| Operation                | Thread Safety                      | When                                 |
| ------------------------ | ---------------------------------- | ------------------------------------ |
| Read LP structure        | ✅ Safe (immutable)                | Forward pass (parallel)              |
| Read cut coefficients    | ✅ Safe (immutable during forward) | Forward pass (parallel)              |
| Update RHS for scenario  | ✅ Thread-local buffer             | Forward pass (parallel)              |
| Enable/disable cuts      | ❌ Single-threaded only            | Backward pass (sequential per stage) |
| Add new cut coefficients | ❌ Single-threaded only            | Backward pass (sequential per stage) |

````rust
/// LpProblem thread-safety documentation
impl LpProblem {
    /// Access pattern during forward pass:
    ///
    /// ```text
    /// Forward Pass (PARALLEL):
    ///   - LpProblem is READ-ONLY (shared reference)
    ///   - RHS updates go to thread-local buffer
    ///   - Solver reads from LpProblem + thread-local RHS
    ///
    /// Backward Pass (SEQUENTIAL per stage):
    ///   - Cuts are enabled/disabled by single thread
    ///   - New cut coefficients written by single thread
    ///   - No parallel access during modification
    /// ```
    ///
    /// This separation eliminates need for locking on LpProblem.
}
````

### 1.6 Workspace vs SolverPool

> **Clarification**: Section 6.9.6 defines `SolverPool` as an alternative pattern for solver instance management. The two approaches serve different scenarios:

| Approach                         | Use Case           | Memory                 | Warm-Start Efficiency         |
| -------------------------------- | ------------------ | ---------------------- | ----------------------------- |
| `ThreadSolverWorkspace` (§5.4.9) | Production SDDP    | Fixed: 1 solver/thread | Optimal (basis persists)      |
| `SolverPool` (§6.9.6)            | Variable workloads | Pooled: N < threads    | Reduced (basis may not match) |

**Use `ThreadSolverWorkspace`** when thread count is fixed, warm-starting is critical, and memory for N solver instances is acceptable.

**Use `SolverPool`** when thread count varies dynamically, memory is severely constrained, or solver instances have high initialization cost.

## 2. LP Scaling Specification

> **Purpose**: Production SDDP LPs often suffer from numerical ill-conditioning due to variables spanning wide magnitudes (10^-6 to 10^9), constraints with mixed coefficient magnitudes, and the future cost variable θ dominating other variables. LP scaling transforms the problem to improve solver numerical stability.

### 2.1 Scaling Transformation

Given original LP: `min c'x  s.t.  Ax ≤ b, x ≥ 0`, apply column scaling `x̃ = D_c⁻¹ x` and row scaling (multiply each constraint by row scale factor) to produce:

Scaled LP: `min c̃'x̃  s.t.  Ã x̃ ≤ b̃`, where `c̃ = D_c × c`, `Ã = D_r × A × D_c`, `b̃ = D_r × b`, and `D_c = diag(col_scale)`, `D_r = diag(row_scale)`.

**Solution transformation** (unscale to physical units):

- Primal: `x = D_c × x̃`
- Row duals: `π = D_r × π̃`
- Reduced costs: `rc = D_c⁻¹ × rc̃`

### 2.2 Scaling Data Structures

```rust
/// Scaling factors for an LP problem
///
/// Stored per-stage to allow different scaling for different problem structures.
/// MUST be persisted for:
/// 1. Reverting primal/dual solutions to physical units
/// 2. Computing cut coefficients in physical variable space
/// 3. Checkpoint/resume with consistent scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpScaling {
    /// Stage identifier
    pub stage_id: u32,

    /// Column (variable) scaling factors
    /// Physical value = scaled_value × col_scale[j]
    pub col_scale: Vec<f64>,

    /// Row (constraint) scaling factors
    /// Scaled constraint = row_scale[i] × original_constraint
    pub row_scale: Vec<f64>,

    /// Scaling method used
    pub method: ScalingMethod,

    /// Whether scaling was applied (false if well-conditioned)
    pub applied: bool,

    /// Diagnostic metrics
    pub metrics: Option<ScalingMetrics>,
}

/// Scaling method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// No scaling
    None,

    /// Geometric mean scaling (recommended default)
    GeometricMean {
        max_iterations: u32,
        tolerance: f64,
    },

    /// Equilibration (scale so max |a_ij| = 1 in each row/col)
    Equilibration,

    /// Let solver choose internally
    SolverAuto,
}

/// Diagnostic metrics for scaling quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingMetrics {
    /// Coefficient range before scaling
    pub max_coef_before: f64,
    pub min_coef_before: f64,
    pub ratio_before: f64,

    /// Coefficient range after scaling
    pub max_coef_after: f64,
    pub min_coef_after: f64,
    pub ratio_after: f64,
}
```

### 2.3 Scaling Impact on Cut Coefficients

> **Critical**: Cut coefficients must be stored in **physical (unscaled) units** for solver-agnostic portability. Scaling transformations are applied when cuts are added to the solver.

**Transformation for Cut Coefficients**:

If the solver operates in scaled space and produces scaled duals $\tilde{\pi}$, the physical cut coefficients are:

$$\beta_{physical} = W' \times D_r \times \tilde{\pi}$$

Where $W$ is the technology matrix linking state variables to constraints.

In terms of scaled state variables, the cut becomes:

$$\theta \geq \alpha + (D_c \times \beta_{physical})' \times \tilde{x}_{scaled}$$

**Implementation**:

```rust
impl LpScaling {
    /// Transform primal solution from scaled to physical space
    /// physical_x = scaled_x × col_scale
    pub fn unscale_primal(&self, scaled_x: &[f64]) -> Vec<f64> {
        scaled_x.iter()
            .zip(&self.col_scale)
            .map(|(x, s)| x * s)
            .collect()
    }

    /// Transform row duals from scaled to physical space
    /// physical_π = scaled_π × row_scale
    pub fn unscale_row_duals(&self, scaled_pi: &[f64]) -> Vec<f64> {
        scaled_pi.iter()
            .zip(&self.row_scale)
            .map(|(pi, s)| pi * s)
            .collect()
    }

    /// Transform cut coefficients from physical to scaled space
    /// (for adding cut to scaled LP)
    /// scaled_β[j] = physical_β[j] × col_scale[j]
    pub fn scale_cut_coefficients(&self, physical_beta: &[f64]) -> Vec<f64> {
        physical_beta.iter()
            .zip(&self.col_scale)
            .map(|(beta, s)| beta * s)
            .collect()
    }
}
```

### 2.4 Scaling Workflow Integration

**SDDP Solve with Scaling Integration:**

| Step | Operation                 | Description                                                         |
| :--: | ------------------------- | ------------------------------------------------------------------- |
|  1   | **Update Problem**        | Set RHS for state and scenario constraints (physical units)         |
|  2   | **Compute/Apply Scaling** | If needed, compute and apply row/column scaling factors             |
|  3   | **Solve LP**              | Call solver; returns Optimal, Infeasible, Unbounded, or Error       |
|  4   | **Unscale Solution**      | Convert primal and dual values back to physical units               |
|  5   | **Compute Cut**           | Extract state duals (β) and compute intercept (α) in physical space |

![Scaling Workflow Integration](../../../diagrams/exports/svg/data/5-5-4-scaling-workflow-integration.svg)

> **Key Point**: Cuts are always stored in physical units (θ ≥ α + β'x). Scaling is only applied during the solve step and immediately reversed.

### 2.5 FlatBuffers Schema for Scaling Persistence

```flatbuffers
// File: schemas/scaling.fbs
namespace powers.solver;

/// Scaling method enum
enum ScalingMethod : byte {
    None = 0,
    GeometricMean = 1,
    Equilibration = 2,
    SolverAuto = 3,
}

/// Diagnostic metrics for scaling quality
table ScalingMetrics {
    max_coef_before: double;
    min_coef_before: double;
    ratio_before: double;
    max_coef_after: double;
    min_coef_after: double;
    ratio_after: double;
}

/// Scaling factors for a single stage
table StageScaling {
    stage_id: uint32;
    col_scale: [double] (required);
    row_scale: [double] (required);
    method: ScalingMethod = GeometricMean;
    applied: bool = true;
    metrics: ScalingMetrics;
}

/// Collection of scaling factors for all stages
table ScalingCollection {
    version: uint32 = 1;
    num_stages: uint32;
    stages: [StageScaling] (required);
}

root_type ScalingCollection;
file_identifier "SCAL";
file_extension "scales";
```

## Cross-References

- [Solver Abstraction](./solver-abstraction.md) — Trait hierarchy, core trait, cut design, error types, and compile-time selection
- [HiGHS Implementation](./solver-highs-impl.md) — HiGHS-specific `LpSolver` implementation, retry strategy, batch operations
- [LP Formulation](../01-math/lp-formulation.md) — Constraint structure that defines LP dimensions and row/column layout
- [Cut Management](../01-math/cut-management.md) — Cut generation algorithms that produce coefficients stored via the pre-allocated design
- [Training Loop](./training-loop.md) — Forward pass (parallel solve) and backward pass (cut addition) orchestration
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — OpenMP threading model requiring thread-local solver workspaces
- [Memory Architecture](../04-hpc/memory-architecture.md) — NUMA topology and first-touch allocation policy for workspace buffers
- [Binary Formats](../02-data-model/binary-formats.md) — FlatBuffers serialization for scaling factor persistence
- [Configuration Reference](../05-config/configuration-reference.md) — Solver configuration parameters (tolerances, thread counts)
