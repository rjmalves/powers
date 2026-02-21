---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §5.4 (5.4.1-5.4.8)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from DATA_MODEL_SPECIFICATION.md §5.4.1-5.4.8"
  - date: 2026-02-20
    description: "Review note (from sddp-algorithm.md review): LP rebuild cost is a critical performance constraint. Cannot keep all stage LPs with full cut sets in memory simultaneously — must rebuild LPs and add cuts when transitioning between stages. The solver abstraction must support efficient incremental cut addition and basis warm-starting to minimize this overhead. Validate that the pre-allocated cut slot design and basis storage adequately address this during P3 review. See sddp-algorithm.md §3.4."
---

# Solver Abstraction Layer

## Purpose

This spec defines the multi-solver abstraction layer: the unified trait hierarchy through which the SDDP algorithm interacts with LP solvers (HiGHS, CPLEX, Gurobi), including the core solver trait, pre-allocated cut constraint design, error types, retry logic, dual normalization, basis storage, and compile-time solver selection via Cargo feature flags.

For thread-local solver infrastructure, the HiGHS reference implementation, and LP scaling, see [Solver Workspaces & LP Scaling](./solver-workspaces.md).

## 1. Design Rationale

The SDDP algorithm must be solver-agnostic. The solver interface abstracts LP solver details (HiGHS, CPLEX, Gurobi) behind a unified trait hierarchy. Key design decisions:

1. **Compile-time solver selection** via Cargo features (avoids vtable overhead on hot path)
2. **Encapsulated retry logic** — each solver handles its own numerical difficulties internally
3. **Pre-allocated cut slots** — LP structure remains static; cuts enabled/disabled via bound toggling
4. **Cuts stored in physical units** — scaling transformations applied at solve time

## 2. Trait Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SOLVER TRAIT HIERARCHY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        LpProblem (Data Holder)                        │  │
│  │  - Owns row/column data, bounds, objective, coefficients              │  │
│  │  - Solver-agnostic problem representation                             │  │
│  │  - Pre-allocates cut constraint rows at construction                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      LpScaling (Pre-processing)                       │  │
│  │  - Row/column scaling factors for numerical stability                 │  │
│  │  - Transforms problem ↔ scaled space                                  │  │
│  │  - Persisted for cut coefficient computation                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        LpSolver (Execution)                           │  │
│  │  - solve() → Result<LpSolution, SolverError>                          │  │
│  │  - Encapsulates retry logic, warm-start, basis management             │  │
│  │  - Compile-time selected via Cargo features                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     LpSolution (Result Data)                          │  │
│  │  - Primal values, dual values (constraint & variable)                 │  │
│  │  - Objective value, solve status                                      │  │
│  │  - Basis information (optional, for warm-starting)                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Static Data vs Transient Solver State:**

| Aspect    | LpProblem (Static Data)           | Solver State (Transient)                       |
| --------- | --------------------------------- | ---------------------------------------------- |
| Lifetime  | Per-stage, entire algorithm       | Per-solve invocation                           |
| Ownership | Shared read-only across threads   | Thread-local                                   |
| Contents  | Matrix, bounds, coefficients      | Basis factors, working arrays                  |
| Memory    | Pre-allocated once                | Created/reused per solve                       |
| Analogies | HiGHS: `HighsLp`, CLP: `ClpModel` | HiGHS: `Highs::run()` state, CLP: `ClpSimplex` |

This separation enables the key optimization: problem data is read-only and shared, while each thread maintains its own solver workspace for warm-starting.

## 3. Core Solver Trait

```rust
/// Core solver trait - the primary interface for SDDP
///
/// Implementations handle:
/// - Problem setup and solving
/// - Retry logic for numerical difficulties (solver-specific)
/// - Warm-starting and basis reuse
/// - Solution extraction with dual normalization
///
/// IMPORTANT: The SDDP algorithm never sees retry details. It calls solve()
/// and receives either a solution or an error.
pub trait LpSolver: Send + Sync {
    /// Solver identifier for logging
    fn name(&self) -> &'static str;

    /// Solve the LP problem
    ///
    /// This is the main entry point. Implementations handle:
    /// - Translating LpProblem to solver format
    /// - Applying solver-specific parameters
    /// - Internal retry logic for numerical issues
    /// - Solution extraction and dual normalization
    ///
    /// Returns `Ok(LpSolution)` on success (possibly after internal retries).
    fn solve(&mut self, problem: &LpProblem) -> Result<LpSolution, SolverError>;

    /// Solve with warm-start from previous basis
    fn solve_with_basis(
        &mut self,
        problem: &LpProblem,
        basis: &Basis,
    ) -> Result<LpSolution, SolverError>;

    /// Update problem and re-solve (incremental for stage re-use)
    ///
    /// More efficient than full solve when only RHS/bounds change.
    fn update_and_solve(
        &mut self,
        updates: &ProblemUpdates,
    ) -> Result<LpSolution, SolverError>;

    /// Reset solver state (clear caches, basis, etc.)
    fn reset(&mut self);

    /// Get accumulated solver statistics
    fn statistics(&self) -> SolverStatistics;
}
```

## 4. Pre-allocated Cut Constraint Design

![Cut Storage Layout](../../../diagrams/exports/svg/data/cut-storage-layout.svg)

> **Key Insight**: Instead of dynamically adding/removing LP rows for Benders cuts, we pre-allocate all cut constraint rows at LP construction time. Cuts are **enabled/disabled by toggling their bounds**, not by row insertion/deletion. This preserves cache locality and enables warm-starting.
>
> **Design Decision**: Full preallocation with dynamic capacity was chosen over dynamic slot management because:
>
> 1. **Reproducibility**: Checkpoint/resume requires identical LP structure for bit-for-bit results
> 2. **Thread Safety**: No runtime allocation eliminates need for atomic operations
> 3. **Performance**: Zero allocation overhead during parallel solve loops
> 4. **HPC Scale**: 10-20 GB memory is acceptable for production HPC nodes (256+ GB)

**Mathematical Foundation**:

A Benders cut has the form: `θ ≥ α + β'x` where:

- `θ` is the future cost variable
- `α` is the cut intercept (RHS)
- `β` is the vector of cut coefficients (dual multipliers)
- `x` is the state vector from the previous stage

Rearranging to standard LP row form: `θ - β'x ≥ α`

For further details on cut generation and selection, see [Cut Management](../01-math/cut-management.md).

**Bound Toggling**:

| State        | Row Lower Bound | Row Upper Bound | Effect                      |
| ------------ | --------------- | --------------- | --------------------------- |
| **Active**   | `α` (cut RHS)   | `+∞`            | Cut enforced                |
| **Inactive** | `-∞`            | `+∞`            | Always satisfied (free row) |

**Capacity Calculation**:

```
capacity = warm_start_cuts + (max_iterations × forward_passes)
         = N + M

Example:
  - Warm-start with 5,000 existing cuts: N = 5,000
  - Training for 50 iterations × 200 forward passes: M = 10,000
  - Total capacity per stage: 15,000 cuts
  - Memory at 1120 state dimension: ~15,000 × 1120 × 8 ≈ 134 MB/stage
  - Total for 120 stages: ~16 GB
```

**Deterministic Slot Assignment**:

````rust
/// Cut slot manager with deterministic indexing
///
/// # Design Rationale
///
/// Slots are COMPUTED, not allocated at runtime. This eliminates:
/// - Thread-safety concerns (no concurrent allocation)
/// - Non-determinism (same inputs → same slot assignments)
/// - Runtime overhead (O(1) slot computation)
///
/// # Slot Layout
///
/// ```text
/// Slots:     [0 .............. N-1] [N ........................ N+M-1]
///            |--- Warm-start ---|   |-------- New Training --------|
///
/// New cut slot = N + iteration × forward_passes + forward_pass_idx
/// ```
pub struct CutSlotManager {
    /// Total pre-allocated cut slots in LP
    /// capacity = warm_start_count + max_iterations × forward_passes
    pub capacity: usize,

    /// Base LP row index for first cut constraint
    /// Cuts occupy rows [base_row, base_row + capacity)
    pub base_row_index: usize,

    /// Number of warm-start cuts (slot offset for new cuts)
    pub warm_start_count: usize,

    /// Number of forward passes per iteration
    pub forward_passes: usize,

    /// Active cut bitmap: bit k = 1 if cut slot k is active
    pub active_bitmap: BitVec,

    /// Count of active cuts (avoids bitmap scan)
    pub active_count: u32,
}

impl CutSlotManager {
    /// Create a new slot manager with computed capacity
    pub fn new(
        warm_start_count: usize,
        max_iterations: usize,
        forward_passes: usize,
        base_row_index: usize,
    ) -> Self {
        let capacity = warm_start_count + max_iterations * forward_passes;
        Self {
            capacity,
            base_row_index,
            warm_start_count,
            forward_passes,
            active_bitmap: BitVec::repeat(false, capacity),
            active_count: 0,
        }
    }

    /// Compute slot for a new cut (deterministic, NOT allocation)
    ///
    /// This is a pure function with no side effects.
    /// Thread-safe by design: no mutable state.
    #[inline]
    pub const fn slot_for_new_cut(&self, iteration: u32, forward_pass_idx: u32) -> u32 {
        self.warm_start_count as u32
            + iteration * self.forward_passes as u32
            + forward_pass_idx
    }

    /// Get LP row index for a cut slot
    #[inline]
    pub const fn row_index(&self, slot: u32) -> usize {
        self.base_row_index + slot as usize
    }

    /// Activate a cut slot (update bitmap, called after LP bound is set)
    ///
    /// Note: This modifies bitmap only. LP bound update is separate.
    /// Called from single-threaded cut addition code.
    pub fn activate(&mut self, slot: u32) {
        debug_assert!((slot as usize) < self.capacity);
        if !self.active_bitmap[slot as usize] {
            self.active_bitmap.set(slot as usize, true);
            self.active_count += 1;
        }
    }

    /// Deactivate a cut slot (for Level 1 selection or cut purging)
    pub fn deactivate(&mut self, slot: u32) {
        debug_assert!((slot as usize) < self.capacity);
        if self.active_bitmap[slot as usize] {
            self.active_bitmap.set(slot as usize, false);
            self.active_count -= 1;
        }
    }

    /// Check if a slot is active
    #[inline]
    pub fn is_active(&self, slot: u32) -> bool {
        self.active_bitmap[slot as usize]
    }

    /// Iterate over all active slots (for checkpoint serialization)
    pub fn active_slots(&self) -> impl Iterator<Item = u32> + '_ {
        self.active_bitmap.iter_ones().map(|i| i as u32)
    }
}

/// LP problem with pre-allocated cut slots
impl LpProblem {
    /// Enable a cut by setting its row bound to the cut RHS
    ///
    /// Changes: row_lb[slot] = -∞  →  row_lb[slot] = α (cut RHS)
    ///
    /// # Thread Safety
    ///
    /// This method modifies a single row bound. Multiple threads can
    /// enable different cuts concurrently (different rows).
    pub fn enable_cut(&mut self, slot: u32, rhs: f64, cut_manager: &CutSlotManager) {
        let row = cut_manager.row_index(slot);
        self.row_lower[row] = rhs;
        // row_upper remains +∞
    }

    /// Disable a cut by setting its row bound to -∞
    ///
    /// Changes: row_lb[slot] = α  →  row_lb[slot] = -∞
    pub fn disable_cut(&mut self, slot: u32, cut_manager: &CutSlotManager) {
        let row = cut_manager.row_index(slot);
        self.row_lower[row] = f64::NEG_INFINITY;
    }

    /// Set cut coefficients for a slot
    ///
    /// Called once when the cut is first created. For full preallocation,
    /// coefficients are written to preallocated rows.
    pub fn set_cut_coefficients(
        &mut self,
        slot: u32,
        rhs: f64,
        coefficients: &[f64],
        cut_manager: &CutSlotManager,
    ) {
        let row = cut_manager.row_index(slot);

        // Set coefficients: θ - β'x ≥ α  →  coefficient for x[j] is -β[j]
        for (state_idx, &coef) in coefficients.iter().enumerate() {
            let col = self.state_variable_columns[state_idx];
            self.set_coefficient(row, col, -coef);
        }

        // θ coefficient is always +1
        self.set_coefficient(row, self.future_cost_column, 1.0);

        // Set the RHS (enables the cut)
        self.row_lower[row] = rhs;
    }

    /// Update cut coefficients in-place (for cut strengthening)
    ///
    /// Note: Only updates coefficients, not the bound.
    /// The cut must be re-enabled with the new RHS after this.
    pub fn update_cut_coefficients(
        &mut self,
        slot: u32,
        coefficients: &[f64],
        cut_manager: &CutSlotManager,
    ) {
        let row = cut_manager.row_index(slot);
        // Update sparse matrix row
        for (state_idx, &coef) in coefficients.iter().enumerate() {
            let col = self.state_variable_columns[state_idx];
            // Coefficient sign: θ - β'x ≥ α  →  coefficient for x[j] is -β[j]
            self.set_coefficient(row, col, -coef);
        }
        // θ coefficient is always +1
        self.set_coefficient(row, self.future_cost_column, 1.0);
    }
}
````

## 5. Solver Error Types

```rust
/// Solver error visible to SDDP algorithm
///
/// This is the only error type SDDP sees. Solver implementations
/// convert internal errors to this type.
#[derive(Debug, Error)]
pub enum SolverError {
    /// Problem is primal infeasible (no feasible solution exists)
    /// SDDP Action: Data error - check bounds, constraints
    #[error("LP infeasible at stage {stage}: {reason}")]
    Infeasible {
        stage: u32,
        reason: String,
        ray: Option<Vec<f64>>,  // Infeasibility ray if available
    },

    /// Problem is dual infeasible (unbounded)
    /// SDDP Action: Modeling error - check objective signs
    #[error("LP unbounded at stage {stage}: {reason}")]
    Unbounded {
        stage: u32,
        reason: String,
        direction: Option<Vec<f64>>,  // Unbounded direction if available
    },

    /// Numerical difficulties (retries exhausted)
    /// SDDP Action: Log warning, may have partial solution
    #[error("Numerical difficulty at stage {stage}: {reason}")]
    NumericalDifficulty {
        stage: u32,
        reason: String,
        partial_solution: Option<LpSolution>,
        suggestion: NumericalRecoverySuggestion,
    },

    /// Time limit exceeded
    #[error("Time limit exceeded at stage {stage} after {elapsed_secs:.2}s")]
    TimeLimit {
        stage: u32,
        elapsed_secs: f64,
        best_solution: Option<LpSolution>,
    },

    /// Iteration limit exceeded
    #[error("Iteration limit at stage {stage} after {iterations} iterations")]
    IterationLimit {
        stage: u32,
        iterations: u64,
        best_solution: Option<LpSolution>,
    },

    /// Unrecoverable internal error
    #[error("Solver internal error: {message}")]
    Internal {
        message: String,
        code: Option<i32>,
        retryable: bool,
    },
}

/// Suggestions for recovering from numerical difficulties
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalRecoverySuggestion {
    /// Problem may be poorly scaled
    ApplyScaling,
    /// Try cold start instead of warm start
    ColdStart,
    /// Tighten solver tolerances
    TightenTolerances,
    /// Problem is inherently ill-conditioned
    ProblemIllConditioned,
    /// No specific suggestion
    Unknown,
}

impl SolverError {
    /// Check if error has a usable (possibly suboptimal) solution
    pub fn has_solution(&self) -> bool {
        match self {
            SolverError::NumericalDifficulty { partial_solution, .. } => partial_solution.is_some(),
            SolverError::TimeLimit { best_solution, .. } => best_solution.is_some(),
            SolverError::IterationLimit { best_solution, .. } => best_solution.is_some(),
            _ => false,
        }
    }
}
```

## 6. Solver-Specific Retry Logic

> **Important**: Retry logic is **encapsulated within each solver implementation**. The SDDP algorithm never sees retry details — it only receives the final result from `solve()`.

Each solver implementation defines its own retry sequence based on its failure modes:

```rust
/// Example: HiGHS retry configuration (internal to HiGHS implementation)
///
/// This is NOT part of the public API - SDDP algorithm doesn't see this.
pub(crate) struct HighsRetryConfig {
    pub max_attempts: u32,
    pub time_budget: Duration,
    pub strategies: Vec<HighsRetryStrategy>,
}

pub(crate) enum HighsRetryStrategy {
    ClearBasis,                              // Attempt 1 failure: clear warm-start
    DisablePresolve,                         // Attempt 2 failure: disable presolve
    SwitchToIPM,                             // Attempt 3 failure: use interior point
    RelaxTolerances { primal: f64, dual: f64 }, // Attempt 4: relax tolerances
}

impl Default for HighsRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            time_budget: Duration::from_secs(30),
            strategies: vec![
                HighsRetryStrategy::ClearBasis,
                HighsRetryStrategy::DisablePresolve,
                HighsRetryStrategy::SwitchToIPM,
                HighsRetryStrategy::RelaxTolerances { primal: 1e-6, dual: 1e-6 },
            ],
        }
    }
}
```

## 7. Dual Variable Normalization

> **Critical**: Different solvers report dual multipliers with different sign conventions. The solver implementation MUST normalize duals to the canonical form before returning to SDDP.

**Canonical Sign Convention**:

- Positive dual means: increasing RHS **increases** objective (binding upper bound)
- For constraint `Ax ≤ b` with positive dual: shadow price is ∂z\*/∂b > 0

```rust
/// Dual normalization within solver implementation
impl HiGHSSolver {
    /// Normalize dual value to canonical form (internal method)
    fn normalize_dual(&self, raw_dual: f64, constraint_type: ConstraintType) -> f64 {
        // HiGHS convention: positive for ≤ constraints
        match constraint_type {
            ConstraintType::LessEqual => raw_dual,      // Already correct
            ConstraintType::GreaterEqual => -raw_dual,  // Flip sign
            ConstraintType::Equality => raw_dual,       // Unrestricted
        }
    }
}
```

## 8. Basis Storage for Warm-Starting

```rust
/// Basis information for warm-starting across iterations
///
/// Stored in ORIGINAL problem space (not presolved) for portability.
#[derive(Debug, Clone)]
pub struct Basis {
    /// Basis status for each column (variable)
    pub column_status: Vec<BasisStatus>,

    /// Basis status for each row (constraint)
    pub row_status: Vec<BasisStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasisStatus {
    /// Variable is at its lower bound
    AtLower = 0,
    /// Variable is basic (in the basis)
    Basic = 1,
    /// Variable is at its upper bound
    AtUpper = 2,
    /// Variable is free (superbasic)
    Free = 3,
    /// Variable is fixed
    Fixed = 4,
}
```

## 9. Compile-Time Solver Selection

```toml
# Cargo.toml feature flags
[features]
default = ["solver-highs"]
solver-highs = ["highs"]
solver-cplex = ["cplex-rs"]
solver-gurobi = ["gurobi-rs"]
```

```rust
/// Type alias selected at compile time via features
#[cfg(feature = "solver-highs")]
pub type ActiveSolver = HighsSolver;

#[cfg(feature = "solver-cplex")]
pub type ActiveSolver = CplexSolver;

#[cfg(feature = "solver-gurobi")]
pub type ActiveSolver = GurobiSolver;

/// Create solver instance (compile-time selected)
pub fn create_solver(config: SolverConfig) -> ActiveSolver {
    ActiveSolver::new(config)
}
```

## Cross-References

- [Solver Workspaces & LP Scaling](./solver-workspaces.md) — Thread-local solver infrastructure and LP scaling specification
- [HiGHS Implementation](./solver-highs-impl.md) — HiGHS-specific `LpSolver` implementation, retry strategy, batch operations
- [LP Formulation](../01-math/lp-formulation.md) — Constraint structure that the solver operates on
- [Cut Management](../01-math/cut-management.md) — How cuts are generated; this spec handles how they are stored and enabled in the LP
- [Training Loop](./training-loop.md) — Forward pass (parallel solve) and backward pass (cut addition) that drive solver invocations
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — OpenMP threading model that requires thread-local solvers
- [Memory Architecture](../04-hpc/memory-architecture.md) — NUMA-aware allocation for solver workspaces
- [Binary Formats](../02-data-model/binary-formats.md) — FlatBuffers schema for LP structure and scaling persistence
- [Configuration Reference](../05-config/configuration-reference.md) — Solver configuration parameters (threads_per_solve, tolerances)
