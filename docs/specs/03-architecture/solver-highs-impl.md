---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §5.4.10 (HiGHS Implementation Guidelines)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from DATA_MODEL_SPECIFICATION.md §5.4.10"
---

# HiGHS Implementation Guidelines

## Purpose

This spec provides implementation guidance specific to HiGHS integration as the default open-source LP solver for POWE.RS. It complements the abstract [`LpSolver` trait](./solver-abstraction.md) with HiGHS-specific patterns for solver lifecycle, warm-starting with bound-toggled cuts, retry strategy, memory footprint, and batch bound operations for the hot path. For thread-local workspace management, see [Solver Workspaces](./solver-workspaces.md).

## 1. Architecture Alignment

| POWE.RS Concept | HiGHS Equivalent              | Notes                              |
| --------------- | ----------------------------- | ---------------------------------- |
| `LpProblem`     | `HighsLp`                     | Data holder, solver-agnostic       |
| `LpSolver`      | `Highs` class                 | Solver controller + internal state |
| `Basis`         | `HighsBasis`                  | Column/row status arrays           |
| `LpSolution`    | `HighsSolution` + `HighsInfo` | Primal, dual, objective            |

## 2. Solver Wrapper

```rust
/// HiGHS solver wrapper implementing LpSolver trait
pub struct HighsSolver {
    /// The HiGHS instance - persists for entire SDDP run
    ///
    /// IMPORTANT: Do NOT create/destroy per solve. The Highs object
    /// maintains internal working memory and basis that enables
    /// efficient warm-starting.
    highs: Highs,

    /// Retry configuration (internal, not exposed to SDDP)
    retry_config: HighsRetryConfig,

    /// Whether LP has been loaded (passModel called)
    model_loaded: bool,

    /// Accumulated statistics
    stats: SolverStatistics,
}

impl HighsSolver {
    /// Create new HiGHS solver instance
    ///
    /// Call once per thread at startup. The instance persists for the
    /// entire SDDP run.
    pub fn new(config: SolverConfig) -> Self {
        let mut highs = Highs::new();

        // Configure for SDDP workload
        highs.set_option("solver", "simplex").unwrap();
        highs.set_option("simplex_strategy", "4").unwrap();  // Dual
        highs.set_option("presolve", "off").unwrap();  // Disable for warm-start
        highs.set_option("parallel", "off").unwrap();  // Thread safety
        highs.set_option("output_flag", false).unwrap();  // Quiet

        // Tolerances
        highs.set_option("primal_feasibility_tolerance", 1e-7).unwrap();
        highs.set_option("dual_feasibility_tolerance", 1e-7).unwrap();

        Self {
            highs,
            retry_config: HighsRetryConfig::default(),
            model_loaded: false,
            stats: SolverStatistics::default(),
        }
    }

    /// Load LP structure (call once, or when structure changes)
    ///
    /// After loading, use update_and_solve() for RHS changes.
    pub fn load_model(&mut self, problem: &LpProblem) {
        let lp = problem.to_highs_lp();
        self.highs.pass_model(lp).expect("Failed to load model");
        self.model_loaded = true;
    }
}
```

## 3. LpSolver Trait Implementation

```rust
impl LpSolver for HighsSolver {
    fn name(&self) -> &'static str {
        "HiGHS"
    }

    fn solve(&mut self, problem: &LpProblem) -> Result<LpSolution, SolverError> {
        if !self.model_loaded {
            self.load_model(problem);
        }

        // Solve with internal retry logic
        self.solve_with_retry()
    }

    fn solve_with_basis(
        &mut self,
        problem: &LpProblem,
        basis: &Basis,
    ) -> Result<LpSolution, SolverError> {
        if !self.model_loaded {
            self.load_model(problem);
        }

        // Set basis for warm-start
        let highs_basis = basis.to_highs_basis();
        self.highs.set_basis(highs_basis).expect("Failed to set basis");

        self.solve_with_retry()
    }

    fn update_and_solve(
        &mut self,
        updates: &ProblemUpdates,
    ) -> Result<LpSolution, SolverError> {
        // Apply RHS changes efficiently
        for &(row, value) in &updates.rhs_changes {
            self.highs.change_row_bounds(row, value, f64::INFINITY).unwrap();
        }

        // Solve (basis from previous solve is automatically used)
        self.solve_with_retry()
    }

    fn reset(&mut self) {
        self.highs.clear_solver();
        self.model_loaded = false;
    }

    fn statistics(&self) -> SolverStatistics {
        self.stats.clone()
    }
}
```

## 4. Warm-Starting with Bound-Toggled Cuts

> **Key Insight**: When cuts are enabled/disabled via bound toggling ([§5.4.3 in solver-abstraction](./solver-abstraction.md)), the LP structure remains unchanged. This means:
>
> 1. **Basis dimensions are constant** — row/column counts don't change
> 2. **Basis status for new cuts** — newly enabled cuts start as non-basic (at lower bound)
> 3. **Warm-start is VALID** — simplex can proceed from current basis

```rust
impl HighsSolver {
    /// Enable cut by updating row bound (warm-start safe)
    ///
    /// The cut row already exists in the LP with bound = -∞.
    /// Enabling sets bound = α (cut RHS).
    ///
    /// Basis impact:
    /// - If row was basic: remains basic (no impact)
    /// - If row was non-basic at -∞: becomes non-basic at α
    /// - Either way, basis is VALID for warm-start
    pub fn enable_cut(&mut self, row: usize, rhs: f64) {
        self.highs.change_row_bounds(row, rhs, f64::INFINITY)
            .expect("Failed to change row bounds");
        // Basis remains valid - no need to clear
    }

    /// Disable cut by resetting row bound (warm-start safe)
    pub fn disable_cut(&mut self, row: usize) {
        self.highs.change_row_bounds(row, f64::NEG_INFINITY, f64::INFINITY)
            .expect("Failed to change row bounds");
        // Basis remains valid
    }
}
```

## 5. Retry Strategy

```rust
impl HighsSolver {
    /// Internal solve with retry logic
    ///
    /// SDDP algorithm never sees retry details - only final result.
    fn solve_with_retry(&mut self) -> Result<LpSolution, SolverError> {
        for (attempt, strategy) in self.retry_config.strategies.iter().enumerate() {
            // Apply strategy
            match strategy {
                HighsRetryStrategy::ClearBasis => {
                    self.highs.clear_basis();
                }
                HighsRetryStrategy::DisablePresolve => {
                    // Already disabled, try enabling briefly
                    self.highs.set_option("presolve", "on").unwrap();
                }
                HighsRetryStrategy::SwitchToIPM => {
                    self.highs.set_option("solver", "ipm").unwrap();
                }
                HighsRetryStrategy::RelaxTolerances { primal, dual } => {
                    self.highs.set_option("primal_feasibility_tolerance", *primal).unwrap();
                    self.highs.set_option("dual_feasibility_tolerance", *dual).unwrap();
                }
            }

            // Attempt solve
            let status = self.highs.run();

            match status {
                HighsStatus::kOk => {
                    let model_status = self.highs.model_status();
                    match model_status {
                        HighsModelStatus::kOptimal => {
                            // Restore default settings for next solve
                            self.restore_default_settings();
                            return Ok(self.extract_solution());
                        }
                        HighsModelStatus::kInfeasible => {
                            return Err(SolverError::Infeasible {
                                stage: 0,  // Filled by caller
                                reason: "LP infeasible".to_string(),
                                ray: self.highs.get_dual_ray().ok(),
                            });
                        }
                        HighsModelStatus::kUnbounded => {
                            return Err(SolverError::Unbounded {
                                stage: 0,
                                reason: "LP unbounded".to_string(),
                                direction: self.highs.get_primal_ray().ok(),
                            });
                        }
                        _ => {
                            // Numerical difficulty, try next strategy
                            self.stats.retries += 1;
                            continue;
                        }
                    }
                }
                _ => {
                    // Solver error, try next strategy
                    self.stats.retries += 1;
                    continue;
                }
            }
        }

        // All retries exhausted
        Err(SolverError::NumericalDifficulty {
            stage: 0,
            reason: format!("Failed after {} attempts", self.retry_config.strategies.len()),
            partial_solution: self.try_extract_partial_solution(),
            suggestion: NumericalRecoverySuggestion::ProblemIllConditioned,
        })
    }

    fn restore_default_settings(&mut self) {
        self.highs.set_option("solver", "simplex").unwrap();
        self.highs.set_option("presolve", "off").unwrap();
        self.highs.set_option("primal_feasibility_tolerance", 1e-7).unwrap();
        self.highs.set_option("dual_feasibility_tolerance", 1e-7).unwrap();
    }

    fn extract_solution(&self) -> LpSolution {
        let info = self.highs.info();
        let solution = self.highs.get_solution();
        let basis = self.highs.get_basis();

        LpSolution {
            status: SolveStatus::Optimal,
            objective_value: info.objective_function_value,
            primal: solution.col_value,
            dual: solution.row_dual,
            reduced_costs: Some(solution.col_dual),
            basis: Some(Basis::from_highs_basis(&basis)),
            simplex_iterations: info.simplex_iteration_count as u64,
        }
    }
}
```

## 6. Memory Footprint

| Component              | Formula                     | Example (1120 states, 15K cuts) |
| ---------------------- | --------------------------- | ------------------------------- |
| LP matrix storage      | nnz × 16 bytes              | ~5 MB                           |
| Working arrays         | (rows + cols) × 3 × 8 bytes | ~1 MB                           |
| Basis storage          | (rows + cols) × 1 byte      | ~50 KB                          |
| Factor storage         | varies with sparsity        | ~5-10 MB                        |
| **Total per instance** |                             | **~15 MB**                      |
| **192 threads**        |                             | **~2.9 GB**                     |

This memory footprint is acceptable for production HPC nodes (256+ GB RAM).

## 7. Batch Bound Operations

> **Performance Critical**: During the forward/backward pass, constraint RHS values must be updated thousands of times per second. Single-row bound changes incur function call overhead per row. Batch operations amortize this overhead.

```rust
impl HighsSolver {
    /// Update multiple row bounds in a single call (hot path optimization)
    ///
    /// This is the preferred method for RHS updates in SDDP:
    /// - Forward pass: Update inflow constraints, state transfer constraints
    /// - Backward pass: Update state constraints for branching scenarios
    ///
    /// # Performance
    ///
    /// | Operation | 500 rows | Overhead |
    /// |-----------|----------|----------|
    /// | 500 × `change_row_bounds()` | ~0.5 ms | High (500 calls) |
    /// | 1 × `change_rows_bounds_batch()` | ~0.05 ms | Low (1 call) |
    ///
    /// # Arguments
    ///
    /// * `row_indices` - Row indices to update (must be pre-allocated, reused)
    /// * `lower_bounds` - New lower bounds (parallel array)
    /// * `upper_bounds` - New upper bounds (parallel array)
    ///
    /// # Thread Safety
    ///
    /// Uses thread-local buffers to avoid allocation. Each thread maintains
    /// its own index/value buffers sized to max constraint count.
    pub fn change_rows_bounds_batch(
        &mut self,
        row_indices: &[i32],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> Result<(), SolverError> {
        debug_assert_eq!(row_indices.len(), lower_bounds.len());
        debug_assert_eq!(row_indices.len(), upper_bounds.len());

        // HiGHS API: changeRowsBounds(num_rows, indices, lower, upper)
        let status = unsafe {
            highs_sys::Highs_changeRowsBounds(
                self.highs.ptr(),
                row_indices.len() as i32,
                row_indices.as_ptr(),
                lower_bounds.as_ptr(),
                upper_bounds.as_ptr(),
            )
        };

        if status == highs_sys::kHighsStatusOk {
            Ok(())
        } else {
            Err(SolverError::InternalError {
                reason: "Failed to change row bounds".to_string(),
            })
        }
    }

    /// Update multiple column bounds in batch (for cut activation)
    ///
    /// Used when enabling/disabling multiple cuts at once (e.g., after cut selection).
    pub fn change_cols_bounds_batch(
        &mut self,
        col_indices: &[i32],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> Result<(), SolverError> {
        debug_assert_eq!(col_indices.len(), lower_bounds.len());
        debug_assert_eq!(col_indices.len(), upper_bounds.len());

        let status = unsafe {
            highs_sys::Highs_changeColsBounds(
                self.highs.ptr(),
                col_indices.len() as i32,
                col_indices.as_ptr(),
                lower_bounds.as_ptr(),
                upper_bounds.as_ptr(),
            )
        };

        if status == highs_sys::kHighsStatusOk {
            Ok(())
        } else {
            Err(SolverError::InternalError {
                reason: "Failed to change column bounds".to_string(),
            })
        }
    }
}
```

## 8. Thread-Local Batch Buffers

```rust
/// Thread-local buffers for batch bound operations
///
/// These buffers are allocated once per thread and reused across all solves.
/// Eliminates allocation overhead on the hot path.
thread_local! {
    /// Row indices buffer (sized to max constraints per stage)
    static BATCH_ROW_INDICES: RefCell<Vec<i32>> = RefCell::new(Vec::with_capacity(2000));

    /// Lower bounds buffer
    static BATCH_LOWER_BOUNDS: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(2000));

    /// Upper bounds buffer
    static BATCH_UPPER_BOUNDS: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(2000));
}

impl ThreadSolverWorkspace {
    /// Prepare batch buffers for RHS update
    ///
    /// Populates thread-local buffers with row indices and new RHS values.
    /// Returns references to the buffers for use with batch API.
    pub fn prepare_batch_rhs_update(
        &self,
        updates: &[(usize, f64)],  // (row_index, new_rhs)
    ) -> (&[i32], &[f64], &[f64]) {
        BATCH_ROW_INDICES.with(|indices| {
            BATCH_LOWER_BOUNDS.with(|lower| {
                BATCH_UPPER_BOUNDS.with(|upper| {
                    let mut indices = indices.borrow_mut();
                    let mut lower = lower.borrow_mut();
                    let mut upper = upper.borrow_mut();

                    indices.clear();
                    lower.clear();
                    upper.clear();

                    for &(row, rhs) in updates {
                        indices.push(row as i32);
                        lower.push(rhs);           // Lower bound = RHS for equality/≥
                        upper.push(f64::INFINITY); // Upper bound = ∞ for ≥ constraints
                    }

                    // Return borrowed slices
                    // Note: In actual implementation, need to handle lifetimes properly
                })
            })
        })
    }
}
```

## Cross-References

- [Solver Abstraction](./solver-abstraction.md) — Core `LpSolver` trait, `CutSlotManager`, error types, retry config, and compile-time solver selection
- [Solver Workspaces & LP Scaling](./solver-workspaces.md) — Thread-local `ThreadSolverWorkspace` that owns an `ActiveSolver` instance, NUMA-aware init, and LP scaling
- [LP Formulation](../01-math/lp-formulation.md) — Constraint structure and row/column layout that HiGHS operates on
- [Cut Management](../01-math/cut-management.md) — Cut generation producing coefficients managed via bound toggling
- [Training Loop](./training-loop.md) — Forward/backward pass orchestration driving solver invocations
- [Hybrid Parallelism](../04-hpc/hybrid-parallelism.md) — OpenMP threading model requiring one HiGHS instance per thread
- [Memory Architecture](../04-hpc/memory-architecture.md) — NUMA topology and memory budget for solver instances
- [Configuration Reference](../05-config/configuration-reference.md) — Solver configuration parameters (tolerances, retry strategies)
