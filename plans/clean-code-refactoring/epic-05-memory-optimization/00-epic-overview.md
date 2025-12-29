# Epic 5: Memory Optimization

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 3 weeks (2 sprints)
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

This epic implements pool-based allocation to eliminate dynamic allocations in hot paths.

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

---

## Summary

This epic implements pool-based memory management to achieve **zero-allocation hot paths** in the training and simulation loops. Based on analysis from previous epics, we introduce mandatory pools for cuts and states (which have a 1:1 relationship), and a formal buffer system for realizations and trajectories.

**Key principle**: Preallocate everything, reuse aggressively. The algorithm sees the same data; only the allocation strategy changes.

**Scope expansion**: This epic also analyzes and potentially implements **direct SoA (Structure of Arrays) migration** for realizations and trajectories, enabled by the formal buffer structure.

---

## Scope

### Included

1. **Cut-State Pool (Mandatory)**
   - Preallocated storage for cuts AND their associated states (1:1 relationship)
   - Slot-based indexing using `(iteration, forward_pass_idx)`
   - Pool recycling for dominated cuts

2. **Realization Buffer System (Mandatory)**
   - Formal `SolutionBuffer` and `BasisBuffer` data structures
   - `get_solution_into(&mut buffer)` and `get_basis_into(&mut buffer)` methods for solver
   - Per-stage buffer allocation
   - Structured field access (not just `Vec<f64>`)

3. **Trajectory Buffer (Mandatory)**
   - Store all solutions and basis for all stages in a single iteration
   - Support for both training and simulation steps
   - Organized access by stage and field

4. **SoA Analysis and Migration**
   - Analyze feasibility of direct SoA conversion for realizations
   - If feasible, implement SoA layout for cache efficiency
   - Document tradeoffs and migration path

5. **Hot Path Verification**
   - Verify zero allocations in forward/backward loops
   - Use DHAT or similar for validation

### Excluded

- Algorithm changes
- External dependency changes

---

## Dependencies

- **Requires**:
  - Epic 4 complete (allocation points documented, pool interfaces ready)
  - Epic 3 complete (clean algorithm boundaries)
  - Epic 2 complete (solution extraction with `extract_into()` APIs)
- **Enables**:
  - Epic 7: Performance Validation (final verification)

---

## Acceptance Criteria

- [ ] **Cut-State pool implemented** with 1:1 relationship preserved
- [ ] **Slot-based indexing** using `(iteration, forward_pass_idx)`
- [ ] **SolutionBuffer and BasisBuffer** formal structures implemented
- [ ] **`get_solution_into()` and `get_basis_into()`** methods on solver
- [ ] **Trajectory buffer** stores all stages for single iteration
- [ ] **Works for both training AND simulation** steps
- [ ] Forward/backward loops have zero heap allocations
- [ ] Memory profile is flat (no growth during training)
- [ ] SoA feasibility analyzed and documented
- [ ] Golden tests pass
- [ ] Performance improved or unchanged (target: +5-10%)

---

## Technical Approach

### 1. Cut-State Pool with 1:1 Relationship

Each cut has exactly one associated state (the state visited when the cut was created). They share the same slot.

#### Slot Indexing Strategy

Use the existing `(iteration, forward_pass_idx)` as the slot key:

```rust
/// Slot index for cut-state pairs.
/// This indexing is already used in the current codebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CutStateSlotId {
    pub iteration: usize,        // 1-based iteration number
    pub forward_pass_idx: usize, // 0-based forward pass index
}

impl CutStateSlotId {
    /// Convert to linear index for array access.
    /// Assumes max_forward_passes is known from config.
    pub fn to_linear(&self, max_forward_passes: usize) -> usize {
        (self.iteration - 1) * max_forward_passes + self.forward_pass_idx
    }
}
```

#### Pool Structure

```rust
/// Pool for cut-state pairs with 1:1 relationship.
/// 
/// # Slot Management
/// 
/// Slots are indexed by (iteration, forward_pass_idx).
/// Each slot contains:
/// - Cut coefficients and RHS
/// - State coefficients (copied, not boxed)
/// - Metadata (active/inactive, creation time)
/// 
/// # Lifecycle
/// 
/// 1. Slot allocated when cut is created
/// 2. Slot marked inactive when cut is dominated
/// 3. Slot recycled when iteration rolls over (if needed)
pub struct CutStatePool {
    /// Maximum slots = max_iterations * forward_passes_per_iteration
    capacity: usize,
    
    /// Cut coefficients: [slot_id][coef_idx]
    /// Stored as contiguous array for cache efficiency
    cut_coefficients: Vec<f64>,
    
    /// Cut RHS values: [slot_id]
    cut_rhs: Vec<f64>,
    
    /// State coefficients: [slot_id][state_coef_idx]
    state_coefficients: Vec<f64>,
    
    /// Slot metadata
    slot_metadata: Vec<SlotMetadata>,
    
    /// Dimensions for indexing
    max_cut_coefficients: usize,
    max_state_coefficients: usize,
    max_forward_passes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SlotMetadata {
    pub active: bool,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}

impl CutStatePool {
    /// Store cut-state pair in slot.
    pub fn store(
        &mut self,
        slot_id: CutStateSlotId,
        cut_coefficients: &[f64],
        cut_rhs: f64,
        state_coefficients: &[f64],
    ) {
        let linear_idx = slot_id.to_linear(self.max_forward_passes);
        
        // Copy cut coefficients
        let cut_start = linear_idx * self.max_cut_coefficients;
        self.cut_coefficients[cut_start..cut_start + cut_coefficients.len()]
            .copy_from_slice(cut_coefficients);
        
        self.cut_rhs[linear_idx] = cut_rhs;
        
        // Copy state coefficients
        let state_start = linear_idx * self.max_state_coefficients;
        self.state_coefficients[state_start..state_start + state_coefficients.len()]
            .copy_from_slice(state_coefficients);
        
        self.slot_metadata[linear_idx] = SlotMetadata {
            active: true,
            iteration: slot_id.iteration,
            forward_pass_idx: slot_id.forward_pass_idx,
        };
    }
    
    /// Mark slot as inactive (cut dominated).
    pub fn deactivate(&mut self, slot_id: CutStateSlotId) {
        let linear_idx = slot_id.to_linear(self.max_forward_passes);
        self.slot_metadata[linear_idx].active = false;
    }
    
    /// Get cut coefficients for a slot.
    pub fn cut_coefficients(&self, slot_id: CutStateSlotId) -> &[f64] {
        let linear_idx = slot_id.to_linear(self.max_forward_passes);
        let start = linear_idx * self.max_cut_coefficients;
        &self.cut_coefficients[start..start + self.max_cut_coefficients]
    }
    
    /// Get state coefficients for a slot.
    pub fn state_coefficients(&self, slot_id: CutStateSlotId) -> &[f64] {
        let linear_idx = slot_id.to_linear(self.max_forward_passes);
        let start = linear_idx * self.max_state_coefficients;
        &self.state_coefficients[start..start + self.max_state_coefficients]
    }
}
```

### 2. Realization Buffer System

Replace allocating `Vec<f64>` with formal buffer structures that support `get_solution_into()`.

#### SolutionBuffer and BasisBuffer

```rust
/// Preallocated buffer for LP solution extraction.
/// 
/// # Usage
/// 
/// Instead of solver allocating a new Vec<f64>:
/// ```rust
/// // OLD: allocates
/// let solution = solver.get_solution();
/// 
/// // NEW: zero allocation
/// solver.get_solution_into(&mut buffer);
/// ```
/// 
/// # Field Access
/// 
/// Provides structured access to solution components:
/// ```rust
/// buffer.deficit();        // &[f64]
/// buffer.thermal_gen();    // &[f64]
/// buffer.final_storage();  // &[f64]
/// ```
pub struct SolutionBuffer {
    /// Raw solution values from solver
    colvalue: Vec<f64>,
    /// Raw dual values from solver
    rowdual: Vec<f64>,
    
    /// Index mappings for field access
    indices: SolutionIndices,
}

/// Index ranges for structured access.
pub struct SolutionIndices {
    pub deficit: Range<usize>,
    pub direct_exchange: Range<usize>,
    pub reverse_exchange: Range<usize>,
    pub thermal_gen: Range<usize>,
    pub spillage: Range<usize>,
    pub turbined_flow: Range<usize>,
    pub stored_volume: Range<usize>,
    pub load: Range<usize>,
    pub inflow: Range<usize>,
    
    // Dual indices
    pub hydro_balance: Range<usize>,
    pub load_balance: Range<usize>,
}

impl SolutionBuffer {
    /// Create buffer with capacity for given problem dimensions.
    pub fn with_capacity(num_cols: usize, num_rows: usize, indices: SolutionIndices) -> Self {
        Self {
            colvalue: vec![0.0; num_cols],
            rowdual: vec![0.0; num_rows],
            indices,
        }
    }
    
    /// Get mutable slice for solver to write into.
    pub fn colvalue_mut(&mut self) -> &mut [f64] {
        &mut self.colvalue
    }
    
    pub fn rowdual_mut(&mut self) -> &mut [f64] {
        &mut self.rowdual
    }
    
    // Structured field access (read-only)
    pub fn deficit(&self) -> &[f64] {
        &self.colvalue[self.indices.deficit.clone()]
    }
    
    pub fn thermal_gen(&self) -> &[f64] {
        &self.colvalue[self.indices.thermal_gen.clone()]
    }
    
    pub fn final_storage(&self) -> &[f64] {
        &self.colvalue[self.indices.stored_volume.clone()]
    }
    
    pub fn water_value(&self) -> &[f64] {
        &self.rowdual[self.indices.hydro_balance.clone()]
    }
    
    pub fn marginal_cost(&self) -> &[f64] {
        &self.rowdual[self.indices.load_balance.clone()]
    }
    
    // ... other accessors
}

/// Preallocated buffer for LP basis information.
pub struct BasisBuffer {
    pub col_status: Vec<i32>,
    pub row_status: Vec<i32>,
}
```

#### Solver Integration

```rust
impl Solver {
    /// Get solution into preallocated buffer.
    /// Zero allocation after initial buffer creation.
    pub fn get_solution_into(&self, buffer: &mut SolutionBuffer) -> Result<(), SolverError> {
        // HiGHS FFI call to copy solution into buffer.colvalue
        highs_get_solution(
            self.model,
            buffer.colvalue_mut().as_mut_ptr(),
            buffer.rowdual_mut().as_mut_ptr(),
            // ...
        );
        Ok(())
    }
    
    /// Get basis into preallocated buffer.
    pub fn get_basis_into(&self, buffer: &mut BasisBuffer) -> Result<(), SolverError> {
        highs_get_basis(
            self.model,
            buffer.col_status.as_mut_ptr(),
            buffer.row_status.as_mut_ptr(),
        );
        Ok(())
    }
}
```

### 3. Trajectory Buffer

Store all solutions and basis for all stages in a single iteration.

```rust
/// Buffer for a complete trajectory (all stages in one iteration).
/// 
/// Used in both training and simulation.
/// 
/// # Structure
/// 
/// ```text
/// Trajectory
/// ├── Stage 0: SolutionBuffer + BasisBuffer
/// ├── Stage 1: SolutionBuffer + BasisBuffer
/// ├── ...
/// └── Stage N: SolutionBuffer + BasisBuffer
/// ```
pub struct TrajectoryBuffer {
    /// Solution buffers per stage
    solutions: Vec<SolutionBuffer>,
    /// Basis buffers per stage
    bases: Vec<BasisBuffer>,
    /// Stage count
    num_stages: usize,
}

impl TrajectoryBuffer {
    /// Create trajectory buffer for given number of stages.
    pub fn with_stages(
        num_stages: usize,
        stage_dimensions: &[StageDimensions],
    ) -> Self {
        let solutions = stage_dimensions.iter()
            .map(|dim| SolutionBuffer::with_capacity(dim.num_cols, dim.num_rows, dim.indices.clone()))
            .collect();
        
        let bases = stage_dimensions.iter()
            .map(|dim| BasisBuffer::with_capacity(dim.num_cols, dim.num_rows))
            .collect();
        
        Self { solutions, bases, num_stages }
    }
    
    /// Get solution buffer for a stage.
    pub fn solution(&self, stage: usize) -> &SolutionBuffer {
        &self.solutions[stage]
    }
    
    pub fn solution_mut(&mut self, stage: usize) -> &mut SolutionBuffer {
        &mut self.solutions[stage]
    }
    
    /// Get basis buffer for a stage.
    pub fn basis(&self, stage: usize) -> &BasisBuffer {
        &self.bases[stage]
    }
    
    pub fn basis_mut(&mut self, stage: usize) -> &mut BasisBuffer {
        &mut self.bases[stage]
    }
}

/// Pool of trajectory buffers for parallel forward passes.
pub struct TrajectoryPool {
    /// One buffer per forward pass
    buffers: Vec<TrajectoryBuffer>,
}

impl TrajectoryPool {
    pub fn with_capacity(
        num_forward_passes: usize,
        num_stages: usize,
        stage_dimensions: &[StageDimensions],
    ) -> Self {
        let buffers = (0..num_forward_passes)
            .map(|_| TrajectoryBuffer::with_stages(num_stages, stage_dimensions))
            .collect();
        Self { buffers }
    }
    
    /// Get buffer for a specific forward pass.
    pub fn get(&mut self, forward_pass_idx: usize) -> &mut TrajectoryBuffer {
        &mut self.buffers[forward_pass_idx]
    }
}
```

### 4. SoA Analysis and Migration

With the formal buffer structure, we can analyze direct SoA migration.

#### Current AoS Layout (Array of Structs)

```rust
// Current: each Realization is a struct with Vec fields
struct Realization {
    deficit: Vec<f64>,           // [bus_0, bus_1, ...]
    thermal_generation: Vec<f64>, // [thermal_0, thermal_1, ...]
    final_storage: Vec<f64>,      // [hydro_0, hydro_1, ...]
    // ... 15+ fields
}

// Array of Realizations
realizations: Vec<Realization>  // [stage_0, stage_1, ...]
```

#### Proposed SoA Layout (Structure of Arrays)

```rust
/// SoA layout for trajectory data.
/// 
/// # Cache Efficiency
/// 
/// When iterating over all stages for a single field (common pattern):
/// - SoA: single contiguous memory access
/// - AoS: strided access with poor cache utilization
/// 
/// # Example: Accessing all final_storage across stages
/// 
/// ```rust
/// // SoA: contiguous
/// for stage in 0..num_stages {
///     let storage = trajectory.final_storage_stage(stage);
/// }
/// 
/// // Equivalent to:
/// let all_storage: &[f64] = &trajectory.final_storage; // contiguous!
/// ```
pub struct TrajectorySoA {
    // Primal variables: [stage][entity]
    deficit: Vec<f64>,           // [stages * buses]
    thermal_generation: Vec<f64>, // [stages * thermals]
    final_storage: Vec<f64>,      // [stages * hydros]
    spillage: Vec<f64>,
    turbined_flow: Vec<f64>,
    // ... all other fields
    
    // Dual variables
    water_value: Vec<f64>,
    marginal_cost: Vec<f64>,
    
    // Dimensions
    num_stages: usize,
    num_buses: usize,
    num_hydros: usize,
    num_thermals: usize,
}

impl TrajectorySoA {
    /// Get final storage for a specific stage.
    pub fn final_storage_stage(&self, stage: usize) -> &[f64] {
        let start = stage * self.num_hydros;
        &self.final_storage[start..start + self.num_hydros]
    }
    
    /// Get all final storage (contiguous, cache-friendly).
    pub fn all_final_storage(&self) -> &[f64] {
        &self.final_storage
    }
    
    /// Mutable access for solver to write into.
    pub fn final_storage_stage_mut(&mut self, stage: usize) -> &mut [f64] {
        let start = stage * self.num_hydros;
        &mut self.final_storage[start..start + self.num_hydros]
    }
}
```

#### SoA Migration Decision

**Analyze during this epic**:
1. Profile memory access patterns in forward/backward passes
2. Measure cache miss rates with AoS vs SoA
3. Estimate implementation effort

**Decision criteria**:
- If >10% speedup from SoA: implement
- If 5-10% speedup: implement if low effort
- If <5% speedup: document and defer

### 5. Application to Simulation Step

All buffer structures must work for simulation, not just training:

```rust
impl Simulation {
    pub fn run_with_buffers(
        &mut self,
        trajectory_pool: &mut TrajectoryPool,
        // ...
    ) -> SimulationResult {
        for scenario_idx in 0..self.num_scenarios {
            let trajectory = trajectory_pool.get(scenario_idx % pool_size);
            
            for stage in 0..self.num_stages {
                let buffer = trajectory.solution_mut(stage);
                self.solver.solve();
                self.solver.get_solution_into(buffer)?;
                // Process results...
            }
        }
    }
}
```

---

## Module Structure After Epic

```
src/memory/
├── mod.rs
├── pools/
│   ├── mod.rs
│   ├── cut_state_pool.rs    # CutStatePool with 1:1 relationship
│   └── slot_index.rs        # CutStateSlotId, linear indexing
├── buffers/
│   ├── mod.rs
│   ├── solution_buffer.rs   # SolutionBuffer, SolutionIndices
│   ├── basis_buffer.rs      # BasisBuffer
│   └── trajectory.rs        # TrajectoryBuffer, TrajectoryPool
├── soa/
│   ├── mod.rs
│   ├── trajectory_soa.rs    # TrajectorySoA (if implemented)
│   └── analysis.rs          # SoA migration analysis docs
└── preallocator.rs          # Upfront allocation from config
```

---

## Sprints

### [Sprint 1: Core Pool Infrastructure](./sprint-01/00-sprint-overview.md) (Week 1-2)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-050 | Design CutStateSlotId and linear indexing | 2 | ⬜ |
| T-051 | Implement CutStatePool | 5 | ⬜ |
| T-052 | Design SolutionBuffer with structured access | 3 | ⬜ |
| T-053 | Implement SolutionBuffer and BasisBuffer | 3 | ⬜ |
| T-054 | Implement get_solution_into/get_basis_into solver methods | 3 | ⬜ |
| T-055 | Implement TrajectoryBuffer and TrajectoryPool | 3 | ⬜ |

**Sprint 1 Points**: 19

### [Sprint 2: Integration and SoA Analysis](./sprint-02/00-sprint-overview.md) (Week 2-3)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-056 | Integrate CutStatePool into backward pass | 5 | ⬜ |
| T-057 | Integrate TrajectoryPool into forward pass | 5 | ⬜ |
| T-058 | Integrate buffers into simulation step | 3 | ⬜ |
| T-059 | SoA feasibility analysis and benchmarking | 3 | ⬜ |
| T-060 | (Conditional) Implement TrajectorySoA | 5 | ⬜ |
| T-061 | Verify zero-allocation hot paths with DHAT | 3 | ⬜ |

**Sprint 2 Points**: 24 (T-060 conditional)

---

## Estimated Effort

- **Duration**: 2 sprints (3 weeks)
- **Story Points**: 43 (38 if SoA deferred)
- **Risk Level**: High (memory management is critical)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pool sizing incorrect | Medium | Medium | Size from config, allow growth with warning |
| Buffer layout incompatible | Low | High | Design review before implementation |
| SoA migration too complex | Medium | Low | Make conditional, can defer |
| Performance regression | Medium | Medium | Benchmark every change |
| Simulation step forgotten | Low | High | Explicit ticket for simulation integration |

---

## Definition of Done

- [ ] **CutStatePool implemented** with 1:1 state-cut relationship
- [ ] **Slot indexing** uses `(iteration, forward_pass_idx)`
- [ ] **SolutionBuffer and BasisBuffer** with structured field access
- [ ] **Solver methods** `get_solution_into()` and `get_basis_into()`
- [ ] **TrajectoryBuffer and TrajectoryPool** implemented
- [ ] **Training loop integrated** with new pools/buffers
- [ ] **Simulation loop integrated** with new pools/buffers
- [ ] **SoA analysis complete** with decision documented
- [ ] Zero allocations in hot paths (verified with DHAT)
- [ ] Memory profile flat during training
- [ ] Golden tests pass
- [ ] Performance target met (+5-10% or no regression)
- [ ] All tests pass
