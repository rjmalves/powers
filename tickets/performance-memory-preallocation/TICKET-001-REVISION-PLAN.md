# TICKET-001 REVISION PLAN

**Decision**: Implement **Option 1: Per-Node Sizing** (Accurate but Complex)  
**Rationale**: Performance-critical system requires accurate sizing for effective pre-allocation  
**Date**: 2025-11-10  
**Status**: Ready for Implementation

---

## Decision Rationale

### Why Option 1 Over Simpler Approaches

**Option 1 (Per-Node Sizing)** chosen because:

1. **Accuracy is Critical**: POWE.RS is performance-focused. 50-300% estimation error is unacceptable
2. **Pre-allocation Depends on Accuracy**: Cannot effectively pre-allocate with wrong sizes
3. **Future-Proof**: Per-node data enables sophisticated optimizations in Phase 2-3
4. **Heterogeneity is Real**: Profiling shows 2.5x variation in state dimensions across nodes
5. **Pay Now, Save Later**: +1 day now prevents 2-3 days debugging in Phase 4

### Trade-offs Accepted

✅ **Accepting**:
- More complex API (but clearer semantics)
- Larger SizingInfo struct (~8KB vs ~120 bytes)
- Graph traversal required (but only at startup, O(n) cost)

✅ **Gaining**:
- <20% memory estimation error (vs 50-300%)
- Stage-aware buffer allocation (2-3x less waste)
- Validation hooks for Phase 4
- Foundation for advanced optimizations

---

## Revised SizingInfo Design

### New Structure

```rust
/// Per-node sizing information for heterogeneous SDDP graphs.
///
/// POWE.RS allows different nodes to have:
/// - Different state implementations (StorageState vs StorageAndInflowState)
/// - Different AR orders per hydro
/// - Different scenario counts
///
/// This struct captures these heterogeneous dimensions for accurate buffer pre-allocation.
pub struct NodeSizing {
    /// Node ID in the graph
    pub node_id: usize,
    
    /// State vector dimension for this node
    /// - StorageState: num_hydros
    /// - StorageAndInflowState: num_hydros + sum(AR_orders)
    pub state_dimension: usize,
    
    /// Number of scenarios branching from this node
    pub num_scenarios: usize,
    
    /// Number of decision variables in subproblem
    pub subproblem_var_count: usize,
    
    /// Number of constraints (excluding cuts)
    pub subproblem_constraint_count: usize,
    
    /// State choice: "storage" or "storage_and_inflow"
    pub state_choice: String,
}

pub struct SizingInfo {
    // ========================================
    // PER-NODE DIMENSIONS (Heterogeneous)
    // ========================================
    
    /// Sizing information for each node in the graph.
    /// Index matches node_id.
    pub node_sizing: Vec<NodeSizing>,
    
    // ========================================
    // AGGREGATE STATISTICS (Derived)
    // ========================================
    
    /// Maximum state dimension across all nodes
    pub max_state_dimension: usize,
    
    /// Minimum state dimension across all nodes
    pub min_state_dimension: usize,
    
    /// Average state dimension across all nodes
    pub avg_state_dimension: f64,
    
    /// Maximum scenarios branching from any node
    pub max_scenarios_per_node: usize,
    
    /// Maximum subproblem variables across all nodes
    pub max_subproblem_vars: usize,
    
    // ========================================
    // SYSTEM-WIDE DIMENSIONS (Uniform)
    // ========================================
    
    /// Number of hydroelectric plants
    pub num_hydros: usize,
    
    /// Number of thermal plants
    pub num_thermals: usize,
    
    /// Number of electrical buses
    pub num_buses: usize,
    
    /// Number of transmission lines
    pub num_lines: usize,
    
    /// Number of decision stages
    pub num_stages: usize,
    
    /// Total nodes in scenario tree
    pub num_nodes: usize,
    
    // ========================================
    // TRAINING/SIMULATION DIMENSIONS
    // ========================================
    
    /// Maximum SDDP iterations
    pub max_iterations: usize,
    
    /// Forward passes per iteration
    pub num_forward_passes: usize,
    
    /// Out-of-sample simulation scenarios
    pub num_simulations: usize,
    
    /// Worker thread count
    pub num_threads: usize,
}
```

---

## Implementation Tasks

### Task 1: Update SizingInfo Struct (2 hours)

**File**: `src/memory/sizing.rs`

```rust
// Add NodeSizing struct
pub struct NodeSizing {
    pub node_id: usize,
    pub state_dimension: usize,
    pub num_scenarios: usize,
    pub subproblem_var_count: usize,
    pub subproblem_constraint_count: usize,
    pub state_choice: String,
}

// Update SizingInfo with per-node data
pub struct SizingInfo {
    pub node_sizing: Vec<NodeSizing>,
    pub max_state_dimension: usize,
    pub min_state_dimension: usize,
    pub avg_state_dimension: f64,
    // ... rest of fields
}
```

### Task 2: Implement Per-Node Computation (3 hours)

**File**: `src/memory/sizing.rs`

```rust
impl SizingInfo {
    pub fn from_input(
        system: &System,
        graph: &DirectedGraph<NodeData>,  // Changed: now uses NodeData
        config: &Config,
    ) -> Self {
        // Compute per-node sizing
        let node_sizing: Vec<NodeSizing> = graph
            .iter_nodes()
            .map(|node| {
                let state_dim = compute_state_dimension_for_node(
                    system.meta.hydros_count,
                    &node.data.state_choice,
                    &node.data.uncertainty_models,
                );
                
                NodeSizing {
                    node_id: node.id,
                    state_dimension: state_dim,
                    num_scenarios: node.data.num_scenarios,
                    subproblem_var_count: compute_variable_count(
                        system.meta.hydros_count,
                        system.meta.thermals_count,
                        system.meta.buses_count,
                    ),
                    subproblem_constraint_count: compute_constraint_count(
                        system.meta.hydros_count,
                        system.meta.buses_count,
                        system.meta.lines_count,
                    ),
                    state_choice: node.data.state_choice.clone(),
                }
            })
            .collect();
        
        // Compute aggregates
        let max_state_dimension = node_sizing
            .iter()
            .map(|ns| ns.state_dimension)
            .max()
            .unwrap_or(0);
        
        let min_state_dimension = node_sizing
            .iter()
            .map(|ns| ns.state_dimension)
            .min()
            .unwrap_or(0);
        
        let avg_state_dimension = node_sizing
            .iter()
            .map(|ns| ns.state_dimension as f64)
            .sum::<f64>() / node_sizing.len() as f64;
        
        // ... rest of construction
    }
}
```

### Task 3: Add Helper Function for Per-Node State Dimension (1 hour)

```rust
/// Computes state dimension for a specific node.
///
/// Handles both StorageState and StorageAndInflowState.
fn compute_state_dimension_for_node(
    num_hydros: usize,
    state_choice: &str,
    uncertainty_models: &[TemporalModel],
) -> usize {
    match state_choice {
        "storage" => num_hydros,
        "storage_and_inflow" => {
            let inflow_lags: usize = uncertainty_models
                .iter()
                .filter(|tm| matches!(
                    tm.entity_type,
                    crate::input::UncertaintyType::Inflow
                ))
                .map(|tm| tm.max_ar_order)
                .sum();
            
            num_hydros + inflow_lags
        }
        _ => {
            log::warn!("Unknown state_choice '{}', defaulting to storage", state_choice);
            num_hydros
        }
    }
}
```

### Task 4: Enhance Memory Estimation (2 hours)

```rust
impl SizingInfo {
    /// Estimates memory usage per node.
    pub fn estimate_memory_per_node(&self) -> Vec<usize> {
        self.node_sizing
            .iter()
            .map(|ns| {
                let cut_memory = self.estimate_cuts_for_node(ns.node_id) 
                    * (std::mem::size_of::<f64>() // objective
                       + ns.state_dimension * std::mem::size_of::<f64>()); // coeffs
                
                let state_memory = ns.state_dimension * std::mem::size_of::<f64>();
                
                cut_memory + state_memory
            })
            .collect()
    }
    
    /// Estimates total memory across all nodes.
    pub fn estimate_memory_bytes(&self) -> usize {
        self.estimate_memory_per_node().iter().sum()
    }
    
    /// Detailed memory breakdown by component.
    pub fn estimate_memory_detailed(&self) -> MemoryBreakdown {
        let cuts = self.estimate_memory_per_node().iter().sum();
        
        let trajectories = self.num_forward_passes
            * self.num_stages
            * self.max_subproblem_vars
            * std::mem::size_of::<f64>()
            * self.max_iterations;
        
        let thread_buffers = self.num_threads
            * self.max_state_dimension
            * std::mem::size_of::<f64>()
            * 10; // ~10 buffers per thread
        
        MemoryBreakdown {
            cuts,
            trajectories,
            thread_buffers,
            total: cuts + trajectories + thread_buffers,
        }
    }
}

pub struct MemoryBreakdown {
    pub cuts: usize,
    pub trajectories: usize,
    pub thread_buffers: usize,
    pub total: usize,
}
```

### Task 5: Implement Cut Estimation Heuristic (2 hours)

```rust
impl SizingInfo {
    /// Estimates number of cuts per node after stabilization.
    ///
    /// Uses exponential stabilization model:
    ///   cuts(t) = C_limit * (1 - exp(-t / tau))
    ///
    /// where:
    ///   - C_limit depends on state complexity
    ///   - tau = 5 iterations (empirical)
    pub fn estimate_cuts_for_node(&self, node_id: usize) -> usize {
        let ns = &self.node_sizing[node_id];
        
        // Base complexity: more state dimensions → more distinct cuts
        let base_cuts = 20.0;
        let complexity_factor = 0.3;
        let c_limit = base_cuts + complexity_factor * ns.state_dimension as f64;
        
        // Exponential approach to limit
        let tau = 5.0;
        let t = self.max_iterations as f64;
        let stabilized_cuts = c_limit * (1.0 - (-t / tau).exp());
        
        // Upper bound: without selection, all cuts survive
        let max_cuts = self.max_iterations * self.num_forward_passes;
        
        stabilized_cuts.min(max_cuts as f64) as usize
    }
    
    /// Estimates cuts assuming no cut selection (worst case).
    pub fn estimate_cuts_without_selection(&self) -> usize {
        self.max_iterations * self.num_forward_passes
    }
}
```

### Task 6: Add Accessor Methods (1 hour)

```rust
impl SizingInfo {
    /// Gets sizing for a specific node.
    pub fn node(&self, node_id: usize) -> Option<&NodeSizing> {
        self.node_sizing.get(node_id)
    }
    
    /// Gets state dimension for a specific node.
    pub fn state_dimension_for_node(&self, node_id: usize) -> Option<usize> {
        self.node(node_id).map(|ns| ns.state_dimension)
    }
    
    /// Checks if all nodes have uniform state dimensions.
    pub fn has_uniform_state_dimensions(&self) -> bool {
        self.max_state_dimension == self.min_state_dimension
    }
    
    /// Gets nodes by state choice.
    pub fn nodes_with_state_choice(&self, choice: &str) -> Vec<usize> {
        self.node_sizing
            .iter()
            .filter(|ns| ns.state_choice == choice)
            .map(|ns| ns.node_id)
            .collect()
    }
}
```

### Task 7: Update log_summary (1 hour)

```rust
impl SizingInfo {
    pub fn log_summary(&self) {
        log::info!("Buffer Sizing Information:");
        log::info!(
            "  System: {} hydros, {} thermals, {} buses, {} lines",
            self.num_hydros, self.num_thermals, self.num_buses, self.num_lines
        );
        
        log::info!(
            "  State dimensions: min={}, max={}, avg={:.1}",
            self.min_state_dimension,
            self.max_state_dimension,
            self.avg_state_dimension
        );
        
        // Show state choice distribution
        let storage_nodes = self.nodes_with_state_choice("storage").len();
        let storage_inflow_nodes = self.nodes_with_state_choice("storage_and_inflow").len();
        log::info!(
            "  State choices: {} storage, {} storage_and_inflow",
            storage_nodes, storage_inflow_nodes
        );
        
        log::info!(
            "  Graph: {} stages, {} nodes, max_scenarios={}",
            self.num_stages, self.num_nodes, self.max_scenarios_per_node
        );
        
        log::info!(
            "  Training: {} iterations, {} forward_passes",
            self.max_iterations, self.num_forward_passes
        );
        
        log::info!("  Parallelism: {} threads", self.num_threads);
        
        let memory = self.estimate_memory_detailed();
        log::info!("  Estimated memory: {} MB total", memory.total / 1_000_000);
        log::info!("    - Cuts: {} MB", memory.cuts / 1_000_000);
        log::info!("    - Trajectories: {} MB", memory.trajectories / 1_000_000);
        log::info!("    - Thread buffers: {} MB", memory.thread_buffers / 1_000_000);
    }
}
```

### Task 8: Update Tests (3 hours)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_heterogeneous_state_dimensions() {
        let system = make_test_system(3, 2, 4, 5);
        let graph = make_heterogeneous_graph(); // Mix of storage/storage_and_inflow
        let config = make_test_config(10, 4, Some(100), Some(4));
        
        let sizing = SizingInfo::from_input(&system, &graph, &config);
        
        // Check per-node dimensions
        assert_eq!(sizing.node_sizing.len(), graph.node_count());
        
        // Storage-only node
        let storage_node = sizing.node(0).unwrap();
        assert_eq!(storage_node.state_dimension, 3);
        assert_eq!(storage_node.state_choice, "storage");
        
        // Storage+inflow node
        let inflow_node = sizing.node(1).unwrap();
        assert!(inflow_node.state_dimension > 3);
        assert_eq!(inflow_node.state_choice, "storage_and_inflow");
        
        // Check aggregates
        assert_eq!(sizing.min_state_dimension, 3);
        assert!(sizing.max_state_dimension > 3);
        assert!(sizing.avg_state_dimension >= 3.0);
    }
    
    #[test]
    fn test_per_node_memory_estimation() {
        let sizing = make_realistic_sizing();
        
        let per_node = sizing.estimate_memory_per_node();
        assert_eq!(per_node.len(), sizing.num_nodes);
        
        // Different nodes should have different memory requirements
        let unique_values: HashSet<_> = per_node.iter().collect();
        assert!(unique_values.len() > 1, "Expected heterogeneous memory");
    }
    
    #[test]
    fn test_cut_estimation_heuristic() {
        let sizing = make_test_sizing(state_dim=390, iterations=32, fp=4);
        
        let estimated = sizing.estimate_cuts_for_node(0);
        let without_selection = sizing.estimate_cuts_without_selection();
        
        // With selection should stabilize below worst case
        assert!(estimated > 100, "Should have reasonable cuts");
        assert!(estimated < without_selection, "Selection should reduce cuts");
    }
    
    #[test]
    fn test_accessor_methods() {
        let sizing = make_heterogeneous_sizing();
        
        // Node accessor
        assert!(sizing.node(0).is_some());
        assert!(sizing.node(999).is_none());
        
        // State dimension accessor
        assert_eq!(
            sizing.state_dimension_for_node(0),
            Some(sizing.node_sizing[0].state_dimension)
        );
        
        // State choice filter
        let storage_nodes = sizing.nodes_with_state_choice("storage");
        assert!(!storage_nodes.is_empty());
    }
    
    #[test]
    fn test_memory_breakdown() {
        let sizing = make_realistic_sizing();
        let breakdown = sizing.estimate_memory_detailed();
        
        assert!(breakdown.cuts > 0);
        assert!(breakdown.trajectories > 0);
        assert!(breakdown.thread_buffers > 0);
        assert_eq!(
            breakdown.total,
            breakdown.cuts + breakdown.trajectories + breakdown.thread_buffers
        );
    }
}
```

---

## Updated Timeline

### Original TICKET-001: 2 days (COMPLETE)
✅ Basic SizingInfo implementation

### TICKET-001-REVISION: +1.5 days (NEW)

**Day 1** (6 hours):
- [ ] Task 1: Update SizingInfo struct (2h)
- [ ] Task 2: Implement per-node computation (3h)
- [ ] Task 3: Add helper for per-node state dimension (1h)

**Day 2** (6 hours):
- [ ] Task 4: Enhance memory estimation (2h)
- [ ] Task 5: Implement cut estimation heuristic (2h)
- [ ] Task 6: Add accessor methods (1h)
- [ ] Task 7: Update log_summary (1h)

**Day 3** (3 hours):
- [ ] Task 8: Update tests (3h)
- [ ] Validation: All tests pass
- [ ] Documentation: Update completion summary

**Total Revision Effort**: 1.5 days

---

## Success Criteria

### Functional Requirements
- [ ] SizingInfo captures per-node dimensions correctly
- [ ] Handles mixed StorageState/StorageAndInflowState nodes
- [ ] Computes aggregates (min/max/avg) correctly
- [ ] Memory estimation uses per-node data

### Quality Requirements
- [ ] All tests pass (existing + new)
- [ ] No clippy warnings
- [ ] Documentation complete
- [ ] Memory estimation error <20% (validated in Phase 4)

### Performance Requirements
- [ ] from_input() completes in <50ms (was <10ms, acceptable increase)
- [ ] No runtime allocations after construction
- [ ] SizingInfo size <10KB (acceptable for startup overhead)

---

## Integration Impact

### Phase 2 (TICKET-002: Buffer Pools)
✅ **Enhanced**: Can now allocate per-stage buffers
```rust
// Before: Single buffer size
let buffer = Buffer::new(sizing.state_dimension);

// After: Stage-aware allocation
let buffer = Buffer::new(sizing.node(stage_id).unwrap().state_dimension);
```

### Phase 2 (TICKET-005: BackwardPassBuffers)
✅ **Enhanced**: Can optimize per-stage
```rust
pub struct BackwardPassBuffers {
    // Per-stage buffers (different sizes)
    state_buffers: Vec<Vec<f64>>,  // Sized per node
}
```

### Phase 4 (TICKET-013: Profiling Validation)
✅ **Easier**: Better estimates to validate
```rust
let estimated = sizing.estimate_memory_detailed();
let actual = measure_with_massif();
let error = validate_estimate(estimated, actual);
assert!(error < 0.20);  // Should pass with per-node data
```

---

## Risk Mitigation

### Risk 1: Breaking Existing Code
**Mitigation**: 
- Keep backward compatibility with accessor methods
- Provide default fallbacks for missing node data
- Comprehensive test coverage

### Risk 2: Performance Regression
**Mitigation**:
- from_input() still O(n), just more work per node
- Only called once at startup
- Profile if >50ms on large systems

### Risk 3: Complexity Burden
**Mitigation**:
- Rich documentation with examples
- Helper methods hide complexity
- Clear API design

---

## Next Steps

1. **Update TICKET-001 status** to "In Revision"
2. **Create branch** `feature/sizing-info-per-node`
3. **Implement tasks** in order (Day 1 → 2 → 3)
4. **Run full test suite** after each day
5. **Update completion summary** when done
6. **Proceed to TICKET-002** with enhanced sizing

---

**Decision Approved**: Option 1 (Per-Node Sizing)  
**Ready to Implement**: Yes  
**Expected Completion**: +1.5 days from start  
**Phase 1 Total Revised**: 3.5 days (was 2 days, acceptable)
