# AR-10: Pre-Study Nodes for Lag Initialization

**Status**: ⏳ NOT STARTED  
**Sprint**: Sprint 2 (State & Process)  
**Effort**: 2 days  
**Priority**: P1 (High)  
**Assignee**: TBD

---

## Context

AR models need historical lag values to start. Rather than special-casing the first stage, we use **multiple pre-study nodes** that represent historical periods. For AR(p), we need p pre-study nodes before the first study stage.

This design naturally represents lag history in the graph structure and simplifies state initialization logic.

**Why this matters**: Clean graph structure prevents special-case code in forward/backward passes and makes lag states explicit.

---

## Objective

Extend graph structure to support pre-study nodes that represent historical periods before the first study stage, enabling natural AR lag initialization.

---

## Acceptance Criteria

### Must Have

- [ ] `is_pre_study` flag in Node struct
- [ ] Graph validation: AR(p) requires ≥ p pre-study nodes
- [ ] Helper methods: `study_nodes()`, `pre_study_nodes()`, `first_study_node()`
- [ ] Pre-study nodes have fixed state (from initial_condition.lag_inflows)
- [ ] No uncertainty in pre-study nodes (deterministic)

### Should Have

- [ ] Graph visualization distinguishes pre-study nodes
- [ ] JSON schema updated for `graph.json`
- [ ] Backward compatibility: graphs without pre-study nodes still work

### Won't Have (Yet)

- Automatic pre-study node generation (user specifies explicitly)
- Multiple pre-study paths (single history for now)

---

## Implementation Tasks

### 1. Extend Node Struct (30 min)

```rust
// In src/graph.rs

#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    pub id: usize,
    pub stage: usize,

    /// New: Flag indicating this is a pre-study (historical) node
    #[serde(default)]
    pub is_pre_study: bool,

    // Existing fields...
    pub season_id: Option<usize>,
    pub children: Vec<usize>,
}

impl Node {
    /// Check if this node is part of the optimization (study period)
    pub fn is_study_node(&self) -> bool {
        !self.is_pre_study
    }
}
```

### 2. Add Graph Helper Methods (1 hour)

```rust
// In src/graph.rs

impl Graph {
    /// Get all pre-study nodes (historical period)
    pub fn pre_study_nodes(&self) -> Vec<&Node> {
        self.nodes.iter().filter(|n| n.is_pre_study).collect()
    }

    /// Get all study nodes (optimization period)
    pub fn study_nodes(&self) -> Vec<&Node> {
        self.nodes.iter().filter(|n| !n.is_pre_study).collect()
    }

    /// Get the first study node (where optimization begins)
    pub fn first_study_node(&self) -> Option<&Node> {
        self.study_nodes().into_iter().min_by_key(|n| n.stage)
    }

    /// Get number of pre-study stages (for AR lag initialization)
    pub fn num_pre_study_stages(&self) -> usize {
        self.pre_study_nodes().len()
    }

    /// Check if graph has sufficient pre-study history for AR(p)
    pub fn has_sufficient_history(&self, lag_order: usize) -> bool {
        self.num_pre_study_stages() >= lag_order
    }
}
```

### 3. Update Graph Validation (1.5 hours)

```rust
// In src/input_validation.rs

pub fn validate_graph_for_ar(
    graph: &Graph,
    recourse: &Recourse,
) -> Result<(), InputError> {
    // Check if any AR models exist
    let max_lag_order = if let Some(noise_models) = &recourse.noise_models {
        noise_models.iter()
            .filter(|m| matches!(m.noise_type, NoiseType::Autoregressive))
            .map(|m| m.lag_order.unwrap_or(0))
            .max()
            .unwrap_or(0)
    } else {
        0 // No AR models
    };

    if max_lag_order == 0 {
        return Ok(()); // No AR models, no pre-study nodes needed
    }

    // Validate sufficient pre-study nodes
    if !graph.has_sufficient_history(max_lag_order) {
        return Err(InputError::InsufficientPreStudyNodes {
            required: max_lag_order,
            found: graph.num_pre_study_stages(),
        });
    }

    // Validate pre-study node structure
    validate_pre_study_structure(graph)?;

    Ok(())
}

fn validate_pre_study_structure(graph: &Graph) -> Result<(), InputError> {
    let pre_study = graph.pre_study_nodes();

    if pre_study.is_empty() {
        return Ok(()); // No pre-study nodes
    }

    // Check: pre-study nodes form a chain (single path)
    for (i, node) in pre_study.iter().enumerate() {
        if i < pre_study.len() - 1 {
            // Not the last pre-study node: must have exactly 1 child
            if node.children.len() != 1 {
                return Err(InputError::InvalidPreStudyStructure {
                    node_id: node.id,
                    message: format!(
                        "Pre-study node {} must have exactly 1 child, found {}",
                        node.id,
                        node.children.len()
                    ),
                });
            }
        } else {
            // Last pre-study node: children are first study nodes
            if node.children.is_empty() {
                return Err(InputError::InvalidPreStudyStructure {
                    node_id: node.id,
                    message: "Last pre-study node must connect to study nodes".to_string(),
                });
            }
        }
    }

    // Check: pre-study stages come before study stages
    let max_pre_study_stage = pre_study.iter().map(|n| n.stage).max().unwrap();
    let min_study_stage = graph.study_nodes().iter().map(|n| n.stage).min().unwrap();

    if max_pre_study_stage >= min_study_stage {
        return Err(InputError::InvalidPreStudyStructure {
            node_id: 0,
            message: format!(
                "Pre-study stages must come before study stages (pre: {}, study: {})",
                max_pre_study_stage, min_study_stage
            ),
        });
    }

    Ok(())
}
```

### 4. Update Initial State Creation (1 hour)

```rust
// In src/initial_condition.rs

impl InitialCondition {
    /// Create initial state for pre-study nodes
    ///
    /// For AR(p), we need p pre-study nodes with known inflow realizations
    pub fn create_pre_study_states(
        &self,
        graph: &Graph,
    ) -> Result<HashMap<usize, Box<dyn State>>, InitialConditionError> {
        let pre_study_nodes = graph.pre_study_nodes();

        if pre_study_nodes.is_empty() {
            return Ok(HashMap::new());
        }

        // Create state for each pre-study node
        let mut states = HashMap::new();

        for (idx, node) in pre_study_nodes.iter().enumerate() {
            // For pre-study nodes, we use lag_inflows to determine state
            // Node at index 0 = oldest (t-p), node at index p-1 = most recent (t-1)

            let state = self.create_pre_study_state(idx, pre_study_nodes.len())?;
            states.insert(node.id, state);
        }

        Ok(states)
    }

    fn create_pre_study_state(
        &self,
        node_index: usize,
        total_pre_study: usize,
    ) -> Result<Box<dyn State>, InitialConditionError> {
        if let Some(lag_inflows) = &self.lag_inflows {
            // Build state with appropriate lag history
            // At node_index, we know inflows from [0..=node_index]

            let volumes = self.volumes.values().cloned().collect();
            let lags: Vec<Vec<f64>> = lag_inflows.iter()
                .map(|(resource, resource_lags)| {
                    // Take lags available up to this point
                    let available = node_index + 1;
                    resource_lags.iter()
                        .take(available.min(resource_lags.len()))
                        .copied()
                        .collect()
                })
                .collect();

            Ok(Box::new(StorageWithInflowState::new(volumes, lags)?))
        } else {
            // No lag inflows: use storage-only state
            let volumes = self.volumes.values().cloned().collect();
            Ok(Box::new(StorageState::new(volumes)))
        }
    }
}
```

### 5. Add Error Types (15 min)

```rust
// In src/error.rs

#[error("Insufficient pre-study nodes: AR requires {required}, found {found}")]
InsufficientPreStudyNodes {
    required: usize,
    found: usize,
},

#[error("Invalid pre-study structure at node {node_id}: {message}")]
InvalidPreStudyStructure {
    node_id: usize,
    message: String,
},
```

### 6. Tests (2 hours)

Comprehensive test suite.

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_pre_study_node_identification() {
    let graph = create_test_graph_with_pre_study();

    assert_eq!(graph.num_pre_study_stages(), 2);
    assert_eq!(graph.study_nodes().len(), 5);
}

#[test]
fn test_sufficient_history_check() {
    let graph = create_test_graph_with_pre_study(); // 2 pre-study nodes

    assert!(graph.has_sufficient_history(1)); // AR(1): OK
    assert!(graph.has_sufficient_history(2)); // AR(2): OK
    assert!(!graph.has_sufficient_history(3)); // AR(3): Not enough
}

#[test]
fn test_validation_insufficient_pre_study() {
    let graph = create_test_graph_with_pre_study(); // 2 pre-study nodes
    let recourse = create_ar3_recourse(); // Requires 3 pre-study nodes

    let result = validate_graph_for_ar(&graph, &recourse);
    assert!(result.is_err());
}

#[test]
fn test_validation_non_chain_pre_study() {
    // Pre-study node with multiple children
    let mut graph = create_test_graph_with_pre_study();
    graph.nodes[0].children = vec![1, 2]; // Branching in pre-study

    let result = validate_pre_study_structure(&graph);
    assert!(result.is_err());
}
```

---

## Documentation Requirements

### Code Documentation

- [ ] Explain pre-study node concept
- [ ] Document lag initialization strategy
- [ ] Provide graph structure examples

### User Documentation

- [ ] Update graph.json specification
- [ ] Explain how to create pre-study nodes
- [ ] Show example with AR(2)

---

## Files to Modify

### Core Implementation

- `src/graph.rs`: Extend Node, add helper methods
- `src/input_validation.rs`: Add graph validation for AR
- `src/initial_condition.rs`: Add pre-study state creation

### Schema

- `schemas/graph.schema.json`: Add is_pre_study field

### Tests

- `tests/test_graph.rs`: Add pre-study tests
- Create fixtures with pre-study nodes

---

## Dependencies

### Depends On

- AR-3 (Initial condition lag support)
- AR-7 (StorageWithInflowState)

### Blocks

- AR-11 (State transition with lag update)
- AR-15 (Forward pass AR integration)

---

## Technical Notes

### Pre-Study Node Design

**Key Insight**: Represent historical lags as actual nodes in the graph.

For AR(2) with 3 study stages:

```
[Pre-2] → [Pre-1] → [Study-1] → [Study-2] → [Study-3]
  t=-2      t=-1       t=0         t=1         t=2
```

**Benefits**:

1. No special-case logic in SDDP
2. State transitions uniform across all nodes
3. Clear visualization of lag history

### State at Pre-Study Nodes

Pre-study nodes have **fixed, known states**:

- Volumes: From initial_condition.volumes
- Lags: From initial_condition.lag_inflows[...][node_index]

No optimization happens at pre-study nodes (deterministic walk).

### Performance Considerations

- Pre-study nodes don't add cuts (no backward pass)
- Forward pass processes them quickly (no uncertainty)
- Memory: +p nodes per scenario tree

---

## Example

### graph.json with Pre-Study Nodes (AR(2))

```json
{
  "nodes": [
    { "id": 0, "stage": -2, "is_pre_study": true, "children": [1] },
    { "id": 1, "stage": -1, "is_pre_study": true, "children": [2, 3] },
    { "id": 2, "stage": 0, "children": [4, 5] },
    { "id": 3, "stage": 0, "children": [4, 5] },
    { "id": 4, "stage": 1, "children": [] },
    { "id": 5, "stage": 1, "children": [] }
  ]
}
```

---

## Success Metrics

- ✅ Pre-study nodes correctly identified
- ✅ Graph validation catches insufficient history
- ✅ Pre-study states created correctly
- ✅ Backward compatibility maintained (no pre-study = storage-only)

---

**Created**: 2025-01-10  
**Last Updated**: 2025-01-10  
**Previous Ticket**: AR-9 (AR stochastic process implementation)  
**Next Ticket**: AR-11 (State transition with lag update)
