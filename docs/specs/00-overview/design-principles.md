---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §1 (1.1-1.4)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Design Principles

## Purpose

This spec defines the foundational design principles governing the POWE.RS data model: format selection criteria, key design goals, the critical declaration order invariance requirement, and the reference to the LP subproblem formulation. All other data model and architecture specs build on these principles.

## 1. Format Selection Criteria

| Data Type                          | Recommended Format | Rationale                                                                         |
| ---------------------------------- | ------------------ | --------------------------------------------------------------------------------- |
| Configuration & Parameters         | JSON               | Human-readable, easily editable, small size                                       |
| Entity Registries                  | JSON               | Structured objects with relationships                                             |
| Time Series Data                   | Parquet            | Columnar, compressed, efficient for large data                                    |
| Policy Data (Cuts/States/Vertices) | FlatBuffers        | Zero-copy deserialization, cache-friendly dense arrays, in-memory during training |
| Simulation Results                 | Parquet            | High volume, per-entity indexing                                                  |
| Dictionaries/Metadata              | CSV                | Human-readable, small, universal                                                  |

## 2. Key Design Goals

1. **Separation of Concerns**: Static system data vs. dynamic algorithm data vs. stochastic data
2. **Scalability**: File formats that scale to production sizes without memory explosion
3. **Reproducibility**: All inputs deterministically produce same outputs
4. **Warm-start Support**: Efficient serialization/deserialization of algorithm state
5. **Distributed I/O**: Rank 0 loads, broadcasts to workers (or parallel loading where beneficial)
6. **Declaration Order Invariance**: Results must not depend on the order entities are declared in input files

## 3. Declaration Order Invariance (Critical Requirement)

> **⚠️ CRITICAL**: The optimization results MUST be identical regardless of the order in which entities are declared in input files. This is a fundamental correctness requirement.

**Principle**: If a user declares hydros A and B, runs the program, then exchanges the declaration order (B before A), the numerical results must be **bit-for-bit identical** (given the same random seed and same IDs).

**What determines identity**: The **entity ID** is the sole identifier. Two runs are equivalent if:

- All entity IDs are the same
- All entity properties are the same
- All relationships (by ID) are the same
- The random seed is the same

**What must NOT affect results**:

- Order of entities in JSON arrays (`hydros`, `thermals`, `buses`, `lines`)
- Order of rows in Parquet tables
- Order of constraints in `generic_constraints.json`
- Order of stages in `stages.json` (sorted by ID internally)
- Order of blocks within a stage (sorted by ID internally)
- Order of correlation blocks or entities within correlation blocks

### 3.1 Implementation Requirements

1. **Canonical Ordering**: After loading, all entity collections must be sorted by ID before any processing
2. **Deterministic Iteration**: All iterations over entities must use the canonical (sorted by ID) order
3. **LP Variable Ordering**: LP variables must be created in canonical order (by entity ID, then by block ID)
4. **LP Constraint Ordering**: LP constraints must be added in canonical order
5. **Random Number Generation**: Scenario generation must iterate entities in canonical order
6. **Cut Coefficients**: Cut coefficient ordering must follow the canonical state variable order

> **Note**: During implementation, one can assume that all the inputs are mostly sorted, so use sorting algorithms that have grater performance if all the inputs are already sorted by ID, ideally returning in few cycles if the data is already sorted by ID.

### 3.2 Validation Requirements

The test suite must include order-invariance tests that:

1. Run the same case with entities in different declaration orders
2. Verify bit-for-bit identical results (costs, decisions, cuts)

### 3.3 Canonical Ordering Example

```rust
// Canonical ordering example
impl System {
    /// Sort all entity collections by ID for order-invariant processing
    pub fn canonicalize(&mut self) {
        self.buses.sort_by_key(|b| b.id);
        self.lines.sort_by_key(|l| l.id);
        self.hydros.sort_by_key(|h| h.id);
        self.thermals.sort_by_key(|t| t.id);
    }
}

impl GenericConstraints {
    /// Sort constraints by ID for order-invariant processing
    pub fn canonicalize(&mut self) {
        self.constraints.sort_by_key(|c| c.id);
    }
}
```

### 3.4 Why This Matters

- **Debugging**: Users can reorganize input files without worrying about result changes
- **Version Control**: Reordering entities for readability doesn't create spurious diffs in results
- **Correctness**: Non-deterministic behavior from ordering is a bug, not a feature
- **Parallelism**: MPI ranks must agree on ordering without communication

## 4. LP Subproblem Formulation Reference

> **Complete Formulation**: See [MATHEMATICAL_FORMULATIONS.md](../../MATHEMATICAL_FORMULATIONS.md) for the authoritative mathematical specification of the SDDP algorithm and LP subproblem.

This data model specification focuses on **data structures and file formats**. The complete LP formulation, including:

- SDDP algorithm (forward/backward passes, convergence)
- Objective function and constraints
- Hydro production function models (constant, FPHA)
- Block formulation variants (parallel, chronological)
- Stochastic inflow modeling (PAR(p))
- Cut generation and aggregation
- Risk measures and advanced features

is documented in the mathematical formulations document.

**Key Cross-References**:

| This Document                       | Mathematical Formulations                                        | Description                 |
| ----------------------------------- | ---------------------------------------------------------------- | --------------------------- |
| hydros.json → productivity          | [Hydro Production Models](../01-math/hydro-production-models.md) | Constant productivity model |
| hydros.json → fpha\_\*              | [Hydro Production Models](../01-math/hydro-production-models.md) | FPHA coefficients           |
| config.json → block_mode            | [Block Formulations](../01-math/block-formulations.md)           | Block formulation variant   |
| scenarios/inflow_models.parquet     | [PAR Inflow Model](../01-math/par-inflow-model.md)               | PAR(p) model parameters     |
| config.json → inflow_non_negativity | [Inflow Non-Negativity](../01-math/inflow-nonnegativity.md)      | Inflow treatment method     |
| stages.json → transitions           | [Discount Rate](../01-math/discount-rate.md)                     | Discount rate               |
| policy/cuts/                        | [Cut Management](../01-math/cut-management.md)                   | Cut coefficients            |

**Variable/Constraint Sizing**: See [Production Scale Reference](./production-scale-reference.md) for production-scale LP dimensions.

## Cross-References

- [Notation Conventions](./notation-conventions.md) — Mathematical notation and symbol definitions used across all specs
- [Production Scale Reference](./production-scale-reference.md) — Production-scale LP dimensions and performance targets
- [LP Formulation](../01-math/lp-formulation.md) — Complete LP subproblem formulation
- [Input Directory Structure](../02-data-model/input-directory-structure.md) — File layout implementing these format choices
- [Validation Architecture](../03-architecture/validation-architecture.md) — Validation of order invariance and other requirements
