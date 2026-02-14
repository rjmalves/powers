---
status: draft
review_priority: 2-high
source_sections:
  - "DATA_MODEL_SPECIFICATION.md §7.1 (Summary Table)"
  - "DATA_MODEL_SPECIFICATION.md §7.2 (FlatBuffers for Policy Data)"
  - "DATA_MODEL_SPECIFICATION.md §7.3 (Parquet Configuration)"
  - "DATA_MODEL_SPECIFICATION.md §5.2 (LP Subproblem Structure)"
  - "DATA_MODEL_SPECIFICATION.md §5.3 (FCF with Replication)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Initial extraction from DATA_MODEL_SPECIFICATION.md §7.1-7.3, §5.2-5.3"
---

# Binary Formats and Internal Structures

## Purpose

This spec defines the binary format decisions (JSON, Parquet, FlatBuffers) used across POWE.RS, the FlatBuffers schema for policy data, Parquet configuration for non-policy data, and the LP subproblem and future cost function (FCF) internal structures that drive format requirements.

For core algorithm Rust structs (system entities, constraints, stage/block definitions), see [Internal Structures](internal-structures.md). For the solver interface and LP scaling, see the solver abstraction spec (Epic 4).

## 1. Format Decision Framework

This framework is the authoritative reference for all format choices across the data model. Each data model spec references this when justifying per-file format choices.

| Data Nature            | Format         | Key Examples                                                            | Rationale                                                                |
| ---------------------- | -------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Registry / catalog     | JSON           | buses.json, hydros.json, thermals.json, lines.json                      | Structured objects with nested fields, human-editable                    |
| Time series            | Parquet        | inflow_models.parquet, hydro_geometry.parquet, fpha_hyperplanes.parquet | Columnar analytics, efficient I/O, partition pruning                     |
| Default-with-overrides | JSON + Parquet | penalties.json (base) + stage overrides (Parquet)                       | Hierarchical defaults in JSON, sparse stage-varying overrides in Parquet |
| Complex nested object  | JSON           | config.json, stages.json, constraints/\*.json                           | Deep nesting, optional sections, human-editable                          |
| Correlation / matrix   | JSON           | correlation.json                                                        | Sparse, small, human-reviewable                                          |
| Policy / binary        | FlatBuffers    | policy cuts, states, vertices, checkpoint data                          | Zero-copy deserialization, SIMD-friendly dense arrays                    |
| High-volume output     | Parquet        | simulation results, training outputs                                    | Columnar compression, partition pruning, analytics tooling               |
| Metadata / dictionary  | CSV / JSON     | variables.csv, entities.csv, codes.json                                 | Human-readable, small volume                                             |

### Format Selection Criteria

| Criterion           | JSON          | Parquet             | FlatBuffers           |
| ------------------- | ------------- | ------------------- | --------------------- |
| Human editable      | ✅            | ❌                  | ❌                    |
| Schema evolution    | Moderate      | Good                | Good                  |
| Compression ratio   | Low           | High (~4×)          | Moderate              |
| Random access       | ❌            | Column + row group  | Field-level           |
| Zero-copy load      | ❌            | ❌                  | ✅                    |
| Analytics tooling   | Limited       | Excellent           | Limited               |
| Dense array storage | Poor          | Poor (many columns) | ✅                    |
| Write frequency     | Config (once) | Output (streaming)  | Checkpoint (periodic) |

## 2. Format Summary by Category

| Data Category      | Read/Write | Format      | Rationale                     |
| ------------------ | ---------- | ----------- | ----------------------------- |
| Algorithm Config   | Read       | JSON        | Small, editable               |
| System Registry    | Read       | JSON        | Structured objects            |
| Stage/Block Def    | Read       | JSON        | Graph structure               |
| Uncertainty Models | Read       | JSON        | Complex nested                |
| Distributions      | Read       | JSON        | Parameters                    |
| Load Profiles      | Read       | Parquet     | Large, indexed                |
| Inflow History     | Read       | Parquet     | Time series                   |
| Policy Cuts        | Read/Write | FlatBuffers | Zero-copy, in-memory training |
| Policy States      | Read/Write | FlatBuffers | Zero-copy, in-memory training |
| Policy Vertices    | Read/Write | FlatBuffers | Zero-copy, in-memory training |
| Training Results   | Write      | Parquet     | Analytics-ready               |
| Simulation Detail  | Write      | Parquet     | Large volume                  |
| Dictionaries       | Write      | CSV         | Human-readable                |

## 3. FlatBuffers for Policy Data

> **Decision Date**: 2026-01-19

Policy data (cuts, states, vertices) has a unique access pattern:

| Characteristic                | Description                                           |
| ----------------------------- | ----------------------------------------------------- |
| **In-memory during training** | Entire cut pool lives in RAM, accessed every LP solve |
| **Checkpointed periodically** | Written only at checkpoint intervals                  |
| **High state dimension**      | 1120 coefficients per cut at production scale         |
| **Large volume**              | Up to 1.2M cuts totaling ~18.6 GB                     |

**Why not Parquet?** Using 1120 individual columns (`coefficient_0` through `coefficient_1119`) is inefficient for Parquet, which is optimized for columnar analytics, not dense fixed-size arrays.

**Why FlatBuffers?**

1. **Zero-copy deserialization**: Load directly into memory without parsing
2. **Cache-friendly layout**: Dense coefficient arrays optimal for SIMD
3. **Simple schema**: Flat structure maps directly to Rust structs
4. **Fast checkpoint writes**: Serialize directly from in-memory structures

### 3.1 FlatBuffers Schema

```flatbuffers
// File: schemas/policy.fbs
namespace powers.policy;

// Benders cut: θ ≥ intercept + Σᵢ coefficients[i] × state[i]
// where intercept = α - β'x̂ (pre-computed)
table BendersCut {
    cut_id: uint64;
    slot_index: uint32;          // LP row position (REQUIRED for reproducibility)
    iteration: uint32;
    forward_pass_idx: uint32;
    scenario_idx: uint32;
    intercept: double;           // α - β'x̂
    coefficients: [double];      // β (length = state_dimension)
    state_at_generation: [double]; // x̂ (for cut selection)
    is_active: bool = true;
    domination_count: uint32 = 0;
}

// All cuts for a single stage (active AND inactive for reproducibility)
table StageCuts {
    stage_id: uint32;
    state_dimension: uint32;
    capacity: uint32;            // Total preallocated slots
    warm_start_count: uint32;    // Slots [0..warm_start_count) from loaded policy
    cuts: [BendersCut];          // Length = populated_count
    active_cut_indices: [uint32]; // O(1) lookup during LP construction
    populated_count: uint32;
}

// Visited state for cut selection
table VisitedState {
    state_id: uint64;
    iteration: uint32;
    forward_pass_idx: uint32;
    scenario_idx: uint32;
    components: [double];        // Length = state_dimension
    dominating_cut_id: uint64;
    dominating_objective: double;
}

table StageStates {
    stage_id: uint32;
    state_dimension: uint32;
    states: [VisitedState];
}

// Vertex for inner approximation (upper bound / SIDP)
table Vertex {
    vertex_id: uint64;
    iteration: uint32;
    forward_pass_idx: uint32;
    scenario_idx: uint32;
    components: [double];        // Length = state_dimension
    upper_bound_value: double;
    lipschitz_constant: double;
}

table StageVertices {
    stage_id: uint32;
    state_dimension: uint32;
    vertices: [Vertex];
    stage_lipschitz: double;
}

// Policy metadata for resume/warm-start
table PolicyMetadata {
    version: string;
    powers_version: string;
    created_at: string;          // ISO 8601
    completed_iterations: uint32;
    last_forward_pass: uint32;
    final_lower_bound: double;
    best_upper_bound: double;
    max_iterations: uint32;
    forward_passes: uint32;
    warm_start_cuts: uint32;
    // capacity = warm_start_cuts + max_iterations × forward_passes
    rng_seed: uint64;
    rng_state: [uint64];         // Full RNG state for resume
    state_dimension: uint32;
    num_stages: uint32;
    config_hash: string;
    system_hash: string;
}

root_type StageCuts;
```

### 3.2 Policy Directory Structure

```
policy/
├── metadata.json               # Human-readable (JSON for editability)
├── state_dictionary.json       # State variable mapping
├── cuts/
│   ├── stage_000.bin          # FlatBuffers StageCuts
│   ├── stage_001.bin
│   └── ...
├── states/
│   ├── stage_000.bin          # FlatBuffers StageStates
│   └── ...
├── vertices/                   # Only if inner approximation enabled
│   ├── stage_000.bin          # FlatBuffers StageVertices
│   └── ...
└── basis/                      # Optional, solver-specific format
    └── ...
```

### 3.3 Encoding Guidelines

| Field Type                        | Encoding          | Rationale                       |
| --------------------------------- | ----------------- | ------------------------------- |
| `cut_id`, `state_id`, `vertex_id` | uint64            | Unique across all iterations    |
| `iteration`, `stage_id`           | uint32            | Sufficient for practical limits |
| `coefficients`, `components`      | `[double]` dense  | SIMD-friendly, no dictionary    |
| `is_active`                       | bool              | Bit-packed by FlatBuffers       |
| Timestamps                        | string (ISO 8601) | Human-readable in metadata      |

**Compression**: `.bin` (uncompressed, fast load) or `.bin.zst` (Zstd-compressed, archival/transfer).

### 3.4 Memory Layout Alignment

```rust
// Rust struct matching FlatBuffers layout for zero-copy access
#[repr(C, align(64))]  // Cache-line aligned
pub struct BendersCutData {
    pub cut_id: u64,
    pub iteration: u32,
    pub forward_pass_idx: u32,
    pub scenario_idx: u32,
    pub is_active: bool,
    pub domination_count: u32,
    _padding: [u8; 3],
    pub intercept: f64,
    pub coefficients_offset: usize,
    pub state_offset: usize,
}

// Coefficient storage: separate dense arrays for SIMD vectorization
pub struct CutCoefficients {
    // All cuts' coefficients packed contiguously:
    // [cut0_coef0, cut0_coef1, ..., cut1_coef0, ...]
    pub data: Vec<f64>,
    pub state_dimension: usize,
    pub num_cuts: usize,
}

impl CutCoefficients {
    #[inline]
    pub fn get_cut_coefficients(&self, cut_idx: usize) -> &[f64] {
        let start = cut_idx * self.state_dimension;
        &self.data[start..start + self.state_dimension]
    }
}
```

### 3.5 Checkpoint Reproducibility

> **Critical**: Checkpoint/resume must produce **bit-for-bit identical** results.

**Why complete state serialization matters:**

```
Run A: Fresh start, 50 iterations → generates cuts in slots [0..10000)
  → deactivates some via Level 1 selection → specific LP row structure
  → specific solver pivots → specific duals → specific new cuts

Run B: Resume from iteration 25 checkpoint
  → MUST reconstruct identical LP structure (same slots, coefficients, bounds)
  → identical pivots → identical duals → identical cuts → identical results
```

| Data                         | Must Serialize | Rationale                |
| ---------------------------- | -------------- | ------------------------ |
| All cuts (active + inactive) | Yes            | LP row structure         |
| Slot indices                 | Yes            | Row mapping              |
| is_active flags              | Yes            | Bound values             |
| Coefficients                 | Yes            | Cut geometry             |
| RNG state                    | Yes            | Scenario reproducibility |
| Solver basis                 | Recommended    | Exact warm-start         |

**Warm-start vs resume modes:**

| Mode         | Cut Loading       | RNG State              | Capacity              | Results                   |
| ------------ | ----------------- | ---------------------- | --------------------- | ------------------------- |
| `fresh`      | None              | From config seed       | max_iter × fwd_passes | Deterministic from seed   |
| `warm_start` | All from policy   | Fresh from config seed | loaded + new          | Different from original   |
| `resume`     | All + exact state | Restored               | Same as checkpoint    | **Bit-for-bit identical** |

## 4. LP Subproblem Structure

```rust
/// Subproblem for a (stage, block) pair
pub struct BlockSubproblem {
    pub stage_id: u32,
    pub block_id: u32,
    pub solver: Box<dyn LpSolver>,
    pub variables: VariableLayout,
    pub constraints: ConstraintLayout,
    pub state: Box<dyn State>,
    pub uncertainty_data: Vec<UncertaintyObservationData>,
    pub cut_constraint_slots: Vec<usize>,
}

/// Variable layout for O(1) extraction
pub struct VariableLayout {
    pub deficit: Range<usize>,
    pub direct_exchange: Option<Range<usize>>,
    pub reverse_exchange: Option<Range<usize>>,
    pub thermal_gen: Range<usize>,
    pub thermal_gen_segments: Vec<Range<usize>>,
    pub turbined_flow: Range<usize>,
    pub spillage: Range<usize>,
    pub storage_final: Range<usize>,
    pub storage_inter_block: Option<Range<usize>>,
    pub alpha: usize,  // Future cost variable
}
```

## 5. FCF with Replication

> **Cut Preallocation Strategy**: Full preallocation with dynamic capacity.
>
> - Bit-for-bit reproducibility: checkpoints restore exact LP state
> - Zero runtime allocation: no thread-safety concerns during parallel solve
> - Warm-start support: capacity = existing_cuts + new_training_cuts

```rust
/// Future Cost Function (replicated per MPI rank)
pub struct FutureCostFunction {
    pub cuts: CutPool,
    pub states: StatePool,
    pub stage_id: u32,
    pub sync_iteration: u32,  // For incremental sync
}

/// Cut pool with full preallocation
/// capacity = warm_start_cuts + (max_iterations × forward_passes)
pub struct CutPool {
    pub pool: Vec<BendersCut>,
    pub warm_start_count: usize,
    pub populated_count: usize,
    pub active_bitmap: BitVec,   // O(1) active lookup
    pub state_dimension: u32,
}

/// Single Benders cut
#[repr(C, align(64))]  // Cache-line aligned
pub struct BendersCut {
    pub id: u64,
    pub rhs: f64,
    pub iteration: u32,
    pub forward_pass_idx: u32,
    pub is_active: AtomicBool,
    pub domination_count: AtomicU32,
    pub slot_index: AtomicU32,
    _padding: [u8; 4],
    pub coefficients_offset: usize, // Into separate SIMD-aligned array
}
```

## 6. Parquet Configuration

```rust
/// Parquet writer settings for simulation and training outputs
pub struct ParquetConfig {
    pub compression: Compression::ZSTD(ZstdLevel::try_new(3).unwrap()),
    pub row_group_size: 100_000,
    pub enable_statistics: true,
    pub dictionary_enabled: true,
    pub dictionary_page_size_limit: 1_048_576,
}
```

## Cross-References

- [Internal Structures](internal-structures.md) — Core algorithm Rust structs (System, Hydro, Thermal, etc.)
- [Output Schemas](output-schemas.md) — Parquet column definitions for output files
- [Output Infrastructure](output-infrastructure.md) — Parquet compression options, production scale
- [Penalty System](penalty-system.md) — Penalty structs referenced by internal structures
- [Input Constraints](input-constraints.md) — Generic constraint definitions and policy directory
- [SDDP Algorithm](../01-math/sddp-algorithm.md) — Algorithm that produces/consumes cuts
- [Cut Management](../01-math/cut-management.md) — Cut selection strategies using cut pool
