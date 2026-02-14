---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §6 (6.1-6.4)"
  - "DATA_MODEL_SPECIFICATION.md §8 (8.1-8.2)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted and merged from ARCHITECTURE §6 and DATA_MODEL §8"
---

# Validation Architecture

## Purpose

This spec defines the POWE.RS multi-layer input validation pipeline, covering the validation layer stack, the five-phase validation sequence, the error collection strategy, the typed error catalog, and the validation report format. It merges the architectural perspective (how validation fits in execution flow) with the data model perspective (what is validated when).

## 1. Validation Pipeline Overview

Validation runs on **rank 0 only** during the Validation phase (typically 1–10 s). It collects all errors before failing, so the user sees every problem in a single report rather than fixing issues one at a time.

The pipeline comprises five sequential phases. Each phase may depend on the output of the previous one (e.g., referential integrity checks require that schema validation has already confirmed field presence).

```
Phase 0          Phase 1         Phase 2           Phase 3          Phase 4
Canonicalize  →  Schema       →  Referential    →  Consistency   →  Semantic
(sort by ID)     (structure)     (foreign keys)    (dimensions)     (business rules)
```

## 2. Validation Layer Stack

The architecture organizes validation into four conceptual layers, from lowest (closest to raw I/O) to highest (domain-specific):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Validation Layer Stack                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Layer 4: Semantic Validation (Business Rules)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Storage min <= initial <= max                                          │   │
│  │ • AR order <= available history length                                   │   │
│  │ • Discount rate required for cycles                                      │   │
│  │ • Sum of block durations > 0 for each stage                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                           │
│  Layer 3: Referential Integrity                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Foreign key references valid (bus_id in hydros → buses)               │   │
│  │ • Cascade references form DAG (no cycles)                                │   │
│  │ • Stage IDs in time-series match stages.json                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                           │
│  Layer 2: Schema Validation                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • JSON conforms to JSON Schema                                           │   │
│  │ • Parquet columns have expected names and types                         │   │
│  │ • Required fields present                                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                           │
│  Layer 1: Structural Validation                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Files exist and are readable                                           │   │
│  │ • Valid JSON/Parquet format                                              │   │
│  │ • UTF-8 encoding                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Five-Phase Validation Sequence

### 3.0 Phase 0 — Canonicalization (Order Invariance)

Before any content validation, all entity collections are sorted into canonical order by ID. This guarantees that results are bit-for-bit identical regardless of the declaration order in input files (see [Design Principles](../00-overview/design-principles.md) §1.3).

- Sort all entity collections by ID (hydros, thermals, buses, lines)
- Sort stages by ID
- Sort blocks within stages by ID
- Sort generic constraints by ID
- Verify IDs are unique within each collection

### 3.1 Phase 1 — Schema Validation

Validates that every input file conforms to its expected structure:

- JSON Schema validation for config, system, and temporal files
- Parquet schema validation for constraints and checkpoint files
- Required field presence
- Type correctness (string vs. number vs. boolean vs. array)

### 3.2 Phase 2 — Referential Integrity

Validates cross-entity references (foreign keys):

- Bus IDs exist for lines, hydros, thermals
- Downstream hydro IDs exist (cascade topology)
- Model IDs in uncertainty models exist
- Stage IDs in transitions exist
- Season IDs match distribution definitions

### 3.3 Phase 3 — Business Rules

Domain-specific validation rules:

| Rule             | Description                                             |
| ---------------- | ------------------------------------------------------- |
| Acyclic cascade  | Hydro cascade graph is a DAG                            |
| Storage bounds   | `min ≤ initial ≤ max` for storage and generation        |
| Probability sums | Probabilities sum to 1.0                                |
| PSD correlation  | Correlation matrices are positive semi-definite         |
| AR stationarity  | AR coefficients ensure stationarity                     |
| Block weights    | Block weights sum to 1.0                                |
| Deficit segments | Monotonically increasing deficit cost segments          |
| Stopping rules   | At least one `iteration_limit` stopping rule present    |
| GNL lag          | `gnl_config.lag_stages` must be ≥ 1                     |
| GNL pipeline     | `thermal_id` must reference a thermal with `gnl_config` |
| GNL bounds       | `committed_mw` must be within thermal bounds            |

#### 3.3b Conditional Validation (mode-dependent)

Certain rules apply only under specific configuration modes:

| Condition                                      | Rules                                                                                                                                              |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `horizon.mode = "infinite_periodic"`           | At least one cycle in transitions; cycle transitions have `discount_rate > 0`; `max_horizon_length` specified                                      |
| `horizon.mode = "markovian"`                   | `markov_states` defined in `stages.json`; all transitions specify valid markov states; Markov transition probabilities sum to 1.0 per source state |
| `simulation.sampling_scheme.type = "external"` | `simulation/external_scenarios/` directory exists; `inflows.parquet` exists with correct schema                                                    |
| Thermal has `gnl_config`                       | `gnl_pipeline` entries cover all `lag_stages`                                                                                                      |

### 3.4 Phase 4 — Dimension Consistency

Cross-file dimensional checks:

- Load profiles cover all (stage, block, bus) combinations
- Inflow history covers all hydros with PAR models
- Seasonal parameters have correct length (`num_seasons`)
- Correlation matrix dimensions match entity count

### 3.5 Phase 5 — Warm-Start Compatibility

When loading a previously trained policy for warm-start:

- State dimension matches current system configuration
- Cut stage IDs exist in current stage graph
- Config hash matches (optional strict mode)

## 4. Error Collection Strategy

Validation collects **all errors** before failing, rather than failing on the first error. This allows users to fix every problem in a single iteration.

```rust
pub struct ValidationContext {
    errors: Vec<ValidationError>,
    warnings: Vec<ValidationWarning>,
    current_file: Option<PathBuf>,
    current_entity: Option<String>,
}

impl ValidationContext {
    /// Record an error without immediately failing
    pub fn error(&mut self, kind: ErrorKind, message: impl Into<String>) {
        self.errors.push(ValidationError {
            file: self.current_file.clone(),
            entity: self.current_entity.clone(),
            kind,
            message: message.into(),
        });
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Generate detailed report
    pub fn into_result(self) -> ValidationResult {
        ValidationResult {
            valid: self.errors.is_empty(),
            errors: self.errors,
            warnings: self.warnings,
        }
    }
}
```

## 5. Validation Error Type Catalog

### 5.1 Error Kind Summary

| Error Kind         | Severity | Description               | Example                     |
| ------------------ | -------- | ------------------------- | --------------------------- |
| `FileNotFound`     | Error    | Required file missing     | `hydros.json` not found     |
| `ParseError`       | Error    | Invalid JSON/Parquet      | Malformed JSON syntax       |
| `SchemaViolation`  | Error    | Schema mismatch           | Missing required field      |
| `InvalidReference` | Error    | Foreign key invalid       | `bus_id: 999` not in buses  |
| `DuplicateId`      | Error    | ID uniqueness violation   | Two hydros with same ID     |
| `InvalidValue`     | Error    | Value out of range        | `storage_max < storage_min` |
| `CycleDetected`    | Error    | Invalid graph structure   | Cascade forms cycle         |
| `MissingData`      | Warning  | Optional data absent      | No FPHA planes for hydro    |
| `UnusedEntity`     | Warning  | Entity defined but unused | Thermal not in any bus      |

### 5.2 Typed Error Enum

```rust
#[derive(Error, Debug)]
pub enum ValidationError {
    // Schema errors
    #[error("JSON schema validation failed for {file}: {details}")]
    JsonSchema { file: String, details: String },

    #[error("Parquet schema mismatch in {file}: expected {expected}, got {actual}")]
    ParquetSchema { file: String, expected: String, actual: String },

    // Reference errors
    #[error("{entity_type} {entity_id} references non-existent {ref_type} {ref_id}")]
    BrokenReference {
        entity_type: String,
        entity_id: u32,
        ref_type: String,
        ref_id: u32,
    },

    // Business rule errors
    #[error("Hydro cascade contains cycle: {cycle:?}")]
    CyclicCascade { cycle: Vec<u32> },

    #[error("Value out of range: {field} = {value}, expected [{min}, {max}]")]
    OutOfRange { field: String, value: f64, min: f64, max: f64 },

    #[error("Correlation matrix is not positive semi-definite for block {block}")]
    NotPositiveSemiDefinite { block: String },

    #[error("Missing required stopping rule: iteration_limit")]
    MissingIterationLimit,

    #[error("Invalid GNL configuration for thermal {thermal_id}: {details}")]
    InvalidGnlConfig { thermal_id: u32, details: String },

    #[error("GNL pipeline references thermal {thermal_id} without gnl_config")]
    GnlPipelineInvalidThermal { thermal_id: u32 },

    // Conditional validation errors
    #[error("Infinite periodic mode requires at least one cycle in transitions")]
    NoCycleInInfiniteMode,

    #[error("Cycle transitions must have discount_rate > 0 for infinite periodic mode")]
    MissingDiscountInCycle,

    #[error("Markovian mode requires markov_states definition in stages.json")]
    MissingMarkovStates,

    #[error("Markov transition probabilities from state {state} don't sum to 1.0: {sum}")]
    InvalidMarkovProbabilities { state: u32, sum: f64 },

    #[error("External sampling scheme requires simulation/external_scenarios/ directory")]
    MissingExternalScenarios,

    // Dimension errors
    #[error("Missing data for ({stage}, {block}, {entity}): expected {expected} rows")]
    MissingTimeSeries {
        stage: u32,
        block: u32,
        entity: String,
        expected: usize,
    },

    // Warm-start errors
    #[error("Warm-start state dimension mismatch: expected {expected}, got {actual}")]
    WarmstartDimensionMismatch { expected: u32, actual: u32 },
}
```

## 6. Validation Report Format

When validation completes (whether it passes or fails), POWE.RS emits a structured JSON report:

```json
{
  "valid": false,
  "timestamp": "2026-01-31T10:30:00Z",
  "case_directory": "/path/to/case",
  "errors": [
    {
      "file": "system/hydros.json",
      "entity": "hydro_042",
      "kind": "InvalidReference",
      "message": "bus_id 'BUS_99' not found in buses.json"
    },
    {
      "file": "scenarios/inflow_models.parquet",
      "entity": null,
      "kind": "MissingData",
      "message": "No PAR coefficients for hydro 'hydro_015' at stage 48"
    }
  ],
  "warnings": [
    {
      "file": "system/thermals.json",
      "entity": "thermal_old",
      "kind": "UnusedEntity",
      "message": "Thermal 'thermal_old' has max_generation=0 for all stages"
    }
  ],
  "summary": {
    "files_checked": 24,
    "entities_validated": 456,
    "error_count": 2,
    "warning_count": 1
  }
}
```

## Cross-References

- [CLI and Lifecycle](./cli-and-lifecycle.md) — Validation phase within the execution lifecycle; `--validate-only` mode
- [Input Loading Pipeline](./input-loading-pipeline.md) — Loading completes before validation begins; loader output feeds into validation context
- [Design Principles](../00-overview/design-principles.md) — Declaration order invariance (§1.3) enforced by Phase 0 canonicalization
- [Scenario Generation](./scenario-generation.md) — PAR model validation rules (AR stationarity, sufficient history)
