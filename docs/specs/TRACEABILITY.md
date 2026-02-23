# Traceability Matrix

This document maps every section heading from the three source documents to the corresponding spec file in `docs/specs/`. It serves as the authoritative record of where content was extracted, merged, split, or intentionally omitted.

**Generated**: 2026-02-14
**Source Documents**:

| Document                                 |  Lines | Specs Derived                                                                    |
| ---------------------------------------- | -----: | :------------------------------------------------------------------------------- |
| `DATA_MODEL_SPECIFICATION.md`            | 10,637 | 15 specs across 00-overview, 02-data-model, 03-architecture, 04-hpc, 06-deferred |
| `MATHEMATICAL_FORMULATIONS.md`           |  4,812 | 16 specs across 00-overview, 01-math, 05-config, 06-deferred                     |
| `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md` |  6,961 | 14 specs across 03-architecture, 04-hpc                                          |

---

## 1. DATA_MODEL_SPECIFICATION.md

### §1 — Design Principles

| Section                                                  | Spec File                                                            | Notes |
| -------------------------------------------------------- | -------------------------------------------------------------------- | ----- |
| §1 Design Principles (top-level)                         | [00-overview/design-principles.md](00-overview/design-principles.md) |       |
| §1.1 Format Selection Criteria                           | [00-overview/design-principles.md](00-overview/design-principles.md) |       |
| §1.2 Key Design Goals                                    | [00-overview/design-principles.md](00-overview/design-principles.md) |       |
| §1.3 Declaration Order Invariance (Critical Requirement) | [00-overview/design-principles.md](00-overview/design-principles.md) |       |
| §1.4 LP Subproblem Formulation Reference                 | [00-overview/design-principles.md](00-overview/design-principles.md) |       |

### §2 — Production Scale Reference

| Section                                   | Spec File                                                                              | Notes |
| ----------------------------------------- | -------------------------------------------------------------------------------------- | ----- |
| §2 Production Scale Reference (top-level) | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |
| §2.1 State Dimension Estimates            | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |
| §2.1.1 State Variables and Dimension      | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |
| §2.2 Variable and Constraint Counts       | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |
| §2.2.1 Variable Count per Subproblem      | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |
| §2.2.2 Constraint Count per Subproblem    | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |
| §2.2.3 Counting Formulas (Exact)          | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |
| §2.2.4 Sizing Calculator Tool             | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |
| §2.3 Performance Expectations by Scale    | [00-overview/production-scale-reference.md](00-overview/production-scale-reference.md) |       |

### §3 — Input Data Model

| Section                                                                                                                                   | Spec File                                                                                | Notes                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| §3 Input Data Model (top-level)                                                                                                           | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) |                                                              |
| §3.1 Directory Structure                                                                                                                  | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) |                                                              |
| §3.2 Configuration (`config.json`)                                                                                                        | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) |                                                              |
| — MPI Configuration (HPC Parameters)                                                                                                      | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Block Mode Configuration                                                                                                                | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Inflow Non-Negativity Methods                                                                                                           | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Horizon Mode Configuration                                                                                                              | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Stopping Rules Configuration                                                                                                            | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Forward Pass Configuration                                                                                                              | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Backward Pass Configuration                                                                                                             | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Cut Formulation Configuration                                                                                                           | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Numerical Tolerances Configuration                                                                                                      | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Simulation Sampling Scheme Configuration                                                                                                | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Policy Directory Configuration                                                                                                          | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — Upper Bound Evaluation (Inner Approximation / SIDP)                                                                                     | [02-data-model/input-directory-structure.md](02-data-model/input-directory-structure.md) | Subsection of §3.2                                           |
| — SDDP Algorithm Variants (DEFERRED)                                                                                                      | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     | Deferred features extracted separately                       |
| — — 1. Markovian Policy Graphs                                                                                                            | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — — 2. Multi-Cut vs Single-Cut Formulation                                                                                                | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — — 3. Objective States (Inner Approximation for Price Processes)                                                                         | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — — 4. Belief States (Partially Observable MDPs)                                                                                          | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — — 5. Duality Handlers (Lagrangian Relaxation for MIP)                                                                                   | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — — 6. Risk-Adjusted Forward Passes                                                                                                       | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — — Extensibility Design                                                                                                                  | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| §3.2.1 Penalties and Costs                                                                                                                | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Global Penalty Defaults (`penalties.json`)                                                                                              | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Constraint Violation Categories                                                                                                         | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Penalty Configuration                                                                                                                   | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Bus Penalties                                                                                                                           | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Bus Penalties Schema (`constraints/bus_penalties.parquet`)                                                                              | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Hydro Penalties Schema (`constraints/hydro_penalties.parquet`)                                                                          | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Penalty Categories                                                                                                                      | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Penalty Semantics                                                                                                                       | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Penalty Resolution Logic                                                                                                                | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Negative Evaporation (Condensation) Handling                                                                                            | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Hydro Variables and Bounds Summary                                                                                                      | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Dead-Volume Filling Specifics                                                                                                           | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — LP Objective Function Impact                                                                                                            | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| — Hydro Water Balance Equation                                                                                                            | [02-data-model/penalty-system.md](02-data-model/penalty-system.md)                       |                                                              |
| §3.3 Buses (`system/buses.json`)                                                                                                          | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| — Bus Fields                                                                                                                              | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| — Deficit Segment Fields                                                                                                                  | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| §3.4 Lines (`system/lines.json`)                                                                                                          | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| — Line Fields                                                                                                                             | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| — Line Operative States                                                                                                                   | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| §3.5 Hydro Registry (`system/hydros.json`)                                                                                                | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         | Core schema only; extensions in separate spec                |
| — Diversion Channel Fields                                                                                                                | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| — Hydro Operative States                                                                                                                  | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| — Hydro LP Variables by State                                                                                                             | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| §3.5.1 Hydro Geometry (`system/hydro_geometry.parquet`)                                                                                   | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| §3.5.2 Hydro Production Models (`system/hydro_production_models.json`)                                                                    | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| — Production Model Types                                                                                                                  | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| — Constant Productivity Model                                                                                                             | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| — Linearized Head Model                                                                                                                   | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| — FPHA Model (Função de Produção Hidrelétrica Aproximada)                                                                                 | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| — Required Data by Model                                                                                                                  | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| — Transition Between Models                                                                                                               | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| §3.5.3 Hydro Production Data (`system/hydro_production_data.parquet`)                                                                     | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| §3.5.4 FPHA Hyperplanes (`system/fpha_hyperplanes.parquet`)                                                                               | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| §3.5.5 Pumping Stations (`system/pumping_stations.json`)                                                                                  | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| §3.5.6 Energy Contracts (`system/energy_contracts.json`)                                                                                  | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| — Contract Bounds (`constraints/contract_bounds.parquet`)                                                                                 | [02-data-model/input-hydro-extensions.md](02-data-model/input-hydro-extensions.md)       |                                                              |
| §3.5.7 Non-Controllable Generation Sources — 🚧 DEFERRED                                                                                  | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — Non-Controllable Generation Model                                                                                                       | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — LP Integration                                                                                                                          | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — Correlation with Inflows                                                                                                                | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| §3.5.8 Battery Storage — 🚧 DEFERRED                                                                                                      | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — LP Variables                                                                                                                            | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — Energy Balance Constraint                                                                                                               | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — Bus Balance Integration                                                                                                                 | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — State Variable in SDDP                                                                                                                  | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| — Battery Bounds (`constraints/battery_bounds.parquet`)                                                                                   | [06-deferred/deferred-features.md](06-deferred/deferred-features.md)                     |                                                              |
| §3.6 Thermal Registry (`system/thermals.json`)                                                                                            | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| — Thermal Operative States                                                                                                                | [02-data-model/input-system-entities.md](02-data-model/input-system-entities.md)         |                                                              |
| §3.7 Stage Definitions (`stages.json`)                                                                                                    | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| — Scenario Sampling Methods                                                                                                               | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| — Stage Field Reference                                                                                                                   | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| §3.8 Uncertainty Models (formerly `inflow_models.parquet`, now split: `inflow_seasonal_stats.parquet` + `inflow_ar_coefficients.parquet`) | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| — Inflow Models Schema                                                                                                                    | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| — Load Models Schema                                                                                                                      | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| §3.9 Initial Conditions (`initial_conditions.json`)                                                                                       | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — GNL Pipeline Initial Conditions                                                                                                         | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Inflow History Schema                                                                                                                   | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| §3.10 Load Factors by Block (`scenarios/load_factors.json`)                                                                               | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| §3.11 Exchange Factors by Block (`scenarios/exchange_factors.json`)                                                                       | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| §3.12 Correlation (`scenarios/correlation.json`)                                                                                          | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| — Correlation Profile Fields                                                                                                              | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| — Time-Varying Correlation Schedule                                                                                                       | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| — Correlation Input Options Summary                                                                                                       | [02-data-model/input-scenarios.md](02-data-model/input-scenarios.md)                     |                                                              |
| §3.13 Constraints (`constraints/`)                                                                                                        | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Thermal Bounds Schema                                                                                                                   | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Hydro Bounds Schema                                                                                                                     | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Line Bounds Schema                                                                                                                      | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — CEPEL Constraint Types Mapping                                                                                                          | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Variable Reference Syntax                                                                                                               | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Expression Grammar                                                                                                                      | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Constraint Definition (`constraints/generic_constraints.json`)                                                                          | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Constraint Fields                                                                                                                       | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Constraint Bounds (`constraints/constraint_bounds.parquet`)                                                                             | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — LP Integration                                                                                                                          | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Validation Rules                                                                                                                        | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| §3.14 Policy Directory (`policy/`)                                                                                                        | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Policy Directory Structure                                                                                                              | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Policy Modes                                                                                                                            | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — State Dictionary (`state_dictionary.json`)                                                                                              | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — State Dictionary Fields                                                                                                                 | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Compatibility Validation                                                                                                                | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Cuts Schema (`policy/cuts/stage_XXX.bin` — FlatBuffers)                                                                                 | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 | Also in [binary-formats.md](02-data-model/binary-formats.md) |
| — Cut Coefficient Sign Convention                                                                                                         | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — States Schema (`policy/states/stage_XXX.bin` — FlatBuffers)                                                                             | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Vertices Schema (`policy/vertices/stage_XXX.bin` — FlatBuffers)                                                                         | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Solver Basis Schema (`policy/basis/stage_XXX.bin`)                                                                                      | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Metadata (`policy/metadata.json`)                                                                                                       | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Metadata Fields                                                                                                                         | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Resume Validation                                                                                                                       | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Warm-Start Validation                                                                                                                   | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |
| — Reproducibility Guarantees                                                                                                              | [02-data-model/input-constraints.md](02-data-model/input-constraints.md)                 |                                                              |

### §4 — Output Data Model

| Section                                       | Spec File                                                                        | Notes                     |
| --------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------- |
| §4 Output Data Model (top-level)              | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.1 Directory Structure Overview             | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.2 Design Principles                        | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.2.1 Hive Partitioning Strategy             | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.2.2 Categorical Encoding                   | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.2.3 Constraint Violation Handling          | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.2.4 File Naming Conventions                | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.3 Categorical Code Definitions             | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.4 Dictionary Files                         | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.4.1 Bounds Dictionary                      | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.4.2 State Dictionary                       | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.4.3 Variables Metadata                     | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.4.4 Entities Metadata                      | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5 Simulation Output Schemas                | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.1 Costs                                  | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.2 Hydros                                 | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.3 Thermals                               | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.4 Exchanges                              | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.5 Buses                                  | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.6 Pumping Stations                       | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.7 Contracts                              | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.8 Batteries — 🚧 DEFERRED                | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               | Marked deferred in schema |
| §4.5.9 Non-Controllables — 🚧 DEFERRED        | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               | Marked deferred in schema |
| §4.5.10 Inflow Lags                           | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.5.11 Generic Violations                    | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.6 Training Output Schemas                  | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.6.1 Convergence Log                        | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.6.2 Iteration Timing                       | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.6.3 MPI Rank Timing                        | [02-data-model/output-schemas.md](02-data-model/output-schemas.md)               |                           |
| §4.7 Manifest Files                           | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.7.1 Simulation Manifest                    | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.7.2 Training Manifest                      | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.8 Metadata File (`training/metadata.json`) | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.9 MPI Direct Hive Partitioning             | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.9.1 Writing Strategy                       | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.9.2 Write Protocol                         | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.9.3 Failure Handling                       | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.9.4 Reading Partitioned Data               | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.10 Output Configuration                    | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.11 Production Scale Reference              | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.11.1 Typical Problem Dimensions            | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.11.2 Output Size Estimates                 | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.11.3 I/O Bandwidth Requirements            | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.12 Validation and Integrity                | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.12.1 Schema Validation                     | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.12.2 Data Integrity Checks                 | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |
| §4.12.3 Reproducibility Verification          | [02-data-model/output-infrastructure.md](02-data-model/output-infrastructure.md) |                           |

### §5 — Internal Data Structures

| Section                                                  | Spec File                                                                        | Notes                                                       |
| -------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| §5 Internal Data Structures (top-level)                  | [02-data-model/internal-structures.md](02-data-model/internal-structures.md)     |                                                             |
| §5.1 Core Algorithm Structures                           | [02-data-model/internal-structures.md](02-data-model/internal-structures.md)     |                                                             |
| §5.2 LP Subproblem Structure                             | [02-data-model/binary-formats.md](02-data-model/binary-formats.md)               | Merged with binary format decisions                         |
| §5.3 FCF with Replication                                | [02-data-model/binary-formats.md](02-data-model/binary-formats.md)               | Merged with binary format decisions                         |
| §5.4 Solver Interface Specification (top-level)          | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.1 Trait Hierarchy                                   | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.2 Core Solver Trait                                 | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.3 Pre-allocated Cut Constraint Design               | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.4 Solver Error Types                                | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.5 Solver-Specific Retry Logic                       | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.6 Dual Variable Normalization                       | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.7 Basis Storage for Warm-Starting                   | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.8 Compile-Time Solver Selection                     | [03-architecture/solver-abstraction.md](03-architecture/solver-abstraction.md)   |                                                             |
| §5.4.9 Thread-Local Solver Infrastructure                | [03-architecture/solver-workspaces.md](03-architecture/solver-workspaces.md)     | Split from solver abstraction                               |
| §5.4.10 HiGHS Implementation Guidelines                  | [03-architecture/solver-highs-impl.md](03-architecture/solver-highs-impl.md)     | Split into own spec                                         |
| §5.4.11 Uncertainty Observation Data (PAR Preprocessing) | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) | Merged into scenario generation (PAR preprocessing content) |
| §5.4.12 Backward Pass Warm-Start Pattern                 | [03-architecture/solver-workspaces.md](03-architecture/solver-workspaces.md)     | Merged into solver workspaces (warm-start content)          |
| §5.4.13 Hot-Path Execution Flow                          | [03-architecture/solver-highs-impl.md](03-architecture/solver-highs-impl.md)     | Merged into HiGHS impl (hot-path content)                   |
| §5.5 LP Scaling Specification                            | [03-architecture/solver-workspaces.md](03-architecture/solver-workspaces.md)     |                                                             |
| §5.5.1 Scaling Transformation                            | [03-architecture/solver-workspaces.md](03-architecture/solver-workspaces.md)     |                                                             |
| §5.5.2 Scaling Data Structures                           | [03-architecture/solver-workspaces.md](03-architecture/solver-workspaces.md)     |                                                             |
| §5.5.3 Scaling Impact on Cut Coefficients                | [03-architecture/solver-workspaces.md](03-architecture/solver-workspaces.md)     |                                                             |
| §5.5.4 Scaling Workflow Integration                      | [03-architecture/solver-workspaces.md](03-architecture/solver-workspaces.md)     |                                                             |
| §5.5.5 FlatBuffers Schema for Scaling Persistence        | [03-architecture/solver-workspaces.md](03-architecture/solver-workspaces.md)     |                                                             |

### §6 — MPI Communication Structures

| Section                                            | Spec File                                                                  | Notes                                                                            |
| -------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| §6 MPI Communication Structures (top-level)        | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md)               |                                                                                  |
| §6.1 Hybrid MPI+OpenMP Architecture Overview       | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md)               |                                                                                  |
| §6.1.1 Architecture Diagram                        | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md)               |                                                                                  |
| §6.1.2 Communication Pattern Summary               | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md)               |                                                                                  |
| §6.1.3 Key Design Decisions                        | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md)               |                                                                                  |
| §6.1.4 Deployment Configuration                    | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md)               |                                                                                  |
| §6.2 Message Structures                            | [04-hpc/communication-patterns.md](04-hpc/communication-patterns.md)       |                                                                                  |
| §6.3 Synchronization Points                        | [04-hpc/synchronization.md](04-hpc/synchronization.md)                     |                                                                                  |
| §6.4 Hierarchical Cut Aggregation                  | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.5 Backward Pass Computation Modes               | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.5.1 Mathematical Foundation                     | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.5.2 Sequential Mode (Default)                   | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.5.3 Pipelined Mode (Future Enhancement)         | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.5.4 Trade-off Analysis                          | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.5.5 Configuration                               | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.6 Intra-Node Shared Memory (MPI Windows)        | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.7 NUMA-Aware Memory Management                  | [04-hpc/memory-architecture.md](04-hpc/memory-architecture.md)             | SLURM template portion also in [slurm-deployment.md](04-hpc/slurm-deployment.md) |
| §6.8 Performance Monitoring                        | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.9 HPC Implementation Requirements (Critical)    | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md)                   |                                                                                  |
| §6.9.1 Cut Slot Management: Thread-Safety Analysis | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md)                   |                                                                                  |
| §6.9.2 NUMA-Aware FCF Allocation                   | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md)                   |                                                                                  |
| §6.9.3 False Sharing Prevention in Cut Evaluation  | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md)                   |                                                                                  |
| §6.9.4 Load Balancing Strategy                     | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md)                   |                                                                                  |
| §6.9.5 Asynchronous Checkpointing                  | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md)                   |                                                                                  |
| §6.9.7 Summary: Critical vs High-Priority Issues   | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md)                   | Note: §6.9.6 missing from source doc                                             |
| §6.10 Dynamic Work Distribution                    | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.10.1 Architecture Overview                      | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.10.2 Rank 0 Bottleneck Mitigation               | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.10.3 Worker Implementation                      | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.10.4 MPI Message Protocol                       | [04-hpc/work-distribution.md](04-hpc/work-distribution.md)                 |                                                                                  |
| §6.11 Shared Memory Scenario Storage               | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.11.1 Shared Memory Architecture                 | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.11.2 Deterministic Scenario Seeding             | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.11.3 Distributed Generation Protocol            | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.11.4 Memory Layout and Access Pattern           | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.12 Two-Level Cut Aggregation                    | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.12.1 Two-Level Reduction Architecture           | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.12.2 Implementation                             | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.12.3 Replicated Cut Selection                   | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.13 Reproducibility Guarantees                   | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.13.1 Reproducibility Mechanisms                 | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.13.2 Potential Reproducibility Issues           | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.13.3 Verification Protocol                      | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |
| §6.13.4 Configuration Requirements                 | [04-hpc/shared-memory-aggregation.md](04-hpc/shared-memory-aggregation.md) |                                                                                  |

### §7 — File Format Decisions

| Section                                        | Spec File                                                          | Notes |
| ---------------------------------------------- | ------------------------------------------------------------------ | ----- |
| §7 File Format Decisions (top-level)           | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |
| §7.1 Summary Table                             | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |
| §7.2 FlatBuffers for Policy Data               | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |
| §7.2.1 FlatBuffers Schema Definitions          | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |
| §7.2.2 File Structure                          | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |
| §7.2.3 FlatBuffers Encoding Guidelines         | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |
| §7.2.4 Memory Layout Alignment                 | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |
| §7.2.5 Checkpoint Reproducibility Requirements | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |
| §7.3 Parquet Configuration                     | [02-data-model/binary-formats.md](02-data-model/binary-formats.md) |       |

### §8 — Validation Requirements

| Section                                | Spec File                                                                                | Notes                       |
| -------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------- |
| §8 Validation Requirements (top-level) | [03-architecture/validation-architecture.md](03-architecture/validation-architecture.md) | Merged with ARCHITECTURE §6 |
| §8.1 Input Validation Phases           | [03-architecture/validation-architecture.md](03-architecture/validation-architecture.md) |                             |
| §8.2 Validation Error Types            | [03-architecture/validation-architecture.md](03-architecture/validation-architecture.md) |                             |

### §9 — Next Steps

| Section                                                 | Spec File | Notes                                                                   |
| ------------------------------------------------------- | --------- | ----------------------------------------------------------------------- |
| §9 Next Steps (top-level)                               | —         | **Intentionally omitted**: Implementation timeline, not a specification |
| §9.1 Implementation Timeline Overview                   | —         | Intentionally omitted: Planning/schedule content                        |
| §9.2 Phase 1: Foundation (Weeks 1-4)                    | —         | Intentionally omitted                                                   |
| §9.3 Phase 2: SDDP Algorithm Core (Weeks 5-8)           | —         | Intentionally omitted                                                   |
| §9.4 Phase 3: MPI/HPC Foundation (Weeks 9-12)           | —         | Intentionally omitted                                                   |
| §9.5 Phase 4: HPC Optimization (Weeks 13-16)            | —         | Intentionally omitted                                                   |
| §9.6 Phase 5: Algorithm Features (Weeks 17-20)          | —         | Intentionally omitted                                                   |
| §9.7 Phase 6: Testing & Validation (Weeks 21-24)        | —         | Intentionally omitted                                                   |
| §9.8 Frontend Development (Parallel Track, Weeks 17-24) | —         | Intentionally omitted                                                   |
| §9.9 Validation Milestones Summary                      | —         | Intentionally omitted                                                   |
| §9.10 Risk Register                                     | —         | Intentionally omitted                                                   |

> **Justification**: §9 "Next Steps" is an implementation timeline and project plan, not a technical specification. It describes development phases, milestones, and a risk register — content that belongs in project management tooling, not in spec files. All technical content (validation requirements, testing criteria) was already captured in other sections.

---

## 2. MATHEMATICAL_FORMULATIONS.md

### Table of Contents and Part Headers

| Section           | Spec File | Notes                           |
| ----------------- | --------- | ------------------------------- |
| Table of Contents | —         | Structural element, not content |
| Part I–VI headers | —         | Structural element, not content |
| Appendices header | —         | Structural element, not content |

### §1 — Introduction

| Section                     | Spec File                                                                  | Notes                     |
| --------------------------- | -------------------------------------------------------------------------- | ------------------------- |
| §1 Introduction (top-level) | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md)                     |                           |
| §1.1 Document Purpose       | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md)                     |                           |
| §1.2 Notation Conventions   | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) | Split from §1 to overview |
| §1.3 Problem Context        | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md)                     |                           |

### §2 — SDDP Algorithm Overview

| Section                                            | Spec File                                              | Notes |
| -------------------------------------------------- | ------------------------------------------------------ | ----- |
| §2 SDDP Algorithm Overview                         | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.1 Multistage Stochastic Programming Formulation | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.2 The SDDP Algorithm                            | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.2.1 Forward Pass                                | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.2.2 Backward Pass                               | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.2.3 Convergence Monitoring                      | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.3 Policy Graph Structure                        | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.3.1 Finite Horizon (Acyclic Graph)              | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.3.2 Cyclic Graph (Infinite Horizon)             | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.4 State Variables and the Markov Property       | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |
| §2.5 Single-Cut vs Multi-Cut Formulation           | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |       |

### §3 — System Element Modeling Overview

| Section                                          | Spec File                                                | Notes                     |
| ------------------------------------------------ | -------------------------------------------------------- | ------------------------- |
| §3 System Element Modeling Overview              | [01-math/system-elements.md](01-math/system-elements.md) |                           |
| §3.1 System Architecture Overview                | [01-math/system-elements.md](01-math/system-elements.md) |                           |
| §3.2 Buses (Regional Subsystems)                 | [01-math/system-elements.md](01-math/system-elements.md) | All sub-headings included |
| §3.3 Transmission Lines                          | [01-math/system-elements.md](01-math/system-elements.md) | All sub-headings included |
| §3.4 Thermal Plants                              | [01-math/system-elements.md](01-math/system-elements.md) | All sub-headings included |
| §3.5 Hydro Plants                                | [01-math/system-elements.md](01-math/system-elements.md) | All sub-headings included |
| §3.6 Non-Controllable Generation Sources         | [01-math/system-elements.md](01-math/system-elements.md) | All sub-headings included |
| §3.7 Pumping Stations                            | [01-math/system-elements.md](01-math/system-elements.md) | All sub-headings included |
| §3.8 Import/Export Contracts                     | [01-math/system-elements.md](01-math/system-elements.md) | All sub-headings included |
| §3.9 Summary: Physical Elements to LP Components | [01-math/system-elements.md](01-math/system-elements.md) |                           |

### §4 — Notation and Sets

| Section                                                       | Spec File                                                                  | Notes |
| ------------------------------------------------------------- | -------------------------------------------------------------------------- | ----- |
| §4 Notation and Sets                                          | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.1 Index Sets                                               | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.2 Parameters                                               | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| — Time Conversion Factor Derivation                           | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.3 Decision Variables                                       | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.4 Dual Variables                                           | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.4.1 LP Formulation Strategy for Efficient Hot-Path Updates | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.4.2 Water Balance: LP Form                                 | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.4.3 AR Lag Constraints: LP Form                            | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.4.4 Cut Coefficient Derivation from Duals                  | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| — Storage Dual ($\pi^{wb}_h$)                                 | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| — AR Lag Dual ($\pi^{lag}_{h,\ell}$)                          | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.4.5 Summary Table                                          | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |
| §4.4.6 Implementation Notes                                   | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |       |

### §5 — Base LP Formulation

| Section                                   | Spec File                                              | Notes                       |
| ----------------------------------------- | ------------------------------------------------------ | --------------------------- |
| §5 Base LP Formulation (top-level)        | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.0 Cost and Penalty Taxonomy            | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.0.1 Cost Categories Overview           | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.0.2 Detailed Cost Definitions          | [01-math/lp-formulation.md](01-math/lp-formulation.md) | All sub-categories included |
| §5.0.3 Penalty Priority and Hierarchy     | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.0.4 Objective Function Structure       | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.1 Objective Function                   | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.2 Load Balance Constraint              | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.3 Hydro Water Balance                  | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.4 AR Inflow Dynamics                   | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.5 Hydro Generation Constraints         | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.6 Outflow Constraints                  | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.7 Minimum Constraints                  | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.8 Slack Penalties and Soft Constraints | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.9 Generic Constraints                  | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |
| §5.10 Benders Cuts                        | [01-math/lp-formulation.md](01-math/lp-formulation.md) |                             |

### §6 — Block Formulation Variants

| Section                                   | Spec File                                                      | Notes |
| ----------------------------------------- | -------------------------------------------------------------- | ----- |
| §6 Block Formulation Variants (top-level) | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| §6.1 Parallel Blocks (Default)            | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| — Water Balance (Parallel)                | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| — Characteristics                         | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| §6.2 Chronological Blocks                 | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| — Additional Variables                    | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| — Block 1 Water Balance                   | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| — Subsequent Blocks Water Balance         | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| — State Variable Definition               | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| — Dual Extraction for Cuts                | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| — Characteristics                         | [01-math/block-formulations.md](01-math/block-formulations.md) |       |
| §6.3 Comparison Summary                   | [01-math/block-formulations.md](01-math/block-formulations.md) |       |

### §7 — Hydro Production Function Models

| Section                                         | Spec File                                                                | Notes                                   |
| ----------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------- |
| §7 Hydro Production Function Models (top-level) | [01-math/hydro-production-models.md](01-math/hydro-production-models.md) |                                         |
| §7.1 Constant Productivity Model                | [01-math/hydro-production-models.md](01-math/hydro-production-models.md) |                                         |
| §7.2 FPHA (Four-Point Head Approximation)       | [01-math/hydro-production-models.md](01-math/hydro-production-models.md) | All sub-sections (6.2.1–6.2.9) included |
| §7.3 Linearized Head Model                      | [01-math/hydro-production-models.md](01-math/hydro-production-models.md) |                                         |
| §7.4 Model Selection Guidelines                 | [01-math/hydro-production-models.md](01-math/hydro-production-models.md) |                                         |
| §7.5 FPHA Data Requirements Summary             | [01-math/hydro-production-models.md](01-math/hydro-production-models.md) |                                         |

### §8 — Equipment-Specific Formulations

| Section                                        | Spec File                                                              | Notes                                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| §8 Equipment-Specific Formulations (top-level) | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) |                                                                                         |
| §8.1 Thermal Plants                            | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) |                                                                                         |
| §8.1.1 Standard Thermals                       | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) |                                                                                         |
| §8.1.2 GNL Thermals (DEFERRED)                 | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) | Marked as deferred, cross-refs [deferred-features.md](06-deferred/deferred-features.md) |
| §8.2 Transmission Lines                        | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) |                                                                                         |
| §8.3 Import/Export Contracts                   | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) |                                                                                         |
| §8.4 Pumping Stations                          | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) |                                                                                         |
| §8.5 Batteries (DEFERRED)                      | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) | Marked as deferred, cross-refs [deferred-features.md](06-deferred/deferred-features.md) |
| §8.6 Non-Controllable Sources (DEFERRED)       | [01-math/equipment-formulations.md](01-math/equipment-formulations.md) | Marked as deferred, cross-refs [deferred-features.md](06-deferred/deferred-features.md) |

### §9 — PAR(p) Inflow Model

| Section                                             | Spec File                                                  | Notes |
| --------------------------------------------------- | ---------------------------------------------------------- | ----- |
| §9 PAR(p) Inflow Model (top-level)                  | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.1 PAR(p) Model Definition                        | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.2 Notation for Fitting                           | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.3 Step 1: Seasonal Means and Standard Deviations | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.4 Step 2: Seasonal Autocorrelations              | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.5 Step 3: Yule-Walker Equations                  | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.6 Step 4: Convert to Original Units              | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.7 Step 5: Residual Standard Deviation            | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.8 Complete PAR(p) Parameter Set                  | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.9 Model Order Selection                          | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.10 CEPEL PAR(p)-A Variant (Future Extension)     | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |
| §9.11 Validation Checks                             | [01-math/par-inflow-model.md](01-math/par-inflow-model.md) |       |

### §10 — Inflow Non-Negativity Solution Methods

| Section                                    | Spec File                                                          | Notes |
| ------------------------------------------ | ------------------------------------------------------------------ | ----- |
| §10 Inflow Non-Negativity (top-level)      | [01-math/inflow-nonnegativity.md](01-math/inflow-nonnegativity.md) |       |
| §10.1 Problem Statement                    | [01-math/inflow-nonnegativity.md](01-math/inflow-nonnegativity.md) |       |
| §10.2 Method 1: None (`sem_relaxacao`)     | [01-math/inflow-nonnegativity.md](01-math/inflow-nonnegativity.md) |       |
| §10.3 Method 2: Penalty (`penalizacao`)    | [01-math/inflow-nonnegativity.md](01-math/inflow-nonnegativity.md) |       |
| §10.4 Method 3: Truncation (`truncamento`) | [01-math/inflow-nonnegativity.md](01-math/inflow-nonnegativity.md) |       |
| §10.5 Method 4: Truncation with Penalty    | [01-math/inflow-nonnegativity.md](01-math/inflow-nonnegativity.md) |       |
| §10.6 Comparison Summary                   | [01-math/inflow-nonnegativity.md](01-math/inflow-nonnegativity.md) |       |
| §10.7 Reference                            | [01-math/inflow-nonnegativity.md](01-math/inflow-nonnegativity.md) |       |

### §11 — Cut Generation and Aggregation

| Section                                        | Spec File                                              | Notes                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| §11 Cut Generation and Aggregation (top-level) | [01-math/cut-management.md](01-math/cut-management.md) |                                                                                           |
| §11.1 Dual Variable Extraction                 | [01-math/cut-management.md](01-math/cut-management.md) |                                                                                           |
| §11.2 Cut Coefficient Computation              | [01-math/cut-management.md](01-math/cut-management.md) |                                                                                           |
| §11.3 Single-Cut Aggregation                   | [01-math/cut-management.md](01-math/cut-management.md) |                                                                                           |
| §11.4 Multi-Cut Formulation (DEFERRED)         | [01-math/cut-management.md](01-math/cut-management.md) | Marked deferred, full details in [deferred-features.md](06-deferred/deferred-features.md) |
| §11.5 Cut Addition Algorithm                   | [01-math/cut-management.md](01-math/cut-management.md) |                                                                                           |
| §11.7 Cut Validity                             | [01-math/cut-management.md](01-math/cut-management.md) | Note: §11.6 absent from source                                                            |

### §12 — Cut Selection Strategies

| Section                                  | Spec File                                              | Notes                            |
| ---------------------------------------- | ------------------------------------------------------ | -------------------------------- |
| §12 Cut Selection Strategies (top-level) | [01-math/cut-management.md](01-math/cut-management.md) | Merged with §11 into single spec |
| §12.1 Motivation                         | [01-math/cut-management.md](01-math/cut-management.md) |                                  |
| §12.2 Cut Activity Definition            | [01-math/cut-management.md](01-math/cut-management.md) |                                  |
| §12.3 Level-1 Cut Selection              | [01-math/cut-management.md](01-math/cut-management.md) |                                  |
| §12.4 Limited Memory Level-1 (LML1)      | [01-math/cut-management.md](01-math/cut-management.md) |                                  |
| §12.5 Dominated Cut Detection            | [01-math/cut-management.md](01-math/cut-management.md) |                                  |
| §12.6 Threshold Parameter                | [01-math/cut-management.md](01-math/cut-management.md) |                                  |
| §12.7 Cut Selection Configuration        | [01-math/cut-management.md](01-math/cut-management.md) |                                  |
| §12.8 Convergence Guarantee              | [01-math/cut-management.md](01-math/cut-management.md) |                                  |
| §12.9 Reference                          | [01-math/cut-management.md](01-math/cut-management.md) |                                  |

### §13 — Stopping Rules Evaluation

| Section                                       | Spec File                                              | Notes |
| --------------------------------------------- | ------------------------------------------------------ | ----- |
| §13 Stopping Rules (top-level)                | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |
| §13.1 Available Stopping Rules                | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |
| §13.2 Iteration Limit (Mandatory)             | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |
| §13.3 Time Limit                              | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |
| §13.4 Statistical Stopping                    | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |
| §13.5 Bound Stalling                          | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |
| §13.6 Simulation-Based Stopping (Recommended) | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |
| §13.7 Combining Rules                         | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |
| §13.8 Output on Termination                   | [01-math/stopping-rules.md](01-math/stopping-rules.md) |       |

### §14 — Discount Rate Formulation

| Section                                         | Spec File                                            | Notes |
| ----------------------------------------------- | ---------------------------------------------------- | ----- |
| §14 Discount Rate Formulation (top-level)       | [01-math/discount-rate.md](01-math/discount-rate.md) |       |
| §14.1 Motivation                                | [01-math/discount-rate.md](01-math/discount-rate.md) |       |
| §14.2 Discounted Bellman Equation               | [01-math/discount-rate.md](01-math/discount-rate.md) |       |
| §14.3 Stage-Dependent Discount Rates            | [01-math/discount-rate.md](01-math/discount-rate.md) |       |
| §14.4 Modified Stage Subproblem                 | [01-math/discount-rate.md](01-math/discount-rate.md) |       |
| §14.5 Cumulative Discounting                    | [01-math/discount-rate.md](01-math/discount-rate.md) |       |
| §14.6 Lower Bound Computation with Discounting  | [01-math/discount-rate.md](01-math/discount-rate.md) |       |
| §14.7 Upper Bound (Simulation) with Discounting | [01-math/discount-rate.md](01-math/discount-rate.md) |       |
| §14.8 Implementation Notes                      | [01-math/discount-rate.md](01-math/discount-rate.md) |       |

### §15 — Infinite Periodic Horizon Formulation

| Section                                   | Spec File                                                  | Notes                                                  |
| ----------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| §15 Infinite Periodic Horizon (top-level) | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) | Split from discount-rate.md during review (2026-02-22) |
| §15.1 Motivation                          | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.2 Periodic Structure                  | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.3 Cycle Detection                     | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.4 Discounting for Convergence         | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.5 Cut Sharing Within Cycles           | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.6 Fixed-Point Iteration               | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.7 Modified Forward Pass               | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.8 Backward Pass Modifications         | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.9 Configuration                       | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |
| §15.10 Reference                          | [01-math/infinite-horizon.md](01-math/infinite-horizon.md) |                                                        |

### §16 — Upper Bound Evaluation LP

| Section                                | Spec File                                                              | Notes |
| -------------------------------------- | ---------------------------------------------------------------------- | ----- |
| §16 Upper Bound Evaluation (top-level) | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.1 Motivation                       | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.2 Vertex-Based Inner Approximation | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.3 Lipschitz Interpolation          | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.4 Lipschitz Constant Computation   | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.5 Vertex Value Computation         | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.6 Upper Bound Evaluation LP        | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.7 Linearized Upper Bound LP        | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.8 Gap Computation                  | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.9 Vertex Storage                   | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.10 Configuration                   | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.11 Computational Considerations    | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |
| §16.12 References                      | [01-math/upper-bound-evaluation.md](01-math/upper-bound-evaluation.md) |       |

### §17 — Risk-Averse SDDP (CVaR) Formulation

| Section                                           | Spec File                                            | Notes |
| ------------------------------------------------- | ---------------------------------------------------- | ----- |
| §17 Risk-Averse SDDP (top-level)                  | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.1 Motivation                                  | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.2 Conditional Value-at-Risk (CVaR)            | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.3 Convex Combination Risk Measure             | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.4 Dual Representation of Convex Risk Measures | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| — CVaR Dual Representation                        | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| — EAVaR Dual Representation                       | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.5 Risk-Averse Subgradient Theorem             | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.6 Risk-Averse Bellman Equation                | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.7 Cut Generation with Risk Measures           | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.8 Per-Stage Risk Profiles                     | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.9 Implementation Notes                        | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.10 Upper Bound with Risk Measures             | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.11 Reference                                  | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| §17.12 Lower Bound Validity with Risk Measures    | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| — Why the Lower Bound Fails                       | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| — What the "Lower Bound" Represents               | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| — Recommendations                                 | [01-math/risk-measures.md](01-math/risk-measures.md) |       |
| — Reference                                       | [01-math/risk-measures.md](01-math/risk-measures.md) |       |

### §18 — Configuration-Driven LP Variants

| Section                                          | Spec File                                                                    | Notes |
| ------------------------------------------------ | ---------------------------------------------------------------------------- | ----- |
| §18 Configuration-Driven LP Variants (top-level) | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.1 Block Mode Configuration                   | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.2 Hydro Production Function                  | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.3 Inflow Non-Negativity Treatment            | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.4 Cut Management                             | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.5 Discount Rate                              | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.6 Horizon Mode                               | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.7 Upper Bound Evaluation                     | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.8 Risk Measures                              | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.9 Penalty Coefficients                       | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |
| §18.10 Complete Example Configuration            | [05-config/configuration-reference.md](05-config/configuration-reference.md) |       |

### §19 — Cross-Reference to Data Model Specification

| Section                                       | Spec File                                                                    | Notes                       |
| --------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------- |
| §19 Cross-Reference to Data Model (top-level) | [05-config/configuration-reference.md](05-config/configuration-reference.md) | Merged with §18             |
| §19.1 Section Mapping                         | [05-config/configuration-reference.md](05-config/configuration-reference.md) |                             |
| §19.2 Variable Correspondence                 | [05-config/configuration-reference.md](05-config/configuration-reference.md) |                             |
| §19.3 Configuration Quick Reference           | [05-config/configuration-reference.md](05-config/configuration-reference.md) | All sub-references included |
| §19.4 Rust Struct Correspondence              | [05-config/configuration-reference.md](05-config/configuration-reference.md) |                             |

### Summary

| Section | Spec File                                                                    | Notes                               |
| ------- | ---------------------------------------------------------------------------- | ----------------------------------- |
| Summary | [05-config/configuration-reference.md](05-config/configuration-reference.md) | Merged into configuration reference |

### Appendix A — Notation Reference

| Section                               | Spec File                                                                  | Notes                   |
| ------------------------------------- | -------------------------------------------------------------------------- | ----------------------- |
| Appendix A Notation Reference         | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) | Merged with §1.2 and §4 |
| A.1 Index Sets                        | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |                         |
| A.2 State Variables                   | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |                         |
| A.3 Control Variables                 | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |                         |
| A.4 Parameters                        | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |                         |
| A.5 Dual Variables (Cut Coefficients) | [00-overview/notation-conventions.md](00-overview/notation-conventions.md) |                         |

### Appendix B — SDDP Algorithm Pseudocode

| Section                              | Spec File                                              | Notes                                |
| ------------------------------------ | ------------------------------------------------------ | ------------------------------------ |
| Appendix B SDDP Algorithm Pseudocode | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) | Merged into main SDDP algorithm spec |
| B.1 Main Training Loop               | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |                                      |
| B.2 Subproblem Solve                 | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |                                      |
| B.3 Cut Coefficient Computation      | [01-math/sddp-algorithm.md](01-math/sddp-algorithm.md) |                                      |

### Appendix C — Deferred Features

| Section                                           | Spec File                                                            | Notes |
| ------------------------------------------------- | -------------------------------------------------------------------- | ----- |
| Appendix C Deferred Features (top-level)          | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.1 GNL Thermal Plants                            | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.2 Battery Energy Storage Systems                | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.3 Multi-Cut Formulation                         | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.4 Markovian Policy Graphs                       | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.5 Non-Controllable Sources (Wind/Solar)         | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.6 FPHA Enhancements                             | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.6.1 Variable Efficiency Curves                  | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.6.2 Pumped Hydro Production Function            | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.6.3 Dynamic FPHA Recomputation                  | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7 Temporal Scope Decoupling                     | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.1 Motivation                                  | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.2 Three Independent Temporal Scopes           | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.3 Mathematical Formulation Impact             | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.4 Data Model Changes Required                 | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.5 LP Subproblem Size Impact                   | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.6 Implementation Considerations               | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.7 Use Cases and Configuration Examples        | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.8 Comparison to Existing Chronological Blocks | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.9 Open Questions and Design Decisions         | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |
| C.7.10 References and Related Work                | [06-deferred/deferred-features.md](06-deferred/deferred-features.md) |       |

---

## 3. PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md

### §1–3 — Program Lifecycle

| Section                                    | Spec File                                                                    | Notes |
| ------------------------------------------ | ---------------------------------------------------------------------------- | ----- |
| §1 Program Entrypoint and CLI Design       | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §1.1 Design Philosophy                     | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §1.2 Invocation Pattern                    | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §1.3 Command-Line Interface                | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §1.4 Exit Codes                            | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §2 Execution Phases Overview               | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §2.1 Phase Diagram                         | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §2.2 Phase Responsibilities                | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §2.3 Conditional Execution                 | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §3 Configuration Resolution and Validation | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §3.1 Configuration Hierarchy               | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |
| §3.2 Scheduler Integration                 | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |       |

### §4–5 — Input Processing

| Section                                 | Spec File                                                                              | Notes          |
| --------------------------------------- | -------------------------------------------------------------------------------------- | -------------- |
| §4 Input Loading Pipeline               | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                |
| §4.1 Loading Architecture               | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                |
| §4.2 File Loading Sequence              | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                |
| §4.3 Loader Interface                   | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                |
| §5 Dependency Resolution and Load Order | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) | Merged with §4 |
| §5.1 Dependency Graph                   | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                |
| §5.2 Conditional Loading                | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                |
| §5.3 Sparse Time-Series Handling        | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                |

### §6 — Validation Architecture

| Section                        | Spec File                                                                                | Notes                     |
| ------------------------------ | ---------------------------------------------------------------------------------------- | ------------------------- |
| §6 Validation Architecture     | [03-architecture/validation-architecture.md](03-architecture/validation-architecture.md) | Also merges DATA_MODEL §8 |
| §6.1 Validation Layers         | [03-architecture/validation-architecture.md](03-architecture/validation-architecture.md) |                           |
| §6.2 Error Collection Strategy | [03-architecture/validation-architecture.md](03-architecture/validation-architecture.md) |                           |
| §6.3 Validation Error Types    | [03-architecture/validation-architecture.md](03-architecture/validation-architecture.md) |                           |
| §6.4 Validation Report Format  | [03-architecture/validation-architecture.md](03-architecture/validation-architecture.md) |                           |

### §7 — Data Broadcasting

| Section                                   | Spec File                                                                              | Notes                               |
| ----------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------- |
| §7 Data Broadcasting                      | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) | Merged with §4-5 (loading pipeline) |
| §7.1 Broadcast Strategy                   | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                                     |
| §7.2 Serialization for Broadcast          | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                                     |
| §7.3 Parallel Policy Loading (Warm-Start) | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                                     |
| §7.4 Memory Layout After Broadcast        | [03-architecture/input-loading-pipeline.md](03-architecture/input-loading-pipeline.md) |                                     |

### §8–11 — Scenario Generation

| Section                                      | Spec File                                                                        | Notes |
| -------------------------------------------- | -------------------------------------------------------------------------------- | ----- |
| §8 PAR Model Preprocessing                   | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §8.1 PAR(p) Model Overview                   | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §8.2 Preprocessing Workflow                  | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §8.3 Memory Layout for Hot-Path Access       | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §8.5 PAR Model Fitting from Historical Data  | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §8.5.1 Fitting Overview                      | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §8.5.2 Yule-Walker Implementation            | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §8.5.3 Validation Requirements               | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §9 Noise Sampling and Correlation            | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §9.1 Correlated Noise Generation             | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §9.2 Reproducible Sampling                   | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §9.3 Noise Caching Strategy                  | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §10 External Scenario Integration            | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §10.1 External Scenario Sources              | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §10.2 Scenario Adapter Interface             | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §10.5 Noise Inversion for External Scenarios | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §10.5.1 The Inversion Problem                | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §10.5.2 Inversion Pipeline                   | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §10.5.3 Implementation                       | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §10.5.4 Validation Report                    | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §11 Scenario Memory Layout                   | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §11.1 Memory Organization                    | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §11.2 Two-Level Work Distribution            | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |
| §11.3 NUMA-Aware Allocation                  | [03-architecture/scenario-generation.md](03-architecture/scenario-generation.md) |       |

### §12–14 — Training Architecture

| Section                                    | Spec File                                                            | Notes           |
| ------------------------------------------ | -------------------------------------------------------------------- | --------------- |
| §12 Training Loop Structure                | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §12.1 SDDP Algorithm Overview              | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §12.2 Core Training Structures             | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §12.3 Trait Abstractions                   | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §13 Forward Pass Execution                 | [03-architecture/training-loop.md](03-architecture/training-loop.md) | Merged with §12 |
| §13.1 Forward Pass Overview                | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §13.2 Forward Pass Implementation          | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §13.3 State Management                     | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §13.4 Parallel Forward Execution           | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §14 Backward Pass Execution                | [03-architecture/training-loop.md](03-architecture/training-loop.md) | Merged with §12 |
| §14.1 Backward Pass Overview               | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §14.2 Backward Pass Implementation         | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §14.3 Dual Extraction for Cut Coefficients | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |
| §14.4 Parallel Backward Execution          | [03-architecture/training-loop.md](03-architecture/training-loop.md) |                 |

### §15 — Cut Management and Storage

| Section                                 | Spec File                                                                        | Notes |
| --------------------------------------- | -------------------------------------------------------------------------------- | ----- |
| §15 Cut Management and Storage          | [03-architecture/cut-management-impl.md](03-architecture/cut-management-impl.md) |       |
| §15.1 Future Cost Function Structure    | [03-architecture/cut-management-impl.md](03-architecture/cut-management-impl.md) |       |
| §15.2 Cut Selection Strategies          | [03-architecture/cut-management-impl.md](03-architecture/cut-management-impl.md) |       |
| §15.3 Cut Serialization for Checkpoints | [03-architecture/cut-management-impl.md](03-architecture/cut-management-impl.md) |       |
| §15.4 Cut Synchronization Across Ranks  | [03-architecture/cut-management-impl.md](03-architecture/cut-management-impl.md) |       |

### §16 — Convergence Monitoring

| Section                                  | Spec File                                                                              | Notes |
| ---------------------------------------- | -------------------------------------------------------------------------------------- | ----- |
| §16 Convergence Monitoring               | [03-architecture/convergence-monitoring.md](03-architecture/convergence-monitoring.md) |       |
| §16.1 Convergence Criteria               | [03-architecture/convergence-monitoring.md](03-architecture/convergence-monitoring.md) |       |
| §16.2 Convergence Monitor Implementation | [03-architecture/convergence-monitoring.md](03-architecture/convergence-monitoring.md) |       |
| §16.3 Bound Computation Details          | [03-architecture/convergence-monitoring.md](03-architecture/convergence-monitoring.md) |       |
| §16.4 Convergence Logging                | [03-architecture/convergence-monitoring.md](03-architecture/convergence-monitoring.md) |       |

### §17–19 — Simulation Architecture

| Section                                   | Spec File                                                                                | Notes           |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- | --------------- |
| §17 Policy Evaluation Mode                | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §17.1 Simulation Overview                 | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §17.2 Simulation Configuration            | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §17.3 Simulation Execution                | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §17.4 Simulation Statistics               | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §18 Non-Convex Extensions                 | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) | Merged with §17 |
| §18.1 Non-Convexity Sources               | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §18.2 Non-Convex Processing Pipeline      | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §18.3 Iterative Head-Dependent Refinement | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §19 Output Streaming                      | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) | Merged with §17 |
| §19.1 Streaming Architecture              | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §19.2 Output Writer Implementation        | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §19.3 Parquet Output Schema               | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |
| §19.4 Distributed Output Coordination     | [03-architecture/simulation-architecture.md](03-architecture/simulation-architecture.md) |                 |

### §20 — MPI+OpenMP Hybrid Strategy

| Section                           | Spec File                                                    | Notes |
| --------------------------------- | ------------------------------------------------------------ | ----- |
| §20 MPI+OpenMP Hybrid Strategy    | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md) |       |
| §20.1 Hybrid Parallelism Overview | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md) |       |
| §20.2 Design Rationale            | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md) |       |
| §20.3 Parallel Configuration      | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md) |       |
| §20.4 OpenMP FFI Bindings         | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md) |       |
| §20.5 Initialization Sequence     | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md) |       |
| §20.6 Build Configuration         | [04-hpc/hybrid-parallelism.md](04-hpc/hybrid-parallelism.md) |       |

### §21 — Work Distribution Patterns

| Section                                | Spec File                                                  | Notes |
| -------------------------------------- | ---------------------------------------------------------- | ----- |
| §21 Work Distribution Patterns         | [04-hpc/work-distribution.md](04-hpc/work-distribution.md) |       |
| §21.1 Forward Pass Distribution        | [04-hpc/work-distribution.md](04-hpc/work-distribution.md) |       |
| §21.2 Backward Pass Distribution       | [04-hpc/work-distribution.md](04-hpc/work-distribution.md) |       |
| §21.3 Work Distribution Implementation | [04-hpc/work-distribution.md](04-hpc/work-distribution.md) |       |

### §22 — Synchronization Architecture

| Section                                    | Spec File                                              | Notes |
| ------------------------------------------ | ------------------------------------------------------ | ----- |
| §22 Synchronization Architecture           | [04-hpc/synchronization.md](04-hpc/synchronization.md) |       |
| §22.1 Synchronization Points               | [04-hpc/synchronization.md](04-hpc/synchronization.md) |       |
| §22.2 Synchronization Summary              | [04-hpc/synchronization.md](04-hpc/synchronization.md) |       |
| §22.3 Thread Synchronization (Within Rank) | [04-hpc/synchronization.md](04-hpc/synchronization.md) |       |
| §22.4 Lock-Free Cut Aggregation            | [04-hpc/synchronization.md](04-hpc/synchronization.md) |       |

### §23 — Communication Patterns

| Section                                                 | Spec File                                                            | Notes                                                                                                |
| ------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| §23 Communication Patterns                              | [04-hpc/communication-patterns.md](04-hpc/communication-patterns.md) |                                                                                                      |
| §23.1 MPI Communication Summary                         | [04-hpc/communication-patterns.md](04-hpc/communication-patterns.md) |                                                                                                      |
| §23.2 MPI 4.0 Persistent Collectives — C Implementation | [04-hpc/communication-patterns.md](04-hpc/communication-patterns.md) |                                                                                                      |
| §23.3 Rust FFI Bindings for Persistent Collectives      | —                                                                    | **Intentionally omitted**: ferroMPI crate provides safe generic bindings; raw FFI wrapper not needed |
| §23.4 Hybrid Shared Memory Architecture                 | [04-hpc/communication-patterns.md](04-hpc/communication-patterns.md) |                                                                                                      |
| §23.5 Communication Volume Analysis                     | [04-hpc/communication-patterns.md](04-hpc/communication-patterns.md) |                                                                                                      |
| §23.6 Asynchronous Communication Overlap                | [04-hpc/communication-patterns.md](04-hpc/communication-patterns.md) |                                                                                                      |
| §23.7 Communication Performance Targets                 | [04-hpc/communication-patterns.md](04-hpc/communication-patterns.md) |                                                                                                      |

### §24 — Memory Architecture

| Section                                     | Spec File                                                      | Notes |
| ------------------------------------------- | -------------------------------------------------------------- | ----- |
| §24 Memory Architecture                     | [04-hpc/memory-architecture.md](04-hpc/memory-architecture.md) |       |
| §24.1 Memory Budget Overview                | [04-hpc/memory-architecture.md](04-hpc/memory-architecture.md) |       |
| §24.2 Memory Layout Strategy                | [04-hpc/memory-architecture.md](04-hpc/memory-architecture.md) |       |
| §24.3 NUMA-Aware Allocation                 | [04-hpc/memory-architecture.md](04-hpc/memory-architecture.md) |       |
| §24.4 Memory Pool for Temporary Allocations | [04-hpc/memory-architecture.md](04-hpc/memory-architecture.md) |       |

### §25–26 — Checkpointing and Output

| Section                               | Spec File                                          | Notes                          |
| ------------------------------------- | -------------------------------------------------- | ------------------------------ |
| §25 Checkpointing and Fault Tolerance | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §25.1 Checkpoint Strategy             | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §25.2 Checkpoint Implementation       | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §25.3 Warm-Start from Checkpoint      | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §25.5 Policy Persistence Architecture | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) | Note: §25.4 absent from source |
| §25.5.1 Policy Components             | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §25.5.2 File Format                   | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §25.5.3 Compatibility Validation      | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §25.5.4 Use Cases                     | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §26 Output Generation                 | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) | Merged with §25                |
| §26.1 Output Directory Structure      | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §26.2 Policy Output                   | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §26.3 Simulation Summary Output       | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |
| §26.4 Performance Logging             | [04-hpc/checkpointing.md](04-hpc/checkpointing.md) |                                |

### §27–29 — Extension Points

| Section                                         | Spec File                                                                  | Notes           |
| ----------------------------------------------- | -------------------------------------------------------------------------- | --------------- |
| §27 Trait Abstractions for Algorithm Variants   | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §27.1 Extensibility Architecture                | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §27.2 Core Trait Definitions                    | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §27.3 Factory Pattern for Configuration         | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §28 Risk Measure Implementations                | [03-architecture/extension-points.md](03-architecture/extension-points.md) | Merged with §27 |
| §28.1 Expected Value (Risk-Neutral)             | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §28.2 Conditional Value-at-Risk (CVaR)          | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §28.3 Convex Combination Risk                   | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §29 Horizon Mode Implementations                | [03-architecture/extension-points.md](03-architecture/extension-points.md) | Merged with §27 |
| §29.1 Finite Horizon                            | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §29.2 Infinite Horizon with Uniform Discounting | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §29.3 Infinite Horizon with Periodic Structure  | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |
| §29.4 Stage Configuration for Periodic Horizon  | [03-architecture/extension-points.md](03-architecture/extension-points.md) |                 |

### Appendix A — SLURM Job Script Patterns

| Section                                   | Spec File                                                | Notes |
| ----------------------------------------- | -------------------------------------------------------- | ----- |
| Appendix A SLURM Job Script Patterns      | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md) |       |
| A.1 Single-Node Job (Development/Testing) | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md) |       |
| A.2 Multi-Node Production Job             | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md) |       |
| A.3 Job Array for Parameter Studies       | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md) |       |

### Appendix B — Performance Monitoring Points

| Section                                  | Spec File                                                | Notes                                |
| ---------------------------------------- | -------------------------------------------------------- | ------------------------------------ |
| Appendix B Performance Monitoring Points | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md) | Merged with Appendix A in SLURM spec |
| B.1 Key Performance Counters             | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md) |                                      |
| B.2 Timing Breakdown                     | [04-hpc/slurm-deployment.md](04-hpc/slurm-deployment.md) |                                      |

### Appendix C — Execution Flow Diagrams

| Section                            | Spec File                                                                    | Notes                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Appendix C Execution Flow Diagrams | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) | Flow diagrams merged into CLI/lifecycle and training loop specs |
| C.1 Complete Execution Flow        | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |                                                                 |
| C.2 Data Flow Diagram              | [03-architecture/cli-and-lifecycle.md](03-architecture/cli-and-lifecycle.md) |                                                                 |

---

## 4. Line Count Report

| Spec File                                  |      Lines | Status                                                  |
| ------------------------------------------ | ---------: | ------------------------------------------------------- |
| 00-overview/design-principles.md           |        145 | ✅ OK                                                   |
| 00-overview/notation-conventions.md        |        398 | ✅ OK                                                   |
| 00-overview/production-scale-reference.md  |        268 | ✅ OK                                                   |
| 01-math/block-formulations.md              |        127 | ✅ OK                                                   |
| 01-math/cut-management.md                  |        295 | ✅ OK                                                   |
| 01-math/discount-rate.md                   |        180 | ✅ OK (split from 298; §15 → infinite-horizon.md)       |
| 01-math/equipment-formulations.md          |        180 | ✅ OK                                                   |
| 01-math/hydro-production-models.md         |        589 | ⚠️ Exceeds 500 lines                                    |
| 01-math/infinite-horizon.md                |        145 | ✅ OK (extracted from discount-rate.md §15, 2026-02-22) |
| 01-math/inflow-nonnegativity.md            |        191 | ✅ OK                                                   |
| 01-math/lp-formulation.md                  |        345 | ✅ OK                                                   |
| 01-math/par-inflow-model.md                |        199 | ✅ OK                                                   |
| 01-math/risk-measures.md                   |        274 | ✅ OK                                                   |
| 01-math/sddp-algorithm.md                  |        221 | ✅ OK                                                   |
| 01-math/stopping-rules.md                  |        250 | ✅ OK                                                   |
| 01-math/system-elements.md                 |        428 | ✅ OK                                                   |
| 01-math/upper-bound-evaluation.md          |        260 | ✅ OK                                                   |
| 02-data-model/binary-formats.md            |        379 | ✅ OK                                                   |
| 02-data-model/input-constraints.md         |        403 | ✅ OK                                                   |
| 02-data-model/input-directory-structure.md |        247 | ✅ OK                                                   |
| 02-data-model/input-hydro-extensions.md    |        418 | ✅ OK                                                   |
| 02-data-model/input-scenarios.md           |        353 | ✅ OK                                                   |
| 02-data-model/input-system-entities.md     |        381 | ✅ OK                                                   |
| 02-data-model/internal-structures.md       |        446 | ✅ OK                                                   |
| 02-data-model/output-infrastructure.md     |        475 | ✅ OK                                                   |
| 02-data-model/output-schemas.md            |        495 | ✅ OK                                                   |
| 02-data-model/penalty-system.md            |        242 | ✅ OK                                                   |
| 03-architecture/cli-and-lifecycle.md       |        177 | ✅ OK                                                   |
| 03-architecture/convergence-monitoring.md  |        288 | ✅ OK                                                   |
| 03-architecture/cut-management-impl.md     |        315 | ✅ OK                                                   |
| 03-architecture/extension-points.md        |        476 | ✅ OK                                                   |
| 03-architecture/input-loading-pipeline.md  |        286 | ✅ OK                                                   |
| 03-architecture/scenario-generation.md     |        490 | ✅ OK                                                   |
| 03-architecture/simulation-architecture.md |        495 | ✅ OK                                                   |
| 03-architecture/solver-abstraction.md      |        588 | ⚠️ Exceeds 500 lines                                    |
| 03-architecture/solver-highs-impl.md       |        448 | ✅ OK                                                   |
| 03-architecture/solver-workspaces.md       |        596 | ⚠️ Exceeds 500 lines                                    |
| 03-architecture/training-loop.md           |        473 | ✅ OK                                                   |
| 03-architecture/validation-architecture.md |        322 | ✅ OK                                                   |
| 04-hpc/checkpointing.md                    |        530 | ⚠️ Exceeds 500 lines                                    |
| 04-hpc/communication-patterns.md           |        316 | ✅ OK                                                   |
| 04-hpc/hybrid-parallelism.md               |        496 | ✅ OK                                                   |
| 04-hpc/memory-architecture.md              |        450 | ✅ OK                                                   |
| 04-hpc/shared-memory-aggregation.md        |        597 | ⚠️ Exceeds 500 lines                                    |
| 04-hpc/slurm-deployment.md                 |        449 | ✅ OK                                                   |
| 04-hpc/synchronization.md                  |        219 | ✅ OK                                                   |
| 04-hpc/work-distribution.md                |        445 | ✅ OK                                                   |
| 05-config/configuration-reference.md       |        331 | ✅ OK                                                   |
| 06-deferred/deferred-features.md           |        468 | ✅ OK                                                   |
| **Total**                                  | **17,562** |                                                         |

### Files Exceeding 500 Lines

| File                                  | Lines | Over By |
| ------------------------------------- | ----: | ------: |
| 04-hpc/shared-memory-aggregation.md   |   597 |      97 |
| 03-architecture/solver-workspaces.md  |   596 |      96 |
| 01-math/hydro-production-models.md    |   589 |      89 |
| 03-architecture/solver-abstraction.md |   588 |      88 |
| 04-hpc/checkpointing.md               |   530 |      30 |

All are within the 600-line hard limit.

---

## 5. Cross-Reference Validation

All markdown links (`[text](path.md)`) in spec files were validated against existing files.

### Broken Links

| File                                    | Line | Link Target                         | Issue                                                                 |
| --------------------------------------- | ---: | ----------------------------------- | --------------------------------------------------------------------- |
| 01-math/stopping-rules.md               |  116 | `risk-averse-sddp.md`               | File does not exist; should be `risk-measures.md`                     |
| 02-data-model/input-hydro-extensions.md |   25 | `../05-config/deferred-features.md` | Wrong directory; file is at `../06-deferred/deferred-features.md`     |
| 02-data-model/input-hydro-extensions.md |   95 | `../01-math/hydro-production.md`    | Incorrect filename; should be `../01-math/hydro-production-models.md` |
| 02-data-model/input-hydro-extensions.md |  407 | `../05-config/deferred-features.md` | Wrong directory; file is at `../06-deferred/deferred-features.md`     |
| 02-data-model/input-hydro-extensions.md |  408 | `../05-config/deferred-features.md` | Wrong directory; file is at `../06-deferred/deferred-features.md`     |
| 02-data-model/input-hydro-extensions.md |  415 | `../01-math/hydro-production.md`    | Incorrect filename; should be `../01-math/hydro-production-models.md` |
| 02-data-model/input-hydro-extensions.md |  417 | `../05-config/deferred-features.md` | Wrong directory; file is at `../06-deferred/deferred-features.md`     |
| 02-data-model/input-scenarios.md        |   24 | `../01-math/par-inflow.md`          | Incorrect filename; should be `../01-math/par-inflow-model.md`        |
| 02-data-model/input-scenarios.md        |  350 | `../01-math/par-inflow.md`          | Incorrect filename; should be `../01-math/par-inflow-model.md`        |
| 02-data-model/input-scenarios.md        |  352 | `../01-math/block-model.md`         | File does not exist; should be `../01-math/block-formulations.md`     |
| 02-data-model/output-schemas.md         |  387 | `../05-config/deferred-features.md` | Wrong directory; file is at `../06-deferred/deferred-features.md`     |
| 02-data-model/output-schemas.md         |  402 | `../05-config/deferred-features.md` | Wrong directory; file is at `../06-deferred/deferred-features.md`     |
| 02-data-model/output-schemas.md         |  495 | `../05-config/deferred-features.md` | Wrong directory; file is at `../06-deferred/deferred-features.md`     |

**Summary**: 13 broken links found across 4 spec files. All are filename or directory path errors — no missing content.

---

## 6. Gap Analysis

### Intentionally Omitted Sections

| Source Section                                                    | Justification                                                                                                                                                   |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DATA_MODEL §9 (Next Steps) — all subsections (§9.1–§9.10)         | Implementation timeline and project plan, not a technical specification. Includes phases, milestones, risk register — belongs in project management, not specs. |
| ARCHITECTURE §23.3 (Rust FFI Bindings for Persistent Collectives) | ferroMPI crate provides safe generic Rust bindings; raw C FFI wrappers are superseded and no longer needed. Noted in communication-patterns.md review_notes.    |

### Content Gaps Found

**None identified.** All technical specification content from all three source documents is accounted for in the traceability matrix. Every section is either:

1. Mapped to a specific spec file, or
2. Explicitly marked as intentionally omitted with justification

### Merge Summary

The following source sections were **merged** into a single spec rather than extracted standalone:

| Merged Sections                  | Target Spec                            | Rationale                                                          |
| -------------------------------- | -------------------------------------- | ------------------------------------------------------------------ |
| MATH §11 + §12                   | cut-management.md                      | Cut generation and cut selection are inseparable lifecycle phases  |
| MATH §14 + §15                   | discount-rate.md + infinite-horizon.md | Originally merged; split during review 2026-02-22                  |
| MATH §18 + §19 + Summary         | configuration-reference.md             | All configuration-driven content consolidated                      |
| MATH §1.2 + §4 + Appendix A      | notation-conventions.md                | All notation/symbol definitions unified                            |
| MATH §1.1/§1.3 + §2 + Appendix B | sddp-algorithm.md                      | Algorithm overview + pseudocode consolidated                       |
| DATA_MODEL §5.2 + §5.3 + §7      | binary-formats.md                      | LP structures drive format decisions                               |
| DATA_MODEL §8 + ARCH §6          | validation-architecture.md             | Same topic from two perspectives                                   |
| ARCH §4 + §5 + §7                | input-loading-pipeline.md              | Loading, dependencies, broadcasting are sequential pipeline        |
| ARCH §8 + §9 + §10 + §11         | scenario-generation.md                 | PAR preprocessing, noise, external, memory layout are one pipeline |
| ARCH §12 + §13 + §14             | training-loop.md                       | Training loop structure includes forward/backward passes           |
| ARCH §17 + §18 + §19             | simulation-architecture.md             | Simulation, non-convex, output are one execution flow              |
| ARCH §25 + §26                   | checkpointing.md                       | Checkpointing and output generation are related persistence topics |
| ARCH §27 + §28 + §29             | extension-points.md                    | All trait abstractions and implementations consolidated            |

### Split Summary

The following source sections were **split** across multiple specs:

| Source Section                     | Split Into                                                                                                                                                            | Rationale                                                            |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| DATA_MODEL §5.4 (Solver Interface) | solver-abstraction.md (§5.4.1–5.4.8), solver-workspaces.md (§5.4.9), solver-highs-impl.md (§5.4.10)                                                                   | Too large for a single spec; natural separation by abstraction level |
| DATA_MODEL §5.4.11                 | scenario-generation.md                                                                                                                                                | PAR preprocessing data belongs with scenario pipeline                |
| DATA_MODEL §5.4.12                 | solver-workspaces.md                                                                                                                                                  | Backward pass warm-start is workspace concern                        |
| DATA_MODEL §5.4.13                 | solver-highs-impl.md                                                                                                                                                  | Hot-path is HiGHS-specific implementation detail                     |
| DATA_MODEL §6 (MPI)                | hybrid-parallelism.md, communication-patterns.md, synchronization.md, shared-memory-aggregation.md, work-distribution.md, memory-architecture.md, slurm-deployment.md | 3,000+ line section split by topic                                   |
| DATA_MODEL §3.5 (Hydro)            | input-system-entities.md (core), input-hydro-extensions.md (§3.5.1–3.5.6)                                                                                             | Core registry vs. optional extension files                           |
