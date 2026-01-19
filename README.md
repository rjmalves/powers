# POWE.RS v2 - Power Optimization for the World of Energy - in pure RuSt

> **Status**: Planning Phase - Ground-up rewrite in progress

## Overview

POWE.RS is a high-performance SDDP (Stochastic Dual Dynamic Programming) solver in Rust for hydrothermal dispatch optimization. This branch (`fresh-start`) represents a complete v2 rewrite focused on:

- **Production-grade data model** - Comprehensive specification for real-world power systems
- **HPC-ready architecture** - MPI parallelization, SIMD optimization, efficient memory layout
- **Modern Rust patterns** - Clean architecture with proper error handling and testing
- **Parity with DECOMP/NEWAVE** - Support for all major features of established solvers

## Documentation

The primary design document is [`docs/DATA_MODEL_SPECIFICATION.md`](docs/DATA_MODEL_SPECIFICATION.md) (~6300 lines), which covers:

| Section | Content |
|---------|---------|
| 1. Design Philosophy | Guiding principles, invariants, notation |
| 2. Mathematical Notation | LP formulation, constraints, variables |
| 3. Input Data Model | All JSON/Parquet input files with schemas |
| 4. State Space | Storage, AR model states, decision variables |
| 5. Algorithm | Forward/backward passes, cut management |
| 6. Output Files | Training, simulation, policy outputs |
| 7. Binary Formats | FlatBuffers schemas for cuts and solutions |
| 8. Validation | Five-phase validation pipeline |
| 9. Implementation Plan | Six-phase development roadmap |

## Implementation Phases

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1: Foundation** | Weeks 1-4 | Core data I/O, validation, project structure |
| **Phase 2: SDDP Core** | Weeks 5-8 | LP model building, HiGHS integration, training loop |
| **Phase 3: MPI/HPC** | Weeks 9-12 | Distributed computing, rank coordination |
| **Phase 4: Optimization** | Weeks 13-16 | SIMD, memory layout, parallel cut sharing |
| **Phase 5: Features** | Weeks 17-20 | FPHA, risk measures, advanced stopping rules |
| **Phase 6: Testing** | Weeks 21-24 | Validation against reference, documentation |

## Project Structure (Planned)

```
powers/
├── Cargo.toml                        # Workspace configuration
├── crates/
│   ├── powers-core/                  # Core data structures and algorithm
│   ├── powers-io/                    # JSON/Parquet/FlatBuffers I/O
│   ├── powers-solver/                # HiGHS integration, LP building
│   ├── powers-mpi/                   # MPI parallelization (optional feature)
│   └── powers-cli/                   # Command-line interface
├── docs/
│   └── DATA_MODEL_SPECIFICATION.md   # Complete specification
├── schemas/
│   └── penalties.schema.json         # JSON Schema for penalties
└── examples/
    └── penalties.example.json        # Example penalty configuration
```

## Key Design Decisions

### Penalty System
- **Deficit is always piecewise** - Multiple cost tiers with final infinite segment for LP feasibility
- **Three-tier cascade** - `penalties.json` → entity JSON overrides → parquet stage overrides
- **Operational costs vs violation penalties** - Clear separation (e.g., `exchange_cost` vs `deficit_cost`)

### Data Model
- **Declaration order invariance** - Results are independent of entity ordering in JSON files
- **Sparse override pattern** - Time-varying bounds/penalties only need rows that differ from base
- **FlatBuffers for binary data** - Zero-copy deserialization for cuts and solutions

### Algorithm
- **Single-cut first** - Robust single-cut implementation before multi-cut
- **HiGHS solver** - Open-source LP solver with warm-starting support
- **Deterministic reproducibility** - Seeded RNG, canonical ordering

## Getting Started

This is currently a planning branch. To contribute:

1. Read [`docs/DATA_MODEL_SPECIFICATION.md`](docs/DATA_MODEL_SPECIFICATION.md)
2. Review the implementation phases in Section 9
3. Check open issues for tasks

## License

MIT License - see [LICENSE](LICENSE)
