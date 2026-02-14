# POWE.RS v2 - Power Optimization for the World of Energy - in pure RuSt

> **Status**: Planning Phase - Ground-up rewrite in progress

## Overview

POWE.RS is a high-performance SDDP (Stochastic Dual Dynamic Programming) solver in Rust for hydrothermal dispatch optimization. This branch (`fresh-start`) represents a complete v2 rewrite focused on:

- **Production-grade data model** - Comprehensive specification for real-world power systems
- **HPC-ready architecture** - MPI parallelization, SIMD optimization, efficient memory layout
- **Modern Rust patterns** - Clean architecture with proper error handling and testing
- **Parity with DECOMP/NEWAVE** - Support for all major features of established solvers

## Documentation

The project specification is organized as **48 atomic spec files** grouped into 7 categories, each focused on a single concern and independently reviewable.

**[View the full Specification Index &rarr;](docs/specs/README.md)**

| Category                                                | Specs | Focus                                                         |
| ------------------------------------------------------- | ----: | ------------------------------------------------------------- |
| [00-overview](docs/specs/README.md#00-overview)         |     3 | Design principles, notation, production-scale reference       |
| [01-math](docs/specs/README.md#01-math)                 |    13 | SDDP algorithm, LP formulation, hydro models, risk measures   |
| [02-data-model](docs/specs/README.md#02-data-model)     |    10 | Input/output schemas, penalty system, binary formats          |
| [03-architecture](docs/specs/README.md#03-architecture) |    12 | Execution flow, solver abstraction, training loop, validation |
| [04-hpc](docs/specs/README.md#04-hpc)                   |     8 | MPI parallelism, memory architecture, SLURM deployment        |
| [05-config](docs/specs/README.md#05-config)             |     1 | Configuration reference for all LP variants                   |
| [06-deferred](docs/specs/README.md#06-deferred)         |     1 | Future features: batteries, multi-cut, wind/solar             |

> **Note**: The original monolithic documentation files (`DATA_MODEL_SPECIFICATION.md`, `MATHEMATICAL_FORMULATIONS.md`, `PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md`) are preserved in `docs/` for reference during the transition period. All new development should reference the atomic specs in `docs/specs/`.

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
│   ├── specs/                        # 48 atomic specification files
│   │   ├── 00-overview/              # Design principles, notation
│   │   ├── 01-math/                  # Mathematical formulations
│   │   ├── 02-data-model/            # Input/output data schemas
│   │   ├── 03-architecture/          # Program architecture
│   │   ├── 04-hpc/                   # HPC and parallelism
│   │   ├── 05-config/                # Configuration reference
│   │   └── 06-deferred/              # Deferred features
│   ├── DATA_MODEL_SPECIFICATION.md   # Original monolithic spec (archived)
│   ├── MATHEMATICAL_FORMULATIONS.md  # Original monolithic spec (archived)
│   └── PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md  # Original monolithic spec (archived)
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

1. Read the [Specification Index](docs/specs/README.md) for an overview of all specs
2. Start with [Design Principles](docs/specs/00-overview/design-principles.md) and [SDDP Algorithm](docs/specs/01-math/sddp-algorithm.md)
3. Check open issues for tasks

## License

MIT License - see [LICENSE](LICENSE)
