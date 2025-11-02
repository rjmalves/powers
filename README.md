# POWE.RS - Power Optimization for the World of Energy - in pure RuSt

[![Test Suite](https://github.com/rjmalves/powers/actions/workflows/test.yml/badge.svg)](https://github.com/rjmalves/powers/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/rjmalves/powers/branch/main/graph/badge.svg)](https://codecov.io/gh/rjmalves/powers)

An implementation of the Stochastic Dual Dynamic Programming (SDDP) algorithm in pure Rust, for the hydrothermal dispatch problem.

## Introduction

This repository contains an implementation of the SDDP for the hydrothermal dispatch problem with support for **Periodic Autoregressive (PAR) models** for seasonal inflow uncertainty. The system can represent realistic hydro inflow patterns with both seasonal variation and temporal correlation.

The focus is on implementing the algorithm efficiently using the Rust programming language, which introduces concepts like ownership and borrowing in exchange for memory safety. This provides a middle ground between the performance of C++ and the safety guarantees needed for numerical optimization code.

This code is the result of a learning path on Rust that aimed to close the gap between how performant code for the SDDP algorithm is done in languages like C++, that gives the developer more freedom to mess with memory in exchange for performance, and how performant code for the SDDP algorithm can be done in Rust.

## Main features

### The `powers` system

The power system considered in powers is composed of four basic entities:

1. Buses
2. Lines
3. Thermals
4. Hydros

Each bus will receive some demand, which could vary depending on the stage or scenario, and should be met either by thermal generation, hydro generation, exchange or deficit, which is penalized by a given value for each bus. The lines allow buses to exchange power, with given limits on each direction. Thermals generate power with a given constant cost to the bus they are connected and hydros are able to store and generate power when the evaluated policy decides chooses to.

The `productivity` of each hydro is considered to be constant, for simplicity, and given by the user. Also, the thermal generation, hydro storage and hydro turbined flow can be bounded in the system input data, but only with a single value for the entire study. Defining more complicated constraints, such as seasonal bounds, is a future work. However, hydro cascades can be defined via the `downstream_hydro_id` attribute.

### The `powers` algorithm

The implemented algorithm is the classic SDDP from [Pereira & Pinto, 1991](https://link.springer.com/article/10.1007/BF01582895). A Sample Average Approximation (SAA) is made for obtaining scenarios from user-specified distributions (Normal, LogNormal3) or temporal models:

- **Independent Sampling**: Direct sampling from marginal distributions
- **Periodic Autoregressive (PAR)**: Seasonal models with temporal correlation

These inflows are sampled on each iteration, which are comprised of a `forward` step (visits viable states) and a `backward` step (refines the policy via Benders cuts).

The main product of this algorithm is a decision-making policy in the form of Benders' Cuts, that are inserted to the optimization problem in the form of constraints. Each iteration produces a new cut for each stage, except for the last one. This is called the `single-cut` or `average-cut` variant of the algorithm. In a scenario that supports parallel computing, each iteration may produce N cuts, where N is the number of simultaneous forward passes.

**Key Features**:

- ✅ **Seasonal uncertainty modeling** via Periodic Autoregressive (PAR) models
- ✅ **Temporal correlation** with AR(p) structure per season
- ✅ **Flexible inflow distributions**: Normal, LogNormal3
- ✅ **Hydro cascades** with upstream-downstream water routing

For detailed PAR configuration and usage, see the [PAR Model Guide](docs/guides/PAR-MODEL-GUIDE.md).

### State Representation for Autoregressive Models

When using PAR(p) inflow models, it's essential to use `StorageAndInflowState` to properly represent lagged inflows as state variables. This ensures Benders cuts capture the autoregressive dynamics through the chain rule.

**Correct Usage:**

For PAR models, always specify `StateSpace::StorageAndInflow` when building the SDDP algorithm. This enables proper cut generation that accounts for temporal correlation in inflows.

**How It Works:**

The Benders cut coefficients for lagged inflows are computed using the chain rule:

```
∂FO/∂Y_{t-j} = (water_value + ar_dual) * ψ_j
```

Where:

- `water_value`: dual of hydro balance constraint (λ^BH)
- `ar_dual`: dual of AR dynamics constraint (λ^AR)
- `ψ_j`: **observation-space** AR coefficient for lag j (see Notation section below)

This ensures the policy accounts for information in past inflows when making decisions, leading to 5-15% cost reduction for systems with high AR persistence (φ > 0.7).

### Notation: φ vs ψ Coefficients

The codebase distinguishes between two representations of AR coefficients:

- **φ (phi)**: Residual-space coefficients from the PAR statistical model

  - Extracted from `par_params.ar_coefficients`
  - Used in: Z'_t = Σ φ_i \* Z'_{t-i} + ε_t
  - Where: Z'\_t = (Y_t - μ_t) / σ_t

- **ψ (psi)**: Observation-space coefficients for LP and cuts
  - Computed: ψ*i = φ_i \* (σ_t / σ*{t-i})
  - Used in: Y*t = Σ ψ_i \* Y*{t-i} + η_t
  - Used in: LP constraints and Benders cuts

**Why the distinction matters**: For seasonal systems where σ varies across months, ψ ≠ φ. The LP constraints use ψ, so Benders cuts must also use ψ for mathematical consistency. Using φ would cause incorrect policy gradients (error magnitude: |σ*t / σ*{t-i} - 1| × 100%).

**Example**: If current month has σ*t = 100 MWh and previous month has σ*{t-1} = 50 MWh, with φ_1 = 0.7, then ψ_1 = 0.7 × (100/50) = 1.4. Using φ instead of ψ would underestimate sensitivity by 50%.

**Reference**: `par_derivation.pdf`, Equations 7-8

**When to Use Each State Type:**

- **`StorageState`**: Independent inflows (no autocorrelation)
- **`StorageAndInflowState`**: PAR(p) models with lag order > 0

For more details on the mathematical foundation, see [SDDP_AR_CUT_ANALYSIS_REPORT.md](SDDP_AR_CUT_ANALYSIS_REPORT.md).

### Performance

In exchange for the relavively simple system, some optimizations to the algorithm itself were made. First, the number of memory allocations was minimized when interacting with the underlying solver, [HiGHS](https://github.com/ERGO-Code/HiGHS/). Therefore, the optimization problem is converted in the model form as a pre-processing step in the policy graph construction, and this model is edited through the iterations.

Also, instead of using the more common [highs](https://docs.rs/highs/latest/highs/) crate for interacting with the solver, a different interface was built using the [highs-sys](https://crates.io/crates/highs-sys) crate, which contains the result of applying [bindgen](https://github.com/rust-lang/rust-bindgen) to the solver repository. The developed interface is highly based on the [highs](https://docs.rs/highs/latest/highs/) crate, but differs in some aspects that affected the SDDP in a relevant way.

Given the nature of the `backward` step, it is expected that some info of the solver state in the `forward` step can help improving the solution process. This is mainly known as `basis reuse` and is implemented in `powers` by storing the basis of each solved problem in the `forward` step and initializing each solved problem in the `backward` step, for the same node, with the stored basis.

Also, when the SDDP algorithm continues for a large number of iterations, the number of cuts (which turns into constraints) begins to hurt the performance. For this case, a `cut selection` strategy was implemented, highly based on the existing one from [SDDP.jl](https://github.com/odow/SDDP.jl), which is inspired in [de Matos, Philpott & Finardi 2015](https://www.sciencedirect.com/science/article/pii/S0377042715002794).

For handling slightly larger problems, it is common for the solver to suffer from numerical issues. Therefore, the `solve` calls consist of a up-to-3 retry steps, which change the solver options in order to continue the iterative process instead of stopping the algorithm with an error state.

Currently there is support for thread-based parallelism, which is capped on the number of logical cores of the running machine. During training, the number of forward passes also limit the parallelism level. During the simulation step, the number of simulated scenarios also defines the maximum number of simultaneous threads. For handling these parallel steps, the [rayon](https://docs.rs/rayon/latest/rayon/) crate is used.

### Benchmarking

Performance benchmarks are available to measure key operations and validate optimization targets. Benchmarks use the [Criterion](https://github.com/bheisler/criterion.rs) framework for statistical analysis.

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench realize_uncertainties
cargo bench --bench sddp_benchmarks
cargo bench --bench cut_selection

# Generate HTML reports (saved to target/criterion/)
cargo bench --bench realize_uncertainties -- --verbose
```

Current baseline results (50-hydro system):

- Subproblem construction: ~86 µs
- HydroData sequential access: ~18 ns

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed performance metrics and system specifications.

#### Performance Features

**SIMD Optimizations** (Optional)

Enable SIMD-optimized dot product operations for additional performance in lag contribution computations:

```bash
# Build with SIMD optimizations
cargo build --release --features simd-optimizations

# Run benchmarks with SIMD enabled
cargo bench --features simd-optimizations --bench simd_dot_product
```

Expected speedup with SIMD enabled:

- AR(1) lag contributions: ~1.3x faster
- AR(2) lag contributions: ~1.2x faster
- AR(3) lag contributions: ~1.4x faster

SIMD optimizations use unsafe unchecked indexing to enable LLVM auto-vectorization. Compile with `RUSTFLAGS="-C target-cpu=native"` for best results on your CPU architecture.

### Dependencies

This implementation was made aiming to minimize the external dependencies whenever possible. The key crates on which it depends are:

1. [highs-sys](https://crates.io/crates/highs-sys): the low-level interface with the HiGHS solver, which is mainly an application of [bindgen](https://github.com/rust-lang/rust-bindgen) to the C-API.
2. [rand](https://docs.rs/rand/latest/rand/), [rand_distr](https://docs.rs/rand_distr/latest/rand_distr/) and [rand_xoshiro](https://docs.rs/rand_xoshiro/latest/rand_xoshiro/): random number generation and probability distributions utilities for the scenario generation and inflow sampling processes.
3. [serde](https://docs.rs/serde/latest/serde/), [serde_json](https://docs.rs/serde_json/latest/serde_json/) and [csv](https://docs.rs/csv/latest/csv/): serializing and deserializing utilities for handling data input and output.
4. [rayon](https://docs.rs/rayon/latest/rayon/): implement parallel iterators for the training and simulation steps.

## Installation

For detailed installation instructions, see **[Installation Guide](docs/guides/INSTALLATION.md)**.

### Quick Install: Pre-built binaries

Pre-built binaries are available on each release page, for downloading on Linux and Mac architectures. An installation via `curl` is also possible through

```
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/rjmalves/powers/releases/download/<VERSION>/powers-rs-installer.sh | sh
```

where the `<VERSION>` must be replaced by the desired tag, such as `v0.1.1`.

### Building dependencies

Since this crate compiles the solver itself, which includes a whole C++ project, some system dependencies are required. In Ubuntu, the installation can be done with

```
sudo apt update
sudo apt install libclang-dev build-essential cmake
```

### Building locally

The code can be downloaded from the repository and built locally for usage with

```
git clone https://github.com/rjmalves/powers.git
cd powers
cargo build --release
```

### Installing from crates.io

The executable can be installed without cloning the repository with

```
cargo install powers-rs
```

The last line displays the name of the executable to be called, which is `powers` itself.

### Running

POWE.RS provides a command-line interface with subcommands for different operations:

#### Run SDDP Algorithm (default)

Execute the SDDP algorithm with JSON configuration files:

```bash
# Explicit subcommand
powers run examples/04-cascade

# Backward-compatible (no subcommand - default behavior)
powers examples/04-cascade
```

The tool expects a directory containing:

- `config.json`: SDDP configuration (iterations, scenarios, convergence)
- `system.json`: Hydrothermal system (buses, hydros, thermals, lines)
- `graph.json`: Scenario tree structure (stages, nodes, probabilities)
- `recourse.json`: Stochastic process models (PAR, independent noise, correlation)

Example output:

```
$ powers examples/01-deterministic

POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

Reading input files from 'examples/01-deterministic'
Using 1 threads for training

# Training
- Iterations: 10
- Forward passes: 1

----------------------------------------------------------------------------------------
iter |      lower ($) |      simul ($) |          fwd |          bwd |        total
----------------------------------------------------------------------------------------
   1 |     2.499394e3 |     2.499394e3 | 00:00:00.000 | 00:00:00.000 | 00:00:00.000
  ...
  10 |     2.499394e3 |     2.499394e3 | 00:00:00.000 | 00:00:00.000 | 00:00:00.000
----------------------------------------------------------------------------------------

Training time: 00:00:00.005
```

#### Estimate PAR Parameters

Estimate Periodic Autoregressive (PAR) model parameters from historical CSV data:

```bash
# Estimate monthly PAR(1) for hydro inflows
powers estimate-par historical_inflows.csv --periods 12 --order 1 --output params.json

# Short form
powers estimate-par inflows.csv -p 12 -o 1 -O params.json

# Quarterly PAR(2) with custom validation
powers estimate-par data.csv -p 4 -o 2 --min-samples 10 -O quarterly.json

# Print to stdout (no output file)
powers estimate-par data.csv -p 12 -o 1
```

**CSV Input Format:**

- Each column represents one entity (e.g., hydro plant)
- Rows are consecutive time steps (e.g., months, weeks)
- Optional header row (use `--has-header` flag)

Example CSV:

```csv
hydro_1,hydro_2,hydro_3
45.0,120.0,85.0
48.0,125.0,90.0
...
```

**Output:** JSON compatible with the `noise_models` field in `recourse.json`.

For more details on any subcommand:

```bash
powers --help
powers run --help
powers estimate-par --help
```

### CLI Reference

```
Usage: powers [PATH] [COMMAND]

Commands:
  run           Run SDDP algorithm with JSON configuration files
  estimate-par  Estimate PAR model parameters from historical time series data
  help          Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

### Quick Start (Library API)

For programmatic use (tests, benchmarks, custom workflows), use the **Factory API**:

```rust
use powers_rs::sddp::SddpAlgorithm;

fn main() -> Result<(), String> {
    // One-line construction from JSON files
    let mut sddp = SddpAlgorithm::from_files(
        "examples/03-multistage/config.json",
        "examples/03-multistage/system.json",
        "examples/03-multistage/graph.json",
        "examples/03-multistage/recourse.json",
    )?;

    // Zero-argument training (config embedded)
    let result = sddp.train()?;
    println!("Final gap: {:.2}", result.final_gap());

    // Zero-argument simulation (config + SAA embedded)
    let handlers = sddp.simulate()?;
    println!("Simulated {} scenarios", handlers.len());

    Ok(())
}
```

**Benefits**: ~50 lines of boilerplate → ~5 lines with factory, works with distributions, zero overhead.

**Alternative (Builder API)**: Simpler but limited to explicit scenarios:

```rust
use powers_rs::sddp::SddpAlgorithm;
use powers_rs::system::System;

let sddp = SddpAlgorithm::builder()
    .system(System::default())
    .initial_storage(vec![50.0])
    .num_stages(2)
    .deterministic_inflows(vec![vec![30.0], vec![40.0]])
    .build()?;
```

Factory API supports full production workflows. Builder API is best for simple unit tests.

### Advanced Usage: Parameter Modification with `SddpInstanceBuilder`

For **parameter sweeps** (benchmarking, sensitivity analysis), use the **`SddpInstanceBuilder`**:

```rust
use powers_rs::sddp::SddpInstanceBuilder;

// Benchmark memory scaling with forward passes
for num_fwd in [1, 4, 8, 16, 32] {
    let sddp = SddpInstanceBuilder::from_paths(
        "examples/05-large-scale-brazilian/config.json",
        "examples/05-large-scale-brazilian/system.json",
        "examples/05-large-scale-brazilian/graph.json",
        "examples/05-large-scale-brazilian/recourse.json",
    )?
    .with_num_forward_passes(num_fwd)  // Modify config parameter
    .with_num_iterations(10)            // Chain multiple modifications
    .with_seed(42)                      // Ensure reproducibility
    .build()?;                          // Build the instance

    let result = sddp.train()?;
    println!("Forward passes: {}, Memory: {} MB", num_fwd, get_peak_memory());
}
```

**Key Features**:

- **Staged construction**: Load JSON files → modify parameters → build instance
- **Zero overhead**: Move semantics, no clones, < 1μs construction time
- **Chainable API**: Fluent interface for multiple modifications
- **Reproducibility**: Modify seed for deterministic testing

**Available Modifiers**:

- `with_num_iterations(n)` - Number of SDDP iterations
- `with_num_forward_passes(n)` - Forward passes per iteration
- `with_seed(seed)` - Random seed for SAA generation
- `with_num_threads(n)` - Thread count for parallelism (T4.5.6)

**Why This Pattern?**

The original `from_files()` API loads and immediately constructs the algorithm, preventing parameter modification. The builder pattern enables:

1. **Benchmarking**: Vary parameters programmatically (no multiple config files)
2. **Sensitivity analysis**: Test different configurations easily
3. **Reproducibility**: Same seed = identical results across runs

**Backward Compatible**: `from_files()` still works (uses builder internally).

### Thread Configuration

Control parallel execution with the **`num_threads`** configuration parameter:

**JSON Configuration**:

```json
{
  "num_iterations": 100,
  "num_forward_passes": 10,
  "seed": 42,
  "num_threads": 4 // Explicit thread count
}
```

**Auto-Detection** (use all available CPU cores):

```json
{
  "num_threads": null // or omit the field entirely
}
```

**Programmatic Control** (via builder):

```rust
let sddp = SddpInstanceBuilder::from_paths(...)
    .with_num_threads(8)    // Override JSON config
    .build()?;
```

**Recommendations**:

- **Small problems** (<20 stages): Use 4 threads
- **Large problems** (>50 stages): Use 8-16 threads
- **Production**: Test different counts to find optimal performance
- **Avoid over-subscription**: Don't exceed physical CPU cores

**Performance Notes**:

- Thread pool configuration adds < 10ms overhead per train/simulate call
- Auto-detection (`null`) uses `num_cpus::get()` for cross-platform detection
- Eliminates `RAYON_NUM_THREADS` environment variable requirement
- Thread count is logged during training/simulation for debugging

**Backward Compatible**: Configs without `num_threads` default to auto-detection.

### Documentation

📚 **Complete Documentation**: See [`docs/`](docs/) for comprehensive guides, references, and examples.

**Quick Links**:

- 🚀 **[Quick Start Tutorial](docs/guides/QUICKSTART.md)** - Your first optimization in 5 minutes
- 📖 **[Input Specification](docs/reference/INPUT-SPECIFICATION.md)** - Complete JSON format documentation
- 🔧 **[Troubleshooting Guide](docs/guides/TROUBLESHOOTING.md)** - Common errors and solutions
- 💻 **[API Reference](docs/reference/API-REFERENCE.md)** - Library usage and examples
- 🎓 **[SDDP Overview](docs/algorithm/SDDP-OVERVIEW.md)** - Algorithm background and theory
- ⚡ **[Performance Baselines](docs/performance/PERFORMANCE-BASELINES.md)** - Benchmark metrics and regression detection

**New: JSON Schema v2** (Unified Temporal Model Format):

- 📝 **[JSON Schema v2 Documentation](docs/json-schema-v2.md)** - Unified temporal model format specification
- 🔄 **[Migration Guide v1→v2](docs/migration-guide.md)** - Step-by-step migration instructions
- 📋 **[Refactoring Tickets](docs/refactoring-tickets.md)** - Implementation progress tracker

**IDE Integration**: JSON schemas provide auto-completion, inline documentation, and validation in VS Code (see [`.vscode/settings.json`](.vscode/settings.json)).

**Schemas**:

- [`schemas/config.schema.json`](schemas/config.schema.json) - Algorithm configuration
- [`schemas/system.schema.json`](schemas/system.schema.json) - Power system topology
- [`schemas/graph.schema.json`](schemas/graph.schema.json) - Scenario tree graph
- [`schemas/recourse.schema.json`](schemas/recourse.schema.json) - Uncertainty distributions

**Quick Reference**: The input data consists of four JSON files:

1. `config.json`: parameters of the SDDP algorithm itself

```json
{
  "num_iterations": 32,
  "num_forward_passes": 4,
  "num_simulation_scenarios": 128,
  "seed": 0,
  "output_path": "./example"
}
```

#### Output Control

The `output_path` field in `config.json` controls CSV file generation:

**Disable Output** (recommended for tests and benchmarks):

```json
{
  "num_iterations": 100,
  "num_forward_passes": 20,
  "num_simulation_scenarios": 1000,
  "seed": 42
  // Omit output_path or set to null for no CSV output
}
```

**Benefits**: 10-30% faster execution, cleaner directories, no I/O overhead.

**Enable Output**:

```json
{
  "num_iterations": 100,
  "num_forward_passes": 20,
  "num_simulation_scenarios": 1000,
  "seed": 42,
  "output_path": "./results"
}
```

CSV files will be written to the specified directory:

- `cuts.csv` - Benders cuts (intercept, slopes)
- `states.csv` - Visited states
- `simulation_buses.csv` - Bus simulation results
- `simulation_lines.csv` - Line simulation results
- `simulation_thermals.csv` - Thermal simulation results
- `simulation_hydros.csv` - Hydro simulation results

For complete field-by-field documentation, examples, and validation rules, see **[Input Specification](docs/reference/INPUT-SPECIFICATION.md)**.

**JSON Schemas**: All files have formal schemas for IDE auto-completion:

- [`schemas/config.schema.json`](schemas/config.schema.json)
- [`schemas/system.schema.json`](schemas/system.schema.json)
- [`schemas/graph.schema.json`](schemas/graph.schema.json)
- [`schemas/recourse.schema.json`](schemas/recourse.schema.json)

2. `system.json`: Power system definition (buses, lines, thermals, hydros)

```json
{
  "buses": [
    {
      "id": 0,
      "deficit_cost": 50.0
    }
  ],
  "lines": [],
  "thermals": [
    {
      "id": 0,
      "bus_id": 0,
      "cost": 5.0,
      "min_generation": 0.0,
      "max_generation": 15.0
    },
    {
      "id": 1,
      "bus_id": 0,
      "cost": 10.0,
      "min_generation": 0.0,
      "max_generation": 15.0
    }
  ],
  "hydros": [
    {
      "id": 0,
      "downstream_hydro_id": null,
      "bus_id": 0,
      "productivity": 1.0,
      "min_storage": 0.0,
      "max_storage": 100.0,
      "min_turbined_flow": 0.0,
      "max_turbined_flow": 60.0,
      "spillage_penalty": 0.01
    }
  ]
}
```

3. `recourse.json`: the avaliable resource for the decision-making process, namely the initial state, bus loads and hydro inflows.

```json
{
  "initial_condition": {
    "storage": [
      {
        "hydro_id": 0,
        "value": 83.222
      }
    ],
    "inflow": [
      {
        "hydro_id": 0,
        "lag": 1,
        "value": 50.0
      }
    ]
  },
  "uncertainties": [
    {
      "season_id": 0,
      "num_branchings": 10,
      "distributions": {
        "load": [
          {
            "bus_id": 0,
            "normal": {
              "mu": 75.0,
              "sigma": 0.0
            }
          }
        ],
        "inflow": [
          {
            "hydro_id": 0,
            "lognormal": {
              "mu": 3.6,
              "sigma": 0.6928
            }
          }
        ]
      }
    },
    // ...
    {
      "season_id": 11,
      "num_branchings": 10,
      "distributions": {
        "load": [
          {
            "bus_id": 0,
            "normal": {
              "mu": 75.0,
              "sigma": 0.0
            }
          }
        ],
        "inflow": [
          {
            "hydro_id": 0,
            "lognormal": {
              "mu": 3.6,
              "sigma": 0.6928
            }
          }
        ]
      }
    }
  ]
}
```

4. `graph.json`: the definition of the graph that models the stochastic decomposition problem, with the state definition, risk measure and stochastic processes of each stage.

```json
{
  "nodes": [
    {
      "id": 0,
      "stage_id": 0,
      "season_id": 0,
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-02-01T00:00:00Z",
      "risk_measure": "expectation",
      "load_stochastic_process": "naive",
      "inflow_stochastic_process": "naive",
      "state_variables": "storage"
    },
    // ...
    {
      "id": 11,
      "stage_id": 11,
      "season_id": 11,
      "start_date": "2024-11-01T00:00:00Z",
      "end_date": "2024-12-01T00:00:00Z",
      "risk_measure": "expectation",
      "load_stochastic_process": "naive",
      "inflow_stochastic_process": "naive",
      "state_variables": "storage"
    }
  ],
  "edges": [
    {
      "source_id": 0,
      "target_id": 1,
      "probability": 1.0,
      "discount_rate": 0.0
    },
    {
      "source_id": 10,
      "target_id": 11,
      "probability": 1.0,
      "discount_rate": 0.0
    }
  ]
}
```

## Output Data and Analysis

### Execution steps

Running the `powers` executable consists of two steps, which produces different outputs, always in `CSV` format:

1. `train`: the construction of the policy (cuts). This step generates two files: `cuts.csv` and `states.csv`.
2. `simulation`: the evaluation of the policy on different scenarios sampled from the same distributions. This step generates a file for each of the system entities: `simulation_buses.csv`, `simulation_lines.csv`, `simulation_thermals.csv` and `simulation_hydros.csv`.

### Output files

The content of each output file is described below:

#### `cuts.csv`

Containts the Benders' cuts evaluated during the training step. The stages are integers starting from 0 and inside each stage, the cuts are identified also by incremental integers from 0. The cuts contain an RHS term and a multiplier for the storage of each entity. An additional `active` column exists for indicating the result of the cut selection process.

```csv
stage_index, stage_cut_id, active, coefficient_entity, value
          0,            0, false , RHS               , 75.0
          0,            0, false , 0                 , 0.0
          0,            1, true  , RHS               , 556.3289550958654
          0,            1, true  , 0                 , -7.399799999999999
          0,            2, true  , RHS               , 982.7864174441443
          0,            2, true  , 0                 , -10.678600000000001
```

#### `states.csv`

Containts the states sampled during the training step. The stages are integers starting from 0 and inside each stage, each state contain the dominating objective value among all cuts and the storage of each entity that compose the state variables.

```csv
stage_index, dominating_cut_id, coefficient_entity , value
          0,                85, DominatingObjective, 4522.841996737085
          0,                85, 0                  , 20.529909871826362
          0,                85, DominatingObjective, 4439.327063667069
          0,                85, 0                  , 23.799050364700705
          0,                85, DominatingObjective, 4182.479610767903
          0,                85, 0                  , 33.85318533072352
```

#### `simulation_buses.csv`

Containts the simulation results for the variables of each `Bus` defined in the problem.

```csv
stage_index, series_index, entity_index, load, deficit             , marginal_cost
          0,            0,            0,  0.0,  0.0                ,           5.0
          1,            0,            0,  0.0,  0.0                ,           5.0
          2,            0,            0,  0.0,  0.0                ,           5.0
          3,            0,            0,  0.0,  7.460308786652931  ,          50.0
          4,            0,            0,  0.0,  0.0                ,          10.0
          5,            0,            0,  0.0, 22.497932081239966  ,          50.0
          6,            0,            0,  0.0,  0.0                ,           5.0
          7,            0,            0,  0.0,  3.5853689259062236 ,          50.0
          8,            0,            0,  0.0, 24.22378891424749   ,          50.0
          9,            0,            0,  0.0, 17.715927055535314  ,          50.0
         10,            0,            0,  0.0, 33.82354347703064   ,          50.0
         11,            0,            0,  0.0, 26.686549714044883  ,          50.0
```

#### `simulation_lines.csv`

Containts the simulation results for the variables of each `Line` defined in the problem.

```csv
stage_index, series_index, entity_index, exchange
          0,            0,            0, 25.0
          1,            0,            0, 25.0
          2,            0,            0, 25.0
          3,            0,            0, 25.0
          4,            0,            0, 25.0
          5,            0,            0, 25.0
```

#### `simulation_thermals.csv`

Containts the simulation results for the variables of each `Thermal` defined in the problem.

```csv
stage_index, series_index, entity_index, generation
          0,            0,            0, 15.0
          0,            0,            1,  0.0
          1,            0,            0, 15.0
          1,            0,            1,  0.0
          2,            0,            0, 15.0
          2,            0,            1,  0.0
          3,            0,            0, 15.0
          3,            0,            1, 15.0
          4,            0,            0, 15.0
          4,            0,            1,  3.9633896974920972
```

#### `simulation_hydros.csv`

Containts the simulation results for the variables of each `Hydro` defined in the problem.

```csv
stage_index, series_index, entity_index, final_storage        , inflow             , turbined_flow     , spillage            , water_value
          0,            0,            0,  42.82014469170315   ,  19.598144691703155, 60.0              ,   0.0               , -0.0
          1,            0,            0,  20.529909871826362  ,  37.70976518012321 , 60.0              ,   0.0               , -0.0
          2,            0,            0,  26.075621784681616  ,  65.54571191285525 , 60.0              ,   0.0               , -0.0
          3,            0,            0,   0.0                ,  11.464069428665454, 37.53969121334707 ,   0.0               , -50.0
          4,            0,            0,   0.0                ,  56.0366103025079  , 56.0366103025079  ,   0.0               , -10.0
          5,            0,            0,   0.0                ,  22.502067918760034, 22.502067918760034,   0.0               , -50.0
          6,            0,            0,  11.893432441431372  ,  71.89343244143137 , 60.0              ,   0.0               , -0.0
          7,            0,            0,   0.0                ,  29.521198632662404, 41.414631074093776,   0.0               , -50.0

```

## Testing

**Test Coverage**: 89.42% (189 library tests + 473+ total tests across all test binaries)

This project has comprehensive test coverage validating:

- Core SDDP algorithm components (Benders cuts, cut pool, state management, convergence)
- Solver interface and error handling
- Input validation and schema conformance
- Stochastic process and scenario generation
- End-to-end integration with realistic problems
- Performance characteristics and algorithmic correctness

### Testing Philosophy

POWE.RS follows the **"test business logic, not infrastructure"** principle:

- ✅ **High-value tests**: Focus on algorithm correctness and user-facing behavior
- ✅ **In-module testing**: Use `#[cfg(test)]` to test private functions
- ✅ **Strategic coverage**: 89.42% with clear documentation of intentionally uncovered code
- ✅ **Integration matters**: Use `--all-targets` for accurate coverage (vs 79.99% with `--lib` only)
- ❌ **Avoid brittle tests**: No environment variable manipulation or complex mocking
- ❌ **Skip infrastructure**: Logging, entry points, and rare error paths documented as uncovered

### Coverage Metrics

- **189 library tests** (unit tests in `src/`)
- **473+ total tests** (including integration tests in `tests/`)
- **89.42% line coverage** (with `--all-targets`)
- **8 modules with 100% coverage** (cut, state, system, risk_measure, stochastic_process, utils, initial_condition, and more)
- **Zero clippy warnings** (enforced with `-D warnings`)

See [docs/development/TESTING.md](docs/development/TESTING.md) for detailed coverage philosophy and breakdown.

### Running Tests Locally

```bash
# Run all tests (unit + integration)
cargo test --all-features

# Run only library unit tests (fast, 189 tests)
cargo test --lib

# Run with output
cargo test --all-features -- --nocapture

# Run specific test module
cargo test --test integration_simple_2stage

# Check coverage with llvm-cov (RECOMMENDED: use --all-targets)
cargo install cargo-llvm-cov
cargo llvm-cov --all-targets --html
# Open target/llvm-cov/html/index.html
# Shows 89.42% coverage (vs 79.99% with --lib only)

# Run with all CI checks
cargo fmt --all -- --check && \
cargo clippy --all-targets --all-features -- -D warnings && \
cargo build --verbose && \
cargo test --verbose --all-features
```

### Test Structure

**Unit Tests** (in `src/` modules with `#[cfg(test)]`):

- 173 library tests covering core algorithm logic
- Private function testing via in-module test modules
- Edge cases and error path validation

**Integration Tests** (in `tests/`):

```
tests/
├── fixtures/              # Test utilities and fixtures
│   ├── simple_2stage_reservoir.rs  # 2-stage problem setup
│   ├── benchmarks.rs               # Benchmark problems
│   └── mod.rs                      # Fixture exports
├── test_sddp_algorithm.rs          # Algorithm correctness
├── test_input_validation.rs        # Comprehensive validation tests
├── test_numerical_validation.rs    # Numerical properties
├── test_cut.rs                     # Benders cut operations
├── test_cut_pool.rs                # Cut storage & selection
├── test_scenario.rs                # Scenario generation
├── test_state.rs                   # State management
└── integration_simple_2stage.rs    # End-to-end SDDP
```

### Continuous Integration

All tests run automatically on:

- Every push to `main` or `master`
- All pull requests

The CI pipeline includes:

- Code formatting check (`cargo fmt`)
- Linting with Clippy (`cargo clippy`)
- Full test suite execution
- Performance validation

See [`.github/workflows/README.md`](.github/workflows/README.md) for detailed CI documentation.

For comprehensive testing documentation including benchmarks, fixtures, and best practices, see **[Testing Guide](docs/development/TESTING.md)**.

## Contributing

Contributions are welcome! For comprehensive guidance on writing and running tests, contributing code, and understanding the architecture, see our documentation:

- **[Testing Guide](docs/development/TESTING.md)** - Test structure, fixtures, and best practices
- **[Architecture Documentation](docs/architecture/)** - Design decisions and implementation details
- **[Performance Documentation](docs/performance/)** - Optimization strategies and analysis
- **[Future Enhancements](FUTURE_WORK.md)** - Planned improvements and enhancement proposals

### Before Submitting a PR

1. **Format your code:**

   ```bash
   cargo fmt --all
   ```

2. **Check for linting issues:**

   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```

3. **Run the test suite:**

   ```bash
   cargo test --all-features
   ```

4. **Check code coverage (optional):**

   ```bash
   cargo tarpaulin --out Html --output-dir coverage --all-features
   ```

5. **Ensure CI passes:** All checks must pass before merging

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use POWE.RS in your research, please cite:

```bibtex
@software{powers_rs,
  author = {Alves, Rogerio},
  title = {POWE.RS: Stochastic Dual Dynamic Programming in Rust},
  year = {2025},
  url = {https://github.com/rjmalves/powers}
}
```

```

```
