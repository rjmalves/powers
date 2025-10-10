# v0.2.1 (Unreleased)

### Added

- **`SddpInstanceBuilder` Pattern**: Enables parameter modification after loading JSON files but before construction

  - `SddpInstanceBuilder::from_paths()` - Load and validate inputs
  - `with_num_iterations(n)` - Modify number of SDDP iterations
  - `with_num_forward_passes(n)` - Modify forward passes per iteration
  - `with_seed(seed)` - Modify random seed for SAA generation
  - `with_num_threads(n)` - Configure thread count for parallel execution
  - `build()` - Construct `SddpInstance` with modified configuration
  - **Use case**: Parameter sweeps for benchmarking and sensitivity analysis
  - **Performance**: Zero-cost abstraction (move semantics, < 1μs construction)
  - **Backward compatible**: `from_files()` still works (uses builder internally)

- **Thread Configuration Parameter** (`num_threads`):
  - New `Config.num_threads: Option<usize>` field for explicit thread control
  - `None` or omitted: Auto-detects available CPU cores using `num_cpus::get()`
  - `Some(n)`: Uses exactly `n` threads (validated > 0 at runtime)
  - Integrated in `SddpInstance::train()` and `simulate()` methods
  - Thread pool configured via `configure_thread_pool()` helper in `utils.rs`
  - Thread count logged during training/simulation for debugging
  - JSON schema updated with validation (minimum: 1, examples: [4, 8, null])
  - All 5 example configs updated with recommended thread counts
  - **Eliminates**: `RAYON_NUM_THREADS` environment variable requirement
  - **Performance**: < 10ms thread pool configuration overhead per train/simulate call
  - **Use cases**: Parameter sweeps, reproducibility, performance tuning
  - **Backward compatible**: Configs without `num_threads` default to auto-detection

### Changed

- `SddpAlgorithm::from_files()` now uses `SddpInstanceBuilder` internally (zero overhead)

### Documentation

- **Performance Tuning Guide** (`docs/guides/PERFORMANCE_TUNING.md`):
  - Comprehensive 914-line guide for optimizing POWE.RS performance
  - **Thread configuration**: Hardware-specific recommendations (94.4%/83.3%/68.9% efficiency at 2/4/8 threads)
  - **Solver tuning**: HiGHS parameters, presolve strategy, tolerance configuration
  - **Memory management**: Scaling formulas, cut pool management, system limits
  - **Problem optimization**: State space reduction, scenario selection, stage aggregation
  - **Benchmarking**: Using Criterion suite for validation
  - **Troubleshooting**: Common issues (poor scaling, slow convergence, high memory, numerical problems)
  - **Quick reference**: Cheat sheets for thread config, memory estimation, benchmarking
  - Cross-references parallel efficiency analysis and memory profiling reports
- Added "Advanced Usage" section to README.md demonstrating parameter sweeps
- Added comprehensive module documentation for `SddpInstanceBuilder` (150+ lines)
- Added 7 unit tests and 7 integration tests for builder pattern

# v0.2.0

- Implements single-node parallelism (thread-based) for both training and simulation steps
- Applies L1 dominance cut selection while locking the future cost function on each thread
- Number of parallel threads is capped either on the logical core count or number of forward passes / simulated scenarios
- Generalizes the underlying data structure for the SDDP algorithm from a vector to a graph
- Creates submodules for scenario generation, risk measure, state, and stochastic process
- Generalizes recourse input for defining inflow and load scenarios, with branchings per node
- Uses trait objects for dynamic risk measure, state and stochastic process definition
- Comprehensive parallel efficiency analysis with production-scale benchmarks (83.3% efficiency at 4 threads)
- Extended memory profiling with production-scale validation: Brazilian hydrothermal (156 reservoirs) uses 3.6 GB; memory scales O(N²) with reservoir count, NOT linearly with stages/iterations

# v0.1.1

- Better handles memory allocation during the SAA generation step in simulation
- Fixes doctests from renaming the crate

# v0.1.0

- Initial release
- Solves a simples hydrothermal dispatch problem with hydro storages as state variables using SDDP
- Inflows are sampled from `LogNormal` distributions, considered the same for all stages
- Loads are constant, given by the user in the input data
- Implements a custom interface to the `HiGHS` solver
- Contains cut selection and basis reuse for improving performance
- Performs simulation by sampling from the same distributions used for training
