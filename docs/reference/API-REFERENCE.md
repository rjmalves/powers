# API Reference

**POWE.RS Library API Documentation**

This document describes the programmatic API for using POWE.RS as a Rust library in your own code.

## Table of Contents

- [Quick Start](#quick-start)
- [Factory API (Recommended)](#factory-api-recommended)
- [Builder API (Simple Cases)](#builder-api-simple-cases)
- [Core Types](#core-types)
- [Training and Simulation](#training-and-simulation)
- [Results](#results)
- [Examples](#examples)

---

## Quick Start

Add POWE.RS to your `Cargo.toml`:

```toml
[dependencies]
powers-rs = "0.2"
```

Basic usage:

```rust
use powers_rs::sddp::SddpAlgorithm;

fn main() -> Result<(), String> {
    // Load from JSON files (recommended)
    let mut sddp = SddpAlgorithm::from_files(
        "config.json",
        "system.json",
        "graph.json",
        "recourse.json",
    )?;

    // Train policy
    let result = sddp.train()?;
    println!("Final gap: {:.2}", result.final_gap());

    Ok(())
}
```

---

## Factory API (Recommended)

The **Factory API** loads SDDP instances from JSON files. This is the recommended approach for most use cases.

### `SddpAlgorithm::from_files()`

**Purpose**: Construct SDDP algorithm from JSON input files.

**Signature**:

```rust
pub fn from_files(
    config_path: &str,
    system_path: &str,
    graph_path: &str,
    recourse_path: &str,
) -> Result<Self, String>
```

**Parameters**:

- `config_path`: Path to `config.json` (algorithm parameters)
- `system_path`: Path to `system.json` (power system definition)
- `graph_path`: Path to `graph.json` (scenario tree)
- `recourse_path`: Path to `recourse.json` (uncertainty distributions)

**Returns**:

- `Ok(SddpAlgorithm)`: Ready-to-train SDDP instance with embedded configuration
- `Err(String)`: Error message if validation fails or files cannot be loaded

**Example**:

```rust
let mut sddp = SddpAlgorithm::from_files(
    "config.json",
    "system.json",
    "graph.json",
    "recourse.json",
)?;
```

**Features**:

- ✅ Full validation of all inputs before construction
- ✅ Supports all distribution types (normal, log-normal)
- ✅ Embeds configuration (zero-argument training)
- ✅ ~50 lines of boilerplate → 1 function call

**See Also**: [Input Specification](INPUT-SPECIFICATION.md) for JSON file format.

---

## Builder API (Simple Cases)

The **Builder API** constructs SDDP instances programmatically. Best for unit tests and simple scenarios with fixed (non-distributed) values.

### `SddpAlgorithm::builder()`

**Purpose**: Start building an SDDP instance programmatically.

**Signature**:

```rust
pub fn builder() -> SddpBuilder
```

**Returns**: `SddpBuilder` instance for method chaining.

### `SddpBuilder` Methods

#### `.system(system: System)`

Set the power system.

**Example**:

```rust
use powers_rs::system::System;

let system = System::default();  // Simple single-bus system
let builder = SddpAlgorithm::builder().system(system);
```

#### `.num_stages(n: usize)`

Set the number of stages (time periods).

**Example**:

```rust
let builder = builder.num_stages(12);  // 12 monthly stages
```

#### `.initial_storage(storage: Vec<f64>)`

Set initial hydro reservoir storage (MWh).

**Example**:

```rust
let builder = builder.initial_storage(vec![50.0, 30.0]);  // 2 reservoirs
```

#### `.deterministic_inflows(inflows: Vec<Vec<f64>>)`

Set deterministic inflows for each stage and hydro.

**Format**: `inflows[stage][hydro_id]`

**Example**:

```rust
let inflows = vec![
    vec![30.0, 20.0],  // Stage 0: hydro 0 = 30, hydro 1 = 20
    vec![40.0, 25.0],  // Stage 1
];
let builder = builder.deterministic_inflows(inflows);
```

#### `.deterministic_loads(loads: Vec<Vec<f64>>)`

Set deterministic loads for each stage and bus.

**Format**: `loads[stage][bus_id]`

**Example**:

```rust
let loads = vec![
    vec![75.0],  // Stage 0: bus 0 = 75 MW
    vec![80.0],  // Stage 1
];
let builder = builder.deterministic_loads(loads);
```

#### `.build()`

Construct the SDDP algorithm.

**Signature**:

```rust
pub fn build(self) -> Result<SddpAlgorithm, String>
```

**Returns**:

- `Ok(SddpAlgorithm)`: Ready-to-train instance
- `Err(String)`: Error if configuration is invalid

**Example**:

```rust
let sddp = SddpAlgorithm::builder()
    .system(System::default())
    .num_stages(2)
    .initial_storage(vec![50.0])
    .deterministic_inflows(vec![vec![30.0], vec![40.0]])
    .build()?;
```

**Limitations**:

- ⚠️ No distribution support (deterministic only)
- ⚠️ Requires explicit `train(iterations, forward_passes, saa)` call
- ⚠️ More verbose than Factory API

**Use Cases**:

- Unit tests with simple scenarios
- Benchmarks with fixed inputs
- Programmatic problem generation

---

## Core Types

### `SddpAlgorithm`

The main SDDP algorithm struct.

**Fields** (private):

- `system`: Power system definition
- `initial_condition`: Initial reservoir storage
- `num_stages`: Number of stages
- `policy_graph`: Scenario tree structure
- `saa`: Sample Average Approximation (scenarios)
- `config`: Optional embedded configuration

**Methods**:

- `from_files()`: Factory constructor
- `builder()`: Builder constructor
- `train()`: Train policy (see below)
- `simulate()`: Simulate policy (see below)

### `System`

Power system definition (buses, lines, thermals, hydros).

**Constructor**:

```rust
let system = System::default();  // Simple test system
```

**See**: `src/system.rs` for full API (to be documented in future release).

### `TrainingResult`

Results from SDDP training.

**Fields**:

```rust
pub struct TrainingResult {
    pub iterations: Vec<IterationResult>,
    pub final_lower_bound: f64,
    pub statistical_upper_bound: f64,
}
```

**Methods**:

```rust
impl TrainingResult {
    pub fn final_gap(&self) -> f64;  // Gap between bounds
    pub fn lower_bounds(&self) -> Vec<f64>;  // Lower bound per iteration
}
```

**Example**:

```rust
let result = sddp.train()?;
println!("Final lower bound: {:.2}", result.final_lower_bound);
println!("Final gap: {:.2}", result.final_gap());
```

---

## Training and Simulation

### Training: `.train()`

Train the SDDP policy via forward-backward iterations.

**Factory API** (zero-argument, uses embedded config):

```rust
pub fn train(&mut self) -> Result<TrainingResult, String>
```

**Example**:

```rust
let mut sddp = SddpAlgorithm::from_files(...)?;
let result = sddp.train()?;  // Uses config from config.json
```

**Builder API** (explicit arguments):

```rust
pub fn train(
    &mut self,
    iterations: usize,
    forward_passes: usize,
    saa: &SAA,
) -> Result<TrainingResult, String>
```

**Parameters**:

- `iterations`: Number of SDDP iterations (typically 50-100)
- `forward_passes`: Number of parallel forward passes per iteration (typically 4-10)
- `saa`: Sample Average Approximation with scenarios

**Example**:

```rust
let saa = generate_scenarios(&system, ...);
let result = sddp.train(100, 10, &saa)?;
```

**Returns**:

- `Ok(TrainingResult)`: Training results with convergence history
- `Err(String)`: Error if solver fails or numerical issues occur

**Side Effects**:

- Modifies internal policy (adds Benders cuts)
- Prints progress table to stdout (iteration, bounds, time)

### Simulation: `.simulate()`

Simulate the trained policy on out-of-sample scenarios.

**Factory API** (zero-argument, uses embedded config and SAA):

```rust
pub fn simulate(&mut self) -> Result<Vec<TrainingHandler>, String>
```

**Example**:

```rust
let mut sddp = SddpAlgorithm::from_files(...)?;
sddp.train()?;  // Train first
let handlers = sddp.simulate()?;  // Simulate with config.num_simulation_scenarios
```

**Builder API** (explicit scenarios):

```rust
pub fn simulate(
    &mut self,
    scenarios: &[Scenario],
) -> Result<Vec<TrainingHandler>, String>
```

**Parameters**:

- `scenarios`: Out-of-sample scenarios to evaluate policy

**Example**:

```rust
let test_scenarios = generate_scenarios(&system, ...);
let handlers = sddp.simulate(&test_scenarios)?;
```

**Returns**:

- `Ok(Vec<TrainingHandler>)`: One handler per scenario with decisions and costs
- `Err(String)`: Error if simulation fails

**Side Effects**:

- Prints progress (expected cost, std dev)
- Uses parallelism (Rayon) for efficiency

---

## Results

### `TrainingResult`

**Fields**:

```rust
pub struct TrainingResult {
    pub iterations: Vec<IterationResult>,  // Per-iteration results
    pub final_lower_bound: f64,            // Best lower bound achieved
    pub statistical_upper_bound: f64,      // Average of all forward pass costs
}
```

**Methods**:

```rust
impl TrainingResult {
    /// Gap between upper and lower bounds
    pub fn final_gap(&self) -> f64 {
        self.statistical_upper_bound - self.final_lower_bound
    }

    /// Extract lower bounds for all iterations
    pub fn lower_bounds(&self) -> Vec<f64> {
        self.iterations.iter().map(|it| it.lower_bound).collect()
    }
}
```

### `IterationResult`

**Fields**:

```rust
pub struct IterationResult {
    pub iteration: usize,           // Iteration number (0-indexed)
    pub lower_bound: f64,           // Lower bound after this iteration
    pub forward_pass_cost: f64,     // Cost from forward pass (sample)
    pub time: Duration,             // Time for this iteration
}
```

**Example** (extract convergence data):

```rust
let result = sddp.train()?;

for iter_result in result.iterations {
    println!("Iteration {}: LB = {:.2}, Cost = {:.2}",
        iter_result.iteration,
        iter_result.lower_bound,
        iter_result.forward_pass_cost
    );
}
```

### `TrainingHandler`

Represents the solution for one simulated scenario.

**Fields** (simplified, see `src/sddp/train_handler.rs` for full API):

```rust
pub struct TrainingHandler {
    pub total_cost: f64,                    // Total cost for this scenario
    pub stage_costs: Vec<f64>,              // Cost per stage
    pub storage_decisions: Vec<Vec<f64>>,   // Storage[stage][hydro]
    pub generation_decisions: Vec<Vec<f64>>, // Generation[stage][hydro]
    // ... more fields
}
```

**Example** (analyze simulation results):

```rust
let handlers = sddp.simulate()?;

let total_costs: Vec<f64> = handlers.iter().map(|h| h.total_cost).collect();
let mean_cost = total_costs.iter().sum::<f64>() / total_costs.len() as f64;
println!("Mean cost: {:.2}", mean_cost);
```

---

## Examples

### Example 1: Load and Train

```rust
use powers_rs::sddp::SddpAlgorithm;

fn main() -> Result<(), String> {
    // Load from JSON
    let mut sddp = SddpAlgorithm::from_files(
        "config.json",
        "system.json",
        "graph.json",
        "recourse.json",
    )?;

    // Train
    let result = sddp.train()?;

    // Print results
    println!("Training complete!");
    println!("  Lower bound: ${:.2}", result.final_lower_bound);
    println!("  Upper bound: ${:.2}", result.statistical_upper_bound);
    println!("  Gap: ${:.2}", result.final_gap());

    Ok(())
}
```

### Example 2: Custom Training Loop

```rust
use powers_rs::sddp::SddpAlgorithm;

fn main() -> Result<(), String> {
    let mut sddp = SddpAlgorithm::from_files(...)?;

    // Train with custom parameters (overrides config.json)
    // Note: Only works with explicit train(iterations, passes, saa) signature
    // Factory API requires modification for this use case

    let result = sddp.train()?;  // Uses config.json parameters

    // Check convergence
    if result.final_gap() < 100.0 {
        println!("Converged!");
    } else {
        println!("Did not converge, gap = {:.2}", result.final_gap());
    }

    Ok(())
}
```

### Example 3: Builder API (Simple Test)

```rust
use powers_rs::sddp::SddpAlgorithm;
use powers_rs::system::System;

fn main() -> Result<(), String> {
    // Build programmatically
    let sddp = SddpAlgorithm::builder()
        .system(System::default())
        .num_stages(2)
        .initial_storage(vec![50.0])
        .deterministic_inflows(vec![
            vec![30.0],  // Stage 0
            vec![40.0],  // Stage 1
        ])
        .build()?;

    // Note: Builder API requires explicit train() call with parameters
    // See development/TESTING.md for examples

    Ok(())
}
```

### Example 4: Simulation and Analysis

```rust
use powers_rs::sddp::SddpAlgorithm;

fn main() -> Result<(), String> {
    let mut sddp = SddpAlgorithm::from_files(...)?;

    // Train
    sddp.train()?;

    // Simulate
    let handlers = sddp.simulate()?;

    // Analyze results
    let costs: Vec<f64> = handlers.iter().map(|h| h.total_cost).collect();
    let mean = costs.iter().sum::<f64>() / costs.len() as f64;
    let variance = costs.iter()
        .map(|c| (c - mean).powi(2))
        .sum::<f64>() / costs.len() as f64;
    let std_dev = variance.sqrt();

    println!("Simulation Results:");
    println!("  Scenarios: {}", handlers.len());
    println!("  Mean cost: ${:.2}", mean);
    println!("  Std dev: ${:.2}", std_dev);
    println!("  Min cost: ${:.2}", costs.iter().copied().fold(f64::INFINITY, f64::min));
    println!("  Max cost: ${:.2}", costs.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    Ok(())
}
```

---

## Error Handling

All API functions return `Result<T, String>`:

```rust
match sddp.train() {
    Ok(result) => {
        println!("Success! Gap: {:.2}", result.final_gap());
    }
    Err(error) => {
        eprintln!("Training failed: {}", error);
        // Error string contains detailed context
    }
}
```

**Common errors**:

- **Validation errors**: Invalid input parameters (see [TROUBLESHOOTING.md](../guides/TROUBLESHOOTING.md))
- **Solver errors**: LP infeasible or numerical issues
- **I/O errors**: Cannot read input files

---

## Best Practices

### Use Factory API for Production

✅ **Recommended**:

```rust
let mut sddp = SddpAlgorithm::from_files(...)?;
let result = sddp.train()?;  // Zero arguments, uses config
```

❌ **Avoid** (more verbose, no validation checkpoint):

```rust
// 50 lines of boilerplate to load and validate manually
```

### Use Builder API for Tests

✅ **Good for tests**:

```rust
let sddp = SddpAlgorithm::builder()
    .system(System::default())
    .num_stages(2)
    .deterministic_inflows(vec![vec![30.0], vec![40.0]])
    .build()?;
```

### Check Convergence

Always check training results:

```rust
let result = sddp.train()?;

if result.final_gap() > 1000.0 {
    println!("Warning: Large gap, consider more iterations");
}

// Check monotonicity
let bounds = result.lower_bounds();
for i in 1..bounds.len() {
    assert!(bounds[i] >= bounds[i-1] - 1e-6, "Lower bound decreased!");
}
```

---

## Future API Changes (v1.0.0)

Planned improvements:

- [ ] Full API documentation with rustdoc
- [ ] More granular control over training (callbacks, early stopping)
- [ ] Policy serialization (save/load trained policies)
- [ ] Advanced risk measures (CVaR, worst-case)
- [ ] Multi-cut variant

---

**Navigation**: [↑ Documentation Index](../README.md) | [Input Specification](INPUT-SPECIFICATION.md) | [Testing Guide](../development/TESTING.md)
