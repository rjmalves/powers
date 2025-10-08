# Quick Start Tutorial

Get started with POWE.RS in 5 minutes! This tutorial will guide you through running your first hydrothermal dispatch optimization.

## Prerequisites

- POWE.RS installed ([Installation Guide](INSTALLATION.md))
- Basic understanding of power systems (optional, but helpful)

---

## Step 1: Get the Example Data

POWE.RS includes a comprehensive example suite in the `examples/` directory. Start with the simplest example:

```bash
git clone https://github.com/rjmalves/powers.git
cd powers
```

### Example Suite

The `examples/` directory contains a progressive series of problems:

- **`01-deterministic/`** - Simplest 2-stage deterministic problem (start here!)
- **`02-stochastic/`** - Introduces uncertainty with stochastic inflows
- **`example/`** - Original 12-stage hydrothermal system

**Recommended learning path**: Start with Example 1, then progress to Example 2, then the full `example/`.

For detailed documentation on each example, see the [Example Suite README](../../examples/README.md).

---

## Step 2: Run Your First Optimization

### Option A: Start with Example 1 (Recommended for Beginners)

Run the simplest deterministic example:

```bash
cargo run --release examples/01-deterministic
```

**Expected output**:

```
POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

Reading input files from 'examples/01-deterministic'

# Training
- Iterations: 3
- Forward passes: 2

--------------------------------------------------------------------------------
 iter |   lower ($) |   simul ($) |  gap (%) |  fwd (s) |  bwd (s) | total (s)
--------------------------------------------------------------------------------
    1 |     3200.00 |     3200.00 |     0.00 |    0.003 |    0.001 |    0.004
    2 |     3200.00 |     3200.00 |     0.00 |    0.000 |    0.001 |    0.001
    3 |     3200.00 |     3200.00 |     0.00 |    0.001 |    0.001 |    0.001

Expected cost ($): 3200.00 +- 0.00
```

This problem converges instantly because it's deterministic! See [Example 1 README](../../examples/01-deterministic/README.md) for details.

### Option B: Run the Full 12-Stage Example

Run POWE.RS on the original example:

```bash
cargo run --release example
```

**Expected output** (first few lines):

```
POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

Reading input files from 'example'

# Training
- Iterations: 32
- Forward passes: 4

------------------------------------------------------------
iteration  | lower bound ($) | simulation ($) |   time (s)
------------------------------------------------------------
         1 |        150.0000 |      8449.5644 |         0.01
         2 |       1934.4894 |      2982.8026 |         0.01
         3 |       2589.7422 |      4579.9037 |         0.01
...
```

The optimization should complete in under 1 second! ⚡

---

## Step 3: Understand the Output

### Training Progress

The table shows SDDP convergence:

| Column            | Meaning                                           |
| ----------------- | ------------------------------------------------- |
| `iteration`       | SDDP iteration number (1-32)                      |
| `lower bound ($)` | Lower bound on optimal expected cost (increasing) |
| `simulation ($)`  | Forward pass cost (Monte Carlo sample)            |
| `time (s)`        | Iteration runtime                                 |

**Key observations**:

- **Lower bound increases** monotonically (SDDP guarantee)
- **Simulation cost** varies (random scenarios)
- **Convergence** happens when gap between bounds is small

### Final Results

After training, you'll see:

```
Training time: 0.29 s

Number of constructed cuts by node: 128

# Simulating
- Scenarios: 128

Expected cost ($): 5230.27 +- 2286.20

Simulation time: 0.08 s
```

**Interpretation**:

- **Expected cost**: $5,230 ± $2,286 (mean ± std dev)
- **Cuts**: 128 Benders cuts learned (policy representation)
- **Total time**: ~0.4 seconds

### CSV Output Files

Results are saved to `example/` directory:

| File                      | Content                                  |
| ------------------------- | ---------------------------------------- |
| `cuts.csv`                | Benders cuts (policy representation)     |
| `states.csv`              | Visited states during training           |
| `simulation_buses.csv`    | Bus-level results (load, deficit)        |
| `simulation_lines.csv`    | Transmission line flows                  |
| `simulation_thermals.csv` | Thermal generation by scenario           |
| `simulation_hydros.csv`   | Hydro generation and storage by scenario |

**Tip**: Open these CSVs in Excel, Python (pandas), or R for analysis.

---

## Step 4: Understand the Problem

The example problem represents a simple hydrothermal dispatch:

### Power System

**1 Bus**:

- Load: 75 MW (demand to meet)
- Deficit cost: $200/MWh (penalty for unmet demand)

**1 Thermal Generator**:

- Capacity: 40 MW
- Cost: $15/MWh

**1 Hydro Reservoir**:

- Storage capacity: 100 MWh
- Turbine capacity: 60 MW
- Initial storage: 50 MWh
- Inflows: Uncertain (log-normal distribution, mean ~30 MWh/stage)

### Optimization Problem

**Objective**: Minimize expected cost over 12 stages (monthly planning)

**Trade-off**:

- Use hydro now (cheap, $0/MWh) → risk running out later
- Save hydro for later (insurance) → use expensive thermal now ($15/MWh)
- Deficit is extremely expensive ($200/MWh) → avoid at all costs

**Uncertainty**: Hydro inflows are random (SDDP learns policy that handles this)

**SDDP Output**: A policy that decides hydro vs. thermal based on current storage and inflows.

---

## Step 5: Customize the Example

### Change Training Parameters

Edit `example/config.json`:

```json
{
  "num_iterations": 50, // More iterations → better convergence
  "num_forward_passes": 10, // More passes → better bounds
  "num_simulation_scenarios": 500, // More scenarios → better policy evaluation
  "seed": 42, // Change seed for different random samples
  "output_path": "./results" // Change output directory
}
```

Run again:

```bash
powers example
```

### Disable CSV Output (Faster)

For benchmarking or repeated runs, disable CSV output:

**Option 1**: Remove `output_path` from `config.json`:

```json
{
  "num_iterations": 100,
  "num_forward_passes": 20,
  "num_simulation_scenarios": 1000,
  "seed": 42
  // No output_path → no CSV files
}
```

**Option 2**: Set to `null`:

```json
{
  "output_path": null
}
```

**Speedup**: 10-30% faster execution (no file I/O)

### Modify the Power System

Edit `example/system.json`:

**Change load**:

```json
{
  "buses": [
    {
      "id": 0,
      "deficit_cost": 200.0,
      "load": 100.0 // Increase from 75 MW → harder problem
    }
  ]
}
```

**Change thermal cost**:

```json
{
  "thermals": [
    {
      "id": 0,
      "bus_id": 0,
      "cost": 25.0, // Increase from 15 → make thermal more expensive
      "min_generation": 0.0,
      "max_generation": 40.0
    }
  ]
}
```

**Change hydro capacity**:

```json
{
  "hydros": [
    {
      "id": 0,
      "bus_id": 0,
      "min_storage": 0.0,
      "max_storage": 150.0, // Increase from 100 MWh → more storage
      "min_turbined_flow": 0.0,
      "max_turbined_flow": 80.0, // Increase from 60 MW → more capacity
      "productivity": 1.0
    }
  ]
}
```

---

## Next Steps

### Learn More About Input Format

📖 **[Input Specification](../reference/INPUT-SPECIFICATION.md)** - Complete documentation of all JSON fields

Key topics:

- Multi-bus systems with transmission
- Multiple hydro reservoirs (cascades)
- Uncertainty distributions (normal, log-normal)
- Stage-varying parameters

### Troubleshoot Issues

🔧 **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common errors and fixes

Common issues:

- JSON syntax errors
- Invalid parameter values
- Numerical issues
- Performance problems

### Use as a Library

For programmatic use in your own code:

```rust
use powers_rs::sddp::SddpAlgorithm;

fn main() -> Result<(), String> {
    // Load from JSON files
    let mut sddp = SddpAlgorithm::from_files(
        "config.json",
        "system.json",
        "graph.json",
        "recourse.json",
    )?;

    // Train policy
    let result = sddp.train()?;
    println!("Final gap: {:.2}", result.final_gap());

    // Simulate policy
    let handlers = sddp.simulate()?;
    println!("Simulated {} scenarios", handlers.len());

    Ok(())
}
```

See **[API Reference](../reference/API-REFERENCE.md)** (coming soon) for library usage.

### Understand the Algorithm

🎓 **[SDDP Overview](../algorithm/SDDP-OVERVIEW.md)** (coming soon) - Mathematical background

Topics:

- Stochastic Dual Dynamic Programming theory
- Forward and backward passes
- Benders cuts and convergence
- Risk measures

---

## Quick Reference

### Command Line Usage

```bash
# Run optimization
powers <input_directory>

# Examples
powers example          # Use example/ directory
powers ./my-problem     # Use custom directory
powers /path/to/data    # Absolute path
```

### Input Files Required

All four JSON files must be present in the input directory:

- `config.json` - Algorithm parameters
- `system.json` - Power system definition
- `graph.json` - Scenario tree structure
- `recourse.json` - Uncertainty and initial conditions

### Output Files Generated

If `output_path` is specified in `config.json`:

- `cuts.csv` - Learned policy (Benders cuts)
- `states.csv` - Visited states
- `simulation_buses.csv` - Bus results
- `simulation_lines.csv` - Line results
- `simulation_thermals.csv` - Thermal results
- `simulation_hydros.csv` - Hydro results

---

## Performance & Memory Usage

### Expected Performance

POWE.RS is highly optimized for performance:

- **Small problems** (5 stages, 1 reservoir): < 1 second
- **Medium problems** (12 stages, 1 reservoir): 1-5 seconds
- **Large problems** (52 stages, multiple reservoirs): 10-60 seconds

**Parallel Performance**:
- Utilizes all available CPU cores via Rayon
- Best performance with 2-4 cores (80% efficiency)
- Scales to 8+ cores (with diminishing returns)

### Memory Requirements

POWE.RS has excellent memory efficiency:

| Problem Size | Stages | Expected Memory |
|--------------|--------|-----------------|
| Small        | 5      | ~10-12 MB       |
| Medium       | 12     | ~15-20 MB       |
| Large        | 24     | ~25-30 MB       |
| Very Large   | 52     | ~45-55 MB       |

**Memory Formula**:
```
Memory (MB) ≈ 7 + (stages × 0.75) + (total_cuts × 0.0001)
```

**Key Characteristics**:
- ✅ **Stable memory**: No growth across iterations
- ✅ **Predictable scaling**: Linear with problem size  
- ✅ **Efficient storage**: ~81 bytes per cut (1 state variable)

### When to Worry About Memory

✅ **You're fine if**:
- Problem has < 100 stages (< 82 MB)
- Running on machine with > 1 GB RAM
- Memory usage is stable during training

⚠️ **Monitor if**:
- Problem has > 200 stages (> 150 MB)
- Many state variables (> 20)
- Running on memory-constrained systems

### Performance Tips

1. **Use release builds** for production:
   ```bash
   cargo build --release
   ./target/release/powers example
   ```

2. **Adjust thread count** if needed:
   ```bash
   RAYON_NUM_THREADS=4 powers example
   ```

3. **Monitor with timing detail**:
   ```bash
   POWERS_TIMING_DETAIL=1 powers example
   ```

4. **Profile if needed**:
   ```bash
   cargo bench --bench memory_profiling  # Memory profiling
   cargo bench --bench sddp_benchmarks    # Performance benchmarks
   ```

For detailed performance analysis, see:
- [Performance Baselines](../performance/PERFORMANCE-BASELINES.md)
- [Memory Profiling](../performance/MEMORY-PROFILING.md)
- [Parallel Efficiency](../performance/PARALLELISM.md)

---

## Tips for Success

### Start Small

- Begin with the provided example (1 hydro, 1 bus, 12 stages)
- Understand the results before scaling up
- Gradually add complexity (more reservoirs, longer horizon)

### Iterate on Configuration

- Start with fewer iterations (10-20) for quick feedback
- Increase iterations (50-100) for production runs
- Use more forward passes (10-20) if convergence is slow

### Monitor Convergence

- Lower bound should increase monotonically
- Gap should decrease over iterations
- If bounds don't converge, problem may be poorly scaled or infeasible

### Validate Results

- Check simulation results make sense (hydro + thermal = load)
- Verify storage constraints are satisfied
- Compare with deterministic solution (if available)

---

**Navigation**: [← Installation](INSTALLATION.md) | [↑ Documentation Index](../README.md) | [→ Input Specification](../reference/INPUT-SPECIFICATION.md)

```
