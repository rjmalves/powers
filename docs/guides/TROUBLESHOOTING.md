# Troubleshooting Guide

This guide helps you diagnose and fix common errors in POWE.RS.

## Table of Contents

- [Validation Errors](#validation-errors)
- [File I/O Errors](#file-io-errors)
- [Solver Errors](#solver-errors)
- [Graph Errors](#graph-errors)

---

## Validation Errors

### Error: "Field 'num_iterations' has invalid value '0'"

**Cause**: Configuration parameter is zero but must be positive.

**Fix**: Set the value to at least 1 in your `config.json`:

```json
{
  "num_iterations": 32, // Change from 0 to a positive number
  "num_forward_passes": 4,
  "num_simulation_scenarios": 128,
  "seed": 42
}
```

**Related Fields**: Same issue can occur with `num_forward_passes` or `num_simulation_scenarios`.

### Error: "Missing required field 'num_iterations'"

**Cause**: Required field is not present in the JSON file.

**Fix**: Add the missing field to your `config.json`:

```json
{
  "num_iterations": 32, // Add this line
  "num_forward_passes": 4,
  "num_simulation_scenarios": 128,
  "seed": 42
}
```

See [INPUT-SPECIFICATION.md](INPUT-SPECIFICATION.md) for complete field documentation.

### Error: "Thermal violates constraint: min_generation <= max_generation"

**Cause**: Minimum generation is greater than maximum generation for a thermal unit.

**Fix**: In `system.json`, ensure min <= max:

```json
{
  "id": 0,
  "bus_id": 0,
  "cost": 50.0,
  "min_generation": 10.0, // Must be <= max_generation
  "max_generation": 100.0
}
```

### Error: "Line references non-existent bus_id=99"

**Cause**: A line references a bus ID that doesn't exist in the system.

**Fix**: Check that bus IDs are contiguous (0, 1, 2, ...) and that all referenced IDs exist:

```json
{
  "buses": [
    { "id": 0, "deficit_cost": 10000.0 },
    { "id": 1, "deficit_cost": 10000.0 }
  ],
  "lines": [
    {
      "id": 0,
      "source_bus_id": 0, // Must reference existing bus
      "target_bus_id": 1, // Must reference existing bus
      "direct_capacity": 100.0,
      "reverse_capacity": 100.0,
      "exchange_penalty": 0.0
    }
  ]
}
```

### Error: "Failed to parse JSON"

**Cause**: JSON syntax error (missing comma, bracket, quote, etc.).

**Fix**:

1. Validate your JSON at https://jsonlint.com/
2. Check for:
   - Missing commas between fields
   - Unmatched brackets `{}` or `[]`
   - Missing quotes around strings
   - Trailing commas (not allowed in strict JSON)

**Example of common mistakes**:

```json
// ❌ WRONG: Missing comma
{
  "num_iterations": 32
  "num_forward_passes": 4
}

// ✅ CORRECT: Comma added
{
  "num_iterations": 32,
  "num_forward_passes": 4
}
```

---

## File I/O Errors

### Error: "File not found: 'examples/03-multistage/config.json'"

**Cause**: File doesn't exist at the specified path.

**Fix**:

1. Check that the file exists: `ls examples/03-multistage/config.json`
2. Check your current directory: `pwd`
3. Use absolute paths if relative paths are confusing
4. Ensure correct file extension (.json, not .txt)

**Debugging**:

```bash
# Check if file exists
ls -l example/config.json

# Check current directory
pwd

# Use absolute path
cargo run /home/user/powers/example
```

### Error: "Permission denied: Cannot read 'example/config.json'"

**Cause**: File exists but you don't have permission to read it.

**Fix**:

```bash
# Check permissions
ls -l example/config.json

# Add read permission
chmod +r example/config.json

# Or add read/write for everyone
chmod 644 example/config.json
```

### Error: "Cannot write to output directory"

**Cause**: Output directory doesn't exist or lacks write permissions.

**Fix**:

```bash
# Create output directory
mkdir -p output/results

# Add write permission
chmod +w output/results
```

---

## Solver Errors

### Error: "Solver failed: Problem is infeasible"

**Cause**: The optimization problem has no feasible solution. Constraints cannot be satisfied simultaneously.

**Common causes**:

- Insufficient generation capacity to meet load
- Transmission capacity too restrictive
- Conflicting constraints (e.g., min > max)
- Storage bounds inconsistent with inflow/outflow

**Fix**:

1. **Check generation capacity**:
   ```
   Total max generation >= Peak load
   ```
2. **Check transmission constraints**:

   - Ensure lines have sufficient capacity
   - Check if islanded buses exist

3. **Check hydro storage bounds**:

   - Initial storage must be within [min_storage, max_storage]
   - Verify productivity is positive
   - Check spillage capacity

4. **Debugging steps**:
   - Reduce load temporarily to isolate issue
   - Increase generation capacities
   - Relax transmission constraints
   - Check for typos in bus_id references

### Error: "Solver failed: Problem is unbounded"

**Cause**: The objective function can go to negative infinity. Usually indicates a modeling error.

**Common causes**:

- Missing upper bounds on generation variables
- Negative costs without capacity constraints
- Missing capacity limits on lines

**Fix**:

1. Ensure all generation has finite upper bounds (max_generation)
2. Check that costs are positive (or zero)
3. Verify all lines have finite capacities

### Error: "Solver failed: Numerical error"

**Cause**: Numerical instability due to poor problem scaling.

**Common causes**:

- Coefficients differ by many orders of magnitude
- Very large values (> 1e10) or very small values (< 1e-10)
- Poor conditioning of constraint matrix

**Fix**:

1. **Scale your data**:

   - Costs: Keep between 1-1000 $/MWh
   - Capacities: Use MW, not W (divide by 1e6)
   - Storage: Use MWh or dam³, not liters

2. **Example scaling**:

   ```json
   // ❌ WRONG: Poor scaling
   {
     "cost": 50000000.0,        // $50M per MWh (too large)
     "max_generation": 0.0001   // 0.1 kW (too small)
   }

   // ✅ CORRECT: Good scaling
   {
     "cost": 50.0,              // $50 per MWh
     "max_generation": 100.0    // 100 MW
   }
   ```

---

## Graph Errors

### Error: "Scenario graph is disconnected. Unreachable nodes: [5, 6, 7]"

**Cause**: Some nodes cannot be reached from the root node (node 0).

**Fix**: Ensure all nodes are connected via edges:

```json
{
  "nodes": [
    {"id": 0, "stage_id": 0, ...},  // Root
    {"id": 1, "stage_id": 1, ...},
    {"id": 2, "stage_id": 1, ...}
  ],
  "edges": [
    {"source_id": 0, "target_id": 1, "probability": 0.5, ...},
    {"source_id": 0, "target_id": 2, "probability": 0.5, ...}
    // Every node (except root) must have at least one incoming edge
  ]
}
```

### Error: "Node 2 has outgoing edge probabilities that don't sum to 1.0. Sum: 0.85"

**Cause**: Outgoing edge probabilities from a node must sum to exactly 1.0.

**Fix**: Adjust probabilities:

```json
{
  "edges": [
    // ❌ WRONG: 0.3 + 0.25 + 0.3 = 0.85
    {"source_id": 2, "target_id": 3, "probability": 0.30, ...},
    {"source_id": 2, "target_id": 4, "probability": 0.25, ...},
    {"source_id": 2, "target_id": 5, "probability": 0.30, ...}

    // ✅ CORRECT: 0.333 + 0.333 + 0.334 = 1.0
    {"source_id": 2, "target_id": 3, "probability": 0.333, ...},
    {"source_id": 2, "target_id": 4, "probability": 0.333, ...},
    {"source_id": 2, "target_id": 5, "probability": 0.334, ...}
  ]
}
```

**Note**: Small floating-point errors (<1e-6) are tolerated.

### Error: "Cycle detected in scenario graph"

**Cause**: The scenario tree must be a Directed Acyclic Graph (DAG). Cycles are not allowed.

**Fix**: Remove edges that create cycles. In a scenario tree:

- Time flows forward (increasing stage_id)
- No edge should point backwards to an earlier stage

---

## Getting More Help

- **Input Format**: See [INPUT-SPECIFICATION.md](INPUT-SPECIFICATION.md) for complete field documentation
- **JSON Schema**: Use `.vscode/settings.json` for IDE auto-completion
- **Examples**: Check `example/` directory for working input files
- **GitHub Issues**: Report bugs at https://github.com/rjmalves/powers/issues
