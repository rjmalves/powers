# Example 1: Simple PAR(1) with Monthly Inflows

## Overview

This example demonstrates a simple single-reservoir system with monthly PAR(1) inflow model.

## System Configuration

- **Single hydro plant**: 200 MWmonth storage capacity, 100 MW max turbining
- **One thermal plant**: 50 MW capacity at $100/MWh
- **Load pattern**: Varies seasonally from 50-80 MW (dry season: 50 MW, wet season: 80 MW)

## PAR Model Configuration

**Period**: 12 (monthly seasonality)  
**Order**: PAR(1) - first-order autoregressive  
**Seasonal Pattern**: 
- Wet season (Dec-Feb): High mean inflows (~90 m³/s), higher variability (σ=25)
- Transition (Mar-May, Sep-Nov): Medium inflows (~70 m³/s), medium variability (σ=20)
- Dry season (Jun-Aug): Low inflows (~50 m³/s), lower variability (σ=15)

**AR Coefficient**: φ = 0.7 for all months (moderate persistence)

## Scenario Tree

- **Stages**: 12 (one per month)
- **Scenarios per stage**: 10 (after initial stage)
- **Total scenarios**: Approximately 10^11 paths (pruned via SDDP)

## Expected Behavior

1. **Wet season**: Hydro dominates, reservoirs fill up
2. **Transition**: Mixed hydro/thermal dispatch
3. **Dry season**: More thermal generation, reservoir drawdown
4. **AR persistence**: High inflow month → likely high next month (φ=0.7)

## Running the Example

```bash
cargo run --release -- examples/06-par-model/01-simple-par1
```

**Expected runtime**: ~5-10 seconds (20 iterations)

## Key Files

- `config.json`: SDDP algorithm parameters
- `system.json`: Hydrothermal system topology
- `graph.json`: 12-stage scenario tree with seasonal indices
- `recourse.json`: PAR(1) parameters for monthly inflows

## Learning Points

1. **Period configuration**: `season_id` in graph nodes maps to PAR periods (0-11 for months)
2. **State variables**: `storage_and_inflow` required for PAR (stores past residuals)
3. **AR coefficients**: Higher φ → more persistence (smoother inflows)
4. **Seasonal means**: Capture annual hydrological cycle
5. **Residual distribution**: Standard normal (can use LogNormal3 for non-negative guarantee)
