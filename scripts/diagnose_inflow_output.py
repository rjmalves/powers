#!/usr/bin/env python3
"""
Diagnostic script to analyze inflow output from simulation_hydros.csv

This script helps identify if changing initial_condition.inflow values
actually affects the simulation output.

Usage:
    python diagnose_inflow_output.py simulation_hydros_1.csv simulation_hydros_2.csv
"""

import sys
import pandas as pd
import numpy as np

def analyze_inflow_csv(filepath):
    """Load and analyze inflow data from simulation CSV"""
    df = pd.read_csv(filepath)
    
    print(f"\n{'='*70}")
    print(f"Analysis of: {filepath}")
    print(f"{'='*70}")
    
    # Basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nNumber of scenarios (series): {df['series_index'].nunique()}")
    print(f"Number of stages: {df['stage_index'].nunique()}")
    print(f"Number of hydro entities: {df['entity_index'].nunique()}")
    
    # Inflow statistics
    print(f"\n{'Inflow Statistics':-^70}")
    print(f"  Mean: {df['inflow'].mean():.4f}")
    print(f"  Std:  {df['inflow'].std():.4f}")
    print(f"  Min:  {df['inflow'].min():.4f}")
    print(f"  Max:  {df['inflow'].max():.4f}")
    
    # Stage 0 analysis (most affected by initial lags)
    stage0 = df[df['stage_index'] == 0]
    print(f"\n{'Stage 0 Inflow Statistics (Most Affected by Initial Lags)':-^70}")
    print(f"  Mean: {stage0['inflow'].mean():.4f}")
    print(f"  Std:  {stage0['inflow'].std():.4f}")
    print(f"  Min:  {stage0['inflow'].min():.4f}")
    print(f"  Max:  {stage0['inflow'].max():.4f}")
    print(f"\n  First 10 stage 0 inflow values:")
    print(f"  {stage0['inflow'].head(10).values}")
    
    # Check for constant values (potential bug indicator)
    unique_inflows = df['inflow'].nunique()
    total_rows = len(df)
    print(f"\n{'Uniqueness Check':-^70}")
    print(f"  Unique inflow values: {unique_inflows} out of {total_rows} total rows")
    if unique_inflows == 1:
        print(f"  ⚠️  WARNING: All inflow values are identical! ({df['inflow'].iloc[0]:.4f})")
        print(f"  This suggests a potential bug in lag buffer initialization or usage.")
    elif unique_inflows < total_rows * 0.1:
        print(f"  ⚠️  WARNING: Very few unique values relative to dataset size.")
        print(f"  This might indicate weak stochastic variation or a bug.")
    else:
        print(f"  ✓ Inflow values show variation (expected for stochastic model)")
    
    # Per-scenario variation
    print(f"\n{'Per-Scenario Variation':-^70}")
    scenario_means = df.groupby('series_index')['inflow'].mean()
    print(f"  Scenario mean inflow std: {scenario_means.std():.4f}")
    print(f"  First 5 scenario means: {scenario_means.head().values}")
    
    return df

def compare_two_csvs(file1, file2):
    """Compare inflow outputs from two different initial lag configurations"""
    df1 = analyze_inflow_csv(file1)
    df2 = analyze_inflow_csv(file2)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON: {file1} vs {file2}")
    print(f"{'='*70}")
    
    # Compare stage 0 (most sensitive to initial lags)
    stage0_1 = df1[df1['stage_index'] == 0]['inflow'].values
    stage0_2 = df2[df2['stage_index'] == 0]['inflow'].values
    
    if len(stage0_1) != len(stage0_2):
        print(f"⚠️  WARNING: Different number of stage 0 observations!")
        print(f"  File 1: {len(stage0_1)}, File 2: {len(stage0_2)}")
        return
    
    mean_diff = abs(stage0_1.mean() - stage0_2.mean())
    max_diff = abs(stage0_1 - stage0_2).max()
    
    print(f"\nStage 0 Inflow Comparison:")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Max difference:  {max_diff:.6f}")
    
    if mean_diff < 0.001 and max_diff < 0.001:
        print(f"\n❌ PROBLEM CONFIRMED: Stage 0 inflows are nearly identical!")
        print(f"   Despite different initial lag values, the simulation produces")
        print(f"   the same inflow values. This indicates a bug in lag buffer")
        print(f"   initialization or usage.")
    elif mean_diff < 1.0:
        print(f"\n⚠️  WEAK EFFECT: Stage 0 inflows show small differences.")
        print(f"   This might indicate:")
        print(f"   - Weak AR coefficient")
        print(f"   - Large stochastic innovation variance dominating AR effect")
        print(f"   - Initial lag effect is small relative to base inflow")
    else:
        print(f"\n✓ EXPECTED BEHAVIOR: Stage 0 inflows show significant differences.")
        print(f"  The initial lag values are correctly affecting the simulation.")
    
    # Show sample differences
    print(f"\n  Sample value differences (first 10):")
    for i in range(min(10, len(stage0_1))):
        diff = stage0_2[i] - stage0_1[i]
        print(f"    Row {i}: {stage0_1[i]:.4f} -> {stage0_2[i]:.4f} (diff: {diff:+.4f})")

def main():
    if len(sys.argv) == 2:
        # Single file analysis
        analyze_inflow_csv(sys.argv[1])
    elif len(sys.argv) == 3:
        # Comparison mode
        compare_two_csvs(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  Single file analysis:")
        print("    python diagnose_inflow_output.py simulation_hydros.csv")
        print()
        print("  Compare two files (different initial lags):")
        print("    python diagnose_inflow_output.py simulation_hydros_lag70.csv simulation_hydros_lag100.csv")
        sys.exit(1)

if __name__ == "__main__":
    main()
