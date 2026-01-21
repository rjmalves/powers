#!/usr/bin/env python3
"""
POWE.RS LP Subproblem Sizing Calculator

Calculates the expected number of variables, constraints, and memory usage
for SDDP subproblems based on system configuration.

Usage:
    python lp_sizing.py config.json
    python lp_sizing.py --interactive
    
Input JSON format:
{
    "n_buses": 6,
    "n_blocks": 3,
    "n_hydros": 160,
    "n_thermals": 130,
    "n_lines": 10,
    "n_pumps": 5,
    "n_contracts_import": 3,
    "n_contracts_export": 2,
    "n_batteries": 0,
    "n_generic_constraints": 50,
    "avg_deficit_segments": 3,
    "avg_thermal_segments": 1.5,
    "n_hydros_with_diversion": 10,
    "n_hydros_with_evaporation": 50,
    "n_hydros_with_withdrawal": 20,
    "n_hydros_with_fpha": 50,
    "avg_fpha_planes": 10,
    "max_ar_order": 12,
    "avg_ar_order": 6,
    "n_gnl_thermals": 0,
    "avg_gnl_lag": 2,
    "n_cuts_capacity": 15000,
    "n_stages": 120,
    "n_iterations": 50,
    "n_forward_passes": 200
}
"""

import json
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemConfig:
    """System configuration for LP sizing calculations."""
    
    # Core entities
    n_buses: int = 6
    n_blocks: int = 3
    n_hydros: int = 160
    n_thermals: int = 130
    n_lines: int = 10
    n_pumps: int = 5
    n_contracts_import: int = 3
    n_contracts_export: int = 2
    n_batteries: int = 0
    n_generic_constraints: int = 50
    
    # Segments and planes
    avg_deficit_segments: float = 3.0
    avg_thermal_segments: float = 1.5
    n_hydros_with_diversion: int = 10
    n_hydros_with_evaporation: int = 50
    n_hydros_with_withdrawal: int = 20
    n_hydros_with_fpha: int = 50
    avg_fpha_planes: float = 10.0
    
    # AR model
    max_ar_order: int = 12
    avg_ar_order: float = 6.0
    
    # GNL
    n_gnl_thermals: int = 0
    avg_gnl_lag: float = 2.0
    
    # Algorithm parameters
    n_cuts_capacity: int = 15000
    n_stages: int = 120
    n_iterations: int = 50
    n_forward_passes: int = 200
    
    # Slack variable types per hydro per block
    n_slack_types: int = 6  # turbined_min, outflow_min/max, gen_min, evap+/-, withdrawal
    
    @classmethod
    def from_json(cls, data: dict) -> "SystemConfig":
        """Create config from JSON dict, using defaults for missing fields."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class LPSizing:
    """LP subproblem sizing results."""
    
    # Variables
    n_vars_theta: int = 1
    n_vars_deficit: int = 0
    n_vars_excess: int = 0
    n_vars_exchange: int = 0
    n_vars_hydro_storage: int = 0
    n_vars_hydro_flow: int = 0  # turbined + spillage + generation + inflow
    n_vars_hydro_diversion: int = 0
    n_vars_hydro_evaporation: int = 0
    n_vars_hydro_withdrawal: int = 0
    n_vars_hydro_slacks: int = 0
    n_vars_thermal: int = 0
    n_vars_contracts: int = 0
    n_vars_pumping: int = 0
    n_vars_battery: int = 0
    
    # Constraints
    n_cons_load_balance: int = 0
    n_cons_water_balance: int = 0
    n_cons_generation_constant: int = 0
    n_cons_generation_fpha: int = 0
    n_cons_outflow_def: int = 0
    n_cons_outflow_bounds: int = 0
    n_cons_turbined_min: int = 0
    n_cons_generation_min: int = 0
    n_cons_evaporation: int = 0
    n_cons_withdrawal: int = 0
    n_cons_generic: int = 0
    n_cons_cuts: int = 0
    
    # State dimension
    n_state_storage: int = 0
    n_state_ar_lags: int = 0
    n_state_battery: int = 0
    n_state_gnl: int = 0
    
    @property
    def total_variables(self) -> int:
        return (
            self.n_vars_theta +
            self.n_vars_deficit +
            self.n_vars_excess +
            self.n_vars_exchange +
            self.n_vars_hydro_storage +
            self.n_vars_hydro_flow +
            self.n_vars_hydro_diversion +
            self.n_vars_hydro_evaporation +
            self.n_vars_hydro_withdrawal +
            self.n_vars_hydro_slacks +
            self.n_vars_thermal +
            self.n_vars_contracts +
            self.n_vars_pumping +
            self.n_vars_battery
        )
    
    @property
    def total_constraints(self) -> int:
        return (
            self.n_cons_load_balance +
            self.n_cons_water_balance +
            self.n_cons_generation_constant +
            self.n_cons_generation_fpha +
            self.n_cons_outflow_def +
            self.n_cons_outflow_bounds +
            self.n_cons_turbined_min +
            self.n_cons_generation_min +
            self.n_cons_evaporation +
            self.n_cons_withdrawal +
            self.n_cons_generic +
            self.n_cons_cuts
        )
    
    @property
    def total_constraints_no_cuts(self) -> int:
        """Constraints excluding pre-allocated cut slots."""
        return self.total_constraints - self.n_cons_cuts
    
    @property
    def state_dimension(self) -> int:
        return (
            self.n_state_storage +
            self.n_state_ar_lags +
            self.n_state_battery +
            self.n_state_gnl
        )


def calculate_sizing(config: SystemConfig) -> LPSizing:
    """Calculate LP sizing from system configuration."""
    
    sizing = LPSizing()
    
    # Variables
    sizing.n_vars_theta = 1
    sizing.n_vars_deficit = int(config.n_buses * config.n_blocks * config.avg_deficit_segments)
    sizing.n_vars_excess = config.n_buses * config.n_blocks
    sizing.n_vars_exchange = 2 * config.n_lines * config.n_blocks
    sizing.n_vars_hydro_storage = config.n_hydros
    sizing.n_vars_hydro_flow = config.n_hydros * config.n_blocks * 4  # q, s, g, inflow
    sizing.n_vars_hydro_diversion = config.n_hydros_with_diversion * config.n_blocks
    sizing.n_vars_hydro_evaporation = config.n_hydros_with_evaporation * config.n_blocks
    sizing.n_vars_hydro_withdrawal = config.n_hydros_with_withdrawal * config.n_blocks
    sizing.n_vars_hydro_slacks = config.n_hydros * config.n_blocks * config.n_slack_types
    sizing.n_vars_thermal = int(config.n_thermals * config.n_blocks * config.avg_thermal_segments)
    sizing.n_vars_contracts = (config.n_contracts_import + config.n_contracts_export) * config.n_blocks
    sizing.n_vars_pumping = config.n_pumps * config.n_blocks * 2  # flow + power
    sizing.n_vars_battery = config.n_batteries * (1 + config.n_blocks * 2)  # SOC + charge/discharge
    
    # Constraints
    sizing.n_cons_load_balance = config.n_buses * config.n_blocks
    sizing.n_cons_water_balance = config.n_hydros
    sizing.n_cons_generation_constant = (config.n_hydros - config.n_hydros_with_fpha) * config.n_blocks
    sizing.n_cons_generation_fpha = int(config.n_hydros_with_fpha * config.n_blocks * config.avg_fpha_planes)
    sizing.n_cons_outflow_def = config.n_hydros * config.n_blocks
    sizing.n_cons_outflow_bounds = 2 * config.n_hydros * config.n_blocks
    sizing.n_cons_turbined_min = config.n_hydros * config.n_blocks
    sizing.n_cons_generation_min = config.n_hydros * config.n_blocks
    sizing.n_cons_evaporation = config.n_hydros_with_evaporation * config.n_blocks
    sizing.n_cons_withdrawal = config.n_hydros_with_withdrawal * config.n_blocks
    sizing.n_cons_generic = config.n_generic_constraints
    sizing.n_cons_cuts = config.n_cuts_capacity
    
    # State dimension
    sizing.n_state_storage = config.n_hydros
    sizing.n_state_ar_lags = int(config.n_hydros * config.avg_ar_order)
    sizing.n_state_battery = config.n_batteries
    sizing.n_state_gnl = int(config.n_gnl_thermals * config.avg_gnl_lag)
    
    return sizing


def estimate_memory(sizing: LPSizing, config: SystemConfig) -> dict:
    """Estimate memory usage in bytes."""
    
    # LP matrix memory (sparse CSC format estimate)
    # Assume average 5 non-zeros per column
    avg_nnz_per_col = 5
    n_nnz = sizing.total_variables * avg_nnz_per_col
    lp_matrix_bytes = (
        n_nnz * 8 +  # values (f64)
        n_nnz * 4 +  # row indices (i32)
        sizing.total_variables * 4  # column pointers (i32)
    )
    
    # Variable bounds and objective
    var_bounds_bytes = sizing.total_variables * 8 * 3  # lb, ub, obj
    
    # Constraint bounds
    con_bounds_bytes = sizing.total_constraints * 8 * 2  # lb, ub
    
    # Cut storage per stage
    cut_coefficient_bytes = config.n_cuts_capacity * sizing.state_dimension * 8
    cut_rhs_bytes = config.n_cuts_capacity * 8
    cut_metadata_bytes = config.n_cuts_capacity * 32  # id, iteration, flags
    cuts_per_stage = cut_coefficient_bytes + cut_rhs_bytes + cut_metadata_bytes
    
    # Total cuts across all stages
    total_cuts_bytes = cuts_per_stage * config.n_stages
    
    # Solver workspace (estimate ~15MB per instance)
    solver_workspace_bytes = 15 * 1024 * 1024
    
    return {
        "lp_matrix_bytes": lp_matrix_bytes,
        "var_bounds_bytes": var_bounds_bytes,
        "con_bounds_bytes": con_bounds_bytes,
        "cuts_per_stage_bytes": cuts_per_stage,
        "total_cuts_bytes": total_cuts_bytes,
        "solver_workspace_bytes": solver_workspace_bytes,
        "per_stage_total_bytes": lp_matrix_bytes + var_bounds_bytes + con_bounds_bytes + cuts_per_stage,
        "per_rank_estimate_bytes": total_cuts_bytes + solver_workspace_bytes,
    }


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def print_report(config: SystemConfig, sizing: LPSizing, memory: dict):
    """Print formatted sizing report."""
    
    print("=" * 70)
    print("POWE.RS LP SUBPROBLEM SIZING REPORT")
    print("=" * 70)
    
    print("\n--- SYSTEM CONFIGURATION ---")
    print(f"  Buses:           {config.n_buses:>6}")
    print(f"  Blocks:          {config.n_blocks:>6}")
    print(f"  Hydros:          {config.n_hydros:>6}")
    print(f"  Thermals:        {config.n_thermals:>6}")
    print(f"  Lines:           {config.n_lines:>6}")
    print(f"  Pumping stations:{config.n_pumps:>6}")
    print(f"  Contracts:       {config.n_contracts_import + config.n_contracts_export:>6}")
    print(f"  Batteries:       {config.n_batteries:>6}")
    print(f"  Generic constr.: {config.n_generic_constraints:>6}")
    print(f"  Stages:          {config.n_stages:>6}")
    print(f"  Cut capacity:    {config.n_cuts_capacity:>6}")
    
    print("\n--- VARIABLE COUNTS ---")
    print(f"  Future cost (θ):       {sizing.n_vars_theta:>8}")
    print(f"  Deficit:               {sizing.n_vars_deficit:>8}")
    print(f"  Excess:                {sizing.n_vars_excess:>8}")
    print(f"  Exchange:              {sizing.n_vars_exchange:>8}")
    print(f"  Hydro storage:         {sizing.n_vars_hydro_storage:>8}")
    print(f"  Hydro flow:            {sizing.n_vars_hydro_flow:>8}")
    print(f"  Hydro diversion:       {sizing.n_vars_hydro_diversion:>8}")
    print(f"  Hydro evaporation:     {sizing.n_vars_hydro_evaporation:>8}")
    print(f"  Hydro withdrawal:      {sizing.n_vars_hydro_withdrawal:>8}")
    print(f"  Hydro slacks:          {sizing.n_vars_hydro_slacks:>8}")
    print(f"  Thermal:               {sizing.n_vars_thermal:>8}")
    print(f"  Contracts:             {sizing.n_vars_contracts:>8}")
    print(f"  Pumping:               {sizing.n_vars_pumping:>8}")
    print(f"  Battery:               {sizing.n_vars_battery:>8}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL VARIABLES:       {sizing.total_variables:>8}")
    
    print("\n--- CONSTRAINT COUNTS ---")
    print(f"  Load balance:          {sizing.n_cons_load_balance:>8}")
    print(f"  Water balance:         {sizing.n_cons_water_balance:>8}")
    print(f"  Generation (constant): {sizing.n_cons_generation_constant:>8}")
    print(f"  Generation (FPHA):     {sizing.n_cons_generation_fpha:>8}")
    print(f"  Outflow definition:    {sizing.n_cons_outflow_def:>8}")
    print(f"  Outflow bounds:        {sizing.n_cons_outflow_bounds:>8}")
    print(f"  Turbined minimum:      {sizing.n_cons_turbined_min:>8}")
    print(f"  Generation minimum:    {sizing.n_cons_generation_min:>8}")
    print(f"  Evaporation:           {sizing.n_cons_evaporation:>8}")
    print(f"  Water withdrawal:      {sizing.n_cons_withdrawal:>8}")
    print(f"  Generic:               {sizing.n_cons_generic:>8}")
    print(f"  Benders cuts (slots):  {sizing.n_cons_cuts:>8}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL CONSTRAINTS:     {sizing.total_constraints:>8}")
    print(f"  (excluding cut slots): {sizing.total_constraints_no_cuts:>8}")
    
    print("\n--- STATE DIMENSION ---")
    print(f"  Storage states:        {sizing.n_state_storage:>8}")
    print(f"  AR lag states:         {sizing.n_state_ar_lags:>8}")
    print(f"  Battery SOC states:    {sizing.n_state_battery:>8}")
    print(f"  GNL pipeline states:   {sizing.n_state_gnl:>8}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL STATE DIM:       {sizing.state_dimension:>8}")
    
    print("\n--- MEMORY ESTIMATES ---")
    print(f"  LP matrix (sparse):    {format_bytes(memory['lp_matrix_bytes']):>12}")
    print(f"  Variable bounds:       {format_bytes(memory['var_bounds_bytes']):>12}")
    print(f"  Constraint bounds:     {format_bytes(memory['con_bounds_bytes']):>12}")
    print(f"  Cuts per stage:        {format_bytes(memory['cuts_per_stage_bytes']):>12}")
    print(f"  Total cuts (all stages): {format_bytes(memory['total_cuts_bytes']):>12}")
    print(f"  Solver workspace:      {format_bytes(memory['solver_workspace_bytes']):>12}")
    print(f"  ─────────────────────────────────")
    print(f"  Per-rank estimate:     {format_bytes(memory['per_rank_estimate_bytes']):>12}")
    
    print("\n--- LP PROBLEM SIZE SUMMARY ---")
    print(f"  Rows × Columns:        {sizing.total_constraints} × {sizing.total_variables}")
    print(f"  Active rows (typical): {sizing.total_constraints_no_cuts + config.n_cuts_capacity // 3}")
    print(f"  Density estimate:      {100 * 5 / sizing.total_constraints:.2f}%")
    
    print("=" * 70)


def interactive_mode():
    """Run in interactive mode, prompting for values."""
    print("POWE.RS LP Sizing Calculator - Interactive Mode")
    print("Press Enter to use default values shown in [brackets]\n")
    
    def prompt(name: str, default: float, is_int: bool = True) -> float:
        val = input(f"  {name} [{default}]: ").strip()
        if not val:
            return default
        return int(val) if is_int else float(val)
    
    config = SystemConfig()
    
    print("--- Core Entities ---")
    config.n_buses = prompt("Number of buses", config.n_buses)
    config.n_blocks = prompt("Number of blocks per stage", config.n_blocks)
    config.n_hydros = prompt("Number of hydros", config.n_hydros)
    config.n_thermals = prompt("Number of thermals", config.n_thermals)
    config.n_lines = prompt("Number of transmission lines", config.n_lines)
    config.n_pumps = prompt("Number of pumping stations", config.n_pumps)
    config.n_contracts_import = prompt("Number of import contracts", config.n_contracts_import)
    config.n_contracts_export = prompt("Number of export contracts", config.n_contracts_export)
    config.n_batteries = prompt("Number of batteries", config.n_batteries)
    config.n_generic_constraints = prompt("Number of generic constraints", config.n_generic_constraints)
    
    print("\n--- Model Details ---")
    config.avg_deficit_segments = prompt("Avg deficit segments per bus", config.avg_deficit_segments, False)
    config.avg_thermal_segments = prompt("Avg cost segments per thermal", config.avg_thermal_segments, False)
    config.n_hydros_with_fpha = prompt("Hydros with FPHA model", config.n_hydros_with_fpha)
    config.avg_fpha_planes = prompt("Avg FPHA planes per hydro", config.avg_fpha_planes, False)
    config.avg_ar_order = prompt("Avg AR order for inflows", config.avg_ar_order, False)
    
    print("\n--- Algorithm Parameters ---")
    config.n_stages = prompt("Number of stages", config.n_stages)
    config.n_cuts_capacity = prompt("Cut capacity per stage", config.n_cuts_capacity)
    
    return config


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print(__doc__)
        sys.exit(0)
    
    if sys.argv[1] == "--interactive":
        config = interactive_mode()
    else:
        # Load from JSON file
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
        config = SystemConfig.from_json(data)
    
    sizing = calculate_sizing(config)
    memory = estimate_memory(sizing, config)
    print_report(config, sizing, memory)
    
    # Output JSON for programmatic use
    if len(sys.argv) > 2 and sys.argv[2] == "--json":
        result = {
            "config": config.__dict__,
            "sizing": {
                "total_variables": sizing.total_variables,
                "total_constraints": sizing.total_constraints,
                "total_constraints_no_cuts": sizing.total_constraints_no_cuts,
                "state_dimension": sizing.state_dimension,
                "variables": {
                    "theta": sizing.n_vars_theta,
                    "deficit": sizing.n_vars_deficit,
                    "excess": sizing.n_vars_excess,
                    "exchange": sizing.n_vars_exchange,
                    "hydro_storage": sizing.n_vars_hydro_storage,
                    "hydro_flow": sizing.n_vars_hydro_flow,
                    "hydro_slacks": sizing.n_vars_hydro_slacks,
                    "thermal": sizing.n_vars_thermal,
                    "contracts": sizing.n_vars_contracts,
                    "pumping": sizing.n_vars_pumping,
                    "battery": sizing.n_vars_battery,
                },
                "constraints": {
                    "load_balance": sizing.n_cons_load_balance,
                    "water_balance": sizing.n_cons_water_balance,
                    "generation": sizing.n_cons_generation_constant + sizing.n_cons_generation_fpha,
                    "outflow": sizing.n_cons_outflow_def + sizing.n_cons_outflow_bounds,
                    "generic": sizing.n_cons_generic,
                    "cuts": sizing.n_cons_cuts,
                },
                "state": {
                    "storage": sizing.n_state_storage,
                    "ar_lags": sizing.n_state_ar_lags,
                    "battery": sizing.n_state_battery,
                    "gnl": sizing.n_state_gnl,
                },
            },
            "memory": memory,
        }
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
