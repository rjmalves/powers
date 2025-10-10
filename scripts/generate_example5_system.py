#!/usr/bin/env python3
"""
Generate large-scale Brazilian system for Example 5.
- ~160 hydro plants across 5 buses with multiple cascades
- ~120 thermal plants across 5 buses
- 5 buses representing Brazilian regions
- Realistic capacity distributions
"""

import json
import random

random.seed(42)

# Brazilian regions (buses)
BUSES = [
    {"id": 0, "name": "North", "deficit_cost": 500.0},
    {"id": 1, "name": "Northeast", "deficit_cost": 500.0},
    {"id": 2, "name": "Southeast", "deficit_cost": 500.0},
    {"id": 3, "name": "South", "deficit_cost": 500.0},
    {"id": 4, "name": "Central-West", "deficit_cost": 500.0}
]

# Transmission lines between regions (simplified topology)
LINES = [
    {"id": 0, "source_bus_id": 0, "target_bus_id": 1, "direct_capacity": 1500.0, "reverse_capacity": 1500.0, "exchange_penalty": 2.0},  # North-Northeast
    {"id": 1, "source_bus_id": 1, "target_bus_id": 2, "direct_capacity": 2500.0, "reverse_capacity": 2500.0, "exchange_penalty": 3.0},  # Northeast-Southeast
    {"id": 2, "source_bus_id": 2, "target_bus_id": 3, "direct_capacity": 3000.0, "reverse_capacity": 3000.0, "exchange_penalty": 2.0},  # Southeast-South
    {"id": 3, "source_bus_id": 3, "target_bus_id": 4, "direct_capacity": 1200.0, "reverse_capacity": 1200.0, "exchange_penalty": 3.0},  # South-CentralWest
]

# Hydro cascade configurations for each region
# Each number represents a hydro plant size (used for capacity scaling)
HYDRO_CASCADES = [
    # North: High hydro concentration (Amazon basin) - Target: 70 hydros
    {"bus_id": 0, "cascades": [
        [10, 9, 8, 7, 6, 5, 4, 3],  # Major Amazon cascade (8 plants)
        [9, 8, 7, 6, 5, 4, 3],       # Large cascade (7 plants)
        [8, 7, 6, 5, 4, 3],          # Medium-large 1 (6 plants)
        [7, 6, 5, 4, 3, 2],          # Medium-large 2 (6 plants)
        [6, 5, 4, 3, 2],             # Medium 1 (5 plants)
        [6, 5, 4, 3],                # Medium 2 (4 plants)
        [5, 4, 3, 2],                # Small-medium 1 (4 plants)
        [5, 4, 3],                   # Small-medium 2 (3 plants)
        [4, 3, 2],                   # Small-medium 3 (3 plants)
        [4, 3],                      # Small 1 (2 plants)
        [3, 2],                      # Small 2 (2 plants)
        [3, 2],                      # Small 3 (2 plants)
        [3, 2]                       # Small 4 (2 plants)
    ]},  # Total: 8+7+6+6+5+4+4+3+3+2+2+2+2 = 54 hydros
    
    # Northeast: Limited hydro - Target: 30 hydros
    {"bus_id": 1, "cascades": [
        [7, 6, 5, 4, 3],             # 5 plants
        [6, 5, 4, 3],                # 4 plants
        [5, 4, 3, 2],                # 4 plants
        [5, 4, 3],                   # 3 plants
        [4, 3, 2],                   # 3 plants
        [4, 3],                      # 2 plants
        [3, 2],                      # 2 plants
        [3, 2],                      # 2 plants
        [3, 2]                       # 2 plants
    ]},  # Total: 5+4+4+3+3+2+2+2+2 = 27 hydros
    
    # Southeast: Major cascades (Paraná basin) - Target: 45 hydros
    {"bus_id": 2, "cascades": [
        [11, 10, 9, 8, 7, 6, 5, 4, 3],  # Major Paraná cascade (9 plants)
        [9, 8, 7, 6, 5, 4],              # Large cascade (6 plants)
        [8, 7, 6, 5, 4],                 # Medium-large 1 (5 plants)
        [7, 6, 5, 4],                    # Medium-large 2 (4 plants)
        [6, 5, 4, 3],                    # Medium 1 (4 plants)
        [6, 5, 4],                       # Medium 2 (3 plants)
        [5, 4, 3],                       # Small-medium 1 (3 plants)
        [5, 4],                          # Small-medium 2 (2 plants)
        [4, 3]                           # Small (2 plants)
    ]},  # Total: 9+6+5+4+4+3+3+2+2 = 38 hydros
    
    # South: Medium hydro - Target: 28 hydros
    {"bus_id": 3, "cascades": [
        [8, 7, 6, 5, 4],                 # 5 plants
        [7, 6, 5, 4],                    # 4 plants
        [6, 5, 4],                       # 3 plants
        [5, 4, 3],                       # 3 plants
        [5, 4],                          # 2 plants
        [4, 3],                          # 2 plants
        [3, 2]                           # 2 plants
    ]},  # Total: 5+4+3+3+2+2+2 = 21 hydros
    
    # Central-West: Medium hydro - Target: 20 hydros
    {"bus_id": 4, "cascades": [
        [7, 6, 5, 4],                    # 4 plants
        [6, 5, 4],                       # 3 plants
        [5, 4, 3],                       # 3 plants
        [5, 4],                          # 2 plants
        [4, 3],                          # 2 plants
        [3, 2]                           # 2 plants
    ]},  # Total: 4+3+3+2+2+2 = 16 hydros
]  # Grand total: 54+27+38+21+16 = 156 hydros

# Thermal plant counts per bus (distribute ~120 thermals)
# Each entry has bus_id and counts for [nuclear, coal, gas, oil]
THERMAL_CONFIG = [
    {"bus_id": 0, "nuclear": 2, "coal": 6, "gas": 10, "oil": 4},   # North: 22 thermals
    {"bus_id": 1, "nuclear": 1, "coal": 5, "gas": 10, "oil": 5},   # Northeast: 21 thermals
    {"bus_id": 2, "nuclear": 4, "coal": 9, "gas": 14, "oil": 7},   # Southeast: 34 thermals - most industrialized
    {"bus_id": 3, "nuclear": 2, "coal": 7, "gas": 11, "oil": 5},   # South: 25 thermals
    {"bus_id": 4, "nuclear": 1, "coal": 5, "gas": 9, "oil": 4}     # Central-West: 19 thermals
]  # Total: 10 nuclear + 32 coal + 54 gas + 25 oil = 121 thermals

def generate_hydros():
    """Generate ~160 hydro plants with cascades."""
    hydros = []
    hydro_id = 0
    
    for region in HYDRO_CASCADES:
        bus_id = region["bus_id"]
        
        for cascade in region["cascades"]:
            upstream_id = None
            
            for reservoir_size in cascade:
                # Storage capacity scales with position in cascade
                # Upstream reservoirs are larger
                storage_capacity = reservoir_size * random.uniform(800, 1200)  # MWh
                turbining_capacity = reservoir_size * random.uniform(80, 120)  # MW
                
                hydro = {
                    "id": hydro_id,
                    "downstream_hydro_id": upstream_id,  # Reversed: points to upstream
                    "bus_id": bus_id,
                    "productivity": round(random.uniform(0.85, 1.05), 3),
                    "min_storage": 0.0,
                    "max_storage": round(storage_capacity, 2),
                    "min_turbined_flow": 0.0,
                    "max_turbined_flow": round(turbining_capacity, 2),
                    "spillage_penalty": 1.0
                }
                
                hydros.append(hydro)
                upstream_id = hydro_id
                hydro_id += 1
    
    return hydros

def generate_thermals():
    """Generate ~120 thermal plants with realistic costs."""
    thermals = []
    thermal_id = 0
    
    # Cost ranges by type ($/MWh)
    costs = {
        "nuclear": (35, 45),
        "coal": (55, 65),
        "gas": (75, 125),
        "oil": (140, 160)
    }
    
    # Capacity ranges by type (MW)
    capacities = {
        "nuclear": (1200, 1500),
        "coal": (600, 900),
        "gas": (300, 600),
        "oil": (100, 250)
    }
    
    for config in THERMAL_CONFIG:
        bus_id = config["bus_id"]
        
        for thermal_type in ["nuclear", "coal", "gas", "oil"]:
            num_plants = config[thermal_type]
            cost_range = costs[thermal_type]
            capacity_range = capacities[thermal_type]
            
            for _ in range(num_plants):
                thermal = {
                    "id": thermal_id,
                    "bus_id": bus_id,
                    "cost": round(random.uniform(*cost_range), 2),
                    "min_generation": 0.0,
                    "max_generation": round(random.uniform(*capacity_range), 2)
                }
                
                thermals.append(thermal)
                thermal_id += 1
    
    return thermals

def main():
    hydros = generate_hydros()
    thermals = generate_thermals()
    
    system = {
        "buses": BUSES,
        "lines": LINES,
        "thermals": thermals,
        "hydros": hydros
    }
    
    print(f"Generated system:")
    print(f"  Buses: {len(BUSES)}")
    print(f"  Lines: {len(LINES)}")
    print(f"  Hydros: {len(hydros)}")
    print(f"  Thermals: {len(thermals)}")
    print(f"  Total capacity (hydro): {sum(h['max_turbined_flow'] for h in hydros):.0f} MW")
    print(f"  Total capacity (thermal): {sum(t['max_generation'] for t in thermals):.0f} MW")
    print(f"  Total storage: {sum(h['max_storage'] for h in hydros):.0f} MWh")
    
    with open("examples/05-large-scale-brazilian/system.json", "w") as f:
        json.dump(system, f, indent=2)
    
    print("\nSystem written to examples/05-large-scale-brazilian/system.json")

if __name__ == "__main__":
    main()
