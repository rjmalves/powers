#!/usr/bin/env python3
"""
Script to migrate example recourse.json files from v0.2.0 to v0.3.0 format.

Changes:
- Rename marginal_distribution -> distribution
- Remove innovation_distribution and residual_distribution fields
- Remove Autoregressive temporal model support (convert to PAR)
"""

import json
import sys
from pathlib import Path

def migrate_noise_model(noise_model):
    """Migrate a single noise model to v0.3.0 format."""
    migrated = {}
    
    # Copy standard fields
    for field in ['uncertainty_type', 'entity_id', 'season_id']:
        if field in noise_model:
            migrated[field] = noise_model[field]
    
    # Migrate distribution field
    if 'marginal_distribution' in noise_model:
        migrated['distribution'] = noise_model['marginal_distribution']
    elif 'distribution' in noise_model:
        migrated['distribution'] = noise_model['distribution']
    else:
        raise ValueError("No distribution field found")
    
    # Copy temporal_model (convert AR to PAR if needed)
    if 'temporal_model' in noise_model:
        temporal = noise_model['temporal_model']
        if temporal.get('type') == 'autoregressive':
            # Convert AR to PAR with num_seasons=1
            lag_order = temporal.get('lag_order', 1)
            coefficients = temporal.get('coefficients', [0.5] * lag_order)
            
            migrated['temporal_model'] = {
                'type': 'periodic_autoregressive',
                'num_seasons': 1,
                'ar_orders': [lag_order],
                'ar_coefficients': [coefficients],
                'seasonal_means': [0.0],
                'seasonal_stds': [1.0]
            }
        else:
            migrated['temporal_model'] = temporal
    
    return migrated

def migrate_recourse_file(file_path):
    """Migrate a recourse.json file to v0.3.0 format."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Migrate noise_models
    if 'noise_models' in data:
        data['noise_models'] = [migrate_noise_model(nm) for nm in data['noise_models']]
    
    # Write back
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Migrated {file_path}")

def main():
    base_path = Path("examples")
    recourse_files = list(base_path.glob("*/recourse.json"))
    recourse_files.extend(list(base_path.glob("**/*/recourse.json")))
    
    for file_path in recourse_files:
        try:
            migrate_recourse_file(file_path)
        except Exception as e:
            print(f"Error migrating {file_path}: {e}")

if __name__ == "__main__":
    main()