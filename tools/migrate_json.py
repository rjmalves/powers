#!/usr/bin/env python3
"""
JSON Migration Tool for powers-rs

Converts old temporal_model format to new unified format:
- Removes "type" field
- For Independent models: extracts means/stds from seasonal_distributions
- For PAR models: keeps existing structure
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import math


def extract_mean_std_from_distribution(dist: Dict[str, Any]) -> tuple[float, float]:
    """Extract mean and std from a marginal distribution."""
    dist_type = dist.get("type", "normal")
    
    if dist_type == "normal":
        return dist["mean"], dist["std_dev"]
    elif dist_type == "lognormal3":
        # LogNormal3: X = γ + exp(μ + σZ)
        # True mean: γ + exp(μ + σ²/2)
        # True std: exp(μ + σ²/2) * sqrt(exp(σ²) - 1)
        gamma = dist["gamma"]
        mu = dist["mu"]
        sigma = dist["sigma"]
        
        exp_term = math.exp(mu + sigma * sigma / 2.0)
        mean = gamma + exp_term
        std = exp_term * math.sqrt(math.exp(sigma * sigma) - 1.0)
        
        return mean, std
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def migrate_independent_model(
    temporal_model: Dict[str, Any],
    seasonal_distributions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Migrate Independent model to new format."""
    num_seasons = len(seasonal_distributions)
    
    # Extract means and stds from seasonal distributions
    seasonal_means = []
    seasonal_stds = []
    
    for dist_spec in seasonal_distributions:
        # Distribution fields might be nested under "distribution" key or directly in the object
        if "distribution" in dist_spec:
            dist = dist_spec["distribution"]
        else:
            dist = dist_spec
        
        mean, std = extract_mean_std_from_distribution(dist)
        seasonal_means.append(mean)
        seasonal_stds.append(std)
    
    # Create new format
    return {
        "num_seasons": num_seasons,
        "seasonal_means": seasonal_means,
        "seasonal_stds": seasonal_stds,
        "ar_orders": [0] * num_seasons,
        "ar_coefficients": [[]] * num_seasons
    }


def migrate_par_model(temporal_model: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate PAR model to new format (just remove 'type' field)."""
    new_model = temporal_model.copy()
    new_model.pop("type", None)
    return new_model


def migrate_temporal_model(
    temporal_model: Dict[str, Any],
    seasonal_distributions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Migrate a temporal model to new format."""
    model_type = temporal_model.get("type")
    
    if model_type == "independent":
        return migrate_independent_model(temporal_model, seasonal_distributions)
    elif model_type == "periodic_ar":
        return migrate_par_model(temporal_model)
    else:
        # Already in new format or unknown
        return temporal_model


def migrate_uncertainty_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a single uncertainty specification."""
    new_spec = spec.copy()
    
    temporal_model = spec.get("temporal_model", {})
    seasonal_distributions = spec.get("seasonal_distributions", [])
    
    # Migrate temporal model
    new_temporal_model = migrate_temporal_model(temporal_model, seasonal_distributions)
    new_spec["temporal_model"] = new_temporal_model
    
    return new_spec


def migrate_recourse_file(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate an entire recourse.json file."""
    new_data = data.copy()
    
    # Migrate uncertainty specifications if present
    if "uncertainty_specifications" in data:
        new_specs = []
        for spec in data["uncertainty_specifications"]:
            new_spec = migrate_uncertainty_spec(spec)
            new_specs.append(new_spec)
        new_data["uncertainty_specifications"] = new_specs
    
    return new_data


def validate_json(data: Dict[str, Any]) -> bool:
    """Basic validation of JSON structure."""
    if "uncertainty_specifications" not in data:
        return True  # No uncertainty specs, nothing to validate
    
    for spec in data["uncertainty_specifications"]:
        temporal_model = spec.get("temporal_model", {})
        
        # Check required fields
        if "num_seasons" not in temporal_model:
            print(f"Warning: Missing num_seasons in temporal_model", file=sys.stderr)
            return False
        
        num_seasons = temporal_model["num_seasons"]
        
        # Validate array lengths
        for field in ["seasonal_means", "seasonal_stds", "ar_orders", "ar_coefficients"]:
            if field in temporal_model:
                if len(temporal_model[field]) != num_seasons:
                    print(f"Error: {field} length != num_seasons", file=sys.stderr)
                    return False
        
        # Validate ar_coefficients match ar_orders
        if "ar_orders" in temporal_model and "ar_coefficients" in temporal_model:
            ar_orders = temporal_model["ar_orders"]
            ar_coefficients = temporal_model["ar_coefficients"]
            
            for i, (order, coeffs) in enumerate(zip(ar_orders, ar_coefficients)):
                if len(coeffs) != order:
                    print(f"Error: Season {i}: ar_coefficients length {len(coeffs)} != ar_order {order}", file=sys.stderr)
                    return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate powers-rs JSON files to new temporal_model format"
    )
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("-o", "--output", help="Output JSON file (default: stdout)")
    parser.add_argument("-i", "--in-place", action="store_true",
                        help="Modify file in place")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Show changes without writing")
    parser.add_argument("-v", "--validate", action="store_true",
                        help="Validate output JSON")
    
    args = parser.parse_args()
    
    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Migrate
    migrated_data = migrate_recourse_file(data)
    
    # Validate if requested
    if args.validate:
        if not validate_json(migrated_data):
            print("Validation failed!", file=sys.stderr)
            return 1
        print("Validation passed!", file=sys.stderr)
    
    # Convert to JSON string
    output_json = json.dumps(migrated_data, indent=2)
    
    # Determine output
    if args.dry_run:
        print("=== DRY RUN - Changes ===")
        print(output_json)
        return 0
    
    if args.in_place:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        # Write to stdout
        print(output_json)
        return 0
    
    # Write output file
    with open(output_path, 'w') as f:
        f.write(output_json)
        f.write('\n')  # Add trailing newline
    
    print(f"Migrated: {input_path} -> {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
