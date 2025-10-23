#!/usr/bin/env python3
"""
Migrate recourse.json files from old noise_models format to new uncertainty_specifications format.

This script performs the breaking change migration for POWE.RS v0.3.0:
- Removes old `noise_models` field
- Generates new `uncertainty_specifications` field with flattened seasonal distributions
- PRESERVES LogNormal3 distributions (does NOT convert to Normal) - TICKET-16
- Creates backups before modifying files
- Validates converted parameters
- Reports distribution type statistics

CRITICAL: LogNormal3 distributions are preserved in their original form.
Previous versions incorrectly converted LogNormal3 → Normal, causing solver
infeasibilities. The new schema (TICKET-15) supports LogNormal3 natively.

Distribution Handling:
  Normal: Preserved as-is (type, mean, std_dev)
  LogNormal3: Preserved as-is (type, gamma, mu, sigma) - NOT converted

Usage:
    python3 scripts/migrate_examples_to_new_format.py [--dry-run] [--no-backup]
    
Examples:
    # Dry run - show what would be changed
    python3 scripts/migrate_examples_to_new_format.py --dry-run
    
    # Migrate with backups (default)
    python3 scripts/migrate_examples_to_new_format.py
    
    # Migrate without backups (dangerous!)
    python3 scripts/migrate_examples_to_new_format.py --no-backup
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import shutil

def find_recourse_files(root_dir: Path) -> List[Path]:
    """Find all recourse.json files recursively."""
    return list(root_dir.glob("**/recourse.json"))

def validate_seasonal_distribution(dist: Dict[str, Any]) -> None:
    """
    Validate seasonal distribution parameters.
    
    Raises ValueError if parameters are invalid.
    """
    if "type" not in dist:
        raise ValueError("Missing 'type' field in seasonal distribution")
    
    dist_type = dist["type"]
    season_id = dist.get("season_id", "unknown")
    
    if dist_type == "normal":
        if "mean" not in dist:
            raise ValueError(f"Normal distribution (season {season_id}) missing 'mean'")
        if "std_dev" not in dist:
            raise ValueError(f"Normal distribution (season {season_id}) missing 'std_dev'")
        
        std_dev = dist["std_dev"]
        if std_dev <= 0:
            raise ValueError(f"Normal std_dev must be > 0, got {std_dev} (season {season_id})")
    
    elif dist_type == "lognormal3":
        if "gamma" not in dist:
            raise ValueError(f"LogNormal3 distribution (season {season_id}) missing 'gamma'")
        if "mu" not in dist:
            raise ValueError(f"LogNormal3 distribution (season {season_id}) missing 'mu'")
        if "sigma" not in dist:
            raise ValueError(f"LogNormal3 distribution (season {season_id}) missing 'sigma'")
        
        sigma = dist["sigma"]
        gamma = dist["gamma"]
        
        if sigma <= 0:
            raise ValueError(f"LogNormal3 sigma must be > 0, got {sigma} (season {season_id})")
        if gamma < 0:
            raise ValueError(f"LogNormal3 gamma must be >= 0, got {gamma} (season {season_id})")
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

def validate_season_ids(seasonal_dists: List[Dict[str, Any]]) -> None:
    """
    Validate season_id sequence is valid.
    
    Checks for duplicates and warns about non-contiguous sequences.
    """
    season_ids = [d["season_id"] for d in seasonal_dists]
    
    # Check for duplicates
    if len(season_ids) != len(set(season_ids)):
        duplicates = [sid for sid in season_ids if season_ids.count(sid) > 1]
        raise ValueError(f"Duplicate season_id values: {set(duplicates)}")
    
    # Check for contiguous sequence (warn only, not error)
    sorted_ids = sorted(season_ids)
    expected = list(range(len(sorted_ids)))
    if sorted_ids != expected:
        print(f"    ⚠️  Non-contiguous season_ids: {sorted_ids} (expected {expected})")

def group_noise_models_by_entity(noise_models: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """
    Group noise models by (uncertainty_type, entity_id).
    
    Returns:
        Dict mapping (uncertainty_type, entity_id) to list of noise models for that entity.
    """
    grouped = defaultdict(list)
    for model in noise_models:
        key = (model["uncertainty_type"], model["entity_id"])
        grouped[key].append(model)
    return grouped

def convert_independent_model(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert Independent noise models to uncertainty_specification format.
    
    Independent models have one entry per season, each with a distribution.
    
    TICKET-16: Preserves LogNormal3 distributions (does NOT convert to Normal).
    TICKET-17: Converts old 2-parameter lognormal to lognormal3 with gamma=0.
    The new schema (TICKET-15) supports both Normal and LogNormal3 natively.
    """
    if not models:
        raise ValueError("Empty models list for Independent conversion")
    
    first_model = models[0]
    
    # Support both old format (noise_type) and new format (temporal_model)
    if "temporal_model" in first_model:
        temporal_type = first_model["temporal_model"]["type"]
    elif "noise_type" in first_model:
        temporal_type = first_model["noise_type"]
    else:
        raise ValueError("Cannot determine temporal model type")
    
    if temporal_type != "independent":
        raise ValueError(f"Expected Independent model, got {temporal_type}")
    
    # Sort by season_id for consistency
    models_sorted = sorted(models, key=lambda m: m["season_id"])
    
    # Build seasonal_distributions from each model's distribution
    # TICKET-15: Use flattened format (type at same level as season_id)
    seasonal_distributions = []
    for model in models_sorted:
        dist = model["distribution"]
        season_id = model["season_id"]
        
        if dist["type"] == "normal":
            # Normal distribution: preserve mean and std_dev
            seasonal_distributions.append({
                "season_id": season_id,
                "type": "normal",
                "mean": dist["mean"],
                "std_dev": dist["std_dev"]
            })
        elif dist["type"] == "lognormal":
            # TICKET-17: Convert old 2-parameter lognormal to lognormal3 with gamma=0
            # This preserves the lognormal distribution type (not convert to Normal)
            # Old format: {"type": "lognormal", "mu": ..., "sigma": ...}
            # New format: {"type": "lognormal3", "gamma": 0, "mu": ..., "sigma": ...}
            seasonal_distributions.append({
                "season_id": season_id,
                "type": "lognormal3",
                "gamma": 0.0,  # 2-parameter lognormal is equivalent to 3-parameter with gamma=0
                "mu": dist["mu"],
                "sigma": dist["sigma"]
            })
        elif dist["type"] == "lognormal3":
            # TICKET-16: PRESERVE LogNormal3 (do NOT convert to Normal)
            # This is critical - conversion causes solver infeasibilities
            # because Normal can produce negative values while LogNormal3 guarantees X ≥ γ
            seasonal_distributions.append({
                "season_id": season_id,
                "type": "lognormal3",
                "gamma": dist["gamma"],
                "mu": dist["mu"],
                "sigma": dist["sigma"]
            })
        else:
            raise ValueError(f"Unsupported distribution type: {dist['type']}")
    
    return {
        "uncertainty_type": first_model["uncertainty_type"],
        "entity_id": first_model["entity_id"],
        "temporal_model": {
            "type": "independent"
        },
        "seasonal_distributions": seasonal_distributions
    }

def convert_par_model(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert Periodic AR noise models to uncertainty_specification format.
    
    PAR models typically have one entry at season_id=0 with embedded seasonal parameters.
    """
    if not models:
        raise ValueError("Empty models list for PAR conversion")
    
    if len(models) != 1:
        raise ValueError(f"PAR models should have exactly 1 entry (at season_id=0), found {len(models)}")
    
    model = models[0]
    
    # Support both old format (noise_type) and new format (temporal_model)
    if "temporal_model" in model:
        temporal_model = model["temporal_model"]
        temporal_type = temporal_model["type"]
    elif "noise_type" in model:
        temporal_type = model["noise_type"]
        # For old format, the PAR parameters are directly in the model
        temporal_model = model
    else:
        raise ValueError("Cannot determine temporal model type")
    
    if temporal_type != "periodic_ar":
        raise ValueError(f"Expected periodic_ar model, got {temporal_type}")
    
    # Extract PAR parameters
    par_params = {
        "type": "periodic_ar",
        "num_seasons": temporal_model["num_seasons"],
        "ar_orders": temporal_model["ar_orders"],
        "ar_coefficients": temporal_model["ar_coefficients"],
        "seasonal_means": temporal_model["seasonal_means"],
        "seasonal_stds": temporal_model["seasonal_stds"]
    }
    
    return {
        "uncertainty_type": model["uncertainty_type"],
        "entity_id": model["entity_id"],
        "temporal_model": par_params,
        "marginal_distribution": model["distribution"]
    }

def convert_noise_models_to_uncertainty_specs(noise_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert old noise_models format to new uncertainty_specifications format.
    
    Algorithm:
    1. Group noise models by (uncertainty_type, entity_id)
    2. For each entity:
       - If Independent: Create one spec with seasonal_distributions
       - If PAR: Create one spec with embedded seasonal parameters
    3. Validate all seasonal distributions
    4. Return list of uncertainty specifications (one per entity)
    """
    if not noise_models:
        return []
    
    grouped = group_noise_models_by_entity(noise_models)
    uncertainty_specs = []
    
    for (uncertainty_type, entity_id), models in sorted(grouped.items()):
        # Determine temporal model type from first model
        # Support both old format (noise_type) and new format (temporal_model.type)
        first_model = models[0]
        if "temporal_model" in first_model:
            temporal_type = first_model["temporal_model"]["type"]
        elif "noise_type" in first_model:
            # Old format: noise_type is at top level
            temporal_type = first_model["noise_type"]
        else:
            raise ValueError(f"Cannot determine temporal model type for {uncertainty_type} entity {entity_id}")
        
        try:
            if temporal_type == "independent":
                spec = convert_independent_model(models)
                
                # Validate seasonal distributions
                for dist in spec.get("seasonal_distributions", []):
                    validate_seasonal_distribution(dist)
                
                # Validate season_id sequence
                if "seasonal_distributions" in spec:
                    validate_season_ids(spec["seasonal_distributions"])
                    
            elif temporal_type == "periodic_ar":
                spec = convert_par_model(models)
            else:
                raise ValueError(f"Unsupported temporal model type: {temporal_type}")
            
            uncertainty_specs.append(spec)
        except Exception as e:
            print(f"❌ Error converting {uncertainty_type} entity {entity_id}: {e}")
            raise
    
    return uncertainty_specs

def collect_statistics(uncertainty_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collect statistics about distribution types in uncertainty specifications.
    
    Returns dictionary with counts by distribution type.
    """
    stats = {
        "normal": 0,
        "lognormal3": 0,
        "lognormal3_entities": []
    }
    
    for spec in uncertainty_specs:
        # Check Independent models with seasonal_distributions
        if "seasonal_distributions" in spec:
            for dist in spec["seasonal_distributions"]:
                dist_type = dist.get("type", "unknown")
                if dist_type == "normal":
                    stats["normal"] += 1
                elif dist_type == "lognormal3":
                    stats["lognormal3"] += 1
                    entity_info = (spec["uncertainty_type"], spec["entity_id"])
                    if entity_info not in stats["lognormal3_entities"]:
                        stats["lognormal3_entities"].append(entity_info)
        
        # Check PAR models with marginal_distribution
        if "marginal_distribution" in spec:
            dist = spec["marginal_distribution"]
            dist_type = dist.get("type", "unknown")
            if dist_type == "lognormal3":
                stats["lognormal3"] += 1
                entity_info = (spec["uncertainty_type"], spec["entity_id"])
                if entity_info not in stats["lognormal3_entities"]:
                    stats["lognormal3_entities"].append(entity_info)
    
    return stats

def migrate_recourse_file(filepath: Path, dry_run: bool = False, create_backup: bool = True) -> tuple[bool, Dict[str, Any]]:
    """
    Migrate a single recourse.json file to new format.
    
    Args:
        filepath: Path to recourse.json file
        dry_run: If True, don't modify files, just report what would change
        create_backup: If True, create .backup file before modifying
    
    Returns:
        Tuple of (success: bool, statistics: Dict)
    """
    empty_stats = {"normal": 0, "lognormal3": 0, "lognormal3_entities": []}
    
    try:
        # Read existing file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if already migrated
        if "uncertainty_specifications" in data and "noise_models" not in data:
            print(f"✓ {filepath}: Already migrated (has uncertainty_specifications, no noise_models)")
            # Collect stats from existing file
            stats = collect_statistics(data.get("uncertainty_specifications", []))
            return True, stats
        
        # Check if has noise_models
        if "noise_models" not in data:
            print(f"⚠️  {filepath}: No noise_models field found (empty file?)")
            return False, empty_stats
        
        # Convert noise_models to uncertainty_specifications
        noise_models = data["noise_models"]
        uncertainty_specs = convert_noise_models_to_uncertainty_specs(noise_models)
        
        # Collect statistics
        stats = collect_statistics(uncertainty_specs)
        
        if dry_run:
            print(f"🔍 {filepath}: Would convert {len(noise_models)} noise_models to {len(uncertainty_specs)} uncertainty_specifications")
            if stats["lognormal3"] > 0:
                print(f"    Found {stats['lognormal3']} LogNormal3 distributions (will be preserved)")
            return True, stats
        
        # Create backup if requested
        if create_backup:
            backup_path = filepath.with_suffix('.json.backup')
            shutil.copy2(filepath, backup_path)
            print(f"💾 Created backup: {backup_path}")
        
        # Update data structure
        data["uncertainty_specifications"] = uncertainty_specs
        del data["noise_models"]
        
        # Write updated file with nice formatting
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            f.write('\n')  # Trailing newline
        
        print(f"✅ {filepath}: Migrated {len(noise_models)} noise_models → {len(uncertainty_specs)} uncertainty_specifications")
        if stats["lognormal3"] > 0:
            print(f"    Preserved {stats['lognormal3']} LogNormal3 distributions")
        
        return True, stats
        
    except Exception as e:
        print(f"❌ {filepath}: Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False, empty_stats

def main():
    parser = argparse.ArgumentParser(
        description="Migrate recourse.json files from noise_models to uncertainty_specifications format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would change
  python3 scripts/migrate_examples_to_new_format.py --dry-run
  
  # Migrate all examples (creates backups)
  python3 scripts/migrate_examples_to_new_format.py
  
  # Migrate without backups (not recommended)
  python3 scripts/migrate_examples_to_new_format.py --no-backup
        """
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files (dangerous!)')
    
    args = parser.parse_args()
    
    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    examples_dir = repo_root / "examples"
    
    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("POWE.RS Format Migration: noise_models → uncertainty_specifications")
    print("=" * 80)
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print()
    
    if args.no_backup and not args.dry_run:
        print("⚠️  WARNING: Running without backups!")
        response = input("Are you sure? Type 'yes' to continue: ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)
        print()
    
    # Find all recourse.json files
    recourse_files = find_recourse_files(examples_dir)
    
    if not recourse_files:
        print(f"❌ No recourse.json files found in {examples_dir}")
        sys.exit(1)
    
    print(f"Found {len(recourse_files)} recourse.json files:")
    for f in recourse_files:
        print(f"  - {f.relative_to(repo_root)}")
    print()
    
    # Migrate each file
    success_count = 0
    failure_count = 0
    total_stats = {"normal": 0, "lognormal3": 0, "lognormal3_files": []}
    
    for filepath in sorted(recourse_files):
        print(f"\nProcessing: {filepath.relative_to(repo_root)}")
        print("-" * 80)
        
        success, stats = migrate_recourse_file(filepath, dry_run=args.dry_run, create_backup=not args.no_backup)
        
        if success:
            success_count += 1
            # Accumulate statistics
            total_stats["normal"] += stats["normal"]
            total_stats["lognormal3"] += stats["lognormal3"]
            if stats["lognormal3"] > 0:
                total_stats["lognormal3_files"].append({
                    "file": filepath.relative_to(repo_root),
                    "entities": stats["lognormal3_entities"],
                    "count": stats["lognormal3"]
                })
        else:
            failure_count += 1
    
    # Summary
    print()
    print("=" * 80)
    print("MIGRATION SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(recourse_files)}")
    print(f"✅ Successful: {success_count}")
    if failure_count > 0:
        print(f"❌ Failed: {failure_count}")
    
    # Distribution statistics
    print()
    print("Distribution Statistics:")
    print(f"  Normal: {total_stats['normal']} seasonal distributions")
    print(f"  LogNormal3: {total_stats['lognormal3']} seasonal distributions")
    
    # LogNormal3 details
    if total_stats["lognormal3_files"]:
        print()
        print("⚠️  LogNormal3 distributions found in:")
        for file_info in total_stats["lognormal3_files"]:
            print(f"  - {file_info['file']} ({file_info['count']} distributions)")
            for unc_type, entity_id in file_info["entities"]:
                print(f"      {unc_type} entity {entity_id}")
        print()
        print("Note: LogNormal3 distributions are preserved (not converted to Normal).")
        print("This maintains non-negativity guarantees and prevents solver infeasibilities.")
    
    if args.dry_run:
        print()
        print("This was a DRY RUN. No files were modified.")
        print("Run without --dry-run to perform actual migration.")
    
    sys.exit(0 if failure_count == 0 else 1)

if __name__ == "__main__":
    main()
