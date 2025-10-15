#!/usr/bin/env python3
"""
Fix test files to use v0.3.0 NoiseModel format.
"""

import re
from pathlib import Path

def fix_noise_model_in_rust_file(file_path):
    """Fix NoiseModel usage in a Rust test file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match NoiseModel construction with old fields
    pattern = r'(NoiseModel\s*\{[^}]*?)distribution:\s*None,\s*//[^\n]*\n\s*marginal_distribution:\s*Some\(([^)]+\([^}]*\})\),[^}]*?innovation_distribution:\s*None,[^}]*?temporal_model:\s*([^}]*\}),[^}]*?residual_distribution:\s*None,([^}]*?\})'
    
    def replace_noise_model(match):
        pre = match.group(1)
        marginal_dist = match.group(2)
        temporal_model = match.group(3)
        post = match.group(4)
        
        # Reconstruct with new format
        return f"{pre}distribution: {marginal_dist},\n        temporal_model: {temporal_model},{post}"
    
    content = re.sub(pattern, replace_noise_model, content, flags=re.DOTALL)
    
    # Remove migration calls
    content = re.sub(r'\s*//\s*Migrate[^\n]*\n\s*for nm in &mut [^}]*\{\s*nm\.migrate_distribution_fields\(\)[^}]*\}\s*\n', '', content, flags=re.DOTALL)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed {file_path}")
        return True
    return False

def main():
    test_files = [
        "tests/test_par_scenario_integration.rs",
        "tests/test_scenario_validation.rs"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            try:
                if fix_noise_model_in_rust_file(file_path):
                    print(f"Successfully updated {file_path}")
                else:
                    print(f"No changes needed for {file_path}")
            except Exception as e:
                print(f"Error fixing {file_path}: {e}")

if __name__ == "__main__":
    main()