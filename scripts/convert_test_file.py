#!/usr/bin/env python3
"""
Convert test_par_scenario_integration.rs to v0.3.0 format.
"""

import re

def convert_noise_model_construction(content):
    """Convert NoiseModel construction from v0.2.0 to v0.3.0 format."""
    
    # Pattern to match NoiseModel with old structure
    pattern = r'''NoiseModel\s*\{\s*
        uncertainty_type:\s*([^,]+),\s*
        entity_id:\s*([^,]+),\s*
        season_id:\s*([^,]+),\s*
        distribution:\s*None,\s*//[^\n]*\n\s*
        marginal_distribution:\s*Some\(([^}]+\})\),\s*
        innovation_distribution:\s*None,\s*
        temporal_model:\s*([^}]*(?:\{[^}]*\})*[^}]*),\s*
        residual_distribution:\s*None,\s*
    \}'''
    
    def replace_noise_model(match):
        uncertainty_type = match.group(1).strip()
        entity_id = match.group(2).strip()
        season_id = match.group(3).strip()
        marginal_dist = match.group(4).strip()
        temporal_model = match.group(5).strip()
        
        return f'''NoiseModel {{
        uncertainty_type: {uncertainty_type},
        entity_id: {entity_id},
        season_id: {season_id},
        distribution: {marginal_dist},
        temporal_model: {temporal_model},
    }}'''
    
    # Apply the conversion
    content = re.sub(pattern, replace_noise_model, content, flags=re.VERBOSE | re.MULTILINE)
    
    # Remove migration calls
    content = re.sub(r'\s*//\s*Migrate[^\n]*\n\s*for nm in &mut [^}]*\{\s*nm\.migrate_distribution_fields\(\)[^}]*\}\s*\n', '', content, flags=re.MULTILINE)
    
    # Fix variable declarations (remove mut when no longer needed)
    content = re.sub(r'let mut noise_models = vec!\[', 'let noise_models = vec![', content)
    
    return content

def main():
    file_path = "tests/test_par_scenario_integration.rs"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Do the conversion manually for this complex case
    # Since regex is complex, let's do a simpler approach
    lines = content.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip migration blocks
        if 'Migrate from legacy fields' in line:
            # Skip the migration comment and loop
            while i < len(lines) and not line.strip().startswith('let recourse'):
                if 'let recourse' in lines[i]:
                    break
                i += 1
            continue
        
        # Convert NoiseModel construction
        if 'NoiseModel {' in line and 'distribution: None' in lines[i+3] if i+3 < len(lines) else False:
            # Found a NoiseModel to convert
            result_lines.append(line.replace('let mut noise_models', 'let noise_models'))
            result_lines.append(lines[i+1])  # uncertainty_type
            result_lines.append(lines[i+2])  # entity_id  
            result_lines.append(lines[i+3])  # season_id
            
            # Skip old distribution line and find marginal_distribution
            i += 4
            while i < len(lines) and 'marginal_distribution:' not in lines[i]:
                i += 1
            
            if i < len(lines):
                # Convert marginal_distribution to distribution
                marginal_line = lines[i].replace('marginal_distribution: Some(', 'distribution: ')
                # Remove the trailing ),
                marginal_line = marginal_line.replace('),', ',')
                result_lines.append('        ' + marginal_line.strip())
                
                # Skip innovation_distribution line
                i += 1
                while i < len(lines) and 'innovation_distribution:' not in lines[i]:
                    result_lines.append(lines[i])
                    i += 1
                i += 1  # Skip innovation_distribution line
                
                # Find and copy temporal_model
                while i < len(lines) and 'temporal_model:' not in lines[i]:
                    result_lines.append(lines[i])
                    i += 1
                
                # Copy temporal_model and everything until residual_distribution
                while i < len(lines) and 'residual_distribution:' not in lines[i]:
                    result_lines.append(lines[i])
                    i += 1
                
                # Skip residual_distribution line
                i += 1
                
                # Add closing brace
                result_lines.append('    },')
            continue
        
        result_lines.append(line)
        i += 1
    
    # Write back
    with open(file_path, 'w') as f:
        f.write('\n'.join(result_lines))
    
    print(f"Converted {file_path}")

if __name__ == "__main__":
    main()