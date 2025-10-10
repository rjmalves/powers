#!/usr/bin/env python3
"""
Extract and analyze memory profiling data from benchmark output
"""

import re
import sys
from collections import defaultdict

def parse_memory_profile(text):
    """Parse memory profile output and extract key metrics"""
    profiles = []
    
    # Pattern to match memory profile blocks
    profile_pattern = r"Memory Profile: (.+?)\n.*?Initial RSS:\s+(.+?)\n.*?Peak RSS:\s+(.+?)\n.*?Final RSS:\s+(.+?)\n.*?Delta RSS:\s+([+-])\s+\((.+?)\)"
    
    matches = re.findall(profile_pattern, text, re.DOTALL)
    
    for match in matches:
        label, initial, peak, final, sign, delta = match
        profiles.append({
            'label': label.strip(),
            'initial': initial.strip(),
            'peak': peak.strip(),
            'final': final.strip(),
            'delta': f"{sign}{delta.strip()}"
        })
    
    return profiles

def format_report(profiles):
    """Format memory profiles into a readable report"""
    if not profiles:
        return "No memory profiles found"
    
    # Group by problem type
    by_type = defaultdict(list)
    for p in profiles:
        # Extract key from label (e.g., "2-stage", "12-stage", etc.)
        if "2-stage" in p['label']:
            key = "2-stage"
        elif "12-stage" in p['label']:
            key = "12-stage"
        elif "24-stage" in p['label']:
            key = "24-stage"
        elif "5-stage" in p['label']:
            key = "5-stage"
        else:
            key = "other"
        by_type[key].append(p)
    
    report = []
    report.append("=" * 80)
    report.append("MEMORY PROFILING SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    for problem_type in sorted(by_type.keys()):
        report.append(f"\n{problem_type.upper()} PROBLEMS")
        report.append("-" * 80)
        
        for profile in by_type[problem_type][:3]:  # Show first 3 of each type
            report.append(f"  {profile['label']}")
            report.append(f"    Initial: {profile['initial']:>12}  Peak: {profile['peak']:>12}")
            report.append(f"    Final:   {profile['final']:>12}  Delta: {profile['delta']:>12}")
            report.append("")
    
    return "\n".join(report)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    
    profiles = parse_memory_profile(text)
    print(format_report(profiles))
    
    # Print CSV for further analysis
    print("\n" + "=" * 80)
    print("CSV FORMAT (for spreadsheet analysis)")
    print("=" * 80)
    print("Label,Initial (MB),Peak (MB),Final (MB),Delta (MB)")
    for p in profiles[:20]:  # First 20 profiles
        # Extract numeric values (rough approximation)
        initial_val = p['initial'].replace(' MB', '').replace(' KB', '')
        peak_val = p['peak'].replace(' MB', '').replace(' KB', '')
        final_val = p['final'].replace(' MB', '').replace(' KB', '')
        delta_val = p['delta'].replace(' MB', '').replace(' KB', '').replace('+', '').replace('(', '').replace(')', '')
        print(f'"{p["label"]}",{initial_val},{peak_val},{final_val},{delta_val}')
