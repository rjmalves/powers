#!/usr/bin/env python3
"""
Simple RSS plotter using matplotlib.

Can be used independently of monitor_rss.py to plot collected data.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def load_data(csv_file: Path) -> Tuple[List[float], List[float], List[dict]]:
    """Load RSS data and iteration events."""
    timestamps = []
    rss_mb = []
    
    # Load CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            rss_mb.append(float(row['rss_mb']))
    
    # Load iterations JSON
    iterations = []
    json_file = csv_file.with_suffix('.iterations.json')
    if json_file.exists():
        with open(json_file, 'r') as f:
            iterations = json.load(f)
    
    return timestamps, rss_mb, iterations


def plot_rss(csv_file: Path, output_file: Path = None, title: str = None):
    """Create RSS timeline plot."""
    timestamps, rss_mb, iterations = load_data(csv_file)
    
    if not timestamps:
        print(f"No data in {csv_file}", file=sys.stderr)
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: RSS timeline
    ax1.plot(timestamps, rss_mb, 'b-', linewidth=1.5, label='RSS')
    ax1.fill_between(timestamps, 0, rss_mb, alpha=0.2)
    
    # Mark iterations
    iter_starts = [e for e in iterations if e['phase'] == 'start']
    iter_ends = [e for e in iterations if e['phase'] == 'end']
    
    for event in iter_starts:
        rss = event.get('rss_kb', 0) / 1024 if event.get('rss_kb') else None
        if rss:
            ax1.plot(event['timestamp'], rss, '^', color='green', markersize=10)
            ax1.annotate(f"{event['iteration']}", 
                        xy=(event['timestamp'], rss),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, color='darkgreen', fontweight='bold')
    
    for event in iter_ends:
        rss = event.get('rss_kb', 0) / 1024 if event.get('rss_kb') else None
        if rss:
            ax1.plot(event['timestamp'], rss, 'v', color='red', markersize=10)
    
    ax1.set_ylabel('RSS (MB)', fontsize=12, fontweight='bold')
    if title:
        ax1.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax1.set_title('RSS Timeline with Iteration Boundaries', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Legend
    start_marker = mpatches.Patch(color='green', label='Iteration start')
    end_marker = mpatches.Patch(color='red', label='Iteration end')
    ax1.legend(handles=[start_marker, end_marker], loc='upper left')
    
    # Plot 2: Per-iteration delta
    if iter_starts and iter_ends:
        iter_nums = []
        deltas = []
        
        for start_ev in iter_starts:
            end_evs = [e for e in iter_ends if e['iteration'] == start_ev['iteration']]
            if end_evs:
                end_ev = end_evs[0]
                start_rss = start_ev.get('rss_kb', 0) / 1024
                end_rss = end_ev.get('rss_kb', 0) / 1024
                if start_rss and end_rss:
                    iter_nums.append(start_ev['iteration'])
                    deltas.append(end_rss - start_rss)
        
        if iter_nums:
            colors = ['green' if abs(d) < 5 else 'orange' if abs(d) < 10 else 'red' for d in deltas]
            ax2.bar(iter_nums, deltas, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            for num, delta in zip(iter_nums, deltas):
                ax2.text(num, delta, f'{delta:+.1f}', 
                        ha='center', va='bottom' if delta >= 0 else 'top',
                        fontsize=8)
    
    ax2.set_ylabel('RSS Delta (MB)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Iteration Memory Delta', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def compare_allocators(files: List[Tuple[str, Path]], output_file: Path = None):
    """Compare multiple RSS traces."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (label, csv_file) in enumerate(files):
        timestamps, rss_mb, _ = load_data(csv_file)
        if timestamps:
            color = colors[i % len(colors)]
            ax.plot(timestamps, rss_mb, '-', linewidth=2, label=label, color=color, alpha=0.7)
            ax.fill_between(timestamps, 0, rss_mb, alpha=0.1, color=color)
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RSS (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Allocator RSS Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot RSS data collected by monitor_rss.py')
    parser.add_argument('input', type=Path, help='Input CSV file')
    parser.add_argument('-o', '--output', type=Path, help='Output plot file (PNG/PDF)')
    parser.add_argument('-t', '--title', help='Plot title')
    parser.add_argument('--compare', nargs='+', help='Additional files to compare (format: label:file.csv)')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1
    
    if args.compare:
        # Comparison mode
        files = [(args.input.stem, args.input)]
        for item in args.compare:
            if ':' in item:
                label, path = item.split(':', 1)
                files.append((label, Path(path)))
            else:
                p = Path(item)
                files.append((p.stem, p))
        
        compare_allocators(files, args.output)
    else:
        # Single plot mode
        plot_rss(args.input, args.output, args.title)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
