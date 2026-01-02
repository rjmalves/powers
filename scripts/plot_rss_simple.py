#!/usr/bin/env python3
"""
Simple RSS plotter - works with monitor_rss_simple.py output.
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


def load_data(csv_file: Path) -> Tuple[List[float], List[float], List[int]]:
    """Load RSS data from CSV."""
    timestamps = []
    rss_mb = []
    thread_counts = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            rss_mb.append(float(row['rss_mb']))
            thread_counts.append(int(row['thread_count']))
    
    return timestamps, rss_mb, thread_counts


def load_statistics(stats_file: Path) -> dict:
    """Load statistics from JSON."""
    if not stats_file.exists():
        return {}
    
    with open(stats_file, 'r') as f:
        return json.load(f)


def plot_rss(csv_file: Path, output_file: Path = None, title: str = None):
    """Create simple RSS timeline plot."""
    timestamps, rss_mb, thread_counts = load_data(csv_file)
    stats_file = csv_file.with_suffix('.stats.json')
    stats = load_statistics(stats_file)
    
    if not timestamps:
        print(f"No data in {csv_file}", file=sys.stderr)
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: RSS timeline
    ax1.plot(timestamps, rss_mb, 'b-', linewidth=1.5, label='RSS')
    ax1.fill_between(timestamps, 0, rss_mb, alpha=0.2)
    
    # Add statistics lines if available
    if stats:
        mean_mb = stats.get('rss_mean', 0) / 1024
        ax1.axhline(y=mean_mb, color='green', linestyle='--', 
                   linewidth=1, label=f'Mean: {mean_mb:.1f} MB', alpha=0.7)
        
        if stats.get('stable_rss_mean'):
            stable_mb = stats['stable_rss_mean'] / 1024
            warmup_time = timestamps[stats.get('warmup_samples', 0)] if stats.get('warmup_samples', 0) < len(timestamps) else timestamps[-1]
            ax1.axvline(x=warmup_time, color='orange', linestyle=':', 
                       linewidth=1, label=f'Warmup end', alpha=0.7)
            ax1.axhline(y=stable_mb, color='purple', linestyle='--',
                       linewidth=1, label=f'Stable mean: {stable_mb:.1f} MB', alpha=0.7)
    
    ax1.set_ylabel('RSS (MB)', fontsize=12, fontweight='bold')
    if title:
        ax1.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax1.set_title('RSS Timeline', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Add stats annotation
    if stats:
        stats_text = f"Final: {stats.get('rss_final', 0)/1024:.1f} MB\n"
        stats_text += f"Delta: {stats.get('rss_delta', 0)/1024:+.1f} MB\n"
        stats_text += f"Stable: {'Yes' if stats.get('is_stable') else 'No'}"
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Per-thread memory usage
    # Calculate RSS per thread (simple division)
    per_thread_mb = []
    for rss, threads in zip(rss_mb, thread_counts):
        if threads > 0:
            per_thread_mb.append(rss / threads)
        else:
            per_thread_mb.append(0)
    
    ax2.plot(timestamps, per_thread_mb, 'g-', linewidth=1.5, label='RSS per thread')
    ax2.fill_between(timestamps, 0, per_thread_mb, alpha=0.2, color='green')
    
    if stats and per_thread_mb:
        mean_per_thread = sum(per_thread_mb) / len(per_thread_mb)
        ax2.axhline(y=mean_per_thread, color='darkgreen', linestyle='--',
                   linewidth=1, label=f'Mean: {mean_per_thread:.1f} MB/thread', alpha=0.7)
        
        # Add thread count annotation
        avg_threads = stats.get('thread_count_mean', 0)
        ax2.text(0.98, 0.98, f'Thread count: {int(avg_threads)}',
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax2.set_ylabel('RSS per Thread (MB)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Thread Memory Usage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def compare_allocators(files: List[Tuple[str, Path]], output_file: Path = None):
    """Compare multiple RSS traces."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot RSS
    for i, (label, csv_file) in enumerate(files):
        timestamps, rss_mb, _ = load_data(csv_file)
        if timestamps:
            color = colors[i % len(colors)]
            ax1.plot(timestamps, rss_mb, '-', linewidth=2, label=label, color=color, alpha=0.7)
            ax1.fill_between(timestamps, 0, rss_mb, alpha=0.1, color=color)
    
    ax1.set_ylabel('RSS (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Allocator RSS Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Statistics comparison table
    stats_data = []
    for label, csv_file in files:
        stats_file = csv_file.with_suffix('.stats.json')
        stats = load_statistics(stats_file)
        if stats:
            stats_data.append((
                label,
                stats.get('rss_final', 0) / 1024,
                stats.get('rss_delta', 0) / 1024,
                'Yes' if stats.get('is_stable') else 'No',
                stats.get('rss_growth_rate_mb_per_sec', 0)
            ))
    
    if stats_data:
        # Create table
        ax2.axis('tight')
        ax2.axis('off')
        
        table_data = [['Allocator', 'Final RSS (MB)', 'Delta (MB)', 'Stable', 'Growth (MB/s)']]
        for row in stats_data:
            table_data.append([
                row[0],
                f'{row[1]:.1f}',
                f'{row[2]:+.1f}',
                row[3],
                f'{row[4]:+.3f}'
            ])
        
        table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.15, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Statistics Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot RSS data from monitor_rss_simple.py')
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
