#!/usr/bin/env python3
"""
External RSS monitoring script for HPC workloads.

Tracks RSS (Resident Set Size) of a running process, parses stdout logs
for iteration boundaries, and generates comprehensive plots.

Usage:
    ./monitor_rss.py -- ./target/release/powers run examples/05-large-scale-brazilian
    ./monitor_rss.py --interval 0.1 --output rss_data.csv -- cargo run --release
    ./monitor_rss.py --plot-only rss_data.csv
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)


@dataclass
class RssSample:
    """Single RSS measurement sample."""
    timestamp: float  # Seconds since start
    rss_kb: int  # RSS in kilobytes
    wall_time: float  # Wall clock time (epoch)


@dataclass
class IterationEvent:
    """Iteration boundary event from program logs."""
    iteration: int
    phase: str  # 'start' or 'end'
    timestamp: float  # Seconds since program start
    rss_kb: Optional[int] = None  # RSS if reported in logs


class RssMonitor:
    """External RSS monitor for a subprocess."""
    
    def __init__(self, interval: float = 0.5):
        """
        Initialize RSS monitor.
        
        Args:
            interval: Sampling interval in seconds (default: 0.5s)
        """
        self.interval = interval
        self.samples: List[RssSample] = []
        self.iterations: List[IterationEvent] = []
        self.start_time: Optional[float] = None
        self.process: Optional[subprocess.Popen] = None
        
    def get_rss_kb(self, pid: int) -> Optional[int]:
        """
        Read RSS from /proc/{pid}/status (Linux only).
        
        Args:
            pid: Process ID to monitor
            
        Returns:
            RSS in kilobytes, or None if unavailable
        """
        try:
            with open(f'/proc/{pid}/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        # VmRSS:	  123456 kB
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1])
        except (FileNotFoundError, PermissionError, ValueError):
            return None
        return None
    
    def parse_log_line(self, line: str, program_start: float) -> Optional[IterationEvent]:
        """
        Parse iteration events from program stdout.
        
        Recognizes patterns like:
        - [DEBUG] RSS at iteration 1 start: VmRSS:	  123456 kB
        - [INFO] Iteration 5 complete
        - Starting iteration 3
        
        Args:
            line: Log line to parse
            program_start: Program start time (epoch)
            
        Returns:
            IterationEvent if recognized, None otherwise
        """
        current_time = time.time()
        timestamp = current_time - program_start
        
        # Pattern 1: RSS at iteration N start/end
        match = re.search(r'RSS at iteration (\d+) (start|end)', line)
        if match:
            iteration = int(match.group(1))
            phase = match.group(2)
            
            # Try to extract RSS value
            rss_kb = None
            rss_match = re.search(r'VmRSS:\s+(\d+)\s+kB', line)
            if rss_match:
                rss_kb = int(rss_match.group(1))
            
            return IterationEvent(iteration, phase, timestamp, rss_kb)
        
        # Pattern 2: Iteration N complete/finished
        match = re.search(r'Iteration (\d+) (?:complete|finished)', line, re.IGNORECASE)
        if match:
            iteration = int(match.group(1))
            return IterationEvent(iteration, 'end', timestamp)
        
        # Pattern 3: Starting iteration N
        match = re.search(r'Starting iteration (\d+)', line, re.IGNORECASE)
        if match:
            iteration = int(match.group(1))
            return IterationEvent(iteration, 'start', timestamp)
        
        return None
    
    def run(self, command: List[str], output_file: Optional[Path] = None) -> int:
        """
        Run command and monitor RSS.
        
        Args:
            command: Command and arguments to run
            output_file: Optional CSV file to save samples
            
        Returns:
            Exit code of the monitored process
        """
        print(f"Starting monitoring: {' '.join(command)}", file=sys.stderr)
        print(f"Sampling interval: {self.interval}s", file=sys.stderr)
        
        self.start_time = time.time()
        program_start = self.start_time
        
        # Start process
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        pid = self.process.pid
        print(f"Monitoring PID: {pid}", file=sys.stderr)
        
        # Monitoring loop
        last_sample_time = 0.0
        
        try:
            while self.process.poll() is None:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Sample RSS at intervals
                if elapsed - last_sample_time >= self.interval:
                    rss_kb = self.get_rss_kb(pid)
                    if rss_kb is not None:
                        sample = RssSample(elapsed, rss_kb, current_time)
                        self.samples.append(sample)
                        last_sample_time = elapsed
                
                # Read and parse stdout (non-blocking)
                try:
                    line = self.process.stdout.readline()
                    if line:
                        # Echo to our stdout
                        print(line, end='')
                        
                        # Parse for iteration events
                        event = self.parse_log_line(line, program_start)
                        if event:
                            self.iterations.append(event)
                except:
                    pass
                
                # Small sleep to avoid busy-waiting
                time.sleep(0.01)
            
            # Read any remaining output
            for line in self.process.stdout:
                print(line, end='')
                event = self.parse_log_line(line, program_start)
                if event:
                    self.iterations.append(event)
            
            # Final RSS sample
            rss_kb = self.get_rss_kb(pid)
            if rss_kb is not None:
                elapsed = time.time() - self.start_time
                sample = RssSample(elapsed, rss_kb, time.time())
                self.samples.append(sample)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user", file=sys.stderr)
            self.process.terminate()
            self.process.wait(timeout=5)
            return 130
        
        exit_code = self.process.wait()
        
        # Save data if requested
        if output_file:
            self.save_data(output_file)
            print(f"\nData saved to {output_file}", file=sys.stderr)
        
        print(f"\nProcess exited with code {exit_code}", file=sys.stderr)
        print(f"Collected {len(self.samples)} RSS samples", file=sys.stderr)
        print(f"Detected {len(self.iterations)} iteration events", file=sys.stderr)
        
        return exit_code
    
    def save_data(self, output_file: Path):
        """Save RSS samples and iteration events to CSV and JSON."""
        # Save RSS samples to CSV
        csv_file = output_file.with_suffix('.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'rss_kb', 'rss_mb', 'wall_time'])
            writer.writeheader()
            for sample in self.samples:
                writer.writerow({
                    'timestamp': f'{sample.timestamp:.3f}',
                    'rss_kb': sample.rss_kb,
                    'rss_mb': f'{sample.rss_kb / 1024:.1f}',
                    'wall_time': sample.wall_time
                })
        
        # Save iteration events to JSON
        json_file = output_file.with_suffix('.iterations.json')
        with open(json_file, 'w') as f:
            json.dump([asdict(event) for event in self.iterations], f, indent=2)
    
    @staticmethod
    def load_data(input_file: Path) -> Tuple[List[RssSample], List[IterationEvent]]:
        """Load RSS samples and iteration events from saved files."""
        samples = []
        iterations = []
        
        # Load RSS samples
        csv_file = input_file.with_suffix('.csv')
        if csv_file.exists():
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(RssSample(
                        timestamp=float(row['timestamp']),
                        rss_kb=int(row['rss_kb']),
                        wall_time=float(row['wall_time'])
                    ))
        
        # Load iteration events
        json_file = input_file.with_suffix('.iterations.json')
        if json_file.exists():
            with open(json_file, 'r') as f:
                events_data = json.load(f)
                for event_dict in events_data:
                    iterations.append(IterationEvent(**event_dict))
        
        return samples, iterations


def plot_rss_timeline(
    samples: List[RssSample],
    iterations: List[IterationEvent],
    output_file: Optional[Path] = None
):
    """
    Create comprehensive RSS timeline plot with iteration markers.
    
    Args:
        samples: RSS samples to plot
        iterations: Iteration events to mark
        output_file: Optional file to save plot
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed", file=sys.stderr)
        return
    
    if not samples:
        print("No samples to plot", file=sys.stderr)
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Extract data
    timestamps = [s.timestamp for s in samples]
    rss_mb = [s.rss_kb / 1024 for s in samples]
    
    # Plot 1: RSS over time
    ax1.plot(timestamps, rss_mb, 'b-', linewidth=1, label='RSS')
    ax1.fill_between(timestamps, 0, rss_mb, alpha=0.3)
    
    # Mark iteration boundaries
    iteration_colors = {'start': 'green', 'end': 'red'}
    iteration_markers = {'start': '^', 'end': 'v'}
    
    for event in iterations:
        color = iteration_colors.get(event.phase, 'gray')
        marker = iteration_markers.get(event.phase, 'o')
        
        # Find nearest RSS sample
        nearest_rss = None
        for sample in samples:
            if abs(sample.timestamp - event.timestamp) < 1.0:
                nearest_rss = sample.rss_kb / 1024
                break
        
        if nearest_rss is None and event.rss_kb:
            nearest_rss = event.rss_kb / 1024
        
        if nearest_rss:
            ax1.plot(event.timestamp, nearest_rss, marker, color=color, 
                    markersize=8, label=f'Iter {event.phase}' if event.iteration == 1 else '')
            
            # Add iteration number label for start events
            if event.phase == 'start':
                ax1.annotate(f'{event.iteration}', 
                           xy=(event.timestamp, nearest_rss),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='darkgreen')
    
    ax1.set_ylabel('RSS (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Memory Usage Timeline with Iteration Boundaries', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot 2: RSS delta between iterations
    if iterations:
        iter_starts = [e for e in iterations if e.phase == 'start']
        iter_ends = [e for e in iterations if e.phase == 'end']
        
        if iter_starts and iter_ends:
            # Calculate per-iteration RSS
            iter_numbers = []
            iter_rss_start = []
            iter_rss_end = []
            iter_delta = []
            
            for start_event in iter_starts:
                # Find corresponding end
                end_events = [e for e in iter_ends if e.iteration == start_event.iteration]
                if not end_events:
                    continue
                
                end_event = end_events[0]
                
                # Find RSS values
                start_rss = None
                end_rss = None
                
                for sample in samples:
                    if abs(sample.timestamp - start_event.timestamp) < 0.5:
                        start_rss = sample.rss_kb / 1024
                    if abs(sample.timestamp - end_event.timestamp) < 0.5:
                        end_rss = sample.rss_kb / 1024
                
                if start_event.rss_kb and not start_rss:
                    start_rss = start_event.rss_kb / 1024
                if end_event.rss_kb and not end_rss:
                    end_rss = end_event.rss_kb / 1024
                
                if start_rss and end_rss:
                    iter_numbers.append(start_event.iteration)
                    iter_rss_start.append(start_rss)
                    iter_rss_end.append(end_rss)
                    iter_delta.append(end_rss - start_rss)
            
            if iter_numbers:
                # Bar plot of deltas
                colors = ['green' if d < 5 else 'orange' if d < 10 else 'red' for d in iter_delta]
                ax2.bar(iter_numbers, iter_delta, color=colors, alpha=0.7, label='RSS Delta')
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.set_ylabel('RSS Delta (MB)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
                ax2.set_title('Per-Iteration Memory Delta', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, (iter_num, delta) in enumerate(zip(iter_numbers, iter_delta)):
                    ax2.text(iter_num, delta, f'{delta:+.1f}', 
                           ha='center', va='bottom' if delta >= 0 else 'top',
                           fontsize=8)
    
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}", file=sys.stderr)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Monitor RSS of a subprocess and plot memory usage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor and save data
  %(prog)s -o data.csv -- ./target/release/powers run examples/05-large-scale-brazilian
  
  # Monitor with custom interval
  %(prog)s -i 0.1 -o data.csv -- cargo run --release
  
  # Plot existing data
  %(prog)s --plot-only data.csv -p memory_plot.png
  
  # Monitor and plot immediately
  %(prog)s -o data.csv -p plot.png -- ./my_program
        """
    )
    
    parser.add_argument('-i', '--interval', type=float, default=0.5,
                       help='RSS sampling interval in seconds (default: 0.5)')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output file for RSS data (CSV)')
    parser.add_argument('-p', '--plot', type=Path,
                       help='Output file for plot (PNG/PDF)')
    parser.add_argument('--plot-only', type=Path,
                       help='Plot data from existing CSV file (skip monitoring)')
    parser.add_argument('command', nargs='*',
                       help='Command to monitor (after --)')
    
    args = parser.parse_args()
    
    # Plot-only mode
    if args.plot_only:
        if not args.plot_only.exists():
            print(f"Error: File not found: {args.plot_only}", file=sys.stderr)
            return 1
        
        samples, iterations = RssMonitor.load_data(args.plot_only)
        plot_rss_timeline(samples, iterations, args.plot)
        return 0
    
    # Monitor mode
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle '--' separator
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        command = sys.argv[idx + 1:]
    else:
        command = args.command
    
    if not command:
        print("Error: No command specified", file=sys.stderr)
        return 1
    
    # Run monitoring
    monitor = RssMonitor(interval=args.interval)
    exit_code = monitor.run(command, args.output)
    
    # Plot if requested
    if args.plot:
        plot_rss_timeline(monitor.samples, monitor.iterations, args.plot)
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
