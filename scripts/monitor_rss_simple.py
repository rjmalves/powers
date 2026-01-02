#!/usr/bin/env python3
"""
External RSS monitoring for HPC workloads - Simple, reliable, no log parsing.

Monitors RSS (Resident Set Size) of a process and all its threads, collecting
detailed statistics without depending on application logs.

Usage:
    ./monitor_rss_simple.py -o data.csv -- ./my_program args
    ./monitor_rss_simple.py -i 0.1 -o data.csv -- cargo run --release
    ./monitor_rss_simple.py --analyze data.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict
import statistics


@dataclass
class RssSample:
    """Single RSS measurement with thread breakdown."""
    timestamp: float  # Seconds since start
    rss_total_kb: int  # Total process RSS
    thread_count: int  # Number of threads
    threads: Dict[int, int]  # Per-thread RSS: {tid: rss_kb}


@dataclass
class RssStatistics:
    """Statistical summary of RSS data."""
    # Basic stats
    duration_seconds: float
    sample_count: int
    sample_interval: float
    
    # RSS statistics (KB)
    rss_min: int
    rss_max: int
    rss_mean: float
    rss_median: float
    rss_stdev: float
    rss_p95: float
    rss_p99: float
    
    # Growth analysis
    rss_initial: int
    rss_final: int
    rss_delta: int
    rss_growth_rate_mb_per_sec: float
    
    # Thread statistics
    thread_count_min: int
    thread_count_max: int
    thread_count_mean: float
    
    # Per-thread RSS statistics (KB)
    rss_per_thread_min: float
    rss_per_thread_max: float
    rss_per_thread_mean: float
    rss_per_thread_median: float
    
    # Stability analysis
    is_stable: bool  # RSS variation < 5% after warmup
    warmup_samples: int
    stable_rss_mean: Optional[float]
    stable_rss_stdev: Optional[float]


class SimpleRssMonitor:
    """Simple external RSS monitor - no log parsing."""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: List[RssSample] = []
        self.start_time: Optional[float] = None
        
    def get_process_rss(self, pid: int) -> Optional[int]:
        """Read total process RSS from /proc/{pid}/status."""
        try:
            with open(f'/proc/{pid}/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1])
        except (FileNotFoundError, PermissionError, ValueError):
            return None
        return None
    
    def get_thread_rss(self, pid: int) -> Dict[int, int]:
        """
        Get per-thread RSS breakdown.
        
        Returns:
            Dictionary mapping thread ID to RSS in KB
        """
        threads = {}
        try:
            task_dir = Path(f'/proc/{pid}/task')
            if not task_dir.exists():
                return threads
            
            for tid_dir in task_dir.iterdir():
                if not tid_dir.is_dir():
                    continue
                
                try:
                    tid = int(tid_dir.name)
                    status_file = tid_dir / 'status'
                    
                    with open(status_file, 'r') as f:
                        for line in f:
                            if line.startswith('VmRSS:'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    # Note: Linux reports same RSS for all threads
                                    # We track thread count for analysis
                                    threads[tid] = int(parts[1])
                                break
                except (ValueError, FileNotFoundError, PermissionError):
                    continue
        except (FileNotFoundError, PermissionError):
            pass
        
        return threads
    
    def sample_rss(self, pid: int) -> Optional[RssSample]:
        """Take a single RSS sample."""
        rss = self.get_process_rss(pid)
        if rss is None:
            return None
        
        threads = self.get_thread_rss(pid)
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        
        return RssSample(
            timestamp=elapsed,
            rss_total_kb=rss,
            thread_count=len(threads),
            threads=threads
        )
    
    def run(self, command: List[str], output_file: Optional[Path] = None) -> int:
        """Run command and monitor RSS."""
        print(f"Starting: {' '.join(command)}", file=sys.stderr)
        print(f"Sampling interval: {self.interval}s", file=sys.stderr)
        
        self.start_time = time.time()
        
        # Start process
        process = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        pid = process.pid
        print(f"Monitoring PID: {pid}", file=sys.stderr)
        
        last_sample_time = 0.0
        
        try:
            while process.poll() is None:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Sample at intervals
                if elapsed - last_sample_time >= self.interval:
                    sample = self.sample_rss(pid)
                    if sample:
                        self.samples.append(sample)
                        last_sample_time = elapsed
                
                # Small sleep to avoid busy-waiting
                time.sleep(min(0.01, self.interval / 10))
            
            # Final sample
            sample = self.sample_rss(pid)
            if sample:
                self.samples.append(sample)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user", file=sys.stderr)
            process.terminate()
            process.wait(timeout=5)
            return 130
        
        exit_code = process.wait()
        
        # Save and analyze
        if output_file:
            self.save_data(output_file)
            stats = self.compute_statistics()
            self.print_statistics(stats)
            self.save_statistics(output_file.with_suffix('.stats.json'), stats)
        
        print(f"\nProcess exited with code {exit_code}", file=sys.stderr)
        print(f"Collected {len(self.samples)} RSS samples", file=sys.stderr)
        
        return exit_code
    
    def save_data(self, output_file: Path):
        """Save RSS samples to CSV."""
        csv_file = output_file.with_suffix('.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'rss_kb', 'rss_mb', 'thread_count',
                'threads_detail'
            ])
            
            for sample in self.samples:
                threads_str = ';'.join(f'{tid}:{rss}' for tid, rss in sample.threads.items())
                writer.writerow([
                    f'{sample.timestamp:.3f}',
                    sample.rss_total_kb,
                    f'{sample.rss_total_kb / 1024:.1f}',
                    sample.thread_count,
                    threads_str
                ])
        
        print(f"Data saved to {csv_file}", file=sys.stderr)
    
    def compute_statistics(self, warmup_fraction: float = 0.1) -> RssStatistics:
        """Compute comprehensive RSS statistics."""
        if not self.samples:
            raise ValueError("No samples collected")
        
        # Extract RSS values
        rss_values = [s.rss_total_kb for s in self.samples]
        thread_counts = [s.thread_count for s in self.samples]
        
        # Basic statistics
        duration = self.samples[-1].timestamp if self.samples else 0.0
        
        rss_min = min(rss_values)
        rss_max = max(rss_values)
        rss_mean = statistics.mean(rss_values)
        rss_median = statistics.median(rss_values)
        rss_stdev = statistics.stdev(rss_values) if len(rss_values) > 1 else 0.0
        
        # Percentiles
        sorted_rss = sorted(rss_values)
        rss_p95 = sorted_rss[int(len(sorted_rss) * 0.95)]
        rss_p99 = sorted_rss[int(len(sorted_rss) * 0.99)]
        
        # Growth analysis
        rss_initial = rss_values[0]
        rss_final = rss_values[-1]
        rss_delta = rss_final - rss_initial
        rss_growth_rate = (rss_delta / 1024) / duration if duration > 0 else 0.0
        
        # Thread statistics
        thread_min = min(thread_counts)
        thread_max = max(thread_counts)
        thread_mean = statistics.mean(thread_counts)
        
        # Per-thread RSS statistics
        rss_per_thread = []
        for rss, threads in zip(rss_values, thread_counts):
            if threads > 0:
                rss_per_thread.append(rss / threads)
            else:
                rss_per_thread.append(0)
        
        rss_per_thread_min = min(rss_per_thread) if rss_per_thread else 0
        rss_per_thread_max = max(rss_per_thread) if rss_per_thread else 0
        rss_per_thread_mean = statistics.mean(rss_per_thread) if rss_per_thread else 0
        rss_per_thread_median = statistics.median(rss_per_thread) if rss_per_thread else 0
        
        # Stability analysis (skip warmup period)
        warmup_count = int(len(rss_values) * warmup_fraction)
        if warmup_count < 1:
            warmup_count = min(5, len(rss_values) // 2)
        
        stable_values = rss_values[warmup_count:]
        stable_mean = statistics.mean(stable_values) if stable_values else rss_mean
        stable_stdev = statistics.stdev(stable_values) if len(stable_values) > 1 else 0.0
        
        # Consider stable if coefficient of variation < 5% after warmup
        cv = (stable_stdev / stable_mean) if stable_mean > 0 else 1.0
        is_stable = cv < 0.05
        
        return RssStatistics(
            duration_seconds=duration,
            sample_count=len(self.samples),
            sample_interval=self.interval,
            rss_min=rss_min,
            rss_max=rss_max,
            rss_mean=rss_mean,
            rss_median=rss_median,
            rss_stdev=rss_stdev,
            rss_p95=rss_p95,
            rss_p99=rss_p99,
            rss_initial=rss_initial,
            rss_final=rss_final,
            rss_delta=rss_delta,
            rss_growth_rate_mb_per_sec=rss_growth_rate,
            thread_count_min=thread_min,
            thread_count_max=thread_max,
            thread_count_mean=thread_mean,
            rss_per_thread_min=rss_per_thread_min,
            rss_per_thread_max=rss_per_thread_max,
            rss_per_thread_mean=rss_per_thread_mean,
            rss_per_thread_median=rss_per_thread_median,
            is_stable=is_stable,
            warmup_samples=warmup_count,
            stable_rss_mean=stable_mean,
            stable_rss_stdev=stable_stdev
        )
    
    def print_statistics(self, stats: RssStatistics):
        """Print human-readable statistics."""
        print("\n" + "="*70, file=sys.stderr)
        print("RSS STATISTICS", file=sys.stderr)
        print("="*70, file=sys.stderr)
        
        print(f"\nDuration: {stats.duration_seconds:.1f}s", file=sys.stderr)
        print(f"Samples: {stats.sample_count} (interval: {stats.sample_interval}s)", file=sys.stderr)
        
        print(f"\n--- RSS Memory Usage ---", file=sys.stderr)
        print(f"Initial:  {stats.rss_initial/1024:>8.1f} MB", file=sys.stderr)
        print(f"Final:    {stats.rss_final/1024:>8.1f} MB", file=sys.stderr)
        print(f"Delta:    {stats.rss_delta/1024:>+8.1f} MB", file=sys.stderr)
        print(f"Min:      {stats.rss_min/1024:>8.1f} MB", file=sys.stderr)
        print(f"Max:      {stats.rss_max/1024:>8.1f} MB", file=sys.stderr)
        print(f"Mean:     {stats.rss_mean/1024:>8.1f} MB", file=sys.stderr)
        print(f"Median:   {stats.rss_median/1024:>8.1f} MB", file=sys.stderr)
        print(f"StdDev:   {stats.rss_stdev/1024:>8.1f} MB", file=sys.stderr)
        print(f"P95:      {stats.rss_p95/1024:>8.1f} MB", file=sys.stderr)
        print(f"P99:      {stats.rss_p99/1024:>8.1f} MB", file=sys.stderr)
        
        print(f"\n--- Growth Rate ---", file=sys.stderr)
        print(f"Rate: {stats.rss_growth_rate_mb_per_sec:+.3f} MB/s", file=sys.stderr)
        
        print(f"\n--- Thread Count ---", file=sys.stderr)
        print(f"Min:  {stats.thread_count_min}", file=sys.stderr)
        print(f"Max:  {stats.thread_count_max}", file=sys.stderr)
        print(f"Mean: {stats.thread_count_mean:.1f}", file=sys.stderr)
        
        print(f"\n--- Per-Thread RSS ---", file=sys.stderr)
        print(f"Min:    {stats.rss_per_thread_min/1024:>8.1f} MB/thread", file=sys.stderr)
        print(f"Max:    {stats.rss_per_thread_max/1024:>8.1f} MB/thread", file=sys.stderr)
        print(f"Mean:   {stats.rss_per_thread_mean/1024:>8.1f} MB/thread", file=sys.stderr)
        print(f"Median: {stats.rss_per_thread_median/1024:>8.1f} MB/thread", file=sys.stderr)
        
        print(f"\n--- Stability Analysis ---", file=sys.stderr)
        print(f"Warmup samples: {stats.warmup_samples}", file=sys.stderr)
        if stats.stable_rss_mean:
            print(f"Stable mean:    {stats.stable_rss_mean/1024:.1f} MB", file=sys.stderr)
            print(f"Stable stdev:   {stats.stable_rss_stdev/1024:.1f} MB", file=sys.stderr)
            cv = (stats.stable_rss_stdev / stats.stable_rss_mean * 100) if stats.stable_rss_mean > 0 else 0
            print(f"Coeff. of var:  {cv:.2f}%", file=sys.stderr)
        print(f"Stable: {'✓ YES' if stats.is_stable else '✗ NO'}", file=sys.stderr)
        
        print("="*70, file=sys.stderr)
    
    def save_statistics(self, output_file: Path, stats: RssStatistics):
        """Save statistics to JSON."""
        with open(output_file, 'w') as f:
            json.dump(asdict(stats), f, indent=2)
        print(f"Statistics saved to {output_file}", file=sys.stderr)
    
    @staticmethod
    def load_and_analyze(csv_file: Path) -> RssStatistics:
        """Load CSV and compute statistics."""
        samples = []
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse thread details
                threads = {}
                if row.get('threads_detail'):
                    for pair in row['threads_detail'].split(';'):
                        if ':' in pair:
                            tid_str, rss_str = pair.split(':', 1)
                            threads[int(tid_str)] = int(rss_str)
                
                sample = RssSample(
                    timestamp=float(row['timestamp']),
                    rss_total_kb=int(row['rss_kb']),
                    thread_count=int(row['thread_count']),
                    threads=threads
                )
                samples.append(sample)
        
        if not samples:
            raise ValueError(f"No data in {csv_file}")
        
        # Reconstruct monitor for statistics
        monitor = SimpleRssMonitor()
        monitor.samples = samples
        return monitor.compute_statistics()


def main():
    parser = argparse.ArgumentParser(
        description='Simple external RSS monitor - no log parsing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor and save data
  %(prog)s -o data.csv -- ./target/release/powers run examples/05-large-scale-brazilian
  
  # Custom sampling interval
  %(prog)s -i 0.1 -o data.csv -- cargo run --release
  
  # Analyze existing data
  %(prog)s --analyze data.csv
        """
    )
    
    parser.add_argument('-i', '--interval', type=float, default=0.5,
                       help='RSS sampling interval in seconds (default: 0.5)')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output file basename (creates .csv and .stats.json)')
    parser.add_argument('--analyze', type=Path,
                       help='Analyze existing CSV file')
    parser.add_argument('command', nargs='*',
                       help='Command to monitor')
    
    args = parser.parse_args()
    
    # Analyze mode
    if args.analyze:
        if not args.analyze.exists():
            print(f"Error: File not found: {args.analyze}", file=sys.stderr)
            return 1
        
        try:
            stats = SimpleRssMonitor.load_and_analyze(args.analyze)
            monitor = SimpleRssMonitor()
            monitor.print_statistics(stats)
            
            if args.output:
                monitor.save_statistics(args.output.with_suffix('.stats.json'), stats)
        except Exception as e:
            print(f"Error analyzing data: {e}", file=sys.stderr)
            return 1
        
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
    monitor = SimpleRssMonitor(interval=args.interval)
    exit_code = monitor.run(command, args.output)
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
