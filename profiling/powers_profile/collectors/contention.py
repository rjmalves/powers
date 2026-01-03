"""Contention detection for parallel workloads.

This module collects and analyzes thread contention metrics using perf
to identify synchronization bottlenecks in parallel execution.
"""

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ProfilingConfig


@dataclass
class ContentionMetrics:
    """Thread contention metrics for a scaling run."""
    
    thread_count: int
    total_wait_time_ms: float
    wait_events: int
    futex_events: int
    lock_contention_events: int
    avg_wait_time_ms: float
    success: bool
    error_message: Optional[str] = None
    raw_perf_data: Optional[str] = None


class ContentionDetector:
    """Detects thread contention using perf events."""
    
    # Perf events related to contention
    CONTENTION_EVENTS = [
        "sched:sched_stat_blocked",
        "syscalls:sys_enter_futex",
    ]
    
    def __init__(self, config: ProfilingConfig):
        """Initialize contention detector.
        
        Args:
            config: Profiling configuration
        """
        self.config = config
        self._perf_available = self._check_perf_available()
    
    def _check_perf_available(self) -> bool:
        """Check if perf is available and has necessary permissions.
        
        Returns:
            True if perf is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["perf", "stat", "--", "echo", "test"],
                capture_output=True,
                timeout=5,
                check=False
            )
            # perf returns non-zero but should still produce output if available
            return result.returncode in [0, 1] and len(result.stderr) > 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def is_available(self) -> bool:
        """Check if contention detection is available.
        
        Returns:
            True if perf is available for contention detection
        """
        return self._perf_available
    
    def collect(
        self,
        binary: Path,
        args: List[str],
        thread_count: int,
        timeout_seconds: int = 600
    ) -> ContentionMetrics:
        """Collect contention metrics for a single run.
        
        Args:
            binary: Path to binary to profile
            args: Command-line arguments
            thread_count: Number of threads being used
            timeout_seconds: Timeout for the run
            
        Returns:
            ContentionMetrics with collected data
        """
        if not self._perf_available:
            return ContentionMetrics(
                thread_count=thread_count,
                total_wait_time_ms=0.0,
                wait_events=0,
                futex_events=0,
                lock_contention_events=0,
                avg_wait_time_ms=0.0,
                success=False,
                error_message="perf not available or insufficient permissions"
            )
        
        # Build perf command
        # Using perf stat with specific events
        perf_cmd = [
            "perf", "stat",
            "-e", "sched:sched_stat_blocked",
            "-e", "syscalls:sys_enter_futex",
            "--",
            str(binary)
        ] + args
        
        try:
            # Set environment with thread count
            import os
            env = os.environ.copy()
            env["RAYON_NUM_THREADS"] = str(thread_count)
            
            # Run perf
            result = subprocess.run(
                perf_cmd,
                cwd=self.config.repo_root,
                env=env,
                capture_output=True,
                timeout=timeout_seconds,
                check=False
            )
            
            # Parse perf output
            perf_output = result.stderr.decode('utf-8', errors='replace')
            
            metrics = self._parse_perf_output(perf_output, thread_count)
            metrics.raw_perf_data = perf_output
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return ContentionMetrics(
                thread_count=thread_count,
                total_wait_time_ms=0.0,
                wait_events=0,
                futex_events=0,
                lock_contention_events=0,
                avg_wait_time_ms=0.0,
                success=False,
                error_message="perf collection timed out"
            )
        except Exception as e:
            return ContentionMetrics(
                thread_count=thread_count,
                total_wait_time_ms=0.0,
                wait_events=0,
                futex_events=0,
                lock_contention_events=0,
                avg_wait_time_ms=0.0,
                success=False,
                error_message=f"perf collection failed: {str(e)}"
            )
    
    def _parse_perf_output(
        self,
        perf_output: str,
        thread_count: int
    ) -> ContentionMetrics:
        """Parse perf stat output for contention metrics.
        
        Args:
            perf_output: Raw perf output
            thread_count: Number of threads
            
        Returns:
            ContentionMetrics parsed from output
        """
        wait_events = 0
        futex_events = 0
        lock_contention_events = 0
        
        # Parse event counts from perf stat output
        # Format: "     12,345      event:name"
        for line in perf_output.split('\n'):
            line = line.strip()
            
            # Match perf stat lines
            # Examples:
            #   "         1,234      sched:sched_stat_blocked"
            #   "           123      syscalls:sys_enter_futex"
            match = re.match(r'^\s*([\d,]+)\s+(\S+)', line)
            if not match:
                continue
            
            count_str = match.group(1).replace(',', '')
            event_name = match.group(2)
            
            try:
                count = int(count_str)
            except ValueError:
                continue
            
            if 'sched_stat_blocked' in event_name:
                wait_events = count
            elif 'sys_enter_futex' in event_name:
                futex_events = count
        
        # Lock contention is approximated by futex calls
        lock_contention_events = futex_events
        
        # Estimate total wait time (very rough approximation)
        # Assume each wait event is ~0.01ms on average (highly variable)
        # This is a placeholder for more sophisticated analysis
        avg_wait_time_per_event_ms = 0.01
        total_wait_time_ms = wait_events * avg_wait_time_per_event_ms
        
        avg_wait_time_ms = (
            total_wait_time_ms / wait_events if wait_events > 0 else 0.0
        )
        
        success = True
        error_message = None
        
        # Check if perf had errors
        if "not supported" in perf_output.lower():
            success = False
            error_message = "Some perf events not supported on this system"
        elif "permission denied" in perf_output.lower():
            success = False
            error_message = "Permission denied for perf events"
        
        return ContentionMetrics(
            thread_count=thread_count,
            total_wait_time_ms=total_wait_time_ms,
            wait_events=wait_events,
            futex_events=futex_events,
            lock_contention_events=lock_contention_events,
            avg_wait_time_ms=avg_wait_time_ms,
            success=success,
            error_message=error_message
        )
    
    def to_dict(self, metrics: ContentionMetrics) -> Dict:
        """Convert contention metrics to dictionary.
        
        Args:
            metrics: ContentionMetrics to convert
            
        Returns:
            Dictionary for JSON serialization
        """
        return {
            "thread_count": metrics.thread_count,
            "total_wait_time_ms": metrics.total_wait_time_ms,
            "wait_events": metrics.wait_events,
            "futex_events": metrics.futex_events,
            "lock_contention_events": metrics.lock_contention_events,
            "avg_wait_time_ms": metrics.avg_wait_time_ms,
            "success": metrics.success,
            "error_message": metrics.error_message
        }


def analyze_contention_trends(
    contention_data: List[ContentionMetrics]
) -> Dict[str, any]:
    """Analyze contention trends across thread counts.
    
    Args:
        contention_data: List of contention metrics for different thread counts
        
    Returns:
        Dictionary with trend analysis
    """
    if not contention_data:
        return {"has_data": False}
    
    successful_data = [m for m in contention_data if m.success]
    if not successful_data:
        return {"has_data": False}
    
    # Sort by thread count
    successful_data.sort(key=lambda m: m.thread_count)
    
    # Analyze trends
    analysis = {
        "has_data": True,
        "thread_counts": [m.thread_count for m in successful_data],
        "total_wait_times_ms": [m.total_wait_time_ms for m in successful_data],
        "wait_events": [m.wait_events for m in successful_data],
        "futex_events": [m.futex_events for m in successful_data],
    }
    
    # Check for increasing contention with thread count
    if len(successful_data) >= 2:
        first = successful_data[0]
        last = successful_data[-1]
        
        # Calculate contention growth rate
        if first.wait_events > 0:
            wait_event_growth = (last.wait_events - first.wait_events) / first.wait_events
            analysis["wait_event_growth_rate"] = wait_event_growth
            
            # Flag high contention growth
            if wait_event_growth > 2.0:  # More than 2x growth
                analysis["high_contention_growth"] = True
                analysis["contention_warning"] = (
                    f"Wait events increased {wait_event_growth * 100:.1f}% "
                    f"from {first.thread_count} to {last.thread_count} threads"
                )
        
        if first.futex_events > 0:
            futex_growth = (last.futex_events - first.futex_events) / first.futex_events
            analysis["futex_growth_rate"] = futex_growth
    
    # Identify peak contention
    peak_contention = max(successful_data, key=lambda m: m.wait_events)
    analysis["peak_contention_thread_count"] = peak_contention.thread_count
    analysis["peak_wait_events"] = peak_contention.wait_events
    
    return analysis
