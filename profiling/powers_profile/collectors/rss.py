"""RSS (Resident Set Size) monitor collector.

Tracks physical memory usage of a process over time by polling /proc/{pid}/status.
Provides lightweight memory monitoring without the overhead of valgrind tools.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ProfilingConfig
from ..schemas import CollectorResult
from .base import Collector

RSS_DATA_JSON = "rss_data.json"


@dataclass
class RssSample:
    """Single RSS measurement sample."""
    
    timestamp: float  # Seconds since process start
    rss_kb: int  # RSS in kilobytes
    wall_time: float  # Wall clock epoch time


class RssMonitor:
    """Lightweight RSS monitor using /proc filesystem."""
    
    def __init__(self, interval_seconds: float = 0.5):
        """
        Initialize RSS monitor.
        
        Args:
            interval_seconds: Sampling interval (default 0.5s)
        """
        self.interval = interval_seconds
        self.samples: List[RssSample] = []
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
    
    def _get_rss_kb(self, pid: int) -> Optional[int]:
        """
        Read RSS from /proc/{pid}/status.
        
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
    
    def _monitor_loop(self, pid: int, start_time: float):
        """Background monitoring loop."""
        last_sample_time = 0.0
        
        while not self._stop_event.is_set():
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Sample at intervals
            if elapsed - last_sample_time >= self.interval:
                rss_kb = self._get_rss_kb(pid)
                if rss_kb is not None:
                    sample = RssSample(
                        timestamp=elapsed,
                        rss_kb=rss_kb,
                        wall_time=current_time
                    )
                    self.samples.append(sample)
                    last_sample_time = elapsed
            
            # Small sleep to avoid busy-waiting
            time.sleep(min(0.01, self.interval / 10))
    
    def start_monitoring(self, pid: int):
        """Start background RSS monitoring thread."""
        start_time = time.time()
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(pid, start_time),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics from collected samples."""
        if not self.samples:
            return {
                'sample_count': 0,
                'min_rss_kb': 0,
                'max_rss_kb': 0,
                'mean_rss_kb': 0,
                'final_rss_kb': 0,
                'peak_rss_mb': 0.0,
                'mean_rss_mb': 0.0,
            }
        
        rss_values = [s.rss_kb for s in self.samples]
        min_rss = min(rss_values)
        max_rss = max(rss_values)
        mean_rss = sum(rss_values) // len(rss_values)
        final_rss = rss_values[-1]
        
        # Detect growth trend (simple heuristic)
        growth_detected = False
        if len(self.samples) >= 10:
            first_half_mean = sum(rss_values[:len(rss_values)//2]) / (len(rss_values)//2)
            second_half_mean = sum(rss_values[len(rss_values)//2:]) / (len(rss_values) - len(rss_values)//2)
            growth_mb = (second_half_mean - first_half_mean) / 1024
            growth_detected = growth_mb > 5.0  # >5MB growth
        
        return {
            'sample_count': len(self.samples),
            'min_rss_kb': min_rss,
            'max_rss_kb': max_rss,
            'mean_rss_kb': mean_rss,
            'final_rss_kb': final_rss,
            'peak_rss_mb': max_rss / 1024,
            'mean_rss_mb': mean_rss / 1024,
            'min_rss_mb': min_rss / 1024,
            'final_rss_mb': final_rss / 1024,
            'growth_detected': growth_detected,
            'duration_seconds': self.samples[-1].timestamp if self.samples else 0,
        }
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Export samples and summary to JSON-serializable dict."""
        return {
            'samples': [asdict(s) for s in self.samples],
            'summary': self.compute_summary(),
        }


class RssCollector(Collector):
    """Collector that monitors RSS during execution."""
    
    name = "rss"
    
    def collect(
        self,
        *,
        binary: Path,
        args: List[str],
        config: ProfilingConfig,
        run_dir: Path,
    ) -> CollectorResult:
        """
        Monitor RSS while running the binary.
        
        Args:
            binary: Target binary to profile
            args: Arguments for the binary
            config: Profiling configuration
            run_dir: Directory to store results
            
        Returns:
            CollectorResult with RSS metrics
        """
        start = time.perf_counter()
        collector_dir = run_dir / self.name
        collector_dir.mkdir(parents=True, exist_ok=True)
        
        errors: List[str] = []
        warnings: List[str] = []
        data: Dict[str, Any] = {}
        raw_files: List[str] = []
        
        # Check /proc availability
        if not Path('/proc').exists():
            errors.append("/proc filesystem not available; RSS monitoring requires Linux")
            duration = time.perf_counter() - start
            return CollectorResult(
                collector_name=self.name,
                success=False,
                duration_seconds=duration,
                data=data,
                errors=errors,
            )
        
        # Validate interval
        interval_seconds = config.rss_interval_ms / 1000.0
        if interval_seconds <= 0:
            errors.append(f"Invalid RSS interval: {config.rss_interval_ms}ms")
            interval_seconds = 0.5
            warnings.append("Using default interval: 500ms")
        
        # Initialize monitor
        monitor = RssMonitor(interval_seconds=interval_seconds)
        
        # Build command
        command = [str(binary), *args]
        data['command'] = ' '.join(command)
        data['interval_ms'] = config.rss_interval_ms
        
        # Start target process
        try:
            process = subprocess.Popen(
                command,
                cwd=config.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            errors.append(f"Failed to start process: {e}")
            duration = time.perf_counter() - start
            return CollectorResult(
                collector_name=self.name,
                success=False,
                duration_seconds=duration,
                data=data,
                errors=errors,
            )
        
        pid = process.pid
        data['pid'] = pid
        
        # Start monitoring in background
        monitor.start_monitoring(pid)
        
        # Wait for process to complete
        try:
            stdout, _ = process.communicate(timeout=600)
            exit_code = process.returncode
            data['exit_code'] = exit_code
            data['stdout_lines'] = len(stdout.splitlines()) if stdout else 0
            
            if exit_code != 0:
                warnings.append(f"Process exited with code {exit_code}")
        
        except subprocess.TimeoutExpired:
            errors.append("Process timeout after 600 seconds")
            process.kill()
            process.wait()
            exit_code = -1
        
        except KeyboardInterrupt:
            warnings.append("Interrupted by user")
            process.terminate()
            process.wait(timeout=5)
            exit_code = 130
        
        finally:
            # Stop monitoring
            monitor.stop_monitoring()
        
        # Compute summary statistics
        summary = monitor.compute_summary()
        data.update(summary)
        
        # Save full data to JSON
        rss_data_path = collector_dir / RSS_DATA_JSON
        with open(rss_data_path, 'w') as f:
            json.dump(monitor.to_json_dict(), f, indent=2)
        raw_files.append(str(rss_data_path))
        
        # Check for warnings
        if summary['sample_count'] < 5:
            warnings.append(f"Only {summary['sample_count']} samples collected; process may have been too short")
        
        if summary.get('growth_detected', False):
            warnings.append(f"Memory growth detected: {summary['peak_rss_mb']:.1f} MB peak")
        
        duration = time.perf_counter() - start
        success = exit_code == 0 and not errors
        
        return CollectorResult(
            collector_name=self.name,
            success=success,
            duration_seconds=duration,
            data=data,
            errors=errors,
            warnings=warnings,
            raw_files=raw_files,
        )
