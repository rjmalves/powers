"""Massif heap profiler collector using valgrind.

Massif tracks heap memory usage over time, providing snapshots of memory
consumption and peak usage. This collector wraps valgrind --tool=massif
and parses the output file.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import ProfilingConfig
from ..schemas import CollectorResult
from .base import Collector

MASSIF_OUT = "massif.out"
MASSIF_SUMMARY = "massif_summary.json"


def _which_tool(candidate: Optional[Path], fallback: str) -> Optional[Path]:
    """Find tool binary by path or fallback to system PATH."""
    if candidate:
        if candidate.exists():
            return candidate
        return None
    resolved = shutil.which(fallback)
    return Path(resolved) if resolved else None


def _run_command(
    command: List[str],
    cwd: Path,
    timeout: int = 600,
) -> Tuple[int, str, str]:
    """Execute command and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _parse_massif_out(massif_path: Path) -> Dict[str, Any]:
    """
    Parse Massif output file and extract key metrics.
    
    Massif output format:
    ------------------------
    Command: ./program
    Massif arguments: ...
    ms_print arguments: ...
    
    n1: 1024 time=123 [details]
    ...
    
    Snapshot format:
    #-----------
    snapshot=N
    #-----------
    time=T
    mem_heap_B=BYTES
    mem_heap_extra_B=EXTRA
    mem_stacks_B=STACKS
    heap_tree=...
    """
    with open(massif_path, 'r') as f:
        content = f.read()
    
    summary: Dict[str, Any] = {}
    snapshots: List[Dict[str, Any]] = []
    
    # Extract command line
    cmd_match = re.search(r'^cmd:\s*(.+)$', content, re.MULTILINE)
    if cmd_match:
        summary['command'] = cmd_match.group(1)
    
    # Extract time unit
    time_unit_match = re.search(r'^time_unit:\s*(\w+)$', content, re.MULTILINE)
    if time_unit_match:
        summary['time_unit'] = time_unit_match.group(1)
    else:
        summary['time_unit'] = 'i'  # Default: instructions
    
    # Parse snapshots
    snapshot_pattern = re.compile(
        r'#-+\s*\n'
        r'snapshot=(\d+)\s*\n'
        r'#-+\s*\n'
        r'time=(\d+)\s*\n'
        r'mem_heap_B=(\d+)\s*\n'
        r'mem_heap_extra_B=(\d+)\s*\n'
        r'mem_stacks_B=(\d+)',
        re.MULTILINE
    )
    
    for match in snapshot_pattern.finditer(content):
        snapshot_num = int(match.group(1))
        time_val = int(match.group(2))
        mem_heap = int(match.group(3))
        mem_heap_extra = int(match.group(4))
        mem_stacks = int(match.group(5))
        
        total_mem = mem_heap + mem_heap_extra + mem_stacks
        
        snapshots.append({
            'snapshot': snapshot_num,
            'time': time_val,
            'mem_heap_bytes': mem_heap,
            'mem_heap_extra_bytes': mem_heap_extra,
            'mem_stacks_bytes': mem_stacks,
            'total_bytes': total_mem,
            'total_mb': total_mem / (1024 * 1024),
        })
    
    # Find peak memory snapshot
    if snapshots:
        peak_snapshot = max(snapshots, key=lambda s: s['total_bytes'])
        summary['peak_snapshot'] = peak_snapshot['snapshot']
        summary['peak_time'] = peak_snapshot['time']
        summary['peak_bytes'] = peak_snapshot['total_bytes']
        summary['peak_mb'] = peak_snapshot['total_mb']
        summary['peak_heap_bytes'] = peak_snapshot['mem_heap_bytes']
        summary['peak_heap_mb'] = peak_snapshot['mem_heap_bytes'] / (1024 * 1024)
        
        # Final snapshot
        final_snapshot = snapshots[-1]
        summary['final_bytes'] = final_snapshot['total_bytes']
        summary['final_mb'] = final_snapshot['total_mb']
        summary['final_time'] = final_snapshot['time']
        
        # Memory growth
        if len(snapshots) >= 2:
            first_snapshot = snapshots[0]
            growth_bytes = final_snapshot['total_bytes'] - first_snapshot['total_bytes']
            summary['growth_bytes'] = growth_bytes
            summary['growth_mb'] = growth_bytes / (1024 * 1024)
    else:
        summary['peak_bytes'] = 0
        summary['peak_mb'] = 0.0
    
    summary['snapshot_count'] = len(snapshots)
    summary['snapshots'] = snapshots[:10]  # Include first 10 snapshots for detail
    
    return summary


class MassifCollector(Collector):
    """Collector that wraps valgrind Massif for heap profiling over time."""
    
    name = "massif"
    
    def collect(
        self,
        *,
        binary: Path,
        args: List[str],
        config: ProfilingConfig,
        run_dir: Path,
    ) -> CollectorResult:
        """
        Run Massif and parse heap usage over time.
        
        Args:
            binary: Target binary to profile
            args: Arguments for the binary
            config: Profiling configuration
            run_dir: Directory to store results
            
        Returns:
            CollectorResult with Massif metrics
        """
        start = time.perf_counter()
        collector_dir = run_dir / self.name
        collector_dir.mkdir(parents=True, exist_ok=True)
        
        errors: List[str] = []
        warnings: List[str] = []
        data: Dict[str, Any] = {}
        raw_files: List[str] = []
        
        # Check if Massif is enabled
        if not config.massif_enabled:
            warnings.append("Massif disabled in config; skipping")
            duration = time.perf_counter() - start
            return CollectorResult(
                collector_name=self.name,
                success=True,
                duration_seconds=duration,
                data={'skipped': True},
                warnings=warnings,
            )
        
        # Find valgrind binary
        valgrind_path = _which_tool(config.valgrind_path, "valgrind")
        if valgrind_path is None:
            errors.append(
                "valgrind not found. Install with: apt install valgrind"
            )
            duration = time.perf_counter() - start
            return CollectorResult(
                collector_name=self.name,
                success=False,
                duration_seconds=duration,
                data=data,
                errors=errors,
            )
        
        # Check valgrind version
        version_cmd = [str(valgrind_path), "--version"]
        rc_ver, out_ver, _ = _run_command(version_cmd, cwd=collector_dir)
        if rc_ver == 0:
            data['valgrind_version'] = out_ver.strip()
        
        # Prepare Massif output path
        massif_out_path = collector_dir / MASSIF_OUT
        
        # Run valgrind with Massif
        massif_cmd = [
            str(valgrind_path),
            "--tool=massif",
            f"--massif-out-file={massif_out_path}",
            f"--time-unit={config.massif_time_unit}",
            "--detailed-freq=1",  # More frequent detailed snapshots
            str(binary),
            *args,
        ]
        
        data['command'] = ' '.join(massif_cmd)
        data['time_unit'] = config.massif_time_unit
        
        rc, out, err = _run_command(massif_cmd, cwd=config.repo_root, timeout=600)
        
        data['exit_code'] = rc
        data['stderr'] = err[:1000] if err else ""  # Truncate stderr
        raw_files.append(str(massif_out_path))
        
        if rc != 0:
            errors.append(f"valgrind Massif failed with exit code {rc}")
        
        # Parse Massif output if successful
        if rc == 0 and massif_out_path.exists():
            try:
                summary = _parse_massif_out(massif_out_path)
                data.update(summary)
                
                # Save summary to separate JSON
                summary_path = collector_dir / MASSIF_SUMMARY
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                raw_files.append(str(summary_path))
                
                # Check for memory growth
                if summary.get('growth_mb', 0) > 10.0:
                    warnings.append(
                        f"Significant memory growth detected: {summary['growth_mb']:.1f} MB"
                    )
                
            except (ValueError, KeyError) as e:
                errors.append(f"Failed to parse Massif output: {e}")
        elif not massif_out_path.exists():
            errors.append(f"Massif output file not created: {massif_out_path}")
        
        duration = time.perf_counter() - start
        success = rc == 0 and not errors
        
        return CollectorResult(
            collector_name=self.name,
            success=success,
            duration_seconds=duration,
            data=data,
            errors=errors,
            warnings=warnings,
            raw_files=raw_files,
        )
