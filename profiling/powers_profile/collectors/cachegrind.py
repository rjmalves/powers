"""Cachegrind cache profiler collector using valgrind.

Cachegrind simulates CPU cache hierarchy and branch prediction to analyze
cache miss rates and instruction counts. This collector wraps valgrind
--tool=cachegrind and parses the output.
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

CACHEGRIND_OUT = "cachegrind.out"
CACHEGRIND_SUMMARY = "cachegrind_summary.json"


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


def _parse_cachegrind_out(cachegrind_path: Path) -> Dict[str, Any]:
    """
    Parse Cachegrind output file and extract cache metrics.
    
    Cachegrind output format:
    ------------------------
    events: Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw
    fl=...
    fn=...
    ...
    summary: 1234567890 12345 123 ...
    
    Where:
    - Ir: Instruction reads (executed instructions)
    - I1mr: I1 cache read misses
    - ILmr: Last-level I cache read misses
    - Dr: Data reads
    - D1mr: D1 cache read misses
    - DLmr: Last-level D cache read misses
    - Dw: Data writes
    - D1mw: D1 cache write misses
    - DLmw: Last-level D cache write misses
    """
    with open(cachegrind_path, 'r') as f:
        content = f.read()
    
    summary: Dict[str, Any] = {}
    
    # Extract events definition
    events_match = re.search(r'^events:\s*(.+)$', content, re.MULTILINE)
    if events_match:
        events = events_match.group(1).split()
        summary['events'] = events
    else:
        # Default event order
        events = ['Ir', 'I1mr', 'ILmr', 'Dr', 'D1mr', 'DLmr', 'Dw', 'D1mw', 'DLmw']
        summary['events'] = events
    
    # Extract summary line
    summary_match = re.search(r'^summary:\s*(.+)$', content, re.MULTILINE)
    if summary_match:
        values_str = summary_match.group(1).split()
        values = [int(v) for v in values_str if v.isdigit()]
        
        # Map event names to values
        event_data = {}
        for i, event_name in enumerate(events):
            if i < len(values):
                event_data[event_name] = values[i]
        
        # Standard cachegrind events
        ir = event_data.get('Ir', 0)  # Instructions
        i1mr = event_data.get('I1mr', 0)  # L1 instruction cache misses
        ilmr = event_data.get('ILmr', 0)  # Last-level instruction cache misses
        dr = event_data.get('Dr', 0)  # Data reads
        d1mr = event_data.get('D1mr', 0)  # L1 data cache read misses
        dlmr = event_data.get('DLmr', 0)  # Last-level data cache read misses
        dw = event_data.get('Dw', 0)  # Data writes
        d1mw = event_data.get('D1mw', 0)  # L1 data cache write misses
        dlmw = event_data.get('DLmw', 0)  # Last-level data cache write misses
        
        summary['instructions'] = ir
        summary['i1_misses'] = i1mr
        summary['ill_misses'] = ilmr
        summary['data_reads'] = dr
        summary['d1_read_misses'] = d1mr
        summary['dll_read_misses'] = dlmr
        summary['data_writes'] = dw
        summary['d1_write_misses'] = d1mw
        summary['dll_write_misses'] = dlmw
        
        # Calculate miss rates
        if ir > 0:
            summary['i1_miss_rate'] = (i1mr / ir) * 100
            summary['ill_miss_rate'] = (ilmr / ir) * 100
        else:
            summary['i1_miss_rate'] = 0.0
            summary['ill_miss_rate'] = 0.0
        
        if dr > 0:
            summary['d1_read_miss_rate'] = (d1mr / dr) * 100
            summary['dll_read_miss_rate'] = (dlmr / dr) * 100
        else:
            summary['d1_read_miss_rate'] = 0.0
            summary['dll_read_miss_rate'] = 0.0
        
        if dw > 0:
            summary['d1_write_miss_rate'] = (d1mw / dw) * 100
            summary['dll_write_miss_rate'] = (dlmw / dw) * 100
        else:
            summary['d1_write_miss_rate'] = 0.0
            summary['dll_write_miss_rate'] = 0.0
        
        # Overall cache metrics
        total_refs = dr + dw
        total_d1_misses = d1mr + d1mw
        total_dll_misses = dlmr + dlmw
        
        if total_refs > 0:
            summary['overall_d1_miss_rate'] = (total_d1_misses / total_refs) * 100
            summary['overall_dll_miss_rate'] = (total_dll_misses / total_refs) * 100
        else:
            summary['overall_d1_miss_rate'] = 0.0
            summary['overall_dll_miss_rate'] = 0.0
        
        summary['total_refs'] = total_refs
        summary['total_d1_misses'] = total_d1_misses
        summary['total_dll_misses'] = total_dll_misses
    
    return summary


class CachegrindCollector(Collector):
    """Collector that wraps valgrind Cachegrind for cache profiling."""
    
    name = "cachegrind"
    
    def collect(
        self,
        *,
        binary: Path,
        args: List[str],
        config: ProfilingConfig,
        run_dir: Path,
    ) -> CollectorResult:
        """
        Run Cachegrind and parse cache metrics.
        
        Args:
            binary: Target binary to profile
            args: Arguments for the binary
            config: Profiling configuration
            run_dir: Directory to store results
            
        Returns:
            CollectorResult with Cachegrind metrics
        """
        start = time.perf_counter()
        collector_dir = run_dir / self.name
        collector_dir.mkdir(parents=True, exist_ok=True)
        
        errors: List[str] = []
        warnings: List[str] = []
        data: Dict[str, Any] = {}
        raw_files: List[str] = []
        
        # Check if Cachegrind is enabled
        if not config.cachegrind_enabled:
            warnings.append("Cachegrind disabled in config; skipping")
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
        
        # Prepare Cachegrind output path
        cachegrind_out_path = collector_dir / CACHEGRIND_OUT
        
        # Run valgrind with Cachegrind
        cachegrind_cmd = [
            str(valgrind_path),
            "--tool=cachegrind",
            f"--cachegrind-out-file={cachegrind_out_path}",
            "--cache-sim=yes",
            "--branch-sim=yes",
            str(binary),
            *args,
        ]
        
        data['command'] = ' '.join(cachegrind_cmd)
        
        rc, out, err = _run_command(cachegrind_cmd, cwd=config.repo_root, timeout=600)
        
        data['exit_code'] = rc
        data['stderr'] = err[:1000] if err else ""  # Truncate stderr
        raw_files.append(str(cachegrind_out_path))
        
        if rc != 0:
            errors.append(f"valgrind Cachegrind failed with exit code {rc}")
        
        # Parse Cachegrind output if successful
        if rc == 0 and cachegrind_out_path.exists():
            try:
                summary = _parse_cachegrind_out(cachegrind_out_path)
                data.update(summary)
                
                # Save summary to separate JSON
                summary_path = collector_dir / CACHEGRIND_SUMMARY
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                raw_files.append(str(summary_path))
                
                # Check for high miss rates
                if summary.get('overall_d1_miss_rate', 0) > 10.0:
                    warnings.append(
                        f"High L1 data cache miss rate: {summary['overall_d1_miss_rate']:.2f}%"
                    )
                
                if summary.get('overall_dll_miss_rate', 0) > 1.0:
                    warnings.append(
                        f"High last-level cache miss rate: {summary['overall_dll_miss_rate']:.2f}%"
                    )
                
            except (ValueError, KeyError) as e:
                errors.append(f"Failed to parse Cachegrind output: {e}")
        elif not cachegrind_out_path.exists():
            errors.append(f"Cachegrind output file not created: {cachegrind_out_path}")
        
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
