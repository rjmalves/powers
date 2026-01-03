"""DHAT (Dynamic Heap Analysis Tool) collector using valgrind.

DHAT profiles heap allocations, providing detailed metrics about allocation
sizes, counts, and hotspots. This collector wraps valgrind --tool=dhat and
parses the JSON output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import ProfilingConfig
from ..schemas import CollectorResult
from .base import Collector

DHAT_OUT = "dhat.out.json"
DHAT_SUMMARY = "dhat_summary.json"


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


def _parse_dhat_json(dhat_path: Path) -> Dict[str, Any]:
    """
    Parse DHAT JSON output and extract key metrics.
    
    DHAT JSON format (valgrind 3.18+):
    {
      "dhatFileVersion": 2,
      "mode": "heap",
      "verb": "Allocated",
      "bklt": true,
      "bkacc": true,
      "tu": "instrs",
      "Mib": false,
      "tot_blocks": 12345,
      "tot_bytes": 67890,
      "max_blocks": 500,
      "max_bytes": 12000,
      "aps": [...],  // Allocation points
      "ftbl": [...]  // Frame table
    }
    """
    with open(dhat_path, 'r') as f:
        data = json.load(f)
    
    summary = {
        'total_blocks': data.get('tot_blocks', 0),
        'total_bytes': data.get('tot_bytes', 0),
        'max_blocks': data.get('max_blocks', 0),
        'max_bytes': data.get('max_bytes', 0),
        'time_unit': data.get('tu', 'instrs'),
        'mode': data.get('mode', 'heap'),
    }
    
    # Extract top allocation points
    allocation_points = data.get('aps', [])
    hotspots = []
    
    for ap in allocation_points[:20]:  # Top 20 hotspots
        hotspot = {
            'total_bytes': ap.get('tb', 0),
            'total_blocks': ap.get('tbk', 0),
            'max_bytes': ap.get('mb', 0),
            'max_blocks': ap.get('mbk', 0),
            'at_tgmax_bytes': ap.get('gb', 0),
            'at_tgmax_blocks': ap.get('gbk', 0),
            'allocation_count': ap.get('ac', 0),
            'total_lifetimes': ap.get('tl', 0),
        }
        
        # Extract stack frame info if available
        if 'fs' in ap:
            frame_indices = ap['fs']
            frame_table = data.get('ftbl', [])
            hotspot['stack_trace'] = [
                frame_table[idx] if idx < len(frame_table) else f"<frame {idx}>"
                for idx in frame_indices[:5]  # Top 5 frames
            ]
        
        hotspots.append(hotspot)
    
    summary['hotspots'] = hotspots
    summary['hotspot_count'] = len(allocation_points)
    
    return summary


class DhatCollector(Collector):
    """Collector that wraps valgrind DHAT for heap profiling."""
    
    name = "dhat"
    
    def collect(
        self,
        *,
        binary: Path,
        args: List[str],
        config: ProfilingConfig,
        run_dir: Path,
    ) -> CollectorResult:
        """
        Run DHAT and parse heap allocation metrics.
        
        Args:
            binary: Target binary to profile
            args: Arguments for the binary
            config: Profiling configuration
            run_dir: Directory to store results
            
        Returns:
            CollectorResult with DHAT metrics
        """
        start = time.perf_counter()
        collector_dir = run_dir / self.name
        collector_dir.mkdir(parents=True, exist_ok=True)
        
        errors: List[str] = []
        warnings: List[str] = []
        data: Dict[str, Any] = {}
        raw_files: List[str] = []
        
        # Check if DHAT is enabled
        if not config.dhat_enabled:
            warnings.append("DHAT disabled in config; skipping")
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
        
        # Prepare DHAT output paths
        dhat_out_path = collector_dir / DHAT_OUT
        
        # Run valgrind with DHAT
        dhat_cmd = [
            str(valgrind_path),
            "--tool=dhat",
            f"--dhat-out-file={dhat_out_path}",
            str(binary),
            *args,
        ]
        
        data['command'] = ' '.join(dhat_cmd)
        
        rc, out, err = _run_command(dhat_cmd, cwd=config.repo_root, timeout=600)
        
        data['exit_code'] = rc
        data['stderr'] = err[:1000] if err else ""  # Truncate stderr
        raw_files.append(str(dhat_out_path))
        
        if rc != 0:
            errors.append(f"valgrind DHAT failed with exit code {rc}")
        
        # Parse DHAT output if successful
        if rc == 0 and dhat_out_path.exists():
            try:
                summary = _parse_dhat_json(dhat_out_path)
                data.update(summary)
                
                # Save summary to separate JSON
                summary_path = collector_dir / DHAT_SUMMARY
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                raw_files.append(str(summary_path))
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                errors.append(f"Failed to parse DHAT output: {e}")
        elif not dhat_out_path.exists():
            errors.append(f"DHAT output file not created: {dhat_out_path}")
        
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
