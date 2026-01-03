"""Unified memory collector that orchestrates all memory profiling tools.

This collector runs DHAT, Massif, Cachegrind, and RSS monitoring, then
aggregates the results into a unified memory_data.json output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ..config import ProfilingConfig
from ..schemas import CollectorResult
from .base import Collector
from .cachegrind import CachegrindCollector
from .dhat import DhatCollector
from .massif import MassifCollector
from .rss import RssCollector

MEMORY_DATA_JSON = "memory_data.json"


class MemoryCollector(Collector):
    """
    Unified memory collector that runs all memory profiling tools.
    
    This collector orchestrates:
    - DHAT: Heap allocation profiling
    - Massif: Heap usage over time
    - Cachegrind: Cache efficiency analysis (optional)
    - RSS: Physical memory monitoring
    
    It aggregates all results into a single JSON output for downstream
    analysis and visualization.
    """
    
    name = "memory"
    
    def __init__(self):
        """Initialize sub-collectors."""
        self.dhat = DhatCollector()
        self.massif = MassifCollector()
        self.cachegrind = CachegrindCollector()
        self.rss = RssCollector()
    
    def collect(
        self,
        *,
        binary: Path,
        args: List[str],
        config: ProfilingConfig,
        run_dir: Path,
    ) -> CollectorResult:
        """
        Run all memory profiling tools and aggregate results.
        
        Args:
            binary: Target binary to profile
            args: Arguments for the binary
            config: Profiling configuration
            run_dir: Directory to store results
            
        Returns:
            CollectorResult with aggregated memory metrics
        """
        start = time.perf_counter()
        collector_dir = run_dir / self.name
        collector_dir.mkdir(parents=True, exist_ok=True)
        
        errors: List[str] = []
        warnings: List[str] = []
        data: Dict[str, Any] = {}
        raw_files: List[str] = []
        
        # Run sub-collectors
        sub_results: Dict[str, CollectorResult] = {}
        
        # 1. DHAT - Heap allocation profiling
        if config.dhat_enabled:
            try:
                dhat_result = self.dhat.collect(
                    binary=binary,
                    args=args,
                    config=config,
                    run_dir=run_dir,
                )
                sub_results['dhat'] = dhat_result
                
                if not dhat_result.success:
                    errors.extend([f"DHAT: {e}" for e in dhat_result.errors])
                warnings.extend([f"DHAT: {w}" for w in dhat_result.warnings])
                raw_files.extend(dhat_result.raw_files)
                
            except Exception as e:
                errors.append(f"DHAT collector failed: {e}")
        
        # 2. Massif - Heap usage over time
        if config.massif_enabled:
            try:
                massif_result = self.massif.collect(
                    binary=binary,
                    args=args,
                    config=config,
                    run_dir=run_dir,
                )
                sub_results['massif'] = massif_result
                
                if not massif_result.success:
                    errors.extend([f"Massif: {e}" for e in massif_result.errors])
                warnings.extend([f"Massif: {w}" for w in massif_result.warnings])
                raw_files.extend(massif_result.raw_files)
                
            except Exception as e:
                errors.append(f"Massif collector failed: {e}")
        
        # 3. Cachegrind - Cache efficiency (optional)
        if config.cachegrind_enabled:
            try:
                cachegrind_result = self.cachegrind.collect(
                    binary=binary,
                    args=args,
                    config=config,
                    run_dir=run_dir,
                )
                sub_results['cachegrind'] = cachegrind_result
                
                if not cachegrind_result.success:
                    errors.extend([f"Cachegrind: {e}" for e in cachegrind_result.errors])
                warnings.extend([f"Cachegrind: {w}" for w in cachegrind_result.warnings])
                raw_files.extend(cachegrind_result.raw_files)
                
            except Exception as e:
                errors.append(f"Cachegrind collector failed: {e}")
        
        # 4. RSS - Physical memory monitoring (always run if available)
        try:
            rss_result = self.rss.collect(
                binary=binary,
                args=args,
                config=config,
                run_dir=run_dir,
            )
            sub_results['rss'] = rss_result
            
            if not rss_result.success:
                # RSS failure is not critical; just warn
                warnings.extend([f"RSS: {e}" for e in rss_result.errors])
            warnings.extend([f"RSS: {w}" for w in rss_result.warnings])
            raw_files.extend(rss_result.raw_files)
            
        except Exception as e:
            warnings.append(f"RSS collector failed: {e}")
        
        # Aggregate results
        data['tools_run'] = list(sub_results.keys())
        data['tools_successful'] = [
            name for name, result in sub_results.items() if result.success
        ]
        
        # Extract key metrics from each tool
        aggregated_metrics: Dict[str, Any] = {}
        
        if 'dhat' in sub_results and sub_results['dhat'].success:
            dhat_data = sub_results['dhat'].data
            aggregated_metrics['dhat'] = {
                'total_blocks': dhat_data.get('total_blocks', 0),
                'total_bytes': dhat_data.get('total_bytes', 0),
                'max_bytes': dhat_data.get('max_bytes', 0),
                'hotspot_count': dhat_data.get('hotspot_count', 0),
                'top_hotspots': dhat_data.get('hotspots', [])[:5],  # Top 5
            }
        
        if 'massif' in sub_results and sub_results['massif'].success:
            massif_data = sub_results['massif'].data
            aggregated_metrics['massif'] = {
                'peak_mb': massif_data.get('peak_mb', 0),
                'peak_heap_mb': massif_data.get('peak_heap_mb', 0),
                'final_mb': massif_data.get('final_mb', 0),
                'growth_mb': massif_data.get('growth_mb', 0),
                'snapshot_count': massif_data.get('snapshot_count', 0),
            }
        
        if 'cachegrind' in sub_results and sub_results['cachegrind'].success:
            cg_data = sub_results['cachegrind'].data
            aggregated_metrics['cachegrind'] = {
                'instructions': cg_data.get('instructions', 0),
                'i1_miss_rate': cg_data.get('i1_miss_rate', 0),
                'overall_d1_miss_rate': cg_data.get('overall_d1_miss_rate', 0),
                'overall_dll_miss_rate': cg_data.get('overall_dll_miss_rate', 0),
            }
        
        if 'rss' in sub_results and sub_results['rss'].success:
            rss_data = sub_results['rss'].data
            aggregated_metrics['rss'] = {
                'peak_mb': rss_data.get('peak_rss_mb', 0),
                'mean_mb': rss_data.get('mean_rss_mb', 0),
                'final_mb': rss_data.get('final_rss_mb', 0),
                'sample_count': rss_data.get('sample_count', 0),
                'growth_detected': rss_data.get('growth_detected', False),
            }
        
        data['metrics'] = aggregated_metrics
        
        # Save aggregated data
        memory_data_path = collector_dir / MEMORY_DATA_JSON
        aggregated_output = {
            'timestamp': time.time(),
            'binary': str(binary),
            'args': args,
            'tools': data['tools_run'],
            'successful_tools': data['tools_successful'],
            'metrics': aggregated_metrics,
            'warnings': warnings,
            'errors': errors,
        }
        
        with open(memory_data_path, 'w') as f:
            json.dump(aggregated_output, f, indent=2)
        raw_files.append(str(memory_data_path))
        
        # Determine overall success
        # Success if at least one tool succeeded
        success = len(data['tools_successful']) > 0
        
        # Add summary to data
        data['summary'] = {
            'total_tools': len(sub_results),
            'successful_tools': len(data['tools_successful']),
            'failed_tools': len(sub_results) - len(data['tools_successful']),
        }
        
        duration = time.perf_counter() - start
        
        return CollectorResult(
            collector_name=self.name,
            success=success,
            duration_seconds=duration,
            data=data,
            errors=errors,
            warnings=warnings,
            raw_files=raw_files,
        )
