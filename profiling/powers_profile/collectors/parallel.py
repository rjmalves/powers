"""Parallel scaling collector for comprehensive parallelism analysis.

This collector orchestrates scaling tests, speedup/efficiency calculation,
Amdahl estimation, and contention detection.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ProfilingConfig
from ..schemas import ProfilingRun
from .scaling_runner import ScalingTestRunner, ScalingTestConfig, ScalingRunResult
from .contention import ContentionDetector, ContentionMetrics
from ..analyzers.scaling import (
    compute_speedup_efficiency,
    estimate_amdahl_serial_fraction,
    detect_scaling_bottlenecks,
    SpeedupMetrics,
    AmdahlEstimate
)


class ParallelCollector:
    """Collector for parallel scaling analysis."""
    
    NAME = "parallel"
    
    def __init__(self, config: ProfilingConfig):
        """Initialize parallel collector.
        
        Args:
            config: Profiling configuration
        """
        self.config = config
        self.contention_detector = ContentionDetector(config)
    
    def collect(
        self,
        binary: Path,
        args: List[str],
        run: ProfilingRun,
        thread_counts: Optional[List[int]] = None,
        warmup_iterations: int = 1,
        measurement_iterations: int = 3,
        enable_contention: bool = False,
        continue_on_error: bool = False,
        timeout_seconds: int = 600
    ) -> Dict:
        """Collect parallel scaling data.
        
        Args:
            binary: Path to binary to profile
            args: Command-line arguments
            run: ProfilingRun metadata
            thread_counts: List of thread counts to test (default: [1, 2, 4, 8])
            warmup_iterations: Number of warmup iterations per thread count
            measurement_iterations: Number of measurement iterations per thread count
            enable_contention: Whether to collect contention metrics
            continue_on_error: Whether to continue testing if a thread count fails
            timeout_seconds: Timeout per iteration
            
        Returns:
            Dictionary with scaling data
        """
        # Default thread counts
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8]
        
        # Create scaling config
        scaling_config = ScalingTestConfig(
            thread_counts=thread_counts,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            timeout_seconds=timeout_seconds,
            enable_contention=enable_contention,
            continue_on_error=continue_on_error
        )
        
        # Run scaling tests
        print(f"🔄 Running scaling tests for thread counts: {thread_counts}")
        runner = ScalingTestRunner(self.config, scaling_config)
        scaling_results = runner.run(binary, args)
        
        # Collect contention data if enabled
        contention_data = []
        if enable_contention:
            if not self.contention_detector.is_available():
                print("⚠️  Contention detection requested but perf is not available")
            else:
                print("🔍 Collecting contention metrics...")
                for result in scaling_results:
                    if result.success:
                        metrics = self.contention_detector.collect(
                            binary,
                            args,
                            result.thread_count,
                            timeout_seconds
                        )
                        contention_data.append(metrics)
                        
                        if metrics.success:
                            # Attach to scaling result
                            result.perf_data = self.contention_detector.to_dict(metrics)
        
        # Compute speedup and efficiency
        results_dicts = runner.to_dict(scaling_results)["results"]
        
        try:
            speedup_metrics = compute_speedup_efficiency(results_dicts)
            print(f"✅ Computed speedup/efficiency for {len(speedup_metrics)} thread counts")
        except ValueError as e:
            print(f"⚠️  Failed to compute speedup/efficiency: {e}")
            speedup_metrics = []
        
        # Estimate Amdahl's law
        amdahl_estimate = None
        if len(speedup_metrics) >= 2:
            try:
                amdahl_estimate = estimate_amdahl_serial_fraction(
                    speedup_metrics,
                    method="harmonic"
                )
                print(f"📊 Amdahl estimate: {amdahl_estimate.serial_fraction * 100:.2f}% serial fraction")
            except ValueError as e:
                print(f"⚠️  Failed to estimate Amdahl: {e}")
        
        # Detect bottlenecks
        bottlenecks = detect_scaling_bottlenecks(speedup_metrics) if speedup_metrics else []
        if bottlenecks:
            print(f"⚠️  Detected {len(bottlenecks)} potential bottleneck(s)")
            for bottleneck in bottlenecks:
                print(f"   - {bottleneck}")
        
        # Build output
        output = {
            "collector": self.NAME,
            "timestamp": datetime.now().isoformat(),
            "run_id": run.run_id,
            "config": {
                "thread_counts": thread_counts,
                "warmup_iterations": warmup_iterations,
                "measurement_iterations": measurement_iterations,
                "timeout_seconds": timeout_seconds,
                "enable_contention": enable_contention
            },
            "scaling_results": results_dicts,
            "speedup_metrics": [
                {
                    "thread_count": m.thread_count,
                    "duration": m.duration,
                    "speedup": m.speedup,
                    "efficiency": m.efficiency,
                    "is_regression": m.is_regression
                }
                for m in speedup_metrics
            ],
            "bottlenecks": bottlenecks
        }
        
        # Add Amdahl estimate if available
        if amdahl_estimate:
            output["amdahl_estimate"] = {
                "serial_fraction": amdahl_estimate.serial_fraction,
                "parallel_fraction": amdahl_estimate.parallel_fraction,
                "predicted_max_speedup": (
                    amdahl_estimate.predicted_max_speedup
                    if not float('inf') == amdahl_estimate.predicted_max_speedup
                    else None
                ),
                "confidence": amdahl_estimate.confidence,
                "estimation_method": amdahl_estimate.estimation_method
            }
        
        # Add contention analysis if available
        if contention_data:
            from .contention import analyze_contention_trends
            contention_analysis = analyze_contention_trends(contention_data)
            output["contention_analysis"] = contention_analysis
        
        # Save to file
        output_dir = self._get_output_dir(run)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "scaling_data.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved scaling data to: {output_file}")
        
        return output
    
    def _get_output_dir(self, run: ProfilingRun) -> Path:
        """Get output directory for this collector.
        
        Args:
            run: ProfilingRun metadata
            
        Returns:
            Path to output directory
        """
        return self.config.output_dir / "runs" / run.run_id / self.NAME


def load_scaling_data(run_dir: Path) -> Optional[Dict]:
    """Load scaling data from a run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Scaling data dictionary or None if not found
    """
    scaling_file = run_dir / "parallel" / "scaling_data.json"
    if not scaling_file.exists():
        return None
    
    with open(scaling_file, 'r') as f:
        return json.load(f)
