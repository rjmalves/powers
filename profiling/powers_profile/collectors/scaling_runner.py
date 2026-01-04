"""Scaling test runner for parallelism analysis.

This module implements automated scaling tests that run a target binary
across multiple thread counts to measure parallel performance characteristics.
"""

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ProfilingConfig


@dataclass
class ScalingRunResult:
    """Results from a single scaling run at a specific thread count."""

    thread_count: int
    durations: List[float]  # Per-iteration durations in seconds
    mean_duration: float
    std_dev: float
    min_duration: float
    max_duration: float
    iterations: int
    warmup_iterations: int
    success: bool
    error_message: Optional[str] = None
    perf_data: Optional[Dict] = None  # Optional contention metrics


@dataclass
class ScalingTestConfig:
    """Configuration for scaling tests."""

    thread_counts: List[int]
    warmup_iterations: int = 1
    measurement_iterations: int = 3
    timeout_seconds: int = 1800
    enable_contention: bool = False
    continue_on_error: bool = False
    env_vars: Dict[str, str] = field(default_factory=dict)


class ScalingTestRunner:
    """Runs scaling tests across multiple thread counts."""

    def __init__(
        self, config: ProfilingConfig, scaling_config: ScalingTestConfig
    ):
        """Initialize scaling test runner.

        Args:
            config: Profiling configuration
            scaling_config: Scaling-specific configuration
        """
        self.config = config
        self.scaling_config = scaling_config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not self.scaling_config.thread_counts:
            raise ValueError("Thread counts list cannot be empty")

        if any(tc <= 0 for tc in self.scaling_config.thread_counts):
            raise ValueError("Thread counts must be positive integers")

        if len(self.scaling_config.thread_counts) != len(
            set(self.scaling_config.thread_counts)
        ):
            raise ValueError("Thread counts must be unique")

        if self.scaling_config.measurement_iterations < 1:
            raise ValueError("Measurement iterations must be at least 1")

        if self.scaling_config.warmup_iterations < 0:
            raise ValueError("Warmup iterations cannot be negative")

    def run(self, binary: Path, args: List[str]) -> List[ScalingRunResult]:
        """Run scaling tests across all configured thread counts.

        Args:
            binary: Path to the binary to execute
            args: Command-line arguments for the binary

        Returns:
            List of ScalingRunResult for each thread count
        """
        if not binary.exists():
            raise FileNotFoundError(f"Binary not found: {binary}")

        results = []
        for thread_count in self.scaling_config.thread_counts:
            try:
                result = self._run_single_thread_count(
                    binary, args, thread_count
                )
                results.append(result)

                if (
                    not result.success
                    and not self.scaling_config.continue_on_error
                ):
                    print(
                        f"⚠️  Stopping scaling test due to failure at {thread_count} threads"
                    )
                    break

            except Exception as e:
                error_result = ScalingRunResult(
                    thread_count=thread_count,
                    durations=[],
                    mean_duration=0.0,
                    std_dev=0.0,
                    min_duration=0.0,
                    max_duration=0.0,
                    iterations=0,
                    warmup_iterations=self.scaling_config.warmup_iterations,
                    success=False,
                    error_message=str(e),
                )
                results.append(error_result)

                if not self.scaling_config.continue_on_error:
                    print(
                        f"⚠️  Stopping scaling test due to exception at {thread_count} threads: {e}"
                    )
                    break

        return results

    def _run_single_thread_count(
        self, binary: Path, args: List[str], thread_count: int
    ) -> ScalingRunResult:
        """Run test at a specific thread count.

        Args:
            binary: Path to binary
            args: Command-line arguments
            thread_count: Number of threads to use

        Returns:
            ScalingRunResult with timing data
        """
        # Prepare environment with thread count
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(thread_count)

        # Add any custom environment variables
        env.update(self.scaling_config.env_vars)

        # Warmup iterations
        for i in range(self.scaling_config.warmup_iterations):
            try:
                subprocess.run(
                    [str(binary)] + args,
                    cwd=self.config.repo_root,
                    env=env,
                    capture_output=True,
                    timeout=self.scaling_config.timeout_seconds,
                    check=True,
                )
            except subprocess.TimeoutExpired:
                return ScalingRunResult(
                    thread_count=thread_count,
                    durations=[],
                    mean_duration=0.0,
                    std_dev=0.0,
                    min_duration=0.0,
                    max_duration=0.0,
                    iterations=0,
                    warmup_iterations=i,
                    success=False,
                    error_message=f"Warmup iteration {i} timed out",
                )
            except subprocess.CalledProcessError as e:
                return ScalingRunResult(
                    thread_count=thread_count,
                    durations=[],
                    mean_duration=0.0,
                    std_dev=0.0,
                    min_duration=0.0,
                    max_duration=0.0,
                    iterations=0,
                    warmup_iterations=i,
                    success=False,
                    error_message=f"Warmup iteration {i} failed: {e.stderr.decode() if e.stderr else str(e)}",
                )

        # Measurement iterations
        durations = []
        for i in range(self.scaling_config.measurement_iterations):
            try:
                start_time = time.perf_counter()
                subprocess.run(
                    [str(binary)] + args,
                    cwd=self.config.repo_root,
                    env=env,
                    capture_output=True,
                    timeout=self.scaling_config.timeout_seconds,
                    check=True,
                )
                end_time = time.perf_counter()

                duration = end_time - start_time
                durations.append(duration)

            except subprocess.TimeoutExpired:
                return ScalingRunResult(
                    thread_count=thread_count,
                    durations=durations,
                    mean_duration=sum(durations) / len(durations)
                    if durations
                    else 0.0,
                    std_dev=self._compute_std_dev(durations)
                    if durations
                    else 0.0,
                    min_duration=min(durations) if durations else 0.0,
                    max_duration=max(durations) if durations else 0.0,
                    iterations=i,
                    warmup_iterations=self.scaling_config.warmup_iterations,
                    success=False,
                    error_message=f"Measurement iteration {i} timed out",
                )
            except subprocess.CalledProcessError as e:
                return ScalingRunResult(
                    thread_count=thread_count,
                    durations=durations,
                    mean_duration=sum(durations) / len(durations)
                    if durations
                    else 0.0,
                    std_dev=self._compute_std_dev(durations)
                    if durations
                    else 0.0,
                    min_duration=min(durations) if durations else 0.0,
                    max_duration=max(durations) if durations else 0.0,
                    iterations=i,
                    warmup_iterations=self.scaling_config.warmup_iterations,
                    success=False,
                    error_message=f"Measurement iteration {i} failed: {e.stderr.decode() if e.stderr else str(e)}",
                )

        # Compute statistics
        mean_duration = sum(durations) / len(durations)
        std_dev = self._compute_std_dev(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        return ScalingRunResult(
            thread_count=thread_count,
            durations=durations,
            mean_duration=mean_duration,
            std_dev=std_dev,
            min_duration=min_duration,
            max_duration=max_duration,
            iterations=len(durations),
            warmup_iterations=self.scaling_config.warmup_iterations,
            success=True,
        )

    @staticmethod
    def _compute_std_dev(values: List[float]) -> float:
        """Compute standard deviation of values.

        Args:
            values: List of numeric values

        Returns:
            Standard deviation
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def to_dict(self, results: List[ScalingRunResult]) -> Dict:
        """Convert results to dictionary for JSON serialization.

        Args:
            results: List of scaling run results

        Returns:
            Dictionary ready for JSON serialization
        """
        return {
            "config": {
                "thread_counts": self.scaling_config.thread_counts,
                "warmup_iterations": self.scaling_config.warmup_iterations,
                "measurement_iterations": self.scaling_config.measurement_iterations,
                "timeout_seconds": self.scaling_config.timeout_seconds,
                "enable_contention": self.scaling_config.enable_contention,
                "env_vars": self.scaling_config.env_vars,
            },
            "results": [
                {
                    "thread_count": r.thread_count,
                    "durations": r.durations,
                    "mean_duration": r.mean_duration,
                    "std_dev": r.std_dev,
                    "min_duration": r.min_duration,
                    "max_duration": r.max_duration,
                    "iterations": r.iterations,
                    "warmup_iterations": r.warmup_iterations,
                    "success": r.success,
                    "error_message": r.error_message,
                    "perf_data": r.perf_data,
                }
                for r in results
            ],
        }
