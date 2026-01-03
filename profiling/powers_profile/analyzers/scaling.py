"""Scaling analysis for parallel performance evaluation.

This module computes speedup, efficiency, and Amdahl's law estimates
from scaling test results.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class SpeedupMetrics:
    """Speedup and efficiency metrics for a specific thread count."""
    
    thread_count: int
    duration: float
    speedup: float
    efficiency: float
    is_regression: bool = False  # True if speedup decreased from lower thread count


@dataclass
class AmdahlEstimate:
    """Amdahl's law estimation results."""
    
    serial_fraction: float
    parallel_fraction: float
    predicted_max_speedup: float
    confidence: str  # "high", "medium", "low"
    estimation_method: str


def compute_speedup_efficiency(
    results: List[Dict],
    baseline_duration: Optional[float] = None
) -> List[SpeedupMetrics]:
    """Compute speedup and efficiency metrics from scaling results.
    
    Speedup is defined as: S(n) = T(1) / T(n)
    Efficiency is defined as: E(n) = S(n) / n
    
    Args:
        results: List of result dictionaries with thread_count and mean_duration
        baseline_duration: Optional baseline duration (T1). If None, uses thread_count=1 result
        
    Returns:
        List of SpeedupMetrics for each thread count
        
    Raises:
        ValueError: If results are empty or baseline cannot be determined
    """
    if not results:
        raise ValueError("Results list cannot be empty")
    
    # Filter successful results
    successful_results = [r for r in results if r.get("success", False)]
    if not successful_results:
        raise ValueError("No successful results to analyze")
    
    # Sort by thread count
    successful_results.sort(key=lambda r: r["thread_count"])
    
    # Determine baseline (T1)
    if baseline_duration is None:
        # Find thread_count=1 result
        baseline_results = [r for r in successful_results if r["thread_count"] == 1]
        if not baseline_results:
            raise ValueError("No thread_count=1 result found and no baseline_duration provided")
        baseline_duration = baseline_results[0]["mean_duration"]
    
    if baseline_duration <= 0:
        raise ValueError(f"Invalid baseline duration: {baseline_duration}")
    
    # Compute metrics
    metrics = []
    prev_speedup = 0.0
    
    for result in successful_results:
        thread_count = result["thread_count"]
        duration = result["mean_duration"]
        
        if duration <= 0:
            # Skip invalid durations
            continue
        
        speedup = baseline_duration / duration
        efficiency = speedup / thread_count
        
        # Check for regression (speedup decreased)
        is_regression = speedup < prev_speedup and thread_count > 1
        
        metrics.append(SpeedupMetrics(
            thread_count=thread_count,
            duration=duration,
            speedup=speedup,
            efficiency=efficiency,
            is_regression=is_regression
        ))
        
        prev_speedup = speedup
    
    return metrics


def estimate_amdahl_serial_fraction(
    speedup_metrics: List[SpeedupMetrics],
    method: str = "harmonic"
) -> AmdahlEstimate:
    """Estimate serial fraction using Amdahl's law.
    
    Amdahl's law: S(n) = 1 / (f_serial + (1 - f_serial) / n)
    Rearranging: f_serial = (1/S - 1/n) / (1 - 1/n)
    
    Args:
        speedup_metrics: List of speedup metrics
        method: Estimation method ("harmonic", "max_threads", "least_squares")
        
    Returns:
        AmdahlEstimate with serial fraction and predictions
        
    Raises:
        ValueError: If insufficient data for estimation
    """
    if not speedup_metrics:
        raise ValueError("Speedup metrics list cannot be empty")
    
    if len(speedup_metrics) < 2:
        raise ValueError("Need at least 2 thread counts for Amdahl estimation")
    
    if method == "max_threads":
        return _estimate_amdahl_max_threads(speedup_metrics)
    elif method == "harmonic":
        return _estimate_amdahl_harmonic(speedup_metrics)
    elif method == "least_squares":
        return _estimate_amdahl_least_squares(speedup_metrics)
    else:
        raise ValueError(f"Unknown estimation method: {method}")


def _estimate_amdahl_max_threads(speedup_metrics: List[SpeedupMetrics]) -> AmdahlEstimate:
    """Estimate using the highest thread count result.
    
    This gives a conservative estimate based on observed scaling at maximum parallelism.
    """
    # Use the result with highest thread count
    max_metric = max(speedup_metrics, key=lambda m: m.thread_count)
    
    n = max_metric.thread_count
    speedup = max_metric.speedup
    
    # f_serial = (1/S - 1/n) / (1 - 1/n)
    if speedup <= 0:
        serial_fraction = 1.0
    else:
        numerator = (1.0 / speedup) - (1.0 / n)
        denominator = 1.0 - (1.0 / n)
        
        if abs(denominator) < 1e-10:
            serial_fraction = 0.0
        else:
            serial_fraction = numerator / denominator
    
    # Clamp to [0, 1]
    serial_fraction = max(0.0, min(1.0, serial_fraction))
    parallel_fraction = 1.0 - serial_fraction
    
    # Predicted max speedup (as n -> infinity)
    if serial_fraction < 1e-10:
        predicted_max_speedup = float('inf')
    else:
        predicted_max_speedup = 1.0 / serial_fraction
    
    # Confidence based on thread count
    if n >= 16:
        confidence = "high"
    elif n >= 4:
        confidence = "medium"
    else:
        confidence = "low"
    
    return AmdahlEstimate(
        serial_fraction=serial_fraction,
        parallel_fraction=parallel_fraction,
        predicted_max_speedup=predicted_max_speedup,
        confidence=confidence,
        estimation_method="max_threads"
    )


def _estimate_amdahl_harmonic(speedup_metrics: List[SpeedupMetrics]) -> AmdahlEstimate:
    """Estimate using harmonic mean across all thread counts.
    
    This balances contributions from all measurements.
    """
    serial_fractions = []
    
    for metric in speedup_metrics:
        if metric.thread_count == 1:
            continue  # Skip baseline
        
        n = metric.thread_count
        speedup = metric.speedup
        
        if speedup <= 0:
            continue
        
        # f_serial = (1/S - 1/n) / (1 - 1/n)
        numerator = (1.0 / speedup) - (1.0 / n)
        denominator = 1.0 - (1.0 / n)
        
        if abs(denominator) >= 1e-10:
            f = numerator / denominator
            serial_fractions.append(max(0.0, min(1.0, f)))
    
    if not serial_fractions:
        # Fallback to max_threads method
        return _estimate_amdahl_max_threads(speedup_metrics)
    
    # Harmonic mean
    serial_fraction = sum(serial_fractions) / len(serial_fractions)
    parallel_fraction = 1.0 - serial_fraction
    
    # Predicted max speedup
    if serial_fraction < 1e-10:
        predicted_max_speedup = float('inf')
    else:
        predicted_max_speedup = 1.0 / serial_fraction
    
    # Confidence based on variance
    variance = sum((f - serial_fraction) ** 2 for f in serial_fractions) / len(serial_fractions)
    std_dev = math.sqrt(variance)
    
    if std_dev < 0.05:
        confidence = "high"
    elif std_dev < 0.15:
        confidence = "medium"
    else:
        confidence = "low"
    
    return AmdahlEstimate(
        serial_fraction=serial_fraction,
        parallel_fraction=parallel_fraction,
        predicted_max_speedup=predicted_max_speedup,
        confidence=confidence,
        estimation_method="harmonic"
    )


def _estimate_amdahl_least_squares(speedup_metrics: List[SpeedupMetrics]) -> AmdahlEstimate:
    """Estimate using least-squares fitting.
    
    Fits the Amdahl curve to observed data points.
    """
    # This is a placeholder for more sophisticated least-squares fitting
    # For now, fallback to harmonic mean
    return _estimate_amdahl_harmonic(speedup_metrics)


def format_scaling_summary(
    speedup_metrics: List[SpeedupMetrics],
    amdahl_estimate: Optional[AmdahlEstimate] = None
) -> str:
    """Format scaling metrics as a readable summary.
    
    Args:
        speedup_metrics: List of speedup metrics
        amdahl_estimate: Optional Amdahl estimate
        
    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SCALING ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Table header
    lines.append(f"{'Threads':>8} | {'Duration (s)':>12} | {'Speedup':>10} | {'Efficiency':>10} | {'Notes':>15}")
    lines.append("-" * 80)
    
    # Data rows
    for metric in speedup_metrics:
        notes = ""
        if metric.is_regression:
            notes = "⚠️  REGRESSION"
        elif metric.efficiency >= 0.9:
            notes = "🟢 Excellent"
        elif metric.efficiency >= 0.7:
            notes = "🟡 Good"
        elif metric.efficiency >= 0.5:
            notes = "🟠 Fair"
        else:
            notes = "🔴 Poor"
        
        lines.append(
            f"{metric.thread_count:>8} | "
            f"{metric.duration:>12.6f} | "
            f"{metric.speedup:>10.2f}x | "
            f"{metric.efficiency * 100:>9.1f}% | "
            f"{notes:>15}"
        )
    
    # Best metrics
    lines.append("")
    lines.append("BEST PERFORMANCE")
    lines.append("-" * 80)
    
    best_speedup = max(speedup_metrics, key=lambda m: m.speedup)
    best_efficiency = max(speedup_metrics, key=lambda m: m.efficiency)
    
    lines.append(f"Best Speedup:    {best_speedup.speedup:.2f}x at {best_speedup.thread_count} threads")
    lines.append(f"Best Efficiency: {best_efficiency.efficiency * 100:.1f}% at {best_efficiency.thread_count} threads")
    
    # Amdahl estimate
    if amdahl_estimate:
        lines.append("")
        lines.append("AMDAHL'S LAW ESTIMATE")
        lines.append("-" * 80)
        lines.append(f"Serial Fraction:     {amdahl_estimate.serial_fraction * 100:.2f}%")
        lines.append(f"Parallel Fraction:   {amdahl_estimate.parallel_fraction * 100:.2f}%")
        
        if math.isinf(amdahl_estimate.predicted_max_speedup):
            lines.append(f"Predicted Max Speedup: ∞ (ideal scaling)")
        else:
            lines.append(f"Predicted Max Speedup: {amdahl_estimate.predicted_max_speedup:.2f}x")
        
        lines.append(f"Confidence:          {amdahl_estimate.confidence.upper()}")
        lines.append(f"Method:              {amdahl_estimate.estimation_method}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def detect_scaling_bottlenecks(speedup_metrics: List[SpeedupMetrics]) -> List[str]:
    """Detect potential scaling bottlenecks from metrics.
    
    Args:
        speedup_metrics: List of speedup metrics
        
    Returns:
        List of bottleneck descriptions
    """
    bottlenecks = []
    
    # Check for regressions
    regressions = [m for m in speedup_metrics if m.is_regression]
    if regressions:
        for reg in regressions:
            bottlenecks.append(
                f"Performance regression at {reg.thread_count} threads "
                f"(speedup: {reg.speedup:.2f}x)"
            )
    
    # Check for poor efficiency at low thread counts
    low_thread_metrics = [m for m in speedup_metrics if m.thread_count <= 4]
    for metric in low_thread_metrics:
        if metric.efficiency < 0.7:
            bottlenecks.append(
                f"Poor efficiency ({metric.efficiency * 100:.1f}%) at {metric.thread_count} threads - "
                f"may indicate high parallelization overhead"
            )
    
    # Check for efficiency cliff (sudden drop)
    for i in range(1, len(speedup_metrics)):
        prev = speedup_metrics[i - 1]
        curr = speedup_metrics[i]
        
        eff_drop = prev.efficiency - curr.efficiency
        if eff_drop > 0.2:  # More than 20% drop
            bottlenecks.append(
                f"Efficiency cliff between {prev.thread_count} and {curr.thread_count} threads "
                f"({prev.efficiency * 100:.1f}% → {curr.efficiency * 100:.1f}%) - "
                f"possible contention or NUMA effects"
            )
    
    # Check for very poor efficiency at high thread counts
    high_thread_metrics = [m for m in speedup_metrics if m.thread_count >= 16]
    for metric in high_thread_metrics:
        if metric.efficiency < 0.3:
            bottlenecks.append(
                f"Very poor efficiency ({metric.efficiency * 100:.1f}%) at {metric.thread_count} threads - "
                f"likely contention or insufficient parallelizable work"
            )
    
    return bottlenecks
