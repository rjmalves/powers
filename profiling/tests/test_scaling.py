"""Unit tests for scaling analysis and test runner."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from powers_profile.collectors.scaling_runner import (
    ScalingTestConfig,
    ScalingTestRunner,
    ScalingRunResult,
)
from powers_profile.analyzers.scaling import (
    compute_speedup_efficiency,
    estimate_amdahl_serial_fraction,
    detect_scaling_bottlenecks,
    format_scaling_summary,
    SpeedupMetrics,
    AmdahlEstimate,
)
from powers_profile.config import ProfilingConfig


# Test Data Fixtures

def create_mock_config():
    """Create a mock profiling config."""
    config = Mock(spec=ProfilingConfig)
    config.repo_root = Path("/mock/repo")
    config.storage_dir = Path("/mock/storage")
    return config


def create_successful_result(thread_count: int, duration: float) -> dict:
    """Create a successful scaling result dict."""
    return {
        "thread_count": thread_count,
        "mean_duration": duration,
        "std_dev": duration * 0.05,
        "min_duration": duration * 0.95,
        "max_duration": duration * 1.05,
        "iterations": 3,
        "warmup_iterations": 1,
        "success": True,
        "error_message": None,
    }


# ScalingTestConfig Tests

def test_scaling_config_validation_empty_thread_counts():
    """Test that empty thread counts raise ValueError."""
    config = create_mock_config()
    
    with pytest.raises(ValueError, match="Thread counts list cannot be empty"):
        scaling_config = ScalingTestConfig(thread_counts=[])
        ScalingTestRunner(config, scaling_config)


def test_scaling_config_validation_negative_threads():
    """Test that negative thread counts raise ValueError."""
    config = create_mock_config()
    
    with pytest.raises(ValueError, match="Thread counts must be positive"):
        scaling_config = ScalingTestConfig(thread_counts=[1, -1, 4])
        ScalingTestRunner(config, scaling_config)


def test_scaling_config_validation_duplicate_threads():
    """Test that duplicate thread counts raise ValueError."""
    config = create_mock_config()
    
    with pytest.raises(ValueError, match="Thread counts must be unique"):
        scaling_config = ScalingTestConfig(thread_counts=[1, 2, 2, 4])
        ScalingTestRunner(config, scaling_config)


def test_scaling_config_validation_zero_iterations():
    """Test that zero measurement iterations raise ValueError."""
    config = create_mock_config()
    
    with pytest.raises(ValueError, match="Measurement iterations must be at least 1"):
        scaling_config = ScalingTestConfig(
            thread_counts=[1, 2],
            measurement_iterations=0
        )
        ScalingTestRunner(config, scaling_config)


# Speedup/Efficiency Tests

def test_compute_speedup_efficiency_ideal_scaling():
    """Test speedup computation with ideal scaling."""
    results = [
        create_successful_result(1, 100.0),
        create_successful_result(2, 50.0),
        create_successful_result(4, 25.0),
        create_successful_result(8, 12.5),
    ]
    
    metrics = compute_speedup_efficiency(results)
    
    assert len(metrics) == 4
    
    # Check thread_count=1 (baseline)
    assert metrics[0].thread_count == 1
    assert metrics[0].speedup == pytest.approx(1.0)
    assert metrics[0].efficiency == pytest.approx(1.0)
    
    # Check thread_count=2
    assert metrics[1].thread_count == 2
    assert metrics[1].speedup == pytest.approx(2.0)
    assert metrics[1].efficiency == pytest.approx(1.0)
    
    # Check thread_count=4
    assert metrics[2].thread_count == 4
    assert metrics[2].speedup == pytest.approx(4.0)
    assert metrics[2].efficiency == pytest.approx(1.0)
    
    # Check thread_count=8
    assert metrics[3].thread_count == 8
    assert metrics[3].speedup == pytest.approx(8.0)
    assert metrics[3].efficiency == pytest.approx(1.0)


def test_compute_speedup_efficiency_sublinear_scaling():
    """Test speedup computation with sublinear scaling."""
    results = [
        create_successful_result(1, 100.0),
        create_successful_result(2, 60.0),  # 1.67x instead of 2x
        create_successful_result(4, 40.0),  # 2.5x instead of 4x
    ]
    
    metrics = compute_speedup_efficiency(results)
    
    assert len(metrics) == 3
    
    # Thread count 2: speedup = 100/60 = 1.67, efficiency = 1.67/2 = 0.835
    assert metrics[1].speedup == pytest.approx(100.0 / 60.0)
    assert metrics[1].efficiency == pytest.approx((100.0 / 60.0) / 2)
    
    # Thread count 4: speedup = 100/40 = 2.5, efficiency = 2.5/4 = 0.625
    assert metrics[2].speedup == pytest.approx(2.5)
    assert metrics[2].efficiency == pytest.approx(0.625)


def test_compute_speedup_efficiency_regression():
    """Test detection of performance regression."""
    results = [
        create_successful_result(1, 100.0),
        create_successful_result(2, 50.0),
        create_successful_result(4, 60.0),  # Regression: slower than 2 threads
    ]
    
    metrics = compute_speedup_efficiency(results)
    
    assert len(metrics) == 3
    assert not metrics[1].is_regression
    assert metrics[2].is_regression  # 4 threads slower than 2


def test_compute_speedup_efficiency_missing_baseline():
    """Test error handling when baseline (T1) is missing."""
    results = [
        create_successful_result(2, 50.0),
        create_successful_result(4, 25.0),
    ]
    
    with pytest.raises(ValueError, match="No thread_count=1 result found"):
        compute_speedup_efficiency(results)


def test_compute_speedup_efficiency_empty_results():
    """Test error handling with empty results."""
    with pytest.raises(ValueError, match="Results list cannot be empty"):
        compute_speedup_efficiency([])


def test_compute_speedup_efficiency_all_failed():
    """Test error handling when all results failed."""
    results = [
        {
            "thread_count": 1,
            "mean_duration": 100.0,
            "success": False,
        },
        {
            "thread_count": 2,
            "mean_duration": 50.0,
            "success": False,
        },
    ]
    
    with pytest.raises(ValueError, match="No successful results"):
        compute_speedup_efficiency(results)


def test_compute_speedup_efficiency_with_custom_baseline():
    """Test speedup computation with custom baseline duration."""
    results = [
        create_successful_result(2, 50.0),
        create_successful_result(4, 25.0),
    ]
    
    # Provide custom baseline of 100.0
    metrics = compute_speedup_efficiency(results, baseline_duration=100.0)
    
    assert len(metrics) == 2
    assert metrics[0].speedup == pytest.approx(2.0)  # 100/50
    assert metrics[1].speedup == pytest.approx(4.0)  # 100/25


# Amdahl Estimation Tests

def test_amdahl_estimation_ideal_scaling():
    """Test Amdahl estimation with ideal scaling (no serial fraction)."""
    speedup_metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 50.0, 2.0, 1.0),
        SpeedupMetrics(4, 25.0, 4.0, 1.0),
        SpeedupMetrics(8, 12.5, 8.0, 1.0),
    ]
    
    estimate = estimate_amdahl_serial_fraction(speedup_metrics, method="harmonic")
    
    # With ideal scaling, serial fraction should be ~0
    assert estimate.serial_fraction < 0.01
    assert estimate.parallel_fraction > 0.99
    assert estimate.confidence in ["low", "medium", "high"]


def test_amdahl_estimation_50_percent_serial():
    """Test Amdahl estimation with 50% serial fraction."""
    # With 50% serial, max speedup is 2.0
    speedup_metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 75.0, 1.33, 0.67),
        SpeedupMetrics(4, 62.5, 1.6, 0.4),
        SpeedupMetrics(8, 56.25, 1.78, 0.22),
    ]
    
    estimate = estimate_amdahl_serial_fraction(speedup_metrics, method="harmonic")
    
    # Should estimate around 50% serial fraction
    # (actual might vary slightly due to discrete data points)
    assert 0.3 < estimate.serial_fraction < 0.7
    assert estimate.predicted_max_speedup < 3.5  # Theoretical max is 2.0


def test_amdahl_estimation_max_threads_method():
    """Test Amdahl estimation using max_threads method."""
    speedup_metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 60.0, 1.67, 0.835),
        SpeedupMetrics(4, 40.0, 2.5, 0.625),
    ]
    
    estimate = estimate_amdahl_serial_fraction(speedup_metrics, method="max_threads")
    
    assert estimate.estimation_method == "max_threads"
    assert 0.0 <= estimate.serial_fraction <= 1.0
    assert estimate.parallel_fraction == pytest.approx(1.0 - estimate.serial_fraction)


def test_amdahl_estimation_insufficient_data():
    """Test error handling with insufficient data points."""
    speedup_metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
    ]
    
    with pytest.raises(ValueError, match="Need at least 2 thread counts"):
        estimate_amdahl_serial_fraction(speedup_metrics)


def test_amdahl_estimation_confidence_levels():
    """Test confidence level assignment based on thread counts."""
    # High confidence: >= 16 threads
    metrics_high = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(16, 20.0, 5.0, 0.3125),
    ]
    estimate_high = estimate_amdahl_serial_fraction(metrics_high, method="max_threads")
    assert estimate_high.confidence == "high"
    
    # Medium confidence: >= 4 threads
    metrics_med = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(8, 25.0, 4.0, 0.5),
    ]
    estimate_med = estimate_amdahl_serial_fraction(metrics_med, method="max_threads")
    assert estimate_med.confidence == "medium"
    
    # Low confidence: < 4 threads
    metrics_low = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 50.0, 2.0, 1.0),
    ]
    estimate_low = estimate_amdahl_serial_fraction(metrics_low, method="max_threads")
    assert estimate_low.confidence == "low"


# Bottleneck Detection Tests

def test_detect_scaling_bottlenecks_regression():
    """Test detection of performance regressions."""
    metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0, is_regression=False),
        SpeedupMetrics(2, 50.0, 2.0, 1.0, is_regression=False),
        SpeedupMetrics(4, 60.0, 1.67, 0.42, is_regression=True),
    ]
    
    bottlenecks = detect_scaling_bottlenecks(metrics)
    
    assert len(bottlenecks) > 0
    assert any("regression" in b.lower() for b in bottlenecks)


def test_detect_scaling_bottlenecks_poor_low_thread_efficiency():
    """Test detection of poor efficiency at low thread counts."""
    metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 80.0, 1.25, 0.625),  # 62.5% efficiency
    ]
    
    bottlenecks = detect_scaling_bottlenecks(metrics)
    
    assert len(bottlenecks) > 0
    assert any("poor efficiency" in b.lower() for b in bottlenecks)


def test_detect_scaling_bottlenecks_efficiency_cliff():
    """Test detection of sudden efficiency drops."""
    metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 50.0, 2.0, 1.0),  # 100% efficiency
        SpeedupMetrics(4, 50.0, 2.0, 0.5),  # 50% efficiency - cliff!
    ]
    
    bottlenecks = detect_scaling_bottlenecks(metrics)
    
    assert len(bottlenecks) > 0
    assert any("efficiency cliff" in b.lower() for b in bottlenecks)


def test_detect_scaling_bottlenecks_high_thread_poor_efficiency():
    """Test detection of very poor efficiency at high thread counts."""
    metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(16, 75.0, 1.33, 0.083),  # 8.3% efficiency at 16 threads
    ]
    
    bottlenecks = detect_scaling_bottlenecks(metrics)
    
    assert len(bottlenecks) > 0
    assert any("very poor efficiency" in b.lower() for b in bottlenecks)


def test_detect_scaling_bottlenecks_none():
    """Test no bottlenecks with good scaling."""
    metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 50.0, 2.0, 1.0),
        SpeedupMetrics(4, 26.0, 3.85, 0.96),
        SpeedupMetrics(8, 14.0, 7.14, 0.89),
    ]
    
    bottlenecks = detect_scaling_bottlenecks(metrics)
    
    # Might have some warnings but shouldn't have severe issues
    assert len(bottlenecks) < 2


# Format Tests

def test_format_scaling_summary():
    """Test formatting of scaling summary."""
    metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 50.0, 2.0, 1.0),
        SpeedupMetrics(4, 27.0, 3.7, 0.925),
    ]
    
    amdahl = AmdahlEstimate(
        serial_fraction=0.05,
        parallel_fraction=0.95,
        predicted_max_speedup=20.0,
        confidence="high",
        estimation_method="harmonic"
    )
    
    summary = format_scaling_summary(metrics, amdahl)
    
    assert "SCALING ANALYSIS SUMMARY" in summary
    assert "Speedup" in summary
    assert "Efficiency" in summary
    assert "AMDAHL'S LAW ESTIMATE" in summary
    assert "5.00%" in summary  # Serial fraction
    assert "high" in summary.lower()  # Confidence


def test_format_scaling_summary_no_amdahl():
    """Test formatting without Amdahl estimate."""
    metrics = [
        SpeedupMetrics(1, 100.0, 1.0, 1.0),
        SpeedupMetrics(2, 50.0, 2.0, 1.0),
    ]
    
    summary = format_scaling_summary(metrics, None)
    
    assert "SCALING ANALYSIS SUMMARY" in summary
    assert "Speedup" in summary
    assert "AMDAHL'S LAW" not in summary  # No Amdahl section
