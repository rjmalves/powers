"""Tests for memory comparison analyzer."""

from pathlib import Path

import pytest

from powers_profile.analyzers.memory_comparison import (
    MemoryComparison,
    MetricDelta,
    compare_memory_metrics,
    format_comparison_markdown,
)
from powers_profile.schemas import CollectorResult, GitInfo, ProfilingRun, SystemInfo


def test_metric_delta_compute_decrease():
    """Test MetricDelta for decreasing value (improvement for lower_is_better)."""
    delta = MetricDelta.compute(100.0, 80.0, lower_is_better=True)
    
    assert delta.baseline == 100.0
    assert delta.target == 80.0
    assert delta.delta == -20.0
    assert delta.percent_change == pytest.approx(-20.0)
    assert delta.improved is True


def test_metric_delta_compute_increase():
    """Test MetricDelta for increasing value (regression for lower_is_better)."""
    delta = MetricDelta.compute(100.0, 120.0, lower_is_better=True)
    
    assert delta.baseline == 100.0
    assert delta.target == 120.0
    assert delta.delta == 20.0
    assert delta.percent_change == pytest.approx(20.0)
    assert delta.improved is False


def test_metric_delta_compute_increase_higher_is_better():
    """Test MetricDelta for increasing value (improvement for higher_is_better)."""
    delta = MetricDelta.compute(100.0, 120.0, lower_is_better=False)
    
    assert delta.baseline == 100.0
    assert delta.target == 120.0
    assert delta.delta == 20.0
    assert delta.percent_change == pytest.approx(20.0)
    assert delta.improved is True


def test_metric_delta_zero_baseline():
    """Test MetricDelta handles zero baseline."""
    delta = MetricDelta.compute(0.0, 100.0, lower_is_better=True)
    
    assert delta.baseline == 0.0
    assert delta.target == 100.0
    assert delta.delta == 100.0
    assert delta.percent_change == 999.9  # Arbitrarily large
    assert delta.improved is False


def test_metric_delta_zero_both():
    """Test MetricDelta handles both values zero."""
    delta = MetricDelta.compute(0.0, 0.0, lower_is_better=True)
    
    assert delta.baseline == 0.0
    assert delta.target == 0.0
    assert delta.delta == 0.0
    assert delta.percent_change == 0.0
    assert delta.improved is False


def create_mock_run(run_id: str, memory_metrics: dict) -> ProfilingRun:
    """Create a mock ProfilingRun with memory metrics."""
    return ProfilingRun(
        run_id=run_id,
        timestamp="2026-01-03T18:00:00Z",
        system_info=SystemInfo(
            hostname="test",
            os="Linux",
            cpu="test-cpu",
            cpu_count=8,
            total_memory_mb=16384.0,
        ),
        git_info=GitInfo(
            commit_hash="abc123",
            commit_short="abc123",
            branch="main",
            is_dirty=False,
            tags=[],
        ),
        config={},
        binary_path="/path/to/binary",
        binary_args=["run", "example"],
        collectors_run=["memory"],
        results={
            "memory": CollectorResult(
                collector_name="memory",
                success=True,
                duration_seconds=10.0,
                data={"metrics": memory_metrics},
            )
        },
        total_duration_seconds=10.0,
        status="success",
    )


def test_compare_memory_metrics_rss():
    """Test memory comparison with RSS metrics."""
    baseline = create_mock_run(
        "baseline-001",
        {
            "rss": {
                "peak_mb": 100.0,
                "mean_mb": 80.0,
                "final_mb": 75.0,
            }
        },
    )
    
    target = create_mock_run(
        "target-001",
        {
            "rss": {
                "peak_mb": 120.0,  # 20% increase (regression)
                "mean_mb": 85.0,   # 6.25% increase
                "final_mb": 70.0,  # 6.67% decrease (improvement)
            }
        },
    )
    
    comparison = compare_memory_metrics(baseline, target, regression_threshold_percent=5.0)
    
    assert comparison.baseline_run_id == "baseline-001"
    assert comparison.target_run_id == "target-001"
    
    # RSS Peak should show regression
    assert comparison.rss_peak_mb is not None
    assert comparison.rss_peak_mb.baseline == 100.0
    assert comparison.rss_peak_mb.target == 120.0
    assert comparison.rss_peak_mb.improved is False
    assert "RSS Peak" in comparison.regressions[0]
    
    # RSS Mean should show regression (above threshold)
    assert comparison.rss_mean_mb is not None
    assert comparison.rss_mean_mb.percent_change == pytest.approx(6.25)
    assert "RSS Mean" in comparison.regressions[1]
    
    # RSS Final computed but below improvement threshold
    assert comparison.rss_final_mb is not None
    assert comparison.rss_final_mb.improved is True


def test_compare_memory_metrics_dhat():
    """Test memory comparison with DHAT metrics."""
    baseline = create_mock_run(
        "baseline-002",
        {
            "dhat": {
                "total_bytes": 1000000,
                "total_blocks": 500,
                "max_bytes": 50000,
            }
        },
    )
    
    target = create_mock_run(
        "target-002",
        {
            "dhat": {
                "total_bytes": 900000,   # 10% decrease (improvement)
                "total_blocks": 450,     # 10% decrease (improvement)
                "max_bytes": 55000,      # 10% increase (regression)
            }
        },
    )
    
    comparison = compare_memory_metrics(baseline, target, improvement_threshold_percent=5.0)
    
    assert comparison.dhat_total_bytes is not None
    assert comparison.dhat_total_bytes.improved is True
    assert "DHAT Total Bytes" in comparison.improvements[0]
    
    assert comparison.dhat_total_blocks is not None
    assert comparison.dhat_total_blocks.improved is True
    
    assert comparison.dhat_max_bytes is not None
    assert comparison.dhat_max_bytes.improved is False


def test_compare_memory_metrics_massif():
    """Test memory comparison with Massif metrics."""
    baseline = create_mock_run(
        "baseline-003",
        {
            "massif": {
                "peak_mb": 150.0,
                "peak_heap_mb": 140.0,
                "growth_mb": 20.0,
            }
        },
    )
    
    target = create_mock_run(
        "target-003",
        {
            "massif": {
                "peak_mb": 160.0,     # 6.67% increase
                "peak_heap_mb": 145.0, # 3.57% increase
                "growth_mb": 15.0,     # 25% decrease
            }
        },
    )
    
    comparison = compare_memory_metrics(baseline, target)
    
    assert comparison.massif_peak_mb is not None
    assert comparison.massif_peak_mb.percent_change == pytest.approx(6.67, rel=0.01)
    
    assert comparison.massif_peak_heap_mb is not None
    assert comparison.massif_peak_heap_mb.percent_change == pytest.approx(3.57, rel=0.01)
    
    assert comparison.massif_growth_mb is not None
    assert comparison.massif_growth_mb.improved is True


def test_compare_memory_metrics_cachegrind():
    """Test memory comparison with Cachegrind metrics."""
    baseline = create_mock_run(
        "baseline-004",
        {
            "cachegrind": {
                "instructions": 1000000000,
                "i1_miss_rate": 2.5,
                "overall_d1_miss_rate": 5.0,
                "overall_dll_miss_rate": 0.5,
            }
        },
    )
    
    target = create_mock_run(
        "target-004",
        {
            "cachegrind": {
                "instructions": 950000000,  # 5% decrease
                "i1_miss_rate": 2.0,        # 20% decrease
                "overall_d1_miss_rate": 5.5, # 10% increase
                "overall_dll_miss_rate": 0.4, # 20% decrease
            }
        },
    )
    
    comparison = compare_memory_metrics(baseline, target, improvement_threshold_percent=5.0)
    
    assert comparison.cachegrind_instructions is not None
    assert comparison.cachegrind_instructions.improved is True
    
    assert comparison.cachegrind_i1_miss_rate is not None
    assert comparison.cachegrind_i1_miss_rate.improved is True
    
    assert comparison.cachegrind_d1_miss_rate is not None
    assert comparison.cachegrind_d1_miss_rate.improved is False
    
    assert comparison.cachegrind_dll_miss_rate is not None
    assert comparison.cachegrind_dll_miss_rate.improved is True


def test_compare_memory_metrics_missing_collector():
    """Test memory comparison when collector not run."""
    baseline = create_mock_run("baseline-005", {})
    target = create_mock_run("target-005", {})
    
    # Set results to empty (collector not run)
    baseline.results = {}
    target.results = {}
    
    comparison = compare_memory_metrics(baseline, target)
    
    assert len(comparison.warnings) > 0
    assert "Memory collector not run" in comparison.warnings[0]


def test_compare_memory_metrics_failed_collector():
    """Test memory comparison when collector failed."""
    baseline = create_mock_run("baseline-006", {})
    target = create_mock_run("target-006", {})
    
    # Set collector as failed
    baseline.results["memory"].success = False
    target.results["memory"].success = False
    
    comparison = compare_memory_metrics(baseline, target)
    
    assert len(comparison.warnings) > 0
    assert "failed" in comparison.warnings[0].lower()


def test_format_comparison_markdown():
    """Test markdown formatting of comparison."""
    baseline = create_mock_run(
        "baseline-007",
        {
            "rss": {
                "peak_mb": 100.0,
                "mean_mb": 80.0,
            },
            "dhat": {
                "total_bytes": 1000000,
            },
        },
    )
    
    target = create_mock_run(
        "target-007",
        {
            "rss": {
                "peak_mb": 110.0,  # 10% regression
                "mean_mb": 75.0,   # 6.25% improvement
            },
            "dhat": {
                "total_bytes": 900000,  # 10% improvement
            },
        },
    )
    
    comparison = compare_memory_metrics(baseline, target, regression_threshold_percent=5.0)
    markdown = format_comparison_markdown(comparison)
    
    # Check structure
    assert "# Memory Comparison" in markdown
    assert "**Baseline**: baseline-007" in markdown
    assert "**Target**: target-007" in markdown
    
    # Check regressions section
    assert "## 🔴 Regressions" in markdown
    assert "RSS Peak" in markdown
    
    # Check improvements section
    assert "## 🟢 Improvements" in markdown
    assert "RSS Mean" in markdown or "DHAT Total Bytes" in markdown
    
    # Check detailed metrics
    assert "### RSS" in markdown
    assert "### DHAT" in markdown
    
    # Check table headers
    assert "| Metric | Baseline | Target | Delta | Change |" in markdown


def test_memory_comparison_to_dict():
    """Test MemoryComparison serialization to dict."""
    baseline = create_mock_run(
        "baseline-008",
        {
            "rss": {
                "peak_mb": 100.0,
            }
        },
    )
    
    target = create_mock_run(
        "target-008",
        {
            "rss": {
                "peak_mb": 110.0,
            }
        },
    )
    
    comparison = compare_memory_metrics(baseline, target)
    result_dict = comparison.to_dict()
    
    assert result_dict["baseline_run_id"] == "baseline-008"
    assert result_dict["target_run_id"] == "target-008"
    assert "metrics" in result_dict
    assert "rss_peak_mb" in result_dict["metrics"]
    
    rss_peak = result_dict["metrics"]["rss_peak_mb"]
    assert rss_peak["baseline"] == 100.0
    assert rss_peak["target"] == 110.0
    assert rss_peak["delta"] == 10.0
    assert rss_peak["percent_change"] == pytest.approx(10.0)
    assert rss_peak["improved"] is False
