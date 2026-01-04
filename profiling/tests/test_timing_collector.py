"""Timing collector parsing tests."""

from powers_profile.collectors.timing import parse_timings


def test_parse_timings_supports_multiple_formats() -> None:
    """Test parsing of legacy timing marker formats."""
    output = "\n".join(
        [
            "[TIMING] load=10ms",
            "compute_time_ms=25",
            "TIMING total = 1.5 s",
        ]
    )
    metrics = parse_timings(output)
    assert metrics["load"] == 0.01
    assert metrics["compute"] == 0.025
    assert metrics["total"] == 1.5


def test_parse_timings_supports_hh_mm_ss_format() -> None:
    """Test parsing of HH:MM:SS.mmm format used by POWE.RS."""
    output = "\n".join(
        [
            "[INFO] Training time: 00:00:19.488",
            "[INFO] Simulation time: 00:00:00.575",
            "[INFO] Total running time: 00:00:20.099",
        ]
    )
    metrics = parse_timings(output)
    
    # Check that all three timings are captured
    assert "training" in metrics
    assert "simulation" in metrics
    assert "total_running" in metrics
    
    # Check values with reasonable tolerance
    assert abs(metrics["training"] - 19.488) < 0.001
    assert abs(metrics["simulation"] - 0.575) < 0.001
    assert abs(metrics["total_running"] - 20.099) < 0.001


def test_parse_timings_handles_hours_minutes() -> None:
    """Test parsing with non-zero hours and minutes."""
    output = "[INFO] Long process time: 01:23:45.678"
    metrics = parse_timings(output)
    
    expected = 1 * 3600 + 23 * 60 + 45 + 0.678
    assert abs(metrics["long_process"] - expected) < 0.001


def test_parse_timings_normalizes_names() -> None:
    """Test that multi-word names are normalized to snake_case."""
    output = "\n".join(
        [
            "[INFO] Total running time: 00:00:10.000",
            "[INFO] Model training time: 00:00:05.000",
        ]
    )
    metrics = parse_timings(output)
    
    assert "total_running" in metrics
    assert "model_training" in metrics
    assert metrics["total_running"] == 10.0
    assert metrics["model_training"] == 5.0


def test_parse_timings_empty_output() -> None:
    """Test that empty output returns empty dict."""
    metrics = parse_timings("")
    assert metrics == {}
    
    metrics = parse_timings("[INFO] No timing data here")
    assert metrics == {}
