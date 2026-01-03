"""Timing collector parsing tests."""

from powers_profile.collectors.timing import parse_timings


def test_parse_timings_supports_multiple_formats() -> None:
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
