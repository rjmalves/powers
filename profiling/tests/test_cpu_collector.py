"""CPU collector behavior tests."""

from pathlib import Path

from powers_profile.collectors.cpu import CPUCollector, _parse_perf_report
from powers_profile.config import load_config


def test_cpu_collector_handles_missing_perf(tmp_path: Path) -> None:
    config = load_config(cli_overrides={"tools": {"perf": "/nonexistent/perf"}})
    collector = CPUCollector()
    result = collector.collect(
        binary=Path("/nonexistent/binary"),
        args=[],
        config=config,
        run_dir=tmp_path,
    )
    assert result.success is False
    assert any("perf not found" in err for err in result.errors)


def test_parse_perf_report_extracts_hotspots() -> None:
    sample = """
    # Samples: 10  of event 'cycles'
    # Event count (approx.): 100000
      30.00%  powers  powers  [.] hot_function
      10.00%  powers  powers  [.] cold_function
    """
    hotspots = _parse_perf_report(sample)
    assert hotspots[0]["symbol"] == "hot_function"
    assert hotspots[0]["percent"] == 30.0
