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
    """Test parsing real perf report format with trailing dashes."""
    sample = """
    # Samples: 10  of event 'cycles'
    # Event count (approx.): 100000
      30.00%     5.00%  [.] hot_function                                -      -
      10.00%     2.00%  [.] cold_function                               -      -
       5.00%     1.00%  [.] malloc                                      -      -
    """
    hotspots = _parse_perf_report(sample)
    assert len(hotspots) == 3
    assert hotspots[0]["symbol"] == "hot_function"
    assert hotspots[0]["percent"] == 30.0
    assert hotspots[1]["symbol"] == "cold_function"
    assert hotspots[1]["percent"] == 10.0
    assert hotspots[2]["symbol"] == "malloc"
    assert hotspots[2]["percent"] == 5.0


def test_parse_perf_report_includes_unknown_symbols() -> None:
    """Test that unknown symbols are included with [unknown] prefix."""
    sample = """
      30.00%     5.00%  [.] known_function                              -      -
      20.00%     3.00%  [.] [unknown]                                   -      -
      15.00%     2.00%  [.] 0x7f8a9b0012ab                              -      -
      10.00%     1.00%  [.] another_known                               -      -
       5.00%     0.50%  [.]                                             -      -
       4.00%     0.40%  [.] 0000000000000000                            -      -
       3.00%     0.30%  [.] 123456789                                   -      -
    """
    hotspots = _parse_perf_report(sample)
    
    # Should include all non-empty symbols (6 total, empty [.] is skipped)
    assert len(hotspots) == 6
    
    # Known functions preserved
    assert hotspots[0]["symbol"] == "known_function"
    assert hotspots[0]["percent"] == 30.0
    
    # [unknown] preserved as-is
    assert hotspots[1]["symbol"] == "[unknown]"
    assert hotspots[1]["percent"] == 20.0
    
    # Hex addresses marked as [unknown:<addr>]
    assert hotspots[2]["symbol"] == "[unknown:0x7f8a9b0012ab]"
    assert hotspots[2]["percent"] == 15.0
    
    # Known function
    assert hotspots[3]["symbol"] == "another_known"
    assert hotspots[3]["percent"] == 10.0
    
    # Numeric addresses marked as [unknown:<addr>]
    assert hotspots[4]["symbol"] == "[unknown:0000000000000000]"
    assert hotspots[4]["percent"] == 4.0
    
    assert hotspots[5]["symbol"] == "[unknown:123456789]"
    assert hotspots[5]["percent"] == 3.0
