"""RSS collector tests."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from powers_profile.collectors.rss import RssCollector, RssMonitor, RssSample
from powers_profile.config import load_config


def test_rss_monitor_computes_summary() -> None:
    """Test RSS monitor computes correct summary statistics."""
    monitor = RssMonitor()
    monitor.samples = [
        RssSample(0.0, 1000, time.time()),
        RssSample(0.5, 1500, time.time()),
        RssSample(1.0, 2000, time.time()),
        RssSample(1.5, 1800, time.time()),
        RssSample(2.0, 2200, time.time()),
    ]
    
    summary = monitor.compute_summary()
    
    assert summary['sample_count'] == 5
    assert summary['min_rss_kb'] == 1000
    assert summary['max_rss_kb'] == 2200
    assert summary['final_rss_kb'] == 2200
    assert summary['peak_rss_mb'] == pytest.approx(2200 / 1024, rel=0.01)
    assert summary['mean_rss_kb'] == 1700  # (1000+1500+2000+1800+2200)/5


def test_rss_monitor_detects_growth() -> None:
    """Test RSS monitor detects memory growth."""
    monitor = RssMonitor()
    
    # Create samples showing growth
    base_time = time.time()
    monitor.samples = [
        RssSample(i * 0.1, 1000 + i * 1024, base_time + i * 0.1)
        for i in range(20)
    ]
    
    summary = monitor.compute_summary()
    assert summary['growth_detected'] is True


def test_rss_monitor_no_growth() -> None:
    """Test RSS monitor doesn't detect growth for stable memory."""
    monitor = RssMonitor()
    
    # Create stable samples
    base_time = time.time()
    monitor.samples = [
        RssSample(i * 0.1, 1000, base_time + i * 0.1)
        for i in range(20)
    ]
    
    summary = monitor.compute_summary()
    assert summary['growth_detected'] is False


def test_rss_monitor_empty_samples() -> None:
    """Test RSS monitor handles empty samples."""
    monitor = RssMonitor()
    summary = monitor.compute_summary()
    
    assert summary['sample_count'] == 0
    assert summary['min_rss_kb'] == 0
    assert summary['max_rss_kb'] == 0
    assert summary['peak_rss_mb'] == 0.0


def test_rss_collector_handles_missing_proc(tmp_path: Path) -> None:
    """Test RSS collector fails gracefully when /proc is unavailable."""
    config = load_config()
    collector = RssCollector()
    
    with patch('pathlib.Path.exists', return_value=False):
        result = collector.collect(
            binary=Path("/bin/true"),
            args=[],
            config=config,
            run_dir=tmp_path,
        )
    
    assert result.success is False
    assert any("/proc" in err for err in result.errors)


def test_rss_collector_validates_interval(tmp_path: Path) -> None:
    """Test RSS collector validates interval configuration."""
    config = load_config(cli_overrides={"memory": {"rss_interval_ms": 0}})
    collector = RssCollector()
    
    # Should not crash, should use default
    with patch('subprocess.Popen') as mock_popen:
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process
        
        with patch.object(RssMonitor, 'start_monitoring'):
            with patch.object(RssMonitor, 'stop_monitoring'):
                result = collector.collect(
                    binary=Path("/bin/true"),
                    args=[],
                    config=config,
                    run_dir=tmp_path,
                )
        
        assert any("Invalid RSS interval" in err for err in result.errors)
        assert any("default interval" in w for w in result.warnings)


@patch('subprocess.Popen')
def test_rss_collector_successful_run(mock_popen, tmp_path: Path) -> None:
    """Test RSS collector successful execution."""
    config = load_config()
    collector = RssCollector()
    
    # Mock process
    mock_process = Mock()
    mock_process.pid = 12345
    mock_process.returncode = 0
    mock_process.communicate.return_value = ("output\n", "")
    mock_popen.return_value = mock_process
    
    # Mock RSS monitor
    with patch.object(RssMonitor, 'start_monitoring'):
        with patch.object(RssMonitor, 'stop_monitoring'):
            with patch.object(RssMonitor, 'compute_summary') as mock_summary:
                mock_summary.return_value = {
                    'sample_count': 10,
                    'peak_rss_mb': 100.0,
                    'mean_rss_mb': 80.0,
                    'growth_detected': False,
                }
                
                result = collector.collect(
                    binary=Path("/bin/true"),
                    args=[],
                    config=config,
                    run_dir=tmp_path,
                )
    
    assert result.success is True
    assert result.data['exit_code'] == 0
    assert result.data['sample_count'] == 10
    assert result.data['peak_rss_mb'] == 100.0


@patch('subprocess.Popen')
def test_rss_collector_handles_timeout(mock_popen, tmp_path: Path) -> None:
    """Test RSS collector handles process timeout."""
    config = load_config()
    collector = RssCollector()
    
    # Mock process that times out
    mock_process = Mock()
    mock_process.pid = 12345
    mock_process.communicate.side_effect = [
        Exception("timeout"),  # Simulate timeout
    ]
    mock_process.kill = Mock()
    mock_process.wait = Mock()
    mock_popen.return_value = mock_process
    
    with patch.object(RssMonitor, 'start_monitoring'):
        with patch.object(RssMonitor, 'stop_monitoring'):
            with patch.object(RssMonitor, 'compute_summary') as mock_summary:
                mock_summary.return_value = {
                    'sample_count': 5,
                    'peak_rss_mb': 50.0,
                    'mean_rss_mb': 40.0,
                    'growth_detected': False,
                }
                
                result = collector.collect(
                    binary=Path("/bin/sleep"),
                    args=["1000"],
                    config=config,
                    run_dir=tmp_path,
                )
    
    # Should still produce results from samples collected
    assert result.data['sample_count'] == 5


@patch('subprocess.Popen')
def test_rss_collector_warns_on_few_samples(mock_popen, tmp_path: Path) -> None:
    """Test RSS collector warns when few samples are collected."""
    config = load_config()
    collector = RssCollector()
    
    mock_process = Mock()
    mock_process.pid = 12345
    mock_process.returncode = 0
    mock_process.communicate.return_value = ("", "")
    mock_popen.return_value = mock_process
    
    with patch.object(RssMonitor, 'start_monitoring'):
        with patch.object(RssMonitor, 'stop_monitoring'):
            with patch.object(RssMonitor, 'compute_summary') as mock_summary:
                mock_summary.return_value = {
                    'sample_count': 3,  # Too few
                    'peak_rss_mb': 10.0,
                    'mean_rss_mb': 8.0,
                    'growth_detected': False,
                }
                
                result = collector.collect(
                    binary=Path("/bin/true"),
                    args=[],
                    config=config,
                    run_dir=tmp_path,
                )
    
    assert any("3 samples" in w for w in result.warnings)
    assert any("too short" in w for w in result.warnings)
