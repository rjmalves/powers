"""Memory collector integration tests."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from powers_profile.collectors.memory import MemoryCollector
from powers_profile.config import load_config
from powers_profile.schemas import CollectorResult


@patch('powers_profile.collectors.memory.DhatCollector.collect')
@patch('powers_profile.collectors.memory.MassifCollector.collect')
@patch('powers_profile.collectors.memory.CachegrindCollector.collect')
@patch('powers_profile.collectors.memory.RssCollector.collect')
def test_memory_collector_orchestrates_all_tools(
    mock_rss, mock_cg, mock_massif, mock_dhat, tmp_path: Path
) -> None:
    """Test memory collector runs all enabled tools."""
    config = load_config(
        cli_overrides={
            "memory": {
                "dhat_enabled": True,
                "massif_enabled": True,
                "cachegrind_enabled": True,
            }
        }
    )
    
    # Mock successful results from all collectors
    mock_dhat.return_value = CollectorResult(
        collector_name="dhat",
        success=True,
        duration_seconds=1.0,
        data={
            'total_blocks': 100,
            'total_bytes': 10000,
            'hotspots': [{'total_bytes': 5000}],
            'hotspot_count': 1,
        },
    )
    
    mock_massif.return_value = CollectorResult(
        collector_name="massif",
        success=True,
        duration_seconds=1.0,
        data={
            'peak_mb': 50.0,
            'peak_heap_mb': 45.0,
            'final_mb': 40.0,
            'growth_mb': 5.0,
            'snapshot_count': 10,
        },
    )
    
    mock_cg.return_value = CollectorResult(
        collector_name="cachegrind",
        success=True,
        duration_seconds=1.0,
        data={
            'instructions': 1000000,
            'i1_miss_rate': 2.5,
            'overall_d1_miss_rate': 5.0,
            'overall_dll_miss_rate': 0.5,
        },
    )
    
    mock_rss.return_value = CollectorResult(
        collector_name="rss",
        success=True,
        duration_seconds=1.0,
        data={
            'peak_rss_mb': 55.0,
            'mean_rss_mb': 45.0,
            'final_rss_mb': 48.0,
            'sample_count': 20,
            'growth_detected': False,
        },
    )
    
    collector = MemoryCollector()
    result = collector.collect(
        binary=Path("/bin/true"),
        args=[],
        config=config,
        run_dir=tmp_path,
    )
    
    assert result.success is True
    assert 'dhat' in result.data['tools_run']
    assert 'massif' in result.data['tools_run']
    assert 'cachegrind' in result.data['tools_run']
    assert 'rss' in result.data['tools_run']
    assert len(result.data['tools_successful']) == 4
    
    # Check aggregated metrics
    metrics = result.data['metrics']
    assert 'dhat' in metrics
    assert 'massif' in metrics
    assert 'cachegrind' in metrics
    assert 'rss' in metrics
    
    assert metrics['dhat']['total_blocks'] == 100
    assert metrics['massif']['peak_mb'] == 50.0
    assert metrics['cachegrind']['instructions'] == 1000000
    assert metrics['rss']['peak_mb'] == 55.0


@patch('powers_profile.collectors.memory.DhatCollector.collect')
@patch('powers_profile.collectors.memory.MassifCollector.collect')
@patch('powers_profile.collectors.memory.CachegrindCollector.collect')
@patch('powers_profile.collectors.memory.RssCollector.collect')
def test_memory_collector_handles_partial_failures(
    mock_rss, mock_cg, mock_massif, mock_dhat, tmp_path: Path
) -> None:
    """Test memory collector succeeds even if some tools fail."""
    config = load_config(
        cli_overrides={
            "memory": {
                "dhat_enabled": True,
                "massif_enabled": True,
                "cachegrind_enabled": False,
            }
        }
    )
    
    # DHAT fails
    mock_dhat.return_value = CollectorResult(
        collector_name="dhat",
        success=False,
        duration_seconds=0.5,
        errors=["valgrind not found"],
    )
    
    # Massif succeeds
    mock_massif.return_value = CollectorResult(
        collector_name="massif",
        success=True,
        duration_seconds=1.0,
        data={'peak_mb': 50.0, 'snapshot_count': 10},
    )
    
    # Cachegrind skipped (disabled)
    mock_cg.return_value = CollectorResult(
        collector_name="cachegrind",
        success=True,
        duration_seconds=0.0,
        data={'skipped': True},
        warnings=["Cachegrind disabled in config; skipping"],
    )
    
    # RSS succeeds
    mock_rss.return_value = CollectorResult(
        collector_name="rss",
        success=True,
        duration_seconds=1.0,
        data={'peak_rss_mb': 55.0, 'sample_count': 20},
    )
    
    collector = MemoryCollector()
    result = collector.collect(
        binary=Path("/bin/true"),
        args=[],
        config=config,
        run_dir=tmp_path,
    )
    
    # Should succeed overall if at least one tool succeeded
    assert result.success is True
    assert 'massif' in result.data['tools_successful']
    assert 'rss' in result.data['tools_successful']
    assert 'dhat' not in result.data['tools_successful']
    
    # Should have errors from DHAT
    assert any("DHAT" in err for err in result.errors)


@patch('powers_profile.collectors.memory.DhatCollector.collect')
@patch('powers_profile.collectors.memory.MassifCollector.collect')
@patch('powers_profile.collectors.memory.CachegrindCollector.collect')
@patch('powers_profile.collectors.memory.RssCollector.collect')
def test_memory_collector_respects_config_flags(
    mock_rss, mock_cg, mock_massif, mock_dhat, tmp_path: Path
) -> None:
    """Test memory collector respects enable/disable flags."""
    config = load_config(
        cli_overrides={
            "memory": {
                "dhat_enabled": False,
                "massif_enabled": False,
                "cachegrind_enabled": False,
            }
        }
    )
    
    # All tools return skipped
    for mock_collector in [mock_dhat, mock_massif, mock_cg]:
        mock_collector.return_value = CollectorResult(
            collector_name="test",
            success=True,
            duration_seconds=0.0,
            data={'skipped': True},
            warnings=["disabled in config; skipping"],
        )
    
    # Only RSS runs
    mock_rss.return_value = CollectorResult(
        collector_name="rss",
        success=True,
        duration_seconds=1.0,
        data={'peak_rss_mb': 55.0, 'sample_count': 20},
    )
    
    collector = MemoryCollector()
    result = collector.collect(
        binary=Path("/bin/true"),
        args=[],
        config=config,
        run_dir=tmp_path,
    )
    
    # Should have run all tools but most are skipped
    assert result.success is True
    assert result.data['summary']['successful_tools'] >= 1


@patch('powers_profile.collectors.memory.DhatCollector.collect')
@patch('powers_profile.collectors.memory.MassifCollector.collect')
@patch('powers_profile.collectors.memory.CachegrindCollector.collect')
@patch('powers_profile.collectors.memory.RssCollector.collect')
def test_memory_collector_saves_aggregated_json(
    mock_rss, mock_cg, mock_massif, mock_dhat, tmp_path: Path
) -> None:
    """Test memory collector saves aggregated JSON output."""
    import json
    
    config = load_config()
    
    mock_dhat.return_value = CollectorResult(
        collector_name="dhat",
        success=True,
        duration_seconds=1.0,
        data={'total_blocks': 100},
    )
    
    mock_massif.return_value = CollectorResult(
        collector_name="massif",
        success=True,
        duration_seconds=1.0,
        data={'peak_mb': 50.0},
    )
    
    mock_cg.return_value = CollectorResult(
        collector_name="cachegrind",
        success=True,
        duration_seconds=1.0,
        data={'skipped': True},
    )
    
    mock_rss.return_value = CollectorResult(
        collector_name="rss",
        success=True,
        duration_seconds=1.0,
        data={'peak_rss_mb': 55.0},
    )
    
    collector = MemoryCollector()
    result = collector.collect(
        binary=Path("/bin/true"),
        args=["--version"],
        config=config,
        run_dir=tmp_path,
    )
    
    # Check JSON file was created
    memory_json = tmp_path / "memory" / "memory_data.json"
    assert memory_json.exists()
    
    # Verify JSON content
    with open(memory_json, 'r') as f:
        data = json.load(f)
    
    assert 'timestamp' in data
    assert data['binary'] == str(Path("/bin/true"))
    assert data['args'] == ["--version"]
    assert 'metrics' in data
    assert 'dhat' in data['metrics']
    assert 'massif' in data['metrics']
