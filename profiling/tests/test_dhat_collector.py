"""DHAT collector tests."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from powers_profile.collectors.dhat import DhatCollector, _parse_dhat_json
from powers_profile.config import load_config


def test_dhat_collector_handles_missing_valgrind(tmp_path: Path) -> None:
    """Test DHAT collector fails gracefully when valgrind is not found."""
    config = load_config(
        cli_overrides={"tools": {"valgrind": "/nonexistent/valgrind"}}
    )
    collector = DhatCollector()
    result = collector.collect(
        binary=Path("/bin/true"),
        args=[],
        config=config,
        run_dir=tmp_path,
    )
    assert result.success is False
    assert any("valgrind not found" in err for err in result.errors)


def test_dhat_collector_skips_when_disabled(tmp_path: Path) -> None:
    """Test DHAT collector skips execution when disabled in config."""
    config = load_config(cli_overrides={"memory": {"dhat_enabled": False}})
    collector = DhatCollector()
    result = collector.collect(
        binary=Path("/bin/true"),
        args=[],
        config=config,
        run_dir=tmp_path,
    )
    assert result.success is True
    assert result.data.get("skipped") is True
    assert any("disabled" in w for w in result.warnings)


def test_parse_dhat_json_extracts_metrics(tmp_path: Path) -> None:
    """Test parsing of DHAT JSON output with real format (pps)."""
    sample_dhat = {
        "dhatFileVersion": 2,
        "mode": "heap",
        "verb": "Allocated",
        "tu": "instrs",
        "Mtu": "Minstr",
        "tuth": 500,
        "cmd": "/usr/bin/test",
        "pid": 12345,
        "te": 1000000,
        "tg": 900000,
        "pps": [  # Use "pps" not "aps"
            {
                "tb": 5000000,  # total bytes
                "tbk": 100,     # total blocks
                "mb": 1000000,  # max bytes
                "mbk": 20,      # max blocks
                "gb": 800000,   # bytes at global max
                "gbk": 15,      # blocks at global max
                "tl": 1000,     # total lifetime
                "fs": [1, 2, 3],
            },
            {
                "tb": 3000000,
                "tbk": 80,
                "mb": 500000,
                "mbk": 10,
                "gb": 400000,
                "gbk": 8,
                "tl": 500,
                "fs": [4, 5],
            },
        ],
        "ftbl": [
            "func1",
            "func2",
            "func3",
            "func4",
            "func5",
        ],
    }
    
    dhat_path = tmp_path / "dhat.out.json"
    with open(dhat_path, 'w') as f:
        json.dump(sample_dhat, f)
    
    summary = _parse_dhat_json(dhat_path)
    
    # Totals should be calculated from pps array
    assert summary['total_blocks'] == 180  # 100 + 80
    assert summary['total_bytes'] == 8000000  # 5000000 + 3000000
    assert summary['max_blocks'] == 20  # max of (20, 10)
    assert summary['max_bytes'] == 1000000  # max of (1000000, 500000)
    assert summary['time_unit'] == 'instrs'
    assert summary['mode'] == 'heap'
    assert len(summary['hotspots']) == 2
    assert summary['hotspot_count'] == 2
    
    # Check first hotspot (should be sorted by total_bytes descending)
    hotspot1 = summary['hotspots'][0]
    assert hotspot1['total_bytes'] == 5000000
    assert hotspot1['total_blocks'] == 100
    assert 'stack_trace' in hotspot1
    assert hotspot1['stack_trace'][0] == 'func2'  # fs[0]=1 -> ftbl[1]


def test_parse_dhat_json_handles_missing_fields(tmp_path: Path) -> None:
    """Test parsing handles minimal DHAT JSON."""
    minimal_dhat = {
        "dhatFileVersion": 2,
        "pps": [],  # Use "pps" not "aps"
    }
    
    dhat_path = tmp_path / "dhat.out.json"
    with open(dhat_path, 'w') as f:
        json.dump(minimal_dhat, f)
    
    summary = _parse_dhat_json(dhat_path)
    
    assert summary['total_blocks'] == 0
    assert summary['total_bytes'] == 0
    assert summary['hotspots'] == []
    assert summary['hotspot_count'] == 0


@patch('powers_profile.collectors.dhat._run_command')
@patch('powers_profile.collectors.dhat._which_tool')
def test_dhat_collector_handles_execution_failure(
    mock_which, mock_run, tmp_path: Path
) -> None:
    """Test DHAT collector handles valgrind execution failure."""
    mock_which.return_value = Path("/usr/bin/valgrind")
    mock_run.side_effect = [
        (0, "valgrind-3.18.0", ""),  # version check
        (1, "", "Error: something went wrong"),  # dhat execution
    ]
    
    config = load_config()
    collector = DhatCollector()
    result = collector.collect(
        binary=Path("/bin/true"),
        args=[],
        config=config,
        run_dir=tmp_path,
    )
    
    assert result.success is False
    assert any("failed with exit code 1" in err for err in result.errors)


@patch('powers_profile.collectors.dhat._run_command')
@patch('powers_profile.collectors.dhat._which_tool')
def test_dhat_collector_parses_successful_run(
    mock_which, mock_run, tmp_path: Path
) -> None:
    """Test DHAT collector parses successful run."""
    mock_which.return_value = Path("/usr/bin/valgrind")
    
    # Create fake DHAT output
    dhat_out_path = tmp_path / "dhat" / "dhat.out.json"
    dhat_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sample_output = {
        "dhatFileVersion": 2,
        "mode": "heap",
        "tu": "instrs",
        "pps": [  # Use "pps" not "aps"
            {
                "tb": 10000,
                "tbk": 100,
                "mb": 5000,
                "mbk": 50,
            }
        ],
        "ftbl": [],
    }
    
    def fake_run(cmd, cwd, timeout=600):
        # Create the output file when DHAT command is run
        if "--tool=dhat" in cmd:
            for arg in cmd:
                if arg.startswith("--dhat-out-file="):
                    out_file = Path(arg.split("=", 1)[1])
                    with open(out_file, 'w') as f:
                        json.dump(sample_output, f)
            return (0, "", "")
        return (0, "valgrind-3.18.0", "")
    
    mock_run.side_effect = fake_run
    
    config = load_config()
    collector = DhatCollector()
    result = collector.collect(
        binary=Path("/bin/true"),
        args=[],
        config=config,
        run_dir=tmp_path,
    )
    
    assert result.success is True
    assert result.data['total_blocks'] == 100
    assert result.data['total_bytes'] == 10000
    assert result.data['max_bytes'] == 5000
