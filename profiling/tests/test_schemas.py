"""Schema serialization round-trip tests."""

from datetime import datetime, timezone

from powers_profile.schemas import (
    Comparison,
    GitInfo,
    HistoryEntry,
    MetricDelta,
    ProfilingRun,
    CollectorResult,
    SystemInfo,
)


def _sample_system_info() -> SystemInfo:
    return SystemInfo(
        hostname="host",
        os_name="Linux",
        os_version="6.0",
        cpu_model="Test CPU",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        cpu_freq_mhz=3200.0,
        ram_total_gb=32.0,
        rust_version="rustc 1.0",
        powers_version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _sample_git_info() -> GitInfo:
    return GitInfo(
        commit_sha="a" * 40,
        commit_short="a" * 7,
        branch="main",
        is_dirty=False,
        commit_date=datetime.now(timezone.utc).isoformat(),
        commit_message="msg",
        tags=["v0.1.0"],
    )


def test_system_info_round_trip() -> None:
    info = _sample_system_info()
    restored = SystemInfo.from_json(info.to_json())
    assert restored == info


def test_profiling_run_nested_round_trip() -> None:
    run = ProfilingRun(
        run_id="run-1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        system_info=_sample_system_info(),
        git_info=_sample_git_info(),
        config={"collectors": ["timing"]},
        binary_path="target/release/powers",
        binary_args=["--help"],
        collectors_run=["timing"],
        results={
            "timing": CollectorResult(
                collector_name="timing",
                success=True,
                duration_seconds=1.2,
                data={"value": 1},
            )
        },
        total_duration_seconds=1.2,
        status="success",
    )
    restored = ProfilingRun.from_json(run.to_json())
    assert restored.run_id == "run-1"
    assert restored.results["timing"].collector_name == "timing"
    assert restored.system_info.cpu_model == run.system_info.cpu_model


def test_comparison_round_trip() -> None:
    delta = MetricDelta(
        metric_name="runtime",
        baseline_value=1.0,
        target_value=0.9,
        absolute_delta=-0.1,
        percent_delta=-10.0,
        improved=True,
        significant=True,
    )
    comp = Comparison(
        baseline_run_id="r0",
        target_run_id="r1",
        baseline_git=_sample_git_info(),
        target_git=_sample_git_info(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        deltas={"perf": [delta]},
        regressions=[],
        improvements=[delta],
    )
    restored = Comparison.from_json(comp.to_json())
    assert restored.deltas["perf"][0].metric_name == "runtime"


def test_history_entry_round_trip() -> None:
    entry = HistoryEntry(
        run_id="r1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit="abc",
        git_branch="main",
        collectors_run=["timing"],
        status="success",
        path="runs/r1",
    )
    restored = HistoryEntry.from_json(entry.to_json())
    assert restored.path == "runs/r1"
