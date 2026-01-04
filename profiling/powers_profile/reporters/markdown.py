"""Markdown report generation for profiling data.

This module generates human-readable markdown reports
summarizing profiling results.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from ..schemas import ProfilingRun


def generate_markdown_report(
    run: ProfilingRun,
    output_path: Path,
    comparison_run: Optional[ProfilingRun] = None
) -> Path:
    """Generate markdown report for profiling run.
    
    Args:
        run: Profiling run to report on
        output_path: Where to write markdown file
        comparison_run: Optional baseline for comparison
        
    Returns:
        Path to generated markdown file
    """
    lines = []
    
    # Header
    lines.append(f"# Profiling Report: {run.run_id}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Overview section
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Run ID:** `{run.run_id}`")
    lines.append(f"- **Timestamp:** {run.timestamp}")
    lines.append(f"- **Status:** {run.status}")
    lines.append(f"- **Binary:** `{run.binary_path}`")
    lines.append(f"- **Arguments:** `{' '.join(run.binary_args)}`")
    lines.append(f"- **Duration:** {run.total_duration_seconds:.2f}s")
    lines.append(f"- **Collectors:** {', '.join(run.collectors_run)}")
    lines.append("")
    
    # Git info
    if hasattr(run, 'git_info') and run.git_info:
        lines.append("### Git Information")
        lines.append("")
        lines.append(f"- **Branch:** `{run.git_info.branch}`")
        lines.append(f"- **Commit:** `{run.git_info.commit_short}` ({run.git_info.commit_sha})")
        lines.append(f"- **Dirty:** {'Yes' if run.git_info.is_dirty else 'No'}")
        if run.git_info.commit_message:
            lines.append(f"- **Message:** {run.git_info.commit_message}")
        lines.append("")
    
    # System info
    if hasattr(run, 'system_info') and run.system_info:
        lines.append("### System Information")
        lines.append("")
        lines.append(f"- **Hostname:** {run.system_info.hostname}")
        lines.append(f"- **OS:** {run.system_info.os_name}")
        lines.append(f"- **CPU:** {run.system_info.cpu_model}")
        lines.append(f"- **Cores:** {run.system_info.cpu_cores_physical} physical, {run.system_info.cpu_cores_logical} logical")
        lines.append(f"- **RAM:** {run.system_info.ram_total_gb:.1f} GB")
        lines.append("")
    
    # Comparison section
    if comparison_run:
        lines.append("## Comparison")
        lines.append("")
        lines.append(f"**Baseline:** {comparison_run.run_id}")
        lines.append("")
        
        # Duration comparison
        baseline_duration = comparison_run.total_duration_seconds
        current_duration = run.total_duration_seconds
        duration_delta = current_duration - baseline_duration
        duration_pct = (duration_delta / baseline_duration * 100) if baseline_duration > 0 else 0
        
        delta_icon = "🟢" if duration_pct < -5 else "🔴" if duration_pct > 5 else "🟡"
        lines.append(f"- **Duration:** {baseline_duration:.2f}s → {current_duration:.2f}s ({duration_pct:+.1f}%) {delta_icon}")
        
        # Memory comparison (if available)
        baseline_rss = _get_peak_rss(comparison_run)
        current_rss = _get_peak_rss(run)
        
        if baseline_rss and current_rss:
            memory_delta = current_rss - baseline_rss
            memory_pct = (memory_delta / baseline_rss * 100) if baseline_rss > 0 else 0
            mem_icon = "🟢" if memory_pct < -5 else "🔴" if memory_pct > 5 else "🟡"
            lines.append(f"- **Peak RSS:** {baseline_rss:.1f} MB → {current_rss:.1f} MB ({memory_pct:+.1f}%) {mem_icon}")
        
        lines.append("")
    
    # Timing section
    if "timing" in run.results:
        timing_result = run.results["timing"]
        if timing_result.success:
            lines.extend(_format_timing_section(timing_result))
    
    # Memory section
    if "memory" in run.results or "rss" in run.results:
        lines.extend(_format_memory_section(run))
    
    # CPU section
    if "cpu" in run.results:
        cpu_result = run.results["cpu"]
        if cpu_result.success:
            lines.extend(_format_cpu_section(cpu_result))
    
    # Parallel section
    if "parallel" in run.results:
        parallel_result = run.results["parallel"]
        if parallel_result.success:
            lines.extend(_format_parallel_section(parallel_result))
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_path


def _get_peak_rss(run: ProfilingRun) -> Optional[float]:
    """Extract peak RSS from run results."""
    if "rss" not in run.results:
        return None
    
    rss_result = run.results["rss"]
    if not rss_result.success:
        return None
    
    rss_data = rss_result.data if hasattr(rss_result, 'data') else {}
    return rss_data.get("peak_rss_mb")


def _format_timing_section(timing_result) -> list:
    """Format timing section of report."""
    lines = ["## Timing Analysis", ""]
    
    data = timing_result.data if hasattr(timing_result, 'data') else {}
    timings = data.get("timings_seconds", {})
    
    if timings:
        lines.append("| Phase | Duration |")
        lines.append("|-------|----------|")
        
        for phase, duration in timings.items():
            lines.append(f"| {phase} | {duration:.3f}s |")
        
        lines.append("")
        
        total = sum(timings.values())
        lines.append(f"**Total:** {total:.3f}s")
        lines.append("")
    
    return lines


def _format_memory_section(run: ProfilingRun) -> list:
    """Format memory section of report."""
    lines = ["## Memory Analysis", ""]
    
    # RSS metrics
    if "rss" in run.results:
        rss_result = run.results["rss"]
        if rss_result.success:
            rss_data = rss_result.data if hasattr(rss_result, 'data') else {}
            
            if rss_data:
                lines.append("### RSS (Resident Set Size)")
                lines.append("")
                
                if rss_data.get("peak_rss_mb"):
                    lines.append(f"- **Peak:** {rss_data['peak_rss_mb']:.1f} MB")
                if rss_data.get("mean_rss_mb"):
                    lines.append(f"- **Mean:** {rss_data['mean_rss_mb']:.1f} MB")
                if rss_data.get("final_rss_mb"):
                    lines.append(f"- **Final:** {rss_data['final_rss_mb']:.1f} MB")
                if rss_data.get("min_rss_mb"):
                    lines.append(f"- **Min:** {rss_data['min_rss_mb']:.1f} MB")
                if rss_data.get("sample_count"):
                    lines.append(f"- **Samples:** {rss_data['sample_count']}")
                
                lines.append("")
    
    # DHAT metrics
    if "memory" in run.results:
        memory_result = run.results["memory"]
        if memory_result.success:
            mem_data = memory_result.data if hasattr(memory_result, 'data') else {}
            metrics = mem_data.get("metrics", {})
            
            if "dhat" in metrics:
                dhat = metrics["dhat"]
                lines.append("### DHAT (Heap Allocation)")
                lines.append("")
                
                if dhat.get("total_bytes"):
                    lines.append(f"- **Total Allocated:** {dhat['total_bytes'] / (1024 * 1024):.1f} MB")
                if dhat.get("total_blocks"):
                    lines.append(f"- **Total Blocks:** {dhat['total_blocks']:,}")
                if dhat.get("max_bytes"):
                    lines.append(f"- **Peak Heap:** {dhat['max_bytes'] / (1024 * 1024):.1f} MB")
                if dhat.get("max_blocks"):
                    lines.append(f"- **Peak Blocks:** {dhat['max_blocks']:,}")
                
                lines.append("")
            
            if "massif" in metrics:
                massif = metrics["massif"]
                lines.append("### Massif (Heap Timeline)")
                lines.append("")
                
                if massif.get("peak_bytes"):
                    lines.append(f"- **Peak Heap:** {massif['peak_bytes'] / (1024 * 1024):.1f} MB")
                if massif.get("peak_snapshot"):
                    lines.append(f"- **Peak Snapshot:** #{massif['peak_snapshot']}")
                
                lines.append("")
    
    return lines


def _format_cpu_section(cpu_result) -> list:
    """Format CPU section of report."""
    lines = ["## CPU Analysis", ""]
    
    data = cpu_result.data if hasattr(cpu_result, 'data') else {}
    hotspots = data.get("hotspots", [])
    
    if hotspots:
        lines.append("### Top Hotspots")
        lines.append("")
        lines.append("| Function | Percent |")
        lines.append("|----------|---------|")
        
        for hotspot in hotspots[:10]:  # Top 10
            symbol = hotspot.get("symbol", "unknown")
            percent = hotspot.get("percent", 0)
            
            # Truncate long function names
            if len(symbol) > 60:
                symbol = symbol[:57] + "..."
            
            lines.append(f"| `{symbol}` | {percent:.2f}% |")
        
        lines.append("")
    
    # Link to flamegraph if available
    lines.append("### FlameGraph")
    lines.append("")
    lines.append("See `flamegraph.svg` for interactive CPU profile visualization.")
    lines.append("")
    
    return lines


def _format_parallel_section(parallel_result) -> list:
    """Format parallel scaling section of report."""
    lines = ["## Parallel Scaling Analysis", ""]
    
    data = parallel_result.data if hasattr(parallel_result, 'data') else {}
    speedup_metrics = data.get("speedup_metrics", [])
    
    if speedup_metrics:
        lines.append("### Speedup & Efficiency")
        lines.append("")
        lines.append("| Threads | Speedup | Efficiency | Status |")
        lines.append("|---------|---------|------------|--------|")
        
        for m in speedup_metrics:
            threads = m["thread_count"]
            speedup = m["speedup"]
            efficiency = m["efficiency"] * 100
            
            # Status indicator
            if m.get("is_regression"):
                status = "🔴 Regression"
            elif efficiency >= 90:
                status = "🟢 Excellent"
            elif efficiency >= 70:
                status = "🟡 Good"
            elif efficiency >= 50:
                status = "🟠 Fair"
            else:
                status = "🔴 Poor"
            
            lines.append(f"| {threads} | {speedup:.2f}x | {efficiency:.1f}% | {status} |")
        
        lines.append("")
    
    # Amdahl's law
    amdahl = data.get("amdahl_estimate", {})
    if amdahl:
        lines.append("### Amdahl's Law Estimate")
        lines.append("")
        
        serial_pct = amdahl.get("serial_fraction", 0) * 100
        parallel_pct = amdahl.get("parallel_fraction", 0) * 100
        confidence = amdahl.get("confidence", "unknown").upper()
        
        lines.append(f"- **Serial Fraction:** {serial_pct:.2f}%")
        lines.append(f"- **Parallel Fraction:** {parallel_pct:.2f}%")
        
        max_speedup = amdahl.get("predicted_max_speedup")
        if max_speedup:
            lines.append(f"- **Predicted Max Speedup:** {max_speedup:.2f}x")
        
        lines.append(f"- **Confidence:** {confidence}")
        lines.append("")
    
    # Bottlenecks
    bottlenecks = data.get("bottlenecks", [])
    if bottlenecks:
        lines.append("### Detected Bottlenecks")
        lines.append("")
        
        for bottleneck in bottlenecks:
            lines.append(f"- ⚠️  {bottleneck}")
        
        lines.append("")
    
    return lines
