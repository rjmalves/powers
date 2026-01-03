"""Memory metrics comparison analyzer.

Compares memory profiling metrics between baseline and target runs,
detecting regressions and improvements in heap usage, cache efficiency,
and physical memory consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas import ProfilingRun


@dataclass
class MetricDelta:
    """Delta for a single numeric metric."""
    
    baseline: float
    target: float
    delta: float  # target - baseline
    percent_change: float  # (delta / baseline) * 100
    improved: bool  # True if lower is better and decreased, or higher is better and increased
    
    @classmethod
    def compute(
        cls,
        baseline: float,
        target: float,
        lower_is_better: bool = True,
    ) -> MetricDelta:
        """
        Compute delta between baseline and target.
        
        Args:
            baseline: Baseline metric value
            target: Target metric value
            lower_is_better: If True, decrease is improvement; if False, increase is improvement
            
        Returns:
            MetricDelta instance
        """
        delta = target - baseline
        
        # Handle zero baseline
        if baseline == 0:
            if target == 0:
                percent_change = 0.0
            else:
                # Arbitrarily large change
                percent_change = 999.9 if delta > 0 else -999.9
        else:
            percent_change = (delta / baseline) * 100.0
        
        # Determine if this is an improvement
        if lower_is_better:
            improved = delta < 0
        else:
            improved = delta > 0
        
        return cls(
            baseline=baseline,
            target=target,
            delta=delta,
            percent_change=percent_change,
            improved=improved,
        )


@dataclass
class MemoryComparison:
    """Comparison results for memory profiling metrics."""
    
    baseline_run_id: str
    target_run_id: str
    
    # RSS metrics
    rss_peak_mb: Optional[MetricDelta] = None
    rss_mean_mb: Optional[MetricDelta] = None
    rss_final_mb: Optional[MetricDelta] = None
    
    # DHAT metrics
    dhat_total_bytes: Optional[MetricDelta] = None
    dhat_total_blocks: Optional[MetricDelta] = None
    dhat_max_bytes: Optional[MetricDelta] = None
    
    # Massif metrics
    massif_peak_mb: Optional[MetricDelta] = None
    massif_peak_heap_mb: Optional[MetricDelta] = None
    massif_growth_mb: Optional[MetricDelta] = None
    
    # Cachegrind metrics
    cachegrind_instructions: Optional[MetricDelta] = None
    cachegrind_i1_miss_rate: Optional[MetricDelta] = None
    cachegrind_d1_miss_rate: Optional[MetricDelta] = None
    cachegrind_dll_miss_rate: Optional[MetricDelta] = None
    
    # Summary flags
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {
            'baseline_run_id': self.baseline_run_id,
            'target_run_id': self.target_run_id,
            'regressions': self.regressions,
            'improvements': self.improvements,
            'warnings': self.warnings,
            'metrics': {},
        }
        
        # Add all metric deltas
        for attr in dir(self):
            if attr.startswith('_') or attr in {'baseline_run_id', 'target_run_id', 'regressions', 'improvements', 'warnings', 'to_dict'}:
                continue
            
            value = getattr(self, attr)
            if isinstance(value, MetricDelta):
                result['metrics'][attr] = {
                    'baseline': value.baseline,
                    'target': value.target,
                    'delta': value.delta,
                    'percent_change': value.percent_change,
                    'improved': value.improved,
                }
        
        return result


def compare_memory_metrics(
    baseline: ProfilingRun,
    target: ProfilingRun,
    regression_threshold_percent: float = 5.0,
    improvement_threshold_percent: float = 5.0,
) -> MemoryComparison:
    """
    Compare memory metrics between baseline and target runs.
    
    Args:
        baseline: Baseline profiling run
        target: Target profiling run
        regression_threshold_percent: Threshold for flagging regressions
        improvement_threshold_percent: Threshold for flagging improvements
        
    Returns:
        MemoryComparison with computed deltas and flags
    """
    comparison = MemoryComparison(
        baseline_run_id=baseline.run_id,
        target_run_id=target.run_id,
    )
    
    # Extract memory results
    baseline_memory = baseline.results.get('memory')
    target_memory = target.results.get('memory')
    
    if not baseline_memory or not target_memory:
        comparison.warnings.append("Memory collector not run in one or both runs")
        return comparison
    
    if not baseline_memory.success or not target_memory.success:
        comparison.warnings.append("Memory collector failed in one or both runs")
        return comparison
    
    baseline_metrics = baseline_memory.data.get('metrics', {})
    target_metrics = target_memory.data.get('metrics', {})
    
    # Compare RSS metrics
    if 'rss' in baseline_metrics and 'rss' in target_metrics:
        baseline_rss = baseline_metrics['rss']
        target_rss = target_metrics['rss']
        
        if 'peak_mb' in baseline_rss and 'peak_mb' in target_rss:
            comparison.rss_peak_mb = MetricDelta.compute(
                baseline_rss['peak_mb'],
                target_rss['peak_mb'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'rss_peak_mb', 'RSS Peak', regression_threshold_percent, improvement_threshold_percent)
        
        if 'mean_mb' in baseline_rss and 'mean_mb' in target_rss:
            comparison.rss_mean_mb = MetricDelta.compute(
                baseline_rss['mean_mb'],
                target_rss['mean_mb'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'rss_mean_mb', 'RSS Mean', regression_threshold_percent, improvement_threshold_percent)
        
        if 'final_mb' in baseline_rss and 'final_mb' in target_rss:
            comparison.rss_final_mb = MetricDelta.compute(
                baseline_rss['final_mb'],
                target_rss['final_mb'],
                lower_is_better=True,
            )
    
    # Compare DHAT metrics
    if 'dhat' in baseline_metrics and 'dhat' in target_metrics:
        baseline_dhat = baseline_metrics['dhat']
        target_dhat = target_metrics['dhat']
        
        if 'total_bytes' in baseline_dhat and 'total_bytes' in target_dhat:
            comparison.dhat_total_bytes = MetricDelta.compute(
                baseline_dhat['total_bytes'],
                target_dhat['total_bytes'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'dhat_total_bytes', 'DHAT Total Bytes', regression_threshold_percent, improvement_threshold_percent)
        
        if 'total_blocks' in baseline_dhat and 'total_blocks' in target_dhat:
            comparison.dhat_total_blocks = MetricDelta.compute(
                baseline_dhat['total_blocks'],
                target_dhat['total_blocks'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'dhat_total_blocks', 'DHAT Total Blocks', regression_threshold_percent, improvement_threshold_percent)
        
        if 'max_bytes' in baseline_dhat and 'max_bytes' in target_dhat:
            comparison.dhat_max_bytes = MetricDelta.compute(
                baseline_dhat['max_bytes'],
                target_dhat['max_bytes'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'dhat_max_bytes', 'DHAT Max Bytes', regression_threshold_percent, improvement_threshold_percent)
    
    # Compare Massif metrics
    if 'massif' in baseline_metrics and 'massif' in target_metrics:
        baseline_massif = baseline_metrics['massif']
        target_massif = target_metrics['massif']
        
        if 'peak_mb' in baseline_massif and 'peak_mb' in target_massif:
            comparison.massif_peak_mb = MetricDelta.compute(
                baseline_massif['peak_mb'],
                target_massif['peak_mb'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'massif_peak_mb', 'Massif Peak', regression_threshold_percent, improvement_threshold_percent)
        
        if 'peak_heap_mb' in baseline_massif and 'peak_heap_mb' in target_massif:
            comparison.massif_peak_heap_mb = MetricDelta.compute(
                baseline_massif['peak_heap_mb'],
                target_massif['peak_heap_mb'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'massif_peak_heap_mb', 'Massif Heap Peak', regression_threshold_percent, improvement_threshold_percent)
        
        if 'growth_mb' in baseline_massif and 'growth_mb' in target_massif:
            comparison.massif_growth_mb = MetricDelta.compute(
                baseline_massif['growth_mb'],
                target_massif['growth_mb'],
                lower_is_better=True,
            )
    
    # Compare Cachegrind metrics
    if 'cachegrind' in baseline_metrics and 'cachegrind' in target_metrics:
        baseline_cg = baseline_metrics['cachegrind']
        target_cg = target_metrics['cachegrind']
        
        if 'instructions' in baseline_cg and 'instructions' in target_cg:
            comparison.cachegrind_instructions = MetricDelta.compute(
                baseline_cg['instructions'],
                target_cg['instructions'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'cachegrind_instructions', 'Instructions', regression_threshold_percent, improvement_threshold_percent)
        
        if 'i1_miss_rate' in baseline_cg and 'i1_miss_rate' in target_cg:
            comparison.cachegrind_i1_miss_rate = MetricDelta.compute(
                baseline_cg['i1_miss_rate'],
                target_cg['i1_miss_rate'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'cachegrind_i1_miss_rate', 'L1 I-Cache Miss Rate', regression_threshold_percent, improvement_threshold_percent)
        
        if 'overall_d1_miss_rate' in baseline_cg and 'overall_d1_miss_rate' in target_cg:
            comparison.cachegrind_d1_miss_rate = MetricDelta.compute(
                baseline_cg['overall_d1_miss_rate'],
                target_cg['overall_d1_miss_rate'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'cachegrind_d1_miss_rate', 'L1 D-Cache Miss Rate', regression_threshold_percent, improvement_threshold_percent)
        
        if 'overall_dll_miss_rate' in baseline_cg and 'overall_dll_miss_rate' in target_cg:
            comparison.cachegrind_dll_miss_rate = MetricDelta.compute(
                baseline_cg['overall_dll_miss_rate'],
                target_cg['overall_dll_miss_rate'],
                lower_is_better=True,
            )
            _check_threshold(comparison, 'cachegrind_dll_miss_rate', 'LL Cache Miss Rate', regression_threshold_percent, improvement_threshold_percent)
    
    return comparison


def _check_threshold(
    comparison: MemoryComparison,
    attr_name: str,
    display_name: str,
    regression_threshold: float,
    improvement_threshold: float,
) -> None:
    """Check if metric delta exceeds thresholds and update flags."""
    delta = getattr(comparison, attr_name)
    if delta is None:
        return
    
    abs_percent = abs(delta.percent_change)
    
    if not delta.improved and abs_percent >= regression_threshold:
        comparison.regressions.append(
            f"{display_name}: {delta.percent_change:+.1f}% (baseline: {delta.baseline:.2f}, target: {delta.target:.2f})"
        )
    elif delta.improved and abs_percent >= improvement_threshold:
        comparison.improvements.append(
            f"{display_name}: {delta.percent_change:+.1f}% (baseline: {delta.baseline:.2f}, target: {delta.target:.2f})"
        )


def format_comparison_markdown(comparison: MemoryComparison) -> str:
    """
    Format memory comparison as markdown for CLI display.
    
    Args:
        comparison: Memory comparison results
        
    Returns:
        Formatted markdown string
    """
    lines = [
        "# Memory Comparison",
        "",
        f"**Baseline**: {comparison.baseline_run_id}",
        f"**Target**: {comparison.target_run_id}",
        "",
    ]
    
    # Regressions
    if comparison.regressions:
        lines.append("## 🔴 Regressions")
        lines.append("")
        for regression in comparison.regressions:
            lines.append(f"- {regression}")
        lines.append("")
    
    # Improvements
    if comparison.improvements:
        lines.append("## 🟢 Improvements")
        lines.append("")
        for improvement in comparison.improvements:
            lines.append(f"- {improvement}")
        lines.append("")
    
    # Warnings
    if comparison.warnings:
        lines.append("## ⚠️ Warnings")
        lines.append("")
        for warning in comparison.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    
    # Detailed metrics
    lines.append("## Detailed Metrics")
    lines.append("")
    
    # RSS
    if comparison.rss_peak_mb or comparison.rss_mean_mb or comparison.rss_final_mb:
        lines.append("### RSS (Physical Memory)")
        lines.append("")
        lines.append("| Metric | Baseline | Target | Delta | Change |")
        lines.append("|--------|----------|--------|-------|--------|")
        
        if comparison.rss_peak_mb:
            _add_metric_row(lines, "Peak", comparison.rss_peak_mb, "MB")
        if comparison.rss_mean_mb:
            _add_metric_row(lines, "Mean", comparison.rss_mean_mb, "MB")
        if comparison.rss_final_mb:
            _add_metric_row(lines, "Final", comparison.rss_final_mb, "MB")
        lines.append("")
    
    # DHAT
    if comparison.dhat_total_bytes or comparison.dhat_total_blocks or comparison.dhat_max_bytes:
        lines.append("### DHAT (Heap Allocations)")
        lines.append("")
        lines.append("| Metric | Baseline | Target | Delta | Change |")
        lines.append("|--------|----------|--------|-------|--------|")
        
        if comparison.dhat_total_bytes:
            _add_metric_row(lines, "Total Bytes", comparison.dhat_total_bytes, "B")
        if comparison.dhat_total_blocks:
            _add_metric_row(lines, "Total Blocks", comparison.dhat_total_blocks, "")
        if comparison.dhat_max_bytes:
            _add_metric_row(lines, "Max Bytes", comparison.dhat_max_bytes, "B")
        lines.append("")
    
    # Massif
    if comparison.massif_peak_mb or comparison.massif_peak_heap_mb or comparison.massif_growth_mb:
        lines.append("### Massif (Heap Timeline)")
        lines.append("")
        lines.append("| Metric | Baseline | Target | Delta | Change |")
        lines.append("|--------|----------|--------|-------|--------|")
        
        if comparison.massif_peak_mb:
            _add_metric_row(lines, "Peak Total", comparison.massif_peak_mb, "MB")
        if comparison.massif_peak_heap_mb:
            _add_metric_row(lines, "Peak Heap", comparison.massif_peak_heap_mb, "MB")
        if comparison.massif_growth_mb:
            _add_metric_row(lines, "Growth", comparison.massif_growth_mb, "MB")
        lines.append("")
    
    # Cachegrind
    if any([comparison.cachegrind_instructions, comparison.cachegrind_i1_miss_rate,
            comparison.cachegrind_d1_miss_rate, comparison.cachegrind_dll_miss_rate]):
        lines.append("### Cachegrind (Cache Efficiency)")
        lines.append("")
        lines.append("| Metric | Baseline | Target | Delta | Change |")
        lines.append("|--------|----------|--------|-------|--------|")
        
        if comparison.cachegrind_instructions:
            _add_metric_row(lines, "Instructions", comparison.cachegrind_instructions, "")
        if comparison.cachegrind_i1_miss_rate:
            _add_metric_row(lines, "L1 I-Miss Rate", comparison.cachegrind_i1_miss_rate, "%")
        if comparison.cachegrind_d1_miss_rate:
            _add_metric_row(lines, "L1 D-Miss Rate", comparison.cachegrind_d1_miss_rate, "%")
        if comparison.cachegrind_dll_miss_rate:
            _add_metric_row(lines, "LL Miss Rate", comparison.cachegrind_dll_miss_rate, "%")
        lines.append("")
    
    return "\n".join(lines)


def _add_metric_row(lines: List[str], name: str, delta: MetricDelta, unit: str) -> None:
    """Add a metric row to the markdown table."""
    icon = "🟢" if delta.improved else "🔴"
    sign = "+" if delta.delta >= 0 else ""
    
    lines.append(
        f"| {name} | {delta.baseline:.2f} {unit} | {delta.target:.2f} {unit} | "
        f"{sign}{delta.delta:.2f} {unit} | {icon} {delta.percent_change:+.1f}% |"
    )
