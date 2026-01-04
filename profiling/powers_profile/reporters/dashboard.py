"""Dashboard generation for interactive profiling visualizations.

This module creates self-contained HTML dashboards using Plotly
for exploring profiling data interactively.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..schemas import ProfilingRun


class DashboardGenerator:
    """Generates interactive Plotly dashboards from profiling data."""

    def __init__(self, offline: bool = True, theme: str = "plotly_white"):
        """Initialize dashboard generator.

        Args:
            offline: If True, embed plotly.js for offline viewing
            theme: Plotly template name ('plotly', 'plotly_white', 'plotly_dark', etc.)
        """
        self.offline = offline
        self.theme = theme

    def generate(
        self,
        run: ProfilingRun,
        output_path: Path,
        comparison_run: Optional[ProfilingRun] = None,
    ) -> Path:
        """Generate complete dashboard HTML.

        Args:
            run: Primary profiling run to visualize
            output_path: Path where HTML should be written
            comparison_run: Optional baseline run for comparison view

        Returns:
            Path to generated HTML file
        """
        # Create figures and tables for each domain
        # Format: list of (tab_name, content, content_type) where content_type is 'plotly' or 'datatable'
        sections = []

        # Summary section at top
        summary_fig = self._create_summary_cards(run, comparison_run)
        if summary_fig:
            sections.append(("Summary", summary_fig, "datatable"))

        # Timing visualizations - keep charts, add DataTable
        if "timing" in run.results or "timing" in run.collectors_run:
            timing_fig = self._create_timing_charts(run)
            if timing_fig:
                sections.append(("Timing Charts", timing_fig, "plotly"))

            timing_table = self._create_timing_table(run)
            if timing_table:
                sections.append(("Timing Breakdown", timing_table, "datatable"))

        # Memory visualizations - charts only, separate DataTable tab
        if "memory" in run.results or any(
            c in run.collectors_run for c in ["memory", "rss", "dhat"]
        ):
            memory_fig = self._create_memory_charts(run)
            if memory_fig:
                sections.append(("Memory Charts", memory_fig, "plotly"))

            # Add memory summary DataTable (replaces buggy Plotly table)
            memory_table = self._create_memory_summary_table(run)
            if memory_table:
                sections.append(("Memory Summary", memory_table, "datatable"))

        # Scaling visualizations
        if "parallel" in run.results or "parallel" in run.collectors_run:
            scaling_fig = self._create_scaling_charts(run)
            if scaling_fig:
                sections.append(("Parallel Scaling", scaling_fig, "plotly"))

        # CPU visualizations
        if "cpu" in run.results or "cpu" in run.collectors_run:
            # FlameGraph (if available)
            flamegraph = self._create_flamegraph_section(run)
            if flamegraph:
                sections.append(("FlameGraph", flamegraph, "datatable"))

            # CPU Hotspots DataTable
            cpu_table = self._create_cpu_section(run)
            if cpu_table:
                sections.append(("CPU Hotspots", cpu_table, "datatable"))

        # Comparison view
        if comparison_run:
            comparison_fig = self._create_comparison_view(run, comparison_run)
            if comparison_fig:
                sections.append(("Comparison", comparison_fig, "plotly"))

        # Build complete HTML with tabs
        html_content = self._build_html(run, sections, comparison_run)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _create_summary_cards(
        self, run: ProfilingRun, comparison_run: Optional[ProfilingRun] = None
    ) -> Optional[str]:
        """Create summary metrics table with DataTables.

        Returns HTML string with DataTables.js table.
        """
        rows_html = ""

        # Total duration
        duration_str = f"{run.total_duration_seconds:.2f}s"
        if comparison_run:
            baseline_duration = f"{comparison_run.total_duration_seconds:.2f}s"
            delta = (
                (
                    run.total_duration_seconds
                    - comparison_run.total_duration_seconds
                )
                / comparison_run.total_duration_seconds
                * 100
            )
            delta_str = f"{delta:+.1f}%"
            rows_html += f"<tr><td>Total Duration</td><td>{duration_str}</td><td>{baseline_duration}</td><td>{delta_str}</td></tr>"
        else:
            rows_html += (
                f"<tr><td>Total Duration</td><td>{duration_str}</td></tr>"
            )

        # Collectors run
        collectors_str = ", ".join(run.collectors_run)
        if comparison_run:
            baseline_collectors = ", ".join(comparison_run.collectors_run)
            rows_html += f"<tr><td>Collectors</td><td>{collectors_str}</td><td>{baseline_collectors}</td><td>-</td></tr>"
        else:
            rows_html += (
                f"<tr><td>Collectors</td><td>{collectors_str}</td></tr>"
            )

        # Status
        if comparison_run:
            rows_html += f"<tr><td>Status</td><td>{run.status}</td><td>{comparison_run.status}</td><td>-</td></tr>"
        else:
            rows_html += f"<tr><td>Status</td><td>{run.status}</td></tr>"

        # Build table HTML
        if comparison_run:
            thead = "<tr><th>Metric</th><th>Value</th><th>Baseline</th><th>Change</th></tr>"
        else:
            thead = "<tr><th>Metric</th><th>Value</th></tr>"

        html = f"""
        <div class="summary-section" style="padding: 20px;">
            <h2 style="color: #667eea; margin-bottom: 20px;">Run Summary</h2>
            <table id="summary-table" class="display compact" style="width:100%">
                <thead>
                    {thead}
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

        return html

    def _create_timing_charts(self, run: ProfilingRun) -> Optional[go.Figure]:
        """Create timing visualization charts.

        Shows timing breakdown and phase analysis if available.
        Excludes 'total_running' from breakdown as it represents the sum.
        """
        timing_data = run.results.get("timing")
        if not timing_data or not timing_data.success:
            return None

        data_dict = timing_data.data if hasattr(timing_data, "data") else {}
        if not data_dict:
            return None

        # Create subplots for timing breakdown
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Phase Duration", "Duration Breakdown"),
            specs=[[{"type": "bar"}, {"type": "pie"}]],
        )

        # Extract timing metrics - exclude total_running as it's the sum
        all_timings = data_dict.get("timings_seconds", {})
        if all_timings:
            # Filter out total/sum entries for the breakdown
            timings = {
                k: v
                for k, v in all_timings.items()
                if not any(
                    keyword in k.lower()
                    for keyword in ["total", "sum", "overall"]
                )
            }

            if timings:
                phases = list(timings.keys())
                durations = [timings[p] for p in phases]

                # Bar chart
                fig.add_trace(
                    go.Bar(
                        x=phases,
                        y=durations,
                        name="Duration",
                        marker_color="rgb(102, 126, 234)",
                        text=[f"{d:.3f}s" for d in durations],
                        textposition="auto",
                    ),
                    row=1,
                    col=1,
                )

                # Pie chart
                fig.add_trace(
                    go.Pie(
                        labels=phases,
                        values=durations,
                        name="Breakdown",
                        hole=0.3,
                    ),
                    row=1,
                    col=2,
                )

        fig.update_xaxes(title_text="Phase", row=1, col=1)
        fig.update_yaxes(title_text="Duration (s)", row=1, col=1)

        fig.update_layout(
            title="Timing Analysis",
            height=500,
            template=self.theme,
            showlegend=True,
            hovermode="closest",
        )

        return fig

    def _create_memory_charts(self, run: ProfilingRun) -> Optional[go.Figure]:
        """Create memory visualization charts.

        Shows RSS timeline, peak memory, and heap metrics if available.
        NOTE: Memory summary table moved to DataTables (separate tab).
        """
        # Try to get memory data from different collectors
        memory_result = run.results.get("memory")
        rss_result = run.results.get("rss")

        if not memory_result and not rss_result:
            return None

        # Check if we have DHAT data
        has_dhat = False
        if memory_result and memory_result.success:
            mem_data = (
                memory_result.data if hasattr(memory_result, "data") else {}
            )
            metrics = mem_data.get("metrics", {})
            has_dhat = "dhat" in metrics and bool(metrics["dhat"])

        # Create subplots - only include Heap Metrics if DHAT data is available
        if has_dhat:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "RSS Over Time",
                    "Peak Memory Usage",
                    "Heap Metrics",
                    "",
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "bar"}],
                    [
                        {"type": "bar"},
                        {"type": "xy"},
                    ],  # Empty subplot for balance
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.20,  # Increased to prevent y-axis label overlap
            )
        else:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("RSS Over Time", "Peak Memory Usage"),
                specs=[[{"type": "scatter"}, {"type": "bar"}]],
                horizontal_spacing=0.20,  # Increased to prevent y-axis label overlap
            )

        # RSS timeline
        if rss_result and rss_result.success:
            rss_data = rss_result.data if hasattr(rss_result, "data") else {}

            # Try to load samples from raw file
            samples = []
            if rss_result.raw_files:
                for raw_file in rss_result.raw_files:
                    if raw_file.endswith("rss_data.json"):
                        try:
                            import json

                            with open(raw_file, "r") as f:
                                rss_json = json.load(f)
                                samples = rss_json.get("samples", [])
                                break
                        except Exception:
                            pass

            if samples:
                # Convert timestamps to relative seconds from start
                if samples:
                    start_time = samples[0].get("wall_time", 0)
                    timestamps = [
                        (s.get("wall_time", i) - start_time)
                        for i, s in enumerate(samples)
                    ]
                else:
                    timestamps = list(range(len(samples)))

                rss_values = [
                    s.get("rss_kb", 0) / 1024.0 for s in samples
                ]  # Convert to MB

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=rss_values,
                        mode="lines",
                        name="RSS",
                        line=dict(color="rgb(118, 75, 162)", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(118, 75, 162, 0.2)",
                    ),
                    row=1,
                    col=1,
                )

        # Peak memory comparison
        memory_types = []
        memory_values = []

        if rss_result and rss_result.success:
            rss_data = rss_result.data if hasattr(rss_result, "data") else {}
            if rss_data.get("peak_rss_mb"):
                memory_types.append("RSS Peak")
                memory_values.append(rss_data["peak_rss_mb"])

        if memory_result and memory_result.success:
            mem_data = (
                memory_result.data if hasattr(memory_result, "data") else {}
            )
            metrics = mem_data.get("metrics", {})

            if "dhat" in metrics:
                dhat = metrics["dhat"]
                if dhat.get("total_bytes"):
                    memory_types.append("DHAT Total")
                    memory_values.append(dhat["total_bytes"] / (1024 * 1024))
                if dhat.get("max_bytes"):
                    memory_types.append("DHAT Peak")
                    memory_values.append(dhat["max_bytes"] / (1024 * 1024))

            if "massif" in metrics:
                massif = metrics["massif"]
                if massif.get("peak_bytes"):
                    memory_types.append("Massif Peak")
                    memory_values.append(massif["peak_bytes"] / (1024 * 1024))

        if memory_types:
            fig.add_trace(
                go.Bar(
                    x=memory_types,
                    y=memory_values,
                    name="Memory",
                    marker_color=["#667eea", "#764ba2", "#f093fb", "#4facfe"],
                    text=[f"{v:.1f} MB" for v in memory_values],
                    textposition="auto",
                ),
                row=1,
                col=2,
            )

        # Heap metrics (DHAT) - only if data is available
        if has_dhat:
            mem_data = (
                memory_result.data if hasattr(memory_result, "data") else {}
            )
            metrics = mem_data.get("metrics", {})
            dhat = metrics["dhat"]

            heap_metrics = []
            heap_values = []

            if dhat.get("total_blocks"):
                heap_metrics.append("Total Blocks")
                heap_values.append(dhat["total_blocks"])

            if dhat.get("total_bytes"):
                heap_metrics.append("Total Bytes")
                heap_values.append(dhat["total_bytes"])

            if dhat.get("max_blocks"):
                heap_metrics.append("Max Blocks")
                heap_values.append(dhat["max_blocks"])

            if heap_metrics:
                fig.add_trace(
                    go.Bar(
                        x=heap_metrics,
                        y=heap_values,
                        name="Heap",
                        marker_color="rgb(102, 126, 234)",
                    ),
                    row=2,
                    col=1,
                )

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="RSS (MB)", row=1, col=1)

        fig.update_xaxes(title_text="Type", row=1, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)

        if has_dhat:
            fig.update_xaxes(title_text="Metric", row=2, col=1)
            fig.update_yaxes(title_text="Count/Bytes", row=2, col=1)

        fig.update_layout(
            title="Memory Analysis",
            height=600 if not has_dhat else 800,
            template=self.theme,
            showlegend=True,
            hovermode="closest",
            margin=dict(
                l=80, r=40, t=80, b=60
            ),  # Increased left margin to prevent y-axis label overlap
        )

        return fig

    def _create_scaling_charts(self, run: ProfilingRun) -> Optional[go.Figure]:
        """Create scaling visualization charts.

        Shows speedup curves, efficiency, and Amdahl analysis.
        """
        parallel_result = run.results.get("parallel")
        if not parallel_result or not parallel_result.success:
            return None

        parallel_data = (
            parallel_result.data if hasattr(parallel_result, "data") else {}
        )
        speedup_metrics = parallel_data.get("speedup_metrics", [])

        if not speedup_metrics:
            return None

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Speedup Curve",
                "Efficiency vs Thread Count",
                "Scaling Metrics Table",
                "Amdahl's Law",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "table"}, {"type": "indicator"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # Extract data
        thread_counts = [m["thread_count"] for m in speedup_metrics]
        speedups = [m["speedup"] for m in speedup_metrics]
        efficiencies = [m["efficiency"] * 100 for m in speedup_metrics]

        # Speedup curve with ideal line
        fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=speedups,
                mode="lines+markers",
                name="Actual Speedup",
                line=dict(color="rgb(102, 126, 234)", width=3),
                marker=dict(size=10),
            ),
            row=1,
            col=1,
        )

        # Ideal speedup line
        max_threads = max(thread_counts)
        fig.add_trace(
            go.Scatter(
                x=[1, max_threads],
                y=[1, max_threads],
                mode="lines",
                name="Ideal Speedup",
                line=dict(color="gray", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Efficiency curve
        fig.add_trace(
            go.Scatter(
                x=thread_counts,
                y=efficiencies,
                mode="lines+markers",
                name="Efficiency",
                line=dict(color="rgb(118, 75, 162)", width=3),
                marker=dict(size=10),
                fill="tozeroy",
                fillcolor="rgba(118, 75, 162, 0.2)",
            ),
            row=1,
            col=2,
        )

        # Add 100% efficiency reference line
        fig.add_trace(
            go.Scatter(
                x=[min(thread_counts), max(thread_counts)],
                y=[100, 100],
                mode="lines",
                name="100% Efficiency",
                line=dict(color="green", width=1, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Metrics table
        table_headers = ["Threads", "Speedup", "Efficiency", "Status"]
        table_data = []

        for m in speedup_metrics:
            threads = m["thread_count"]
            speedup = f"{m['speedup']:.2f}x"
            efficiency = f"{m['efficiency'] * 100:.1f}%"

            # Status indicator
            if m.get("is_regression"):
                status = "🔴 Regression"
            elif m["efficiency"] >= 0.9:
                status = "🟢 Excellent"
            elif m["efficiency"] >= 0.7:
                status = "🟡 Good"
            elif m["efficiency"] >= 0.5:
                status = "🟠 Fair"
            else:
                status = "🔴 Poor"

            table_data.append([str(threads), speedup, efficiency, status])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_headers,
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color="lavender",
                    align="left",
                    font=dict(size=11),
                ),
            ),
            row=2,
            col=1,
        )

        # Amdahl's law estimate
        amdahl = parallel_data.get("amdahl_estimate", {})
        if amdahl:
            serial_fraction = amdahl.get("serial_fraction", 0) * 100
            confidence = amdahl.get("confidence", "unknown").upper()

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=serial_fraction,
                    title={
                        "text": f"Serial Fraction<br><span style='font-size:0.8em'>{confidence} Confidence</span>"
                    },
                    delta={
                        "reference": 0,
                        "increasing": {"color": "red"},
                        "decreasing": {"color": "green"},
                    },
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 10], "color": "lightgreen"},
                            {"range": [10, 30], "color": "yellow"},
                            {"range": [30, 100], "color": "lightcoral"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                    number={"suffix": "%"},
                ),
                row=2,
                col=2,
            )

        fig.update_xaxes(title_text="Thread Count", row=1, col=1)
        fig.update_yaxes(title_text="Speedup", row=1, col=1)

        fig.update_xaxes(title_text="Thread Count", row=1, col=2)
        fig.update_yaxes(
            title_text="Efficiency (%)", row=1, col=2, range=[0, 110]
        )

        fig.update_layout(
            title="Parallel Scaling Analysis",
            height=800,
            template=self.theme,
            showlegend=True,
            hovermode="closest",
        )

        return fig

    def _create_timing_table(self, run: ProfilingRun) -> Optional[str]:
        """Create timing breakdown table with DataTables.

        Returns HTML string with DataTables.js table.
        """
        timing_data = run.results.get("timing")
        if not timing_data or not timing_data.success:
            return None

        data_dict = timing_data.data if hasattr(timing_data, "data") else {}
        all_timings = data_dict.get("timings_seconds", {})

        if not all_timings:
            return None

        # Filter out total/sum entries
        timings = {
            k: v
            for k, v in all_timings.items()
            if not any(
                keyword in k.lower() for keyword in ["total", "sum", "overall"]
            )
        }

        if not timings:
            return None

        # Sort by duration descending
        sorted_timings = sorted(
            timings.items(), key=lambda x: x[1], reverse=True
        )

        rows_html = ""
        for phase, duration in sorted_timings:
            rows_html += f"""
                <tr>
                    <td>{phase}</td>
                    <td>{duration:.6f}</td>
                </tr>
            """

        html = f"""
        <div class="timing-section" style="padding: 20px;">
            <h2 style="color: #667eea; margin-bottom: 20px;">Timing Breakdown</h2>
            <table id="timing-table" class="display compact" style="width:100%">
                <thead>
                    <tr>
                        <th>Phase</th>
                        <th>Duration (s)</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

        return html

    def _create_memory_summary_table(self, run: ProfilingRun) -> Optional[str]:
        """Create memory summary table with DataTables.

        Returns HTML string with DataTables.js table.
        """
        rss_result = run.results.get("rss")
        if not rss_result or not rss_result.success:
            return None

        rss_data = rss_result.data if hasattr(rss_result, "data") else {}

        rows_html = ""
        if rss_data.get("peak_rss_mb"):
            rows_html += f"<tr><td>RSS Peak</td><td>{rss_data['peak_rss_mb']:.1f} MB</td></tr>"
        if rss_data.get("mean_rss_mb"):
            rows_html += f"<tr><td>RSS Mean</td><td>{rss_data['mean_rss_mb']:.1f} MB</td></tr>"
        if rss_data.get("min_rss_mb"):
            rows_html += f"<tr><td>RSS Min</td><td>{rss_data['min_rss_mb']:.1f} MB</td></tr>"
        if rss_data.get("final_rss_mb"):
            rows_html += f"<tr><td>RSS Final</td><td>{rss_data['final_rss_mb']:.1f} MB</td></tr>"

        if not rows_html:
            return None

        html = f"""
        <div class="memory-summary" style="padding: 20px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">Memory Summary</h3>
            <table id="memory-summary-table" class="display compact" style="width:100%">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

        return html

    def _create_flamegraph_section(self, run: ProfilingRun) -> Optional[str]:
        """Create FlameGraph section by embedding the SVG.

        Returns HTML string with embedded FlameGraph SVG.
        """
        cpu_result = run.results.get("cpu")
        if not cpu_result or not cpu_result.success:
            return None

        # Find flamegraph SVG in raw files
        flamegraph_path = None
        if cpu_result.raw_files:
            for raw_file in cpu_result.raw_files:
                if raw_file.endswith("flamegraph.svg"):
                    flamegraph_path = Path(raw_file)
                    break

        if not flamegraph_path or not flamegraph_path.exists():
            return None

        # Read SVG content
        try:
            svg_content = flamegraph_path.read_text()
        except Exception as e:
            return f"<p>Error loading FlameGraph: {e}</p>"

        html = f"""
        <div class="flamegraph-section" style="padding: 20px;">
            <h2 style="color: #667eea; margin-bottom: 20px;">CPU FlameGraph</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Interactive visualization of call stacks. Hover over boxes to see function names and percentages.
            </p>
            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: white; overflow-x: auto; width: 100%;">
                <div style="width: 100%;">
                    {svg_content}
                </div>
            </div>
        </div>
        """

        return html

    def _create_cpu_section(self, run: ProfilingRun) -> Optional[str]:
        """Create CPU profiling section with hotspot DataTable.

        Returns HTML string with DataTables.js table instead of Plotly figure.
        """
        cpu_result = run.results.get("cpu")
        if not cpu_result or not cpu_result.success:
            return None

        cpu_data = cpu_result.data if hasattr(cpu_result, "data") else {}
        hotspots = cpu_data.get("hotspots", [])

        if not hotspots:
            return None

        # Generate DataTables HTML
        rows_html = ""
        for hotspot in hotspots[:50]:  # Top 50 for better analysis
            symbol = hotspot.get("symbol", "unknown")
            percent = hotspot.get("percent", 0)
            rows_html += f"""
                <tr>
                    <td>{symbol}</td>
                    <td>{percent:.2f}%</td>
                </tr>
            """

        html = f"""
        <div class="cpu-section" style="padding: 20px;">
            <h2 style="color: #667eea; margin-bottom: 20px;">CPU Hotspots (Top 50)</h2>
            <table id="cpu-hotspots-table" class="display compact" style="width:100%">
                <thead>
                    <tr>
                        <th>Function</th>
                        <th>CPU %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

        return html

    def _create_comparison_view(
        self, run: ProfilingRun, comparison_run: ProfilingRun
    ) -> Optional[go.Figure]:
        """Create comparison visualization.

        Shows side-by-side metrics with deltas and before/after charts.
        """
        # Create comprehensive comparison view
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Duration Comparison",
                "Memory Comparison",
                "Delta Summary",
                "Performance Changes",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "bar"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # Duration comparison
        baseline_duration = comparison_run.total_duration_seconds
        current_duration = run.total_duration_seconds

        fig.add_trace(
            go.Bar(
                x=["Baseline", "Current"],
                y=[baseline_duration, current_duration],
                name="Duration",
                marker_color=["#764ba2", "#667eea"],
                text=[f"{baseline_duration:.2f}s", f"{current_duration:.2f}s"],
                textposition="auto",
            ),
            row=1,
            col=1,
        )

        # Memory comparison (if available)
        baseline_rss = None
        current_rss = None

        if "rss" in comparison_run.results:
            baseline_rss_result = comparison_run.results["rss"]
            if baseline_rss_result.success:
                baseline_rss_data = (
                    baseline_rss_result.data
                    if hasattr(baseline_rss_result, "data")
                    else {}
                )
                summary = baseline_rss_data.get("summary", {})
                baseline_rss = summary.get("peak_mb", 0)

        if "rss" in run.results:
            current_rss_result = run.results["rss"]
            if current_rss_result.success:
                current_rss_data = (
                    current_rss_result.data
                    if hasattr(current_rss_result, "data")
                    else {}
                )
                summary = current_rss_data.get("summary", {})
                current_rss = summary.get("peak_mb", 0)

        if baseline_rss and current_rss:
            fig.add_trace(
                go.Bar(
                    x=["Baseline", "Current"],
                    y=[baseline_rss, current_rss],
                    name="Peak RSS",
                    marker_color=["#764ba2", "#667eea"],
                    text=[f"{baseline_rss:.1f} MB", f"{current_rss:.1f} MB"],
                    textposition="auto",
                ),
                row=1,
                col=2,
            )

        # Delta summary table
        delta_headers = ["Metric", "Baseline", "Current", "Delta", "Change"]
        delta_rows = []

        # Duration delta
        duration_delta = current_duration - baseline_duration
        duration_pct = (
            (duration_delta / baseline_duration * 100)
            if baseline_duration > 0
            else 0
        )
        delta_rows.append([
            "Duration",
            f"{baseline_duration:.2f}s",
            f"{current_duration:.2f}s",
            f"{duration_delta:+.2f}s",
            f"{duration_pct:+.1f}%",
        ])

        # Memory delta
        if baseline_rss and current_rss:
            memory_delta = current_rss - baseline_rss
            memory_pct = (
                (memory_delta / baseline_rss * 100) if baseline_rss > 0 else 0
            )
            delta_rows.append([
                "Peak RSS",
                f"{baseline_rss:.1f} MB",
                f"{current_rss:.1f} MB",
                f"{memory_delta:+.1f} MB",
                f"{memory_pct:+.1f}%",
            ])

        # Color code the cells based on improvement/regression
        cell_colors = []
        for row in delta_rows:
            change_pct = float(row[4].rstrip("%"))
            if change_pct < -5:  # > 5% improvement
                color = "lightgreen"
            elif change_pct > 5:  # > 5% regression
                color = "lightcoral"
            else:
                color = "lightyellow"
            cell_colors.append([color] * len(row))

        fig.add_trace(
            go.Table(
                header=dict(
                    values=delta_headers,
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=list(zip(*delta_rows)),
                    fill_color=[c[0] for c in cell_colors]
                    if cell_colors
                    else "lavender",
                    align="left",
                    font=dict(size=11),
                ),
            ),
            row=2,
            col=1,
        )

        # Performance changes bar
        changes = []
        change_labels = []
        change_colors = []

        if duration_pct:
            changes.append(duration_pct)
            change_labels.append("Duration")
            change_colors.append("red" if duration_pct > 0 else "green")

        if baseline_rss and current_rss and memory_pct:
            changes.append(memory_pct)
            change_labels.append("Memory")
            change_colors.append("red" if memory_pct > 0 else "green")

        if changes:
            fig.add_trace(
                go.Bar(
                    x=change_labels,
                    y=changes,
                    name="Change %",
                    marker_color=change_colors,
                    text=[f"{c:+.1f}%" for c in changes],
                    textposition="auto",
                ),
                row=2,
                col=2,
            )

            # Add zero reference line
            fig.add_hline(
                y=0, line_dash="dash", line_color="gray", row=2, col=2
            )

        fig.update_xaxes(title_text="Run", row=1, col=1)
        fig.update_yaxes(title_text="Duration (s)", row=1, col=1)

        fig.update_xaxes(title_text="Run", row=1, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)

        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_yaxes(title_text="Change (%)", row=2, col=2)

        fig.update_layout(
            title=f"Comparison: {comparison_run.run_id} → {run.run_id}",
            height=800,
            template=self.theme,
            showlegend=True,
            hovermode="closest",
        )

        return fig

    def _build_html(
        self,
        run: ProfilingRun,
        sections: List[tuple],
        comparison_run: Optional[ProfilingRun] = None,
    ) -> str:
        """Build complete HTML with tabs for all sections.

        Args:
            run: Profiling run
            sections: List of (tab_name, content, content_type) tuples
            comparison_run: Optional comparison run

        Returns:
            Complete HTML string
        """
        # Include plotly.js and DataTables.js
        if self.offline:
            plotly_js = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'
        else:
            plotly_js = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

        # DataTables.js dependencies
        datatables_css = """
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
        """

        datatables_js = """
        <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
        <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
        <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.print.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.2.7/pdfmake.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.2.7/vfs_fonts.js"></script>
        """

        # Build header
        header_html = f"""
        <div class="header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; margin-bottom: 20px; border-radius: 10px;">
            <h1 style="margin: 0; font-size: 32px;">POWE.RS Profiling Dashboard</h1>
            <div style="margin-top: 15px; opacity: 0.9;">
                <p style="margin: 5px 0;"><strong>Run ID:</strong> {run.run_id}</p>
                <p style="margin: 5px 0;"><strong>Timestamp:</strong> {run.timestamp}</p>
                <p style="margin: 5px 0;"><strong>Binary:</strong> {run.binary_path}</p>
                <p style="margin: 5px 0;"><strong>Args:</strong> {" ".join(run.binary_args)}</p>
                {f'<p style="margin: 5px 0;"><strong>Git:</strong> {run.git_info.branch}@{run.git_info.commit_short} {"(dirty)" if run.git_info.is_dirty else ""}</p>' if hasattr(run, "git_info") and run.git_info else ""}
            </div>
        </div>
        """

        # Build tabs
        tabs_html = '<div class="tabs" style="margin-bottom: 20px;">'
        tab_buttons = []
        tab_contents = []

        for idx, (tab_name, content, content_type) in enumerate(sections):
            active_class = "active" if idx == 0 else ""
            display_style = "display: block;" if idx == 0 else "display: none;"

            tab_id = f"tab-{tab_name.lower().replace(' ', '-')}"

            # Tab button
            tab_buttons.append(f"""
                <button class="tab-button {active_class}" onclick="openTab(event, '{tab_id}')" 
                        style="background-color: {"#667eea" if idx == 0 else "#f0f0f0"}; 
                               color: {"white" if idx == 0 else "black"}; 
                               padding: 12px 24px; 
                               margin: 0 5px 5px 0; 
                               border: none; 
                               cursor: pointer; 
                               border-radius: 5px 5px 0 0;
                               font-size: 14px;
                               font-weight: 500;
                               transition: all 0.3s;">
                    {tab_name}
                </button>
            """)

            # Tab content based on type
            if content_type == "plotly":
                content_html = content.to_html(
                    include_plotlyjs=False,
                    div_id=f"{tab_id}-plot",
                    config={"responsive": True},
                )
            else:  # datatable
                content_html = content

            tab_contents.append(f'''
                <div id="{tab_id}" class="tab-content" style="{display_style} padding: 20px; border: 1px solid #ddd; border-radius: 0 5px 5px 5px;">
                    {content_html}
                </div>
            ''')

        tabs_html += "".join(tab_buttons) + "</div>"
        tabs_html += "".join(tab_contents)

        # Footer
        footer_html = f"""
        <div class="footer" style="margin-top: 40px; padding: 20px; background-color: #f5f5f5; border-radius: 5px; text-align: center;">
            <p style="margin: 5px; color: #666; font-size: 12px;">
                Generated by powers-profile on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
            <p style="margin: 5px; color: #666; font-size: 12px;">
                Dashboard version: 1.1.0 (with DataTables.js)
            </p>
        </div>
        """

        # JavaScript for tab switching and DataTables initialization
        tab_script = """
        <script>
        function openTab(evt, tabId) {
            // Hide all tab contents
            var contents = document.getElementsByClassName("tab-content");
            for (var i = 0; i < contents.length; i++) {
                contents[i].style.display = "none";
            }
            
            // Remove active class from all buttons
            var buttons = document.getElementsByClassName("tab-button");
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].style.backgroundColor = "#f0f0f0";
                buttons[i].style.color = "black";
                buttons[i].classList.remove("active");
            }
            
            // Show selected tab and activate button
            document.getElementById(tabId).style.display = "block";
            evt.currentTarget.style.backgroundColor = "#667eea";
            evt.currentTarget.style.color = "white";
            evt.currentTarget.classList.add("active");
        }
        
        // Initialize DataTables when document is ready
        $(document).ready(function() {
            // Summary table
            if ($('#summary-table').length) {
                $('#summary-table').DataTable({
                    paging: false,
                    searching: false,
                    info: false,
                    dom: 'Bt',
                    buttons: [
                        'copy',
                        { extend: 'csv', title: 'Run_Summary' },
                        'print'
                    ]
                });
            }
            
            // Timing Breakdown table
            if ($('#timing-table').length) {
                $('#timing-table').DataTable({
                    pageLength: 25,
                    order: [[1, 'desc']],  // Sort by duration descending
                    dom: 'Bfrtip',
                    buttons: [
                        'copy',
                        { extend: 'csv', title: 'Timing_Breakdown' },
                        { extend: 'excel', title: 'Timing_Breakdown' },
                        'print'
                    ],
                    language: {
                        search: "Filter phases:"
                    }
                });
            }
            
            // CPU Hotspots table
            if ($('#cpu-hotspots-table').length) {
                $('#cpu-hotspots-table').DataTable({
                    pageLength: 25,
                    order: [[1, 'desc']],  // Sort by CPU % descending
                    dom: 'Bfrtip',
                    buttons: [
                        'copy', 
                        { extend: 'csv', title: 'CPU_Hotspots' },
                        { extend: 'excel', title: 'CPU_Hotspots' },
                        'print'
                    ],
                    language: {
                        search: "Filter hotspots:"
                    }
                });
            }
            
            // Memory Summary table
            if ($('#memory-summary-table').length) {
                $('#memory-summary-table').DataTable({
                    paging: false,
                    searching: false,
                    info: false,
                    dom: 'Bt',
                    buttons: [
                        'copy',
                        { extend: 'csv', title: 'Memory_Summary' },
                        'print'
                    ]
                });
            }
        });
        </script>
        """

        # Complete HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Profiling Dashboard - {run.run_id}</title>
            {datatables_css}
            {plotly_js}
            {datatables_js}
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #fafafa;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .tab-button:hover {{
                    opacity: 0.8;
                }}
                /* DataTables styling */
                .dataTables_wrapper {{
                    margin-top: 20px;
                }}
                .dt-buttons {{
                    margin-bottom: 15px;
                }}
                .dt-button {{
                    background-color: #667eea !important;
                    color: white !important;
                    border: none !important;
                    padding: 8px 16px !important;
                    margin-right: 8px !important;
                    border-radius: 4px !important;
                    cursor: pointer !important;
                }}
                .dt-button:hover {{
                    background-color: #5568d3 !important;
                }}
                table.dataTable thead th {{
                    background-color: #667eea;
                    color: white;
                }}
                table.dataTable tbody tr:hover {{
                    background-color: #f0f4ff;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {header_html}
                {tabs_html}
                {footer_html}
            </div>
            {tab_script}
        </body>
        </html>
        """

        return html


def generate_dashboard(
    run: ProfilingRun,
    output_path: Path,
    comparison_run: Optional[ProfilingRun] = None,
    offline: bool = True,
    theme: str = "plotly_white",
) -> Path:
    """Generate interactive dashboard for profiling run.

    Args:
        run: Profiling run to visualize
        output_path: Where to write HTML file
        comparison_run: Optional baseline for comparison
        offline: Include plotly.js for offline viewing
        theme: Plotly theme name

    Returns:
        Path to generated dashboard
    """
    generator = DashboardGenerator(offline=offline, theme=theme)
    return generator.generate(run, output_path, comparison_run)
