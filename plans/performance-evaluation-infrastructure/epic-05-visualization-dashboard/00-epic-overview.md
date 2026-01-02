# Epic 5: Visualization & Reporting Dashboard

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: ⬜ Not Started

---

## Summary

This epic implements interactive Plotly-based dashboards for visualizing profiling data, along with markdown report generation and CLI summaries. It enables developers to explore performance data interactively, generate shareable reports, and get quick terminal-based insights.

---

## Scope

### Included

1. **Interactive Plotly Dashboards**
   - HTML file generation (no server required)
   - CPU timeline and flame graph embedding
   - Memory usage charts (RSS, heap over time)
   - Scaling efficiency curves
   - Comparison views (baseline vs target)

2. **Dashboard Components**
   - Summary metrics panel
   - Timing breakdown charts (bar, pie)
   - Memory timeline (line chart)
   - Scaling efficiency chart
   - Hotspot tables
   - FlameGraph SVG embedding

3. **Markdown Reports**
   - Human-readable summary
   - Tables with key metrics
   - Embedded images/links to SVGs
   - Regression/improvement highlights

4. **CLI Summaries**
   - Rich terminal formatting
   - Quick metrics display
   - Colored regression/improvement indicators
   - Progress bars for comparisons

5. **Comparison Views**
   - Side-by-side metrics
   - Delta percentages with color coding
   - Before/after charts

### Excluded

- Real-time dashboards
- Server-based deployment
- Grafana integration

---

## Dependencies

- **Requires**: Epics 1-4 (data from all collectors)
- **Enables**: Epic 6 (Documentation)

---

## Acceptance Criteria

- [ ] `powers-profile dashboard` generates interactive HTML
- [ ] Dashboard includes all profiling domains (CPU, memory, parallel, timing)
- [ ] FlameGraph SVG embedded in dashboard
- [ ] Markdown report generated with key metrics
- [ ] `powers-profile summary` shows rich CLI output
- [ ] Comparison dashboard shows baseline vs target
- [ ] Charts are interactive (hover, zoom, pan)
- [ ] Dashboard works offline (no CDN dependencies)

---

## Technical Approach

### Plotly Dashboard Structure

```
dashboard.html
├── Header
│   ├── Run ID, timestamp, git info
│   └── System info summary
├── Summary Cards
│   ├── Total time
│   ├── Peak memory
│   ├── Parallel efficiency
│   └── Hotspot count
├── Tabs
│   ├── Timing
│   │   ├── Phase breakdown (bar chart)
│   │   └── Iteration timeline
│   ├── CPU
│   │   ├── FlameGraph (embedded SVG)
│   │   └── Hotspot table
│   ├── Memory
│   │   ├── RSS timeline
│   │   ├── DHAT summary
│   │   └── Massif peak chart
│   ├── Parallel
│   │   ├── Scaling curve
│   │   └── Efficiency table
│   └── Comparison (if applicable)
│       ├── Delta table
│       └── Before/after charts
└── Footer
    └── Generation timestamp, version
```

### Plotly Offline Mode

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Generate self-contained HTML
pio.write_html(fig, "dashboard.html", include_plotlyjs="cdn")
# Or for offline:
pio.write_html(fig, "dashboard.html", include_plotlyjs=True)
```

### Rich CLI Output

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()

def show_summary(run: ProfilingRun):
    table = Table(title=f"Run {run.run_id[:8]}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Status")
    
    table.add_row("Total Time", "25.3s", "[green]✓[/green]")
    table.add_row("Peak RSS", "253 MB", "[green]✓[/green]")
    table.add_row("Hotspots", "15", "[yellow]![/yellow]")
    
    console.print(table)
```

---

## Sprints

### [Sprint 1: Visualization](./sprint-01/00-sprint-overview.md)

| ID | Title | Points | Status |
|----|-------|--------|--------|
| T-033 | Implement dashboard base template | 5 | ⬜ |
| T-034 | Add timing visualization charts | 3 | ⬜ |
| T-035 | Add memory visualization charts | 3 | ⬜ |
| T-036 | Add scaling visualization charts | 3 | ⬜ |
| T-037 | Embed FlameGraph in dashboard | 3 | ⬜ |
| T-038 | Implement markdown report generator | 3 | ⬜ |
| T-039 | Implement rich CLI summary | 3 | ⬜ |
| T-040 | Add comparison dashboard view | 5 | ⬜ |

**Sprint Points**: 28

---

## Estimated Effort

- **Duration**: 1 sprint (2 weeks)
- **Story Points**: 28
- **Risk Level**: Low (Plotly well-documented, Rich straightforward)

---

## Definition of Done

- [ ] All tickets complete
- [ ] Interactive dashboard generated
- [ ] All chart types working
- [ ] Markdown report generated
- [ ] CLI summary shows colored output
- [ ] Comparison view works
- [ ] Dashboard works offline
