# Quick Start Guide

Get started with POWE.RS performance profiling in 5 minutes.

## Prerequisites

### System Requirements

- **Operating System**: Linux (WSL2 on Windows works)
- **Python**: 3.10 or newer
- **Rust**: 1.70 or newer (for building POWE.RS)

### Required Tools

```bash
# Linux perf tools (for CPU profiling)
sudo apt install linux-tools-common linux-tools-generic

# Valgrind (for memory profiling)
sudo apt install valgrind

# Python development headers
sudo apt install python3-dev python3-pip
```

### Optional Tools

```bash
# FlameGraph for CPU visualization
git clone https://github.com/brendangregg/FlameGraph.git ~/FlameGraph
export PATH="$HOME/FlameGraph:$PATH"
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/powers.git
cd powers
```

### 2. Install Profiling Tool

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install powers-profile
pip install -e profiling/
```

### 3. Build POWE.RS

```bash
cargo build --release
```

## First Profiling Run

### Quick Profile

Run a quick profile with default collectors:

```bash
python -m powers_profile run
```

This will:
- Run the default example (`examples/05-large-scale-brazilian`)
- Collect timing and basic metrics
- Save results to `profiling_results/runs/`

### View Results

```bash
# Show summary in terminal
python -m powers_profile summary

# Generate interactive dashboard
python -m powers_profile dashboard

# Open in browser
firefox profiling_results/runs/latest/dashboard.html
```

## Common Workflows

### Full Profiling Suite

Run comprehensive profiling with all collectors:

```bash
# CPU + Memory + Timing
python -m powers_profile run -c cpu,memory,timing

# Or use specific collectors
python -m powers_profile run -c timing,rss  # Lightweight
python -m powers_profile run -c cpu         # CPU only
```

### Memory Profiling

```bash
# Lightweight RSS monitoring (fast)
python -m powers_profile run -c rss

# Comprehensive memory analysis (slow, use small example)
python -m powers_profile run -c memory -- run examples/01-deterministic
```

### Parallel Scaling Analysis

```bash
# Test scaling from 1 to 16 threads
python -m powers_profile scaling --threads 1,2,4,8,16

# With contention detection
python -m powers_profile scaling --threads 1,2,4,8 --contention
```

### Version Comparison

```bash
# Profile baseline
git checkout v0.2.0
cargo build --release
python -m powers_profile run -c cpu,memory
BASELINE_ID=$(python -m powers_profile history -n 1 --format json | jq -r '.[0].run_id')

# Profile new version
git checkout main
cargo build --release
python -m powers_profile run -c cpu,memory
CURRENT_ID=$(python -m powers_profile history -n 1 --format json | jq -r '.[0].run_id')

# Compare
python -m powers_profile compare $BASELINE_ID $CURRENT_ID

# Generate comparison dashboard
python -m powers_profile dashboard --baseline $BASELINE_ID $CURRENT_ID
```

## Understanding Output

### Run Directory Structure

```
profiling_results/runs/20260103-180250-abc1234/
├── run.json              # Complete run metadata
├── timing/
│   ├── stdout.log        # Program output
│   └── stderr.log
├── rss/
│   └── rss_data.json     # RSS timeline + summary
├── cpu/
│   ├── perf.data
│   ├── flamegraph.svg
│   └── hotspots.json
├── memory/
│   ├── dhat/
│   ├── massif/
│   └── memory_data.json
├── parallel/
│   └── scaling_data.json
├── dashboard.html        # Interactive visualization
└── report.md            # Markdown summary
```

### Key Metrics

| Metric | Location | Interpretation |
|--------|----------|----------------|
| **Total Duration** | `run.json` | Overall execution time |
| **Peak RSS** | `rss/rss_data.json` → `summary.peak_mb` | Maximum memory usage |
| **CPU Hotspots** | `cpu/hotspots.json` | Functions consuming most CPU |
| **Heap Allocation** | `memory/memory_data.json` → `dhat` | Dynamic memory allocation patterns |
| **Speedup** | `parallel/scaling_data.json` | Parallel efficiency |

## Next Steps

- 📖 **[Tools Reference](TOOLS_REFERENCE.md)** - Complete list of collectors and options
- 📊 **[Analysis Guide](ANALYSIS_GUIDE.md)** - How to interpret profiling results
- 🔍 **[Comparison Guide](COMPARISON_GUIDE.md)** - Detecting regressions between versions
- 🔧 **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- 🛠️ **[Extending Guide](EXTENDING.md)** - Adding custom collectors

## Quick Reference

### Most Common Commands

```bash
# Basic profiling
python -m powers_profile run                              # Default profile
python -m powers_profile run -c timing,rss               # Fast profile
python -m powers_profile run -c cpu,memory               # Comprehensive

# Specific example
python -m powers_profile run -- run examples/01-deterministic

# View results
python -m powers_profile summary                          # Terminal output
python -m powers_profile summary --verbose               # Detailed tables
python -m powers_profile dashboard                        # Interactive HTML
python -m powers_profile report                           # Markdown report

# History
python -m powers_profile history                          # Recent runs
python -m powers_profile history --limit 50              # More runs

# Scaling
python -m powers_profile scaling --threads 1,2,4,8       # Thread scaling
python -m powers_profile scaling --contention            # With lock analysis

# Comparison
python -m powers_profile compare <baseline> <target>     # Compare two runs
python -m powers_profile dashboard --baseline <id> <id>  # Visual comparison
```

### Configuration

Create `~/.config/powers-profile/config.toml`:

```toml
[general]
output_dir = "profiling_results"
binary = "target/release/powers"
default_example = "examples/05-large-scale-brazilian"

[cpu]
perf_frequency = 99  # Hz

[memory]
dhat_enabled = true
massif_enabled = true
cachegrind_enabled = false  # Slow, opt-in only
rss_interval_ms = 500

[parallel]
thread_counts = [1, 2, 4, 8, 16, 32]
warmup_iterations = 1
```

## Tips

💡 **Performance**: Start with lightweight collectors (`timing`, `rss`) for quick feedback, then use heavy collectors (`memory`, `cpu`) for deep dives.

💡 **Examples**: Use small examples (`01-deterministic`) for memory profiling with valgrind tools (10-50x slower).

💡 **Baseline**: Always profile both baseline and current versions with the same system conditions for accurate comparison.

💡 **Visualization**: Interactive dashboards are great for exploration, markdown reports are better for documentation.

💡 **Automation**: Use `--continue-on-error` in CI to collect partial results even if one collector fails.
