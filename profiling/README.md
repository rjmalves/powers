# POWE.RS Performance Profiling Infrastructure

Comprehensive performance evaluation toolkit for the POWE.RS power optimization framework.

## Quick Start

### Installation

```bash
# From repository root
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e profiling/
```

### Basic Usage

```bash
# Profile with default collectors and example
python -m powers_profile run

# Profile with specific collectors
python -m powers_profile run -c timing,rss

# Profile with custom binary arguments
python -m powers_profile run -- run examples/01-deterministic

# Profile with all collectors
python -m powers_profile run -c all

# View profiling history
python -m powers_profile history

# View latest run summary
python -m powers_profile summary
```

## Available Collectors

### Lightweight (Fast)
- **timing**: Extract timing metrics from program output (~0% overhead)
- **rss**: Track physical memory usage via /proc polling (<1% overhead)

### CPU Profiling
- **cpu**: Linux `perf` integration with flamegraph generation (1-5% overhead)

### Memory Profiling (Valgrind-based, slower)
- **dhat**: Heap allocation profiling (3-10x slowdown)
- **massif**: Heap usage over time (5-20x slowdown)
- **cachegrind**: Cache efficiency analysis (10-50x slowdown, opt-in)
- **memory**: Unified memory collector (runs all enabled memory tools)

## Configuration

Configuration is loaded from (in order):
1. `profiling/config/default.toml` (repository defaults)
2. `~/.config/powers-profile/config.toml` (user overrides)
3. `--config` CLI option (session override)

### Key Configuration Options

```toml
[general]
output_dir = "profiling_results"
binary = "target/release/powers"
default_example = "examples/05-large-scale-brazilian"

[memory]
dhat_enabled = true
massif_enabled = true
cachegrind_enabled = false  # Slow, opt-in only
rss_interval_ms = 500

[cpu]
perf_frequency = 99  # Hz
flamegraph_colors = "hot"
```

## Output Structure

```
profiling_results/
├── runs/
│   └── 20260103-180250-dfe8823/  # Run ID: YYYYMMDD-HHMMSS-commit
│       ├── run.json              # Complete run metadata
│       ├── timing/
│       │   ├── stdout.log
│       │   └── stderr.log
│       ├── rss/
│       │   └── rss_data.json     # RSS samples + summary
│       ├── cpu/
│       │   ├── perf.data
│       │   ├── flamegraph.svg
│       │   └── hotspots.json
│       └── memory/
│           ├── dhat/
│           │   └── dhat.out.json
│           ├── massif/
│           │   └── massif.out
│           ├── rss/
│           │   └── rss_data.json
│           └── memory_data.json  # Aggregated metrics
└── history.jsonl                  # Run history log
```

## Examples

### Profiling CPU Performance

```bash
# Profile CPU with flamegraph
python -m powers_profile run -c cpu -- run examples/01-deterministic

# View results
ls profiling_results/runs/latest/cpu/
# flamegraph.svg  perf.data  hotspots.json
```

### Profiling Memory Usage

```bash
# Lightweight RSS monitoring
python -m powers_profile run -c rss -- run examples/01-deterministic

# Full memory profiling (slow, small workload recommended)
python -m powers_profile run -c memory -- run examples/01-deterministic

# View RSS summary
jq '.summary' profiling_results/runs/latest/rss/rss_data.json
```

### Comparing Two Runs

```bash
# Run baseline
python -m powers_profile run -c cpu

# Make changes to code...
# cargo build --release

# Run target
python -m powers_profile run -c cpu

# Compare (generates differential flamegraph)
python -m powers_profile compare <baseline-id> <target-id>
```

## Requirements

### System Tools
- **Linux**: Required for `perf` and `/proc` filesystem
- **perf** (`linux-tools-common`): For CPU profiling
- **valgrind**: For memory profiling (DHAT, Massif, Cachegrind)
- **FlameGraph** (optional): For flamegraph visualization

### Python Dependencies
- Python >=3.10
- typer >=0.9.0
- rich >=13.0.0
- toml >=0.10.0
- plotly >=5.18.0 (for future dashboard)
- pandas >=2.0.0 (for future dashboard)

## Troubleshooting

### "Binary not found" Error
```bash
# Build the release binary first
cargo build --release

# Or specify custom binary
python -m powers_profile run --binary path/to/binary
```

### "perf not found" Error
```bash
# Install Linux perf tools
sudo apt install linux-tools-common linux-tools-generic

# Or configure perf path
# In ~/.config/powers-profile/config.toml:
[tools]
perf = "/usr/bin/perf"
```

### "valgrind not found" Error
```bash
# Install valgrind
sudo apt install valgrind

# Or disable memory profiling
python -m powers_profile run -c timing,rss
```

### Collector Fails
```bash
# Check run details
python -m powers_profile summary latest

# View full run JSON
jq '.' profiling_results/runs/latest/run.json
```

## Development

### Running Tests
```bash
cd profiling
python -m pytest tests/
```

### Adding a New Collector

1. Create `profiling/powers_profile/collectors/mycollector.py`
2. Implement `Collector` interface
3. Add to `collectors/__init__.py` registry
4. Add tests in `tests/test_mycollector.py`
5. Update `config/default.toml` with collector config

See existing collectors for examples.

## Architecture

The profiling framework consists of:

- **CLI** (`cli.py`): Typer-based command-line interface
- **Collectors** (`collectors/`): Data collection plugins
- **Runtime** (`runtime.py`): Run management and history
- **Schemas** (`schemas/`): Data models and serialization
- **Reporters** (`reporters/`): Result analysis (future)
- **Config** (`config.py`): Configuration management

## License

Part of the POWE.RS project. See top-level LICENSE file.
