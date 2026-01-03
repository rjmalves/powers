# POWE.RS Performance Profiling (powers-profile)

Python-based CLI framework for POWE.RS performance evaluation. Provides placeholder commands that will be expanded with collectors, analyzers, and reporters across CPU, memory, parallelism, and timing domains.

## Installation

```bash
cd profiling
pip install -e .
```

## Usage

```bash
# Install
cd profiling
pip install -e .

# Run the timing collector against a binary
powers-profile run --collectors timing --binary /path/to/your/binary --output /tmp/profiling_results

# Run CPU profiling (requires perf + FlameGraph)
powers-profile run --collectors cpu --binary /path/to/your/binary --output /tmp/profiling_results \
  --config profiling/config/default.toml

# Inspect recorded runs
powers-profile history --output /tmp/profiling_results
powers-profile summary --output /tmp/profiling_results
```

### What works now
- Timing collector executes the target binary, parses `[TIMING]` markers, and persists raw stdout/stderr
- CPU collector wraps `perf record/script/report`, parses hotspots, and generates FlameGraphs when scripts are available
- Runs are stored under `<output>/runs/<run-id>/run.json` with machine-readable schemas
- History tracking via `<output>/history.json`
- CLI commands: `run`, `history`, and `summary`

### Roadmap (next up)
- Memory and parallel collectors
- JSON/Markdown reporters
- Dashboard generation

## Prerequisites for CPU profiling
- `perf` installed (e.g., `sudo apt install linux-tools-$(uname -r)`)
- FlameGraph scripts available locally; set `tools.flamegraph` in `profiling/config/default.toml` to the directory containing `flamegraph.pl`, `stackcollapse-perf.pl`, and `difffolded.pl`
- On WSL2, ensure `perf_event_paranoid` permits sampling or run with elevated privileges
