# Troubleshooting Guide

Solutions to common issues when using the profiling infrastructure.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Collection Failures](#collection-failures)
- [Permission Errors](#permission-errors)
- [Performance Issues](#performance-issues)
- [Data Interpretation](#data-interpretation)
- [Platform-Specific](#platform-specific)

---

## Installation Issues

### "pip install -e profiling/" fails

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement plotly>=5.18.0
```

**Solution:**
```bash
# Update pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -e profiling/ -v

# If still fails, install dependencies manually
pip install typer rich toml plotly pandas
```

### "python: command not found"

**Solution:**
```bash
# Use python3 explicitly
python3 -m powers_profile run

# Or create alias
alias python=python3

# Or activate virtual environment
source .venv/bin/activate
```

### Module import errors

**Symptom:**
```
ModuleNotFoundError: No module named 'powers_profile'
```

**Solution:**
```bash
# Ensure you're in the repository root
cd /path/to/powers

# Install in development mode
pip install -e profiling/

# Verify installation
python -c "import powers_profile; print(powers_profile.__version__)"
```

---

## Collection Failures

### "Binary not found"

**Symptom:**
```
Binary not found: target/release/powers
```

**Solutions:**
```bash
# Build the release binary
cargo build --release

# Or specify binary explicitly
python -m powers_profile run --binary path/to/binary

# Or update config
# ~/.config/powers-profile/config.toml:
[general]
binary = "/full/path/to/binary"
```

### Collector fails silently

**Symptom:**
```
Status: partial
Collectors run: timing, rss
# (cpu missing despite being requested)
```

**Diagnosis:**
```bash
# Check run.json for error messages
cat profiling_results/runs/latest/run.json | jq '.results.cpu'

# Common causes:
# - perf not available
# - valgrind not installed
# - binary crashed during collection
```

**Solutions:**
```bash
# Install missing tools
sudo apt install linux-tools-generic  # perf
sudo apt install valgrind              # memory tools

# Test binary independently
target/release/powers run examples/01-deterministic

# Use --continue-on-error to see partial results
python -m powers_profile run --continue-on-error
```

### Timeout errors

**Symptom:**
```
ERROR: Collection timed out after 600s
```

**Solutions:**
```bash
# Increase timeout in config
# config.toml:
[general]
timeout_seconds = 1200  # 20 minutes

# Or use smaller example
python -m powers_profile run -- run examples/01-deterministic

# For memory collectors (very slow), increase further
[memory]
timeout_seconds = 3600  # 1 hour
```

---

## Permission Errors

### "perf: Permission denied"

**Symptom:**
```
perf_event_open(...) failed: Permission denied
```

**Solutions:**
```bash
# Option 1: Reduce paranoid level (preferred)
sudo sysctl kernel.perf_event_paranoid=1

# Make permanent
echo 'kernel.perf_event_paranoid=1' | sudo tee -a /etc/sysctl.conf

# Option 2: Run with sudo (not recommended for benchmarks)
sudo python -m powers_profile run -c cpu

# Option 3: Add CAP_SYS_ADMIN capability (advanced)
sudo setcap cap_sys_admin+ep /usr/bin/perf
```

### "Operation not permitted" (valgrind)

**Symptom:**
```
valgrind: mmap(0x100000000, 2097152) failed in UME
```

**Solutions:**
```bash
# Increase memory limits
ulimit -v unlimited

# Or run with less aggressive settings
# Disable Massif if DHAT works
[memory]
massif_enabled = false
```

### "/proc not accessible"

**Symptom:**
```
ERROR: Cannot read /proc/PID/status
```

**Solutions:**
```bash
# Check /proc is mounted (WSL2)
mount | grep proc

# If not mounted
sudo mount -t proc proc /proc

# Check permissions
ls -la /proc/self/status
```

---

## Performance Issues

### Profiling takes too long

**Problem:** Full memory profiling is very slow (10-50x overhead).

**Solutions:**
```bash
# Use lightweight collectors for iteration
python -m powers_profile run -c timing,rss

# Reserve heavy collectors for final validation
python -m powers_profile run -c memory -- run examples/01-deterministic

# Disable slow collectors
[memory]
cachegrind_enabled = false  # 10-50x slowdown

# Use smaller examples
python -m powers_profile run -- run examples/01-deterministic
```

### Out of memory during profiling

**Symptom:**
```
ERROR: Out of memory
valgrind: failed to allocate memory
```

**Solutions:**
```bash
# Use smaller example
python -m powers_profile run -c memory -- run examples/01-deterministic

# Reduce Massif snapshot frequency
[memory]
massif_time_unit = "ms"  # Change to "B" for fewer snapshots

# Disable memory collectors temporarily
python -m powers_profile run -c timing,rss,cpu
```

### Dashboard generation fails

**Symptom:**
```
ERROR: Failed to generate dashboard
ImportError: No module named 'plotly'
```

**Solutions:**
```bash
# Install plotly
pip install plotly>=5.18.0

# Or reinstall profiling package
pip install -e profiling/

# If offline, install plotly offline bundle
pip install plotly-orca
```

---

## Data Interpretation

### "No timing data" but program ran

**Cause:** Timing collector parses stdout/stderr, but program doesn't output expected format.

**Solution:**
```bash
# Check stdout/stderr manually
cat profiling_results/runs/latest/timing/stdout.log

# Ensure program outputs timing info
# POWE.RS should output lines like:
# "Phase X completed in Y seconds"

# Configure parser in config.toml if needed
[timing]
parse_stdout = true
```

### Inconsistent performance across runs

**Causes:**
1. System load varies
2. Turbo boost enabled
3. Thermal throttling
4. Different NUMA nodes

**Solutions:**
```bash
# 1. Reduce system load
# Close other applications
# Disable background services

# 2. Disable turbo boost
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# 3. Set performance governor
sudo cpupower frequency-set --governor performance

# 4. Pin to NUMA node (multi-socket)
numactl --cpunodebind=0 --membind=0 python -m powers_profile run

# 5. Increase measurement iterations
python -m powers_profile scaling --iterations 10
```

### Comparison shows large delta but performance feels similar

**Causes:**
1. Measurement noise
2. Different system conditions
3. Non-deterministic timing

**Solutions:**
```bash
# Increase iterations to reduce noise
python -m powers_profile scaling --iterations 5 --warmup 2

# Profile both versions under same conditions
# - Same time of day
# - Same system load
# - Same CPU governor

# Use relative thresholds
[thresholds]
regression_percent = 10.0  # Ignore <10% changes
```

---

## Platform-Specific

### WSL2 Issues

**Problem:** perf doesn't work on WSL2.

**Solution:**
```bash
# Option 1: Use WSL2 with kernel 5.10.16.3+
# Check kernel version
uname -r

# Update if needed
wsl --update

# Option 2: Skip CPU collector on WSL2
python -m powers_profile run -c timing,rss,memory

# Option 3: Use bare-metal Linux for CPU profiling
```

**Problem:** Valgrind is very slow on WSL2.

**Solution:**
```bash
# Use smaller examples
python -m powers_profile run -c memory -- run examples/01-deterministic

# Or skip memory collectors on WSL2
python -m powers_profile run -c timing,rss,cpu
```

### macOS Not Supported

**Problem:** Most collectors require Linux-specific tools.

**Current Status:**
- ❌ CPU profiling (perf is Linux-only)
- ❌ Memory profiling (valgrind unreliable on macOS)
- ✅ Timing collector (works)
- ⚠️  RSS collector (requires /proc, may not work)

**Workarounds:**
```bash
# Use Docker with Linux image
docker run -it --rm -v $(pwd):/work ubuntu:22.04
# Inside container: install tools and run profiling

# Or use remote Linux machine
ssh linux-box "cd powers && python -m powers_profile run"
```

### Docker Issues

**Problem:** perf doesn't work in Docker.

**Solution:**
```bash
# Run Docker with --privileged
docker run --privileged -it ...

# Or with specific capabilities
docker run --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE -it ...

# Or skip perf
python -m powers_profile run -c timing,rss,memory
```

---

## Debugging Tips

### Enable verbose logging

```bash
# Set log level in config
[timing]
log_level = "debug"

# Or use Python logging
export PYTHONUNBUFFERED=1
python -m powers_profile run -c timing 2>&1 | tee profiling.log
```

### Inspect JSON output

```bash
# Pretty-print run metadata
cat profiling_results/runs/latest/run.json | jq '.'

# Check collector results
jq '.results' profiling_results/runs/latest/run.json

# Check for errors
jq '.results.cpu.error_message' profiling_results/runs/latest/run.json
```

### Test individual collectors

```bash
# Test timing only
python -m powers_profile run -c timing

# Test RSS only
python -m powers_profile run -c rss

# Test CPU only (requires perf)
python -m powers_profile run -c cpu

# Test memory only (very slow)
python -m powers_profile run -c memory -- run examples/01-deterministic
```

### Validate installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Check installed packages
pip list | grep -E "typer|rich|plotly|toml"

# Check system tools
which perf
which valgrind
perf --version
valgrind --version

# Test import
python -c "from powers_profile import __version__; print(__version__)"
```

---

## Getting Help

If you encounter an issue not covered here:

1. **Check logs**: `profiling_results/runs/latest/run.json` for error messages
2. **Search issues**: Check GitHub issues for similar problems
3. **Minimal reproduction**: Create smallest example that reproduces issue
4. **Report**: Open GitHub issue with:
   - Error message (full traceback)
   - System info (`uname -a`, Python version)
   - Configuration (`cat ~/.config/powers-profile/config.toml`)
   - Steps to reproduce

---

## See Also

- **[Quick Start](QUICK_START.md)** - Getting started guide
- **[Tools Reference](TOOLS_REFERENCE.md)** - Complete collector documentation
- **[Analysis Guide](ANALYSIS_GUIDE.md)** - Interpreting results
- **[Scaling Guide](../SCALING_GUIDE.md)** - Multi-socket systems
