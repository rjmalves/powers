# Installation Guide

This guide explains how to install POWE.RS on your system.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Options](#installation-options)
  - [Pre-built Binaries](#option-1-pre-built-binaries-easiest)
  - [cargo install](#option-2-cargo-install)
  - [Building from Source](#option-3-building-from-source)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Operating Systems

- **Linux** (Ubuntu 20.04+, Debian 11+, Fedora 35+, or equivalent)
- **macOS** (10.15+ Catalina or later)
- **Windows** (Not officially supported yet, but may work with WSL2)

### Hardware

- **CPU**: Any modern x86_64 processor
- **RAM**: Minimum 2 GB, recommended 8+ GB for large problems
- **Disk**: ~100 MB for binary, ~500 MB for build from source

### Software Dependencies (for building from source)

POWE.RS requires these system dependencies to build:

- **Rust toolchain** (1.70.0 or later)
- **C++ compiler** (GCC 7+, Clang 6+, or MSVC 2019+)
- **CMake** (3.15 or later)
- **Build tools** (make, pkg-config)
- **libclang** (for bindgen)

---

## Installation Options

### Option 1: Pre-built Binaries (Easiest)

Pre-built binaries are available on each release page for Linux and macOS.

#### Using curl (Recommended)

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/rjmalves/powers/releases/download/<VERSION>/powers-rs-installer.sh | sh
```

Replace `<VERSION>` with the desired release tag (e.g., `v0.2.0`):

```bash
# Example: Install v0.2.0
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/rjmalves/powers/releases/download/v0.2.0/powers-rs-installer.sh | sh
```

This will:

- Download the appropriate binary for your platform
- Install it to `~/.cargo/bin/powers`
- Add `~/.cargo/bin` to your PATH (if needed)

#### Manual Download

1. Go to [Releases](https://github.com/rjmalves/powers/releases)
2. Download the binary for your platform:
   - `powers-x86_64-unknown-linux-gnu.tar.gz` (Linux)
   - `powers-x86_64-apple-darwin.tar.gz` (macOS Intel)
   - `powers-aarch64-apple-darwin.tar.gz` (macOS Apple Silicon)
3. Extract and move to your PATH:

```bash
# Linux/macOS
tar -xzf powers-*.tar.gz
sudo mv powers /usr/local/bin/
```

---

### Option 2: cargo install

If you have Rust installed, you can install directly from crates.io:

```bash
cargo install powers-rs
```

This will:

- Download and compile POWE.RS and all dependencies
- Install the `powers` binary to `~/.cargo/bin/`
- Take 5-15 minutes depending on your system

**Note**: This requires build dependencies (see [System Requirements](#system-requirements)).

---

### Option 3: Building from Source

For development or customization, build from source:

#### 1. Install Build Dependencies

**Ubuntu/Debian**:

```bash
sudo apt update
sudo apt install libclang-dev build-essential cmake pkg-config
```

**Fedora/RHEL**:

```bash
sudo dnf install clang-devel gcc-c++ cmake make
```

**macOS** (with Homebrew):

```bash
brew install cmake llvm
```

**Arch Linux**:

```bash
sudo pacman -S base-devel cmake clang
```

#### 2. Install Rust

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:

```bash
rustc --version  # Should show 1.70.0 or later
```

#### 3. Clone and Build

```bash
git clone https://github.com/rjmalves/powers.git
cd powers
cargo build --release
```

Build time: 5-20 minutes depending on your system.

The binary will be at `target/release/powers`.

#### 4. Optional: Install to PATH

```bash
cargo install --path .
```

This installs the binary to `~/.cargo/bin/powers`.

---

## Verifying Installation

Test that POWE.RS is correctly installed:

```bash
# Check version
powers --version

# Should output: powers-rs 0.2.0 (or similar)
```

Run the example problem:

```bash
# Clone repository if you haven't (to get example data)
git clone https://github.com/rjmalves/powers.git
cd powers

# Run example
powers example
```

**Expected output**:

```
POWE.RS - Power Optimization for the World of Energy - in pure RuSt
--------------------------------------------------------------------

Reading input files from 'example'

# Training
- Iterations: 32
- Forward passes: 4

------------------------------------------------------------
iteration  | lower bound ($) | simulation ($) |   time (s)
------------------------------------------------------------
         1 |        150.0000 |      8449.5644 |         0.01
         2 |       1934.4894 |      2982.8026 |         0.01
...
```

If you see this output, installation was successful! ✅

---

## Troubleshooting

### Error: "command not found: powers"

**Cause**: Binary not in PATH.

**Fix**:

```bash
# Check if ~/.cargo/bin is in PATH
echo $PATH | grep cargo

# If not, add to ~/.bashrc or ~/.zshrc:
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Error: "could not compile `highs-sys`"

**Cause**: Missing build dependencies (libclang, cmake, C++ compiler).

**Fix**: Install dependencies for your platform (see [Building from Source](#option-3-building-from-source)).

**Ubuntu/Debian**:

```bash
sudo apt install libclang-dev build-essential cmake
```

**macOS**:

```bash
brew install cmake llvm
```

### Error: "linker `cc` not found"

**Cause**: No C compiler installed.

**Fix**:

**Ubuntu/Debian**:

```bash
sudo apt install build-essential
```

**Fedora/RHEL**:

```bash
sudo dnf install gcc gcc-c++
```

**macOS**:

```bash
xcode-select --install
```

### Build is Very Slow

**Cause**: HiGHS solver is a large C++ project that takes time to compile.

**Tips**:

- Use pre-built binaries instead (much faster)
- Enable parallel compilation: `cargo build --release -j $(nproc)`
- First build is slow, subsequent builds are incremental (faster)

### Error: "error: failed to run custom build command for `highs-sys`"

**Cause**: CMake version too old or not found.

**Fix**: Install CMake 3.15 or later:

**Ubuntu 20.04**:

```bash
sudo apt install cmake
cmake --version  # Should be 3.15+
```

**macOS**:

```bash
brew install cmake
```

If your distribution's CMake is too old, install from https://cmake.org/download/

### macOS: "xcrun: error: invalid active developer path"

**Cause**: Xcode command line tools not installed.

**Fix**:

```bash
xcode-select --install
```

### Still Having Issues?

1. Check [Troubleshooting Guide](TROUBLESHOOTING.md) for common errors
2. Search [GitHub Issues](https://github.com/rjmalves/powers/issues)
3. Open a new issue with:
   - Your OS and version
   - Rust version (`rustc --version`)
   - Full error message
   - Output of `cargo build --release --verbose`

---

## Next Steps

Once installed, continue with:

- **[Quick Start Tutorial](QUICKSTART.md)** - Run your first optimization
- **[Input Specification](../reference/INPUT-SPECIFICATION.md)** - Understand the input format
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions

---

**Navigation**: [↑ Documentation Index](../README.md) | [Next: Quick Start →](QUICKSTART.md)
