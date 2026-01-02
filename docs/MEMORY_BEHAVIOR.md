# Memory Behavior

## Overview

After Epic 5 Sprint 5 optimizations, powers achieves deterministic memory allocation during SDDP training. After an initialization and warmup phase, the training loop performs near-zero heap allocations in the hot path.

## Memory Phases

### 1. Initialization Phase

During initialization, all data structures are allocated:

- Model construction (HiGHS LP problem)
- Subproblem graphs
- Realization graphs
- Initial constraint matrices

**Expected behavior**: Memory grows as models are constructed.

### 2. Preallocation Phase

Cut constraint slots are preallocated:

- `preallocate_cut_constraints()` creates placeholder rows in HiGHS
- Pool data structures are sized for maximum capacity
- Handler buffers are allocated

**Expected behavior**: Memory jumps to final working set size.

### 3. Warmup Phase

Solvers are warmed up to pre-allocate internal structures:

- `warmup_solver()` performs a single solve to force HiGHS to allocate:
  - LU factorization storage
  - Work vectors for pricing
  - Steepest-edge weights
  - Row/column pricing vectors

**Expected behavior**: Memory may increase slightly, then stabilize.

### 4. Training Phase (Stable Memory)

During training iterations:

- Cut coefficients updated via `change_coefficient()` (no allocation)
- Cut bounds updated via `change_rows_bounds()` (no allocation)
- Thread-local buffers reused for any row operations
- Coordinator uses preallocated result buffers

**Expected behavior**: RSS remains constant (±1% variation acceptable for OS page management).

### 5. Finalization Phase

After training:

- Results are constructed and serialized
- Memory may increase temporarily for output generation
- Cleanup reduces memory

**Expected behavior**: Memory may fluctuate during output, then decrease.

## Monitoring Memory

### Using RSS Monitoring

Monitor resident set size during execution:

```bash
./target/release/powers run examples/05-large-scale-brazilian &
PID=$!
watch -n 1 "ps -o rss= -p $PID"
```

### Using /usr/bin/time

Get peak memory usage:

```bash
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | grep "Maximum resident"
```

### Using DHAT for Detailed Analysis

Profile heap allocations:

```bash
# Build with debug info
RUSTFLAGS="-g" cargo build --release

# Run with DHAT
valgrind --tool=dhat --dhat-out-file=dhat.out ./target/release/powers run examples/05-large-scale-brazilian

# View results in browser
# Open dhat.out.* file with dh_view.html
```

## Expected Allocation Sites

After warmup, the following allocations are acceptable:

| Category | Description | When |
|----------|-------------|------|
| Logging | Tracing/log output buffers | During training |
| I/O | File write buffers | At training end |
| JSON | Serialization buffers | At training end |
| History | Realization clones | Only if `record_history` enabled |

## Troubleshooting

### Memory Growing During Training

If you observe memory growth during training:

1. **Check if history recording is enabled** - This causes realization cloning
2. **Verify cut preallocation was called** - Check logs for preallocation messages
3. **Run DHAT to identify allocation sites** - Profile to find unexpected allocators

### Peak Memory Too High

If peak memory is higher than expected:

1. **Reduce num_forward_passes** - Fewer parallel handlers = less memory
2. **Reduce num_iterations** - Fewer preallocated cut slots
3. **Check for unnecessary history recording** - Disable if not needed

### Allocations in Hot Path

If DHAT shows allocations in hot path functions:

1. Check that all cuts use preallocation path
2. Verify thread-local buffers are being reused
3. Look for temporary Vec creations in loops

## Technical Details

### Thread-Local Buffers

The following operations use thread-local buffers:

- `try_add_row()` - Column indices and values
- `delete_row()` - Row index set
- `get_basis_into()` - Raw status buffers
- Cut evaluation - Computation buffers

### Preallocated Structures

| Structure | Purpose |
|-----------|---------|
| `CutStagingBuffer` | Per-handler cut computation staging |
| `CoordinatorBuffers` | Result collection during backward pass |
| Cut constraint rows | HiGHS model placeholder constraints |
| Pool slots | Cut and state storage |

### Zero-Allocation Path

The training hot path uses:

```
evaluate_cut_ref() → staging buffer
  ↓
compute_cut_into_slot() → pool slot update
  ↓
add_cut_with_preallocation() → HiGHS coefficient/bound changes
  ↓
realize_and_solve() → thread-local solution buffer
```

No heap allocation occurs in this path after initialization.
