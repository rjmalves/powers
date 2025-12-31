# HEkkDual and HSimplexNla Investigation Report

> **Sprint**: Epic 5, Sprint 7
> **Tickets**: T-096, T-097
> **Date**: 2025-12-30

---

## Executive Summary

This document summarizes the investigation into HEkkDual and HSimplexNla allocations,
which together account for ~21 GB of the remaining 45 GB total allocations after Sprint 6.

**Conclusion**: These allocations are **inherent to the HiGHS dual simplex algorithm** and
cannot be reduced without modifying HiGHS source code or switching to a different solver.

---

## T-096: HEkkDual Allocation Investigation

### Current State

| Metric | Value |
|--------|-------|
| Total Bytes | 17.96 GB |
| Total Blocks | 19.1 million |
| Primary Functions | `_M_fill_assign`, `_M_default_append` |

### Analysis

HEkkDual is the core dual simplex implementation in HiGHS. The allocations occur in:

1. **Working vectors**: Dual values, reduced costs, edge weights
2. **Pivot operations**: Row/column selection data structures  
3. **Update buffers**: Basis update intermediate results

These allocations are fundamental to the dual simplex algorithm and occur proportionally
to the number of pivots and iterations performed.

### Options Tested

| Option | Current | Tested | Impact on Allocations |
|--------|---------|--------|----------------------|
| `simplex_strategy` | 1 (dual) | 4 (primal) | Not viable - primal slower for SDDP |
| `simplex_update_limit` | 5000 | 1000, 10000 | Minimal impact on allocations |
| `simplex_dual_edge_weight_strategy` | -1 (auto) | 0 (Dantzig), 1 (Devex), 2 (Steepest) | Steepest slightly worse |

### Conclusion

HEkkDual allocations are **inherent to the dual simplex algorithm**. The allocations
scale with problem complexity and cannot be reduced without:

- Switching to a different LP solver algorithm
- Modifying HiGHS source code to pool/reuse internal vectors
- Reducing problem size (fewer constraints/cuts)

**Recommendation**: Accept as inherent limitation. Document for future reference.

---

## T-097: HSimplexNla Investigation  

### Current State

| Metric | Value |
|--------|-------|
| Total Bytes | 3.05 GB |
| Total Blocks | 0.4 million |
| Primary Functions | Debug-related (suspected), NLA operations |

### Analysis

HSimplexNla (Simplex Numerical Linear Algebra) handles:

1. **Factorization operations**: LU decomposition maintenance
2. **Basis management**: Valid basis verification
3. **Debug checks**: `debugCheckData` calls

### Debug Investigation

Checked if debug allocations could be disabled:

1. **HiGHS Build Type**: Release mode confirmed (no debug symbols in allocations)
2. **Output Flags**: Already set `output_flag=false`, `log_to_console=0`
3. **Debug Options**: No runtime option to disable internal consistency checks

The `debugCheckData` references in DHAT may be function names compiled into the binary
rather than actual debug code execution.

### Conclusion

HSimplexNla allocations are **inherent to HiGHS factorization operations**. The 3.05 GB
allocation is proportional to:

- Number of LP solves
- Constraint matrix density
- Factorization update frequency

**Recommendation**: Accept as inherent limitation. No actionable optimization found.

---

## Combined Impact

| Category | Before Sprint 6 | After Sprint 6 | Reduction |
|----------|-----------------|----------------|-----------|
| HEkkDual | 18.32 GB | 17.96 GB | 2.0% |
| HSimplexNla | 3.07 GB | 3.05 GB | 0.5% |
| **Combined** | **21.39 GB** | **21.01 GB** | **1.8%** |

These components are stable and not affected by the `reuse_forward_basis` changes
or batch bounds optimization, confirming they are internal HiGHS operations.

---

## Future Considerations

1. **Alternative Solvers**: GLPK, CLP, or commercial solvers may have different allocation patterns
2. **HiGHS Contribution**: Could propose PR to HiGHS for vector pooling in hot paths
3. **Problem Reduction**: Aggressive cut selection could reduce model size and thus allocations
4. **Incremental Solving**: HiGHS `run()` with warm-start may reduce some overhead (already used)

---

## References

- HiGHS Source: `HEkkDual.cpp`, `HEkkDualRow.cpp`, `HSimplexNla.cpp`
- DHAT Analysis: [DHAT_SPRINT6_ANALYSIS.md](./DHAT_SPRINT6_ANALYSIS.md)
- HiGHS Documentation: https://ergo-code.github.io/HiGHS/
