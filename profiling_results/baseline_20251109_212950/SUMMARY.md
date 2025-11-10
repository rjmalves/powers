# Performance Profiling Summary

**Date**: $(date)
**Example**: $EXAMPLE_DIR
**Rust Version**: $(rustc --version)
**CPU**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
**RAM**: $(free -h | grep Mem | awk '{print $2}')

## Files Generated

1. `benchmark_output.txt` - Criterion benchmark results
2. `flamegraph.svg` - CPU profiling visualization
3. `massif_report.txt` - Memory usage analysis
4. `perf_stat.txt` - Cache performance statistics
5. `perf_report.txt` - Detailed perf analysis
6. `timing.md` - Quick timing comparison

## Quick Analysis

### Top CPU Consumers (from flamegraph)
TODO: Open flamegraph.svg and list top 5 functions

### Memory Usage (from massif)
**Peak Memory**: 

### Cache Performance (from perf)
   <not supported>      cache-misses:u                                                        
   <not supported>      cache-references:u                                                    
   <not supported>      instructions:u                                                        
   <not supported>      cycles:u                                                              

## Next Steps

1. Review flamegraph.svg to identify CPU hotspots
2. Review massif_report.txt to identify allocation hotspots
3. Review perf_report.txt for cache analysis
4. Document top 5 bottlenecks in PROFILING_RESULTS.md
5. Prioritize optimizations based on data

## Benchmark Baseline

Benchmarks saved to: `target/criterion/before_refactoring/`

To compare after changes:
```bash
cargo bench --baseline before_refactoring
```
