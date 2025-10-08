#!/bin/bash
# reproducibility_test.sh - Comprehensive reproducibility validation for POWE.RS
#
# This script tests the deterministic behavior of the SDDP algorithm by running
# the same example multiple times and comparing results at multiple levels:
# - Log output comparison
# - CSV coefficient precision analysis  
# - Performance consistency evaluation
#
# Usage: ./reproducibility_test.sh [example_path]
# Example: ./reproducibility_test.sh examples/05-large-scale-brazilian

set -e

# Configuration
DEFAULT_EXAMPLE="examples/05-large-scale-brazilian"
EXAMPLE="${1:-$DEFAULT_EXAMPLE}"
RUNS=3  # Test multiple runs for statistical confidence

echo "🔍 POWE.RS Reproducibility Test Suite"
echo "====================================="
echo "Testing example: $EXAMPLE"
echo "Number of runs: $RUNS"
echo ""

# Verify example exists
if [ ! -d "$EXAMPLE" ]; then
    echo "❌ Error: Example directory '$EXAMPLE' not found"
    echo "Available examples:"
    ls -1d examples/*/ 2>/dev/null || echo "No examples found"
    exit 1
fi

# Cleanup function
cleanup() {
    rm -f run*.log run*_cuts.csv run*_states.csv run*_time.txt
    rm -f run*_math.txt run*_sim.txt run*_bounds.txt run*_final_mean.txt
    rm -f temp_*.csv
}
trap cleanup EXIT

echo "📊 Testing Basic Reproducibility..."

# Run multiple times and collect outputs
for i in $(seq 1 $RUNS); do
    echo "  Running iteration $i/$RUNS..."
    start_time=$(date +%s.%N)
    
    # Run the example
    if ! cargo run --release "$EXAMPLE/" > "run${i}.log" 2>/dev/null; then
        echo "❌ Error: Failed to run example on iteration $i"
        exit 1
    fi
    
    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc -l)
    echo "$runtime" > "run${i}_time.txt"
    
    # Save CSV files if they exist
    if [ -f "$EXAMPLE/cuts.csv" ]; then
        cp "$EXAMPLE/cuts.csv" "run${i}_cuts.csv"
    fi
    if [ -f "$EXAMPLE/states.csv" ]; then
        cp "$EXAMPLE/states.csv" "run${i}_states.csv"
    fi
done

echo ""
echo "🔍 Analyzing Results..."

# Extract and compare mathematically relevant results (ignore timing)
echo "🔍 Extracting mathematical results (ignoring timing)..."

all_identical=true
for i in $(seq 1 $RUNS); do
    # Extract lower bounds, simulation values, and gaps (columns after | separators)
    # Format: "   1 |      239.19 |      308.10 |    28.81 |  timing... |"
    # Ignore timing columns as they naturally vary
    grep -E "^\s+[0-9]+\s+\|" "run${i}.log" | awk -F'|' '{
        gsub(/[ \t]+/, "", $2);  # lower bound
        gsub(/[ \t]+/, "", $3);  # simulation value  
        gsub(/[ \t]+/, "", $4);  # gap
        print $2, $3, $4
    }' > "run${i}_math.txt" 2>/dev/null || echo "" > "run${i}_math.txt"
    
    # Extract final simulation statistics
    grep -A 10 "Simulation statistics" "run${i}.log" | grep -E "(Mean|Std)" > "run${i}_sim.txt" 2>/dev/null || echo "" > "run${i}_sim.txt"
done

# Compare mathematical results (not timing)
for i in $(seq 2 $RUNS); do
    if ! diff -q "run1_math.txt" "run${i}_math.txt" > /dev/null; then
        echo "❌ Mathematical results differ between runs 1 and $i"
        all_identical=false
        
        echo "   Lower bounds and simulation values comparison:"
        echo "   Run 1 vs Run $i:"
        paste "run1_math.txt" "run${i}_math.txt" | head -5 | while read line; do
            echo "   $line"
        done
        echo ""
    fi
    
    # Compare simulation statistics
    if [ -s "run1_sim.txt" ] && [ -s "run${i}_sim.txt" ]; then
        if ! diff -q "run1_sim.txt" "run${i}_sim.txt" > /dev/null; then
            echo "❌ Simulation statistics differ between runs 1 and $i"
            all_identical=false
            echo "   Run 1 simulation stats:"
            cat "run1_sim.txt" | head -3
            echo "   Run $i simulation stats:"
            cat "run${i}_sim.txt" | head -3
            echo ""
        fi
    fi
done

# Check CSV file reproducibility (if available)
if [ -f "run1_cuts.csv" ]; then
    echo "🔍 Checking Cut CSV reproducibility..."
    for i in $(seq 2 $RUNS); do
        if ! diff -q "run1_cuts.csv" "run${i}_cuts.csv" > /dev/null; then
            echo "❌ Cut CSV files differ between runs 1 and $i"
            all_identical=false
            
            # Analyze coefficient differences
            echo "   Analyzing coefficient precision..."
            cut_diffs=$(awk '
                FNR==NR {a[NR]=$0; next}
                FNR in a && $0 != a[FNR] {
                    split(a[FNR], arr1, ",")
                    split($0, arr2, ",")
                    for(i=1; i<=NF; i++) {
                        if(arr1[i] != arr2[i] && arr1[i] ~ /^[0-9.-]+$/ && arr2[i] ~ /^[0-9.-]+$/) {
                            diff = arr1[i] - arr2[i]
                            if(diff != 0) {
                                printf "Line %d, Field %d: %.16f vs %.16f (diff: %.2e)\n", 
                                       FNR, i, arr1[i], arr2[i], diff
                            }
                        }
                    }
                }
            ' "run1_cuts.csv" "run${i}_cuts.csv" | head -5)
            
            if [ -n "$cut_diffs" ]; then
                echo "$cut_diffs"
            fi
            echo ""
        fi
    done
fi

# Check states CSV (if available)
if [ -f "run1_states.csv" ]; then
    echo "🔍 Checking State CSV reproducibility..."
    for i in $(seq 2 $RUNS); do
        if ! diff -q "run1_states.csv" "run${i}_states.csv" > /dev/null; then
            echo "❌ State CSV files differ between runs 1 and $i"
            all_identical=false
        fi
    done
fi

# Performance consistency check
echo ""
echo "⏱️ Performance Consistency Analysis..."

# Try to extract runtime from logs first, fallback to measured times
runtimes=()
for i in $(seq 1 $RUNS); do
    # Try to extract from log file
    runtime=$(grep -o "Total elapsed time: [0-9.]*s" "run${i}.log" 2>/dev/null | sed 's/Total elapsed time: //;s/s//' || echo "")
    
    # Fallback to measured time
    if [ -z "$runtime" ] && [ -f "run${i}_time.txt" ]; then
        runtime=$(cat "run${i}_time.txt")
    fi
    
    if [ -n "$runtime" ]; then
        runtimes+=("$runtime")
    fi
done

if [ ${#runtimes[@]} -eq $RUNS ]; then
    # Calculate runtime statistics
    runtime_stats=$(printf '%s\n' "${runtimes[@]}" | awk '
        {
            sum += $1
            sumsq += $1*$1
            values[NR] = $1
        }
        END {
            if(NR > 1) {
                mean = sum/NR
                variance = (sumsq - sum*sum/NR)/(NR-1)
                stddev = sqrt(variance)
                cv = stddev/mean * 100
                
                printf "  Mean: %.2fs, StdDev: %.3fs, CV: %.1f%%\n", mean, stddev, cv
                
                if(cv < 5.0) {
                    print "  ✅ Runtime consistency good (CV < 5%)"
                } else {
                    print "  ⚠️ Runtime variability high (CV >= 5%)"
                }
            } else {
                printf "  Single run: %.2fs\n", sum
            }
        }
    ')
    echo "$runtime_stats"
else
    echo "  ⚠️ Could not extract runtime information for analysis"
fi

# Extract and compare key metrics (focusing on algorithmic results)
echo ""
echo "📈 Key Metrics Summary..."

# Extract lower bounds progression (most critical for reproducibility)
for i in $(seq 1 $RUNS); do
    # Extract lower bounds from iteration table (column after first |)
    # Format: "   1 |      239.19 |      308.10 |..."
    # We want the first numerical value after the iteration number
    grep -E "^\s+[0-9]+\s+\|" "run${i}.log" | awk -F'|' '{gsub(/[ \t]+/, "", $2); print $2}' | paste -s -d',' > "run${i}_bounds.txt" 2>/dev/null || echo "" > "run${i}_bounds.txt"
done

if [ -f "run1_bounds.txt" ] && [ -s "run1_bounds.txt" ]; then
    bounds_identical=true
    for i in $(seq 2 $RUNS); do
        if [ -f "run${i}_bounds.txt" ] && ! diff -q "run1_bounds.txt" "run${i}_bounds.txt" > /dev/null; then
            echo "❌ Lower bounds progression differs between runs 1 and $i"
            echo "   Run 1: $(cat run1_bounds.txt)"
            echo "   Run $i: $(cat run${i}_bounds.txt)"
            bounds_identical=false
            all_identical=false
        fi
    done
    
    if [ "$bounds_identical" = true ]; then
        echo "✅ Lower bounds progression identical across all runs"
        echo "   Bounds: [$(cat run1_bounds.txt)]"
    fi
else
    echo "  ⚠️ Could not extract lower bounds progression for comparison"
fi

# Extract final simulation mean (critical result)
for i in $(seq 1 $RUNS); do
    grep "Mean:" "run${i}.log" | awk '{print $2}' | tail -1 > "run${i}_final_mean.txt" 2>/dev/null || echo "" > "run${i}_final_mean.txt"
done

if [ -f "run1_final_mean.txt" ] && [ -s "run1_final_mean.txt" ]; then
    final_mean_identical=true
    for i in $(seq 2 $RUNS); do
        if [ -f "run${i}_final_mean.txt" ] && ! diff -q "run1_final_mean.txt" "run${i}_final_mean.txt" > /dev/null; then
            echo "❌ Final simulation mean differs between runs 1 and $i"
            echo "   Run 1: $(cat run1_final_mean.txt)"
            echo "   Run $i: $(cat run${i}_final_mean.txt)"
            final_mean_identical=false
            all_identical=false
        fi
    done
    
    if [ "$final_mean_identical" = true ]; then
        echo "✅ Final simulation mean identical across all runs"
        echo "   Mean: $(cat run1_final_mean.txt)"
    fi
fi

# Final assessment
echo ""
echo "📋 Final Assessment:"
echo "==================="
if [ "$all_identical" = true ]; then
    echo "✅ PERFECT REPRODUCIBILITY ACHIEVED"
    echo "   ✓ Lower bounds progression identical"
    echo "   ✓ Simulation results identical"
    echo "   ✓ Mathematical results consistent across runs"
    echo "   ✓ CSV outputs identical (if generated)"
    echo ""
    echo "🎯 System Status: READY FOR PRODUCTION USE"
    echo "   The SDDP algorithm produces deterministic results"
    echo "   Safe for academic research and regulatory compliance"
    echo ""
    echo "ℹ️  Note: Timing variations are normal and expected"
    echo "   Only mathematical results are compared for reproducibility"
    exit 0
else
    echo "❌ NON-DETERMINISTIC BEHAVIOR DETECTED" 
    echo "   ✗ Mathematical results vary between runs"
    echo "   ✗ System NOT ready for production use"
    echo ""
    echo "🔧 Debugging Suggestions:"
    echo "   1. Try sequential execution:"
    echo "      export RAYON_NUM_THREADS=1"
    echo "      ./reproducibility_test.sh $EXAMPLE"
    echo ""
    echo "   2. Check solver configuration:"
    echo "      Verify set_default_solver_options() in src/subproblem.rs"
    echo ""
    echo "   3. Verify deterministic algorithms:"
    echo "      - Kahan summation in critical accumulations"
    echo "      - Deterministic dot product for cut heights"
    echo "      - Stable sorting algorithms"
    echo ""
    echo "   4. Enable detailed debugging:"
    echo "      RUST_LOG=debug cargo run --release $EXAMPLE/ > debug.log 2>&1"
    echo ""
    echo "   5. See detailed debugging guide:"
    echo "      docs/performance/DETERMINISTIC-REPRODUCIBILITY.md"
    echo ""
    echo "ℹ️  Note: Timing differences are ignored - only mathematical results matter"
    echo "⚠️ Do NOT use this build for production until reproducibility is fixed!"
    exit 1
fi