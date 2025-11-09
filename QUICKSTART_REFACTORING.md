# Quick Start: Performance Refactoring

**tl;dr**: Profile first, optimize hot paths, measure improvements.

---

## ⚡ Quick Command Reference

```bash
# Step 1: Profile the baseline (5-10 minutes)
./scripts/profile_baseline.sh examples/fourbus/

# Step 2: View results
firefox profiling_results/baseline_*/flamegraph.svg
cat profiling_results/baseline_*/SUMMARY.md

# Step 3: Save benchmark baseline
cargo bench --save-baseline before_refactoring

# Step 4: Make optimizations (following the plan)

# Step 5: Measure improvement
cargo bench --baseline before_refactoring
cargo test --release

# Step 6: Commit
git add -A
git commit -m "perf: [optimization description] - X% improvement"
```

---

## 📚 Documentation Map

| File | Purpose | When to Read |
|------|---------|--------------|
| **REFACTORING_SUMMARY.md** | Overview of all plans | **Read first** ⭐ |
| **PERFORMANCE_REFACTORING_PLAN.md** | Detailed perf plan (44KB) | Main guide |
| **REFACTORING_PLAN.md** | Clean code alternative | For comparison |
| **REFACTORING_APPROACHES_COMPARISON.md** | Side-by-side | Understand trade-offs |
| **PROFILING_RESULTS.md** | Template for findings | After profiling |

---

## 🎯 Recommended Path

### For POWE.RS (HPC Application)

```
1. Read REFACTORING_SUMMARY.md (5 min)
   └→ Understand the approach
   
2. Run profiling (10 min)
   └→ ./scripts/profile_baseline.sh
   
3. Document findings (15 min)
   └→ Fill in PROFILING_RESULTS.md
   
4. Start Phase 1: Memory optimization (1-2 weeks)
   └→ See PERFORMANCE_REFACTORING_PLAN.md Phase 1
   
5. Measure & iterate
   └→ cargo bench --baseline before_refactoring
```

---

## 🔍 Finding What You Need

### "I want to understand the overall strategy"
→ Read `REFACTORING_SUMMARY.md`

### "I want detailed optimization steps"
→ Read `PERFORMANCE_REFACTORING_PLAN.md`

### "I want to see the trade-offs"
→ Read `REFACTORING_APPROACHES_COMPARISON.md`

### "I want to start optimizing NOW"
1. Run `./scripts/profile_baseline.sh`
2. Read Phase 1 of `PERFORMANCE_REFACTORING_PLAN.md`
3. Start with buffer pre-allocation

### "I want clean code, not performance"
→ Read `REFACTORING_PLAN.md` instead

---

## 📊 Performance Targets

After completing all phases:

| Metric | Target |
|--------|--------|
| Overall runtime | **-40% to -60%** |
| Forward pass | **-25%** |
| Backward pass | **-30%** |
| Allocations | **-80%** |
| Memory per iteration | **Constant** |
| Cache miss rate | **-20%** |

---

## 🛠️ Required Tools

```bash
# Install once
cargo install flamegraph criterion
sudo apt install valgrind linux-tools-generic

# Verify
flamegraph --version
valgrind --version
perf --version
```

---

## ⚠️ Important Principles

1. **Profile First** - Don't guess what's slow
2. **Measure Everything** - Benchmarks before & after
3. **Hot Path Focus** - 80% effort on 20% of code
4. **Document Trade-offs** - Explain performance choices
5. **Keep Tests Passing** - Correctness > Speed

---

## 🚦 When to Stop

Stop optimizing when:
- ✅ Performance goals met
- ✅ No clear bottlenecks in profiling
- ❌ Further optimization too complex
- ❌ Diminishing returns (<5% gain)

---

## 📞 Quick Help

### "Profiling script failed"
- Check if `examples/fourbus/` exists
- Try different example: `./scripts/profile_baseline.sh examples/other/`
- Check tool installation

### "Benchmarks show regression"
- Verify correctness: `cargo test`
- Check if profiling supports the change
- Consider reverting

### "Code is getting messy"
- Add PERFORMANCE comments
- Document trade-offs
- Keep hot/cold paths separate

---

## 🎓 Learning Path

### Beginner
1. Read REFACTORING_SUMMARY.md
2. Run profiling script
3. Do Phase 1 (memory optimization)

### Intermediate  
1. Read full PERFORMANCE_REFACTORING_PLAN.md
2. Profile and identify bottlenecks
3. Do Phases 1-3

### Advanced
1. Read both plans + comparison
2. Use hybrid approach
3. Do all phases + custom optimizations

---

## ✅ Quick Checklist

Before starting:
- [ ] Read REFACTORING_SUMMARY.md
- [ ] Install profiling tools
- [ ] Run `./scripts/profile_baseline.sh`
- [ ] Document findings
- [ ] Choose Phase 1 target

After each phase:
- [ ] Benchmark improvement
- [ ] Run tests
- [ ] Document with PERFORMANCE comments
- [ ] Git commit with measurements

---

## 🚀 Get Started

```bash
# 1. Profile (do this now!)
./scripts/profile_baseline.sh examples/fourbus/

# 2. While that runs, read:
cat REFACTORING_SUMMARY.md

# 3. After profiling, check results:
ls profiling_results/baseline_*/

# 4. Document findings:
$EDITOR PROFILING_RESULTS.md

# 5. Start optimizing! Follow Phase 1:
$EDITOR PERFORMANCE_REFACTORING_PLAN.md
# (Search for "Phase 1")
```

---

**Ready? Run this command to start:**

```bash
./scripts/profile_baseline.sh examples/fourbus/
```

Then open the flamegraph and see what's slow! 🔥

---

**Pro Tip**: Bookmark `REFACTORING_SUMMARY.md` - it has everything you need.
