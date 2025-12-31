# [T-108] Add Rebuild Configuration Options

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 8: Model Rebuild Strategy](./00-sprint-overview.md)
> **Dependencies**: T-107
> **Blocks**: T-106
> **Priority**: 1 (Medium)
> **Status**: 🔵 Ready

## Files to Read Before Starting

- `src/sddp/mod.rs` - Training loop with hardcoded rebuild interval
- Sprint overview for design context

---

## Context

### Background

T-107 integrates model rebuild with a hardcoded interval. This ticket makes it configurable so users can tune the tradeoff between memory usage and rebuild overhead.

### Configuration Options

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| `model_rebuild_interval` | 100 | 0 (disabled), 10-1000 | Iterations between rebuilds |
| `enable_model_rebuild` | true | bool | Feature toggle |

---

## Specification

### API Changes

```rust
impl SddpAlgorithm {
    /// Train with configuration options.
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration including rebuild options
    /// * `saa` - Scenario approximation
    ///
    /// # Rebuild Behavior
    ///
    /// When `config.model_rebuild_interval > 0`, models are rebuilt every
    /// N iterations to reclaim HiGHS internal memory. Set to 0 to disable.
    pub fn train_with_config(
        &mut self,
        config: TrainingConfig,
        saa: &scenario::ScenarioTree,
    ) -> Result<TrainingResult, String>;
}

/// Configuration for SDDP training.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// Number of training iterations.
    pub num_iterations: usize,
    /// Number of forward passes per iteration.
    pub num_forward_passes: usize,
    /// Enable cut selection (dominated cut removal).
    pub enable_cut_selection: bool,
    /// Preserve forward pass trajectories for export.
    pub preserve_forward_detail: bool,
    /// Preserve backward pass details for export.
    pub preserve_backward_detail: bool,
    /// Interval for model rebuild (0 = disabled).
    /// Rebuilding reclaims HiGHS internal memory at the cost of ~1% overhead.
    pub model_rebuild_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_iterations: 100,
            num_forward_passes: 10,
            enable_cut_selection: true,
            preserve_forward_detail: false,
            preserve_backward_detail: false,
            model_rebuild_interval: 100,  // Rebuild every 100 iterations
        }
    }
}

impl TrainingConfig {
    /// Create config with rebuild disabled.
    pub fn without_rebuild(mut self) -> Self {
        self.model_rebuild_interval = 0;
        self
    }
    
    /// Set rebuild interval.
    pub fn with_rebuild_interval(mut self, interval: usize) -> Self {
        self.model_rebuild_interval = interval;
        self
    }
}
```

### Backward Compatibility

Keep existing `train()` signature, delegate to `train_with_config()`:

```rust
pub fn train(
    &mut self,
    num_iterations: usize,
    num_forward_passes: usize,
    enable_cut_selection: bool,
    saa: &scenario::ScenarioTree,
    preserve_forward_detail: bool,
    preserve_backward_detail: bool,
) -> Result<TrainingResult, String> {
    let config = TrainingConfig {
        num_iterations,
        num_forward_passes,
        enable_cut_selection,
        preserve_forward_detail,
        preserve_backward_detail,
        model_rebuild_interval: 100,  // Default enabled
    };
    self.train_with_config(config, saa)
}
```

---

## Acceptance Criteria

- [ ] `TrainingConfig` struct defined
- [ ] `train_with_config()` implemented
- [ ] Existing `train()` preserved with default config
- [ ] `model_rebuild_interval = 0` disables rebuild
- [ ] Tests for configuration options
- [ ] Documentation for new API

---

## Implementation Guide

### Suggested Approach

1. **Add `TrainingConfig` struct** to `src/sddp/mod.rs`

2. **Implement `train_with_config()`**:
   - Move current `train()` logic into this method
   - Replace hardcoded interval with `config.model_rebuild_interval`
   - Add check for `interval == 0` to disable rebuild

3. **Update `train()` to delegate**:
   - Create default config from parameters
   - Call `train_with_config()`

4. **Add tests** for configuration options

### Training Loop Change

```rust
// In train_with_config():

// Periodic model rebuild for memory reclaim
if config.model_rebuild_interval > 0 
    && iteration % config.model_rebuild_interval == 0 
    && iteration < config.num_iterations 
{
    // ... rebuild logic from T-107 ...
}
```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/sddp/mod.rs` | Add `TrainingConfig`, `train_with_config()`, update `train()` |

---

## Testing Requirements

### Unit Tests

- [ ] `TrainingConfig::default()` has expected values
- [ ] `without_rebuild()` sets interval to 0
- [ ] `with_rebuild_interval()` sets correct value

### Integration Tests

- [ ] Training with `model_rebuild_interval = 0` doesn't rebuild
- [ ] Training with `model_rebuild_interval = 10` rebuilds multiple times
- [ ] Existing `train()` calls work unchanged

---

## Documentation Requirements

- [ ] Doc comments for `TrainingConfig`
- [ ] Doc comments for `train_with_config()`
- [ ] Example usage in doc comment
- [ ] Update any README/guide that shows training API

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Straightforward refactoring, low risk

---

## Definition of Done

- [ ] `TrainingConfig` implemented
- [ ] `train_with_config()` working
- [ ] Backward compatibility preserved
- [ ] Tests passing
- [ ] Documentation complete
- [ ] PR merged
