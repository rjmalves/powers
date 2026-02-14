---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §15 (15.1-15.4)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: null
    description: ""
---

# Cut Management Implementation

## Purpose

This spec defines the POWE.RS cut management implementation: the Future Cost Function (FCF) data structure, cut selection strategies for maintaining tractability, binary serialization for checkpoints, and cross-rank cut synchronization via MPI.

## 1. Future Cost Function Structure

The FCF is stored as a collection of cuts per stage:

```rust
/// Future Cost Function: piecewise-linear approximation of cost-to-go
pub struct FutureCostFunction {
    /// Cuts indexed by stage
    cuts_by_stage: Vec<CutPool>,

    /// State dimension information
    n_storage_vars: usize,
    n_inflow_vars: usize,

    /// Global statistics
    total_cuts: usize,
    cuts_added_this_iteration: usize,
}

/// Pool of cuts for a single stage
pub struct CutPool {
    stage: StageId,
    cuts: Vec<Cut>,
    selection_strategy: CutSelectionStrategy,
    max_cuts: Option<usize>,
}

impl FutureCostFunction {
    /// Create empty FCF
    pub fn new(n_stages: usize, n_storage: usize, n_inflow: usize) -> Self {
        let cuts_by_stage = (0..n_stages)
            .map(|s| CutPool::new(StageId(s)))
            .collect();

        Self {
            cuts_by_stage,
            n_storage_vars: n_storage,
            n_inflow_vars: n_inflow,
            total_cuts: 0,
            cuts_added_this_iteration: 0,
        }
    }

    /// Add a cut to the specified stage
    pub fn add_cut(&mut self, stage: StageId, cut: Cut) {
        self.cuts_by_stage[stage.0].add(cut);
        self.total_cuts += 1;
        self.cuts_added_this_iteration += 1;
    }

    /// Get cuts for a stage (for LP construction)
    pub fn get_cuts(&self, stage: StageId) -> &[Cut] {
        self.cuts_by_stage[stage.0].active_cuts()
    }

    /// Evaluate lower bound on θ at a state
    pub fn evaluate(&self, stage: StageId, state: &StatePoint) -> f64 {
        self.cuts_by_stage[stage.0]
            .cuts
            .iter()
            .map(|cut| cut.evaluate(state))
            .fold(f64::NEG_INFINITY, f64::max)
    }
}
```

## 2. Cut Selection Strategies

As iterations progress, the number of cuts can become unwieldy. Cut selection strategies maintain tractability:

```rust
pub enum CutSelectionStrategy {
    /// Keep all cuts (no selection)
    KeepAll,

    /// Keep most recently generated cuts
    MostRecent { max_cuts: usize },

    /// Keep cuts that were binding most often
    MostActive {
        max_cuts: usize,
        decay_factor: f64,  // Weight recent activity more
    },

    /// Level-1 cuts: statistical significance test
    Level1 {
        max_cuts: usize,
        significance: f64,  // e.g., 0.05
    },

    /// Hybrid: keep recent + most active
    Hybrid {
        recent_fraction: f64,  // e.g., 0.3
        max_cuts: usize,
    },
}

impl CutPool {
    /// Add cut and potentially prune
    pub fn add(&mut self, cut: Cut) {
        self.cuts.push(cut);
        self.maybe_prune();
    }

    /// Prune cuts based on selection strategy
    fn maybe_prune(&mut self) {
        if let Some(max) = self.max_cuts {
            if self.cuts.len() > max {
                match &self.selection_strategy {
                    CutSelectionStrategy::MostRecent { .. } => {
                        // Keep newest cuts
                        let drain_count = self.cuts.len() - max;
                        self.cuts.drain(0..drain_count);
                    }

                    CutSelectionStrategy::MostActive { decay_factor, .. } => {
                        // Sort by weighted activity, keep top
                        self.cuts.sort_by(|a, b| {
                            let score_a = a.weighted_activity(*decay_factor);
                            let score_b = b.weighted_activity(*decay_factor);
                            score_b.partial_cmp(&score_a).unwrap()
                        });
                        self.cuts.truncate(max);
                    }

                    CutSelectionStrategy::Hybrid { recent_fraction, .. } => {
                        let n_recent = (max as f64 * recent_fraction) as usize;
                        let n_active = max - n_recent;

                        // Partition: recent (by iteration) vs active (by binding count)
                        self.cuts.sort_by_key(|c| std::cmp::Reverse(c.iteration));
                        let recent: Vec<_> = self.cuts.drain(0..n_recent).collect();

                        self.cuts.sort_by_key(|c| std::cmp::Reverse(c.active_count));
                        self.cuts.truncate(n_active);

                        self.cuts.extend(recent);
                    }

                    _ => {}
                }
            }
        }
    }

    /// Update activity counters after LP solve
    pub fn update_activity(&mut self, binding_cuts: &[usize]) {
        for &idx in binding_cuts {
            if idx < self.cuts.len() {
                self.cuts[idx].active_count += 1;
            }
        }
    }

    /// Get cuts for LP (may return subset based on strategy)
    pub fn active_cuts(&self) -> &[Cut] {
        &self.cuts
    }
}
```

## 3. Cut Serialization for Checkpoints

```rust
/// Binary format for cut storage (efficient I/O)
impl Cut {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            8 +  // stage (u64)
            8 +  // intercept (f64)
            8 +  // iteration (u64)
            8 +  // active_count (u64)
            8 +  // n_storage (u64)
            8 +  // n_inflow (u64)
            self.storage_coef.len() * 8 +
            self.inflow_coef.len() * 8
        );

        buf.extend(&(self.stage.0 as u64).to_le_bytes());
        buf.extend(&self.intercept.to_le_bytes());
        buf.extend(&(self.iteration as u64).to_le_bytes());
        buf.extend(&(self.active_count as u64).to_le_bytes());
        buf.extend(&(self.storage_coef.len() as u64).to_le_bytes());
        buf.extend(&(self.inflow_coef.len() as u64).to_le_bytes());

        for &c in &self.storage_coef {
            buf.extend(&c.to_le_bytes());
        }
        for &c in &self.inflow_coef {
            buf.extend(&c.to_le_bytes());
        }

        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut offset = 0;

        let stage = StageId(read_u64(bytes, &mut offset) as usize);
        let intercept = read_f64(bytes, &mut offset);
        let iteration = read_u64(bytes, &mut offset) as usize;
        let active_count = read_u64(bytes, &mut offset) as usize;
        let n_storage = read_u64(bytes, &mut offset) as usize;
        let n_inflow = read_u64(bytes, &mut offset) as usize;

        let storage_coef = (0..n_storage)
            .map(|_| read_f64(bytes, &mut offset))
            .collect();
        let inflow_coef = (0..n_inflow)
            .map(|_| read_f64(bytes, &mut offset))
            .collect();

        Self {
            stage,
            intercept,
            storage_coef,
            inflow_coef,
            iteration,
            active_count,
        }
    }
}

/// Stage cuts file format
/// Header: magic (8 bytes) + version (4 bytes) + n_cuts (4 bytes)
/// Body: [cut_size (4 bytes) + cut_bytes]...
pub fn save_stage_cuts(path: &Path, cuts: &[Cut]) -> io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);

    // Header
    file.write_all(b"PWRSCUTS")?;  // Magic
    file.write_all(&1u32.to_le_bytes())?;  // Version
    file.write_all(&(cuts.len() as u32).to_le_bytes())?;  // Count

    // Cuts
    for cut in cuts {
        let bytes = cut.to_bytes();
        file.write_all(&(bytes.len() as u32).to_le_bytes())?;
        file.write_all(&bytes)?;
    }

    Ok(())
}
```

## 4. Cut Synchronization Across Ranks

```rust
impl<R: RiskMeasure, C: CutFormulation, H: HorizonMode> TrainingLoop<R, C, H> {
    /// Synchronize newly generated cuts across all ranks
    fn sync_cuts(&mut self) {
        // Serialize local new cuts
        let local_cuts: Vec<u8> = self.fcf.drain_new_cuts()
            .into_iter()
            .flat_map(|c| c.to_bytes())
            .collect();

        // Gather sizes from all ranks
        let local_size = local_cuts.len() as i32;
        let mut sizes = vec![0i32; self.comm.size() as usize];
        self.comm.all_gather(&local_size, &mut sizes);

        // Compute displacements
        let mut displs = vec![0i32; self.comm.size() as usize];
        for i in 1..displs.len() {
            displs[i] = displs[i-1] + sizes[i-1];
        }
        let total_size = displs.last().unwrap() + sizes.last().unwrap();

        // Gather all cuts
        let mut all_cuts_bytes = vec![0u8; total_size as usize];
        self.comm.all_gather_varcount(&local_cuts, &mut all_cuts_bytes, &sizes, &displs);

        // Deserialize and add to FCF (skipping own cuts, already added)
        let my_rank = self.comm.rank();
        for (rank, &size) in sizes.iter().enumerate() {
            if rank != my_rank as usize && size > 0 {
                let start = displs[rank] as usize;
                let end = start + size as usize;
                let cuts = deserialize_cuts(&all_cuts_bytes[start..end]);
                for cut in cuts {
                    self.fcf.add_cut(cut.stage, cut);
                }
            }
        }
    }
}
```

## Cross-References

- [Cut Management](../01-math/cut-management.md) — Mathematical foundations for cut coefficients, selection theory, and dominance criteria
- [Training Loop](training-loop.md) — The SDDP training loop that drives cut generation via forward and backward passes
- [Work Distribution](../04-hpc/work-distribution.md) — MPI+OpenMP parallelism patterns for distributing cut generation and synchronization across ranks
