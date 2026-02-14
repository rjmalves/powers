---
status: draft
review_priority: 2-high
source_sections:
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §17 (17.1-17.4)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §18 (18.1-18.3)"
  - "PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §19 (19.1-19.4)"
last_reviewed: null
reviewed_by: null
review_notes: ""
change_log:
  - date: 2026-02-14
    description: "Extracted from PROGRAM_ARCHITECTURE_EXECUTION_FLOW.md §17-19"
---

# Simulation Architecture

## Purpose

This spec defines the simulation phase of the POWE.RS SDDP solver: how trained policies are evaluated on large scenario sets, how non-convex operational constraints are handled during simulation, and how results are streamed to Parquet output files across distributed MPI ranks.

## 1. Policy Evaluation Mode

### 1.1 Simulation Overview

The simulation phase evaluates the trained SDDP policy on a large number of scenarios to assess:

1. **Policy quality**: Expected cost, variance, and risk metrics
2. **Operational behavior**: Storage trajectories, generation mix, deficit frequency
3. **Robustness**: Performance across diverse hydrological conditions

```
┌───────────────────────────────────────────────────────────────┐
│                    Simulation Architecture                     │
├───────────────────────────────────────────────────────────────┤
│  Input: Trained FCF (cuts), Simulation scenarios              │
│                                                               │
│  SCENARIO GENERATION                                          │
│  Monte Carlo | Historical replay | External file | Hybrid    │
│                                                               │
│  PARALLEL EXECUTION                                           │
│  Scenarios statically distributed across MPI ranks            │
│  Each rank solves LP sequence for assigned scenarios          │
│                                                               │
│  PER-SCENARIO (for stage t = 1..T):                           │
│  1. Realize uncertainties  2. Solve stage LP with FCF cuts    │
│  3. Stream results to output  4. Non-convexities if enabled   │
│                                                               │
│  OUTPUT AGGREGATION                                           │
│  Streaming write | Statistics across scenarios | Risk metrics │
└───────────────────────────────────────────────────────────────┘
```

### 1.2 Simulation Configuration

```rust
pub struct SimulationConfig {
    pub enabled: bool,
    pub n_scenarios: usize,                   // e.g., 2000
    pub scenario_source: ScenarioSource,
    pub output: SimulationOutputConfig,
    pub non_convex: Option<NonConvexConfig>,
    pub chunk_size: usize,                    // Scenarios per work unit
}

pub enum ScenarioSource {
    MonteCarlo { seed: u64 },
    Historical { start_year: u32, end_year: u32 },
    External { path: PathBuf, format: ScenarioFormat },
    Hybrid { historical_weight: f64, synthetic_tail_years: u32 },
}

pub struct SimulationOutputConfig {
    pub detail: OutputDetail,
    pub streaming: bool,
    pub compress: bool,
    pub variables: Vec<OutputVariable>,
}

pub enum OutputDetail {
    Summary,     // Only aggregate statistics
    StageLevel,  // Per-stage aggregates
    Full,        // Per-scenario, per-stage details
}
```

### 1.3 Simulation Execution

```rust
pub struct SimulationRunner {
    fcf: Arc<FutureCostFunction>,
    config: SimulationConfig,
    comm: WorldCommunicator,
    output_writer: OutputWriter,
}

impl SimulationRunner {
    pub fn run(&mut self, case_data: &CaseData) -> SimulationResult {
        let scenarios = self.prepare_scenarios(case_data);
        let my_scenarios = self.distribute_scenarios(&scenarios);
        let mut local_stats = SimulationStats::new();

        for chunk in my_scenarios.chunks(self.config.chunk_size) {
            let chunk_results: Vec<ScenarioResult> = chunk
                .par_iter()
                .map(|scenario| self.simulate_scenario(case_data, scenario))
                .collect();

            for result in &chunk_results {
                self.output_writer.write_scenario(result);
            }
            for result in chunk_results {
                local_stats.accumulate(&result);
            }
        }

        let global_stats = self.aggregate_stats(&local_stats);
        SimulationResult {
            stats: global_stats,
            output_path: self.output_writer.finalize(),
        }
    }

    fn simulate_scenario(
        &self, case_data: &CaseData, scenario: &Scenario,
    ) -> ScenarioResult {
        let mut state = case_data.initial_state();
        let mut stage_results = Vec::with_capacity(case_data.num_stages());
        let mut total_cost = 0.0;

        for (stage_idx, stage) in case_data.stages.iter().enumerate() {
            let inflows = scenario.inflows_at_stage(stage_idx);
            state.set_inflows(&inflows);

            let lp = self.build_simulation_lp(case_data, stage, &state);
            let solution = lp.solve().expect("Simulation LP should be feasible");

            let final_solution = if let Some(nc_config) = &self.config.non_convex {
                self.apply_non_convex(case_data, stage, &solution, nc_config)
            } else {
                solution
            };

            let stage_result = StageResult::from_solution(&final_solution, stage);
            total_cost += stage_result.cost;
            stage_results.push(stage_result);
            state = final_solution.extract_end_state();
        }

        ScenarioResult { scenario_id: scenario.id, total_cost, stage_results }
    }
}
```

### 1.4 Simulation Statistics

```rust
pub struct SimulationStats {
    pub n_scenarios: usize,
    pub total_cost_sum: f64,
    pub total_cost_sum_sq: f64,
    pub min_cost: f64,
    pub max_cost: f64,
    costs: Vec<f64>,               // Sorted costs for percentiles (kept on rank 0)
    pub deficit_scenarios: usize,
    pub deficit_mwh_sum: f64,
    pub spill_mwh_sum: f64,
    stage_stats: Option<Vec<StageStats>>,
}

impl SimulationStats {
    pub fn accumulate(&mut self, result: &ScenarioResult) {
        self.n_scenarios += 1;
        self.total_cost_sum += result.total_cost;
        self.total_cost_sum_sq += result.total_cost.powi(2);
        self.min_cost = self.min_cost.min(result.total_cost);
        self.max_cost = self.max_cost.max(result.total_cost);
        self.costs.push(result.total_cost);

        let has_deficit = result.stage_results.iter().any(|s| s.deficit > 0.0);
        if has_deficit {
            self.deficit_scenarios += 1;
            self.deficit_mwh_sum += result.stage_results.iter()
                .map(|s| s.deficit).sum::<f64>();
        }
    }

    pub fn mean_cost(&self) -> f64 { self.total_cost_sum / self.n_scenarios as f64 }

    pub fn std_cost(&self) -> f64 {
        let mean = self.mean_cost();
        ((self.total_cost_sum_sq / self.n_scenarios as f64) - mean.powi(2)).sqrt()
    }

    pub fn cvar(&self, alpha: f64) -> f64 {
        let mut sorted = self.costs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cutoff_idx = ((1.0 - alpha) * self.n_scenarios as f64) as usize;
        let tail = &sorted[cutoff_idx..];
        tail.iter().sum::<f64>() / tail.len() as f64
    }
}
```

## 2. Non-Convex Extensions

### 2.1 Non-Convexity Sources

SDDP produces an optimal policy for the convex relaxation. Simulation can incorporate non-convex operational constraints through post-processing:

| Non-Convexity Source      | Modeling Approach                          |
| ------------------------- | ------------------------------------------ |
| Thermal unit commitment   | MIP with binary on/off variables           |
| Minimum generation        | Big-M constraints or indicator constraints |
| Startup/shutdown costs    | Multi-period linking constraints           |
| Transmission switching    | Binary line switching variables            |
| Head-dependent generation | Piecewise-linear or iterative refinement   |
| Forbidden operating zones | Disjunctive constraints                    |

**Strategy**: Solve LP for policy decisions, then refine with MIP/heuristics.

### 2.2 Non-Convex Processing Pipeline

```rust
pub struct NonConvexConfig {
    pub thermal_commitment: Option<ThermalCommitmentConfig>,
    pub transmission_switching: Option<TransmissionSwitchingConfig>,
    pub head_dependent: Option<HeadDependentConfig>,
    pub mip_settings: MipSettings,
}

pub struct ThermalCommitmentConfig {
    pub min_updown_time: bool,
    pub startup_costs: bool,
    pub time_limit_seconds: f64,
    pub gap_tolerance: f64,
}

impl SimulationRunner {
    fn apply_non_convex(
        &self, case_data: &CaseData, stage: &Stage,
        lp_solution: &LpSolution, config: &NonConvexConfig,
    ) -> Solution {
        let mut refined = lp_solution.clone();
        if let Some(tc_config) = &config.thermal_commitment {
            refined = self.refine_thermal_commitment(case_data, stage, &refined, tc_config);
        }
        if let Some(hd_config) = &config.head_dependent {
            refined = self.refine_head_dependent(case_data, stage, &refined, hd_config);
        }
        refined
    }

    fn refine_thermal_commitment(
        &self, case_data: &CaseData, stage: &Stage,
        lp_solution: &LpSolution, config: &ThermalCommitmentConfig,
    ) -> Solution {
        let mut mip = MipBuilder::new();
        for thermal in &case_data.thermals {
            let lp_gen = lp_solution.get_generation(thermal.id);
            let commit = mip.add_binary(&format!("commit_{}", thermal.id));
            let gen = mip.add_continuous(
                &format!("gen_{}", thermal.id), 0.0, thermal.max_generation,
            );
            // gen ∈ [min_gen × commit, max_gen × commit]
            mip.add_constraint(gen - thermal.min_generation * commit >= 0.0);
            mip.add_constraint(gen - thermal.max_generation * commit <= 0.0);
            if lp_gen > 0.0 {
                mip.set_start_value(commit, 1.0);
                mip.set_start_value(gen, lp_gen);
            }
        }
        mip.set_objective(/* deviation terms */);
        let mip_solution = mip.solve_with_timeout(config.time_limit_seconds);
        Self::merge_mip_solution(lp_solution, &mip_solution)
    }
}
```

### 2.3 Iterative Head-Dependent Refinement

```rust
pub struct HeadDependentConfig {
    pub max_iterations: usize,
    pub head_tolerance: f64,
    pub method: HeadDependentMethod,
}

pub enum HeadDependentMethod {
    FixedPoint,                       // Average of start and end storage
    EndOfStage,                       // Use end-of-stage storage for head
    AverageStorage { weight: f64 },   // Weighted interpolation
}

impl SimulationRunner {
    fn refine_head_dependent(
        &self, case_data: &CaseData, stage: &Stage,
        initial_solution: &LpSolution, config: &HeadDependentConfig,
    ) -> Solution {
        let mut solution = initial_solution.clone();
        for _iter in 0..config.max_iterations {
            let heads: Vec<f64> = case_data.hydros.iter()
                .map(|h| self.compute_head(h, &solution, config))
                .collect();
            let efficiencies: Vec<f64> = case_data.hydros.iter()
                .zip(heads.iter())
                .map(|(h, &head)| h.efficiency_at_head(head))
                .collect();
            let lp = self.build_lp_with_efficiencies(
                case_data, stage, &solution, &efficiencies
            );
            let new_solution = lp.solve().expect("LP should be feasible");
            let max_head_change = heads.iter()
                .zip(self.compute_heads(&new_solution, case_data, config))
                .map(|(&old, new)| (old - new).abs())
                .fold(0.0, f64::max);
            if max_head_change < config.head_tolerance { return new_solution; }
            solution = new_solution;
        }
        solution
    }

    fn compute_head(
        &self, hydro: &Hydro, solution: &LpSolution, config: &HeadDependentConfig,
    ) -> f64 {
        let (start, end) = (
            solution.get_initial_storage(hydro.id),
            solution.get_storage(hydro.id),
        );
        let storage = match config.method {
            HeadDependentMethod::FixedPoint => (start + end) / 2.0,
            HeadDependentMethod::EndOfStage => end,
            HeadDependentMethod::AverageStorage { weight } => {
                weight * start + (1.0 - weight) * end
            }
        };
        hydro.head_from_storage(storage)
    }
}
```

## 3. Output Streaming

### 3.1 Streaming Architecture

With potentially thousands of scenarios, storing all results in memory is impractical. The output writer streams results to disk as they are computed via a background I/O thread connected by a bounded channel:

![Output Streaming Pipeline](../../diagrams/exports/svg/data/output-streaming-pipeline.svg)

### 3.2 Output Writer Implementation

```rust
pub struct OutputWriter {
    sender: Sender<OutputMessage>,
    writer_handle: Option<JoinHandle<()>>,
    config: SimulationOutputConfig,
}

enum OutputMessage {
    ScenarioResult(ScenarioResult),
    Flush,
    Finish,
}

impl OutputWriter {
    pub fn new(output_dir: &Path, config: SimulationOutputConfig) -> Self {
        let (sender, receiver) = bounded(100);
        let writer_handle = std::thread::spawn(move || {
            let mut writer = ParquetWriter::new(output_dir);
            loop {
                match receiver.recv() {
                    Ok(OutputMessage::ScenarioResult(result)) => writer.write_scenario(&result),
                    Ok(OutputMessage::Flush) => writer.flush(),
                    Ok(OutputMessage::Finish) => { writer.finalize(); break; }
                    Err(_) => break,
                }
            }
        });
        Self { sender, writer_handle: Some(writer_handle), config }
    }

    pub fn write_scenario(&self, result: &ScenarioResult) {
        let filtered = match self.config.detail {
            OutputDetail::Summary => result.summary_only(),
            OutputDetail::StageLevel => result.stage_aggregates(),
            OutputDetail::Full => result.filter_variables(&self.config.variables),
        };
        self.sender.send(OutputMessage::ScenarioResult(filtered))
            .expect("Writer thread should be alive");
    }

    pub fn finalize(mut self) -> PathBuf {
        self.sender.send(OutputMessage::Finish).ok();
        if let Some(handle) = self.writer_handle.take() {
            handle.join().expect("Writer thread should complete");
        }
        self.output_path()
    }
}
```

### 3.3 Parquet Output Schema

Output file: `results/scenario_results.parquet`

| Column                    | Type    | Detail Level | Description                       |
| ------------------------- | ------- | ------------ | --------------------------------- |
| `scenario_id`             | INT32   | All          | Scenario identifier               |
| `stage_id`                | INT32   | Stage+       | Stage identifier                  |
| `total_cost`              | DOUBLE  | All          | Total scenario cost               |
| `immediate_cost`          | DOUBLE  | Stage+       | Stage immediate cost              |
| `future_cost`             | DOUBLE  | Stage+       | Stage future cost estimate        |
| `deficit_mwh`             | DOUBLE  | Stage+       | Total deficit in MWh              |
| `spill_mwh`               | DOUBLE  | Stage+       | Total spill in MWh                |
| `hydro_{id}_storage`      | DOUBLE  | Full         | End-of-stage storage              |
| `hydro_{id}_generation`   | DOUBLE  | Full         | Hydro generation                  |
| `hydro_{id}_turbined`     | DOUBLE  | Full         | Turbined outflow                  |
| `hydro_{id}_spilled`      | DOUBLE  | Full         | Spilled outflow                   |
| `thermal_{id}_generation` | DOUBLE  | Full         | Thermal generation                |
| `thermal_{id}_committed`  | BOOLEAN | Full (UC)    | Commitment status (if UC enabled) |
| `bus_{id}_deficit`        | DOUBLE  | Full         | Bus deficit                       |
| `bus_{id}_marginal_cost`  | DOUBLE  | Full         | Bus marginal cost                 |

```rust
impl ParquetWriter {
    pub fn new(output_dir: &Path, schema: &SchemaRef, row_group_size: usize) -> Self {
        let file = File::create(output_dir.join("scenario_results.parquet"))
            .expect("Could not create output file");
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
            .set_dictionary_enabled(true)
            .build();
        let writer = SerializedFileWriter::new(file, schema.clone(), Arc::new(props))
            .expect("Could not create Parquet writer");
        Self { writer, row_group_builder: RowGroupBuilder::new(schema),
               rows_in_group: 0, row_group_size }
    }

    pub fn write_scenario(&mut self, result: &ScenarioResult) {
        for (stage_idx, stage_result) in result.stage_results.iter().enumerate() {
            self.row_group_builder.append_row(
                result.scenario_id, stage_idx as i32, result.total_cost, stage_result,
            );
            self.rows_in_group += 1;
            if self.rows_in_group >= self.row_group_size { self.flush_row_group(); }
        }
    }

    pub fn finalize(mut self) {
        self.flush_row_group();
        self.writer.close().expect("Could not close Parquet file");
    }
}
```

### 3.4 Distributed Output Coordination

Each MPI rank writes simulation results independently. Two modes are supported:

- **PerRank**: Each rank writes to its own file (`scenarios_0000.parquet`, `scenarios_0001.parquet`, ...)
- **Collected**: All results are sent to rank 0, which writes a single `all_scenarios.parquet`

```rust
impl SimulationRunner {
    fn setup_output_writer(&self, case_data: &CaseData) -> OutputWriter {
        let output_dir = case_data.output_dir().join("simulation");
        std::fs::create_dir_all(&output_dir).expect("Could not create output dir");

        match self.config.output.distributed_mode {
            DistributedOutputMode::PerRank => {
                let rank_file = output_dir.join(
                    format!("scenarios_{:04}.parquet", self.comm.rank())
                );
                OutputWriter::new(&rank_file, self.config.output.clone())
            }
            DistributedOutputMode::Collected => {
                if self.comm.rank() == 0 {
                    OutputWriter::new(
                        &output_dir.join("all_scenarios.parquet"),
                        self.config.output.clone(),
                    )
                } else {
                    OutputWriter::new_sender(0, &self.comm)
                }
            }
        }
    }
}
```

## Cross-References

- [CLI and Lifecycle](./cli-and-lifecycle.md) — execution phases and conditional simulation mode
- [Risk Measures](../01-math/risk-measures.md) — mathematical foundation for CVaR and risk metrics computed during simulation
- [Extension Points](./extension-points.md) — trait abstractions that parameterize simulation behavior
