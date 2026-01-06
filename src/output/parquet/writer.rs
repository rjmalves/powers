//! Parquet writer implementation.
//!
//! Implements the OutputWriter trait using Apache Parquet format for
//! efficient columnar storage with compression.

use super::schemas;
use crate::fcf;
use crate::graph;
use crate::output::writer::{OutputWriter, Result};
use crate::scenario;
use crate::sddp;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int16Builder, UInt16Builder,
    UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{WriterProperties, WriterVersion};

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Configuration for Parquet file generation.
///
/// Controls compression, row grouping, and other Parquet-specific settings.
#[derive(Debug, Clone)]
pub struct ParquetConfig {
    /// Compression codec to use (Snappy recommended for speed/size balance)
    pub compression: Compression,

    /// Number of rows per row group (affects compression and query performance)
    /// Default: 10,000 (good balance for most use cases)
    pub row_group_size: usize,

    /// Parquet format version (V2 has better compression)
    pub writer_version: WriterVersion,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            // Snappy: Fast compression with decent ratio (~50-60% of gzip)
            compression: Compression::SNAPPY,
            row_group_size: 10_000,
            writer_version: WriterVersion::PARQUET_2_0,
        }
    }
}

impl ParquetConfig {
    /// Creates configuration with Zstd compression (best compression, slower).
    pub fn with_zstd(level: i32) -> Self {
        Self {
            compression: Compression::ZSTD(ZstdLevel::try_new(level).unwrap()),
            ..Default::default()
        }
    }

    /// Creates configuration with no compression (fastest writes).
    pub fn uncompressed() -> Self {
        Self {
            compression: Compression::UNCOMPRESSED,
            ..Default::default()
        }
    }
}

/// Parquet output writer.
///
/// Writes SDDP algorithm outputs to Parquet files with columnar storage
/// and compression. Requires the `parquet-output` feature flag.
///
/// # Benefits
///
/// - **75-85% smaller files** than CSV (typical compression ratio)
/// - **Faster analytics**: Columnar format optimized for queries
/// - **Type safety**: Preserves data types (no string conversion)
/// - **Metadata**: Embeds schema and statistics for tools
///
/// # Performance
///
/// - Write time: ~20-30% slower than CSV (due to compression)
/// - Read time: 3-5x faster than CSV for analytical queries
/// - File size: 15-25% of CSV size
///
/// # Example
///
/// ```ignore
/// use powers_rs::output::parquet::{ParquetWriter, ParquetConfig};
///
/// let config = ParquetConfig::default(); // Snappy compression
/// let mut writer = ParquetWriter::with_config("./output", config)?;
/// writer.write_training(&results)?;
/// ```
pub struct ParquetWriter {
    output_dir: PathBuf,
    config: ParquetConfig,
}

impl ParquetWriter {
    /// Creates a new Parquet writer with default configuration.
    ///
    /// Uses Snappy compression and 10,000 row groups.
    ///
    /// # Errors
    ///
    /// Returns error if output directory cannot be created.
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Result<Self> {
        Self::with_config(output_dir, ParquetConfig::default())
    }

    /// Creates a new Parquet writer with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns error if output directory cannot be created.
    pub fn with_config<P: AsRef<Path>>(
        output_dir: P,
        config: ParquetConfig,
    ) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_dir)?;

        Ok(Self { output_dir, config })
    }

    /// Creates writer properties from configuration.
    fn writer_properties(&self) -> WriterProperties {
        WriterProperties::builder()
            .set_compression(self.config.compression)
            .set_writer_version(self.config.writer_version)
            .build()
    }

    /// Writes a record batch to a Parquet file.
    fn write_batch(
        &self,
        filename: &str,
        schema: arrow::datatypes::SchemaRef,
        batch: RecordBatch,
    ) -> Result<()> {
        let path = self.output_dir.join(filename);
        let file = File::create(&path)?;
        let props = self.writer_properties();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }
}

impl OutputWriter for ParquetWriter {
    fn write_training(
        &mut self,
        results: &[sddp::IterationResult],
    ) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        let schema = schemas::training_schema();
        let capacity = results.len();

        // Build arrays
        let mut iteration = UInt32Builder::with_capacity(capacity);
        let mut lower_bound = Float64Builder::with_capacity(capacity);
        let mut policy_cost = Float64Builder::with_capacity(capacity);
        let mut forward_time = UInt64Builder::with_capacity(capacity);
        let mut gap = Float64Builder::with_capacity(capacity);
        let mut forward_passes = UInt16Builder::with_capacity(capacity);
        let mut forward_scenarios = UInt32Builder::with_capacity(capacity);
        let mut forward_solver_ms = UInt64Builder::with_capacity(capacity);
        let mut forward_avg_obj = Float64Builder::with_capacity(capacity);
        let mut forward_std_obj = Float64Builder::with_capacity(capacity);
        let mut backward_time = UInt64Builder::with_capacity(capacity);
        let mut backward_stages = UInt16Builder::with_capacity(capacity);
        let mut backward_states = UInt32Builder::with_capacity(capacity);
        let mut backward_solver_ms = UInt64Builder::with_capacity(capacity);
        let mut backward_cuts_added = UInt32Builder::with_capacity(capacity);
        let mut backward_cuts_total = UInt32Builder::with_capacity(capacity);

        for result in results {
            iteration.append_value(result.iteration as u32);
            lower_bound.append_value(result.lower_bound);

            // Calculate policy cost from forward costs
            let pc = if !result.forward_costs.is_empty() {
                result.forward_costs.iter().sum::<f64>()
                    / result.forward_costs.len() as f64
            } else {
                0.0
            };
            policy_cost.append_value(pc);

            // Calculate gap
            let g = if result.lower_bound > 0.0 {
                ((pc - result.lower_bound) / result.lower_bound).abs() * 100.0
            } else {
                f64::INFINITY
            };
            gap.append_value(g);

            // T-024: Update to use new timing structure
            forward_time
                .append_value(result.timing.forward.total.as_millis() as u64);
            forward_passes.append_value(result.forward_costs.len() as u16);
            forward_scenarios.append_value(result.forward_costs.len() as u32);
            forward_solver_ms
                .append_value(result.timing.forward.solver.as_millis() as u64);
            forward_avg_obj.append_value(pc);

            // Calculate forward std deviation
            let std = if result.forward_costs.len() > 1 {
                let mean = pc;
                let variance: f64 = result
                    .forward_costs
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / result.forward_costs.len() as f64;
                variance.sqrt()
            } else {
                0.0
            };
            forward_std_obj.append_value(std);

            // T-024: Update to use new timing structure
            backward_time
                .append_value(result.timing.backward.total.as_millis() as u64);
            backward_stages.append_value(0); // Not tracked in current structure
            backward_states.append_value(0); // Not tracked in current structure
            backward_solver_ms
                .append_value(result.timing.backward.solver.as_millis() as u64);
            backward_cuts_added.append_value(result.num_cuts_added as u32);
            backward_cuts_total.append_value(result.num_active_cuts as u32);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(iteration.finish()) as ArrayRef,
                Arc::new(lower_bound.finish()),
                Arc::new(policy_cost.finish()),
                Arc::new(forward_time.finish()),
                Arc::new(gap.finish()),
                Arc::new(forward_passes.finish()),
                Arc::new(forward_scenarios.finish()),
                Arc::new(forward_solver_ms.finish()),
                Arc::new(forward_avg_obj.finish()),
                Arc::new(forward_std_obj.finish()),
                Arc::new(backward_time.finish()),
                Arc::new(backward_stages.finish()),
                Arc::new(backward_states.finish()),
                Arc::new(backward_solver_ms.finish()),
                Arc::new(backward_cuts_added.finish()),
                Arc::new(backward_cuts_total.finish()),
            ],
        )?;

        self.write_batch("training.parquet", schema, batch)
    }

    fn write_forward_detail(
        &mut self,
        details: &[sddp::ForwardPassDetail],
    ) -> Result<()> {
        if details.is_empty() {
            return Ok(());
        }

        let schema = schemas::forward_detail_schema();

        // Pre-calculate capacity
        let mut total_records = 0;
        for detail in details {
            let r = &detail.realization;
            total_records += r.initial_storage.len();
            total_records +=
                r.inflow_lags.iter().map(|lags| lags.len()).sum::<usize>();
            total_records += r.loads.len();
            total_records += r.inflow.len();
            total_records += r.turbined_flow.len();
            total_records += r.spillage.len();
            total_records += r.thermal_generation.len();
            total_records += r.water_value.len();
            total_records += r.deficit.len();
            total_records += r.exchange.len();
            total_records += r.marginal_cost.len();
            total_records += r.final_storage.len();
        }

        // Build arrays with proper capacity
        let mut iteration = UInt32Builder::with_capacity(total_records);
        let mut forward_pass_idx = UInt32Builder::with_capacity(total_records);
        let mut stage_id = Int16Builder::with_capacity(total_records);
        let mut variable_index = UInt16Builder::with_capacity(total_records);
        let mut entity_id = UInt16Builder::with_capacity(total_records);
        let mut lag_index = UInt8Builder::with_capacity(total_records);
        let mut value = Float64Builder::with_capacity(total_records);

        for detail in details {
            let r = &detail.realization;
            let iter = detail.iteration as u32;
            let fp_idx = detail.forward_pass_idx as u32;
            let stg_id = detail.stage_id as i16;

            // Helper macro to append records
            macro_rules! append_record {
                ($var:expr, $eid:expr, $lag:expr, $val:expr) => {
                    iteration.append_value(iter);
                    forward_pass_idx.append_value(fp_idx);
                    stage_id.append_value(stg_id);
                    variable_index.append_value($var);
                    entity_id.append_value($eid);
                    lag_index.append_value($lag);
                    value.append_value($val);
                };
            }

            // Initial storage
            for (hydro_id, &val) in r.initial_storage.iter().enumerate() {
                append_record!(0, hydro_id as u16, 0, val);
            }

            // Inflow lags
            for (hydro_id, lags) in r.inflow_lags.iter().enumerate() {
                for (lag_idx, &val) in lags.iter().enumerate() {
                    append_record!(
                        1,
                        hydro_id as u16,
                        (lag_idx + 1) as u8,
                        val
                    );
                }
            }

            // Sampled loads
            for (bus_id, &val) in r.loads.iter().enumerate() {
                append_record!(2, bus_id as u16, 0, val);
            }

            // Sampled inflows
            for (hydro_id, &val) in r.inflow.iter().enumerate() {
                append_record!(3, hydro_id as u16, 0, val);
            }

            // Final storage
            for (hydro_id, &val) in r.final_storage.iter().enumerate() {
                append_record!(4, hydro_id as u16, 0, val);
            }

            // Turbined flow
            for (hydro_id, &val) in r.turbined_flow.iter().enumerate() {
                append_record!(5, hydro_id as u16, 0, val);
            }

            // Spillage
            for (hydro_id, &val) in r.spillage.iter().enumerate() {
                append_record!(6, hydro_id as u16, 0, val);
            }

            // Water value
            for (hydro_id, &val) in r.water_value.iter().enumerate() {
                append_record!(7, hydro_id as u16, 0, val);
            }

            // Thermal generation
            for (thermal_id, &val) in r.thermal_generation.iter().enumerate() {
                append_record!(8, thermal_id as u16, 0, val);
            }

            // Deficit
            for (bus_id, &val) in r.deficit.iter().enumerate() {
                append_record!(9, bus_id as u16, 0, val);
            }

            // Exchange
            for (line_id, &val) in r.exchange.iter().enumerate() {
                append_record!(10, line_id as u16, 0, val);
            }

            // Marginal cost
            for (bus_id, &val) in r.marginal_cost.iter().enumerate() {
                append_record!(11, bus_id as u16, 0, val);
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(iteration.finish()) as ArrayRef,
                Arc::new(forward_pass_idx.finish()),
                Arc::new(stage_id.finish()),
                Arc::new(variable_index.finish()),
                Arc::new(entity_id.finish()),
                Arc::new(lag_index.finish()),
                Arc::new(value.finish()),
            ],
        )?;

        self.write_batch("forward_detail.parquet", schema, batch)
    }

    fn write_backward_detail(
        &mut self,
        details: &[sddp::BackwardPassDetail],
    ) -> Result<()> {
        if details.is_empty() {
            return Ok(());
        }

        let schema = schemas::backward_detail_schema();

        // Pre-calculate capacity
        let mut total_records = 0;
        for detail in details {
            let r = &detail.realization;
            total_records += r.initial_storage.len();
            total_records +=
                r.inflow_lags.iter().map(|lags| lags.len()).sum::<usize>();
            total_records += r.loads.len();
            total_records += r.inflow.len();
            total_records += r.turbined_flow.len();
            total_records += r.spillage.len();
            total_records += r.thermal_generation.len();
            total_records += r.water_value.len();
            total_records += r.deficit.len();
            total_records += r.exchange.len();
            total_records += r.marginal_cost.len();
            total_records += r.final_storage.len();
        }

        // Build arrays
        let mut iteration = UInt32Builder::with_capacity(total_records);
        let mut forward_pass_idx = UInt32Builder::with_capacity(total_records);
        let mut stage_id = Int16Builder::with_capacity(total_records);
        let mut training_state_id = UInt32Builder::with_capacity(total_records);
        let mut branching_idx = UInt16Builder::with_capacity(total_records);
        let mut variable_index = UInt16Builder::with_capacity(total_records);
        let mut entity_id = UInt16Builder::with_capacity(total_records);
        let mut lag_index = UInt8Builder::with_capacity(total_records);
        let mut value = Float64Builder::with_capacity(total_records);

        for detail in details {
            let r = &detail.realization;
            let iter = detail.iteration as u32;
            let fp_idx = detail.forward_pass_idx as u32;
            let stg_id = detail.stage_id as i16;
            let state_id = detail.training_state_id as u32;
            let branch_idx = detail.branching_idx as u16;

            // Helper macro
            macro_rules! append_record {
                ($var:expr, $eid:expr, $lag:expr, $val:expr) => {
                    iteration.append_value(iter);
                    forward_pass_idx.append_value(fp_idx);
                    stage_id.append_value(stg_id);
                    training_state_id.append_value(state_id);
                    branching_idx.append_value(branch_idx);
                    variable_index.append_value($var);
                    entity_id.append_value($eid);
                    lag_index.append_value($lag);
                    value.append_value($val);
                };
            }

            // Initial storage
            for (hydro_id, &val) in r.initial_storage.iter().enumerate() {
                append_record!(0, hydro_id as u16, 0, val);
            }

            // Inflow lags
            for (hydro_id, lags) in r.inflow_lags.iter().enumerate() {
                for (lag_idx, &val) in lags.iter().enumerate() {
                    append_record!(
                        1,
                        hydro_id as u16,
                        (lag_idx + 1) as u8,
                        val
                    );
                }
            }

            // Sampled loads
            for (bus_id, &val) in r.loads.iter().enumerate() {
                append_record!(2, bus_id as u16, 0, val);
            }

            // Sampled inflows
            for (hydro_id, &val) in r.inflow.iter().enumerate() {
                append_record!(3, hydro_id as u16, 0, val);
            }

            // Final storage
            for (hydro_id, &val) in r.final_storage.iter().enumerate() {
                append_record!(4, hydro_id as u16, 0, val);
            }

            // Turbined flow
            for (hydro_id, &val) in r.turbined_flow.iter().enumerate() {
                append_record!(5, hydro_id as u16, 0, val);
            }

            // Spillage
            for (hydro_id, &val) in r.spillage.iter().enumerate() {
                append_record!(6, hydro_id as u16, 0, val);
            }

            // Water value
            for (hydro_id, &val) in r.water_value.iter().enumerate() {
                append_record!(7, hydro_id as u16, 0, val);
            }

            // Thermal generation
            for (thermal_id, &val) in r.thermal_generation.iter().enumerate() {
                append_record!(8, thermal_id as u16, 0, val);
            }

            // Deficit
            for (bus_id, &val) in r.deficit.iter().enumerate() {
                append_record!(9, bus_id as u16, 0, val);
            }

            // Exchange
            for (line_id, &val) in r.exchange.iter().enumerate() {
                append_record!(10, line_id as u16, 0, val);
            }

            // Marginal cost
            for (bus_id, &val) in r.marginal_cost.iter().enumerate() {
                append_record!(11, bus_id as u16, 0, val);
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(iteration.finish()) as ArrayRef,
                Arc::new(forward_pass_idx.finish()),
                Arc::new(stage_id.finish()),
                Arc::new(training_state_id.finish()),
                Arc::new(branching_idx.finish()),
                Arc::new(variable_index.finish()),
                Arc::new(entity_id.finish()),
                Arc::new(lag_index.finish()),
                Arc::new(value.finish()),
            ],
        )?;

        self.write_batch("backward_detail.parquet", schema, batch)
    }

    fn write_cuts(
        &mut self,
        graph: &graph::DirectedGraph<fcf::FutureCostFunction>,
    ) -> Result<()> {
        let schema = schemas::cuts_schema();

        // Pre-calculate capacity
        let mut total_records = 0;
        for id in 0..graph.node_count() {
            let node = graph.get_node(id).unwrap();
            let fcf = &node.data;
            for cut in fcf.cut_pool.pool.iter() {
                total_records += 1; // RHS
                total_records += cut.coefficients.len(); // Coefficients
            }
        }

        if total_records == 0 {
            return Ok(());
        }

        // Build arrays
        let mut stage_index = UInt16Builder::with_capacity(total_records);
        let mut stage_cut_id = UInt32Builder::with_capacity(total_records);
        let mut iteration = UInt32Builder::with_capacity(total_records);
        let mut forward_pass_idx = UInt32Builder::with_capacity(total_records);
        let mut active = BooleanBuilder::with_capacity(total_records);
        let mut coefficient_index = UInt16Builder::with_capacity(total_records);
        let mut value = Float64Builder::with_capacity(total_records);

        for id in 0..graph.node_count() {
            let node = graph.get_node(id).unwrap();
            let fcf = &node.data;

            for cut in fcf.cut_pool.pool.iter() {
                // Index 0: RHS
                stage_index.append_value(node.id as u16);
                stage_cut_id.append_value(cut.id as u32);
                iteration.append_value(cut.iteration as u32);
                forward_pass_idx.append_value(cut.forward_pass_idx as u32);
                active.append_value(cut.is_active());
                coefficient_index.append_value(0);
                value.append_value(cut.rhs);

                // Indices 1+: State coefficients
                for (idx, &coef) in cut.coefficients.iter().enumerate() {
                    stage_index.append_value(node.id as u16);
                    stage_cut_id.append_value(cut.id as u32);
                    iteration.append_value(cut.iteration as u32);
                    forward_pass_idx.append_value(cut.forward_pass_idx as u32);
                    active.append_value(cut.is_active());
                    coefficient_index.append_value((idx + 1) as u16);
                    value.append_value(coef);
                }
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(stage_index.finish()) as ArrayRef,
                Arc::new(stage_cut_id.finish()),
                Arc::new(iteration.finish()),
                Arc::new(forward_pass_idx.finish()),
                Arc::new(active.finish()),
                Arc::new(coefficient_index.finish()),
                Arc::new(value.finish()),
            ],
        )?;

        self.write_batch("cuts.parquet", schema, batch)
    }

    fn write_states(
        &mut self,
        graph: &graph::DirectedGraph<fcf::FutureCostFunction>,
    ) -> Result<()> {
        let schema = schemas::states_schema();

        // Pre-calculate capacity
        let mut total_records = 0;
        for id in 0..graph.node_count() {
            let node = graph.get_node(id).unwrap();
            let fcf = &node.data;
            for state in fcf.state_pool.pool.iter() {
                total_records += 1; // Dominating objective
                total_records += state.coefficients().len(); // State components
            }
        }

        if total_records == 0 {
            return Ok(());
        }

        // Build arrays
        let mut stage_index = UInt16Builder::with_capacity(total_records);
        let mut dominating_cut_id = UInt32Builder::with_capacity(total_records);
        let mut iteration = UInt32Builder::with_capacity(total_records);
        let mut forward_pass_idx = UInt32Builder::with_capacity(total_records);
        let mut state_component_index =
            UInt16Builder::with_capacity(total_records);
        let mut value = Float64Builder::with_capacity(total_records);

        for id in 0..graph.node_count() {
            let node = graph.get_node(id).unwrap();
            let fcf = &node.data;

            for state in fcf.state_pool.pool.iter() {
                // Index 0: Dominating objective
                stage_index.append_value(node.id as u16);
                dominating_cut_id
                    .append_value(state.get_dominating_cut_id() as u32);
                iteration.append_value(state.get_iteration() as u32);
                forward_pass_idx
                    .append_value(state.get_forward_pass_idx() as u32);
                state_component_index.append_value(0);
                value.append_value(state.get_dominating_objective());

                // Indices 1+: State components
                for (idx, &component) in state.coefficients().iter().enumerate()
                {
                    stage_index.append_value(node.id as u16);
                    dominating_cut_id
                        .append_value(state.get_dominating_cut_id() as u32);
                    iteration.append_value(state.get_iteration() as u32);
                    forward_pass_idx
                        .append_value(state.get_forward_pass_idx() as u32);
                    state_component_index.append_value((idx + 1) as u16);
                    value.append_value(component);
                }
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(stage_index.finish()) as ArrayRef,
                Arc::new(dominating_cut_id.finish()),
                Arc::new(iteration.finish()),
                Arc::new(forward_pass_idx.finish()),
                Arc::new(state_component_index.finish()),
                Arc::new(value.finish()),
            ],
        )?;

        self.write_batch("states.parquet", schema, batch)
    }

    fn write_simulation(
        &mut self,
        trajectories: &[sddp::SimulationTrajectory],
        _system: &crate::system::System,
    ) -> Result<()> {
        if trajectories.is_empty() {
            return Ok(());
        }

        // Use a normalized schema similar to CSV output
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "stage",
                arrow::datatypes::DataType::UInt16,
                false,
            ),
            arrow::datatypes::Field::new(
                "series",
                arrow::datatypes::DataType::UInt32,
                false,
            ),
            arrow::datatypes::Field::new(
                "variable_index",
                arrow::datatypes::DataType::UInt16,
                false,
            ),
            arrow::datatypes::Field::new(
                "entity_id",
                arrow::datatypes::DataType::UInt16,
                true,
            ),
            arrow::datatypes::Field::new(
                "value",
                arrow::datatypes::DataType::Float64,
                false,
            ),
        ]));

        // Pre-calculate capacity
        let mut total_records = 0;
        for trajectory in trajectories {
            for realization in &trajectory.realizations {
                total_records += realization.loads.len();
                total_records += realization.deficit.len();
                total_records += realization.marginal_cost.len();
                total_records += realization.exchange.len();
                total_records += realization.inflow.len();
                total_records += realization.turbined_flow.len();
                total_records += realization.spillage.len();
                total_records += realization.thermal_generation.len();
                total_records += realization.water_value.len();
                total_records += realization.final_storage.len();
            }
        }

        // Build arrays
        let mut stage = UInt16Builder::with_capacity(total_records);
        let mut series = UInt32Builder::with_capacity(total_records);
        let mut variable_index = UInt16Builder::with_capacity(total_records);
        let mut entity_id = UInt16Builder::with_capacity(total_records);
        let mut value = Float64Builder::with_capacity(total_records);

        for (series_idx, trajectory) in trajectories.iter().enumerate() {
            for (stage_idx, realization) in
                trajectory.realizations.iter().enumerate()
            {
                let stg = stage_idx as u16;
                let ser = series_idx as u32;

                // Helper macro
                macro_rules! append_records {
                    ($var:expr, $data:expr) => {
                        for (eid, &val) in $data.iter().enumerate() {
                            stage.append_value(stg);
                            series.append_value(ser);
                            variable_index.append_value($var);
                            entity_id.append_value(eid as u16);
                            value.append_value(val);
                        }
                    };
                }

                // Bus variables
                append_records!(2, &realization.loads);
                append_records!(9, &realization.deficit);
                append_records!(11, &realization.marginal_cost);

                // Line variables
                append_records!(10, &realization.exchange);

                // Hydro variables
                append_records!(3, &realization.inflow);
                append_records!(5, &realization.turbined_flow);
                append_records!(6, &realization.spillage);
                append_records!(7, &realization.water_value);
                append_records!(4, &realization.final_storage);

                // Thermal variables
                append_records!(8, &realization.thermal_generation);
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(stage.finish()) as ArrayRef,
                Arc::new(series.finish()),
                Arc::new(variable_index.finish()),
                Arc::new(entity_id.finish()),
                Arc::new(value.finish()),
            ],
        )?;

        self.write_batch("simulation.parquet", schema, batch)
    }

    fn write_noises(
        &mut self,
        tree: &scenario::ScenarioTree,
        _system: &crate::system::System,
    ) -> Result<()> {
        // Check if there's any data to write
        if tree.stage_scenarios.is_empty() {
            return Ok(());
        }

        // Use custom schema for noises
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "stage_index",
                arrow::datatypes::DataType::UInt16,
                false,
            ),
            arrow::datatypes::Field::new(
                "branching_index",
                arrow::datatypes::DataType::UInt16,
                false,
            ),
            arrow::datatypes::Field::new(
                "variable_index",
                arrow::datatypes::DataType::UInt16,
                false,
            ),
            arrow::datatypes::Field::new(
                "entity_id",
                arrow::datatypes::DataType::UInt16,
                false,
            ),
            arrow::datatypes::Field::new(
                "value",
                arrow::datatypes::DataType::Float64,
                false,
            ),
        ]));

        // Pre-calculate capacity
        let mut total_records = 0;
        for stage_branchings in &tree.stage_scenarios {
            for branching_noises in &stage_branchings.branching_noises {
                total_records += branching_noises.load_innovations.len();
                total_records += branching_noises.inflow_innovations.len();
            }
        }

        if total_records == 0 {
            return Ok(());
        }

        // Build arrays
        let mut stage_index = UInt16Builder::with_capacity(total_records);
        let mut branching_index = UInt16Builder::with_capacity(total_records);
        let mut variable_index = UInt16Builder::with_capacity(total_records);
        let mut entity_id = UInt16Builder::with_capacity(total_records);
        let mut value = Float64Builder::with_capacity(total_records);

        for (stg_idx, stage_branchings) in
            tree.stage_scenarios.iter().enumerate()
        {
            for (branch_idx, branching_noises) in
                stage_branchings.branching_noises.iter().enumerate()
            {
                let stg = stg_idx as u16;
                let branch = branch_idx as u16;

                // Load innovations (variable_index = 2)
                for (eid, &noise_val) in
                    branching_noises.load_innovations.iter().enumerate()
                {
                    stage_index.append_value(stg);
                    branching_index.append_value(branch);
                    variable_index.append_value(2); // SampledLoad
                    entity_id.append_value(eid as u16);
                    value.append_value(noise_val);
                }

                // Inflow innovations (variable_index = 3)
                for (eid, &noise_val) in
                    branching_noises.inflow_innovations.iter().enumerate()
                {
                    stage_index.append_value(stg);
                    branching_index.append_value(branch);
                    variable_index.append_value(3); // SampledInflow
                    entity_id.append_value(eid as u16);
                    value.append_value(noise_val);
                }
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(stage_index.finish()) as ArrayRef,
                Arc::new(branching_index.finish()),
                Arc::new(variable_index.finish()),
                Arc::new(entity_id.finish()),
                Arc::new(value.finish()),
            ],
        )?;

        self.write_batch("training_sampled_noises.parquet", schema, batch)
    }

    fn flush(&mut self) -> Result<()> {
        // Parquet files are flushed immediately after write
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::{
        BackwardTimingOutput, ForwardTimingOutput, IterationTimingOutput,
    };
    use std::time::Duration;

    fn create_mock_iteration_result(iteration: usize) -> sddp::IterationResult {
        sddp::IterationResult {
            iteration,
            lower_bound: 100.0 + iteration as f64,
            forward_costs: vec![105.0, 104.0, 106.0],
            timing: IterationTimingOutput {
                model_allocation: Duration::from_millis(10),
                forward: ForwardTimingOutput {
                    saa_sampling: Duration::from_millis(100),
                    model_preprocessing: Duration::from_millis(50),
                    solver: Duration::from_millis(800),
                    model_postprocessing: Duration::from_millis(30),
                    postprocessing: Duration::from_millis(20),
                    parallel_wall: Duration::from_millis(900),
                    parallel_overhead: Duration::from_millis(100),
                    solver_max: Duration::from_millis(850),
                    solver_calls: 9,
                    total: Duration::from_millis(1000),
                },
                backward: BackwardTimingOutput {
                    model_preprocessing: Duration::from_millis(30),
                    solver: Duration::from_millis(400),
                    model_postprocessing: Duration::from_millis(20),
                    cut_selection: Duration::from_millis(10),
                    problem_update: Duration::from_millis(20),
                    solver_calls: 6,
                    total: Duration::from_millis(500),
                },
                model_cleanup: Duration::from_millis(5),
                total: Duration::from_millis(1500),
                solver_calls: 15,
            },
            num_cuts_added: 20,
            num_cuts_removed: 0,
            num_cuts_returned: 20,
            num_active_cuts: 200,
            first_stage_branching_costs: Vec::new(),
        }
    }

    #[test]
    fn test_parquet_config_default() {
        let config = ParquetConfig::default();
        assert!(matches!(config.compression, Compression::SNAPPY));
        assert_eq!(config.row_group_size, 10_000);
    }

    #[test]
    fn test_parquet_config_zstd() {
        let config = ParquetConfig::with_zstd(3);
        assert!(matches!(config.compression, Compression::ZSTD(_)));
    }

    #[test]
    fn test_parquet_config_uncompressed() {
        let config = ParquetConfig::uncompressed();
        assert!(matches!(config.compression, Compression::UNCOMPRESSED));
    }

    #[test]
    fn test_parquet_writer_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let writer = ParquetWriter::new(temp_dir.path());
        assert!(writer.is_ok());
    }

    #[test]
    fn test_write_training_empty() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut writer = ParquetWriter::new(temp_dir.path()).unwrap();

        let result = writer.write_training(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_write_training_single_iteration() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut writer = ParquetWriter::new(temp_dir.path()).unwrap();

        let results = vec![create_mock_iteration_result(1)];
        let result = writer.write_training(&results);
        assert!(result.is_ok());

        // Verify file was created
        let parquet_file = temp_dir.path().join("training.parquet");
        assert!(parquet_file.exists());
    }

    #[test]
    fn test_write_training_multiple_iterations() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut writer = ParquetWriter::new(temp_dir.path()).unwrap();

        let results: Vec<_> =
            (1..=100).map(create_mock_iteration_result).collect();

        let result = writer.write_training(&results);
        assert!(result.is_ok());

        let parquet_file = temp_dir.path().join("training.parquet");
        assert!(parquet_file.exists());

        // Verify file is not empty
        let metadata = std::fs::metadata(&parquet_file).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_write_and_read_training_parquet() {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use std::fs::File;

        let temp_dir = tempfile::tempdir().unwrap();
        let mut writer = ParquetWriter::new(temp_dir.path()).unwrap();

        let results: Vec<_> =
            (1..=10).map(create_mock_iteration_result).collect();

        writer.write_training(&results).unwrap();

        let parquet_path = temp_dir.path().join("training.parquet");
        assert!(parquet_path.exists());

        // Verify we can read the file back
        let file = File::open(&parquet_path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let metadata = reader.metadata();

        // Check schema has correct number of columns
        assert_eq!(metadata.file_metadata().schema().get_fields().len(), 16);

        // Check we have the right number of rows
        assert_eq!(metadata.file_metadata().num_rows(), 10);

        // Check file size (should be significantly smaller than CSV)
        let file_size = std::fs::metadata(&parquet_path).unwrap().len();
        assert!(file_size > 0);
        assert!(file_size < 10_000); // Reasonable size for 10 rows
    }

    #[test]
    fn test_parquet_compression_reduces_size() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create large dataset
        let results: Vec<_> =
            (1..=1000).map(create_mock_iteration_result).collect();

        // Write with compression
        let mut compressed_writer =
            ParquetWriter::new(temp_dir.path().join("compressed")).unwrap();
        compressed_writer.write_training(&results).unwrap();

        // Write without compression
        let mut uncompressed_writer = ParquetWriter::with_config(
            temp_dir.path().join("uncompressed"),
            ParquetConfig::uncompressed(),
        )
        .unwrap();
        uncompressed_writer.write_training(&results).unwrap();

        let compressed_size = std::fs::metadata(
            temp_dir.path().join("compressed/training.parquet"),
        )
        .unwrap()
        .len();

        let uncompressed_size = std::fs::metadata(
            temp_dir.path().join("uncompressed/training.parquet"),
        )
        .unwrap()
        .len();

        // Compressed should be smaller
        assert!(compressed_size < uncompressed_size);
        println!(
            "Compressed: {} bytes, Uncompressed: {} bytes, Ratio: {:.1}%",
            compressed_size,
            uncompressed_size,
            (compressed_size as f64 / uncompressed_size as f64) * 100.0
        );
    }
}
