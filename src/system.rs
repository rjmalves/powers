#[derive(Debug)]
pub struct Bus {
    pub id: usize,
    pub deficit_cost: f64,
    pub hydro_ids: Vec<usize>,
    pub thermal_ids: Vec<usize>,
    pub source_line_ids: Vec<usize>,
    pub target_line_ids: Vec<usize>,
}

impl Bus {
    pub fn new(id: usize, deficit_cost: f64) -> Self {
        Self {
            id,
            deficit_cost,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        }
    }

    pub fn add_hydro(&mut self, hydro_id: usize) {
        self.hydro_ids.push(hydro_id);
    }

    pub fn add_thermal(&mut self, thermal_id: usize) {
        self.thermal_ids.push(thermal_id);
    }

    pub fn add_source_line(&mut self, line_id: usize) {
        self.source_line_ids.push(line_id);
    }

    pub fn add_target_line(&mut self, line_id: usize) {
        self.target_line_ids.push(line_id);
    }
}

#[derive(Debug)]
pub struct Line {
    pub id: usize,
    pub source_bus_id: usize,
    pub target_bus_id: usize,
    pub direct_capacity: f64,
    pub reverse_capacity: f64,
    pub exchange_penalty: f64,
}

impl Line {
    pub fn new(
        id: usize,
        source_bus_id: usize,
        target_bus_id: usize,
        direct_capacity: f64,
        reverse_capacity: f64,
        exchange_penalty: f64,
    ) -> Self {
        Self {
            id,
            source_bus_id,
            target_bus_id,
            direct_capacity,
            reverse_capacity,
            exchange_penalty,
        }
    }
}

#[derive(Debug)]
pub struct Thermal {
    pub id: usize,
    pub bus_id: usize,
    pub cost: f64,
    pub min_generation: f64,
    pub max_generation: f64,
}

impl Thermal {
    pub fn new(
        id: usize,
        bus_id: usize,
        cost: f64,
        min_generation: f64,
        max_generation: f64,
    ) -> Self {
        Self {
            id,
            bus_id,
            cost,
            min_generation,
            max_generation,
        }
    }
}

#[derive(Debug)]
pub struct Hydro {
    pub id: usize,
    pub downstream_hydro_id: Option<usize>,
    pub bus_id: usize,
    pub productivity: f64,
    pub min_storage: f64,
    pub max_storage: f64,
    pub min_turbined_flow: f64,
    pub max_turbined_flow: f64,
    pub spillage_penalty: f64,
    pub upstream_hydro_ids: Vec<usize>,
}

impl Hydro {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: usize,
        downstream_hydro_id: Option<usize>,
        bus_id: usize,
        productivity: f64,
        min_storage: f64,
        max_storage: f64,
        min_turbined_flow: f64,
        max_turbined_flow: f64,
        spillage_penalty: f64,
    ) -> Self {
        Self {
            id,
            downstream_hydro_id,
            bus_id,
            productivity,
            min_storage,
            max_storage,
            min_turbined_flow,
            max_turbined_flow,
            spillage_penalty,
            upstream_hydro_ids: vec![],
        }
    }

    pub fn add_upstream_hydro(&mut self, hydro_id: usize) {
        self.upstream_hydro_ids.push(hydro_id);
    }
}

#[derive(Debug)]
pub struct SystemMetadata {
    pub buses_count: usize,
    pub lines_count: usize,
    pub thermals_count: usize,
    pub hydros_count: usize,
}

#[derive(Debug)]
pub struct System {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub thermals: Vec<Thermal>,
    pub hydros: Vec<Hydro>,
    pub meta: SystemMetadata,
}

impl System {
    pub fn new(
        mut buses: Vec<Bus>,
        lines: Vec<Line>,
        thermals: Vec<Thermal>,
        hydros: Vec<Hydro>,
    ) -> Self {
        for l in lines.iter() {
            buses[l.source_bus_id].add_source_line(l.id);
            buses[l.target_bus_id].add_target_line(l.id);
        }
        for t in thermals.iter() {
            buses[t.bus_id].add_thermal(t.id);
        }
        for h in hydros.iter() {
            buses[h.bus_id].add_hydro(h.id);
        }

        let buses_count = buses.len();
        let lines_count = lines.len();
        let thermals_count = thermals.len();
        let hydros_count = hydros.len();

        Self {
            buses,
            lines,
            thermals,
            hydros,
            meta: SystemMetadata {
                buses_count,
                lines_count,
                thermals_count,
                hydros_count,
            },
        }
    }

    /// Creates an empty system for testing purposes
    #[cfg(test)]
    pub fn new_empty() -> Self {
        Self {
            buses: Vec::new(),
            lines: Vec::new(),
            thermals: Vec::new(),
            hydros: Vec::new(),
            meta: SystemMetadata {
                buses_count: 0,
                lines_count: 0,
                thermals_count: 0,
                hydros_count: 0,
            },
        }
    }

    /// Validates the system configuration and returns errors if any
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check for empty system
        if self.buses.is_empty() {
            errors.push("System must have at least one bus".to_string());
        }

        if self.hydros.is_empty() && self.thermals.is_empty() {
            errors.push(
                "System must have at least one hydro or thermal unit"
                    .to_string(),
            );
        }

        // Check unique IDs
        self.validate_unique_ids(&mut errors);

        // Check valid references
        self.validate_references(&mut errors);

        // Check capacity constraints
        self.validate_capacity_constraints(&mut errors);

        // Check positive values
        self.validate_positive_values(&mut errors);

        // Check hydro cascade acyclicity
        self.validate_hydro_cascade_acyclic(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_unique_ids(&self, errors: &mut Vec<String>) {
        // Check bus IDs
        let mut bus_ids = std::collections::HashSet::new();
        for bus in &self.buses {
            if !bus_ids.insert(bus.id) {
                errors.push(format!("Duplicate bus ID: {}", bus.id));
            }
        }

        // Check line IDs
        let mut line_ids = std::collections::HashSet::new();
        for line in &self.lines {
            if !line_ids.insert(line.id) {
                errors.push(format!("Duplicate line ID: {}", line.id));
            }
        }

        // Check thermal IDs
        let mut thermal_ids = std::collections::HashSet::new();
        for thermal in &self.thermals {
            if !thermal_ids.insert(thermal.id) {
                errors.push(format!("Duplicate thermal ID: {}", thermal.id));
            }
        }

        // Check hydro IDs
        let mut hydro_ids = std::collections::HashSet::new();
        for hydro in &self.hydros {
            if !hydro_ids.insert(hydro.id) {
                errors.push(format!("Duplicate hydro ID: {}", hydro.id));
            }
        }
    }

    fn validate_references(&self, errors: &mut Vec<String>) {
        let bus_ids: std::collections::HashSet<_> =
            self.buses.iter().map(|b| b.id).collect();
        let hydro_ids: std::collections::HashSet<_> =
            self.hydros.iter().map(|h| h.id).collect();

        // Check line references
        for line in &self.lines {
            if !bus_ids.contains(&line.source_bus_id) {
                errors.push(format!(
                    "Line {} references non-existent source bus {}",
                    line.id, line.source_bus_id
                ));
            }
            if !bus_ids.contains(&line.target_bus_id) {
                errors.push(format!(
                    "Line {} references non-existent target bus {}",
                    line.id, line.target_bus_id
                ));
            }
        }

        // Check thermal references
        for thermal in &self.thermals {
            if !bus_ids.contains(&thermal.bus_id) {
                errors.push(format!(
                    "Thermal {} references non-existent bus {}",
                    thermal.id, thermal.bus_id
                ));
            }
        }

        // Check hydro references
        for hydro in &self.hydros {
            if !bus_ids.contains(&hydro.bus_id) {
                errors.push(format!(
                    "Hydro {} references non-existent bus {}",
                    hydro.id, hydro.bus_id
                ));
            }
            if let Some(downstream_id) = hydro.downstream_hydro_id {
                if !hydro_ids.contains(&downstream_id) {
                    errors.push(format!(
                        "Hydro {} references non-existent downstream hydro {}",
                        hydro.id, downstream_id
                    ));
                }
            }
        }
    }

    fn validate_capacity_constraints(&self, errors: &mut Vec<String>) {
        // Check hydro constraints
        for hydro in &self.hydros {
            if hydro.min_storage > hydro.max_storage {
                errors.push(format!(
                    "Hydro {}: min_storage ({}) > max_storage ({})",
                    hydro.id, hydro.min_storage, hydro.max_storage
                ));
            }
            if hydro.min_turbined_flow > hydro.max_turbined_flow {
                errors.push(format!(
                    "Hydro {}: min_turbined_flow ({}) > max_turbined_flow ({})",
                    hydro.id, hydro.min_turbined_flow, hydro.max_turbined_flow
                ));
            }
        }

        // Check thermal constraints
        for thermal in &self.thermals {
            if thermal.min_generation > thermal.max_generation {
                errors.push(format!(
                    "Thermal {}: min_generation ({}) > max_generation ({})",
                    thermal.id, thermal.min_generation, thermal.max_generation
                ));
            }
        }
    }

    fn validate_positive_values(&self, errors: &mut Vec<String>) {
        // Check bus deficit costs
        for bus in &self.buses {
            if bus.deficit_cost < 0.0 {
                errors.push(format!(
                    "Bus {}: deficit_cost must be non-negative, got {}",
                    bus.id, bus.deficit_cost
                ));
            }
        }

        // Check line capacities
        for line in &self.lines {
            if line.direct_capacity < 0.0 {
                errors.push(format!(
                    "Line {}: direct_capacity must be non-negative, got {}",
                    line.id, line.direct_capacity
                ));
            }
            if line.reverse_capacity < 0.0 {
                errors.push(format!(
                    "Line {}: reverse_capacity must be non-negative, got {}",
                    line.id, line.reverse_capacity
                ));
            }
            if line.exchange_penalty < 0.0 {
                errors.push(format!(
                    "Line {}: exchange_penalty must be non-negative, got {}",
                    line.id, line.exchange_penalty
                ));
            }
        }

        // Check thermal costs and generation
        for thermal in &self.thermals {
            if thermal.cost < 0.0 {
                errors.push(format!(
                    "Thermal {}: cost must be non-negative, got {}",
                    thermal.id, thermal.cost
                ));
            }
            if thermal.min_generation < 0.0 {
                errors.push(format!(
                    "Thermal {}: min_generation must be non-negative, got {}",
                    thermal.id, thermal.min_generation
                ));
            }
            if thermal.max_generation < 0.0 {
                errors.push(format!(
                    "Thermal {}: max_generation must be non-negative, got {}",
                    thermal.id, thermal.max_generation
                ));
            }
        }

        // Check hydro values
        for hydro in &self.hydros {
            if hydro.productivity <= 0.0 {
                errors.push(format!(
                    "Hydro {}: productivity must be positive, got {}",
                    hydro.id, hydro.productivity
                ));
            }
            if hydro.min_storage < 0.0 {
                errors.push(format!(
                    "Hydro {}: min_storage must be non-negative, got {}",
                    hydro.id, hydro.min_storage
                ));
            }
            if hydro.max_storage < 0.0 {
                errors.push(format!(
                    "Hydro {}: max_storage must be non-negative, got {}",
                    hydro.id, hydro.max_storage
                ));
            }
            if hydro.min_turbined_flow < 0.0 {
                errors.push(format!(
                    "Hydro {}: min_turbined_flow must be non-negative, got {}",
                    hydro.id, hydro.min_turbined_flow
                ));
            }
            if hydro.max_turbined_flow < 0.0 {
                errors.push(format!(
                    "Hydro {}: max_turbined_flow must be non-negative, got {}",
                    hydro.id, hydro.max_turbined_flow
                ));
            }
            if hydro.spillage_penalty < 0.0 {
                errors.push(format!(
                    "Hydro {}: spillage_penalty must be non-negative, got {}",
                    hydro.id, hydro.spillage_penalty
                ));
            }
        }
    }

    fn validate_hydro_cascade_acyclic(&self, errors: &mut Vec<String>) {
        // Use DFS to detect cycles
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for hydro in &self.hydros {
            if !visited.contains(&hydro.id)
                && self.has_cycle_dfs(hydro.id, &mut visited, &mut rec_stack)
            {
                errors.push(format!(
                    "Cycle detected in hydro cascade involving hydro {}",
                    hydro.id
                ));
            }
        }
    }

    fn has_cycle_dfs(
        &self,
        hydro_id: usize,
        visited: &mut std::collections::HashSet<usize>,
        rec_stack: &mut std::collections::HashSet<usize>,
    ) -> bool {
        visited.insert(hydro_id);
        rec_stack.insert(hydro_id);

        // Find the hydro by ID
        if let Some(hydro) = self.hydros.iter().find(|h| h.id == hydro_id) {
            if let Some(downstream_id) = hydro.downstream_hydro_id {
                if !visited.contains(&downstream_id) {
                    if self.has_cycle_dfs(downstream_id, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&downstream_id) {
                    return true;
                }
            }
        }

        rec_stack.remove(&hydro_id);
        false
    }
}

impl Default for System {
    fn default() -> Self {
        let buses = vec![Bus::new(0, 50.0)];
        let lines: Vec<Line> = vec![];
        let thermals = vec![
            Thermal::new(0, 0, 5.0, 0.0, 15.0),
            Thermal::new(1, 0, 10.0, 0.0, 15.0),
        ];
        let hydros =
            vec![Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];

        Self::new(buses, lines, thermals, hydros)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_default_system() {
        let system = System::default();
        assert_eq!(system.buses.len(), 1);
        assert_eq!(system.lines.len(), 0);
        assert_eq!(system.thermals.len(), 2);
        assert_eq!(system.hydros.len(), 1);
    }

    // ========================================================================
    // TEST-005: System Validation Tests
    // ========================================================================

    #[test]
    fn test_empty_system_rejected() {
        // System with no buses should be rejected
        let system = System::new(vec![], vec![], vec![], vec![]);
        let result = system.validate();

        assert!(result.is_err(), "Empty system should be rejected");
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("at least one bus")),
            "Should report missing buses"
        );
    }

    #[test]
    fn test_system_validation_unique_ids() {
        // Create system with duplicate bus IDs
        let bus1 = Bus {
            id: 0,
            deficit_cost: 1000.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        };
        let bus2 = Bus {
            id: 0, // Duplicate ID
            deficit_cost: 900.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        };

        let system = System::new(vec![bus1, bus2], vec![], vec![], vec![]);
        let result = system.validate();

        assert!(
            result.is_err(),
            "System with duplicate bus IDs should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("Duplicate bus ID")),
            "Should report duplicate bus ID, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_hydro_cascade_acyclic() {
        // Create system with circular cascade: hydro 0 -> 1 -> 2 -> 0
        let bus = Bus {
            id: 0,
            deficit_cost: 1000.0,
            hydro_ids: vec![0, 1, 2],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        };

        let hydro0 = Hydro::new(0, Some(1), 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.0);
        let hydro1 = Hydro::new(1, Some(2), 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.0);
        let hydro2 = Hydro::new(2, Some(0), 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.0); // Cycle!

        let system = System::new(
            vec![bus],
            vec![],
            vec![],
            vec![hydro0, hydro1, hydro2],
        );
        let result = system.validate();

        assert!(
            result.is_err(),
            "System with circular cascade should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("cycle") || e.contains("Cycle")),
            "Should report cascade cycle, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_system_no_generation_rejected() {
        // System with bus but no generation (hydro or thermal)
        let bus = Bus::new(0, 1000.0);
        let system = System::new(vec![bus], vec![], vec![], vec![]);
        let result = system.validate();

        assert!(
            result.is_err(),
            "System with no generation should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("at least one hydro or thermal")),
            "Should report missing generation"
        );
    }

    #[test]
    fn test_duplicate_thermal_ids_rejected() {
        let bus = Bus::new(0, 1000.0);
        let thermal1 = Thermal::new(0, 0, 10.0, 0.0, 100.0);
        let thermal2 = Thermal::new(0, 0, 20.0, 0.0, 100.0); // Duplicate ID

        let system =
            System::new(vec![bus], vec![], vec![thermal1, thermal2], vec![]);
        let result = system.validate();

        assert!(result.is_err(), "Duplicate thermal IDs should be rejected");
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("Duplicate thermal ID")));
    }

    #[test]
    fn test_duplicate_hydro_ids_rejected() {
        let bus = Bus::new(0, 1000.0);
        let hydro1 = Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.0);
        let hydro2 = Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.0); // Duplicate ID

        let system =
            System::new(vec![bus], vec![], vec![], vec![hydro1, hydro2]);
        let result = system.validate();

        assert!(result.is_err(), "Duplicate hydro IDs should be rejected");
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("Duplicate hydro ID")));
    }

    #[test]
    fn test_duplicate_line_ids_rejected() {
        let bus1 = Bus::new(0, 1000.0);
        let bus2 = Bus::new(1, 1000.0);
        let thermal = Thermal::new(0, 0, 10.0, 0.0, 100.0);
        let line1 = Line::new(0, 0, 1, 50.0, 50.0, 1.0);
        let line2 = Line::new(0, 1, 0, 50.0, 50.0, 1.0); // Duplicate ID

        let system = System::new(
            vec![bus1, bus2],
            vec![line1, line2],
            vec![thermal],
            vec![],
        );
        let result = system.validate();

        assert!(result.is_err(), "Duplicate line IDs should be rejected");
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("Duplicate line ID")));
    }

    #[test]
    fn test_thermal_invalid_bus_reference_rejected() {
        let bus = Bus::new(0, 1000.0);
        let thermal = Thermal {
            id: 0,
            bus_id: 99, // Invalid bus ID
            cost: 10.0,
            min_generation: 0.0,
            max_generation: 100.0,
        };

        let system = System {
            buses: vec![bus],
            lines: vec![],
            thermals: vec![thermal],
            hydros: vec![],
            meta: SystemMetadata {
                buses_count: 1,
                lines_count: 0,
                thermals_count: 1,
                hydros_count: 0,
            },
        };
        let result = system.validate();

        assert!(
            result.is_err(),
            "Thermal with invalid bus reference should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(
                |e| e.contains("Thermal") && e.contains("non-existent bus")
            ),
            "Should report invalid bus reference, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_hydro_invalid_bus_reference_rejected() {
        let bus = Bus::new(0, 1000.0);
        let hydro = Hydro {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 99, // Invalid bus ID
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        };

        let system = System {
            buses: vec![bus],
            lines: vec![],
            thermals: vec![],
            hydros: vec![hydro],
            meta: SystemMetadata {
                buses_count: 1,
                lines_count: 0,
                thermals_count: 0,
                hydros_count: 1,
            },
        };
        let result = system.validate();

        assert!(
            result.is_err(),
            "Hydro with invalid bus reference should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("Hydro") && e.contains("non-existent bus")),
            "Should report invalid bus reference, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_line_invalid_source_bus_rejected() {
        let bus = Bus::new(0, 1000.0);
        let thermal = Thermal::new(0, 0, 10.0, 0.0, 100.0);
        let line = Line {
            id: 0,
            source_bus_id: 99, // Invalid source bus
            target_bus_id: 0,
            direct_capacity: 50.0,
            reverse_capacity: 50.0,
            exchange_penalty: 1.0,
        };

        let system = System {
            buses: vec![bus],
            lines: vec![line],
            thermals: vec![thermal],
            hydros: vec![],
            meta: SystemMetadata {
                buses_count: 1,
                lines_count: 1,
                thermals_count: 1,
                hydros_count: 0,
            },
        };
        let result = system.validate();

        assert!(
            result.is_err(),
            "Line with invalid source bus should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("Line")
                    && e.contains("non-existent source bus")),
            "Should report invalid source bus, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_line_invalid_target_bus_rejected() {
        let bus = Bus::new(0, 1000.0);
        let thermal = Thermal::new(0, 0, 10.0, 0.0, 100.0);
        let line = Line {
            id: 0,
            source_bus_id: 0,
            target_bus_id: 99, // Invalid target bus
            direct_capacity: 50.0,
            reverse_capacity: 50.0,
            exchange_penalty: 1.0,
        };

        let system = System {
            buses: vec![bus],
            lines: vec![line],
            thermals: vec![thermal],
            hydros: vec![],
            meta: SystemMetadata {
                buses_count: 1,
                lines_count: 1,
                thermals_count: 1,
                hydros_count: 0,
            },
        };
        let result = system.validate();

        assert!(
            result.is_err(),
            "Line with invalid target bus should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("Line")
                    && e.contains("non-existent target bus")),
            "Should report invalid target bus, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_hydro_invalid_downstream_reference_rejected() {
        let bus = Bus::new(0, 1000.0);
        let hydro = Hydro {
            id: 0,
            downstream_hydro_id: Some(99), // Invalid downstream
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        };

        let system = System {
            buses: vec![bus],
            lines: vec![],
            thermals: vec![],
            hydros: vec![hydro],
            meta: SystemMetadata {
                buses_count: 1,
                lines_count: 0,
                thermals_count: 0,
                hydros_count: 1,
            },
        };
        let result = system.validate();

        assert!(
            result.is_err(),
            "Hydro with invalid downstream reference should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("Hydro")
                && e.contains("non-existent downstream")),
            "Should report invalid downstream hydro, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_thermal_capacity_validation() {
        let bus = Bus::new(0, 1000.0);
        // Min > Max should be invalid
        let thermal = Thermal::new(0, 0, 10.0, 100.0, 50.0);

        let system = System::new(vec![bus], vec![], vec![thermal], vec![]);
        let result = system.validate();

        assert!(result.is_err(), "Thermal with min > max should be rejected");
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("Thermal")
                && (e.contains("min_generation") || e.contains("capacity"))),
            "Should report capacity constraint violation"
        );
    }

    #[test]
    fn test_hydro_capacity_validation() {
        let bus = Bus::new(0, 1000.0);
        // Min > Max storage should be invalid
        let hydro = Hydro::new(0, None, 0, 1.0, 100.0, 50.0, 0.0, 50.0, 0.0);

        let system = System::new(vec![bus], vec![], vec![], vec![hydro]);
        let result = system.validate();

        assert!(
            result.is_err(),
            "Hydro with min_storage > max_storage should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("Hydro")
                && (e.contains("min_storage") || e.contains("capacity"))),
            "Should report storage capacity constraint violation"
        );
    }

    #[test]
    fn test_valid_system_accepted() {
        // Create a valid system
        let system = System::default();
        let result = system.validate();

        assert!(
            result.is_ok(),
            "Valid system should be accepted, got errors: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_valid_system_with_cascade_accepted() {
        let bus = Bus::new(0, 1000.0);
        let hydro1 = Hydro::new(0, Some(1), 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.0);
        let hydro2 = Hydro::new(1, None, 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.0);

        let system =
            System::new(vec![bus], vec![], vec![], vec![hydro1, hydro2]);
        let result = system.validate();

        assert!(
            result.is_ok(),
            "Valid system with cascade should be accepted"
        );
    }

    #[test]
    fn test_validation_performance() {
        // Create a reasonably sized system
        let mut buses = Vec::new();
        let mut hydros = Vec::new();
        let mut thermals = Vec::new();

        for i in 0..10 {
            buses.push(Bus::new(i, 1000.0));
            hydros.push(Hydro::new(
                i,
                None,
                i % 10,
                1.0,
                0.0,
                100.0,
                0.0,
                50.0,
                0.0,
            ));
            thermals.push(Thermal::new(i, i % 10, 10.0, 0.0, 100.0));
        }

        let system = System::new(buses, vec![], thermals, hydros);

        let start = std::time::Instant::now();
        let result = system.validate();
        let duration = start.elapsed();

        assert!(result.is_ok(), "Valid system should be accepted");
        assert!(
            duration.as_millis() < 10,
            "Validation should complete in < 10ms for 30 components, took {:?}",
            duration
        );
    }

    #[test]
    fn test_self_referencing_hydro_rejected() {
        let bus = Bus::new(0, 1000.0);
        let hydro = Hydro::new(0, Some(0), 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.0); // Self-reference

        let system = System::new(vec![bus], vec![], vec![], vec![hydro]);
        let result = system.validate();

        assert!(
            result.is_err(),
            "Hydro with self-reference should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("Cycle")),
            "Should report cycle (self-reference)"
        );
    }

    #[test]
    fn test_negative_deficit_cost_rejected() {
        let bus = Bus {
            id: 0,
            deficit_cost: -100.0, // Negative deficit cost
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        };
        let thermal = Thermal::new(0, 0, 10.0, 0.0, 100.0);

        let system = System {
            buses: vec![bus],
            lines: vec![],
            thermals: vec![thermal],
            hydros: vec![],
            meta: SystemMetadata {
                buses_count: 1,
                lines_count: 0,
                thermals_count: 1,
                hydros_count: 0,
            },
        };
        let result = system.validate();

        assert!(result.is_err(), "Negative deficit cost should be rejected");
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("deficit_cost")
                    && e.contains("non-negative")),
            "Should report negative deficit cost, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_negative_thermal_cost_rejected() {
        let bus = Bus::new(0, 1000.0);
        let thermal = Thermal {
            id: 0,
            bus_id: 0,
            cost: -10.0, // Negative cost
            min_generation: 0.0,
            max_generation: 100.0,
        };

        let system = System {
            buses: vec![bus],
            lines: vec![],
            thermals: vec![thermal],
            hydros: vec![],
            meta: SystemMetadata {
                buses_count: 1,
                lines_count: 0,
                thermals_count: 1,
                hydros_count: 0,
            },
        };
        let result = system.validate();

        assert!(result.is_err(), "Negative thermal cost should be rejected");
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("Thermal")
                && e.contains("cost")
                && e.contains("non-negative")),
            "Should report negative thermal cost, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_negative_hydro_productivity_rejected() {
        let bus = Bus::new(0, 1000.0);
        let hydro = Hydro {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: -1.0, // Negative productivity
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        };

        let system = System {
            buses: vec![bus],
            lines: vec![],
            thermals: vec![],
            hydros: vec![hydro],
            meta: SystemMetadata {
                buses_count: 1,
                lines_count: 0,
                thermals_count: 0,
                hydros_count: 1,
            },
        };
        let result = system.validate();

        assert!(
            result.is_err(),
            "Negative hydro productivity should be rejected"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("Hydro")
                && e.contains("productivity")
                && e.contains("positive")),
            "Should report negative productivity, got: {:?}",
            errors
        );
    }
}
