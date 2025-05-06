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

#[allow(dead_code)]
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

    pub fn default() -> Self {
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
}
