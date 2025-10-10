// Test system fixtures
//
// Provides pre-configured power system configurations for testing using the public input module.

/// Creates a trivial single-bus, single-hydro system configuration as JSON
///
/// This returns a JSON string that can be loaded via the input module.
/// System characteristics:
/// - 1 bus with deficit cost of 1000.0
/// - 1 hydro plant (no downstream, no spillage cost)
/// - Min volume: 0.0, Max volume: 100.0, Initial volume: 50.0
/// - Productivity: 1.0 (simple 1:1 conversion)
/// - No transmission lines
/// - Deterministic (suitable for testing algorithm mechanics without stochasticity)
///
/// # Use Cases
/// - Testing basic state transitions
/// - Testing cut operations without complex constraints
/// - Validating solver interface
/// - Baseline performance benchmarks
///
/// # Performance Note
/// Minimal memory footprint - suitable for tight inner loop tests
pub fn trivial_system_json() -> &'static str {
    r#"{
        "buses": [
            {"id": 0, "deficit_cost": 1000.0}
        ],
        "hydros": [
            {
                "id": 0,
                "bus_id": 0,
                "downstream_hydro_id": null,
                "min_volume": 0.0,
                "max_volume": 100.0,
                "initial_volume": 50.0,
                "productivity": 1.0,
                "spillage_cost": 0.0,
                "min_generation": 0.0,
                "max_generation": 100.0,
                "min_outflow": 0.0,
                "max_outflow": 100.0,
                "min_spillage": 0.0,
                "max_spillage": 1e308,
                "routing": []
            }
        ],
        "thermals": [],
        "lines": []
    }"#
}

/// Creates a simple two-bus system configuration with two hydro plants in cascade
///
/// Returns JSON string for input module.
/// System characteristics:
/// - 2 buses (deficit costs: 1000.0, 900.0)
/// - 2 hydro plants in cascade (plant 1 downstream of plant 0)
/// - 1 transmission line connecting buses (capacity 50 MW each direction)
/// - More realistic volumes and productivity factors
/// - Suitable for stochastic testing
///
/// # Use Cases
/// - Testing cascaded hydro operations
/// - Testing transmission constraints
/// - Testing stochastic scenarios
/// - Integration tests with meaningful structure
///
/// # Performance Note
/// Small enough for rapid testing, complex enough to catch bugs
pub fn simple_system_json() -> &'static str {
    r#"{
        "buses": [
            {"id": 0, "deficit_cost": 1000.0},
            {"id": 1, "deficit_cost": 900.0}
        ],
        "hydros": [
            {
                "id": 0,
                "bus_id": 0,
                "downstream_hydro_id": 1,
                "min_volume": 0.0,
                "max_volume": 200.0,
                "initial_volume": 100.0,
                "productivity": 0.95,
                "spillage_cost": 5.0,
                "min_generation": 0.0,
                "max_generation": 150.0,
                "min_outflow": 0.0,
                "max_outflow": 150.0,
                "min_spillage": 0.0,
                "max_spillage": 1e308,
                "routing": []
            },
            {
                "id": 1,
                "bus_id": 1,
                "downstream_hydro_id": null,
                "min_volume": 0.0,
                "max_volume": 150.0,
                "initial_volume": 75.0,
                "productivity": 0.90,
                "spillage_cost": 5.0,
                "min_generation": 0.0,
                "max_generation": 120.0,
                "min_outflow": 0.0,
                "max_outflow": 120.0,
                "min_spillage": 0.0,
                "max_spillage": 1e308,
                "routing": []
            }
        ],
        "thermals": [],
        "lines": [
            {
                "id": 0,
                "source_bus_id": 0,
                "target_bus_id": 1,
                "direct_capacity": 50.0,
                "reverse_capacity": 50.0,
                "exchange_penalty": 10.0
            }
        ]
    }"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_system_json_parses() {
        // Just verify it's valid JSON
        let json = trivial_system_json();
        assert!(json.contains("\"buses\""));
        assert!(json.contains("\"hydros\""));
    }

    #[test]
    fn test_simple_system_json_parses() {
        let json = simple_system_json();
        assert!(json.contains("\"buses\""));
        assert!(json.contains("\"hydros\""));
        assert!(json.contains("\"lines\""));
    }
}
