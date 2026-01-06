//! Statistics and timing formatting utilities.
//!
//! Provides consistent formatting for costs, durations, and statistical summaries
//! across all display renderers.

use crate::display::context::CostStatistics;
use std::time::Duration;

/// Options for formatting statistics.
#[derive(Debug, Clone, Copy)]
pub struct StatisticsFormat {
    /// Use scientific notation for large numbers
    pub scientific: bool,
    /// Number of significant digits
    pub precision: usize,
    /// Include sample count
    pub show_count: bool,
}

impl Default for StatisticsFormat {
    fn default() -> Self {
        Self {
            scientific: true,
            precision: 2,
            show_count: true,
        }
    }
}

/// Format a single cost value with appropriate precision.
///
/// # Arguments
///
/// * `value` - Cost value to format
/// * `scientific` - If true, use scientific notation for values >= 1e4
///
/// # Returns
///
/// Formatted string like "1.23e+05" or "1234.56"
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::statistics::format_cost;
///
/// assert_eq!(format_cost(123456.0, true), "1.23e5");
/// assert_eq!(format_cost(1234.56, false), "1234.56");
/// assert_eq!(format_cost(f64::NAN, true), "NaN");
/// ```
#[must_use]
pub fn format_cost(value: f64, scientific: bool) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "NaN".to_string()
        } else if value > 0.0 {
            "+∞".to_string()
        } else {
            "-∞".to_string()
        };
    }

    if scientific || value.abs() >= 1e4 {
        // Use scientific notation without the '+' in exponent
        let formatted = format!("{:.2e}", value);
        // Remove '+' from exponent: 1.23e+05 -> 1.23e5
        formatted.replace("e+0", "e").replace("e+", "e")
    } else {
        format!("{:.2}", value)
    }
}

/// Format cost statistics in compact form.
///
/// # Arguments
///
/// * `stats` - Cost statistics to format
/// * `format` - Formatting options
///
/// # Returns
///
/// String like "μ=1.28e5 σ=3.2e3 [1.24e5..1.35e5] n=4"
///
/// # Examples
///
/// ```
/// use powers_rs::display::{CostStatistics, components::statistics::{format_cost_stats, StatisticsFormat}};
///
/// let stats = CostStatistics {
///     mean: 128000.0,
///     std_dev: 3200.0,
///     min: 124000.0,
///     max: 135000.0,
///     count: 4,
/// };
/// let result = format_cost_stats(&stats, &StatisticsFormat::default());
/// assert!(result.contains("μ=1.28e5"));
/// assert!(result.contains("σ=3.20e3"));
/// assert!(result.contains("n=4"));
/// ```
#[must_use]
pub fn format_cost_stats(
    stats: &CostStatistics,
    format: &StatisticsFormat,
) -> String {
    let mut parts = vec![
        format!("μ={}", format_cost(stats.mean, format.scientific)),
        format!("σ={}", format_cost(stats.std_dev, format.scientific)),
        format!(
            "[{}..{}]",
            format_cost(stats.min, format.scientific),
            format_cost(stats.max, format.scientific)
        ),
    ];

    if format.show_count {
        parts.push(format!("n={}", stats.count));
    }

    parts.join(" ")
}

/// Format cost statistics on one line with explicit labels.
///
/// # Examples
///
/// ```
/// use powers_rs::display::{CostStatistics, components::statistics::{format_cost_stats_labeled, StatisticsFormat}};
///
/// let stats = CostStatistics {
///     mean: 128000.0,
///     std_dev: 3200.0,
///     min: 124000.0,
///     max: 135000.0,
///     count: 4,
/// };
/// let result = format_cost_stats_labeled(&stats, &StatisticsFormat::default());
/// assert!(result.contains("mean:"));
/// assert!(result.contains("std:"));
/// assert!(result.contains("range:"));
/// ```
#[must_use]
pub fn format_cost_stats_labeled(
    stats: &CostStatistics,
    format: &StatisticsFormat,
) -> String {
    format!(
        "mean: {} | std: {} | range: [{}, {}]",
        format_cost(stats.mean, format.scientific),
        format_cost(stats.std_dev, format.scientific),
        format_cost(stats.min, format.scientific),
        format_cost(stats.max, format.scientific)
    )
}

/// Format a duration in compact form.
///
/// # Formatting Rules
///
/// - < 1s: `0.XXXs` (3 decimal places)
/// - < 60s: `X.XXs` (2 decimal places)
/// - < 60m: `Xm XXs`
/// - >= 60m: `Xh XXm`
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use powers_rs::display::components::statistics::format_duration_compact;
///
/// assert_eq!(format_duration_compact(Duration::from_millis(34)), "0.034s");
/// assert_eq!(format_duration_compact(Duration::from_secs(65)), "1m 05s");
/// assert_eq!(format_duration_compact(Duration::from_secs(3725)), "1h 02m");
/// ```
#[must_use]
pub fn format_duration_compact(duration: Duration) -> String {
    let total_secs = duration.as_secs_f64();

    if total_secs < 1.0 {
        format!("{:.3}s", total_secs)
    } else if total_secs < 60.0 {
        format!("{:.2}s", total_secs)
    } else if total_secs < 3600.0 {
        let minutes = (total_secs / 60.0) as u64;
        let seconds = total_secs % 60.0;
        format!("{}m {:02.0}s", minutes, seconds)
    } else {
        let hours = (total_secs / 3600.0) as u64;
        let minutes = ((total_secs % 3600.0) / 60.0) as u64;
        format!("{}h {:02}m", hours, minutes)
    }
}

/// Format timing as forward/backward pair.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use powers_rs::display::components::statistics::format_timing_pair;
///
/// let fwd = Duration::from_millis(18);
/// let bwd = Duration::from_millis(34);
/// assert_eq!(format_timing_pair(fwd, bwd), "0.018s / 0.034s");
/// ```
#[must_use]
pub fn format_timing_pair(forward: Duration, backward: Duration) -> String {
    format!(
        "{} / {}",
        format_duration_compact(forward),
        format_duration_compact(backward)
    )
}

/// Format a percentage value with sign.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::statistics::format_percentage_change;
///
/// assert_eq!(format_percentage_change(18.6), "+18.6%");
/// assert_eq!(format_percentage_change(-5.2), "-5.2%");
/// assert_eq!(format_percentage_change(0.0), "+0.0%");
/// ```
#[must_use]
pub fn format_percentage_change(value: f64) -> String {
    format!("{:+.1}%", value)
}

/// Format gap percentage.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::statistics::format_gap;
///
/// assert_eq!(format_gap(26.4), "26.4%");
/// assert_eq!(format_gap(2.47), "2.5%");
/// ```
#[must_use]
pub fn format_gap(gap_percentage: f64) -> String {
    format!("{:.1}%", gap_percentage)
}

/// Format duration as HH:MM:SS.mmm.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use powers_rs::display::components::statistics::format_duration_hms;
///
/// assert_eq!(format_duration_hms(Duration::from_millis(511)), "00:00:00.511");
/// assert_eq!(format_duration_hms(Duration::from_secs(3661)), "01:01:01.000");
/// ```
#[must_use]
pub fn format_duration_hms(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = duration.subsec_millis();

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_cost_scientific() {
        assert_eq!(format_cost(123456.0, true), "1.23e5");
        assert_eq!(format_cost(0.00123, true), "1.23e-3");
    }

    #[test]
    fn test_format_cost_fixed() {
        assert_eq!(format_cost(1234.56, false), "1234.56");
        assert_eq!(format_cost(42.0, false), "42.00");
    }

    #[test]
    fn test_format_cost_edge_cases() {
        assert_eq!(format_cost(f64::NAN, true), "NaN");
        assert_eq!(format_cost(f64::INFINITY, true), "+∞");
        assert_eq!(format_cost(f64::NEG_INFINITY, true), "-∞");
    }

    #[test]
    fn test_format_duration_compact() {
        assert_eq!(
            format_duration_compact(Duration::from_millis(34)),
            "0.034s"
        );
        assert_eq!(format_duration_compact(Duration::from_secs(65)), "1m 05s");
        assert_eq!(
            format_duration_compact(Duration::from_secs(3725)),
            "1h 02m"
        );
        assert_eq!(format_duration_compact(Duration::from_secs(0)), "0.000s");
    }

    #[test]
    fn test_format_cost_stats() {
        let stats = CostStatistics {
            mean: 128000.0,
            std_dev: 3200.0,
            min: 124000.0,
            max: 135000.0,
            count: 4,
        };
        let result = format_cost_stats(&stats, &StatisticsFormat::default());
        assert!(result.contains("μ=1.28e5"));
        assert!(result.contains("σ=3.20e3"));
        assert!(result.contains("n=4"));
    }

    #[test]
    fn test_format_cost_stats_no_count() {
        let stats = CostStatistics {
            mean: 128000.0,
            std_dev: 3200.0,
            min: 124000.0,
            max: 135000.0,
            count: 4,
        };
        let format = StatisticsFormat {
            show_count: false,
            ..Default::default()
        };
        let result = format_cost_stats(&stats, &format);
        assert!(!result.contains("n="));
    }

    #[test]
    fn test_format_percentage_change() {
        assert_eq!(format_percentage_change(18.6), "+18.6%");
        assert_eq!(format_percentage_change(-5.2), "-5.2%");
        assert_eq!(format_percentage_change(0.0), "+0.0%");
    }

    #[test]
    fn test_format_timing_pair() {
        let fwd = Duration::from_millis(18);
        let bwd = Duration::from_millis(34);
        assert_eq!(format_timing_pair(fwd, bwd), "0.018s / 0.034s");
    }

    #[test]
    fn test_format_duration_hms() {
        assert_eq!(
            format_duration_hms(Duration::from_millis(511)),
            "00:00:00.511"
        );
        assert_eq!(
            format_duration_hms(Duration::from_secs(3661)),
            "01:01:01.000"
        );
        assert_eq!(format_duration_hms(Duration::from_secs(0)), "00:00:00.000");
    }

    #[test]
    fn test_format_gap() {
        assert_eq!(format_gap(26.4), "26.4%");
        assert_eq!(format_gap(2.47), "2.5%");
    }
}
