//! Trend indicators and status icons.
//!
//! Provides visual indicators for trends (arrows), bound changes, and training status.

use crate::display::components::color::{colorize, ColorConfig, SemanticColor};

/// Direction of a trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Improving (gap decreasing, bound increasing)
    Improving,
    /// Worsening (gap increasing, bound decreasing)
    Worsening,
    /// Stable (minimal change)
    Stable,
    /// Unknown (no previous value to compare)
    Unknown,
}

impl TrendDirection {
    /// Returns true if this trend is positive.
    #[must_use]
    pub const fn is_positive(self) -> bool {
        matches!(self, Self::Improving)
    }
}

/// Configuration for trend calculation.
#[derive(Debug, Clone, Copy)]
pub struct TrendConfig {
    /// Threshold for considering a change significant (percentage)
    pub significance_threshold: f64,
}

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            significance_threshold: 0.1, // 0.1% change is significant
        }
    }
}

/// Get trend arrow based on direction.
///
/// # Returns
///
/// - "↓" (improving)
/// - "↑" (worsening)
/// - "→" (stable)
/// - "" (unknown)
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::indicators::{trend_arrow, TrendDirection};
///
/// assert_eq!(trend_arrow(TrendDirection::Improving), "↓");
/// assert_eq!(trend_arrow(TrendDirection::Worsening), "↑");
/// assert_eq!(trend_arrow(TrendDirection::Stable), "→");
/// assert_eq!(trend_arrow(TrendDirection::Unknown), "");
/// ```
#[must_use]
pub const fn trend_arrow(direction: TrendDirection) -> &'static str {
    match direction {
        TrendDirection::Improving => "↓",
        TrendDirection::Worsening => "↑",
        TrendDirection::Stable => "→",
        TrendDirection::Unknown => "",
    }
}

/// Get trend arrow with color applied.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::{
///     indicators::{trend_arrow_colored, TrendDirection},
///     color::ColorConfig,
/// };
///
/// let config = ColorConfig::new(true);
/// let arrow = trend_arrow_colored(TrendDirection::Improving, &config);
/// assert!(arrow.contains("↓"));
/// ```
#[must_use]
pub fn trend_arrow_colored(
    direction: TrendDirection,
    color_config: &ColorConfig,
) -> String {
    let arrow = trend_arrow(direction);
    if arrow.is_empty() {
        return String::new();
    }

    match direction {
        TrendDirection::Improving => {
            colorize(arrow, SemanticColor::Good, color_config)
        }
        TrendDirection::Worsening => {
            colorize(arrow, SemanticColor::Bad, color_config)
        }
        TrendDirection::Stable => {
            colorize(arrow, SemanticColor::Caution, color_config)
        }
        TrendDirection::Unknown => arrow.to_string(),
    }
}

/// Calculate trend direction for gap (lower is better).
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::indicators::{gap_trend, TrendConfig, TrendDirection};
///
/// let config = TrendConfig::default();
///
/// // Gap decreased from 20% to 15% = improving
/// assert_eq!(gap_trend(15.0, Some(20.0), &config), TrendDirection::Improving);
///
/// // Gap increased from 10% to 15% = worsening
/// assert_eq!(gap_trend(15.0, Some(10.0), &config), TrendDirection::Worsening);
///
/// // No previous value
/// assert_eq!(gap_trend(10.0, None, &config), TrendDirection::Unknown);
/// ```
#[must_use]
pub fn gap_trend(
    current: f64,
    previous: Option<f64>,
    config: &TrendConfig,
) -> TrendDirection {
    let Some(prev) = previous else {
        return TrendDirection::Unknown;
    };

    if prev == 0.0 {
        return if current < 0.0 {
            TrendDirection::Improving
        } else if current > 0.0 {
            TrendDirection::Worsening
        } else {
            TrendDirection::Stable
        };
    }

    let change_pct = ((current - prev) / prev.abs()) * 100.0;

    if change_pct < -config.significance_threshold {
        TrendDirection::Improving // Gap decreased
    } else if change_pct > config.significance_threshold {
        TrendDirection::Worsening // Gap increased
    } else {
        TrendDirection::Stable
    }
}

/// Calculate trend direction for bound (higher is better).
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::indicators::{bound_trend, TrendConfig, TrendDirection};
///
/// let config = TrendConfig::default();
///
/// // Bound increased = improving
/// assert_eq!(bound_trend(120000.0, Some(100000.0), &config), TrendDirection::Improving);
///
/// // Bound decreased = worsening
/// assert_eq!(bound_trend(80000.0, Some(100000.0), &config), TrendDirection::Worsening);
/// ```
#[must_use]
pub fn bound_trend(
    current: f64,
    previous: Option<f64>,
    config: &TrendConfig,
) -> TrendDirection {
    let Some(prev) = previous else {
        return TrendDirection::Unknown;
    };

    if prev == 0.0 {
        return if current > 0.0 {
            TrendDirection::Improving
        } else if current < 0.0 {
            TrendDirection::Worsening
        } else {
            TrendDirection::Stable
        };
    }

    let change_pct = ((current - prev) / prev.abs()) * 100.0;

    if change_pct > config.significance_threshold {
        TrendDirection::Improving // Bound increased
    } else if change_pct < -config.significance_threshold {
        TrendDirection::Worsening // Bound decreased
    } else {
        TrendDirection::Stable
    }
}

/// Get bound change indicator.
///
/// # Returns
///
/// - "▲" (increased)
/// - "▼" (decreased)
/// - "─" (stable)
/// - "" (no previous value)
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::indicators::bound_change_indicator;
///
/// assert_eq!(bound_change_indicator(120.0, Some(100.0)), "▲");
/// assert_eq!(bound_change_indicator(80.0, Some(100.0)), "▼");
/// assert_eq!(bound_change_indicator(100.0, Some(100.0)), "─");
/// assert_eq!(bound_change_indicator(100.0, None), "");
/// ```
#[must_use]
pub fn bound_change_indicator(
    current: f64,
    previous: Option<f64>,
) -> &'static str {
    let Some(prev) = previous else {
        return "";
    };

    if current > prev {
        "▲"
    } else if current < prev {
        "▼"
    } else {
        "─"
    }
}

/// Format bound change with percentage and indicator.
///
/// # Returns
///
/// - "▲ +18.6%" (increased)
/// - "▼ -5.2%" (decreased)
/// - "" (no previous value)
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::{
///     indicators::format_bound_change,
///     color::ColorConfig,
/// };
///
/// let config = ColorConfig::new(false);
/// assert_eq!(format_bound_change(120.0, Some(100.0), &config), "▲ +20.0%");
/// assert_eq!(format_bound_change(80.0, Some(100.0), &config), "▼ -20.0%");
/// assert_eq!(format_bound_change(100.0, None, &config), "");
/// ```
#[must_use]
pub fn format_bound_change(
    current: f64,
    previous: Option<f64>,
    color_config: &ColorConfig,
) -> String {
    let Some(prev) = previous else {
        return String::new();
    };

    if prev == 0.0 {
        return String::new();
    }

    let change_pct = ((current - prev) / prev.abs()) * 100.0;
    let indicator = bound_change_indicator(current, Some(prev));
    let formatted = format!("{} {:+.1}%", indicator, change_pct);

    let color = if change_pct > 0.0 {
        SemanticColor::Good
    } else if change_pct < 0.0 {
        SemanticColor::Bad
    } else {
        SemanticColor::Normal
    };

    colorize(&formatted, color, color_config)
}

/// Get status icon for training state.
///
/// # Returns
///
/// - "✓" (converged)
/// - "⋯" (not converged)
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::indicators::status_icon;
///
/// assert_eq!(status_icon(true), "✓");
/// assert_eq!(status_icon(false), "⋯");
/// ```
#[must_use]
pub const fn status_icon(converged: bool) -> &'static str {
    if converged {
        "✓"
    } else {
        "⋯"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trend_arrow() {
        assert_eq!(trend_arrow(TrendDirection::Improving), "↓");
        assert_eq!(trend_arrow(TrendDirection::Worsening), "↑");
        assert_eq!(trend_arrow(TrendDirection::Stable), "→");
        assert_eq!(trend_arrow(TrendDirection::Unknown), "");
    }

    #[test]
    fn test_gap_trend_improving() {
        let config = TrendConfig::default();
        // Gap went from 20% to 15% = improving
        assert_eq!(
            gap_trend(15.0, Some(20.0), &config),
            TrendDirection::Improving
        );
    }

    #[test]
    fn test_gap_trend_worsening() {
        let config = TrendConfig::default();
        // Gap went from 10% to 15% = worsening
        assert_eq!(
            gap_trend(15.0, Some(10.0), &config),
            TrendDirection::Worsening
        );
    }

    #[test]
    fn test_gap_trend_stable() {
        let config = TrendConfig {
            significance_threshold: 1.0,
        };
        // Gap changed by 0.5% which is below 1% threshold
        assert_eq!(
            gap_trend(10.05, Some(10.0), &config),
            TrendDirection::Stable
        );
    }

    #[test]
    fn test_gap_trend_unknown() {
        let config = TrendConfig::default();
        assert_eq!(gap_trend(10.0, None, &config), TrendDirection::Unknown);
    }

    #[test]
    fn test_bound_trend_improving() {
        let config = TrendConfig::default();
        // Bound increased = improving
        assert_eq!(
            bound_trend(120000.0, Some(100000.0), &config),
            TrendDirection::Improving
        );
    }

    #[test]
    fn test_bound_trend_worsening() {
        let config = TrendConfig::default();
        // Bound decreased = worsening
        assert_eq!(
            bound_trend(80000.0, Some(100000.0), &config),
            TrendDirection::Worsening
        );
    }

    #[test]
    fn test_bound_change_indicator() {
        assert_eq!(bound_change_indicator(120.0, Some(100.0)), "▲");
        assert_eq!(bound_change_indicator(80.0, Some(100.0)), "▼");
        assert_eq!(bound_change_indicator(100.0, Some(100.0)), "─");
        assert_eq!(bound_change_indicator(100.0, None), "");
    }

    #[test]
    fn test_format_bound_change() {
        let config = ColorConfig::new(false);

        assert_eq!(
            format_bound_change(120.0, Some(100.0), &config),
            "▲ +20.0%"
        );
        assert_eq!(format_bound_change(80.0, Some(100.0), &config), "▼ -20.0%");
        assert_eq!(format_bound_change(100.0, None, &config), "");
    }

    #[test]
    fn test_status_icon() {
        assert_eq!(status_icon(true), "✓");
        assert_eq!(status_icon(false), "⋯");
    }

    #[test]
    fn test_trend_direction_is_positive() {
        assert!(TrendDirection::Improving.is_positive());
        assert!(!TrendDirection::Worsening.is_positive());
        assert!(!TrendDirection::Stable.is_positive());
        assert!(!TrendDirection::Unknown.is_positive());
    }
}
