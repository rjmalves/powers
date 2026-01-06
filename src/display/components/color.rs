//! Color utilities for semantic terminal styling.
//!
//! Provides consistent color application across all display renderers with
//! semantic color categories (Good, Bad, Caution, etc.) that adapt based on
//! terminal capabilities.

use crossterm::style::Stylize;

/// Semantic color categories for display elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticColor {
    /// Good/positive values (e.g., improving gap)
    Good,
    /// Bad/negative values (e.g., worsening gap)
    Bad,
    /// Caution/neutral values
    Caution,
    /// Informational text
    Info,
    /// Muted/secondary text
    Muted,
    /// Emphasis/highlight
    Emphasis,
    /// Standard text (no special styling)
    Normal,
}

/// Color configuration for styled output.
#[derive(Debug, Clone, Copy)]
pub struct ColorConfig {
    /// Whether colors are enabled
    pub enabled: bool,
}

impl ColorConfig {
    /// Create a new color configuration.
    #[must_use]
    pub const fn new(enabled: bool) -> Self {
        Self { enabled }
    }
}

impl Default for ColorConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Apply semantic color to text.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::color::{colorize, SemanticColor, ColorConfig};
///
/// let config = ColorConfig::new(true);
/// let colored = colorize("Success", SemanticColor::Good, &config);
/// // Returns ANSI-colored green text
/// ```
#[must_use]
pub fn colorize(
    text: &str,
    color: SemanticColor,
    config: &ColorConfig,
) -> String {
    if !config.enabled {
        return text.to_string();
    }

    match color {
        SemanticColor::Good => text.green().to_string(),
        SemanticColor::Bad => text.red().to_string(),
        SemanticColor::Caution => text.yellow().to_string(),
        SemanticColor::Info => text.cyan().to_string(),
        SemanticColor::Muted => text.dark_grey().to_string(),
        SemanticColor::Emphasis => text.white().bold().to_string(),
        SemanticColor::Normal => text.to_string(),
    }
}

/// Apply color for trend direction.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::color::{color_trend, ColorConfig};
///
/// let config = ColorConfig::new(true);
/// let improving = color_trend("↓", true, &config); // Green
/// let worsening = color_trend("↑", false, &config); // Red
/// ```
#[must_use]
pub fn color_trend(
    text: &str,
    improving: bool,
    config: &ColorConfig,
) -> String {
    if !config.enabled {
        return text.to_string();
    }

    if improving {
        text.green().to_string()
    } else {
        text.red().to_string()
    }
}

/// Apply color for gap percentage values.
///
/// Color thresholds:
/// - gap < 5%: Green (good)
/// - gap < 20%: Yellow (caution)
/// - gap >= 20%: Red (bad)
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::color::{color_gap_percentage, ColorConfig};
///
/// let config = ColorConfig::new(true);
/// let low = color_gap_percentage(3.0, "3.0%", &config);  // Green
/// let mid = color_gap_percentage(10.0, "10.0%", &config); // Yellow
/// let high = color_gap_percentage(25.0, "25.0%", &config); // Red
/// ```
#[must_use]
pub fn color_gap_percentage(
    gap: f64,
    text: &str,
    config: &ColorConfig,
) -> String {
    if !config.enabled {
        return text.to_string();
    }

    if gap < 5.0 {
        text.green().to_string()
    } else if gap < 20.0 {
        text.yellow().to_string()
    } else {
        text.red().to_string()
    }
}

/// Format a value with optional color based on comparison to previous value.
///
/// Colors the formatted text green if improved, red if worsened, normal otherwise.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::color::{color_comparison, ColorConfig};
///
/// let config = ColorConfig::new(true);
/// let improved = color_comparison(120.0, Some(100.0), "120.0", &config); // Green (increased)
/// let worsened = color_comparison(80.0, Some(100.0), "80.0", &config);   // Red (decreased)
/// let first = color_comparison(100.0, None, "100.0", &config);           // Normal
/// ```
#[must_use]
pub fn color_comparison(
    current: f64,
    previous: Option<f64>,
    formatted: &str,
    config: &ColorConfig,
) -> String {
    if !config.enabled {
        return formatted.to_string();
    }

    let Some(prev) = previous else {
        return formatted.to_string();
    };

    if current > prev {
        formatted.green().to_string()
    } else if current < prev {
        formatted.red().to_string()
    } else {
        formatted.to_string()
    }
}

/// Style text as bold.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::color::{bold, ColorConfig};
///
/// let config = ColorConfig::new(true);
/// let emphasized = bold("Important", &config);
/// ```
#[must_use]
pub fn bold(text: &str, config: &ColorConfig) -> String {
    if !config.enabled {
        return text.to_string();
    }

    text.bold().to_string()
}

/// Style text as dim/muted.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::color::{dim, ColorConfig};
///
/// let config = ColorConfig::new(true);
/// let muted = dim("Secondary info", &config);
/// ```
#[must_use]
pub fn dim(text: &str, config: &ColorConfig) -> String {
    if !config.enabled {
        return text.to_string();
    }

    text.dim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colorize_disabled() {
        let config = ColorConfig::new(false);
        assert_eq!(colorize("test", SemanticColor::Good, &config), "test");
        assert_eq!(colorize("test", SemanticColor::Bad, &config), "test");
    }

    #[test]
    fn test_colorize_enabled_contains_ansi() {
        let config = ColorConfig::new(true);
        let result = colorize("test", SemanticColor::Good, &config);
        assert!(result.contains("\x1b[")); // ANSI escape
        assert!(result.contains("test"));
    }

    #[test]
    fn test_color_trend() {
        let config = ColorConfig::new(false);
        assert_eq!(color_trend("↓", true, &config), "↓");
        assert_eq!(color_trend("↑", false, &config), "↑");

        let config = ColorConfig::new(true);
        let improving = color_trend("↓", true, &config);
        assert!(improving.contains("\x1b["));
        assert!(improving.contains("↓"));
    }

    #[test]
    fn test_color_gap_percentage_thresholds() {
        let config = ColorConfig::new(true);

        // Low gap (< 5%) should be green
        let low = color_gap_percentage(4.0, "4.0%", &config);
        assert!(low.contains("\x1b["));

        // Medium gap (< 20%) should be yellow
        let mid = color_gap_percentage(15.0, "15.0%", &config);
        assert!(mid.contains("\x1b["));

        // High gap (>= 20%) should be red
        let high = color_gap_percentage(25.0, "25.0%", &config);
        assert!(high.contains("\x1b["));
    }

    #[test]
    fn test_color_gap_percentage_disabled() {
        let config = ColorConfig::new(false);
        assert_eq!(color_gap_percentage(4.0, "4.0%", &config), "4.0%");
        assert_eq!(color_gap_percentage(15.0, "15.0%", &config), "15.0%");
        assert_eq!(color_gap_percentage(25.0, "25.0%", &config), "25.0%");
    }

    #[test]
    fn test_color_comparison_no_previous() {
        let config = ColorConfig::new(true);
        assert_eq!(color_comparison(100.0, None, "100.0", &config), "100.0");
    }

    #[test]
    fn test_color_comparison_improved() {
        let config = ColorConfig::new(true);
        let result = color_comparison(120.0, Some(100.0), "120.0", &config);
        assert!(result.contains("\x1b[")); // Has ANSI codes
        assert!(result.contains("120.0"));
    }

    #[test]
    fn test_color_comparison_worsened() {
        let config = ColorConfig::new(true);
        let result = color_comparison(80.0, Some(100.0), "80.0", &config);
        assert!(result.contains("\x1b["));
        assert!(result.contains("80.0"));
    }

    #[test]
    fn test_color_comparison_same() {
        let config = ColorConfig::new(true);
        let result = color_comparison(100.0, Some(100.0), "100.0", &config);
        // Should be plain (no improvement or worsening)
        assert_eq!(result, "100.0");
    }

    #[test]
    fn test_bold() {
        let config = ColorConfig::new(false);
        assert_eq!(bold("test", &config), "test");

        let config = ColorConfig::new(true);
        let result = bold("test", &config);
        assert!(result.contains("\x1b["));
        assert!(result.contains("test"));
    }

    #[test]
    fn test_dim() {
        let config = ColorConfig::new(false);
        assert_eq!(dim("test", &config), "test");

        let config = ColorConfig::new(true);
        let result = dim("test", &config);
        assert!(result.contains("\x1b["));
        assert!(result.contains("test"));
    }
}
