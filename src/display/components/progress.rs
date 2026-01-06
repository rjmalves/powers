//! Progress bar component for visualizing training progress.
//!
//! Provides customizable progress bars with ETA calculation and multiple visual styles.

use crate::display::components::{
    color::ColorConfig, statistics::format_duration_compact,
};
use std::time::{Duration, Instant};

/// Style variants for progress bars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressStyle {
    /// Standard block characters: [████░░░░░░]
    #[default]
    Block,
    /// ASCII compatible: [====------]
    Ascii,
    /// Thin bar: [▓▓▓▓░░░░░░]
    Thin,
}

/// Configuration for progress bar rendering.
#[derive(Debug, Clone)]
pub struct ProgressBarConfig {
    /// Total width of the bar (including brackets)
    pub width: u16,
    /// Style of fill characters
    pub style: ProgressStyle,
    /// Show percentage after bar
    pub show_percentage: bool,
    /// Show ETA after bar
    pub show_eta: bool,
    /// Show iteration count (e.g., "3/10")
    pub show_count: bool,
}

impl Default for ProgressBarConfig {
    fn default() -> Self {
        Self {
            width: 30,
            style: ProgressStyle::Block,
            show_percentage: true,
            show_eta: true,
            show_count: true,
        }
    }
}

/// Progress bar state for rendering.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::{
///     progress::{ProgressBar, ProgressBarConfig},
///     color::ColorConfig,
/// };
///
/// let mut bar = ProgressBar::new(10, ProgressBarConfig::default());
/// bar.set(5);
/// assert_eq!(bar.fraction(), 0.5);
///
/// let config = ColorConfig::new(false);
/// let rendered = bar.render(&config);
/// assert!(rendered.contains("50%"));
/// ```
pub struct ProgressBar {
    /// Current progress value
    pub current: usize,
    /// Total expected value
    pub total: usize,
    /// Start time for ETA calculation
    pub started_at: Instant,
    /// Configuration
    pub config: ProgressBarConfig,
}

impl ProgressBar {
    /// Create a new progress bar.
    ///
    /// # Arguments
    ///
    /// * `total` - Total number of iterations/steps
    /// * `config` - Progress bar configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::display::components::progress::{ProgressBar, ProgressBarConfig};
    ///
    /// let bar = ProgressBar::new(100, ProgressBarConfig::default());
    /// assert_eq!(bar.total, 100);
    /// assert_eq!(bar.current, 0);
    /// ```
    #[must_use]
    pub fn new(total: usize, config: ProgressBarConfig) -> Self {
        Self {
            current: 0,
            total,
            started_at: Instant::now(),
            config,
        }
    }

    /// Update the current progress.
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::display::components::progress::{ProgressBar, ProgressBarConfig};
    ///
    /// let mut bar = ProgressBar::new(10, ProgressBarConfig::default());
    /// bar.set(5);
    /// assert_eq!(bar.current, 5);
    /// ```
    pub fn set(&mut self, current: usize) {
        self.current = current;
    }

    /// Increment progress by 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::display::components::progress::{ProgressBar, ProgressBarConfig};
    ///
    /// let mut bar = ProgressBar::new(10, ProgressBarConfig::default());
    /// bar.inc();
    /// bar.inc();
    /// assert_eq!(bar.current, 2);
    /// ```
    pub fn inc(&mut self) {
        self.current = self.current.saturating_add(1);
    }

    /// Get the completion fraction (0.0 to 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::display::components::progress::{ProgressBar, ProgressBarConfig};
    ///
    /// let mut bar = ProgressBar::new(10, ProgressBarConfig::default());
    /// assert_eq!(bar.fraction(), 0.0);
    /// bar.set(5);
    /// assert_eq!(bar.fraction(), 0.5);
    /// bar.set(10);
    /// assert_eq!(bar.fraction(), 1.0);
    /// ```
    #[must_use]
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            return 1.0; // Avoid division by zero
        }
        (self.current as f64 / self.total as f64).clamp(0.0, 1.0)
    }

    /// Calculate estimated time remaining.
    ///
    /// Returns `None` if ETA cannot be calculated (e.g., no progress yet).
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::display::components::progress::{ProgressBar, ProgressBarConfig};
    ///
    /// let bar = ProgressBar::new(10, ProgressBarConfig::default());
    /// // No progress yet, can't calculate ETA
    /// assert!(bar.eta().is_none());
    /// ```
    #[must_use]
    pub fn eta(&self) -> Option<Duration> {
        if self.current == 0 {
            return None; // Can't calculate yet
        }

        let elapsed = self.started_at.elapsed();
        let rate = self.current as f64 / elapsed.as_secs_f64();

        if rate <= 0.0 {
            return None;
        }

        let remaining = self.total.saturating_sub(self.current);
        let eta_secs = remaining as f64 / rate;

        Some(Duration::from_secs_f64(eta_secs))
    }

    /// Render the progress bar as a string.
    ///
    /// # Arguments
    ///
    /// * `color_config` - Color configuration for styling
    ///
    /// # Returns
    ///
    /// Formatted progress bar string with optional percentage, count, and ETA
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::display::components::{
    ///     progress::{ProgressBar, ProgressBarConfig},
    ///     color::ColorConfig,
    /// };
    ///
    /// let mut bar = ProgressBar::new(10, ProgressBarConfig::default());
    /// bar.set(5);
    /// let config = ColorConfig::new(false);
    /// let rendered = bar.render(&config);
    /// assert!(rendered.contains("50%"));
    /// assert!(rendered.contains("5/10"));
    /// ```
    #[must_use]
    pub fn render(&self, _color_config: &ColorConfig) -> String {
        let bar = self.render_bar_only(self.config.width);
        let mut parts = vec![bar];

        if self.config.show_percentage {
            parts.push(format!("{:3.0}%", self.fraction() * 100.0));
        }

        if self.config.show_count {
            parts.push(format!("{}/{} iter", self.current, self.total));
        }

        if self.config.show_eta {
            if let Some(eta) = self.eta() {
                parts.push(format!("ETA: {}", format_duration_compact(eta)));
            } else if self.current == 0 {
                parts.push("ETA: calculating...".to_string());
            }
        }

        parts.join(" | ")
    }

    /// Render a compact version (just the bar, no extras).
    ///
    /// # Arguments
    ///
    /// * `width` - Total width of the bar including brackets
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::display::components::progress::{ProgressBar, ProgressBarConfig, ProgressStyle};
    ///
    /// let mut bar = ProgressBar::new(10, ProgressBarConfig {
    ///     width: 12,
    ///     style: ProgressStyle::Ascii,
    ///     ..Default::default()
    /// });
    /// bar.set(5);
    /// let rendered = bar.render_bar_only(12);
    /// assert_eq!(rendered, "[=====-----]");
    /// ```
    #[must_use]
    pub fn render_bar_only(&self, width: u16) -> String {
        let (filled_char, empty_char) = match self.config.style {
            ProgressStyle::Block => ('█', '░'),
            ProgressStyle::Ascii => ('=', '-'),
            ProgressStyle::Thin => ('▓', '░'),
        };

        let inner_width = width.saturating_sub(2) as usize; // Account for [ ]
        let filled = (self.fraction() * inner_width as f64).round() as usize;
        let empty = inner_width.saturating_sub(filled);

        format!(
            "[{}{}]",
            filled_char.to_string().repeat(filled),
            empty_char.to_string().repeat(empty)
        )
    }
}

/// Create a simple inline progress indicator.
///
/// # Arguments
///
/// * `fraction` - Completion fraction (0.0 to 1.0)
/// * `width` - Total width including brackets
/// * `style` - Visual style for the bar
///
/// # Returns
///
/// Formatted progress bar string
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::progress::{simple_progress, ProgressStyle};
///
/// let bar = simple_progress(0.5, 12, ProgressStyle::Ascii);
/// assert_eq!(bar, "[=====-----]");
///
/// let bar = simple_progress(0.75, 12, ProgressStyle::Block);
/// assert_eq!(bar, "[███████░░░]");
/// ```
#[must_use]
pub fn simple_progress(
    fraction: f64,
    width: u16,
    style: ProgressStyle,
) -> String {
    let (filled_char, empty_char) = match style {
        ProgressStyle::Block => ('█', '░'),
        ProgressStyle::Ascii => ('=', '-'),
        ProgressStyle::Thin => ('▓', '░'),
    };

    let fraction = fraction.clamp(0.0, 1.0);
    let inner_width = width.saturating_sub(2) as usize;
    let filled = (fraction * inner_width as f64).round() as usize;
    let empty = inner_width.saturating_sub(filled);

    format!(
        "[{}{}]",
        filled_char.to_string().repeat(filled),
        empty_char.to_string().repeat(empty)
    )
}

/// Calculate progress toward a target gap percentage.
///
/// Returns a percentage (0-100) indicating how much progress has been made
/// from the initial gap to the target gap.
///
/// # Arguments
///
/// * `current_gap` - Current optimality gap percentage
/// * `initial_gap` - Initial gap at start of training (if known)
/// * `target_gap` - Target gap percentage to achieve
///
/// # Returns
///
/// Progress percentage from 0.0 to 100.0:
/// - 0% at start (current gap = initial gap)
/// - 100% when current gap ≤ target gap
/// - Proportional progress in between
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::progress::gap_progress;
///
/// // At start: 0% progress
/// assert_eq!(gap_progress(50.0, Some(50.0), 5.0), 0.0);
///
/// // Halfway there: 50% progress
/// let progress = gap_progress(27.5, Some(50.0), 5.0);
/// assert!((progress - 50.0).abs() < 0.1);
///
/// // Target achieved: 100% progress
/// assert_eq!(gap_progress(4.0, Some(50.0), 5.0), 100.0);
/// assert_eq!(gap_progress(5.0, Some(50.0), 5.0), 100.0);
/// ```
#[must_use]
pub fn gap_progress(
    current_gap: f64,
    initial_gap: Option<f64>,
    target_gap: f64,
) -> f64 {
    // Default to 100% initial gap if unknown
    let initial = initial_gap.unwrap_or(100.0);

    // Already at or below target
    if current_gap <= target_gap {
        return 100.0;
    }

    // Started at or below target
    if initial <= target_gap {
        return 100.0;
    }

    // Calculate progress as percentage of distance covered
    let total_distance = initial - target_gap;
    let distance_covered = initial - current_gap;

    ((distance_covered / total_distance) * 100.0).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_fraction() {
        let mut bar = ProgressBar::new(10, ProgressBarConfig::default());
        assert_eq!(bar.fraction(), 0.0);
        bar.set(5);
        assert_eq!(bar.fraction(), 0.5);
        bar.set(10);
        assert_eq!(bar.fraction(), 1.0);
    }

    #[test]
    fn test_progress_fraction_zero_total() {
        let bar = ProgressBar::new(0, ProgressBarConfig::default());
        assert_eq!(bar.fraction(), 1.0);
    }

    #[test]
    fn test_render_bar_block_style() {
        let mut bar = ProgressBar::new(
            10,
            ProgressBarConfig {
                width: 12,
                style: ProgressStyle::Block,
                ..Default::default()
            },
        );
        bar.set(5);
        let rendered = bar.render_bar_only(12);
        assert_eq!(rendered, "[█████░░░░░]");
    }

    #[test]
    fn test_render_bar_ascii_style() {
        let mut bar = ProgressBar::new(
            10,
            ProgressBarConfig {
                width: 12,
                style: ProgressStyle::Ascii,
                ..Default::default()
            },
        );
        bar.set(5);
        let rendered = bar.render_bar_only(12);
        assert_eq!(rendered, "[=====-----]");
    }

    #[test]
    fn test_render_bar_thin_style() {
        let mut bar = ProgressBar::new(
            10,
            ProgressBarConfig {
                width: 12,
                style: ProgressStyle::Thin,
                ..Default::default()
            },
        );
        bar.set(5);
        let rendered = bar.render_bar_only(12);
        assert_eq!(rendered, "[▓▓▓▓▓░░░░░]");
    }

    #[test]
    fn test_simple_progress() {
        let result = simple_progress(0.5, 12, ProgressStyle::Ascii);
        assert_eq!(result, "[=====-----]");

        let result = simple_progress(0.0, 12, ProgressStyle::Block);
        assert_eq!(result, "[░░░░░░░░░░]");

        let result = simple_progress(1.0, 12, ProgressStyle::Block);
        assert_eq!(result, "[██████████]");
    }

    #[test]
    fn test_eta_none_at_start() {
        let bar = ProgressBar::new(10, ProgressBarConfig::default());
        assert!(bar.eta().is_none());
    }

    #[test]
    fn test_increment() {
        let mut bar = ProgressBar::new(10, ProgressBarConfig::default());
        bar.inc();
        assert_eq!(bar.current, 1);
        bar.inc();
        assert_eq!(bar.current, 2);
    }

    #[test]
    fn test_render_with_percentage() {
        let mut bar = ProgressBar::new(
            10,
            ProgressBarConfig {
                width: 12,
                show_percentage: true,
                show_count: false,
                show_eta: false,
                ..Default::default()
            },
        );
        bar.set(5);
        let config = ColorConfig::new(false);
        let rendered = bar.render(&config);
        assert!(rendered.contains("50%"));
    }

    #[test]
    fn test_render_with_count() {
        let mut bar = ProgressBar::new(
            10,
            ProgressBarConfig {
                width: 12,
                show_percentage: false,
                show_count: true,
                show_eta: false,
                ..Default::default()
            },
        );
        bar.set(5);
        let config = ColorConfig::new(false);
        let rendered = bar.render(&config);
        assert!(rendered.contains("5/10"));
    }

    #[test]
    fn test_gap_progress_at_start() {
        // Initial gap 50%, target 5%, current 50% = 0% progress
        assert_eq!(gap_progress(50.0, Some(50.0), 5.0), 0.0);
    }

    #[test]
    fn test_gap_progress_halfway() {
        // Initial 50%, target 5%, current 27.5% = 50% progress
        // (50 - 27.5) / (50 - 5) = 22.5 / 45 = 50%
        let progress = gap_progress(27.5, Some(50.0), 5.0);
        assert!((progress - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gap_progress_achieved() {
        // Current gap at or below target = 100%
        assert_eq!(gap_progress(4.0, Some(50.0), 5.0), 100.0);
        assert_eq!(gap_progress(5.0, Some(50.0), 5.0), 100.0);
    }

    #[test]
    fn test_gap_progress_no_initial() {
        // No initial gap defaults to 100%
        let progress = gap_progress(50.0, None, 5.0);
        assert!(progress > 0.0); // Some progress from assumed 100%
                                 // (100 - 50) / (100 - 5) = 50 / 95 ≈ 52.6%
        assert!((progress - 52.63).abs() < 0.1);
    }

    #[test]
    fn test_gap_progress_already_at_target() {
        // Already at target = 100%
        assert_eq!(gap_progress(3.0, Some(3.0), 5.0), 100.0);
    }

    #[test]
    fn test_gap_progress_clamped() {
        // Gap worse than initial shouldn't go negative
        assert_eq!(gap_progress(60.0, Some(50.0), 5.0), 0.0);
    }

    #[test]
    fn test_gap_progress_large_gap() {
        // Large initial gap
        let progress = gap_progress(75.0, Some(100.0), 10.0);
        // (100 - 75) / (100 - 10) = 25 / 90 ≈ 27.8%
        assert!((progress - 27.78).abs() < 0.1);
    }
}
