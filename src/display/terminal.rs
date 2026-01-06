//! Terminal capability detection utilities.
//!
//! Provides functions to detect terminal features like color support
//! and interactivity for automatic output adaptation.

use std::io::{stdout, IsTerminal};

/// Terminal capability information.
#[derive(Debug, Clone)]
pub struct TerminalCapabilities {
    /// Whether stdout is connected to an interactive terminal.
    pub is_interactive: bool,

    /// Whether the terminal supports ANSI colors.
    pub supports_color: bool,

    /// Whether the terminal supports Unicode (UTF-8).
    pub supports_unicode: bool,

    /// Terminal width in columns (if detectable).
    pub width: Option<u16>,

    /// Terminal height in rows (if detectable).
    pub height: Option<u16>,
}

impl TerminalCapabilities {
    /// Detect terminal capabilities.
    ///
    /// Checks stdout for interactivity, color support, and dimensions.
    /// Uses conservative defaults when detection fails.
    pub fn detect() -> Self {
        let is_interactive = stdout().is_terminal();

        // Color support: interactive terminal and not explicitly disabled
        let supports_color = is_interactive && !is_color_disabled_by_env();

        // Unicode: assume supported on modern terminals
        // Could check LANG/LC_ALL but this is conservative enough
        let supports_unicode = is_interactive;

        // Terminal dimensions
        let (width, height) = crossterm::terminal::size()
            .map(|(w, h)| (Some(w), Some(h)))
            .unwrap_or((None, None));

        Self {
            is_interactive,
            supports_color,
            supports_unicode,
            width,
            height,
        }
    }

    /// Create capabilities for non-interactive (pipe/file) output.
    pub fn non_interactive() -> Self {
        Self {
            is_interactive: false,
            supports_color: false,
            supports_unicode: false,
            width: None,
            height: None,
        }
    }

    /// Create capabilities forcing color output.
    pub fn force_color() -> Self {
        let mut caps = Self::detect();
        caps.supports_color = true;
        caps
    }

    /// Apply ColorMode override.
    pub fn with_color_mode(mut self, mode: super::config::ColorMode) -> Self {
        match mode {
            super::config::ColorMode::Auto => { /* keep detected */ }
            super::config::ColorMode::Always => {
                self.supports_color = true;
            }
            super::config::ColorMode::Never => {
                self.supports_color = false;
            }
        }
        self
    }
}

/// Check if color is disabled by environment variables.
///
/// Respects common conventions:
/// - `NO_COLOR` environment variable (https://no-color.org/)
/// - `TERM=dumb`
fn is_color_disabled_by_env() -> bool {
    // NO_COLOR standard: presence disables color (value doesn't matter)
    if std::env::var("NO_COLOR").is_ok() {
        return true;
    }

    // TERM=dumb is a conventional signal for minimal terminal
    if let Ok(term) = std::env::var("TERM") {
        if term == "dumb" {
            return true;
        }
    }

    false
}

/// Check if output is being piped (quick check without full detection).
#[inline]
pub fn is_piped() -> bool {
    !stdout().is_terminal()
}

/// Get terminal width, or default if not detectable.
pub fn terminal_width_or(default: u16) -> u16 {
    crossterm::terminal::size()
        .map(|(w, _)| w)
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_interactive() {
        let caps = TerminalCapabilities::non_interactive();
        assert!(!caps.is_interactive);
        assert!(!caps.supports_color);
        assert!(!caps.supports_unicode);
    }

    #[test]
    fn test_force_color() {
        let caps = TerminalCapabilities::force_color();
        assert!(caps.supports_color);
    }

    #[test]
    fn test_with_color_mode_always() {
        let caps = TerminalCapabilities::non_interactive()
            .with_color_mode(super::super::config::ColorMode::Always);
        assert!(caps.supports_color);
    }

    #[test]
    fn test_with_color_mode_never() {
        let caps = TerminalCapabilities::force_color()
            .with_color_mode(super::super::config::ColorMode::Never);
        assert!(!caps.supports_color);
    }

    #[test]
    fn test_with_color_mode_auto() {
        let original = TerminalCapabilities::detect();
        let with_auto = original
            .clone()
            .with_color_mode(super::super::config::ColorMode::Auto);
        assert_eq!(original.supports_color, with_auto.supports_color);
    }

    // Environment variable tests
    // Note: These tests modify environment and should run with --test-threads=1
    #[test]
    fn test_is_color_disabled_by_env_no_color() {
        let original = std::env::var("NO_COLOR").ok();

        std::env::set_var("NO_COLOR", "1");
        assert!(is_color_disabled_by_env());

        // Restore
        match original {
            Some(v) => std::env::set_var("NO_COLOR", v),
            None => std::env::remove_var("NO_COLOR"),
        }
    }

    #[test]
    fn test_is_color_disabled_by_env_term_dumb() {
        let original = std::env::var("TERM").ok();

        std::env::set_var("TERM", "dumb");
        assert!(is_color_disabled_by_env());

        // Restore
        match original {
            Some(v) => std::env::set_var("TERM", v),
            None => std::env::remove_var("TERM"),
        }
    }

    #[test]
    fn test_terminal_width_or() {
        let width = terminal_width_or(80);
        assert!(width > 0);
    }
}
