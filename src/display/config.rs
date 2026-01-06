//! Display configuration types.
//!
//! Defines the display profile selection and configuration options
//! that control how output is rendered.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Display configuration from config.json file.
///
/// This is the serde-compatible input type that gets deserialized from JSON.
/// It gets converted to `DisplayConfig` at runtime with terminal detection applied.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DisplayConfigInput {
    /// Display profile selection.
    pub profile: DisplayProfile,

    /// Color output mode.
    pub color: ColorMode,

    /// Target convergence gap (optional).
    /// When set, enables progress visualization toward this target.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_gap: Option<f64>,
}

impl From<DisplayConfigInput> for DisplayConfig {
    fn from(input: DisplayConfigInput) -> Self {
        Self {
            profile: input.profile,
            color: input.color,
            target_gap: input.target_gap,
            // Runtime fields set later
            is_interactive: true,
            color_enabled: true,
        }
    }
}

/// Output display profile controlling verbosity and format.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum DisplayProfile {
    /// Full metrics with colors, real-time statistics, and trend indicators.
    /// Shows forward cost distributions, first-stage branching stats, timing breakdown.
    #[default]
    Advanced,

    /// Key metrics with simplified layout.
    /// Shows bounds, gap, timing without detailed statistics.
    Standard,

    /// Minimal output with progress bar and final summary only.
    /// Suitable for long runs where per-iteration detail is not needed.
    Minimal,

    /// Machine-readable JSON lines format.
    /// No ANSI codes, structured for parsing by external tools.
    Automation,
}

impl FromStr for DisplayProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "advanced" => Ok(Self::Advanced),
            "standard" => Ok(Self::Standard),
            "minimal" => Ok(Self::Minimal),
            "automation" | "json" => Ok(Self::Automation),
            _ => Err(format!(
                "Invalid display profile: '{}'. Valid options: advanced, standard, minimal, automation",
                s
            )),
        }
    }
}

/// Color output mode for terminal display.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ColorMode {
    /// Detect terminal capabilities automatically.
    /// Disables color for non-interactive terminals and pipes.
    #[default]
    Auto,

    /// Always use colors (even in pipes).
    Always,

    /// Never use colors.
    Never,
}

impl FromStr for ColorMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            _ => Err(format!(
                "Invalid color mode: '{}'. Valid options: auto, always, never",
                s
            )),
        }
    }
}

/// Configuration for the display system.
#[derive(Debug, Clone)]
pub struct DisplayConfig {
    /// Selected display profile.
    pub profile: DisplayProfile,

    /// Color output mode.
    pub color: ColorMode,

    /// Target convergence gap (optional).
    /// When set, enables progress visualization toward this target.
    pub target_gap: Option<f64>,

    /// Whether terminal is interactive (computed at runtime).
    /// Set by terminal detection, not user configuration.
    pub is_interactive: bool,

    /// Whether color output is actually enabled (computed at runtime).
    /// Combines `color` mode with terminal detection.
    pub color_enabled: bool,
}

impl DisplayConfig {
    /// Apply terminal capabilities to determine runtime display settings.
    pub fn apply_terminal_caps(
        &mut self,
        caps: &super::terminal::TerminalCapabilities,
    ) {
        self.is_interactive = caps.is_interactive;

        // Color enabled if: user wants color AND terminal supports it
        self.color_enabled = match self.color {
            ColorMode::Auto => caps.supports_color,
            ColorMode::Always => true,
            ColorMode::Never => false,
        };
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            profile: DisplayProfile::default(),
            color: ColorMode::default(),
            target_gap: None,
            is_interactive: true,
            color_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_from_str() {
        assert_eq!(
            "advanced".parse::<DisplayProfile>().unwrap(),
            DisplayProfile::Advanced
        );
        assert_eq!(
            "MINIMAL".parse::<DisplayProfile>().unwrap(),
            DisplayProfile::Minimal
        );
        assert_eq!(
            "json".parse::<DisplayProfile>().unwrap(),
            DisplayProfile::Automation
        );
        assert!("invalid".parse::<DisplayProfile>().is_err());
    }

    #[test]
    fn test_profile_serde() {
        let json = r#""advanced""#;
        let profile: DisplayProfile = serde_json::from_str(json).unwrap();
        assert_eq!(profile, DisplayProfile::Advanced);

        let serialized =
            serde_json::to_string(&DisplayProfile::Automation).unwrap();
        assert_eq!(serialized, r#""automation""#);
    }

    #[test]
    fn test_color_mode_from_str() {
        assert_eq!("auto".parse::<ColorMode>().unwrap(), ColorMode::Auto);
        assert_eq!("ALWAYS".parse::<ColorMode>().unwrap(), ColorMode::Always);
        assert_eq!("never".parse::<ColorMode>().unwrap(), ColorMode::Never);
        assert!("invalid".parse::<ColorMode>().is_err());
    }

    #[test]
    fn test_color_mode_serde() {
        let json = r#""always""#;
        let mode: ColorMode = serde_json::from_str(json).unwrap();
        assert_eq!(mode, ColorMode::Always);
    }

    #[test]
    fn test_config_default() {
        let config = DisplayConfig::default();
        assert_eq!(config.profile, DisplayProfile::Advanced);
        assert_eq!(config.color, ColorMode::Auto);
        assert!(config.target_gap.is_none());
    }

    #[test]
    fn test_display_config_input_defaults() {
        let json = r#"{}"#;
        let config: DisplayConfigInput = serde_json::from_str(json).unwrap();
        assert_eq!(config.profile, DisplayProfile::Advanced);
        assert_eq!(config.color, ColorMode::Auto);
        assert!(config.target_gap.is_none());
    }

    #[test]
    fn test_display_config_input_full() {
        let json =
            r#"{"profile": "minimal", "color": "never", "target_gap": 5.0}"#;
        let config: DisplayConfigInput = serde_json::from_str(json).unwrap();
        assert_eq!(config.profile, DisplayProfile::Minimal);
        assert_eq!(config.color, ColorMode::Never);
        assert_eq!(config.target_gap, Some(5.0));
    }

    #[test]
    fn test_display_config_input_null_target() {
        let json = r#"{"target_gap": null}"#;
        let config: DisplayConfigInput = serde_json::from_str(json).unwrap();
        assert!(config.target_gap.is_none());
    }

    #[test]
    fn test_display_config_from_input() {
        let input = DisplayConfigInput {
            profile: DisplayProfile::Automation,
            color: ColorMode::Never,
            target_gap: Some(2.5),
        };
        let config = DisplayConfig::from(input);
        assert_eq!(config.profile, DisplayProfile::Automation);
        assert_eq!(config.color, ColorMode::Never);
        assert_eq!(config.target_gap, Some(2.5));
    }
}
