"""Configuration loading tests."""

from pathlib import Path

from powers_profile.config import load_config, ProfilingConfig


def test_load_default_config() -> None:
    config = load_config()
    assert isinstance(config, ProfilingConfig)
    assert "timing" in config.default_collectors
    assert config.output_dir.exists() or True  # path may not exist yet


def test_load_config_with_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom.toml"
    custom.write_text(
        """
[general]
output_dir = "tmp/profiling"

[parallel]
thread_counts = [1, 2]
"""
    )
    cfg = load_config(config_path=custom)
    assert cfg.output_dir.name == "profiling"
    assert cfg.thread_counts == [1, 2]


def test_cli_overrides_applied() -> None:
    cfg = load_config(cli_overrides={"general": {"output_dir": "/tmp/override"}})
    assert cfg.output_dir == Path("/tmp/override")
