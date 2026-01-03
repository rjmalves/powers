"""Ensure powers_profile package imports cleanly."""

from powers_profile import __version__
from powers_profile import analyzers, collectors, config, reporters, schemas, utils  # noqa: F401


def test_version_exposed() -> None:
    assert __version__ == "0.1.0"


def test_submodules_importable() -> None:
    # Modules are imported in module-level import; reaching this point is success.
    assert analyzers is not None
    assert collectors is not None
    assert config is not None
    assert reporters is not None
    assert schemas is not None
    assert utils is not None
