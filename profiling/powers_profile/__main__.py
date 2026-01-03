"""Entry point for `python -m powers_profile`."""

from powers_profile.cli import app


def main() -> None:
    """Dispatch to the Typer application."""
    app()


if __name__ == "__main__":
    main()
