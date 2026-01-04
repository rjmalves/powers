"""Report generation modules."""

from .dashboard import DashboardGenerator, generate_dashboard
from .markdown import generate_markdown_report

__all__ = [
    "DashboardGenerator",
    "generate_dashboard",
    "generate_markdown_report",
]
