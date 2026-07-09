"""reports — automated reporting.

Generates reproducible research and performance tearsheets — factor summaries,
backtest results, risk decompositions, and monitoring dashboards — as figures
and documents. Turns pipeline output into artifacts suitable for review and
distribution.
"""

from quantlab.reports.tearsheet import (
    metrics_markdown,
    metrics_table_png,
    plot_drawdown,
    plot_nav_comparison,
)

__all__ = [
    "plot_nav_comparison",
    "plot_drawdown",
    "metrics_table_png",
    "metrics_markdown",
]
