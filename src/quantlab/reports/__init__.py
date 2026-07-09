"""reports — automated reporting.

Generates reproducible research and performance tearsheets — factor summaries,
backtest results, risk decompositions, and monitoring dashboards — as figures
and documents. Turns pipeline output into artifacts suitable for review and
distribution.
"""

from quantlab.reports.tearsheet import (
    metrics_comparison_markdown,
    metrics_comparison_table_png,
    metrics_markdown,
    metrics_table_png,
    plot_drawdown,
    plot_multi_nav,
    plot_nav_comparison,
)
from quantlab.reports.factor_report import (
    plot_decile_returns,
    plot_factor_correlation_heatmap,
    plot_ic_timeseries,
)
from quantlab.reports.risk_report import (
    plot_exposure_profile,
    plot_risk_decomposition,
)
from quantlab.reports.monitor_report import (
    plot_factor_health,
    plot_regime_heatmap,
)

__all__ = [
    "plot_nav_comparison",
    "plot_drawdown",
    "metrics_table_png",
    "metrics_markdown",
    "plot_multi_nav",
    "metrics_comparison_table_png",
    "metrics_comparison_markdown",
    "plot_ic_timeseries",
    "plot_decile_returns",
    "plot_factor_correlation_heatmap",
    "plot_exposure_profile",
    "plot_risk_decomposition",
    "plot_factor_health",
    "plot_regime_heatmap",
]
