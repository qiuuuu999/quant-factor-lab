"""tearsheet — figures and tables for a backtest performance review.

Renders the three standard artifacts from a NAV series and its
:class:`~quantlab.backtest.metrics.PerformanceMetrics`:

* :func:`plot_nav_comparison` — strategy vs. benchmark equity curves on a log
  y-axis (so a decade of compounding is legible),
* :func:`plot_drawdown` — the underwater (drawdown-from-peak) curve,
* :func:`metrics_table_png` / :func:`metrics_markdown` — the summary metrics
  table as a PNG and as Markdown.

Everything uses matplotlib's non-interactive ``Agg`` backend so it runs
head-less (CI, scripts) and only ever writes files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402

from quantlab.backtest.metrics import PerformanceMetrics, drawdown_series  # noqa: E402

__all__ = [
    "plot_nav_comparison",
    "plot_drawdown",
    "metrics_table_png",
    "metrics_markdown",
]

_STRAT_COLOR = "#1f77b4"
_BENCH_COLOR = "#888888"
_DD_COLOR = "#d62728"


def plot_nav_comparison(
    strategy: pd.Series,
    benchmark: pd.Series | None,
    path: str | Path,
    *,
    title: str = "Equity Curve",
    strategy_label: str = "Strategy",
    benchmark_label: str = "Benchmark",
    log_scale: bool = True,
) -> Path:
    """Plot strategy vs. benchmark NAV; log y-axis by default."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(strategy.index, strategy.values, color=_STRAT_COLOR,
            lw=1.6, label=strategy_label)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, color=_BENCH_COLOR,
                lw=1.4, ls="--", label=benchmark_label)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("NAV (log scale)")
    else:
        ax.set_ylabel("NAV")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def plot_drawdown(
    nav: pd.Series,
    path: str | Path,
    *,
    title: str = "Drawdown",
    label: str = "Strategy",
) -> Path:
    """Plot the underwater curve (drawdown from running peak)."""
    dd = drawdown_series(nav) * 100.0  # percent
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.values, 0.0, color=_DD_COLOR, alpha=0.35)
    ax.plot(dd.index, dd.values, color=_DD_COLOR, lw=1.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def metrics_table_png(
    metrics: PerformanceMetrics,
    path: str | Path,
    *,
    title: str = "Performance Summary",
) -> Path:
    """Render the metrics summary as a standalone table image."""
    s = metrics.to_series()
    fig, ax = plt.subplots(figsize=(6, 0.42 * len(s) + 1.1))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    table = ax.table(
        cellText=[[k, v] for k, v in s.items()],
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    # Header styling.
    for col in (0, 1):
        cell = table[0, col]
        cell.set_facecolor("#1f77b4")
        cell.set_text_props(color="white", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def metrics_markdown(
    metrics: PerformanceMetrics,
    path: str | Path,
    *,
    title: str = "Performance Summary",
    preamble: str | None = None,
) -> Path:
    """Write the metrics summary as a Markdown document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}", ""]
    if preamble:
        parts += [preamble, ""]
    parts.append(metrics.to_markdown())
    parts.append("")
    path.write_text("\n".join(parts))
    return path


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
