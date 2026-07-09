"""tearsheet — figures and tables for a backtest performance review.

Renders the three standard artifacts from a NAV series and its
:class:`~quantlab.backtest.metrics.PerformanceMetrics`:

* :func:`plot_nav_comparison` — strategy vs. benchmark equity curves on a log
  y-axis (so a decade of compounding is legible),
* :func:`plot_drawdown` — the underwater (drawdown-from-peak) curve,
* :func:`metrics_table_png` / :func:`metrics_markdown` — the summary metrics
  table as a PNG and as Markdown.

Plus the N-way analogues for comparing several strategies built on the same
underlying signal (e.g. different portfolio-construction methods) side by
side: :func:`plot_multi_nav`, :func:`metrics_comparison_table_png`,
:func:`metrics_comparison_markdown`.

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
    "plot_multi_nav",
    "metrics_comparison_table_png",
    "metrics_comparison_markdown",
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


def plot_multi_nav(
    navs: dict[str, pd.Series],
    path: str | Path,
    *,
    title: str = "Equity Curve Comparison",
    benchmark: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
    log_scale: bool = True,
) -> Path:
    """Plot several strategy NAV series on one chart (log y-axis by default).

    Colors cycle through matplotlib's default palette, so this scales
    cleanly to a handful of strategies without manual color assignment.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, nav in navs.items():
        ax.plot(nav.index, nav.values, lw=1.6, label=label)
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


def _comparison_rows(metrics: dict[str, PerformanceMetrics]) -> tuple[list[str], dict[str, pd.Series]]:
    """Metric rows common to every strategy, in the first strategy's order.

    Strategies without a benchmark leave ``has_benchmark`` rows out of their
    ``to_series()``; intersecting keeps the comparison table well-formed
    (every row present for every column) even when only some strategies were
    scored against a benchmark.
    """
    series = {name: m.to_series() for name, m in metrics.items()}
    base_order = list(next(iter(series.values())).index)
    common = set.intersection(*(set(s.index) for s in series.values()))
    rows = [m for m in base_order if m in common]
    return rows, series


def metrics_comparison_table_png(
    metrics: dict[str, PerformanceMetrics],
    path: str | Path,
    *,
    title: str = "Strategy Comparison",
) -> Path:
    """Render several strategies' metrics side by side as one table image."""
    rows, series = _comparison_rows(metrics)
    names = list(series.keys())
    fig, ax = plt.subplots(figsize=(2.2 * len(names) + 3.0, 0.42 * len(rows) + 1.1))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    col_labels = ["Metric"] + names
    cell_text = [[m] + [series[name][m] for name in names] for m in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    for col in range(len(col_labels)):
        cell = table[0, col]
        cell.set_facecolor("#1f77b4")
        cell.set_text_props(color="white", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def metrics_comparison_markdown(
    metrics: dict[str, PerformanceMetrics],
    path: str | Path,
    *,
    title: str = "Strategy Comparison",
    preamble: str | None = None,
) -> Path:
    """Write several strategies' metrics side by side as one Markdown table."""
    rows, series = _comparison_rows(metrics)
    names = list(series.keys())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}", ""]
    if preamble:
        parts += [preamble, ""]
    parts.append("| Metric | " + " | ".join(names) + " |")
    parts.append("| --- | " + " | ".join("---" for _ in names) + " |")
    for m in rows:
        parts.append("| " + m + " | " + " | ".join(series[name][m] for name in names) + " |")
    parts.append("")
    path.write_text("\n".join(parts))
    return path


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
