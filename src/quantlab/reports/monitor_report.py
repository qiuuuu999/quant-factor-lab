"""monitor_report — figures for a factor health / regime-fit review.

Renders the two standard artifacts from :mod:`quantlab.monitor`:

* :func:`plot_factor_health` — the per-period IC with its rolling mean
  (:class:`~quantlab.monitor.decay.FactorHealthReport`), the CUSUM break date
  marked if a decay alert fired.
* :func:`plot_regime_heatmap` — the factor-x-regime mean-IC table
  (:class:`~quantlab.monitor.regime.RegimeICMatrix`) as a heatmap.

Like :mod:`quantlab.reports.tearsheet` this uses matplotlib's non-interactive
``Agg`` backend so it runs head-less and only ever writes files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

from quantlab.monitor.decay import FactorHealthReport  # noqa: E402
from quantlab.monitor.regime import RegimeICMatrix  # noqa: E402

__all__ = ["plot_factor_health", "plot_regime_heatmap"]

_POS_COLOR = "#1f77b4"
_NEG_COLOR = "#d62728"
_ROLL_COLOR = "#111111"
_ALERT_COLOR = "#d62728"


def plot_factor_health(
    report: FactorHealthReport,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Per-period IC bars with the rolling mean overlaid, CUSUM break marked."""
    ic = report.ic.dropna()
    colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in ic.values]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(ic.index, ic.values, width=20, color=colors, alpha=0.5)

    roll = report.rolling.dropna()
    if len(roll):
        ax.plot(roll.index, roll.values, color=_ROLL_COLOR, lw=1.8,
                label="rolling mean IC")
    ax.axhline(0.0, color="#999999", lw=0.8)

    if report.decay_alert and report.alert_date is not None:
        ax.axvline(report.alert_date, color=_ALERT_COLOR, lw=1.5, ls="--",
                    label=f"CUSUM decay alert ({report.alert_date.date()})")

    ax.set_title(title or f"{report.name} — Factor Health "
                 f"(current {report.current_rolling_ic:+.4f} vs. "
                 f"historical {report.historical_mean_ic:+.4f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rank IC")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def plot_regime_heatmap(
    matrix: RegimeICMatrix,
    path: str | Path,
    *,
    title: str = "Factor – Regime Fit (mean IC)",
) -> Path:
    """Heatmap of mean IC per factor (rows) x regime (columns)."""
    data = matrix.mean_ic.to_numpy(dtype=float)
    rows, cols = list(matrix.mean_ic.index), list(matrix.mean_ic.columns)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3.0, 0.7 * len(rows) + 2.0))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = data[i, j]
            if np.isnan(v):
                continue
            n = matrix.counts.iloc[i, j]
            ax.text(j, i, f"{v:+.3f}\n(n={n})", ha="center", va="center",
                    color="white" if abs(v) > vmax * 0.5 else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean rank IC")
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
