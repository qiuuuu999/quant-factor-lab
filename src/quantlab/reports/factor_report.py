"""factor_report — figures for a factor-evaluation review.

Renders the standard factor-scoring artifacts from the outputs of
:mod:`quantlab.factors.evaluation`:

* :func:`plot_ic_timeseries` — the per-period IC bar series with its rolling and
  full-sample means,
* :func:`plot_decile_returns` — annualised return by factor quantile (the
  monotonicity check at a glance),
* :func:`plot_factor_correlation_heatmap` — the average cross-sectional rank
  correlation matrix between factors.

Like :mod:`quantlab.reports.tearsheet` this uses matplotlib's non-interactive
``Agg`` backend so it runs head-less and only ever writes files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = [
    "plot_ic_timeseries",
    "plot_decile_returns",
    "plot_factor_correlation_heatmap",
]

_POS_COLOR = "#1f77b4"
_NEG_COLOR = "#d62728"
_MEAN_COLOR = "#111111"
_ROLL_COLOR = "#ff7f0e"


def plot_ic_timeseries(
    ic: pd.Series,
    path: str | Path,
    *,
    title: str = "Information Coefficient",
    roll: int = 12,
) -> Path:
    """Bar plot of the per-period IC with a rolling mean and full-sample mean."""
    ic = ic.dropna()
    colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in ic.values]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(ic.index, ic.values, width=20, color=colors, alpha=0.65)
    if len(ic) >= roll:
        rolling = ic.rolling(roll).mean()
        ax.plot(rolling.index, rolling.values, color=_ROLL_COLOR, lw=1.8,
                label=f"{roll}-period rolling mean")
    mean = float(ic.mean())
    ax.axhline(mean, color=_MEAN_COLOR, lw=1.2, ls="--",
               label=f"mean {mean:+.3f}")
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rank IC")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def plot_decile_returns(
    annualized: pd.Series,
    path: str | Path,
    *,
    title: str = "Annualised Return by Quantile",
) -> Path:
    """Bar chart of annualised return per factor quantile (1 = low … n = high)."""
    ann = annualized.dropna()
    colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in ann.values]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([str(int(b)) for b in ann.index], ann.values * 100.0,
           color=colors, alpha=0.8)
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Factor quantile (1 = lowest, {} = highest)".format(
        int(ann.index.max())))
    ax.set_ylabel("Annualised return (%)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_factor_correlation_heatmap(
    corr: pd.DataFrame,
    path: str | Path,
    *,
    title: str = "Factor Rank-Correlation",
) -> Path:
    """Heatmap of the average cross-sectional rank correlation between factors."""
    labels = list(corr.columns)
    data = corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2.5,
                                    1.4 * len(labels) + 2.0))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=10)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
