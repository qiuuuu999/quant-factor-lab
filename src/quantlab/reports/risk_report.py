"""risk_report — figures for a factor risk-model review.

Renders the two standard artifacts for a risk attribution review:

* :func:`plot_exposure_profile` — the portfolio's factor-exposure profile
  (:class:`~quantlab.risk.attribution.ExposureProfile`) as a horizontal bar
  chart, in cross-sectional standard deviations.
* :func:`plot_risk_decomposition` — the factor-vs-specific variance split
  (:class:`~quantlab.risk.factor_risk.FactorRiskDecomposition`), with the
  factor share broken out per factor.

Like :mod:`quantlab.reports.tearsheet` this uses matplotlib's non-interactive
``Agg`` backend so it runs head-less and only ever writes files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from quantlab.risk.attribution import ExposureProfile  # noqa: E402
from quantlab.risk.factor_risk import FactorRiskDecomposition  # noqa: E402

__all__ = ["plot_exposure_profile", "plot_risk_decomposition"]

_POS_COLOR = "#1f77b4"
_NEG_COLOR = "#d62728"
_FACTOR_COLOR = "#1f77b4"
_SPECIFIC_COLOR = "#888888"


def plot_exposure_profile(
    profile: ExposureProfile,
    path: str | Path,
    *,
    title: str = "Portfolio Factor Exposure",
) -> Path:
    """Horizontal bar chart of a portfolio's factor-exposure profile."""
    exp = profile.exposure
    colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in exp.values]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(exp) + 1.5))
    ax.barh(list(exp.index), exp.values, color=colors, alpha=0.85)
    ax.axvline(0.0, color="#999999", lw=0.8)
    ax.set_title(f"{title}\n{profile.as_of.date()} ({profile.n_names} names)")
    ax.set_xlabel("Portfolio exposure (cross-sectional std. dev.)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_risk_decomposition(
    decomposition: FactorRiskDecomposition,
    path: str | Path,
    *,
    title: str = "Risk Decomposition",
) -> Path:
    """Bar chart of variance contribution: specific + one bar per factor."""
    labels = ["Specific"] + list(decomposition.factor_contributions.index)
    values = [decomposition.specific_variance] + list(
        decomposition.factor_contributions.values
    )
    colors = [_SPECIFIC_COLOR] + [_FACTOR_COLOR] * len(decomposition.factor_contributions)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.set_title(
        f"{title}\nFactor {decomposition.factor_variance_pct:.0%} / "
        f"Specific {decomposition.specific_variance_pct:.0%} of total variance"
    )
    ax.set_ylabel("Variance contribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
