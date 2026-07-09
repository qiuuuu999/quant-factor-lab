"""attribution — portfolio factor-exposure profiling.

Answers one question: given a portfolio's holdings and a snapshot of
cross-sectionally standardised factor exposures, how many standard deviations
of momentum / low-vol / reversal / liquidity (etc.) does the *portfolio* carry,
net of its individual names?

The exposure of a portfolio to factor ``k`` is simply its holdings-weighted
average of each name's exposure to that factor:

    exposure_k = sum_i  w_i * exposure_{i,k}

When ``exposure`` is a cross-sectional z-score (mean 0, std 1 across names on
that date — the convention used throughout this platform, see
``quantlab.factors.preprocess.zscore``), the result reads directly in units of
"standard deviations of the universe": an equal-weight top-decile-momentum
book will show a large positive momentum exposure and, because winsorized
z-scores keep everything else roughly centered, only modest exposure to the
other factors -- unless the selection is correlated with them too.

This module deliberately has no dependency on how the exposures were computed
(:mod:`quantlab.risk.factor_risk` builds them from the factor library) or on
the risk model (:mod:`quantlab.risk.factor_risk` consumes this module's
:func:`portfolio_factor_exposure` for its own decomposition), so it stays a
plain, dependency-free leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

__all__ = ["ExposureProfile", "portfolio_factor_exposure"]


def portfolio_factor_exposure(
    weights: pd.Series, exposures: pd.DataFrame
) -> pd.Series:
    """Holdings-weighted average factor exposure of a portfolio.

    Parameters
    ----------
    weights:
        Portfolio weights indexed by ticker (need not sum to 1; short weights
        are fine). Tickers absent from ``exposures`` contribute nothing.
    exposures:
        A ``ticker x factor`` frame of (typically winsorized, z-scored)
        cross-sectional factor values for the same formation date. Tickers
        held but missing an exposure value are treated as ``0`` (neutral)
        rather than dropped, so a name with too little history to score a
        given factor does not silently exclude itself from the portfolio's
        weight base.

    Returns
    -------
    A Series indexed by factor name, in the same units as ``exposures``
    (standard deviations, if ``exposures`` was z-scored).
    """
    w = weights.reindex(exposures.index).fillna(0.0)
    exp = exposures.fillna(0.0)
    return exp.T.dot(w).rename("exposure")


@dataclass
class ExposureProfile:
    """A portfolio's factor-exposure profile at one point in time."""

    as_of: pd.Timestamp
    exposure: pd.Series          # factor name -> weighted-average exposure
    weights: pd.Series           # the holdings this profile was built from
    n_names: int                 # number of names with non-zero weight

    @classmethod
    def build(
        cls,
        weights: pd.Series,
        exposures: pd.DataFrame,
        as_of: str | date | datetime,
    ) -> "ExposureProfile":
        """Build the profile for ``weights`` against a single date's ``exposures``."""
        exposure = portfolio_factor_exposure(weights, exposures)
        held = weights[weights.fillna(0.0) != 0.0]
        return cls(
            as_of=pd.Timestamp(as_of),
            exposure=exposure,
            weights=weights,
            n_names=int(len(held)),
        )

    def summary(self) -> str:
        lines = [
            f"Factor exposure profile as of {self.as_of.date()} "
            f"({self.n_names} names held):"
        ]
        for name, value in self.exposure.items():
            lines.append(f"  {name:>18}: {value:+.2f}σ")
        return "\n".join(lines)
