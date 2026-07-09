"""factor_risk — style-factor risk decomposition (Barra-style, price-factor version).

Turns the four price factors already in :mod:`quantlab.factors` (momentum,
low volatility, reversal, liquidity) into exposures for every name in the
universe, estimates each factor's **realised return** via a per-period
cross-sectional regression, and uses the resulting factor-return time series to
split a portfolio's total variance into a **factor** component (systematic,
driven by the shared style tilts) and a **specific** component (idiosyncratic,
diversifiable across names).

Pipeline
--------
1. :func:`build_exposure_panel` computes, for every rebalance date, a
   ``ticker x factor`` frame of winsorized, z-scored factor exposures (reusing
   :func:`quantlab.factors.evaluation.build_factor_panel` for the point-in-time
   factor values and forward returns).
2. :func:`cross_sectional_regression` regresses that period's forward returns
   on the exposures (with an intercept absorbing the equal-weighted average
   return): the slope coefficients are that period's **factor returns**, the
   residuals are that period's **specific returns**.
3. :class:`FactorRiskModel` runs step 2 across every date, then estimates the
   factor covariance from the factor-return time series (any
   :class:`~quantlab.risk.covariance.CovarianceEstimator`) and the specific
   variance from the time series of residuals (per ticker).
4. :func:`decompose_portfolio_risk` combines a snapshot of portfolio weights
   and exposures with the fitted factor covariance and specific variance into
   a :class:`FactorRiskDecomposition`: total variance = factor variance +
   specific variance, with an exact (Euler) per-factor contribution breakdown.

This is a simplified, price-factor-only cousin of a commercial (Barra/Axioma)
fundamental risk model: no industry dummies, no market-cap weighting of the
cross-sectional regression, four style factors instead of dozens. The
mechanics -- cross-sectional regression -> factor covariance + specific
variance -> quadratic-form decomposition -- are the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from quantlab.factors.base import Factor
from quantlab.factors.evaluation import build_factor_panel
from quantlab.factors.preprocess import winsorize, zscore
from quantlab.risk.attribution import portfolio_factor_exposure
from quantlab.risk.covariance import CovarianceEstimator, SampleCovariance

__all__ = [
    "build_exposure_panel",
    "cross_sectional_regression",
    "FactorRiskModel",
    "FactorRiskDecomposition",
    "decompose_portfolio_risk",
]

MembersFn = Callable[[pd.Timestamp], Iterable[str]]


# --------------------------------------------------------------------------- #
# Exposure panel construction
# --------------------------------------------------------------------------- #

def build_exposure_panel(
    factors: Sequence[Factor],
    long_prices: pd.DataFrame,
    rebal_dates: Sequence[pd.Timestamp],
    members_by_date: MembersFn,
    *,
    history_months: int = 16,
    price_col: str = "adj_close",
    winsorize_limits: tuple[float, float] = (0.01, 0.99),
) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Build per-date factor exposures and the shared forward-return panel.

    Each factor's raw point-in-time values (via
    :func:`~quantlab.factors.evaluation.build_factor_panel`, which enforces the
    no-look-ahead contract) are winsorized then z-scored *cross-sectionally on
    each date* -- the standard preparation for regression/risk-model inputs
    (see ``docs/factors.md``).

    Returns
    -------
    exposures_by_date:
        ``{date: DataFrame}`` where each frame is ``ticker x factor``.
    forward_returns:
        The ``date x ticker`` forward-return panel (formation close to next
        formation close). Identical regardless of which factor produced it --
        it is a pure function of ``long_prices`` and ``rebal_dates`` -- so it
        is computed once and shared.
    """
    if not factors:
        raise ValueError("need at least one factor")

    panels = [
        build_factor_panel(
            f, long_prices, rebal_dates, members_by_date,
            history_months=history_months, price_col=price_col,
        )
        for f in factors
    ]
    forward_returns = panels[0].forward_returns

    exposures_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for dt in panels[0].dates:
        cols = {
            p.name: zscore(winsorize(p.factor_values.loc[dt], *winsorize_limits))
            for p in panels
        }
        exposures_by_date[dt] = pd.DataFrame(cols)
    return exposures_by_date, forward_returns


# --------------------------------------------------------------------------- #
# Cross-sectional regression
# --------------------------------------------------------------------------- #

def cross_sectional_regression(
    forward_returns: pd.Series,
    exposures: pd.DataFrame,
    *,
    add_intercept: bool = True,
    min_names: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """OLS of one period's forward returns on that period's factor exposures.

    The slope coefficients are the period's estimated **factor returns**; the
    residuals are the period's **specific returns**. An intercept is included
    by default to absorb the equal-weighted average return (the exposures are
    z-scored, so without an intercept that level effect would leak into the
    factor coefficients); it is dropped from the returned factor-return
    Series.

    Parameters
    ----------
    forward_returns:
        Realised return for the period, indexed by ticker.
    exposures:
        ``ticker x factor`` exposures for the same period.
    add_intercept:
        Include an intercept column in the regression (default ``True``).
    min_names:
        Minimum number of names with both a return and full exposures required
        to run the regression; below this, factor returns and residuals are
        all ``NaN``. Defaults to ``exposures.shape[1] + (2 if add_intercept
        else 1)`` -- at least one surplus degree of freedom.

    Returns
    -------
    ``(factor_returns, residuals)`` -- a Series indexed by factor name and a
    Series indexed by ``exposures.index`` (``NaN`` for names that were dropped
    or unregressed).
    """
    factor_names = list(exposures.columns)
    k = len(factor_names)
    if min_names is None:
        min_names = k + (2 if add_intercept else 1)

    frame = exposures.copy()
    frame["__ret__"] = forward_returns.reindex(exposures.index)
    frame = frame.dropna()

    empty_factor_returns = pd.Series(np.nan, index=factor_names)
    empty_residuals = pd.Series(np.nan, index=exposures.index)
    if len(frame) < min_names:
        return empty_factor_returns, empty_residuals

    y = frame.pop("__ret__").to_numpy(dtype=float)
    names = list(frame.columns)
    x = frame.to_numpy(dtype=float)
    if add_intercept:
        x = np.column_stack([np.ones(len(x)), x])
        names = ["intercept"] + names

    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    resid = pd.Series(y - fitted, index=frame.index).reindex(exposures.index)

    coef_s = pd.Series(coef, index=names)
    factor_returns = coef_s.drop("intercept") if add_intercept else coef_s
    return factor_returns.reindex(factor_names), resid


# --------------------------------------------------------------------------- #
# Fitted risk model
# --------------------------------------------------------------------------- #

class FactorRiskModel:
    """Fits per-period factor/specific returns, then risk (covariance/variance).

    Attributes (populated by :meth:`fit`)
    --------------------------------------
    factor_returns_:
        ``date x factor`` DataFrame of estimated factor returns.
    specific_returns_:
        ``date x ticker`` DataFrame of regression residuals.
    """

    def __init__(self) -> None:
        self.factor_returns_: pd.DataFrame = pd.DataFrame()
        self.specific_returns_: pd.DataFrame = pd.DataFrame()

    def fit(
        self,
        exposures_by_date: dict[pd.Timestamp, pd.DataFrame],
        forward_returns: pd.DataFrame,
        *,
        add_intercept: bool = True,
        min_names: int | None = None,
    ) -> "FactorRiskModel":
        """Run :func:`cross_sectional_regression` on every date in ``exposures_by_date``."""
        factor_rows: dict[pd.Timestamp, pd.Series] = {}
        resid_rows: dict[pd.Timestamp, pd.Series] = {}
        for dt, exposures in exposures_by_date.items():
            if dt not in forward_returns.index:
                continue
            fr, resid = cross_sectional_regression(
                forward_returns.loc[dt], exposures,
                add_intercept=add_intercept, min_names=min_names,
            )
            factor_rows[dt] = fr
            resid_rows[dt] = resid

        self.factor_returns_ = pd.DataFrame(factor_rows).T.sort_index()
        self.specific_returns_ = pd.DataFrame(resid_rows).T.sort_index()
        return self

    def factor_covariance(
        self, estimator: CovarianceEstimator | None = None
    ) -> pd.DataFrame:
        """Covariance of the fitted factor-return time series."""
        estimator = estimator or SampleCovariance()
        return estimator.estimate(self.factor_returns_)

    def specific_variance(self, min_periods: int = 2) -> pd.Series:
        """Per-ticker variance of the fitted specific-return time series.

        Tickers with fewer than ``min_periods`` non-NaN residuals return
        ``NaN`` rather than an unreliable estimate.
        """
        counts = self.specific_returns_.count()
        var = self.specific_returns_.var(ddof=1)
        return var.where(counts >= min_periods)

    def decompose(
        self,
        weights: pd.Series,
        exposures: pd.DataFrame,
        *,
        covariance_estimator: CovarianceEstimator | None = None,
        min_periods: int = 2,
    ) -> "FactorRiskDecomposition":
        """Decompose a portfolio's risk using this model's fitted factor risk.

        Convenience wrapper around :func:`decompose_portfolio_risk` using
        :meth:`factor_covariance` and :meth:`specific_variance`.
        """
        factor_cov = self.factor_covariance(covariance_estimator)
        specific_var = self.specific_variance(min_periods)
        return decompose_portfolio_risk(weights, exposures, factor_cov, specific_var)


# --------------------------------------------------------------------------- #
# Portfolio risk decomposition
# --------------------------------------------------------------------------- #

@dataclass
class FactorRiskDecomposition:
    """A portfolio's variance, split into factor and specific components.

    ``factor_contributions`` is an *Euler* (marginal-contribution) split of
    ``factor_variance`` across individual factors: contribution_k = exposure_k
    * (factor_covariance @ exposure)_k, which sums *exactly* to
    ``factor_variance`` (it is the first-order Taylor / homogeneity-of-degree-2
    identity for a quadratic form).
    """

    total_variance: float
    factor_variance: float
    specific_variance: float
    factor_contributions: pd.Series   # factor name -> contribution to factor_variance
    exposure: pd.Series               # portfolio factor exposure (same units as attribution)

    @property
    def factor_variance_pct(self) -> float:
        return self.factor_variance / self.total_variance if self.total_variance else float("nan")

    @property
    def specific_variance_pct(self) -> float:
        return self.specific_variance / self.total_variance if self.total_variance else float("nan")

    def summary(self) -> str:
        lines = [
            f"Total variance:    {self.total_variance:.6f}  "
            f"(annualised vol {np.sqrt(max(self.total_variance, 0.0) * 12):.2%})",
            f"  Factor:          {self.factor_variance:.6f}  "
            f"({self.factor_variance_pct:.1%} of total)",
            f"  Specific:        {self.specific_variance:.6f}  "
            f"({self.specific_variance_pct:.1%} of total)",
            "Factor contributions to factor variance:",
        ]
        for name, value in self.factor_contributions.items():
            share = value / self.factor_variance if self.factor_variance else float("nan")
            lines.append(f"  {name:>18}: {value:+.6f}  ({share:.1%} of factor variance)")
        return "\n".join(lines)


def decompose_portfolio_risk(
    weights: pd.Series,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_variance: pd.Series,
) -> FactorRiskDecomposition:
    """Split a portfolio's variance into factor and specific components.

    ``Var(portfolio) = e' Sigma_f e + w' D w``, where ``e`` is the portfolio's
    factor exposure (:func:`~quantlab.risk.attribution.portfolio_factor_exposure`),
    ``Sigma_f`` is the factor covariance, ``w`` is the (name-level) weight
    vector, and ``D`` is the diagonal of per-name specific variances (assumed
    uncorrelated across names -- the standard factor-model assumption that all
    cross-sectional correlation is captured by the shared factors).

    Names held but missing a specific-variance estimate contribute zero
    specific variance (their risk is carried entirely by their factor
    exposures) rather than raising.
    """
    factor_names = list(factor_covariance.columns)
    exposure = portfolio_factor_exposure(weights, exposures).reindex(factor_names)

    sigma_f = factor_covariance.reindex(index=factor_names, columns=factor_names).to_numpy()
    e = exposure.to_numpy(dtype=float)
    marginal = sigma_f @ e
    factor_contributions = pd.Series(e * marginal, index=factor_names)
    factor_var = float(factor_contributions.sum())

    w = weights.reindex(specific_variance.index).fillna(0.0)
    sv = specific_variance.reindex(w.index).fillna(0.0)
    specific_var = float((w.to_numpy() ** 2 * sv.to_numpy()).sum())

    return FactorRiskDecomposition(
        total_variance=factor_var + specific_var,
        factor_variance=factor_var,
        specific_variance=specific_var,
        factor_contributions=factor_contributions,
        exposure=exposure,
    )
