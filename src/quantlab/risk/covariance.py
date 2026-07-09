"""covariance — return-covariance estimators behind one unified interface.

Every estimator here answers the same question — "what is the covariance
matrix of these asset (or factor) returns?" — with different bias/variance
trade-offs:

* :class:`SampleCovariance` — the textbook estimator. Unbiased, but noisy: with
  ``N`` assets and ``T`` periods it has ``N(N+1)/2`` free parameters estimated
  from ``N*T`` numbers, and is **singular** whenever ``N > T`` (more assets than
  observations) — a routine situation for an equity risk model with hundreds of
  names and a few years of monthly history.
* :class:`EWMACovariance` — an exponentially-weighted sample covariance
  (RiskMetrics-style), trading a little bias for responsiveness to recent
  regime shifts.
* :class:`LedoitWolfCovariance` — shrinks the sample covariance toward a
  well-conditioned target (a scaled identity matrix) with an analytically
  optimal shrinkage intensity (Ledoit & Wolf, 2004). This is always invertible
  and has lower expected estimation error than the raw sample covariance,
  which is exactly the fix needed when ``N`` approaches or exceeds ``T``.

All three implement :class:`CovarianceEstimator` — a single ``estimate(returns)
-> DataFrame`` method — so callers (e.g. :mod:`quantlab.risk.factor_risk`) can
swap estimators without changing call sites. :func:`estimate_covariance` is a
convenience dispatcher keyed by method name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

__all__ = [
    "CovarianceEstimator",
    "SampleCovariance",
    "EWMACovariance",
    "LedoitWolfCovariance",
    "estimate_covariance",
]


def _clean(returns: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    r = returns.dropna(how="any")
    if len(r) < min_periods:
        raise ValueError(
            f"need >= {min_periods} fully-populated periods, got {len(r)}"
        )
    return r


class CovarianceEstimator(ABC):
    """Common interface: turn a ``period x asset`` return frame into a covariance."""

    @abstractmethod
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate the ``asset x asset`` covariance matrix of ``returns``.

        ``returns`` is a period x asset frame of periodic (e.g. monthly) simple
        or log returns. Rows with any ``NaN`` are dropped before estimation.
        """


class SampleCovariance(CovarianceEstimator):
    """The ordinary sample covariance matrix.

    Parameters
    ----------
    ddof:
        Delta degrees of freedom for the covariance normalisation (default
        ``1``, i.e. divide by ``T - 1``).
    min_periods:
        Minimum number of fully-populated periods required.
    """

    def __init__(self, ddof: int = 1, min_periods: int = 2):
        self.ddof = ddof
        self.min_periods = min_periods

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        r = _clean(returns, self.min_periods)
        return r.cov(ddof=self.ddof)


class EWMACovariance(CovarianceEstimator):
    """Exponentially-weighted covariance (RiskMetrics-style).

    The most recent observation gets weight ``1``, each period further back is
    discounted by ``decay = 0.5 ** (1 / halflife)``, and weights are normalised
    to sum to one. The (weighted) mean is subtracted before the outer product so
    this remains a proper covariance (not a raw second-moment) estimate, and the
    result is bias-corrected by ``1 - sum(w**2)`` (the weighted analogue of
    dividing by ``T - 1``).

    Parameters
    ----------
    halflife:
        Number of periods for a weight to decay to one half (e.g. ``36`` months
        gives roughly three years of effective memory). Larger halflife ->
        closer to the plain sample covariance.
    min_periods:
        Minimum number of fully-populated periods required.
    """

    def __init__(self, halflife: float = 36.0, min_periods: int = 2):
        if halflife <= 0:
            raise ValueError("halflife must be > 0")
        self.halflife = halflife
        self.min_periods = min_periods

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        r = _clean(returns, self.min_periods)
        n = len(r)
        decay = 0.5 ** (1.0 / self.halflife)
        # Oldest observation (row 0) gets decay**(n-1); most recent gets decay**0 = 1.
        weights = decay ** np.arange(n - 1, -1, -1)
        weights = weights / weights.sum()

        x = r.to_numpy(dtype=float)
        mean = weights @ x
        centered = x - mean
        cov = (centered * weights[:, None]).T @ centered
        denom = 1.0 - float(np.sum(weights ** 2))
        cov = cov / denom if denom > 0 else cov
        return pd.DataFrame(cov, index=r.columns, columns=r.columns)


class LedoitWolfCovariance(CovarianceEstimator):
    """Ledoit-Wolf shrinkage toward a scaled-identity target.

    Reference: Ledoit, O., & Wolf, M. (2004), "Honey, I Shrunk the Sample
    Covariance Matrix", *Journal of Portfolio Management* 30(4), 110-119.

    The shrunk estimate is a convex combination of the sample covariance ``S``
    and a well-conditioned target ``F = mu * I`` (``mu`` = average sample
    variance): ``S_hat = shrinkage * F + (1 - shrinkage) * S``. The shrinkage
    intensity is chosen analytically to minimise expected quadratic loss
    ``E||S_hat - Sigma||_F^2`` and is estimated from the data itself (no
    cross-validation, no tuning parameter). ``shrinkage`` is clipped to
    ``[0, 1]``; the fitted value is stored on ``last_shrinkage_`` after each
    call to :meth:`estimate`.

    Because the target is full-rank by construction, ``S_hat`` is invertible
    even when the number of assets ``N`` exceeds the number of periods ``T``
    (where the raw sample covariance is singular).

    Parameters
    ----------
    min_periods:
        Minimum number of fully-populated periods required.
    """

    def __init__(self, min_periods: int = 2):
        self.min_periods = min_periods
        self.last_shrinkage_: float | None = None

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        r = _clean(returns, self.min_periods)
        t, n = r.shape
        x = r.to_numpy(dtype=float)
        x = x - x.mean(axis=0)

        # Biased (1/T-normalised) sample covariance, per Ledoit & Wolf's own
        # convention -- the shrinkage formula below is derived against it.
        sample = (x.T @ x) / t
        mu = float(np.trace(sample)) / n
        target = mu * np.eye(n)

        if n == 1:
            self.last_shrinkage_ = 0.0
            shrunk = sample
        else:
            # pi_hat: sum over (i, j) of the sample variance of T * s_ij (how
            # noisy each entry of `sample` is), estimated from the data.
            x2 = x ** 2
            phi_mat = (x2.T @ x2) / t - sample ** 2
            phi = float(phi_mat.sum())

            # gamma_hat: squared Frobenius distance from sample to target --
            # how far off the (well-conditioned but biased) target already is.
            gamma = float(np.linalg.norm(sample - target, ord="fro") ** 2)

            shrinkage = 0.0 if gamma == 0.0 else phi / gamma / t
            shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
            self.last_shrinkage_ = shrinkage
            shrunk = shrinkage * target + (1.0 - shrinkage) * sample

        return pd.DataFrame(shrunk, index=r.columns, columns=r.columns)


_ESTIMATORS = {
    "sample": SampleCovariance,
    "ewma": EWMACovariance,
    "ledoit_wolf": LedoitWolfCovariance,
}


def estimate_covariance(
    returns: pd.DataFrame, method: str = "sample", **kwargs
) -> pd.DataFrame:
    """Dispatch to a :class:`CovarianceEstimator` by name.

    ``method`` is one of ``"sample"``, ``"ewma"``, ``"ledoit_wolf"``. Extra
    ``kwargs`` (e.g. ``halflife=``) are forwarded to the estimator's
    constructor.
    """
    if method not in _ESTIMATORS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_ESTIMATORS)}")
    estimator = _ESTIMATORS[method](**kwargs)
    return estimator.estimate(returns)
