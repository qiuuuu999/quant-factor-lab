"""preprocess — cross-sectional factor cleaning utilities.

Standard steps applied to a raw factor Series before it is used in ranking,
risk models, or portfolio construction:

* :func:`winsorize` — clip extreme values to given quantiles (default 1%/99%).
* :func:`zscore` — standardize to zero mean / unit standard deviation.
* :func:`deciles` — assign each name to one of ``n`` equal-count buckets.

All functions ignore ``NaN`` (undefined factor values) and preserve the input
index, so they compose cleanly on the Series returned by ``Factor.compute``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["winsorize", "zscore", "deciles"]


def winsorize(
    s: pd.Series, lower: float = 0.01, upper: float = 0.99
) -> pd.Series:
    """Clip values to the ``[lower, upper]`` quantile range.

    Quantiles are computed on non-NaN values; NaNs are left untouched.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("require 0 <= lower < upper <= 1")
    valid = s.dropna()
    if valid.empty:
        return s.copy()
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series, ddof: int = 1) -> pd.Series:
    """Standardize to mean 0, standard deviation 1 (NaNs preserved)."""
    valid = s.dropna()
    std = valid.std(ddof=ddof)
    if valid.empty or std == 0 or np.isnan(std):
        # No dispersion (or nothing to standardize): center only.
        return s - valid.mean() if not valid.empty else s.copy()
    return (s - valid.mean()) / std


def deciles(s: pd.Series, n: int = 10) -> pd.Series:
    """Assign each non-NaN value to a bucket in ``1..n`` (n = highest values).

    Uses first-tie-broken ranks so buckets are of (near-)equal count even with
    duplicate factor values. NaNs stay NaN. Bucket 1 holds the lowest factor
    values, bucket ``n`` the highest.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    valid = s.dropna()
    out = pd.Series(np.nan, index=s.index, name=s.name)
    if valid.empty:
        return out
    if len(valid) < n:
        # Too few names to fill n buckets; rank into as many as we can.
        n = len(valid)
    ranks = valid.rank(method="first")
    buckets = pd.qcut(ranks, n, labels=range(1, n + 1))
    out.loc[valid.index] = buckets.astype(float)
    return out
