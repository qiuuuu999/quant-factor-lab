"""low_volatility — the low-volatility (idiosyncratic/total volatility) factor.

Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006), "The Cross-Section of
Volatility and Expected Returns", *Journal of Finance* 61(1), 259-299.

Ang et al. document the **low-volatility anomaly**: stocks with high past
volatility earn *lower* subsequent returns than the CAPM predicts, and low-
volatility stocks earn higher risk-adjusted returns. The tradable signal is
therefore the **negative** of trailing return volatility, so that — like every
other factor in this library — a *higher* factor value corresponds to a *higher*
expected return (long the top, short the bottom):

    low_vol(t) = - stdev( r_d ,  d in the last `window` trading days up to t )

where ``r_d`` is the daily total return from **adjusted** closes (so splits and
dividends do not inflate the volatility estimate). We use total return
volatility as the baseline proxy; it is the simplest price-only version of the
Ang et al. signal and is highly correlated with the idiosyncratic-volatility
measure they emphasise.

Stocks with fewer than ``min_periods`` daily returns in the window return
``NaN`` (recent IPOs, delisted names with short history) rather than raising.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantlab.factors.base import DATE_COL, PRICE_COL, Factor

__all__ = ["LowVolatilityFactor"]


class LowVolatilityFactor(Factor):
    """Cross-sectional low-volatility factor (negative trailing return std).

    Parameters
    ----------
    window:
        Trailing look-back in **trading days** (default 252 ≈ one year).
    min_periods:
        Minimum daily returns required in the window; below this the value is
        ``NaN``. Defaults to ``max(2, window // 2)``.
    """

    def __init__(self, window: int = 252, min_periods: int | None = None):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.min_periods = (
            min_periods if min_periods is not None else max(2, window // 2)
        )
        self.name = f"low_vol_{window}"

    def _compute(
        self, prices: pd.DataFrame, universe: list[str], as_of_date: date
    ) -> pd.Series:
        if prices.empty:
            return pd.Series(np.nan, index=universe, name=self.name)

        wide = prices.pivot_table(
            index=DATE_COL, columns="ticker", values=PRICE_COL, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()

        # Daily total returns, then the trailing window of them.
        rets = wide.pct_change()
        recent = rets.tail(self.window)

        vol = recent.std(ddof=1)                 # per-ticker, NaN-aware
        counts = recent.count()
        vol = vol.where(counts >= self.min_periods)   # too little history -> NaN

        # Negate: low volatility -> high factor value -> long leg.
        return (-vol).reindex(universe)
