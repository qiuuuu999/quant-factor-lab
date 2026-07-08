"""momentum — the classic 12-1 price momentum factor.

Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers:
Implications for Stock Market Efficiency", *Journal of Finance* 48(1), 65-91.

The factor is the cumulative total return over the past ``lookback`` months
*excluding* the most recent ``skip`` month(s). The one-month skip removes the
well-documented short-term reversal effect so that momentum, not micro-structure
mean-reversion, is what is measured:

    mom_{12-1}(t) = P_adj(t - 1 month) / P_adj(t - 12 months) - 1

Computed from **adjusted** close prices (so splits and dividends do not create
spurious jumps). Stocks with insufficient history (fewer than ``lookback + 1``
month-end observations up to the formation date) return ``NaN`` rather than
raising.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantlab.factors.base import DATE_COL, PRICE_COL, Factor

__all__ = ["MomentumFactor"]

# pandas month-end resample alias ("ME" on modern pandas, "M" on older).
try:  # pragma: no cover - trivial version shim
    pd.Series(dtype=float, index=pd.DatetimeIndex([])).resample("ME")
    _MONTH_END = "ME"
except ValueError:  # pragma: no cover
    _MONTH_END = "M"


class MomentumFactor(Factor):
    """Cross-sectional 12-1 momentum.

    Parameters
    ----------
    lookback:
        Total look-back window in months (default 12).
    skip:
        Most-recent months to exclude (default 1).
    """

    def __init__(self, lookback: int = 12, skip: int = 1):
        if skip < 0 or lookback <= skip:
            raise ValueError("require lookback > skip >= 0")
        self.lookback = lookback
        self.skip = skip
        self.name = f"momentum_{lookback}_{skip}"

    def _compute(
        self, prices: pd.DataFrame, universe: list[str], as_of_date: date
    ) -> pd.Series:
        if prices.empty:
            return pd.Series(np.nan, index=universe, name=self.name)

        # Long -> wide adjusted-close panel, then month-end sampling.
        wide = prices.pivot_table(
            index=DATE_COL, columns="ticker", values=PRICE_COL, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index)
        monthly = wide.resample(_MONTH_END).last()
        monthly = monthly[monthly.index <= pd.Timestamp(as_of_date)]

        # Need the formation month plus `lookback` prior months.
        if len(monthly) < self.lookback + 1:
            return pd.Series(np.nan, index=universe, name=self.name)

        recent = monthly.iloc[-1 - self.skip]        # price `skip` months ago
        base = monthly.iloc[-1 - self.lookback]      # price `lookback` months ago
        mom = recent / base - 1.0

        # Tickers missing either endpoint (e.g. IPO'd mid-window) become NaN.
        return mom.reindex(universe)
