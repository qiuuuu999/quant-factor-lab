"""reversal — the short-term (one-month) reversal factor.

Jegadeesh, N. (1990), "Evidence of Predictable Behavior of Security Returns",
*Journal of Finance* 45(3), 881-898.

Jegadeesh documents **short-term reversal**: a stock's return over the most
recent month is *negatively* related to its return over the following month —
last month's losers tend to bounce and last month's winners tend to give back,
a mean-reversion driven by liquidity provision and microstructure effects. (It
is precisely this effect that 12-1 momentum *skips over* with its one-month
gap.) The tradable signal is the **negative** of the trailing one-month return,
so that a *higher* factor value corresponds to a *higher* expected return:

    reversal(t) = - ( P_adj(t) / P_adj(t - `lookback_months`) - 1 )

computed from month-end **adjusted** closes. Stocks with fewer than
``lookback_months + 1`` month-end observations up to the formation date return
``NaN`` rather than raising.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantlab.factors.base import DATE_COL, PRICE_COL, Factor

__all__ = ["ShortTermReversalFactor"]

# pandas month-end resample alias ("ME" on modern pandas, "M" on older).
try:  # pragma: no cover - trivial version shim, mirrors factors.momentum
    pd.Series(dtype=float, index=pd.DatetimeIndex([])).resample("ME")
    _MONTH_END = "ME"
except ValueError:  # pragma: no cover
    _MONTH_END = "M"


class ShortTermReversalFactor(Factor):
    """Cross-sectional short-term reversal (negative trailing month return).

    Parameters
    ----------
    lookback_months:
        Length of the trailing return window in months (default 1 — the classic
        one-month reversal).
    """

    def __init__(self, lookback_months: int = 1):
        if lookback_months < 1:
            raise ValueError("lookback_months must be >= 1")
        self.lookback_months = lookback_months
        self.name = f"reversal_{lookback_months}m"

    def _compute(
        self, prices: pd.DataFrame, universe: list[str], as_of_date: date
    ) -> pd.Series:
        if prices.empty:
            return pd.Series(np.nan, index=universe, name=self.name)

        wide = prices.pivot_table(
            index=DATE_COL, columns="ticker", values=PRICE_COL, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index)
        monthly = wide.resample(_MONTH_END).last()
        monthly = monthly[monthly.index <= pd.Timestamp(as_of_date)]

        if len(monthly) < self.lookback_months + 1:
            return pd.Series(np.nan, index=universe, name=self.name)

        recent = monthly.iloc[-1]                         # price now
        base = monthly.iloc[-1 - self.lookback_months]    # price `lookback` ago
        past_return = recent / base - 1.0

        # Negate: last month's losers -> high factor value -> long leg.
        return (-past_return).reindex(universe)
