"""liquidity — the Amihud (2002) illiquidity factor.

Amihud, Y. (2002), "Illiquidity and stock returns: cross-section and time-series
effects", *Journal of Financial Markets* 5(1), 31-56.

Amihud's **ILLIQ** measures price impact as the average daily ratio of absolute
return to dollar volume: how much the price moves per dollar traded. Illiquid
stocks (a large move per dollar) command an **illiquidity premium** — higher
expected returns to compensate for higher trading costs. The measure is:

    ILLIQ(t) = mean( |r_d| / (P_d * V_d) ,  d in the last `window` days up to t )

where ``r_d`` is the daily return from **adjusted** closes and ``P_d * V_d`` is
the day's **dollar volume** (raw close × raw share volume — the actual value
traded that session). The raw mean is scaled by ``scale`` (Amihud's 1e6) purely
for readable magnitudes; scaling is monotone so it does not affect ranks, ICs,
or decile sorts.

Higher illiquidity ⇒ higher expected return, so — consistent with the rest of
the library — a higher factor value already corresponds to the long leg and no
sign flip is needed.

Days with zero dollar volume are dropped from the average. Stocks with fewer
than ``min_periods`` valid days in the window return ``NaN``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantlab.factors.base import DATE_COL, PRICE_COL, Factor

__all__ = ["AmihudIlliquidityFactor"]

#: Extra columns (beyond date/adj_close) this factor needs from the price frame.
_REQUIRED = ("close", "volume")


class AmihudIlliquidityFactor(Factor):
    """Cross-sectional Amihud illiquidity factor.

    Parameters
    ----------
    window:
        Trailing look-back in **trading days** (default 21 ≈ one month).
    min_periods:
        Minimum valid (non-zero-volume) days required in the window; below this
        the value is ``NaN``. Defaults to ``max(2, window // 2)``.
    scale:
        Multiplier applied to the raw mean for readability (default 1e6, as in
        Amihud 2002). Does not affect ranking-based analytics.
    """

    def __init__(
        self,
        window: int = 21,
        min_periods: int | None = None,
        scale: float = 1e6,
    ):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.min_periods = (
            min_periods if min_periods is not None else max(2, window // 2)
        )
        self.scale = scale
        self.name = f"amihud_illiq_{window}"

    def _compute(
        self, prices: pd.DataFrame, universe: list[str], as_of_date: date
    ) -> pd.Series:
        if prices.empty:
            return pd.Series(np.nan, index=universe, name=self.name)

        missing = [c for c in _REQUIRED if c not in prices.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__} needs column(s) {missing}; dollar volume "
                f"requires raw 'close' and 'volume'."
            )

        def _wide(field: str) -> pd.DataFrame:
            w = prices.pivot_table(
                index=DATE_COL, columns="ticker", values=field, aggfunc="last"
            )
            w.index = pd.to_datetime(w.index)
            return w.sort_index()

        adj = _wide(PRICE_COL)
        close = _wide("close")
        volume = _wide("volume")

        abs_ret = adj.pct_change().abs()
        dollar_volume = (close * volume).replace(0.0, np.nan)  # avoid /0
        illiq_daily = abs_ret / dollar_volume

        recent = illiq_daily.tail(self.window)
        illiq = recent.mean()                     # per-ticker, NaN-aware
        counts = recent.count()
        illiq = illiq.where(counts >= self.min_periods)   # too little -> NaN

        return (illiq * self.scale).reindex(universe)
