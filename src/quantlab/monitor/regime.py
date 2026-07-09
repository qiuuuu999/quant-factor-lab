"""regime — market-regime detection and the factor-regime fit matrix.

A factor's IC is rarely regime-invariant: momentum tends to work in trending
markets and get crushed in sharp reversals, low-volatility tends to
outperform in drawdowns, and so on. Averaging IC over a full history that
mixes regimes hides this and can make a factor look mediocre everywhere when
it is actually excellent in one regime and useless (or harmful) in another.

This module classifies each trading day into one of four market regimes from
a single benchmark series (SPY), crossing **volatility level** with **trend
direction**:

* volatility: rolling realised (21-trading-day) volatility, split into
  *low* / *high* at its own **sample median** — a self-referential threshold
  so it adapts to the benchmark's own history rather than an arbitrary fixed
  cutoff;
* trend: price above / below its 200-day moving average — the standard
  "risk-on / risk-off" trend filter.

giving four regimes: ``low_vol_up``, ``low_vol_down``, ``high_vol_up``,
``high_vol_down``. :func:`factor_regime_matrix` then aggregates a factor's
per-period IC by the regime active at each period, producing a
factor-x-regime mean-IC table — the "where does this factor actually work"
map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "REGIMES",
    "classify_regime",
    "regime_as_of",
    "RegimeICMatrix",
    "factor_regime_matrix",
]

REGIMES = ["low_vol_up", "low_vol_down", "high_vol_up", "high_vol_down"]

_TRADING_DAYS = 252


def classify_regime(
    prices: pd.Series,
    *,
    vol_window: int = 21,
    trend_window: int = 200,
    annualize: int = _TRADING_DAYS,
    expanding: bool = False,
) -> pd.Series:
    """Classify each trading day into one of :data:`REGIMES` from a price series.

    ``prices`` is a benchmark (e.g. SPY) close/adj-close series indexed by
    date. Days without a full ``vol_window`` / ``trend_window`` lookback are
    dropped rather than guessed at.

    ``expanding`` controls how the volatility level is split into low/high:

    * ``False`` (default) — the **full-sample** median, i.e. every day's label
      is relative to the whole series' own history including days *after* it.
      This is what a retrospective health report wants (see
      ``quantlab.monitor``'s module docstring): "was this a calm or turbulent
      day, viewed with the full benefit of hindsight."
    * ``True`` — an **expanding** median, using only observations up to and
      including that day. Each day's label then depends only on data known by
      that day, which is the point-in-time-safe mode required whenever the
      label feeds a live trading decision (e.g.
      ``quantlab.portfolio.regime_adaptive``) rather than a backward-looking
      report — using the full-sample median there would leak future
      volatility information into today's regime call.
    """
    px = prices.dropna().sort_index()
    ret = px.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(annualize)
    trend_ma = px.rolling(trend_window).mean()

    valid = vol.notna() & trend_ma.notna()
    threshold = vol.expanding().median() if expanding else vol.median()
    is_high_vol = vol > threshold
    is_up = px > trend_ma

    label = pd.Series(np.where(
        is_high_vol,
        np.where(is_up, "high_vol_up", "high_vol_down"),
        np.where(is_up, "low_vol_up", "low_vol_down"),
    ), index=px.index)
    return label.where(valid).dropna()


def regime_as_of(regime: pd.Series, dates: Sequence[pd.Timestamp]) -> pd.Series:
    """Map arbitrary dates (e.g. a monthly rebalance calendar) to the most
    recently known regime label at or before each date.

    Dates before the first classified regime day have no label and are
    dropped.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    unioned = regime.reindex(regime.index.union(idx)).sort_index().ffill()
    return unioned.reindex(idx).dropna()


@dataclass
class RegimeICMatrix:
    """Per-factor, per-regime mean IC (and period counts) table."""

    mean_ic: pd.DataFrame     # factor x regime
    counts: pd.DataFrame      # factor x regime
    best_regime: pd.Series    # factor -> regime with highest mean IC
    worst_regime: pd.Series   # factor -> regime with lowest mean IC

    def summary(self) -> str:
        lines = []
        for name in self.mean_ic.index:
            best, worst = self.best_regime[name], self.worst_regime[name]
            lines.append(
                f"{name:>16}  best {best:>13} ({self.mean_ic.loc[name, best]:+.4f})  "
                f"worst {worst:>13} ({self.mean_ic.loc[name, worst]:+.4f})"
            )
        return "\n".join(lines)


def factor_regime_matrix(
    ic_by_factor: Mapping[str, pd.Series], regime_by_date: pd.Series
) -> RegimeICMatrix:
    """Aggregate each factor's per-period IC by the regime active on that date.

    ``ic_by_factor`` maps factor name -> per-period IC series (as produced by
    :func:`quantlab.factors.evaluation.information_coefficient`).
    ``regime_by_date`` is a regime label series already aligned to the same
    dates, e.g. via :func:`regime_as_of` on the factor's rebalance calendar.
    """
    mean_rows: dict[str, pd.Series] = {}
    count_rows: dict[str, pd.Series] = {}
    for name, ic in ic_by_factor.items():
        paired = pd.concat([ic.rename("ic"), regime_by_date.rename("regime")], axis=1).dropna()
        grouped = paired.groupby("regime")["ic"]
        mean_rows[name] = grouped.mean().reindex(REGIMES)
        count_rows[name] = grouped.count().reindex(REGIMES).fillna(0).astype(int)

    mean_ic = pd.DataFrame(mean_rows).T.reindex(columns=REGIMES)
    counts = pd.DataFrame(count_rows).T.reindex(columns=REGIMES)
    best = mean_ic.idxmax(axis=1)
    worst = mean_ic.idxmin(axis=1)
    return RegimeICMatrix(mean_ic=mean_ic, counts=counts, best_regime=best, worst_regime=worst)
