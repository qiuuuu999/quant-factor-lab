"""evaluation — a factor-scoring toolkit (IC, quantile sorts, turnover, overlap).

Given a :class:`~quantlab.factors.base.Factor` and a point-in-time price history,
this module answers the four questions that decide whether a signal is worth
trading, *before* a full backtest is run:

1. **Information Coefficient (IC)** — does the factor rank next period's winners?
   The per-period Spearman rank correlation between the factor value at ``t`` and
   the realised forward return over ``[t, t+1]``. Its mean gauges sign/strength;
   its ``ICIR`` (mean / std) gauges *consistency*; a ``t``-stat and hit-rate come
   for free.
2. **Quantile (decile) returns** — sort names into ``n`` equal-count buckets each
   period, equal-weight, and compound. A good factor produces a **monotone**
   spread from the bottom bucket to the top and a positive top-minus-bottom
   (long/short) return.
3. **Rank autocorrelation** — the period-over-period Spearman correlation of the
   factor's own cross-sectional ranks. High autocorrelation ⇒ stable rankings ⇒
   low turnover ⇒ low trading cost.
4. **Factor correlation** — the average cross-sectional rank correlation *between*
   factors, to see how much unique information each adds.

The workhorse is :func:`build_factor_panel`, which turns a factor + universe +
rebalance calendar into two aligned ``date × ticker`` panels — factor values and
forward returns — that every analytic consumes.

**Point-in-time.** Factor values are produced through ``Factor.compute`` with the
look-ahead guard on, so each is formed from data on or before its formation date.
Forward returns look *strictly forward* (formation close to next-formation close)
and are therefore never part of the signal — this is a signal-quality diagnostic,
not a tradable P&L (the backtest engine, which fills at ``t+1`` open with costs,
is the tradable measure).

All rank-based analytics are invariant to winsorization/standardisation, so raw
factor values are fed in directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from quantlab.factors.base import Factor
from quantlab.factors.preprocess import deciles

__all__ = [
    "FactorPanel",
    "build_factor_panel",
    "ICResult",
    "information_coefficient",
    "DecileResult",
    "decile_returns",
    "rank_autocorrelation",
    "factor_correlation",
]

_MONTHS_PER_YEAR = 12

MembersFn = Callable[[pd.Timestamp], Iterable[str]]


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #

@dataclass
class FactorPanel:
    """Aligned factor-value and forward-return panels for one factor.

    Both frames are indexed by rebalance date with one column per ticker.
    ``forward_returns[t, i]`` is the return of ticker ``i`` from the formation
    close at ``t`` to the next formation close (``NaN`` on the last date).
    """

    name: str
    factor_values: pd.DataFrame
    forward_returns: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.factor_values.index)


def build_factor_panel(
    factor: Factor,
    long_prices: pd.DataFrame,
    rebal_dates: Sequence[pd.Timestamp],
    members_by_date: MembersFn,
    *,
    history_months: int = 16,
    price_col: str = "adj_close",
) -> FactorPanel:
    """Compute a factor and its forward returns across a rebalance calendar.

    Parameters
    ----------
    factor:
        The factor to score. Its ``compute`` is called per date with PIT
        enforcement on.
    long_prices:
        Tidy long price frame (``date``, ``ticker``, ``adj_close``, …) as
        returned by :func:`quantlab.data.prices.get_prices`.
    rebal_dates:
        Formation dates (typically month-ends), ascending.
    members_by_date:
        Callable ``dt -> iterable of tickers`` giving the point-in-time universe
        eligible on ``dt`` (e.g. index membership ∩ priced names).
    history_months:
        How many months of price history to hand each ``compute`` call. Must
        cover the factor's longest look-back (16 comfortably covers 12-1
        momentum and 252-day volatility).
    price_col:
        Column used to build forward returns (adjusted close).
    """
    rebal_dates = [pd.Timestamp(d) for d in rebal_dates]

    # Wide adjusted-close panel -> forward returns between consecutive formation
    # dates. rebal_dates are drawn from the panel's own index, so .reindex hits
    # real rows; a name not trading exactly on a formation date becomes NaN and
    # is simply dropped by the downstream pairwise analytics.
    wide = long_prices.pivot_table(
        index="date", columns="ticker", values=price_col, aggfunc="last"
    )
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    at_rebal = wide.reindex(rebal_dates)
    forward_returns = at_rebal.shift(-1) / at_rebal - 1.0

    # Factor values per formation date (PIT-enforced inside compute).
    by_ticker = {t: g for t, g in long_prices.groupby("ticker", sort=False)}
    values: dict[pd.Timestamp, pd.Series] = {}
    for dt in rebal_dates:
        members = sorted(members_by_date(dt))
        if not members:
            continue
        lo = dt - pd.DateOffset(months=history_months)
        frames = [
            by_ticker[t] for t in members if t in by_ticker
        ]
        if not frames:
            continue
        sub = pd.concat(frames, ignore_index=True)
        sub = sub[(sub["date"] >= lo) & (sub["date"] <= dt)]
        values[dt] = factor.compute(sub, members, dt)   # PIT-enforced

    factor_values = pd.DataFrame(values).T
    factor_values.index = pd.to_datetime(factor_values.index)
    factor_values = factor_values.sort_index()

    # Share a common column set so the two panels align cleanly.
    cols = factor_values.columns.union(forward_returns.columns)
    factor_values = factor_values.reindex(columns=cols)
    forward_returns = forward_returns.reindex(columns=cols).reindex(
        index=factor_values.index
    )
    return FactorPanel(factor.name, factor_values, forward_returns)


# --------------------------------------------------------------------------- #
# Information Coefficient
# --------------------------------------------------------------------------- #

def _paired_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def information_coefficient(
    panel: FactorPanel, *, method: str = "spearman"
) -> pd.Series:
    """Per-period rank IC: corr(factor value at ``t``, forward return over ``t``).

    Returns a Series indexed by formation date. Dates with fewer than two paired
    (factor, forward-return) observations are dropped.
    """
    fv, fwd = panel.factor_values, panel.forward_returns
    ic: dict[pd.Timestamp, float] = {}
    for dt in fv.index:
        value = _paired_corr(fv.loc[dt], fwd.loc[dt], method)
        if not np.isnan(value):
            ic[dt] = value
    return pd.Series(ic, name=f"{panel.name}_ic", dtype=float)


@dataclass
class ICResult:
    """Summary statistics of an IC time series."""

    name: str
    series: pd.Series
    mean: float
    std: float
    icir: float          # mean / std of the per-period IC
    t_stat: float        # icir * sqrt(n)
    hit_rate: float      # fraction of periods with IC > 0
    n: int

    @classmethod
    def from_series(cls, ic: pd.Series, name: str) -> "ICResult":
        clean = ic.dropna()
        n = int(len(clean))
        mean = float(clean.mean()) if n else np.nan
        std = float(clean.std(ddof=1)) if n > 1 else np.nan
        icir = mean / std if std and not np.isnan(std) else np.nan
        t_stat = icir * np.sqrt(n) if not np.isnan(icir) else np.nan
        hit_rate = float((clean > 0).mean()) if n else np.nan
        return cls(name, ic, mean, std, icir, t_stat, hit_rate, n)

    def summary(self) -> str:
        return (
            f"{self.name:>16}  mean IC {self.mean:+.4f}  "
            f"ICIR {self.icir:+.3f}  t {self.t_stat:+.2f}  "
            f"hit {self.hit_rate:.0%}  (n={self.n})"
        )


# --------------------------------------------------------------------------- #
# Quantile (decile) returns
# --------------------------------------------------------------------------- #

def _annualize(monthly: pd.Series, periods_per_year: int) -> float:
    """Geometric annualisation of a series of per-period simple returns."""
    r = monthly.dropna()
    if r.empty:
        return np.nan
    growth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if growth <= 0 or years <= 0:
        return np.nan
    return growth ** (1.0 / years) - 1.0


@dataclass
class DecileResult:
    """Quantile-sort diagnostics for one factor."""

    name: str
    n_buckets: int
    periods_per_year: int
    monthly_returns: pd.DataFrame     # date × bucket (1..n), equal-weight
    annualized: pd.Series             # bucket -> annualised return
    long_short_monthly: pd.Series     # per-period (top − bottom) return
    long_short_annualized: float
    monotonicity: float               # Spearman corr(bucket index, ann. return)

    def summary(self) -> str:
        return (
            f"{self.name:>16}  L/S ann {self.long_short_annualized:+.2%}  "
            f"monotonicity {self.monotonicity:+.2f}  "
            f"(Q{self.n_buckets} {self.annualized.iloc[-1]:+.1%} vs "
            f"Q1 {self.annualized.iloc[0]:+.1%})"
        )


def decile_returns(
    panel: FactorPanel, *, n: int = 10, periods_per_year: int = _MONTHS_PER_YEAR
) -> DecileResult:
    """Equal-weight forward returns of each factor quantile, rebuilt each period.

    Bucket ``1`` holds the lowest factor values, bucket ``n`` the highest, so a
    factor aligned to expected return produces an increasing return profile and
    a positive ``n − 1`` (long/short) spread.
    """
    fv, fwd = panel.factor_values, panel.forward_returns
    rows: dict[pd.Timestamp, pd.Series] = {}
    for dt in fv.index:
        buckets = deciles(fv.loc[dt], n)
        frame = pd.DataFrame({"bucket": buckets, "ret": fwd.loc[dt]}).dropna()
        if frame.empty:
            continue
        rows[dt] = frame.groupby("bucket")["ret"].mean()

    monthly = pd.DataFrame(rows).T.sort_index()
    monthly.index = pd.to_datetime(monthly.index)
    # Ensure every bucket column 1..n exists and is ordered.
    monthly = monthly.reindex(columns=[float(b) for b in range(1, n + 1)])

    annualized = monthly.apply(lambda c: _annualize(c, periods_per_year))
    annualized.index = [int(b) for b in annualized.index]

    top, bottom = float(n), 1.0
    long_short_monthly = (monthly[top] - monthly[bottom]).dropna()
    long_short_monthly.name = f"{panel.name}_long_short"
    ls_ann = _annualize(long_short_monthly, periods_per_year)

    ann = annualized.dropna()
    if len(ann) >= 2:
        monotonicity = float(
            pd.Series(ann.index, index=ann.index).corr(ann, method="spearman")
        )
    else:
        monotonicity = np.nan

    return DecileResult(
        name=panel.name,
        n_buckets=n,
        periods_per_year=periods_per_year,
        monthly_returns=monthly,
        annualized=annualized,
        long_short_monthly=long_short_monthly,
        long_short_annualized=ls_ann,
        monotonicity=monotonicity,
    )


# --------------------------------------------------------------------------- #
# Turnover (rank autocorrelation) and cross-factor overlap
# --------------------------------------------------------------------------- #

def rank_autocorrelation(
    panel: FactorPanel, *, lag: int = 1, method: str = "spearman"
) -> pd.Series:
    """Period-over-period rank autocorrelation of the factor's own values.

    A value near ``1`` means the ranking barely moves (low turnover); a value
    near ``0`` means it is reshuffled every period (high turnover). Indexed by
    the *later* date of each pair.
    """
    fv = panel.factor_values
    dates = list(fv.index)
    out: dict[pd.Timestamp, float] = {}
    for i in range(lag, len(dates)):
        value = _paired_corr(fv.loc[dates[i - lag]], fv.loc[dates[i]], method)
        if not np.isnan(value):
            out[dates[i]] = value
    return pd.Series(out, name=f"{panel.name}_rank_autocorr", dtype=float)


def factor_correlation(
    panels: Sequence[FactorPanel], *, method: str = "spearman"
) -> pd.DataFrame:
    """Average cross-sectional rank correlation *between* factors.

    For each date shared by all panels, the factors' cross-sectional values form
    a ``ticker × factor`` frame whose correlation matrix is computed; the return
    is the element-wise mean of those matrices over time (NaN-aware).
    """
    if not panels:
        return pd.DataFrame()
    names = [p.name for p in panels]
    common = set(panels[0].factor_values.index)
    for p in panels[1:]:
        common &= set(p.factor_values.index)

    mats: list[pd.DataFrame] = []
    for dt in sorted(common):
        frame = pd.DataFrame({p.name: p.factor_values.loc[dt] for p in panels})
        c = frame.corr(method=method)
        mats.append(c)

    if not mats:
        return pd.DataFrame(np.nan, index=names, columns=names)

    avg = pd.concat(mats).groupby(level=0).mean()
    return avg.reindex(index=names, columns=names)
