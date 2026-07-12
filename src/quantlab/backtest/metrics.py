"""metrics — performance analytics for a backtested NAV series.

Turns the daily net-asset-value series produced by
:func:`quantlab.backtest.engine.run_backtest` into the standard battery of
performance statistics, optionally relative to a benchmark NAV (e.g. SPY).

Conventions
-----------
* **Returns** are simple period-over-period returns of the NAV.
* **Annualisation** uses a ``periods_per_year`` factor. When not supplied it is
  inferred from the index spacing (daily -> 252, weekly -> 52, monthly -> 12,
  ...). CAGR annualises by the *number of periods*, so a series that doubles over
  exactly two years of monthly points reports ``2 ** (1/2) - 1 = 41.42%``.
* **Drawdown** is measured off the running peak of the NAV; the reported maximum
  drawdown carries the peak and trough dates that bracket it.
* **Benchmark-relative** stats (information ratio, annualised excess return) use
  the active return series ``strategy - benchmark`` on the aligned dates.

Every statistic is also exposed as a standalone function so callers can compute
just one; :func:`compute_metrics` bundles them into a :class:`PerformanceMetrics`
record with tidy ``to_series`` / ``to_frame`` / ``summary`` views.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "PerformanceMetrics",
    "infer_periods_per_year",
    "to_returns",
    "total_return",
    "cagr",
    "annual_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "drawdown_series",
    "calmar_ratio",
    "monthly_win_rate",
    "annual_turnover",
    "information_ratio",
    "tracking_error",
    "compute_metrics",
]

_MONTH_END = "ME"
try:  # pragma: no cover - version shim, mirrors factors.momentum
    pd.Series(dtype=float, index=pd.DatetimeIndex([])).resample("ME")
except ValueError:  # pragma: no cover
    _MONTH_END = "M"


# --------------------------------------------------------------------------- #
# Frequency inference
# --------------------------------------------------------------------------- #

def infer_periods_per_year(index: pd.Index) -> int:
    """Guess the number of periods per year from a DatetimeIndex's spacing.

    Returns one of ``{252, 52, 12, 4, 1}`` (trading-daily, weekly, monthly,
    quarterly, annual). Defaults to ``252`` when there is too little to infer.
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return 252
    med_days = float(np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int)))
    if med_days <= 3:
        return 252
    if med_days <= 10:
        return 52
    if med_days <= 45:
        return 12
    if med_days <= 120:
        return 4
    return 1


def _ppy(returns_or_nav: pd.Series, periods_per_year: int | None) -> int:
    return periods_per_year or infer_periods_per_year(returns_or_nav.index)


# --------------------------------------------------------------------------- #
# Core building blocks
# --------------------------------------------------------------------------- #

def to_returns(nav: pd.Series) -> pd.Series:
    """Simple period returns of a NAV series (first NaN dropped)."""
    return nav.pct_change().dropna()


def total_return(nav: pd.Series) -> float:
    """Cumulative return over the whole series: ``nav[-1] / nav[0] - 1``."""
    if len(nav) < 2:
        return 0.0
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def cagr(nav: pd.Series, periods_per_year: int | None = None) -> float:
    """Compound annual growth rate, annualised by number of periods."""
    if len(nav) < 2:
        return 0.0
    ppy = _ppy(nav, periods_per_year)
    n_periods = len(nav) - 1
    growth = nav.iloc[-1] / nav.iloc[0]
    if growth <= 0:
        return -1.0
    years = n_periods / ppy
    return float(growth ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series, periods_per_year: int | None = None) -> float:
    """Annualised standard deviation of returns (sample std, ddof=1)."""
    if len(returns) < 2:
        return 0.0
    ppy = _ppy(returns, periods_per_year)
    return float(returns.std(ddof=1) * np.sqrt(ppy))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int | None = None,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio. ``risk_free`` is an *annual* rate."""
    if len(returns) < 2:
        return 0.0
    ppy = _ppy(returns, periods_per_year)
    excess = returns - risk_free / ppy
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(ppy))


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak at each point (<= 0)."""
    peak = nav.cummax()
    return nav / peak - 1.0


def max_drawdown(nav: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Maximum drawdown and the ``(peak_date, trough_date)`` that bracket it.

    The value is negative (or ``0`` for a monotonically non-decreasing series).
    """
    if len(nav) < 2:
        return 0.0, None, None
    dd = drawdown_series(nav)
    trough = dd.idxmin()
    mdd = float(dd.loc[trough])
    if mdd == 0.0:
        return 0.0, None, None
    peak = nav.loc[:trough].idxmax()
    return mdd, peak, trough


def calmar_ratio(nav: pd.Series, periods_per_year: int | None = None) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    mdd, _, _ = max_drawdown(nav)
    if mdd == 0.0:
        return float("nan")
    return cagr(nav, periods_per_year) / abs(mdd)


def monthly_win_rate(nav: pd.Series) -> float:
    """Fraction of calendar months with a positive return."""
    if len(nav) < 2:
        return 0.0
    monthly = nav.resample(_MONTH_END).last()
    rets = monthly.pct_change().dropna()
    if rets.empty:
        return 0.0
    return float((rets > 0).mean())


def annual_turnover(
    turnover: pd.Series, rebalances_per_year: int | None = None
) -> float:
    """Average turnover per rebalance scaled to an annual figure.

    ``turnover`` is the per-rebalance series from
    :class:`~quantlab.backtest.engine.BacktestResult` (traded notional / pre-trade
    value). The annual figure is ``mean(turnover) * rebalances_per_year``, with
    the rebalance cadence inferred from the index when not given.
    """
    if turnover is None or len(turnover) == 0:
        return 0.0
    rpy = rebalances_per_year or infer_periods_per_year(turnover.index)
    return float(turnover.mean() * rpy)


# --------------------------------------------------------------------------- #
# Benchmark-relative
# --------------------------------------------------------------------------- #

def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    return joined.iloc[:, 0], joined.iloc[:, 1]


def tracking_error(
    returns: pd.Series, bench_returns: pd.Series, periods_per_year: int | None = None
) -> float:
    """Annualised standard deviation of the active return (strategy - bench)."""
    r, b = _align(returns, bench_returns)
    active = r - b
    return annual_volatility(active, periods_per_year)


def information_ratio(
    returns: pd.Series, bench_returns: pd.Series, periods_per_year: int | None = None
) -> float:
    """Annualised active return divided by tracking error."""
    r, b = _align(returns, bench_returns)
    active = r - b
    if len(active) < 2:
        return 0.0
    ppy = _ppy(active, periods_per_year)
    sd = active.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(active.mean() / sd * np.sqrt(ppy))


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #

@dataclass
class PerformanceMetrics:
    """A full performance record for one strategy (optionally vs. a benchmark)."""

    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    max_dd_peak: pd.Timestamp | None
    max_dd_trough: pd.Timestamp | None
    calmar: float
    monthly_win_rate: float
    annual_turnover: float
    periods_per_year: int
    # Benchmark-relative (NaN when no benchmark supplied).
    benchmark_cagr: float = float("nan")
    excess_cagr: float = float("nan")
    information_ratio: float = float("nan")
    tracking_error: float = float("nan")

    @property
    def has_benchmark(self) -> bool:
        return not np.isnan(self.excess_cagr)

    def to_series(self) -> pd.Series:
        """Metrics as a labelled Series (dates/percentages formatted for display)."""
        def _d(ts: pd.Timestamp | None) -> str:
            return "" if ts is None else pd.Timestamp(ts).date().isoformat()

        rows: dict[str, object] = {
            "Total Return": f"{self.total_return:.2%}",
            "CAGR": f"{self.cagr:.2%}",
            "Annual Volatility": f"{self.annual_volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Max DD Peak": _d(self.max_dd_peak),
            "Max DD Trough": _d(self.max_dd_trough),
            "Calmar Ratio": f"{self.calmar:.2f}",
            "Monthly Win Rate": f"{self.monthly_win_rate:.2%}",
            "Annual Turnover": f"{self.annual_turnover:.2f}x",
        }
        if self.has_benchmark:
            rows.update({
                "Benchmark CAGR": f"{self.benchmark_cagr:.2%}",
                "Excess CAGR": f"{self.excess_cagr:.2%}",
                "Information Ratio": f"{self.information_ratio:.2f}",
                "Tracking Error": f"{self.tracking_error:.2%}",
            })
        return pd.Series(rows, name="value")

    def to_frame(self) -> pd.DataFrame:
        """Two-column (Metric, Value) DataFrame — handy for tables/markdown."""
        s = self.to_series()
        return pd.DataFrame({"Metric": s.index, "Value": s.values})

    def to_markdown(self) -> str:
        """Markdown table of the metrics."""
        lines = ["| Metric | Value |", "| --- | --- |"]
        for metric, value in self.to_series().items():
            lines.append(f"| {metric} | {value} |")
        return "\n".join(lines)

    def summary(self) -> str:
        return self.to_series().to_string()


def compute_metrics(
    nav: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    periods_per_year: int | None = None,
    risk_free: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full :class:`PerformanceMetrics` for a NAV series.

    Parameters
    ----------
    nav:
        Strategy net-asset-value series indexed by date.
    benchmark:
        Optional benchmark NAV (e.g. SPY). Aligned to ``nav``'s dates before
        computing excess return, information ratio, and tracking error.
    turnover:
        Optional per-rebalance turnover series (for annual turnover).
    periods_per_year:
        Annualisation factor; inferred from ``nav`` when omitted.
    risk_free:
        Annual risk-free rate used in the Sharpe ratio.
    """
    nav = nav.dropna()
    ppy = _ppy(nav, periods_per_year)
    rets = to_returns(nav)
    mdd, peak, trough = max_drawdown(nav)

    m = PerformanceMetrics(
        total_return=total_return(nav),
        cagr=cagr(nav, ppy),
        annual_volatility=annual_volatility(rets, ppy),
        sharpe=sharpe_ratio(rets, ppy, risk_free),
        max_drawdown=mdd,
        max_dd_peak=peak,
        max_dd_trough=trough,
        calmar=calmar_ratio(nav, ppy),
        monthly_win_rate=monthly_win_rate(nav),
        annual_turnover=annual_turnover(turnover) if turnover is not None else 0.0,
        periods_per_year=ppy,
    )

    if benchmark is not None:
        bench = benchmark.dropna()
        n_aligned, b_aligned = _align(nav, bench)
        bench_rets = to_returns(b_aligned)
        m.benchmark_cagr = cagr(b_aligned, ppy)
        m.excess_cagr = m.cagr - m.benchmark_cagr
        m.information_ratio = information_ratio(rets, bench_rets, ppy)
        m.tracking_error = tracking_error(rets, bench_rets, ppy)

    return m
