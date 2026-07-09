"""Tests for the performance-metrics module.

Every statistic is asserted against a hand-built series with a closed-form
answer (e.g. a NAV that doubles over exactly two years annualises to
``2 ** (1/2) - 1 = 41.42%``), so the annualisation and drawdown arithmetic is
pinned exactly rather than checked against itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.metrics import (
    PerformanceMetrics,
    annual_turnover,
    annual_volatility,
    calmar_ratio,
    cagr,
    compute_metrics,
    infer_periods_per_year,
    information_ratio,
    max_drawdown,
    monthly_win_rate,
    sharpe_ratio,
    total_return,
    tracking_error,
)

SQRT252 = np.sqrt(252)


def _daily(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


# --------------------------------------------------------------------------- #
# Total return & CAGR
# --------------------------------------------------------------------------- #

def test_total_return():
    nav = pd.Series([100.0, 150.0], index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
    assert total_return(nav) == pytest.approx(0.5)


def test_cagr_two_year_doubling_is_41_42_percent():
    # 25 monthly points doubling from 100 to 200 => exactly two years of returns.
    dates = pd.date_range("2020-01-31", periods=25, freq="ME")
    nav = pd.Series(100.0 * 2.0 ** (np.arange(25) / 24.0), index=dates)
    assert nav.iloc[-1] == pytest.approx(200.0)
    assert total_return(nav) == pytest.approx(1.0)
    # 2 ** (1/2) - 1 = 0.414213...
    assert cagr(nav, periods_per_year=12) == pytest.approx(0.41421356, abs=1e-6)
    # Monthly spacing is inferred as 12 periods/year, giving the same answer.
    assert cagr(nav) == pytest.approx(0.41421356, abs=1e-6)


def test_cagr_single_point_is_zero():
    nav = pd.Series([100.0], index=pd.to_datetime(["2020-01-31"]))
    assert cagr(nav) == 0.0
    assert total_return(nav) == 0.0


# --------------------------------------------------------------------------- #
# Volatility & Sharpe
# --------------------------------------------------------------------------- #

def test_annual_volatility_known_value():
    rets = _daily([0.01, -0.01])
    # sample std = sqrt(0.0002) = 0.0141421; annualised * sqrt(252).
    assert annual_volatility(rets, periods_per_year=252) == pytest.approx(
        np.sqrt(0.0002) * SQRT252, abs=1e-9
    )
    assert annual_volatility(rets, periods_per_year=252) == pytest.approx(0.2244994, abs=1e-6)


def test_sharpe_ratio_known_value():
    rets = _daily([0.02, 0.00])           # mean 0.01, std sqrt(0.0002)
    # 0.01 / 0.0141421 * sqrt(252) = 11.22497
    assert sharpe_ratio(rets, periods_per_year=252) == pytest.approx(11.22497, abs=1e-4)


def test_sharpe_zero_variance_is_zero():
    rets = _daily([0.01, 0.01, 0.01])
    assert sharpe_ratio(rets, periods_per_year=252) == 0.0


def test_sharpe_risk_free_reduces_ratio():
    rets = _daily([0.02, 0.00])
    assert sharpe_ratio(rets, 252, risk_free=0.10) < sharpe_ratio(rets, 252, risk_free=0.0)


# --------------------------------------------------------------------------- #
# Drawdown / Calmar
# --------------------------------------------------------------------------- #

def test_max_drawdown_value_and_dates():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    nav = pd.Series([100.0, 120.0, 90.0, 150.0], index=dates)
    mdd, peak, trough = max_drawdown(nav)
    assert mdd == pytest.approx(-0.25)          # 90 / 120 - 1
    assert peak == dates[1]                     # ran up to 120 first
    assert trough == dates[2]                    # bottomed at 90


def test_max_drawdown_monotonic_is_zero():
    nav = _daily([100.0, 101.0, 102.0])
    mdd, peak, trough = max_drawdown(nav)
    assert mdd == 0.0 and peak is None and trough is None


def test_calmar_ratio_known_value():
    # Yearly points: 100 -> 50 -> 200. CAGR = 2**(1/2)-1; max DD = -0.5.
    dates = pd.date_range("2020-12-31", periods=3, freq="YE")
    nav = pd.Series([100.0, 50.0, 200.0], index=dates)
    assert calmar_ratio(nav, periods_per_year=1) == pytest.approx(0.41421356 / 0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Win rate & turnover
# --------------------------------------------------------------------------- #

def test_monthly_win_rate():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    nav = pd.Series([100.0, 110.0, 99.0, 108.0, 120.0], index=dates)
    # monthly returns: +, -, +, + => 3 of 4 positive.
    assert monthly_win_rate(nav) == pytest.approx(0.75)


def test_annual_turnover_scales_by_cadence():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    turn = pd.Series([0.5] * 12, index=dates)
    # mean 0.5 * 12 rebalances/year = 6.0
    assert annual_turnover(turn) == pytest.approx(6.0)
    assert annual_turnover(turn, rebalances_per_year=12) == pytest.approx(6.0)


def test_annual_turnover_empty_is_zero():
    assert annual_turnover(pd.Series(dtype=float)) == 0.0


# --------------------------------------------------------------------------- #
# Benchmark-relative
# --------------------------------------------------------------------------- #

def test_information_ratio_and_tracking_error_known():
    rets = _daily([0.03, 0.01])
    bench = _daily([0.01, 0.01])
    # active = [0.02, 0.00]: same shape as the Sharpe case.
    assert information_ratio(rets, bench, periods_per_year=252) == pytest.approx(11.22497, abs=1e-4)
    assert tracking_error(rets, bench, periods_per_year=252) == pytest.approx(
        np.sqrt(0.0002) * SQRT252, abs=1e-9
    )


def test_information_ratio_zero_when_matching_benchmark():
    rets = _daily([0.01, 0.02, -0.01])
    assert information_ratio(rets, rets.copy(), periods_per_year=252) == 0.0


# --------------------------------------------------------------------------- #
# Frequency inference
# --------------------------------------------------------------------------- #

def test_infer_periods_per_year():
    assert infer_periods_per_year(pd.bdate_range("2020-01-01", periods=300)) == 252
    assert infer_periods_per_year(pd.date_range("2020-01-01", periods=60, freq="W")) == 52
    assert infer_periods_per_year(pd.date_range("2020-01-31", periods=36, freq="ME")) == 12
    assert infer_periods_per_year(pd.date_range("2020-03-31", periods=12, freq="QE")) == 4


# --------------------------------------------------------------------------- #
# compute_metrics bundle
# --------------------------------------------------------------------------- #

def test_compute_metrics_without_benchmark():
    dates = pd.date_range("2020-01-31", periods=25, freq="ME")
    nav = pd.Series(100.0 * 2.0 ** (np.arange(25) / 24.0), index=dates)
    m = compute_metrics(nav, periods_per_year=12)
    assert isinstance(m, PerformanceMetrics)
    assert m.cagr == pytest.approx(0.41421356, abs=1e-6)
    assert m.total_return == pytest.approx(1.0)
    assert not m.has_benchmark
    assert np.isnan(m.information_ratio)
    # Views render without error.
    assert "CAGR" in m.to_series().index
    assert m.to_markdown().startswith("| Metric | Value |")


def test_compute_metrics_with_benchmark_excess():
    dates = pd.date_range("2020-01-31", periods=25, freq="ME")
    strat = pd.Series(100.0 * 2.0 ** (np.arange(25) / 24.0), index=dates)   # doubles
    bench = pd.Series(100.0 * 1.5 ** (np.arange(25) / 24.0), index=dates)   # +50%
    m = compute_metrics(strat, benchmark=bench, periods_per_year=12)
    assert m.has_benchmark
    assert m.benchmark_cagr == pytest.approx(1.5 ** 0.5 - 1.0, abs=1e-6)
    assert m.excess_cagr == pytest.approx(m.cagr - m.benchmark_cagr, abs=1e-12)
    assert "Information Ratio" in m.to_series().index


def test_compute_metrics_with_turnover():
    dates = pd.date_range("2020-01-31", periods=13, freq="ME")
    nav = pd.Series(100.0 * 1.01 ** np.arange(13), index=dates)
    turn = pd.Series([0.4] * 12, index=dates[1:])
    m = compute_metrics(nav, turnover=turn, periods_per_year=12)
    assert m.annual_turnover == pytest.approx(0.4 * 12)
