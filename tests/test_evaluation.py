"""Tests for the factor-evaluation toolkit (quantlab.factors.evaluation).

Hermetic: IC, quantile sorts, turnover, and cross-factor correlation are checked
against constructed panels whose answers are known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import Factor
from quantlab.factors.evaluation import (
    FactorPanel,
    ICResult,
    build_factor_panel,
    decile_returns,
    factor_correlation,
    information_coefficient,
    rank_autocorrelation,
)


def _panel(factor_values, forward_returns, name="f"):
    return FactorPanel(name, pd.DataFrame(factor_values), pd.DataFrame(forward_returns))


def _dates(n):
    return pd.date_range("2020-01-31", periods=n, freq="ME")


# --------------------------------------------------------------------------- #
# Information Coefficient
# --------------------------------------------------------------------------- #

def test_ic_perfect_positive_is_one():
    dates, tickers = _dates(3), ["A", "B", "C", "D"]
    fv = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 3, index=dates, columns=tickers)
    # Forward return strictly increases with the factor -> Spearman IC = 1.
    fwd = pd.DataFrame([[0.01, 0.02, 0.03, 0.04]] * 3, index=dates, columns=tickers)
    ic = information_coefficient(_panel(fv, fwd))
    assert ic.dropna().eq(1.0).all()


def test_ic_perfect_negative_is_minus_one():
    dates, tickers = _dates(3), ["A", "B", "C", "D"]
    fv = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 3, index=dates, columns=tickers)
    fwd = pd.DataFrame([[0.04, 0.03, 0.02, 0.01]] * 3, index=dates, columns=tickers)
    ic = information_coefficient(_panel(fv, fwd))
    assert ic.dropna().eq(-1.0).all()


def test_ic_rank_based_ignores_monotone_scaling():
    # Spearman only sees ranks: a monotone (non-linear) forward map still gives 1.
    dates, tickers = _dates(2), ["A", "B", "C", "D"]
    fv = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 2, index=dates, columns=tickers)
    fwd = pd.DataFrame([[0.001, 0.5, 0.51, 9.0]] * 2, index=dates, columns=tickers)
    ic = information_coefficient(_panel(fv, fwd))
    assert ic.dropna().eq(1.0).all()


def test_ic_drops_dates_with_too_few_pairs():
    dates, tickers = _dates(2), ["A", "B"]
    fv = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], index=dates, columns=tickers)
    # Second date has only one non-NaN forward return -> dropped.
    fwd = pd.DataFrame([[0.1, 0.2], [0.1, np.nan]], index=dates, columns=tickers)
    ic = information_coefficient(_panel(fv, fwd))
    assert list(ic.index) == [dates[0]]


def test_ic_result_summary_stats():
    ic = pd.Series([0.1, 0.2, 0.3, 0.2], index=_dates(4))
    res = ICResult.from_series(ic, "f")
    assert res.n == 4
    assert res.mean == pytest.approx(0.2)
    assert res.hit_rate == 1.0
    # ICIR = mean / std, and t-stat = ICIR * sqrt(n).
    assert res.icir == pytest.approx(res.mean / ic.std(ddof=1))
    assert res.t_stat == pytest.approx(res.icir * np.sqrt(4))


# --------------------------------------------------------------------------- #
# Quantile (decile) returns
# --------------------------------------------------------------------------- #

def test_decile_monotone_factor_is_monotone_and_positive_spread():
    dates = _dates(3)
    tickers = [f"T{i:02d}" for i in range(20)]
    # Factor value = rank; forward return increases with it, every date.
    fv = pd.DataFrame([[float(i) for i in range(20)]] * 3,
                      index=dates, columns=tickers)
    fwd = pd.DataFrame([[0.001 * i for i in range(20)]] * 3,
                       index=dates, columns=tickers)
    res = decile_returns(_panel(fv, fwd), n=5)

    # Five buckets, equal-count -> each holds 4 of the 20 names.
    assert res.n_buckets == 5
    # Top bucket beats bottom; ranking is perfectly monotone.
    assert res.annualized[5] > res.annualized[1]
    assert res.monotonicity == pytest.approx(1.0)
    assert res.long_short_annualized > 0.0


def test_decile_bucket_assignment_counts_and_membership():
    # A single date; check the top bucket holds the highest-factor names and the
    # bottom bucket the lowest, in equal counts.
    dates = _dates(1)
    tickers = [f"T{i:02d}" for i in range(10)]
    fv = pd.DataFrame([[float(i) for i in range(10)]], index=dates, columns=tickers)
    fwd = pd.DataFrame([[1.0] * 10], index=dates, columns=tickers)
    res = decile_returns(_panel(fv, fwd), n=5)
    # With a flat forward return every bucket earns the same -> spread 0.
    assert res.long_short_monthly.iloc[0] == pytest.approx(0.0)
    # Monthly frame has one row per date and one column per bucket.
    assert res.monthly_returns.shape == (1, 5)
    assert list(res.monthly_returns.columns) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_decile_inverse_factor_has_negative_spread():
    dates = _dates(3)
    tickers = [f"T{i:02d}" for i in range(20)]
    fv = pd.DataFrame([[float(i) for i in range(20)]] * 3,
                      index=dates, columns=tickers)
    # Forward return DECREASES with the factor -> top-minus-bottom is negative.
    fwd = pd.DataFrame([[0.001 * (19 - i) for i in range(20)]] * 3,
                       index=dates, columns=tickers)
    res = decile_returns(_panel(fv, fwd), n=5)
    assert res.long_short_annualized < 0.0
    assert res.monotonicity == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Turnover and cross-factor correlation
# --------------------------------------------------------------------------- #

def test_rank_autocorrelation_stable_ranking_is_one():
    dates, tickers = _dates(4), ["A", "B", "C", "D"]
    fv = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 4, index=dates, columns=tickers)
    ac = rank_autocorrelation(_panel(fv, fv))
    assert ac.dropna().eq(1.0).all()
    assert list(ac.index) == list(dates[1:])   # indexed by the later date


def test_rank_autocorrelation_reshuffled_ranking_is_negative():
    dates, tickers = _dates(2), ["A", "B", "C", "D"]
    fv = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                      index=dates, columns=tickers)
    ac = rank_autocorrelation(_panel(fv, fv))
    assert ac.iloc[0] == pytest.approx(-1.0)


def test_factor_correlation_matrix_identity_and_opposite():
    dates, tickers = _dates(3), ["A", "B", "C", "D"]
    up = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 3, index=dates, columns=tickers)
    down = pd.DataFrame([[4.0, 3.0, 2.0, 1.0]] * 3, index=dates, columns=tickers)
    p_up = FactorPanel("up", up, up)
    p_down = FactorPanel("down", down, down)
    corr = factor_correlation([p_up, p_down])
    assert corr.loc["up", "up"] == pytest.approx(1.0)
    assert corr.loc["up", "down"] == pytest.approx(-1.0)
    assert list(corr.columns) == ["up", "down"]


# --------------------------------------------------------------------------- #
# Panel construction (forward-return correctness + PIT)
# --------------------------------------------------------------------------- #

class _PriceFactor(Factor):
    """Trivial factor: returns each ticker's latest adjusted close as of date."""

    name = "last_price"

    def _compute(self, prices, universe, as_of_date):
        wide = prices.pivot_table(index="date", columns="ticker",
                                  values="adj_close", aggfunc="last")
        return wide.iloc[-1] if len(wide) else pd.Series(dtype=float)


def test_build_panel_forward_returns_match_price_ratios():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    a = [100.0, 110.0, 121.0, 133.1]     # +10% each month
    b = [50.0, 55.0, 60.5, 66.55]
    rows = []
    for t, series in {"A": a, "B": b}.items():
        for d, px in zip(dates, series, strict=True):
            rows.append({"date": d, "ticker": t, "adj_close": px,
                         "close": px, "volume": 1_000})
    long_prices = pd.DataFrame(rows)

    panel = build_factor_panel(
        _PriceFactor(), long_prices, list(dates),
        members_by_date=lambda dt: ["A", "B"], history_months=12,
    )
    # Forward return over each month is +10% for both, NaN on the last date.
    fwd = panel.forward_returns
    assert fwd.loc[dates[0], "A"] == pytest.approx(0.10)
    assert fwd.loc[dates[2], "B"] == pytest.approx(0.10)
    assert np.isnan(fwd.loc[dates[-1], "A"])
    # Factor value on the first date is that date's price (PIT: no future rows).
    assert panel.factor_values.loc[dates[0], "A"] == pytest.approx(100.0)
