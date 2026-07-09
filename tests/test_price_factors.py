"""Tests for the pure-price factors: low volatility, reversal, Amihud illiquidity.

Hermetic: each factor's sign and ranking are asserted against hand-built price
series with known closed-form behaviour, plus the shared PIT/NaN contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import (
    AmihudIlliquidityFactor,
    LookaheadBiasError,
    LowVolatilityFactor,
    ShortTermReversalFactor,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _daily_long(prices_by_ticker, end="2020-06-30", volume=1_000_000):
    """Long daily price frame ending at `end`. Each ticker maps to a list of
    (adjusted == raw) closes, oldest first; volume constant unless overridden."""
    n = max(len(v) for v in prices_by_ticker.values())
    dates = pd.bdate_range(end=end, periods=n)
    vol_map = volume if isinstance(volume, dict) else {t: volume for t in prices_by_ticker}
    rows = []
    for ticker, series in prices_by_ticker.items():
        pad = n - len(series)
        for i, px in enumerate(series):
            rows.append({
                "date": dates[pad + i], "ticker": ticker,
                "adj_close": px, "close": px, "volume": vol_map[ticker],
            })
    return pd.DataFrame(rows)


def _monthly_long(prices_by_ticker, end="2020-06-30"):
    n = max(len(v) for v in prices_by_ticker.values())
    dates = pd.date_range(end=end, periods=n, freq="ME")
    rows = []
    for ticker, series in prices_by_ticker.items():
        pad = n - len(series)
        for i, px in enumerate(series):
            rows.append({"date": dates[pad + i], "ticker": ticker,
                         "adj_close": px, "close": px, "volume": 1_000})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Low volatility
# --------------------------------------------------------------------------- #

def test_low_vol_ranks_calm_stock_above_choppy():
    # CALM drifts smoothly; CHOPPY has the same trend but big daily swings.
    rng = np.arange(60)
    calm = [100.0 * (1.0005 ** i) for i in rng]
    choppy = [100.0 * (1.0005 ** i) * (1.0 + 0.05 * (-1) ** i) for i in rng]
    prices = _daily_long({"CALM": calm, "CHOPPY": choppy})
    f = LowVolatilityFactor(window=40, min_periods=20)
    lv = f.compute(prices, ["CALM", "CHOPPY"], "2020-06-30")
    # Higher factor value = lower volatility = the calm name.
    assert lv["CALM"] > lv["CHOPPY"]


def test_low_vol_is_negative_of_std():
    # Deterministic two-point return pattern -> checkable std.
    series = [100.0, 101.0, 102.01, 103.0301, 104.060401]  # exact +1%/day
    prices = _daily_long({"UP": series})
    f = LowVolatilityFactor(window=10, min_periods=2)
    lv = f.compute(prices, ["UP"], "2020-06-30")
    # Constant daily return => zero dispersion => factor 0 (== -0.0).
    assert lv["UP"] == pytest.approx(0.0, abs=1e-12)


def test_low_vol_insufficient_history_is_nan():
    prices = _daily_long({"NEW": [100.0, 101.0, 102.0]})
    lv = LowVolatilityFactor(window=252, min_periods=100).compute(
        prices, ["NEW"], "2020-06-30")
    assert np.isnan(lv["NEW"])


# --------------------------------------------------------------------------- #
# Short-term reversal
# --------------------------------------------------------------------------- #

def test_reversal_sign_is_negative_of_past_return():
    # WIN rose 20% last month, LOSE fell 20%. Reversal flips the sign, so LOSE
    # (the recent loser) gets the higher factor value.
    win = [100.0, 100.0, 120.0]      # last month +20%
    lose = [100.0, 100.0, 80.0]      # last month -20%
    prices = _monthly_long({"WIN": win, "LOSE": lose})
    rev = ShortTermReversalFactor(lookback_months=1).compute(
        prices, ["WIN", "LOSE"], "2020-06-30")
    assert rev["LOSE"] > rev["WIN"]
    assert rev["WIN"] == pytest.approx(-0.20, rel=1e-9)
    assert rev["LOSE"] == pytest.approx(0.20, rel=1e-9)


def test_reversal_insufficient_history_is_nan():
    prices = _monthly_long({"NEW": [100.0]})
    rev = ShortTermReversalFactor().compute(prices, ["NEW"], "2020-06-30")
    assert np.isnan(rev["NEW"])


# --------------------------------------------------------------------------- #
# Amihud illiquidity
# --------------------------------------------------------------------------- #

def test_amihud_ranks_thin_volume_as_more_illiquid():
    # Identical price path; THIN trades 100x less volume => higher price impact
    # => more illiquid => higher factor value.
    rng = np.arange(40)
    path = [100.0 * (1.0 + 0.02 * (-1) ** i) for i in rng]
    prices = _daily_long(
        {"THICK": path, "THIN": path},
        volume={"THICK": 10_000_000, "THIN": 100_000},
    )
    f = AmihudIlliquidityFactor(window=30, min_periods=10)
    illiq = f.compute(prices, ["THICK", "THIN"], "2020-06-30")
    assert illiq["THIN"] > illiq["THICK"] > 0


def test_amihud_scale_is_monotone_not_rank_changing():
    rng = np.arange(40)
    a = [100.0 * (1.0 + 0.02 * (-1) ** i) for i in rng]
    b = [100.0 * (1.0 + 0.01 * (-1) ** i) for i in rng]
    prices = _daily_long({"A": a, "B": b},
                         volume={"A": 1_000_000, "B": 1_000_000})
    hi = AmihudIlliquidityFactor(window=30, min_periods=10, scale=1e6).compute(
        prices, ["A", "B"], "2020-06-30")
    lo = AmihudIlliquidityFactor(window=30, min_periods=10, scale=1.0).compute(
        prices, ["A", "B"], "2020-06-30")
    # Different scale, same ordering.
    assert (hi["A"] > hi["B"]) == (lo["A"] > lo["B"])
    assert hi["A"] == pytest.approx(lo["A"] * 1e6, rel=1e-9)


def test_amihud_requires_volume_column():
    prices = _daily_long({"X": [100.0, 101.0, 102.0]}).drop(columns=["volume"])
    with pytest.raises(ValueError):
        AmihudIlliquidityFactor().compute(prices, ["X"], "2020-06-30")


# --------------------------------------------------------------------------- #
# Shared contract: the PIT guard applies to every factor
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("factor", [
    LowVolatilityFactor(window=10, min_periods=2),
    ShortTermReversalFactor(),
    AmihudIlliquidityFactor(window=5, min_periods=2),
])
def test_price_factors_enforce_pit(factor):
    prices = _daily_long({"X": [100.0 * (1.01 ** i) for i in range(30)]},
                         end="2020-07-31")
    with pytest.raises(LookaheadBiasError):
        factor.compute(prices, ["X"], "2020-06-30")
